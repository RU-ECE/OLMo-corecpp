cmake_minimum_required(VERSION 3.18)
project(olmo_cpp LANGUAGES CXX)

# Default: "release with debug info" → RelWithDebInfo (-O2 + debug symbols on GCC/Clang).
# Single-config generators only (Ninja/Make); Visual Studio/Xcode use --config RelWithDebInfo.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
    "Release with debug info (optimized + symbols). Alternatives: Debug, Release, MinSizeRel." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

set(CMAKE_CXX_STANDARD 17)







# Link-time optimization for better inlining across translation units
# Disabled on Linux/GCC due to cross-library LTO linker issues with vendored deps
option(OLMO_ENABLE_LTO "Enable link-time optimization" OFF)
if(OLMO_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
    if(ipo_supported)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
        message(STATUS "LTO enabled")
    else()
        message(STATUS "LTO not supported: ${ipo_error}")
    endif()
else()
    message(STATUS "LTO disabled (use -DOLMO_ENABLE_LTO=ON to enable)")
endif()

# Enable CUDA if available (for Jetstream H100, etc.)
include(CheckLanguage)

# Prefer nvcc from CUDAToolkit_ROOT when set (-DCUDAToolkit_ROOT=...), so we don't pick
# /usr/bin/nvcc while a newer CUDA's ptxas appears earlier on PATH (mixed toolchains break).
if(NOT CMAKE_CUDA_COMPILER AND DEFINED CUDAToolkit_ROOT AND EXISTS "${CUDAToolkit_ROOT}/bin/nvcc")
  set(CMAKE_CUDA_COMPILER "${CUDAToolkit_ROOT}/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
endif()

# CMake's CUDA compiler probe defaults to sm_52. CUDA 13+ ptxas rejects sm_52; CUDA 12+
# also needs real archs for BF16 kernels. Set before check_language / enable_language.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90" CACHE STRING
    "CUDA arch list (e.g. 86 for RTX 3060, 90 for H100). Must be set before first CUDA probe.")
endif()

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    add_compile_definitions(USE_CUDA=1)
    message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER} (version ${CMAKE_CUDA_COMPILER_VERSION})")
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

    include(FindCUDAToolkit)
    if(CUDAToolkit_VERSION)
      message(STATUS "CUDA toolkit version: ${CUDAToolkit_VERSION}")
    endif()
endif()

# Find LibTorch (CPU or CUDA build)
find_package(Torch REQUIRED)
find_package(ZLIB REQUIRED)

# Detect which autocast C++ API is available (changed across PyTorch versions).
# Use STATIC_LIBRARY target type so try_compile only compiles (no link step),
# avoiding protobuf and other transitive dependency issues.
include(CheckCXXSourceCompiles)
set(_SAVE_TARGET_TYPE ${CMAKE_TRY_COMPILE_TARGET_TYPE})
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(CMAKE_REQUIRED_INCLUDES ${TORCH_INCLUDE_DIRS})
set(CMAKE_REQUIRED_FLAGS "${TORCH_CXX_FLAGS} -std=c++17")

# Probe 1: newest API — is_autocast_enabled(at::kCUDA) (PyTorch >= 2.4)
check_cxx_source_compiles("
#include <ATen/autocast_mode.h>
#include <c10/core/DeviceType.h>
void test() { (void)at::autocast::is_autocast_enabled(at::kCUDA); }
" HAVE_AUTOCAST_DEVICE_API)

# Probe 2: mid-era — is_autocast_gpu_enabled() (some PyTorch 2.x builds)
if(NOT HAVE_AUTOCAST_DEVICE_API)
    check_cxx_source_compiles("
    #include <ATen/autocast_mode.h>
    void test() { (void)at::autocast::is_autocast_gpu_enabled(); }
    " HAVE_AUTOCAST_GPU_API)
endif()

# Probe 3: older PyTorch 2.1 — is_enabled() + set_enabled(bool) (no device param, GPU-only)
if(NOT HAVE_AUTOCAST_DEVICE_API AND NOT HAVE_AUTOCAST_GPU_API)
    check_cxx_source_compiles("
    #include <ATen/autocast_mode.h>
    void test() { at::autocast::set_enabled(true); (void)at::autocast::is_enabled(); }
    " HAVE_AUTOCAST_LEGACY_API)
endif()

if(HAVE_AUTOCAST_DEVICE_API)
    add_compile_definitions(OLMO_AUTOCAST_DEVICE_API=1)
    message(STATUS "Autocast API: device-parameterized (PyTorch >= 2.4)")
elseif(HAVE_AUTOCAST_GPU_API)
    add_compile_definitions(OLMO_AUTOCAST_GPU_API=1)
    message(STATUS "Autocast API: GPU-specific (PyTorch 2.x)")
elseif(HAVE_AUTOCAST_LEGACY_API)
    add_compile_definitions(OLMO_AUTOCAST_LEGACY_API=1)
    message(STATUS "Autocast API: legacy is_enabled/set_enabled (PyTorch 2.0-2.1)")
else()
    message(STATUS "Autocast API: dispatch-key fallback (autocast may not work)")
endif()

set(CMAKE_TRY_COMPILE_TARGET_TYPE ${_SAVE_TARGET_TYPE})
unset(CMAKE_REQUIRED_INCLUDES)
unset(CMAKE_REQUIRED_FLAGS)

# cnpy for .npy token files (compiled directly into olmo_cpp to avoid GCC link-order issues)
# add_subdirectory(third_party/cnpy)

# Optional: nlohmann_json for config and prepare_data (BPE tokenizer)
find_path(JSON_INCLUDE_DIR nlohmann/json.hpp
    HINTS /usr/include /usr/local/include /opt/homebrew/include
          $ENV{HOME}/.local/include
)
if(NOT JSON_INCLUDE_DIR)
    include(FetchContent)
    FetchContent_Declare(nlohmann_json
        URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    )
    FetchContent_MakeAvailable(nlohmann_json)
    get_target_property(JSON_INCLUDE_DIR nlohmann_json::nlohmann_json INTERFACE_INCLUDE_DIRECTORIES)
    list(GET JSON_INCLUDE_DIR 0 JSON_INCLUDE_DIR)
endif()
if(JSON_INCLUDE_DIR)
    add_compile_definitions(HAS_NLOHMANN_JSON=1)
endif()

# DDP requires Gloo (not in pip LibTorch). Check for gloo.
find_path(GLOO_INCLUDE_DIR gloo/algorithm.h HINTS /usr/local/include /opt/homebrew/include)
option(OLMO_USE_DDP "Build with DDP (requires Gloo)" OFF)

# Main library: model components + data + training infrastructure
set(OLMO_SOURCES
    # cnpy (vendored .npy loader)
    third_party/cnpy/cnpy.cpp
    # Config
    src/config.cpp
    # Data pipeline
    src/data/token_dataset.cpp
    src/data/bpe_tokenizer.cpp
    src/data/collator.cpp
    src/data/source_mixture.cpp
    src/data/vsl_dataset.cpp
    src/data/structural_tokenizer.cpp
    src/data/composable/data_loader.cpp
    src/data/composable/document_source.cpp
    src/data/composable/instance_source.cpp
    src/data/composable/token_source.cpp
    # Model: core
    src/model/layer_norm.cpp
    src/model/layer_norm_variants.cpp
    src/model/rope.cpp
    src/model/rope_scaling.cpp
    src/model/feed_forward.cpp
    src/model/attention.cpp
    src/model/attention_config.cpp
    src/model/block.cpp
    src/model/block_variants.cpp
    src/model/lm_head.cpp
    src/model/mtp_head.cpp
    src/model/transformer.cpp
    src/model/fused_attention.cpp
    src/model/fused_block.cpp
    src/model/fused_transformer.cpp
    src/model/mup_init.cpp
    src/model/convolution.cpp
    src/model/gated_attention.cpp
    # Model: MoE
    src/model/moe/router.cpp
    src/model/moe/loss.cpp
    src/model/moe/mlp.cpp
    src/model/moe/moe.cpp
    # Optimizers
    src/optim/lion.cpp
    src/optim/muon.cpp
    src/optim/dion.cpp
    src/optim/foreach_adamw.cpp
    src/optim/grad_clip.cpp
    src/optim/skip_step.cpp
    src/optim/scheduler.cpp
    # Evaluation
    src/eval/metrics.cpp
    src/eval/evaluator.cpp
    src/eval/lm_evaluator.cpp
    # Float8
    src/float8/float8.cpp
    # Training infrastructure
    src/train/checkpoint.cpp
    src/train/activation_checkpoint.cpp
    src/train/grad_scaler.cpp
    src/train/callbacks/all_callbacks.cpp
    # IO abstraction
    src/io/filesystem.cpp
    # HuggingFace conversion
    src/nn/hf_convert.cpp
    # Multi-Resolution Embedding (DC-MRE)
    src/nn/multi_res_embedding.cpp
    # Compute backend
    src/backend/backend.cpp
    src/backend/simd_backend.cpp
    src/backend/arena.cpp
    src/backend/fused_ops.cpp
    src/backend/kernels/simd_elementwise.cpp
    src/backend/cuda_backend.cpp
    # Seed and profiling infrastructure
    src/seed.cpp
    src/profiler.cpp
    # Speculative decoding
    src/generate/speculative_decode.cpp
)
if(OLMO_USE_DDP AND GLOO_INCLUDE_DIR)
    list(APPEND OLMO_SOURCES
        src/distributed/ddp.cpp
        src/distributed/tensor_parallel.cpp
        src/distributed/pipeline_parallel.cpp
        src/distributed/fsdp.cpp
        src/distributed/context_parallel.cpp
        src/distributed/expert_parallel.cpp
    )
else()
    list(APPEND OLMO_SOURCES src/distributed/ddp_stub.cpp)
endif()
add_library(olmo_cpp STATIC ${OLMO_SOURCES})
if(OLMO_USE_DDP AND GLOO_INCLUDE_DIR)
    target_compile_definitions(olmo_cpp PRIVATE OLMO_HAS_DDP=1)
endif()
target_include_directories(olmo_cpp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cnpy
    ${TORCH_INCLUDE_DIRS}
    ${JSON_INCLUDE_DIR}
)
target_link_libraries(olmo_cpp PUBLIC ${TORCH_LIBRARIES} ZLIB::ZLIB)
target_compile_options(olmo_cpp PRIVATE -O3 -march=native)

# CUDA kernels (auto-enabled when CUDA compiler is found)
option(OLMO_BUILD_KERNELS "Build custom CUDA kernels" ON)
if(CMAKE_CUDA_COMPILER AND OLMO_BUILD_KERNELS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/CMakeLists.txt")
    add_subdirectory(kernels)
    target_link_libraries(olmo_cpp PUBLIC olmo_kernels)
    target_compile_definitions(olmo_cpp PRIVATE OLMO_HAS_CUDA_KERNELS=1)
    message(STATUS "Building custom CUDA kernels")
endif()

# Training executable
add_executable(olmo_train
    src/main.cpp
    src/train.cpp
)
target_link_libraries(olmo_train PRIVATE olmo_cpp)
# if(CMAKE_CUDA_COMPILER AND OLMO_BUILD_KERNELS)
#     target_link_libraries(olmo_train PRIVATE olmo_kernels)
# endif()
target_compile_options(olmo_train PRIVATE -O3 -march=native)

# Checkpoint converter tool
add_executable(convert_checkpoint tools/convert_checkpoint.cpp)
target_link_libraries(convert_checkpoint PRIVATE olmo_cpp)
target_compile_options(convert_checkpoint PRIVATE -O3)

# Interactive chat CLI
add_executable(chat tools/chat.cpp)
target_link_libraries(chat PRIVATE olmo_cpp)
target_include_directories(chat PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${JSON_INCLUDE_DIR})
target_compile_options(chat PRIVATE -O3 -march=native)
if(JSON_INCLUDE_DIR)
  target_compile_definitions(chat PRIVATE HAS_NLOHMANN_JSON=1)
endif()

# Param dumper (debug utility)
add_executable(dump_params tools/dump_params.cpp)
target_link_libraries(dump_params PRIVATE olmo_cpp)

# HF checkpoint converter
add_executable(convert_hf tools/convert_hf.cpp)
target_link_libraries(convert_hf PRIVATE olmo_cpp)
target_compile_options(convert_hf PRIVATE -O3)
if(JSON_INCLUDE_DIR)
  target_compile_definitions(convert_hf PRIVATE HAS_NLOHMANN_JSON=1)
endif()

# Data preparation tool (no LibTorch - standalone, all C++)
add_executable(prepare_data
  tools/prepare_data.cpp
  src/data/simple_tokenizer.cpp
  src/data/bpe_tokenizer.cpp
  src/data/structural_tokenizer.cpp
  third_party/cnpy/cnpy.cpp
)
target_include_directories(prepare_data PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cnpy
  ${JSON_INCLUDE_DIR}
)
target_link_libraries(prepare_data PRIVATE ZLIB::ZLIB)
target_compile_options(prepare_data PRIVATE -O3 -march=native)
if(JSON_INCLUDE_DIR)
  target_compile_definitions(prepare_data PRIVATE HAS_NLOHMANN_JSON=1)
endif()

# Token inspection tool (no LibTorch - standalone, all C++)
add_executable(inspect_tokens
  tools/inspect_tokens.cpp
  src/data/bpe_tokenizer.cpp
)
target_include_directories(inspect_tokens PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${JSON_INCLUDE_DIR}
)
target_compile_options(inspect_tokens PRIVATE -O3 -march=native)
if(JSON_INCLUDE_DIR)
  target_compile_definitions(inspect_tokens PRIVATE HAS_NLOHMANN_JSON=1)
endif()

# Pattern mining tool (no LibTorch - standalone, all C++)
add_executable(mine_patterns
  tools/mine_patterns.cpp
)
target_include_directories(mine_patterns PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_options(mine_patterns PRIVATE -O3 -march=native)

# Tokenizer benchmark (no LibTorch - standalone, all C++)
add_executable(benchmark_tokenizer
  tools/benchmark_tokenizer.cpp
  src/data/bpe_tokenizer.cpp
  src/data/structural_tokenizer.cpp
)
target_include_directories(benchmark_tokenizer PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${JSON_INCLUDE_DIR}
)
target_compile_options(benchmark_tokenizer PRIVATE -O3 -march=native)
if(JSON_INCLUDE_DIR)
  target_compile_definitions(benchmark_tokenizer PRIVATE HAS_NLOHMANN_JSON=1)
endif()
