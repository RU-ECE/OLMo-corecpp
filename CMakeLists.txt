cmake_minimum_required(VERSION 3.18)
project(olmo_cpp LANGUAGES CXX)

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
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    add_compile_definitions(USE_CUDA=1)
    message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER} (version ${CMAKE_CUDA_COMPILER_VERSION})")

    # GPU architecture: auto-detect or use H100 (sm_90) as default
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # Detect GPU at build time; fall back to broad compatibility
        include(FindCUDAToolkit)
        if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.0")
            set(CMAKE_CUDA_ARCHITECTURES "70;80;90")  # V100, A100, H100
            message(STATUS "CUDA 12+: targeting sm_70, sm_80, sm_90 (H100)")
        elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.0")
            set(CMAKE_CUDA_ARCHITECTURES "70;80")
            message(STATUS "CUDA 11: targeting sm_70, sm_80")
        else()
            set(CMAKE_CUDA_ARCHITECTURES "70")
            message(STATUS "CUDA <11: targeting sm_70")
        endif()
    else()
        message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    endif()
endif()

# Find LibTorch (CPU or CUDA build)
find_package(Torch REQUIRED)
find_package(ZLIB REQUIRED)

# cnpy for .npy token files
add_subdirectory(third_party/cnpy)

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
    ${TORCH_INCLUDE_DIRS}
    ${JSON_INCLUDE_DIR}
)
target_link_libraries(olmo_cpp PUBLIC ${TORCH_LIBRARIES} cnpy)
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
)
target_include_directories(prepare_data PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${JSON_INCLUDE_DIR}
)
target_link_libraries(prepare_data PRIVATE cnpy)
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
