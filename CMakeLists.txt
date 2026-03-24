cmake_minimum_required(VERSION 3.18)
project(olmo_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Find LibTorch (CPU or CUDA build)
find_package(Torch REQUIRED)
find_package(ZLIB REQUIRED)

# cnpy for .npy token files
# Save/restore output dirs: PyTorch's cmake config may override these globally,
# which breaks the cnpy target's output path in the Makefile generator.
set(_saved_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")
set(_saved_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
set(_saved_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
unset(CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
unset(CMAKE_LIBRARY_OUTPUT_DIRECTORY)
unset(CMAKE_RUNTIME_OUTPUT_DIRECTORY)
add_subdirectory(third_party/cnpy)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${_saved_ARCHIVE_OUTPUT_DIRECTORY}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${_saved_LIBRARY_OUTPUT_DIRECTORY}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${_saved_RUNTIME_OUTPUT_DIRECTORY}")

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

# Main library
set(OLMO_SOURCES
    # Config
    src/config.cpp
    # Data pipeline
    src/data/token_dataset.cpp
    src/data/bpe_tokenizer.cpp
    src/data/collator.cpp
    src/data/source_mixture.cpp
    src/data/vsl_dataset.cpp
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
    src/model/transformer.cpp
    src/model/convolution.cpp
    src/model/gated_attention.cpp
    # Model: MoE
    src/model/moe/router.cpp
    src/model/moe/loss.cpp
    src/model/moe/mlp.cpp
    src/model/moe/moe.cpp
    # Optimizers
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
target_compile_options(olmo_cpp PRIVATE -O3)

# CUDA kernels (optional)
# if(CMAKE_CUDA_COMPILER)
#     add_subdirectory(kernels)
#     target_link_libraries(olmo_cpp PUBLIC olmo_kernels)
#     target_compile_definitions(olmo_cpp PRIVATE OLMO_HAS_CUDA_KERNELS=1)
# endif()

# Helper: on Linux, GNU ld is order-sensitive with static libraries.
# Executables linking olmo_cpp need cnpy repeated after olmo_cpp so the
# linker resolves cnpy symbols that olmo_cpp references.
macro(olmo_add_executable target_name)
    add_executable(${target_name} ${ARGN})
    target_link_libraries(${target_name} PRIVATE olmo_cpp cnpy)
endmacro()

# Training executable
olmo_add_executable(olmo_train src/main.cpp src/train.cpp)
target_compile_options(olmo_train PRIVATE -O3)

# Checkpoint converter tool
olmo_add_executable(convert_checkpoint tools/convert_checkpoint.cpp)
target_compile_options(convert_checkpoint PRIVATE -O3)

# Interactive chat CLI
olmo_add_executable(chat tools/chat.cpp)
target_include_directories(chat PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${JSON_INCLUDE_DIR})
target_compile_options(chat PRIVATE -O3)
if(JSON_INCLUDE_DIR)
  target_compile_definitions(chat PRIVATE HAS_NLOHMANN_JSON=1)
endif()

# Param dumper (debug utility)
olmo_add_executable(dump_params tools/dump_params.cpp)

# HF checkpoint converter
olmo_add_executable(convert_hf tools/convert_hf.cpp)
target_compile_options(convert_hf PRIVATE -O3)
if(JSON_INCLUDE_DIR)
  target_compile_definitions(convert_hf PRIVATE HAS_NLOHMANN_JSON=1)
endif()

# Data preparation tool (no LibTorch - standalone, all C++)
add_executable(prepare_data
  tools/prepare_data.cpp
  src/data/simple_tokenizer.cpp
  src/data/bpe_tokenizer.cpp
)
target_include_directories(prepare_data PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${JSON_INCLUDE_DIR}
)
target_link_libraries(prepare_data PRIVATE cnpy ZLIB::ZLIB)
target_compile_options(prepare_data PRIVATE -O3)
if(JSON_INCLUDE_DIR)
  target_compile_definitions(prepare_data PRIVATE HAS_NLOHMANN_JSON=1)
endif()
