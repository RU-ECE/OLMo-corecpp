# CUDA kernels for fused operations
# Use TORCH_LIBRARY for registration with LibTorch

add_library(olmo_kernels SHARED
    rms_norm.cu
    rope.cu
)
target_include_directories(olmo_kernels PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(olmo_kernels PRIVATE ${TORCH_LIBRARIES} CUDA::cudart)
target_compile_options(olmo_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)
set_target_properties(olmo_kernels PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    PREFIX ""
    CUDA_SEPARABLE_COMPILATION ON
)
