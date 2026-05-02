# CUDA kernels for fused operations
# Auto-detected and linked when CUDA compiler is available

add_library(olmo_kernels SHARED
    rms_norm.cu
    rope.cu
    silu_mul.cu
    topp_radix.cu              # fast-inference [7]
    fused_lm_head_sample.cu    # fast-inference [6]
)
target_include_directories(olmo_kernels PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(olmo_kernels PRIVATE ${TORCH_LIBRARIES} CUDA::cudart)

# CUDA compilation flags: fast math + architecture-aware optimization
target_compile_options(olmo_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
        --expt-relaxed-constexpr
    >
)
set_target_properties(olmo_kernels PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    PREFIX ""
    CUDA_SEPARABLE_COMPILATION ON
)
