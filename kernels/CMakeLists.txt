# CUDA kernels for fused operations
# Auto-detected and linked when CUDA compiler is available

add_library(olmo_kernels SHARED
    rms_norm.cu
    rope.cu
    silu_mul.cu
    topp_radix.cu              # fast-inference [7]
    fused_lm_head_sample.cu    # fast-inference [6]
    paged_attention.cu         # fast-inference [9]
    lm_head_gemv.cu            # fast-inference [11]
    flash_decode.cu            # fast-inference [12]
    sparse_attn_decode.cu      # SubQ Phase-1: fused content-selected SDPA
    quant_dequant.cu           # FP8 / INT4 AWQ device-resident dequant
    topk_argmax.cu             # K (fused GEMV+argmax, no logits materialize)
    fused_qkv_rope.cu          # G (fused QKV projection + reshape + RoPE)
    fused_qkv_rope_wmma.cu     # G + item 2: tensor-core (WMMA) variant
    fused_ffn.cu               # I (fused SwiGLU FFN macro)
    fused_ffn_wmma.cu          # I + item 2: tensor-core (WMMA) variant
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
