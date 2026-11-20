# test_cuda_parity on Colab A100-SXM4-40GB, 2026-11-20

Environment: driver 580.82.07, torch 2.14.1+cu130 (CUDA 13.0), llm-cpp speed/phase0 @ b4f553a plus the
uncommitted tools/test_cuda_parity.cpp (final version sha256 f7373d01497f7fd63baa56e4063db3730287812a5c2221620de7b1f5c0aafaea). EXIT 0,
three runs byte-identical (fixed seed; md5 8fb982a72238feda230cad5e3edc4635).

The two "open bugs" of 2026-11-15 were the test, not the kernels: unscaled randn inputs make the FFN output
reach 2.3e4 (one bf16 step = 128 there) and q/k reach ~150 (step 0.5-1), so a fixed absolute tolerance
failed on rounding. Against an fp64 truth from the same bf16 inputs the fused kernels match plain ATen bf16:

| Output | fused rel err | ATen bf16 rel err | max abs diff vs ATen | planted 2% error |
|---|---|---|---|---|
| fused FFN forward y | 3.464e-03 | 3.464e-03 | 128 | 2.045e-02 (fails, as it must) |
| fused QKV+RoPE forward q | 4.004e-03 | 3.969e-03 | 0.5 | 1.712e-02 (fails) |
| fused QKV+RoPE forward k | 2.972e-03 | 3.857e-03 | 0.5 | 1.886e-02 (fails) |

Gate: fused rel err <= 2 x ATen's + 1e-3 and < 2%. All backward checks were already exact (max |diff| 0).

Discriminating check (the kernels on inputs scaled so outputs are O(1), under the ORIGINAL absolute
tolerances): rounding shrinks with magnitude, a real bug would not.

| Output (O(1) inputs) | max |y| | max abs diff | original tolerance |
|---|---|---|---|
| fused FFN y | 2.547 | 1.562e-02 | 0.2 |
| fused QKV+RoPE q | 4.312 | 3.125e-02 | 0.05 |
| fused QKV+RoPE k | - | 1.562e-02 | 0.05 |

Side finding while adding it: the RMSNorm bf16 grad_w check at d = 24576 used a fixed 0.15, below one bf16
step (0.25) for its largest entries (~32-64, a 64-row sum); with the random stream shifted it failed at
exactly 0.25. Its bf16 tolerance is now max(fixed, max|ref|/128) (one step), and a planted 5% grad_w error
(1.3-2.0 abs) fails it at every width, bf16 and fp32.

Full output of the FIRST test version (fp64 check only). The final version (with the O(1) check and the
RMSNorm tolerance fix) was summarised from the terminal; its raw output files (md5 above) were on the
runtime, which has been deleted.
```
=== Fused FFN (WMMA path) — CUDA bf16 ===
(info) forward y max |diff| 128
forward y (vs fp64) rel err fused 3.464e-03, ATen bf16 3.464e-03, |truth| max 2.303e+04, planted 2% 2.045e-02 [ok]
backward grad_x max |diff| = 0.000e+00 [ok]
backward grad_w_gu max |diff| = 0.000e+00 [ok]
backward grad_w_dn max |diff| = 0.000e+00 [ok]
=== Fused QKV+RoPE — CUDA bf16 ===
(info) forward q/k max |diff| 5.000e-01 / 5.000e-01
forward q (vs fp64) rel err fused 4.004e-03, ATen bf16 3.969e-03, |truth| max 1.434e+02, planted 2% 1.712e-02 [ok]
forward k (vs fp64) rel err fused 2.972e-03, ATen bf16 3.857e-03, |truth| max 1.522e+02, planted 2% 1.886e-02 [ok]
forward v max |diff| = 0.000e+00 [ok]
backward grad_x max |diff| = 0.000e+00 [ok]
backward grad_W max |diff| = 0.000e+00 [ok]
=== Fused LM-head + CE — CUDA bf16 ===
loss scalar max |diff| = 5.951e-04 [ok]
backward grad_h max |diff| = 1.221e-04 [ok]
backward grad_W max |diff| = 1.221e-04 [ok]
=== RMSNorm gradient flow (d = 768, 5120, 8192, 24576; bf16 and fp32) ===
all ok (bf16 max |diff| <= 1.25e-01, fp32 <= 9.537e-07)
=== silu_mul gradient flow === all 0.000e+00 [ok]
=== apply_rope with per-position tables ===
fp32 forward 2.384e-07, bf16 forward 1.379e-02, planted position-0-only rotation differs by 7.5 / 7.35 [ok: the check can fail]
ALL CUDA PARITY TESTS PASSED
EXIT 0
```
