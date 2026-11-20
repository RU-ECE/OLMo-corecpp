# colibri baseline: Qwen3-Coder-30B-A3B Q4_K_M on a Colab CPU runtime (2 vCPU, 12 GB RAM), 2026-11-20

Machine: Intel Xeon @ 2.20GHz, 2 vCPU, 12 GB RAM, no swap; model file 18.6 GB (does not fit in RAM).
colibri 0a5504a1 (`qwen36` engine, GGUF read in place via `coli import`), measured with its own
`tools/datapoint.py` persistent mode, `--memory-gb 10 --max-new 64 --warm-runs 1 --rotating-runs 2`.
Page cache evicted before load (datapoint writes 13 GB to a temp file; no direct eviction on this platform).
`coli run`/`coli plan` refuse on Colab ("unnormalized cgroup membership path"), hence the explicit budget.
OpenMP team 1. Engine load 6.67 s.

| cap (experts/layer) | request | prompt tok | new tok | seconds | decode tok/s | expert hit | RSS |
|---|---|---|---|---|---|---|---|
| 32 | cold (cache evicted) | 31 | 64 | 108.01 | 1.02 | 88.7% | 5.22 GB |
| 32 | warm, same prompt (upper bound) | 31 | 64 | 91.16 | 1.17 | 88.7% | 5.22 GB |
| 32 | rotating prompts 1, 2 | 35, 36 | 0, 0 | 41.29, 47.98 | - | - | 5.22 GB |
| 16 | cold | 31 | 64 | 126.86 | 0.77 | 68.0% | 3.14 GB |
| 16 | warm, same prompt | 31 | 64 | 106.29 | 0.95 | 68.0% | 3.14 GB |
| 8 | cold | 31 | 64 | 106.21 | 0.91 | 38.0% | 2.09 GB |
| 8 | warm, same prompt | 31 | 64 | 82.93 | 1.22 | 41.5% | 2.09 GB |

Warm per-request profile (engine telemetry): expert disk 27.6 / 40.3 / 26.8 s, expert matmul 15.6 / 15.8 /
14.9 s, attention 7.2 / 7.2 / 7.0 s at cap 32 / 16 / 8. colibri's own lookup speculation: 1.05-1.07
tokens/forward. Rotating-prompt rows returned 0 tokens at every cap (excluded).

Reading: on this machine colibri decodes ~0.8-1.2 tok/s in 2-5 GB of RAM for a 30B-A3B model that does
not fit in RAM; cap 8 is as fast as cap 32 (the 2-vCPU matmul and attention, ~22 s/request, bound it as
much as the disk does). Not comparable with colibri's README figures (other machines).

The rotating-prompt rows returned 0 tokens: not a measurement (excluded; cause not yet found).
Warm profile per request: expert disk 27.6 s, expert matmul 15.6 s, attention 7.2 s: disk is the largest
cost even at 88.7% hits, which is where an engine of ours would have to win.
An earlier smoke run with `coli run --ram 10` (contaminated by a parallel compile) reported 2.13 s per
forward and 1.27 tokens/forward from colibri's lookup speculation; not used as a number.

Control: llama.cpp b11900 `llama-bench -t 2 -p 32 -n 64 -r 1` (mmap) on the same GGUF, same machine, page
cache dropped first: **did not finish in 20 min** (timeout; the 18.6 GB mapping thrashed 12 GB of RAM and the
VM stopped responding meanwhile). colibri produced 64 tokens in 83-127 s per request on the same machine.
Caveats still open: one prompt of 64 tokens per colibri cell; colibri's RSS figures do not count ~9 GB of
page cache, so "fits in 2-5 GB" is not claimed until a run with DIRECT=1 binds the budget.

## Follow-up: DIRECT=1 (O_DIRECT expert reads), fresh 2 vCPU / 12 GB runtime, same evening

`scripts/colab_colibri_direct.sh`: model file's pages dropped with posix_fadvise before each cap, datapoint
`--no-evict` (so the first request is labelled "page cache retained" by datapoint), 64 new tokens, one prompt.

| cap | request | seconds | decode tok/s | expert hit | engine RSS | expert disk / matmul / attention s |
|---|---|---|---|---|---|---|
| 8 | first | 90.02 | 1.22 | 38.0% | 2.09 GB | |
| 8 | repeat (upper bound) | 46.31 | 2.34 | 38.0% | 2.09 GB | 14.5 / 7.1 / 3.4 |
| 32 | first | 62.66 | 1.92 | 88.7% | 5.23 GB | |
| 32 | repeat (upper bound) | 50.00 | 2.38 | 88.7% | 5.23 GB | 13.9 / 7.3 / 3.3 |

System memory sampled every 5 s over both caps: max "used" 6,548 MB, max buff/cache 10,430 MB. So even with
O_DIRECT expert reads the page cache still grew to ~10 GB (the dense/attention weights and the GGUF reads
outside the expert path still go through it), and the RAM colibri really needs on this machine is not yet
pinned down: the honest statement is "engine RSS 2.1-5.2 GB; total used <= 6.5 GB; page cache up to 10.4 GB".
Not an A/B: this was a different VM and a different eviction method from the first run, and matmul and
attention time also halved (15.6 -> 7.1 s, 7.2 -> 3.3 s), which O_DIRECT cannot cause, so most of the speed-up
is the VM. Conclusion: decode is 1-2.4 tok/s on this class of machine and the baseline varies ~2x across Colab
VMs; a same-VM DIRECT on/off comparison is still owed.
Rotating-prompt rows again produced 0 tokens (excluded).
