#!/usr/bin/env bash
# scripts/race/00_env_check.sh
#
# Pre-flight: verify the 5060 Ti environment is ready before the race.
# Bails on any missing prereq.

set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root

say() { printf "\033[1;36m[env]\033[0m %s\n" "$*"; }
ok()  { printf "\033[1;32m  ✓\033[0m  %s\n" "$*"; }
fail(){ printf "\033[1;31m  ✗\033[0m  %s\n" "$*"; exit 1; }

say "── GPU ──"
command -v nvidia-smi >/dev/null || fail "nvidia-smi not on PATH"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
[[ "$gpu_count" -ge 1 ]] || fail "no CUDA device detected"
ok "$gpu_count CUDA device(s) visible"

# Check compute capability. 5060 Ti = sm_120. A100 = sm_80. H100 = sm_90.
# Race works on sm_80+; warn on lower.
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
if [[ "$cc" -lt 80 ]]; then
  fail "compute capability sm_$cc < sm_80, kernels will not run"
fi
ok "compute capability sm_$cc"

say "── Build deps ──"
for tool in cmake make gcc g++ git python3 pip3; do
  command -v "$tool" >/dev/null || fail "missing: $tool"
  ok "$tool: $($tool --version 2>&1 | head -1)"
done

# CUDA toolkit (for nvcc, headers).
if ! command -v nvcc >/dev/null; then
  fail "nvcc not on PATH — install CUDA Toolkit ≥ 12.0"
fi
nvcc_v=$(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | head -1)
ok "nvcc: $nvcc_v"

say "── Python ──"
py_v=$(python3 --version | awk '{print $2}')
ok "python3: $py_v"

say "── PyTorch ──"
if ! python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())" 2>/dev/null; then
  fail "torch not installed. Run:  pip3 install torch --index-url https://download.pytorch.org/whl/cu121"
fi

torch_cuda=$(python3 -c "import torch; print(torch.cuda.is_available())")
[[ "$torch_cuda" == "True" ]] || fail "torch.cuda.is_available() == False — torch+cu mismatch"
ok "torch sees CUDA"

say "── Disk space ──"
free_gb=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
[[ "$free_gb" -ge 10 ]] || fail "need ≥ 10 GB free, have ${free_gb} GB"
ok "${free_gb} GB free"

say "── Repo state ──"
[[ -f CLAUDE.md ]] || fail "not at repo root"
[[ -f conf/olmo_125M.conf ]] || fail "conf/ missing — wrong directory?"
[[ -f scripts/race/configs/race_250m_cpp.conf ]] || fail "race config missing"
ok "repo layout looks right"

say "── OLMo-core (Python reference) ──"
if [[ -d OLMo-corecpp ]]; then
  ok "OLMo-corecpp/ present at $(pwd)/OLMo-corecpp"
else
  printf "\033[1;33m  ⚠\033[0m  OLMo-corecpp/ missing — Python training side will need to be set up\n"
  printf "       (see scripts/race/05_train_python.sh)\n"
fi

printf "\n\033[1;32mALL ENV CHECKS PASSED\033[0m\n"
