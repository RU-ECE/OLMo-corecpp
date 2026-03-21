#!/bin/bash
# Benchmark: Standard vs Fused vs Fused+µP
#
# Runs all three configurations on the same seed and compares throughput,
# loss convergence, and per-operation timing.
#
# Usage:
#   ./scripts/benchmark_optimizations.sh           # auto device
#   ./scripts/benchmark_optimizations.sh mps       # explicit device
#   ./scripts/benchmark_optimizations.sh cpu 100   # device + steps

set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${1:-cpu}"
STEPS="${2:-50}"
SEED=42
BATCH_SIZE=4
SEQ_LEN=256

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  OLMo C++ Optimization Benchmark                           ║"
echo "║  Device: $DEVICE | Steps: $STEPS | Seed: $SEED             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check build exists
if [ ! -f "build/olmo_train" ]; then
    echo "Building project..."
    cmake --build build -j$(sysctl -n hw.ncpu 2>/dev/null || nproc)
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  1/3  STANDARD MODEL (baseline)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
./build/olmo_train --train --seed $SEED --profile \
    --steps $STEPS --batch-size $BATCH_SIZE --seq-len $SEQ_LEN \
    --device $DEVICE --optimizer muon --lr 3e-4 --warmup-steps 10 \
    2>&1 | tee /tmp/bench_standard.txt

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  2/3  FUSED MODEL (fused QKV + fused gate_up)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
./build/olmo_train --train --fused --seed $SEED --profile \
    --steps $STEPS --batch-size $BATCH_SIZE --seq-len $SEQ_LEN \
    --device $DEVICE --optimizer muon --lr 3e-4 --warmup-steps 10 \
    2>&1 | tee /tmp/bench_fused.txt

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  3/3  FUSED + µP MODEL (fused ops + µP initialization)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
./build/olmo_train --train --fused --mup --seed $SEED --profile \
    --steps $STEPS --batch-size $BATCH_SIZE --seq-len $SEQ_LEN \
    --device $DEVICE --optimizer muon --lr 3e-4 --warmup-steps 10 \
    2>&1 | tee /tmp/bench_fused_mup.txt

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  COMPARISON SUMMARY                                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Extract throughput from each run
STANDARD_TOKS=$(grep "Throughput:" /tmp/bench_standard.txt | awk '{print $2}')
FUSED_TOKS=$(grep "Throughput:" /tmp/bench_fused.txt | awk '{print $2}')
FUSED_MUP_TOKS=$(grep "Throughput:" /tmp/bench_fused_mup.txt | awk '{print $2}')

# Extract final loss
STANDARD_LOSS=$(grep "Step" /tmp/bench_standard.txt | tail -1 | grep -o 'loss: [0-9.]*' | awk '{print $2}')
FUSED_LOSS=$(grep "Step" /tmp/bench_fused.txt | tail -1 | grep -o 'loss: [0-9.]*' | awk '{print $2}')
FUSED_MUP_LOSS=$(grep "Step" /tmp/bench_fused_mup.txt | tail -1 | grep -o 'loss: [0-9.]*' | awk '{print $2}')

printf "  %-25s %10s %10s\n" "Configuration" "tok/s" "final_loss"
printf "  %-25s %10s %10s\n" "─────────────────────────" "──────────" "──────────"
printf "  %-25s %10s %10s\n" "Standard" "${STANDARD_TOKS:-N/A}" "${STANDARD_LOSS:-N/A}"
printf "  %-25s %10s %10s\n" "Fused" "${FUSED_TOKS:-N/A}" "${FUSED_LOSS:-N/A}"
printf "  %-25s %10s %10s\n" "Fused + µP" "${FUSED_MUP_TOKS:-N/A}" "${FUSED_MUP_LOSS:-N/A}"

echo ""
echo "  Note: On CUDA with larger models (1B+), expect 5-20x speedup from"
echo "  fused ops + BF16 + Flash Attention (SDPA dispatch) + CUDA Graphs."
echo "  CPU/MPS improvements are more modest (1.2-2x) due to less kernel overhead."
echo ""
