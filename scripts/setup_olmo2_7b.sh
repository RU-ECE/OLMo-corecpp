#!/usr/bin/env bash
# scripts/setup_olmo2_7b.sh — fetch OLMo-2-1124-7B, export its tokenizer to our
# format, and convert it to our .pt. Run on a box with the model on a big disk.
# Everything lives under $VOL (default = the kuiper volume; root is too small).
#
#   VOL=/media/volume/Prep_and_Voice_Training bash scripts/setup_olmo2_7b.sh
set -euo pipefail
VOL="${VOL:-/media/volume/Prep_and_Voice_Training}"
export HF_HOME="$VOL/hf_cache"            # keep the ~14GB HF cache OFF the root disk
REPO="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="$VOL/olmo2-7b"
mkdir -p "$MODEL"

echo "[1/3] download allenai/OLMo-2-1124-7B -> $MODEL"
VOL="$VOL" python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
vol = os.environ["VOL"]
snapshot_download("allenai/OLMo-2-1124-7B", local_dir=os.path.join(vol, "olmo2-7b"),
                  allow_patterns=["*.safetensors","*.json","*.txt","tokenizer*","config*"])
print("download ok")
PY

echo "[2/3] export tokenizer -> $VOL/olmo2-tok"
python3 "$REPO/scripts/export_hf_tokenizer.py" "$MODEL/tokenizer.json" "$VOL/olmo2-tok"

echo "[3/3] convert_hf -> $MODEL/olmo2_7b.pt"
"$REPO/build/convert_hf" --hf-dir "$MODEL" \
  --config "$REPO/configs/olmo2_1124_7B.json" --output "$MODEL/olmo2_7b.pt"

echo "ALL DONE -> $MODEL/olmo2_7b.pt"
ls -la "$MODEL/olmo2_7b.pt" "$VOL/olmo2-tok/"
