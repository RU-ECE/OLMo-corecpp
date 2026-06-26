#!/usr/bin/env python3
"""scripts/prep_sft_data.py — build masked SFT data for the OLMo-2 finetune.

Formats an instruction dataset as ChatML, tokenizes with OLMo-2's tokenizer, and
produces a PACKED token stream + a loss mask (1 on assistant tokens + their
<|im_end|>, 0 elsewhere) so finetuning trains ONLY on the assistant responses.

Outputs (under <out_dir>):
  sft_tokens.npy  uint32  [N]   packed token ids
  sft_mask.npy    uint8   [N]   1 = compute loss here, 0 = ignore (prompt)

Usage:
  pip install -U datasets tokenizers numpy
  python scripts/prep_sft_data.py \
      --tokenizer $VOL/olmo2-7b/tokenizer.json \
      --dataset allenai/tulu-3-sft-mixture --max-examples 100000 \
      --seq-len 2048 --out $VOL/sft

ChatML per turn:
  <|im_start|>user\\n{user}<|im_end|>\\n<|im_start|>assistant\\n{asst}<|im_end|>\\n
mask = 0 over the prompt (through "assistant\\n"), 1 over {asst}<|im_end|>\\n.
"""
import argparse, numpy as np
from tokenizers import Tokenizer
from datasets import load_dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="path to OLMo-2 tokenizer.json")
    ap.add_argument("--dataset", default="allenai/tulu-3-sft-mixture")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-examples", type=int, default=100000)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tok = Tokenizer.from_file(a.tokenizer)
    def ids(s): return tok.encode(s, add_special_tokens=False).ids
    # Special-token ids (atomic).
    def sid(t):
        i = tok.token_to_id(t)
        assert i is not None, f"missing special token {t}"
        return i
    IM_START, IM_END = sid("<|im_start|>"), sid("<|im_end|>")
    NL = ids("\n")

    ds = load_dataset(a.dataset, split=a.split, streaming=True)
    toks, mask = [], []
    n = 0
    for ex in ds:
        msgs = ex.get("messages") or ex.get("conversations")
        if not msgs:
            continue
        for m in msgs:
            role = m.get("role") or m.get("from")
            content = m.get("content") or m.get("value") or ""
            role = {"human":"user","gpt":"assistant"}.get(role, role)
            if role not in ("user","assistant","system"):
                continue
            # <|im_start|> role \n  content  <|im_end|> \n
            head = [IM_START] + ids(role) + NL
            body = ids(content)
            tail = [IM_END] + NL
            seg = head + body + tail
            is_asst = (role == "assistant")
            # train only on the assistant's content + its <|im_end|>\n
            seg_mask = ([0]*len(head)) + ([1 if is_asst else 0]*len(body)) + \
                       ([1 if is_asst else 0]*len(tail))
            toks.extend(seg); mask.extend(seg_mask)
        n += 1
        if n >= a.max_examples:
            break
        if n % 5000 == 0:
            print(f"  {n} examples, {len(toks)/1e6:.1f}M tokens", flush=True)

    # Truncate to a multiple of seq_len (packed).
    L = (len(toks) // a.seq_len) * a.seq_len
    toks = np.asarray(toks[:L], dtype=np.uint32)
    mask = np.asarray(mask[:L], dtype=np.uint8)
    import os; os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, "sft_tokens.npy"), toks)
    np.save(os.path.join(a.out, "sft_mask.npy"), mask)
    print(f"DONE: {len(toks)/1e6:.1f}M tokens, {mask.mean()*100:.1f}% trainable -> {a.out}")

if __name__ == "__main__":
    main()
