#!/usr/bin/env python3
"""Reference logits from a Hugging Face model, for tools/test_hf_logits.

  hf_logits.py --model DIR --out OUT [--seq 48] [--batch 2]
  hf_logits.py --make-tiny-qwen2 DIR       # a small random Qwen2 that exercises GQA,
                                           # q/k/v bias and an FFN width that is not a
                                           # multiple of 256, plus its llm-cpp config

Writes OUT/ids.npy (int64 [B, S]) and OUT/ref.npy (float32 [B, S, V]) from a forward
with eager attention (--dtype fp32 by default; bf16 gives the bf16 noise floor). Row 0 is tokenized text when the model dir has a
tokenizer, other rows are seeded random ids. Also writes OUT/llmcpp_config.json, the
llm-cpp config derived from the HF config.json (Qwen2 / Llama-style pre-norm).
"""
import argparse
import json
import os

import numpy as np
import torch


def llmcpp_config(hf):
    if hf.get("model_type") not in ("qwen2", "llama"):
        raise SystemExit(f"unsupported model_type {hf.get('model_type')}")
    if hf.get("rope_scaling"):
        raise SystemExit(f"rope_scaling {hf['rope_scaling']} is not ported")
    theta = hf.get("rope_theta") or (hf.get("rope_parameters") or {}).get("rope_theta")
    return {
        "_comment": f"derived from HF {hf.get('model_type')} config.json by scripts/hf_logits.py",
        "d_model": hf["hidden_size"],
        "vocab_size": hf["vocab_size"],
        "n_layers": hf["num_hidden_layers"],
        "n_heads": hf["num_attention_heads"],
        "n_kv_heads": hf.get("num_key_value_heads", hf["num_attention_heads"]),
        "head_dim": -1,
        "rope_theta": theta,
        "layer_norm_eps": hf["rms_norm_eps"],
        "init_std": 0.02,
        "use_qk_norm": False,
        "use_head_qk_norm": False,
        "block_type": "pre_norm",
        "attention_backend": "sdpa",
        "layer_norm_type": "rms_norm",
        "ffn_hidden_size": hf["intermediate_size"],
        "attention_bias": hf.get("model_type") == "qwen2" or bool(hf.get("attention_bias")),
        "use_embedding_norm": False,
        "num_mtp_heads": 0,
    }


def make_tiny(d):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=1000, hidden_size=128, intermediate_size=344,
                      num_hidden_layers=3, num_attention_heads=8, num_key_value_heads=2,
                      rms_norm_eps=1e-6, rope_theta=1000000.0, tie_word_embeddings=False,
                      max_position_embeddings=512)
    m = Qwen2ForCausalLM(cfg)
    with torch.no_grad():  # HF zero-inits biases and sets norms to 1: randomize so they count
        for n, p in m.named_parameters():
            if n.endswith(".bias") or "norm" in n:
                p.add_(torch.randn_like(p) * 0.5)
    m.save_pretrained(d, safe_serialization=True)
    print("tiny qwen2 ->", d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--out")
    ap.add_argument("--seq", type=int, default=48)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--make-tiny-qwen2")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if a.make_tiny_qwen2:
        make_tiny(a.make_tiny_qwen2)
        return
    from transformers import AutoModelForCausalLM
    os.makedirs(a.out, exist_ok=True)
    hf = json.load(open(os.path.join(a.model, "config.json")))
    json.dump(llmcpp_config(hf), open(os.path.join(a.out, "llmcpp_config.json"), "w"), indent=2)
    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt, attn_implementation="eager").to(a.device).eval()
    V = hf["vocab_size"]
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, min(V, 150000), (a.batch, a.seq), generator=g)
    if os.path.exists(os.path.join(a.model, "tokenizer.json")):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model)
        text = ("def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n\n"
                "# Two's complement of an 8-bit value, then render it as hex.\n") * 4
        t = tok(text)["input_ids"][: a.seq]
        ids[0, : len(t)] = torch.tensor(t)
    with torch.no_grad():
        logits = m(ids.to(a.device)).logits.float().cpu()
    np.save(os.path.join(a.out, "ids.npy"), ids.numpy().astype(np.int64))
    np.save(os.path.join(a.out, "ref.npy"), logits.numpy().astype(np.float32))
    print(f"ids {tuple(ids.shape)} logits {tuple(logits.shape)} -> {a.out}")


if __name__ == "__main__":
    main()
