#!/usr/bin/env python3
"""Convert a zwt `ZWTCKPT1` binary checkpoint into a HuggingFace Llama
safetensors directory, so the model can be loaded with `transformers` and
chatted with locally.

The zwt save format on the `zero-wait-trainer-sg` branch writes every
parameter with the literal `name = "weight"` (no hierarchical prefix).
Records collide on name, so name-based lookup is unsafe — we walk records
in **ordinal order** matching `Transformer::collect_params()`:

    [0]   tok_emb.weight                  shape [vocab, d_model]
    for each layer L in 0..n_layers-1:
      norm1.weight                        [d_model]
      attn.q_proj.weight                  [d_model, d_model]
      attn.k_proj.weight                  [d_model, d_model]
      attn.v_proj.weight                  [d_model, d_model]
      attn.out_proj.weight                [d_model, d_model]
      norm2.weight                        [d_model]
      ffn.gate.weight                     [d_ffn,  d_model]
      ffn.up.weight                       [d_ffn,  d_model]
      ffn.down.weight                     [d_model, d_ffn ]
    [N+1] final_norm.weight               [d_model]
    [N+2] lm_head.weight                  [vocab, d_model]

Each parameter generates THREE on-disk records: <value>, <.m>, <.v>.
We only consume value records (every 3rd) — moments are training state
we don't need for inference.

Usage:
    python3 scripts/zwt_bin_to_hf.py \\
        --ckpt downloads/h100_pull/owt_1B.bin \\
        --conf downloads/h100_pull/owt_1B_h100.conf \\
        --vocab downloads/h100_pull/vocab.json \\
        --merges downloads/h100_pull/merges.txt \\
        --out exports/owt_1B_hf
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import time
from pathlib import Path

# zwt DType enum (zwt/include/zwt/core/dtype.hpp)
DT_F32, DT_F16, DT_BF16, DT_I32, DT_I64, DT_U8, DT_BOOL = 0, 1, 2, 3, 4, 5, 6
DT_NAME = {0: "f32", 1: "f16", 2: "bf16", 3: "i32", 4: "i64", 5: "u8", 6: "bool"}
DT_BYTES = {0: 4, 1: 2, 2: 2, 3: 4, 4: 8, 5: 1, 6: 1}

HEADER_FMT = "<8sIIqqQiIffQ"          # 64 bytes
TREC_FMT   = "<IBBHQ6qQQ"             # 80 bytes
HEADER_SIZE = struct.calcsize(HEADER_FMT)
TREC_SIZE   = struct.calcsize(TREC_FMT)
assert HEADER_SIZE == 64, HEADER_SIZE
assert TREC_SIZE == 80, TREC_SIZE


def pad8(n: int) -> int:
    return (n + 7) & ~7


def parse_conf(path: Path) -> dict:
    """INI-ish config parser matching zwt/src/train/config.cpp behavior."""
    out: dict = {}
    section = None
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.match(r"^\[(\w+)\]$", line)
        if m:
            section = m.group(1)
            out.setdefault(section, {})
            continue
        if section is None or "=" not in line:
            continue
        k, v = (s.strip() for s in line.split("=", 1))
        # cast: try int (incl. 0x), then float, else string.
        try:
            out[section][k] = int(v, 0)
            continue
        except ValueError:
            pass
        try:
            out[section][k] = float(v)
            continue
        except ValueError:
            pass
        out[section][k] = v.lower() if v.lower() in ("true", "false") else v
    return out


def read_records(bin_path: Path):
    """Yield (rec_index, name, dtype, dims, payload_bytes) for every record."""
    f = open(bin_path, "rb")
    try:
        hdr = f.read(HEADER_SIZE)
        magic, version, flags, step, data_cursor, seed, n_records, _pad, lr, loss, _res = \
            struct.unpack(HEADER_FMT, hdr)
        if magic != b"ZWTCKPT1":
            raise ValueError(f"bad magic: {magic!r}")
        if version != 1:
            raise ValueError(f"unsupported version: {version}")
        print(f"[hdr] step={step}  seed=0x{seed:x}  n_records={n_records}  "
              f"lr={lr:.3e}  loss={loss:.4f}")

        for i in range(n_records):
            rec = f.read(TREC_SIZE)
            if len(rec) < TREC_SIZE:
                raise EOFError(f"truncated TRec at index {i}")
            (name_len, dtype, rank, _pad0, nbytes,
             d0, d1, d2, d3, d4, d5, _r0, _r1) = struct.unpack(TREC_FMT, rec)
            name = f.read(name_len).decode("utf-8", errors="replace")
            f.seek(pad8(name_len) - name_len, os.SEEK_CUR)
            payload = f.read(nbytes)
            if len(payload) < nbytes:
                raise EOFError(f"truncated payload at record {i} '{name}'")
            f.seek(pad8(nbytes) - nbytes, os.SEEK_CUR)
            dims = [d0, d1, d2, d3, d4, d5][:rank]
            yield i, name, dtype, dims, payload
    finally:
        f.close()


def build_param_schema(model_cfg: dict) -> list:
    """The expected ordering / shape of every PARAMETER (not record).

    Each entry: (hf_tensor_name, expected_dims_list, must_be_bf16).
    """
    V  = int(model_cfg["vocab_size"])
    D  = int(model_cfg["d_model"])
    H  = int(model_cfg["d_ffn"])
    N  = int(model_cfg["n_layers"])
    schema = [("model.embed_tokens.weight", [V, D], True)]
    for i in range(N):
        p = f"model.layers.{i}"
        schema += [
            (f"{p}.input_layernorm.weight",          [D],     True),
            (f"{p}.self_attn.q_proj.weight",         [D, D],  True),
            (f"{p}.self_attn.k_proj.weight",         [D, D],  True),
            (f"{p}.self_attn.v_proj.weight",         [D, D],  True),
            (f"{p}.self_attn.o_proj.weight",         [D, D],  True),
            (f"{p}.post_attention_layernorm.weight", [D],     True),
            (f"{p}.mlp.gate_proj.weight",            [H, D],  True),
            (f"{p}.mlp.up_proj.weight",              [H, D],  True),
            (f"{p}.mlp.down_proj.weight",            [D, H],  True),
        ]
    schema += [
        ("model.norm.weight",  [D],     True),
        ("lm_head.weight",     [V, D],  True),
    ]
    return schema


def write_safetensors(out_path: Path, tensors: dict, dtype_str: str = "BF16"):
    """Minimal safetensors writer. tensors: name -> bytes (raw little-endian).

    safetensors v1 layout:
      [u64 header_len][JSON header][raw payloads concatenated]
    """
    header = {}
    offset = 0
    payload_chunks = []
    for name, blob_info in tensors.items():
        shape, raw = blob_info
        size = len(raw)
        header[name] = {
            "dtype":         dtype_str,
            "shape":         list(shape),
            "data_offsets":  [offset, offset + size],
        }
        payload_chunks.append(raw)
        offset += size
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # safetensors requires the header to be 8-byte aligned via the leading u64.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for chunk in payload_chunks:
            f.write(chunk)


def write_hf_config(model_cfg: dict, out_dir: Path):
    V = int(model_cfg["vocab_size"])
    D = int(model_cfg["d_model"])
    H = int(model_cfg["d_ffn"])
    L = int(model_cfg["n_layers"])
    Q = int(model_cfg["n_heads"])
    M = int(model_cfg["max_seq"])
    rope_base = float(model_cfg.get("rope_base", 10000))
    norm_eps  = float(model_cfg.get("norm_eps", 1e-5))
    cfg = {
        "architectures":            ["LlamaForCausalLM"],
        "model_type":               "llama",
        "vocab_size":               V,
        "hidden_size":              D,
        "intermediate_size":        H,
        "num_hidden_layers":        L,
        "num_attention_heads":      Q,
        "num_key_value_heads":      Q,            # no GQA on this branch
        "max_position_embeddings":  M,
        "rms_norm_eps":             norm_eps,
        "hidden_act":               "silu",
        "rope_theta":               rope_base,
        "attention_bias":           False,
        "mlp_bias":                 False,
        "tie_word_embeddings":      False,
        "torch_dtype":              "bfloat16",
        "transformers_version":     "4.41.0",
        "bos_token_id":             50256,        # GPT-2 <|endoftext|>
        "eos_token_id":             50256,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))


def write_tokenizer(out_dir: Path, vocab_src: Path, merges_src: Path):
    import shutil
    shutil.copyfile(vocab_src,  out_dir / "vocab.json")
    shutil.copyfile(merges_src, out_dir / "merges.txt")
    tok_cfg = {
        "tokenizer_class": "GPT2Tokenizer",
        "model_max_length": 2048,
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
        "add_prefix_space": False,
    }
    (out_dir / "tokenizer_config.json").write_text(json.dumps(tok_cfg, indent=2))
    sp_cfg = {
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
    }
    (out_dir / "special_tokens_map.json").write_text(json.dumps(sp_cfg, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",   required=True, type=Path)
    ap.add_argument("--conf",   required=True, type=Path)
    ap.add_argument("--vocab",  required=True, type=Path)
    ap.add_argument("--merges", required=True, type=Path)
    ap.add_argument("--out",    required=True, type=Path)
    args = ap.parse_args()

    cfg = parse_conf(args.conf)
    if "model" not in cfg:
        print(f"conf missing [model]: {args.conf}", file=sys.stderr)
        return 2
    model_cfg = cfg["model"]
    schema = build_param_schema(model_cfg)
    expected_records = len(schema) * 3   # value + .m + .v per parameter
    print(f"[plan] {len(schema)} parameters → {expected_records} records expected")

    tensors_out: dict = {}
    param_idx  = 0    # index into schema
    record_idx = 0    # current record being processed
    bytes_total = 0
    t0 = time.perf_counter()

    for i, name, dtype, dims, payload in read_records(args.ckpt):
        # The save loop emits records in groups of 3:
        #   3*P     -> value
        #   3*P + 1 -> .m moment
        #   3*P + 2 -> .v moment
        kind = i % 3
        if kind != 0:
            continue
        if param_idx >= len(schema):
            print(f"[warn] extra value record at {i} (name={name!r}); ignoring tail",
                  file=sys.stderr)
            break

        hf_name, expected_dims, must_bf16 = schema[param_idx]
        if list(dims) != expected_dims:
            raise ValueError(
                f"shape mismatch at record {i} (param #{param_idx}, hf={hf_name}):"
                f" file={dims}  expected={expected_dims}"
            )
        if must_bf16 and dtype != DT_BF16:
            raise ValueError(
                f"dtype mismatch at record {i} ({hf_name}): "
                f"file={DT_NAME.get(dtype, dtype)}  expected=bf16"
            )
        # safetensors stores raw little-endian. zwt writes little-endian on x86,
        # so we copy bytes directly. Each bf16 element is 2 bytes.
        expected_bytes = 1
        for d in expected_dims:
            expected_bytes *= d
        expected_bytes *= DT_BYTES[dtype]
        if len(payload) != expected_bytes:
            raise ValueError(
                f"payload size mismatch at record {i} ({hf_name}): "
                f"got {len(payload)}, expected {expected_bytes}"
            )

        tensors_out[hf_name] = (expected_dims, payload)
        bytes_total += len(payload)
        param_idx += 1
        if param_idx % 16 == 0:
            print(f"  [{param_idx:>3}/{len(schema)}] {hf_name}  "
                  f"{dims}  ({len(payload)/1e6:.1f} MB)")

    if param_idx != len(schema):
        raise ValueError(f"only consumed {param_idx} value records, "
                         f"expected {len(schema)}")

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"\n[write] {args.out / 'model.safetensors'}  ({bytes_total/1e9:.2f} GB)")
    write_safetensors(args.out / "model.safetensors", tensors_out, dtype_str="BF16")

    print(f"[write] {args.out / 'config.json'}")
    write_hf_config(model_cfg, args.out)

    print(f"[write] tokenizer files into {args.out}")
    write_tokenizer(args.out, args.vocab, args.merges)

    dt = time.perf_counter() - t0
    print(f"\n[done] {param_idx} parameters in {dt:.1f}s "
          f"({bytes_total/1e9/dt:.2f} GB/s)")
    print(f"\nload it with:")
    print(f"  from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"  m = AutoModelForCausalLM.from_pretrained('{args.out}', torch_dtype='bfloat16')")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
