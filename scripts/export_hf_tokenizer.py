#!/usr/bin/env python3
"""scripts/export_hf_tokenizer.py

Export a HuggingFace fast tokenizer (tokenizer.json) to the GPT-2-style
vocab.json + merges.txt that our C++ BPETokenizer loads. Works for any
byte-level BPE tokenizer (GPT-2, GPT-NeoX, OLMo-2/dolma2, Llama-3*).

  python scripts/export_hf_tokenizer.py <tokenizer.json> <out_dir>
  # -> <out_dir>/vocab.json  +  <out_dir>/merges.txt

It also PRINTS the pre-tokenizer regex + special tokens so we can confirm our
C++ tokenizer's GPT-2 regex matches (it does for GPT-2/NeoX; cl100k-style needs
a regex tweak in bpe_tokenizer.cpp).

(*Llama-3 uses a cl100k-style regex; vocab/merges still export fine but the C++
pre-tokenizer regex would need updating — this script flags that.)
"""
import json, sys, os

def main():
    if len(sys.argv) != 3:
        print("usage: export_hf_tokenizer.py <tokenizer.json> <out_dir>"); sys.exit(1)
    tok_path, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    with open(tok_path) as f:
        tok = json.load(f)

    model = tok.get("model", {})
    vocab = model.get("vocab")
    merges = model.get("merges")
    if vocab is None or merges is None:
        print("ERROR: tokenizer.json has no model.vocab / model.merges (not a BPE tokenizer?)")
        sys.exit(2)

    # vocab.json: {token_str: id}
    with open(os.path.join(out_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f, ensure_ascii=False)

    # merges.txt: "#version" header + one merge per line ("a b").
    # HF stores merges as ["a b", ...] (old) or [["a","b"], ...] (new) — handle both.
    with open(os.path.join(out_dir, "merges.txt"), "w") as f:
        f.write("#version: 0.2 - exported from tokenizer.json\n")
        for m in merges:
            f.write((m if isinstance(m, str) else " ".join(m)) + "\n")

    print(f"wrote {out_dir}/vocab.json ({len(vocab)} tokens) + merges.txt ({len(merges)} merges)")

    # Diagnostics for the C++ regex/special-token check.
    pre = tok.get("pre_tokenizer")
    print("\n--- pre_tokenizer (must be GPT-2/ByteLevel for our C++ regex) ---")
    print(json.dumps(pre, indent=2)[:1200])
    added = tok.get("added_tokens", [])
    specials = [t.get("content") for t in added if t.get("special")]
    print("\n--- special tokens ---")
    print(specials[:20])
    print("\nNOTE: ensure an EOS-like token (e.g. <|endoftext|>) exists; our C++"
          " tokenizer reads eos_id from '<|endoftext|>'. If OLMo-2's EOS differs,"
          " we add it to the C++ special-token handling.")

if __name__ == "__main__":
    main()
