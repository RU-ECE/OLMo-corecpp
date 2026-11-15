"""hf_sft_loss.py <hf_model_dir> <forge-sft dir> <sft.jsonl> <seq_len> [n_windows]

Ground truth for the llm-cpp training loss: the same windows, the same assistant-only
mask, scored by Hugging Face transformers. Prints the main next-token loss overall, on
the turn-end labels, and on everything else, then greedily continues one Forge task.

Format (env TEMPLATE, default chatml): chatml = <|im_start|>role ... <|im_end|> (the base
model's demo format); tulu = the model's own chat template (OLMo-2-Instruct), turns ending
in <|endoftext|>. Generation stops at the format's turn end.
"""
import json, os, sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path, d, sft, S = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
n_win = int(sys.argv[5]) if len(sys.argv) > 5 else 12
DEV = os.environ.get("DEVICE", "cuda")
L = int(os.environ.get("MAXLEN", S))  # score only the first L tokens of each window (CPU spot check)
GEN = int(os.environ.get("GEN", 160))
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(DEV).eval()
t = np.load(f"{d}/sft_tokens.npy").astype(np.int64)
m = np.load(f"{d}/sft_mask.npy").astype(np.int64)
TEMPLATE = os.environ.get("TEMPLATE", "chatml")
END = tok.convert_tokens_to_ids("<|im_end|>") if TEMPLATE == "chatml" else tok.eos_token_id
IM_END = END  # turn-end token of this format

tot = {"all": [0.0, 0], "turn_end": [0.0, 0], "other": [0.0, 0]}
win_means = []  # llm-cpp's per-step loss is the mean over windows of each window's token-mean
with torch.no_grad():
    for k in range(min(n_win, (len(t) - 1) // S)):
        x = torch.tensor(t[k * S: k * S + L]).to(DEV)[None]
        y = torch.tensor(t[k * S + 1: k * S + L + 1]).to(DEV)
        keep = torch.tensor(m[k * S + 1: k * S + L + 1]).to(DEV).bool()
        logits = model(x).logits[0].float()
        ce = torch.nn.functional.cross_entropy(logits, y, reduction="none")
        for name, sel in (("all", keep), ("turn_end", keep & (y == IM_END)), ("other", keep & (y != IM_END))):
            tot[name][0] += ce[sel].sum().item(); tot[name][1] += int(sel.sum())
        if keep.any():
            win_means.append(ce[keep].mean().item())
for name, (s, n) in tot.items():
    print(f"{path}: main loss on {name} labels = {s / max(n, 1):.4f} over {n} tokens ({n_win} windows x {L})")
print(f"{path}: mean of per-window main loss = {sum(win_means) / len(win_means):.4f} over {len(win_means)} windows")

if GEN <= 0:  # loss only
    sys.exit(0)
conv = json.loads(open(sft).readline())["messages"]
if TEMPLATE == "chatml":
    prompt = "".join(f"<|im_start|>{mm['role']}\n{mm['content']}<|im_end|>\n" for mm in conv[:2]) + "<|im_start|>assistant\n"
else:
    prompt = tok.apply_chat_template(conv[:2], tokenize=False, add_generation_prompt=True)
ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(DEV)
out = model.generate(ids, max_new_tokens=GEN, do_sample=False, eos_token_id=END)
stopped = int(out[0, -1]) == END
print(f"stopped at the turn end: {stopped} ({out.shape[1] - ids.shape[1]} new tokens)")
print("---- greedy continuation of Forge task 1 ----")
print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False))
