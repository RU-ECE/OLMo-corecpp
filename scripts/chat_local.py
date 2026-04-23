#!/usr/bin/env python3
"""zwt local chat — demo-friendly REPL for the converted HF Llama model.

Streaming output, slash commands, Ctrl-C interrupts the in-flight
generation (not the program). No external deps beyond torch + transformers.

  python3 scripts/chat_local.py exports/owt_1B_hf
"""
from __future__ import annotations

# Silence warnings BEFORE importing torch/transformers so the env vars
# actually take effect. urllib3/LibreSSL, transformers deprecations, and
# tokenizers parallelism noise all go to /dev/null.
import os
os.environ["TRANSFORMERS_VERBOSITY"]   = "error"
os.environ["TOKENIZERS_PARALLELISM"]   = "false"
os.environ["PYTHONWARNINGS"]           = "ignore"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import argparse
import logging
import sys
import threading
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)


# ── ANSI helpers ─────────────────────────────────────────────────────────
USE_COLOR = sys.stdout.isatty()
ANSI_RE = None
import re as _re
ANSI_RE = _re.compile(r"\x1b\[[0-9;]*m")


def c(s: str, code: str) -> str:
    return f"\x1b[{code}m{s}\x1b[0m" if USE_COLOR else s


def visible_len(s: str) -> int:
    return len(ANSI_RE.sub("", s))


def dim(s):     return c(s, "2")
def bold(s):    return c(s, "1")
def cyan(s):    return c(s, "36")
def green(s):   return c(s, "32")
def yellow(s):  return c(s, "33")
def magenta(s): return c(s, "35")
def red(s):     return c(s, "31")


# ── runtime config the user can poke via /set ────────────────────────────
class Settings:
    temperature   = 0.8
    top_p         = 0.9
    top_k         = 50
    rep_penalty   = 1.2
    no_repeat_ng  = 3       # block any 3-gram from repeating verbatim
    max_new       = 200
    min_new       = 40
    seed          = 0       # 0 = no manual seed (nondeterministic)

    keys = ("temperature", "top_p", "top_k", "rep_penalty", "no_repeat_ng",
            "max_new", "min_new", "seed")

    @classmethod
    def show(cls) -> str:
        return "  ".join(f"{dim(k)}={getattr(cls, k)}" for k in cls.keys)

    @classmethod
    def set_kv(cls, key: str, val: str) -> str:
        if key not in cls.keys:
            return red(f"unknown key '{key}' — try one of: {', '.join(cls.keys)}")
        cur = getattr(cls, key)
        try:
            new = type(cur)(val) if not isinstance(cur, float) else float(val)
        except ValueError:
            return red(f"can't parse {val!r} as {type(cur).__name__}")
        setattr(cls, key, new)
        return green(f"  {key} = {new}")


# ── banner + help ────────────────────────────────────────────────────────
BOX_W = 56


def line(s: str) -> str:
    # box drawing chars + 2-space inset on each side = 6 chars of frame
    pad = max(BOX_W - visible_len(s) - 6, 0)
    return f"{dim('│')}  {s}{' ' * pad}  {dim('│')}"


def banner(model_basename: str, n_params: float, device: str,
           dtype, model_cfg: dict):
    top = dim("╭" + "─" * (BOX_W - 2) + "╮")
    bot = dim("╰" + "─" * (BOX_W - 2) + "╯")
    title = (f"{bold(magenta('zwt'))} {dim('·')} "
             f"{bold(f'{n_params/1e9:.2f}B')} params {dim('·')} "
             f"{cyan(device.upper())} {dim('·')} "
             f"{cyan(str(dtype).split('.')[-1])}")
    arch = (f"{model_cfg.get('num_hidden_layers','?')} layers"
            f" {dim('·')} d={model_cfg.get('hidden_size','?')}"
            f" {dim('·')} {model_cfg.get('num_attention_heads','?')} heads"
            f" {dim('·')} v={model_cfg.get('vocab_size','?')}")
    cmds = (f"{dim('/help')}  {dim('/show')}  {dim('/set k=v')}"
            f"  {dim('/quit')}")
    print(top)
    print(line(title))
    print(line(dim(arch)))
    print(line(""))
    print(line(cmds))
    print(bot)


HELP_LINES = [
    ("",          ""),
    ("commands",  ""),
    ("/help",     "show this"),
    ("/show",     "current sampling settings"),
    ("/set k=v",  "change a setting   e.g. /set temperature=0.6 top_p=0.85"),
    ("/reset",    "(stateless — no-op, kept for muscle memory)"),
    ("/quit",     "leave  (also: /exit  /q  blank line  Ctrl-D)"),
    ("",          ""),
    ("input",     ""),
    ("...\\",     "trailing backslash continues on the next line"),
    ("Ctrl-C",    "while generating: stop output, return to prompt"),
    ("",          ""),
    ("settings",  ""),
    ("temperature",  "sampling temperature   (0 = greedy)"),
    ("top_p",        "nucleus mass            (1.0 disables)"),
    ("top_k",        "top-k cutoff            (0 disables)"),
    ("rep_penalty",  "repetition penalty      (1.0 disables)"),
    ("no_repeat_ng", "ban any n-gram repeat   (0 disables)"),
    ("max_new",      "max tokens generated"),
    ("min_new",      "floor before EOS can fire"),
    ("seed",         "RNG seed                (0 = random)"),
    ("",          ""),
    ("tip",       "this is a base LM. story-style prompts work best:"),
    ("",          dim("  Once upon a time")),
    ("",          dim("  The lighthouse keeper noticed")),
    ("",          dim("  Dr. Chen opened the journal and read:")),
    ("",          ""),
]


def print_help():
    for k, v in HELP_LINES:
        if not k and not v:
            print()
            continue
        if not v:
            print("  " + bold(k))
            continue
        if not k:
            print(f"      {v}")
            continue
        print(f"  {cyan(k):<22} {dim('·')} {v}")


# ── interruptible generation ─────────────────────────────────────────────
class StopFlag(StoppingCriteria):
    """Cooperatively stop generate() when the main thread sets `.stop`."""
    def __init__(self):
        self.stop = False

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self.stop


def generate_streaming(mdl, tok, prompt: str, device, settings: Settings) -> tuple[int, float]:
    """Stream tokens to stdout. Returns (n_new_tokens, seconds)."""
    enc = tok(prompt, return_tensors="pt")
    ids = enc.input_ids.to(device)
    attn = enc.attention_mask.to(device)

    streamer = TextIteratorStreamer(
        tok, skip_prompt=True, skip_special_tokens=True,
        decode_kwargs={"clean_up_tokenization_spaces": True},
    )
    flag = StopFlag()
    if settings.seed:
        torch.manual_seed(settings.seed)

    gen_kwargs = dict(
        input_ids=ids,
        attention_mask=attn,
        max_new_tokens=settings.max_new,
        min_new_tokens=settings.min_new,
        do_sample=settings.temperature > 0,
        temperature=settings.temperature,
        top_p=settings.top_p,
        top_k=settings.top_k if settings.top_k > 0 else None,
        repetition_penalty=settings.rep_penalty,
        no_repeat_ngram_size=settings.no_repeat_ng if settings.no_repeat_ng > 0 else None,
        pad_token_id=tok.pad_token_id,
        streamer=streamer,
        stopping_criteria=StoppingCriteriaList([flag]),
    )
    holder: dict = {}

    def run():
        try:
            holder["out"] = mdl.generate(**gen_kwargs)
        except Exception as exc:
            holder["err"] = exc

    thread = threading.Thread(target=run, daemon=True)
    t0 = time.perf_counter()
    thread.start()

    # Cheap degenerate-loop guard: if the same short line repeats >=4 times
    # in a row (typical undertrained-model failure mode where it hammers a
    # single BBPE byte token forever), set the stop flag.
    # Filter out U+FFFD (replacement char) before printing — that's the `��`
    # the model emits when sampling a lone UTF-8 continuation byte.
    buf = []
    last_line = ""
    repeat_count = 0
    try:
        for piece in streamer:
            piece = piece.replace("�", "")
            if not piece:
                continue
            sys.stdout.write(piece)
            sys.stdout.flush()
            buf.append(piece)
            if "\n" in piece:
                joined = "".join(buf).split("\n")
                for ln in joined[:-1]:
                    s = ln.strip()
                    if s and s == last_line:
                        repeat_count += 1
                        if repeat_count >= 4:
                            flag.stop = True
                            sys.stdout.write(yellow("  [loop-stopped]"))
                            sys.stdout.flush()
                            break
                    else:
                        last_line = s
                        repeat_count = 1 if s else 0
                buf = [joined[-1]]
    except KeyboardInterrupt:
        flag.stop = True
        sys.stdout.write(yellow("  [stopped]"))
        sys.stdout.flush()

    thread.join()
    dt = time.perf_counter() - t0
    if "err" in holder:
        print(red(f"\n[gen error] {holder['err']}"))
        return 0, dt
    out = holder.get("out")
    n_new = (out.shape[1] - ids.shape[1]) if out is not None else 0
    return n_new, dt


# ── input plumbing ───────────────────────────────────────────────────────
def read_prompt(prompt_str: str) -> str | None:
    """Read a possibly multi-line prompt. Lines ending in \\ continue."""
    try:
        first = input(prompt_str)
    except (EOFError, KeyboardInterrupt):
        return None
    if not first.strip():
        return ""
    parts = [first]
    while parts[-1].endswith("\\"):
        parts[-1] = parts[-1][:-1]   # drop the trailing slash
        try:
            cont = input(dim("… "))
        except (EOFError, KeyboardInterrupt):
            break
        parts.append(cont)
    return "\n".join(parts)


# ── slash dispatcher ─────────────────────────────────────────────────────
def handle_slash(line: str) -> bool:
    """Returns True if we should keep going, False to exit."""
    cmd, *rest = line[1:].split(maxsplit=1)
    cmd = cmd.lower()
    args = rest[0] if rest else ""
    if cmd in ("quit", "exit", "q"):
        return False
    if cmd == "help":
        print_help()
    elif cmd == "show":
        for k in Settings.keys:
            print(f"  {cyan(k):<22} {dim('·')} {getattr(Settings, k)}")
    elif cmd == "reset":
        print(dim("  (stateless — nothing to reset)"))
    elif cmd == "set":
        if not args:
            print(red("  usage: /set k=v [k=v ...]"))
        else:
            for kv in args.split():
                if "=" not in kv:
                    print(red(f"  ignoring '{kv}' — expected k=v"))
                    continue
                k, v = kv.split("=", 1)
                print(Settings.set_kv(k.strip(), v.strip()))
    else:
        print(red(f"  unknown command '/{cmd}' — /help for the list"))
    return True


# ── load + run ───────────────────────────────────────────────────────────
def pick_device(arg: str) -> tuple[str, "torch.dtype"]:
    if arg != "auto":
        if arg == "cuda":
            return "cuda", torch.bfloat16
        if arg == "mps":
            return "mps", torch.float16
        return "cpu", torch.float32
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = ap.parse_args()

    device, dtype = pick_device(args.device)
    ckpt = Path(args.ckpt_dir).expanduser().resolve()

    sys.stdout.write(dim(f"  loading {ckpt.name} ... "))
    sys.stdout.flush()
    t0 = time.perf_counter()
    # use_fast=True converts vocab.json+merges.txt to a fast tokenizer on the
    # fly. Fast GPT-2 tokenizer handles byte-level decoding correctly across
    # token boundaries, which kills most `��` fragments the slow one emits.
    try:
        tok = AutoTokenizer.from_pretrained(ckpt, use_fast=True, local_files_only=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(ckpt, use_fast=False, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    mdl = AutoModelForCausalLM.from_pretrained(
        ckpt, dtype=dtype, local_files_only=True,
    ).to(device).eval()
    n_params = sum(p.numel() for p in mdl.parameters())
    load_dt = time.perf_counter() - t0
    sys.stdout.write(dim(f"{load_dt:.1f}s\n\n"))

    model_cfg = mdl.config.to_dict() if hasattr(mdl, "config") else {}
    banner(ckpt.name, n_params, device, dtype, model_cfg)
    print()

    while True:
        prompt = read_prompt(cyan("» "))
        if prompt is None:
            print()
            break
        if not prompt.strip():
            continue
        if prompt.startswith("/"):
            if not handle_slash(prompt.strip()):
                break
            continue

        n_new, dt = generate_streaming(mdl, tok, prompt, device, Settings)
        rate = (n_new / dt) if dt > 0 else 0
        print()  # newline after stream
        print(dim(f"  ─ {n_new} tok · {dt:.1f}s · {rate:.0f} tok/s"))
        print()

    print(dim("\n  bye.\n"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
