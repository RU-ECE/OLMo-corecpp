#!/usr/bin/env python3
"""zwt 1B — guided demo.

Walks through a curated set of scenes that each:
  1. Show the sampling settings about to be used
  2. Echo the prompt
  3. Stream the model's continuation live
  4. Pause for audience reaction (press Enter)

At the end drops into the same interactive REPL as chat_local.py so the
demo can take audience prompts.

  python3 scripts/chat_demo.py exports/owt_1B_hf
"""
from __future__ import annotations

# Set env vars BEFORE chat_local imports torch/transformers, so the
# warning-suppression in chat_local actually takes effect when run via
# `python3 -m` or direct script invocation.
import os
os.environ["TRANSFORMERS_VERBOSITY"]   = "error"
os.environ["TOKENIZERS_PARALLELISM"]   = "false"
os.environ["PYTHONWARNINGS"]           = "ignore"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Make sibling-script imports work no matter where this is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chat_local import (  # noqa: E402
    Settings,
    banner,
    bold,
    cyan,
    dim,
    generate_streaming,
    green,
    handle_slash,
    magenta,
    pick_device,
    print_help,
    read_prompt,
    yellow,
)
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


# ── scenes ───────────────────────────────────────────────────────────────
@dataclass
class Scene:
    title: str
    blurb: str                       # one-line context for the audience
    prompt: str
    settings: dict = field(default_factory=dict)


SCENES: list[Scene] = [
    Scene(
        title="news wire",
        blurb="OpenWebText is ~30% news. Default settings, AP-wire voice.",
        prompt="WASHINGTON — Federal regulators announced on Tuesday that",
        settings=dict(temperature=0.8, top_p=0.9, rep_penalty=1.15,
                      no_repeat_ng=3, max_new=180, min_new=60, top_k=50),
    ),
    Scene(
        title="encyclopedia",
        blurb="Same model, slightly tighter sampling — Wikipedia voice.",
        prompt="The Krebs cycle is a series of chemical reactions used by",
        settings=dict(temperature=0.6, top_p=0.9, rep_penalty=1.1,
                      no_repeat_ng=3, max_new=160, min_new=60, top_k=40),
    ),
    Scene(
        title="story open",
        blurb="Bumping temperature; OWT has some fiction so it cooperates.",
        prompt="The lighthouse keeper noticed something strange in the water that morning.",
        settings=dict(temperature=0.9, top_p=0.92, rep_penalty=1.15,
                      no_repeat_ng=3, max_new=180, min_new=80, top_k=60),
    ),
    Scene(
        title="same prompt — wild",
        blurb="Same prompt, temperature=1.1. Watch how the knob changes the world.",
        prompt="The lighthouse keeper noticed something strange in the water that morning.",
        settings=dict(temperature=1.1, top_p=0.95, rep_penalty=1.05,
                      no_repeat_ng=0, max_new=180, min_new=80, top_k=80),
    ),
    Scene(
        title="code mode",
        blurb="Anti-repetition off (code repeats by nature) + lower temperature.",
        prompt="def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
        settings=dict(temperature=0.5, top_p=0.95, rep_penalty=1.05,
                      no_repeat_ng=0, max_new=120, min_new=40, top_k=40),
    ),
    Scene(
        title="quote / interview",
        blurb="OWT trained on a lot of quoted dialogue inside articles.",
        prompt="\"I never expected this to happen,\" she said. \"When the",
        settings=dict(temperature=0.85, top_p=0.9, rep_penalty=1.15,
                      no_repeat_ng=3, max_new=180, min_new=60, top_k=50),
    ),
    Scene(
        title="numbered list",
        blurb="Web text loves listicles. Model defaults to that genre easily.",
        prompt="Here are five things every new programmer should know:\n1.",
        settings=dict(temperature=0.7, top_p=0.9, rep_penalty=1.2,
                      no_repeat_ng=4, max_new=200, min_new=100, top_k=50),
    ),
]


# ── presentation helpers ────────────────────────────────────────────────
def hr():
    print(dim("─" * 60))


def apply_settings(s: dict):
    for k, v in s.items():
        setattr(Settings, k, v)


def settings_one_liner() -> str:
    return "  ".join(f"{dim(k)}={getattr(Settings, k)}"
                     for k in ("temperature", "top_p", "rep_penalty",
                               "no_repeat_ng", "max_new"))


def wait(prompt_text: str = "press Enter") -> bool:
    """Pause for the audience. Returns False if user wants to abort."""
    try:
        cmd = input(dim(f"\n  ({prompt_text})  ")).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    if cmd in ("q", "quit", "exit", "skip"):
        return False
    return True


def play_scene(idx: int, total: int, scene: Scene, mdl, tok, device) -> bool:
    print()
    hr()
    print(f"  {dim(f'scene {idx}/{total}')}    {bold(magenta(scene.title))}")
    print(f"  {dim(scene.blurb)}")
    apply_settings(scene.settings)
    print(f"  {settings_one_liner()}")
    hr()
    print()
    print(f"  {green('»')} {scene.prompt}")
    print()
    n_new, dt = generate_streaming(mdl, tok, scene.prompt, device, Settings)
    rate = (n_new / dt) if dt > 0 else 0
    print()
    print(dim(f"  ─ {n_new} tok · {dt:.1f}s · {rate:.0f} tok/s"))
    return wait("Enter for next scene · type 'skip' to jump to free chat")


# ── free-chat REPL (mirrors chat_local.main, sharing the loaded model) ─
def free_chat(mdl, tok, device):
    hr()
    print(f"  {bold(green('free chat'))} {dim('— audience prompts welcome')}")
    print(f"  {dim('/help for commands · /quit when done')}")
    hr()
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
        print()
        print(dim(f"  ─ {n_new} tok · {dt:.1f}s · {rate:.0f} tok/s"))
        print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--scenes", default="all",
                    help="comma-separated indices (1-based) or 'all'")
    args = ap.parse_args()

    device, dtype = pick_device(args.device)
    ckpt = Path(args.ckpt_dir).expanduser().resolve()

    sys.stdout.write(dim(f"  loading {ckpt.name} ... "))
    sys.stdout.flush()
    t0 = time.perf_counter()
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
    sys.stdout.write(dim(f"{(time.perf_counter()-t0):.1f}s\n\n"))

    model_cfg = mdl.config.to_dict()
    banner(ckpt.name, n_params, device, dtype, model_cfg)
    print()
    print(f"  {bold('guided demo')} {dim('— curated scenes, then free chat')}")
    print(f"  {dim(f'{len(SCENES)} scenes queued · press Enter between · ')}"
          f"{dim('type ')}{cyan('skip')}{dim(' anytime to jump to free chat')}")
    if not wait("Enter to begin"):
        free_chat(mdl, tok, device)
        return 0

    if args.scenes == "all":
        indices = list(range(len(SCENES)))
    else:
        indices = [int(i) - 1 for i in args.scenes.split(",") if i.strip()]
        indices = [i for i in indices if 0 <= i < len(SCENES)]

    for n, i in enumerate(indices, start=1):
        if not play_scene(n, len(indices), SCENES[i], mdl, tok, device):
            print(dim(f"\n  jumping to free chat...\n"))
            break
    else:
        print()
        print(dim(f"  all {len(indices)} scenes done. switching to free chat.\n"))

    free_chat(mdl, tok, device)
    print(dim("\n  bye.\n"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
