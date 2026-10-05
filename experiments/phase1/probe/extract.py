"""Capture Qwen2.5-1.5B-Instruct activations for every prompt in the probe splits.

Run from the repo root, after prepare_splits.py:
    source scripts/dev-env.sh
    uv run python experiments/phase1/probe/extract.py --config fp16
    uv run python experiments/phase1/probe/extract.py --config fp16 --limit 16   # smoke test

Writes .cache/probe/<config>/{activations.npy, ids.json, manifest.json}. Activations are
(n, layers, positions, d) float16 in splits.parquet order.
"""

import argparse
import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import vllm

from hulder.capture.vllm import CaptureConfig, capture

SPLITS = Path(".cache/probe/splits.parquet")
CONFIGS = {
    "fp16": CaptureConfig(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        layers=(4, 8, 12, 16, 20, 24, 28),
        positions=(-1, -5),  # last templated token; end of user turn (<|im_end|>)
        system_prompt="You are a helpful assistant.",
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=CONFIGS, required=True)
    ap.add_argument("--limit", type=int, help="first N prompts only, for a smoke test")
    args = ap.parse_args()
    cfg = CONFIGS[args.config]

    df = pd.read_parquet(SPLITS)
    if args.limit:
        df = df.head(args.limit)
    t0 = time.time()
    cap = capture(df["text"].tolist(), cfg)
    elapsed = time.time() - t0

    bad = [i for i, toks in zip(df["id"], cap.position_tokens) if toks[1] != "<|im_end|>"]
    assert not bad, f"position -5 is not <|im_end|> for {len(bad)} prompts, e.g. {bad[:3]}"
    assert np.isfinite(cap.activations).all(), "non-finite activation"

    out = Path(".cache/probe") / (args.config if not args.limit else f"{args.config}_smoke")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "activations.npy", cap.activations)
    (out / "ids.json").write_text(json.dumps(df["id"].tolist()))
    manifest = {
        "capture_config": asdict(cfg),
        "n_prompts": len(df),
        "shape": list(cap.activations.shape),
        "last_token_examples": sorted({t[0] for t in cap.position_tokens}),
        "elapsed_s": round(elapsed, 1),
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "splits_sha256": subprocess.run(["sha256sum", str(SPLITS)], capture_output=True, text=True).stdout.split()[0],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
