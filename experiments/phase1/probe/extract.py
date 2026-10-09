"""Capture Qwen2.5-1.5B-Instruct activations for every prompt in the probe splits.

Run from the repo root, after prepare_splits.py:
    source scripts/dev-env.sh
    uv run python experiments/phase1/probe/extract.py --config fp16
    uv run python experiments/phase1/probe/extract.py --config fp16 --limit 16   # smoke test
    uv run python experiments/phase1/probe/extract.py --config awq --save-layer 20 --save-position -1

Writes .cache/probe/<config>/{activations.npy, ids.json, manifest.json}. Activations are
(n, layers, positions, d) float16 in splits.parquet order. --save-layer and --save-position
write one layer and position only; the engine still extracts every configured one, so its
work matches the fp16 runs.

Every config runs all prompts in file order in batches of batch_size, so each prompt has
the same batch-mates in every run. A subset capture would change them at the edges.
"""

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from common import CAPTURE_FILES, SPLITS, run_manifest, sha256_file

from hulder.capture.hidden_states import CaptureConfig, capture

CONFIGS = {
    "fp16": CaptureConfig(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        layers=(4, 8, 12, 16, 20, 24, 28),
        positions=(-1, -5),  # last templated token; end of user turn (<|im_end|>)
        system_prompt="You are a helpful assistant.",
    ),
}
# The official AWQ checkpoint of the same model; only model, revision and quantization change.
CONFIGS["awq"] = replace(
    CONFIGS["fp16"],
    model="Qwen/Qwen2.5-1.5B-Instruct-AWQ",
    revision="3ecffa0ceb27851800f45519bab9c457a04405e1",
    quantization="awq",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=CONFIGS, required=True)
    ap.add_argument("--limit", type=int, help="first N prompts only, for a smoke test")
    ap.add_argument("--save-layer", type=int, help="write only this layer (disk rule)")
    ap.add_argument("--save-position", type=int, help="write only this position (disk rule)")
    args = ap.parse_args()
    cfg = CONFIGS[args.config]

    df = pd.read_parquet(SPLITS)
    if args.limit:
        df = df.head(args.limit)
    t0 = time.time()
    cap = capture(df["text"].tolist(), cfg)
    elapsed = time.time() - t0

    end_of_user_turn = cfg.positions.index(-5)
    bad = [i for i, toks in zip(df["id"], cap.position_tokens) if toks[end_of_user_turn] != "<|im_end|>"]
    assert not bad, f"position -5 is not <|im_end|> for {len(bad)} prompts, e.g. {bad[:3]}"
    assert np.isfinite(cap.activations).all(), "non-finite activation"

    acts, saved = cap.activations, {}
    if args.save_layer is not None:
        acts = acts[:, [cfg.layers.index(args.save_layer)]]
        saved["saved_layers"] = [args.save_layer]
    if args.save_position is not None:
        acts = acts[:, :, [cfg.positions.index(args.save_position)]]
        saved["saved_positions"] = [args.save_position]
    out = Path(".cache/probe") / (args.config if not args.limit else f"{args.config}_smoke")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "activations.npy", acts)
    (out / "ids.json").write_text(json.dumps(df["id"].tolist()))
    manifest = {
        "capture_config": asdict(cfg),
        "n_prompts": len(df),
        "shape": list(acts.shape),
        **saved,
        "files_sha256": {name: sha256_file(out / name) for name in CAPTURE_FILES},
        "last_token_examples": sorted({t[0] for t in cap.position_tokens}),
        "elapsed_s": round(elapsed, 1),
        "gpu": torch.cuda.get_device_name(0),
        **run_manifest(seed=cfg.seed, inputs=(SPLITS,)),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
