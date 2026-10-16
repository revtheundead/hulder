"""One capture inside a Modal container, run as its own process by capture.py so that its log,
including the vLLM engine-core process's, lands in one file. Imports vLLM.

    python worker.py SPEC.json OUT_DIR

SPEC holds the CaptureConfig fields, the canary, the prompts, an optional base_model override,
metadata to add to every snapshot and whether to save snapshots. Writes one Snapshot per captured
layer (<layer>.npz) and worker.json to OUT_DIR.

The canary (smoke_lib.canary_prompts) runs as the engine's first batch, unless the prompts already
begin with it, so it runs once. Its hash (smoke_lib.canary_sha256) goes into every snapshot's metadata
as canary_sha256, and the snapshots hold exactly the requested prompts. If the canary's tokens exceed
one scheduler step, the worker fails before writing anything (smoke_lib.check_canary_budget).
"""

import json
import platform
import sys
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np


def main(spec_path: str, out_dir: str) -> None:
    import smoke_lib  # beside this file, in the container and the checkout
    import torch

    import hulder
    from hulder.capture import CaptureConfig, base_model_for, capture, snapshot_metadata

    spec = json.loads(Path(spec_path).read_text())
    out = Path(out_dir)
    fields = {k: tuple(v) if k in ("layers", "positions") else v for k, v in spec["config"].items()}
    cfg = CaptureConfig(**fields)
    rows, canary = spec["prompts"], spec["canary"]
    if len(canary) != cfg.batch_size:
        raise SystemExit(f"the canary must fill one batch: {len(canary)} prompts at batch size {cfg.batch_size}")
    lead = [] if [r["id"] for r in rows[: len(canary)]] == [r["id"] for r in canary] else canary
    start = time.time()
    captured = capture([r["text"] for r in lead + rows], cfg)
    seconds = time.time() - start
    # vLLM 0.30.0 budgets max_num_batched_tokens per step, less at most one draft slot for the
    # extraction's speculative config (config/speculative.py:1853).
    canary_tokens = captured.prompt_token_counts[: len(canary)]
    smoke_lib.check_canary_budget(canary_tokens, cfg.max_num_batched_tokens - 1)
    canary_sha256 = smoke_lib.canary_sha256(captured.activations[: len(canary)], cfg.layers, cfg.positions)
    activations = captured.activations[len(lead) :]
    base_model, warning = base_model_for(cfg.model, cfg.revision, cfg.quantization, override=spec["base_model"])
    if warning:
        raise SystemExit(smoke_lib.base_model_error(spec["config_name"], warning))
    saved = []
    if spec["save"]:
        for i, layer in enumerate(cfg.layers):
            hulder.Snapshot(
                ids=np.array([r["id"] for r in rows]),
                labels=np.array([r["label"] for r in rows]),
                activations=activations[:, i, 0].astype(np.float16),  # the config's only position
                metadata={
                    **snapshot_metadata(cfg, captured, base_model, version("vllm"), layer=layer),
                    **spec.get("extra_metadata", {}),  # the GPU's UUID and driver, so another GPU warns
                    "canary_sha256": canary_sha256,  # the numerical path; another hash warns when checked
                },
            ).save(out / f"{layer}.npz")
            saved.append(f"{layer}.npz")
    worker = {
        "capture_config": asdict(cfg),
        "base_model": base_model,
        "n_prompts": len(rows),
        "canary_sha256": canary_sha256,
        "canary_prompt_tokens": sum(canary_tokens),
        "gpu": captured.gpu,
        "capture_seconds": round(seconds, 1),  # engine start, load, compile and capture together
        "last_tokens": sorted({tokens[0] for tokens in captured.position_tokens}),
        "snapshots": saved,
        "packages": {p: version(p) for p in ("hulder", "vllm", "torch", "numpy")},
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
    }
    (out / "worker.json").write_text(json.dumps(worker, indent=2) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
