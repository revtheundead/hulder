"""One capture inside a Modal container, run as its own process by capture.py so that its log,
including the vLLM engine-core process's, lands in one file. Imports vLLM.

    python worker.py SPEC.json OUT_DIR

SPEC holds the CaptureConfig fields, the prompts, an optional base_model override, metadata to add
to every snapshot and whether to save snapshots. Writes one Snapshot per captured layer
(<layer>.npz) and worker.json to OUT_DIR.
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
    import torch

    import hulder
    from hulder.capture import CaptureConfig, base_model_for, capture, snapshot_metadata

    spec = json.loads(Path(spec_path).read_text())
    out = Path(out_dir)
    fields = {k: tuple(v) if k in ("layers", "positions") else v for k, v in spec["config"].items()}
    cfg = CaptureConfig(**fields)
    rows = spec["prompts"]
    start = time.time()
    captured = capture([r["text"] for r in rows], cfg)
    seconds = time.time() - start
    base_model, warning = base_model_for(cfg.model, cfg.revision, cfg.quantization, override=spec.get("base_model"))
    if warning:
        raise SystemExit(warning)
    saved = []
    if spec["save"]:
        for i, layer in enumerate(cfg.layers):
            hulder.Snapshot(
                ids=np.array([r["id"] for r in rows]),
                labels=np.array([r["label"] for r in rows]),
                activations=captured.activations[:, i, 0].astype(np.float16),  # the config's only position
                metadata={
                    **snapshot_metadata(cfg, captured, base_model, version("vllm"), layer=layer),
                    **spec.get("extra_metadata", {}),  # the GPU's UUID and driver, so another GPU warns
                },
            ).save(out / f"{layer}.npz")
            saved.append(f"{layer}.npz")
    worker = {
        "capture_config": asdict(cfg),
        "base_model": base_model,
        "n_prompts": len(rows),
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
