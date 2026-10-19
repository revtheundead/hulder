"""One capture inside a Modal container, run as its own process by capture.py so that its log,
including the vLLM engine-core process's, lands in one file. Imports vLLM.

    python worker.py SPEC.json OUT_DIR

SPEC holds the CaptureConfig fields, the canary, the prompts, an optional base_model override,
metadata to add to every snapshot and whether to save snapshots. Writes one Snapshot per captured
layer (<layer>.npz), or per layer and position (<layer>_<position>.npz) when the config captures
more than one position, and worker.json to OUT_DIR.

The canary runs as the engine's first batch (smoke_lib.capture_order), its hash goes into every
snapshot's metadata as canary_sha256, and the snapshots hold exactly the requested prompts
(smoke_lib.split_canary). If the canary's tokens exceed one scheduler step, the worker fails before
writing anything.
"""

import json
import os
import platform
import sys
import time
from dataclasses import asdict
from importlib.metadata import distributions, version
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
    ids = [r["id"] for r in rows]
    order = smoke_lib.capture_order(ids, [r["id"] for r in canary], cfg.batch_size)
    text = {r["id"]: r["text"] for r in canary + rows}
    start = time.time()
    captured = capture([text[i] for i in order], cfg)
    seconds = time.time() - start
    # vLLM 0.30.0 budgets max_num_batched_tokens per step, less at most one draft slot for the
    # extraction's speculative config (config/speculative.py:1853).
    split = smoke_lib.split_canary(
        order,
        ids,
        [r["id"] for r in canary],
        captured.activations,
        captured.prompt_token_counts,
        cfg.layers,
        cfg.positions,
        cfg.max_num_batched_tokens - 1,
    )
    base_model, warning = base_model_for(cfg.model, cfg.revision, cfg.quantization, override=spec["base_model"])
    if warning:
        raise SystemExit(smoke_lib.base_model_error(spec["config_name"], warning))
    # The engine's own tokens: the requested prompts', and the canary's apart.
    first = len(order) - len(ids)
    requested_token_hash = smoke_lib.token_ids_sha256(captured.prompt_token_ids[first:])
    saved = []
    if spec["save"]:
        for i, layer in enumerate(cfg.layers):
            for j, position in enumerate(cfg.positions):
                # One position keeps the name every earlier experiment used.
                name = f"{layer}.npz" if len(cfg.positions) == 1 else f"{layer}_{position}.npz"
                hulder.Snapshot(
                    ids=np.array([r["id"] for r in rows]),
                    labels=np.array([r["label"] for r in rows]),
                    activations=split.activations[:, i, j].astype(np.float16),
                    metadata={
                        **snapshot_metadata(cfg, captured, base_model, version("vllm"), layer=layer, position=position),
                        **spec.get("extra_metadata", {}),  # the GPU's UUID and driver, so another GPU warns
                        "canary_sha256": split.canary_sha256,  # the numerical path; another hash warns when checked
                        "prompt_token_ids_sha256": requested_token_hash,  # the tokens these rows were fed
                    },
                ).save(out / name)
                saved.append(name)
        # The rows' own token ids, so the hash of any subset of them can be recomputed.
        smoke_lib.save_token_ids(out / smoke_lib.TOKEN_IDS_FILE, ids, captured.prompt_token_ids[first:])
    # The requested prompts' tokens at each position, in request order, without the canary's.
    requested_tokens = captured.position_tokens[first:]
    worker = {
        "capture_config": asdict(cfg),
        "base_model": base_model,
        "n_prompts": len(rows),
        "prompt_ids_sha256": hulder.prompt_ids_sha256(ids),
        "canary_sha256": split.canary_sha256,
        "canary_prompt_tokens": split.canary_prompt_tokens,
        "prompt_token_ids_sha256": requested_token_hash,
        "canary_token_ids_sha256": smoke_lib.token_ids_sha256(captured.prompt_token_ids[: len(canary)]),
        "token_ids_file": smoke_lib.TOKEN_IDS_FILE if spec["save"] else None,
        "gpu": captured.gpu,
        **spec.get("extra_metadata", {}),  # gpu_uuid and gpu_driver
        "capture_seconds": round(seconds, 1),  # engine start, load, compile and capture together
        "engine_seconds": captured.engine_seconds,  # constructing the LLM
        "batch_seconds": captured.batch_seconds,  # each batch in capture order, the canary's first
        "last_tokens": sorted({tokens[0] for tokens in captured.position_tokens}),
        "position_tokens": {str(p): [t[j] for t in requested_tokens] for j, p in enumerate(cfg.positions)},
        "snapshots": sorted(saved),
        "packages": {p: version(p) for p in ("hulder", "vllm", "torch", "numpy")},
        # The CUDA wheels in this environment, cuBLAS among them. The engine-core process loads its
        # libraries from the same environment.
        "nvidia_packages": dict(
            sorted(
                (d.metadata["Name"], d.version)
                for d in distributions()
                if (d.metadata["Name"] or "").startswith("nvidia-")
            )
        ),
        # Modal's identifiers for the image and container (None outside Modal). Not a content digest.
        "modal": {k: os.environ.get(f"MODAL_{k.upper()}") for k in ("image_id", "task_id", "region")},
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
    }
    (out / "worker.json").write_text(json.dumps(worker, indent=2) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
