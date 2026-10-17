"""Report a canary_repeat run (smoke_lib.EXPERIMENTS["canary_repeat"]) batch by batch.

    uv run python experiments/modal/canary_repeat_report.py .cache/modal-runs/<run id> .cache/modal-runs/<pilot run id>

Writes canary_repeat_report.json next to the run's manifest.json and prints it.

Each container's canary_check is compared, batch by batch, with the majority bf16 path (recorded
cluster 2.0), with the pilot's canary_check and with the other container. Batch 1 is the canary,
compared by its hash since its rows aren't saved, and batches 2 to 8 bit for bit. A failed or
timed-out container is a stop. A batch that differs is a finding, since that is what the run
measures. Each container's cuBLASLt algorithm set is matched against the recorded traces of the
majority path, and the containers' token hashes against each other.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pilot_report as pilot
import smoke_lib as lib

import hulder

# The majority bf16 path, 16 captures on 2026-10-13 and 2026-10-15 (docs/results/phase0/modal_smoke.md).
MAJORITY = "2.0"
N_CANARY = 8


def _rows(capture: Path, skip: int = 0) -> tuple[list[str], dict[int, np.ndarray]]:
    snapshots = {layer: hulder.Snapshot.load(capture / f"{layer}.npz") for layer in lib.LAYERS}
    ids = [str(i) for i in snapshots[lib.LAYERS[0]].ids[skip:]]
    return ids, {layer: s.activations[skip:] for layer, s in snapshots.items()}


def _trace(capture: Path) -> set[str] | None:
    traces = [capture / name for name in ("cublas_fingerprint.txt", "cublas_summary.txt")]
    found = next((p for p in traces if p.exists()), None)
    return lib.matmul_signatures(found.read_text(errors="replace")) if found else None


def summarise(
    run_dir: Path,
    pilot_dir: Path,
    clusters_path: Path = pilot.CLUSTERS,
    runs_dir: Path = pilot.RUNS,
    reference_cluster: str = MAJORITY,
) -> dict:
    run = json.loads((run_dir / "manifest.json").read_text())
    recorded = pilot.recorded_hashes(clusters_path, runs_dir)
    [(reference_hash, reference)] = [(h, e) for h, e in recorded.items() if e["cluster"] == reference_cluster]
    reference_ids, reference_rows = _rows(runs_dir / reference["members"][0], skip=N_CANARY)
    pilot_ids, pilot_rows = _rows(pilot_dir / "canary_check")
    traced = {m: t for m in reference["members"] if (t := _trace(runs_dir / m)) is not None}

    stops = [f"the run stopped: {run['stopped']}"] if run.get("stopped") else []
    findings: list[str] = []
    configs, rows_by_label = {}, {}
    for label, entry in run["configs"].items():
        manifest = json.loads((run_dir / label / "manifest.json").read_text())
        if failed := pilot.failure_reason(run_dir, label, manifest):
            stops.append(failed)
        if "worker" not in manifest:
            continue
        worker = manifest["worker"]
        container = f"{run['run_id']}#{entry.get('container', 0)}"
        ids, rows = _rows(run_dir / label)
        if ids != reference_ids or ids != pilot_ids:
            raise ValueError(f"{label} holds other prompts than the majority path's rows 8 on or the pilot's")
        rows_by_label[label] = rows
        canary = {
            "batch": 1,
            "rows": N_CANARY,
            "identical": worker["canary_sha256"] == reference_hash,
            "by": "canary hash",
        }
        vs_majority = [canary, *lib.compare_batches(rows, reference_rows, first_batch=2)]
        if worker["canary_sha256"] not in recorded:
            findings.append(pilot.unknown_path(label, worker, container))
        differing = [b["batch"] for b in vs_majority if not b["identical"]]
        if differing:
            findings.append(f"{label}: batches {differing} differ from the majority path (cluster {reference_cluster})")
        signatures = _trace(run_dir / label)
        configs[label] = {
            "container": container,
            "gpu_uuid": worker.get("gpu_uuid"),
            "gpu_driver": worker.get("gpu_driver"),
            "modal": worker.get("modal"),
            "nvidia_cublas": worker.get("nvidia_packages", {}).get("nvidia-cublas"),
            "engine_seconds": worker.get("engine_seconds"),
            "canary_sha256": worker["canary_sha256"],
            "prompt_token_ids_sha256": worker.get("prompt_token_ids_sha256"),
            "canary_token_ids_sha256": worker.get("canary_token_ids_sha256"),
            "vs_majority": vs_majority,
            "vs_pilot": lib.compare_batches(rows, pilot_rows, first_batch=2),
            "cublas_same_as": sorted(m for m, t in traced.items() if t == signatures) if signatures else None,
            "cublas_signatures": len(signatures) if signatures else None,
        }

    labels = list(rows_by_label)
    between = (
        lib.compare_batches(rows_by_label[labels[0]], rows_by_label[labels[1]], first_batch=2)
        if len(labels) == 2
        else None
    )
    tokens = {(c["prompt_token_ids_sha256"], c["canary_token_ids_sha256"]) for c in configs.values()}
    keep = ("run_id", "commit", "estimated_usd", "stopped")
    return {
        **{k: run.get(k) for k in keep},
        "stop_reasons": stops,
        "findings": findings,
        "reference": {"cluster": reference_cluster, "member": reference["members"][0], "traced": sorted(traced)},
        "configs": configs,
        "between_containers": between,
        # None when no container ran or a hash wasn't recorded, rather than agreement by absence.
        "token_hashes_agree": len(tokens) == 1 if configs and None not in {x for t in tokens for x in t} else None,
    }


if __name__ == "__main__":
    run_dir = Path(sys.argv[1])
    summary = summarise(run_dir, Path(sys.argv[2]))
    (run_dir / "canary_repeat_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
