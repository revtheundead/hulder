"""Report a pilot run (smoke_lib.EXPERIMENTS["pilot"]) against its stop rules.

    uv run python experiments/modal/pilot_report.py .cache/modal-runs/<run id>

Writes pilot_report.json next to the run's manifest.json and prints it.

Stop reasons: the run stopped, a config failed or timed out, the canary budget guard fired, or
canary_check hashes like a recorded cluster while its 56 rows differ from that cluster's. Any of
them means stop, report and wait.

Findings are reported without stopping. They cover a canary hash no recorded capture has, which is
an unknown path and is given with the GPU UUID, driver and container, and pilot prompts without
<|im_end|> at position -9.

The pilot's timing leaves out the canary's batch, which is reported apart, and is projected with
smoke_lib.timing_estimate from the slower of the two engines' startups. The band covers noise
between batches in one engine only, not between containers or GPUs.

The recorded clusters are docs/results/phase0/modal/bf16_clusters.json, and their snapshots are
read from .cache/modal-runs.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import smoke_lib as lib

import hulder

CLUSTERS = lib.ROOT / "docs" / "results" / "phase0" / "modal" / "bf16_clusters.json"
RUNS = lib.ROOT / ".cache" / "modal-runs"
LABELS = ("canary_check", "pilot")
IM_END = "<|im_end|>"
GUARD = "exceed the scheduler's per-step budget"  # smoke_lib.check_canary_budget's message


def recorded_hashes(clusters_path: Path = CLUSTERS, runs_dir: Path = RUNS) -> dict[str, dict]:
    """canary hash -> the recorded cluster ("<group>.<cluster>") with that hash and its members,
    hashed from each member's first batch. Members of one cluster must hash alike, and no two
    clusters may share a hash, since a hash would then name either."""
    found: dict[str, dict] = {}
    for g, group in enumerate(json.loads(clusters_path.read_text())):
        for c, cluster in enumerate(group["clusters"]):
            members = [m["name"] for m in cluster]
            hashes = {lib.recorded_canary_hash(runs_dir / name) for name in members}
            if len(hashes) != 1:
                raise ValueError(f"recorded cluster {g}.{c} holds {len(hashes)} canary hashes")
            hash_ = hashes.pop()
            if hash_ in found:
                raise ValueError(f"recorded clusters {found[hash_]['cluster']} and {g}.{c} share a canary hash")
            found[hash_] = {"cluster": f"{g}.{c}", "members": members}
    return found


def match_canary(canary_sha256: str, recorded: dict[str, dict]) -> dict | str:
    """The recorded cluster with this canary hash, or "unknown path"."""
    return recorded.get(canary_sha256, "unknown path")


def compare_rows(rows: dict[int, np.ndarray], reference: dict[int, np.ndarray]) -> dict:
    """Whether each layer's rows equal the reference's bit for bit, counted per row."""
    layers, identical = {}, True
    for layer in lib.LAYERS:
        a, b = rows[layer], reference[layer]
        if a.shape != b.shape:
            raise ValueError(f"layer {layer}: rows of shape {a.shape} against {b.shape}")
        same = int((a.view(np.uint16) == b.view(np.uint16)).all(axis=1).sum())  # bits, so -0.0 != 0.0
        layers[str(layer)] = f"{same}/{len(a)} identical"
        identical = identical and same == len(a)
    return {"status": "bit-identical" if identical else "differs", "layers": layers}


def position_findings(position_tokens: dict[str, list[str]], ids: list[str]) -> dict:
    """Per position, how many distinct tokens and the commonest. At -9 also every prompt whose token
    isn't <|im_end|>, which Qwen3's template with thinking off should always put there."""
    out = {}
    for position, tokens in position_tokens.items():
        counts = Counter(tokens)
        out[position] = {"distinct": len(counts), "commonest": counts.most_common(5)}
    if "-9" in position_tokens:
        missing = [{"id": i, "token": t} for i, t in zip(ids, position_tokens["-9"], strict=True) if t != IM_END]
        out["-9"].update(im_end_on_every_prompt=not missing, not_im_end=missing)
    return out


def summarise(run_dir: Path, clusters_path: Path = CLUSTERS, runs_dir: Path = RUNS) -> dict:
    run = json.loads((run_dir / "manifest.json").read_text())
    recorded = recorded_hashes(clusters_path, runs_dir)
    stops = [f"the run stopped: {run['stopped']}"] if run.get("stopped") else []
    findings: list[str] = []
    configs, workers = {}, {}
    for label in LABELS:
        path = run_dir / label / "manifest.json"
        if not path.exists():
            stops.append(f"{label} did not run")
            continue
        manifest = json.loads(path.read_text())
        if manifest["returncode"] != 0:
            log = (run_dir / label / "capture.log").read_text(errors="replace")
            why = "the canary budget guard fired" if GUARD in log else manifest.get("error", "see capture.log")
            stops.append(f"{label} failed (returncode {manifest['returncode']}): {why}")
        if "worker" not in manifest:
            continue
        worker = workers[label] = manifest["worker"]
        container = f"{run['run_id']}#{run['configs'].get(label, {}).get('container', 0)}"
        match = match_canary(worker["canary_sha256"], recorded)
        configs[label] = {
            "returncode": manifest["returncode"],
            "seconds": manifest.get("seconds"),
            "engine_seconds": worker.get("engine_seconds"),
            "canary_sha256": worker["canary_sha256"],
            "canary_match": match,
            "gpu_uuid": worker.get("gpu_uuid"),
            "gpu_driver": worker.get("gpu_driver"),
            "container": container,
        }
        if match == "unknown path":
            findings.append(
                f"{label}: unknown path, canary hash {worker['canary_sha256'][:12]}… on {worker.get('gpu_uuid')}, "
                f"driver {worker.get('gpu_driver')}, container {container}"
            )

    rows = {"status": "not applicable"}
    check = configs.get("canary_check")
    if check and check["canary_match"] != "unknown path":
        against = check["canary_match"]["members"][0]
        mine = {layer: hulder.Snapshot.load(run_dir / "canary_check" / f"{layer}.npz") for layer in lib.LAYERS}
        theirs = {layer: hulder.Snapshot.load(runs_dir / against / f"{layer}.npz") for layer in lib.LAYERS}
        n = len(lib.canary_prompts())
        if any(list(mine[la].ids) != list(theirs[la].ids[n:]) for la in lib.LAYERS):
            raise ValueError(f"canary_check holds other prompts than rows {n} on of {against}")
        compared = compare_rows(
            {la: s.activations for la, s in mine.items()}, {la: s.activations[n:] for la, s in theirs.items()}
        )
        rows = {**compared, "against": against}
        if compared["status"] == "differs":
            stops.append(
                f"canary_check hashes like recorded cluster {check['canary_match']['cluster']} "
                f"but its rows differ from {against}"
            )

    pilot = None
    if "pilot" in workers:
        worker = workers["pilot"]
        engines = {label: workers[label].get("engine_seconds") for label in LABELS if label in workers}
        # position_tokens are in request order, as the snapshots' ids.
        ids = [str(i) for i in hulder.Snapshot.load(run_dir / "pilot" / f"{lib.LAYERS[0]}_-1.npz").ids]
        n = len(ids)
        positions = position_findings(worker["position_tokens"], ids)
        if "-9" in positions and not positions["-9"]["im_end_on_every_prompt"]:
            findings.append(f"pilot: {len(positions['-9']['not_im_end'])} of {n} prompts lack {IM_END} at position -9")
        same = check and check["canary_sha256"] == worker["canary_sha256"]
        pilot = {
            "prompt_ids_sha256": worker.get("prompt_ids_sha256"),
            "canary_hash_vs_canary_check": "same" if same else "different" if check else "canary_check missing",
            "canary_batch_seconds": worker["batch_seconds"][0],
            "engine_seconds": engines,
            "timing": lib.timing_estimate(max(engines.values()), worker["batch_seconds"][1:]),
            "positions": positions,
        }

    keep = ("run_id", "commit", "estimated_usd", "stopped")
    return {
        **{k: run.get(k) for k in keep},
        "stop_reasons": stops,
        "findings": findings,
        "configs": configs,
        "canary_check_rows": rows,
        "pilot": pilot,
    }


if __name__ == "__main__":
    run_dir = Path(sys.argv[1])
    summary = summarise(run_dir)
    (run_dir / "pilot_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
