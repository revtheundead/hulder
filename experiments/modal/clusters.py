"""Group captures of the same config, from any number of runs, into bit-exact clusters, and measure
how far apart the clusters are. Runs with different engine settings (engine_env, leaving out
logging) are kept apart.

    uv run python experiments/modal/clusters.py OUT.json RUN_DIR [RUN_DIR ...]

Writes OUT.json and prints it. Each cluster lists its members as <run id>/<label>, with GPU UUID,
driver, container and position in the container, so what a cluster's members share can be read off.
Captures with a cuBLASLt trace (cublas_fingerprint.txt, or the 2026-10-13 run's cublas_summary.txt)
are also grouped by the set of matmul algorithms they chose, each set diffed against the first.
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import smoke_lib as lib

import hulder


def engine_settings(env: dict) -> dict:
    """The engine environment that can change results: cuBLAS logging variables only record."""
    return {k: v for k, v in env.items() if not k.startswith("CUBLAS")}


def members(run_dirs: list[Path]) -> list[dict]:
    """Every saved capture in the runs, with what identifies its config and where it ran."""
    found = []
    for run_dir in run_dirs:
        run = json.loads((run_dir / "manifest.json").read_text())
        position: dict[int, int] = {}
        for label, entry in run["configs"].items():
            manifest = json.loads((run_dir / label / "manifest.json").read_text())
            if "worker" not in manifest or not manifest["worker"]["snapshots"]:
                continue
            container = entry.get("container", 0)
            position[container] = position.get(container, 0) + 1
            _, uuid, driver, _ = (f.strip() for f in manifest["nvidia_smi"].split(","))
            config = manifest["worker"]["capture_config"]
            traces = [run_dir / label / name for name in ("cublas_fingerprint.txt", "cublas_summary.txt")]
            trace = next((p for p in traces if p.exists()), None)
            found.append(
                {
                    "name": f"{run['run_id']}/{label}",
                    "path": run_dir / label,
                    "identity": json.dumps([config, engine_settings(manifest.get("engine_env", {}))], sort_keys=True),
                    "gpu_uuid": uuid,
                    "gpu_driver": driver,
                    "container": f"{run['run_id']}#{container}",
                    "position_in_container": position[container],
                    "cublas": lib.matmul_signatures(trace.read_text(errors="replace")) if trace else None,
                }
            )
    return found


def clusters(found: list[dict]) -> list[dict]:
    """Bit-exact clusters within each identity, and the mean relative L2 between clusters per layer."""
    out = []
    for identity in sorted({m["identity"] for m in found}):
        group = [m for m in found if m["identity"] == identity]
        acts = {
            m["name"]: {layer: hulder.Snapshot.load(m["path"] / f"{layer}.npz").activations for layer in lib.LAYERS}
            for m in group
        }
        exact: list[list[dict]] = []
        for m in group:
            home = next(
                (
                    c
                    for c in exact
                    if all(np.array_equal(acts[c[0]["name"]][la], acts[m["name"]][la]) for la in lib.LAYERS)
                ),
                None,
            )
            if home is None:
                exact.append([m])
            else:
                home.append(m)
        between = {}
        for (i, a), (j, b) in combinations(enumerate(exact), 2):
            between[f"{i} vs {j}"] = {
                f"layer {la}": float(lib.relative_l2(acts[a[0]["name"]][la], acts[b[0]["name"]][la]).mean())
                for la in lib.LAYERS
            }
        keep = ("name", "gpu_uuid", "gpu_driver", "container", "position_in_container")
        out.append(
            {
                "identity": json.loads(identity),
                "clusters": [[{k: m[k] for k in keep} for m in c] for c in exact],
                "between_clusters_relative_l2_mean": between,
                "cublas_signature_sets": signature_sets(group),
                "untraced": [m["name"] for m in group if m["cublas"] is None],
            }
        )
    return out


def signature_sets(group: list[dict]) -> list[dict]:
    """The traced captures grouped by the exact set of cuBLASLt matmul algorithms they chose, in
    order of first appearance, each later set with what it has and lacks against the first."""
    sets: list[tuple[set[str], list[str]]] = []
    for m in group:
        if m["cublas"] is None:
            continue
        home = next((names for sigs, names in sets if sigs == m["cublas"]), None)
        if home is None:
            sets.append((m["cublas"], [m["name"]]))
        else:
            home.append(m["name"])
    out = []
    for i, (sigs, names) in enumerate(sets):
        entry: dict = {"members": names, "signatures": len(sigs)}
        if i:
            entry["only_here"] = sorted(sigs - sets[0][0])
            entry["only_in_first"] = sorted(sets[0][0] - sigs)
        out.append(entry)
    return out


if __name__ == "__main__":
    result = clusters(members([Path(p) for p in sys.argv[2:]]))
    Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
