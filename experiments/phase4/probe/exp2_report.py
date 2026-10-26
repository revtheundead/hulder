"""Experiment 2: the first gate comparison on Qwen3-4B, bf16 against vLLM's online FP8 (W8A8).

    source scripts/dev-env.sh && uv run python experiments/phase4/probe/exp2_report.py .cache/modal-runs/<run id>

Reads an exp2 run (experiments/modal/smoke_lib.py, EXPERIMENTS["exp2"]) and the full capture's
kept/, and stops at the first step that fails:
  1. control   the bf16 control equals kept/ bit for bit on every array and batch it holds.
  2. majority  the three FP8 replicas, batch by batch on every array, each batch kept from its
               majority. R1, they were fed the same prompts, tokens and metadata as each other and
               the control; R2, at most 1% of batches have no majority; R3, their canaries have one.
  3. floor     the shuffled bf16 rerun, its rows put in the control's order by prompt id, against
               the control: the batch-composition noise floor.
  4. check     hulder check through its CLI, as a user runs it: the control as reference, the
               shuffled rerun as --noise-floor and the FP8 majority as candidate, on the
               calibration split, for the frozen primary probe and then the secondary, chosen
               after the fact. Its exit code, printed report and --json report are kept whole.
  5. bpairs    the same on all 1,200 boundary-pairs prompts. Their threshold wasn't set there, so
               nothing on this slice counts toward the verdict.

Writes the check inputs to <run>/exp2_report/ and the report to <run>/exp2_report.json, and
refuses to run if either exists.
"""

import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import train_qwen3 as t

import hulder
from hulder.gate.probe import probe_logits

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "modal"))
import full_capture_report as fcr
import pilot_report as pilot
import smoke_lib as lib

KEPT = Path(".cache/modal-runs/20261022T125644Z/kept")
FULL_CAPTURE_REPORT = KEPT.parent / "full_capture_report.json"
SELECTION = Path("docs/results/phase4/probe_qwen3_selection.json")
CELLS = {"primary": "18_-1", "secondary": "27_-1"}  # the secondary was chosen after experiment 1
FP8 = ("fp8_r1", "fp8_r2", "fp8_r3")
NO_MAJORITY_LIMIT = fcr.NO_MAJORITY_LIMIT  # R2
BATCH_SIZE = lib.COMMON["batch_size"]
BPAIRS_NOTE = (
    "Not counted toward the verdict: the threshold was set on S-Labs calibration, not on "
    "boundary-pairs, where the primary already flags most benign prompts."
)


class Snapshots(Mapping):
    """A capture's snapshots by array name, each loaded only when read."""

    def __init__(self, directory: Path, names: list[str]):
        self.directory, self.names = directory, list(names)

    def __getitem__(self, name: str) -> hulder.Snapshot:
        if name not in self.names:
            raise KeyError(name)
        return hulder.Snapshot.load(self.directory / f"{name}.npz")

    def __iter__(self):
        return iter(self.names)

    def __len__(self) -> int:
        return len(self.names)


def _bits(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a).view(np.uint16)  # so -0.0 differs from 0.0


def control_vs_kept(control: Mapping, kept: Mapping, rows: list[int], batch_size: int = BATCH_SIZE) -> dict:
    """Per array, whether the control holds kept's rows at `rows`, prompt for prompt, with the same
    labels and the same bits, and which batches (numbered from 1) differ."""
    arrays, n = {}, None
    for name in control:
        mine, theirs = control[name], kept[name]
        ids = [str(i) for i in mine.ids]
        theirs_ids = [str(theirs.ids[i]) for i in rows]
        n = -(-len(ids) // batch_size)
        same_shape = mine.activations.shape == (len(rows), theirs.activations.shape[1])
        differing = list(range(1, n + 1))
        if same_shape:
            a, b = (
                lib.batch_digests(_bits(mine.activations), batch_size),
                lib.batch_digests(_bits(np.asarray(theirs.activations)[rows]), batch_size),
            )
            differing = [k + 1 for k, (x, y) in enumerate(zip(a, b, strict=True)) if x != y]
        arrays[name] = {
            "ids_match": ids == theirs_ids,
            "labels_match": len(mine.labels) == len(rows) and bool(np.array_equal(mine.labels, theirs.labels[rows])),
            "differing_batches": differing,
        }
    identical = bool(arrays) and all(
        a["ids_match"] and a["labels_match"] and not a["differing_batches"] for a in arrays.values()
    )
    return {"identical": identical, "batches": n, "arrays": arrays}


def fp8_majority(
    replicas: dict[str, Mapping], workers: dict[str, dict], recorded: dict, batch_size: int = BATCH_SIZE
) -> dict:
    """Each batch's majority across the replicas, from every array together, with R1 to R3."""
    labels = sorted(replicas)
    names = list(replicas[labels[0]])
    stops = []
    for key in ("prompt_token_ids_sha256", "canary_token_ids_sha256"):
        if len({workers[label].get(key) for label in labels}) != 1 or workers[labels[0]].get(key) is None:
            stops.append(f"R1: the replicas' {key} differ")
    digests: dict[str, dict[str, list[str]]] = {label: {} for label in labels}
    for name in names:
        snapshots = {label: replicas[label][name] for label in labels}
        if any([str(i) for i in s.ids] != [str(i) for i in snapshots[labels[0]].ids] for s in snapshots.values()):
            stops.append(f"R1: the replicas' {name} prompt ids differ")
        keys = {k for s in snapshots.values() for k in s.metadata} - set(fcr.CHECKED_APART)
        differing = sorted(
            k for k in keys if len({json.dumps(s.metadata.get(k), sort_keys=True) for s in snapshots.values()}) > 1
        )
        if differing:
            stops.append(f"R1: the replicas' {name} metadata differs in {differing}")
        for label, s in snapshots.items():
            digests[label][name] = lib.batch_digests(_bits(s.activations), batch_size)
    batches = lib.batch_agreement(digests, first_batch=1)
    kinds = [b["kind"] for b in batches]
    counts = {
        "batches": len(batches),
        "agree": kinds.count("agree"),
        "split": kinds.count("split"),
        "no_majority": kinds.count("no majority"),
    }
    if counts["no_majority"] > NO_MAJORITY_LIMIT * len(batches):
        stops.append(f"R2: {counts['no_majority']} of {len(batches)} batches have no majority, over the 1% limit")
    by_hash: dict[str, list[str]] = {}
    for label in labels:
        by_hash.setdefault(workers[label]["canary_sha256"], []).append(label)
    top = max(by_hash, key=lambda h: len(by_hash[h]))
    majority_canary = None
    if 2 * len(by_hash[top]) > len(labels):
        match = pilot.match_canary(top, recorded)
        cluster = match["cluster"] if isinstance(match, dict) else match
        majority_canary = {"canary_sha256": top, "cluster": cluster, "replicas": by_hash[top]}
    else:
        stops.append("R3: the replicas' canaries have no majority")
    kept = None
    if not stops:
        kept = {}
        for name in names:
            snapshots = {label: replicas[label][name] for label in labels}
            first = snapshots[next(label for b in batches for label in b["majority"])]
            ids = lib.majority_rows({k: s.ids for k, s in snapshots.items()}, batches, batch_size, 1)
            metadata = {
                **first.metadata,
                "gpu_uuid": sorted(str(s.metadata.get("gpu_uuid")) for s in snapshots.values()),
                "canary_sha256": majority_canary["canary_sha256"],
                "prompt_ids_sha256": hulder.prompt_ids_sha256([str(i) for i in ids]),
            }
            if len(ids) != len(first.ids):
                metadata.pop("prompt_token_ids_sha256", None)  # select() recomputes it for what is checked
            kept[name] = hulder.Snapshot(
                ids=ids,
                labels=lib.majority_rows({k: s.labels for k, s in snapshots.items()}, batches, batch_size, 1),
                activations=lib.majority_rows({k: s.activations for k, s in snapshots.items()}, batches, batch_size, 1),
                metadata=metadata,
            )
    return {
        "stops": stops,
        "counts": counts,
        "batches": batches,
        "outliers_per_replica": {label: sum(label in b["outliers"] for b in batches) for label in labels},
        "majority_canary": majority_canary,
        "kept": kept,
    }


def select(snapshot: hulder.Snapshot, ids: list[str], tokens: dict[str, list[int]]) -> hulder.Snapshot:
    """The snapshot's rows for `ids`, in that order, with the id and token hashes recomputed."""
    row = {str(i): k for k, i in enumerate(snapshot.ids)}
    picked = [row[i] for i in ids]
    return hulder.Snapshot(
        ids=np.array(ids),
        labels=np.asarray(snapshot.labels)[picked],
        activations=np.asarray(snapshot.activations)[picked],
        metadata={
            **snapshot.metadata,
            "prompt_ids_sha256": hulder.prompt_ids_sha256(ids),
            "prompt_token_ids_sha256": lib.token_ids_sha256([tokens[i] for i in ids]),
        },
    )


def paired_shift(probe: hulder.Probe, threshold: float, reference: hulder.Snapshot, other: hulder.Snapshot) -> dict:
    """Another run against the reference on the same prompts in the same order: rows bit for bit,
    score and logit shifts, and decisions flipped at the threshold."""
    if [str(i) for i in reference.ids] != [str(i) for i in other.ids]:
        raise ValueError("the snapshots don't pair: other prompts or another order")
    a, b = np.asarray(reference.activations), np.asarray(other.activations)
    ds = np.abs(hulder.apply_probe(probe, b) - hulder.apply_probe(probe, a))
    dl = np.abs(probe_logits(probe, b) - probe_logits(probe, a))
    before, after = hulder.apply_probe(probe, a) > threshold, hulder.apply_probe(probe, b) > threshold
    flips, y = before != after, np.asarray(reference.labels)
    return {
        "n": len(a),
        "rows_bit_identical": int((_bits(a) == _bits(b)).all(axis=1).sum()),
        "mean_abs_dscore": float(ds.mean()),
        "max_abs_dscore": float(ds.max()),
        "mean_abs_dlogit": float(dl.mean()),
        "max_abs_dlogit": float(dl.max()),
        "flips": t.rate(int(flips.sum()), len(a)),
        "flips_by_class": {
            name: {
                "newly_flagged": int((~before & after & (y == label)).sum()),
                "unflagged": int((before & ~after & (y == label)).sum()),
            }
            for name, label in (("attack", 1), ("benign", 0))
        },
    }


def split_rates(fp8: dict, bf16: dict) -> dict:
    """The FP8 replicas' rate of 2-to-1 split batches against the bf16 full capture's, by Fisher's
    exact test. Post hoc, and the two ran on different slices of the request."""
    from scipy.stats import fisher_exact

    a, b = (fp8["split"], fp8["batches"]), (bf16["split"], bf16["batches"])
    table = [[a[0], a[1] - a[0]], [b[0], b[1] - b[0]]]
    return {
        "fp8": t.rate(*a),
        "bf16_full_capture": t.rate(*b),
        "fisher_two_sided_p": float(fisher_exact(table).pvalue),
        "note": (
            "A post hoc test, chosen after seeing the splits. The FP8 replicas ran this experiment's 414 batches "
            "and the bf16 replicas the full capture's 2,042, so the slices differ."
        ),
    }


def run_check(probe: Path, reference: Path, rerun: Path, candidate: Path) -> dict:
    """hulder check as a user runs it, from this environment's console script: printed, then --json.
    Both runs read the same files with the default seed, so they report the same verdict."""
    exe = Path(sys.executable).parent / "hulder"
    command = [
        str(exe),
        "check",
        "--probe", str(probe),
        "--reference", str(reference),
        "--noise-floor", str(rerun),
        "--candidate", str(candidate),
    ]  # fmt: skip
    text = subprocess.run(command, capture_output=True, text=True, check=False)
    js = subprocess.run([*command, "--json"], capture_output=True, text=True, check=False)
    if text.returncode != js.returncode:
        raise RuntimeError(f"hulder check exited {text.returncode} printed and {js.returncode} with --json")
    return {
        "command": ["hulder", *command[1:]],
        "exit_code": text.returncode,
        "text": text.stdout,
        "stderr": text.stderr,
        "json": json.loads(js.stdout),
    }


def boundary_pairs(
    probe: hulder.Probe,
    threshold: float,
    reference: hulder.Snapshot,
    candidate: hulder.Snapshot,
    rerun: hulder.Snapshot,
) -> dict:
    """Flag rates at the frozen threshold, pair-clustered AUROC and paired shift on boundary-pairs."""
    clusters = [t.pair_of(str(i)) for i in reference.ids]
    out = {"counts_toward_verdict": False, "note": BPAIRS_NOTE}
    for side, s in (("reference", reference), ("candidate", candidate)):
        scores, y = hulder.apply_probe(probe, np.asarray(s.activations)), np.asarray(s.labels)
        out[side] = {**t.at_threshold(scores, y, threshold), "auroc": t.auroc_ci(scores, y, clusters)}
    out["shift"] = {
        "candidate": paired_shift(probe, threshold, reference, candidate),
        "rerun": paired_shift(probe, threshold, reference, rerun),
    }
    return out


def load_run(run_dir: Path) -> dict:
    """Each config's worker record and token ids, and what stops the report before it reads a
    snapshot: a stopped run, a failed config, files not fetched intact, or token ids that don't
    match their recorded hash."""
    run = json.loads((run_dir / "manifest.json").read_text())
    stops = [f"the run stopped: {run['stopped']}"] if run.get("stopped") else []
    workers, token_ids = {}, {}
    for label in run["configs"]:
        manifest = json.loads((run_dir / label / "manifest.json").read_text())
        if failed := pilot.failure_reason(run_dir, label, manifest):
            stops.append(failed)
        elif problem := (run["configs"][label].get("volume") or {}).get("problem"):
            stops.append(f"{label}'s snapshots weren't fetched intact: {problem}")
        else:
            workers[label] = manifest["worker"]
            ids, tokens = lib.load_token_ids(run_dir / label / lib.TOKEN_IDS_FILE)
            if lib.token_ids_sha256(tokens) != workers[label]["prompt_token_ids_sha256"]:
                stops.append(f"{label}'s {lib.TOKEN_IDS_FILE} doesn't match its prompt_token_ids_sha256")
            token_ids[label] = dict(zip(ids, tokens, strict=True))
    names = sorted(n.removesuffix(".npz") for n in next(iter(workers.values()))["snapshots"]) if workers else []
    return {"run": run, "stops": stops, "workers": workers, "token_ids": token_ids, "names": names}


def _frozen_probe(probe_dir: Path, cell: str, selection: dict) -> tuple[Path, hulder.Probe, float]:
    path = probe_dir / f"{cell}.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != selection["cells"][cell]["probe_sha256"]:
        raise ValueError(f"{path} isn't the frozen probe for {cell}")
    probe = hulder.Probe.load(path)
    if probe.metadata["threshold"] != selection["cells"][cell]["threshold"]:
        raise ValueError(f"{path}'s threshold isn't the frozen one")
    return path, probe, probe.metadata["threshold"]


def report(
    run_dir: Path,
    kept_dir: Path,
    probe_dir: Path,
    out_dir: Path,
    *,
    splits: dict[str, str],
    selection_path: Path = SELECTION,
    full_capture_report: Path = FULL_CAPTURE_REPORT,
) -> dict:
    loaded = load_run(run_dir)
    out: dict = {"run_id": run_dir.name, "stops": list(loaded["stops"])}
    if out["stops"]:
        return {**out, "stopped_at": "run"}
    names = loaded["names"]
    rows = list(range(BATCH_SIZE)) + [i for start, stop in lib.EXP2_ROWS for i in range(start, stop)]

    # 1. The control against kept/.
    out["control"] = control_vs_kept(Snapshots(run_dir / "control", names), Snapshots(kept_dir, names), rows)
    if not out["control"]["identical"]:
        return {**out, "stopped_at": "control"}

    # 2. The FP8 majority, with the control's tokens as one more R1 check.
    workers = loaded["workers"]
    majority = fp8_majority(
        {label: Snapshots(run_dir / label, names) for label in FP8},
        {label: workers[label] for label in FP8},
        pilot.recorded_hashes(),
    )
    if workers[FP8[0]]["prompt_token_ids_sha256"] != workers["control"]["prompt_token_ids_sha256"]:
        majority["stops"].append("R1: the FP8 replicas were fed other tokens than the control")
    kept = majority.pop("kept")
    if full_capture_report.exists():
        bf16 = json.loads(full_capture_report.read_text())["counts"]
        majority["split_rate_vs_bf16_full_capture"] = split_rates(majority["counts"], bf16)
    out["fp8_majority"] = majority
    if majority["stops"]:
        return {**out, "stops": out["stops"] + majority["stops"], "stopped_at": "fp8_majority"}

    # 3 to 5. The slices, each in the control's order, without any batch the majority dropped.
    tokens = loaded["token_ids"]["control"]
    control_ids = [str(i) for i in Snapshots(run_dir / "control", names)[CELLS["primary"]].ids]
    kept_ids = {str(i) for i in kept[CELLS["primary"]].ids}
    slices = {
        "calibration": [i for i in control_ids if splits.get(i) == "calibration" and i in kept_ids],
        "bpairs": [i for i in control_ids if splits.get(i, "").startswith("bpairs_") and i in kept_ids],
    }
    out["slices"] = {name: len(ids) for name, ids in slices.items()}
    out["dropped_by_majority"] = sorted(set(control_ids) - kept_ids)
    selection = json.loads(selection_path.read_text())
    out_dir.mkdir(parents=True)
    out["probes"], out["floor"], out["checks"], out["bpairs"] = {}, {}, {}, {}
    for role, cell in CELLS.items():
        probe_path, probe, threshold = _frozen_probe(probe_dir, cell, selection)
        out["probes"][role] = {
            "cell": cell,
            "threshold": threshold,
            "probe_sha256": selection["cells"][cell]["probe_sha256"],
        }
        control = hulder.Snapshot.load(run_dir / "control" / f"{cell}.npz")
        shuffled = hulder.Snapshot.load(run_dir / "shuffled" / f"{cell}.npz")
        inputs = {}
        for slice_name, ids in slices.items():
            ref, rerun, cand = (
                select(control, ids, tokens),
                select(shuffled, ids, tokens),
                select(kept[cell], ids, tokens),
            )
            paths = {k: out_dir / f"{role}_{slice_name}_{k}.npz" for k in ("reference", "rerun", "candidate")}
            for k, s in zip(paths, (ref, rerun, cand), strict=True):
                s.save(paths[k])
            inputs[slice_name] = (ref, rerun, cand, paths)
        ref, rerun, cand, paths = inputs["calibration"]
        out["floor"][role] = {
            "rerun": paired_shift(probe, threshold, ref, rerun),
            "candidate": paired_shift(probe, threshold, ref, cand),
        }
        out["checks"][role] = run_check(probe_path, paths["reference"], paths["rerun"], paths["candidate"])
        ref, rerun, cand, paths = inputs["bpairs"]
        out["bpairs"][role] = {
            **boundary_pairs(probe, threshold, ref, cand, rerun),
            "check": run_check(probe_path, paths["reference"], paths["rerun"], paths["candidate"]),
        }
    out["stopped_at"] = None
    return out


if __name__ == "__main__":
    run_dir = Path(sys.argv[1])
    out_dir, out_json = run_dir / "exp2_report", run_dir / "exp2_report.json"
    if out_dir.exists() or out_json.exists():
        sys.exit(f"{out_dir} or {out_json} exists; this report runs once per run")
    if subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True
    ).stdout:
        sys.exit("the tracked tree has changes; commit them so the report's commit is what ran")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    result = {"commit": commit, **report(run_dir, KEPT, t.PROBES, out_dir, splits=fcr.splits_by_id())}
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("fp8_majority",)}, indent=2, ensure_ascii=False)[
            :20000
        ]
    )
