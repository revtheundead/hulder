"""Sweep step C, AWQ int4 (docs/results/phase4/sweep.md): its stop rules, the AWQ majority and every
comparison the plan declared, against docs/results/phase4/sweep_expected.json.

    source scripts/dev-env.sh
    uv run python experiments/phase4/probe/sweep_c_report.py .cache/modal-runs/<C run id> .cache/modal-runs/<B run id>

Inputs are checked first: kept/ against the plan's hashes, Experiment 2's control and shuffled
rerun against their committed manifest, B against its committed manifest, and kept/ against the
control bit for bit. Then:

  C0  the run or a replica failed, or its files didn't verify (exp2_report.load_run);
  C1  the replicas weren't fed the same prompts, tokens and metadata, or their tokens differ from
      kept/'s for the same rows;
  C2  more than 1% of batches have no majority.

AWQ has no recorded path, so the majority canary is recorded, not checked: a canary without a
majority is reported, not a stop.

Comparisons, each `hulder check` with kept/ as reference and the AWQ majority as candidate:
  primary      calibration, primary probe at its frozen 0.8025, Experiment 2's shuffled rerun as floor;
  replication  test, primary probe at 0.4524, B as floor;
  secondary    calibration at its own frozen threshold, reported beside;
  bpairs       boundary-pairs for both probes, Experiment 2's shuffled rerun as floor, not counted.
The secondary on test is paired shift only, since no test threshold was declared for it.
CPU only, in under a minute, so it writes its result once at the end. Refuses a dirty tree and runs
once per run.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import exp2_report as e2
import train_qwen3 as t

import hulder

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "modal"))
import full_capture_report as fcr
import pilot_report as pilot
import smoke_lib as lib

EXPECTED = Path("docs/results/phase4/sweep_expected.json")
EXP2_RUN = Path(".cache/modal-runs/20261026T114520Z")
EXP2_MANIFEST = Path("docs/results/phase4/modal/20261026T114520Z/manifest.json")
REPLICAS = ("awq_r1", "awq_r2", "awq_r3")
B_LABEL = "test_rerun"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files_match(directory: Path, expected: dict[str, str]) -> list[str]:
    """The files in `expected` whose sha256 in `directory` differs, or that are missing."""
    return sorted(n for n, h in expected.items() if not (directory / n).exists() or sha256(directory / n) != h)


def relabel(stops: list[str]) -> tuple[list[str], list[str]]:
    """exp2_report's majority stops as the plan's: R1 is C1, R2 is C2. R3, a canary without a
    majority, is a note here, since AWQ's canary is recorded and not checked."""
    out, notes = [], []
    for s in stops:
        if s.startswith("R3"):
            notes.append(s.replace("R3: ", "", 1))
        else:
            out.append("C" + s[1:] if s[:2] in ("R1", "R2") else s)
    return out, notes


def tokens_differ(replica: dict[str, list[int]], kept: dict[str, list[int]]) -> list[str]:
    """C1: the replica's prompt ids whose tokens differ from kept/'s, or that kept/ lacks."""
    return [i for i, tok in replica.items() if kept.get(i) != tok]


def check(probe: Path, reference: Path, rerun: Path, candidate: Path, threshold: float, target_fpr: float) -> dict:
    """hulder check as a user runs it, printed and with --json, at the declared threshold."""
    exe = Path(sys.executable).parent / "hulder"
    command = [str(exe), "check", "--probe", str(probe), "--reference", str(reference), "--noise-floor", str(rerun),
               "--candidate", str(candidate), "--threshold", repr(threshold), "--target-fpr", repr(target_fpr)]  # fmt: skip
    text = subprocess.run(command, capture_output=True, text=True, check=False)
    js = subprocess.run([*command, "--json"], capture_output=True, text=True, check=False)
    if text.returncode != js.returncode:
        raise RuntimeError(f"hulder check exited {text.returncode} printed and {js.returncode} with --json")
    report = json.loads(js.stdout)
    return {"command": ["hulder", *command[1:]], "exit_code": text.returncode, "verdict": report.get("verdict"),
            "text": text.stdout, "stderr": text.stderr, "json": report}  # fmt: skip


def report(run_dir: Path, b_dir: Path, out_dir: Path) -> dict:
    expected = json.loads(EXPECTED.read_text())
    out: dict = {"run_id": run_dir.name, "b_run_id": b_dir.name, "stops": [], "notes": []}

    # The inputs, before anything reads them.
    kept_dir = Path(".cache/modal-runs") / expected["inputs"]["kept"]["run"] / "kept"
    exp2 = json.loads(EXP2_MANIFEST.read_text())["configs"]
    b_manifest = json.loads((Path("docs/results/phase4/modal") / b_dir.name / "manifest.json").read_text())
    out["inputs"] = {
        "kept_mismatched": files_match(kept_dir, expected["inputs"]["kept"]["files_sha256"]),
        "exp2_control_mismatched": files_match(EXP2_RUN / "control", exp2["control"]["volume"]["files"]),
        "exp2_shuffled_mismatched": files_match(EXP2_RUN / "shuffled", exp2["shuffled"]["volume"]["files"]),
        "b_mismatched": files_match(b_dir / B_LABEL, b_manifest["configs"][B_LABEL]["volume"]["files"]),
    }
    if any(out["inputs"].values()):
        return {**out, "stops": ["an input doesn't match its recorded hash"], "stopped_at": "inputs"}
    names = sorted(p.stem for p in kept_dir.glob("*.npz") if p.name != lib.TOKEN_IDS_FILE)
    rows = list(range(e2.BATCH_SIZE)) + [i for start, stop in lib.EXP2_ROWS for i in range(start, stop)]
    out["kept_vs_exp2_control"] = e2.control_vs_kept(
        e2.Snapshots(EXP2_RUN / "control", names), e2.Snapshots(kept_dir, names), rows
    )
    if not out["kept_vs_exp2_control"]["identical"]:
        return {**out, "stops": ["kept/ differs from Experiment 2's control"], "stopped_at": "inputs"}

    # C0.
    loaded = e2.load_run(run_dir)
    out["stops"] = [s.replace("the run stopped", "C0: the run stopped") for s in loaded["stops"]]
    if out["stops"]:
        return {**out, "stopped_at": "C0"}
    workers = loaded["workers"]

    # C1 and C2, and the majority.
    majority = e2.fp8_majority(
        {label: e2.Snapshots(run_dir / label, names) for label in REPLICAS},
        {label: workers[label] for label in REPLICAS},
        pilot.recorded_hashes(),
    )
    kept_awq = majority.pop("kept")
    stops, notes = relabel(majority.pop("stops"))
    kept_ids, kept_tok = lib.load_token_ids(kept_dir / lib.TOKEN_IDS_FILE)
    kept_tokens = dict(zip(kept_ids, kept_tok, strict=True))
    for label in REPLICAS:
        if bad := tokens_differ(loaded["token_ids"][label], kept_tokens):
            stops.append(f"C1: {label} was fed other tokens than kept/ on {len(bad)} rows, e.g. {bad[:3]}")
    out["awq_majority"] = majority
    out["stops"], out["notes"] = stops, notes
    if stops:
        return {**out, "stopped_at": "C1/C2"}

    # The slices, in kept/'s order, without any batch the majority dropped.
    splits = fcr.splits_by_id()
    majority_ids = {str(i) for i in kept_awq[e2.CELLS["primary"]].ids}
    slices = {
        "calibration": [i for i in kept_ids if splits.get(i) == "calibration"],
        "test": [i for i in kept_ids if splits.get(i) == "test"],
        "bpairs": [i for i in kept_ids if splits.get(i, "").startswith("bpairs_")],
    }
    out["dropped_by_majority"] = {k: [i for i in ids if i not in majority_ids] for k, ids in slices.items()}
    slices = {k: [i for i in ids if i in majority_ids] for k, ids in slices.items()}
    out["slices"] = {k: len(ids) for k, ids in slices.items()}

    selection = json.loads(e2.SELECTION.read_text())
    comparisons = expected["comparisons"]
    target = expected["target_fpr"]
    reruns = {"calibration": EXP2_RUN / "shuffled", "test": b_dir / B_LABEL, "bpairs": EXP2_RUN / "shuffled"}
    out_dir.mkdir(parents=True)
    out["probes"], out["checks"], out["shift"], out["bpairs"] = {}, {}, {}, {}
    for role, cell in e2.CELLS.items():
        probe_path, probe, frozen = e2._frozen_probe(t.PROBES, cell, selection)
        out["probes"][role] = {"cell": cell, "frozen_threshold": frozen, "probe_sha256": sha256(probe_path)}
        snaps = {
            "reference": hulder.Snapshot.load(kept_dir / f"{cell}.npz"),
            "candidate": kept_awq[cell],
        }
        for slice_name, ids in slices.items():
            if role == "secondary" and slice_name == "test":
                threshold = None
            elif slice_name == "test":
                threshold = comparisons["replication"]["threshold"]
            else:
                threshold = frozen
            parts = {
                "reference": e2.select(snaps["reference"], ids, kept_tokens),
                "rerun": e2.select(hulder.Snapshot.load(reruns[slice_name] / f"{cell}.npz"), ids, kept_tokens),
                "candidate": e2.select(snaps["candidate"], ids, kept_tokens),
            }
            key = f"{role}_{slice_name}"
            shift_at = threshold if threshold is not None else frozen
            out["shift"][key] = {
                "threshold": shift_at,
                "candidate": e2.paired_shift(probe, shift_at, parts["reference"], parts["candidate"]),
                "rerun": e2.paired_shift(probe, shift_at, parts["reference"], parts["rerun"]),
            }
            if threshold is None:
                continue
            paths = {k: out_dir / f"{key}_{k}.npz" for k in parts}
            for k, s in parts.items():
                s.save(paths[k])
            result = check(probe_path, paths["reference"], paths["rerun"], paths["candidate"], threshold, target)
            if slice_name == "bpairs":
                out["bpairs"][role] = {
                    **e2.boundary_pairs(probe, threshold, parts["reference"], parts["candidate"], parts["rerun"]),
                    "check": result,
                }
            else:
                out["checks"][key] = result

    pred = expected["steps"]["C"]["prediction"]
    primary = out["checks"]["primary_calibration"]["verdict"]
    replication = out["checks"]["primary_test"]["verdict"]
    out["against_expected"] = {
        "primary": {"verdict": primary, "predicted": pred["primary"], "within_prediction": primary in pred["primary"]},
        "replication": {"verdict": replication, "predicted": pred["replication"]},
        "replication_agrees_with_primary": primary == replication,
        "note": "predictions, not bars; the replication can't change the primary",
    }
    out["stopped_at"] = None
    return out


if __name__ == "__main__":
    run_dir, b_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir, out_json = run_dir / "sweep_c_report", run_dir / "sweep_c_report.json"
    if out_dir.exists() or out_json.exists():
        sys.exit(f"{out_dir} or {out_json} exists; this report runs once per run")
    if subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True
    ).stdout:
        sys.exit("the tracked tree has changes; commit them so the report's commit is what ran")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    result = {"commit": commit, **report(run_dir, b_dir, out_dir)}
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: result.get(k) for k in ("stops", "notes", "slices", "against_expected")}, indent=2))
    for key, c in result.get("checks", {}).items():
        print(f"\n=== {key}\n{c['text']}")
