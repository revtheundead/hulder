"""Sweep step B, the bf16 test rerun (docs/results/phase4/sweep.md): its stop rules and the null
check on test, against the rules committed in docs/results/phase4/sweep_expected.json.

    source scripts/dev-env.sh
    uv run python experiments/phase4/probe/sweep_b_report.py .cache/modal-runs/<run id>

Before reading B it checks kept/ against the file hashes the plan recorded, and Experiment 2's
control against its committed manifest, then that kept/ equals the control bit for bit on the
control's batches. Then:

  B0  the run or its config failed, or its files didn't verify (exp2_report.load_run);
  B1  B's test rows weren't fed kept/'s prompts and tokens;
  B2  B's canary isn't cluster 2.0, the bf16 majority path; B is then reported as a deviation
      and isn't used as a floor.

The null is `hulder check` with kept/'s test rows as reference and B's as candidate, at the
replication threshold 0.4524 for a 1% target, with no noise floor. The primary probe gives the
verdict, the secondary is reported as paired shift only, since no test threshold was declared for
it. CPU only, in seconds, so it writes its result once at the end. Refuses a dirty tree and runs
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
LABEL = "test_rerun"
MAJORITY_CLUSTER = fcr.MAJORITY  # "2.0"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files_match(directory: Path, expected: dict[str, str]) -> list[str]:
    """The files in `expected` whose sha256 in `directory` differs, or that are missing."""
    return sorted(n for n, h in expected.items() if not (directory / n).exists() or sha256(directory / n) != h)


def tokens_match(b: dict[str, list[int]], kept: dict[str, list[int]], ids: list[str]) -> list[str]:
    """B1: the ids among `ids` that B lacks, or whose tokens differ from kept/'s."""
    return [i for i in ids if i not in b or i not in kept or b[i] != kept[i]]


def canary_stop(canary_sha256: str, recorded: dict) -> tuple[str | None, object]:
    """B2: a stop message unless the canary is the bf16 majority path's cluster, and the match."""
    match = pilot.match_canary(canary_sha256, recorded)
    cluster = match["cluster"] if isinstance(match, dict) else match
    if cluster != MAJORITY_CLUSTER:
        return f"B2: B's canary is {cluster}, not cluster {MAJORITY_CLUSTER}", match
    return None, match


def check_null(probe: Path, reference: Path, candidate: Path, threshold: float, target_fpr: float) -> dict:
    """hulder check as a user runs it, printed and with --json, at the declared threshold."""
    exe = Path(sys.executable).parent / "hulder"
    command = [str(exe), "check", "--probe", str(probe), "--reference", str(reference), "--candidate", str(candidate),
               "--threshold", repr(threshold), "--target-fpr", repr(target_fpr)]  # fmt: skip
    text = subprocess.run(command, capture_output=True, text=True, check=False)
    js = subprocess.run([*command, "--json"], capture_output=True, text=True, check=False)
    if text.returncode != js.returncode:
        raise RuntimeError(f"hulder check exited {text.returncode} printed and {js.returncode} with --json")
    return {"command": ["hulder", *command[1:]], "exit_code": text.returncode, "text": text.stdout,
            "stderr": text.stderr, "json": json.loads(js.stdout)}  # fmt: skip


def report(run_dir: Path, out_dir: Path) -> dict:
    expected = json.loads(EXPECTED.read_text())
    out: dict = {"run_id": run_dir.name, "stops": []}

    # The inputs, before anything reads them.
    kept_dir = Path(".cache/modal-runs") / expected["inputs"]["kept"]["run"] / "kept"
    out["inputs"] = {
        "kept_mismatched": files_match(kept_dir, expected["inputs"]["kept"]["files_sha256"]),
        "exp2_control_mismatched": files_match(
            EXP2_RUN / "control", json.loads(EXP2_MANIFEST.read_text())["configs"]["control"]["volume"]["files"]
        ),
    }
    if out["inputs"]["kept_mismatched"] or out["inputs"]["exp2_control_mismatched"]:
        return {**out, "stops": ["an input doesn't match its recorded hash"], "stopped_at": "inputs"}
    names = sorted(p.stem for p in kept_dir.glob("*.npz") if p.name != lib.TOKEN_IDS_FILE)
    rows = list(range(e2.BATCH_SIZE)) + [i for start, stop in lib.EXP2_ROWS for i in range(start, stop)]
    out["kept_vs_exp2_control"] = e2.control_vs_kept(
        e2.Snapshots(EXP2_RUN / "control", names), e2.Snapshots(kept_dir, names), rows
    )
    if not out["kept_vs_exp2_control"]["identical"]:
        return {**out, "stops": ["kept/ differs from Experiment 2's control"], "stopped_at": "inputs"}

    # B0.
    loaded = e2.load_run(run_dir)
    out["stops"] = [s.replace("the run stopped", "B0: the run stopped") for s in loaded["stops"]]
    if out["stops"]:
        return {**out, "stopped_at": "B0"}
    worker = loaded["workers"][LABEL]

    # B1.
    splits = fcr.splits_by_id()
    b_tokens = loaded["token_ids"][LABEL]
    kept_ids, kept_tok = lib.load_token_ids(kept_dir / lib.TOKEN_IDS_FILE)
    kept_tokens = dict(zip(kept_ids, kept_tok, strict=True))
    test_ids = [i for i in kept_ids if splits.get(i) == "test"]  # kept/'s order
    out["test_rows"] = len(test_ids)
    if bad := tokens_match(b_tokens, kept_tokens, test_ids):
        out["stops"].append(f"B1: {len(bad)} test rows weren't fed kept/'s prompt and tokens, e.g. {bad[:3]}")

    # B2.
    stop, match = canary_stop(worker["canary_sha256"], pilot.recorded_hashes())
    out["canary"] = {"canary_sha256": worker["canary_sha256"], "match": match}
    if stop:
        out["stops"].append(stop)
    if out["stops"]:
        return {**out, "stopped_at": "B1/B2"}

    # The null on test.
    rep = expected["comparisons"]["replication"]
    selection = json.loads(e2.SELECTION.read_text())
    out_dir.mkdir(parents=True)
    out["probes"], out["shift"] = {}, {}
    for role, cell in e2.CELLS.items():
        probe_path, probe, frozen = e2._frozen_probe(t.PROBES, cell, selection)
        threshold = rep["threshold"] if role == "primary" else frozen
        ref = e2.select(hulder.Snapshot.load(kept_dir / f"{cell}.npz"), test_ids, kept_tokens)
        cand = e2.select(hulder.Snapshot.load(run_dir / LABEL / f"{cell}.npz"), test_ids, kept_tokens)
        out["probes"][role] = {"cell": cell, "threshold": threshold, "probe_sha256": sha256(probe_path)}
        out["shift"][role] = e2.paired_shift(probe, threshold, ref, cand)
        if role == "primary":
            paths = {k: out_dir / f"primary_test_{k}.npz" for k in ("reference", "candidate")}
            ref.save(paths["reference"])
            cand.save(paths["candidate"])
            out["null"] = {
                **check_null(probe_path, paths["reference"], paths["candidate"], threshold, expected["target_fpr"]),
                "expected": expected["steps"]["B"]["null"]["expected"],
            }
    out["stopped_at"] = None
    return out


if __name__ == "__main__":
    run_dir = Path(sys.argv[1])
    out_dir, out_json = run_dir / "sweep_b_report", run_dir / "sweep_b_report.json"
    if out_dir.exists() or out_json.exists():
        sys.exit(f"{out_dir} or {out_json} exists; this report runs once per run")
    if subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True
    ).stdout:
        sys.exit("the tracked tree has changes; commit them so the report's commit is what ran")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    result = {"commit": commit, **report(run_dir, out_dir)}
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "null"}, indent=2, ensure_ascii=False))
    if "null" in result:
        print(result["null"]["text"])
