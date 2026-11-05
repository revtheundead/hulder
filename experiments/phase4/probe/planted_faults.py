"""Planted-fault control: how often `hulder check` reaches the correct verdict on real Qwen3-4B
scores with a fault of known size (docs/results/phase4/planted_faults.md).

Two stages, with the expected verdicts committed between them:

    source scripts/dev-env.sh
    uv run python experiments/phase4/probe/planted_faults.py expected
    git add docs/results/phase4/planted_faults_expected.json && git commit
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 uv run python experiments/phase4/probe/planted_faults.py run

`expected` plants each fault on every row of a split and reads the verdict rule on the resulting
population values, without calling `check`. `run` refuses unless that file is committed and matches
what it recomputes, then draws 1,000 pair-preserving resamples of each row through `hulder.check`,
and runs a few cases through the CLI as a user would. CPU only and free. Each finished row goes to
.cache/planted-faults/rows.jsonl at once, and a rerun resumes from there unless the expected
verdicts or the code changed, which it refuses.

Calibration is the primary verdict and test its pre-declared replication. The bar applies to rows
whose threshold is refit per resample, the gate's own assumption. Rows at the frozen threshold are
reported and don't count.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from resumable_rows import RowStore, run_rows
from scipy.special import expit
from scipy.stats import binomtest, rankdata

import hulder
from hulder.gate.check import AUROC_TOLERANCE, FNR_TOLERANCE, FPR_BAND_HIGH, FPR_BAND_LOW
from hulder.gate.probe import probe_logits

ROOT = Path(__file__).resolve().parents[3]
EXPECTED = ROOT / "docs/results/phase4/planted_faults_expected.json"
RESULTS = ROOT / "docs/results/phase4/planted_faults.json"
KEPT = ROOT / ".cache/modal-runs/20261022T125644Z/kept"
RERUN = ROOT / ".cache/modal-runs/20261026T114520Z/shuffled"
PROBE = ROOT / ".cache/probe/qwen3/18_-1.npz"
WORK = ROOT / ".cache/planted-faults"
CODE = (
    Path(__file__).resolve(),
    Path(__file__).resolve().parent / "resumable_rows.py",
    ROOT / "src/hulder/gate/check.py",
    ROOT / "src/hulder/gate/stats.py",
)

TARGET_FPR = 0.01
FROZEN_THRESHOLD = 0.8025219440460205  # the primary's, docs/results/phase4/probe_qwen3_selection.json
RESAMPLES = 1000
SEED = 0
SPLITS = ("calibration", "test")  # the primary verdict and its replication
MODES = ("refit", "frozen")
DETECTION_BAR, FALSE_ALARM_BAR = 0.95, 0.05
GROSS_DELTA_AUROC, GROSS_FPR, GROSS_FNR_RISE, GROSS_CLEAR_OF_AUROC = -0.05, 0.05, 0.15, -0.015
SHIFTS = (-3.0, -2.0, -1.0, -0.75, -0.5, -0.3, -0.1, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0)
ATTACK_SHIFTS = (-1.0, -3.0)
SCRAMBLES = (-0.02, -0.025, -0.03, -0.05, -0.1)
NOISES = {"noise, FP8's size": 0.184, "noise, Phase 1 AWQ's size": 0.93}  # mean |Δlogit| measured on each


def auroc(logits: np.ndarray, labels: np.ndarray) -> float:
    ranks = rankdata(logits)
    n_attack, n_benign = int((labels == 1).sum()), int((labels == 0).sum())
    return float((ranks[labels == 1].sum() - n_attack * (n_attack + 1) / 2) / (n_attack * n_benign))


def shift(logits: np.ndarray, delta: float) -> np.ndarray:
    return logits + delta


def attack_shift(logits: np.ndarray, labels: np.ndarray, delta: float) -> np.ndarray:
    return np.where(labels == 1, logits + delta, logits)


def sign_flip(logits: np.ndarray) -> np.ndarray:
    return -logits


def scramble(logits: np.ndarray, order: np.ndarray, n_rows: int, seed: int) -> np.ndarray:
    """The first n_rows of `order` swap logits among themselves, so a row's score stops meaning anything."""
    out, rows = logits.copy(), order[:n_rows]
    out[rows] = np.random.default_rng(seed).permutation(logits[rows])
    return out


def scramble_to(logits: np.ndarray, labels: np.ndarray, target_delta: float, seed: int) -> np.ndarray:
    """A scramble of the size whose ΔAUROC on these rows lands closest to the target."""
    order = np.random.default_rng(seed).permutation(len(logits))
    reference = auroc(logits, labels)

    def delta(n_rows: int) -> float:
        return auroc(scramble(logits, order, n_rows, seed), labels) - reference

    n_rows = min(range(len(logits) + 1), key=lambda n: abs(delta(n) - target_delta))
    return scramble(logits, order, n_rows, seed)


def noise(logits: np.ndarray, mean_abs: float, seed: int) -> np.ndarray:
    """Independent Gaussian noise, whose mean absolute size is sd·√(2/π)."""
    return logits + np.random.default_rng(seed).normal(0, mean_abs * np.sqrt(np.pi / 2), logits.shape)


def with_residuals(logits: np.ndarray, labels: np.ndarray, residuals: dict[int, np.ndarray], seed: int) -> np.ndarray:
    """A rerun's differences drawn onto other rows, within each class."""
    rng, out = np.random.default_rng(seed), logits.astype(float).copy()
    for label, pool in residuals.items():
        rows = labels == label
        out[rows] += rng.choice(pool, int(rows.sum()))
    return out


def population_values(reference: np.ndarray, candidate: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    """On every row, with no sampling, the values the verdict rule reads. Logits in, a score threshold."""
    benign, attack = labels == 0, labels == 1
    return {
        "fpr": float((expit(candidate[benign]) > threshold).mean()),
        "fnr": float((expit(candidate[attack]) <= threshold).mean()),
        "reference_fpr": float((expit(reference[benign]) > threshold).mean()),
        "reference_fnr": float((expit(reference[attack]) <= threshold).mean()),
        "delta_auroc": auroc(candidate, labels) - auroc(reference, labels),
    }


def expected_verdict(values: dict, target_fpr: float = TARGET_FPR) -> str:
    """The rule of docs/design.md §6 on population values, where every interval has shrunk to its point."""
    if values["delta_auroc"] < -AUROC_TOLERANCE:
        return hulder.Verdict.SCORER_DEGRADED.value
    if (
        values["fpr"] > FPR_BAND_HIGH * target_fpr
        or values["fpr"] < FPR_BAND_LOW * target_fpr
        or values["fnr"] > values["reference_fnr"] + FNR_TOLERANCE
    ):
        return hulder.Verdict.THRESHOLD_STALE.value
    return hulder.Verdict.PASS.value


def is_gross(values: dict) -> bool:
    """Far from every line, so sampling alone shouldn't change the verdict. Fixed before any verdict."""
    if values["delta_auroc"] <= GROSS_DELTA_AUROC:
        return True
    if values["delta_auroc"] < GROSS_CLEAR_OF_AUROC:
        return False
    return values["fpr"] >= GROSS_FPR or values["fnr"] >= values["reference_fnr"] + GROSS_FNR_RISE


def is_gross_either_verdict(values: dict) -> bool:
    """Plainly broken, but close enough to the AUROC line that DEGRADED and STALE are both correct."""
    if not GROSS_DELTA_AUROC < values["delta_auroc"] < GROSS_CLEAR_OF_AUROC:
        return False
    return values["fpr"] >= GROSS_FPR or values["fnr"] >= values["reference_fnr"] + GROSS_FNR_RISE


def resample_rows(labels: np.ndarray, seed: int) -> np.ndarray:
    """Rows drawn with replacement within each class. One draw indexes both runs, so pairs stay pairs."""
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.choice(np.flatnonzero(labels == c), int((labels == c).sum())) for c in (0, 1)])


def refit_threshold(benign_scores: np.ndarray, target_fpr: float) -> float:
    return float(np.quantile(benign_scores, 1 - target_fpr, method="higher"))


def meets_bar(kind: str, rates: dict) -> bool | None:
    if kind == "gross":
        return rates["expected_rate"] >= DETECTION_BAR
    if kind == "gross, either verdict":
        return rates["non_pass_rate"] >= DETECTION_BAR
    if kind == "null":
        return rates["non_pass_rate"] <= FALSE_ALARM_BAR
    return None


def tally(verdicts: list[str], expected: str) -> dict:
    counts = {v.value: verdicts.count(v.value) for v in hulder.Verdict}
    ci = binomtest(counts[expected], len(verdicts)).proportion_ci(method="exact")
    return {
        "counts": counts,
        "expected_rate": counts[expected] / len(verdicts),
        "expected_ci95": [ci.low, ci.high],
        "non_pass_rate": 1 - counts[hulder.Verdict.PASS.value] / len(verdicts),
    }


def uncommitted(path: Path) -> bool:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", path.name], cwd=path.parent, capture_output=True, text=True, check=True
    )
    return bool(status.stdout.strip())


def load_split(split: str, probe: hulder.Probe) -> dict:
    """A split's logits under the primary probe: from its own cell, and from the two wrong arrays."""
    import pandas as pd

    frame = pd.read_parquet(ROOT / ".cache/probe/splits.parquet", columns=["id", "split"])
    wanted = set(frame["id"][frame["split"] == split].astype(str))
    reference = hulder.Snapshot.load(KEPT / "18_-1.npz")
    ids = np.array([str(i) for i in reference.ids])
    rows = np.flatnonzero([i in wanted for i in ids])
    labels = np.asarray(reference.labels)[rows]
    out = {"ids": ids[rows], "labels": labels, "reference": probe_logits(probe, reference.activations[rows])}
    for name, cell in (("wrong layer", "27_-1"), ("wrong position", "18_-9")):
        other = hulder.Snapshot.load(KEPT / f"{cell}.npz")
        if [str(i) for i in other.ids] != ids.tolist():
            raise ValueError(f"{cell} holds other prompts")
        out[name] = probe_logits(probe, other.activations[rows])
    return out


def load_rerun(ids: np.ndarray, probe: hulder.Probe) -> np.ndarray:
    """The shuffled bf16 rerun's logits, in the reference's order."""
    rerun = hulder.Snapshot.load(RERUN / "18_-1.npz")
    position = {str(i): k for k, i in enumerate(rerun.ids)}
    return probe_logits(probe, rerun.activations[[position[str(i)] for i in ids]])


def planted(data: dict, rerun_residuals: dict[int, np.ndarray], rerun: np.ndarray | None) -> dict[str, tuple]:
    """Each fault's candidate logits, by name, with the kind it was planned as."""
    reference, labels = data["reference"], data["labels"]
    faults: dict[str, tuple] = {"rerun residuals": ("null", with_residuals(reference, labels, rerun_residuals, SEED))}
    if rerun is not None:
        faults["real rerun"] = ("null", rerun)
    for delta in SHIFTS:
        faults[f"shift {delta:+g}"] = ("fault", shift(reference, delta))
    for delta in ATTACK_SHIFTS:
        faults[f"attack-only shift {delta:+g}"] = ("fault", attack_shift(reference, labels, delta))
    for target in SCRAMBLES:
        faults[f"scramble to ΔAUROC {target:g}"] = ("fault", scramble_to(reference, labels, target, SEED))
    for name, size in NOISES.items():
        faults[name] = ("fault", noise(reference, size, SEED))
    faults["sign flip"] = ("fault", sign_flip(reference))
    faults["wrong layer, 27 for 18"] = ("fault", data["wrong layer"])
    faults["wrong position, -9 for -1"] = ("fault", data["wrong position"])
    return faults


def build() -> tuple[list[dict], dict]:
    """Every row's population values and expected verdict, and its logits by (split, fault)."""
    probe = hulder.Probe.load(PROBE)
    rows, arrays = [], {}
    residuals = None
    for split in SPLITS:
        data = load_split(split, probe)
        rerun = load_rerun(data["ids"], probe) if split == "calibration" else None
        if residuals is None:
            difference = rerun - data["reference"]
            residuals = {c: difference[data["labels"] == c] for c in (0, 1)}
        arrays[split] = {"reference": data["reference"], "labels": data["labels"]}
        for fault, (planned, candidate) in planted(data, residuals, rerun).items():
            arrays[split][fault] = candidate
            for mode in MODES:
                threshold = (
                    FROZEN_THRESHOLD
                    if mode == "frozen"
                    else refit_threshold(expit(data["reference"][data["labels"] == 0]), TARGET_FPR)
                )
                values = population_values(data["reference"], candidate, data["labels"], threshold)
                kind = "frozen" if mode == "frozen" else planned if planned == "null" else None
                kind = kind or (
                    "gross"
                    if is_gross(values)
                    else "gross, either verdict"
                    if is_gross_either_verdict(values)
                    else "graded"
                )
                rows.append(
                    {
                        "split": split,
                        "mode": mode,
                        "fault": fault,
                        "kind": kind,
                        "threshold": threshold,
                        "n_benign": int((data["labels"] == 0).sum()),
                        "n_attack": int((data["labels"] == 1).sum()),
                        "population": values,
                        "expected": expected_verdict(values),
                    }
                )
    return rows, arrays


def expected_document(rows: list[dict]) -> dict:
    return {
        "bar": {
            "gross": f"the expected verdict in at least {DETECTION_BAR:.0%} of resamples",
            "gross, either verdict": f"non-PASS in at least {DETECTION_BAR:.0%} of resamples",
            "null": f"non-PASS in at most {FALSE_ALARM_BAR:.0%} of resamples",
            "applies_to": "rows with mode refit, on each split separately",
        },
        "gross_means": {
            "delta_auroc_at_most": GROSS_DELTA_AUROC,
            "or_clear_of_auroc_line_above": GROSS_CLEAR_OF_AUROC,
            "with_fpr_at_least": GROSS_FPR,
            "or_fnr_rise_at_least": GROSS_FNR_RISE,
        },
        "resamples": RESAMPLES,
        "seed": SEED,
        "target_fpr": TARGET_FPR,
        "rows": rows,
    }


def stage_expected() -> int:
    rows, _ = build()
    EXPECTED.write_text(json.dumps(expected_document(rows), indent=2) + "\n")
    print(f"wrote {EXPECTED}; review it and commit it before `run`")
    return 0


_ARRAYS: dict = {}


def run_row(row: dict) -> dict:
    """One row's verdict over its resamples. The resample seed is shared across rows and sets each bootstrap."""
    data = _ARRAYS[row["split"]]
    labels, reference, candidate = data["labels"], expit(data["reference"]), expit(data[row["fault"]])
    verdicts = []
    for i in range(RESAMPLES):
        rows = resample_rows(labels, SEED + i)
        ref, cand, y = reference[rows], candidate[rows], labels[rows]
        threshold = row["threshold"] if row["mode"] == "frozen" else refit_threshold(ref[y == 0], TARGET_FPR)
        verdicts.append(hulder.check(ref, cand, y, threshold, TARGET_FPR, seed=SEED + i).verdict.value)
    out = {**row, **tally(verdicts, row["expected"])}
    out["key"], out["meets_bar"] = row_key(row), meets_bar(row["kind"], out)
    print(row["split"], row["mode"], row["fault"], row["expected"], out["counts"], flush=True)
    return out


def row_key(row: dict) -> str:
    return f"{row['split']}/{row['mode']}/{row['fault']}"


def identity_of(expected: Path, code: tuple[Path, ...], resamples: int, seed: int) -> dict:
    """What a resume must share with the rows already written."""
    digest = hashlib.sha256()
    for path in code:
        digest.update(path.read_bytes())
    return {
        "expected_sha256": hashlib.sha256(expected.read_bytes()).hexdigest(),
        "code_sha256": digest.hexdigest(),
        "resamples": resamples,
        "seed": seed,
    }


def snapshot(ids, labels, metadata, **arrays) -> hulder.Snapshot:
    return hulder.Snapshot(
        ids=ids, labels=labels, metadata={**metadata, "prompt_ids_sha256": hulder.prompt_ids_sha256(ids)}, **arrays
    )


def end_to_end(store: RowStore) -> list[dict]:
    """A few faults through `hulder check` as files, at the frozen threshold on calibration. Each has
    its exit code fixed here, from the exit codes and the rule. Each case is written as it finishes."""
    WORK.mkdir(parents=True, exist_ok=True)
    probe = hulder.Probe.load(PROBE)
    data = load_split("calibration", probe)
    ids, labels, logits = data["ids"], data["labels"], data["reference"]
    kept = hulder.Snapshot.load(KEPT / "18_-1.npz")
    keep = {k: v for k, v in kept.metadata.items() if k != "prompt_token_ids_sha256"}
    position = {str(i): k for k, i in enumerate(kept.ids)}
    rows = [position[str(i)] for i in ids]
    rerun = load_rerun(ids, probe)

    def scores(name: str, values: np.ndarray, order: np.ndarray | None = None) -> Path:
        path = WORK / f"{name}.npz"
        i, y, v = (ids, labels, values) if order is None else (ids[order], labels[order], values[order])
        snapshot(i, y, keep, scores=expit(v)).save(path)
        return path

    def activations(name: str, cell: str) -> Path:
        path = WORK / f"{name}.npz"
        source = hulder.Snapshot.load(KEPT / f"{cell}.npz")
        metadata = {k: v for k, v in source.metadata.items() if k != "prompt_token_ids_sha256"}
        snapshot(ids, labels, metadata, activations=np.asarray(source.activations[rows])).save(path)
        return path

    exe = Path(sys.executable).parent / "hulder"
    score_args = ["--threshold", str(FROZEN_THRESHOLD), "--target-fpr", str(TARGET_FPR)]
    reference = scores("reference", logits)
    probe_reference = activations("reference_activations", "18_-1")
    cases = [
        ("real rerun", ["--candidate", scores("rerun", rerun), "--reference", reference, *score_args], 0),
        ("shift +3", ["--candidate", scores("shift", shift(logits, 3.0)), "--reference", reference, *score_args], 10),
        ("sign flip", ["--candidate", scores("flip", sign_flip(logits)), "--reference", reference, *score_args], 11),
        ("shift +1", ["--candidate", scores("shift1", shift(logits, 1.0)), "--reference", reference, *score_args], 12),
        (
            "reordered prompts",
            [
                "--candidate",
                scores("order", logits, np.random.default_rng(SEED).permutation(len(ids))),
                "--reference",
                reference,
                *score_args,
            ],
            1,
        ),
        (
            "wrong layer, recorded in metadata",
            ["--candidate", activations("layer", "27_-1"), "--reference", probe_reference, "--probe", PROBE],
            1,
        ),
        (
            "wrong position, recorded in metadata",
            ["--candidate", activations("position", "18_-9"), "--reference", probe_reference, "--probe", PROBE],
            1,
        ),
    ]
    out = []
    for name, args, expected in cases:
        if f"end_to_end/{name}" in store.done:
            out.append(store.done[f"end_to_end/{name}"])
            continue
        run = subprocess.run([str(exe), "check", *map(str, args)], capture_output=True, text=True, check=False)
        last = (run.stderr.strip().splitlines() or [""])[-1]
        out.append(
            {
                "key": f"end_to_end/{name}",
                "case": name,
                "expected_exit": expected,
                "exit": run.returncode,
                "stderr_last_line": last[:300],
            }
        )
        store.add(out[-1])
        print(out[-1], flush=True)
    return out


def stage_run(workers: int) -> int:
    if uncommitted(EXPECTED):
        print(f"{EXPECTED.name} isn't committed. Commit the expected verdicts before running.", file=sys.stderr)
        return 1
    committed = json.loads(EXPECTED.read_text())
    rows, arrays = build()
    if committed != json.loads(json.dumps(expected_document(rows))):
        print(f"{EXPECTED.name} differs from what the code recomputes.", file=sys.stderr)
        return 1
    _ARRAYS.update(arrays)
    sys.path.insert(0, str(ROOT / "experiments/phase1/probe"))
    from common import run_manifest

    git_log = ["git", "log", "-1", "--format=%H", "--", EXPECTED.name]
    expected_commit = subprocess.run(git_log, cwd=EXPECTED.parent, capture_output=True, text=True, check=True)
    store = RowStore(
        WORK / "rows.jsonl",
        identity_of(EXPECTED, CODE, RESAMPLES, SEED),
        extra={"expected_file_commit": expected_commit.stdout.strip(), "run_manifest": run_manifest(seed=SEED)},
    )
    print(f"{len(store.done)} rows already written", flush=True)
    results = run_rows([{**row, "key": row_key(row)} for row in rows], run_row, store, workers)
    report = {**store.extra, "rows": results, "end_to_end": end_to_end(store)}
    RESULTS.write_text(json.dumps(report, indent=2) + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("expected", "run"))
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    return stage_expected() if args.stage == "expected" else stage_run(args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
