"""The hulder command: parse, call check_snapshots, print the Report, exit with its code.

No statistics live here. Exit codes follow the exit codes: the verdict's code, 1 for
a tool failure such as a PairingError or a damaged file, 2 for a usage error.
"""

import argparse
import dataclasses
import json
import sys
from importlib.resources import files
from pathlib import Path

from hulder.gate import check as gate
from hulder.gate.check import Report, Verdict, check_snapshots
from hulder.gate.pairing import PairingError
from hulder.gate.probe import Probe
from hulder.gate.snapshot import Snapshot

# The verdict rule presentation rule: past this many times the calibration set's benign count,
# more_benign_needed is printed with a caveat that it is unstable.
EDGE_FACTOR = 10
COLUMN = 26  # width of the reference and candidate columns
EDGE_CAVEAT = "the FPR estimate sits at the tolerance edge; this number is unstable and more data may not settle it"


def main() -> None:
    sys.exit(run())


def run(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "quickstart":
        return _quickstart()
    try:
        probe = Probe.load(args.probe) if args.probe else None
        threshold = _setting(parser, args.threshold, probe, "threshold")
        target_fpr = _setting(parser, args.target_fpr, probe, "target_fpr")
        report = _check(
            args.reference, args.candidate, args.noise_floor, probe, threshold, target_fpr, args.seed, args.under_test
        )
    except PairingError as e:
        print(f"hulder: pairing error: {e}", file=sys.stderr)
        return 1
    except (ValueError, OSError) as e:
        print(f"hulder: error: {e}", file=sys.stderr)
        return 1
    inputs = {"reference": args.reference, "candidate": args.candidate, "rerun": args.noise_floor, "probe": args.probe}
    if args.json:
        print(to_json(report, inputs, threshold, target_fpr))
    else:
        print(render(report, inputs, threshold, target_fpr))
    return report.verdict.exit_code


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hulder", description="Release gate for thresholded scorers.")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser(
        "check",
        help="compare a candidate serving config's snapshot against the reference's",
        description="Exit codes: 0 PASS, 10 THRESHOLD STALE, 11 SCORER DEGRADED, 12 INCONCLUSIVE, "
        "1 tool failure (including a pairing error), 2 usage error.",
    )
    check.add_argument("--reference", required=True, help="reference snapshot (.npz)")
    check.add_argument("--candidate", required=True, help="candidate snapshot (.npz)")
    check.add_argument("--noise-floor", metavar="RERUN", help="a rerun of the reference, to measure the noise floor")
    check.add_argument("--probe", help="probe (.npz) for snapshots of activations; supplies the threshold too")
    check.add_argument("--threshold", type=float, help="defaults to the probe's")
    check.add_argument("--target-fpr", type=float, help="defaults to the probe's, e.g. 0.01 for 1%%")
    check.add_argument(
        "--under-test", action="append", default=[], metavar="KEY", help="a further metadata key that may differ"
    )
    check.add_argument("--seed", type=int, default=0, help="bootstrap seed (default 0)")
    check.add_argument("--json", action="store_true", help="print the Report as JSON")
    commands.add_parser("quickstart", help="run two checks on the bundled demo data")
    return parser


def _setting(parser: argparse.ArgumentParser, given: float | None, probe: Probe | None, key: str) -> float:
    if given is not None:
        return given
    if probe is not None and key in probe.metadata:
        return float(probe.metadata[key])
    parser.error(f"--{key.replace('_', '-')} is required unless the probe records it")


def _check(reference, candidate, rerun, probe, threshold, target_fpr, seed=0, under_test=()) -> Report:
    return check_snapshots(
        Snapshot.load(reference),
        Snapshot.load(candidate),
        None if rerun is None else Snapshot.load(rerun),
        threshold=threshold,
        target_fpr=target_fpr,
        probe=probe,
        seed=seed,
        changes_under_test=tuple(under_test),
    )


def to_json(report: Report, inputs: dict, threshold: float, target_fpr: float) -> str:
    out = {
        "verdict": report.verdict.value,
        "exit_code": report.verdict.exit_code,
        "threshold": threshold,
        "target_fpr": target_fpr,
        "inputs": {k: None if v is None else str(v) for k, v in inputs.items()},
        "report": {**dataclasses.asdict(report), "verdict": report.verdict.value},
    }
    return json.dumps(out, indent=2)


def render(report: Report, inputs: dict, threshold: float, target_fpr: float) -> str:
    """The human-readable report. Changes under test come near the top, so a swapped
    reference and candidate shows at a glance."""
    lines = []
    if report.verdict is Verdict.INCONCLUSIVE and report.shift_exceeds_noise_floor:
        lines += [
            (
                f"Scores shifted beyond the noise floor: {report.decision_flips} decisions flipped "
                f"vs {report.noise_floor_decision_flips} on the rerun, "
                f"mean |Δscore| {report.mean_abs_shift:.3g} vs {report.noise_floor_mean_abs_shift:.3g}."
            ),
            "The serving change moved the scorer; whether its FPR crossed the tolerance is undecided.",
            "",
        ]
    for role in ("reference", "candidate", "rerun", "probe"):
        if inputs.get(role) is not None:
            lines.append(f"{role.capitalize():<10} {inputs[role]}")
    lines += [
        (
            f"Threshold  {threshold:.4g} at a {target_fpr:.1%} target FPR; "
            f"{report.n_benign:,} benign, {report.n_attack:,} attack"
        ),
        "",
    ]
    if report.changes_under_test:
        width = max(len(key) for key, _, _ in report.changes_under_test)
        lines.append("Changes under test (reference -> candidate)")
        lines += [f"  {key:<{width}}  {ref!r} -> {cand!r}" for key, ref, cand in report.changes_under_test]
    else:
        lines.append("Changes under test: none found in the metadata")
    lines += ["", f"Verdict: {report.verdict.value}", *_more_benign(report), ""]
    lines += _performance(report, threshold, target_fpr)
    lines += [""] + _shift(report)
    if report.warnings:
        lines += ["", "Warnings"] + [f"  - {w}" for w in report.warnings]
    lines += ["", f"Exit code {report.verdict.exit_code} ({report.verdict.value})"]
    return "\n".join(line.rstrip() for line in lines)


def _more_benign(report: Report) -> list[str]:
    if report.verdict is not Verdict.INCONCLUSIVE:
        return []
    if report.more_benign_needed == 0:
        return ["  The FPR condition is already decided; more benign examples alone won't resolve this."]
    if report.more_benign_beyond_limit:
        return [
            f"  More than {gate.MORE_BENIGN_LIMIT:,} benign examples in all would be needed to settle the FPR condition:",
            f"  {EDGE_CAVEAT}.",
        ]
    lines = [
        (
            f"  {report.more_benign_needed:,} more benign examples would settle the FPR condition, "
            "if the observed rate held."
        )
    ]
    if report.more_benign_needed > EDGE_FACTOR * report.n_benign:
        lines.append(f"  Caveat: {EDGE_CAVEAT}.")
    return lines


def _performance(report: Report, threshold: float, target_fpr: float) -> list[str]:
    def rate(e):
        return f"{e.value:.1%} [{e.low:.1%}, {e.high:.1%}]"

    def auroc(e):
        return f"{e.value:.4f} [{e.low:.4f}, {e.high:.4f}]"

    return [
        f"{'Performance at threshold ' + format(threshold, '.4g'):<34}{'reference':<{COLUMN}}candidate",
        f"  {'FPR':<32}{rate(report.reference_fpr):<{COLUMN}}{rate(report.candidate_fpr)}",
        f"  {'FNR':<32}{rate(report.reference_fnr):<{COLUMN}}{rate(report.candidate_fnr)}",
        f"  {'AUROC':<32}{auroc(report.reference_auroc):<{COLUMN}}{auroc(report.candidate_auroc)}",
        (
            f"  {'ΔAUROC (candidate - reference)':<32}{'':<{COLUMN}}{report.delta_auroc.value:+.4f} "
            f"[{report.delta_auroc.low:+.4f}, {report.delta_auroc.high:+.4f}]"
        ),
        (
            f"  Bands: candidate FPR upper bound <= {gate.FPR_BAND_HIGH * target_fpr:.1%}, "
            f"FNR upper bound <= reference FNR + {gate.FNR_TOLERANCE}, ΔAUROC lower bound >= -{gate.AUROC_TOLERANCE}"
        ),
    ]


def _shift(report: Report) -> list[str]:
    if report.noise_floor_measured:
        rerun_flips = str(report.noise_floor_decision_flips)
        rerun_shift = f"{report.noise_floor_mean_abs_shift:.3g}"
        exceeds = "yes" if report.shift_exceeds_noise_floor else "no"
    else:
        rerun_flips = rerun_shift = ""
        exceeds = (
            "not measured (rerun identical to reference)" if report.rerun_identical_to_reference else "not measured"
        )
    lines = [
        f"{'Shift (paired, same prompts)':<34}{'candidate':<{COLUMN}}rerun",
        f"  {'decisions flipped':<32}{f'{report.decision_flips} ({report.decision_flip_rate:.1%})':<{COLUMN}}{rerun_flips}",
        f"  {'mean |Δscore|':<32}{f'{report.mean_abs_shift:.3g}':<{COLUMN}}{rerun_shift}",
    ]
    if report.mean_abs_logit_shift is not None:
        rerun_logit = report.noise_floor_mean_abs_logit_shift
        lines.append(
            f"  {'mean |Δlogit|':<32}{f'{report.mean_abs_logit_shift:.3g}':<{COLUMN}}"
            f"{'' if rerun_logit is None else f'{rerun_logit:.3g}'}"
        )
    lines.append(f"  {'exceeds the noise floor':<32}{exceeds}")
    return lines


# The bundled demo: the rerun as candidate (nothing changed but run-to-run noise), then
# AWQ int4 as candidate with the rerun as noise floor. Expected verdicts from
# docs/results/phase1/real_check.md.
QUICKSTART = (
    ("The fp16 rerun as candidate: nothing changed but run-to-run noise.", "demo_fp16_run1.npz", None, Verdict.PASS),
    (
        "AWQ int4 as candidate, with the fp16 rerun as noise floor.",
        "demo_awq.npz",
        "demo_fp16_run1.npz",
        Verdict.INCONCLUSIVE,
    ),
)


def _quickstart() -> int:
    data = files("hulder.data")
    probe_path = Path(str(data / "probe_demo.npz"))
    probe = Probe.load(probe_path)
    threshold, target_fpr = probe.metadata["threshold"], probe.metadata["target_fpr"]
    reference = Path(str(data / "demo_fp16.npz"))
    as_expected = True
    for number, (title, candidate, rerun, expected) in enumerate(QUICKSTART, 1):
        candidate = Path(str(data / candidate))
        rerun = None if rerun is None else Path(str(data / rerun))
        report = _check(reference, candidate, rerun, probe, threshold, target_fpr)
        inputs = {"reference": reference.name, "candidate": candidate.name, "probe": probe_path.name}
        inputs["rerun"] = None if rerun is None else rerun.name
        print(f"=== Check {number}: {title} Expect {expected.value}.\n")
        print(render(report, inputs, threshold, target_fpr))
        print()
        as_expected &= report.verdict is expected
    if not as_expected:
        print("hulder: a quickstart check did not give its expected verdict", file=sys.stderr)
    return 0 if as_expected else 1
