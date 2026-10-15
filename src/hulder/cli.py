"""The hulder command: parse, call check_snapshots, print the Report, exit with its code.

No statistics live here (docs/design.md §4). Exit codes follow docs/design.md §4: the verdict's code, 1 for
a tool failure such as a PairingError or a damaged file, 2 for a usage error.
"""

import argparse
import dataclasses
import json
import math
import sys
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
from typing import NoReturn

import hulder.gate.check as gate_check
from hulder.gate.check import Report, Verdict, check_snapshots
from hulder.gate.jsonl import load_scores_jsonl
from hulder.gate.pairing import PairingError
from hulder.gate.probe import Probe
from hulder.gate.snapshot import Snapshot

# The tolerance-edge presentation rule (docs/design.md §6): past this many times the calibration set's benign count,
# more_benign_needed is printed with a caveat that it is unstable.
EDGE_FACTOR = 10
COLUMN = 26  # width of the reference and candidate columns
EDGE_CAVEAT = "the FPR estimate sits at the tolerance edge; this number is unstable and more data may not settle it"


# Input formats by file extension: snapshots, or bring-your-own scores (docs/design.md §4).
FORMATS = {".npz": "npz", ".jsonl": "jsonl"}

# Version of --json's layout. Bump it when a key is renamed or removed, not when one is added.
JSON_SCHEMA_VERSION = 1


class UsageError(Exception):
    """A usage error, exit 2 (docs/design.md §4). argparse's own exit is replaced so --json can report it."""

    def __init__(self, message: str, usage: str):
        super().__init__(message)
        self.usage = usage


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise UsageError(message, self.format_usage())


def main() -> None:
    sys.exit(run())


def run(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    as_json = "--json" in argv
    parser = _parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "quickstart":
            return _quickstart()
        if forbidden := sorted(set(args.under_test) & set(gate_check.NEVER_UNDER_TEST)):
            parser.error(
                f"--under-test {forbidden[0]}: {forbidden[0]} can't be a change under test, "
                "since it says both runs serve the same model"
            )
        formats = {
            path: _format(parser, path, args.input_format)
            for path in (args.reference, args.candidate, args.noise_floor)
            if path is not None
        }
        probe = Probe.load(args.probe) if args.probe else None
        threshold = _setting(parser, args.threshold, probe, "threshold")
        target_fpr = _setting(parser, args.target_fpr, probe, "target_fpr")
        report = _check(
            args.reference,
            args.candidate,
            args.noise_floor,
            probe,
            threshold,
            target_fpr,
            args.seed,
            args.under_test,
            formats,
        )
    except UsageError as e:
        print(e.usage, end="", file=sys.stderr)
        return _fail(str(e), 2, as_json)
    except PairingError as e:
        return _fail(f"pairing error: {e}", 1, as_json, prefix="")
    except (ValueError, OSError) as e:
        return _fail(str(e), 1, as_json)
    except Exception as e:
        if as_json:  # a bug: the JSON still says so, and the traceback follows on stderr
            print(_error_json(f"internal error: {e!r}", 1))
        raise
    inputs = _inputs(args.reference, args.candidate, args.noise_floor, args.probe)
    print(to_json(report, inputs) if args.json else render(report, inputs))
    return report.verdict.exit_code


def _fail(message: str, code: int, as_json: bool, prefix: str = "error: ") -> int:
    """Print a failure to stderr, and with --json to stdout too, then give its exit code."""
    print(f"hulder: {prefix}{message}", file=sys.stderr)
    if as_json:
        print(_error_json(message, code))
    return code


def _error_json(message: str, code: int) -> str:
    return json.dumps(
        {
            "schema_version": JSON_SCHEMA_VERSION,
            "hulder_version": version("hulder"),
            "error": message,
            "exit_code": code,
        },
        indent=2,
    )


def _parser() -> _Parser:
    parser = _Parser(prog="hulder", description="Release gate for thresholded scorers.")
    commands = parser.add_subparsers(dest="command", required=True)
    check_parser = commands.add_parser(
        "check",
        help="compare a candidate serving config's snapshot against the reference's",
        description="Exit codes: 0 PASS, 10 THRESHOLD STALE, 11 SCORER DEGRADED, 12 INCONCLUSIVE, "
        "1 tool failure (including a pairing error), 2 usage error.",
    )
    check_parser.add_argument("--reference", required=True, help="reference snapshot (.npz) or scores (.jsonl)")
    check_parser.add_argument("--candidate", required=True, help="candidate snapshot (.npz) or scores (.jsonl)")
    check_parser.add_argument(
        "--noise-floor", metavar="RERUN", help="a rerun of the reference, to measure the noise floor"
    )
    check_parser.add_argument("--probe", help="probe (.npz) for snapshots of activations; supplies the threshold too")
    check_parser.add_argument("--threshold", type=_finite, help="defaults to the probe's")
    check_parser.add_argument("--target-fpr", type=_rate, help="defaults to the probe's, e.g. 0.01 for 1%%")
    check_parser.add_argument(
        "--under-test", action="append", default=[], metavar="KEY", help="a further metadata key that may differ"
    )
    check_parser.add_argument(
        "--input-format",
        choices=sorted(set(FORMATS.values())),
        help="read every input in this format, whatever its extension (default: from each file's extension)",
    )
    check_parser.add_argument("--seed", type=_seed, default=0, help="bootstrap seed (default 0)")
    check_parser.add_argument("--json", action="store_true", help="print the Report as JSON")
    commands.add_parser("quickstart", help="run two checks on the bundled demo data")
    return parser


# argparse reports an ArgumentTypeError from a type= function as a usage error naming the argument.
def _number(text: str, kind=float):
    try:
        return kind(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None


def _finite(text: str) -> float:
    value = _number(text)
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"{text} is not a finite number")
    return value


def _rate(text: str) -> float:
    value = _number(text)
    if not 0 < value <= gate_check.MAX_TARGET_FPR:  # also false for nan
        raise argparse.ArgumentTypeError(f"{text} is not a rate in (0, {gate_check.MAX_TARGET_FPR}], e.g. 0.01 for 1%")
    return value


def _seed(text: str) -> int:
    value = _number(text, int)
    if value < 0:
        raise argparse.ArgumentTypeError(f"{text} is negative; the seed must be 0 or more")
    return value


def _setting(parser: _Parser, given: float | None, probe: Probe | None, key: str) -> float:
    if given is not None:
        return given
    if probe is not None and key in probe.metadata:
        return float(probe.metadata[key])
    parser.error(f"--{key.replace('_', '-')} is required unless the probe records it")


def _format(parser: _Parser, path: str, override: str | None) -> str:
    if override is not None:
        return override
    if (fmt := FORMATS.get(Path(path).suffix.lower())) is None:
        parser.error(
            f"can't tell the format of {path} from its extension: name it .npz (a snapshot) or .jsonl "
            "(scores), or pass --input-format"
        )
    return fmt


def _load(path, fmt: str, order=None) -> Snapshot:
    """A snapshot, or a JSONL of scores put in the reference's id order when order is given."""
    return load_scores_jsonl(path, order) if fmt == "jsonl" else Snapshot.load(path)


def _check(reference, candidate, rerun, probe, threshold, target_fpr, seed=0, under_test=(), formats=None) -> Report:
    formats = formats or {}
    reference_snapshot = _load(reference, formats.get(reference, "npz"))
    order = reference_snapshot.ids
    return check_snapshots(
        reference_snapshot,
        _load(candidate, formats.get(candidate, "npz"), order),
        None if rerun is None else _load(rerun, formats.get(rerun, "npz"), order),
        threshold=threshold,
        target_fpr=target_fpr,
        probe=probe,
        seed=seed,
        changes_under_test=tuple(under_test),
    )


def _inputs(reference, candidate, rerun, probe) -> dict:
    """The input files by role, as render and to_json show them; None for a role not given."""
    return {"reference": reference, "candidate": candidate, "rerun": rerun, "probe": probe}


def to_json(report: Report, inputs: dict) -> str:
    out = {
        "schema_version": JSON_SCHEMA_VERSION,
        "hulder_version": version("hulder"),
        "verdict": report.verdict.value,
        "exit_code": report.verdict.exit_code,
        "inputs": {k: None if v is None else str(v) for k, v in inputs.items()},
        "report": {**dataclasses.asdict(report), "verdict": report.verdict.value},
    }
    return json.dumps(out, indent=2)


def render(report: Report, inputs: dict) -> str:
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
            f"Threshold  {report.threshold:.4g} at a {report.target_fpr:.1%} target FPR; "
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
    if report.recalibration is not None:
        lines += [*_recalibration(report), ""]
    lines += _performance(report)
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
            f"  More than {gate_check.MORE_BENIGN_LIMIT:,} benign examples in all would be needed to settle the FPR condition:",
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


def _recalibration(report: Report) -> list[str]:
    r = report.recalibration
    title = "Recalibrated threshold" + (" (optional)" if report.verdict is Verdict.INCONCLUSIVE else "")
    if r.threshold is None:
        return [
            (
                f"{title}: none. {report.n_benign:,} benign can't certify a {report.target_fpr:.1%} FPR "
                f"at {r.confidence:.0%} confidence; {r.benign_needed:,} benign in all could."
            )
        ]
    fnr = r.candidate_fnr
    return [
        f"{title}  {r.threshold!r} for a {report.target_fpr:.1%} target FPR ({r.method}, {r.confidence:.0%} confidence)",
        (
            f"  {r.benign_above} of {report.n_benign:,} benign above it. With {r.confidence:.0%} confidence "
            f"its FPR is at most {report.target_fpr:.1%} on traffic like this calibration set."
        ),
        f"  Candidate FNR at {r.threshold!r}: {fnr.value:.1%} [{fnr.low:.1%}, {fnr.high:.1%}]",
        (
            "  It was fitted on this candidate's calibration scores, "
            "so confirm it on a fresh candidate snapshot before relying on it."
        ),
    ]


def _performance(report: Report) -> list[str]:
    def rate(e):
        return f"{e.value:.1%} [{e.low:.1%}, {e.high:.1%}]"

    def auroc(e):
        return f"{e.value:.4f} [{e.low:.4f}, {e.high:.4f}]"

    return [
        f"{'Performance at threshold ' + format(report.threshold, '.4g'):<34}{'reference':<{COLUMN}}candidate",
        f"  {'FPR':<32}{rate(report.reference_fpr):<{COLUMN}}{rate(report.candidate_fpr)}",
        f"  {'FNR':<32}{rate(report.reference_fnr):<{COLUMN}}{rate(report.candidate_fnr)}",
        f"  {'AUROC':<32}{auroc(report.reference_auroc):<{COLUMN}}{auroc(report.candidate_auroc)}",
        (
            f"  {'ΔAUROC (candidate - reference)':<32}{'':<{COLUMN}}{report.delta_auroc.value:+.4f} "
            f"[{report.delta_auroc.low:+.4f}, {report.delta_auroc.high:+.4f}]"
        ),
        (
            f"  Bands: candidate FPR upper bound <= {gate_check.FPR_BAND_HIGH * report.target_fpr:.1%}, "
            f"FNR upper bound <= reference FNR + {gate_check.FNR_TOLERANCE}, ΔAUROC lower bound >= -{gate_check.AUROC_TOLERANCE}"
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
    if report.shift_over_noise_floor is not None:
        e = report.shift_over_noise_floor
        lines.append(f"  {'candidate - rerun mean |Δscore|':<32}{e.value:.3g} [{e.low:.3g}, {e.high:.3g}]")
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
        inputs = _inputs(reference.name, candidate.name, None if rerun is None else rerun.name, probe_path.name)
        print(f"=== Check {number}: {title} Expect {expected.value}.\n")
        print(render(report, inputs))
        print()
        as_expected &= report.verdict is expected
    if not as_expected:
        print("hulder: a quickstart check did not give its expected verdict", file=sys.stderr)
    return 0 if as_expected else 1
