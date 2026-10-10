import json

import numpy as np
import pytest
from synthetic import unchanged_scorer

import hulder
from hulder import cli

_META = {"base_model": "Qwen/Qwen2.5-1.5B-Instruct", "quantization": None}


def _benign_then_attack(n_benign, k_flagged, n_attack=910, k_missed=120):
    """Scores with exactly k_flagged benign above 0.5 and k_missed attacks below it."""
    benign = np.r_[np.full(k_flagged, 0.6), np.linspace(0.01, 0.4, n_benign - k_flagged)]
    attack = np.r_[np.full(k_missed, 0.45), np.linspace(0.7, 0.99, n_attack - k_missed)]
    return np.r_[benign, attack], np.r_[np.zeros(n_benign, int), np.ones(n_attack, int)]


def _write(tmp_path, name, scores, labels, metadata=_META, ids=None):
    ids = np.array([f"slabs/{i}" for i in range(len(labels))]) if ids is None else ids
    path = tmp_path / f"{name}.npz"
    hulder.Snapshot(ids=ids, labels=labels, scores=scores, metadata=metadata).save(path)
    return str(path)


def _run(tmp_path, reference, candidate, labels, rerun=None, extra=(), candidate_meta=None):
    args = [
        "check",
        "--reference",
        _write(tmp_path, "reference", reference, labels),
        "--candidate",
        _write(tmp_path, "candidate", candidate, labels, {**_META, "quantization": "awq", **(candidate_meta or {})}),
        "--threshold",
        "0.5",
        "--target-fpr",
        "0.01",
        *extra,
    ]
    if rerun is not None:
        args += ["--noise-floor", _write(tmp_path, "rerun", rerun, labels)]
    return cli.run(args)


@pytest.mark.parametrize(("candidate_fpr", "code"), [(None, 0), (0.05, 10)])
def test_the_exit_code_is_the_verdicts(tmp_path, capsys, candidate_fpr, code):
    data = unchanged_scorer(seed=0, candidate_fpr=candidate_fpr)
    args = ["--threshold", str(data.threshold)]
    assert _run(tmp_path, data.reference, data.candidate, data.labels, extra=args) == code


def test_a_pairing_error_exits_1_with_the_reason(tmp_path, capsys):
    reference, labels = _benign_then_attack(1191, 11)
    ids = np.array([f"slabs/{i}" for i in range(len(labels))])
    args = [
        "check",
        "--reference",
        _write(tmp_path, "reference", reference, labels, ids=ids),
        "--candidate",
        _write(tmp_path, "candidate", reference, labels, ids=ids[::-1]),
        "--threshold",
        "0.5",
        "--target-fpr",
        "0.01",
    ]
    assert cli.run(args) == 1
    assert "pairing error" in capsys.readouterr().err


def test_a_missing_file_exits_1(tmp_path, capsys):
    args = ["check", "--reference", str(tmp_path / "no.npz"), "--candidate", str(tmp_path / "no.npz")]
    assert cli.run([*args, "--threshold", "0.5", "--target-fpr", "0.01"]) == 1
    assert "hulder: error" in capsys.readouterr().err


def test_a_threshold_neither_given_nor_in_a_probe_is_a_usage_error(tmp_path, capsys):
    reference, labels = _benign_then_attack(1191, 11)
    path = _write(tmp_path, "reference", reference, labels)
    with pytest.raises(SystemExit) as exit:
        cli.run(["check", "--reference", path, "--candidate", path, "--target-fpr", "0.01"])
    assert exit.value.code == 2
    assert "--threshold is required" in capsys.readouterr().err


def test_changes_under_test_are_printed_before_the_verdict(tmp_path, capsys):
    # A swapped reference and candidate would show as quantization 'awq' -> None.
    reference, labels = _benign_then_attack(1191, 11)
    _run(tmp_path, reference, reference, labels)
    out = capsys.readouterr().out
    assert "  quantization  None -> 'awq'" in out
    assert out.index("Changes under test") < out.index("Verdict:")


def test_under_test_declares_a_further_key(tmp_path, capsys):
    reference, labels = _benign_then_attack(1191, 11)
    _run(tmp_path, reference, reference, labels, extra=["--under-test", "team"], candidate_meta={"team": "b"})
    out = capsys.readouterr().out
    assert "  team          None -> 'b'" in out
    assert "Warnings" not in out


def test_an_inconclusive_verdict_with_a_real_shift_leads_with_the_shift(tmp_path, capsys):
    reference, labels = _benign_then_attack(1191, 11)
    candidate, _ = _benign_then_attack(1191, 22)
    rerun = reference.copy()
    rerun[100] += 0.01  # an independent rerun that moved one score a little
    assert _run(tmp_path, reference, candidate, labels, rerun=rerun) == 12
    first = capsys.readouterr().out.splitlines()[0]
    assert first.startswith("Scores shifted beyond the noise floor: 11 decisions flipped vs 0 on the rerun")


def test_a_decided_verdict_leads_with_the_inputs(tmp_path, capsys):
    data = unchanged_scorer(seed=0, candidate_fpr=0.05)
    _run(tmp_path, data.reference, data.candidate, data.labels, rerun=data.reference + 1e-6)
    assert capsys.readouterr().out.startswith("Reference")


@pytest.mark.parametrize(
    ("k_flagged", "number", "caveat"),
    [
        (20, "6,407 more benign examples", False),  # under 10 x 1,191
        (22, "31,752 more benign examples", True),  # over 10 x 1,191
        (24, "More than 1,000,000 benign examples in all", True),
    ],
)
def test_more_benign_needed_past_ten_times_the_calibration_set_carries_the_caveat(
    tmp_path, capsys, k_flagged, number, caveat
):
    reference, labels = _benign_then_attack(1191, 11)
    candidate, _ = _benign_then_attack(1191, k_flagged)
    _run(tmp_path, reference, candidate, labels)
    out = capsys.readouterr().out
    assert number in out
    assert (cli.EDGE_CAVEAT in out) is caveat


def test_json_output_carries_the_report_and_its_intervals(tmp_path, capsys):
    reference, labels = _benign_then_attack(1191, 11)
    candidate, _ = _benign_then_attack(1191, 22)
    rerun = reference.copy()
    rerun[100] += 0.01
    assert _run(tmp_path, reference, candidate, labels, rerun=rerun, extra=["--json"]) == 12
    out = json.loads(capsys.readouterr().out)
    assert (out["verdict"], out["exit_code"]) == ("INCONCLUSIVE", 12)
    report = out["report"]
    assert report["candidate_fpr"]["value"] == pytest.approx(22 / 1191)
    keys = ("reference_fpr", "candidate_fpr", "reference_fnr", "candidate_fnr", "delta_auroc", "shift_over_noise_floor")
    for key in keys:
        assert report[key]["low"] <= report[key]["value"] <= report[key]["high"]
    assert report["changes_under_test"] == [["quantization", None, "awq"]]


def test_quickstart_gives_pass_for_the_rerun_and_inconclusive_with_the_shift_for_awq(capsys):
    assert cli.run(["quickstart"]) == 0
    out = capsys.readouterr().out
    first, second = out.split("=== Check 2")
    assert "Verdict: PASS" in first
    assert "Verdict: INCONCLUSIVE" in second
    assert second.splitlines()[2].startswith("Scores shifted beyond the noise floor: 11 decisions flipped vs 0")
