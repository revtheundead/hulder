import json
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest
from synthetic import missed_attack_shift_for, unchanged_scorer

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


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({}, 0),
        ({"candidate_fpr": 0.05}, 10),
        ({"missed_attack_shift": missed_attack_shift_for(-0.05)}, 11),
    ],
)
def test_the_exit_code_is_the_verdicts(tmp_path, capsys, change, code):
    data = unchanged_scorer(seed=0, **change)
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
    assert cli.run(["check", "--reference", path, "--candidate", path, "--target-fpr", "0.01"]) == 2
    assert "--threshold is required" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--target-fpr", "0.7"),
        ("--target-fpr", "0"),
        ("--target-fpr", "nan"),
        ("--threshold", "nan"),
        ("--threshold", "inf"),
        ("--seed", "-1"),
    ],
)
def test_an_argument_out_of_range_is_a_usage_error(tmp_path, capsys, flag, value):
    reference, labels = _benign_then_attack(1191, 11)
    assert _run(tmp_path, reference, reference + 1e-6, labels, extra=[flag, value]) == 2
    assert f"argument {flag}" in capsys.readouterr().err


def test_base_model_under_test_is_a_usage_error(tmp_path, capsys):
    reference, labels = _benign_then_attack(1191, 11)
    assert _run(tmp_path, reference, reference + 1e-6, labels, extra=["--under-test", "base_model"]) == 2
    assert "base_model can't be a change under test" in capsys.readouterr().err


def test_changes_under_test_are_printed_before_the_verdict(tmp_path, capsys):
    # A swapped reference and candidate would show as quantization 'awq' -> None.
    reference, labels = _benign_then_attack(1191, 11)
    _run(tmp_path, reference, reference, labels)
    out = capsys.readouterr().out
    change = "  quantization  None -> 'awq'"
    assert change in out
    # Every line of the list, not only its heading, comes before the verdict.
    assert out.index("Changes under test") < out.index(change) < out.index("Verdict:")


def test_under_test_declares_a_further_key(tmp_path, capsys):
    reference, labels = _benign_then_attack(1191, 11)
    candidate = reference + 1e-6  # not bit-identical, which would warn
    _run(tmp_path, reference, candidate, labels, extra=["--under-test", "team"], candidate_meta={"team": "b"})
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
    assert (report["threshold"], report["target_fpr"]) == (0.5, 0.01)
    assert (out["schema_version"], out["hulder_version"]) == (1, version("hulder"))


def test_json_output_on_a_tool_failure_is_the_error_and_its_exit_code(tmp_path, capsys):
    args = ["check", "--reference", str(tmp_path / "no.npz"), "--candidate", str(tmp_path / "no.npz")]
    assert cli.run([*args, "--threshold", "0.5", "--target-fpr", "0.01", "--json"]) == 1
    captured = capsys.readouterr()
    out = json.loads(captured.out)
    assert out["exit_code"] == 1
    assert "No such file" in out["error"]
    assert out["schema_version"] == 1
    assert "hulder: error" in captured.err


def test_json_output_on_a_usage_error_is_the_error_and_exit_code_2(tmp_path, capsys):
    reference, labels = _benign_then_attack(1191, 11)
    path = _write(tmp_path, "reference", reference, labels)
    assert cli.run(["check", "--reference", path, "--candidate", path, "--target-fpr", "0.01", "--json"]) == 2
    out = json.loads(capsys.readouterr().out)
    assert out["exit_code"] == 2
    assert "--threshold is required" in out["error"]


def test_quickstart_gives_pass_for_the_rerun_and_inconclusive_with_the_shift_for_awq(capsys):
    assert cli.run(["quickstart"]) == 0
    out = capsys.readouterr().out
    first, second = out.split("=== Check 2")
    assert "Verdict: PASS" in first
    assert "Verdict: INCONCLUSIVE" in second
    assert second.splitlines()[2].startswith("Scores shifted beyond the noise floor: 11 decisions flipped vs 0")


def _damaged(tmp_path, kind):
    """A file that isn't a readable hulder npz, in one of the ways a real one gets damaged."""
    path = tmp_path / f"{kind}.npz"
    if kind == "text":
        path.write_text("ids,labels,scores\n")
    elif kind == "empty":
        path.write_bytes(b"")
    elif kind == "truncated":
        reference, labels = _benign_then_attack(1191, 11)
        whole = Path(_write(tmp_path, "whole", reference, labels)).read_bytes()
        path.write_bytes(whole[: len(whole) // 2])
    elif kind == "no_metadata":
        np.savez(path, ids=np.array(["a"]))
    elif kind == "object_array":
        np.savez(path, metadata=np.array([{"format_version": 2}], dtype=object), allow_pickle=True)
    return str(path)


@pytest.mark.parametrize("kind", ["text", "empty", "truncated", "no_metadata", "object_array"])
@pytest.mark.parametrize("role", ["--reference", "--probe"])
def test_a_damaged_file_exits_1_with_a_plain_message(tmp_path, capsys, kind, role):
    reference, labels = _benign_then_attack(1191, 11)
    good = _write(tmp_path, "reference", reference, labels)
    args = {"--reference": good, "--candidate": good, "--threshold": "0.5", "--target-fpr": "0.01"}
    args[role] = _damaged(tmp_path, kind)
    assert cli.run(["check", *[x for pair in args.items() for x in pair]]) == 1
    err = capsys.readouterr().err
    assert err.startswith(f"hulder: error: {args[role]}: ")
    assert "allow_pickle" not in err
    assert len(err.splitlines()) == 1


def test_every_warning_is_printed_in_the_human_output_and_the_json(tmp_path, capsys):
    # An undeclared metadata difference and a candidate identical to the reference: two warnings.
    reference, labels = _benign_then_attack(1191, 11)
    expected = hulder.check_snapshots(
        hulder.Snapshot.load(_write(tmp_path, "reference", reference, labels)),
        hulder.Snapshot.load(
            _write(tmp_path, "candidate", reference, labels, {**_META, "quantization": "awq", "team": "b"})
        ),
        threshold=0.5,
        target_fpr=0.01,
    ).warnings
    assert len(expected) == 2
    _run(tmp_path, reference, reference, labels, candidate_meta={"team": "b"})
    human = capsys.readouterr().out
    _run(tmp_path, reference, reference, labels, candidate_meta={"team": "b"}, extra=["--json"])
    as_json = json.loads(capsys.readouterr().out)
    for warning in expected:
        assert f"  - {warning}" in human
    assert tuple(as_json["report"]["warnings"]) == expected


def test_the_same_file_as_reference_and_candidate_passes_with_a_warning(tmp_path, capsys):
    reference, labels = _benign_then_attack(1191, 11)
    path = _write(tmp_path, "reference", reference, labels)
    assert (
        cli.run(["check", "--reference", path, "--candidate", path, "--threshold", "0.5", "--target-fpr", "0.01"]) == 0
    )
    assert "candidate is bit-identical to the reference" in capsys.readouterr().out


def test_a_probe_for_another_layer_exits_1_with_a_pairing_error(tmp_path, capsys):
    data = files("hulder.data")
    probe = hulder.Probe.load(Path(str(data / "probe_demo.npz")))
    wrong_layer = tmp_path / "probe_layer_10.npz"
    hulder.Probe(probe.weights, probe.bias, probe.mean, probe.std, {**probe.metadata, "layer": 10}).save(wrong_layer)
    snapshots = [str(data / name) for name in ("demo_fp16.npz", "demo_awq.npz")]
    args = ["check", "--probe", str(wrong_layer), "--reference", snapshots[0], "--candidate", snapshots[1]]
    assert cli.run(args) == 1
    assert "pairing error: The probe's layer is 10 but the reference's is 20" in capsys.readouterr().err


def test_quickstart_exits_1_when_a_bundled_check_gives_an_unexpected_verdict(capsys, monkeypatch):
    # The bundled data is fixed, so another verdict means a broken install.
    title, candidate, rerun, _ = cli.QUICKSTART[0]
    monkeypatch.setattr(cli, "QUICKSTART", ((title, candidate, rerun, hulder.Verdict.THRESHOLD_STALE),))
    assert cli.run(["quickstart"]) == 1
    assert "did not give its expected verdict" in capsys.readouterr().err


def _probe_lacking(tmp_path, field):
    """The demo probe saved without one array (weights, bias) or metadata key (threshold, target_fpr)."""
    probe = hulder.Probe.load(Path(str(files("hulder.data") / "probe_demo.npz")))
    path = tmp_path / f"probe_without_{field}.npz"
    if field in ("weights", "bias"):
        arrays = {"weights": probe.weights, "bias": probe.bias, "mean": probe.mean, "std": probe.std}
        del arrays[field]
        metadata = {"format_version": hulder.Probe.FORMAT_VERSION, **probe.metadata}
        np.savez(path, metadata=np.array(json.dumps(metadata)), **arrays)
    else:
        metadata = {k: v for k, v in probe.metadata.items() if k != field}
        hulder.Probe(probe.weights, probe.bias, probe.mean, probe.std, metadata).save(path)
    return str(path)


def _probe_check(probe, extra=()):
    data = files("hulder.data")
    snapshots = [str(data / name) for name in ("demo_fp16.npz", "demo_awq.npz")]
    return cli.run(["check", "--probe", probe, "--reference", snapshots[0], "--candidate", snapshots[1], *extra])


@pytest.mark.parametrize("field", ["weights", "bias"])
@pytest.mark.parametrize("overrides", [(), ("--threshold", "0.4", "--target-fpr", "0.05")])
def test_a_probe_file_without_an_array_exits_1_naming_it(tmp_path, capsys, field, overrides):
    # No flag can supply a probe's weights or bias, so overrides change nothing.
    probe = _probe_lacking(tmp_path, field)
    assert _probe_check(probe, overrides) == 1
    err = capsys.readouterr().err
    assert err == f"hulder: error: {probe}: lacks the arrays {field}, so it isn't a hulder probe file\n"


@pytest.mark.parametrize("field", ["threshold", "target_fpr"])
def test_a_probe_file_without_a_setting_and_no_flag_for_it_is_a_usage_error(tmp_path, capsys, field):
    flag = f"--{field.replace('_', '-')}"
    other = {"--threshold": "0.4", "--target-fpr": "0.05"}
    del other[flag]
    assert _probe_check(_probe_lacking(tmp_path, field), [*other.popitem()]) == 2
    assert f"{flag} is required unless the probe records it" in capsys.readouterr().err


@pytest.mark.parametrize("field", ["threshold", "target_fpr"])
def test_a_probe_file_without_a_setting_runs_when_the_flag_supplies_it(tmp_path, capsys, field):
    flag = f"--{field.replace('_', '-')}"
    value = {
        "--threshold": str(hulder.Probe.load(Path(str(files("hulder.data") / "probe_demo.npz"))).metadata["threshold"]),
        "--target-fpr": "0.05",
    }[flag]
    assert _probe_check(_probe_lacking(tmp_path, field), [flag, value]) == 12  # the demo's INCONCLUSIVE, README §4


def test_a_probe_file_whose_threshold_is_null_exits_1_naming_it(tmp_path, capsys):
    probe = hulder.Probe.load(Path(str(files("hulder.data") / "probe_demo.npz")))
    path = tmp_path / "probe_null_threshold.npz"
    hulder.Probe(probe.weights, probe.bias, probe.mean, probe.std, {**probe.metadata, "threshold": None}).save(path)
    assert _probe_check(str(path)) == 1
    assert capsys.readouterr().err == f"hulder: error: {path}: its threshold is None, not a number\n"
