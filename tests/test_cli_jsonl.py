"""hulder check on bring-your-own scores in JSONL (docs/design.md §4)."""

import json

import numpy as np
import pytest
from synthetic import unchanged_scorer

import hulder
from hulder import cli

META = {"model": "scorer", "base_model": "scorer"}


@pytest.fixture
def stale(tmp_path):
    """Reference, candidate and rerun JSONL files for a synthetic THRESHOLD STALE case."""
    d = unchanged_scorer(seed=0, target_fpr=0.05, n_benign=400, n_attack=300, candidate_fpr=0.2)
    rerun = unchanged_scorer(seed=0, target_fpr=0.05, n_benign=400, n_attack=300, noise_seed=1).candidate
    ids = [f"q{i}" for i in range(len(d.labels))]
    paths = {}
    for role, scores, meta in (
        ("reference", d.reference, META),
        ("candidate", d.candidate, {**META, "quantization": "int8"}),
        ("rerun", rerun, META),
    ):
        rows = [
            {"id": i, "score": float(s), "label": int(lab)} for i, s, lab in zip(ids, scores, d.labels, strict=True)
        ]
        paths[role] = _write(tmp_path / f"{role}.jsonl", rows, meta)
    paths["threshold"] = repr(d.threshold)
    return paths, d


def _write(path, rows, meta=None):
    lines = ([json.dumps({"metadata": meta})] if meta else []) + [json.dumps(r) for r in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def _args(paths, *extra):
    return [
        "check",
        "--reference", str(paths["reference"]),
        "--candidate", str(paths["candidate"]),
        "--noise-floor", str(paths["rerun"]),
        "--threshold", paths["threshold"], "--target-fpr", "0.05",
        *extra,
    ]  # fmt: skip


def test_jsonl_runs_end_to_end_and_matches_check_on_the_same_scores(stale, capsys):
    paths, d = stale
    assert cli.run(_args(paths, "--json")) == 10
    out = json.loads(capsys.readouterr().out)
    direct = hulder.check(d.reference, d.candidate, d.labels, threshold=d.threshold, target_fpr=0.05)
    assert out["report"]["candidate_fpr"]["value"] == direct.candidate_fpr.value
    assert out["report"]["recalibration"]["threshold"] == direct.recalibration.threshold
    assert out["report"]["changes_under_test"] == [["quantization", None, "int8"]]
    assert out["report"]["warnings"] == []


def test_candidate_rows_in_another_order_are_aligned_by_id(stale, tmp_path, capsys):
    paths, _ = stale
    cli.run(_args(paths, "--json"))
    in_order = json.loads(capsys.readouterr().out)["report"]
    lines = paths["candidate"].read_text().splitlines()
    paths["candidate"] = tmp_path / "shuffled.jsonl"
    paths["candidate"].write_text("\n".join([lines[0], *reversed(lines[1:])]) + "\n")
    cli.run(_args(paths, "--json"))
    assert json.loads(capsys.readouterr().out)["report"] == in_order


def test_input_format_overrides_the_extension(stale, tmp_path, capsys):
    paths, _ = stale
    paths["candidate"] = paths["candidate"].rename(tmp_path / "candidate.txt")
    assert cli.run(_args(paths, "--input-format", "jsonl")) == 10
    assert "Verdict: THRESHOLD STALE" in capsys.readouterr().out


def test_an_unknown_extension_is_a_usage_error_naming_the_file_and_the_override(stale, tmp_path, capsys):
    paths, _ = stale
    paths["candidate"] = paths["candidate"].rename(tmp_path / "candidate.txt")
    assert cli.run(_args(paths)) == 2
    err = capsys.readouterr().err
    assert "candidate.txt" in err and "--input-format" in err and ".npz" in err and ".jsonl" in err


def test_an_unknown_extension_with_json_still_prints_one_object(stale, tmp_path, capsys):
    paths, _ = stale
    paths["candidate"] = paths["candidate"].rename(tmp_path / "candidate.csv")
    assert cli.run(_args(paths, "--json")) == 2
    assert json.loads(capsys.readouterr().out)["exit_code"] == 2


def test_ids_the_reference_lacks_are_a_pairing_error(stale, tmp_path, capsys):
    paths, _ = stale
    text = paths["candidate"].read_text().replace('"id": "q0"', '"id": "other"')
    paths["candidate"] = tmp_path / "renamed.jsonl"
    paths["candidate"].write_text(text)
    assert cli.run(_args(paths)) == 1
    assert "pairing error" in capsys.readouterr().err


def test_labels_that_differ_are_a_pairing_error(stale, tmp_path, capsys):
    paths, _ = stale
    rows = [json.loads(line) for line in paths["candidate"].read_text().splitlines()]
    rows[1]["label"] = 1 - rows[1]["label"]
    paths["candidate"] = tmp_path / "relabelled.jsonl"
    paths["candidate"].write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert cli.run(_args(paths)) == 1
    assert "labels differ" in capsys.readouterr().err


def test_a_malformed_file_is_a_tool_failure_naming_its_line(stale, tmp_path, capsys):
    paths, _ = stale
    paths["candidate"] = tmp_path / "broken.jsonl"
    paths["candidate"].write_text('{"id": "q0", "score": NaN, "label": 0}\n')
    assert cli.run(_args(paths)) == 1
    assert "broken.jsonl line 1" in capsys.readouterr().err


def test_an_npz_reference_takes_a_jsonl_candidate(stale, tmp_path, capsys):
    paths, d = stale
    ids = np.array([f"q{i}" for i in range(len(d.labels))])
    hulder.Snapshot(ids=ids, labels=d.labels, scores=d.reference, metadata=META).save(tmp_path / "reference.npz")
    paths["reference"] = tmp_path / "reference.npz"
    assert cli.run(_args(paths)) == 10
