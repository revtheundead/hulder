"""Bring-your-own scores: one JSONL file per serving config (docs/design.md §4)."""

import json

import numpy as np
import pytest

from hulder.gate.jsonl import load_scores_jsonl
from hulder.gate.pairing import PairingError

ROWS = [
    {"id": "a", "score": 0.1, "label": 0},
    {"id": "b", "score": 0.9, "label": 1},
    {"id": 3, "score": 0.2, "label": 0},
]


def _write(tmp_path, rows, name="run.jsonl", metadata=None, raw=None):
    path = tmp_path / name
    lines = ([json.dumps({"metadata": metadata})] if metadata is not None else []) + [json.dumps(r) for r in rows]
    path.write_text(raw if raw is not None else "\n".join(lines) + "\n")
    return path


def test_rows_become_a_snapshot_of_scores_in_file_order(tmp_path):
    snapshot = load_scores_jsonl(_write(tmp_path, ROWS))
    assert list(snapshot.ids) == ["a", "b", "3"]
    assert np.array_equal(snapshot.scores, [0.1, 0.9, 0.2]) and np.array_equal(snapshot.labels, [0, 1, 0])
    assert snapshot.activations is None and snapshot.metadata == {}


def test_an_optional_first_line_carries_the_runs_metadata(tmp_path):
    meta = {"model": "bge-small", "base_model": "bge-small", "quantization": "int8"}
    assert load_scores_jsonl(_write(tmp_path, ROWS, metadata=meta)).metadata == meta


def test_blank_lines_and_extra_keys_are_ignored(tmp_path):
    raw = '\n{"id": "a", "score": 0.1, "label": 0, "text": "hi"}\n\n{"id": "b", "score": 0.9, "label": 1}\n'
    assert list(load_scores_jsonl(_write(tmp_path, [], raw=raw)).ids) == ["a", "b"]


def test_order_aligns_the_rows_to_the_references_ids(tmp_path):
    snapshot = load_scores_jsonl(_write(tmp_path, ROWS), order=np.array(["3", "a", "b"]))
    assert list(snapshot.ids) == ["3", "a", "b"] and np.array_equal(snapshot.scores, [0.2, 0.1, 0.9])


def test_ids_that_differ_from_the_references_raise_a_pairing_error_naming_both_sides(tmp_path):
    with pytest.raises(
        PairingError, match=r"1 id the reference lacks, e\.g\. 'b'.*lacks 1 of the reference's, e\.g\. 'c'"
    ):
        load_scores_jsonl(_write(tmp_path, ROWS), order=np.array(["a", "3", "c"]))


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ('{"id": "a", "score": 0.1, "label": 0}\nnot json\n', r"run\.jsonl line 2: not valid JSON"),
        ('{"id": "a", "label": 0}\n', r"line 1: no 'score'"),
        ('{"id": "a", "score": "0.1", "label": 0}\n', r"line 1: 'score' must be a number"),
        ('{"id": "a", "score": true, "label": 0}\n', r"line 1: 'score' must be a number"),
        ('{"id": "a", "score": NaN, "label": 0}\n', r"line 1: 'score' is nan; scores must be finite"),
        ('{"id": "a", "score": Infinity, "label": 0}\n', r"line 1: 'score' is inf; scores must be finite"),
        ('{"id": "a", "score": 0.1, "label": 2}\n', r"line 1: 'label' must be 0 \(benign\) or 1 \(attack\)"),
        ('{"id": "a", "score": 0.1, "label": false}\n', r"line 1: 'label' must be 0 \(benign\) or 1 \(attack\)"),
        ('{"id": null, "score": 0.1, "label": 0}\n', r"line 1: 'id' must be a string or an integer"),
        ("[1, 2]\n", r"line 1: each line must be a JSON object"),
        ('{"id": "a", "score": 0.1, "label": 0}\n{"metadata": {}}\n', r"line 2: a metadata line must come first"),
        ('{"metadata": [1]}\n{"id": "a", "score": 0.1, "label": 0}\n', r"line 1: 'metadata' must be an object"),
        ("\n\n", r"run\.jsonl: no rows"),
    ],
)
def test_malformed_rows_raise_naming_the_file_and_line(tmp_path, raw, message):
    with pytest.raises(ValueError, match=message):
        load_scores_jsonl(_write(tmp_path, [], raw=raw))


def test_repeated_ids_raise(tmp_path):
    with pytest.raises(ValueError, match="repeats 1 ids"):
        load_scores_jsonl(_write(tmp_path, [*ROWS, ROWS[0]]))
