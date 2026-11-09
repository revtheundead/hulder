"""Sweep step B's report: its input and stop-rule checks. CPU only."""

import hashlib

import pytest

pytest.importorskip("sklearn")
import sweep_b_report as r


def test_files_match_names_each_file_missing_or_changed(tmp_path):
    (tmp_path / "a.npz").write_bytes(b"a")
    (tmp_path / "b.npz").write_bytes(b"changed")
    expected = {n: hashlib.sha256(n[0].encode()).hexdigest() for n in ("a.npz", "b.npz", "c.npz")}
    assert r.files_match(tmp_path, expected) == ["b.npz", "c.npz"]


def test_b1_names_test_rows_missing_from_b_or_fed_other_tokens():
    kept = {"t1": [1, 2], "t2": [3, 4], "t3": [5]}
    b = {"t1": [1, 2], "t2": [3, 9], "x": [7]}
    assert r.tokens_match(b, kept, ["t1", "t2", "t3"]) == ["t2", "t3"]
    assert r.tokens_match(kept, kept, ["t1", "t2", "t3"]) == []


def test_b2_passes_only_the_bf16_majority_cluster():
    recorded = {"h20": {"cluster": "2.0"}, "h21": {"cluster": "2.1"}}
    assert r.canary_stop("h20", recorded) == (None, {"cluster": "2.0"})
    assert r.canary_stop("h21", recorded)[0] == "B2: B's canary is 2.1, not cluster 2.0"
    assert r.canary_stop("new", recorded)[0] == "B2: B's canary is unknown path, not cluster 2.0"
