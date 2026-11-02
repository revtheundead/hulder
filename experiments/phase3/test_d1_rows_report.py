"""The rows report on small synthetic captures: rows that didn't move are kept apart from those that did. CPU only."""

import numpy as np
import pytest

import hulder

pytest.importorskip("sklearn")  # exp2_report's train_qwen3
import d1_rows_report as r

PROBE = hulder.Probe(weights=np.ones(2), bias=0.0, mean=np.zeros(2), std=np.ones(2), metadata={})
REFERENCE = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]], dtype=np.float16)


def test_a_stages_mean_over_changed_rows_ignores_the_rows_that_didnt_move():
    other = REFERENCE.copy()
    other[0] = [1.0, 0.0]  # logit 0 to 1, so |Δscore| is sigmoid(1) - 0.5, or 0.2310586
    out = r.row_shift(PROBE, REFERENCE, other)
    assert out["rows_bit_identical"] == 3
    assert out["changed_mean_abs_dscore"] == pytest.approx(0.2310586)
    assert out["mean_abs_dscore"] == pytest.approx(0.2310586 / 4)


def test_a_capture_identical_to_the_reference_has_no_changed_rows_to_average():
    out = r.row_shift(PROBE, REFERENCE, REFERENCE.copy())
    assert out["rows_bit_identical"] == 4
    assert out["mean_abs_dscore"] == 0
    assert out["changed_mean_abs_dscore"] == 0


def test_identical_rows_counts_every_pair():
    other = REFERENCE.copy()
    other[1] = [1.0, 0.0]
    counts = r.identical_rows({"a": REFERENCE, "b": other})
    assert counts == {"a": {"a": 4, "b": 3}, "b": {"a": 3, "b": 4}}
