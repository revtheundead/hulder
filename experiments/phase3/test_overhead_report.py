"""The overhead table's score report on synthetic scores, through the real hulder.check_snapshots."""

import numpy as np
import overhead_report as o
import pytest

import hulder

N = 600
LABELS = np.array([0] * (N // 2) + [1] * (N // 2))
THRESHOLD = 0.5


def reference_scores():
    rng = np.random.default_rng(0)
    benign = rng.uniform(0.0, 0.45, N // 2)
    attack = rng.uniform(0.55, 1.0, N // 2)
    return np.concatenate([benign, attack])


def snapshot(scores):
    ids = np.array([f"p{i}" for i in range(N)])
    return hulder.Snapshot(ids=ids, labels=LABELS, scores=scores, metadata={"model": "m", "prefix_caching": False})


def stage(scores, prefix_caching=False, **fields):
    return {"ids": [f"p{i}" for i in range(N)], "scores": scores, "prefix_caching": prefix_caching, **fields}


def report(scores, **fields):
    ref = reference_scores()
    return o.stage_report(snapshot(ref), snapshot(ref + 1e-4), stage(scores, **fields), THRESHOLD, 0.01)


def test_an_unchanged_stage_passes_with_no_flips():
    result = report(reference_scores())
    assert result["verdict"] == "PASS"
    assert result["decision_flips"] == 0
    assert result["without_score"] == 0


def test_caching_on_is_listed_as_a_change_under_test_and_doesnt_warn():
    result = report(reference_scores(), prefix_caching=True)
    assert result["changes"] == [["prefix_caching", False, True]]
    assert not [w for w in result["warnings"] if "prefix_caching" in w]


def test_scores_moved_up_past_the_threshold_are_threshold_stale():
    moved = reference_scores()
    moved[: N // 2] = np.clip(moved[: N // 2] + 0.2, 0, 1)  # about half the benign cross 0.5
    result = report(moved)
    assert result["verdict"] == "THRESHOLD STALE"
    assert result["decision_flips"] > 0


def test_rows_without_a_score_are_left_out_of_all_three_runs_and_counted():
    scores = reference_scores()
    scores[0] = np.nan
    result = report(scores)
    assert result["without_score"] == 1
    assert result["rows"] == N - 1
    assert result["verdict"] == "PASS"


def test_a_stage_in_another_order_is_put_in_the_references_order_first():
    ref = reference_scores()
    order = np.random.default_rng(1).permutation(N)
    ids = [f"p{i}" for i in order]
    result = o.stage_report(
        snapshot(ref),
        snapshot(ref + 1e-4),
        {"ids": ids, "scores": ref[order], "prefix_caching": False},
        THRESHOLD,
        0.01,
    )
    assert result["decision_flips"] == 0


@pytest.mark.parametrize(
    ("verdicts", "expected"),
    [
        (["PASS", "PASS"], "holds at this threshold on this set"),
        (["PASS", "INCONCLUSIVE"], "needs its own calibration"),
        (["THRESHOLD STALE", "PASS"], "needs its own calibration"),
        ([], "not measured"),
    ],
)
def test_the_caching_outcome_needs_every_stage_to_pass(verdicts, expected):
    assert o.caching_outcome(verdicts) == expected
