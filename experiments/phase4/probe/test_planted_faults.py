"""The pure part of the planted-fault control. CPU only, no captured data."""

import subprocess

import numpy as np
import planted_faults as p
import pytest

LABELS = np.array([0, 0, 0, 0, 1, 1, 1, 1])
LOGITS = np.array([-3.0, -2.0, -1.0, 0.5, 0.0, 1.0, 2.0, 3.0])


def population(**over):
    return {"fpr": 0.01, "fnr": 0.2, "reference_fnr": 0.2, "delta_auroc": 0.0, **over}


@pytest.mark.parametrize(
    ("values", "verdict"),
    [
        ({}, "PASS"),
        ({"fpr": 0.02}, "PASS"),
        ({"fpr": 0.021}, "THRESHOLD STALE"),
        ({"fpr": 0.004}, "THRESHOLD STALE"),
        ({"fnr": 0.25}, "PASS"),
        ({"fnr": 0.26}, "THRESHOLD STALE"),
        ({"delta_auroc": -0.02}, "PASS"),
        ({"delta_auroc": -0.021}, "SCORER DEGRADED"),
        ({"delta_auroc": -0.05, "fpr": 0.2}, "SCORER DEGRADED"),
    ],
)
def test_the_expected_verdict_reads_the_rule_on_population_values(values, verdict):
    assert p.expected_verdict(population(**values), target_fpr=0.01) == verdict


@pytest.mark.parametrize(
    ("values", "gross"),
    [
        ({"delta_auroc": -0.05}, True),
        ({"delta_auroc": -0.049}, False),
        ({"fpr": 0.05}, True),
        ({"fpr": 0.049}, False),
        ({"fnr": 0.35}, True),
        ({"fnr": 0.349}, False),
        ({"fpr": 0.2, "delta_auroc": -0.03}, False),
        ({"fpr": 0.2, "delta_auroc": -0.016}, False),
        ({}, False),
    ],
)
def test_a_fault_is_gross_only_far_from_every_line(values, gross):
    assert p.is_gross(population(**values)) is gross


@pytest.mark.parametrize(
    ("values", "either"),
    [
        ({"fpr": 0.12, "delta_auroc": -0.0255}, True),
        ({"fnr": 0.35, "delta_auroc": -0.03}, True),
        ({"fpr": 0.03, "delta_auroc": -0.03}, False),
        ({"fpr": 0.12, "delta_auroc": -0.01}, False),
        ({"delta_auroc": -0.05, "fpr": 0.12}, False),
    ],
)
def test_a_fault_past_two_lines_near_the_auroc_one_is_gross_on_either_verdict(values, either):
    assert p.is_gross_either_verdict(population(**values)) is either


def test_population_values_come_from_the_rows_at_the_threshold():
    candidate = LOGITS + np.array([0, 0, 0, 0, 0, -2.0, 0, 0])
    values = p.population_values(LOGITS, candidate, LABELS, threshold=0.5)
    assert values["fpr"] == 0.25  # logit 0.5 is the one benign above score 0.5
    assert values["reference_fnr"] == 0.25  # logit 0.0 isn't strictly above it
    assert values["fnr"] == 0.5
    assert values["delta_auroc"] == pytest.approx(-0.09375)


def test_a_uniform_shift_moves_every_logit_and_keeps_the_order():
    assert p.shift(LOGITS, 1.5).tolist() == [-1.5, -0.5, 0.5, 2.0, 1.5, 2.5, 3.5, 4.5]


def test_an_attack_only_shift_leaves_benign_rows_alone():
    out = p.attack_shift(LOGITS, LABELS, -1.0)
    assert out.tolist() == [-3.0, -2.0, -1.0, 0.5, -1.0, 0.0, 1.0, 2.0]


def test_a_sign_flip_inverts_the_logits():
    assert p.sign_flip(LOGITS).tolist() == (-LOGITS).tolist()


def test_a_scramble_permutes_only_the_chosen_rows():
    order = np.array([7, 0, 3, 1, 2, 4, 5, 6])
    out = p.scramble(LOGITS, order, n_rows=3, seed=0)
    assert sorted(out[[7, 0, 3]].tolist()) == sorted(LOGITS[[7, 0, 3]].tolist())
    assert out[[1, 2, 4, 5, 6]].tolist() == LOGITS[[1, 2, 4, 5, 6]].tolist()


def test_a_scramble_is_sized_to_land_close_to_its_target_delta_auroc():
    rng = np.random.default_rng(0)
    labels = np.array([0] * 500 + [1] * 500)
    logits = np.concatenate([rng.normal(-2, 1, 500), rng.normal(2, 1, 500)])
    out = p.scramble_to(logits, labels, target_delta=-0.05, seed=0)
    assert p.auroc(out, labels) - p.auroc(logits, labels) == pytest.approx(-0.05, abs=0.002)


def test_noise_is_sized_by_its_mean_absolute_change():
    out = p.noise(np.zeros(200_000), mean_abs=0.184, seed=0)
    assert np.abs(out).mean() == pytest.approx(0.184, abs=0.001)


def test_residuals_are_drawn_within_each_class():
    residuals = {0: np.array([1.0]), 1: np.array([5.0])}
    assert p.with_residuals(np.zeros(8), LABELS, residuals, seed=0).tolist() == [1.0] * 4 + [5.0] * 4


def test_a_resample_keeps_each_class_size_and_keeps_rows_paired():
    rows = p.resample_rows(LABELS, seed=3)
    assert LABELS[rows].tolist() == sorted(LABELS.tolist())
    assert p.resample_rows(LABELS, seed=3).tolist() == rows.tolist()


def test_a_refit_threshold_flags_at_most_the_target_share_of_benign():
    benign = np.linspace(0, 1, 1191)
    assert (benign > p.refit_threshold(benign, 0.01)).sum() == 11


def test_the_bar_needs_both_the_detection_and_the_false_alarm_rates():
    assert p.meets_bar("gross", {"expected_rate": 0.95, "non_pass_rate": 1.0}) is True
    assert p.meets_bar("gross", {"expected_rate": 0.949, "non_pass_rate": 1.0}) is False
    assert p.meets_bar("null", {"expected_rate": 0.95, "non_pass_rate": 0.05}) is True
    assert p.meets_bar("null", {"expected_rate": 0.94, "non_pass_rate": 0.06}) is False
    assert p.meets_bar("gross, either verdict", {"expected_rate": 0.1, "non_pass_rate": 0.95}) is True
    assert p.meets_bar("gross, either verdict", {"expected_rate": 1.0, "non_pass_rate": 0.94}) is False
    assert p.meets_bar("frozen", {"expected_rate": 0.0, "non_pass_rate": 1.0}) is None
    assert p.meets_bar("graded", {"expected_rate": 0.0, "non_pass_rate": 0.0}) is None


def test_a_verdict_tally_counts_all_four_with_exact_intervals():
    out = p.tally(["PASS"] * 9 + ["THRESHOLD STALE"], expected="PASS")
    assert out["counts"] == {"PASS": 9, "THRESHOLD STALE": 1, "SCORER DEGRADED": 0, "INCONCLUSIVE": 0}
    assert out["expected_rate"] == 0.9
    assert out["non_pass_rate"] == pytest.approx(0.1)
    assert out["expected_ci95"][0] < 0.9 < out["expected_ci95"][1]


def test_the_run_refuses_expected_verdicts_that_arent_committed(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    path = tmp_path / "expected.json"
    path.write_text("{}")
    assert p.uncommitted(path) is True  # untracked
    git("add", "expected.json")
    git("commit", "-qm", "x")
    assert p.uncommitted(path) is False
    path.write_text('{"a": 1}')
    assert p.uncommitted(path) is True
