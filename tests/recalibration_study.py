"""The recalibration study on synthetic scores with exact ground truth (docs/design.md §6).

Scores are sigmoid(logit), benign logits N(-3.5, 1) and attack logits N(0, 1), as in synthetic.py.
Three score populations: continuous, rounded to 2 decimals (heavy ties near a 1% threshold), and
float16 (the precision snapshots store activations in). For each, a threshold's true FPR and FNR
are computed exactly, so each fitted threshold is judged against the truth, not a sample.

Bounds are fixed in BOUNDS and checked by bounds_hold before the study is run
(tests/test_recalibration_study.py, experiments/phase2/recalibration.py).
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.special import expit, logit
from scipy.stats import binomtest, norm
from synthetic import FRESH_ENGINE_NOISE, FRESH_ENGINE_P_ZERO, unchanged_scorer

import hulder
from hulder.gate.recalibrate import CONFIDENCE, binomial_tail_threshold, conformal_threshold

SEPARATION = 3.5
SIZES = (180, 360, 1191, 5000)
TARGETS = (0.01, 0.05)
CASES = ("continuous", "rounded_2dp", "float16")
# Attacks per benign in each study cell, as in the S-Labs calibration set (910 / 1,191).
ATTACK_RATIO = 910 / 1191
# Promotion round trip: (benign, attack, target, candidate FPR at the old threshold).
PROMOTIONS = ((180, 180, 0.05, 0.25), (1191, 910, 0.01, 0.05))
NOT_CALIBRATED = "wasn't calibrated on this snapshot"

# Pre-registered 2026-10-15, before any run. SE is the Monte Carlo standard error at the seeds used.
BOUNDS = {
    "binomial_tail_exceedance": "P(true FPR > target) <= 0.05 + 3 SE, every cell",
    "conformal_mean_fpr": "mean true FPR <= target + 3 SE, every cell; >= target - 1/(n+1) - 3 SE, continuous cells",
    "binomial_tail_refusal": "refuses exactly when (1 - target)**n > 0.05: only n = 180 at 1%",
    "promotion_non_pass": "non-PASS <= 10% after promotion, each round-trip cell",
    "promotion_not_calibrated_warning": "the 'not calibrated on this snapshot' warning in <= 1%, each round-trip cell",
}


@dataclass(frozen=True)
class Population:
    """Benign and attack score distributions with exact rates at any threshold."""

    name: str
    # Sorted values scores snap to, or None for continuous scores.
    grid: np.ndarray | None

    def sample(self, rng: np.random.Generator, n_benign: int, n_attack: int) -> tuple[np.ndarray, np.ndarray]:
        benign = expit(rng.normal(-SEPARATION, 1, n_benign))
        attack = expit(rng.normal(0, 1, n_attack))
        return self._snap(benign), self._snap(attack)

    def rates(self, threshold: float) -> tuple[float, float]:
        """True (FPR, FNR) when scores strictly above threshold are flagged."""
        edge = threshold if self.grid is None else self._upper_edge(threshold)
        x = logit(np.clip(edge, 0, 1))
        return float(norm.sf(x + SEPARATION)), float(norm.cdf(x))

    def _snap(self, scores: np.ndarray) -> np.ndarray:
        if self.grid is None:
            return scores
        mids = (self.grid[1:] + self.grid[:-1]) / 2
        return self.grid[np.searchsorted(mids, scores)]

    def _upper_edge(self, threshold: float) -> float:
        """The continuous score above which a sample snaps to a grid value above threshold."""
        i = int(np.searchsorted(self.grid, threshold, side="right"))
        if i == len(self.grid):
            return 1.0
        return float((self.grid[i - 1] + self.grid[i]) / 2) if i else 0.0


POPULATIONS = {
    "continuous": Population("continuous", None),
    "rounded_2dp": Population("rounded_2dp", np.arange(101) / 100),
    "float16": Population("float16", np.arange(0x3C01, dtype=np.uint16).view(np.float16).astype(np.float64)),
}


def plug_in_threshold(benign: np.ndarray, target_fpr: float) -> float:
    """v0.1's fit (experiments/phase1/probe/train.py), for context only."""
    return float(np.quantile(benign, 1 - target_fpr, method="higher"))


METHODS: dict[str, Callable] = {
    "binomial_tail": lambda b, a, t: binomial_tail_threshold(b, t, attack_scores=a).threshold,
    "conformal": lambda b, a, t: conformal_threshold(b, t),
    "plug_in": lambda b, a, t: plug_in_threshold(b, t),
}


def study_cell(case: str, n: int, target_fpr: float, seeds: range) -> dict:
    """Per method: how often it gives no threshold, and over the seeds where it does, the mean
    true FPR and FNR, and the share whose true FPR exceeds the target, each with its SE or CI."""
    population = POPULATIONS[case]
    n_attack = round(n * ATTACK_RATIO)
    rates: dict[str, list] = {m: [] for m in METHODS}
    refused = dict.fromkeys(METHODS, 0)
    for seed in seeds:
        benign, attack = population.sample(np.random.default_rng((seed, n)), n, n_attack)
        for method, fit in METHODS.items():
            threshold = fit(benign, attack, target_fpr)
            if threshold is None:
                refused[method] += 1
            else:
                rates[method].append(population.rates(threshold))
    out = {}
    for method in METHODS:
        entry: dict = {"refused": refused[method] / len(seeds)}
        if rates[method]:
            fpr, fnr = np.array(rates[method]).T
            k = int((fpr > target_fpr).sum())
            ci = binomtest(k, len(fpr)).proportion_ci(method="exact")
            entry |= {
                "exceedance": k / len(fpr),
                "exceedance_ci": [float(ci.low), float(ci.high)],
                "exceedance_se": math.sqrt((1 - CONFIDENCE) * CONFIDENCE / len(fpr)),
                "mean_fpr": float(fpr.mean()),
                "mean_fpr_se": float(fpr.std(ddof=1) / math.sqrt(len(fpr))) if len(fpr) > 1 else 0.0,
                "mean_fnr": float(fnr.mean()),
            }
        out[method] = entry
    return out


def promotion_round_trip(n_benign: int, n_attack: int, target_fpr: float, candidate_fpr: float, seeds: range) -> dict:
    """Recalibrate a stale candidate, promote it with that threshold, and check an unchanged
    rerun against it. The rerun differs by fresh-engine noise only, so the check should PASS
    without saying the threshold wasn't calibrated on the new reference."""
    verdicts, warned, refused = [], 0, 0
    for seed in seeds:
        d = unchanged_scorer(seed, target_fpr, n_benign, n_attack, candidate_fpr=candidate_fpr)
        benign = d.labels == 0
        fit = binomial_tail_threshold(d.candidate[benign], target_fpr, attack_scores=d.candidate[~benign])
        if fit.threshold is None:
            refused += 1
            continue
        rng = np.random.default_rng((seed, 2))
        noise = rng.normal(0, FRESH_ENGINE_NOISE, d.candidate.shape) * (
            rng.random(d.candidate.shape) >= FRESH_ENGINE_P_ZERO
        )
        rerun = expit(logit(d.candidate) + noise)
        report = hulder.check(d.candidate, rerun, d.labels, threshold=fit.threshold, target_fpr=target_fpr)
        verdicts.append(report.verdict.value)
        warned += any(NOT_CALIBRATED in w for w in report.warnings)
    checked = len(verdicts)
    return {
        "checked": checked,
        "refused": refused,
        "non_pass": sum(v != "PASS" for v in verdicts) / checked,
        "verdicts": {v.value: verdicts.count(v.value) / checked for v in hulder.Verdict},
        "not_calibrated_warning": warned / checked,
    }


def bounds_hold(cells: dict, promotions: dict) -> dict[str, bool]:
    """Each pre-registered bound against the results. cells is keyed (case, n, target)."""
    held = {key: True for key in BOUNDS}
    for (case, n, target), cell in cells.items():
        bt, cf = cell["binomial_tail"], cell["conformal"]
        if "exceedance" in bt:
            held["binomial_tail_exceedance"] &= bt["exceedance"] <= 1 - CONFIDENCE + 3 * bt["exceedance_se"]
        refuse_expected = (1 - target) ** n > 1 - CONFIDENCE
        held["binomial_tail_refusal"] &= bt["refused"] == (1.0 if refuse_expected else 0.0)
        if "mean_fpr" in cf:
            held["conformal_mean_fpr"] &= cf["mean_fpr"] <= target + 3 * cf["mean_fpr_se"]
            if case == "continuous":
                held["conformal_mean_fpr"] &= cf["mean_fpr"] >= target - 1 / (n + 1) - 3 * cf["mean_fpr_se"]
    for result in promotions.values():
        held["promotion_non_pass"] &= result["non_pass"] <= 0.10
        held["promotion_not_calibrated_warning"] &= result["not_calibrated_warning"] <= 0.01
    return held
