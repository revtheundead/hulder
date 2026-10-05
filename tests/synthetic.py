"""Synthetic paired scores with known ground truth, for testing the gate (README §7).

Scores are sigmoid(logit). Benign logits are N(-separation, 1) and attack logits N(0, 1),
so the scorer's AUROC is Phi(separation / sqrt(2)). The candidate sees the same examples
as the reference, plus run-to-run noise in the logit.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import expit
from scipy.stats import norm

# Sizes of the S-Labs calibration set (docs/results/phase1/probe_splits.json).
N_BENIGN = 1191
N_ATTACK = 910


@dataclass
class Paired:
    reference: np.ndarray
    candidate: np.ndarray
    labels: np.ndarray
    threshold: float


def unchanged_scorer(
    seed: int,
    target_fpr: float = 0.01,
    n_benign: int = N_BENIGN,
    n_attack: int = N_ATTACK,
    separation: float = 3.5,
    noise: float = 1e-3,
) -> Paired:
    """A null case: the candidate differs from the reference by noise only.

    The threshold is the population (1 - target_fpr) quantile of benign scores, so it is
    exactly right for the reference's distribution rather than fitted to this sample.
    With the defaults the population FNR is about 12% and the AUROC about 0.993.
    """
    rng = np.random.default_rng(seed)
    labels = np.r_[np.zeros(n_benign, int), np.ones(n_attack, int)]
    logits = np.r_[rng.normal(-separation, 1, n_benign), rng.normal(0, 1, n_attack)]
    noisy = logits + rng.normal(0, noise, logits.shape)
    threshold = float(expit(-separation + norm.ppf(1 - target_fpr)))
    return Paired(expit(logits), expit(noisy), labels, threshold)
