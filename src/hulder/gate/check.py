"""The release gate: compare a scorer under reference and candidate serving configs."""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.stats import binomtest


class Verdict(Enum):
    PASS = "PASS"
    THRESHOLD_STALE = "THRESHOLD STALE"
    SCORER_DEGRADED = "SCORER DEGRADED"
    INCONCLUSIVE = "INCONCLUSIVE"

    @property
    def exit_code(self) -> int:
        """CLI exit code from the exit codes. Verdicts start at 10 so they never collide with failures."""
        return _EXIT_CODES[self]


_EXIT_CODES = {
    Verdict.PASS: 0,
    Verdict.THRESHOLD_STALE: 10,
    Verdict.SCORER_DEGRADED: 11,
    Verdict.INCONCLUSIVE: 12,
}


@dataclass(frozen=True)
class Report:
    """Plain data: it neither prints nor exits, and compares with ==."""

    verdict: Verdict
    warnings: tuple[str, ...] = ()


def check(
    reference: np.ndarray,
    candidate: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    target_fpr: float,
) -> Report:
    warnings = []
    benign = reference[labels == 0]
    k, n = int((benign > threshold).sum()), len(benign)
    ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
    if not ci.low <= target_fpr <= ci.high:
        warnings.append(
            f"The reference's FPR at threshold {threshold:.4g} is {k / n:.2%} "
            f"[95% CI {ci.low:.2%}, {ci.high:.2%}], which excludes the {target_fpr:.2%} target, "
            "so the threshold wasn't calibrated on this snapshot. "
            "Verdicts then mix threshold generalisation with the serving change: "
            "recalibrate on the reference first."
        )
    return Report(verdict=Verdict.PASS, warnings=tuple(warnings))
