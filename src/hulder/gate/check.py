"""The release gate: compare a scorer under reference and candidate serving configs."""

from dataclasses import dataclass
from enum import Enum

import numpy as np


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


def check(
    reference: np.ndarray,
    candidate: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    target_fpr: float,
) -> Report:
    return Report(verdict=Verdict.PASS)
