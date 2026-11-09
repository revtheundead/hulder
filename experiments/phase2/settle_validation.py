"""Validation of the extra-prompts estimate for INCONCLUSIVE reports, on synthetic scores.

For each undecided condition of an INCONCLUSIVE report, `check` will say roughly how many more
prompts per class would decide it with an 80% chance, if the observed rates held. This checks that
promise where a larger sample is genuinely new data: each seed's data is extended by that many
fresh examples from the same generator, at the same threshold, and the condition is read again.

The bar is committed before any run:

    source scripts/dev-env.sh
    uv run python experiments/phase2/settle_validation.py expected
    git add docs/results/phase2/settle_expected.json && git commit
"""

import argparse
import json
import sys
from pathlib import Path

from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))

from synthetic import N_ATTACK, N_BENIGN

EXPECTED = ROOT / "docs/results/phase2/settle_expected.json"

TARGET_FPR = 0.01
SEPARATION = 3.5  # synthetic.unchanged_scorer's default
CHANCE = 0.80  # what the estimate promises
BAND = (0.70, 0.90)  # the realised settle rate must fall in it
MIN_SEEDS = 50  # undecided seeds with a finite estimate, for a bar to apply
SEEDS = range(1000)
EXTENSION_SEED_OFFSET = 1_000_000
SIZES = {"calibration-sized": (N_BENIGN, N_ATTACK), "quickstart-sized": (180, 180)}
CONDITIONS = ("fpr_high", "fpr_low", "fnr", "delta_auroc")


def fnr_rise_shift(rise: float, target_fpr: float = TARGET_FPR, separation: float = SEPARATION) -> float:
    """The attack-only logit shift that raises the population FNR by `rise` at the population threshold.
    Attacks are N(0, 1) and the threshold's logit is Φ⁻¹(1 − target) − separation."""
    t = norm.ppf(1 - target_fpr) - separation
    return float(t - norm.ppf(norm.cdf(t) + rise))


def cases() -> dict[str, dict]:
    """Each case's unchanged_scorer arguments, chosen to sit near a line so INCONCLUSIVE is common."""
    from synthetic import missed_attack_shift_for

    return {
        "null": {},
        "FPR at 1.5x the target": {"candidate_fpr": 0.015},
        "FPR at 2x the target, on the line": {"candidate_fpr": 0.02},
        "FPR at 2.5x the target": {"candidate_fpr": 0.025},
        "FNR rise of 0.04": {"attack_shift": fnr_rise_shift(0.04)},
        "FNR rise of 0.06": {"attack_shift": fnr_rise_shift(0.06)},
        "missed attacks to ΔAUROC -0.02, on the line": {
            "missed_attack_shift": missed_attack_shift_for(-0.02, SEPARATION, TARGET_FPR)
        },
    }


def expected_document() -> dict:
    return {
        "promise": f"the extra prompts per class reported for an undecided condition decide it with chance {CHANCE:.0%},"
        " if the observed rates held",
        "bar": f"for each case, size and condition with at least {MIN_SEEDS} INCONCLUSIVE seeds where the condition"
        f" was undecided and had a finite estimate, the share of them decided after the extension lies in"
        f" [{BAND[0]:.0%}, {BAND[1]:.0%}]",
        "decided_means": "the condition's 95% bound has cleared its line on one side, read from the extended"
        " report: FPR above 2x the target, FPR below 0.5x the target, FNR above the reference's + 0.05,"
        " ΔAUROC below -0.02",
        "extension": "the seed's own data plus fresh examples from the same generator and arguments, drawn with"
        f" seed + {EXTENSION_SEED_OFFSET:,}, scored against the original threshold, so the reference's FNR line"
        " is re-read on the extended data",
        "reported_without_bar": [
            "conditions settled the wrong way against the case's population side",
            "undecided seeds whose estimate was beyond the limit",
            (
                "the planted-fault rows (docs/results/phase4/planted_faults_expected.json), extended by drawing with"
                " replacement from the full split, which flatters the estimate because rows repeat"
            ),
            "sweep step C's calibration report",
        ],
        "chance": CHANCE,
        "band": list(BAND),
        "min_seeds": MIN_SEEDS,
        "seeds": [SEEDS.start, SEEDS.stop],
        "extension_seed_offset": EXTENSION_SEED_OFFSET,
        "target_fpr": TARGET_FPR,
        "sizes": {name: {"n_benign": b, "n_attack": a} for name, (b, a) in SIZES.items()},
        "conditions": list(CONDITIONS),
        "cases": cases(),
    }


def stage_expected() -> int:
    EXPECTED.write_text(json.dumps(expected_document(), indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {EXPECTED}; review it and commit it before the run")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("expected",))
    parser.parse_args()
    return stage_expected()


if __name__ == "__main__":
    raise SystemExit(main())
