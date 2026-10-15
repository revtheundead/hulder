# Recalibration on synthetic scores (v0.2)

How the recalibrated threshold behaves, what it costs, and what happens when a candidate is promoted with it. Full numbers are in `recalibration.json`. The method is in docs/design.md §6.

Reproduce from the repo root:

```
source scripts/dev-env.sh && uv run python experiments/phase2/recalibration.py
```

The run used commit `03b69c2` with a clean tree, 10,000 seeds per cell and 1,000 per promotion case. An earlier run at `84ca1cb` gave the same numbers, without the attacks-needed measurement added since. The generators and bounds are in `tests/recalibration_study.py`, committed at `c300c52` before any run.

## Setup

- **Scores.** Benign logits are N(−3.5, 1) and attack logits N(0, 1), scored through a sigmoid, as in the v0.1 gate calibration. There are 910 attacks per 1,191 benign.
- **Three score populations.** Continuous scores, scores rounded to 2 decimals (heavy ties, fewer than 40 distinct values among 1,191 benign), and scores rounded to float16, the precision snapshots store activations in.
- **Exact truth.** Each population's true FPR and FNR at any threshold are computed in closed form, so every fitted threshold is judged against the truth rather than a second sample. The formulas agree with 2-million-sample estimates (`tests/test_recalibration_study.py`).
- **Methods.**
  - *Binomial-tail*, which `check` uses. It allows the largest number j of benign scores above the threshold with P(Bin(n, target) ≤ j) ≤ 0.05, so the true FPR exceeds the target in at most 5% of calibration sets.
  - *Conformal*, the ⌈(n + 1)(1 − target)⌉-th smallest benign score, whose FPR is at most the target on average over calibration sets.
  - *Plug-in*, the benign scores' (1 − target) quantile, v0.1's training-time fit. It is shown for context only.

## Results

"Exceeds" is the share of seeds whose threshold's true FPR is above the target. FPR and FNR are true rates, averaged over seeds. The float16 rows match the continuous ones to within 0.001 in FPR and FNR and 0.01 in the share exceeding, so only the continuous and rounded rows are shown.

| Scores | Benign | Target | Binomial-tail exceeds | FPR | FNR | Conformal exceeds | FPR | FNR | Plug-in exceeds | FNR |
|---|---|---|---|---|---|---|---|---|---|---|
| continuous | 180 | 1% | refuses | | | 16.3% | 0.56% | 23.0% | 46.1% | 14.0% |
| continuous | 180 | 5% | 1.6% | 2.14% | 8.1% | 45.2% | 4.96% | 3.5% | 45.2% | 3.5% |
| continuous | 360 | 1% | 2.4% | 0.27% | 29.9% | 29.1% | 0.82% | 15.4% | 50.8% | 12.6% |
| continuous | 360 | 5% | 2.2% | 2.99% | 5.6% | 45.8% | 4.97% | 3.4% | 45.8% | 3.4% |
| continuous | 1,191 | 1% | 4.5% | 0.58% | 17.2% | 35.4% | 0.92% | 13.1% | 47.0% | 12.4% |
| continuous | 1,191 | 5% | 3.7% | 3.93% | 4.2% | 45.2% | 4.95% | 3.3% | 50.4% | 3.2% |
| continuous | 5,000 | 1% | 4.3% | 0.78% | 14.1% | 48.0% | 1.00% | 12.1% | 48.0% | 12.1% |
| continuous | 5,000 | 5% | 4.4% | 4.49% | 3.6% | 48.7% | 5.00% | 3.2% | 48.7% | 3.2% |
| rounded | 180 | 1% | refuses | | | 15.9% | 0.52% | 23.6% | 45.2% | 14.6% |
| rounded | 180 | 5% | 1.9% | 2.04% | 8.4% | 44.8% | 4.55% | 3.9% | 44.8% | 3.9% |
| rounded | 360 | 1% | 2.4% | 0.26% | 30.5% | 28.1% | 0.77% | 16.0% | 49.3% | 13.2% |
| rounded | 360 | 5% | 2.3% | 2.79% | 6.0% | 44.9% | 4.56% | 3.7% | 44.9% | 3.7% |
| rounded | 1,191 | 1% | 4.0% | 0.55% | 17.8% | 32.9% | 0.86% | 13.7% | 44.3% | 13.0% |
| rounded | 1,191 | 5% | 3.5% | 3.63% | 4.5% | 43.7% | 4.54% | 3.6% | 49.1% | 3.5% |
| rounded | 5,000 | 1% | 3.4% | 0.72% | 14.7% | 42.9% | 0.93% | 12.7% | 42.9% | 12.7% |
| rounded | 5,000 | 5% | 3.8% | 4.14% | 3.9% | 45.5% | 4.58% | 3.5% | 45.5% | 3.5% |

**Promotion round trip.** A candidate whose FPR rose to five times the target is recalibrated, promoted with its new threshold, and checked against an unchanged rerun of itself that differs by fresh-engine noise only.

| Benign | Attack | Target | PASS | INCONCLUSIVE | "Not calibrated" warning |
|---|---|---|---|---|---|
| 180 | 180 | 5% | 56.7% | 43.3% | 0% |
| 1,191 | 910 | 1% | 100% | 0% | 0% |

**Predicting it.** Added after the pre-registered run, with no bound. When the candidate's FNR interval at the new threshold is wider than the 0.05 FNR tolerance, an unchanged rerun can't PASS, so `Recalibration.attacks_needed` gives the attacks in all from which it could, and the report warns. Across the same 1,000 seeds per case:

| Benign | Attack | Warned, then INCONCLUSIVE | Not warned, then INCONCLUSIVE | Warned, then PASS | Attacks needed, median [max] |
|---|---|---|---|---|---|
| 180 | 180 | 416 of 433 | 17 | 3 of 567 | 224 [363] |
| 1,191 | 910 | 0 | 0 | 0 | none needed |

The 17 it misses are reruns whose noise moved an attack across the threshold.

## Bounds fixed before running

Four of five hold.

| Bound | Result |
|---|---|
| Binomial-tail exceeds the target in at most 5% + 3 SE (5.65%) of seeds, every cell | Holds. Highest 4.5% |
| Conformal mean true FPR at most the target + 3 SE in every cell, and at least target − 1/(n + 1) − 3 SE in continuous cells | Holds |
| Binomial-tail refuses exactly when (1 − target)ⁿ > 0.05, which here is 180 benign at 1% | Holds |
| After promotion, an unchanged rerun fails to PASS in at most 10% of seeds, each case | **Fails at 180 benign and 5%: 43.3%.** Holds at 1,191: 0% |
| After promotion, the "not calibrated on this snapshot" warning in at most 1%, each case | Holds. 0% in both. The v0.1 rule warned in 100% of the 180-benign case and 0% at 1,191, in a 400-seed run at `c300c52` that wasn't saved |

## What it shows

- **The guarantee holds, with ties and in float16.** Binomial-tail never exceeded its 5%, and runs below it at small n because the allowed count is an integer. Ties only make it more conservative.
- **Conformal and the plug-in quantile often exceed the target.** Conformal's FPR is right on average over calibration sets, which is its guarantee, but one calibration set gives a threshold above the target with probability 16% to 49%. The plug-in quantile has no guarantee and exceeds the target 43% to 51% of the time. The two give the same threshold in 12 of the 24 cells.
- **The price is FNR, and it is steep at small n.** At 1,191 benign and 1%, binomial-tail misses 17.2% of attacks against 12.4% for the plug-in quantile. At 360 benign and 1% it misses 29.9% against 12.6%, because 360 benign allow at most one false positive. At 5,000 benign the gap narrows to 14.1% against 12.1%. At 180 benign and 1% it refuses, and needs 299 benign in all.
- **Promotion fails to PASS at the demo's size, and the FNR condition is why.** In all 92 non-PASS seeds of a 200-seed rerun of the 180-benign case, the FPR and ΔAUROC conditions passed and only the FNR condition failed. The conservative threshold raises the reference's FNR to a median 7.2%. With 180 attacks the candidate's FNR upper bound then sits about 5 points above its point value, and the rule allows the reference's FNR + 0.05, so an unchanged rerun is INCONCLUSIVE 43% of the time. The FNR condition compares the candidate's upper bound with the reference's point value, so this happens whenever the FNR is several percent and attacks number in the low hundreds, recalibrated or not. Recalibration exposes it by raising the FNR. At 910 attacks the round trip PASSes every time.
- **The relaxed warning works.** A promoted binomial-tail threshold flags as many reference benign scores as their own binomial-tail fit, which `check` now accepts as calibrated on that snapshot. A threshold flagging fewer still warns.

## Where it loses

- At 180 benign, the demo's size, a recalibrated threshold can't be confirmed by an unchanged rerun 43% of the time (above). The report warns in 96% of those cases and says how many attacks would avoid it, typically about 220.
- At a 1% target, binomial-tail needs at least 299 benign examples, and below about 1,000 the FNR it costs is large.
- The guarantee is about traffic like the calibration set. The threshold is fitted on the same scores the verdict read, so `check` says to confirm it on a fresh candidate snapshot before relying on it.
- These are synthetic scores from one pair of normal distributions. Binomial-tail's FPR guarantee doesn't depend on the distribution, but the FNR costs do.
