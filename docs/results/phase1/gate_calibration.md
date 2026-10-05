# The gate's own calibration on synthetic data (Phase 1)

How often `hulder check` fails to PASS an unchanged scorer, and how often it calls THRESHOLD STALE when the candidate's false-positive rate has risen. Full numbers in `gate_calibration.json`. Design in the verdict rule.

Reproduce from the repo root:

```
source scripts/dev-env.sh && uv run python experiments/phase1/gate/calibration.py
```

1,000 seeds per row, 1,191 benign and 910 attack (the S-Labs calibration set's sizes), a 1% target, and a threshold fitted on the reference's own benign scores. The generators are in `tests/synthetic.py`. The candidate's FPR shift moves every logit by the same amount, which leaves AUROC unchanged and lowers FNR.

**Status: partial rule.** At the time of this run `check` implemented only the first STALE condition (candidate FPR lower bound > 2 × target), and every other case fell through to PASS. Under a pure upward shift no other condition can fire, so the STALE column is final. The PASS column is not. Once the PASS rule (FPR upper bound ≤ 2 × target) lands, a case that is neither STALE nor PASS becomes INCONCLUSIVE.

| Case | THRESHOLD STALE [95% CI] | PASS now | PASS under the full rule (projected) |
|---|---|---|---|
| Null (logit noise sd 1e-3) | 0.0% [0.0, 0.4] | 100% | not yet measured |
| Noise at the fresh-engine noise level | 0.0% [0.0, 0.4] | 100% | not yet measured |
| Candidate FPR 1.5% | 0.0% [0.0, 0.4] | 100% | about 22% |
| Candidate FPR 2% | 2.1% [1.3, 3.2] | 97.9% | about 2% |
| Candidate FPR 3% | 57.0% [53.9, 60.1] | 43.0% | about 0% |
| Candidate FPR 5% | 99.9% [99.4, 100] | 0.1% | about 0% |

The projected PASS share is P(k ≤ 14) for k ~ Binomial(1,191, true FPR), since 14 false positives is the most whose Clopper-Pearson upper bound stays ≤ 2%. It ignores the small sampling error in the fitted threshold. The remeasured rows replace it once the PASS rule is implemented.

**Noise at the fresh-engine noise level.** Two fresh fp16 engines (`fp16` vs `fp16_run1`), main probe, calibration split: 44% of scores bit-identical, logit differences with sd 3.65e-3, mean |Δscore| 7.8e-5, 1 decision flip of 2,101. The synthetic version is zero-inflated N(0, 4.9e-3) in the logit, which matches the mean and p99 of |Δlogit| (2.2e-3 and 1.16e-2 against a measured 2.17e-3 and 1.15e-2).

**Where it loses.** At this sample size a tripled FPR, which triples escalations to the judge, is called STALE only 57% of the time. The rest is projected to come out INCONCLUSIVE rather than PASS, so the gate doesn't wave it through, but neither does it name the cause. A candidate at 1.5% FPR, inside the tolerance band, is projected to PASS only about 22% of the time and to come out INCONCLUSIVE otherwise.
