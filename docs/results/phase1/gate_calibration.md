# The gate's own calibration on synthetic data (Phase 1)

How `hulder check` behaves on an unchanged scorer and on candidates whose false-positive rate has risen by known amounts. Full numbers in `gate_calibration.json`. Design in the verdict rule.

Reproduce from the repo root:

```
source scripts/dev-env.sh && uv run python experiments/phase1/gate/calibration.py
```

1,000 seeds per row, 1,191 benign and 910 attack (the S-Labs calibration set's sizes), a 1% target, and a threshold fitted on the reference's own benign scores. The generators are in `tests/synthetic.py`. The FPR shift moves every candidate logit by the same amount, which leaves AUROC unchanged and lowers FNR. The run used the complete the verdict rule rule (commit `51ffc46`). An earlier run with only the first STALE condition is in this file's git history.

| Case | PASS | THRESHOLD STALE | SCORER DEGRADED | INCONCLUSIVE | More benign asked for, median [IQR] |
|---|---|---|---|---|---|
| Null (logit noise sd 1e-3) | 100% [99.6, 100] | 0% | 0% | 0% | |
| Noise at the fresh-engine noise level | 100% [99.6, 100] | 0% | 0% | 0% | |
| Candidate FPR 1.5% | 20.5% [18.0, 23.1] | 0% | 0% | 79.5% | 2,154 [1,236, 12,677] |
| Candidate FPR 2% | 1.8% [1.1, 2.8] | 2.1% [1.3, 3.2] | 0% | 96.1% | 10,389 [3,389, 31,752] |
| Candidate FPR 3% | 0% [0, 0.4] | 57.0% [53.9, 60.1] | 0% | 43.0% | 1,326 [416, 3,389] |
| Candidate FPR 5% | 0% [0, 0.4] | 99.9% [99.4, 100] | 0% | 0.1% | 123 (one run) |

Brackets on rates are 95% Clopper-Pearson intervals over seeds; zero rates all have the interval [0, 0.4%]. The last column covers INCONCLUSIVE runs whose FPR condition is undecided. Some INCONCLUSIVE runs (27 at 1.5%, 76 at 2%, 8 at 3%) observed an FPR within about 0.02 points of the 2% line, which would need more than a million benign to decide. Those report none and are left out of the median.

**Bounds fixed before running** (`tests/test_calibration.py`): non-PASS ≤ 10% for the null and for fresh-engine noise (measured 0%), STALE ≥ 90% at 5% (measured 99.9%), and PASS ≤ 1% at 3% and at 5% (measured 0% at both).

**Noise at the fresh-engine noise level.** Two fresh fp16 engines (`fp16` vs `fp16_run1`), main probe, calibration split: 44% of scores bit-identical, logit differences with sd 3.65e-3, mean |Δscore| 7.8e-5, 1 decision flip of 2,101. The synthetic version is zero-inflated N(0, 4.9e-3) in the logit, which matches the mean and p99 of |Δlogit| (2.2e-3 and 1.16e-2 against a measured 2.17e-3 and 1.15e-2).

**Where it loses.**
- A tripled FPR, which triples escalations to the judge, is named THRESHOLD STALE only 57% of the time. It is never passed, but in the other 43% the gate only says it can't decide.
- Between 1.5% and 2%, the gate almost always says INCONCLUSIVE. At about 1,200 benign it can't tell an acceptable 1.5% from an unacceptable 2.5%, and settling it takes thousands to tens of thousands more benign examples.
- The 0% false-alarm rate holds because the threshold was fitted on the reference snapshot and the candidate differs only by noise. A threshold fitted on other data fails to PASS about 22% of the time.
