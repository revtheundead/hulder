# The gate's own calibration on synthetic data (Phase 1)

How `hulder check` behaves on an unchanged scorer, on candidates whose false-positive rate has risen by known amounts, and on candidates whose AUROC has fallen by known amounts. Full numbers in `gate_calibration.json`. The rule is in docs/design.md §6.

Reproduce from the repo root:

```
source scripts/dev-env.sh && uv run python experiments/phase1/gate/calibration.py
```

1,000 seeds per row, 1,191 benign and 910 attack (the S-Labs calibration set's sizes), a 1% target, and a threshold fitted on the reference's own benign scores. The generators are in `tests/synthetic.py`. The FPR shift moves every candidate logit by the same amount, which leaves AUROC unchanged and lowers FNR. The run used the complete verdict rule at commit `4d08e4b`, and its first six rows match the earlier run at `51ffc46` exactly. The manifest says `git_dirty: true` because this file was being edited during the run; no code differed from `4d08e4b`. An earlier run with only the first STALE condition is in this file's git history.

| Case | PASS | THRESHOLD STALE | SCORER DEGRADED | INCONCLUSIVE | More benign asked for, median [IQR] |
|---|---|---|---|---|---|
| Null (logit noise sd 1e-3) | 100% [99.6, 100] | 0% | 0% | 0% | |
| Fresh-engine noise | 100% [99.6, 100] | 0% | 0% | 0% | |
| Candidate FPR 1.5% | 20.5% [18.0, 23.1] | 0% | 0% | 79.5% | 2,154 [1,236, 12,677] |
| Candidate FPR 2% | 1.8% [1.1, 2.8] | 2.1% [1.3, 3.2] | 0% | 96.1% | 10,389 [3,389, 31,752] |
| Candidate FPR 3% | 0% [0, 0.4] | 57.0% [53.9, 60.1] | 0% | 43.0% | 1,326 [416, 3,389] |
| Candidate FPR 5% | 0% [0, 0.4] | 99.9% [99.4, 100] | 0% | 0.1% | 123 (one run) |
| Missed attacks, ΔAUROC −0.01 | 100% [99.6, 100] | 0% | 0% | 0% | |
| Missed attacks, ΔAUROC −0.02 | 6.7% [5.2, 8.4] | 0% | 4.6% [3.4, 6.1] | 88.7% [86.6, 90.6] | 0 (FPR condition already decided) |
| Missed attacks, ΔAUROC −0.05 | 0% | 0% | 100% [99.6, 100] | 0% | |
| Missed attacks, ΔAUROC −0.1 | 0% | 0% | 100% [99.6, 100] | 0% | |
| Blur, ΔAUROC −0.01 | 0.1% [0.0, 0.6] | 11.7% [9.8, 13.9] | 0% | 88.2% [86.0, 90.1] | 5,679 [2,124, 22,790] |
| Blur, ΔAUROC −0.02 | 0% | 78.8% [76.1, 81.3] | 2.3% [1.5, 3.4] | 18.9% [16.5, 21.5] | 763 [123, 3,389] |
| Blur, ΔAUROC −0.05 | 0% | 0% | 100% [99.6, 100] | 0% | |
| Blur, ΔAUROC −0.1 | 0% | 0% | 100% [99.6, 100] | 0% | |

Brackets on rates are 95% Clopper-Pearson intervals over seeds; zero rates all have the interval [0, 0.4%]. The last column covers INCONCLUSIVE runs whose FPR condition is undecided. Some INCONCLUSIVE runs (27 at 1.5%, 76 at 2%, 8 at 3%) observed an FPR within about 0.02 points of the 2% line, which would need more than a million benign to decide. Those report none and are left out of the median (69 in the blurred −0.01 case, 5 at −0.02).

**Bounds fixed before running** (`tests/test_calibration.py`): non-PASS ≤ 10% for the null and for fresh-engine noise (measured 0%), STALE ≥ 90% at 5% (measured 99.9%), and PASS ≤ 1% at 3% and at 5% (measured 0% at both). For the missed-attack cases (pre-registered 2026-10-05): at ΔAUROC −0.05 and −0.1, SCORER DEGRADED ≥ 80% and PASS ≤ 1% (measured 100% and 0% at both); at −0.01, SCORER DEGRADED ≤ 5% (measured 0%). The −0.02 case and the blurred cases have no bound. All bounds hold.

**What the degraded cases test.** The missed-attack shift lowers AUROC while FPR and FNR stay unchanged, so it measures SCORER DEGRADED on its own. A real degradation would rarely be that clean. It would usually raise FNR, and often FPR, so it can also trip THRESHOLD STALE. Blur is closer to that and shows how the two interact. Rule 1 is checked first, so a drop well beyond the tolerance (−0.05 or worse) still comes out as SCORER DEGRADED in every run. Near the tolerance it mostly surfaces as THRESHOLD STALE or INCONCLUSIVE instead: at −0.02, 78.8% STALE and 2.3% DEGRADED. That is still a non-PASS, and at −0.02 pointing the user at recalibration is defensible, since the drop is the tolerance itself. Blurred runs between −0.02 and −0.05 were not measured, so where STALE gives way to DEGRADED is unknown.

**Sizing the degraded candidates.** Each degraded case is set by its population ΔAUROC, not by what one seed shows. On seed 0 a missed-attack shift of −1.0 gives ΔAUROC −0.022 and −1.5 gives −0.044, while the population values are −0.0205 and −0.0392.
- *Missed-attack shift.* Attack logits A ~ N(0, 1) at or below the threshold's logit T = Φ⁻¹(0.99) − 3.5 ≈ −1.174 move down by s, and benign logits B ~ N(−3.5, 1) stay put. So ΔAUROC(s) = E[1{A ≤ T} (Φ(A + s + 3.5) − Φ(A + 3.5))], integrated numerically and inverted by root-finding (`missed_attack_shift_for` in `tests/synthetic.py`). Only the 12% of attacks below the threshold move, so ΔAUROC can never pass −0.1145, and −0.1 sits near that floor.
- *Blur.* Noise of sd σ on every candidate logit gives classes N(−3.5, 1 + σ²) and N(0, 1 + σ²), so AUROC = Φ(3.5 / √(2(1 + σ²))), inverted in closed form (`blur_for`).
- *Check.* `tests/test_synthetic.py` runs the generator itself at 1M benign and 1M attack for each size and requires the measured ΔAUROC within 0.001 of the target. An independent 2M-sample simulation of the missed-attack integral gave −0.0204 at s = −1.0 and −0.0391 at s = −1.5.

| Population ΔAUROC | −0.01 | −0.02 | −0.05 | −0.1 |
|---|---|---|---|---|
| Missed-attack shift (logit) | −0.616 | −0.983 | −1.749 | −3.098 |
| Blur noise sd (logit) | 0.594 | 0.800 | 1.201 | 1.719 |

**Fresh-engine noise.** Two fresh fp16 engines (`fp16` vs `fp16_run1`), main probe, calibration split: 44% of scores bit-identical, logit differences with sd 3.65e-3, mean |Δscore| 7.8e-5, 1 decision flip of 2,101. The synthetic version is zero-inflated N(0, 4.9e-3) in the logit, which matches the mean and p99 of |Δlogit| (2.2e-3 and 1.16e-2 against a measured 2.17e-3 and 1.15e-2).

**Where it loses.**
- At a missed-attack drop of exactly the −0.02 tolerance, the ΔAUROC lower bound should clear −0.02, and so allow PASS, in about 2.5% of runs. The measured PASS rate is 6.7% [5.2, 8.4]. Most of the excess comes from the threshold being refitted per seed, which scatters each seed's true ΔAUROC around −0.02 (sd 0.0017) and alone predicts 6.1%. The bootstrap CI covers each seed's own truth 94.8% of the time, though its lower bound sits above that truth 3.8% of the time rather than 2.5%. Details in docs/design.md §6 and `delta_auroc_coverage.json`.
- A tripled FPR, which triples escalations to the judge, is named THRESHOLD STALE only 57% of the time. It is never passed, but in the other 43% the gate only says it can't decide.
- Between 1.5% and 2%, the gate almost always says INCONCLUSIVE. At about 1,200 benign it can't tell an acceptable 1.5% from an unacceptable 2.5%, and settling it takes thousands to tens of thousands more benign examples.
- The 0% false-alarm rate holds because the threshold was fitted on the reference snapshot and the candidate differs only by noise. A threshold fitted on other data fails to PASS about 22% of the time (docs/design.md §6).
