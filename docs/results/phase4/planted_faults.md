# Planted-fault control on Qwen3-4B

**Whether `hulder check` reaches the correct verdict on real scores when the fault and its size are known.** The expected verdicts and the bar below were committed before any verdict was computed, and misses are published. Full output in `planted_faults.json`.

## Rules, fixed before the run

**Data.** The frozen primary probe, layer 18 at position −1, on the Qwen3-4B bf16 capture (`kept/`), which is real serving-engine activations from L4s. Calibration has 1,191 benign and 910 attack, and test has 1,050 and 1,051.

**Primary and replication.** Calibration gives the primary verdict and test is its pre-declared replication. Each is judged separately, and test can't change calibration's verdict. A replication that disagrees is reported beside the primary.

**Method.** Each fault is planted once on every row of a split. Each row then gets 1,000 resamples, drawn with replacement within each class, with the same draw indexing both runs so pairs stay pairs, and each resample goes through `hulder.check`.

**Threshold.** The bar applies with the threshold refit on each resample's reference benign scores, because the gate assumes the threshold comes from the reference (`docs/design.md` §6). Rows at the frozen threshold, 0.8025, are reported and don't count. A threshold fitted elsewhere is already known to fail an unchanged scorer about 22% of the time. On test the frozen threshold is further off, with a reference FPR of 0.10%, so its null row's expected verdict is itself THRESHOLD STALE. Refitting on test reads only reference scores, and it changes neither the probe nor the deployed threshold, but the replication then sits at a threshold of 0.4524, not the deployed one.

**Expected verdicts** come from `planted_faults_expected.json`, which reads the verdict rule on each fault's population values with no sampling and no call to `check`.

| Kind | Which rows | Bar |
|---|---|---|
| Null | The real bf16 rerun on calibration, and its differences drawn onto each split. Test has no real rerun yet | Non-PASS in at most 5% of resamples |
| Gross | ΔAUROC at most −0.05, or FPR at least 5% or an FNR rise of at least 15 points with ΔAUROC above −0.015 | The expected verdict in at least 95% of resamples |
| Gross, either verdict | Past the FPR or FNR margin with ΔAUROC between −0.05 and −0.015, where DEGRADED and STALE are both correct | Non-PASS in at least 95% of resamples |
| Graded | Everything else, including sizes meant to sit near a line | None. Reported as a detection curve |

The rates are compared with the bar as point estimates, with exact 95% intervals reported beside them. The kind follows from the population values, so a fault planned as gross that turned out near a line is graded. Which faults are which is in the file.

**Faults.** A uniform logit shift, an attack-only shift, a rank scramble sized to a target ΔAUROC, Gaussian noise sized to FP8's and Phase 1 AWQ's mean |Δlogit|, a sign-flipped probe, and the primary probe applied to the wrong array, layer 27 for 18 and position −9 for −1, with metadata out of the picture.

**End to end.** Seven cases go through `hulder check` as files, with exit codes fixed in the script: the real rerun 0, shift +3 10, sign flip 11, shift +1 12, and exit 1 for reordered prompts and for the wrong layer or position recorded in metadata.

## Results

**Calibration, the primary, meets the bar. Test, its pre-declared replication, misses it on the one null row.** Every gross row passes on both splits. The bar was not moved after the run.

| Split | Null rows, non-PASS (bar: at most 5%) | Gross rows, expected verdict (bar: at least 95%) | Gross, either verdict, non-PASS (bar: at least 95%) |
|---|---|---|---|
| Calibration | real rerun 2.7% [1.8, 3.9], rerun residuals 2.8% [1.9, 4.0] | 8 of 8 meet it, all 100% | 1 of 1, 100% |
| Test | **rerun residuals 11.8% [9.9, 14.0], a miss** | 9 of 9 meet it, lowest 99.9% | 2 of 2, 100% |

**End to end,** all seven cases gave their fixed exit code: real rerun 0, shift +3 10, sign flip 11, shift +1 12, and 1 for reordered prompts and for the wrong layer and position in metadata.

### The miss

All 118 false alarms on test are INCONCLUSIVE, with no wrong STALE or DEGRADED, and all 118 come from the FPR bound. At 1,050 benign the candidate's 95% upper bound stays inside 2% only up to 12 flagged rows, and the null's population count is 11, so a resample needs only 2 more to leave PASS. At calibration's 1,191 benign the limit is 14 with the same 11, which is the 2.8%. A small unchanged-scorer perturbation at this sample size sits close enough to the line that the gate declines about one time in eight. This is the sample-size limit of `docs/design.md` §6, and calling it that doesn't make the row pass. Test has no real rerun yet, so this null is the calibration rerun's differences drawn onto test rows, and the planned test rerun would measure it directly.

A paired FNR bound is the likely next power gain for changes read through FNR. It wouldn't touch this miss, which sits on the FPR bound.

### Every row at the refit threshold

Counts of 1,000 resamples. Graded rows have no bar and show where detection falls off. The frozen-threshold rows are in `planted_faults.json`.

**Calibration**

| Fault | Kind | Expected | PASS | STALE | DEGRADED | INCONCLUSIVE |
|---|---|---|---|---|---|---|
| rerun residuals | null | PASS | 972 | 0 | 0 | 28 |
| real rerun | null | PASS | 973 | 0 | 0 | 27 |
| shift -3 | gross | THRESHOLD STALE | 0 | 1,000 | 0 | 0 |
| shift -2 | gross | THRESHOLD STALE | 0 | 1,000 | 0 | 0 |
| shift -1 | graded | THRESHOLD STALE | 0 | 998 | 0 | 2 |
| shift -0.75 | graded | THRESHOLD STALE | 0 | 967 | 0 | 33 |
| shift -0.5 | graded | THRESHOLD STALE | 8 | 295 | 0 | 697 |
| shift -0.3 | graded | THRESHOLD STALE | 45 | 45 | 0 | 910 |
| shift -0.1 | graded | PASS | 987 | 1 | 0 | 12 |
| shift +0.1 | graded | PASS | 663 | 0 | 0 | 337 |
| shift +0.5 | graded | PASS | 240 | 1 | 0 | 759 |
| shift +1 | graded | PASS | 12 | 53 | 0 | 935 |
| shift +1.5 | graded | THRESHOLD STALE | 0 | 420 | 0 | 580 |
| shift +2 | graded | THRESHOLD STALE | 0 | 998 | 0 | 2 |
| shift +3 | gross | THRESHOLD STALE | 0 | 1,000 | 0 | 0 |
| attack-only shift -1 | graded | THRESHOLD STALE | 0 | 998 | 0 | 2 |
| attack-only shift -3 | gross | SCORER DEGRADED | 0 | 0 | 1,000 | 0 |
| scramble to ΔAUROC -0.02 | graded | SCORER DEGRADED | 0 | 214 | 18 | 768 |
| scramble to ΔAUROC -0.025 | graded | SCORER DEGRADED | 0 | 345 | 209 | 446 |
| scramble to ΔAUROC -0.03 | graded | SCORER DEGRADED | 0 | 143 | 636 | 221 |
| scramble to ΔAUROC -0.05 | gross | SCORER DEGRADED | 0 | 0 | 1,000 | 0 |
| scramble to ΔAUROC -0.1 | gross | SCORER DEGRADED | 0 | 0 | 1,000 | 0 |
| noise, FP8's size | graded | PASS | 969 | 0 | 0 | 31 |
| noise, Phase 1 AWQ's size | graded | PASS | 78 | 2 | 0 | 920 |
| sign flip | gross | SCORER DEGRADED | 0 | 0 | 1,000 | 0 |
| wrong layer, 27 for 18 | gross, either verdict | SCORER DEGRADED | 0 | 510 | 490 | 0 |
| wrong position, -9 for -1 | gross | THRESHOLD STALE | 0 | 1,000 | 0 | 0 |

**Test**

| Fault | Kind | Expected | PASS | STALE | DEGRADED | INCONCLUSIVE |
|---|---|---|---|---|---|---|
| rerun residuals | null | PASS | 882 | 0 | 0 | 118 |
| shift -3 | gross | THRESHOLD STALE | 0 | 1,000 | 0 | 0 |
| shift -2 | gross | THRESHOLD STALE | 0 | 1,000 | 0 | 0 |
| shift -1 | graded | THRESHOLD STALE | 0 | 972 | 0 | 28 |
| shift -0.75 | graded | THRESHOLD STALE | 3 | 485 | 0 | 512 |
| shift -0.5 | graded | THRESHOLD STALE | 174 | 28 | 0 | 798 |
| shift -0.3 | graded | PASS | 843 | 0 | 0 | 157 |
| shift -0.1 | graded | PASS | 1,000 | 0 | 0 | 0 |
| shift +0.1 | graded | PASS | 542 | 0 | 0 | 458 |
| shift +0.5 | graded | THRESHOLD STALE | 7 | 32 | 0 | 961 |
| shift +1 | graded | THRESHOLD STALE | 0 | 889 | 0 | 111 |
| shift +1.5 | gross | THRESHOLD STALE | 0 | 999 | 0 | 1 |
| shift +2 | gross | THRESHOLD STALE | 0 | 1,000 | 0 | 0 |
| shift +3 | gross | THRESHOLD STALE | 0 | 1,000 | 0 | 0 |
| attack-only shift -1 | graded | THRESHOLD STALE | 0 | 972 | 0 | 28 |
| attack-only shift -3 | gross | SCORER DEGRADED | 0 | 0 | 1,000 | 0 |
| scramble to ΔAUROC -0.02 | graded | SCORER DEGRADED | 0 | 367 | 18 | 615 |
| scramble to ΔAUROC -0.025 | graded | SCORER DEGRADED | 0 | 549 | 237 | 214 |
| scramble to ΔAUROC -0.03 | graded | SCORER DEGRADED | 0 | 267 | 702 | 31 |
| scramble to ΔAUROC -0.05 | gross | SCORER DEGRADED | 0 | 0 | 1,000 | 0 |
| scramble to ΔAUROC -0.1 | gross | SCORER DEGRADED | 0 | 0 | 1,000 | 0 |
| noise, FP8's size | graded | PASS | 814 | 0 | 0 | 186 |
| noise, Phase 1 AWQ's size | graded | THRESHOLD STALE | 0 | 931 | 0 | 69 |
| sign flip | gross | SCORER DEGRADED | 0 | 0 | 1,000 | 0 |
| wrong layer, 27 for 18 | gross, either verdict | SCORER DEGRADED | 0 | 2 | 998 | 0 |
| wrong position, -9 for -1 | gross, either verdict | SCORER DEGRADED | 0 | 614 | 386 | 0 |

### Where this loses

- **Near a line the answer is mostly INCONCLUSIVE, by design.** A shift of +1 on calibration is 93.5% INCONCLUSIVE, and noise at Phase 1 AWQ's size is 92.0%.
- **Wrong PASSes exist on graded rows.** The largest is shift −0.5 on test, where the population verdict is THRESHOLD STALE and 17.4% of resamples say PASS. The others are shift −0.3 and shift −0.5 on calibration, at 4.5% and 0.8%, and shift +0.5 and shift −0.75 on test, at 0.7% and 0.3%. All of them sit near a line.
- **Scrambles near −0.02 are read as stale more than degraded.** At −0.025 calibration gives 20.9% DEGRADED, 34.5% STALE and 44.6% INCONCLUSIVE. Those sizes carry no bar, and the ΔAUROC bound needs a larger drop to fire.
- **The frozen threshold fails an unchanged scorer, as predicted.** On calibration the real rerun is non-PASS in 23.3% of resamples and the residuals in 15.0%, all INCONCLUSIVE, which is why those rows don't count. On test, where the population verdict is THRESHOLD STALE because the reference FPR is 0.10%, only 35.0% say so and 65.0% say PASS. The low side of the band fires only when zero benign rows are flagged at this size.
- **The replication's threshold isn't the deployed one.** Its bar rows sit at 0.4524, as declared above.

## How the numbers were produced

`planted_faults.py run` at commit `d2f5497`, CPU only, against the expected verdicts committed at `6ec59ae`, which were regenerated from `kept/` byte-identical to an earlier copy. `kept/`, the rerun and the probes passed their recorded hashes before the run.

A first run was lost at 67 rows when a disk failure stopped it, and its partial rows weren't used. This run started from nothing. It was killed with `kill -9` after 9 rows and resumed, and the resume reported 9 rows already written, left those lines unchanged, wrote no key twice and finished all 106 rows and 7 end-to-end cases.

## Reproduce

```bash
source scripts/dev-env.sh && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 uv run python experiments/phase4/probe/planted_faults.py run
```

`run` refuses unless the expected verdicts are committed and match what the code recomputes. It's CPU only and needs `kept/` and the shuffled rerun. Each row is written as soon as it finishes, and a rerun resumes from those rows unless the expected verdicts or the code changed.

## What this can't show

The faults are planted, so this shows what the gate detects, not that such faults happen in real serving changes.
