# Planted-fault control on Qwen3-4B

**Whether `hulder check` reaches the correct verdict on real scores when the fault and its size are known.** The expected verdicts and the bar below were committed before any verdict was computed, and misses are published. Results follow in a later commit, so this page holds only the rules for now.

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

## Reproduce

```bash
source scripts/dev-env.sh && uv run python experiments/phase4/probe/planted_faults.py run
```

`run` refuses unless the expected verdicts are committed and match what the code recomputes. It's CPU only and needs `kept/` and the shuffled rerun.

## What this can't show

The faults are planted, so this shows what the gate detects, not that such faults happen in real serving changes.
