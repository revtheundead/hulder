# Paired flip check: validation on the planted-fault harness

**Whether a direction test on changed decisions can resolve INCONCLUSIVE into THRESHOLD STALE without making the gate stricter than the current rule's population verdict.** It can't, at any candidate margin, so the rule doesn't ship. The bars were committed before the run (`flip_check_expected.json`, commit `f5d78c6`) and haven't moved. Full output in `flip_check.json`.

## The rule and the bars

`up` and `down` are the decisions the candidate newly flags and newly drops, across both classes, at the threshold the verdict reads. Only when the current rule says INCONCLUSIVE, the verdict becomes THRESHOLD STALE if an exact two-sided sign test of `up` against `down` gives p ≤ 0.05 and the one-sided 95% lower bound on |up − down| / n is above a margin m.

| Rows (threshold refit per resample, each split) | Bar |
|---|---|
| Nulls: real rerun, rerun residuals | The rule fires in at most 2% of resamples |
| Path-switch nulls: uniform shifts of ±0.02 and ±0.04 logits | At most 2% |
| Shifts of ±0.1 | At most 5% |
| Population verdict PASS under the current rule | At most 5%, so the gate never gets stricter as the sample gets smaller |
| Gross | The planted-fault control's gross bars hold on the combined verdict |
| Every row | The only change is INCONCLUSIVE to STALE |
| Synthetic nulls (`null`, `fresh_engine_noise`, 1,000 seeds) | At most 2% at the selected margin |

m is the smallest of 0.5%, 1%, 2%, 3%, 4% and 5% that meets every calibration bar, with test as the replication. If none does, nothing ships and nothing is retuned.

## Result

**No margin meets every calibration bar.** The population-PASS bar fails at every candidate.

| m | Calibration bars missed |
|---|---|
| 0.5% | shift +0.1 (17.6%, against both its bars), shift +0.5, shift +1, noise at Phase 1 AWQ's size |
| 1% | shift +0.5, shift +1, noise at Phase 1 AWQ's size (35.9%) |
| 2% | shift +0.5 (57.6%), shift +1 (93.4%) |
| 3% | shift +0.5 (19.8%), shift +1 (83.1%) |
| 4% | shift +1, 63.4% [60.3, 66.4] |
| 5% | shift +1, 13.6% [11.5, 15.9] |

Test would have met every bar from m = 2% up, but it can't choose the margin. Everything else held at every margin:
- **Nulls:** the rule never fired on the real rerun or the rerun residuals, on either split.
- **Path-switch nulls:** it fired only on calibration's +0.04, in 9 resamples at m = 0.5%.
- **Synthetic nulls:** it fired in 0 of 1,000 seeds on both.
- **Gross rows:** every gross bar held on the combined verdict.
- **Every row:** the only change was INCONCLUSIVE to STALE.

The current verdicts reproduce the planted-fault control's counts on all 106 shared rows.

## Why no margin works

On calibration, the motivating case and the row that fails are the same kind of change.

| | up | down | Lower bound on net shift | Current population verdict |
|---|---|---|---|---|
| Sweep step C, AWQ int4, full set | 85 | 0 | 3.8% | INCONCLUSIVE as recorded |
| Planted shift +1, full set | 98 | 0 | 4.4% | PASS |

Both are one-way moves of a few percent of decisions with the FPR inside its band. A margin low enough to fire on C fires even more on shift +1, which the current rule calls PASS. So the population-PASS bar and the rule's purpose can't both be met by tuning m. It's a conflict in the definition, not a sample-size miss. At m = 5%, the only margin where shift +1 comes close, C wouldn't fire either.

What the rule would have added, as graded rows with no bar, at m = 2% on calibration:
- **INCONCLUSIVE resolved correctly to STALE:** 660 of 697 at shift −0.5, 183 of 910 at shift −0.3, all 580 at +1.5.
- **Scrambles:** it fired at most once in 1,000 at any margin, since their flips go both ways.

## An observation about the current rule

The current rule alone is often INCONCLUSIVE on a uniform shift at a whole-capture path switch's size. That's 4.7% and 12.4% of resamples at +0.02 and +0.04 on calibration, and 19.0% and 24.2% on test. Downward shifts of the same size were almost always PASS, with one STALE in 1,000 at −0.04 on calibration. These rows weren't bars for the current rule, and they don't change its recorded results. They bear on numerical-path detection, since a path switch alone can cost a declined verdict.

## What this doesn't change

Sweep step C stays INCONCLUSIVE in `sweep_c.json`, `sweep_results.md` and `sweep_expected.json`. Experiment 2 and the planted-fault control are unchanged. Nothing in `hulder.check` changed.

## Reproduce

```bash
source scripts/dev-env.sh && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 uv run python experiments/phase4/probe/flip_check.py run
```

Run at commit `bdc26dc`, CPU only, against the bars committed at `f5d78c6`. `kept/`, the shuffled rerun, the splits and the primary probe matched their recorded hashes and scanned clean before the run. Each row is written to `.cache/flip-check/rows.jsonl` as it finishes, and a rerun resumes unless the bars or the code changed.
