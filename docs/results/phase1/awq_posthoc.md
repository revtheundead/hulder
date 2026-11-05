# Post hoc: the Phase 1 AWQ candidate on the test split

**This analysis is post hoc and doesn't change any verdict.** The primary result is still `real_check.md`, on the calibration split, which is INCONCLUSIVE. Nobody planned this look before `real_check.json` came out, and the choice to look followed that result, so what follows is a lead for the Qwen3 sweep and not a finding about AWQ.

The model, revision and probe are those of `real_check.md`, so Qwen2.5-1.5B-Instruct at fp16 against the official AWQ int4 checkpoint, with the main probe at layer 20, position −1, on an RTX 2070 at a target FPR of 1%. Those captures held the test split, and `real_check.py` never read it. Full output in `awq_posthoc.json`.

## Results

Every set and threshold tried is below, with nothing dropped. Rates are Clopper-Pearson 95% intervals for the candidate. The frozen threshold is the deployed one from `real_check.json`, 0.769. A refit threshold is the one giving at most 1% FPR on that set's fp16 benign scores. On calibration the two are the same threshold.

| Set | Threshold | Benign / attack | FPR fp16 | FPR AWQ | FNR fp16 → AWQ | Flips: AWQ vs rerun | Verdict |
|---|---|---|---|---|---|---|---|
| Calibration (primary) | 0.769 | 1,191 / 910 | 11 = 0.92% | 25 = 2.10% [1.36, 3.08] | 8.9% → 4.5% | 58 vs 1 | INCONCLUSIVE |
| Test | frozen 0.769 | 1,050 / 1,051 | 4 = 0.38% | 37 = 3.52% [2.49, 4.82] | 15.5% → 5.8% | 135 vs 0 | THRESHOLD STALE |
| Test | refit 0.590 | 1,050 / 1,051 | 10 = 0.95% | 74 = 7.05% [5.57, 8.77] | 7.8% → 2.9% | 115 vs 1 | THRESHOLD STALE |
| Pool | frozen 0.769 | 2,241 / 1,961 | 15 = 0.67% | 62 = 2.77% [2.13, 3.53] | 12.4% → 5.2% | 193 vs 1 | THRESHOLD STALE |
| Pool | refit 0.682 | 2,241 / 1,961 | 22 = 0.98% | 84 = 3.75% [3.00, 4.62] | 8.8% → 3.9% | 166 vs 0 | THRESHOLD STALE |

ΔAUROC stays inside tolerance everywhere, at −0.0020 [−0.0031, −0.0011] on calibration, −0.0026 [−0.0039, −0.0015] on test and −0.0021 [−0.0030, −0.0014] on the pool.

How stable each verdict is across 1,000 resamples. The last row of each set is the check that should stay quiet, with the fp16 rerun as the candidate.

| Set | Candidate, threshold | PASS | THRESHOLD STALE | SCORER DEGRADED | INCONCLUSIVE |
|---|---|---|---|---|---|
| Calibration | AWQ, frozen | 10 | 30 | 0 | 960 |
| Calibration | AWQ, refit per resample | 9 | 74 | 0 | 917 |
| Calibration | fp16 rerun, refit per resample | 994 | 0 | 0 | 6 |
| Test | AWQ, frozen | 0 | 883 | 0 | 117 |
| Test | AWQ, refit per resample | 0 | 1,000 | 0 | 0 |
| Test | fp16 rerun, refit per resample | 973 | 0 | 0 | 27 |
| Pool | AWQ, frozen | 0 | 673 | 0 | 327 |
| Pool | AWQ, refit per resample | 0 | 998 | 0 | 2 |
| Pool | fp16 rerun, refit per resample | 1,000 | 0 | 0 | 0 |

## How to read this

On the split `real_check.py` never read, the same change crosses the line that calibration sat on, in the same direction: FPR up, FNR down, ΔAUROC inside tolerance. That is the pattern of a stale threshold. The fp16 rerun stays quiet on the same rows, so the rule isn't simply firing on these sets.

## Why it stays a lead

- **Nothing was declared first.** Five distinct checks across three sets were tried, and none of them is corrected for that. The look was prompted by the INCONCLUSIVE it might overturn.
- **The pool isn't independent evidence.** It contains calibration, and its verdict adds sample size and no separate replication.
- **The size of the rise differs by split, unexplained.** At the same threshold FPR rose 2.3× on calibration and 9.3× on test, from a reference FPR that was already lower on test, though the candidate's intervals overlap. Split composition may matter, and this page doesn't test that.
- **Resampling measures stability and not truth.** The resamples redraw the same prompts, so 1,000 of 1,000 says the test verdict isn't fragile to which of those rows are drawn, and says nothing about other prompts.
- **One model on one GPU.** This is Qwen2.5-1.5B on Turing through the Marlin kernel, and it says nothing about Qwen3-4B on an L4.

## How the numbers were produced

A one-off script, not kept in this repository, since nothing else depends on it and the method fits here. It ran CPU only on 2026-11-05 at commit `d2f5497`, with vLLM unused and the manifest in `awq_posthoc.json`, whose `inputs_sha256` is empty, so the input hashes are given here instead.

1. **Inputs.** The Phase 1 captures, with these SHA-256 prefixes: `fp16/activations.npy` `2764025d`, `fp16_run1/activations.npy` `e012ff52`, `awq/activations.npy` `42d525ad`, identical `ids.json` in all three `5af14cca`, and `probe_main.npz` `fed66cae`. Each run's layer 20, position −1 rows were compared with the saved snapshots hashed in `real_check.json`, and all three matched exactly. Splits and labels came from `splits.parquet`.
2. **Scores.** Probe logits from `probe_logits`, through `expit` to scores.
3. **Point checks.** `hulder.check(reference, candidate, labels, threshold, 0.01, seed=0, noise_floor=rerun)` for each set at each threshold. The script checked that the three runs hold the same ids in the same order, and it passed scores without ids, which is why the rows carry pairing warnings.
4. **Resamples.** Resample `i` of 1,000 uses `resample_rows(labels, i)` and `refit_threshold` from `experiments/phase4/probe/planted_faults.py`, then `hulder.check(..., seed=i)` with no noise floor. Those functions keep each class's size, keep pairs together, and refit on the resample's reference benign scores.
5. **Persistence.** Each row went to disk as it finished, through `experiments/phase4/probe/resumable_rows.py`.

An earlier run of this analysis was stopped by a disk failure that same night, and none of its output was kept or used.
