# Main probe on S-Labs, and the quickstart rule (Phase 1)

Qwen2.5-1.5B-Instruct @ `989aa798`, fp16, vLLM 0.30.0, RTX 2070. System prompt "You are a helpful assistant." Splits in `probe_splits.json`, full numbers in `probe_main.json`.

Reproduce from the repo root:

```
source scripts/dev-env.sh
uv run python experiments/phase1/probe/prepare_splits.py
uv run python experiments/phase1/probe/extract.py --config fp16
uv run python experiments/phase1/probe/train.py
uv run python experiments/phase1/probe/near_dups.py
```

Capture of 16,332 prompts at 7 layers and 2 positions takes about 250 s. The committed numbers come from the capture at commit `eaa1356`. Fresh-engine captures are not bit-identical (`extraction_reruns.md`), which moves these numbers by at most one count.

## Selection (validation, position −1)

Selected: layer 20, C = 0.001, validation AUROC 0.9987. Selection is close to arbitrary, for two reasons:
- Every layer from 4 to 28 at either position scores 0.990 to 0.999. Layer 4 alone reaches 0.996, which suggests the S-Labs classes are separable from surface features of the text.
- Across layers 16 to 24 the differences are around 1e-4, within noise.

**Limitation: C = 0.001 is the edge of the grid** (1e-3 to 1). A smaller C might score marginally higher on validation. The grid is deliberately left as is: it was not extended after the boundary-pairs result below was seen, so that result cannot steer selection.

## Results at threshold 0.769 (1% FPR target on calibration benign)

| Set | AUROC [95% CI] | FPR [95% CI] | FNR [95% CI] |
|---|---|---|---|
| Calibration (1,191 benign / 910 attack) | | 0.92% [0.46, 1.65] | 8.9% [7.1, 10.9] |
| S-Labs test (1,050 / 1,051) | 0.994 [0.992, 0.997] | 0.38% [0.10, 0.97] | 15.5% [13.4, 17.8] |
| Boundary-pairs, all 1,200 (600 / 600) | **0.839** [0.819, 0.860] | **95.3%** [93.3, 96.9] | 0.0% [0.0, 0.6] |

AUROC CIs are percentile bootstrap (2,000 resamples); boundary-pairs resamples whole pairs. Rate CIs are Clopper-Pearson. The threshold is the empirical quantile, provisional until recalibration is settled in v0.2.

## Quickstart rule

Point estimate 0.839 < 0.85, so **the quickstart gets a demo probe trained on boundary-pairs**. The S-Labs probe is unchanged and boundary-pairs stays held out from it. The CI straddles 0.85. The rule uses the point estimate, as fixed in advance (docs/design.md §7).

## What this shows

- **The hard-negative failure is total at this threshold.** Benign boundary-pairs prompts score a median of 0.986 (S-Labs calibration benign: 0.011), so 95% of them would escalate. The probe still ranks attacks above benign (AUROC 0.84), so a threshold tuned on boundary-pairs would do better, but an S-Labs-calibrated threshold is useless on agentic text. This is the distribution-shift case the dataset choice anticipated, and it belongs on the benchmarks page.
- **Test is harder than calibration, and near-duplicates don't explain it.** FNR is 15.5% on test against 8.9% on calibration, with non-overlapping CIs. `near_dups.json` checks whether calibration items copy train (character 3-5-gram TF-IDF cosine on normalised text, `experiments/phase1/probe/near_dups.py`):

  | Split | Train near-dup ≥ 0.9 | ≥ 0.8 | FNR, items without a near-dup ≥ 0.8 | Median similarity to nearest train item (benign / attack) |
  |---|---|---|---|---|
  | Calibration | 1.2% | 3.3% | 9.0% | 0.49 / 0.49 |
  | Validation | 0.5% | 2.1% | 9.1% | 0.44 / 0.48 |
  | Test | 0.1% | 0.8% | 15.6% | 0.41 / 0.46 |

  Removing near-duplicates leaves calibration's FNR unchanged, so the probe was not refit. **S-Labs test is further from train than calibration is**, for benign and attack alike, which points to a distribution difference between S-Labs' train and test splits rather than leakage. That distance raises test FNR (15.5% against 8.9%), while test FPR (0.38%) stays under the 1% target.
- **Validation and test overlap.** 19.6% of validation items have a near-duplicate (≥ 0.9) in test. Cross-split deduplication matched only whitespace-normalised text, so this overlap includes texts that differ only in case or punctuation as well as true near-duplicates. Validation only chose layer and C, over a grid flat to about 1e-4, so the effect on the test numbers should be small. But the test split is not fully independent of selection.
