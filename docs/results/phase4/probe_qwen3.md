# The Qwen3-4B probe: in-distribution near ceiling, and hard negatives worse than Qwen2.5

A prompt-attack probe trained on the full capture's `kept/` (`full_capture.md`): Qwen3-4B at `1cfa9a72` in bf16 on L4s, vLLM 0.30.0, `enable_thinking=False`, system prompt "You are a helpful assistant." Six cells were trained, layers 18 and 27 at positions −1, −9 and −10, on the Phase 1 splits (`../phase1/probe_splits.json`, docs/design.md §7). Selection was frozen and committed before test or boundary-pairs was scored. Full numbers are in `probe_qwen3_selection.json`, `probe_qwen3.json` and `probe_qwen3_deviations.json`.

## Reproduce

```bash
source scripts/dev-env.sh
uv run python experiments/phase4/probe/train_qwen3.py select
uv run python experiments/phase4/probe/train_qwen3.py score
uv run python experiments/phase4/probe/deviations.py
```

`select` refuses to overwrite a frozen selection. `score` runs only on the selection with sha256 `fbc7306b…` and only once. Both run on CPU in under 2 minutes each.

## The boundary-pairs failure repeats, and the selected probe is the worst at it

The primary probe, chosen on validation at position −1 as in Phase 1 (layer 18, C = 1e-4, threshold 0.8025 for a 1% FPR target), has boundary-pairs AUROC **0.685** [0.657, 0.713], against **0.839** [0.819, 0.860] for the Qwen2.5 probe. At its frozen threshold it flags **88.3%** of benign boundary-pairs prompts, whose median score is 0.973, against 0.023 for benign calibration prompts.

Every cell fails this way at its own threshold, flagging 81% to 97% of benign boundary-pairs. The cells differ in how badly they rank, though, and validation selection picked the one that generalises worst.

| Cell | Validation AUROC | Test AUROC [95% CI] | Boundary-pairs AUROC [95% CI] | Benign boundary-pairs flagged |
|---|---|---|---|---|
| **18, −1 (primary)** | 0.9990 | 0.994 [0.991, 0.996] | **0.685** [0.657, 0.713] | 88.3% [85.5, 90.8] |
| 18, −9 | 0.9983 | 0.995 [0.992, 0.997] | 0.741 [0.712, 0.767] | 95.8% [93.9, 97.3] |
| 18, −10 | 0.9967 | 0.994 [0.992, 0.996] | 0.822 [0.798, 0.843] | 80.8% [77.5, 83.9] |
| 27, −1 | 0.9981 | 0.994 [0.991, 0.996] | 0.845 [0.824, 0.866] | 92.3% [89.9, 94.3] |
| 27, −9 | 0.9982 | 0.996 [0.993, 0.998] | 0.804 [0.779, 0.827] | 96.5% [94.7, 97.8] |
| 27, −10 | 0.9960 | 0.996 [0.994, 0.998] | 0.809 [0.788, 0.830] | 81.0% [77.6, 84.1] |

Boundary-pairs AUROC intervals use a pair-clustered bootstrap (2,000 resamples, seed 0). Test AUROC intervals use a percentile bootstrap, and rates use Clopper-Pearson intervals.

- **Validation can't separate the cells.** All six score between 0.996 and 0.999, and the primary leads layer 27 at −1 by 0.0009. Validation measures in-distribution separation, and all six are near its ceiling. It says nothing about hard negatives.
- **Layer 18 at −1 ranks boundary-pairs worst; layer 27 at −1 ranks it best,** at about the Qwen2.5 probe's level. That ordering was seen only after selection was frozen.
- **The surface-feature reading of Phase 1 stands** (`../phase1/probe_main.md`). It can't be retested directly here, because only layers 18 and 27 were captured, so there is no layer-4 check.

## Test is harder than calibration

| Primary probe at 0.8025 | FPR [95% CI] | FNR [95% CI] |
|---|---|---|
| Calibration (1,191 benign / 910 attack) | 0.92% [0.46, 1.65] | 16.2% [13.8, 18.7] |
| Test (1,050 / 1,051) | 0.10% [0.00, 0.53] | **26.5%** [23.8, 29.2] |

Test misses 26.5% of attacks, against 16.2% on calibration, while its FPR stays under the target. Phase 1 saw the same direction (15.5% against 8.9%) and traced it to S-Labs test being further from train than calibration is. The primary's FNR is also the highest of the six cells on both sets. On test, the other cells' FNR runs from 8.9% (layer 27, −9) to 16.8%.

## What a numerical-path deviation does to the probe (experiment 1b)

Each deviated capture was scored against a twin on the majority path that saw the same prompts in the same batches. Only rows whose activations differ were scored.

| Deviation (`full_capture.md`) | Rows deviated | Cells | Mean \|Δlogit\| | Max \|Δlogit\| | Max \|Δscore\| | Flips |
|---|---|---|---|---|---|---|
| Whole-capture path switch (64 smoke prompts) | 64 of 64 | 18,−1 / 27,−1 | 0.028 / 0.022 | 0.104 / 0.076 | 0.012 / 0.004 | 0 |
| Pilot `canary_check`, one batch | 8 of 56 | 18,−1 / 27,−1 | 0.014 / 0.040 | 0.026 / 0.085 | 0.003 / 0.008 | 0 |
| Pilot, one prompt (`slabs/train/682`) | 1 of 1,000 | all 6 | | 0.013 to 0.057 | at most 0.001 | 0 |

The two smoke captures on the second path are bit-identical to each other, so they count as one set of 64 rows.

- **Single-prompt deviations are negligible.** One prompt moved by at most 0.057 in logit and 0.001 in score.
- **A whole-capture path switch can matter at the threshold.**
  - **Observed:** none of the 64 rows flipped, because none was near the threshold. The closest sat 0.25 logits away for the primary and 0.11 for layer 27 at −1.
  - **What's in reach:** on calibration, 26 of 2,101 rows (1.2%) lie within the largest observed shift, 0.104 logits, of the primary's threshold, and 8 (0.4%) within 0.085 of layer 27's.
  - **The worst case:** if a capture switched path and every such row flipped, the flag rate would move by up to about 1 point. With a mean shift of 0.02 to 0.04 logits, the typical effect is far smaller.
- **What this means for a gate check.** A shift of that size would read as a serving-config change at the threshold, not as noise. So replication stays the check for captures (docs/design.md §10).

## Next

- **The first serving-config comparison.** Online FP8 (W8A8) against this bf16 reference, on the calibration set.
- **Probes.** It keeps the frozen primary probe. It also reports layer 27 at −1 as a secondary probe, chosen after the boundary-pairs results above were seen, and labelled as such.
- **Reporting.** Validation AUROC is near its ceiling in every cell, so AUROC will barely move under a serving change. The comparison leads with flag rates (FPR and FNR at the frozen threshold) and the decision flip rate, with ΔAUROC as context.
