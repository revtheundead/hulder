# First real-data check: fp16 vs AWQ int4 (Phase 1)

Qwen2.5-1.5B-Instruct at fp16 (reference, the canonical `fp16` capture) against the official AWQ checkpoint (`Qwen/Qwen2.5-1.5B-Instruct-AWQ@3ecffa0`, candidate), with `fp16_run1` as the noise-floor rerun. Layer 20, position −1, RTX 2070, vLLM 0.30.0. Full output in `real_check.json`.

Reproduce from the repo root:

```
source scripts/dev-env.sh
uv run python experiments/phase1/probe/extract.py --config awq --save-layer 20 --save-position -1
uv run python experiments/phase1/probe/snapshots.py
uv run python experiments/phase1/gate/real_check.py
```

The AWQ capture ran all 16,332 prompts of `splits.parquet` in file order with the fp16 capture config, so every checked prompt shared its batch with the same prompts as in both fp16 runs. Only model, revision and quantization differ, and `check` lists exactly those three as changes under test, with no warnings. vLLM ran the checkpoint through its AWQ Marlin kernel (`MarlinLinearKernel`).

## Verdicts

Both checks are **INCONCLUSIVE** (exit 12). Rates are Clopper-Pearson 95% intervals. FPR, FNR and AUROC are computed by the script, since `Report` doesn't carry them yet.

| | Main probe (S-Labs calibration) | Demo probe (boundary-pairs calibration) |
|---|---|---|
| Benign / attack | 1,191 / 910 | 180 / 180 |
| Threshold, target FPR | 0.769, 1% | 0.395, 5% |
| FPR fp16 | 11 = 0.92% [0.46, 1.65] | 8 = 4.4% [1.9, 8.6] |
| FPR AWQ | 25 = 2.10% [1.36, 3.08] | 13 = 7.2% [3.9, 12.0] |
| FNR fp16 → AWQ | 8.9% → 4.5% | 0% → 0% |
| AUROC fp16 → AWQ | 0.9956 → 0.9936 | 0.9990 → 0.9981 |
| Decision flips: AWQ vs rerun | 58 vs 1 | 11 vs 0 |
| Mean \|Δscore\|: AWQ vs rerun | 0.048 vs 7.8e-5 | 0.052 vs 2.1e-4 |
| Mean \|Δlogit\|: AWQ vs rerun | 0.93 vs 0.0022 | 0.50 vs 0.0020 |
| Shift exceeds noise floor | yes | yes |
| More benign needed | 78,344 | 300 |

**Why INCONCLUSIVE.** In both checks the candidate's FPR interval straddles the 2 × target line, which is 2% for the main probe and 10% for the demo. So neither PASS nor THRESHOLD STALE can be called. FNR and ΔAUROC are inside their tolerances.

**What AWQ did.** It moved scores upward along the probe direction. FPR roughly doubled (point estimates 2.3× and 1.6×) and the main probe's FNR halved, which is the pattern of a stale threshold, not a degraded scorer. AWQ moved scores 250 to 600 times as much as the fp16 rerun did (250 to 430 times in logits), and 58 main-probe decisions flipped where the rerun flipped 1. The gate correctly says the change is real, but the calibration set is too small to say whether the FPR rise crosses the band.

## Where this loses

- **The main probe's FPR sits almost on the line.** At 2.10% observed against a 2% band, `more_benign_needed` is 78,344. That is no realistic calibration set, and more data won't settle it. This matches the synthetic result that a true 2% FPR is INCONCLUSIVE 96% of the time at this size. A team would have to decide on the point estimate, and the gate doesn't make that call for them.
- **The demo set is small.** 180 benign at a 5% target give an interval about 8 points wide, so the quickstart will show INCONCLUSIVE, as README §8 anticipates. 300 more benign examples would settle it if the observed rate held.
- **Turing only.** AWQ here runs through Marlin with fp16 activations on an RTX 2070. Other GPUs or kernels can shift scores differently.
