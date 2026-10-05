# Demo probe on boundary-pairs (Phase 1)

**Demo only.** The quickstart's probe, trained because the main probe failed the dataset decision's AUROC rule on boundary-pairs (`probe_main.md`). It is not a prompt-attack monitor. Full numbers in `probe_demo.json`; design in the dataset decision's Outcome section.

Reproduce from the repo root, after the three steps in `probe_main.md`:

```
source scripts/dev-env.sh && uv run python experiments/phase1/probe/train_demo.py
```

Layer 20, position −1, C = 0.001, all reused from the main probe. Split by scenario: train 600 (25 scenarios), calibration 360 (15), test 240 (the dataset's own 10).

| Set | AUROC [95% CI] | FPR [95% CI] | FNR [95% CI] |
|---|---|---|---|
| Calibration (180 / 180) | 0.999 | 4.4% [1.9, 8.6] | 0.0% [0.0, 2.0] |
| Test (120 / 120) | 0.999 [0.996, 1.000] | 5.0% [1.9, 10.6] | 0.8% [0.0, 4.6] |

Threshold 0.394, for 5% FPR on calibration benign. AUROC CI is a pair-clustered bootstrap; rate CIs are Clopper-Pearson.

Near-perfect separation on 240 synthetic, curated examples says the dataset is easy for a probe trained on it, not that the probe generalises. Its FPR interval spans 1.9% to 10.6% at n = 120, so the quickstart's fp16 vs AWQ check may well return INCONCLUSIVE. If it does, the quickstart shows that verdict as is.

Contains data derived from 3nesdeniz/agentic-prompt-injection-boundary-pairs (Deniz 2026, doi:10.5281/zenodo.21378693), CC-BY-4.0.
