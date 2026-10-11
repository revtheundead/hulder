# Live tutorial run on the RTX 2070

The live tutorial (`docs/tutorial-live.md`) run once on the local GPU, recorded for the v0.1 exit criterion (README §11).

**Command** (from the repo root, at commit `f97124e`, clean tree):

```bash
source scripts/dev-env.sh && uv run python -m hulder.capture.demo --out .cache/live_demo
```

The outputs are copied to `live_demo/` (`check.txt`, `check.json`, `manifest.json`). The snapshots stay in `.cache/` and are not committed. vLLM 0.30.0, torch 2.13.0 (CUDA 13.0), NVIDIA GeForce RTX 2070, with the models already in `HF_HOME`. The captures took 112 s, 41 s and 81 s, and the whole command 4 min 48 s.

## Exit criterion

| Condition | Result |
|---|---|
| Runs end to end on the local GPU | Yes, exit 0 |
| The rerun gives a noise floor | 0 decisions flipped, mean \|Δscore\| 0.000223 |
| The AWQ shift exceeds it | 11 flipped, mean \|Δscore\| 0.0514; candidate minus rerun 0.0512 [0.0456, 0.0576] |
| Verdict, reported whatever it is | **INCONCLUSIVE** (exit 12), FPR 7.2% [3.9%, 12.0%] against a 10% band |

The verdict, flip counts and FPR interval match the bundled quickstart (README §4).

## Against the bundled snapshots

Not bit-identical, as expected. These 360 prompts were captured on their own, while the bundled snapshots came from a capture of all 16,332 probe prompts, so batch-mates differ. From `experiments/phase1/probe/compare_live.py`, written to `live_demo/vs_bundled.json`:

| Live run | Bundled | Median relative L2 | Max | Mean \|Δscore\| | Decisions differing |
|---|---|---|---|---|---|
| fp16 | demo_fp16 | 1.1e-3 | 2.1e-3 | 2.7e-4 | 0 |
| fp16_rerun | demo_fp16_run1 | 1.2e-3 | 2.0e-3 | 3.3e-4 | 0 |
| awq | demo_awq | 4.7e-3 | 8.9e-3 | 1.2e-3 | 0 |

The batch-composition effect is about the size the earlier rerun study found at layer 20 (around 3e-3), and about four times larger under AWQ.

## Where this loses

- One run on one GPU. It shows the tutorial works on Turing, not what the verdict is on other hardware. Near a boundary (13 of 180 benign flagged, PASS at 10 or fewer), another GPU could give PASS.
- No bf16 and no FP8, which Turing lacks.
