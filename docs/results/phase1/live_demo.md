# Live tutorial run on the RTX 2070

The live tutorial (`docs/tutorial-live.md`) run on the local GPU, recorded for the v0.1 exit criterion (README §11).

**Command** (from the repo root, at commit `5f88802`, clean tree, which `manifest.json` records). The same run today:

```bash
source scripts/dev-env.sh && uv run python examples/live_demo/live_demo.py --out .cache/live_demo --force
```

At `5f88802` the tutorial was a module inside the package, so the recorded `manifest.json` names that module's command, not this one. It has since moved to `examples/live_demo/` and calls `hulder check` as a command rather than through the CLI's internals. The configs, prompts and check are unchanged. The recorded files in `live_demo/` are left as they were written.

The outputs are copied to `live_demo/` (`check.txt`, `check.json`, `manifest.json`). The snapshots stay in `.cache/` and are not committed. vLLM 0.30.0, torch 2.13.0 (CUDA 13.0), NVIDIA GeForce RTX 2070, with the models already in `HF_HOME`. The captures took 50 s, 40 s and 49 s, and the whole command 3 min 10 s.

## Exit criterion

| Condition | Result |
|---|---|
| Runs end to end on the local GPU | Yes, exit 0 |
| The rerun gives a noise floor | 0 decisions flipped, mean \|Δscore\| 0.000286 |
| The AWQ shift exceeds it | 11 flipped, mean \|Δscore\| 0.0514; candidate minus rerun 0.0511 [0.0455, 0.0576] |
| Verdict, reported whatever it is | **INCONCLUSIVE** (exit 12), FPR 7.2% [3.9%, 12.0%] against a 10% band |

The verdict, flip counts and FPR interval match the bundled quickstart (README §4).

This is the second recorded run. The first, at `f97124e` before the resume and manifest changes, gave the same verdict, flips and FPR, but its rerun's mean \|Δscore\| was 0.000223, not 0.000286. Every capture here is a fresh engine on the same 360 prompts, so two fp16 reruns on one GPU differ by about a quarter in the noise floor. The AWQ shift is 180 times the noise floor in this run and 230 times in the first.

## Against the bundled snapshots

Not bit-identical, as expected. These 360 prompts were captured on their own, while the bundled snapshots came from a capture of all 16,332 probe prompts, so batch-mates differ. From `experiments/phase1/probe/compare_live.py`, written to `live_demo/vs_bundled.json`:

| Live run | Bundled | Median relative L2 | Max | Mean \|Δscore\| | Decisions differing |
|---|---|---|---|---|---|
| fp16 | demo_fp16 | 1.2e-3 | 2.1e-3 | 2.7e-4 | 0 |
| fp16_rerun | demo_fp16_run1 | 1.2e-3 | 2.1e-3 | 3.2e-4 | 0 |
| awq | demo_awq | 4.6e-3 | 8.5e-3 | 1.2e-3 | 0 |

The batch-composition effect is about the size the earlier rerun study found at layer 20 (around 3e-3), and about four times larger under AWQ.

## Where this loses

- Two runs on one GPU. It shows the tutorial works on Turing, not what the verdict is on other hardware. Near a boundary (13 of 180 benign flagged, PASS at 10 or fewer), another GPU could give PASS.
- No bf16 and no FP8, which Turing lacks.
