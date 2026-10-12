# Live tutorial run on the RTX 2070

The live tutorial (`docs/tutorial-live.md`) run on the local GPU, recorded for the v0.1 milestone (docs/design.md §11). There are two recordings.

## Recorded runs

| Run | Commit | Recorded in | Command it records |
|---|---|---|---|
| 2026-10-11 | `5f88802`, clean tree | `live_demo/` | the tutorial's old in-package module |
| 2026-10-12 | `82ae95e`, clean tree | `live_demo/2026-10-12/` | `python examples/live_demo/live_demo.py` |

Each folder holds the run's `check.txt`, `check.json` and `manifest.json`. The snapshots are not committed. The first run's stay in `.cache/live_demo/` for `compare_live.py`, and the second's were deleted after recording. Both runs used vLLM 0.30.0, torch 2.13.0 (CUDA 13.0) and an NVIDIA GeForce RTX 2070, with the models already in `HF_HOME`. Both `check.txt` files show the report's tolerance line with the label the CLI prints now, `Bands:`.

The 2026-10-11 run, and the earlier one at `f97124e`, predate metadata rule 9's probe-file check (docs/design.md §6), which rejects a probe file whose recorded threshold or target FPR isn't a number. It doesn't affect capture or the verdict.

**The move.** Between the two runs the tutorial moved out of the package into `examples/live_demo/`. It now calls the `hulder check` command rather than the CLI's internals. The configs, prompts and check are unchanged. So the first run's manifest names the old command, and its files are left as they were written. To repeat a run today, from the repo root:

```bash
source scripts/dev-env.sh && uv run python examples/live_demo/live_demo.py --out .cache/live_demo --force
```

The 2026-10-12 run used `--out .cache/live_demo_2026-10-12`. Its captures took 64 s, 63 s and 68 s, and the whole command 4 min 32 s. The 2026-10-11 run took 50 s, 40 s and 49 s, and 3 min 10 s in all. Neither timing is a benchmark.

## Exit criterion

| Condition | 2026-10-11 | 2026-10-12 |
|---|---|---|
| Runs end to end on the local GPU | Yes, exit 0 | Yes, exit 0 |
| The rerun gives a noise floor: decisions flipped, mean \|Δscore\| | 0, 0.000286 | 0, 0.000205 |
| The AWQ shift exceeds it: decisions flipped, mean \|Δscore\| | 11, 0.0514 | 11, 0.0514 |
| Candidate minus rerun mean \|Δscore\| | 0.0511 [0.0455, 0.0576] | 0.0512 [0.0455, 0.0576] |
| Verdict, reported whatever it is | **INCONCLUSIVE** (exit 12) | **INCONCLUSIVE** (exit 12) |
| Candidate FPR against a 10% band | 7.2% [3.9%, 12.0%] | 7.2% [3.9%, 12.0%] |

In both runs the verdict, flip counts and FPR interval match the bundled quickstart (the README's "How to read a report").

**The noise floor varies between runs.** An earlier run at `f97124e`, before the resume and manifest changes and with no files kept, gave the same verdict, flips and FPR. Across the three runs the rerun's mean \|Δscore\| was 0.000223, 0.000286 and 0.000205. Every capture is a fresh engine on the same 360 prompts, so the noise floor of two fp16 reruns on one GPU varies by up to about 40%, because captures are not bit-identical across engine runs (docs/design.md §10). The AWQ shift is 180 to 250 times the noise floor across the three runs.

## Against the bundled snapshots

Not bit-identical, as expected. These 360 prompts were captured on their own, while the bundled snapshots came from a capture of all 16,332 probe prompts, so batch-mates differ, and batch composition changes activations slightly. For the 2026-10-11 run, from `experiments/phase1/probe/compare_live.py`, written to `live_demo/vs_bundled.json`. It was not rerun for the 2026-10-12 run, whose verdict and flips are the same:

| Live run | Bundled | Median relative L2 | Max | Mean \|Δscore\| | Decisions differing |
|---|---|---|---|---|---|
| fp16 | demo_fp16 | 1.2e-3 | 2.1e-3 | 2.7e-4 | 0 |
| fp16_rerun | demo_fp16_run1 | 1.2e-3 | 2.1e-3 | 3.2e-4 | 0 |
| awq | demo_awq | 4.6e-3 | 8.5e-3 | 1.2e-3 | 0 |

The batch-composition effect is about the size the earlier rerun study found at layer 20 (around 3e-3), and about four times larger under AWQ, matching AWQ's run-to-run noise, about four times fp16's (docs/design.md §10).

## Where this loses

- Three runs on one GPU. It shows the tutorial works on Turing, not what the verdict is on other hardware. Near a boundary (13 of 180 benign flagged, PASS at 10 or fewer), another GPU could give PASS.
- No bf16 and no FP8, which Turing lacks.
