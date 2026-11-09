# Qwen3-4B sweep: results so far

**A: AWQ runs in bf16, and FP8 KV is unsettled. B: the test rerun is PASS, as predicted, and gives the replication its noise floor.** The rules and predictions were committed before any run, in `sweep.md` and `sweep_expected.json`. C and D haven't run.

## A. Smoke

Run `20261109T112130Z`, one L4 container, estimated at $0.10. Manifest in `modal/20261109T112130Z/`.

| Arm | Predicted | Outcome |
|---|---|---|
| AWQ in bf16, 64 smoke prompts | Runs | **As predicted.** vLLM 0.30.0 routed `Qwen/Qwen3-4B-AWQ` to `auto_awq` with `MarlinLinearKernel`, cast the checkpoint's float16 to bfloat16, and used FLASH_ATTN. All activations finite, the largest absolute value 87.5. The snapshots record `quantization: awq`, `dtype: bfloat16` and `base_model: Qwen/Qwen3-4B` |
| bf16 with an FP8 KV cache, 1 prompt | Engine fails on `CacheOnlyAttentionLayer`'s KV-dtype assertion | **Prediction wrong, unsettled.** No assertion fired. The engine built the extraction layers with an FP8 KV cache, switched attention to FLASHINFER, and died at CUDA-graph profiling on a missing nvcc when FlashInfer tried to compile its FP8 KV prefill kernel |

So C runs AWQ in bf16, and A2 and C's fp16 arm don't apply. FP8 KV is revisited only if D runs, and nothing goes into `docs/design.md` §10. AWQ's canary is a new hash, recorded in the run's logs, as expected for a new checkpoint. Its first engine start took 160 s, most of it the first compile.

## B. Test rerun

Run `20261109T115553Z`, one L4 container, bf16 on the 2,112 rows of the test batches behind the canary, in `default_rng(1)` order. Report in `sweep_b.json`, manifest in `modal/20261109T115553Z/`.

**Inputs and stop rules.** All pass.

- `kept/` matches the file hashes the plan recorded, Experiment 2's control matches its committed manifest, and `kept/` equals the control bit for bit on the control's batches.
- B0. The container finished and its files verified.
- B1. All 2,101 test rows were fed `kept/`'s prompts and tokens.
- B2. B's canary is cluster 2.0, the bf16 majority path, with the same hash as 16 earlier captures.

**The null, at the replication threshold 0.4524.** `hulder check` with `kept/`'s test rows as reference and B's as candidate gives **PASS, exit 0**, as predicted. The planted-fault control predicted INCONCLUSIVE for about 12% of draws at this size, so a PASS was the likely outcome, not a strong one.

| Primary probe, test, 1,050 benign and 1,051 attack | `kept/` | B |
|---|---|---|
| FPR | 1.0% [0.5, 1.7] | 1.0% [0.5, 1.7] |
| FNR | 6.0% [4.6, 7.6] | 6.0% [4.6, 7.6] |
| AUROC | 0.9940 [0.9915, 0.9963] | 0.9940 [0.9915, 0.9963] |

The check warns about the rerun's `batch_order_seed` and its GPU, both expected.

**The noise floor C's replication will use.**

| B against `kept/` on test | Primary, at 0.4524 | Secondary, at its frozen threshold |
|---|---|---|
| Rows bit-identical | 1,630 of 2,101 | 1,630 of 2,101 |
| Decisions flipped | 2 (0.10%), one attack each way | 0 |
| Mean \|Δscore\| | 0.00054 | 0.00034 |
| Mean \|Δlogit\| | 0.0082 | 0.0060 |
| Largest \|Δlogit\| | 0.149 | 0.135 |

**The test floor is smaller than calibration's, so the test replication is the stricter check.** Experiment 2's shuffled rerun left only 355 of 2,101 calibration rows bit-identical, against 1,630 of 2,101 test rows here. Its mean |Δlogit| on calibration was 0.031, against 0.0082 here. Both used `default_rng(1)` on their own slices. With a smaller floor, a smaller shift on test counts as exceeding it, so C's replication can call a shift real where the primary wouldn't. The floor is a diagnostic and doesn't enter the verdict (`docs/design.md` §6), so this affects the shift line of the report, not the verdict rule.

**Best guess at why, unverified.** B shuffled only test prompts among themselves, all from one S-Labs split. Experiment 2's shuffle mixed calibration prompts with boundary-pairs prompts, which differ more in length. So B's shuffled batches probably changed their total token counts less, and with them the matrix shapes and kernel choices that move a row's activations. No run has tested this.

## Cost

Estimates from wall time, which include time before the engine starts that Modal may not bill. A came to $0.10. B came to $0.23, about three times the $0.08 expected, because 575 of its 662 s passed before the worker started, while its engine ran 86 s. A new commit means a new wheel and so an image rebuild, which probably accounts for much of that. The actual charge is on Modal's usage page.

## Reproduce

```bash
source scripts/dev-env.sh
uv run --with modal==1.6.1 modal run --detach experiments/modal/capture.py --experiment sweep_a --spent-usd <today> --backup-dir <dir>
uv run --with modal==1.6.1 modal run --detach experiments/modal/capture.py --experiment sweep_b --spent-usd <today> --backup-dir <dir>
uv run python experiments/phase4/probe/sweep_b_report.py .cache/modal-runs/20261109T115553Z
```

A ran at `ec0cde7` and B at `f42f232`. B's report ran at `bd93786` on CPU, refuses a dirty tree, and runs once per capture.
