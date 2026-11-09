# Qwen3-4B sweep: results so far

**A: AWQ runs in bf16, and FP8 KV is unsettled. B: the test rerun is PASS, as predicted. C: AWQ int4 is INCONCLUSIVE on calibration, the primary verdict, and THRESHOLD STALE on test, its replication, both within the predictions.** AWQ pushes the probe's scores up on every flipped prompt, with no measurable loss of ranking. The rules and predictions were committed before any run, in `sweep.md` and `sweep_expected.json`. D is skipped, since C answered the question, and FP8 KV stays unsettled.

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

## C. AWQ int4

Run `20261109T130049Z`, three single-use L4 containers on three GPUs, each AWQ in bf16 on the canary and 675 batches, rows 10,936 to 16,331 of the full request. Report in `sweep_c.json`, manifest in `modal/20261109T130049Z/`.

**Inputs and stop rules.** All pass, and no rule changed after the run.

- `kept/`, Experiment 2's control and shuffled rerun, and B match their recorded hashes, and `kept/` equals the control bit for bit.
- C0. All three replicas finished and their files verified against their containers' hashes.
- C1. The replicas were fed the same prompts, tokens and metadata, and every row's tokens equal `kept/`'s.
- C2. Of 676 batches, 655 are unanimous, 21 split 2 to 1, and none lacks a majority. Replica r2 is the outlier on 20 of the splits and r3 on 1.
- All three canaries are the hash A's AWQ smoke recorded, a path no bf16 capture took, as expected.

**Verdicts against the predictions.**

| | Predicted | Verdict |
|---|---|---|
| Primary, calibration at 0.8025 | INCONCLUSIVE or THRESHOLD STALE | **INCONCLUSIVE**, exit 12. 3,612 more benign would settle the FPR condition at the observed rate |
| Replication, test at 0.4524 | THRESHOLD STALE more likely than not | **THRESHOLD STALE**, exit 10 |

The replication disagrees with the primary and is reported beside it. It can't change the primary verdict.

| Primary probe | Calibration, `kept/` | Calibration, AWQ | Test, `kept/` | Test, AWQ |
|---|---|---|---|---|
| FPR | 0.9% [0.5, 1.6] | 1.6% [1.0, 2.5] | 1.0% [0.5, 1.7] | **5.6% [4.3, 7.2]** |
| FNR | 16.2% [13.8, 18.7] | 7.7% [6.0, 9.6] | 6.0% [4.6, 7.6] | 2.8% [1.9, 3.9] |
| ΔAUROC | | +0.0002 [−0.0004, +0.0008] | | −0.0003 [−0.0010, +0.0005] |
| Decisions flipped, AWQ against the floor | | 85 against 4 | | 83 against 2 |
| Of the flips, newly flagged attack / benign | | 77 / 8 | | 34 / 49 |
| Mean \|Δlogit\|, AWQ against the floor | | 0.94 against 0.031 | | 0.94 against 0.0082 |

- **Every flip goes the same way.** Across both splits and both probes, AWQ newly flags prompts and unflags none. The scores move up, about 0.94 logits on average for the primary, while the ranking holds, with both ΔAUROC intervals inside ±0.001. That is what THRESHOLD STALE describes, a threshold that no longer means what it did on a scorer that still separates.
- **Why the splits differ.** Calibration's threshold, 0.8025, sits where benign scores are sparse, so the upward shift adds 8 benign flags, and an FPR of 1.6% leaves the upper bound at 2.5%, past the 2% band but not clearly. Test's threshold, 0.4524, sits lower, where more benign prompts lie within reach of the shift, so it adds 49 and the lower bound, 4.3%, clears twice the target. So the replication's sharper verdict owes part of its strength to its lower threshold, not only to AWQ.
- **The trade at the deployed threshold.** On calibration AWQ catches 77 more attacks for 8 more false positives. A team reading only the FPR line would call AWQ worse, and only the FNR line, better. The gate doesn't yet say the threshold is stale on calibration. It offers 0.9102 as an optional recalibration for a 1% target, at an FNR of 19.5% [16.9, 22.2], fitted on these same scores and to be confirmed on a fresh AWQ capture.
- **The floors don't decide anything here.** AWQ's mean |Δscore| is 35 times calibration's floor and 131 times test's, so the smaller test floor noted under B makes no difference to C.

**Secondary probe, reported beside.** On calibration at its frozen 0.6855 it is INCONCLUSIVE, with FPR 0.9% to 2.3% [1.5, 3.3], 35 flips against 3, all newly flagged. On test it has paired shift only, 73 flips against 0, all newly flagged.

**Boundary-pairs, not counted.** Both probes say THRESHOLD STALE, as in Experiment 2, because the bf16 reference already flags 88% and 92% of benign there. AWQ adds 33 benign flags for each probe. The primary's AUROC moves from 0.685 to 0.674 and the secondary's from 0.845 to 0.860, both with overlapping intervals.

**Compared with Phase 1.** Qwen2.5-1.5B with its official AWQ checkpoint showed the same pattern, INCONCLUSIVE on calibration and STALE on test, but only post hoc (`docs/results/phase1/awq_posthoc.md`). Here the pattern was predicted before the run and recurs on a second model.

**Seen after the run, not part of any rule.** 21 of 676 AWQ batches split, 3.1%, against 4 of 414 for Experiment 2's FP8, 0.97%, and none of 2,042 for the full capture's bf16. One replica, r2, accounts for 20. The majority caught them, and nothing was dropped. Whether AWQ's kernels deviate more often, or r2's GPU did, one run can't say.

**How the run came back.** The local process died on its first poll of the containers, from a bug in the runner, now fixed. The run was detached, so the containers finished on their own. `recover_run.py` rebuilt all three from the Volume, checked against their containers' hashes, so this is the first real use of the recovery path. The containers' wall times were lost with the local process, so C has no cost estimate here.

## Cost

Estimates from wall time, which include time before the engine starts that Modal may not bill. A came to $0.10. B came to $0.23, about three times the $0.08 expected, because 575 of its 662 s passed before the worker started, while its engine ran 86 s. A new commit means a new wheel and so an image rebuild, which probably accounts for much of that. C's three engines ran 127, 102 and 112 s. The containers waited about 13 minutes for GPUs before starting, which Modal shouldn't bill. The day's actual charge before C was $0.14, and C's is on Modal's usage page.

## Reproduce

```bash
source scripts/dev-env.sh
uv run --with modal==1.6.1 modal run --detach experiments/modal/capture.py --experiment sweep_a --spent-usd <today> --backup-dir <dir>
uv run --with modal==1.6.1 modal run --detach experiments/modal/capture.py --experiment sweep_b --spent-usd <today> --backup-dir <dir>
uv run python experiments/phase4/probe/sweep_b_report.py .cache/modal-runs/20261109T115553Z
uv run --with modal==1.6.1 modal run --detach experiments/modal/capture.py --experiment sweep_c --spent-usd <today> --backup-dir <dir>
uv run python experiments/phase4/probe/sweep_c_report.py .cache/modal-runs/20261109T130049Z .cache/modal-runs/20261109T115553Z
```

A ran at `ec0cde7` and B at `f42f232`. B's report ran at `bd93786` on CPU. C ran at `78c2030`, was recovered from the Volume at `eea45e9`, and its report ran at `eea45e9` on CPU. Both reports refuse a dirty tree and run once per capture. If a run's local process dies, `experiments/modal/recover_run.py --run-id <run id>` rebuilds it from the Volume.
