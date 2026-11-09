# Qwen3-4B serving-config sweep: the plan

Written and committed before any of its runs. Each run's rules and predictions are in `sweep_expected.json`, committed with this page, and no rule changes after a run. Every config that runs is reported, including where the gate or the prediction is wrong.

## What it adds

Experiment 2 (`exp2.md`) compared bf16 with online FP8 on calibration only. The sweep adds a measured noise floor on test, then the official AWQ int4 checkpoint, then one harsher config chosen after AWQ.

| Step | What runs | Expected | Worst case |
|---|---|---|---|
| A. Smoke | One container. AWQ in bf16 on the 64 smoke prompts, then one bf16 engine start with an FP8 KV cache through the extraction path | ~$0.06 | $0.45 |
| A2. Only if AWQ fails in bf16 | One container. AWQ in fp16 and unquantized Qwen3-4B in fp16, each on the 64 smoke prompts | ~$0.06 | $0.45 |
| B. Test rerun | One container. bf16 on the full capture's test batches in a seeded shuffled order | ~$0.08 | $0.28 |
| C. AWQ int4 | Three single-use containers, kept by batch majority, on the batches holding calibration, test and boundary-pairs | ~$0.45 | $1.09 |
| C's fp16 arm. Only if A shows AWQ can't run in bf16 | Four single-use containers. Unquantized fp16 as three replicas and one shuffled fp16 rerun, on C's batches | ~$0.60 | $1.45 |
| D. One harsher non-KV config | Decided after C has been read | ~$0.45 | $1.09 |

Worst cases come from `smoke_lib.worst_case_cost`, each config's timeout plus 333 s of container startup at Modal's published L4 rates. The guard in `capture.py` refuses any container whose worst case, added to the day's spend and the run so far, would pass $3. So C and its fp16 arm can't share a day, and the fp16 arm runs on a later day if needed. Expected costs scale Experiment 2's containers, about $0.11 for 3,300 prompts, by batch count. The actual charge is read from Modal's usage page.

## Data and probes

- **Prompts.** The full capture's request (`full_capture.md`), canary first, from `splits.parquet` with sha256 `13122a029b5442592da011ace5d59e43009fb599793d93ab20a9593c5e0c04b4`, the hash `probe_qwen3.json` records.
- **Reference.** `kept/` of run `20261022T125644Z`. Its per-file hashes weren't committed before, so `sweep_expected.json` records them now, from its `SHA256SUMS`, which still verifies. Experiment 2's control, whose hashes its committed manifest holds, equalled `kept/` bit for bit on the calibration and boundary-pairs batches. The reports check that again before reading `kept/`.
- **Probes.** The frozen primary, layer 18 at position −1, gives the verdict. Layer 27 at −1 is reported beside it, as in Experiment 2. Their hashes match `exp2.json`.
- **Row ranges.** In the full request, calibration is rows 10,938 to 13,038 and test 13,039 to 15,139. Batches of 8 straddle both ends of test, so the test batches are rows 13,032 to 15,143, and C's slice, calibration through boundary-pairs, is one range, rows 10,936 to 16,331, 675 batches. Every batch keeps its full-capture composition.

## Verdicts

The verdict rule, its tolerances and the target FPR of 1% are unchanged (`docs/design.md` §6).

- **Primary.** Calibration, 1,191 benign and 910 attack, at the frozen threshold 0.8025, with Experiment 2's shuffled rerun as the noise floor.
- **Replication.** Test, 1,050 benign and 1,051 attack, at the threshold refit on `kept/`'s test benign scores, 0.4524, as `planted_faults.md` declared it, with B as the noise floor. It's judged on its own and can't change the primary. A replication that disagrees is reported beside the primary.
- **Boundary-pairs.** Reported as in Experiment 2, not counted toward any verdict.

## Each step

### A. Smoke

AWQ is `Qwen/Qwen3-4B-AWQ` at `74d4bd2b`, with `quantization="awq"` as in Phase 1. vLLM 0.30.0 routes AWQ to `auto_awq`, which lists float16 and bfloat16 as supported activation dtypes (`model_executor/layers/quantization/auto_awq.py:226`). So the prediction is that AWQ runs in bf16. A settles it, and records the kernel vLLM selects.

The FP8 KV arm runs the full capture's bf16 config with `kv_cache_dtype="fp8"` passed to the engine by the runner, not through `CaptureConfig`, which has no field for it. It requests one prompt behind the canary and saves no snapshot. The prediction is that the engine fails to start on `CacheOnlyAttentionLayer`'s assertion that the KV dtype is `auto`, `bfloat16` or `float16` (`model_executor/models/extract_hidden_states.py:253`). The container runs both arms even if the first fails.

- **If the FP8 KV arm fails as predicted,** `docs/design.md` §10 records that at vLLM 0.30.0 the extraction path can't capture, or score live, under an FP8 KV cache. D is then a harsher non-KV config.
- **If it starts,** that limit doesn't exist, and FP8 KV becomes a candidate for D.

**Outcome, added after run `20261109T112130Z`.** AWQ ran in bf16, as predicted, so A2 and C's fp16 arm don't apply. The FP8 KV prediction was wrong, and the question is unsettled. No assertion fired. The engine built the extraction layers with an FP8 KV cache, switched attention from FLASH_ATTN to FLASHINFER, and died at CUDA-graph profiling on a missing nvcc, when FlashInfer tried to compile its FP8 KV prefill kernel. Neither branch above applies, so nothing goes into `docs/design.md` §10. It is revisited only if D runs.

### A2 and C's fp16 arm, pre-declared

If AWQ can't run in bf16, C runs AWQ in fp16 and the comparison with bf16 changes two things at once, the weights and the dtype. To separate them, C then adds an unquantized fp16 arm. A2 first checks that AWQ and unquantized Qwen3-4B both start in fp16 and capture the smoke prompts without overflow, since a value beyond float16's range stops a capture (`docs/design.md` §5).

| Comparison | Reference | Candidate | Noise floor | Role |
|---|---|---|---|---|
| The change a team would ship | bf16, `kept/` | AWQ in fp16 | as for C | Primary and replication, as declared above |
| The dtype alone | bf16, `kept/` | unquantized fp16 | as for C | Reported, no verdict claim |
| AWQ alone | unquantized fp16 | AWQ in fp16 | the shuffled fp16 rerun | Reported, no verdict claim |

The AWQ-alone row keeps the deployed thresholds, which weren't set on fp16, so `check` warns if fp16's own FPR interval excludes the target. Its verdict is reported with that warning and isn't read as the gate's answer. If A2 fails, C doesn't run, and the plan comes back for a decision.

### B. Test rerun

The full capture's bf16 config on the test batches, in `default_rng(1)` order with the canary still batch 1, as Experiment 2's rerun was. It is the null for the replication and its noise floor.

- **The null.** B against `kept/` on test at 0.4524. PASS is predicted. In the planted-fault control the same comparison, with rerun residuals drawn onto test, was non-PASS in 11.8% of resamples, all INCONCLUSIVE, so a single INCONCLUSIVE here doesn't count as the null failing. STALE or DEGRADED would.
- **Stop rules.** B0, the container failed or its files didn't verify. B1, B's test rows weren't fed the same prompts and tokens as `kept/`'s. B2, B's canary isn't the bf16 majority path's, cluster 2.0. On B2 B is reported as a deviation and isn't used as a floor, and a repeat needs a new go-ahead with its cost.

### C. AWQ int4

AWQ on C's slice, three replicas, each in its own fresh container, each batch kept from its majority, as Experiment 2's FP8 was.

- **Prediction.** INCONCLUSIVE or THRESHOLD STALE on calibration, and STALE more likely on test, from Phase 1's pattern on Qwen2.5 (`docs/results/phase1/awq_posthoc.md`). A prediction, not a bar.
- **Stop rules.** C0, a replica failed or its files didn't verify. C1, the replicas weren't fed the same prompts, tokens and metadata, or their rows' tokens differ from `kept/`'s. C2, more than 1% of batches have no majority. AWQ has no recorded numerical path, so its canary hash is recorded rather than checked.
- **Changes under test.** `model`, `revision` and `quantization`, and `dtype` too if A2 applies. Any other metadata difference warns and is reported.

### D

Chosen after C has been read, and only after A's FP8 KV answer. The options, none of them settled: bitsandbytes NF4 at load, which needs `bitsandbytes` added to `uv.lock`; a community GPTQ checkpoint, which is unofficial; or AWQ alone. D gets its own rules and predictions, committed before it runs.

**Outcome, added after C.** D is skipped. C answered the sweep's question, whether the gate reads a real quantization change on Qwen3-4B, and no harsher config is needed for that. FP8 KV stays unsettled, as step A left it.

## How each run keeps what it has

The machine that launches runs has lost files and in-memory results before (`docs/design.md` §10 covers what that means for captures). So:

- **On Modal,** each config writes its snapshots, log, manifest and file hashes to the `hulder-runs` Volume and commits it as soon as its engine finishes, before the next config starts.
- **Locally,** each config's folder and the run manifest are written as soon as that config's results arrive, and single-use containers are collected in the order they finish. A multi-config container streams each config back as it finishes, instead of returning them all at the end.
- **If the local process dies,** `experiments/modal/recover_run.py` rebuilds the run's folders from the Volume, checking each file against the hash the container recorded. Runs start with `modal run --detach`, so the containers finish without the local process.
- **CPU reports** that take more than a few minutes write each row as it finishes, through `experiments/phase4/probe/resumable_rows.py`.
- **After each run,** the folder is scanned with `export/scan_cache.py`, backed up to a new dated folder with `SHA256SUMS`, and the copy is verified and scanned.

A single engine's capture still returns its activations at the end. One config at C's size takes about 3 minutes on the GPU, and writing per batch would change `hulder.capture`'s public API, so the config stays the smallest unit kept.
