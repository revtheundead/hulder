# Modal L4 captures: smoke run, FP8 paths and two bf16 numerical paths

First captures on the remote GPU, 2026-10-13, and a census of the bf16 paths on 2026-10-15. Every run used an NVIDIA L4 on Modal (Ada, sm_89, 24 GB) with vLLM 0.30.0, Qwen3-4B at `1cfa9a72` with `enable_thinking=False`, layers 18 and 27 at the last prompt token, batch 8, prefix caching and chunked prefill off, and the first 32 benign and 32 attack prompts of `examples/live_demo/demo_prompts.jsonl`. 64 prompts and no Qwen3 probe are too few for a verdict, so these runs compare activations only.

Run manifests, per-run manifests and comparisons are in [`modal/`](modal/), one directory per run, and `modal/bf16_clusters.json` groups every bf16 capture. Snapshots are not committed.

## Reproduce

```bash
source scripts/dev-env.sh
uv run --with modal==1.6.1 modal run experiments/modal/capture.py --experiment <name> --spent-usd <spent today>
uv run python experiments/modal/compare.py .cache/modal-runs/<run id>
uv run python experiments/modal/clusters.py docs/results/phase0/modal/bf16_clusters.json .cache/modal-runs/<run id> ...
```

| Run | Experiment | Commit | What ran | Estimated cost |
|---|---|---|---|---|
| `20261013T141533Z` | smoke | `dfd9be3` | bf16, bf16 again, Qwen3-4B-FP8 checkpoint, Qwen3-8B bf16 load with one prompt | $0.31 |
| `20261013T144543Z` | fp8_online | `581d867` | bf16, then vLLM's online per-tensor FP8 of the same weights, in one container | $0.11 |
| `20261013T145138Z` | cross_container | `581d867` | bf16 in two single-use containers started at once | $0.13 |
| `20261013T150939Z` | batch_invariant | `a20a7c0` | bf16 three times with cuBLASLt tracing, then three times with `VLLM_BATCH_INVARIANT=1`, in one container | $0.36 |
| `20261015T114213Z` | two_path_census | `4298438` (manifest says `a04e0708074b`, below) | Two single-use containers, each running bf16 five times with cuBLASLt tracing, then once with `VLLM_BATCH_INVARIANT=1` | $0.24 |

The census manifest records commit `a04e0708074b`, but `4298438` ran. The checkout was switched to another branch while the runner started, and the runner reads the commit after starting the app. The two commits differ only in `experiments/modal/`, in the experiment definition and the analysis helpers. `worker.py` and the hulder wheel the containers ran are identical at both.

The smoke run predates named experiments. At `dfd9be3` it was `modal run experiments/modal/capture.py` with no arguments, four calls that Modal happened to serve from one warm container. `--experiment smoke` now runs the same configs in one container on purpose.

## Memory and kernels

| Config | KV cache | Linear kernel | Notes |
|---|---|---|---|
| Qwen3-4B bf16 | 11.42 to 12.07 GiB | default cuBLAS | 81,648 tokens at 11.99 GiB |
| Qwen3-4B-FP8 checkpoint | 14.69 GiB | `MarlinFP8ScaledMMLinearKernel` | Weight-only FP8 (W8A16) |
| Qwen3-4B, online FP8 | 14.50 GiB | `CutlassFP8ScaledMMLinearKernel` | FP8 weights and activations (W8A8) |
| Qwen3-8B bf16 | 2.05 GiB | default cuBLAS | 13,440 tokens, 13× concurrency at 1,024 tokens |

- **FP8 on Ada.** vLLM 0.30.0 tries DeepGEMM and CUTLASS block-FP8 kernels for Qwen's 128×128 block-quantized checkpoint, and on the L4 selects Marlin, which dequantizes the weights and computes in bf16. Online per-tensor quantization of the bf16 weights selects CUTLASS, a real W8A8 path. So an FP8 result on an L4 depends on which of the two a team deploys.
- **Qwen3-8B fits for capture only.** At 0.9 memory utilization its bf16 weights (15.27 GiB loaded) leave 2.05 GiB of KV cache, below the 5 GiB estimated beforehand. Capture at batch 8 and 1,024 tokens needs 8,192 tokens and fits, but large-batch or throughput configs would be cramped.
- **Drivers differ between hosts.** The L4s ran driver 580.95.05 or 610.57.04, and identical bf16 configs logged 11.42 to 12.07 GiB of KV cache across hosts and modes here, and 10.67 GiB in an earlier failed attempt on another host. Snapshots now record `gpu_uuid` and `gpu_driver`, so checking runs from two hosts warns (docs/design.md §6, metadata rule 2).

## Activation shift against bf16

Relative L2 per prompt, mean and maximum over the 64 prompts, each candidate in the same container as its bf16 reference.

| Candidate | Layer 18 | Layer 27 |
|---|---|---|
| FP8 checkpoint (Marlin, W8A16) | 0.065 / 0.082 | 0.059 / 0.120 |
| Online FP8 (CUTLASS, W8A8) | 0.088 / 0.111 | 0.080 / 0.132 |
| bf16 on the other numerical path (below) | 0.013 / 0.017 | 0.012 / 0.019 |

## Two numerical paths in bf16

Eighteen default bf16 captures of one config fall into two clusters. Each cluster is bit-identical on all 64 prompts at both layers, and the clusters differ on every prompt.

| Cluster | Captures | GPUs | Drivers | Days |
|---|---|---|---|---|
| 1 | 16 | 6 | 580.95.05, 610.57.04 | 2026-10-13, 2026-10-15 |
| 2 | 2 | 2 | 580.95.05, 610.57.04 | 2026-10-13 |

- **Size.** Between clusters the relative L2 has median 0.0131 at layer 18 (range 0.0090 to 0.0173) and 0.0111 at layer 27, with about 90% of dimensions changed, evenly across all eight batches. That is a switch of computation path, not run-to-run noise.
- **What it is not.** Not the GPU or driver, since both clusters span both drivers and two single-use containers on different GPUs and drivers agreed bit for bit. Not the container, since one container moved from cluster 1 to cluster 2 between consecutive engine processes. Not the engine arguments, identical once random paths are removed. Not the compiled graph, since all eight load the same AOT artifact.
- **How often.** The census ran ten default engines, five in each of two fresh containers on two GPUs, to catch a cluster-2 run with tracing on. All ten landed on cluster 1. At the 2 in 8 seen on 2026-10-13 that happens 5.6% of the time (0.75¹⁰). Pooled, cluster 2 is 2 of 18 captures, 11% [1.4%, 34.7%] (Clopper-Pearson). Both cluster-2 captures came on 2026-10-13, in the runs started at 14:15 and 14:45 UTC, and all 15 default captures since, over two days, landed on cluster 1. Whether cluster 2 is rare or was tied to something transient that day is unknown.
- **Cause.** Unidentified. No cluster-2 run has been traced. All 13 traced engines, three on 2026-10-13 and ten in the census, chose the same set of 99 cuBLASLt matmul shapes and algorithms, which is the baseline for the first traced cluster-2 run. CUDA-graph replays make no cuBLAS calls, so a replay's kernels are covered only if capturing the graph logged them.
- **Batch-invariant mode reproduces across containers.** Its five captures, three in one container on 2026-10-13 and one in each census container on 2026-10-15, are bit-identical, on three GPUs. All five ran on driver 610.57.04, so a change of driver is untested.
- **Batch-invariant mode takes effect on sm_89.** vLLM logs nothing when it turns on. Its effect is inferred from two signals. The AOT-load line, present in every default run, is absent, as the mode's `VLLM_USE_AOT_COMPILE=0` implies. And all three invariant captures are bit-identical to each other and sit 0.0139 (layer 18) from cluster 1 and 0.0141 from cluster 2, a third path. On sm_8x the mode swaps the GEMMs for Triton persistent kernels and runs FlashAttention with one split. The census then tested it across containers (above).
- **Score space is untested.** The 1.3% is in activation space. How far it moves a probe's scores and decisions needs a Qwen3 probe.

So the earlier reading of the smoke rerun, a bf16 noise floor about ten times fp16's, was wrong. Within a path the floor measured here is zero. A rerun that lands on the reference's path then measures no floor, while a candidate on the other path shows a shift the serving change didn't cause.

## Timing and cost

- **Cold start dominates the first run.** On a freshly built image the first config took 572 s of wall time, of which the worker ran 116 s. Later configs in a warm image took 60 to 130 s of wall time.
- **Engine start.** A warm bf16 engine initialises in 12 to 47 s. Compiling online FP8 from cold took 122 s, and batch-invariant mode 201 s on first compile and about 33 s after. cuBLASLt tracing added about 50 to 80 s to engine start in the batch_invariant run, so its default runs are not timed fairly against the others.
- **Cost.** Modal's usage page shows $0.90 for the day, failed attempts included. The runner estimated $1.19, the runs above $0.90 and earlier failed attempts about $0.29, by charging list rates for the full CPU and memory requested over container time measured locally. So the estimates run about 30% high, which errs on the safe side for a stop rule.
- **The census was cheaper than estimated beforehand.** Both containers started warm, engines initialised in 9 to 15 s (43 to 56 s with batch-invariant mode, mostly compilation), and the containers ran 312 s and 381 s. The runner estimated $0.24 against $0.75 forecast.
- **Unreliable timer.** vLLM's second "Loading weights took" line reported up to 404 s, more than whole worker runs, and is left out of the comparisons.

## What it took to get here

Four failures came before the first full run, each fixed in the runner or the library:
1. Modal refuses to mount a Volume on a non-empty path, and setting cache directories inside the mount path in the image's environment populated it. Cache paths are now set at run time.
2. Copying hulder's source into the image left out its package metadata, which `hulder.__version__` reads. The image now installs a wheel built from the checkout.
3. On Ada, vLLM's warmup samples through a FlashInfer kernel that compiles with `nvcc`, which the image lacks. Captures set `VLLM_USE_FLASHINFER_SAMPLER=0`. Sampling follows the prefill the activations come from, so it can't change them.
4. `hulder.capture` called `.numpy()` on bf16 tensors, which numpy can't represent. Fixed in 0.1.1, which converts through float32 and stores float16.

## Open

- The cause of the two bf16 paths. Two of 18 default captures landed on cluster 2, both on 2026-10-13.
- Whether batch-invariant mode keeps one path across drivers.
- The score-space size of a path switch, once a Qwen3 probe exists.
- With `enable_thinking=False`, Qwen3's template ends in an empty thinking block, so the last prompt token is always `"\n\n"`. The token-position ablation (docs/design.md §5, D2) compares it with the last content token.
