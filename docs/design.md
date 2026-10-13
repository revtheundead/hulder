# Hulder design

How Hulder works and why: the problem, the deliverables, the design decisions, the statistical method behind `hulder check`, the experiment and the roadmap. The [README](../README.md) is the user guide. Results are in [`docs/results/`](results/), and the terms used here are defined in [`CONTEXT.md`](../CONTEXT.md).

## 1. The problem

> **Status of this section.** The prior-art claims below were gathered from search results and paper summaries, not from reading every source in full. Items marked † are listed in §12 and must be verified before anything is built on them.

Frontier labs now screen traffic with **activation probes**, small classifiers that read the served model's own hidden states. Because those activations are computed during generation anyway, a probe costs almost nothing compared with running a second LLM as a judge. Anthropic (Constitutional Classifiers++) and Google DeepMind (production probes for Gemini, arXiv 2601.11516) both published in January 2026 on making probes production-ready, and both converged on a **probe-first cascade**: a cheap probe screens all traffic, and an expensive LLM classifier only sees flagged cases. GDM reports using the LLM less than 10% of the time while getting a lower false-negative rate than the LLM alone.†

Two gaps follow from that.

**Gap 1. Self-hosters don't have this.** Teams serving open models on vLLM typically bolt on guardrails via NeMo Guardrails, LLM Guard or LlamaFirewall, and use a separate guard model (e.g. Llama Guard on its own vLLM server) for model-based checks. That is the expensive pattern. Existing open-source probe tooling is either research-grade or not in a production engine:
- `vllm-lens` runs probes and hooks inside vLLM but forces `enforce_eager=True`, disabling CUDA graphs, which costs throughput.†
- IBM's `vLLM-Hook` targets inspection and steering for research.†
- `AIsafePy` has a probe tier in its cascade, but probe features only work on HF Transformers, not a serving engine.†

Gap 1 is the adoption argument for the whole project, so it is the first thing to verify.

**Gap 2. Probes are calibrated against one serving configuration, and teams keep changing it.** Weight quantization (FP8, AWQ int4), FP8 KV cache, engine upgrades and batching changes all perturb the hidden states a probe reads. The probe's threshold was chosen on the old numbers. The failure is silent in one of two ways:
- scores drift up, the false-positive rate climbs, escalations to the expensive judge multiply, and the cost savings evaporate (and the judge can be overloaded); or
- scores drift down, real attacks slip under the threshold, and nobody notices because the monitor is the thing that degraded.

GDM explicitly names robustness to distribution shift as a hard open problem for probes. Related work shows ordinary typos break single-position probes ("Latent Undertow", arXiv 2609.15994) and studies probe robustness to inference-time numerical non-determinism ("Same Probe, Different Numbers", arXiv 2609.31796). **The deliberate serving-config changes teams make to cut cost are the gap this project targets.** 2609.31796 is the closest prior work and has been read in full. It keeps weights fixed, runs in HF Transformers rather than a serving engine, and reports accuracy and verdict flips rather than FPR/FNR at a tuned threshold or recalibration, so the plan is unchanged.

## 2. What this project is

A tool people should actually use. Three deliverables, in priority order.

1. **`hulder check`**, a release gate shipped as a pip-installable library with a thin CLI and a GitHub Action (§4). It checks any **scorer**, meaning any thresholded score derived from a model, under two serving configs. Activation probes are the flagship case. Given reference and candidate scores (or activations plus a probe) on the same labelled examples, it answers three questions with honest uncertainty. Did the scores shift? Does the scorer still separate the classes? If only the threshold is stale, what is the recalibrated threshold for the same target FPR? It returns a verdict, and the CLI turns that into a CI exit code (§4).
2. **The vLLM runtime plugin.** In-server probe scoring at production settings (CUDA graphs intact, no host-device sync on the hot path), with metrics and drift alarms.
3. **The benchmarks page.** The experiment, run with the tool: train a linear probe on open models, serve them under several serving configs, and publish what `hulder check` says about each. It demonstrates the tool and is reported in full whatever it shows.

The probe-first cascade (gateway, bounded escalation queue, LLM judge, failure policies) is an **example deployment** that shows the plugin and gate in a realistic stack, not a deliverable of its own.

**What the gate can detect so far (measured on synthetic scores).** These numbers come from 1,000 simulated runs per row, with about 1,200 benign and 900 attack examples and a 1% false-positive target. They assume the threshold was set on the reference itself (§6). Real serving configs come next.
- An unchanged serving config passes in every run, including at the run-to-run noise measured between two fresh fp16 vLLM engines.
- If a change pushes the false-positive rate to 5%, the gate says THRESHOLD STALE in 999 of 1,000 runs.
- At 3% it never passes. It says THRESHOLD STALE in 57% of runs and INCONCLUSIVE in the rest.
- At 1.5% to 2% it mostly says INCONCLUSIVE, and reports how many more benign examples would settle it. The median is about 2,200 more at 1.5% and 10,400 at 2%, so small rises need a much bigger calibration set to judge.
- If a change lowers AUROC by 0.05 or more, the gate says SCORER DEGRADED in every run, and it never does for a drop of 0.01.

Details and the cases where it loses are in `docs/results/phase1/gate_calibration.md`.

**Positioning.** This is a component, not a replacement stack. People adopt pieces that slot into what they already run. The target adopter is a team already self-hosting on vLLM who would otherwise run Llama Guard on every request.

**Scorers the gate applies to.** The gate needs only scores and labels, so it applies wherever a threshold was tuned on one serving config and the config then changes.
- *Activation probes* (flagship, v0.1). Hulder captures the activations itself through the `vllm` extra.
- *Semantic-cache similarity* (v0.2 worked example). Paraphrase-based caching: a small embedding model at fp32 and at int8, comparing the cosine-similarity threshold above which a cached answer is served. It uses PAWS-Wiki sentence pairs, whose negatives differ from their pair only in word order, so a false positive is a cached answer served for a sentence that means something else. It runs on CPU and downloads PAWS at runtime (§7).
- *Further use cases, not in scope:* guard-model probabilities (for example Llama Guard's probability of "unsafe") and embedding classifiers. Both work through bring-your-own scores but get no worked example.

### Non-goals

- Producing scores for every kind of scorer. Beyond activation probes, users bring their own scores.
- Inventing new probe architectures. A linear (logistic regression) probe is deliberately simple; the project is about operating probes, not designing them.
- A general guardrail framework. Integrate with existing ones as a scorer instead.
- Per-token streaming monitoring in the first version. Start with one score per request at the end of prefill.
- Multi-GPU, tensor parallelism, or datacenter-scale load. One GPU, realistic loads, honest extrapolation.
- Beating Llama Guard on accuracy. The claim is cost and latency at acceptable accuracy, plus knowing when accuracy has silently dropped.
- A llama.cpp or custom C++ engine. Possible later port; not in scope now.

## 3. Worked example (the behaviour we are building toward)

A team serves a ~9B open model on vLLM in bf16, with a prompt-attack probe on layer 20, thresholded at 0.730 for a 1% false-positive rate. They want to switch to FP8 weights and FP8 KV cache to halve memory.

They capture layer-20 activations for a labelled calibration set under each serving config: the reference twice, so the gate can measure the noise floor, and the candidate once. v0.1 has no capture command. The `vllm` extra's `hulder.capture` runs vLLM offline with the extraction settings of §5, or any other capture works. Its `capture`, `CaptureConfig` and `snapshot_metadata` are public but experimental in v0.1 and may change in 0.x (§4). Each run is saved as a `Snapshot`, and `hulder check` compares the candidate with the reference on CPU. The live tutorial's "Your own serving change" (`docs/tutorial-live.md`) shows the code, and `docs/reading-a-report.md` walks through real output on the bundled fp16 vs AWQ int4 demo. In this FP8 scenario the verdict would read THRESHOLD STALE and the exit code 10.

A THRESHOLD STALE verdict means the scorer still separates the classes but the threshold no longer gives the target FPR, so it needs recalibrating on the candidate. v0.1 reports the verdict only. A recalibrated threshold for the same target FPR is planned for v0.2 (§11). From Python, `hulder.check_snapshots(reference, candidate, rerun, threshold=..., target_fpr=..., probe=probe)` does what the CLI does on loaded `Snapshot`s and returns the `Report`. `hulder.check` takes plain arrays. Scores from any other scorer go in as JSONL instead of snapshots (v0.2, §11).

The other important verdict is `SCORER DEGRADED` (e.g. AUROC falls to 0.81), where no threshold fixes it and the probe must be retrained on the new config. **The scenario's numbers (0.730, 3.4%, 0.81) are illustrative.**

In production the runtime emits:

```
probe_score_p50{probe="attack_v3"}             0.12
probe_flag_rate{probe="attack_v3"}             1.1%
probe_drift_stat{probe="attack_v3"}            0.08   (alarm at 0.25)
probe_latency_ms_p99{probe="attack_v3"}        0.3
probe_errors_total{probe="attack_v3"}          0
escalation_queue_depth                         3
judge_circuit_state                            closed
```

The capacity link is what makes this a systems problem. At 2,000 req/s with a judge that handles 50 req/s, a 1% flag rate is 20 req/s (fine). A stale threshold pushing flags to 3.4% is 68 req/s, and an unbounded queue grows until something falls over. The statistical failure causes the capacity failure.

## 4. Architecture

```
                   ┌──────────────────────────────────────────────┐
client ──► gateway │ vLLM generation server                       │
  ▲       (policy, │  forward pass ──► layer-L tap ──► probe ──►  │── score
  │        routing)│  (continuous batching, CUDA graphs intact)   │
  │                └──────────────────────────────────────────────┘
  │                                   │ score > threshold
  │                                   ▼
  │                       escalation queue (bounded)
  │                                   │
  │                                   ▼
  │                       judge service (LLM classifier;
  │                        slow, expensive, can fail)
  │                                   │
  └──────── allow / block / degrade ◄─┘

control plane (stretch): probe registry (versions, thresholds) · rollout controller
               (canary + hulder check) · Prometheus/Grafana · alerts
```

### Package layering

```
 producers of arrays                   core library (numpy, scipy; no GPU)

 hulder[vllm] capture ──┐
  (§5)                  ├─activations─► apply_probe(probe, activations)
 bundled sample data ───┘                          │ scores
                                                   ▼
 bring-your-own scores ───────scores──► check(reference, candidate, labels,
  (JSONL, v0.2)                               threshold, target_fpr) ──► Report
                                                   │
                                                   ▼
                              CLI: parse args · call check · print Report · exit code
```

### Components

| Component | Responsibility | Depth |
|---|---|---|
| Gate library (`check`, `apply_probe`) | Paired shift test, performance CIs, recalibration, verdict, returned as a `Report`. Numpy only | **Deep (stats)** |
| CLI | Parse arguments, load inputs, call the library, print the report, set the exit code. No statistics | Thin |
| Snapshot capture (`hulder[vllm]`) | Extract activations from vLLM for a calibration set and save them with a run manifest | Medium |
| Score loader | Read bring-your-own scores (JSONL of id, score and label per serving config) and align the two configs by id (v0.2) | Thin |
| Probe training (`hulder[train]`) | Train a logistic regression probe on captured activations, export it | Simple on purpose |
| In-server scoring | Compute probe score inside vLLM without breaking CUDA graphs or syncing the hot path; return per-request scores | **Deep (systems)** |
| Gateway + policy | Per-route thresholds, fail-open/fail-closed, escalation decisions | Medium |
| Escalation queue + judge | Bounded queue, admission control, degraded modes, circuit breaker on the judge | **Deep (robustness)** |
| Probe registry | Versioned probes and thresholds, hot swap, shadow scoring | Stretch |
| Rollout controller | Canary a config change, run `hulder check` on the canary, promote or roll back | Stretch |
| Observability | Metrics above, dashboards, drift alarms | Medium |

### Public API (v0.1)

v0.1's public API is `hulder check` and `hulder quickstart`, with their arguments, exit codes and `--json` layout, and the names in `hulder.__all__`: `Estimate`, `PairingError`, `Probe`, `Report`, `Snapshot`, `Verdict`, `apply_probe`, `check`, `check_snapshots` and `prompt_ids_sha256`. A test keeps `__all__` equal to this list.

- `hulder.check(reference, candidate, labels, threshold, target_fpr, *, noise_floor=None, ...)` takes plain arrays. `reference` and `candidate` hold the same examples in the same order, as 1-D scores, or as 2-D activations with `probe=`. Everything after `target_fpr` is keyword-only, so a rerun can't bind to the wrong positional argument.
- `hulder.check_snapshots(reference, candidate, rerun=None, *, threshold, target_fpr, probe=None, seed=0, changes_under_test=())` is `check` on `Snapshot`s. It takes labels from the reference, raises `PairingError` if another snapshot's labels differ, and applies the metadata rules of §6. The CLI calls it.
- `hulder.apply_probe(probe, activations)` returns the scores `check` computes from activations.
- `Report` is a plain data object that neither prints nor exits. It carries the verdict with the threshold and target FPR it was judged at, FPR and FNR with Clopper-Pearson intervals, AUROC and ΔAUROC with paired bootstrap intervals, decision flips, mean |Δscore| (and mean |Δlogit| when Hulder applies the probe) and the noise-floor comparison.

`hulder.capture`, in the `vllm` extra, is public but experimental in v0.1, so its names (`capture`, `CaptureConfig`, `Capture`, `snapshot_metadata`, `base_model_for`) may change in any 0.x release. Importing it needs no vLLM. Everything else is internal, including `hulder.cli`'s functions and the submodules of `hulder.capture`.

| Install | Pulls in | Gives |
|---|---|---|
| `pip install hulder` | numpy, scipy | `check`, `check_snapshots`, `apply_probe`, the CLI, the quickstart on bundled data |
| `pip install "hulder[vllm]"` | vLLM, pinned | `hulder.capture`, the extraction path of §5 |
| `pip install "hulder[train]"` | scikit-learn | probe training |

The library is the core and the CLI only parses, calls and reports. Other tools (CI scripts, notebooks, a rollout controller) then call the same code instead of parsing text, and every statistical test targets the library. Nothing in the base install imports vLLM or torch, which a test enforces. vLLM as a core dependency was rejected because its torch and CUDA wheels run to several GB, which would break the CPU quickstart and exclude users who bring their own scores.

### Exit codes

`hulder check` exits 0 for PASS, 10 for THRESHOLD STALE, 11 for SCORER DEGRADED and 12 for INCONCLUSIVE. Exit 1 is a tool failure and 2 a usage error. Verdicts start at 10 because argparse exits 2 on a usage error and an uncaught exception exits 1, so a verdict on 1 or 2 could be a crash or a mistyped command in disguise. With 2 meaning THRESHOLD STALE, a typo in a CI job could trigger a recalibration on a run that never happened.

- A command-line value out of range, such as a `--target-fpr` outside (0, 0.5], a non-finite `--threshold` or a negative `--seed`, is a usage error. The same value read from a probe file is a tool failure, because the file is at fault.
- `hulder quickstart` exits 1 when a bundled check gives an unexpected verdict, since fixed data giving another verdict means a broken install.
- With `--json`, stdout always holds one JSON object. A verdict gives the whole `Report` with `schema_version` and `hulder_version`. A failure gives `schema_version`, `hulder_version`, `error` and `exit_code`, with the message on stderr as well. `schema_version` changes only when a key is renamed or removed.

## 5. Key design decisions

**D1. Where the probe runs in the server.** Settled for the runtime plugin (v0.3). This is separate from the offline extraction path the gate uses in v0.1 (below), where CUDA graphs don't matter. Candidate options, to be measured rather than argued:
- *A. Fused into the forward pass.* Register an out-of-tree model class via a vLLM plugin that computes `w·h + b` at layer L into a persistent, graph-capturable buffer. Lowest overhead; tight coupling to engine internals; the hard part is mapping batch rows back to requests.
- *B. vLLM's native hidden-state extraction* (added in v0.18 via the KV-connector path, built for Eagle-3 draft training), with a consumer that scores. Uses supported plumbing; may be built around disk/offline use and add latency. It requires chunked prefill off, unlike typical production servers, which run with it on (acceptable offline; for in-server use it is a cost to measure).
- *C. Async sidecar.* Ship hidden states over shared memory to a separate process. Failure isolation and hot swap; copy bandwidth cost; cannot block before tokens stream.

Deliverable is a measured comparison (throughput, p50/p99 latency, GPU memory) against a no-probe baseline and against `vllm-lens` eager mode.

**D2. Which token position.** Start with the last prompt token at end of prefill (one score per request, enables blocking before generation). Per-token streaming is a stretch goal. The default position is the last token of the full chat-templated prompt, which the engine is guaranteed to compute. Published probe work often reads the end of the user turn instead, before the generation prompt (e.g. arXiv 2602.14161 uses Llama-3's `<|eot_id|>` at position −5), so the two positions are compared as an ablation on the benchmarks page on the same probe task. Prefix caching on/off is part of the serving config and is tested in the v0.1 extraction prototype (below).

**D3. How scores leave the server.** Extra response field, response header, or side-channel endpoint. Must stay OpenAI-API compatible for clients that ignore it.

**D4. Failure policy.** When the probe errors or times out, per route: `fail_closed` (escalate to judge) or `fail_open` (allow and log). When the judge is overloaded or down: block, allow-and-log, or hold with timeout. Every path is explicit, tested and observable. *Deferred: settled before the example deployment (v1.0).*

**D5. Custom kernels.** No custom CUDA/Triton kernel unless D1 measurements show the plain implementation costs meaningful throughput. A probe is one small matrix-vector product; the expectation is that a kernel is unnecessary. If measurement says otherwise, the kernel is justified work and is recorded here with its measurement.

**D6. Gateway language.** Default Python (FastAPI, asyncio). Only change if measurement shows the gateway is the bottleneck. Language chosen per layer on evidence, and said so. *Deferred: settled before the example deployment (v1.0).*

**Offline extraction (v0.1).** The gate reads hidden states through vLLM's KV-connector hidden-state extraction (`speculative_config.method = "extract_hidden_states"` with `ExampleHiddenStatesConnector`), pinned at vLLM 0.30.0, with prefix caching off and a fixed batch composition (prompt order and batch size, recorded in the snapshot). The evidence is in `docs/results/phase1/extraction_prototype.md`.
- It extracts any layer and returns every prompt position, so both D2 positions come from one pass.
- Batch composition moves the last-token hidden state by up to 0.0625 absolute (relative L2 about 1e-3), so it is fixed.
- With prefix caching on, full 16-token blocks come back from stored hidden states and only the tail is recomputed, so values depend on which requests ran before, which a snapshot can't record. Caching on also changes values at position 0 before any cache hit, which is unexplained.
- One batch of 6 prompts was bit-identical across fresh engines, but at scale captures are not (§10). A snapshot reproduces to within a noise floor, which `check` measures from a rerun.
- Hidden states are stored as float16. A bf16 engine's values pass through float32, and convert exactly wherever they lie in float16's normal range (6.1e-5 to 65504 in magnitude). Smaller values round to float16's subnormals or to zero, negligible next to a hidden state's typical size, and a value beyond 65504 stops the capture with an error naming its prompt, layer and position rather than storing infinity. The probe reads positions −1 and −5, where the massive activations at position 0 never appear (`docs/results/phase1/massive_activations.md`).

The connector needs chunked prefill off, unlike typical production servers, and writes one file per request. Neither matters offline, and for in-server scoring (D1) both are costs to measure. HF Transformers hooks were rejected because engine mismatch is itself a source of shift, and `vllm-lens` is not yet measured at 0.30.0. Revisit if the connector breaks or changes its output on a vLLM upgrade.

## 6. Statistical method for `hulder check`

- **Paired design.** Both configs see the same prompts, so compare per-prompt score differences (paired bootstrap or Wilcoxon signed-rank), which has far more power than a two-sample test on unpaired score distributions.
- **Noise floor first.** Run the baseline config twice (and across batch sizes) to measure run-to-run variation from numerical non-determinism. A shift only counts if it exceeds the floor. This also positions the work relative to arXiv 2609.31796.
- **For a linear probe, the projection is sufficient.** The probe's output depends only on `w·h + b`, so the 1D projection onto `w` fully determines probe behaviour. A multivariate test (e.g. MMD on the layer-L activations) is reported as a general-shift diagnostic, not as the verdict.
- **Performance with intervals.** AUROC with bootstrap CI; FPR and FNR at threshold with exact binomial (Clopper-Pearson) or Wilson intervals.
- **Recalibration.** Choose the new threshold from benign calibration scores with a finite-sample guarantee. Decide between a conformal-style quantile (controls expected FPR) and a binomial-tail bound (controls FPR with high probability). The choice is v0.2 work (§11).
- **Decision flip rate (diagnostic).** The fraction of examples whose probe decision (score above or below the threshold) differs between reference and candidate on the same prompts. Reported alongside the verdict, never an input to it. arXiv 2609.31796 recommends it as the main per-example measure, because aggregate accuracy understated the number of changed decisions by a factor of two to nine.
- **Multiple configs.** Holm correction when checking several candidates at once.
- **Validation.** Every statistical routine is tested on synthetic data with known ground truth (injected shifts of known size, null cases) before real data. False-alarm and detection rates of the gate itself are reported.

### The verdict rule

`hulder check` tests whether a serving change moved the scorer's decisions on the same prompts. It doesn't test whether the threshold generalises to new data, so the threshold is assumed to be set on the reference snapshot itself. A "bound" below is a 95% confidence bound: Clopper-Pearson for FPR and FNR, and a stratified paired bootstrap (1,000 resamples) for ΔAUROC = candidate − reference. The verdicts are checked in this order.

1. **SCORER DEGRADED** if the ΔAUROC upper bound is below −0.02.
2. **THRESHOLD STALE** if the candidate FPR lower bound is above twice the target, or its upper bound is below half the target, or the candidate FNR lower bound is above the reference FNR + 0.05.
3. **PASS** if the candidate FPR upper bound is at most twice the target, the candidate FNR upper bound at most the reference FNR + 0.05, and the ΔAUROC lower bound at least −0.02.
4. Otherwise **INCONCLUSIVE**, with the number of extra benign examples that would settle the FPR condition if the observed rate held.

The tolerances are fixed in v0.1, as constants in `hulder/gate/check.py` (`FPR_BAND_HIGH`, `FPR_BAND_LOW`, `FNR_TOLERANCE`, `AUROC_TOLERANCE`). Shift, the noise floor and the decision flip rate are diagnostics, never inputs to the verdict.

- **One-sided bands.** At about 1,150 benign and a 1% target, an unchanged scorer's FPR interval is about [0.5%, 1.7%]. A symmetric band such as [target / 1.5, target × 1.5] would almost never PASS, because the interval alone is wider than the band. So each condition asks only whether a bound has crossed its tolerance in the harmful direction.
- **Ties.** An example is flagged when its score is strictly above the threshold, here and in the decision flip rate. The threshold is fitted as the benign scores' 1 − target quantile (`method="higher"`), so with heavily tied scores no benign example may sit above it, and an unchanged scorer comes out THRESHOLD STALE with the calibration warning below.
- **The noise floor.** The candidate's shift exceeds the floor when the lower bound of a 95% paired bootstrap interval of mean |candidate − reference| − |rerun − reference| is above 0. Shift is measured in score space, which works for any scorer, and when Hulder applies the probe it also reports mean |Δlogit|. Without a rerun the floor is "not measured". A rerun or a candidate bit-identical to the reference warns, since a zero floor, or the same file passed twice, would otherwise mislead.
- **The rerun must be of the reference.** A rerun of the candidate would count the whole shift as noise. A reference rerun can understate a noisier candidate's own noise (§10).
- **The threshold must come from the reference snapshot.** If the reference's own FPR interval excludes the target, `check` warns that the threshold wasn't calibrated on this snapshot. A threshold fitted elsewhere fails to PASS an unchanged scorer far more often, 21.9% of the time at 1,191 benign even at the true quantile, because the FPR condition alone then allows at most 14 false positives. Bringing that below 10% needs about 1,600 benign examples.
- **The tolerance edge.** When an INCONCLUSIVE report would need more than 10 times its own benign count to settle, the CLI adds that the estimate sits at the tolerance edge and the number is unstable. This is presentation only.

The gate's own calibration was measured on synthetic scores over 1,000 seeds at 1,191 benign and a 1% target, against bounds fixed before running (`docs/results/phase1/gate_calibration.md`). The null and fresh-engine noise PASS in every run. A fivefold FPR rise is THRESHOLD STALE in 99.9% of runs, a threefold one never PASSes, and between 1.5% and 2% the gate is mostly INCONCLUSIVE. A drop in AUROC of 0.05 or more is SCORER DEGRADED in every run. At exactly −0.02 the missed-attack case PASSes 6.7% of the time, more than a 95% bound implies, because each seed fits its own threshold and so has its own true ΔAUROC. Scored against that, the interval's coverage is nominal (`docs/results/phase1/delta_auroc_coverage.json`).

### Metadata rules

`check` compares the runs' metadata so that a verdict about one serving change doesn't quietly mix in another, such as a different system prompt or probe layer. The rules live in `hulder/gate/check.py` and `hulder/capture/base_model.py`, and tests cover each one.

1. **Changes under test are listed, never warned about.** `model`, `revision`, `dtype`, `quantization`, `kv_cache_dtype`, `vllm_version` and `batch_size` may differ, as may any key declared with `changes_under_test=` (`--under-test KEY`). Each difference is reported as (key, reference value, candidate value).
2. **Every other difference warns**, including keys `check` doesn't know and keys on one side only. The verdict is unchanged. `gpu` is deliberately not under test, so runs on two GPU models warn unless the check declares `--under-test gpu`. Nor is `enable_thinking`, recorded for hybrid-thinking models such as Qwen3, since the chat template and so the scored token depend on it.
3. **`base_model` must match.** The AWQ checkpoint of a model and the model itself share one `base_model` while their `model` and `revision` differ. A run without it warns. It can't be declared under test (`ValueError`, or exit 2 from the CLI), because that would hide a model swap.
4. **How capture sets `base_model`.** An unquantized capture uses the model itself. A quantized one uses an explicit override, or else the model card's `base_model` at the pinned revision, and with neither it warns rather than guessing. An Instruct model's own card names its pretrained parent, which is why an unquantized capture doesn't copy the field.
5. **`prompt_ids_sha256` is pairing, not metadata.** Known hashes that differ raise `PairingError`.
6. **Metadata on one side only** warns that it wasn't compared. With metadata on neither side, nothing is compared.
7. **A rerun must match the reference on every key**, changes under test included, or `check_snapshots` warns.
8. **The probe must match every snapshot it scores** on `layer`, `position` and `base_model`, or `check_snapshots` raises `PairingError`, because a probe of the right width applied to another layer scores nonsense without failing. A probe's `base_model` defaults to its `model`. Other keys both record, such as a system prompt, warn when they differ. The probe's own keys (threshold, training data) are never compared.
9. **A probe file lacking a field.** A missing `layer`, `position` or `base_model` warns. A missing `weights` or `bias` array is a damaged file and exits 1. `threshold` and `target_fpr` are optional in the file, a flag overrides them, and with neither file nor flag it is a usage error. A recorded one that isn't a number is a damaged file, exit 1, even when a flag would override it.

Warning on every undeclared difference fails loud, where an allow-list of keys that must match would let a new key, such as a chat template added later, pass silently. Raising instead would block every comparison between snapshots from older and newer capture code. So per-run provenance (commit, timestamps) stays in the capture manifest, not in snapshot metadata.

## 7. The experiment

- **Models.** A small (~1 to 2B) open model for v0.1 and fast iteration, and Qwen3-4B or Qwen3-8B in bf16 for the benchmarks page, chosen once the L4 smoke capture reports the 8B model's KV-cache headroom (§8.1, §12). Qwen3-4B is 8.04 GB of bf16 weights at revision `1cfa9a72`, with official AWQ int4 (`Qwen/Qwen3-4B-AWQ` at `74d4bd2b`) and block-quantized FP8 (`Qwen/Qwen3-4B-FP8` at `96b30dc1`) checkpoints. Qwen3-8B is 16.38 GB at `b968826d`, which fits the 24 GB L4 but not a 12 GB card, with official AWQ (`Qwen/Qwen3-8B-AWQ` at `4da05a8e`, 6.10 GB) and FP8 (`Qwen/Qwen3-8B-FP8` at `220b46e3`, 9.44 GB) checkpoints. Each quantized card names its base model in `base_model`, the AWQ cards as a string and the FP8 cards as a one-element list, which `hulder.capture.base_model_for` doesn't read, so FP8 captures pass the base model explicitly. All six cards declare Apache-2.0. Sizes are the safetensors totals from the Hugging Face API at those revisions (§14). Its hybrid-thinking chat template is captured with `enable_thinking` pinned in `CaptureConfig`. Each must have an official AWQ checkpoint of the same base model from the same publisher, otherwise the AWQ comparison measures a different model rather than a serving config. Self-quantizing is the fallback. Each must also be fp16-stable, which rules out Gemma-3 (arXiv 2609.31796 §3.4 reports its fp16 residual activations overflow to NaN).
- **Task.** Prompt attack (injection or jailbreak) detection, named so because the chosen datasets' attack classes mix both. Benign examples include hard negatives (benign text that looks like an instruction) so a 1% FPR target means something. S-Labs for training, calibration and in-distribution test, 3nesdeniz boundary-pairs as a held-out hard-negative test set that also supplies the quickstart's sample activations (datasets below).
- **Critical detail.** Train the probe on activations extracted from the serving engine itself (vLLM, bf16), not from HF Transformers. Engine mismatch is itself a source of shift, and training on HF would contaminate the baseline.
- **Configs** (each vs the bf16 reference): FP8 weights, AWQ int4, FP8 KV cache, batch size 1 vs large, two vLLM versions, and bf16 rerun (noise floor). Add more only if cheap.
- **vLLM versions.** The pin is `vllm==0.30.0`. The second version for the engine-version config is chosen by rule, not by hand: the most recent stable minor release before the pin, at its latest patch, that has the extraction path of §5 and supports both experiment models. Against the current pin that is v0.29.x, the one-step upgrade a team realistically faces. v0.18.x, the first release with the connector, is an optional large-jump config, labelled as confounded wherever it appears, because twelve minor releases sit between it and the pin.
- **Layers.** Report for 2 to 3 layers; the effect of quantization may depend on depth.
- **Output.** A table of AUROC, FPR/FNR at the original threshold, recalibrated threshold, and gate verdict per config, plus plots of score distributions.
- **Quickstart demo probe (demo only).** The main S-Labs probe failed the quickstart rule on boundary-pairs (AUROC 0.839 < 0.85, datasets below), so the quickstart ships a separate demo probe trained on 600 boundary-pairs examples with a 5% FPR target (`docs/results/phase1/probe_demo.md`). It shows how the gate works and is not a prompt-attack monitor. If its fp16 vs AWQ check comes out INCONCLUSIVE at this sample size, the quickstart shows that verdict as is.
- **The gate's own calibration.** Its false-alarm rate on unchanged scorers and its THRESHOLD STALE detection rate against the candidate's true FPR, at the benchmark's calibration-set size. The v0.1 synthetic curve at 1,191 benign is in `docs/results/phase1/gate_calibration.md` (§6).
- **Finding to report: a probe that separates on surface features fails on hard negatives.** The S-Labs probe reaches test AUROC 0.994, and even layer 4 separates S-Labs validation at 0.996, which points to surface features of the text. At its 1% FPR threshold it flags 95% of benign boundary-pairs prompts, whose median score is 0.986 against 0.011 for S-Labs calibration benign. The benchmarks page reports this finding alongside the serving-config results, because it bounds what any in-distribution AUROC says about deployment (`docs/results/phase1/probe_main.md`).

### Datasets

| Role | Dataset | Revision |
|---|---|---|
| Training, calibration, in-distribution test | `S-Labs/prompt-injection-dataset`, MIT | `002a9dd1` |
| Held-out hard-negative test set, never trained on | `3nesdeniz/agentic-prompt-injection-boundary-pairs`, CC-BY-4.0 | `a5682e75` |
| Quickstart sample activations and the demo probe | boundary-pairs only | same |

S-Labs is the only candidate with natural-looking hard negatives at enough size (8,358 benign and 6,774 attack after deduplication), but its card gives no provenance, so its MIT licence rests on the uploader's word. Boundary-pairs pairs each attack with a legitimate request from the same scenario, which tests whether a 1% FPR survives instruction-like benign text, but at 600 per class it is too small to train on alone. Shipping only boundary-pairs data keeps the package independent of S-Labs' provenance, and everything shipped from it carries the CC-BY-4.0 attribution (Deniz 2026, doi:10.5281/zenodo.21378693). "Never trained on" covers everything that fits the main probe or picks its settings, from weights and layer to regularisation and threshold.

The quickstart rule was fixed in advance. If the S-Labs probe's AUROC on all 1,200 boundary-pairs examples fell below 0.85, the quickstart would get its own demo probe trained on boundary-pairs. It scored 0.839 [0.819, 0.860]. So the demo probe reuses the main probe's layer, position and C, splits boundary-pairs by scenario (train 600, calibration 360, and the dataset's own test split of 240) so each attack stays with its paired request, and targets 5% FPR, because 180 benign calibration examples can't support 1%.

Rejected candidates were `neuralchemy/Prompt-injection-dataset`, because 933 of its 2,538 benign examples follow one template and its labels mix tasks; `deepset/prompt-injections` and `NotInject` as too small; `xTRam1/safe-guard-prompt-injection`, `jayavibhav/prompt-injection-safety` and `rogue-security/prompt-injections-benchmark` for their licences; and SPML, whose authors advise against training detectors on it.

**Semantic cache (v0.2).** PAWS-Wiki (`google-research-datasets/paws`, config `labeled_final`, revision `161ece95`) is downloaded at runtime and never redistributed. Its licence allows any use, with acknowledgement of Google LLC as the source, and the example cites Zhang, Baldridge and He, NAACL 2019. Its negatives differ from their pair only in word order, close to the worst case for an embedding model, so absolute FPRs from it say nothing about a production cache, and its sentences are Wikipedia statements rather than user questions. Quora Question Pairs and `redis/langcache-sentencepairs-v1` were rejected for their licences (the latter bundles CC-BY-NC-SA SICK), Stack Exchange duplicates because only positives are marked, and AskUbuntu duplicates because no licence for the annotations was found.

## 8. Tech stack

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.12 | `uv` for envs, `ruff`, type hints, `pytest` |
| Core library | numpy, scipy | The base install. No black-box stats libraries for the gate's core logic, and no torch or vLLM import (§4) |
| Serving | vLLM, pinned version, in the `hulder[vllm]` extra | Upgrade deliberately; version change is one of the experiment configs |
| ML | scikit-learn in the `hulder[train]` extra | Probe training is ~50 lines by design |
| Embedding model (v0.2 example) | A small sentence embedding model, e.g. `BAAI/bge-small-en-v1.5` (MIT, 33M parameters), at fp32 and int8 on CPU | Model and int8 path chosen in v0.2 (§12). In an examples extra, not the base install |
| Gateway | FastAPI + asyncio | See D6 |
| Judge | Llama Guard (current version) on a second vLLM server; plus a mock judge that can be made slow or broken on demand | Mock is essential for fault injection |
| Observability | Prometheus, Grafana, OpenTelemetry traces | |
| Load / chaos | Locust or k6; scripted fault injection | |
| Packaging | Docker Compose locally | GKE deployment is an optional stretch that doubles as Google Cloud PMLE cert practice |
| CI | GitHub Actions for CPU tests; GPU tests run manually with recorded results | |
| Compute | Laptop for development; an NVIDIA L4 on Modal for experiments | See §8.1 |

### 8.1 Development environment

Two tiers, because the local GPU cannot run the headline experiment.

**Local (development).** Windows laptop with an RTX 2070 (Turing, compute capability 7.5, 8 GB VRAM), 32 GB RAM. vLLM runs on Turing, but Turing has no bf16 support and no native FP8 (FP8 there is weight-only W8A16 via Marlin kernels). So the laptop cannot produce the bf16 reference or the native FP8 configs, and its throughput numbers are not representative.

Used for:
- all `hulder` statistics, unit tests and synthetic validation (CPU)
- gateway, queue, mock judge and the Prometheus/Grafana harness
- runtime plumbing prototypes on a ~1B model in fp16 with conservative `--gpu-memory-utilization` and `--max-model-len`
- the v0.1 demo: fp16 vs AWQ int4 on a ~1B model (vLLM's AWQ path needs compute capability 7.5, `auto_awq.py:230` at v0.30.0)

Workspace rules:
- A dedicated WSL2 distro (`hulder`), created fresh from an official Ubuntu image, **not cloned** from an existing distro. Teardown is `wsl --unregister hulder`.
- Code lives on the distro's Linux filesystem (`~/hulder`), never under `/mnt/c`. Edit via VS Code's WSL extension.
- GPU access comes from the Windows NVIDIA driver only. No Linux NVIDIA driver and no CUDA toolkit inside WSL; PyTorch and vLLM wheels bundle the CUDA runtime.
- Python via `uv` with a project `.venv` and committed `uv.lock`. The interpreter is uv-managed CPython 3.12.14 (pinned in `.python-version`), installed inside the repo under `.cache/uv-python`, and `pyproject.toml` forbids implicit downloads. So `source scripts/dev-env.sh` before any `uv` command, which also keeps `HF_HOME` and the compile caches inside the repo. Without it, `uv run` fails with "No interpreter found for Python 3.12.14 in managed installations", and following uv's hint without the script would install Python outside the repo. First-time setup:

  ```bash
  source scripts/dev-env.sh && uv python install 3.12.14 && uv sync
  ```
- Tests. `uv run pytest` is the fast run, CPU only. Two groups are opt-in. `--runslow` adds the gate's seeded calibration runs (about 9 minutes), and `--runvllm` adds the tests that import vLLM, which skip cleanly where the `[vllm]` extra isn't installed, as in GitHub CI. `uv run pytest --runvllm -m vllm` runs that group alone.

  ```bash
  source scripts/dev-env.sh && uv run pytest --runvllm --runslow
  ```
- `HF_HOME` points inside the project (`.cache/hf`, gitignored) so model weights are deleted with the workspace.
- Docker Engine and the NVIDIA Container Toolkit installed inside the distro, not Docker Desktop. Compose volumes as bind mounts under the project.
- `.wslconfig` caps WSL memory and CPUs so Windows stays usable.

**Remote (experiments and benchmarks).** An NVIDIA L4 on Modal: Ada (sm_89), 24 GB, native bf16 and FP8. It is a datacenter card that teams serve on, so its throughput and overhead numbers represent L4 deployments, and it holds an 8B model in bf16. vLLM 0.30.0's docs list FP8 W8A8 on Ada, Hopper and Blackwell (compute capability 8.9 and up), so Ada is supported but not documented as better supported than the others. Which FP8 kernel vLLM selects for Qwen3's 128×128 block-quantized weights on the L4 is unverified† (§12). Every comparison runs on one GPU model, since a different GPU is itself a serving change. Snapshots record `gpu`, and a difference warns (§6, metadata rule 2). Two Modal containers may land on different physical L4s, so each capture's manifest records the GPU UUID and driver.

Captures run through `experiments/modal/capture.py`: the image is this repository's `uv.lock` with the `vllm` extra, model weights and compile caches live on a Modal Volume, each serving config runs in its own container with a timeout, and snapshots, logs and a run manifest with an estimated cost come back to the repository. Remote runs use Modal's monthly free credit under a hard usage cap. A teaching-lab RTX 4070 (Ada, 12 GB) is the fallback, never mixed with L4 runs in one comparison. Without either, the overhead table and the benchmarks page shrink to what the 2070 can show, which is fp16 and AWQ configs, the noise floor and engine versions, with no bf16 or FP8 results and no throughput claims.

## 9. Repository layout (target)

```
hulder/
  README.md            user guide: install, quickstart, one example
  CONTEXT.md           domain glossary, kept current
  docs/
    design.md          this file
    reading-a-report.md  the quickstart's report, line by line
    tutorial-live.md   the demo captured with vLLM, and your own snapshots
    results/           experiment tables, plots, run manifests
  src/hulder/
    gate/              check, apply_probe, Report, recalibrate, stats (base install, numpy only)
    inputs/            loaders: bundled sample data, bring-your-own scores (JSONL)
    cli.py             thin wrapper: parse, call gate, print, exit code
    data/              bundled quickstart sample activations, example probe
    capture/           vLLM snapshot capture (hulder[vllm] extra)
    train/             probe training + export (hulder[train] extra)
    runtime/           vLLM plugin / in-server scoring
    gateway/           policy, routing, escalation queue, judge client
    control/           probe registry, rollout controller (stretch)
  harness/             docker-compose, load tests, fault injection
  tests/               unit (CPU), stats validation (synthetic), gpu/ (manual)
```

## 10. Known limits

**Tested on one GPU and one model.** v0.1 has run only on an RTX 2070 (Turing) with Qwen2.5-1.5B-Instruct under vLLM 0.30.0, fp16 against the official AWQ int4 checkpoint. No bf16, FP8, FP8 KV-cache or engine-upgrade results exist yet.

**Captures are not bit-identical across engine runs.** Three fresh-engine fp16 captures of 16,332 prompts at an identical config differed on 53 to 56% of prompts (mean |Δscore| about 8e-5, up to 2 decision flips; `docs/results/phase1/extraction_reruns.md`). Batch composition also moves activations slightly. The cause is unknown. So pass a rerun of the reference as `--noise-floor`, and expect the floor itself to vary: three runs of the live tutorial gave 0.000205 to 0.000286, up to about 40% apart (`docs/results/phase1/live_demo.md`). The verdict was the same in all three.

**The rerun must be of the reference.** A noisier candidate's own noise is not measured. AWQ's run-to-run noise is about 4 times fp16's, so an fp16 rerun can understate it (§6). The AWQ shift in the demo is still 180 to 250 times the floor.

**Prefix caching changes activations even without cache hits.** The cause is unexplained (`docs/results/phase1/extraction_prototype.md`). `hulder.capture` requires it off (§5).

**Fixed tolerance bands.** The candidate FPR must stay between 0.5 and 2 times the target, FNR within 0.05 of the reference, and ΔAUROC above -0.02. These are constants in `hulder/gate/check.py`, not options.

**No recalibration.** THRESHOLD STALE says the threshold is stale but doesn't propose a new one.

**Small FPR rises need large calibration sets.** With about 1,200 benign examples, a rise from 1% to 1.5% or 2% is usually INCONCLUSIVE (§2). The report says how many more examples would settle it.

**Swapped roles go undetected.** Passing the candidate as reference turns a degradation into PASS. Check the "Changes under test" lines, which show a swap as `quantization: 'awq' -> None`.

**API rough edges.** `Snapshot == Snapshot` raises `ValueError`, and `Report` isn't hashable. There is no `hulder snapshot` command, and capture is the experimental `hulder.capture` Python API.

## 11. Roadmap

v0.1 is the gate on activation probes: `hulder check` and `hulder quickstart` on CPU, `hulder.capture` for vLLM, and the live tutorial on a local GPU. Next, in priority order:

- **v0.2, gate complete.** A recalibrated threshold for THRESHOLD STALE, choosing between a conformal quantile and a binomial-tail bound on measured coverage. Bring-your-own scores as JSONL of id, score and label per serving config. A GitHub Action wrapping the CLI. A second worked example, semantic-cache similarity at fp32 vs int8 on CPU (§7).
- **After v0.1, smaller items.** A `hulder snapshot` command around `hulder.capture`. A rerun of the candidate as the noise floor (`--rerun-of candidate`). Configurable tolerance bands, recorded in the report. The multivariate shift diagnostic of §6 (MMD on activations). Equality and hashing for `Snapshot` and `Report`.
- **v0.3, runtime plugin.** In-server probe scoring with CUDA graphs on (D1, D3), metrics and drift alarms, with overhead measured against a no-probe baseline and `vllm-lens`.
- **Benchmarks page.** Qwen3-4B or Qwen3-8B in bf16 against FP8, AWQ and FP8 KV cache on an L4, with the noise floor, 2 to 3 layers and the token-position ablation (§5 D2, §7).
- **v1.0, example deployment.** The probe-first cascade with failure policies (D4, D6), a docs site and a write-up.

## 12. Assumptions to verify before building on them

- ~~Contents of arXiv 2609.31796 ("Same Probe, Different Numbers").~~ Resolved: read in full. It keeps weights fixed and doesn't measure thresholded FPR/FNR or recalibration, so the plan is unchanged.
- Prior-art claims in §1 (marked †): the Anthropic and GDM cascade details and the GDM "<10% LLM usage" figure; `vllm-lens` forcing eager mode; IBM `vLLM-Hook` being research-oriented; `AIsafePy` probes being HF-Transformers-only. Read the actual sources and correct §1 where they differ.
- The current state of vLLM hidden-state extraction (v0.18+), whether it works at serving time for arbitrary layers, and its latency cost.
- Whether an out-of-tree model class can compute and export per-request probe scores with CUDA graphs enabled.
- FP8 weights, FP8 KV cache and AWQ under the pinned vLLM on the L4. Checked by a smoke capture on the L4. The local Turing GPU is ruled out for FP8 and bf16 configs (§8.1).
- The FP8 kernel vLLM 0.30.0 selects for Qwen3-4B-FP8 on the L4 (§8.1), from its `Selected <kernel> for <module>` startup line in that smoke capture. Its docs try DeepGEMM and CUTLASS before Marlin and Triton for block-quantized FP8, and the first two may need Hopper or newer, so Ada may run a less exercised kernel than datacenter Hopper.
- The KV-cache headroom on the L4 for Qwen3-4B and Qwen3-8B in bf16. For 8B it is an estimate of about 5 GiB: 0.9 of the L4's memory less 15.26 GiB of weights, before activations and CUDA graphs. Read the KV-cache size vLLM logs in that smoke capture.
- The noise floor between two Modal containers, which may run on different physical L4s, against the single-machine floor of §10.
- Current `vllm-lens` behaviour (still eager-only?) and whether anyone has shipped production-speed probes for vLLM since September 2026.
- Dataset licences and label quality.
- Semantic-cache example (v0.2): that int8 quantization of the chosen embedding model runs on CPU and actually changes the similarities. The dataset is settled (§7).

## 13. Success criteria

**Usability**
- A quickstart that goes from `pip install` to a first verdict in under a minute without a GPU, using shipped pre-extracted sample activations. The live vLLM run is a second tutorial, run from a clone.
- Pretrained example probes for 1 to 2 open models, each shipped with its reference snapshot and training config.
- An end-to-end demo: a serving-config change caught by the GitHub Action before rollout.
- Versioned releases on PyPI, tests and CI on every push, and a short docs site.

**Technical**
- Benchmarks page across at least five serving configs with noise floor and confidence intervals, reproducible from a documented command per result. It reports every config run, including PASS, INCONCLUSIVE, and cases where the gate is wrong.
- Gate with measured false-alarm and detection rates on synthetic shifts.
- Runtime plugin with measured overhead against a no-probe baseline and against eager-mode hooks, including honest cases where it loses.
- Example deployment that stays responsive under judge failure and overload, with the behaviour documented and tested.
- A design doc covering every decision in §5, with the alternatives it rejected.
- A public write-up a vLLM user could act on.

---

## 14. References

- Building Production-Ready Probes For Gemini, arXiv 2601.11516 · https://arxiv.org/abs/2601.11516
- AI Safety at the Frontier, paper highlights January 2026 (Anthropic and GDM probe cascades) · https://aisafetyfrontier.substack.com/p/paper-highlights-of-january-2026
- Same Probe, Different Numbers, arXiv 2609.31796 · https://arxiv.org/abs/2609.31796
- Latent Undertow: How Ordinary Typos Break Probes, arXiv 2609.15994 · https://arxiv.org/abs/2609.15994
- Stop Early, Spend Less: hidden-state probes for streaming moderation, arXiv 2606.10487 · https://arxiv.org/abs/2606.10487
- vllm-lens · https://pypi.org/project/vllm-lens/
- IBM vLLM-Hook · https://github.com/IBM/vLLM-Hook
- AIsafePy · https://pypi.org/project/aisafepy
- Extracting hidden states from vLLM (vLLM blog, March 2026) · https://vllm-project.github.io/2026/03/30/extract-hidden-states.html
- NeMo Guardrails · https://github.com/NVIDIA-NeMo/Guardrails
- Llama Guard self-hosting with vLLM (NeMo docs) · https://docs.nvidia.com/nemo/guardrails/user_guides/advanced/llama-guard-deployment.html
- vLLM GPU compute capability matrix (Speediyo) · https://www.speediyo.com/ai-infra/vllm-gpu-compute-capability-matrix
- Turing microarchitecture (Wikipedia) · https://en.wikipedia.org/wiki/Turing_(microarchitecture)
- Qwen3 Technical Report, arXiv 2505.09388 · https://arxiv.org/abs/2505.09388
- Qwen3-4B at `1cfa9a7208912126459214e8b04321603b3df60c` (bf16, Apache-2.0) · https://huggingface.co/Qwen/Qwen3-4B/tree/1cfa9a7208912126459214e8b04321603b3df60c
- Qwen3-4B-AWQ at `74d4bd2bd4bff9cafc9345221320bffb08b406a3` (Apache-2.0) · https://huggingface.co/Qwen/Qwen3-4B-AWQ/tree/74d4bd2bd4bff9cafc9345221320bffb08b406a3
- Qwen3-4B-FP8 at `96b30dc13593a244a5e59e84687309f53c375cfa` (e4m3, 128×128 blocks, Apache-2.0) · https://huggingface.co/Qwen/Qwen3-4B-FP8/tree/96b30dc13593a244a5e59e84687309f53c375cfa
- Qwen3-8B at `b968826d9c46dd6066d109eabc6255188de91218` (bf16, Apache-2.0) · https://huggingface.co/Qwen/Qwen3-8B/tree/b968826d9c46dd6066d109eabc6255188de91218
- Qwen3-8B-AWQ at `4da05a8edb55c6046cce958586c33b61da07bb79` (Apache-2.0) · https://huggingface.co/Qwen/Qwen3-8B-AWQ/tree/4da05a8edb55c6046cce958586c33b61da07bb79
- Qwen3-8B-FP8 at `220b46e3b2180893580a4454f21f22d3ebb187d3` (Apache-2.0) · https://huggingface.co/Qwen/Qwen3-8B-FP8/tree/220b46e3b2180893580a4454f21f22d3ebb187d3
- Modal pricing (L4 $0.000222/s) and budgets (checked 2026-10-13) · https://modal.com/pricing · https://modal.com/docs/guide/budgets
- Model sizes and licences: the Hugging Face model API, e.g. https://huggingface.co/api/models/Qwen/Qwen3-4B?blobs=true (checked 2026-10-11; Qwen3-8B AWQ and FP8 2026-10-13)
- vLLM 0.30.0, FP8 W8A8 docs · https://github.com/vllm-project/vllm/blob/v0.30.0/docs/features/quantization/llm_compressor/fp8.md
- NVIDIA L4 and L40S (Ada Lovelace) · https://www.nvidia.com/en-us/data-center/l4/ · https://www.nvidia.com/en-us/data-center/l40s/
