# In-server scoring with `hulder.runtime`

`hulder.runtime` scores a linear probe inside `vllm serve`, on each request's last prompt token at the end of prefill, with CUDA graphs on. It is **experimental**. It runs only on the pinned vLLM 0.30.0, and its names may change in any 0.x release.

It is **monitor-only**. The probe can flag a request but can't stop it, since the reply may already be streaming before the score exists. Scores go into non-streamed replies and onto `/metrics`, and drift alarms are Prometheus rules over those metrics. Blocking a request before tokens stream is not built (design §5 D3).

## Serve

```bash
pip install "hulder[vllm]"
```

`hulder.runtime.serve_args` turns a probe file into the two arguments `vllm serve` needs. The probe file's metadata must record its `layer`, `position` -1 (the last prompt token) and a `threshold`, or it raises `ValueError`.

```python
import shlex
from hulder.runtime import serve_args

print(shlex.join(["vllm", "serve", "Qwen/Qwen2.5-1.5B-Instruct", "--dtype", "float16", *serve_args("probe.npz")]))
```

For the bundled demo probe (layer 20) that prints, with the probe's absolute path:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype float16 \
  --speculative-config '{"method": "extract_hidden_states", "num_speculative_tokens": 1, "num_speculative_tokens_per_batch_size": [[1, 1, 0]], "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": [20]}}}' \
  --kv-transfer-config '{"kv_connector": "ScoringHiddenStatesConnector", "kv_connector_module_path": "hulder.runtime.connector", "kv_role": "kv_producer", "kv_connector_extra_config": {"probe_path": "/abs/path/probe.npz", "use_synchronization_lock": false}}'
```

Add your usual serving flags. Leave async scheduling on, which is vLLM's default here.

**Why the speculative config.** vLLM reads hidden states through its hidden-state extraction path, which is built as a speculative decoding method and drafts one token per decode step by default. That token cost about 38.5% of saturation throughput on an L4. The schedule `[[1, 1, 0]]` sets drafts to 0 at every batch size, which on Model Runner V2 with async scheduling on took that cost to about 1.5% ([D1 confirmation](results/phase3/d1_confirmation_l4.md)). The probe's work itself is one matrix-vector product per request.

## Read the score

A non-streamed chat or completion reply carries the score in `kv_transfer_params`:

```json
{"id": "...", "choices": [...], "kv_transfer_params": {"hulder_score": 0.0412}}
```

The request is flagged when `hulder_score` is strictly above the probe's threshold. A reply can also come back with `"hulder_score": null`. Scores travel back with engine steps, so a reply that finishes first goes without one. That means the reply ended early, **not** that the prompt is benign.
- **Streamed replies** never carry a score.
- **Very short replies** can miss it. In a local check on an RTX 2070, no 1-token reply carried one, and 53% to 96% of 2-token replies went without. Every reply of 3 tokens or more carried one ([D3 spike](results/phase3/d3_spike.md)).

`/metrics` counts every score, including those a reply missed.

## Metrics

| Series | Type | Meaning |
|---|---|---|
| `hulder_probe_score` | histogram, buckets every 0.05 | The probe's score at the end of prefill, for every scored request |
| `hulder_probe_flagged_total` | counter | Scores strictly above the threshold |
| `hulder_probe_replies_with_score_total` | counter | Finished requests whose reply carried a score |
| `hulder_probe_replies_without_score_total` | counter | Finished requests whose reply went without one, streamed replies included |
| `hulder_draft_tokens_total` | counter | Draft tokens scheduled, which should stay at 0 (below) |

They carry vLLM's `model_name` and `engine` labels.

## Drift alarms

[`examples/prometheus/hulder-rules.yml`](../examples/prometheus/hulder-rules.yml) has example alerting rules:
- **Flag rate** over an hour against the gate's bands, twice and half the probe's target FPR.
- **No scores** while requests still succeed, meaning the connector isn't loaded or scoring has stopped.
- **Replies without a score,** for non-streamed traffic.
- **Draft tokens** above 0.

They are examples, not a tested artefact. A test checks that they load as YAML and use only metrics that exist, but none has run against a live Prometheus. **A flag rate is not a false-positive rate.** Live traffic mixes benign prompts and attacks in unknown proportions, so a move outside the band says scores or traffic changed, and `hulder check` on a labelled calibration set says which. There is no drift statistic on the score distribution itself.

## The draft counter

vLLM 0.30.0 can drop the zero-draft schedule without an error, and each decode step then carries a drafted token again.
- **Async scheduling off** (`--no-async-scheduling`): no warning from vLLM.
- **`data_parallel_size` above 1:** one startup warning from vLLM, read from its code and not run.

The connector counts the draft tokens each step schedules, from the same scheduler output vLLM's own `vllm:spec_decode_num_draft_tokens_total` is summed from. Every one goes into `hulder_draft_tokens_total`, and the first drafted step logs one warning naming the likely causes. Scores are unaffected, and only throughput suffers.

## Measured cost

The connector on zero-draft against a no-probe baseline on an L4 ([overhead table](results/phase3/overhead_l4.md)):
- **Qwen3-4B in bf16:** passes the cost rule on all three metrics, with saturation throughput within −1.68% to 0.88% of the baseline, p99 TTFT −1.57% to −0.72% and p99 TPOT 0.84% to 1.72%.
- **Qwen2.5-1.5B in fp16:** passes p99 TPOT. Throughput and p99 TTFT are inconclusive, because one baseline server was disturbed for a reason not found, though every overhead was under its threshold.

On Qwen3-4B, in-server scores of the 2,101 calibration prompts moved slightly past an offline rerun's floor on 4 of 6 stages, with at most 5 decision flips against the rerun's 4, and `hulder check` passed every stage.

## Limits

- **Monitor-only.** Nothing waits for the score, so a flagged request is still answered.
- **vLLM 0.30.0 only.** The connector uses vLLM internals: the v1 KV connector API, worker metadata and Model Runner V2. A vLLM upgrade needs the connector and the overhead table rerun.
- **Model Runner V2 with async scheduling on.** V1 is untested and treated as unsupported. With async scheduling off or data parallelism the server keeps working but drafts again, which the draft counter shows.
- **One GPU.** Tensor and data parallelism are untested.
- **One probe per server, at one layer, on the last prompt token.** Per-token streaming scores are not built.
- **Prefix caching on** held on both models at the frozen threshold, with 21% to 26% of tokens served from the cache. A workload built around long shared prefixes is untested.
- **FP8 KV cache** is unsettled. It failed to start in the extraction setup ([sweep](results/phase4/sweep_results.md)).
- **Tested on two models and one GPU type for cost:** Qwen3-4B and Qwen2.5-1.5B on an L4, with one sequence length and 128-token replies.

**Upstream.** vLLM's extraction method drafts a token by default, and zero-draft relies on its per-batch-size schedule accepting 0, which isn't documented as a way to run extraction. An open vLLM pull request, [vllm-project/vllm#65991](https://github.com/vllm-project/vllm/pull/65991), proposes an option to turn those decode drafts off. If a release ships something like it, zero-draft could become a plain option there, but nothing here depends on it or assumes it will merge.
