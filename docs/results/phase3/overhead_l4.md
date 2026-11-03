# Phase 3 overhead table on the L4

**Not run yet.** This page fixes the setup, the rules and the cost before any data exists. Only the local smoke below has run, and it says nothing about overhead.

Each model gets its own call, on its own day, with every arm-run in one container and each a fresh `vllm serve`. Arm order rotates by round. Round 0 calibrates, with no fixed-rate bench. Every later round runs a greedy check, a saturation bench of 300 prompts and a fixed-rate bench of up to 75 s, each request 128 output tokens with EOS ignored.

| | Qwen3-4B at `1cfa9a72` | Qwen2.5-1.5B-Instruct at `989aa798` |
|---|---|---|
| Serving | bf16, layer 18 | fp16 to match its probe, layer 20 |
| Probe | Frozen primary `18_-1`, threshold 0.8025 | Bundled demo probe, threshold 0.3952 |
| Arms | `baseline`, `scoring`, `scoring_caching` | The same and `lens` |
| Rounds | 1 calibration + 3 measured | 1 calibration + 2 measured |
| Score stages | 6, on 2,101 calibration prompts, at 64 in flight | 4, on the 360 demo prompts, at 64 in flight |
| Score reference | The full capture's `kept/`, on L4s | The bundled snapshot, from a 2070, so any shift includes the GPU |

Both use `--max-model-len 2048`, chunked prefill and vLLM 0.30.0, with the 2,101 S-Labs calibration prompts as bench prompts.

- `baseline`: no probe, prefix caching off.
- `scoring`: `experiments/phase3/scoring_connector.py` on zero-draft B, no file write, prefix caching off.
- `scoring_caching`: `scoring` with prefix caching on.
- `lens`: vllm-lens 1.3.0, eager, with a persistent hook scoring a linear probe.

Reproduce from the repository root once the run has happened:

```bash
source scripts/dev-env.sh && uv run --with modal==1.6.1 modal run experiments/phase3/d1_modal.py --experiment overhead_qwen3 --spent-usd <today>
uv run --with modal==1.6.1 modal run experiments/phase3/d1_modal.py --experiment overhead_15b --spent-usd <today>
uv run python experiments/phase3/overhead_report.py .cache/modal-runs/<run id> --model qwen3
uv run python experiments/phase3/overhead_report.py .cache/modal-runs/<run id> --model 1.5b
```

## vllm-lens on Qwen3-4B is cross-day

Qwen3-4B has no lens arm here. Its figure is the 2026-10-31 L4 run (`d1_l4.md`), about 4.9 req/s against the baseline's 20 at saturation, roughly 75% lost, with no latency figures because lens didn't sustain the fixed rate. That comparison crosses days, which is against this table's one-container rule. It's accepted because a 75% effect is far larger than anything that moves between days here, where the baseline's range has been under 4% on throughput. The small model has no L4 lens data, so it keeps the arm.

## What `scoring_caching` tests

**It tests whether turning caching on moves scores, not how caching behaves on a shared-prefix workload.** The prompts are unique, with no shared system prompt beyond the chat template's and no conversation history, so nothing here resembles the traffic that caching is meant for. Its overhead is reported against the caching-off baseline, and doesn't measure what caching saves.

It isn't a run with no hits, though. The local smoke's counts show two sources:

- **The chat template.** All prompts share at least one 16-token block. On short prompts that's a real share: 20% of tokens in the first saturation bench, and 22% in a score stage, where 4,896 hit tokens are about one block per request.
- **Repeats.** The benches reuse prompts on one server. 46% to 52% of tokens hit by the end of the fixed-rate bench.

On the L4, the Qwen3-4B score stage reuses the bench's prompts on the same server, so it gets repeat hits too. The small model's stage uses prompts the benches don't. The run reports every stage's hit rate, and the Qwen3-4B stage's shift should be read with it.

## Rules, fixed before the run

**Cost.** D1's rule (`d1_l4.md`): at most 5% on saturation throughput and 10% on fixed-rate p99 TTFT and p99 TPOT, each round paired with that round's baseline. A metric fails if its smallest overhead exceeds its threshold plus the baseline's range, and is otherwise inconclusive if that range exceeds half its threshold. `scoring` is judged by this. `scoring_caching`'s figures are reported beside it, labelled as caching on against off. Lens not sustaining the fixed rate is a result.

**The fixed rate** is 70% of the slowest arm's calibration saturation, leaving lens out, since it would set a rate well below serving load.

**Scores.** Each stage goes through `hulder.check_snapshots` with the reference and its rerun, with prefix caching declared a change under test. If every caching-on stage is PASS, caching holds at this threshold on this set. Anything else means it needs its own calibration, which isn't an overhead failure.

**Stop rules.**
- From D1: a startup or measurement timeout, a failed request, and an arm's greedy replies diverging beyond a fresh baseline's. Lens running out of measurement time is a result, not a stop.
- A scoring arm off Model Runner V2, with async scheduling off, or with any draft token.
- More than 1% of a stage's 16-token replies without a score, or `/metrics` counting fewer scores than requests.
- On Qwen3-4B, a `scoring` stage's mean |Δscore| against `kept/` above 0.00585, three times D1's highest zero-draft stage, which would suggest the connector is reading the wrong thing.
- `scoring_caching` is exempt from the greedy rule, since caching can change replies, and from the shift rule, since its shift is what's being measured. Its divergences are reported.

## Cost

At about $0.000345/s for the L4 with 4 CPUs and 32 GiB, from `experiments/modal/smoke_lib.py`'s published rates.

| | Qwen3-4B | Qwen2.5-1.5B |
|---|---|---|
| Arm-runs | 12 | 12 |
| Timeouts: cold, warm, measure, stage (s) | 270, 270, 240, 120 | 420, 270, 240, 120 |
| Measure call's limit, plus 333 s of container startup | 7,500 s | 7,860 s |
| The guard's worst case, with download and pre-flight | $2.78 | $2.90 |
| Room under the $3 daily guard | $0.22 | $0.10 |
| Expected, arm time only | About 3,300 s, $1.15 | About 3,500 s, $1.20, not measured on an L4 |
| With about 57 minutes of queueing charged, as an upper estimate | About $2.30 | About $2.40 |

## Local smoke (2026-11-03)

`d1_l4.py --smoke --overhead` and `--overhead-lens` ran each experiment's path on the RTX 2070, on the small model with a calibration round and one more, through `in_container`, and `overhead_report.py` scored the stages. The workload is tiny, 16 requests of 16 tokens at saturation.

```bash
source scripts/dev-env.sh && uv run python experiments/phase3/d1_l4.py --smoke --overhead --out .cache/overhead-smoke
uv run --with vllm-lens==1.3.0 python experiments/phase3/d1_l4.py --smoke --overhead-lens --out .cache/overhead-smoke-lens
uv run python experiments/phase3/overhead_report.py .cache/overhead-smoke --model 1.5b
```

| | Three arms | Four arms |
|---|---|---|
| Stop rules fired | None | None |
| Draft tokens, scoring arms | 0 | 0 |
| Replies without a score, of 360, each stage | 0 | 0 |
| Greedy replies diverged from round 0's baseline | 0 | 0 |
| `scoring` stage, mean \|Δscore\| against the bundled reference | 0.000252, PASS | 0.000246, PASS |
| `scoring_caching` stage | 0.000298, PASS | 0.000298, PASS |
| Lens saturation, req/s, against the baseline's 17.8 to 18.1 | | 3.45 to 3.55 |

The rerun's floor is 0.000208, and every stage is above it with 0 flips. Caching on is the highest of the four.

**One thing to watch, not a result.** Saturation fell from 15 to 18 req/s on the baseline to 5.8 to 12 on both scoring arms. That's 16 short requests on a GPU whose throughput isn't representative (docs/design.md §8.1), and the workload is mostly prefill, unlike the L4's 128-token replies, where zero-draft B lost 1.5%. It still could be a real cost of the connector or of extraction on prefill-heavy traffic, and the cause isn't known. The L4 run should show it if it's the connector.

## What the draft doesn't settle

- Whether `scoring_caching` needs its own compile. If it does, Qwen3-4B's 270 s cold start stops the run in round 0, for about $0.22.
- The small model's fixed rate and timing on an L4.
- That the Modal path works. `d1_modal.py`'s experiments have been imported and costed, not called.
- Two measured rounds on the small model give a thin range, and the baseline's noise could leave more metrics inconclusive.
