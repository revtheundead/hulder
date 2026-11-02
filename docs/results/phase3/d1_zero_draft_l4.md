# D1 zero-draft cost on the L4 (Phase 3)

**Zero-draft B costs roughly 1% of saturation throughput against a no-probe baseline, and passes the cost rule on all three metrics.** The worst round lost 2.41%, and the mean over four rounds is 1.32%. It drafted 0 tokens in every arm-run.

Two limits go with that number:

- **The comparison with `b_stock`'s roughly 40% is across days.** `b_stock` wasn't in this run. Its figure comes from `d1_l4.md`, measured the day before on a different GPU, where the baseline itself ran 4.7% faster.
- **Hidden states weren't checked on the L4.** This run measures cost and greedy replies only. The connector wrote hidden states during the benches, but they weren't kept or scored. The only evidence that zero-draft's scores hold is the 2070's, with a 1.5B model (`d1_local.md`, zero-draft section).

Zero-draft B is `b_stock`'s command (the stock `ExampleHiddenStatesConnector`, writing to disk) with `"num_speculative_tokens_per_batch_size": [[1, 1, 0]]` in the speculative config, which sets the draft count to 0 at every batch size on Model Runner V2 with async scheduling on. Async scheduling was at vLLM's default, on, in all 8 arm-runs. The manifest predates recording that, so it's read afterwards from the server logs, which hold no flag or warning turning it off. With it off, zero-draft drafts as much as plain B on the 2070 (`d1_local.md`, zero-draft section).

Modal L4 (Ada, 24 GB, driver 610.57.04), 4 CPU, 32 GiB, vLLM 0.30.0, torch 2.13.0. Qwen3-4B at `1cfa9a72` in bf16, probe layer 18, `--max-model-len 2048`, chunked prefill on, prefix caching off, CUDA graphs on. Both arms ran in one container, each arm-run a fresh `vllm serve`. Round 0 is a calibration round, with 16 greedy replies of 64 tokens and a saturation bench of 300 prompts. Rounds 1 to 3 add a fixed-rate bench for 75 s at 13.25 req/s, 70% of the slower arm's round 0 saturation. Every bench reply is 128 tokens. Records are in [`modal/20261101T224613Z/`](modal/20261101T224613Z/), and the server and bench logs stayed local. Estimated cost $0.97, 2,805 s of container time.

Reproduce from the repository root, at `aa6fe4933d89`:

```bash
source scripts/dev-env.sh && uv run --with modal==1.6.1 modal run experiments/phase3/d1_modal.py --experiment zero_draft --spent-usd 0.0
```

## Verdicts

The run completed all 8 arm-runs with 0 failed requests and no stop rule fired. The rule and the stop rules were fixed before the run. It's the revised rule from `d1_l4.md`, so its thresholds are 5% for throughput and 10% for each latency, and a metric fails outright when its smallest overhead exceeds the threshold plus the baseline's range.

| Metric | Overhead by round | Threshold | Baseline's range | Verdict |
|---|---|---|---|---|
| Saturation throughput | 0.30%, 0.89%, 2.41%, 1.69% (rounds 0 to 3) | 5% | 0.71% | Pass |
| Fixed-rate p99 TTFT | −0.50%, 1.85%, 4.21% (rounds 1 to 3) | 10% | 0.29% | Pass |
| Fixed-rate p99 TPOT | 0.30%, 0.90%, 1.92% (rounds 1 to 3) | 10% | 0.77% | Pass |

**A small cost is probably real.** Zero-draft was slower than the same round's baseline in all four rounds, and its worst round is more than three times the baseline's range. That's within the rule, but it isn't zero.

## Throughput and latency

| Round | Baseline req/s | Zero-draft req/s | Baseline p99 TTFT / TPOT (ms) | Zero-draft p99 TTFT / TPOT (ms) |
|---|---|---|---|---|
| 0 | 18.99 | 18.93 | | |
| 1 | 19.13 | 18.95 | 190.1 / 51.86 | 189.2 / 52.02 |
| 2 | 19.08 | 18.62 | 190.0 / 51.72 | 193.5 / 52.19 |
| 3 | 19.12 | 18.80 | 189.6 / 51.46 | 197.5 / 52.45 |

Both arms finished all 994 fixed-rate requests in about 80 s in every round, with up to 113 to 117 in flight. So these latencies aren't queueing above capacity, unlike `b_stock`'s in `d1_l4.md`.

**What this does and doesn't say about the draft token.** Removing it is the only change from `b_stock`'s command, and the loss goes from about 40% to about 1%, so the draft token is now the strong candidate for B's cost. It isn't a paired measurement. The two figures come from separate runs on separate GPUs, the baselines were 19.08 req/s here and 19.98 on the day before (means across rounds), and the fixed rate differed too.

## Greedy replies

Each arm-run's 16 greedy replies are compared with round 0's baseline and classed as in `d1_l4.md`, by the baseline's top-two logprob margin at the first differing token.

| Arm-run | Identical | Exact tie | Within bound | Change |
|---|---|---|---|---|
| Zero-draft, round 0 | 12 of 16 | 2 | 2 | 0 |
| Zero-draft, rounds 1 to 3, each | 12 of 16 | 3 | 1 | 0 |
| Baseline, rounds 1 to 3, each | 16 of 16 | 0 | 0 | 0 |

No divergence reaches a change, so the generation rule passes. Every within-bound divergence has a baseline margin of 0.125, one bf16 step, so none relies on the second step of the bound.

**The replies do differ from the baseline's, and one part of that is unexplained.** Rounds 1 to 3 diverge at the same prompts and tokens (prompts 1, 5, 8 and 9). Round 0 diverges at prompts 7, 8, 9 and 13, and at a different token on prompt 8. `b_stock`'s divergences were the same in every round in `d1_l4.md`. The cause of round 0's difference isn't known, and one fresh server in four isn't enough to say how often it happens.

## Decode counters

From each server's `/metrics`, over both benches. The counters start after the greedy check, so its prompts aren't counted.

| Arm | Runner | Draft tokens | Accepted | Steps, rounds 1 to 3 |
|---|---|---|---|---|
| Zero-draft, each of 4 | V2 | 0 | 0 | 1,937 to 1,943 |
| Baseline, each of 4 | V2 | 0 | 0 | 1,955 to 1,960 |

For comparison, `b_stock` drafted about 173,000 tokens over about 1,200 steps at its heavier fixed-rate workload (`d1_l4.md`).

## KV cache, concurrency and the token budget

| Arm | KV-cache tokens | Max concurrency at 2,048 | Peak running | Logged scheduled-token budget |
|---|---|---|---|---|
| Baseline | 86,768 | 42.37× | 256 | None logged |
| Zero-draft | 83,984 | 41.01× | 256 | 2048 |

Zero-draft's KV cache is 3.2% below the baseline, the same as `b_stock`'s warm figure, since it still holds a hidden state per token. vLLM logs the scheduled-token budget only under a speculative config. From its code at 0.30.0, the baseline's unset value falls back to the same `max_num_batched_tokens` of 2048 (`v1/core/sched/scheduler.py:125`), so the budget isn't a cost of B. The baseline's value is read from the code, not from this run's logs.

## Time

| Stage | Seconds |
|---|---|
| Server start, round 0 | baseline 100, zero-draft 88 |
| Server start, rounds 1 to 3 | 88 to 92 |
| Arm-run, round 0 | 168 to 181 |
| Arm-run, rounds 1 to 3 | 277 to 282 |

Zero-draft's first start wasn't a cold compile, so it reused a compile cache already on the Volume.

Two of the 8 server logs, one per arm, end with an `EngineDeadError` traceback. Both come after the server's shutdown had force-killed its engine, once the benches were over, and no request failed.

## Where this falls short

- `b_stock` wasn't run alongside, so the 40% to 1% comparison is across days and GPUs (above).
- Hidden states and probe scores weren't checked on the L4 (above).
- One GPU, one model, one sequence length and one output length, with 128-token replies at up to 256 running requests.
- Round 0's different greedy divergences are unexplained.
- Four rounds of throughput and three of latency give a range, not an interval.
- The tie bound's second step is reasoned, not measured, though no divergence here depends on it.
