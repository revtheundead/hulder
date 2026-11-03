# The saturation drop on short replies (Phase 3)

**On short replies on the RTX 2070, the extraction path costs about 15% to 18% of saturation throughput, and the scoring connector adds nothing measurable. The 40% to 60% drop in the overhead smoke (`overhead_l4.md`) didn't reproduce.** This says which step costs what on this workload, not what it costs on an L4, since throughput on this GPU isn't representative (docs/design.md §8.1).

Qwen2.5-1.5B-Instruct at `989aa798` in fp16, `--max-num-seqs 8`, chunked prefill on, prefix caching off, vLLM 0.30.0, Model Runner V2 with async scheduling on. The smoke's workload: 4 greedy replies, then saturation benches of 16 requests of 16 tokens. Three arms interleaved, each a fresh `vllm serve`: `baseline`, `b_zero_draft` (the stock connector on zero-draft, which writes files) and `scoring` (the scoring connector, which doesn't). Results are in [`d3_drop.json`](d3_drop.json) and [`d3_drop_first_run.json`](d3_drop_first_run.json).

Reproduce from the repository root, at `6c8fe7be747a` (the first run at `8974831af18d`):

```bash
source scripts/dev-env.sh && uv run python experiments/phase3/d3_drop.py --out .cache/d3-drop-2
```

**Why there are two runs.** The first, with 3 rounds, came out "no drop" under the rule, because one round's baseline and `b_zero_draft` ran far slower than the rest and decided the mean. That result stands, below. It couldn't say which step costs what, since rounds 1 and 2 pointed the other way from the mean, and the disturbance had no explanation. So a second run of 5 rounds, with the rule unchanged, a disturbed-round check added and the GPU's state recorded, went looking for an answer. It wasn't run because the first result was unwelcome, but the decision to run it did come after seeing that data, and that's why both are reported.

## The rule

Fixed in the script before the first run, on each server's first bench, in req/s lost. Under a 10% mean drop from baseline to scoring, nothing reproduced. A step accounts for the drop if it holds at least two thirds of the loss and the rounds agree on which step is larger.

## The second run, 5 rounds

No stop rule fired, no round was disturbed, and the rule says **extraction** on both benches.

| | `baseline` | `b_zero_draft` | `scoring` |
|---|---|---|---|
| First bench, median req/s (range) | 16.42 (15.67 to 17.73) | 14.09 (13.76 to 14.51) | 14.42 (13.31 to 14.99) |
| Second bench, median req/s (range) | 17.97 (15.89 to 18.37) | 14.73 (14.23 to 15.65) | 14.59 (13.92 to 14.95) |
| p50 TPOT, median of first bench (ms) | 14.3 | 19.9 | 18.9 |

| Bench | Mean drop, baseline to scoring | Share at extraction | Share at scoring | Scoring's drop from `b_zero_draft`, by round |
|---|---|---|---|---|
| First | 14.9% | 100% | 0% | 4.9, 7.2, −4.5, −6.4, −6.2% |
| Second | 18.4% | 90% | 10% | 11.0, 1.0, 1.0, 2.2, −4.6% |

**Every arm ran the same 34 engine steps on the same 1,626 tokens, with no drafts,** so the cost is per step, not extra work. TPOT rises by about 5 ms a step. **The file write isn't it either,** since `b_zero_draft` writes and `scoring` doesn't, and they match. What's left is the extraction path itself, such as the hidden-state cache and the drafter's forward. That's a process of elimination, and nothing here splits it further.

The connector isn't the cause, so it wasn't profiled.

## The first run, 3 rounds

The rule said **no drop**, and that stands as its result. Round 0's baseline ran at 5.4 req/s and its `b_zero_draft` at 2.3, with p50 TPOTs of 52 and 180 ms against about 14 and 21 ms in later rounds, while its `scoring` ran normally at 13.4. Those two rows decided the mean. Rounds 1 and 2 had total drops of 35% and 3% on the first bench.

So, before the second run, a round with a bench at over twice its arm's median TPOT was set to count as disturbed, with results reported with and without such rounds, and the GPU's state recorded per bench. That change came after seeing the first run's data. The second run had no disturbed rounds, so it made no difference to the result.

**What disturbed round 0 isn't known.** It's the likeliest explanation for the smoke's drop too, whose slowest scoring arm-runs, 5.7 to 6.9 req/s, sit outside everything in the second run. That's an inference, since the smoke recorded no GPU state. In the second run the GPU held P2 at 1,890 to 1,905 MHz and about 50 W throughout.

## Where this falls short

- One model and GPU, at 8 running requests. The L4's 128-token replies at up to 256 running requests lost 1.5% (`d1_confirmation_l4.md`), and whether that gap comes from reply length, batch size or the GPU isn't separated.
- 16 requests per bench is short, so one bench's rate is noisy, and the arms' ranges overlap between `b_zero_draft` and `scoring`. A small scoring cost, a few percent, can't be ruled out.
- Each server's benches aren't independent.
