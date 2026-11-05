# Phase 3 overhead table on the L4

**On Qwen3-4B on an L4, the scoring connector on zero-draft B passes the cost rule on all three metrics, with no drafts and no stop rule fired, and turning prefix caching on holds at the frozen threshold.** **On Qwen2.5-1.5B it passes only p99 TPOT.** Throughput and p99 TTFT are inconclusive because one baseline server was disturbed, cause unknown, though every overhead is under its threshold, and caching on holds there too. The small model's greedy check was judged by a flawed classifier, now fixed, and under the fixed one that run's stop rule would fire, as its section describes. The setup, the rules and the cost were fixed on this page before the runs.

| Model | Run | Saturation loss | p99 TTFT | p99 TPOT | `scoring` | Short replies, `scoring` | Caching on |
|---|---|---|---|---|---|---|---|
| Qwen3-4B | `20261103T212652Z` | −1.68 to 0.88% | −1.57 to −0.72% | 0.84 to 1.72% | **Passes** | −1.70 to 3.35%, within 5% | **Holds**, PASS on 3 of 3 stages, 26.1% of tokens hit |
| Qwen2.5-1.5B | `20261104T175409Z` | −0.37 to 2.86% | −39.8 to 1.89% | 2.31 to 2.62% | **Inconclusive** on throughput and p99 TTFT, p99 TPOT passes | −2.00 to −0.28%, within 5% | **Holds**, PASS on 2 of 2 stages, 21.1% of tokens hit |

Records are in [`modal/20261103T212652Z/`](modal/20261103T212652Z/) and [`modal/20261104T175409Z/`](modal/20261104T175409Z/).

## Qwen3-4B

Modal L4 (Ada, 24 GB, driver 610.57.04), one GPU and one container, at `afb17ce4ba16`. 12 arm-runs and 6 score stages, with 0 failed requests, all on Model Runner V2 with async scheduling on. The fixed rate was 13.2 req/s, 70% of the slowest arm's calibration saturation.

**Estimated cost $1.21,** from the measure call's 3,487 s of wall time, with almost no wait for a GPU. The actual charge is on Modal's usage page.

| Metric | Threshold | Baseline's range | `scoring` by round | Verdict | `scoring_caching` by round |
|---|---|---|---|---|---|
| Saturation throughput loss | 5% | 1.17% | −0.79, −1.68, 0.08, 0.88% | Pass | −0.41, −0.85, 0.46, 0.24% |
| Fixed-rate p99 TTFT | 10% | 2.07% | −0.83, −0.72, −1.57% | Pass | 1.71, −1.24, −0.96% |
| Fixed-rate p99 TPOT | 10% | 0.86% | 1.72, 0.84, 0.96% | Pass | 1.90, 0.89, 0.56% |

The baseline stayed within half of every threshold, so these are passes rather than D1 confirmation's inconclusive verdict. p99 TTFT was 191 to 198 ms and p99 TPOT 51.8 to 52.8 ms. `scoring_caching` is caching on against the caching-off baseline, so it doesn't separate the two effects.

**Draft tokens: 0** in every arm-run, benches and score stages. **Greedy replies: 0 changes.** `scoring` diverged from the baseline on 4 prompts per server and `scoring_caching` on 4 to 6, all at exact ties or within the tie bound. Fresh baselines diverged on 0. **Margin:** measurement took at most 208 s of its 240 s limit.

### Score stages against `kept/`

All 2,101 rows at threshold 0.8025, with 0 replies without a score and `/metrics` counting every score. The rerun's floor is 0.001641 with 4 flips.

| Stage | `hulder check` | Flips | Mean \|Δscore\| | Past the floor | Tokens hit |
|---|---|---|---|---|---|
| r1 `scoring` | PASS | 1 | 0.001831 | Yes | |
| r2 `scoring` | PASS | 5 | 0.001862 | Yes | |
| r3 `scoring` | PASS | 4 | 0.001777 | Yes | |
| r1 `scoring_caching` | PASS | 4 | 0.001880 | Yes | 26.1% |
| r2 `scoring_caching` | PASS | 2 | 0.001742 | No | 26.1% |
| r3 `scoring_caching` | PASS | 2 | 0.001679 | No | 26.1% |

**Caching on holds at this threshold on this set,** under the rule fixed before the run, with prefix caching declared a change under test. Its shift falls in the same range as the caching-off stages. `scoring` in round 2 has one flip more than the floor, and flips aren't an input to the verdict.

**Hit rates.** 18,912 of 72,360 tokens in each score stage, 26.1%. 3,232 of 5,760 in each short bench, 56.1%, since it repeats the saturation bench's prompts. 6,464 of 26,880 from the saturation bench through the fixed-rate one in measured rounds, 24.0%.

### Short replies

| Arm | Loss against baseline, rounds 0 to 3 | Past 5% |
|---|---|---|
| `scoring` | 1.61, 3.35, −1.70, −0.01% | No |
| `scoring_caching` | −12.3, −17.6, −13.9, −18.7%, faster on its hits | No |

**This is a known limit: it doesn't test prefill-heavy traffic.** The short bench's prompts come to about 19 tokens each (5,760 tokens over 300 requests), so with 16-token replies it spends little on prefill. The 2070's check, which found about 15% to 18% at the extraction path (`d3_drop.md`), had prompts of about 85 tokens. So this run neither confirms nor refutes that cost, and long prompts on an L4 are unmeasured. Each bench also lasted only about 3 s.

### Where this falls short

- **Numerical paths weren't observable.** This run keeps scores, not hidden states, so it can't make the bit comparisons that showed D1's servers changing path, 4 of 12 extraction servers in the confirmation and zero-draft runs (`d1_confirmation_l4.md`). Any such change here sits inside the stages' shifts.
- **vllm-lens on Qwen3-4B is cross-day** (below).
- Three latency rounds and four throughput rounds give ranges, not intervals. One model, one GPU, and 128-token replies at up to 256 running requests.
- Caching on is tested only with unique, short prompts, and not on a shared-prefix workload (below).
- The 60 s fixed-rate bench means about 790 requests behind each p99, fewer than D1's.

## Qwen2.5-1.5B

Modal L4 (Ada, 24 GB, driver 610.57.04), one GPU and one container, at `063dceb32dbe`. 12 arm-runs and 4 score stages, with 0 failed requests in any completed bench. The scoring arms and the baseline ran on Model Runner V2 and lens on V1, all with async scheduling on. The fixed rate was 28.8 req/s, 70% of `scoring`'s calibration saturation of 41.2.

**Estimated cost $2.01,** from the measure call's 5,807 s of wall time. The arm-runs took 3,126 s, about $1.08, so about 45 minutes went on something else, which is probably waiting for a GPU. The actual charge is on Modal's usage page.

| Metric | Threshold | Baseline's range | `scoring` by round | Verdict | `scoring_caching` by round |
|---|---|---|---|---|---|
| Saturation throughput loss | 5% | 4.35% | 2.86, 1.08, −0.37% | Inconclusive | −1.74, −5.47, −3.99% |
| Fixed-rate p99 TTFT | 10% | 55.9% | −39.8, 1.89% | Inconclusive | −45.8, −4.57% |
| Fixed-rate p99 TPOT | 10% | 0.20% | 2.62, 2.31% | Pass | −1.77, −1.64% |

**So throughput and p99 TTFT are recorded as inconclusive, and the run isn't being repeated for now.** `scoring` doesn't pass the cost rule here, and nothing fails it. The rule calls a metric inconclusive when the baseline's range is over half its threshold.

The reason is one baseline server, in round 1, and what disturbed it is unknown. Its p99 TTFT was 166.8 ms against round 2's 94.0, with the other arms at 89.7 to 100.4, which is why round 1's overheads look like large gains. It also had the lowest baseline saturation, 40.6 req/s against 41.4 and 42.4. Leaving it out would bring throughput's range to 2.46%, just under its 2.5% allowance, but that's arithmetic after the fact and not a pass. With two measured rounds there's nothing to outvote it. Saturation benches lasted about 7 s.

**Draft tokens: 0** in every arm-run, benches and score stages. **Margin:** the first server's cold start took 132 s of its 420 s, and lens's first took 316 s. Measurement took at most 176 s of its 240 s limit, leaving out lens.

### The greedy check

**Under the check as it now stands, this run's stop rule would fire.** The run isn't reclassified, and its records stay as they came out. `scoring` and `scoring_caching` diverged from the baseline on one prompt of 16 on all 6 servers, at the same token, prompt 0's token 39, `" will"` against `" your"`. Fresh baselines diverged on 0. Lens diverged on one prompt at an exact tie.

The run filed that divergence as "within bound", at a margin of −2.6875 against a tie bound of 0.25. That margin is the wrong sign. vLLM lists the sampled token first, so the baseline's chosen token sits 2.69 below its alternative, which a greedy step shouldn't do on the values it chose from. The check only tested that the margin was at most the bound, so a negative one passed. A negative or non-finite margin now counts as a change, which makes it 1 against the fresh baselines' 0.

Repetition penalty explains the sign. This model's `generation_config.json` sets `repetition_penalty` to 1.1, the check didn't override it, and vLLM's logprobs are raw by default (`logprobs_mode`), so they don't show the penalised values that chose the token. The check now refuses to classify margins on a config with a logit processor unless the server reports them after it, and this model's greedy check asks for `repetition_penalty` 1.0. Neither has run on an L4.

**The divergence looks like a near-tie, on a 2070 and not on the L4.** A local rerun on the same 16 prompts and 64 tokens, with the GPU otherwise idle, gave:

| Setting | Token at prompt 0's position 39 | Top two logprobs | Margin | Fresh baseline and `scoring` identical to baseline |
|---|---|---|---|---|
| Penalty 1.1, raw, as the run | `" your"` | `" your"` −0.3244, `" the"` −1.6838 | 1.3594 | 16 of 16, both |
| Penalty 1.1, after the penalty | `" your"` | `" your"` −1.0725, `" will"` −1.0767 | 0.0043 | 16 of 16, both |
| Penalty 1.0, raw | `" define"` | `" define"` −0.6639, `" understand"` −1.0545 | 0.3906 | 16 of 16, both |

The first row's margin is the one the L4's `scoring` servers recorded at that token, 1.3593750298 in both, so the 2070 appears to be at the same decision. After the penalty, `" your"` and `" will"` are the same two tokens the L4 split on, 0.0043 apart, against a bound of 0.25. That's the evidence for a near-tie. It has three limits. The divergence itself didn't recur here, so what tipped the L4 is unobserved. The margin was read on a 2070, not on the L4's baseline. And at penalty 1.0 the reply has already taken another path by token 39, so that row isn't the same decision, and 16 of 16 there says nothing about whether the L4 would diverge at penalty 1.0.

It doesn't touch the cost figures or the scores. A score is read at the end of prefill, before any generated token.

### Score stages against the bundled snapshot

All 360 rows at threshold 0.3952, with 0 replies without a score and `/metrics` counting every score. The rerun's floor is 0.000208 with 0 flips, and it comes from a 2070, like the reference.

| Stage | `hulder check` | Flips | Mean \|Δscore\| | Past the floor | Tokens hit |
|---|---|---|---|---|---|
| r1 `scoring` | PASS | 0 | 0.000355 | Yes | |
| r2 `scoring` | PASS | 0 | 0.000344 | Yes | |
| r1 `scoring_caching` | PASS | 0 | 0.000354 | Yes | 21.1% |
| r2 `scoring_caching` | PASS | 0 | 0.000345 | Yes | 21.1% |

**Caching on holds at this threshold on this set,** under the rule fixed before the run. Its shift matches the caching-off stages to within the rounds' own difference. The shift includes the change of GPU, from a 2070 to an L4, as well as batch size and caching, and every stage is about 1.7 times the floor. This is the demo probe on 360 prompts at a 5% target, so it can't resolve small changes in FPR.

**Hit rates.** 4,736 of 22,422 tokens in each score stage, 21.1%. 10,048 of 12,060 in each short bench, 83.3%, since it repeats the saturation bench's prompts. 47,792 of 94,037 from the saturation bench through the fixed-rate one in measured rounds, 50.8%.

### Short replies and lens

| Arm | Loss against baseline, rounds 0 to 2 | Past 5% |
|---|---|---|
| `scoring` | −0.84, −2.00, −0.28% | No |
| `scoring_caching` | −21.4, −16.8, −14.9%, faster on its hits | No |
| `lens` | 70.7, 69.7, 70.7% | Yes |

The short bench's prompts come to about 40 tokens each, against 19 on Qwen3-4B and about 85 in the 2070's check (`d3_drop.md`), and each bench lasted about 2 s. So this neither confirms nor refutes the 2070's 15% to 18%, for the same reason as on Qwen3-4B, and prefill-heavy traffic stays unmeasured.

**Lens lost 84% of saturation throughput,** at 6.57 to 6.72 req/s against the baseline's 40.6 to 42.4, 84.2, 83.8 and 83.7% by round. It failed the cost rule and didn't sustain the fixed rate, completing 285 and 296 of 1,729 requests in time, so it has no latency figures. The report's throughput verdict reads only round 0, since the two later rounds timed out at the fixed-rate bench, and the other two rounds' figures here come from their saturation benches.

**The 0 from lens's hook was a failed collect, so the hook very probably did run, and the 84% very probably includes it.** `collect_hook_results` returned 0 in round 0, the only round that reached that call. The server's log shows why: `/v1/hooks/collect` answered 500, with a CUDA out-of-memory error in the API process while it unpickled the results, and the client reads a failed answer as no results. Results to unpickle means the hook had produced some, which is an inference from vllm-lens's source and that traceback, since the run has no count of them. This page earlier said the loss might be eager mode's alone and could only grow with a working hook, and that no longer holds. As on Qwen3-4B, the loss can't be split between the two.

**Why it ran out of memory is unexplained.** vllm-lens keeps the hook's GPU tensors as GPU tensors through the pickle, so the API process has to allocate on the GPU, and that much is in the traceback. The failure didn't reproduce locally. On an otherwise idle 2070, `/v1/hooks/collect` answered 200 at every setting:

| `--gpu-memory-utilization` | Requests | Free before collect | Results |
|---|---|---|---|
| 0.7 | 16 | 2,174 MiB | 16 |
| 0.87 | 16 | 782 MiB | 16 |
| 0.87 | 616, the L4 server's count | 778 MiB | 616 |

0.87 is as high as this GPU starts, since vLLM sees 6.98 of its 8 GiB free, and the L4 ran at 0.92, which by vLLM's figures left it about 1.6 GiB, twice the 2070's room here. So low free memory at startup doesn't explain it on its own, and 616 scores cost almost none. The L4's free memory at the collect itself wasn't recorded. Differences left untested are the L4's concurrency, its driver and what its engine had allocated by then. The failure doesn't change the throughput figures, since the collect comes after the benches.

### Where this falls short

- **Two of three cost metrics are inconclusive,** on two measured rounds, one of them with a disturbed baseline.
- **The greedy check's stop rule would now fire on this run,** and the fix hasn't run on an L4 (above).
- **Lens's out-of-memory failure at collect is unexplained** and has no local reproduction (above).
- **The reference is from another GPU,** so the score shift doesn't separate the connector from the 2070-to-L4 change.
- Numerical paths weren't observable, as on Qwen3-4B.
- Caching on is tested only with unique, short prompts.

The local checks' records are in [`greedy_lens_local.json`](greedy_lens_local.json):

```bash
source scripts/dev-env.sh && uv run --with vllm-lens==1.3.0 python experiments/phase3/greedy_lens_local.py --out .cache/greedy-lens-local
```

## Setup

Each model gets its own call, on its own day, with every arm-run in one container and each a fresh `vllm serve`. Arm order rotates by round. Round 0 calibrates, with no fixed-rate bench. Every later round runs a greedy check, a saturation bench of 300 prompts, a short-reply saturation bench of the same 300 prompts at 16 output tokens, and a fixed-rate bench of up to 60 s, with EOS ignored and 128 output tokens outside the short bench. Round 0 has both saturation benches.

**Why the short bench.** Every other timed bench is 128-token replies, and a local check found a cost there that those wouldn't show: about 15% to 18% of saturation throughput on short replies on the 2070, all of it at the extraction path (`d3_drop.md`). **Why 60 s.** The short bench takes about 30 s, and that's an estimate from a bench process's 24 s start. D1's fixed-rate bench ran 75 s. At 60 s, a Qwen3-4B arm-run's measurement was at most 208 s of its 240 s limit, against 188 s in D1, which keeps over 30 s of margin. It means about 790 requests behind each p99, so these latencies aren't from the same length of bench as D1's.

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
- **Repeats.** The benches reuse prompts on one server. 46% to 52% of tokens hit by the end of the fixed-rate bench, measured before the short bench was added.
- **The short bench itself.** It sends the prompts the saturation bench just sent, so with caching on it's almost all hits: 90% of tokens in the smoke, where `scoring_caching` ran 33% to 40% faster than the baseline. That says nothing about the connector on short replies, so read `scoring` for that. The run reports this bench's hit rate too.

On the L4, the Qwen3-4B score stage reuses the bench's prompts on the same server, though its measured 26.1% is lower than this suggested, because these prompts are short and a hit needs a full 16-token block. The small model's stage uses prompts the benches don't.

## Rules, fixed before the run

**Cost.** D1's rule (`d1_l4.md`): at most 5% on saturation throughput and 10% on fixed-rate p99 TTFT and p99 TPOT, each round paired with that round's baseline. A metric fails if its smallest overhead exceeds its threshold plus the baseline's range, and is otherwise inconclusive if that range exceeds half its threshold. `scoring` is judged by this. `scoring_caching`'s figures are reported beside it, labelled as caching on against off. Lens not sustaining the fixed rate is a result.

**Short replies.** Each arm's short-bench throughput is reported against the same round's baseline, with the 5% threshold marked, and stays out of the arm's pass or fail, so `scoring`'s verdict stays comparable with D1's. Past 5% is recorded as a limit on prefill-heavy traffic. It would be a finding about zero-draft B rather than the connector unless `scoring` differs from D1's figures.

**The fixed rate** is 70% of the slowest arm's calibration saturation, leaving lens out, since it would set a rate well below serving load.

**Scores.** Each stage goes through `hulder.check_snapshots` with the reference and its rerun, with prefix caching declared a change under test. If every caching-on stage is PASS, caching holds at this threshold on this set. Anything else means it needs its own calibration, which isn't an overhead failure.

**Stop rules.**
- From D1: a startup or measurement timeout, a failed request in any bench, and an arm's greedy replies diverging beyond a fresh baseline's. Lens running out of measurement time is a result, not a stop.
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
| Room under the $3 daily guard | $0.22 | $0.096 |
| Expected, arm time only | About 3,480 s, $1.20. Came to 3,487 s, $1.21 | About 3,680 s, $1.27. Came to 3,126 s, $1.08 |
| With about 57 minutes of queueing charged, as an upper estimate | About $2.40 | About $2.45. The run's estimate, with 45 minutes, was $2.01 |

The short bench and the shorter fixed-rate bench add about 15 s to each arm-run. The guard's worst cases don't change, because the 240 s limit on measurement stays.

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

A third smoke, the four arms again at `4e20b31d3726`, ran the short bench at 8 tokens. No stop rule fired. On it, against the same round's baseline, `scoring` was 5.5% faster and 7.1% slower, `lens` 53% and 52% slower, and `scoring_caching` 40% and 33% faster on its 90% hits.

The rerun's floor is 0.000208, and every stage is above it with 0 flips. Caching on is the highest of the four.

**The saturation drop this smoke seemed to show didn't reproduce.** Saturation fell from 15 to 18 req/s on the baseline to 5.7 to 12 on both scoring arms. A 5-round check on the same workload (`d3_drop.md`) found about 15% to 18%, all of it at the extraction path and none at the connector.
