# D3 monitor-only spike (Phase 3)

**A connector can score a probe inside vLLM at the end of prefill and return the score in a non-streamed reply and on `/metrics`, with matching scores and unchanged replies. Replies of 1 token never carry a score, and most of 2 tokens don't.** This tests whether the path works on the local GPU. It doesn't measure overhead, and caching on, bf16 and the L4 are untested.

RTX 2070 (Turing), vLLM 0.30.0. Qwen2.5-1.5B-Instruct at `989aa798` in fp16 under `vllm serve`, zero-draft B, chunked prefill and CUDA graphs on, prefix caching off, `--max-num-seqs 8`. The bundled demo probe (layer 20, position −1, threshold 0.3952) on its 360 boundary-pairs prompts. Each server is a fresh process. Results are in [`d3_spike.json`](d3_spike.json).

Reproduce from the repository root, at `f3b92ec6000f`:

```bash
source scripts/dev-env.sh && uv run python experiments/phase3/d3_spike.py --out .cache/d3-spike
```

## How it works

`experiments/phase3/scoring_connector.py` subclasses vLLM's `ExampleHiddenStatesConnector`. On the step that schedules a request's last prompt token, the scheduler side tells the worker where its hidden state will be. The worker queues the probe on its copy stream and polls the result on later steps, using the stock connector's stream pattern. Scores return through worker metadata. A request's blocks stay held until its score is read.

## Results

The rule was fixed in the script before the first run. All three checks pass.

| Check | Rule | Result |
|---|---|---|
| Correct, default budget | Each reply's score within 1e-6 of `hulder.apply_probe` on the same request's saved hidden state, 360 of 360 | 360 of 360, largest gap 1.2e-7, 0 flips |
| Correct, `--max-num-batched-tokens 64` | The same, with the 30 prompts over 64 tokens prefilled over several steps | 360 of 360, largest gap 1.2e-7, 0 flips |
| Replies unchanged | Without the file write, 32 greedy replies of 64 tokens match the stock connector on zero-draft by token id, with no drafts | 32 of 32, 0 draft tokens |
| Surfaces | `/metrics` counts every score and reply, agrees with the replies without a score, no files written, and 16 streamed replies count a score each with none in a chunk | Pass at all 12 settings, and on the 16 streamed |

## Replies without a score

Scores travel with engine steps, so a reply can end before its score arrives. The first column of counts is this run, and the second is the run before it, which finished and then failed writing its results file.

| Reply tokens | Without a score, of 360, concurrency 1 | Concurrency 8 |
|---|---|---|
| 1 | 360, 360 | 360, 360 |
| 2 | 190, 181 | 344, 338 |
| 3, 4, 8, 16 | 0, 0 | 0, 0 |

A late score still reaches `/metrics`, which counted all 360 and the same 188 flagged at every setting. A client limiting replies to 1 or 2 tokens should read `/metrics`, not the reply. Where the line falls in steps depends on timing, so 3 tokens on this GPU doesn't promise anything on another.

## Where this falls short

- **Overhead isn't measured.** No throughput or latency here is representative (docs/design.md §8.1). The overhead table is the next measurement, on the L4.
- **Caching on is untested,** though the logic handles a cache hit in its unit test. The overhead table's caching-on arm is its first run.
- **No host sync is by construction, not by check.** The code never waits on the GPU, but vLLM's GPU sync debugging wasn't turned on to confirm it.
- **Preemption and aborts weren't exercised on a server.** A preempted request is reported once, and that's only unit-tested. Nothing here forced either case.
- **One model, one GPU, fp16 only.** bf16 and Qwen3-4B are untested with this connector.
- **It can flag a request, not stop it** (docs/design.md §5 D3, §10).
