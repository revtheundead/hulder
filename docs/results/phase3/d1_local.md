# D1 local checks (Phase 3)

Checks of the in-server options for D1 (docs/design.md §5) on the local GPU, before any remote run. They test whether option B, vLLM's hidden-state connector, serves a probe correctly with production settings, and whether `vllm-lens` runs at the pin. They don't measure overhead, which needs the L4.

RTX 2070 (Turing, driver 621.01), vLLM 0.30.0, vllm-lens 1.3.0. Qwen2.5-1.5B-Instruct at `989aa798` in fp16 under `vllm serve`, with chunked prefill and CUDA graphs on (piecewise) and prefix caching off. The bundled demo probe (layer 20, position −1, threshold 0.3952) and the bundled demo snapshots (`demo_fp16.npz` as the reference and `demo_fp16_run1.npz` as its rerun) on their 360 boundary-pairs prompts. Every server is a fresh process with `--max-num-seqs 8`. Results are in [`d1_local.json`](d1_local.json).

Reproduce from the repository root, at `9220bb37003c` for checks 1 to 5, at `2120a6eaa1d0` for the generation repeats and at `9541fad1aff1` for the no-op connector:

```bash
source scripts/dev-env.sh && uv run python experiments/phase3/d1_local.py --out .cache/d1-local
uv run --with vllm-lens==1.3.0 python experiments/phase3/d1_local.py --out .cache/d1-local --lens
uv run python experiments/phase3/d1_local.py --out .cache/d1-local --repeats
uv run python experiments/phase3/d1_local.py --out .cache/d1-local --no-write
```

## Results

| Check | What ran | Result |
|---|---|---|
| 1. Skew | 360 prompts through B at concurrency 8, `max_tokens=1`, scored against the offline reference | PASS, 0 flips. Mean \|Δscore\| 0.000229 against the rerun's 0.000208, not beyond it. Max 0.0033 |
| 2. Generation | 32 prompts, greedy, 64 tokens, concurrency 1, B against a server without extraction | 31 of 32 identical (below) |
| 2. Streaming | One streamed reply from B, 13 chunks | No chunk carries `kv_transfer_params` |
| 3. Chunked prefill | Check 1 with `--max-num-batched-tokens 64`, so the 30 prompts over 64 tokens prefill over several steps | Serves, token ids match. PASS, 0 flips. Mean \|Δscore\| 0.000332, beyond the rerun's 0.000208. Max 0.0024 from check 1's scores |
| 4. Decode | One greedy request of 64 tokens on each server, from `/metrics` | Plain: 64 steps, no drafts. B: 64 steps, 63 draft tokens, 0 accepted |
| 5. vllm-lens | 360 prompts at concurrency 1 through the completions API, residual stream at lens layers 19 and 20 | Lens layer 19 is hulder's layer 20: PASS, 0 flips, mean \|Δscore\| 0.00047. Lens layer 20 is the next layer: 7 flips, mean 0.032. Eager mode forced, CUDA graphs off |
| 5. Lens disabled | `VLLM_LENS_DISABLE=1`, check 2's generation against the plain server | 32 of 32 identical |
| No-op connector | `experiments/phase3/no_write_connector.py`, one 16-token request, then 16 more at concurrency 8 | Serves all 17. No `kv_transfer_params`, no files written, 15 draft tokens over 15 decode steps |

**Chunked prefill works with the connector.** docs/design.md §5 and the offline extraction decision said it needed chunked prefill off. At 0.30.0 it doesn't: in check 3, prompts prefilled over several steps came back with the prompt's token ids, and their scores flipped no decision. The shift beyond the floor can't be pinned on chunking alone, since a 64-token budget also changes which prompts share a step for all 360.

**B's scores match the offline reference.** Check 1's shift stays inside the rerun's floor, which itself varied from 0.000205 to 0.000286 across three live demo runs (`docs/results/phase1/live_demo.md`). No prompt in check 1 was long enough to chunk at the default budget. The longest is 303 tokens.

**B drafts a token on every decode step and always rejects it.** The extract_hidden_states drafter proposes the token it was just given, and verification rejects it (0 of 63 accepted). So each decode step schedules one extra token per request. `vllm:iteration_tokens_total` doesn't show it, because it counts computed prompt tokens and accepted output tokens only. The draft counters do.

**B's score reaches the client only at the end of a non-streamed reply.** A streamed reply carries no `kv_transfer_params` at all.

## The one generation difference

Prompt 17 diverges at its 47th generated token (index 46). In the plain server, `" properly"` and `" thoroughly"` tie exactly (logprob margin 0.0) and `" properly"` is picked. In B, `" thoroughly"` leads by 0.0156 and is picked. The rest of each reply follows from there.

The difference is deterministic, not run-to-run noise. Two fresh plain servers gave 32 of 32 identical replies, two fresh B servers did too, and each plain server against each B server diverged on prompt 17 at token 46 alone, with the same margins. B computes its decode steps two tokens wide, so its numbers differ slightly from the plain server's, enough to break an exact tie.

## Zero-draft B (2026-11-01)

B on the L4 lost about 40% of its saturation throughput (`d1_l4.md`), and the draft token it schedules on every decode step is the leading suspect. On Model Runner V2, which the 2070 and the L4 both use, adding `"num_speculative_tokens_per_batch_size": [[1, 1, 0]]` to B's speculative config sets the draft count to 0 at every batch size. Results are in [`d1_zero_draft.json`](d1_zero_draft.json).

```bash
source scripts/dev-env.sh && uv run python experiments/phase3/d1_local.py --zero-draft --out .cache/d1-zero-draft
uv run python experiments/phase3/d1_local.py --zero-draft-scores --out .cache/d1-zero-draft-scores-3
```

The first ran at `d6e6995ca9cb` and the second at `7c0dade77597`.

**Zero-draft schedules no drafts and changes no reply.** On the 32 generation prompts at 16 tokens each, B drafted 479 tokens and accepted 1, and zero-draft drafted 0, one at a time and at concurrency 8. All 32 replies and saved token ids match B's.

**One request at a time, the saved hidden states are bit-identical to B's.** That held for all 32 prompts, and two fresh B servers also matched bit for bit.

**At concurrency 8 they differ slightly, and its probe scores stay within B's own range.** The 360 demo prompts were served at concurrency 8 with 16 tokens each, so decode steps share batches with prefills. B and zero-draft were each served three times, alternating, and scored against the offline reference.

| | Flips | Mean \|Δscore\| | Max \|Δscore\| |
|---|---|---|---|
| Rerun floor | 0 | 0.000208 | 0.00205 |
| B, 3 runs | 0 | 0.000255 to 0.000292 | 0.00187 to 0.00242 |
| Zero-draft, 3 runs | 0 | 0.000259 to 0.000274 | 0.00242 to 0.00333 |

Every run gets PASS from `hulder check`, and every run's shift exceeds the floor, B's included, as in check 1. Zero-draft passes the rule below. It has 0 flips and its mean |Δscore| stays within B's range.

**The rule was revised after seeing the data.** The first rule had zero-draft no worse than a single B run on flips, mean and max. In that first comparison one zero-draft run came out at 0.000286 against B's 0.000276, a fail by 3.9%. A second zero-draft run came out at 0.000272. The first rule had no allowance for run-to-run noise, so it was replaced by B's range over three runs (`zero_draft_verdict`). Under the first rule, the third zero-draft run here (0.000259) would pass and the second (0.000274) would depend on which B run it met.

**At concurrency 8, saved hidden states aren't deterministic across fresh servers, with or without the draft.** Per prompt, the last position's relative L2 difference is 0 for most rows and at most 0.0022 between any two of the six runs.

| Pair of runs | Rows bit-identical, of 360 |
|---|---|
| B against B | 197 to 222 |
| Zero-draft against zero-draft | 179 to 192 |
| B against zero-draft | 140 to 237 |

The 32-prompt check's two B servers matched bit for bit at concurrency 8, so it was too small to show this. The likely cause is that request arrival timing changes which prompts share a step from one server start to the next, the batch-composition effect in docs/design.md §5, which keeps capture at a fixed batch composition. That's not verified here. It means a served score moves between server starts by about as much as zero-draft moves it, and that the bundled rerun floor, captured at a fixed batch composition, understates serving's own noise.

## Where this falls short

- Overhead isn't measured. Throughput and latency on the 2070 aren't representative (docs/design.md §8.1).
- KV-cache capacity isn't comparable between server starts on this GPU. The plain server reported 60,128 tokens and the lens-disabled one 87,616 with the same flags. B's 53,776 is 10.6% below the plain server's, close to the 10.7% its hidden-state cache should cost for this model (3,072 bytes per token against 28,672 for KV), but the variation between starts is larger than that.
- vllm-lens's extra shift (0.00047, about twice the floor) comes with eager mode, one request at a time and the completions endpoint at once, so its cause isn't separated.
- One model, one GPU, 360 prompts and 32 generations.
- Zero-draft's throughput isn't measured here, and B's 40% isn't yet attributed to the draft token.
- Three runs per arm give a range, not an interval.
