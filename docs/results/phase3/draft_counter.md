# The draft counter on an L4

**With async scheduling on, zero-draft scheduled no draft tokens and `hulder.runtime` stayed quiet. With `--no-async-scheduling`, vLLM drafted again with no warning of its own, and the runtime logged one warning and counted 894 drafts, the same number as vLLM's counter. The check passes under the rule fixed before the run.**

## Setup

- **Hardware:** Modal L4 (Ada, 24 GB, driver 610.57.04), one container.
- **Code and model:** commit `9d4f05b2987c`, Qwen2.5-1.5B-Instruct in fp16 at `989aa798`, with `experiments/phase3/d1_local.py`'s settings: chunked prefill on, prefix caching off, at most 8 sequences.
- **Probe:** the bundled demo probe (layer 20), loaded through `hulder.runtime.serve_args`.
- **Servers:** two fresh `vllm serve` processes, both on Model Runner V2. The first left async scheduling at its default, on, and the second added `--no-async-scheduling`.
- **Requests:** each server answered the same 32 demo prompts, 8 at a time, with 32-token greedy replies.

The rule, in `experiments/phase3/draft_counter.py`'s docstring before the run:
- **Async on** passes with no runtime warning, and both `hulder_draft_tokens_total` and `vllm:spec_decode_num_draft_tokens_total` at 0.
- **Async off** passes with exactly one runtime warning and both counters above 0, hulder's at least vLLM's.

## Results

| Server | Startup | Replies with a score | Scores on `/metrics` | `hulder_draft_tokens` | vLLM's draft count | Runtime warnings | Passes |
|---|---|---|---|---|---|---|---|
| Async on | 206 s | 32 of 32 | 32 | 0 | 0 | 0 | Yes |
| Async off | 94 s | 32 of 32 | 32 | 894 | 894 | 1 | Yes |

The warning came from the engine core process as the first drafted step was scheduled. That step had 1 draft. The message's wording was tidied after the run, from "draft tokens were scheduled" to "draft token(s) scheduled", with no other change. Scoring was unaffected: every reply on both servers carried a score.

The two counters agree exactly here. vLLM leaves out drafts whose step produced no token, so hulder's can run higher, and the rule allows that.

## Not tested

- **Data parallelism,** the other known fallback. It's read from vLLM's code (`config/vllm.py:991-1008`), and one GPU can't run it.
- **Model Runner V1.**
- **Other models and GPUs.**

## Cost

The manifest estimates $0.57, from 1,652 s of wall time. That's above the $0.31 worst case the run was guarded against, because the wall time includes about 20 minutes waiting for a GPU before the container started. The server logs start at 15:49 UTC for a call made at 15:25. The container's own time was under the 15-minute call timeout. The actual charge is on Modal's usage page.

## Reproduce

```bash
source scripts/dev-env.sh
uv run --with modal==1.6.1 modal run experiments/phase3/draft_counter_modal.py --spent-usd 0.0
```

Records are in [`modal/20261110T152540Z/`](modal/20261110T152540Z/). The server logs are kept with the run, not here.
