# D1 confirmation on the L4 (Phase 3)

**In one container, `b_stock` lost 38.5% of saturation throughput against a no-probe baseline, and zero-draft B lost 1.5%.** Those are means over four rounds, paired with the same round's baseline. In every round, `b_stock` was about 37.5% slower than zero-draft.

Two results fall short of a clean pass:

- **Zero-draft's cost verdict is inconclusive, not a pass.** Every overhead is under its threshold, but the baseline moved too much on two of the three metrics for the rule to call them (below).
- **The score rule came out "needs decision".** Zero-draft narrowly missed both pass conditions, and it was then judged equivalent to `b_stock` in practice (below). That's a judgement, not a pass under the rule.

Modal L4 (Ada, 24 GB, driver 610.57.04), 4 CPU, 32 GiB, vLLM 0.30.0, torch 2.13.0. Qwen3-4B at `1cfa9a72` in bf16, probe layer 18, `--max-model-len 2048`, chunked prefill on, prefix caching off, CUDA graphs on. All 12 arm-runs ran in one container, each a fresh `vllm serve`, on Model Runner V2 with async scheduling on. Round 0 is a calibration round, with 16 greedy replies of 64 tokens and a saturation bench of 300 prompts. Rounds 1 to 3 add a fixed-rate bench for 75 s at 8.75 req/s, 70% of the slowest arm's round 0 saturation, and a score stage on both B arms. Every bench reply is 128 tokens. Records are in [`modal/20261102T130854Z/`](modal/20261102T130854Z/), and the server and bench logs and each stage's activations stayed local.

Reproduce from the repository root, at `aa6fe4933d89`:

```bash
source scripts/dev-env.sh && uv run --with modal==1.6.1 modal run experiments/phase3/d1_modal.py --experiment confirmation --spent-usd 0.0
uv run python experiments/phase3/d1_scores_report.py .cache/modal-runs/<run id>
uv run python experiments/phase3/d1_rows_report.py .cache/modal-runs/<run id>
```

The two reports need `kept/` and the shuffled rerun locally.

## Cost

The run completed with 0 failed requests and no stop rule fired. The thresholds and stop rules were fixed before the run, and are those of `d1_zero_draft_l4.md`.

| Metric | Threshold | Baseline's range | `b_stock` by round | Verdict | Zero-draft by round | Verdict |
|---|---|---|---|---|---|---|
| Saturation throughput | 5% | 3.10% | 39.1, 37.4, 38.9, 38.4% | Fail | 2.56, −0.03, 1.93, 1.67% | **Inconclusive** |
| Fixed-rate p99 TTFT | 10% | 6.85% | 18.3, 25.9, 26.7% | Fail | −3.04, 2.47, 3.56% | **Inconclusive** |
| Fixed-rate p99 TPOT | 10% | 0.55% | 32.1, 33.1, 32.6% | Fail | 1.05, 1.63, 1.37% | Pass |

**Why inconclusive.** The rule gives no pass or fail below the threshold when the baseline's own range is more than half the threshold, and 3.10% and 6.85% are both over that line. Zero-draft's worst rounds are still under the thresholds, at 2.56% and 3.56%. The previous run had the baseline within 0.71% and 0.29% and passed all three (`d1_zero_draft_l4.md`). What moved the baseline here isn't known: round 0 saturated at 20.53 req/s against about 20 later, and round 1's p99 TTFT was 157.5 ms against about 147.5 ms in rounds 2 and 3.

| Round | Baseline req/s | `b_stock` req/s | Zero-draft req/s | Baseline p99 TTFT / TPOT (ms) | `b_stock` p99 TTFT / TPOT (ms) | Zero-draft p99 TTFT / TPOT (ms) |
|---|---|---|---|---|---|---|
| 0 | 20.53 | 12.50 | 20.00 | | | |
| 1 | 19.90 | 12.45 | 19.91 | 157.5 / 43.05 | 186.4 / 56.85 | 152.7 / 43.50 |
| 2 | 20.21 | 12.34 | 19.82 | 147.7 / 42.82 | 185.9 / 57.00 | 151.3 / 43.51 |
| 3 | 20.20 | 12.44 | 19.86 | 147.2 / 42.90 | 186.5 / 56.87 | 152.4 / 43.49 |

All three arms kept up with the fixed rate, so these are the first valid latencies for `b_stock`. Zero-draft's latency wasn't re-measured at the previous run's 13.25 req/s, so that figure stands only as that run's result.

## Decode counters

From each server's `/metrics`. The score stage's are counted apart from the benches'.

| Arm | Bench draft tokens, rounds 1 to 3 | Accepted | Score stage draft tokens, each | Accepted |
|---|---|---|---|---|
| `b_stock` | 121,374 / 121,372 / 121,369 | 38 / 40 / 43 | 31,507 | 8 |
| Zero-draft | 0 | 0 | 0 | 0 |
| Baseline | 0 | 0 | | |

Round 0's benches drafted 38,087 tokens on `b_stock`, with 14 accepted, and 0 on zero-draft.

## Scores

Each score stage ran the 2,101 calibration prompts through the server, in a `default_rng(round)` order at 64 in flight, and kept each prompt's layer 18 hidden state at position −1. They're scored against `kept/` with the frozen primary probe (`18_-1`) at threshold 0.8025. The floor is experiment 2's shuffled rerun against `kept/`: 4 flips, mean |Δscore| 0.00164, 355 rows bit-identical. It reproduced exactly.

| Stage | Flips | Mean \|Δscore\| | Mean \|Δlogit\| | Max \|Δscore\| | Rows bit-identical | `hulder check` | Past the rerun's floor, by its CI |
|---|---|---|---|---|---|---|---|
| `b_stock`, round 1 | 3 | 0.00176 | 0.0322 | 0.0304 | 193 | PASS | No |
| `b_stock`, round 2 | 3 | 0.00171 | 0.0314 | 0.0362 | 263 | PASS | No |
| `b_stock`, round 3 | 3 | 0.00186 | 0.0353 | 0.0288 | 0 | PASS | Yes |
| Zero-draft, round 1 | 3 | 0.00165 | 0.0297 | 0.0280 | 311 | PASS | No |
| Zero-draft, round 2 | 4 | 0.00191 | 0.0342 | 0.0296 | 158 | PASS | Yes |
| Zero-draft, round 3 | 2 | 0.00195 | 0.0356 | 0.0257 | 0 | PASS | Yes |

Against the reference's FPR of 0.9% and FNR of 16.2%, stages ranged over 0.9% to 1.0% and 16.0% to 16.3%, and AUROC stayed at 0.9954.

**The rule, fixed before the run.** Pass needs zero-draft's median stage mean |Δscore| at most `b_stock`'s highest, and its flips per stage at most `b_stock`'s maximum. Fail needs its lowest stage above `b_stock`'s highest. Anything else needs a decision.

- The median, 0.001906, is above `b_stock`'s highest, 0.001857, by 2.6%. So the first condition isn't met.
- The most flips, 4, are above `b_stock`'s 3. So the second isn't either.
- The lowest stage, 0.001649, is below `b_stock`'s highest, so it isn't a fail.

**The decision: zero-draft and `b_stock` are equivalent in practice.** The stages overlap, with zero-draft from 0.00165 to 0.00195 and `b_stock` from 0.00171 to 0.00186, so zero-draft holds both the lowest and the highest. Its flips stay within the floor's 4, and FPR, FNR and AUROC don't move beyond the 0.1-point spread above.

This is a judgement on three stages per arm, which give a range rather than an interval. **Neither arm matches `kept/`.** All six stages are above the floor's mean |Δscore|, though none exceeds its flips, and the floor itself is one rerun.

## Round 3's zero bit-identical rows

Both round 3 stages have no row bit-identical to `kept/`, against 158 to 311 in rounds 1 and 2. **The evidence points to those two servers being on a different numerical path, not to the arm or the arrival order.** What selects the path isn't known. This is a reading of the run after the fact, and it changes neither the rule nor its outcome.

**Round 3 rows match nothing outside round 3.** Rows bit-identical between captures, from [`rows.json`](modal/20261102T130854Z/rows.json):

| | `kept/` | Shuffled | r1 `b_stock` | r1 zero | r2 `b_stock` | r2 zero | r3 `b_stock` | r3 zero |
|---|---|---|---|---|---|---|---|---|
| `kept/` | | 355 | 193 | 311 | 263 | 158 | 0 | 0 |
| r1 `b_stock` | 193 | 110 | | 244 | 169 | 162 | 4 | 6 |
| r2 `b_stock` | 263 | 120 | 169 | 147 | | 248 | 2 | 0 |
| r3 `b_stock` | 0 | 2 | 4 | 3 | 2 | 0 | | 353 |
| r3 zero | 0 | 2 | 6 | 1 | 0 | 1 | 353 | |

**The difference comes before the score stage.** Each arm-run's greedy replies run first, one request at a time, so arrival order can't reach them. Both round 3 servers diverge at different places from the same arm's servers in rounds 1 and 2. Baseline servers matched their round 0's on 16 of 16 in every round.

| Arm | Servers | First divergences (prompt, token) |
|---|---|---|
| `b_stock` | Rounds 0, 1, 2 | (5, 35), (8, 63), (9, 48), (12, 6), (13, 59) |
| `b_stock` | Round 3 | (8, 30), (9, 48), (12, 61), (13, 59) |
| Zero-draft | Rounds 1, 2 | (1, 24), (5, 35), (8, 30), (9, 48) |
| Zero-draft | Rounds 0, 3 | (7, 36), (8, 38), (9, 48), (13, 11) |

So the different greedy divergence that `d1_zero_draft_l4.md` reported in its round 0 recurs here, at the same prompts and tokens. **Across the two runs, 4 of 12 extraction servers changed numerical path, against 0 of 6 baseline servers.** That's 3 of zero-draft's 8 and 1 of `b_stock`'s 4, read from these greedy patterns. It's an observation too small to test, so it doesn't say whether extraction makes a path change more likely. The earlier cost run (`d1_l4.md`) points the same way but dilutes it: its 6 `b_stock` and `b_noop` servers all kept the usual pattern and its 2 baseline servers compared matched, which makes 4 of 18 against 0 of 8.

**The size of the shift doesn't change, only how many rows have none.** Rows that aren't bit-identical move about the same in every stage, and the same as the floor's:

| Capture | Rows bit-identical | Mean \|Δscore\|, all rows | Mean \|Δscore\|, changed rows | Median, changed rows | Relative L2, changed rows |
|---|---|---|---|---|---|
| Shuffled rerun (the floor) | 355 | 0.00164 | 0.00197 | 0.00068 | 1.54% |
| `b_stock`, round 1 | 193 | 0.00176 | 0.00193 | 0.00067 | 1.50% |
| `b_stock`, round 2 | 263 | 0.00171 | 0.00195 | 0.00068 | 1.49% |
| `b_stock`, round 3 | 0 | 0.00186 | 0.00186 | 0.00065 | 1.48% |
| Zero-draft, round 1 | 311 | 0.00165 | 0.00194 | 0.00065 | 1.48% |
| Zero-draft, round 2 | 158 | 0.00191 | 0.00206 | 0.00070 | 1.49% |
| Zero-draft, round 3 | 0 | 0.00195 | 0.00195 | 0.00065 | 1.47% |

A row bit-identical to `kept/` adds exactly 0 to a stage's mean, so a stage's mean depends on how many of them it happens to have, as well as on how far its other rows moved. That's why round 3 reads high, and why `hulder check` calls 2 of the 3 stages past the rerun's floor round 3's. Round 2's zero-draft stage is the third, and the median the rule judged. It has few identical rows, and its changed rows also moved the most.

**This doesn't favour zero-draft.** On changed rows alone, zero-draft's stages average 0.00198 and `b_stock`'s 0.00191, about 3.5% higher. With three stages each, that's not distinguishable from the stages' own spread.

**The other checks found nothing.** Which rows match `kept/` looks random. Between zero-draft's rounds 1 and 2, 26 rows are identical to `kept/` in both, against about 23 expected by chance. They spread evenly across arrival order and only slightly favour shorter prompts. Each arm's server logs from round 3 match rounds 1 and 2 once timestamps, ids and numbers are removed, apart from the count of periodic throughput lines.

The path reading fits `full_capture.md`, where the same model left its usual path by about 1% in activation space, and one container moved between paths from one engine to the next. It's an inference, from greedy replies and bit comparisons. Nothing here observed the path directly. Round 3's servers weren't rerun, and round 0 has no score stage to compare.

## Greedy replies

Each arm-run's 16 greedy replies are compared with round 0's baseline, and classed by the baseline's top-two logprob margin at the first differing token.

| Arm-runs | Identical, each | Exact tie, total | Within bound, total | Change |
|---|---|---|---|---|
| `b_stock`, 4 | 11 or 12 of 16 | 8 | 11 | 0 |
| Zero-draft, 4 | 12 of 16 | 10 | 6 | 0 |
| Baseline, rounds 1 to 3 | 16 of 16 | 0 | 0 | 0 |

No divergence reaches a change, so the generation rule passes.

## Environment

**`python -m vllm.collect_env` failed on the L4 until uv's cache was turned off.** It exited 1 in the run and again alone on a second L4 ([`modal/20261102T192211Z/`](modal/20261102T192211Z/)), with an `AttributeError` at `collect_env.py:672` at 0.30.0. The venv has no pip, so it falls back to `uv pip list`, discards any error and crashes when that command fails. Putting the image's uv on `PATH` and setting `VIRTUAL_ENV` didn't fix it.

A CPU check in the same environment found the cause ([`modal/20261102T202134Z/`](modal/20261102T202134Z/)). uv exits 2 with "Could not acquire lock", because its cache is on the Modal Volume through `XDG_CACHE_HOME` and it can't persist a temporary file there. With `UV_NO_CACHE=1`, or a cache under `/tmp`, it lists 209 packages. With `UV_NO_CACHE=1` added, a third L4 exited 0 in 10.7 s ([`modal/20261102T202226Z/`](modal/20261102T202226Z/)). The failure came before the arm-runs and doesn't touch them.

```bash
source scripts/dev-env.sh && uv run --with modal==1.6.1 modal run experiments/phase3/d1_modal.py --spent-usd 0.0 --environment-only
```

**That report comes from a later container than the run's**, on the same image definition, so it describes the image and not the run's own container. It gives Debian 12, glibc 2.36, GCC 12.2, Python 3.12.14 under gVisor, torch 2.13.0+cu130 built for CUDA 13.0, numpy 2.3.5, transformers 5.18.0, triton 3.7.1, flashinfer-python 0.6.18.post1 and vLLM 0.30.0. Every library version it lists matches `packages.txt`. It sees 20 CPUs in a container asked for 4, with `OMP_NUM_THREADS` and `MKL_NUM_THREADS` at 4. It couldn't collect the driver or CUDA runtime version, so those come from `nvidia-smi`: NVIDIA L4, driver 610.57.04, 23034 MiB. The raw report stays local.

**`packages.txt` isn't exactly the venv's list.** It reads package metadata from everything on the Python path, and 13 names appear twice at different versions, `aiohttp` 3.12.7 and 3.14.3 among them. uv lists 209 packages in the venv, the file holds 226 lines for 213 names, and the run's pre-flight counted 238. Where the extras come from, and which version of a doubled name a server imported, wasn't recorded. No library in `collect_env`'s list is doubled.

The 2070 used for the local checks is on driver 621.01 (WDDM), CUDA UMD 13.5, 8192 MiB, read from `nvidia-smi` on Windows, since `collect_env` can't read the driver under WSL.

## Time and cost

| Stage | Seconds |
|---|---|
| Server start | 88 to 100 |
| Arm-run, round 0 | 169 to 181 |
| Arm-run, rounds 1 to 3, baseline | 280 to 281 |
| Arm-run, rounds 1 to 3, zero-draft | 312 to 314 |
| Arm-run, rounds 1 to 3, `b_stock` | 328 to 342 |
| Score stage, against a 120 s timeout | 35 to 55 |

**Estimated cost $2.33, from the call's 6,751 s of wall time.** It's an upper estimate. The 12 arm-runs took 3,307 s, and the first server started about 57 minutes after the call did. That wait looks like queueing for a GPU, which wouldn't be charged, but this run didn't record it separately. The same rate over the arm-runs alone gives about $1.14. The two later L4 calls support that reading. They returned after 1,952 s and 4,117 s of wall time despite a 180 s function timeout, so most of their time wasn't spent in the container. By wall time they estimate $0.67 and $1.42, and that timeout should hold each to about $0.18.

## Where this falls short

- Zero-draft's cost verdict is inconclusive on two of three metrics, because of the baseline's range (above).
- The score outcome is a judgement after "needs decision", not a pass under the rule, and neither arm matches `kept/`.
- Three stages per arm and four rounds of throughput give ranges, not intervals. The floor is one rerun.
- The path reading of round 3 is an inference, and what selects a path isn't known. A server landing on another path changes which rows match `kept/`, and a path switch can matter at the threshold (`../phase4/probe_qwen3.md`), though here it flipped no more decisions than the other stages did.
- Zero-draft's latency at the higher 13.25 req/s wasn't re-measured.
- One GPU type, one model, one sequence length and one output length, with 128-token replies at up to 256 running requests.
