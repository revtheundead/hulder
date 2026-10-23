# The full capture: three replicas, and deviation within a batch

The training and calibration activations for the Qwen3 probe, captured on 2026-10-22 as three replicas on three L4s, with the two runs that shaped the design: the timing pilot (2026-10-17) and canary_repeat (2026-10-19). Every run used an NVIDIA L4 on Modal with vLLM 0.30.0, Qwen3-4B at `1cfa9a72` in bf16 with `enable_thinking=False`, layers 18 and 27 at positions −1, −9 and −10, batch 8, prefix caching and chunked prefill off, and the path canary of [`../phase0/modal_smoke.md`](../phase0/modal_smoke.md#the-path-canary) as batch 1. There is no probe yet, so everything here compares activations bit for bit.

Run manifests and reports are in [`modal/`](modal/), one directory per run. Snapshots are not committed.

## Reproduce

```bash
source scripts/dev-env.sh
uv run --with modal==1.6.1 modal run experiments/modal/capture.py --experiment <name> --spent-usd <spent today> [--backup-dir <dir>]
uv run python experiments/modal/pilot_report.py .cache/modal-runs/<pilot run id>
uv run python experiments/modal/canary_repeat_report.py .cache/modal-runs/<run id> .cache/modal-runs/20261017T145226Z
uv run python experiments/modal/full_capture_report.py .cache/modal-runs/<run id> .cache/modal-runs/20261017T145226Z
```

`full_capture_report.py` writes the kept dataset, `kept/`, and refuses a run that already has one unless given `--force`.

| Run | Experiment | Commit | What ran | Estimated cost |
|---|---|---|---|---|
| `20261017T145226Z` | pilot | `3309187` | One container: `canary_check` (the 64 smoke prompts) in one engine, then 1,000 S-Labs training prompts in a second | $0.04 |
| `20261019T112159Z` | canary_repeat | `0688b6b` | `canary_check` in 4 single-use containers, one engine each, 1 with cuBLASLt tracing | $0.12 |
| `20261022T110135Z` | full_capture | `50fe051` | Stopped before capturing, below | $0.04 |
| `20261022T125644Z` | full_capture | `84c25c5` | All 16,332 prompts in 3 single-use containers, `r1` to `r3`, one engine each | $0.54 |

The estimates charge list rates for the full CPU and memory requested over container time measured locally, from the shared spawn time, so they include Modal's startup. The full capture's three replicas came to $0.154, $0.182 and $0.202. It was expected at $0.30 ($0.25 to $0.42), mostly because the forecast counted only the workers, which captured for 252 to 332 s. The pilot projected 224 s.

## The pilot left the path on one batch

`canary_check` repeats the 64 prompts every Phase 0 capture ran, so its batches can be compared bit for bit with the 16 recorded captures on the majority path, cluster 2.0. Its canary hash matched cluster 2.0 and so did batches 2 to 7. Batch 8 differed on all 8 of its prompts, at both layers, by 1.1% to 1.7% relative L2, and its rows match no row of any recorded capture, the batch-invariant and FP8 ones included.

The differences are smooth, with no NaNs, infinities or blocks of far-off dimensions, the largest sit in the large-magnitude dimensions, and per token they correlate with the minority path's own differences at 0.29 to 0.58. That reads as a different reduction order on that batch rather than corruption, which nothing has tested.

So a canary on batch 1 can't label a whole capture. Its hash matched while a later batch left the path.

## canary_repeat: batch 8 didn't recur

Four single-use containers ran `canary_check` again, three of them in exactly the pilot's engine environment. All four matched cluster 2.0 on every batch, batch 1 by its canary hash and batches 2 to 8 bit for bit, and the traced container chose the same 99 cuBLASLt signatures as all 8 traced majority-path captures. They ran on four GPUs, none of them the pilot's, all on driver 610.57.04, three in us-east and one in ap-south.

That left the pilot's batch 8 as 1 in 35 non-canary batches compared bit for bit, 95% Clopper-Pearson [0.07%, 14.9%].

## The full capture

**Why three replicas.** One engine can't be trusted to stay on the path for 2,042 batches, and two that disagree on a batch can't say which is right. With three, each batch is kept from its majority. The report compares the replicas batch by batch on all 6 arrays together and classes each batch as agree, a 2-to-1 split or no majority. It writes `kept/` only if no stop rule fires: R0, a replica failed or its files didn't verify; R1, the replicas weren't fed the same prompts, tokens and metadata; R2, more than 1% of batches have no majority; R3, the majority canary isn't cluster 2.0.

**The first attempt stopped on R0.** All three replicas refused the request after about 40 s, because the canary's 8 prompts belong to boundary-pairs' training split, so the request held them away from the front, which the runner refuses. The request now begins with the canary, so batch 1 is saved and classed like any batch, and a test passes every experiment's request through the same check.

**Result.**

| | |
|---|---|
| Stop rules | R0 to R3 passed |
| Batches | 2,042, every one unanimous |
| 2-to-1 splits | 0, 95% Clopper-Pearson [0, 0.18%] |
| Outliers per replica | 0, 0, 0 |
| Canaries | all three cluster 2.0, `b9dfd9d5ad63…` |
| Kept | 16,332 prompts (8,957 benign, 7,375 attack), none dropped, bit-identical to `r1` |
| GPUs | `d53bf06a`, `4f34788d`, `e615a974`, driver 610.57.04, in us-east, eu-west and us-east |
| Tokens | prompts `cd3a70fc…`, canary `1ae044fb…` (canary_repeat's) |

`kept/` holds all 6 arrays and `token_ids.npz`. It is the dataset for probe training and calibration.

**The independence guide was wrong.** Planned from 1 deviation in 35 batches, and treating each replica as leaving the path independently, a 2-to-1 split was expected on 8.1% of batches, about 165. None came. At 1 in 35 that has a probability of about e⁻¹⁷². Under the same independence, each of these three replicas left the path on at most about 0.06% of batches.

**Batch 106 differs from the pilot.** The pilot's saved batches are the full capture's batches 2 to 126, and 124 of those 125 are bit-identical. Batch 106 differs on one prompt only, `slabs/train/682` (benign), on all 6 arrays, by 0.79% to 1.32% relative L2 per array, with the largest single difference 0.125 to 0.375. The batch's other 7 prompts are identical. The three replicas agree with each other there, so the pilot is the likely outlier. Whether the difference lies along the direction between the two Phase 0 paths is unchecked.

## How deviation shows up

| Deviation | Run | Scope | Size (relative L2) | GPU |
|---|---|---|---|---|
| Cluster 2, 2 captures | 2026-10-13 smoke and fp8_online | every prompt of the capture | median 1.3% (layer 18), 1.1% (layer 27) | 2 GPUs, 2 drivers |
| Batch 8 | pilot `canary_check` | all 8 prompts of one batch | 1.1% to 1.7% | `0c4c4bb9` |
| Batch 106 | pilot `pilot` | 1 prompt of one batch | 0.8% to 1.3% | `0c4c4bb9` |

The two cluster-2 captures switched path for the whole capture, and one container moved between the paths from one engine to the next, so that switch wasn't tied to hardware. The two deviations since are local, one batch and then a single prompt, while the engine stayed on the majority path on either side. Both are about the size of the gap between the Phase 0 paths, about 1% in activation space. So deviation can't be assumed to be a property of the capture, and checking it needs evidence for each batch, which three replicas give.

## GPU `0c4c4bb9` (an inference, not a tested cause)

Both local deviations came from the pilot's one container, on GPU `0c4c4bb9`, in two separate engine processes. Across the 7 other GPUs since, no deviation has been found.

| | Batches compared | Deviated |
|---|---|---|
| `0c4c4bb9` (pilot) | 7 in `canary_check`, 125 in `pilot` | 2 |
| canary_repeat's 4 GPUs | 28, against cluster 2.0 | 0 |
| The full capture's 3 GPUs | 2,041 non-canary batches each, against each other; 124 of them also against the pilot | 0 found |

That ties the deviations to that GPU or that container. It isn't a tested cause, for three reasons.

- **The grouping was chosen after seeing the deviations,** so no test on these counts would mean what it says, and none is reported.
- **The pilot differs in more than the GPU.** It ran two engines in one container where the others ran one, and its image had another hulder wheel. The driver, 610.57.04, was the same everywhere.
- **Nothing has run on `0c4c4bb9` again,** so the tie hasn't been tried a second time.

If the tie holds, it matters for the check below. Replicas on separate GPUs would then deviate mostly independently, which is what a majority needs.

## Replication as the check, and its limit

A per-batch majority over three replicas on separate GPUs catches a deviation in one replica. It can't catch a deviation that all three share, because a batch where all three agree is kept whatever path it took. Two replicas deviating alike show up as a split, but the kept rows are then the deviant pair's. The canary and R3 check only batch 1 against the recorded path, and the comparison with the pilot covers batches 2 to 126.

The 0 splits bound how often each replica deviated on its own, not how often they deviated together. Deviations tied to a GPU or container would make a shared one less likely across three GPUs, but that rests on the inference above.

## The pilot is timing-only

The pilot's snapshots are used for their timing projection and canary hashes, never as data. `canary_check` left the path on batch 8, and `pilot` differs from the full capture on batch 106. The same prompts are in the full capture's `kept/`.

## Open

- What a local deviation does to a probe's scores and decisions, once a Qwen3 probe is trained on `kept/`.
- Whether batch 106's difference lies along the direction between the two Phase 0 paths.
- Whether `0c4c4bb9`, or a second engine in one container, is what the deviations depend on.
- How often three replicas deviate alike, which three replicas can't measure.
