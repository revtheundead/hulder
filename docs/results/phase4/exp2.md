# Experiment 2: bf16 against online FP8 on Qwen3-4B

**PASS: FP8 keeps flag rates but changes which prompts are flagged (17 flips against a noise floor of 4).**

The first gate comparison on Qwen3-4B. The reference is bf16, and the candidate is vLLM's online per-tensor FP8 of the same weights, which runs as W8A8 (CUTLASS) on the L4 (docs/design.md §7). Both run Qwen3-4B at `1cfa9a72` under vLLM 0.30.0 with `enable_thinking=False` on L4s. The verdict comes from the frozen primary probe, layer 18 at position −1 with threshold 0.8025 for a 1% FPR target (`probe_qwen3.md`). Layer 27 at −1 is reported beside it as a secondary probe. It was chosen after the fact, because it ranked boundary-pairs best. Full numbers, including each `hulder check` report in full, are in `exp2.json`.

## Reproduce

```bash
source scripts/dev-env.sh
uv run --with modal==1.6.1 modal run experiments/modal/capture.py --experiment exp2 --spent-usd 0.00 --backup-dir <backup dir>
uv run python experiments/phase4/probe/exp2_report.py .cache/modal-runs/20261026T114520Z
```

The capture ran at commit `fda7034` as run `20261026T114520Z`, and its manifest is in `modal/20261026T114520Z/`. The report ran at `544a50f` on CPU. It refuses a dirty tree, records its commit, and runs once per capture.

## The run

Five single-use containers each captured the canary, then the full capture's batches that hold the calibration split and all of boundary-pairs. That is 3,308 prompts in 414 batches, each batch exactly as in the full capture (`full_capture.md`).

| Container | Serving config | Role |
|---|---|---|
| `control` | bf16, the full capture's config | the reference |
| `shuffled` | bf16, prompts in a `default_rng(1)` order, the canary still batch 1 | the rerun, so the noise floor includes batch composition |
| `fp8_r1` to `fp8_r3` | online FP8 | the candidate, each batch kept from the three replicas' majority |

The run was estimated at $0.55, counted from the shared spawn time, which overstates later containers.

## The checks before the comparison

- **The control equals the full capture bit for bit** on all 6 arrays and all 414 batches, with the same prompts and labels. So slicing the request changed nothing, and the reference is `kept/`'s own rows.
- **The FP8 majority passes R1 to R3.** The replicas were fed the same tokens as each other and the control. 410 batches are unanimous, 4 split 2 to 1, and none lacks a majority. Their canary matches the one recorded online-FP8 capture, from 2026-10-13.
- **FP8 replicas split more often than bf16 replicas.** That's 4 of 414 batches, 0.97% [0.26%, 2.46%], against 0 of 2,042 for the full capture's bf16 replicas [0, 0.18%]. Fisher's exact test gives a two-sided p of 0.0008. The test is post hoc, chosen after seeing the splits, and the two ran on different slices of the request. All 4 splits are in the boundary-pairs slice, which was also noticed after the fact.
- **The shuffled rerun moves most rows.** Only 355 of 2,101 calibration rows are bit-identical to the control, since a prompt's activations depend on the other prompts in its batch.

## Calibration: flag rates and flips

On the calibration split, 1,191 benign and 910 attack, at each probe's frozen threshold. Rates have Clopper-Pearson intervals.

| | Primary, bf16 | Primary, FP8 | Secondary, bf16 | Secondary, FP8 |
|---|---|---|---|---|
| FPR | 0.92% (11) [0.46, 1.65] | 0.76% (9) [0.35, 1.43] | 0.92% (11) [0.46, 1.65] | 1.01% (12) [0.52, 1.75] |
| FNR | 16.2% (147) [13.8, 18.7] | 15.8% (144) [13.5, 18.4] | 5.2% (47) [3.8, 6.8] | 4.9% (45) [3.6, 6.6] |

| Against the reference, same prompts | Primary, FP8 | Primary, rerun | Secondary, FP8 | Secondary, rerun |
|---|---|---|---|---|
| Decisions flipped | **17** (0.81%) | 4 | **5** (0.24%) | 3 |
| Attacks newly flagged / newly missed | 9 / 6 | 2 / 1 | 3 / 1 | 0 / 2 |
| Benign newly flagged / unflagged | 0 / 2 | 1 / 0 | 1 / 0 | 1 / 0 |
| Mean \|Δlogit\| | 0.184 | 0.031 | 0.267 | 0.026 |
| Largest \|Δlogit\| | 0.765 | 0.194 | 1.148 | 0.148 |

- **The rates barely move.** The primary's FPR falls by 2 benign and its FNR by 3 attacks. The secondary's FPR rises by 1 and its FNR falls by 2. Every change is far inside its interval.
- **The decisions do move.** FP8 flips 17 of the primary's decisions against the rerun's 4. Its mean |Δscore| exceeds the rerun's by 0.0084 [0.0078, 0.0090], so the shift is real. 15 of the 17 are attacks, 9 newly caught and 6 newly missed, so the net change in missed attacks is −3.
- **A path switch alone could move the primary's FNR by up to 2.31 points** (`probe_qwen3.md`, experiment 1b). FP8's −0.33 points is inside that, but the three replicas agree, which argues against a deviation here.

**ΔAUROC, as context.** Primary −0.0002 [−0.0005, +0.0001], from 0.9954 to 0.9952. Secondary +0.0002 [−0.0000, +0.0004], from 0.9969 to 0.9971. Both probes sit near the ceiling of in-distribution AUROC, so it says little here.

**Verdicts.** Both are **PASS**, exit 0, with `quantization: None -> 'fp8'` the only change under test. Each check warns about four expected metadata differences:
- FP8's canary hash, since FP8 takes its own numerical path;
- the candidate's GPUs;
- the rerun's `batch_order_seed`;
- the rerun's GPU.

### What the gate can't see here

Decision flips are a diagnostic, never a verdict input (docs/design.md §6). So the gate passes a change that swaps 6 caught attacks for 9 others. The paired FNR condition planned in docs/design.md §11 compares missed attacks on the same attacks. If it bounds the net change, −3 of 910 here, it passes this run too. A one-sided bound on newly missed attacks, 6 of 910 against the rerun's 1, is the form that would flag it.

## Boundary-pairs, not counted toward the verdict

All 1,200 boundary-pairs prompts, 600 benign and 600 attack. The thresholds were set on S-Labs calibration, so the bf16 reference already flags most benign prompts here. Nothing in this section counts toward the verdict. AUROC uses a pair-clustered bootstrap (2,000 resamples, seed 0).

| | Primary, bf16 | Primary, FP8 | Secondary, bf16 | Secondary, FP8 |
|---|---|---|---|---|
| FPR | 88.3% (530) [85.5, 90.8] | 89.3% (536) [86.6, 91.7] | 92.3% (554) [89.9, 94.3] | 93.5% (561) [91.2, 95.3] |
| FNR | 1.8% (11) [0.9, 3.3] | 2.0% (12) [1.0, 3.5] | 0.0% (0) [0.0, 0.6] | 0.0% (0) [0.0, 0.6] |
| AUROC | 0.685 [0.657, 0.713] | 0.683 [0.655, 0.712] | 0.845 [0.824, 0.866] | 0.841 [0.820, 0.862] |
| Flips, FP8 against rerun | | 9 against 3 | | 7 against 0 |

- **`hulder check` says THRESHOLD STALE, exit 10, for both.** It also warns that the reference's FPR excludes the target, so the threshold wasn't calibrated on this snapshot. The verdict reflects that, not FP8. The proposed thresholds would miss 90.2% and 79.8% of attacks.
- **FP8 adds 6 and 7 benign flags,** a 1-point rise whose interval overlaps the reference's.
- **The secondary's ΔAUROC is −0.0044 [−0.0073, −0.0013].** It's the only ΔAUROC interval in this experiment that excludes zero, a small ranking loss on hard negatives, well inside the −0.02 band.

## Limits

- **In distribution only.** The PASS covers S-Labs calibration, where both probes are near the AUROC ceiling. On hard negatives the probes fail in bf16 already, whatever the serving config.
- **One rerun.** The noise floor comes from a single shuffled container without replicas, so a deviation in it would inflate the floor.
- **Shared deviations go unseen.** The FP8 majority can't see a deviation all three replicas share, and the control check covers only bf16.
- **One candidate.** AWQ int4, FP8 KV cache and a vLLM upgrade are still to run (docs/design.md §7).
