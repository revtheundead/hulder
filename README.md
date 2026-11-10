# Hulder

[![CI](https://github.com/revtheundead/hulder/actions/workflows/ci.yml/badge.svg)](https://github.com/revtheundead/hulder/actions/workflows/ci.yml)

**A release gate for activation probes, and any other thresholded scorer, across LLM serving configs.**

Activation probes are small classifiers that read a served model's own hidden states, for example to flag prompt attacks, and send only flagged requests to an expensive LLM judge. Each probe's threshold is tuned on one serving config to give a target false-positive rate. Then the config changes to cut cost (FP8 or AWQ int4 weights, an FP8 KV cache, a vLLM upgrade), the hidden states move, and the threshold silently stops meaning what it did. If scores drift up, false positives flood the judge. If they drift down, attacks slip through unnoticed, because the monitor is what degraded.

`hulder check` catches this before the change ships. It compares a scorer on the config live now (the reference) and the change you want to ship (the candidate), over the same labelled examples, and returns PASS, THRESHOLD STALE, SCORER DEGRADED or INCONCLUSIVE with confidence intervals and a CI exit code.

## Install

```bash
pip install hulder
```

The base install is numpy and scipy and runs on CPU, on Python 3.12 or newer. Capturing activations from vLLM, or scoring inside `vllm serve`, needs `pip install "hulder[vllm]"`, which adds the experimental `hulder.capture` and `hulder.runtime`.

## Example

```bash
hulder quickstart
```

It runs two checks on bundled activations from Qwen2.5-1.5B-Instruct in a few seconds on CPU, with a demo probe trained on 600 boundary-pairs examples. The demo probe shows how the gate works and is not a prompt-attack monitor. Check 1 passes a second fp16 run as the candidate, so nothing changed but run-to-run noise:

```
=== Check 1: The fp16 rerun as candidate: nothing changed but run-to-run noise. Expect PASS.

Reference  demo_fp16.npz
Candidate  demo_fp16_run1.npz
Probe      probe_demo.npz
Threshold  0.3952 at a 5.0% target FPR; 180 benign, 180 attack

Changes under test: none found in the metadata

Verdict: PASS

Performance at threshold 0.3952   reference                 candidate
  FPR                             4.4% [1.9%, 8.6%]         4.4% [1.9%, 8.6%]
  FNR                             0.0% [0.0%, 2.0%]         0.0% [0.0%, 2.0%]
  AUROC                           0.9990 [0.9973, 1.0000]   0.9990 [0.9973, 1.0000]
  ΔAUROC (candidate - reference)                            +0.0000 [+0.0000, +0.0000]
  Bands: candidate FPR upper bound <= 10.0%, FNR upper bound <= reference FNR + 0.05, ΔAUROC lower bound >= -0.02

Shift (paired, same prompts)      candidate                 rerun
  decisions flipped               0 (0.0%)
  mean |Δscore|                   0.000208
  mean |Δlogit|                   0.00197
  exceeds the noise floor         not measured

Exit code 0 (PASS)
```

Check 2 swaps in the official AWQ int4 checkpoint. It flips 11 decisions and moves scores by 0.0515 on average against 0.000208 on the rerun, so the change is real, but with 180 benign examples the gate can't yet say whether the false-positive rate crossed its 10.0% tolerance. The verdict is INCONCLUSIVE, exit 12, and the report says what would settle it:

```text
Verdict: INCONCLUSIVE
  FPR 3.9% to 12.0% against the 10.0% line; at least 16 more benign prompts are needed, as fewer cannot settle it.
```

**Reading a report.** The brackets are 95% confidence intervals, and PASS needs the pessimistic end of each one inside the band printed under it.
- **THRESHOLD STALE** means scores moved but the ranking survived, so a new threshold fixes it, and the report proposes one.
- **SCORER DEGRADED** means the ranking itself is damaged and the probe needs retraining.
- **INCONCLUSIVE** means an interval straddles its line. The report gives one line per undecided condition, with its interval against its line. For FPR and FNR the line adds a floor, the fewest extra benign or attack prompts that could settle it. Fewer cannot, and more may not, so it is not an estimate of how many will. If PASS is impossible at the current size even with no errors, the report says so first, with the size at which it becomes possible.

PASS, THRESHOLD STALE, SCORER DEGRADED and INCONCLUSIVE exit 0, 10, 11 and 12, leaving 1 and 2 for crashes and usage errors. [Reading a report](docs/reading-a-report.md) walks through check 2 line by line.

## Your own serving change

Capture activations for a labelled calibration set under the reference twice, so the gate can measure the noise floor, and the candidate once. Save each run as a `hulder.Snapshot`, then:

```
hulder check --probe probe.npz --reference ref.npz --noise-floor ref_rerun.npz --candidate cand.npz
```

The [live tutorial](docs/tutorial-live.md) captures the demo with vLLM on a GPU and shows how to save your own snapshots. `hulder check --help` lists the options, and `hulder.check_snapshots` and `hulder.check` do the same from Python.

Any other scorer, such as an embedding similarity or a guard model's probability, can bring its own scores instead, one JSONL file per serving config with an `id`, `score` and `label` per line. The [JSONL example](examples/scores_jsonl/README.md) gives the format and a synthetic stale threshold to try.

## In-server scoring (experimental)

`hulder.runtime`, in the `vllm` extra, scores a probe inside `vllm serve` with CUDA graphs on, on vLLM 0.30.0. Each non-streamed reply carries `hulder_score`, and `/metrics` counts scores, flags and replies that went without one, with example Prometheus rules for drift alarms. It is monitor-only, so it can flag a request but not stop it. On Qwen3-4B on an L4 it passed the cost rule against a no-probe baseline on throughput, p99 TTFT and p99 TPOT. [In-server scoring](docs/runtime.md) gives the `vllm serve` command and the limits.

## What it has been tested on

Every result below had its rules and expected verdicts committed before it ran, and misses are published.
- **Faults of known size.** On real Qwen3-4B bf16 activations captured on Modal L4s, a probe's scores were given planted faults and checked over 1,000 resamples each. Every gross fault got the right verdict in at least 99.9% of resamples. An unchanged scorer was flagged in 2.7% to 2.8% of resamples on the calibration set, but 11.8% on the smaller test set, missing its 5% bar ([planted faults](docs/results/phase4/planted_faults.md)).
- **Real serving changes on Qwen3-4B.** Online FP8 passes, while changing which prompts are flagged (17 decisions against a rerun's 4, [Experiment 2](docs/results/phase4/exp2.md)). AWQ int4 is INCONCLUSIVE on calibration and THRESHOLD STALE on the test replication ([sweep](docs/results/phase4/sweep_results.md)).
- **Qwen2.5-1.5B on an RTX 2070,** in fp16 and AWQ int4: the v0.1 results and the quickstart ([phase 1](docs/results/phase1/)).
- **Estimates of how much data would settle INCONCLUSIVE.** Two estimators were tested on fresh synthetic data against bars fixed beforehand, and both missed ([first](docs/results/phase2/settle.md), [second](docs/results/phase2/settle_v2.md)). So the report states only the floor and the size check, which are exact.

## Limits

- **Tolerance bands are fixed.** The candidate's FPR must stay under twice the target, its FNR within 0.05 of the reference's, and ΔAUROC above -0.02.
- **Small FPR rises need large calibration sets.** With about 1,200 benign prompts, a rise from 1% to 1.5% or 2% is usually INCONCLUSIVE.
- **At about 180 prompts per class, FNR and ΔAUROC can settle the wrong way.** In the settle runs, extending small INCONCLUSIVE checks with fresh synthetic data settled the FNR condition on the wrong side 9% to 12% of the time and ΔAUROC 13% of the time. At 1,191 benign and 910 attacks every condition stayed at 0.4% or below.
- **A one-way shift inside tolerance isn't flagged.** A change can newly flag several percent of prompts while every rate stays within its band. Whether that should count as a stale threshold is an open question.
- **A deviation from the numerical path isn't detected.** A whole capture can take a different numerical path and pass for a shift.
- **The proposed threshold costs FNR.** The threshold THRESHOLD STALE proposes misses more attacks, in exchange for its FPR guarantee.

The full list is in [docs/design.md §10](docs/design.md#10-known-limits).

## More

- [Reading a report](docs/reading-a-report.md): check 2 of the quickstart, line by line
- [Live tutorial](docs/tutorial-live.md): the demo captured with vLLM on a GPU, and your own snapshots
- [In-server scoring](docs/runtime.md): `hulder.runtime`, its metrics, example drift alarms and limits
- [Design](docs/design.md): the problem and prior art, the public API, every design decision with the alternatives it rejected, the verdict rule, known limits and roadmap
- [Results](docs/results/): every measurement, with the command that reproduces it, including where the gate loses
- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)

Apache-2.0. The bundled demo data is derived from a CC-BY-4.0 dataset, attributed in `src/hulder/data/NOTICE.md`.
