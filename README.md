# Hulder

[![CI](https://github.com/revtheundead/hulder/actions/workflows/ci.yml/badge.svg)](https://github.com/revtheundead/hulder/actions/workflows/ci.yml)

**A release gate for activation probes, and any other thresholded scorer, across LLM serving configs.**

Activation probes are small classifiers that read a served model's own hidden states, for example to flag prompt attacks, and send only flagged requests to an expensive LLM judge. Each probe's threshold is tuned on one serving config to give a target false-positive rate. Then the config changes to cut cost (FP8 or AWQ int4 weights, an FP8 KV cache, a vLLM upgrade), the hidden states move, and the threshold silently stops meaning what it did. If scores drift up, false positives flood the judge. If they drift down, attacks slip through unnoticed, because the monitor is what degraded.

`hulder check` catches this before the change ships. It compares a scorer on the config live now (the reference) and the change you want to ship (the candidate), over the same labelled examples, and returns PASS, THRESHOLD STALE, SCORER DEGRADED or INCONCLUSIVE with confidence intervals and a CI exit code.

## Install

```bash
pip install hulder
```

The base install is numpy and scipy and runs on CPU, on Python 3.12 or newer. Capturing activations from vLLM needs `pip install "hulder[vllm]"`, which adds the experimental `hulder.capture`.

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

Check 2 swaps in the official AWQ int4 checkpoint. It flips 11 decisions and moves scores by 0.0515 on average against 0.000208 on the rerun, so the change is real, but with 180 benign examples the gate can't yet say whether the false-positive rate crossed its 10.0% tolerance. The verdict is INCONCLUSIVE, exit 12, with an estimate of how many more examples would settle it.

**Reading a report.** The brackets are 95% confidence intervals, and PASS needs the pessimistic end of each one inside the band printed under it. THRESHOLD STALE means scores moved but the ranking survived, so a new threshold fixes it, and the report proposes one. SCORER DEGRADED means the ranking itself is damaged and the probe needs retraining. PASS, THRESHOLD STALE, SCORER DEGRADED and INCONCLUSIVE exit 0, 10, 11 and 12, leaving 1 and 2 for crashes and usage errors. [Reading a report](docs/reading-a-report.md) walks through check 2 line by line.

## Your own serving change

Capture activations for a labelled calibration set under the reference twice, so the gate can measure the noise floor, and the candidate once. Save each run as a `hulder.Snapshot`, then:

```
hulder check --probe probe.npz --reference ref.npz --noise-floor ref_rerun.npz --candidate cand.npz
```

The [live tutorial](docs/tutorial-live.md) captures the demo with vLLM on a GPU and shows how to save your own snapshots. `hulder check --help` lists the options, and `hulder.check_snapshots` and `hulder.check` do the same from Python.

Any other scorer, such as an embedding similarity or a guard model's probability, can bring its own scores instead, one JSONL file per serving config with an `id`, `score` and `label` per line. The [JSONL example](examples/scores_jsonl/README.md) gives the format and a synthetic stale threshold to try.

## Limits

v0.1 has run only on an RTX 2070 (Turing) in fp16 and AWQ int4, with Qwen2.5-1.5B-Instruct under vLLM 0.30.0. Its tolerance bands are fixed, the threshold THRESHOLD STALE proposes misses more attacks in exchange for its FPR guarantee, and small FPR rises need large calibration sets to judge. The full list is in [docs/design.md §10](docs/design.md#10-known-limits).

## More

- [Reading a report](docs/reading-a-report.md): check 2 of the quickstart, line by line
- [Live tutorial](docs/tutorial-live.md): the demo captured with vLLM on a GPU, and your own snapshots
- [Design](docs/design.md): the problem and prior art, the public API, every design decision with the alternatives it rejected, the verdict rule, known limits and roadmap
- [Results](docs/results/phase1/): every measurement behind v0.1, with the command that reproduces it, including where the gate loses
- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)

Apache-2.0. The bundled demo data is derived from a CC-BY-4.0 dataset, attributed in `src/hulder/data/NOTICE.md`.
