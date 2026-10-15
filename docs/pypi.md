# Hulder

[![CI](https://github.com/revtheundead/hulder/actions/workflows/ci.yml/badge.svg)](https://github.com/revtheundead/hulder/actions/workflows/ci.yml)

A release gate for activation probes, and any other thresholded scorer, across LLM serving configs.

Teams tune a probe's threshold on one serving config, then change it: FP8 or AWQ weights, an FP8 KV cache, a vLLM upgrade. The hidden states move, and the threshold silently stops giving its target false-positive rate. `hulder check` compares a reference and a candidate on the same labelled examples and returns PASS, THRESHOLD STALE, SCORER DEGRADED or INCONCLUSIVE, with confidence intervals and a CI exit code.

## Install

```bash
pip install hulder
```

The base install is numpy and scipy and runs on CPU, on Python 3.12 or newer. Capturing activations from vLLM needs `pip install "hulder[vllm]"`, which adds the experimental `hulder.capture`.

## Quickstart

```bash
hulder quickstart
```

It runs two checks on bundled activations from Qwen2.5-1.5B-Instruct in a few seconds on CPU, with a demo probe that shows how the gate works and is not a prompt-attack monitor. A second fp16 run as the candidate gives PASS. The official AWQ int4 checkpoint flips 11 decisions and moves scores far beyond the rerun's noise floor, but with 180 benign examples the gate can't yet say whether the false-positive rate crossed its tolerance, so it gives INCONCLUSIVE. `hulder check --help` lists the options.

v0.1 has run only on an RTX 2070 (Turing) in fp16 and AWQ int4. Its tolerance bands are fixed, and the threshold THRESHOLD STALE proposes misses more attacks in exchange for its FPR guarantee.

## More

Reading a report, the design and statistical method, results and the cases where it loses: https://github.com/revtheundead/hulder

Apache-2.0. The bundled demo data is derived from a CC-BY-4.0 dataset, attributed in `hulder/data/NOTICE.md`.
