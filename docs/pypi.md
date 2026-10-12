# Hulder

[![CI](https://github.com/revtheundead/hulder/actions/workflows/ci.yml/badge.svg)](https://github.com/revtheundead/hulder/actions/workflows/ci.yml)

A release gate for activation probes, and any other thresholded scorer, across LLM serving configs.

Teams tune a probe's threshold on one serving config, then change it: FP8 or AWQ weights, an FP8 KV cache, a vLLM upgrade. The hidden states move, and the threshold silently stops giving its target false-positive rate. `hulder check` compares a reference and a candidate on the same labelled examples and returns PASS, THRESHOLD STALE, SCORER DEGRADED or INCONCLUSIVE, with confidence intervals and a CI exit code.

## Install

```bash
pip install hulder
```

The base install is numpy and scipy only and runs on CPU. Capturing activations from vLLM needs the `vllm` extra (`pip install "hulder[vllm]"`).

## Quickstart

```bash
hulder quickstart
```

It runs two checks on bundled data with a demo probe, under a minute on CPU. An fp16 rerun as candidate gives PASS. The official AWQ int4 checkpoint of Qwen2.5-1.5B-Instruct gives INCONCLUSIVE, with the shift flagged against the rerun's noise floor. `hulder check --help` lists the options.

v0.1 is an early release. It has run only on a Turing GPU in fp16 and AWQ, and its tolerance bands are fixed.

## More

Design, statistical method, results and the cases where it loses: https://github.com/revtheundead/hulder

Apache-2.0. The bundled demo data is derived from a CC-BY-4.0 dataset, attributed in `hulder/data/NOTICE.md`.
