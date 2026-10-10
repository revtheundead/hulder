# Bundled quickstart data

`hulder quickstart` checks these files. They are for demonstration only. The demo probe is not a prompt-attack monitor (README §8).

| File | What it is |
|---|---|
| `probe_demo.npz` | The demo probe: logistic regression on Qwen2.5-1.5B-Instruct's layer-20 residual stream at the last prompt token, trained on 600 boundary-pairs examples. Threshold 0.395 for a 5% FPR target. |
| `demo_fp16.npz` | Reference snapshot: that layer's fp16 activations for the 360 boundary-pairs calibration examples (180 benign, 180 attack), from vLLM 0.30.0 on an RTX 2070. |
| `demo_fp16_run1.npz` | The same serving config run again in a fresh engine (the rerun). |
| `demo_awq.npz` | Candidate snapshot: the same examples under the official AWQ int4 checkpoint, `Qwen/Qwen2.5-1.5B-Instruct-AWQ`. |

They were produced by `experiments/phase1/probe/snapshots.py` (snapshot format 2) and `experiments/phase1/probe/train_demo.py`.

Contains data derived from 3nesdeniz/agentic-prompt-injection-boundary-pairs (Deniz 2026, doi:10.5281/zenodo.21378693), licensed under CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/). The files here are activations and a probe computed from that data, not the data itself. The activations come from Qwen2.5-1.5B-Instruct and its AWQ checkpoint, both Apache-2.0.
