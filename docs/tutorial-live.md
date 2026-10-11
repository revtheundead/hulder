# Live tutorial: the demo check, captured with vLLM on your GPU

`hulder quickstart` checks activations that ship with the package. This tutorial captures the same examples yourself with vLLM, under the same three serving configs, and runs the same check. It shows the whole path a real release check takes, from serving config to verdict.

## Run it

```bash
pip install "hulder[vllm]"
python -m hulder.capture.demo
```

It needs an NVIDIA GPU of compute capability 7.5 or newer (Turing onwards, which vLLM's AWQ path requires) with about 6 GB free, Linux or WSL2, and about 4.5 GB of model downloads from Hugging Face. On an RTX 2070 with the models already downloaded it takes about 5 minutes. `--out DIR` picks where results go (default `hulder-demo/`). An interrupted run resumes from the snapshots already there. A snapshot is reused only if its config, vLLM version, GPU and prompts match this run, so anything else is refused, naming what differs, and `--force` recaptures it.

## What it does

1. **Three captures, one fresh vLLM engine each.** Each runs in its own subprocess, since one process can't reliably free an engine's GPU memory for the next.
   - `fp16` is the reference, `Qwen/Qwen2.5-1.5B-Instruct` in fp16.
   - `fp16_rerun` is the same serving config again. Comparing it with the reference measures the noise floor, the variation you get without changing anything.
   - `awq` is the candidate, the official AWQ int4 checkpoint `Qwen/Qwen2.5-1.5B-Instruct-AWQ`.

   Every capture reads layer 20 at the last prompt token for the 360 bundled boundary-pairs prompts (`hulder/data/demo_prompts.jsonl`), with vLLM's hidden-state connector and the settings of the offline extraction decision. Each is saved as a `Snapshot` with its metadata, including the GPU it ran on.
2. **One check.** It runs `hulder check` with the bundled demo probe, the AWQ capture as candidate and the rerun as noise floor, and prints the report.
3. **A record.** `check.txt`, `check.json` and `manifest.json` (configs, GPU, package versions, timings, prompt hash) are written next to the snapshots.

The command exits 0 whatever the verdict, since the point is to see it. Exit 1 means a capture or the check failed.

## What to expect

On the RTX 2070 it gives **INCONCLUSIVE**, the same as the quickstart. AWQ flipped 11 of 360 decisions where the rerun flipped none, and its scores moved about 230 times as much as the rerun's, so the change is real. The candidate's false-positive rate is 7.2% [3.9%, 12.0%] against a 10% band, so the gate can't yet say whether it crossed the line. The recorded run is in `docs/results/phase1/live_demo.md`.

## Why your verdict can differ from the bundled one

Your captures will not be bit-identical to the bundled snapshots, and on another GPU the verdict itself can change.

- **Batch composition.** vLLM's numbers depend slightly on which prompts share a batch. The bundled snapshots came from a capture of all 16,332 probe prompts in fixed batches, so these 360 had other batch-mates. On the same RTX 2070 that alone moves the activations by about 0.1% (fp16) to 0.5% (AWQ) in relative L2, with no decision changed.
- **Another GPU.** Different hardware runs different kernels, so every number moves. That is itself a serving change, which is why snapshots record `gpu` and a GPU difference between reference and candidate warns.
- **A verdict near a boundary.** The demo's AWQ candidate flags 13 of 180 benign examples. At 10 or fewer the FPR interval would sit under the 10% band and the verdict would be PASS, and at 27 or more it would be THRESHOLD STALE. Three fewer flags on your GPU would give PASS.

What should hold everywhere is the shape of the result. The rerun's shift is tiny, the AWQ shift is far beyond it, and the report says so.

## Next

Your own probe and serving configs follow the same steps. Capture each config with `hulder.capture.hidden_states.capture`, save a `Snapshot` per run (README §4), and run `hulder check`. `src/hulder/capture/demo.py` is a short worked example of all three.
