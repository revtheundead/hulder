# Live demo: the quickstart's check, captured with vLLM

This example captures the 360 demo prompts with vLLM under fp16, an fp16 rerun and the official AWQ int4 checkpoint, then runs `hulder check` on them with the demo probe bundled in hulder. The walkthrough, what to expect and why your verdict can differ are in [`docs/tutorial-live.md`](../../docs/tutorial-live.md).

It runs from a clone of this repository, not from a pip install. It reads the demo probe and snapshots from `src/hulder/data/`, and the prompts live here.

## Run it

You need an NVIDIA GPU of compute capability 7.5 or newer with about 6 GB free, Linux or WSL2, and about 4.5 GB of model downloads. Clone the repository and run it from the root:

```bash
git clone https://github.com/revtheundead/hulder && cd hulder
source scripts/dev-env.sh && uv python install 3.12.14 && uv sync
uv run python examples/live_demo/live_demo.py
```

`uv sync` installs hulder from this checkout with its `vllm` extra. Results go to `hulder-demo/`, or `--out DIR`. `--force` recaptures snapshots already there.

## Files

| File | What it is |
|---|---|
| `live_demo.py` | The script. It uses only hulder's public API: `hulder.Snapshot`, the experimental `hulder.capture`, and the `hulder check` command. |
| `demo_prompts.jsonl` | The 360 calibration prompts, with their ids and labels, in the order of the bundled demo snapshots. Written by `experiments/phase1/probe/demo_prompts.py`. |
| `test_live_demo.py` | Its tests, with a fake capture in place of vLLM. They run in the repository's fast suite, `uv run pytest`. |

## Data and licence

`demo_prompts.jsonl` contains data from 3nesdeniz/agentic-prompt-injection-boundary-pairs (Deniz 2026, doi:10.5281/zenodo.21378693), licensed under CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/). It holds 360 of the dataset's prompts, unmodified, from revision a5682e75. The changes are the selection of rows, their order, and the prefix `bpairs/` on each id.
