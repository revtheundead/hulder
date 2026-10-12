# Contributing

Bug reports, questions and pull requests are welcome. For anything larger than a fix, please open an issue first so we can agree on the approach.

## Setup

Python comes from `uv` and stays inside the repository, as do model weights and compile caches. Source `scripts/dev-env.sh` before any `uv` command:

```bash
source scripts/dev-env.sh && uv python install 3.12.14 && uv sync
```

`uv sync` installs both extras, so it pulls in vLLM. CI uses the base install only:

```bash
source scripts/dev-env.sh && uv sync --no-default-groups --group test
```

## Checks

```bash
source scripts/dev-env.sh && uv run ruff check && uv run ruff format --check && uv run pytest
```

`uv run pytest` is the fast suite, CPU only. `--runslow` adds the gate's seeded calibration runs (about 9 minutes), and `--runvllm` adds the tests that import vLLM, which need the `vllm` extra and a GPU.

## Conventions

- Statistical code is tested on synthetic data with known ground truth before it touches real data (`tests/synthetic.py`).
- A design decision is written up in `docs/design.md`, with the alternatives it rejected. Terms follow `CONTEXT.md`.
- A result in `docs/results/` states the one command that reproduces it, with pinned versions and seeds.
- Tests that import vLLM are marked `vllm`, so CI on the base install skips them.
