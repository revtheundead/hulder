"""Two follow-ups to the Qwen2.5-1.5B overhead run (docs/results/phase3/overhead_l4.md), on the local GPU
or, through d1_modal.py --greedy-lens, on an L4.

1. The greedy comparison. That run's scoring servers all diverged from its baseline at prompt 0, token
   39, with a raw margin of -2.6875, and repetition_penalty 1.1 is the likely reason. Each mode runs a
   baseline, a fresh baseline and `scoring` on the same 16 prompts and 64 tokens:
   - rp11_raw: the run's settings, the model's penalty and raw logprobs.
   - rp10_raw: repetition_penalty 1.0, so raw logprobs are what the token was chosen from.
   - rp11_processed: the model's penalty, served with --logprobs-mode processed_logprobs, so the
     margins come after the penalty.

2. Why vllm-lens's hook results came back empty. The L4's log has /v1/hooks/collect failing with a 500,
   a CUDA out-of-memory error in the API process. Each --gpu-memory-utilization here registers the probe
   hook, sends the same greedy prompts and calls /v1/hooks/collect directly. 0.87 is the most this GPU
   starts at, since vLLM sees 6.98 of its 8 GiB free with nothing running, and the L4 ran at 0.92. The
   last run adds the L4 server's traffic before its collect, 300 prompts at 128 tokens and at 16.

    source scripts/dev-env.sh
    uv run --with vllm-lens==1.3.0 python experiments/phase3/greedy_lens_local.py --out .cache/greedy-lens-local

The GPU has to be otherwise idle throughout. --lens-only repeats part 2 into an existing results file.

Writes <out>/greedy_lens_local.json. This is an RTX 2070 and the run was an L4, so a divergence that
doesn't recur here says nothing about whether it was a near-tie there.

L4_PLAN is the same check on an L4 at the run's memory fraction, 0.92: the two penalty-1.1 modes, and lens
with and without the run's traffic, sent all at once as its saturation benches did.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import NamedTuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import d1_l4
import d1_local

N_PROMPTS, N_TOKENS = 16, 64
VOLUME_PROMPTS, VOLUME_LENGTHS = 300, (128, 16)
STARTUP_S = 900
MODES = {
    "rp11_raw": {"params": {}, "flags": []},
    "rp10_raw": {"params": {"repetition_penalty": 1.0}, "flags": []},
    "rp11_processed": {"params": {}, "flags": ["--logprobs-mode", "processed_logprobs"]},
}


class Plan(NamedTuple):
    """Which greedy modes run at which memory fraction, and each lens server as (fraction, with traffic)."""

    modes: dict
    utilization: float
    lens: tuple[tuple[float, bool], ...]
    volume_concurrency: int


LOCAL_PLAN = Plan(MODES, 0.7, ((0.7, False), (0.87, False), (0.87, True)), 32)
L4_PLAN = Plan(
    {k: MODES[k] for k in ("rp11_raw", "rp11_processed")}, 0.92, ((0.92, False), (0.92, True)), VOLUME_PROMPTS
)
# The L4 run's divergence, to read in every mode that reaches it with the same tokens before it.
L4_PROMPT, L4_TOKEN = 0, 39


def step_at(reply: dict, token: int) -> dict | None:
    """A reply's token at this position with its top candidates, or None past its end."""
    steps = reply["choices"][0]["logprobs"]["content"]
    if token >= len(steps):
        return None
    step = steps[token]
    return {
        "token": step["token"],
        "top": [{"token": t["token"], "logprob": t["logprob"]} for t in step["top_logprobs"]],
    }


def parsed_body(text: str) -> dict:
    """A collect response's JSON. A failed collect can answer in plain text, which holds no results."""
    try:
        return json.loads(text)
    except ValueError:
        return {}


def prompts_texts(n: int = N_PROMPTS) -> list[str]:
    """The overhead run's greedy prompts: the first of S-Labs calibration, sorted by id."""
    import pandas as pd

    rows = pd.read_parquet(HERE.parents[1] / ".cache/probe/splits.parquet", columns=["id", "split", "text"])
    calibration = rows[rows["split"] == "calibration"].sort_values("id", kind="stable")
    return [str(t) for t in calibration["text"]][:n]


def local_cfg(extra_flags: list[str], utilization: float = 0.7) -> dict:
    """The overhead run's serve flags on this GPU, with the given memory fraction."""
    cfg = {k: v for k, v in d1_l4.OVERHEAD_15B_CFG.items() if k != "greedy_params"}
    cfg["serve"] = [*cfg["serve"], "--gpu-memory-utilization", str(utilization), *extra_flags]
    return cfg


def greedy_arm(arm: str, cfg: dict, params: dict, texts: list[str], out: Path) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    command, env = d1_l4.arm_command(arm, cfg, out / "hidden_states")
    request = {"model": cfg["model"], "max_tokens": N_TOKENS, "logprobs": True, "top_logprobs": 2, **params}
    with d1_l4.serve(command, {**env, "VLLM_LENS_DISABLE": "1"}, out / "server.log", STARTUP_S) as url:
        return d1_local.asyncio.run(d1_local.chat_all(url, texts, 1, **request))


def greedy_mode(name: str, mode: dict, texts: list[str], out: Path, utilization: float = 0.7) -> dict:
    cfg = local_cfg(mode["flags"], utilization)
    print(f"greedy {name}", flush=True)
    replies = {
        label: greedy_arm(arm, cfg, mode["params"], texts, out / name / label)
        for label, arm in (("baseline", "baseline"), ("fresh_baseline", "baseline"), ("scoring", "scoring"))
    }
    reference = replies["baseline"]
    comparisons = {k: d1_local.compare_generations(reference, replies[k]) for k in ("fresh_baseline", "scoring")}
    return {
        "comparisons": comparisons,
        "classes": {
            k: [d1_l4.divergence_class(d["margins"][0]) for d in c["diverged"]] for k, c in comparisons.items()
        },
        "l4_position": {k: step_at(replies[k][L4_PROMPT], L4_TOKEN) for k in replies},
        "tokens_before_l4_position": {
            k: [s["token"] for s in replies[k][L4_PROMPT]["choices"][0]["logprobs"]["content"][:L4_TOKEN]]
            for k in replies
        },
    }


def lens_collect(
    utilization: float, texts: list[str], out: Path, volume: list[str] | None = None, concurrency: int = 32
) -> dict:
    """The probe hook through a server at this memory fraction, and what /v1/hooks/collect answers."""
    import httpx
    import torch
    from vllm_lens.client import VLLMLensClient

    run_dir = out / f"lens_{utilization}{'_volume' if volume else ''}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"lens at {utilization}, {'with' if volume else 'without'} volume", flush=True)
    cfg = local_cfg([], utilization)
    command, env = d1_l4.arm_command("lens", cfg, run_dir / "hidden_states")
    try:
        with d1_l4.serve(command, env, run_dir / "server.log", STARTUP_S) as url:
            generator = torch.Generator().manual_seed(0)
            weights = torch.randn(cfg["hidden_size"], generator=generator) / math.sqrt(cfg["hidden_size"])
            VLLMLensClient(url).register_hooks([d1_l4.make_probe_hook(weights, 0.0, cfg["layer"] - 1)])
            d1_local.asyncio.run(d1_local.chat_all(url, texts, 1, model=cfg["model"], max_tokens=N_TOKENS))
            for length in VOLUME_LENGTHS if volume else ():
                d1_local.asyncio.run(d1_local.chat_all(url, volume, concurrency, model=cfg["model"], max_tokens=length))
            memory = d1_l4.free_gpu_memory()
            response = httpx.post(f"{url}/v1/hooks/collect", timeout=120)
            body = parsed_body(response.text)
            client_count = len(VLLMLensClient(url).collect_hook_results())
    except (RuntimeError, TimeoutError) as error:
        return {"utilization": utilization, "error": str(error)}
    log = (run_dir / "server.log").read_text(errors="replace")
    return {
        "utilization": utilization,
        "requests": len(texts) + (len(volume) * len(VOLUME_LENGTHS) if volume else 0),
        "free_mib_before_collect": memory["free_mib"],
        "status": response.status_code,
        "direct_results": len(body.get("results", {})),
        "body_keys": sorted(body),
        "client_results": client_count,
        "out_of_memory_in_log": "CUDA error: out of memory" in log,
    }


def run_greedy(plan: Plan, texts: list[str], out: Path) -> dict:
    return {
        "prompts": N_PROMPTS,
        "tokens": N_TOKENS,
        "gpu": d1_l4.free_gpu_memory(),
        "greedy": {name: greedy_mode(name, mode, texts, out, plan.utilization) for name, mode in plan.modes.items()},
    }


def run_lens(plan: Plan, texts: list[str], volume: list[str], out: Path) -> list[dict]:
    return [
        lens_collect(u, texts, out, volume if traffic else None, plan.volume_concurrency) for u, traffic in plan.lens
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lens-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    texts = prompts_texts()
    path = args.out / "greedy_lens_local.json"
    results = json.loads(path.read_text()) if args.lens_only else run_greedy(LOCAL_PLAN, texts, args.out)
    results["lens"] = run_lens(LOCAL_PLAN, texts, prompts_texts(VOLUME_PROMPTS), args.out)
    path.write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
