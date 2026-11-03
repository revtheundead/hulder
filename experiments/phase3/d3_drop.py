"""Why short replies lost saturation throughput in the overhead smoke (docs/results/phase3/overhead_l4.md).

The smoke's workload on the local GPU: Qwen2.5-1.5B-Instruct in fp16, 4 greedy replies, then saturation
benches of 16 requests of 16 tokens at --max-num-seqs 8. Three arms interleaved over 3 rounds, each a
fresh `vllm serve`: the baseline, the stock connector on zero-draft (b_zero_draft), and the scoring
connector (scoring). Each server runs the bench 3 times, so the first bench is the smoke's measurement
and the later two show whether it was warm-up.

The rule, fixed before the first run, on the first bench of each server, in req/s lost:
- Under a 10% mean drop from the baseline to scoring, nothing reproduced.
- A step accounts for the drop if it holds at least two thirds of the req/s lost over all rounds, and
  every round has more of its loss at that step than at the other. Otherwise, if the rounds disagree on
  which step is larger, that's reported instead. If not, it's both.

The stock connector still writes files, and the scoring one doesn't, so scoring can come out faster.

    source scripts/dev-env.sh && uv run python experiments/phase3/d3_drop.py --out .cache/d3-drop

Writes <out>/d3_drop.json. Throughput on this GPU isn't representative (docs/design.md §8.1), so this
says which step costs what on this workload, not what it costs on an L4.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import d1_l4
import d1_local

ARMS = ("baseline", "b_zero_draft", "scoring")
ROUNDS = 3
BENCHES = 3
REPRODUCED = 0.10
ACCOUNTS = 2 / 3


def attribute(rows: list[dict]) -> dict:
    """The rule above, on (round, baseline, b_zero_draft, scoring) saturation in req/s."""
    extraction = [r["baseline"] - r["b_zero_draft"] for r in rows]
    scoring = [r["b_zero_draft"] - r["scoring"] for r in rows]
    total = sum(r["baseline"] - r["scoring"] for r in rows)
    lost = max(sum(extraction), 0) + max(sum(scoring), 0)
    result = {
        "extraction_drops": [e / r["baseline"] for e, r in zip(extraction, rows, strict=True)],
        "scoring_drops": [s / r["b_zero_draft"] for s, r in zip(scoring, rows, strict=True)],
        "total_drops": [1 - r["scoring"] / r["baseline"] for r in rows],
        "mean_total_drop": total / sum(r["baseline"] for r in rows),
        "extraction_share": max(sum(extraction), 0) / lost if lost else 0,
        "scoring_share": max(sum(scoring), 0) / lost if lost else 0,
    }
    if result["mean_total_drop"] < REPRODUCED:
        finding = "no drop"
    elif any(e > s for e, s in zip(extraction, scoring, strict=True)) and any(
        s > e for e, s in zip(extraction, scoring, strict=True)
    ):
        finding = "rounds disagree"
    elif result["extraction_share"] >= ACCOUNTS:
        finding = "extraction"
    elif result["scoring_share"] >= ACCOUNTS:
        finding = "scoring"
    else:
        finding = "both"
    return {**result, "finding": finding}


def workload(texts: list[str]) -> dict:
    """d1_l4.py --smoke's workload."""
    return {
        "gen_prompts": texts[:4],
        "gen_tokens": 16,
        "bench_prompts": texts,
        "saturation_prompts": 16,
        "output_len": 16,
    }


def arm_run(arm: str, rnd: int, cfg: dict, load: dict, out: Path, deadline_s: float = 900) -> dict:
    """A fresh server, the smoke's greedy replies, then its bench BENCHES times with counters for each."""
    import httpx

    run_dir = out / f"r{rnd}_{arm}"
    run_dir.mkdir(parents=True, exist_ok=True)
    storage = run_dir / "hidden_states"
    command, env = d1_l4.arm_command(arm, cfg, storage)
    dataset = run_dir / "bench.jsonl"
    dataset.write_text("".join(json.dumps({"prompt": p}) + "\n" for p in load["bench_prompts"]))
    benches = []
    with d1_l4.serve(command, env, run_dir / "server.log", deadline_s) as url:
        params = {"model": cfg["model"], "max_tokens": load["gen_tokens"], "logprobs": True, "top_logprobs": 2}
        d1_local.asyncio.run(d1_local.chat_all(url, load["gen_prompts"], 1, **params))
        for i in range(BENCHES):
            before = httpx.get(f"{url}/metrics").text
            result = d1_l4.bench(
                url,
                cfg,
                load,
                dataset,
                run_dir,
                f"bench_{i}",
                None,
                load["saturation_prompts"],
                time.time() + deadline_s,
            )
            benches.append({**result, "counters": d1_local.step_tokens(before, httpx.get(f"{url}/metrics").text)})
    shutil.rmtree(storage, ignore_errors=True)
    startup = d1_l4.startup_facts((run_dir / "server.log").read_text(errors="replace"))
    if failed := sum(b["failed"] for b in benches):
        raise RuntimeError(f"round {rnd} {arm}: {failed} failed requests")
    if arm != "baseline" and any(b["counters"]["draft_tokens"] for b in benches):
        raise RuntimeError(f"round {rnd} {arm}: draft tokens on zero-draft")
    return {"arm": arm, "round": rnd, "startup": startup, "benches": benches}


def run(out: Path) -> dict:
    cfg = {**d1_l4.SMOKE, "probe_path": d1_l4.bundled_probe_path()}
    load = workload([p["text"] for p in d1_local.prompts()])
    runs = []
    for rnd, arm in d1_l4.schedule(ARMS, ROUNDS):
        print(f"round {rnd}: {arm}", flush=True)
        runs.append(arm_run(arm, rnd, cfg, load, out))
        print([round(b["request_throughput"], 2) for b in runs[-1]["benches"]], flush=True)

    def by_bench(index: int) -> list[dict]:
        return [
            {
                "round": rnd,
                **{r["arm"]: r["benches"][index]["request_throughput"] for r in runs if r["round"] == rnd},
            }
            for rnd in range(ROUNDS)
        ]

    return {"runs": runs, "attribution": {f"bench_{i}": attribute(by_bench(i)) for i in range(BENCHES)}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=d1_l4.HERE.parents[1] / ".cache" / "d3-drop")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    results = run(args.out)
    (args.out / "d3_drop.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    print(json.dumps({k: v["finding"] for k, v in results["attribution"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
