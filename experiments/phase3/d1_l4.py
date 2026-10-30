"""D1's cost measurement (docs/design.md §5 D1, docs/results/phase3/d1_local.md): four arms served
in turn on one GPU, interleaved over rounds, each arm-run in a fresh `vllm serve`.

- baseline: no probe
- b_stock: vLLM's hidden-state connector, writing every prompt's hidden states to disk
- b_noop: the same connector with its write removed (no_write_connector.py)
- lens: vllm-lens 1.3.0 in eager mode, with a persistent hook scoring a linear probe

Each arm-run records the GPU's free memory before the server starts, the server's KV-cache size
and maximum concurrency, a greedy generation check, a saturation bench (throughput), a fixed-rate
bench at 70% of round 0's baseline saturation rate (TTFT and TPOT), the peak running requests and
KV-cache use during the benches, and the spec-decode counters.

The cost rule (verdicts): an arm passes when the worst round's paired overhead against that round's
baseline is at most 5% for saturation throughput and 10% for fixed-rate p99 TTFT and p99 TPOT. A
metric is inconclusive when the baseline's own range across rounds exceeds half its threshold.

Stop rules, checked after every arm-run: a server that doesn't start, a run past its timeout, a
bench with failed requests, or an arm whose greedy replies diverge from round 0's baseline at more
non-tie tokens than a later baseline does (generation_fails). Every arm except lens runs with
VLLM_LENS_DISABLE=1, which left greedy replies unchanged in the local check.

The L4 run goes through d1_modal.py. A smoke run of the whole path on the local GPU, with the
small model and a tiny workload, needs vllm-lens layered on:

    source scripts/dev-env.sh
    uv run --with vllm-lens==1.3.0 python experiments/phase3/d1_l4.py --smoke --out .cache/d1-l4-smoke
"""

import argparse
import contextlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import d1_local
from prometheus_client.parser import text_string_to_metric_families

HERE = Path(__file__).resolve().parent
ARMS = ("baseline", "b_stock", "b_noop", "lens")
BASELINE = "baseline"
THRESHOLDS = {"throughput": 0.05, "p99_ttft": 0.10, "p99_tpot": 0.10}
FIXED_RATE_FRACTION = 0.7
# A greedy divergence starting where the reference's top two candidates are this close is a tie
# broken by rounding, not a change. The top two logprobs differ by exactly the difference of their
# logits, so the gap moves in steps of the logits' rounding: 2^-6 at |logit| 16 to 32 in fp16, as
# measured locally, and 0.125 to 0.25 in bf16 at |logit| 16 to 64. So two bf16 steps at 16 to 32.
TIE_BOUND = 0.25
PORT = 8766


def schedule(arms: tuple[str, ...], rounds: int) -> list[tuple[int, str]]:
    """(round, arm) in run order: every arm once a round, the order rotated by one each round so no
    arm always follows another, and round 0 starting with the baseline."""
    return [(r, arms[(r + i) % len(arms)]) for r in range(rounds) for i in range(len(arms))]


def verdicts(runs: dict, baseline: str = BASELINE) -> dict:
    """The cost rule for every arm with a baseline in the same round. runs maps (round, arm) to
    {"saturation": bench result, "fixed": bench result}."""
    metrics = {
        "throughput": ("saturation", "request_throughput", lambda arm, base: 1 - arm / base),
        "p99_ttft": ("fixed", "p99_ttft_ms", lambda arm, base: arm / base - 1),
        "p99_tpot": ("fixed", "p99_tpot_ms", lambda arm, base: arm / base - 1),
    }
    rounds = sorted({r for r, a in runs if a == baseline})
    out = {}
    for arm in sorted({a for _, a in runs if a != baseline}):
        paired = [r for r in rounds if (r, arm) in runs]
        if not paired:
            continue
        result = {}
        for name, (bench, key, overhead) in metrics.items():
            base = [runs[(r, baseline)][bench][key] for r in rounds]
            spread = (max(base) - min(base)) / (sum(base) / len(base))
            overheads = [overhead(runs[(r, arm)][bench][key], runs[(r, baseline)][bench][key]) for r in paired]
            threshold = THRESHOLDS[name]
            verdict = "inconclusive" if spread > threshold / 2 else "fail" if max(overheads) > threshold else "pass"
            result[name] = {
                "overheads": overheads,
                "upper": max(overheads),
                "threshold": threshold,
                "baseline_spread": spread,
                "verdict": verdict,
            }
        settled = [m["verdict"] for m in result.values()]
        result["passes"] = None if "inconclusive" in settled else all(v == "pass" for v in settled)
        out[arm] = result
    return out


def generation_fails(arm_vs_plain: dict, plain_vs_plain: list[dict]) -> bool | None:
    """Whether an arm's greedy replies diverge from the baseline's at more non-tie tokens than a
    fresh baseline does (d1_local.compare_generations results). None until a baseline has been
    compared with another."""
    if not plain_vs_plain:
        return None

    def changes(comparison: dict) -> int:
        return sum(d["margins"][0] is None or d["margins"][0] > TIE_BOUND for d in comparison["diverged"])

    return changes(arm_vs_plain) > max(map(changes, plain_vs_plain))


def startup_facts(log: str) -> dict:
    """The KV-cache size in tokens and the maximum concurrency a server logged at startup."""
    kv = re.search(r"GPU KV cache size: ([\d,]+) tokens", log)
    concurrency = re.search(r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x", log)
    return {
        "kv_cache_tokens": int(kv.group(1).replace(",", "")) if kv else None,
        "max_concurrency": float(concurrency.group(1)) if concurrency else None,
    }


def free_gpu_memory() -> dict:
    line = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    used, free = (int(x) for x in line.splitlines()[0].split(","))
    return {"used_mib": used, "free_mib": free}


def arm_command(arm: str, cfg: dict, storage: Path) -> tuple[list[str], dict]:
    """The `vllm serve` command and extra environment for one arm."""
    command = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", cfg["model"], *cfg["serve"]]
    command += ["--port", str(PORT)]
    env = {} if arm == "lens" else {"VLLM_LENS_DISABLE": "1"}
    if arm in ("b_stock", "b_noop"):
        speculative = {
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": [cfg["layer"]]}},
        }
        transfer = {
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {"shared_storage_path": str(storage)},
        }
        if arm == "b_noop":
            transfer["kv_connector"] = "NoWriteHiddenStatesConnector"
            transfer["kv_connector_module_path"] = "no_write_connector"
            transfer["kv_connector_extra_config"]["use_synchronization_lock"] = False
            env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(HERE), os.environ.get("PYTHONPATH")]))
        command += ["--speculative-config", json.dumps(speculative), "--kv-transfer-config", json.dumps(transfer)]
    return command, env


@contextlib.contextmanager
def serve(command: list[str], env: dict, log_path: Path, deadline: float):
    """A server in its own process group, yielding its URL once /health answers, then killed."""
    import httpx

    log = log_path.open("w")
    log.write(" ".join(command) + "\n")
    log.flush()
    proc = subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **env}, start_new_session=True
    )
    url = f"http://127.0.0.1:{PORT}"
    try:
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"the server exited with {proc.returncode}")
            with contextlib.suppress(httpx.HTTPError):
                if httpx.get(f"{url}/health", timeout=2).status_code == 200:
                    break
            if time.time() > deadline:
                raise TimeoutError("the server didn't start before the run's timeout")
            time.sleep(2)
        yield url
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        log.close()


GAUGES = {"running": "vllm:num_requests_running", "kv_cache_usage": "vllm:kv_cache_usage_perc"}


class Peaks(threading.Thread):
    """Polls /metrics once a second for the peak running requests and KV-cache use."""

    def __init__(self, url: str):
        super().__init__(daemon=True)
        self.url, self.stop, self.peaks = url, threading.Event(), dict.fromkeys(GAUGES, 0.0)

    def run(self) -> None:
        import httpx

        while not self.stop.wait(1.0):
            with contextlib.suppress(httpx.HTTPError):
                for family in text_string_to_metric_families(httpx.get(f"{self.url}/metrics", timeout=2).text):
                    for sample in family.samples:
                        for key, name in GAUGES.items():
                            if sample.name == name:
                                self.peaks[key] = max(self.peaks[key], sample.value)


def bench(url: str, cfg: dict, workload: dict, dataset: Path, out: Path, name: str, rate: float | None, n: int,
          deadline: float) -> dict:  # fmt: skip
    """One `vllm bench serve` run of n prompts, at rate requests per second or all at once."""
    command = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
        "--backend", "openai-chat", "--endpoint", "/v1/chat/completions", "--base-url", url,
        "--model", cfg["model"], "--tokenizer", cfg["tokenizer"],
        "--dataset-name", "custom", "--dataset-path", str(dataset), "--skip-chat-template",
        "--custom-output-len", str(workload["output_len"]), "--ignore-eos", "--temperature", "0",
        "--num-prompts", str(n), "--request-rate", "inf" if rate is None else f"{rate:.3f}", "--seed", "0",
        "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,99",
        "--save-result", "--result-dir", str(out), "--result-filename", f"{name}.json",
    ]  # fmt: skip
    with (out / f"{name}.log").open("w") as log:
        subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=max(1, deadline - time.time())
        )
    result = json.loads((out / f"{name}.json").read_text())
    keep = ("duration", "completed", "failed", "request_throughput", "output_throughput", "total_token_throughput",
            "max_concurrent_requests", "mean_ttft_ms", "p50_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "p50_tpot_ms",
            "p99_tpot_ms", "p50_itl_ms", "p99_itl_ms", "p99_e2el_ms")  # fmt: skip
    return {k: result.get(k) for k in keep}


def make_probe_hook(weights, bias: float, layer: int):
    """A vllm-lens persistent hook that scores w·h + b at the last position of each request's
    first forward pass, and keeps the score on the GPU, so the hook adds no host sync."""
    from vllm_lens import Hook

    on_device = {}  # the weights, copied to the GPU once per process

    def score(ctx, h):
        if "score" not in ctx.saved:
            if (w := on_device.get(h.device)) is None:
                w = on_device[h.device] = weights.to(h.device, h.dtype)
            ctx.saved["score"] = h[-1] @ w + bias

    return Hook(fn=score, layer_indices=[layer])


def run_arm(arm: str, rnd: int, cfg: dict, workload: dict, out: Path, rate: float | None, timeout_s: float) -> dict:
    """One arm-run. rate is the fixed-rate bench's, None in round 0's baseline, which sets it."""
    import httpx

    deadline = time.time() + timeout_s
    run_dir = out / f"r{rnd}_{arm}"
    run_dir.mkdir(parents=True, exist_ok=True)
    storage = run_dir / "hidden_states"
    command, env = arm_command(arm, cfg, storage)
    result = {"arm": arm, "round": rnd, "free_gpu_before": free_gpu_memory(), "command": command, "env": env}
    started = time.time()
    try:
        with serve(command, env, run_dir / "server.log", deadline) as url:
            result["startup_seconds"] = round(time.time() - started, 1)
            if arm == "lens":
                import torch
                from vllm_lens.client import VLLMLensClient

                generator = torch.Generator().manual_seed(0)
                weights = torch.randn(cfg["hidden_size"], generator=generator) / math.sqrt(cfg["hidden_size"])
                # vllm-lens's layer index is one below hulder's layer id (d1_local.md, check 5).
                VLLMLensClient(url).register_hooks([make_probe_hook(weights, 0.0, cfg["layer"] - 1)])
            params = {"model": cfg["model"], "max_tokens": workload["gen_tokens"], "logprobs": True, "top_logprobs": 2}
            result["generate"] = d1_local.asyncio.run(d1_local.chat_all(url, workload["gen_prompts"], 1, **params))
            dataset = run_dir / "bench.jsonl"
            dataset.write_text("".join(json.dumps({"prompt": p}) + "\n" for p in workload["bench_prompts"]))
            before = httpx.get(f"{url}/metrics").text
            peaks = Peaks(url)
            peaks.start()
            n_sat = workload["saturation_prompts"]
            result["saturation"] = bench(url, cfg, workload, dataset, run_dir, "saturation", None, n_sat, deadline)
            rate = rate or FIXED_RATE_FRACTION * result["saturation"]["request_throughput"]
            n_fixed = min(len(workload["bench_prompts"]), max(1, round(rate * workload["fixed_seconds"])))
            result["fixed_rate"] = rate
            result["fixed"] = bench(url, cfg, workload, dataset, run_dir, "fixed", rate, n_fixed, deadline)
            peaks.stop.set()
            peaks.join()
            result["peaks"] = peaks.peaks
            result["counters"] = d1_local.step_tokens(before, httpx.get(f"{url}/metrics").text)
            if arm == "lens":
                result["hook_results"] = len(VLLMLensClient(url).collect_hook_results())
    except (RuntimeError, TimeoutError, subprocess.SubprocessError, httpx.HTTPError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        shutil.rmtree(storage, ignore_errors=True)
    result["seconds"] = round(time.time() - started, 1)
    result["startup"] = startup_facts((run_dir / "server.log").read_text(errors="replace"))
    return result


def run_all(cfg: dict, workload: dict, out: Path, rounds: int, timeout_s: float) -> dict:
    """Every arm-run of the schedule in order, stopping at the first stop rule that fires."""
    runs, plan = {}, schedule(ARMS, rounds)
    results = {"schedule": plan, "runs": [], "stopped": None}
    rate = None
    for rnd, arm in plan:
        print(f"round {rnd}: {arm}", flush=True)
        run = run_arm(arm, rnd, cfg, workload, out, rate, timeout_s)
        replies = run.pop("generate", None)
        results["runs"].append(run)
        if "error" in run:
            results["stopped"] = f"round {rnd} {arm}: {run['error']}"
            break
        if failed := run["saturation"]["failed"] or run["fixed"]["failed"]:
            results["stopped"] = f"round {rnd} {arm}: {failed} failed requests"
            break
        runs[(rnd, arm)] = {"saturation": run["saturation"], "fixed": run["fixed"], "replies": replies}
        if rnd == 0 and arm == BASELINE:
            rate = run["fixed_rate"]
        reference = runs[(0, BASELINE)]["replies"]
        comparisons = {
            key: d1_local.compare_generations(reference, value["replies"])
            for key, value in runs.items()
            if key != (0, BASELINE)
        }
        plain = [c for (r, a), c in comparisons.items() if a == BASELINE]
        results["generation"] = {f"r{r}_{a}": c for (r, a), c in comparisons.items()}
        fails = [f"r{r}_{a}" for (r, a), c in comparisons.items() if a != BASELINE and generation_fails(c, plain)]
        if fails:
            results["stopped"] = f"greedy replies changed beyond a fresh baseline's: {', '.join(fails)}"
            break
    results["verdicts"] = verdicts(runs)
    return results


SMOKE = {
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "tokenizer": "Qwen/Qwen2.5-1.5B-Instruct",
    "hidden_size": 1536,
    "layer": 20,
    "serve": [
        "--revision",
        "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "--dtype",
        "float16",
        "--gpu-memory-utilization",
        "0.7",
        "--max-model-len",
        "1024",
        "--max-num-seqs",
        "8",
        "--enable-chunked-prefill",
        "--no-enable-prefix-caching",
        "--seed",
        "0",
    ],
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--smoke", action="store_true", help="the whole path on the local GPU, tiny workload")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.smoke:
        raise SystemExit("the L4 run goes through d1_modal.py; here only --smoke runs")
    texts = [p["text"] for p in d1_local.prompts()]
    workload = {
        "gen_prompts": texts[:4],
        "gen_tokens": 16,
        "bench_prompts": texts,
        "saturation_prompts": 16,
        "fixed_seconds": 5,
        "output_len": 16,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    results = run_all(SMOKE, workload, args.out, rounds=1, timeout_s=900)
    (args.out / "d1_l4.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    print(json.dumps({k: results[k] for k in ("stopped", "verdicts")}, indent=2))
    return 0 if results["stopped"] is None else 1


if __name__ == "__main__":
    sys.exit(main())
