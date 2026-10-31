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

Stop rules, checked after every arm-run: a server that doesn't answer /health within its startup
timeout, a baseline or B measurement past its own timeout, a bench with failed requests, or an arm whose greedy replies diverge from round 0's baseline at more
non-tie tokens than a later baseline does (generation_fails). Each prompt counts once, at its
first divergence, and every first divergence is reported with the baseline's margin there
(divergence_table). A lens measurement past its timeout is a lens result instead, recorded with
how far it got (measurement_timeout), and the remaining arm-runs go on. Every arm except lens runs with VLLM_LENS_DISABLE=1, which left greedy
replies unchanged in the local check.

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
from typing import NamedTuple

import d1_local

HERE = Path(__file__).resolve().parent
ARMS = ("baseline", "b_stock", "b_noop", "lens")
BASELINE = "baseline"
LENS = "lens"
THRESHOLDS = {"throughput": 0.05, "p99_ttft": 0.10, "p99_tpot": 0.10}
FIXED_RATE_FRACTION = 0.7
# A greedy divergence starting where the reference's top two candidates are this close is a tie
# broken by rounding, not a change. The top two logprobs differ by exactly the difference of their
# logits, so the gap moves in steps of the logits' rounding: 2^-6 at |logit| 16 to 32 in fp16, as
# measured locally, and 0.125 to 0.25 in bf16 at |logit| 16 to 64. So two bf16 steps at 16 to 32.
TIE_BOUND = 0.25
PORT = 8766
STOP_GRACE_S = 30  # from SIGTERM to SIGKILL when a server is stopped
# A fixed-rate bench keeping up has completed every request that arrived this long ago: about four
# 128-token replies at an eager 60 ms a token. One that hasn't did not sustain the rate.
SUSTAIN_GRACE_S = 30
UNSUSTAINED = "did not sustain the fixed rate"


def schedule(arms: tuple[str, ...], rounds: int) -> list[tuple[int, str]]:
    """(round, arm) in run order: every arm once a round, the order rotated by one each round so no
    arm always follows another, and round 0 starting with the baseline."""
    return [(r, arms[(r + i) % len(arms)]) for r in range(rounds) for i in range(len(arms))]


class Timeouts(NamedTuple):
    """An arm-run's timeouts in seconds. Startup runs from launch until /health answers, and
    measurement from then to the end of the last bench. Round 0 runs every arm first in the
    container, so its servers may compile cold; later rounds reuse the compile cache it wrote."""

    cold_startup: float
    warm_startup: float
    measure: float

    def startup(self, rnd: int) -> float:
        return self.cold_startup if rnd == 0 else self.warm_startup


def worst_case_seconds(timeouts: Timeouts, rounds: int) -> float:
    """The longest the schedule can run: every arm-run at both its timeouts, then stopped."""
    return sum(timeouts.startup(rnd) + timeouts.measure + STOP_GRACE_S for rnd, _ in schedule(ARMS, rounds))


def verdicts(runs: dict, baseline: str = BASELINE, measurement_timeouts: dict | None = None) -> dict:
    """The cost rule for every arm with a baseline in the same round. runs maps (round, arm) to
    {"saturation": bench result, "fixed": bench result}, and measurement_timeouts maps an arm to its
    rounds' measurement_timeout results. An arm that didn't sustain the fixed rate in any round
    fails, and one that ran out of measurement time can't pass."""
    measurement_timeouts = measurement_timeouts or {}
    metrics = {
        "throughput": ("saturation", "request_throughput", lambda arm, base: 1 - arm / base),
        "p99_ttft": ("fixed", "p99_ttft_ms", lambda arm, base: arm / base - 1),
        "p99_tpot": ("fixed", "p99_tpot_ms", lambda arm, base: arm / base - 1),
    }
    rounds = sorted({r for r, a in runs if a == baseline})
    out = {}
    for arm in sorted({a for _, a in runs if a != baseline} | set(measurement_timeouts)):
        paired = [r for r in rounds if (r, arm) in runs]
        timeouts = measurement_timeouts.get(arm, [])
        if not paired and not timeouts:
            continue
        result = {}
        for name, (bench, key, overhead) in metrics.items() if paired else ():
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
        passes = None if "inconclusive" in settled else all(v == "pass" for v in settled)
        if timeouts:
            result["measurement_timeouts"] = timeouts
            passes = False if any(t["outcome"] == UNSUSTAINED for t in timeouts) else passes and None
        result["passes"] = passes
        out[arm] = result
    return out


def divergence_class(reference_margin: float | None) -> str:
    """A first divergence by the reference's top-two margin there. None means a reply ended early."""
    if reference_margin == 0:
        return "exact tie"
    if reference_margin is not None and reference_margin <= TIE_BOUND:
        return "within bound"
    return "change"


def non_tie_divergences(comparison: dict) -> int:
    """Prompts whose first divergence isn't a tie. compare_generations records only each prompt's
    first divergence, since everything after it differs anyway, so a prompt counts once."""
    return sum(divergence_class(d["margins"][0]) == "change" for d in comparison["diverged"])


def generation_fails(arm_vs_plain: dict, plain_vs_plain: list[dict]) -> bool | None:
    """Whether an arm's greedy replies diverge from the baseline's at more non-tie tokens than a
    fresh baseline does (d1_local.compare_generations results). None until a baseline has been
    compared with another."""
    if not plain_vs_plain:
        return None
    return non_tie_divergences(arm_vs_plain) > max(map(non_tie_divergences, plain_vs_plain))


def divergence_table(generation: dict) -> dict:
    """Every run's first divergences from round 0's baseline, each with the reference's margin and
    class, and the counts per class, so it shows whether TIE_BOUND excused any ("within bound")."""
    classes = ("exact tie", "within bound", "change")
    rows = [
        {
            "run": run,
            "prompt": d["prompt"],
            "token": d["token"],
            "tokens": d["tokens"],
            "reference_margin": d["margins"][0],
            "arm_margin": d["margins"][1],
            "class": divergence_class(d["margins"][0]),
        }
        for run, comparison in sorted(generation.items())
        for d in comparison["diverged"]
    ]
    counts = {run: dict.fromkeys(classes, 0) for run in sorted(generation)}
    for row in rows:
        counts[row["run"]][row["class"]] += 1
    return {"tie_bound": TIE_BOUND, "rows": rows, "counts": counts}


def measurement_timeout(stage: str, stage_seconds: float, rate: float | None, bench_log: str | None) -> dict:
    """A measurement timeout as a result: the stage it hit, how long that stage had run, the bench's
    last progress (its own seconds and requests completed of its total, None outside a bench) and
    the outcome. A fixed-rate bench that had completed fewer requests than had arrived
    SUSTAIN_GRACE_S before did not sustain the rate."""
    progress = re.findall(r"(\d+)/(\d+) \[(\d+):(\d+)", bench_log or "")
    completed, of, minutes, seconds = (int(n) for n in progress[-1]) if progress else (None,) * 4
    bench_seconds = None if minutes is None else 60 * minutes + seconds
    behind = (
        stage == "fixed" and bench_seconds is not None and completed < min(of, rate * (bench_seconds - SUSTAIN_GRACE_S))
    )
    return {
        "stage": stage,
        "stage_seconds": stage_seconds,
        "bench_seconds": bench_seconds,
        "completed": completed,
        "of": of,
        "outcome": UNSUSTAINED if behind else f"ran out of measurement time in {stage}",
    }


def startup_facts(log: str) -> dict:
    """The KV-cache size in tokens and the maximum concurrency a server logged at startup."""
    kv = re.search(r"GPU KV cache size: ([\d,]+) tokens", log)
    concurrency = re.search(r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x", log)
    return {
        "kv_cache_tokens": int(kv.group(1).replace(",", "")) if kv else None,
        "max_concurrency": float(concurrency.group(1)) if concurrency else None,
    }


def installed_packages() -> str:
    """Every installed distribution as name==version, one per line, sorted, read from package
    metadata as worker.py reads it, so recording the image's packages needs no subprocess."""
    from importlib.metadata import distributions

    lines = {f"{d.metadata['Name']}=={d.version}" for d in distributions() if d.metadata["Name"]}
    return "\n".join(sorted(lines, key=str.lower)) + "\n"


def preflight(model: str, revision: str) -> dict:
    """Everything an arm-run needs except a GPU, checked on CPU: the package list, the tokenizer
    read offline from the HF cache, the imports of the no-op connector and vllm-lens, and the
    bench command. Raises RuntimeError on the first that fails."""
    import importlib

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    started = time.time()
    try:
        path = snapshot_download(model, revision=revision, local_files_only=True)
    except LocalEntryNotFoundError as error:
        raise RuntimeError(f"{model} at {revision[:12]} isn't in the HF cache: {error}") from error
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}], tokenize=False, add_generation_prompt=True
    )
    templated = tokenizer(text, add_special_tokens=False)["input_ids"]
    sys.path.insert(0, str(HERE))
    for module in ("no_write_connector", "vllm_lens", "vllm_lens.client", "prometheus_client.parser", "httpx"):
        try:
            importlib.import_module(module)
        except ImportError as error:
            raise RuntimeError(f"can't import {module}: {error}") from error
    command = [sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "serve", "--help"]
    bench_help = subprocess.run(command, capture_output=True, text=True, check=False)
    if bench_help.returncode != 0:
        raise RuntimeError(f"`vllm bench serve --help` exited {bench_help.returncode}: {bench_help.stderr[-500:]}")
    packages = installed_packages()
    return {
        "tokenizer_path": path,
        "tokenizer_vocab": len(tokenizer),
        "templated_tokens": len(templated),
        "packages": len(packages.splitlines()),
        "versions": {line.split("==")[0]: line.split("==")[1] for line in packages.splitlines()
                     if line.split("==")[0].lower() in ("vllm", "vllm-lens", "hulder", "torch", "transformers")},
        "seconds": round(time.time() - started, 1),
    }  # fmt: skip


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
def serve(command: list[str], env: dict, log_path: Path, startup_timeout_s: float):
    """A server in its own process group, yielding its URL once /health answers, then stopped."""
    import httpx

    log = log_path.open("w")
    log.write(" ".join(command) + "\n")
    log.flush()
    proc = subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **env}, start_new_session=True
    )
    url = f"http://127.0.0.1:{PORT}"
    deadline = time.time() + startup_timeout_s
    try:
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"the server exited with {proc.returncode}")
            with contextlib.suppress(httpx.HTTPError):
                if httpx.get(f"{url}/health", timeout=2).status_code == 200:
                    break
            if time.time() > deadline:
                raise TimeoutError(
                    f"the server didn't answer /health within its {startup_timeout_s:g} s startup timeout"
                )
            time.sleep(2)
        yield url
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=STOP_GRACE_S)
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
        from prometheus_client.parser import text_string_to_metric_families

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


def run_arm(arm: str, rnd: int, cfg: dict, workload: dict, out: Path, rate: float | None,
            timeouts: Timeouts) -> dict:  # fmt: skip
    """One arm-run. rate is the fixed-rate bench's, None in round 0's baseline, which sets it."""
    import httpx

    run_dir = out / f"r{rnd}_{arm}"
    run_dir.mkdir(parents=True, exist_ok=True)
    storage = run_dir / "hidden_states"
    command, env = arm_command(arm, cfg, storage)
    result = {"arm": arm, "round": rnd, "free_gpu_before": free_gpu_memory(), "command": command, "env": env}
    started = time.time()
    stage, peaks = None, None
    try:
        with serve(command, env, run_dir / "server.log", timeouts.startup(rnd)) as url:
            result["startup_seconds"] = round(time.time() - started, 1)
            deadline = time.time() + timeouts.measure
            stage, stage_started = "generate", time.time()
            if arm == "lens":
                import torch
                from vllm_lens.client import VLLMLensClient

                generator = torch.Generator().manual_seed(0)
                weights = torch.randn(cfg["hidden_size"], generator=generator) / math.sqrt(cfg["hidden_size"])
                # vllm-lens's layer index is one below hulder's layer id (d1_local.md, check 5).
                VLLMLensClient(url).register_hooks([make_probe_hook(weights, 0.0, cfg["layer"] - 1)])
            params = {"model": cfg["model"], "max_tokens": workload["gen_tokens"], "logprobs": True, "top_logprobs": 2}
            replies = d1_local.chat_all(url, workload["gen_prompts"], 1, **params)
            result["generate"] = d1_local.asyncio.run(d1_local.asyncio.wait_for(replies, deadline - time.time()))
            dataset = run_dir / "bench.jsonl"
            dataset.write_text("".join(json.dumps({"prompt": p}) + "\n" for p in workload["bench_prompts"]))
            before = httpx.get(f"{url}/metrics").text
            peaks = Peaks(url)
            peaks.start()
            stage, stage_started = "saturation", time.time()
            n_sat = workload["saturation_prompts"]
            result["saturation"] = bench(url, cfg, workload, dataset, run_dir, "saturation", None, n_sat, deadline)
            rate = rate or FIXED_RATE_FRACTION * result["saturation"]["request_throughput"]
            n_fixed = min(len(workload["bench_prompts"]), max(1, round(rate * workload["fixed_seconds"])))
            result["fixed_rate"] = rate
            stage, stage_started = "fixed", time.time()
            result["fixed"] = bench(url, cfg, workload, dataset, run_dir, "fixed", rate, n_fixed, deadline)
            result["counters"] = d1_local.step_tokens(before, httpx.get(f"{url}/metrics").text)
            if arm == "lens":
                result["hook_results"] = len(VLLMLensClient(url).collect_hook_results())
    except (RuntimeError, TimeoutError, subprocess.SubprocessError, httpx.HTTPError) as error:
        result["error"] = f"{type(error).__name__}: {str(error) or 'the measurement timeout passed'}"
        if stage and isinstance(error, (TimeoutError, subprocess.TimeoutExpired)):
            log = run_dir / f"{stage}.log"
            result["timeout"] = measurement_timeout(
                stage,
                round(time.time() - stage_started, 1),
                result.get("fixed_rate"),
                log.read_text(errors="replace") if log.exists() else None,
            )
    finally:
        if peaks:
            peaks.stop.set()
            peaks.join()
            result["peaks"] = peaks.peaks
        shutil.rmtree(storage, ignore_errors=True)
    result["seconds"] = round(time.time() - started, 1)
    result["startup"] = startup_facts((run_dir / "server.log").read_text(errors="replace"))
    return result


def run_all(cfg: dict, workload: dict, out: Path, rounds: int, timeouts: Timeouts) -> dict:
    """Every arm-run of the schedule in order, stopping at the first stop rule that fires."""
    runs, replies, lens_timeouts, plan = {}, {}, [], schedule(ARMS, rounds)
    results = {"schedule": plan, "runs": [], "stopped": None}
    rate = None
    for rnd, arm in plan:
        print(f"round {rnd}: {arm}", flush=True)
        run = run_arm(arm, rnd, cfg, workload, out, rate, timeouts)
        generated = run.pop("generate", None)
        results["runs"].append(run)
        # Lens runs eager, so it may not keep up with the baseline's fixed rate. That is its result.
        lens_timed_out = arm == LENS and "timeout" in run
        if "error" in run and not lens_timed_out:
            results["stopped"] = f"round {rnd} {arm}: {run['error']}"
            break
        if failed := sum(run[b]["failed"] for b in ("saturation", "fixed") if b in run):
            results["stopped"] = f"round {rnd} {arm}: {failed} failed requests"
            break
        if lens_timed_out:
            lens_timeouts.append({"round": rnd, **run["timeout"]})
        else:
            runs[(rnd, arm)] = {"saturation": run["saturation"], "fixed": run["fixed"]}
        if generated is not None:
            replies[(rnd, arm)] = generated
        if rnd == 0 and arm == BASELINE:
            rate = run["fixed_rate"]
        reference = replies[(0, BASELINE)]
        comparisons = {
            key: d1_local.compare_generations(reference, value)
            for key, value in replies.items()
            if key != (0, BASELINE)
        }
        plain = [c for (r, a), c in comparisons.items() if a == BASELINE]
        results["generation"] = {f"r{r}_{a}": c for (r, a), c in comparisons.items()}
        fails = [f"r{r}_{a}" for (r, a), c in comparisons.items() if a != BASELINE and generation_fails(c, plain)]
        if fails:
            results["stopped"] = f"greedy replies changed beyond a fresh baseline's: {', '.join(fails)}"
            break
    results["verdicts"] = verdicts(runs, measurement_timeouts={LENS: lens_timeouts} if lens_timeouts else None)
    results["divergences"] = divergence_table(results.get("generation", {}))
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
    results = run_all(SMOKE, workload, args.out, rounds=1, timeouts=Timeouts(900, 900, 900))
    (args.out / "d1_l4.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    print(json.dumps({k: results[k] for k in ("stopped", "verdicts", "divergences")}, indent=2))
    return 0 if results["stopped"] is None else 1


if __name__ == "__main__":
    sys.exit(main())
