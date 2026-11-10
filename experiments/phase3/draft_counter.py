"""The draft counter's check on a server (hulder.runtime, written decision-0010): with async scheduling on, zero-draft
schedules no draft tokens and the runtime stays quiet. With it off, vLLM 0.30.0 drafts again with no
warning of its own, and the runtime must warn once and count the drafts on /metrics.

Two fresh `vllm serve` processes in one container, Qwen2.5-1.5B-Instruct in fp16 with d1_local.py's
settings and the bundled demo probe through hulder.runtime.serve_args. The first leaves async
scheduling at its default, on, and the second adds --no-async-scheduling. Each answers the same 32
demo prompts, 8 at a time, with 32-token greedy replies, then /metrics is read once its score
count is in or SETTLE_SECONDS have passed.

The rule, fixed before the run:
- async on passes with no runtime warning in its log, hulder_draft_tokens_total at 0 and vLLM's
  spec_decode_num_draft_tokens_total at 0;
- async off passes with exactly one runtime warning, and both counters above 0 with hulder's at
  least vLLM's, since vLLM leaves out drafts whose step produced no token;
- the check passes when both arms pass, and fails otherwise.

Run on Modal through draft_counter_modal.py.
"""

import asyncio
import contextlib
import subprocess
import sys
import time
from pathlib import Path

import d1_local

ARMS = {"async_on": [], "async_off": ["--no-async-scheduling"]}
N_REQUESTS, CONCURRENCY, MAX_TOKENS = 32, 8, 32
SETTLE_SECONDS = 30
WARNING_MARK = "hulder.runtime: "
COUNTERS = {
    "hulder_draft_tokens": "hulder_draft_tokens_total",
    "vllm_draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "scores": "hulder_probe_score_count",
}


def counters(text: str) -> dict:
    """COUNTERS' totals in a /metrics scrape, summed over engines. Absent ones are 0."""
    from prometheus_client.parser import text_string_to_metric_families

    totals = dict.fromkeys(COUNTERS.values(), 0.0)
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name in totals:
                totals[sample.name] += sample.value
    return {key: round(totals[name]) for key, name in COUNTERS.items()}


def runtime_warnings(log: str) -> int:
    return sum(WARNING_MARK in line for line in log.splitlines())


def arm_passes(arm: str, result: dict) -> bool:
    hulder, vllm = result["hulder_draft_tokens"], result["vllm_draft_tokens"]
    if arm == "async_on":
        return result["warnings"] == 0 and hulder == 0 and vllm == 0
    return result["warnings"] == 1 and vllm > 0 and hulder >= vllm


def verdict(results: dict) -> dict:
    passed = {arm: arm_passes(arm, results[arm]) if "error" not in results[arm] else False for arm in ARMS}
    return {**passed, "overall": "pass" if all(passed.values()) else "fail"}


@contextlib.contextmanager
def server(name: str, out: Path, extra: list[str], startup_timeout_s: float):
    """A fresh `vllm serve` with d1_local's model and settings, yielding its URL once /health answers."""
    import httpx

    command = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", d1_local.MODEL, *d1_local.SERVE, *extra]
    command += ["--port", str(d1_local.PORT)]
    log = (out / f"{name}.log").open("w")
    log.write(" ".join(command) + "\n")
    log.flush()
    proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{d1_local.PORT}"
    started = time.monotonic()
    try:
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"{name} exited with {proc.returncode}")
            with contextlib.suppress(httpx.HTTPError):
                if httpx.get(f"{url}/health", timeout=2).status_code == 200:
                    break
            if time.monotonic() - started > startup_timeout_s:
                raise RuntimeError(f"{name} didn't answer /health in {startup_timeout_s:.0f} s")
            time.sleep(2)
        yield url, round(time.monotonic() - started, 1)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log.close()


def run_arm(name: str, flags: list[str], probe: Path, out: Path, startup_timeout_s: float) -> dict:
    import httpx

    from hulder.runtime import serve_args

    texts = [r["text"] for r in d1_local.prompts()][:N_REQUESTS]
    with server(name, out, [*serve_args(probe), *flags], startup_timeout_s) as (url, startup_s):
        before = counters(httpx.get(f"{url}/metrics").text)
        responses = asyncio.run(d1_local.chat_all(url, texts, CONCURRENCY, max_tokens=MAX_TOKENS))
        deadline = time.monotonic() + SETTLE_SECONDS
        while True:
            after = counters(httpx.get(f"{url}/metrics").text)
            if after["scores"] - before["scores"] >= N_REQUESTS or time.monotonic() > deadline:
                break
            time.sleep(1)
    with_score = sum((r.get("kv_transfer_params") or {}).get("hulder_score") is not None for r in responses)
    return {
        "startup_s": startup_s,
        "requests": len(responses),
        "replies_with_score": with_score,
        **{key: after[key] - before[key] for key in COUNTERS},
        "warnings": runtime_warnings((out / f"{name}.log").read_text(errors="replace")),
    }


def run(probe: Path, out: Path, startup_timeout_s: float) -> dict:
    """Both arms, each recorded even if the other fails, then the verdict."""
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, flags in ARMS.items():
        try:
            results[name] = run_arm(name, flags, probe, out, startup_timeout_s)
        except Exception as error:  # noqa: BLE001 - the other arm and the logs must still come back
            results[name] = {"error": f"{type(error).__name__}: {error}"}
    return {"arms": results, "verdict": verdict(results)}
