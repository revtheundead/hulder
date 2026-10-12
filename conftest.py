"""Opt-in test groups, skipped by default so the fast run stays fast and CPU-only:
slow tests (the gate's own calibration, README §7) run with --runslow, and tests that
import vLLM (the capture extra) with --runvllm."""

import pytest

GROUPS = {
    "slow": ("--runslow", "seeded calibration runs"),
    "vllm": ("--runvllm", "imports vLLM (the [vllm] extra); skips cleanly where it isn't installed"),
}


def pytest_addoption(parser):
    for marker, (flag, what) in GROUPS.items():
        parser.addoption(flag, action="store_true", help=f"run tests marked {marker}: {what}")


def pytest_configure(config):
    for marker, (flag, what) in GROUPS.items():
        config.addinivalue_line("markers", f"{marker}: {what}; enable with {flag}")


def pytest_collection_modifyitems(config, items):
    for marker, (flag, what) in GROUPS.items():
        if config.getoption(flag):
            continue
        skip = pytest.mark.skip(reason=f"{what}; enable with {flag}")
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)
