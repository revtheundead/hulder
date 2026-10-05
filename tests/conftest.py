"""Slow tests (the gate's own calibration, README §7) run only with --runslow."""

import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="run tests marked slow")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: seeded calibration runs; enable with --runslow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="slow; enable with --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
