"""v0.1's public API is hulder.__all__, hulder.capture.__all__ (experimental) and the CLI
commands (docs/design.md §4). Changing either list means amending that section."""

import subprocess
import sys

import pytest

import hulder
import hulder.capture

PUBLIC = [
    "Estimate",
    "PairingError",
    "Probe",
    "Recalibration",
    "Report",
    "Snapshot",
    "Verdict",
    "apply_probe",
    "check",
    "check_snapshots",
    "prompt_ids_sha256",
]
CAPTURE = ["Capture", "CaptureConfig", "base_model_for", "capture", "snapshot_metadata"]


def test_hulder_all_is_the_public_api_recorded_in_adr_0005():
    assert sorted(hulder.__all__) == PUBLIC
    assert all(hasattr(hulder, name) for name in PUBLIC)


def test_hulder_capture_all_is_the_experimental_api_recorded_in_adr_0005():
    assert sorted(hulder.capture.__all__) == CAPTURE


def test_importing_hulder_capture_loads_no_vllm_or_torch():
    code = "import sys, hulder.capture; print(','.join(m for m in ('vllm', 'torch') if m in sys.modules))"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip() == ""


@pytest.mark.vllm
def test_the_capture_names_that_need_vllm_resolve_to_hidden_states():
    hidden_states = pytest.importorskip("hulder.capture.hidden_states")
    for name in ("Capture", "CaptureConfig", "capture"):
        assert getattr(hulder.capture, name) is getattr(hidden_states, name)
