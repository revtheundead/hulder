"""The base install must not import vLLM, torch, scikit-learn or the capture extra's dependencies (docs/design.md §4)."""

import subprocess
import sys

HEAVY = ("vllm", "torch", "sklearn", "yaml", "huggingface_hub", "pandas")


def test_importing_hulder_and_running_the_quickstart_loads_no_heavy_dependency():
    code = (
        "import contextlib, io, sys\n"
        "import hulder, hulder.cli\n"
        "hulder.apply_probe\n"
        "with contextlib.redirect_stdout(io.StringIO()):\n"
        "    assert hulder.cli.run(['quickstart']) == 0\n"
        f"print(','.join(m for m in {HEAVY!r} if m in sys.modules))"
    )  # fmt: skip
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""
