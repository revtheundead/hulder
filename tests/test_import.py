"""The base install must not import vLLM, torch or scikit-learn."""

import subprocess
import sys

HEAVY = ("vllm", "torch", "sklearn")


def test_importing_hulder_loads_no_heavy_dependency():
    code = (
        "import sys, hulder\n"
        "hulder.apply_probe\n"
        f"print(','.join(m for m in {HEAVY!r} if m in sys.modules))"
    )  # fmt: skip
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""
