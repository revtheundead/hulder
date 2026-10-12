"""Activation capture from vLLM, the hulder[vllm] extra. Public but experimental
in v0.1: these names may change in any 0.x release.

capture, CaptureConfig and Capture import vLLM and torch, so they load on first use. The
other names, and importing this package, need neither.
"""

from hulder.capture.base_model import base_model_for
from hulder.capture.metadata import snapshot_metadata

__all__ = ["Capture", "CaptureConfig", "base_model_for", "capture", "snapshot_metadata"]

_NEEDS_VLLM = ("Capture", "CaptureConfig", "capture")


def __getattr__(name: str):
    if name in _NEEDS_VLLM:
        from hulder.capture import hidden_states

        return getattr(hidden_states, name)
    raise AttributeError(f"module 'hulder.capture' has no attribute {name!r}")
