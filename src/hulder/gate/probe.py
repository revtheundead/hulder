"""Linear probes and the probe-application helper. Numpy and scipy only."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.special import expit

from hulder.gate.atomic import savez_atomically
from hulder.gate.npz import read_npz


@dataclass(frozen=True)
class Probe:
    """Logistic regression on standardised activations: sigmoid(w . (h - mean) / std + b)."""

    weights: np.ndarray
    bias: float
    mean: np.ndarray
    std: np.ndarray
    metadata: dict = field(default_factory=dict)

    FORMAT_VERSION = 1

    def save(self, path: str | Path) -> None:
        """Write an .npz of plain arrays plus JSON metadata. Never pickled, so safe to share.
        Written atomically, so an interrupted save leaves any previous file whole."""
        savez_atomically(
            path,
            weights=np.asarray(self.weights, dtype=np.float64),
            bias=np.float64(self.bias),
            mean=np.asarray(self.mean, dtype=np.float64),
            std=np.asarray(self.std, dtype=np.float64),
            metadata=np.array(json.dumps({"format_version": self.FORMAT_VERSION, **self.metadata})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Probe":
        arrays, metadata = read_npz(path, "probe", ("weights", "bias", "mean", "std"))
        if "format_version" not in metadata:
            raise ValueError(f"{path}: no format_version in its metadata, so it isn't a hulder probe")
        version = metadata.pop("format_version")
        if version != cls.FORMAT_VERSION:
            raise ValueError(f"{path}: probe file format {version}, this hulder reads {cls.FORMAT_VERSION}")
        # Optional, but a recorded one must be a number; its range is check's to judge (metadata rule 9).
        for key in ("threshold", "target_fpr"):
            value = metadata.get(key, 0.0)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{path}: its {key} is {value!r}, not a number")  # noqa: TRY004, a damaged file
        return cls(
            weights=arrays["weights"],
            bias=float(arrays["bias"]),
            mean=arrays["mean"],
            std=arrays["std"],
            metadata=metadata,
        )


def apply_probe(probe: Probe, activations: np.ndarray) -> np.ndarray:
    """Score a (n, d) batch of activations. Computes in float32 whatever the input dtype."""
    return expit(probe_logits(probe, activations))


def probe_logits(probe: Probe, activations: np.ndarray) -> np.ndarray:
    """The probe's logits for a (n, d) batch, before the sigmoid, in float32. Computed
    directly, since inverting a saturated float32 score gives inf."""
    h = np.asarray(activations, dtype=np.float32)
    if h.ndim != 2 or h.shape[1] != probe.weights.shape[0]:
        raise ValueError(
            f"activations have shape {h.shape}, width {h.shape[-1]}; "
            f"probe expects 2-D input of width {probe.weights.shape[0]}"
        )
    z = (h - probe.mean.astype(np.float32)) / probe.std.astype(np.float32)
    return z @ probe.weights.astype(np.float32) + np.float32(probe.bias)
