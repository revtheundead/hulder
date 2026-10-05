"""Linear probes and the probe-application helper. Numpy only."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.special import expit


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
        """Write an .npz of plain arrays plus JSON metadata. Never pickled, so safe to share."""
        np.savez(
            path,
            weights=np.asarray(self.weights, dtype=np.float64),
            bias=np.float64(self.bias),
            mean=np.asarray(self.mean, dtype=np.float64),
            std=np.asarray(self.std, dtype=np.float64),
            metadata=np.array(json.dumps({"format_version": self.FORMAT_VERSION, **self.metadata})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Probe":
        with np.load(path, allow_pickle=False) as f:
            metadata = json.loads(str(f["metadata"]))
            version = metadata.pop("format_version")
            if version != cls.FORMAT_VERSION:
                raise ValueError(f"probe file format {version}, this hulder reads {cls.FORMAT_VERSION}")
            return cls(
                weights=f["weights"],
                bias=float(f["bias"]),
                mean=f["mean"],
                std=f["std"],
                metadata=metadata,
            )


def apply_probe(probe: Probe, activations: np.ndarray) -> np.ndarray:
    """Score a (n, d) batch of activations. Computes in float32 whatever the input dtype."""
    h = np.asarray(activations, dtype=np.float32)
    if h.ndim != 2 or h.shape[1] != probe.weights.shape[0]:
        raise ValueError(
            f"activations have shape {h.shape}, width {h.shape[-1]}; "
            f"probe expects 2-D input of width {probe.weights.shape[0]}"
        )
    z = (h - probe.mean.astype(np.float32)) / probe.std.astype(np.float32)
    return expit(z @ probe.weights.astype(np.float32) + np.float32(probe.bias))
