"""Shared pieces of the Phase 1 probe scripts: dataset pins, loaders, run manifests."""

import json
import platform
import subprocess
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

import hulder

SPLITS = Path(".cache/probe/splits.parquet")
ACTS = Path(".cache/probe/fp16")


class HfDataset(NamedTuple):
    repo: str
    revision: str


SLABS = HfDataset("S-Labs/prompt-injection-dataset", "002a9dd18514abd021869823d6b0429b38606d99")
BPAIRS = HfDataset("3nesdeniz/agentic-prompt-injection-boundary-pairs", "a5682e7573e1c7bc4b12e64d49c0dcd90ca776cf")
BPAIRS_ATTRIBUTION = (
    "Contains data derived from 3nesdeniz/agentic-prompt-injection-boundary-pairs "
    "(Deniz 2026, doi:10.5281/zenodo.21378693), CC-BY-4.0."
)


def load_bpairs() -> pd.DataFrame:
    """All 1,200 boundary-pairs rows, ids prefixed 'bpairs/' as in splits.parquet."""
    parts = []
    for split in ["train", "validation", "test"]:
        path = hf_hub_download(BPAIRS.repo, f"data/{split}.parquet", repo_type="dataset", revision=BPAIRS.revision)
        parts.append(pd.read_parquet(path))
    df = pd.concat(parts, ignore_index=True)
    df["id"] = "bpairs/" + df["id"]
    return df


@dataclass(frozen=True)
class Activations:
    """The captured activations of one serving config, addressable by example id."""

    array: np.ndarray  # (n, layers, positions, d), memory-mapped
    row_of: dict[str, int]
    capture_manifest: dict

    @classmethod
    def load(cls, path: Path = ACTS) -> "Activations":
        ids = json.loads((path / "ids.json").read_text())
        return cls(
            array=np.load(path / "activations.npy", mmap_mode="r"),
            row_of={i: r for r, i in enumerate(ids)},
            capture_manifest=json.loads((path / "manifest.json").read_text()),
        )

    def at(self, ids, layer: int, position: int) -> np.ndarray:
        cfg = self.capture_manifest["capture_config"]
        rows = [self.row_of[i] for i in ids]
        return np.asarray(self.array[rows, cfg["layers"].index(layer), cfg["positions"].index(position)])

    def for_probe(self, ids, probe: hulder.Probe) -> np.ndarray:
        return self.at(ids, probe.metadata["layer"], probe.metadata["position"])


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def run_manifest(seed: int | None = None, inputs: tuple[Path, ...] = ()) -> dict:
    """Provenance for a committed result: code version, dirty tree, packages, input hashes."""
    packages = {}
    for name in ("numpy", "scipy", "scikit-learn", "pandas", "huggingface-hub", "vllm", "torch"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "python": platform.python_version(),
        "packages": packages,
        "seed": seed,
        "inputs_sha256": {
            str(p): subprocess.run(["sha256sum", str(p)], capture_output=True, text=True, check=True).stdout.split()[0]
            for p in inputs
        },
    }
