"""Reading Snapshot and Probe files. A damaged or foreign file raises ValueError naming it."""

import json
import zipfile
from pathlib import Path

import numpy as np

# What np.load and NpzFile raise on a file that isn't an intact .npz of plain arrays:
# BadZipFile when truncated, EOFError when empty, ValueError when it would need pickle.
_UNREADABLE = (zipfile.BadZipFile, EOFError, ValueError)


def read_npz(path: str | Path, kind: str, required: tuple[str, ...]) -> tuple[dict[str, np.ndarray], dict]:
    """Every array in the file, loaded without pickle, and its JSON metadata as a dict. A
    missing file stays an OSError."""
    unreadable = f"{path}: can't be read as a hulder {kind} file, since it isn't an intact .npz of plain arrays"
    try:
        loaded = np.load(path, allow_pickle=False)
        if not isinstance(loaded, np.lib.npyio.NpzFile):
            raise ValueError(unreadable)  # noqa: TRY004, a damaged file, not a caller's type error
        with loaded as f:
            arrays = {name: f[name] for name in f.files}
    except _UNREADABLE as e:
        raise ValueError(unreadable) from e
    missing = [name for name in ("metadata", *required) if name not in arrays]
    if missing:
        raise ValueError(f"{path}: lacks the arrays {', '.join(missing)}, so it isn't a hulder {kind} file")
    try:
        metadata = json.loads(str(arrays.pop("metadata")))
    except ValueError as e:
        raise ValueError(f"{path}: its metadata isn't valid JSON, so the file is damaged") from e
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: its metadata isn't a JSON object, so it isn't a hulder {kind} file")  # noqa: TRY004
    return arrays, metadata
