"""Files that appear whole or not at all, so an interrupted write never leaves half a file."""

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO


@contextmanager
def atomic_write(path: str | Path) -> Iterator[BinaryIO]:
    """A binary file to write that replaces path only once it is complete. It is written to a
    temporary file in path's directory, so the final os.replace is a rename on one filesystem."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            yield f
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_text_atomically(path: str | Path, text: str) -> None:
    with atomic_write(path) as f:
        f.write(text.encode())
