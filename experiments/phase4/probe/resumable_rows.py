"""Rows of a long job, each on disk as soon as it finishes, so an interruption loses only what was running.

A store is a JSONL file. Its first line records the identity of what's being run, and a resume with
any other identity is refused, so results from other inputs or code can't be mixed in. A last line
torn by a crash is dropped and run again. Damage anywhere else is refused, since it isn't explained
by an interruption.
"""

import json
import multiprocessing
import os
from collections.abc import Callable
from pathlib import Path


class RowStore:
    def __init__(self, path: Path, identity: dict, extra: dict | None = None):
        self.path, self.done = Path(path), {}
        if not self.path.exists() or self.path.stat().st_size == 0:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.extra = extra or {}
            self._append({"identity": identity, "extra": self.extra})
            _sync_directory(self.path.parent)
            return
        header, rows, keep = self._read()
        if header["identity"] != identity:
            raise ValueError(f"{self.path} holds rows from other inputs or code: {header['identity']} vs {identity}")
        self.extra = header.get("extra", {})
        self.done = {row["key"]: row for row in rows}
        if keep != self.path.stat().st_size:
            with self.path.open("r+b") as f:
                f.truncate(keep)
                f.flush()
                os.fsync(f.fileno())

    def _read(self) -> tuple[dict, list[dict], int]:
        """The header, the complete rows, and the bytes they take. A torn last line isn't counted."""
        raw = self.path.read_bytes()
        lines = raw.split(b"\n")
        complete, torn = lines[:-1], lines[-1]  # with a final newline, torn is b""
        if not complete:
            raise ValueError(f"{self.path} is damaged: its identity line is incomplete")
        parsed = []
        for number, line in enumerate(complete, start=1):
            try:
                parsed.append(json.loads(line))
            except ValueError as error:
                raise ValueError(f"{self.path} is damaged at line {number}: {error}") from error
        if "identity" not in parsed[0]:
            raise ValueError(f"{self.path} is damaged: its first line has no identity")
        return parsed[0], parsed[1:], len(raw) - len(torn)

    def _append(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def add(self, row: dict) -> None:
        self._append(row)
        self.done[row["key"]] = row


def run_rows(rows: list[dict], work: Callable[[dict], dict], store: RowStore, workers: int) -> list[dict]:
    """Every row's result, in the order asked for. Rows the store holds aren't run again. `work` runs
    in forked workers and returns a dict with the row's "key", and only this process writes the store."""
    pending = [row for row in rows if row["key"] not in store.done]
    if pending:
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            for result in pool.imap_unordered(work, pending, chunksize=1):
                store.add(result)
    return [store.done[row["key"]] for row in rows]


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
