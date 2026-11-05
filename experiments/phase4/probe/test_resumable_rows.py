"""A long job's rows survive an interruption, and a resume can't mix in other inputs. CPU only."""

import json
import os

import pytest
import resumable_rows as r

IDENTITY = {"expected": "abc", "code": "def"}


def double(row):
    return {**row, "value": row["n"] * 2}


def fail_on_two(row):
    if row["n"] == 2:
        raise RuntimeError("interrupted")
    return double(row)


def rows(*ns):
    return [{"key": f"row{n}", "n": n} for n in ns]


def test_a_new_store_is_empty_and_keeps_what_was_added(tmp_path):
    path = tmp_path / "rows.jsonl"
    store = r.RowStore(path, IDENTITY)
    assert store.done == {}
    store.add({"key": "a", "value": 1})
    assert r.RowStore(path, IDENTITY).done == {"a": {"key": "a", "value": 1}}


def test_a_row_is_on_disk_as_soon_as_it_is_added(tmp_path, monkeypatch):
    synced = []
    real = os.fsync
    monkeypatch.setattr(r.os, "fsync", lambda fd: (synced.append(fd), real(fd)))
    store = r.RowStore(tmp_path / "rows.jsonl", IDENTITY)
    before = len(synced)
    store.add({"key": "a"})
    assert len(synced) > before


def test_a_store_from_other_inputs_or_code_is_refused(tmp_path):
    path = tmp_path / "rows.jsonl"
    r.RowStore(path, IDENTITY).add({"key": "a"})
    with pytest.raises(ValueError, match="other inputs or code"):
        r.RowStore(path, {**IDENTITY, "code": "changed"})


def test_a_torn_last_line_is_dropped_and_the_store_stays_appendable(tmp_path):
    path = tmp_path / "rows.jsonl"
    store = r.RowStore(path, IDENTITY)
    store.add({"key": "a"})
    with path.open("a") as f:
        f.write('{"key": "b", "val')
    resumed = r.RowStore(path, IDENTITY)
    assert list(resumed.done) == ["a"]
    resumed.add({"key": "c"})
    assert list(r.RowStore(path, IDENTITY).done) == ["a", "c"]


def test_a_complete_last_line_without_its_newline_is_also_dropped(tmp_path):
    path = tmp_path / "rows.jsonl"
    r.RowStore(path, IDENTITY).add({"key": "a"})
    with path.open("a") as f:
        f.write('{"key": "b"}')
    assert list(r.RowStore(path, IDENTITY).done) == ["a"]


def test_damage_before_the_last_line_is_refused(tmp_path):
    path = tmp_path / "rows.jsonl"
    store = r.RowStore(path, IDENTITY)
    store.add({"key": "a"})
    store.add({"key": "b"})
    lines = path.read_text().splitlines()
    lines[1] = "not json"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="damaged"):
        r.RowStore(path, IDENTITY)


def test_a_torn_identity_line_is_refused_rather_than_read_as_an_empty_store(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"identity"')
    with pytest.raises(ValueError, match="damaged"):
        r.RowStore(path, IDENTITY)


def test_running_returns_every_row_in_the_order_asked_for(tmp_path):
    store = r.RowStore(tmp_path / "rows.jsonl", IDENTITY)
    out = r.run_rows(rows(3, 1, 2), double, store, workers=2)
    assert [row["key"] for row in out] == ["row3", "row1", "row2"]
    assert [row["value"] for row in out] == [6, 2, 4]


def test_an_interruption_keeps_finished_rows_and_a_resume_runs_only_the_rest(tmp_path):
    path = tmp_path / "rows.jsonl"
    with pytest.raises(RuntimeError, match="interrupted"):
        r.run_rows(rows(1, 2), fail_on_two, r.RowStore(path, IDENTITY), workers=1)
    assert list(r.RowStore(path, IDENTITY).done) == ["row1"]
    # row1 would raise here if it ran again, so finishing shows it didn't.
    out = r.run_rows(
        [{"key": "row1", "n": 2}, {"key": "row2", "n": 3}], fail_on_two, r.RowStore(path, IDENTITY), workers=1
    )
    assert [row["value"] for row in out] == [2, 6]


def test_extra_header_fields_are_kept_from_the_first_start_and_dont_block_a_resume(tmp_path):
    path = tmp_path / "rows.jsonl"
    r.RowStore(path, IDENTITY, extra={"manifest": {"commit": "first"}})
    resumed = r.RowStore(path, IDENTITY, extra={"manifest": {"commit": "second"}})
    assert resumed.extra == {"manifest": {"commit": "first"}}
    assert json.loads(path.read_text().splitlines()[0])["extra"]["manifest"]["commit"] == "first"
