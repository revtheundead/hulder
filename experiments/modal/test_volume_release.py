"""volume_release.py on a fake run and Volume: a backup is re-verified after a reboot before its
files leave the Volume. CPU only, no Modal."""

import hashlib
import json

import smoke_lib as lib
import volume_release

FILES = {"18_-1.npz": b"layer 18 rows", "27_-1.npz": b"layer 27 rows"}
SHAS = {name: hashlib.sha256(data).hexdigest() for name, data in FILES.items()}


def fetched_run(tmp_path, labels=("r1", "r2")):
    """A run whose replicas were fetched and backed up in boot "boot-a"."""
    run_dir, backup = tmp_path / "runs" / "R", tmp_path / "backup"
    configs = {}
    for label in labels:
        status = lib.settle_volume_files(SHAS, lambda name: [FILES[name]], run_dir / label, backup / "R" / label)
        configs[label] = {"returncode": 0, "volume": status}
    (run_dir / "manifest.json").write_text(json.dumps({"run_id": "R", "boot_id": "boot-a", "configs": configs}))
    return run_dir, backup


def release(run_dir, backup, boot):
    removed = []
    record = volume_release.release_run(run_dir, backup, lambda path: removed.append(path), boot)
    return record, removed


def test_after_a_reboot_each_replicas_files_leave_the_volume_and_the_run_records_it(tmp_path):
    run_dir, backup = fetched_run(tmp_path)
    record, removed = release(run_dir, backup, "boot-b")
    assert removed == [f"R/{label}/{name}" for label in ("r1", "r2") for name in sorted(FILES)]
    assert record["boot_id"] == "boot-b"
    assert {label: r["released"] for label, r in record["configs"].items()} == {"r1": True, "r2": True}
    assert json.loads((run_dir / volume_release.RECORD).read_text()) == record


def test_in_the_boot_that_fetched_them_nothing_leaves_the_volume(tmp_path):
    run_dir, backup = fetched_run(tmp_path)
    record, removed = release(run_dir, backup, "boot-a")
    assert removed == []
    assert all(not r["released"] and "reboot" in r["problem"] for r in record["configs"].values())


def test_a_replica_whose_backup_no_longer_verifies_keeps_its_files_and_the_others_go(tmp_path):
    run_dir, backup = fetched_run(tmp_path)
    (backup / "R" / "r2" / "27_-1.npz").write_bytes(b"layer 27 rowz")
    record, removed = release(run_dir, backup, "boot-b")
    assert removed == [f"R/r1/{name}" for name in sorted(FILES)]
    assert (record["configs"]["r1"]["released"], record["configs"]["r2"]["released"]) == (True, False)


def test_a_replica_released_before_isnt_released_again(tmp_path):
    run_dir, backup = fetched_run(tmp_path)
    release(run_dir, backup, "boot-b")
    record, removed = release(run_dir, backup, "boot-c")
    assert removed == []
    assert record["configs"]["r1"]["released"] is True and record["boot_id"] == "boot-b"
