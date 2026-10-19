"""Remove a run's snapshots from the hulder-runs Volume once their backup verifies again after a reboot.

    uv run --with modal==1.6.1 python experiments/modal/volume_release.py .cache/modal-runs/<run id> <backup-dir>

<backup-dir> is the --backup-dir the run was given. capture.py fetches each replica's snapshots,
checks them against the container's sha256 and backs them up, but never removes them. This does,
in a later boot than the run's (the manifest records the boot id), and only for a replica whose
SHA256SUMS and every backed-up file still match the container's sha256. A replica that fails keeps
its files and the others go. The outcome is written to <run>/volume_release.json, which
full_capture_report.py reads.
"""

import json
import sys
from collections.abc import Callable
from pathlib import Path

import smoke_lib as lib

RECORD = "volume_release.json"


def release_run(run_dir: Path, backup_root: Path, remove: Callable[[str], None], current_boot_id: str | None) -> dict:
    """Release each config of the run whose backup verifies again; remove takes a Volume path."""
    run = json.loads((run_dir / "manifest.json").read_text())
    path = run_dir / RECORD
    record = json.loads(path.read_text()) if path.exists() else {"boot_id": None, "configs": {}}
    changed = False
    for label, config in sorted(run["configs"].items()):
        if record["configs"].get(label, {}).get("released"):
            continue
        record["configs"][label] = lib.release_volume_files(
            config.get("volume", {"files": {}, "backed_up": False}),
            backup_root / run["run_id"] / label,
            lambda name, label=label: remove(f"{run['run_id']}/{label}/{name}"),
            run.get("boot_id"),
            current_boot_id,
        )
        changed = True
    if changed:
        record["boot_id"] = current_boot_id
        path.write_text(json.dumps(record, indent=2) + "\n")
    return record


if __name__ == "__main__":
    import modal

    runs = modal.Volume.from_name("hulder-runs")
    print(json.dumps(release_run(Path(sys.argv[1]), Path(sys.argv[2]), runs.remove_file, lib.boot_id()), indent=2))
