"""Rebuild a capture run's config folders from the hulder-runs Volume after the local process of
capture.py died. Its own file, so capture.py keeps a single local entrypoint and its documented
command. Starts no container: it only reads the Volume.

    source scripts/dev-env.sh
    uv run --with modal==1.6.1 modal run experiments/modal/recover_run.py --run-id <run id> --backup-dir <dir>

Each config whose manifest reached the Volume is fetched and checked against the snapshot hashes
its container recorded (smoke_lib.recover_config), and backed up with --backup-dir. A config folder
that already exists locally is left alone. Writes <run>/recovered.json and adds the recovered
configs to <run>/manifest.json. Containers' costs aren't recovered: read them from Modal's usage page.
"""

import json
from pathlib import Path, PurePosixPath

import modal
import smoke_lib as lib

ROOT = Path(__file__).resolve().parents[2]
app = modal.App("hulder-recover")
runs = modal.Volume.from_name("hulder-runs")


@app.local_entrypoint()
def main(run_id: str, out: str = ".cache/modal-runs", backup_dir: str = "") -> None:
    run_dir = ROOT / out / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    labels = sorted({PurePosixPath(e.path).name for e in runs.listdir(run_id)})

    def reader(label: str):
        def read(name: str) -> bytes:
            try:
                return b"".join(runs.read_file(f"{run_id}/{label}/{name}"))
            except (modal.exception.NotFoundError, FileNotFoundError) as e:
                raise FileNotFoundError(name) from e

        return read

    recovered = {}
    for label in labels:
        if (run_dir / label).exists():
            recovered[label] = "already local, left alone"
            continue
        backup = Path(backup_dir) / run_id / label if backup_dir else None
        entry = lib.recover_config(reader(label), run_dir / label, backup)
        recovered[label] = entry if entry is not None else "no manifest on the Volume: its engine didn't finish"
    record = {"boot_id": lib.boot_id(), "configs": recovered}
    (run_dir / "recovered.json").write_text(json.dumps(record, indent=2) + "\n")
    # The run manifest gains the recovered configs, so reports read the run as if collected.
    path = run_dir / "manifest.json"
    if path.exists():
        merged = lib.merge_recovered(json.loads(path.read_text()), recovered)
        path.write_text(json.dumps(merged, indent=2) + "\n")
    print(json.dumps(recovered, indent=2))
