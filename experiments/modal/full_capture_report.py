"""Report a full_capture run (smoke_lib.EXPERIMENTS["full_capture"]) and keep each batch's majority.

    uv run python experiments/modal/full_capture_report.py .cache/modal-runs/<run id> .cache/modal-runs/<pilot run id>

Writes full_capture_report.json next to the run's manifest.json and prints it. A run that already
has kept/ is refused before anything is read or written, unless --force is passed. With it, the new
kept dataset is written to kept.new/ and swapped in only once every rule has passed, so a rerun that
stops, or fails part way, leaves the earlier kept/ as it was and the report says so.

The three replicas are compared batch by batch, bit for bit on every saved array together, from
batch 1 on. Batch 1 is the canary: the request begins with its 8 boundary-pairs prompts, so their
rows are saved like any other batch's, and each replica's canary hash is listed as well. Each batch
is "agree", "split" (2 to 1, the outlier recorded) or "no majority". The kept dataset, under kept/, is
each batch's majority without the no-majority batches. It is written only if no stop rule fires:
R1, the replicas weren't fed the same prompts and tokens; R2, more than 1% of batches have no
majority; R3, the replicas' majority canary isn't the majority bf16 path's. A failed or timed-out
replica stops it too. The report reads one array file at a time.
"""

import json
import shutil
import sys
from pathlib import Path

import pilot_report as pilot
import smoke_lib as lib

import hulder

# The majority bf16 path, 16 captures on 2026-10-13 and 2026-10-15 (docs/results/phase0/modal_smoke.md).
MAJORITY = "2.0"
# Batches seen leaving the majority path, of those compared bit for bit: the pilot's batch 8, of 7
# batches each in the pilot's canary_check and canary_repeat's 4 (2026-10-17 and 2026-10-19).
DEVIATIONS = {"batches_off_path": 1, "batches": 35}
NO_MAJORITY_LIMIT = 0.01  # R2
# Metadata that may differ between replicas, and which the kept dataset records its own way.
PER_REPLICA = ("gpu_uuid", "gpu_driver", "canary_sha256")
# Left out of R1's metadata comparison: those, and the token hash R1 reads from the worker records.
CHECKED_APART = (*PER_REPLICA, "prompt_token_ids_sha256")


def _expected() -> dict:
    p = DEVIATIONS["batches_off_path"] / DEVIATIONS["batches"]
    interval = hulder.clopper_pearson(DEVIATIONS["batches_off_path"], DEVIATIONS["batches"])
    rates = lib.expected_split_rates(p)
    return {
        "deviations": DEVIATIONS,
        "per_batch_rate": p,
        "per_batch_interval": list(interval),
        "split_rate": rates["split"],
        "two_or_more_deviate_rate": rates["two_or_more_deviate"],
        "split_rate_range": [lib.expected_split_rates(q)["split"] for q in interval],
        "note": (
            "If each replica leaves the path on a batch independently at the rough rate, a 2-to-1 split "
            "is expected on about this share of batches, so a split rate near it is not alarming. The "
            "rate rests on one deviation in 35 batches, so the range is wide. If two replicas ever "
            "deviated alike, their batch would be a split with an off-path majority, which nothing here "
            "can detect."
        ),
    }


def summarise(
    run_dir: Path,
    pilot_dir: Path,
    clusters_path: Path = pilot.CLUSTERS,
    runs_dir: Path = pilot.RUNS,
    *,
    splits: dict[str, str] | None = None,
    reference_cluster: str = MAJORITY,
    force: bool = False,
) -> dict:
    if (run_dir / "kept.old").exists():
        # An interrupted swap: it may hold the only copy of the earlier kept dataset, so a person decides.
        raise FileExistsError(f"{run_dir / 'kept.old'} is left from an interrupted swap; restore or remove it first")
    if (run_dir / "kept").exists() and not force:
        raise FileExistsError(f"{run_dir / 'kept'} exists, and a rerun would replace it; pass --force to allow that")
    run = json.loads((run_dir / "manifest.json").read_text())
    recorded = pilot.recorded_hashes(clusters_path, runs_dir)
    stops = [f"the run stopped: {run['stopped']}"] if run.get("stopped") else []
    workers = {}
    for label in run["configs"]:
        manifest = json.loads((run_dir / label / "manifest.json").read_text())
        if failed := pilot.failure_reason(run_dir, label, manifest):
            stops.append(failed)
        problem = (run["configs"][label].get("volume") or {}).get("problem")
        if problem:
            # Its local files may be partial or wrong, so like a failed replica's they are never read.
            stops.append(f"{label}'s snapshots weren't fetched intact: {problem}")
        elif "worker" in manifest:
            workers[label] = manifest["worker"]
    labels = sorted(workers)
    record = run_dir / "volume_release.json"
    released = json.loads(record.read_text())["configs"] if record.exists() else {}
    head = {
        **{k: run.get(k) for k in ("run_id", "commit", "estimated_usd", "stopped")},
        "stop_reasons": stops,
        "volume": {
            label: _volume(run["configs"][label].get("volume"), released.get(label)) for label in sorted(run["configs"])
        },
        "expected": _expected(),
    }
    if not labels:
        return {
            **head,
            "findings": [],
            "kept": None,
            "batches": [],
            "counts": {"batches": 0, "agree": 0, "split": 0, "no_majority": 0},
            "outliers_per_replica": {},
            "vs_pilot": [],
            "canaries": {},
            "majority_canary": None,
            "kept_note": _earlier_kept(run_dir),
        }
    names = workers[labels[0]]["snapshots"]
    # The request begins with the canary, so the saved rows start at batch 1 (smoke_lib.canary_first).
    # R1 below checks every replica's batch_size, which snapshot metadata records, against the others.
    batch_size = workers[labels[0]]["capture_config"]["batch_size"]
    first_batch = 1

    # R1: the same prompts and tokens, and the same metadata apart from what is per replica.
    for key in ("prompt_token_ids_sha256", "canary_token_ids_sha256"):
        if len({workers[label].get(key) for label in labels}) != 1 or workers[labels[0]].get(key) is None:
            stops.append(f"R1: the replicas' {key} differ")
    digests: dict[str, dict[str, list[str]]] = {label: {} for label in labels}
    snapshot_ids: dict[str, list[str]] = {}
    for name in names:
        snapshots = {label: hulder.Snapshot.load(run_dir / label / name) for label in labels}
        for label, snapshot in snapshots.items():
            snapshot_ids.setdefault(label, [str(i) for i in snapshot.ids])
        if any(list(s.ids) != list(snapshots[labels[0]].ids) for s in snapshots.values()):
            stops.append(f"R1: the replicas' {name} prompt ids differ")
        keys = {k for snapshot in snapshots.values() for k in snapshot.metadata} - set(CHECKED_APART)
        differing = sorted(
            k for k in keys if len({json.dumps(x.metadata.get(k), sort_keys=True) for x in snapshots.values()}) > 1
        )
        if differing:
            stops.append(f"R1: the replicas' {name} metadata differs in {differing}")
        for label, snapshot in snapshots.items():
            digests[label][name] = lib.batch_digests(snapshot.activations, batch_size)
        del snapshots
    # R1 on each replica's token ids: their hash as recorded, and filed under its snapshots' prompts.
    for label in labels:
        token_ids, recorded_hash = run_dir / label / lib.TOKEN_IDS_FILE, workers[label].get("prompt_token_ids_sha256")
        if not token_ids.exists():
            stops.append(f"R1: {label} has no {lib.TOKEN_IDS_FILE}")
            continue
        ids, tokens = lib.load_token_ids(token_ids)
        if lib.token_ids_sha256(tokens) != recorded_hash:
            stops.append(f"R1: {label}'s {lib.TOKEN_IDS_FILE} doesn't match its prompt_token_ids_sha256")
        if ids != snapshot_ids[label]:
            stops.append(f"R1: {label}'s {lib.TOKEN_IDS_FILE} holds other prompt ids than its snapshots")
    batches = lib.batch_agreement(digests, first_batch)
    n = len(batches)
    kinds = [b["kind"] for b in batches]
    not_unanimous = n - kinds.count("agree")
    if kinds.count("no majority") > NO_MAJORITY_LIMIT * n:
        stops.append(
            f"R2: {kinds.count('no majority')} of {n} batches ({kinds.count('no majority') / n:.1%}) have no "
            f"majority, over the {NO_MAJORITY_LIMIT:.0%} limit"
        )

    outliers = {label: sum(label in b["outliers"] for b in batches) for label in labels}

    canaries = {}
    for label in labels:
        h = workers[label]["canary_sha256"]
        match = pilot.match_canary(h, recorded)
        canaries[label] = {"canary_sha256": h, "cluster": match["cluster"] if isinstance(match, dict) else match}
    by_hash: dict[str, list[str]] = {}
    for label in labels:
        by_hash.setdefault(canaries[label]["canary_sha256"], []).append(label)
    top = max(by_hash, key=lambda h: len(by_hash[h]))
    majority_canary = (
        {"canary_sha256": top, "cluster": canaries[by_hash[top][0]]["cluster"], "replicas": by_hash[top]}
        if 2 * len(by_hash[top]) > len(labels)
        else None
    )
    # R3: a review stop, so a kept dataset on a path no recorded capture took isn't used unread.
    if majority_canary is None:
        stops.append("R3: the replicas' canaries have no majority")
    elif majority_canary["cluster"] != reference_cluster:
        stops.append(
            f"R3: the replicas' majority canary is {majority_canary['cluster']}, not cluster {reference_cluster}"
        )

    findings = [
        f"{label} is the outlier on {count} of {n} batches, a path switch rather than sporadic batches"
        for label, count in outliers.items()
        if 2 * count > n
    ]
    vs_pilot = _vs_pilot(pilot_dir / "pilot", run_dir / labels[0], names, digests, batches, batch_size)
    if unlike := [b["batch"] for b in vs_pilot if b["identical"] is False]:
        findings.append(f"the kept batches {unlike} differ from the pilot's")
    kept = None
    if not stops:
        kept = _keep(run_dir, labels, names, batches, workers, majority_canary, splits or {}, batch_size, first_batch)

    return {
        **head,
        "findings": findings,
        "kept": kept,
        "batches": batches,
        "counts": {
            "batches": n,
            "agree": kinds.count("agree"),
            "split": kinds.count("split"),
            "no_majority": kinds.count("no majority"),
            "disagreement_rate": not_unanimous / n,
            "disagreement_interval": list(hulder.clopper_pearson(not_unanimous, n)),
        },
        "outliers_per_replica": outliers,
        "vs_pilot": vs_pilot,
        "canaries": canaries,
        "majority_canary": majority_canary,
        "kept_note": None if kept else _earlier_kept(run_dir),
    }


def _earlier_kept(run_dir: Path) -> str | None:
    """Why a kept/ is there when this run kept nothing: only a forced rerun that stopped leaves one."""
    if not (run_dir / "kept").exists():
        return None
    return "kept/ is from an earlier report, left untouched because this run stopped; it is not this run's majority"


def _vs_pilot(
    pilot_capture: Path, replica: Path, names: list[str], digests: dict, batches: list[dict], batch_size: int
) -> list:
    """The kept batches against the pilot's. The pilot ran the canary in front of its request
    without saving its rows, so its saved rows are the replicas' from the batch after the canary
    on, in the same batches. None for a batch with no majority, which isn't kept."""
    out = []
    pilot_digests = {}
    skip = len(lib.canary_prompts())
    for name in names:
        snapshot = hulder.Snapshot.load(pilot_capture / name)
        ids = [str(i) for i in hulder.Snapshot.load(replica / name).ids]
        if [str(i) for i in snapshot.ids] != ids[skip : skip + len(snapshot.ids)]:
            raise ValueError(
                f"the pilot's {name} holds other prompts than the replicas' {len(snapshot.ids)} after the canary"
            )
        pilot_digests[name] = lib.batch_digests(snapshot.activations, batch_size)
    for b, entry in enumerate(batches[skip // batch_size :][: len(pilot_digests[names[0]])]):
        same = None
        if entry["majority"]:
            member = digests[entry["majority"][0]]
            same = all(member[name][b + skip // batch_size] == pilot_digests[name][b] for name in names)
        out.append({"batch": entry["batch"], "identical": same})
    return out


def _volume(status: dict | None, release: dict | None) -> str:
    """Where a replica's snapshots stand: capture.py fetches and backs them up and leaves them on
    the Volume, and volume_release.py removes them after a reboot (smoke_lib.release_volume_files)."""
    if release and release["released"]:
        return "removed from the hulder-runs Volume after its backup verified again in a later boot"
    if status is None:
        return "not recorded"
    if not status["backed_up"]:
        return f"on the hulder-runs Volume, not backed up: {status['problem'] or 'no backup dir was given'}"
    if release:
        return f"on the hulder-runs Volume, backed up and verified; volume_release.py refused: {release['problem']}"
    return "on the hulder-runs Volume, backed up and verified; volume_release.py removes it after a reboot"


def _keep(run_dir, labels, names, batches, workers, majority_canary, splits, batch_size, first_batch) -> dict:
    """Write each array's per-batch majority to kept.new/, swap it in as kept/, and say what was
    dropped. A failure part way leaves kept.new/ behind and any earlier kept/ as it was."""
    contributing = sorted({label for b in batches for label in b["majority"]})
    drivers = sorted({workers[label]["gpu_driver"] for label in contributing})
    out, kept, old = run_dir / "kept.new", run_dir / "kept", run_dir / "kept.old"
    if out.exists():  # a partial write from a run that failed part way
        shutil.rmtree(out)
    out.mkdir()
    # R1 checked every replica's token ids against its hash, and the hashes against each other.
    token_ids = dict(zip(*lib.load_token_ids(run_dir / contributing[0] / lib.TOKEN_IDS_FILE), strict=True))
    for name in names:
        snapshots = {label: hulder.Snapshot.load(run_dir / label / name) for label in labels}
        first = snapshots[contributing[0]]
        ids = lib.majority_rows({label: s.ids for label, s in snapshots.items()}, batches, batch_size, first_batch)
        metadata = {
            **first.metadata,
            "gpu_uuid": sorted(workers[label]["gpu_uuid"] for label in contributing),
            "gpu_driver": drivers[0] if len(drivers) == 1 else drivers,
            "canary_sha256": majority_canary["canary_sha256"],
            "prompt_ids_sha256": hulder.prompt_ids_sha256([str(i) for i in ids]),
            "prompt_token_ids_sha256": lib.token_ids_sha256([token_ids[str(i)] for i in ids]),
        }
        hulder.Snapshot(
            ids=ids,
            labels=lib.majority_rows(
                {label: s.labels for label, s in snapshots.items()}, batches, batch_size, first_batch
            ),
            activations=lib.majority_rows(
                {label: s.activations for label, s in snapshots.items()}, batches, batch_size, first_batch
            ),
            metadata=metadata,
        ).save(out / name)
        all_ids = [str(i) for i in first.ids]
    kept_ids = [str(i) for i in ids]
    lib.save_token_ids(out / lib.TOKEN_IDS_FILE, kept_ids, [token_ids[i] for i in kept_ids])
    if kept.exists():
        kept.rename(old)
    out.rename(kept)
    if old.exists():
        shutil.rmtree(old)
    dropped_batches = [b["batch"] for b in batches if not b["majority"]]
    dropped = [
        {"id": i, "split": splits.get(i)}
        for b in dropped_batches
        for i in all_ids[(b - first_batch) * batch_size : (b - first_batch + 1) * batch_size]
    ]
    return {
        "dir": "kept",
        "prompts": len(all_ids) - len(dropped),
        "dropped_batches": dropped_batches,
        "dropped": dropped,
    }


def splits_by_id(path: Path = lib.SLABS_SPLITS) -> dict[str, str]:
    """Each prompt id's split, to name what dropped batches cost. Calibration ids look like train's."""
    import pandas as pd

    rows = pd.read_parquet(path, columns=["id", "split"])
    return dict(zip(rows["id"].astype(str), rows["split"].astype(str), strict=True))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--force"]
    run_dir = Path(args[0])
    try:
        summary = summarise(run_dir, Path(args[1]), splits=splits_by_id(), force="--force" in sys.argv[1:])
    except FileExistsError as e:
        sys.exit(str(e))
    (run_dir / "full_capture_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
