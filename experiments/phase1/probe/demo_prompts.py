"""The live tutorial's prompts: the 360 boundary-pairs calibration examples of the demo
snapshots, in their order, from the pinned dataset revision.

Run from the repo root:
    source scripts/dev-env.sh && uv run python experiments/phase1/probe/demo_prompts.py

Writes examples/live_demo/demo_prompts.jsonl, one {"id", "label", "text"} per line.
"""

import json
from importlib.resources import files
from pathlib import Path

from common import load_bpairs

import hulder

OUT = Path("examples/live_demo/demo_prompts.jsonl")


def main() -> None:
    reference = hulder.Snapshot.load(str(files("hulder.data") / "demo_fp16.npz"))
    rows = load_bpairs().set_index("id").loc[list(reference.ids)]
    if (rows["label"].to_numpy() != reference.labels).any():
        raise SystemExit("boundary-pairs labels disagree with demo_fp16.npz")
    with OUT.open("w") as f:
        for i, row in rows.iterrows():
            f.write(json.dumps({"id": i, "label": int(row["label"]), "text": row["text"]}, ensure_ascii=False) + "\n")
    print(f"{OUT}: {len(rows)} prompts ({int((rows['label'] == 0).sum())} benign)")


if __name__ == "__main__":
    main()
