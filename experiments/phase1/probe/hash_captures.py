"""Record sha256 hashes of capture files in their manifests, for captures that predate it.

extract.py records them for new captures, and Activations.load verifies them. Run from
the repo root with the capture directories:
    source scripts/dev-env.sh
    uv run python experiments/phase1/probe/hash_captures.py .cache/probe/fp16 .cache/probe/fp16_run1 .cache/probe/awq

A manifest that already has hashes is verified, not rewritten.
"""

import json
import sys
from pathlib import Path

from common import CAPTURE_FILES, sha256_file


def main() -> None:
    for capture in map(Path, sys.argv[1:]):
        manifest_path = capture / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        hashes = {name: sha256_file(capture / name) for name in CAPTURE_FILES}
        if "files_sha256" in manifest:
            status = "verified" if manifest["files_sha256"] == hashes else "MISMATCH"
        else:
            manifest["files_sha256"] = hashes
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            status = "recorded"
        print(f"{capture}: {status} {hashes}")
        if status == "MISMATCH":
            sys.exit(1)


if __name__ == "__main__":
    main()
