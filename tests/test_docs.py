"""The live tutorial is an example in the repository, not part of the package (docs/design.md §4), so
no shipped file or prose doc may point at its old in-package module or data path. Recorded
run output is a historical record and is left as written."""

from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
GONE = ("hulder.capture.demo", "hulder/capture/demo", "hulder/data/demo_prompts")


def prose_and_package() -> list[Path]:
    package = [p for p in (ROOT / "src" / "hulder").rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    prose = [ROOT / "README.md", ROOT / "CONTEXT.md", *ROOT.glob("examples/**/*.md")]
    prose += [*ROOT.glob("docs/*.md"), *ROOT.glob("docs/results/**/*.md")]
    return package + prose


@pytest.mark.parametrize("path", prose_and_package(), ids=lambda p: str(p.relative_to(ROOT)))
def test_no_shipped_file_or_doc_names_the_old_live_tutorial_location(path):
    text = path.read_bytes().decode(errors="replace")
    assert not [old for old in GONE if old in text]


def test_the_live_tutorials_prompts_are_not_in_the_package():
    assert not (ROOT / "src" / "hulder" / "data" / "demo_prompts.jsonl").exists()
