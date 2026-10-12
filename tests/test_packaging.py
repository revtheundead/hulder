"""What the installed distribution declares: its licence, and the bundled data's attribution."""

import re
from importlib.metadata import metadata, requires
from importlib.resources import files

import hulder
from hulder.capture import demo


def test_hulder_is_licensed_apache_2():
    assert metadata("hulder")["License-Expression"] == "Apache-2.0"


def test_the_bundled_data_notice_links_its_cc_by_licence():
    notice = (files("hulder.data") / "NOTICE.md").read_text()
    assert "CC-BY-4.0" in notice
    assert "https://creativecommons.org/licenses/by/4.0/" in notice


def test_the_vllm_extra_declares_what_hulder_capture_imports():
    # hulder.capture imports yaml and huggingface_hub itself, so they can't rely on vllm pulling them in.
    vllm_extra = [
        r.split(";")[0].strip() for r in requires("hulder") if "extra == 'vllm'" in r or 'extra == "vllm"' in r
    ]
    names = {re.split(r"[<>=!~ ]", r, maxsplit=1)[0].lower() for r in vllm_extra}
    assert {"vllm", "pyyaml", "huggingface-hub"} <= names


def test_the_bundled_prompts_are_the_demo_snapshots_examples_in_order():
    rows = demo.prompts()
    reference = hulder.Snapshot.load(str(files("hulder.data") / "demo_fp16.npz"))
    assert [r["id"] for r in rows] == list(reference.ids)
    assert [r["label"] for r in rows] == reference.labels.tolist()
    assert all(r["text"].strip() for r in rows)


def test_the_pypi_description_is_the_short_page_with_absolute_links():
    # README.md is also the agent brief and its relative links break on PyPI.
    description = metadata("hulder")["Description"]
    assert description.startswith("# Hulder")
    assert "https://github.com/revtheundead/hulder" in description
    assert not re.search(r"\]\((?!https?://)", description)
