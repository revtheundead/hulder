"""What the installed distribution declares: its licence, and the bundled data's attribution."""

import re
from importlib.metadata import metadata, requires
from importlib.resources import files


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
