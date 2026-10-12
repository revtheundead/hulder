"""What the installed distribution declares: its version, licence, links and classifiers, and the
bundled data's attribution."""

import re
import tomllib
from importlib.metadata import metadata, requires
from importlib.resources import files
from pathlib import Path

import hulder

PYPROJECT = Path(__file__).parents[1] / "pyproject.toml"


def test_hulder_version_is_the_one_in_pyproject():
    assert hulder.__version__ == tomllib.loads(PYPROJECT.read_text())["project"]["version"]


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


def test_the_pypi_description_is_the_short_page_with_absolute_links():
    # README.md's relative links break on PyPI, so PyPI gets docs/pypi.md.
    description = metadata("hulder")["Description"]
    assert description.startswith("# Hulder")
    assert "https://github.com/revtheundead/hulder" in description
    assert not re.search(r"\]\((?!https?://)", description)


def test_the_pypi_sidebar_links_the_source_and_issues():
    urls = dict(u.split(", ", 1) for u in metadata("hulder").get_all("Project-URL"))
    assert urls["Source"] == "https://github.com/revtheundead/hulder"
    assert urls["Issues"] == "https://github.com/revtheundead/hulder/issues"


def test_the_classifiers_name_python_3_12_and_the_ai_topic_but_no_licence():
    # PEP 639: with a License-Expression, License :: classifiers are an error on upload.
    classifiers = metadata("hulder").get_all("Classifier")
    assert "Programming Language :: Python :: 3.12" in classifiers
    assert "Topic :: Scientific/Engineering :: Artificial Intelligence" in classifiers
    assert not [c for c in classifiers if c.startswith("License ::")]
