"""What the installed distribution declares: its licence, and the bundled data's attribution."""

from importlib.metadata import metadata
from importlib.resources import files


def test_hulder_is_licensed_apache_2():
    assert metadata("hulder")["License-Expression"] == "Apache-2.0"


def test_the_bundled_data_notice_links_its_cc_by_licence():
    notice = (files("hulder.data") / "NOTICE.md").read_text()
    assert "CC-BY-4.0" in notice
    assert "https://creativecommons.org/licenses/by/4.0/" in notice
