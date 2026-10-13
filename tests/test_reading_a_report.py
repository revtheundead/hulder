"""docs/reading-a-report.md quotes the real quickstart, and its hulder check command prints the same report."""

import re
from pathlib import Path

from hulder import cli

ROOT = Path(__file__).parents[1]
DOC = ROOT / "docs" / "reading-a-report.md"

# Numbers the walkthrough uses to explain, not to quote: the CI level and the AUROC scale.
EXPLANATORY = {"95%", "1.0", "0.5"}
NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:[.,]\d+)*%?(?![\w])")


def walkthrough() -> tuple[str, str]:
    """The report the walkthrough reads, and the prose around it, up to "Reproduce it"."""
    pattern = r"^# Reading a report\n(.*?)^## Reproduce it"
    section = re.search(pattern, DOC.read_text(), re.DOTALL | re.MULTILINE).group(1)
    [report] = re.findall(r"```\n(.*?)\n```", section, re.DOTALL)
    prose = re.sub(r"```.*?```", "", section, flags=re.DOTALL)
    return report, re.sub(r"(?m)^\d+\. ", "", prose)  # list markers aren't quoted numbers


def quickstart(capsys) -> str:
    assert cli.run(["quickstart"]) == 0
    return capsys.readouterr().out


def test_the_walkthroughs_report_is_check_2_of_the_quickstart_as_printed(capsys):
    report, _ = walkthrough()
    assert "\n" + report + "\n" in quickstart(capsys)


def test_every_number_the_walkthrough_quotes_is_in_the_quickstart_output(capsys):
    _, prose = walkthrough()
    printed = set(re.findall(r"[-+]?\d+(?:[.,]\d+)*%?", quickstart(capsys)))
    quoted = set(NUMBER.findall(prose)) - EXPLANATORY
    assert quoted, "the walkthrough quotes no numbers"
    assert sorted(quoted - printed) == []


def test_the_hulder_check_command_prints_the_walkthroughs_report(capsys, monkeypatch):
    [command] = re.findall(r"```\n\$ (hulder check .*?)\n```", DOC.read_text(), re.DOTALL)
    monkeypatch.chdir(ROOT)
    argv = command.replace("\\\n", " ").split()[1:]
    assert cli.run(argv) == 12
    report, _ = walkthrough()
    assert capsys.readouterr().out.rstrip("\n").replace("src/hulder/data/", "") == report
