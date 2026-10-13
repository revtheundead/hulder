"""The README's example is the real quickstart, and its prose quotes only numbers the quickstart prints."""

import re
from pathlib import Path

from hulder import cli

README = Path(__file__).parents[1] / "README.md"

# Numbers the example uses to explain, not to quote: the CI level and the exit codes. And the
# demo probe's training-set size, which the quickstart doesn't print (docs/results/phase1/probe_demo.md).
EXPLANATORY = {"95%", "0", "1", "2", "10", "11", "12", "600"}
NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:[.,]\d+)*%?(?![\w])")


def example() -> tuple[str, str]:
    """The output shown under "Example", and the prose around it."""
    section = re.search(r"^## Example\n(.*?)^## ", README.read_text(), re.DOTALL | re.MULTILINE).group(1)
    [output] = [
        body for lang, body in re.findall(r"^```(\w*)\n(.*?)\n```", section, re.DOTALL | re.MULTILINE) if not lang
    ]
    return output, re.sub(r"```.*?```", "", section, flags=re.DOTALL)


def quickstart(capsys) -> str:
    assert cli.run(["quickstart"]) == 0
    return capsys.readouterr().out


def test_the_examples_output_is_check_1_of_the_quickstart_as_printed(capsys):
    output, _ = example()
    assert output.startswith("=== Check 1")
    assert output + "\n" in quickstart(capsys)


def test_every_number_the_example_quotes_is_in_the_quickstart_output(capsys):
    _, prose = example()
    printed = set(re.findall(r"[-+]?\d+(?:[.,]\d+)*%?", quickstart(capsys)))
    quoted = set(NUMBER.findall(prose)) - EXPLANATORY
    assert quoted, "the example quotes no numbers"
    assert sorted(quoted - printed) == []
