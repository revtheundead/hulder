"""The example Prometheus rules (examples/prometheus/hulder-rules.yml) load as YAML, and every metric
they name is one the runtime exposes, or vLLM's. Nothing here runs Prometheus or promtool."""

import re
from pathlib import Path

import pytest

from hulder.runtime.scoring import exposed_metric_names

yaml = pytest.importorskip("yaml")

RULES = Path(__file__).parents[1] / "examples" / "prometheus" / "hulder-rules.yml"
VLLM_METRICS = {"vllm:request_success_total"}


def expressions() -> list[str]:
    groups = yaml.safe_load(RULES.read_text())["groups"]
    return [rule["expr"] for group in groups for rule in group["rules"]]


def named(expr: str) -> set[str]:
    return set(re.findall(r"\b(?:hulder_[a-z_]+|vllm:[a-z_:]+)", expr))


def test_every_rule_is_an_alert_with_an_expression_and_a_summary():
    for group in yaml.safe_load(RULES.read_text())["groups"]:
        for rule in group["rules"]:
            assert rule["alert"] and rule["expr"] and rule["annotations"]["summary"]


def test_every_metric_the_rules_name_is_on_metrics():
    used = set().union(*map(named, expressions()))
    assert used <= exposed_metric_names() | VLLM_METRICS, used - exposed_metric_names() - VLLM_METRICS


def test_the_draft_counter_has_a_rule():
    assert any("hulder_draft_tokens_total" in expr for expr in expressions())


@pytest.mark.vllm
def test_vllms_metrics_the_rules_name_exist_at_the_pin():
    vllm = pytest.importorskip("vllm")
    source = (Path(vllm.__file__).parent / "v1" / "metrics" / "loggers.py").read_text()
    for name in VLLM_METRICS:  # counters are registered without _total
        assert f'name="{name.removesuffix("_total")}"' in source
