from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rag_sentinel.config import GateSettings
from rag_sentinel.exceptions import ConfigurationError
from rag_sentinel.gate import QualityGate, parse_threshold_overrides
from rag_sentinel.models import MetricDetail, MetricName, MetricResult
from rag_sentinel.report import COMMENT_MARKER, render_markdown, write_json
from rag_sentinel.storage import RunSummary

from .conftest import make_report

F = MetricName.FAITHFULNESS
R = MetricName.ANSWER_RELEVANCE


def gate(**kwargs: object) -> QualityGate:
    return QualityGate(thresholds={F: 0.85}, **kwargs)  # type: ignore[arg-type]


def test_gate_passes_at_threshold() -> None:
    result = gate().evaluate(make_report({F: 0.85}))
    assert result.passed
    assert result.violations == []


def test_gate_fails_below_faithfulness_threshold() -> None:
    result = gate().evaluate(make_report({F: 0.84}))
    assert not result.passed
    (violation,) = result.violations
    assert violation.metric == "faithfulness"
    assert violation.actual == 0.84
    assert "below the threshold 0.85" in violation.message


def test_gate_fails_when_metric_was_not_evaluated() -> None:
    result = gate().evaluate(make_report({R: 0.9}))
    assert not result.passed
    assert "not evaluated" in result.violations[0].message


def test_gate_fails_when_no_sample_was_scored() -> None:
    result = gate().evaluate(make_report({F: None}))
    assert "no sample produced a score" in result.violations[0].message


def test_gate_enforces_error_budget() -> None:
    result = gate(max_error_rate=0.2).evaluate(make_report({F: 0.95}, errors=2, total=4))
    assert not result.passed
    assert result.violations[0].metric == "error_rate"


def test_gate_reports_all_violations_and_warnings() -> None:
    g = QualityGate(thresholds={F: 0.85}, warn_thresholds={R: 0.8}, max_error_rate=0.1)
    result = g.evaluate(make_report({F: 0.5, R: 0.6}, errors=1, total=4))
    assert {v.metric for v in result.violations} == {"faithfulness", "error_rate"}
    assert result.warnings == ["answer_relevance 0.600 is below the advisory 0.80"]


def test_gate_from_settings_applies_overrides() -> None:
    g = QualityGate.from_settings(
        GateSettings(), overrides={"faithfulness": 0.9}, max_error_rate=0.0
    )
    assert g.thresholds == {F: 0.9}
    assert g.max_error_rate == 0.0
    assert MetricName.CONTEXT_RECALL in g.warn_thresholds


@pytest.mark.parametrize(
    "item", ["faithfulness", "faithfulness=abc", "bogus=0.5", "faithfulness=1.5"]
)
def test_parse_threshold_overrides_rejects_bad_input(item: str) -> None:
    with pytest.raises(ConfigurationError):
        parse_threshold_overrides([item])


def test_parse_threshold_overrides() -> None:
    assert parse_threshold_overrides(["faithfulness=0.9", " context_recall =0.7"]) == {
        "faithfulness": 0.9,
        "context_recall": 0.7,
    }


def test_markdown_report_contains_scores_deltas_and_evidence() -> None:
    report = make_report({F: 0.7, R: 0.9})
    worst = report.samples[0].model_copy(
        update={
            "metrics": {
                F: MetricResult(
                    name=F,
                    score=0.5,
                    details=[
                        MetricDetail(
                            item="Refunds take 2 days.", score=0.0, reason="not in context"
                        ),
                        MetricDetail(item="Window is 14 days.", score=1.0),
                    ],
                )
            }
        }
    )
    report = report.model_copy(update={"samples": [worst, *report.samples[1:]]})
    baseline = RunSummary(
        run_id="base",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        dataset="golden_set",
        git_sha="abc",
        git_ref="main",
        trigger="ci:push",
        judge_model="j",
        generator_model="g",
        sample_count=4,
        error_count=0,
        gate_passed=True,
        duration_s=1.0,
        aggregates={"faithfulness": 0.9},
    )
    result = gate().evaluate(report)
    md = render_markdown(report, result, thresholds={F: 0.85}, baseline=baseline)

    assert md.startswith(COMMENT_MARKER)
    assert "## RAG Sentinel: quality gate failed ❌" in md
    assert "| Faithfulness | 0.700 | ▼ -0.200 | ≥ 0.85 | ❌ |" in md
    assert "| Answer relevance | 0.900 | — | — | n/a |" in md
    assert "Refunds take 2 days." in md
    assert "Window is 14 days." not in md.split("Lowest-faithfulness")[1]
    assert "run `base` on `main`" in md


def test_markdown_report_without_baseline_and_with_errors() -> None:
    report = make_report({F: 0.9}, errors=1, total=3)
    md = render_markdown(report, gate().evaluate(report), thresholds={F: 0.85})
    assert "No baseline run on `main` yet" in md
    assert "Samples with errors" in md
    assert "✅" in md


def test_write_json(tmp_path: Path) -> None:
    report = make_report({F: 0.9})
    target = tmp_path / "out" / "report.json"
    write_json(report, gate().evaluate(report), target)
    payload = json.loads(target.read_text())
    assert payload["gate"]["passed"] is True
    assert payload["report"]["aggregates"]["faithfulness"] == 0.9
