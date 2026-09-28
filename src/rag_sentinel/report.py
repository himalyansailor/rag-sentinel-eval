"""Render run results as JSON (machines) and GitHub-flavoured Markdown (PR comments)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from rag_sentinel.models import MetricName

if TYPE_CHECKING:
    from collections.abc import Mapping

    from rag_sentinel.models import GateResult, RunReport, SampleResult
    from rag_sentinel.storage import RunSummary

#: Hidden marker so CI can find and update ("sticky") its previous PR comment.
COMMENT_MARKER = "<!-- rag-sentinel-report -->"

_LABELS: dict[MetricName, str] = {
    MetricName.FAITHFULNESS: "Faithfulness",
    MetricName.ANSWER_RELEVANCE: "Answer relevance",
    MetricName.CONTEXT_PRECISION: "Context precision",
    MetricName.CONTEXT_RECALL: "Context recall",
}


def metric_label(name: MetricName | str) -> str:
    try:
        return _LABELS[MetricName(name)]
    except ValueError:
        return str(name)


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _delta(current: float | None, baseline: float | None) -> str:
    if current is None or baseline is None:
        return "—"
    diff = current - baseline
    if abs(diff) < 0.0005:
        return "±0.000"
    arrow = "▲" if diff > 0 else "▼"
    return f"{arrow} {diff:+.3f}"


def _escape(text: str, limit: int = 140) -> str:
    flat = " ".join(text.split()).replace("|", "\\|")
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _lowest(samples: list[SampleResult], metric: MetricName, n: int) -> list[SampleResult]:
    scored = [
        s for s in samples if (m := s.metrics.get(metric)) is not None and m.score is not None
    ]
    return sorted(scored, key=lambda s: s.metrics[metric].score or 0.0)[:n]


def render_markdown(
    report: RunReport,
    gate: GateResult,
    *,
    thresholds: Mapping[MetricName, float] | None = None,
    baseline: RunSummary | None = None,
    max_samples: int = 5,
) -> str:
    """Markdown summary suitable for a PR comment and ``$GITHUB_STEP_SUMMARY``."""
    thresholds = thresholds or {}
    status = "quality gate passed ✅" if gate.passed else "quality gate failed ❌"
    prov = report.provenance
    lines = [
        COMMENT_MARKER,
        f"## RAG Sentinel: {status}",
        "",
        f"Dataset **{report.dataset}** · {report.sample_count} samples · "
        f"judge `{report.config.judge_model}` · generator `{report.config.generator_model}` · "
        f"run `{report.run_id}` · {report.duration_s:.0f}s",
    ]
    if prov.git_sha:
        lines[-1] += f" · commit `{prov.git_sha[:8]}`"
    lines += [
        "",
        "| Metric | Score | Δ vs baseline | Threshold | Status |",
        "|---|---:|---:|---:|:---:|",
    ]

    for name in report.config.metrics:
        value = report.aggregates.get(name)
        base = baseline.aggregates.get(name.value) if baseline else None
        threshold = thresholds.get(name)
        if threshold is None:
            state = "n/a"
        elif value is not None and value >= threshold:
            state = "✅"
        else:
            state = "❌"
        lines.append(
            f"| {metric_label(name)} | {_fmt(value)} | {_delta(value, base)} | "
            f"{'—' if threshold is None else f'≥ {threshold:.2f}'} | {state} |"
        )

    lines.append("")
    if baseline:
        lines.append(
            f"Baseline: run `{baseline.run_id}` on `{baseline.git_ref or 'unknown'}` "
            f"({baseline.created_at:%Y-%m-%d %H:%M} UTC)."
        )
    else:
        lines.append("No baseline run found yet — deltas will appear once `main` has history.")
    lines.append(
        f"Errors: {report.error_count}/{report.sample_count} samples ({report.error_rate:.0%})."
    )

    if gate.violations:
        lines += ["", "### Gate violations", *(f"- {v.message}" for v in gate.violations)]
    if gate.warnings:
        lines += ["", "### Warnings", *(f"- ⚠️ {w}" for w in gate.warnings)]

    worst = _lowest(report.samples, MetricName.FAITHFULNESS, max_samples)
    if worst:
        lines += [
            "",
            "<details><summary>Lowest-faithfulness samples</summary>",
            "",
            "| Sample | Faithfulness | Unsupported statements |",
            "|---|---:|---|",
        ]
        for s in worst:
            result = s.metrics[MetricName.FAITHFULNESS]
            unsupported = [d.item for d in result.details if d.score < 1.0]
            lines.append(
                f"| `{s.sample.id}` {_escape(s.sample.question, 80)} | {_fmt(result.score)} | "
                f"{_escape('; '.join(unsupported)) if unsupported else '—'} |"
            )
        lines += ["", "</details>"]

    failed = [s for s in report.samples if s.has_error]
    if failed:
        lines += ["", "<details><summary>Samples with errors</summary>", ""]
        for s in failed[:max_samples]:
            errors = (
                [s.error]
                if s.error
                else [f"{k.value}: {m.error}" for k, m in s.metrics.items() if m.error]
            )
            lines.append(f"- `{s.sample.id}`: {_escape('; '.join(e for e in errors if e), 200)}")
        lines += ["", "</details>"]

    if prov.ci_run_url:
        lines += ["", f"[Workflow run]({prov.ci_run_url})"]
    return "\n".join(lines) + "\n"


def write_json(report: RunReport, gate: GateResult, path: Path) -> None:
    """Write the full report plus gate verdict as pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "report": json.loads(report.model_dump_json()),
        "gate": json.loads(gate.model_dump_json()),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(content: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
