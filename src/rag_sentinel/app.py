"""Streamlit dashboard (see docs/LLD.md §11.3).

Run with ``rag-sentinel dashboard`` or ``streamlit run src/rag_sentinel/app.py``.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from pathlib import Path
from typing import Any, TypeVar

import altair as alt
import pandas as pd
import streamlit as st

from rag_sentinel.config import Settings, get_settings
from rag_sentinel.evaluator import Evaluator
from rag_sentinel.exceptions import SentinelError
from rag_sentinel.llm.base import BaseHTTPJudge
from rag_sentinel.llm.embeddings import build_embeddings
from rag_sentinel.llm.factory import build_judge
from rag_sentinel.log import configure_logging
from rag_sentinel.metrics import build_metrics
from rag_sentinel.models import EvalSample, MetricName, RunReport, SampleResult
from rag_sentinel.pipeline import LangChainRAGPipeline
from rag_sentinel.report import metric_label
from rag_sentinel.storage import RunRepository

T = TypeVar("T")

# Categorical slots 1-4 of the reference palette, in fixed order (colour follows the metric).
METRIC_COLORS: dict[str, str] = {
    MetricName.FAITHFULNESS.value: "#2a78d6",
    MetricName.ANSWER_RELEVANCE.value: "#eb6834",
    MetricName.CONTEXT_PRECISION.value: "#1baf7a",
    MetricName.CONTEXT_RECALL.value: "#eda100",
}
THRESHOLD_RULE_COLOR = "#8a8984"


# ------------------------------------------------------------------ async bridge ---


class BackgroundLoop:
    """One long-lived event loop on a daemon thread.

    Async HTTP clients (the judge, ChatOllama) are bound to the loop that first uses them, and
    Streamlit reruns the script on new threads; funnelling every coroutine through one loop
    keeps cached clients valid across reruns.
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, name="sentinel-loop", daemon=True).start()

    def run(self, coro: Coroutine[Any, Any, T], timeout: float | None = None) -> T:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)


# ------------------------------------------------------------------- resources ---


@st.cache_resource
def _settings() -> Settings:
    settings = get_settings()
    configure_logging(settings.log_level, json_logs=settings.log_json)
    return settings


@st.cache_resource
def _loop() -> BackgroundLoop:
    return BackgroundLoop()


@st.cache_resource(show_spinner="Indexing the corpus into ChromaDB…")
def _pipeline(_settings: Settings, cache_key: str) -> LangChainRAGPipeline:
    return LangChainRAGPipeline.from_settings(
        _settings.pipeline, build_embeddings(_settings.embeddings)
    )


@st.cache_resource
def _judge(_settings: Settings, cache_key: str) -> BaseHTTPJudge:
    return build_judge(_settings.judge)


def _repo(db_path: Path) -> RunRepository | None:
    try:
        return RunRepository(db_path)
    except SentinelError as exc:
        st.error(f"Cannot open history database `{db_path}`: {exc}")
        return None


# ------------------------------------------------------------------ formatting ---


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _score_frame(samples: list[SampleResult], metrics: list[MetricName]) -> pd.DataFrame:
    rows = []
    for s in samples:
        row: dict[str, Any] = {"sample": s.sample.id, "question": s.sample.question}
        for m in metrics:
            result = s.metrics.get(m)
            row[metric_label(m)] = None if result is None else result.score
        row["error"] = s.error or "; ".join(
            f"{k.value}: {v.error}" for k, v in s.metrics.items() if v.error
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _details_frame(sample: SampleResult, metric: MetricName) -> pd.DataFrame | None:
    result = sample.metrics.get(metric)
    if result is None or not result.details:
        return None
    return pd.DataFrame(
        [
            {
                "item": d.item,
                "verdict": ("✅ supported" if d.score >= 1.0 else "❌ unsupported")
                if metric is not MetricName.ANSWER_RELEVANCE
                else f"{d.score:.3f}",
                "reason": d.reason,
            }
            for d in result.details
        ]
    )


# ------------------------------------------------------------------------ tabs ---


def _history_chart(
    points: pd.DataFrame, thresholds: dict[str, float]
) -> alt.LayerChart | alt.FacetChart:
    metrics = [m for m in METRIC_COLORS if m in set(points["metric"])]
    color = alt.Color(
        "label:N",
        title="Metric",
        scale=alt.Scale(
            domain=[metric_label(m) for m in metrics], range=[METRIC_COLORS[m] for m in metrics]
        ),
        legend=alt.Legend(orient="top"),
    )
    hover = alt.selection_point(fields=["created_at"], nearest=True, on="pointerover", empty=False)
    base = alt.Chart(points).encode(
        x=alt.X("created_at:T", title=None),
        y=alt.Y("value:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
        color=color,
    )
    lines = base.mark_line(strokeWidth=2)
    dots = (
        base.mark_point(size=64, filled=True)
        .encode(
            opacity=alt.condition(hover, alt.value(1), alt.value(0.35)),
            tooltip=[
                alt.Tooltip("label:N", title="Metric"),
                alt.Tooltip("value:Q", title="Score", format=".3f"),
                alt.Tooltip("created_at:T", title="When", format="%Y-%m-%d %H:%M"),
                alt.Tooltip("git_ref:N", title="Ref"),
                alt.Tooltip("run_id:N", title="Run"),
                alt.Tooltip("gate:N", title="Gate"),
            ],
        )
        .add_params(hover)
    )
    rule_data = pd.DataFrame(
        [{"label": f"{metric_label(k)} gate {v:.2f}", "value": v} for k, v in thresholds.items()]
    )
    layers: list[alt.Chart] = [lines, dots]
    if not rule_data.empty:
        rules = (
            alt.Chart(rule_data)
            .mark_rule(strokeDash=[6, 4], strokeWidth=1.5, color=THRESHOLD_RULE_COLOR)
            .encode(y="value:Q", tooltip=["label:N"])
        )
        labels = (
            alt.Chart(rule_data)
            .mark_text(align="left", dx=4, dy=-6, color=THRESHOLD_RULE_COLOR)
            .encode(y="value:Q", x=alt.value(0), text="label:N")
        )
        layers += [rules, labels]
    return alt.layer(*layers).properties(height=360)


def history_tab(repo: RunRepository, dataset: str | None, thresholds: dict[str, float]) -> None:
    runs = repo.list_runs(dataset=dataset, limit=200)
    if not runs:
        st.info(
            "No runs recorded yet. Run `rag-sentinel evaluate`, or `rag-sentinel seed-demo` "
            "to explore the dashboard with synthetic history."
        )
        return

    latest, previous = runs[0], (runs[1] if len(runs) > 1 else None)
    st.caption(
        f"Latest run `{latest.run_id}` on `{latest.git_ref or '—'}` · "
        f"{latest.created_at:%Y-%m-%d %H:%M} UTC · gate "
        + {True: "✅ passed", False: "❌ failed", None: "not evaluated"}[latest.gate_passed]
    )
    columns = st.columns(len(MetricName))
    for column, metric in zip(columns, MetricName, strict=True):
        value = latest.aggregates.get(metric.value)
        before = previous.aggregates.get(metric.value) if previous else None
        delta = None if value is None or before is None else round(value - before, 3)
        column.metric(metric_label(metric), _fmt(value), delta=delta)

    points = pd.DataFrame([p.model_dump() for p in repo.metric_history(dataset=dataset)])
    points = points.dropna(subset=["value"])
    if points.empty:
        st.warning("Runs exist but none has metric values yet.")
        return
    points["label"] = points["metric"].map(metric_label)
    points["gate"] = points["gate_passed"].map({True: "passed", False: "failed"}).fillna("—")
    points["git_ref"] = points["git_ref"].fillna("—")
    st.subheader("Metric trend")
    st.altair_chart(_history_chart(points, thresholds), width="stretch")

    st.subheader("Runs")
    table = pd.DataFrame(
        [
            {
                "run": r.run_id,
                "when (UTC)": r.created_at.strftime("%Y-%m-%d %H:%M"),
                "ref": r.git_ref,
                "sha": (r.git_sha or "")[:8],
                **{metric_label(m): r.aggregates.get(m.value) for m in MetricName},
                "errors": f"{r.error_count}/{r.sample_count}",
                "gate": {True: "✅", False: "❌", None: "—"}[r.gate_passed],
                "judge": r.judge_model,
            }
            for r in runs
        ]
    )
    st.dataframe(table, hide_index=True, width="stretch")


def _sample_panel(sample: SampleResult, metrics: list[MetricName]) -> None:
    if sample.error:
        st.error(sample.error)
    if sample.response is None:
        return
    st.markdown(f"**Answer**\n\n{sample.response.answer}")
    if sample.sample.ground_truth:
        st.markdown(f"**Reference answer**\n\n{sample.sample.ground_truth}")
    for metric in metrics:
        result = sample.metrics.get(metric)
        if result is None:
            continue
        st.markdown(f"**{metric_label(metric)}: {_fmt(result.score)}** — {result.reason or ''}")
        if result.error:
            st.error(result.error)
        frame = _details_frame(sample, metric)
        if frame is not None:
            st.dataframe(frame, hide_index=True, width="stretch")
    with st.expander(f"Retrieved context ({len(sample.response.contexts)} chunks)"):
        for ctx in sample.response.contexts:
            st.markdown(f"**#{ctx.rank}** `{ctx.source}` · similarity {_fmt(ctx.score)}")
            st.text(ctx.content)


def explorer_tab(repo: RunRepository, dataset: str | None) -> None:
    runs = repo.list_runs(dataset=dataset, limit=200)
    if not runs:
        st.info("No runs to explore yet.")
        return
    labels = {
        r.run_id: f"{r.created_at:%Y-%m-%d %H:%M} · {r.run_id} · {r.git_ref or '—'}" for r in runs
    }
    run_id = st.selectbox("Run", options=list(labels), format_func=labels.__getitem__)
    report: RunReport | None = repo.get_run(run_id) if run_id else None
    if report is None:
        st.warning("Run not found.")
        return
    gate = repo.get_gate(report.run_id)
    if gate is not None:
        if gate.passed:
            st.success("Quality gate passed")
        else:
            st.error(
                "Quality gate failed:\n\n" + "\n".join(f"- {v.message}" for v in gate.violations)
            )
        for warning in gate.warnings:
            st.warning(warning)

    st.caption(
        f"judge `{report.config.judge_model}` · generator `{report.config.generator_model}` · "
        f"embeddings `{report.config.embedding_model}` · top_k {report.config.top_k} · "
        f"{report.duration_s:.0f}s"
    )
    frame = _score_frame(report.samples, report.config.metrics)
    st.dataframe(frame, hide_index=True, width="stretch")

    by_id = {s.sample.id: s for s in report.samples}
    if by_id:
        chosen = st.selectbox("Inspect sample", options=list(by_id))
        if chosen:
            st.markdown(f"#### {by_id[chosen].sample.question}")
            _sample_panel(by_id[chosen], report.config.metrics)


def live_tab(settings: Settings) -> None:
    st.markdown(
        "Ask the reference pipeline a question and score its answer in real time. "
        f"Judge: `{settings.judge.model}` via `{settings.judge.provider}`."
    )
    with st.form("live"):
        question = st.text_input(
            "Query", placeholder="How long do customers have to request a refund?", key="live_query"
        )
        ground_truth = st.text_area(
            "Reference answer (optional — enables context recall)", height=80, key="live_reference"
        )
        extra = st.multiselect(
            "Additional metrics",
            options=[m for m in MetricName if m is not MetricName.FAITHFULNESS],
            format_func=metric_label,
        )
        submitted = st.form_submit_button("Score it", type="primary")
    if not submitted:
        return
    if not question.strip():
        st.warning("Enter a query first.")
        return

    cache_key = settings.model_dump_json()
    try:
        pipeline = _pipeline(settings, cache_key)
        judge = _judge(settings, cache_key)
    except SentinelError as exc:
        st.error(f"Could not start the pipeline: {exc}")
        return

    metrics = build_metrics(
        [MetricName.FAITHFULNESS, *extra],
        judge=judge,
        embeddings=build_embeddings(settings.embeddings),
        settings=settings.evaluation,
    )
    evaluator = Evaluator(pipeline, metrics)
    sample = EvalSample(
        id="live", question=question.strip(), ground_truth=ground_truth.strip() or None
    )
    with st.spinner("Retrieving, generating and judging…"):
        try:
            result = _loop().run(
                evaluator.evaluate_sample(sample), timeout=settings.judge.timeout_s * 4
            )
        except TimeoutError:
            st.error("Timed out waiting for the judge. Is Ollama/vLLM running?")
            return

    faith = result.metrics.get(MetricName.FAITHFULNESS)
    threshold = settings.gate.thresholds.get(MetricName.FAITHFULNESS.value)
    if faith is not None:
        left, right = st.columns([1, 2])
        with left:
            st.metric("Faithfulness", _fmt(faith.score))
            if faith.score is not None:
                st.progress(faith.score)
                if threshold is not None:
                    if faith.score >= threshold:
                        st.success(f"✅ Meets the CI gate (≥ {threshold:.2f})")
                    else:
                        st.error(f"❌ Would fail the CI gate (≥ {threshold:.2f})")
            elif faith.error:
                st.error(faith.error)
            else:
                st.info(faith.reason)
        with right:
            for metric in extra:
                r = result.metrics.get(metric)
                if r is not None:
                    st.metric(metric_label(metric), _fmt(r.score), help=r.error or r.reason)
    _sample_panel(result, [MetricName.FAITHFULNESS, *extra])


# ------------------------------------------------------------------------ main ---


def main() -> None:
    st.set_page_config(page_title="RAG Sentinel", layout="wide")
    settings = _settings()
    st.title("RAG Sentinel")

    with st.sidebar:
        st.header("Settings")
        db_path = Path(
            st.text_input("History database", value=str(settings.storage.db_path), key="db_path")
        )
        repo = _repo(db_path)
        datasets = repo.datasets() if repo else []
        dataset = st.selectbox("Dataset", options=datasets, index=0) if datasets else None
        st.caption("Blocking thresholds")
        st.json(settings.gate.thresholds)

    history, explorer, live = st.tabs(["Regression history", "Run explorer", "Live scorecard"])
    with history:
        if repo:
            history_tab(repo, dataset, settings.gate.thresholds)
    with explorer:
        if repo:
            explorer_tab(repo, dataset)
    with live:
        live_tab(settings)


main()
