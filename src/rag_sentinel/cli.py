"""Command-line interface — the composition root used by CI (see docs/LLD.md §11).

Exit codes: 0 gate passed · 1 gate failed · 2 configuration error · 3 infrastructure error.
"""

from __future__ import annotations

import asyncio
import random
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from enum import IntEnum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from rag_sentinel import __version__
from rag_sentinel.config import Settings, get_settings
from rag_sentinel.datasets import dataset_name, load_golden_set
from rag_sentinel.evaluator import Evaluator, aggregate, new_run_id
from rag_sentinel.exceptions import (
    ConfigurationError,
    DatasetError,
    JudgeConfigurationError,
    PipelineError,
    SentinelError,
)
from rag_sentinel.gate import QualityGate, parse_threshold_overrides
from rag_sentinel.llm.embeddings import build_embeddings, embedding_model_label
from rag_sentinel.llm.factory import build_judge
from rag_sentinel.log import configure_logging, get_logger
from rag_sentinel.metrics import build_metrics
from rag_sentinel.models import (
    EvalSample,
    MetricDetail,
    MetricName,
    MetricResult,
    Provenance,
    RAGResponse,
    RetrievedContext,
    RunConfig,
    RunReport,
    SampleResult,
)
from rag_sentinel.pipeline import LangChainRAGPipeline
from rag_sentinel.provenance import detect_provenance
from rag_sentinel.report import metric_label, render_markdown, write_json, write_text
from rag_sentinel.storage import RunRepository

log = get_logger(__name__)
console = Console()

app = typer.Typer(
    name="rag-sentinel",
    help="CI quality gates and regression tracking for RAG pipelines.",
    no_args_is_help=True,
    add_completion=False,
)


class ExitCode(IntEnum):
    OK = 0
    GATE_FAILED = 1
    CONFIG_ERROR = 2
    INFRA_ERROR = 3


def exit_code_for(exc: SentinelError) -> ExitCode:
    if isinstance(exc, ConfigurationError | DatasetError | JudgeConfigurationError):
        return ExitCode.CONFIG_ERROR
    return ExitCode.INFRA_ERROR


def _load_settings() -> Settings:
    try:
        settings = get_settings()
    except Exception as exc:  # pydantic ValidationError from env vars
        console.print(f"[red]Invalid configuration:[/red] {exc}")
        raise typer.Exit(ExitCode.CONFIG_ERROR) from exc
    configure_logging(settings.log_level, json_logs=settings.log_json)
    return settings


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"rag-sentinel {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """rag-sentinel: evaluate RAG pipelines with open-source LLM judges."""


# ----------------------------------------------------------------------- evaluate ---


async def _run_evaluation(
    settings: Settings,
    samples: list[EvalSample],
    metric_names: list[MetricName],
    dataset: str,
    provenance: Provenance,
) -> RunReport:
    async with build_judge(settings.judge) as judge:
        await judge.healthcheck()
        embeddings = build_embeddings(settings.embeddings)
        pipeline = await asyncio.to_thread(
            LangChainRAGPipeline.from_settings, settings.pipeline, embeddings
        )
        metrics = build_metrics(
            metric_names, judge=judge, embeddings=embeddings, settings=settings.evaluation
        )
        evaluator = Evaluator(
            pipeline, metrics, max_concurrency=settings.evaluation.max_concurrency
        )
        config = RunConfig(
            judge_model=judge.model_name,
            generator_model=pipeline.generator_label,
            embedding_model=embedding_model_label(settings.embeddings),
            top_k=settings.pipeline.top_k,
            metrics=evaluator.metric_names,
        )

        def progress(done: int, total: int) -> None:
            log.info("run.progress", done=done, total=total)

        report = await evaluator.run(
            samples, dataset=dataset, config=config, provenance=provenance, on_progress=progress
        )

    if report.samples and all(s.error for s in report.samples):
        msg = f"the pipeline failed on every sample; first error: {report.samples[0].error}"
        raise PipelineError(msg)
    return report


@app.command()
def evaluate(
    dataset: Annotated[Path, typer.Option(help="Golden dataset (.jsonl or .json).")] = Path(
        "data/golden/golden_set.jsonl"
    ),
    corpus: Annotated[
        Path | None,
        typer.Option(help="Corpus directory (overrides SENTINEL_PIPELINE__CORPUS_DIR)."),
    ] = None,
    metric: Annotated[
        list[MetricName] | None,
        typer.Option("--metric", "-m", help="Metric to compute (repeatable). Default: all."),
    ] = None,
    threshold: Annotated[
        list[str] | None,
        typer.Option("--threshold", "-t", help="Blocking threshold METRIC=VALUE (repeatable)."),
    ] = None,
    max_error_rate: Annotated[
        float | None, typer.Option(min=0.0, max=1.0, help="Maximum share of failed samples.")
    ] = None,
    limit: Annotated[
        int | None, typer.Option(min=1, help="Evaluate only the first N samples.")
    ] = None,
    db: Annotated[Path | None, typer.Option(help="SQLite history database path.")] = None,
    save: Annotated[bool, typer.Option(help="Persist the run to the history database.")] = True,
    output_json: Annotated[
        Path | None, typer.Option(help="Write the full JSON report here.")
    ] = None,
    output_markdown: Annotated[
        Path | None, typer.Option(help="Write the Markdown summary here.")
    ] = None,
    baseline_ref: Annotated[
        str, typer.Option(help="Git ref whose latest run is the baseline.")
    ] = "main",
    report_only: Annotated[
        bool, typer.Option("--report-only", help="Always exit 0 on gate failure (still reports).")
    ] = False,
) -> None:
    """Run the golden dataset through the pipeline, score it and enforce the quality gate."""
    settings = _load_settings()
    if corpus is not None:
        settings = settings.model_copy(
            update={"pipeline": settings.pipeline.model_copy(update={"corpus_dir": corpus})}
        )
    metric_names = metric or list(MetricName)

    try:
        gate = QualityGate.from_settings(
            settings.gate,
            overrides=parse_threshold_overrides(threshold or []),
            max_error_rate=max_error_rate,
        )
        samples = load_golden_set(dataset, limit=limit)
        name = dataset_name(dataset)
        report = asyncio.run(
            _run_evaluation(settings, samples, metric_names, name, detect_provenance())
        )
        verdict = gate.evaluate(report)

        baseline = None
        if save:
            repo = RunRepository(db or settings.storage.db_path)
            baseline = repo.latest_run(
                dataset=name, git_ref=baseline_ref, exclude_run_id=report.run_id
            )
            repo.save_run(
                report, verdict, thresholds={k.value: v for k, v in gate.thresholds.items()}
            )
    except SentinelError as exc:
        code = exit_code_for(exc)
        log.error(
            "evaluate.failed", error=str(exc), error_type=type(exc).__name__, exit_code=int(code)
        )
        console.print(f"[red]{type(exc).__name__}:[/red] {exc}")
        raise typer.Exit(code) from exc

    markdown = render_markdown(report, verdict, thresholds=gate.thresholds, baseline=baseline)
    if output_json:
        write_json(report, verdict, output_json)
    if output_markdown:
        write_text(markdown, output_markdown)
    _print_summary(report, verdict.passed, gate.thresholds)

    if verdict.passed:
        raise typer.Exit(ExitCode.OK)
    for violation in verdict.violations:
        console.print(f"[red]✗[/red] {violation.message}")
    raise typer.Exit(ExitCode.OK if report_only else ExitCode.GATE_FAILED)


def _print_summary(report: RunReport, passed: bool, thresholds: Mapping[MetricName, float]) -> None:
    table = Table(title=f"RAG Sentinel run {report.run_id} — {report.dataset}")
    table.add_column("Metric")
    table.add_column("Score", justify="right")
    table.add_column("Threshold", justify="right")
    for name in report.config.metrics:
        value = report.aggregates.get(name)
        limit = thresholds.get(name)
        table.add_row(
            metric_label(name),
            "—" if value is None else f"{value:.3f}",
            "—" if limit is None else f"≥ {limit:.2f}",
        )
    console.print(table)
    console.print(
        f"Samples: {report.sample_count} · errors: {report.error_count} · "
        f"duration: {report.duration_s:.1f}s · gate: "
        + ("[green]PASSED[/green]" if passed else "[red]FAILED[/red]")
    )


# ------------------------------------------------------------------------ history ---


@app.command()
def history(
    dataset: Annotated[str | None, typer.Option(help="Filter by dataset name.")] = None,
    limit: Annotated[int, typer.Option(min=1, help="Number of runs to show.")] = 20,
    db: Annotated[Path | None, typer.Option(help="SQLite history database path.")] = None,
) -> None:
    """Show recent evaluation runs."""
    settings = _load_settings()
    try:
        runs = RunRepository(db or settings.storage.db_path).list_runs(dataset=dataset, limit=limit)
    except SentinelError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(exit_code_for(exc)) from exc
    if not runs:
        console.print(
            "No runs recorded yet. Try `rag-sentinel evaluate` or `rag-sentinel seed-demo`."
        )
        return
    table = Table(title="Evaluation history (newest first)")
    for column in (
        "Run",
        "When (UTC)",
        "Dataset",
        "Ref",
        *(metric_label(m) for m in MetricName),
        "Gate",
    ):
        table.add_column(column)
    for run in runs:
        table.add_row(
            run.run_id,
            f"{run.created_at:%Y-%m-%d %H:%M}",
            run.dataset,
            run.git_ref or "—",
            *(
                "—" if (v := run.aggregates.get(m.value)) is None else f"{v:.3f}"
                for m in MetricName
            ),
            {True: "[green]pass[/green]", False: "[red]fail[/red]", None: "—"}[run.gate_passed],
        )
    console.print(table)


# ---------------------------------------------------------------------- seed-demo ---


def _synthetic_run(rng: random.Random, when: datetime, index: int, total: int) -> RunReport:
    """A plausible run whose faithfulness dips below the gate part-way through."""
    dip = total // 2 <= index < total // 2 + 2
    base = {
        MetricName.FAITHFULNESS: 0.78 if dip else 0.9 + 0.05 * index / max(total, 1),
        MetricName.ANSWER_RELEVANCE: 0.84,
        MetricName.CONTEXT_PRECISION: 0.74 + 0.1 * index / max(total, 1),
        MetricName.CONTEXT_RECALL: 0.8,
    }
    samples: list[SampleResult] = []
    for i in range(8):
        sample = EvalSample(
            id=f"demo-{i:02d}", question=f"Demo question {i + 1}?", ground_truth="Demo."
        )
        metrics = {
            name: MetricResult(
                name=name,
                score=round(min(1.0, max(0.0, rng.gauss(mean, 0.06))), 3),
                reason="synthetic demo value",
                details=[MetricDetail(item="Synthetic statement.", score=1.0)],
            )
            for name, mean in base.items()
        }
        response = RAGResponse(
            question=sample.question,
            answer="Synthetic demo answer.",
            contexts=[RetrievedContext(content="Synthetic context.", source="demo.md", rank=1)],
            latency_ms=round(rng.uniform(300, 1500), 1),
        )
        samples.append(SampleResult(sample=sample, response=response, metrics=metrics))
    names = list(MetricName)
    return RunReport(
        run_id=new_run_id(),
        created_at=when,
        dataset="demo",
        provenance=Provenance(
            git_sha=f"{rng.getrandbits(160):040x}",
            git_ref="main" if index % 3 else f"feature/change-{index}",
            trigger="seed-demo",
        ),
        config=RunConfig(
            judge_model="demo-judge",
            generator_model="demo-generator",
            embedding_model="demo-embeddings",
            top_k=4,
            metrics=names,
        ),
        samples=samples,
        aggregates=aggregate(samples, names),
        duration_s=round(rng.uniform(60, 240), 1),
    )


@app.command("seed-demo")
def seed_demo(
    runs: Annotated[int, typer.Option(min=2, max=365, help="Number of synthetic runs.")] = 14,
    db: Annotated[Path | None, typer.Option(help="SQLite history database path.")] = None,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 7,
) -> None:
    """Populate the history DB with synthetic runs (dataset 'demo') to explore the dashboard."""
    settings = _load_settings()
    repo = RunRepository(db or settings.storage.db_path)
    gate = QualityGate.from_settings(settings.gate)
    rng = random.Random(seed)  # noqa: S311 - not cryptographic
    start = datetime.now(UTC) - timedelta(days=runs)
    for index in range(runs):
        report = _synthetic_run(rng, start + timedelta(days=index), index, runs)
        repo.save_run(
            report,
            gate.evaluate(report),
            thresholds={k.value: v for k, v in gate.thresholds.items()},
        )
    console.print(f"Seeded {runs} demo runs into {repo.db_path}.")


# ---------------------------------------------------------------------- dashboard ---


@app.command()
def dashboard(
    port: Annotated[int, typer.Option(help="Port for the Streamlit server.")] = 8501,
) -> None:
    """Launch the Streamlit dashboard."""
    app_path = Path(__file__).with_name("app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)]
    raise typer.Exit(subprocess.call(cmd))  # noqa: S603 - fixed argv


if __name__ == "__main__":  # pragma: no cover
    app()
