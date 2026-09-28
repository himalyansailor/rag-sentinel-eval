"""SQLite run-history repository (see docs/LLD.md §10)."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import closing, contextmanager
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from rag_sentinel.exceptions import StorageError
from rag_sentinel.log import get_logger
from rag_sentinel.models import GateResult, RunReport

log = get_logger(__name__)

MIGRATIONS: tuple[str, ...] = (
    # 1 — initial schema
    """
    CREATE TABLE runs (
        run_id          TEXT PRIMARY KEY,
        created_at      TEXT NOT NULL,
        dataset         TEXT NOT NULL,
        git_sha         TEXT,
        git_ref         TEXT,
        trigger         TEXT NOT NULL,
        judge_model     TEXT NOT NULL,
        generator_model TEXT NOT NULL,
        sample_count    INTEGER NOT NULL,
        error_count     INTEGER NOT NULL,
        gate_passed     INTEGER,
        duration_s      REAL NOT NULL,
        report_json     TEXT NOT NULL,
        gate_json       TEXT
    );
    CREATE INDEX idx_runs_created_at ON runs (created_at);
    CREATE INDEX idx_runs_dataset_ref ON runs (dataset, git_ref, created_at);

    CREATE TABLE run_metrics (
        run_id    TEXT NOT NULL REFERENCES runs (run_id) ON DELETE CASCADE,
        metric    TEXT NOT NULL,
        value     REAL,
        threshold REAL,
        PRIMARY KEY (run_id, metric)
    );

    CREATE TABLE sample_results (
        run_id       TEXT NOT NULL REFERENCES runs (run_id) ON DELETE CASCADE,
        sample_id    TEXT NOT NULL,
        question     TEXT NOT NULL,
        answer       TEXT,
        error        TEXT,
        latency_ms   REAL,
        metrics_json TEXT NOT NULL,
        PRIMARY KEY (run_id, sample_id)
    );
    """,
)


class RunSummary(BaseModel):
    """One row of the run history, with aggregates pivoted into a dict."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    created_at: datetime
    dataset: str
    git_sha: str | None
    git_ref: str | None
    trigger: str
    judge_model: str
    generator_model: str
    sample_count: int
    error_count: int
    gate_passed: bool | None
    duration_s: float
    aggregates: dict[str, float | None]


class MetricPoint(BaseModel):
    """A single (run, metric) value for time-series charts."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    created_at: datetime
    dataset: str
    git_ref: str | None
    git_sha: str | None
    gate_passed: bool | None
    metric: str
    value: float | None
    threshold: float | None


class RunRepository:
    """Persists and queries evaluation runs. Safe to use from multiple threads."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        try:
            db_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            msg = f"cannot create directory for {db_path}: {exc}"
            raise StorageError(msg) from exc
        self._migrate()

    # ------------------------------------------------------------------ plumbing ---

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection inside a transaction; map sqlite errors to StorageError."""
        try:
            with closing(sqlite3.connect(self.db_path, timeout=5.0)) as conn:
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA foreign_keys = ON")
                conn.execute("PRAGMA busy_timeout = 5000")
                with conn:
                    yield conn
        except sqlite3.Error as exc:
            log.error("storage.error", db=str(self.db_path), error=str(exc))
            msg = f"SQLite error on {self.db_path}: {exc}"
            raise StorageError(msg) from exc

    def _migrate(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            version: int = conn.execute("PRAGMA user_version").fetchone()[0]
            for number, script in enumerate(MIGRATIONS[version:], start=version + 1):
                conn.executescript(script)
                conn.execute(f"PRAGMA user_version = {int(number)}")
                log.info("storage.migrated", db=str(self.db_path), version=number)

    # --------------------------------------------------------------------- writes ---

    def save_run(
        self,
        report: RunReport,
        gate: GateResult | None = None,
        thresholds: Mapping[str, float] | None = None,
    ) -> None:
        """Insert a run with its aggregates and per-sample results (atomic)."""
        thresholds = thresholds or {}
        prov = report.provenance
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (run_id, created_at, dataset, git_sha, git_ref, trigger,
                                  judge_model, generator_model, sample_count, error_count,
                                  gate_passed, duration_s, report_json, gate_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.run_id,
                    report.created_at.isoformat(),
                    report.dataset,
                    prov.git_sha,
                    prov.git_ref,
                    prov.trigger,
                    report.config.judge_model,
                    report.config.generator_model,
                    report.sample_count,
                    report.error_count,
                    None if gate is None else int(gate.passed),
                    report.duration_s,
                    report.model_dump_json(),
                    None if gate is None else gate.model_dump_json(),
                ),
            )
            conn.executemany(
                "INSERT INTO run_metrics (run_id, metric, value, threshold) VALUES (?, ?, ?, ?)",
                [
                    (report.run_id, name.value, value, thresholds.get(name.value))
                    for name, value in report.aggregates.items()
                ],
            )
            conn.executemany(
                """
                INSERT INTO sample_results (run_id, sample_id, question, answer, error,
                                            latency_ms, metrics_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        report.run_id,
                        s.sample.id,
                        s.sample.question,
                        s.response.answer if s.response else None,
                        s.error,
                        s.response.latency_ms if s.response else None,
                        json.dumps({k.value: v.score for k, v in s.metrics.items()}),
                    )
                    for s in report.samples
                ],
            )
        log.info("storage.run_saved", run_id=report.run_id, db=str(self.db_path))

    # ---------------------------------------------------------------------- reads ---

    def get_run(self, run_id: str) -> RunReport | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT report_json FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return None if row is None else RunReport.model_validate_json(row["report_json"])

    def get_gate(self, run_id: str) -> GateResult | None:
        with self._connect() as conn:
            row = conn.execute("SELECT gate_json FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None or row["gate_json"] is None:
            return None
        return GateResult.model_validate_json(row["gate_json"])

    def list_runs(
        self, *, dataset: str | None = None, git_ref: str | None = None, limit: int = 200
    ) -> list[RunSummary]:
        """Most recent runs first."""
        clauses: list[str] = []
        params: list[object] = []
        if dataset is not None:
            clauses.append("dataset = ?")
            params.append(dataset)
        if git_ref is not None:
            clauses.append("git_ref = ?")
            params.append(git_ref)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM runs {where} ORDER BY created_at DESC LIMIT ?",  # noqa: S608 - static clauses
                (*params, limit),
            ).fetchall()
            metrics = self._metrics_for(conn, [r["run_id"] for r in rows])
        return [self._summary(row, metrics.get(row["run_id"], {})) for row in rows]

    def latest_run(
        self, *, dataset: str, git_ref: str | None, exclude_run_id: str | None = None
    ) -> RunSummary | None:
        """The newest run for ``dataset`` on ``git_ref`` (the regression baseline)."""
        for run in self.list_runs(dataset=dataset, git_ref=git_ref, limit=5):
            if run.run_id != exclude_run_id:
                return run
        return None

    def metric_history(self, *, dataset: str | None = None, limit: int = 500) -> list[MetricPoint]:
        """Metric values of the most recent ``limit`` runs, oldest first (chart order)."""
        where = "WHERE r.dataset = ?" if dataset is not None else ""
        params: tuple[object, ...] = (dataset, limit) if dataset is not None else (limit,)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT r.run_id, r.created_at, r.dataset, r.git_ref, r.git_sha, r.gate_passed,
                       m.metric, m.value, m.threshold
                FROM (SELECT * FROM runs r {where} ORDER BY created_at DESC LIMIT ?) AS r
                JOIN run_metrics m ON m.run_id = r.run_id
                ORDER BY r.created_at ASC, m.metric ASC
                """,  # noqa: S608 - static clause, values are parameterised
                params,
            ).fetchall()
        return [
            MetricPoint(
                run_id=row["run_id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                dataset=row["dataset"],
                git_ref=row["git_ref"],
                git_sha=row["git_sha"],
                gate_passed=None if row["gate_passed"] is None else bool(row["gate_passed"]),
                metric=row["metric"],
                value=row["value"],
                threshold=row["threshold"],
            )
            for row in rows
        ]

    def datasets(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT DISTINCT dataset FROM runs ORDER BY dataset").fetchall()
        return [row["dataset"] for row in rows]

    # -------------------------------------------------------------------- helpers ---

    @staticmethod
    def _metrics_for(
        conn: sqlite3.Connection, run_ids: list[str]
    ) -> dict[str, dict[str, float | None]]:
        if not run_ids:
            return {}
        placeholders = ",".join("?" for _ in run_ids)
        rows = conn.execute(
            f"SELECT run_id, metric, value FROM run_metrics WHERE run_id IN ({placeholders})",  # noqa: S608
            run_ids,
        ).fetchall()
        out: dict[str, dict[str, float | None]] = {}
        for row in rows:
            out.setdefault(row["run_id"], {})[row["metric"]] = row["value"]
        return out

    @staticmethod
    def _summary(row: sqlite3.Row, aggregates: dict[str, float | None]) -> RunSummary:
        return RunSummary(
            run_id=row["run_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            dataset=row["dataset"],
            git_sha=row["git_sha"],
            git_ref=row["git_ref"],
            trigger=row["trigger"],
            judge_model=row["judge_model"],
            generator_model=row["generator_model"],
            sample_count=row["sample_count"],
            error_count=row["error_count"],
            gate_passed=None if row["gate_passed"] is None else bool(row["gate_passed"]),
            duration_s=row["duration_s"],
            aggregates=aggregates,
        )
