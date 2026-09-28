from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rag_sentinel.exceptions import StorageError
from rag_sentinel.gate import QualityGate
from rag_sentinel.models import MetricName, Provenance
from rag_sentinel.storage import MIGRATIONS, RunRepository

from .conftest import make_report

F = MetricName.FAITHFULNESS


@pytest.fixture
def repo(tmp_path: Path) -> RunRepository:
    return RunRepository(tmp_path / "nested" / "history.db")


def _save(repo: RunRepository, run_id: str, value: float, *, ref: str, days_ago: int) -> None:
    report = make_report({F: value}, run_id=run_id).model_copy(
        update={
            "created_at": datetime.now(UTC) - timedelta(days=days_ago),
            "provenance": Provenance(git_ref=ref, git_sha="deadbeef"),
        }
    )
    gate = QualityGate(thresholds={F: 0.85})
    repo.save_run(report, gate.evaluate(report), thresholds={"faithfulness": 0.85})


def test_migrations_are_applied_once(repo: RunRepository) -> None:
    RunRepository(repo.db_path)  # re-open: must not re-run migrations
    with sqlite3.connect(repo.db_path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == len(MIGRATIONS)


def test_round_trip_run_and_gate(repo: RunRepository) -> None:
    _save(repo, "r1", 0.9, ref="main", days_ago=1)
    report = repo.get_run("r1")
    gate = repo.get_gate("r1")
    assert report is not None and gate is not None
    assert report.aggregates[F] == 0.9
    assert len(report.samples) == 4
    assert gate.passed
    assert repo.get_run("missing") is None
    assert repo.get_gate("missing") is None


def test_list_runs_filters_and_orders(repo: RunRepository) -> None:
    _save(repo, "old-main", 0.9, ref="main", days_ago=3)
    _save(repo, "feature", 0.7, ref="feature/x", days_ago=2)
    _save(repo, "new-main", 0.95, ref="main", days_ago=1)

    assert [r.run_id for r in repo.list_runs()] == ["new-main", "feature", "old-main"]
    assert [r.run_id for r in repo.list_runs(git_ref="main")] == ["new-main", "old-main"]
    feature = repo.list_runs(git_ref="feature/x")[0]
    assert feature.gate_passed is False
    assert feature.aggregates == {"faithfulness": 0.7}
    assert repo.datasets() == ["golden_set"]


def test_latest_run_is_the_baseline(repo: RunRepository) -> None:
    _save(repo, "old-main", 0.9, ref="main", days_ago=3)
    _save(repo, "new-main", 0.95, ref="main", days_ago=1)
    baseline = repo.latest_run(dataset="golden_set", git_ref="main", exclude_run_id="new-main")
    assert baseline is not None and baseline.run_id == "old-main"
    assert repo.latest_run(dataset="other", git_ref="main") is None


def test_metric_history_is_chronological_with_thresholds(repo: RunRepository) -> None:
    _save(repo, "b", 0.8, ref="main", days_ago=1)
    _save(repo, "a", 0.9, ref="main", days_ago=2)
    points = repo.metric_history(dataset="golden_set")
    assert [p.run_id for p in points] == ["a", "b"]
    assert points[0].threshold == 0.85
    assert points[1].gate_passed is False


def test_duplicate_run_id_raises_storage_error(repo: RunRepository) -> None:
    _save(repo, "dup", 0.9, ref="main", days_ago=1)
    with pytest.raises(StorageError):
        _save(repo, "dup", 0.9, ref="main", days_ago=1)
