from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from rag_sentinel import cli
from rag_sentinel.config import Settings, get_settings
from rag_sentinel.datasets import load_golden_set
from rag_sentinel.exceptions import DatasetError, JudgeConfigurationError, JudgeUnavailableError
from rag_sentinel.models import MetricName, RunReport
from rag_sentinel.provenance import detect_provenance
from rag_sentinel.storage import RunRepository

from .conftest import GOLDEN_SET, make_report

runner = CliRunner()


# --------------------------------------------------------------------- config ---


def test_settings_read_nested_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SENTINEL_JUDGE__MODEL", "qwen2.5:3b")
    monkeypatch.setenv("SENTINEL_GATE__THRESHOLDS", '{"faithfulness": 0.9}')
    settings = get_settings()
    assert settings.judge.model == "qwen2.5:3b"
    assert settings.gate.thresholds == {"faithfulness": 0.9}


def test_settings_reject_unknown_threshold_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SENTINEL_GATE__THRESHOLDS", '{"vibes": 0.9}')
    with pytest.raises(ValueError, match="unknown metric"):
        Settings()


# ------------------------------------------------------------------- datasets ---


def test_bundled_golden_set_is_valid() -> None:
    samples = load_golden_set(GOLDEN_SET)
    assert len(samples) == 10
    assert all(s.ground_truth for s in samples)
    assert load_golden_set(GOLDEN_SET, limit=2)[1].id == samples[1].id


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ('{"id": "a", "question": "q"}\n{"id": "a", "question": "q"}', "duplicate"),
        ('{"id": "a"}', "invalid sample"),
        ("{not json", "invalid JSON"),
        ("// only a comment\n", "no samples"),
    ],
)
def test_dataset_errors(tmp_path: Path, content: str, match: str) -> None:
    path = tmp_path / "set.jsonl"
    path.write_text(content)
    with pytest.raises(DatasetError, match=match):
        load_golden_set(path)


def test_json_array_datasets(tmp_path: Path) -> None:
    path = tmp_path / "set.json"
    path.write_text('[{"id": "a", "question": "q"}]')
    assert load_golden_set(path)[0].id == "a"
    path.write_text('{"id": "a"}')
    with pytest.raises(DatasetError, match="array"):
        load_golden_set(path)


# ----------------------------------------------------------------- provenance ---


def test_provenance_from_github_pull_request() -> None:
    prov = detect_provenance(
        {
            "GITHUB_ACTIONS": "true",
            "GITHUB_SHA": "abc123",
            "GITHUB_HEAD_REF": "feature/better-chunks",
            "GITHUB_REF_NAME": "42/merge",
            "GITHUB_EVENT_NAME": "pull_request",
            "GITHUB_SERVER_URL": "https://github.com",
            "GITHUB_REPOSITORY": "org/repo",
            "GITHUB_RUN_ID": "99",
        }
    )
    assert prov.git_ref == "feature/better-chunks"
    assert prov.trigger == "ci:pull_request"
    assert prov.ci_run_url == "https://github.com/org/repo/actions/runs/99"


def test_provenance_local_never_raises() -> None:
    assert detect_provenance({}, trigger="dashboard").trigger == "dashboard"


# ------------------------------------------------------------------------ cli ---


def _fake_evaluation(aggregate: float):  # type: ignore[no-untyped-def]
    async def fake(settings, samples, metric_names, dataset, provenance) -> RunReport:  # type: ignore[no-untyped-def]
        return make_report({MetricName.FAITHFULNESS: aggregate}, dataset=dataset)

    return fake


@pytest.mark.parametrize(("score", "code"), [(0.95, 0), (0.80, 1)])
def test_evaluate_exit_code_follows_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, score: float, code: int
) -> None:
    monkeypatch.setattr(cli, "_run_evaluation", _fake_evaluation(score))
    md, js, db = tmp_path / "summary.md", tmp_path / "report.json", tmp_path / "h.db"
    result = runner.invoke(
        cli.app,
        [
            "evaluate",
            "--dataset",
            str(GOLDEN_SET),
            "--db",
            str(db),
            "--output-markdown",
            str(md),
            "--output-json",
            str(js),
        ],
    )
    assert result.exit_code == code, result.output
    assert md.read_text().startswith("<!-- rag-sentinel-report -->")
    assert js.exists()
    assert len(RunRepository(db).list_runs()) == 1


def test_evaluate_report_only_never_fails_on_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "_run_evaluation", _fake_evaluation(0.1))
    result = runner.invoke(
        cli.app,
        ["evaluate", "--dataset", str(GOLDEN_SET), "--no-save", "--report-only"],
    )
    assert result.exit_code == 0, result.output


def test_evaluate_threshold_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_run_evaluation", _fake_evaluation(0.9))
    result = runner.invoke(
        cli.app, ["evaluate", "--dataset", str(GOLDEN_SET), "--no-save", "-t", "faithfulness=0.95"]
    )
    assert result.exit_code == 1


@pytest.mark.parametrize(
    ("error", "code"),
    [(JudgeUnavailableError("down"), 3), (JudgeConfigurationError("pull the model"), 2)],
)
def test_evaluate_maps_errors_to_exit_codes(
    monkeypatch: pytest.MonkeyPatch, error: Exception, code: int
) -> None:
    async def boom(*_: object) -> RunReport:
        raise error

    monkeypatch.setattr(cli, "_run_evaluation", boom)
    result = runner.invoke(cli.app, ["evaluate", "--dataset", str(GOLDEN_SET), "--no-save"])
    assert result.exit_code == code


def test_evaluate_bad_dataset_is_config_error(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["evaluate", "--dataset", str(tmp_path / "nope.jsonl")])
    assert result.exit_code == 2


def test_seed_demo_then_history(tmp_path: Path) -> None:
    db = tmp_path / "demo.db"
    seeded = runner.invoke(cli.app, ["seed-demo", "--runs", "6", "--db", str(db)])
    assert seeded.exit_code == 0, seeded.output
    runs = RunRepository(db).list_runs()
    assert len(runs) == 6
    assert any(r.gate_passed is False for r in runs)  # the demo includes a regression

    shown = runner.invoke(cli.app, ["history", "--db", str(db), "--limit", "3"])
    assert shown.exit_code == 0
    assert "demo" in shown.output


def test_history_on_empty_db(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["history", "--db", str(tmp_path / "empty.db")])
    assert result.exit_code == 0
    assert "No runs recorded yet" in result.output


def test_version() -> None:
    result = runner.invoke(cli.app, ["--version"])
    assert result.exit_code == 0
    assert "rag-sentinel" in result.output
