from __future__ import annotations

import json

import pytest

from rag_sentinel.log import configure_logging, get_logger

# Created at import time, before configure_logging(), exactly like module-level loggers.
module_logger = get_logger("tests.module")


def test_module_level_loggers_honour_later_configuration(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("WARNING", json_logs=True)
    module_logger.info("hidden.event")
    module_logger.warning("shown.event", answer=42)

    captured = capsys.readouterr()
    assert captured.out == ""  # stdout is reserved for CLI output
    lines = [json.loads(line) for line in captured.err.splitlines() if line.strip()]
    assert [line["event"] for line in lines] == ["shown.event"]
    assert lines[0]["module"] == "tests.module"
    assert lines[0]["answer"] == 42
    assert lines[0]["level"] == "warning"
