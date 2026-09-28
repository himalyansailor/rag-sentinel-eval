"""CI quality gate (see docs/LLD.md §9)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rag_sentinel.exceptions import ConfigurationError
from rag_sentinel.models import GateResult, GateViolation, MetricName

if TYPE_CHECKING:
    from rag_sentinel.config import GateSettings
    from rag_sentinel.models import RunReport


def _to_metric_map(raw: Mapping[str, float]) -> dict[MetricName, float]:
    out: dict[MetricName, float] = {}
    for key, value in raw.items():
        try:
            name = MetricName(key)
        except ValueError as exc:
            msg = f"unknown metric {key!r}; expected one of {[m.value for m in MetricName]}"
            raise ConfigurationError(msg) from exc
        if not 0.0 <= value <= 1.0:
            msg = f"threshold for {key!r} must be within [0, 1], got {value}"
            raise ConfigurationError(msg)
        out[name] = value
    return out


def parse_threshold_overrides(items: Iterable[str]) -> dict[str, float]:
    """Parse CLI overrides such as ``["faithfulness=0.9", "context_recall=0.7"]``."""
    parsed: dict[str, float] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            msg = f"invalid threshold {item!r}; expected METRIC=VALUE"
            raise ConfigurationError(msg)
        try:
            parsed[key.strip()] = float(value)
        except ValueError as exc:
            msg = f"invalid threshold value in {item!r}"
            raise ConfigurationError(msg) from exc
    _to_metric_map(parsed)  # validate names and ranges
    return parsed


@dataclass(frozen=True)
class QualityGate:
    """Blocking thresholds, advisory thresholds and an error budget."""

    thresholds: Mapping[MetricName, float]
    warn_thresholds: Mapping[MetricName, float] = field(default_factory=dict)
    max_error_rate: float = 0.2

    @classmethod
    def from_settings(
        cls,
        settings: GateSettings,
        overrides: Mapping[str, float] | None = None,
        max_error_rate: float | None = None,
    ) -> QualityGate:
        thresholds = {**settings.thresholds, **(overrides or {})}
        return cls(
            thresholds=_to_metric_map(thresholds),
            warn_thresholds=_to_metric_map(settings.warn_thresholds),
            max_error_rate=settings.max_error_rate if max_error_rate is None else max_error_rate,
        )

    def evaluate(self, report: RunReport) -> GateResult:
        violations: list[GateViolation] = []
        warnings: list[str] = []
        evaluated = set(report.config.metrics)

        for metric, threshold in self.thresholds.items():
            actual = report.aggregates.get(metric)
            if metric not in evaluated:
                message = f"{metric.value} has a blocking threshold but was not evaluated"
            elif actual is None:
                message = f"{metric.value}: no sample produced a score"
            elif actual < threshold:
                message = f"{metric.value} {actual:.3f} is below the threshold {threshold:.2f}"
            else:
                continue
            violations.append(
                GateViolation(
                    metric=metric.value, threshold=threshold, actual=actual, message=message
                )
            )

        if report.error_rate > self.max_error_rate:
            violations.append(
                GateViolation(
                    metric="error_rate",
                    threshold=self.max_error_rate,
                    actual=round(report.error_rate, 4),
                    message=(
                        f"{report.error_count}/{report.sample_count} samples failed to evaluate "
                        f"(error rate {report.error_rate:.0%} > budget {self.max_error_rate:.0%})"
                    ),
                )
            )

        for metric, threshold in self.warn_thresholds.items():
            if metric in self.thresholds or metric not in evaluated:
                continue
            actual = report.aggregates.get(metric)
            if actual is not None and actual < threshold:
                warnings.append(
                    f"{metric.value} {actual:.3f} is below the advisory {threshold:.2f}"
                )

        return GateResult(passed=not violations, violations=violations, warnings=warnings)
