"""에이전트 입출력 스키마 정의.

에이전트 간 통신에 사용되는 구조화된 데이터 계약.
Pydantic 미설치 환경(SiS)을 고려하여 dataclass 기반으로 구현.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Analyst Agent 스키마
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalystResult:
    """AnalystAgent의 분석 결과 구조."""

    analysis: str = ""
    data_used: tuple[str, ...] = ()
    confidence: str = "low"  # "high" | "medium" | "low"
    key_findings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis": self.analysis,
            "data_used": list(self.data_used),
            "confidence": self.confidence,
            "key_findings": list(self.key_findings),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnalystResult:
        return cls(
            analysis=data.get("analysis", ""),
            data_used=tuple(data.get("data_used", [])),
            confidence=data.get("confidence", "low"),
            key_findings=tuple(data.get("key_findings", [])),
        )

    @classmethod
    def error(cls, message: str) -> AnalystResult:
        return cls(analysis=message, confidence="low")


# ---------------------------------------------------------------------------
# Strategy Agent 스키마
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategyResult:
    """StrategyAgent의 전략 결과 구조."""

    strategy: str = ""
    scenarios: dict[str, str] = field(default_factory=dict)
    confidence: str = "low"  # "high" | "medium" | "low"
    action_items: tuple[str, ...] = ()
    risk_level: str = "unknown"  # "low" | "medium" | "high" | "critical"

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "scenarios": dict(self.scenarios),
            "confidence": self.confidence,
            "action_items": list(self.action_items),
            "risk_level": self.risk_level,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StrategyResult:
        return cls(
            strategy=data.get("strategy", ""),
            scenarios=data.get("scenarios", {}),
            confidence=data.get("confidence", "low"),
            action_items=tuple(data.get("action_items", [])),
            risk_level=data.get("risk_level", "unknown"),
        )

    @classmethod
    def error(cls, message: str) -> StrategyResult:
        return cls(strategy=message, confidence="low", risk_level="unknown")


# ---------------------------------------------------------------------------
# Orchestrator 최종 결과
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OrchestratorResult:
    """AgentOrchestrator의 전체 분석 결과."""

    executive_summary: str = ""
    analyst_report: AnalystResult = field(default_factory=lambda: AnalystResult())
    strategy_report: StrategyResult = field(default_factory=lambda: StrategyResult())
    recommended_actions: tuple[str, ...] = ()
    confidence_level: str = "low"

    def to_dict(self) -> dict[str, Any]:
        return {
            "executive_summary": self.executive_summary,
            "analyst_report": self.analyst_report.to_dict(),
            "strategy_report": self.strategy_report.to_dict(),
            "recommended_actions": list(self.recommended_actions),
            "confidence_level": self.confidence_level,
        }


# ---------------------------------------------------------------------------
# 전이 행렬 검증
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransitionMatrixValidation:
    """마르코프 전이 행렬 검증 결과."""

    is_valid: bool = False
    row_sums_ok: bool = False
    non_negative: bool = False
    max_row_sum_error: float = 0.0
    warnings: tuple[str, ...] = ()

    @classmethod
    def validate(cls, matrix) -> TransitionMatrixValidation:
        """pandas DataFrame 전이 행렬을 검증.

        검증 항목:
        - 각 행의 합이 1.0 (허용 오차 1e-6)
        - 모든 값이 0 이상
        - 대각선 값이 1.0 미만
        """
        import numpy as np

        warnings: list[str] = []

        if matrix is None or matrix.empty:
            return cls(is_valid=False, warnings=("행렬이 비어 있습니다",))

        values = matrix.values.astype(float)

        non_negative = bool(np.all(values >= -1e-10))
        if not non_negative:
            neg_count = int(np.sum(values < -1e-10))
            warnings.append(f"음수 값 {neg_count}개 발견")

        row_sums = values.sum(axis=1)
        max_error = float(np.max(np.abs(row_sums - 1.0)))
        row_sums_ok = max_error < 1e-4

        if not row_sums_ok:
            warnings.append(
                f"행 합 오차 최대 {max_error:.6f} (허용: 1e-4)"
            )

        diag = np.diag(values)
        if np.any(diag >= 1.0 - 1e-10):
            warnings.append("대각선에 1.0 값 존재 (완전 흡수 상태)")

        is_valid = non_negative and row_sums_ok

        return cls(
            is_valid=is_valid,
            row_sums_ok=row_sums_ok,
            non_negative=non_negative,
            max_row_sum_error=max_error,
            warnings=tuple(warnings),
        )
