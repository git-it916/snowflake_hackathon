"""ML 모델 검증 및 메트릭 관리 모듈.

학습된 모델의 품질을 검증하고, 메트릭을 구조화하여 관리한다.
Model Registry 저장 시 메트릭을 함께 기록할 수 있도록 지원.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 검증 결과 데이터 클래스
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelMetrics:
    """학습/검증 메트릭 불변 컨테이너."""

    accuracy: float
    f1_macro: float
    f1_weighted: float
    precision_macro: float
    recall_macro: float
    auc_ovr: float | None  # One-vs-Rest AUC (확률 미지원 시 None)
    cv_scores: tuple[float, ...] = ()
    confusion_matrix: tuple[tuple[int, ...], ...] = ()
    classification_report_text: str = ""
    n_train_samples: int = 0
    n_features: int = 0
    feature_columns: tuple[str, ...] = ()

    @property
    def cv_mean(self) -> float:
        return float(np.mean(self.cv_scores)) if self.cv_scores else 0.0

    @property
    def cv_std(self) -> float:
        return float(np.std(self.cv_scores)) if self.cv_scores else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": round(self.accuracy, 4),
            "f1_macro": round(self.f1_macro, 4),
            "f1_weighted": round(self.f1_weighted, 4),
            "precision_macro": round(self.precision_macro, 4),
            "recall_macro": round(self.recall_macro, 4),
            "auc_ovr": round(self.auc_ovr, 4) if self.auc_ovr is not None else None,
            "cv_mean": round(self.cv_mean, 4),
            "cv_std": round(self.cv_std, 4),
            "n_train_samples": self.n_train_samples,
            "n_features": self.n_features,
        }

    @property
    def is_acceptable(self) -> bool:
        """최소 품질 기준 충족 여부 (F1 >= 0.4)."""
        return self.f1_macro >= 0.4


@dataclass(frozen=True)
class FeatureValidationResult:
    """피처 검증 결과."""

    total_features: int
    available_features: int
    missing_features: tuple[str, ...] = ()
    null_ratio: dict[str, float] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.available_features >= 3

    @property
    def high_null_features(self) -> list[str]:
        """NULL 비율 50% 이상인 피처 목록."""
        return [f for f, r in self.null_ratio.items() if r > 0.5]


# ---------------------------------------------------------------------------
# 검증 함수
# ---------------------------------------------------------------------------


def validate_features(
    df: pd.DataFrame,
    expected_features: list[str],
    target_col: str = "TARGET_CLASS",
) -> FeatureValidationResult:
    """학습 데이터의 피처 유효성을 검증.

    Args:
        df: 학습 데이터 DataFrame
        expected_features: 기대 피처 컬럼 목록
        target_col: 타겟 컬럼명

    Returns:
        FeatureValidationResult
    """
    available = [c for c in expected_features if c in df.columns]
    missing = tuple(c for c in expected_features if c not in df.columns)

    null_ratio = {}
    for col in available:
        ratio = float(df[col].isna().mean())
        if ratio > 0.1:
            null_ratio[col] = round(ratio, 3)

    if missing:
        logger.warning(
            "누락 피처 %d개: %s", len(missing), ", ".join(missing[:5])
        )

    if not available:
        logger.error("사용 가능한 피처가 없습니다")

    return FeatureValidationResult(
        total_features=len(expected_features),
        available_features=len(available),
        missing_features=missing,
        null_ratio=null_ratio,
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    cv_scores: list[float] | None = None,
    n_features: int = 0,
    feature_columns: list[str] | None = None,
    target_names: list[str] | None = None,
) -> ModelMetrics:
    """전체 모델 메트릭을 계산.

    Args:
        y_true: 실제 라벨 (정수)
        y_pred: 예측 라벨 (정수)
        y_proba: 클래스별 확률 (shape: [n_samples, n_classes])
        cv_scores: CV fold별 F1 점수
        n_features: 피처 수
        feature_columns: 피처 컬럼명 리스트
        target_names: 클래스명 리스트

    Returns:
        ModelMetrics
    """
    acc = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_wt = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)

    auc = None
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        except (ValueError, IndexError):
            logger.warning("AUC 계산 실패 — 클래스 수 불일치 가능")

    cm = confusion_matrix(y_true, y_pred)
    cm_tuple = tuple(tuple(int(x) for x in row) for row in cm)

    names = target_names or ["LOW", "MEDIUM", "HIGH"]
    report = classification_report(
        y_true, y_pred, target_names=names, zero_division=0
    )

    return ModelMetrics(
        accuracy=acc,
        f1_macro=f1_mac,
        f1_weighted=f1_wt,
        precision_macro=prec,
        recall_macro=rec,
        auc_ovr=auc,
        cv_scores=tuple(cv_scores or []),
        confusion_matrix=cm_tuple,
        classification_report_text=report,
        n_train_samples=len(y_true),
        n_features=n_features,
        feature_columns=tuple(feature_columns or []),
    )
