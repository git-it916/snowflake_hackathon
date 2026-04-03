"""텔레콤 가입 퍼널 ML 모듈.

Snowpark ML 기반 채널 전환율 예측, What-if 시뮬레이션, 모델 해석 시스템.
Snowflake Model Registry를 통한 모델 버전 관리 포함.

구성요소:
    - FeatureEngineer: Snowpark DataFrame API 기반 피처 엔지니어링
    - ConversionModel: XGBoost 3-class 채널 전환율 예측 모델
    - SimulationEngine: What-if 채널 예산 시뮬레이션
    - ModelExplainer: SHAP 기반 모델 해석
    - ModelRegistryManager: Snowflake Model Registry 생명주기 관리
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from snowflake.snowpark import Session

logger = logging.getLogger(__name__)

_IMPORT_ERRORS: dict[str, str] = {}

try:
    from ml.feature_engineering import FeatureEngineer
except Exception as exc:
    _IMPORT_ERRORS["FeatureEngineer"] = str(exc)
    FeatureEngineer = None  # type: ignore[assignment, misc]

try:
    from ml.conversion_model import ConversionModel
except Exception as exc:
    _IMPORT_ERRORS["ConversionModel"] = str(exc)
    ConversionModel = None  # type: ignore[assignment, misc]

try:
    from ml.simulation_engine import SimulationEngine
except Exception as exc:
    _IMPORT_ERRORS["SimulationEngine"] = str(exc)
    SimulationEngine = None  # type: ignore[assignment, misc]

try:
    from ml.explainer import ModelExplainer
except Exception as exc:
    _IMPORT_ERRORS["ModelExplainer"] = str(exc)
    ModelExplainer = None  # type: ignore[assignment, misc]

try:
    from ml.model_registry import ModelRegistryManager
except Exception as exc:
    _IMPORT_ERRORS["ModelRegistryManager"] = str(exc)
    ModelRegistryManager = None  # type: ignore[assignment, misc]

if _IMPORT_ERRORS:
    for component, error in _IMPORT_ERRORS.items():
        logger.warning("ML 컴포넌트 '%s' 임포트 실패: %s", component, error)

__all__ = [
    "FeatureEngineer",
    "ConversionModel",
    "SimulationEngine",
    "ModelExplainer",
    "ModelRegistryManager",
    "create_ml_pipeline",
]


def create_ml_pipeline(session: Session) -> dict[str, Any]:
    """ML 파이프라인 전체 컴포넌트를 초기화하여 반환."""
    pipeline: dict[str, Any] = {
        "feature_eng": None,
        "model": None,
        "simulator": None,
        "explainer": None,
        "registry": None,
    }

    if FeatureEngineer is not None:
        try:
            pipeline["feature_eng"] = FeatureEngineer(session)
        except Exception as exc:
            logger.warning("FeatureEngineer 초기화 실패: %s", exc)

    model_instance = None
    if ConversionModel is not None:
        try:
            model_instance = ConversionModel(session)
            pipeline["model"] = model_instance
        except Exception as exc:
            logger.warning("ConversionModel 초기화 실패: %s", exc)

    if SimulationEngine is not None and model_instance is not None:
        try:
            pipeline["simulator"] = SimulationEngine(session, model_instance)
        except Exception as exc:
            logger.warning("SimulationEngine 초기화 실패: %s", exc)

    if ModelExplainer is not None:
        pipeline["explainer"] = ModelExplainer

    if ModelRegistryManager is not None:
        try:
            pipeline["registry"] = ModelRegistryManager(session)
        except Exception as exc:
            logger.warning("ModelRegistryManager 초기화 실패: %s", exc)

    return pipeline
