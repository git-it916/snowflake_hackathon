"""Snowflake Model Registry 연동 모듈.

학습된 ML 모델을 Snowflake Model Registry에 등록, 조회, 관리하는 모듈.
Registry API를 통해 모델 버전 관리, 메트릭 로깅, 최적 모델 탐색을 지원한다.

사용 예시:
    from config.settings import get_session
    from ml.model_registry import ModelRegistryManager

    session = get_session()
    mgr = ModelRegistryManager(session)

    # 모델 등록
    mgr.register_model(
        model=trained_model,
        model_name="apt_price_classifier",
        version="v1",
        metrics={"f1_macro": 0.75, "accuracy": 0.82},
    )

    # 최적 모델 로드
    best_model, best_version = mgr.get_best_model("apt_price_classifier")
"""

from __future__ import annotations

import logging
from typing import Any

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore[assignment]

try:
    from snowflake.ml.model import ModelVersion
    from snowflake.ml.registry import Registry
except ImportError as _ml_err:
    raise ImportError(
        "snowflake-ml-python 패키지가 필요합니다. "
        "'pip install snowflake-ml-python'으로 설치하세요."
    ) from _ml_err

from snowflake.snowpark import Session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

_DEFAULT_DATABASE = "TELECOM_DB"
_DEFAULT_SCHEMA = "ANALYTICS"
_DEFAULT_METRIC = "f1_macro"


class ModelRegistryManager:
    """Snowflake Model Registry를 통한 모델 생명주기 관리 클래스.

    학습된 모델의 등록, 버전 관리, 메트릭 로깅, 최적 모델 탐색을
    Snowflake Model Registry API로 수행한다.
    """

    def __init__(
        self,
        session: Session,
        database: str = _DEFAULT_DATABASE,
        schema: str = _DEFAULT_SCHEMA,
    ) -> None:
        """ModelRegistryManager 초기화.

        Args:
            session: Snowpark 세션 (Snowflake 연결)
            database: 모델 레지스트리가 위치할 데이터베이스
            schema: 모델 레지스트리가 위치할 스키마
        """
        self._session = session
        self._database = database
        self._schema = schema
        self._registry = Registry(
            session=session,
            database_name=database,
            schema_name=schema,
        )
        logger.info(
            "ModelRegistryManager 초기화 완료: %s.%s",
            database,
            schema,
        )

    # ------------------------------------------------------------------
    # 모델 등록
    # ------------------------------------------------------------------

    def register_model(
        self,
        model: Any,
        model_name: str,
        version: str,
        metrics: dict[str, float],
        description: str = "",
        tags: dict[str, str] | None = None,
        sample_input_data: pd.DataFrame | None = None,
        signatures: dict[str, Any] | None = None,
    ) -> ModelVersion:
        """학습된 모델을 Snowflake Model Registry에 등록.

        모델 객체를 직렬화하여 Registry에 저장하고
        평가 메트릭을 함께 기록한다.

        Args:
            model: 학습된 모델 객체 (sklearn, xgboost 등)
            model_name: 레지스트리 내 모델 이름
            version: 모델 버전 문자열 (예: "v1", "v2.1")
            metrics: 평가 메트릭 딕셔너리
                예: {"f1_macro": 0.75, "accuracy": 0.82}
            description: 모델 설명 (선택)
            tags: 모델 태그 딕셔너리 (선택)
                예: {"stage": "production", "district": "서초구"}
            sample_input_data: 모델 입력 샘플 (시그니처 추론용, 선택)
            signatures: 명시적 모델 시그니처 (선택, sample_input_data보다 우선)

        Returns:
            등록된 ModelVersion 객체
        """
        logger.info(
            "모델 등록 시작: %s (version=%s)",
            model_name,
            version,
        )

        # Registry에 모델 등록
        register_kwargs: dict[str, Any] = {
            "model_name": model_name,
            "version_name": version,
            "model": model,
        }

        if signatures is not None:
            register_kwargs["signatures"] = signatures
        elif sample_input_data is not None:
            register_kwargs["sample_input_data"] = sample_input_data

        mv = self._registry.log_model(**register_kwargs)

        # 설명 설정
        if description and hasattr(mv, "description"):
            try:
                mv.description = description
            except Exception:
                pass

        # 메트릭 로깅
        mv.set_metric(metric_name="metrics", metric_value=metrics)

        # 태그 설정
        if tags is not None:
            for tag_key, tag_value in tags.items():
                mv.set_metric(
                    metric_name=f"tag_{tag_key}",
                    metric_value=tag_value,
                )

        logger.info(
            "모델 등록 완료: %s/%s (metrics=%s)",
            model_name,
            version,
            metrics,
        )

        return mv

    # ------------------------------------------------------------------
    # 모델 로드
    # ------------------------------------------------------------------

    def load_model(
        self,
        model_name: str,
        version: str | None = None,
    ) -> Any:
        """Registry에서 모델을 로드.

        버전을 지정하지 않으면 최신 버전(default)을 로드한다.

        Args:
            model_name: 레지스트리 내 모델 이름
            version: 로드할 버전. None이면 default 버전 로드.

        Returns:
            로드된 모델 객체

        Raises:
            ValueError: 모델 또는 버전을 찾을 수 없는 경우
        """
        logger.info(
            "모델 로드: %s (version=%s)",
            model_name,
            version or "default",
        )

        model_ref = self._registry.get_model(model_name)

        if version is not None:
            mv = model_ref.version(version)
        else:
            mv = model_ref.default
            if mv is None:
                # default가 없으면 가장 최근 버전 사용
                versions = model_ref.versions()
                if not versions:
                    raise ValueError(
                        f"모델 '{model_name}'에 등록된 버전이 없습니다."
                    )
                mv = versions[-1]

        loaded_model = mv.load_model()
        logger.info("모델 로드 완료: %s/%s", model_name, mv.version_name)
        return loaded_model

    # ------------------------------------------------------------------
    # 모델 목록 조회
    # ------------------------------------------------------------------

    def list_models(self) -> pd.DataFrame:
        """Registry에 등록된 모든 모델 목록을 반환.

        Returns:
            모델 이름, 버전, 생성일 등을 포함한 pandas DataFrame
        """
        models = self._registry.models()

        records: list[dict[str, Any]] = []
        for model in models:
            model_name = model.name
            for mv in model.versions():
                record = {
                    "MODEL_NAME": model_name,
                    "VERSION": mv.version_name,
                    "DESCRIPTION": getattr(mv, "description", "") or "",
                    "CREATED_ON": getattr(mv, "created_on", None),
                }

                # 메트릭 조회 (존재하는 경우)
                try:
                    metrics = mv.get_metric("metrics")
                    record["METRICS"] = str(metrics)
                except Exception:
                    record["METRICS"] = ""

                records.append(record)

        if not records:
            return pd.DataFrame(
                columns=["MODEL_NAME", "VERSION", "DESCRIPTION",
                         "CREATED_ON", "METRICS"]
            )

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # 모델 버전 목록 조회
    # ------------------------------------------------------------------

    def list_versions(self, model_name: str) -> pd.DataFrame:
        """특정 모델의 모든 버전 목록과 메트릭을 반환.

        Args:
            model_name: 레지스트리 내 모델 이름

        Returns:
            버전명, 설명, 생성일, 메트릭을 포함한 pandas DataFrame

        Raises:
            ValueError: 모델을 찾을 수 없는 경우
        """
        logger.info("모델 버전 목록 조회: %s", model_name)

        model_ref = self._registry.get_model(model_name)
        versions = model_ref.versions()

        if not versions:
            logger.warning("모델 '%s'에 등록된 버전이 없습니다.", model_name)
            return pd.DataFrame(
                columns=["VERSION", "DESCRIPTION", "CREATED_ON", "METRICS"]
            )

        records: list[dict[str, Any]] = []
        for mv in versions:
            record: dict[str, Any] = {
                "VERSION": mv.version_name,
                "DESCRIPTION": getattr(mv, "description", "") or "",
                "CREATED_ON": getattr(mv, "created_on", None),
            }

            try:
                metrics = mv.get_metric("metrics")
                record["METRICS"] = metrics if isinstance(metrics, dict) else str(metrics)
            except Exception:
                record["METRICS"] = {}

            records.append(record)

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # 메트릭 로깅
    # ------------------------------------------------------------------

    def log_metrics(
        self,
        model_name: str,
        version: str,
        metrics: dict[str, float],
    ) -> None:
        """기존 모델 버전에 추가 메트릭을 기록.

        Args:
            model_name: 레지스트리 내 모델 이름
            version: 메트릭을 추가할 버전
            metrics: 기록할 메트릭 딕셔너리
                예: {"f1_macro": 0.78, "recall": 0.80}
        """
        logger.info(
            "메트릭 로깅: %s/%s -> %s",
            model_name,
            version,
            metrics,
        )

        model_ref = self._registry.get_model(model_name)
        mv = model_ref.version(version)
        mv.set_metric(metric_name="metrics", metric_value=metrics)

        logger.info("메트릭 로깅 완료")

    # ------------------------------------------------------------------
    # 최적 모델 탐색
    # ------------------------------------------------------------------

    def get_best_model(
        self,
        model_name: str,
        metric: str = _DEFAULT_METRIC,
    ) -> tuple[Any, str]:
        """지정 메트릭 기준으로 최적 모델 버전을 반환.

        모든 버전의 메트릭을 비교하여 가장 높은 값을 가진
        모델을 로드하여 반환한다.

        Args:
            model_name: 레지스트리 내 모델 이름
            metric: 비교 기준 메트릭 키 (예: "f1_macro", "accuracy")

        Returns:
            (최적 모델 객체, 최적 버전 문자열) 튜플

        Raises:
            ValueError: 모델이 없거나, 지정 메트릭이 기록된 버전이 없는 경우
        """
        logger.info(
            "최적 모델 탐색: %s (metric=%s)",
            model_name,
            metric,
        )

        model_ref = self._registry.get_model(model_name)
        versions = model_ref.versions()

        if not versions:
            raise ValueError(
                f"모델 '{model_name}'에 등록된 버전이 없습니다."
            )

        best_score: float = -float("inf")
        best_version: ModelVersion | None = None

        for mv in versions:
            try:
                stored_metrics = mv.get_metric("metrics")
                if isinstance(stored_metrics, dict) and metric in stored_metrics:
                    score = float(stored_metrics[metric])
                    if score > best_score:
                        best_score = score
                        best_version = mv
            except Exception:
                # 메트릭이 없는 버전은 건너뛰기
                continue

        if best_version is None:
            raise ValueError(
                f"모델 '{model_name}'에서 메트릭 '{metric}'이 "
                f"기록된 버전을 찾을 수 없습니다."
            )

        loaded_model = best_version.load_model()
        version_name = best_version.version_name

        logger.info(
            "최적 모델 선택: %s/%s (score=%.4f)",
            model_name,
            version_name,
            best_score,
        )

        return loaded_model, version_name
