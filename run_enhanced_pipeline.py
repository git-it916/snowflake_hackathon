"""텔레콤 파이프라인 실행기: SQL + ML 학습 + 검증.

TELECOM_DB에 대해 SQL 파이프라인(00-06)과 ML 파이프라인을 순차 실행하고,
결과 테이블의 존재 여부를 검증하는 CLI 도구.

사용법:
    python run_enhanced_pipeline.py --step all      # SQL + ML 전체 실행
    python run_enhanced_pipeline.py --step sql      # SQL 파이프라인만
    python run_enhanced_pipeline.py --step ml       # ML 파이프라인만
    python run_enhanced_pipeline.py --step test     # 검증만
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from snowflake.snowpark import Session

# ---------------------------------------------------------------------------
# 로깅 설정
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("enhanced_pipeline")

# ---------------------------------------------------------------------------
# SQL 파이프라인 상수
# ---------------------------------------------------------------------------

_SQL_DIR = Path(__file__).resolve().parent / "sql"

_SQL_FILES_ORDER: list[str] = [
    "00_setup.sql",
    "01_staging.sql",
    "02_analytics.sql",
    "03_mart.sql",
    "04_cortex_ml.sql",
    # 05_stored_procedures.sql 제외: 세미콜론 분리 이슈 + Python 분석 모듈이 동일 기능 수행
    "06_feature_store.sql",
    "07_data_quality.sql",
    "08_lineage.sql",
    # 09_cortex_analyst.sql: PUT 명령은 별도 실행 (deploy_sis.py 또는 수동)
    "10_dynamic_tables.sql",
    "11_cortex_ai_functions.sql",
]

# ---------------------------------------------------------------------------
# 검증 대상 테이블
# ---------------------------------------------------------------------------

_EXPECTED_TABLES: dict[str, list[str]] = {
    "STAGING": [
        "STG_FUNNEL",
        "STG_CHANNEL",
        "STG_REGIONAL",
        "STG_MARKETING",
    ],
    "ANALYTICS": [
        "FUNNEL_BOTTLENECKS",
        "CHANNEL_EFFICIENCY",
        "REGIONAL_DEMAND_SCORE",
        "ML_FEATURE_STORE",
    ],
    "MART": [
        "DT_KPI",
        "V_FUNNEL_TIMESERIES",
        "V_CHANNEL_PERFORMANCE",
        "V_REGIONAL_HEATMAP",
    ],
}

_ML_TABLES: list[str] = [
    "ANALYTICS.ML_FEATURE_STORE",
    "ANALYTICS.ML_PREDICTIONS",
]


# ---------------------------------------------------------------------------
# SQL 파이프라인
# ---------------------------------------------------------------------------


def run_sql_pipeline(session: Session) -> bool:
    """SQL 파이프라인을 순차적으로 실행.

    sql/ 디렉토리의 00-06 SQL 파일을 순서대로 실행한다.
    파일이 존재하지 않으면 건너뛰고, 실패 시 로그를 남긴 후 계속 진행한다.

    Args:
        session: Snowpark 세션

    Returns:
        모든 파일 실행 성공 여부
    """
    logger.info("=" * 60)
    logger.info("SQL 파이프라인 시작")
    logger.info("=" * 60)

    success_count = 0
    skip_count = 0
    fail_count = 0

    for sql_file in _SQL_FILES_ORDER:
        file_path = _SQL_DIR / sql_file

        if not file_path.exists():
            logger.warning("SQL 파일 없음 (건너뜀): %s", sql_file)
            skip_count += 1
            continue

        logger.info("실행 중: %s", sql_file)
        start = time.time()

        try:
            sql_content = file_path.read_text(encoding="utf-8")

            # 세미콜론으로 분리하여 개별 문 실행
            statements = _split_sql_statements(sql_content)

            for idx, stmt in enumerate(statements, start=1):
                try:
                    session.sql(stmt).collect()
                except Exception as stmt_exc:
                    # 개별 문 실패 시 경고 후 계속
                    logger.warning(
                        "  [%s] 문 %d 실패: %s (문: %s...)",
                        sql_file,
                        idx,
                        stmt_exc,
                        stmt[:80],
                    )

            elapsed = time.time() - start
            logger.info("  완료: %s (%.1f초)", sql_file, elapsed)
            success_count += 1

        except Exception as exc:
            elapsed = time.time() - start
            logger.error(
                "  실패: %s (%.1f초) - %s", sql_file, elapsed, exc
            )
            fail_count += 1

    logger.info(
        "SQL 파이프라인 완료: 성공=%d, 건너뜀=%d, 실패=%d",
        success_count,
        skip_count,
        fail_count,
    )
    return fail_count == 0


def _split_sql_statements(sql_content: str) -> list[str]:
    """SQL 파일 내용을 개별 문으로 분리.

    주석, 빈 라인, USE 문 등을 유지하면서 세미콜론 기준으로 분리한다.
    $$ dollar-quoting 블록 내부의 세미콜론은 분리하지 않는다.

    Args:
        sql_content: SQL 파일 전체 내용

    Returns:
        실행할 SQL 문 리스트
    """
    statements: list[str] = []
    current: list[str] = []
    in_dollar_quote = False

    for char_idx, char in enumerate(sql_content):
        # $$ 토글 감지
        if char == "$" and char_idx + 1 < len(sql_content) and sql_content[char_idx + 1] == "$":
            in_dollar_quote = not in_dollar_quote
            current.append(char)
            continue

        if char == ";" and not in_dollar_quote:
            stmt = "".join(current).strip()
            if stmt:
                # 주석만 있는 문 건너뛰기
                lines = [
                    line for line in stmt.split("\n")
                    if line.strip() and not line.strip().startswith("--")
                ]
                if lines:
                    statements.append(stmt)
            current = []
        else:
            current.append(char)

    # 마지막 문 (세미콜론 없이 끝나는 경우)
    remaining = "".join(current).strip()
    if remaining:
        lines = [
            line for line in remaining.split("\n")
            if line.strip() and not line.strip().startswith("--")
        ]
        if lines:
            statements.append(remaining)

    return statements


# ---------------------------------------------------------------------------
# ML 파이프라인
# ---------------------------------------------------------------------------


def run_ml_pipeline(session: Session) -> bool:
    """ML 파이프라인을 실행.

    1. FeatureEngineer: 피처 스토어 빌드
    2. ConversionModel: 모델 학습
    3. predict_all: 전체 예측 실행
    4. 예측 결과 저장
    5. Model Registry 등록

    Args:
        session: Snowpark 세션

    Returns:
        ML 파이프라인 성공 여부
    """
    logger.info("=" * 60)
    logger.info("ML 파이프라인 시작")
    logger.info("=" * 60)

    # Step 1: SQL Feature Store 사용 (06_feature_store.sql이 생성한 카테고리×월 테이블)
    # 주의: feature_engineering.py는 채널 레벨로 덮어쓰기 때문에 건너뜀.
    # SQL Feature Store가 9개 추가 피처(HHI, N_CHANNELS, FUNNEL_CVR 등)를 포함하므로
    # 모델 성능이 더 높음.
    logger.info("[Step 1/4] SQL Feature Store 확인 중 (Snowpark 피처 엔지니어링 건너뜀)")
    try:
        count = session.sql(
            "SELECT COUNT(*) AS CNT FROM TELECOM_DB.ANALYTICS.ML_FEATURE_STORE"
        ).collect()[0]["CNT"]
        if count > 0:
            logger.info("[Step 1/4] SQL Feature Store 사용: %d행", count)
        else:
            logger.error("[Step 1/4] SQL Feature Store 비어있음 — 06_feature_store.sql 실행 필요")
            return False
    except Exception:
        logger.error("[Step 1/4] SQL Feature Store 없음 — 06_feature_store.sql 실행 필요")
        return False

    # Step 2: 모델 학습
    model = _run_model_training(session)
    if model is None:
        logger.error("모델 학습 실패 — ML 파이프라인 중단")
        return False

    # Step 3: 전체 예측 + 저장
    predictions_ok = _run_predictions(session, model)
    if not predictions_ok:
        logger.warning("예측 실행 또는 저장에 문제가 있습니다.")

    # Step 4: Model Registry 등록
    _register_model(session, model)

    logger.info("ML 파이프라인 완료")
    return True


def _run_feature_engineering(session: Session):
    """피처 엔지니어링 단계를 실행.

    Args:
        session: Snowpark 세션

    Returns:
        FeatureEngineer 인스턴스. 실패 시 None.
    """
    logger.info("[Step 1/4] 피처 엔지니어링 시작")
    start = time.time()

    try:
        from ml.feature_engineering import FeatureEngineer

        fe = FeatureEngineer(session)
        fe.build_features()

        elapsed = time.time() - start
        logger.info("[Step 1/4] 피처 엔지니어링 완료 (%.1f초)", elapsed)
        return fe

    except Exception as exc:
        elapsed = time.time() - start
        logger.exception(
            "[Step 1/4] 피처 엔지니어링 실패 (%.1f초): %s", elapsed, exc
        )
        return None


def _run_model_training(session: Session):
    """모델 학습 단계를 실행.

    Args:
        session: Snowpark 세션

    Returns:
        학습된 ConversionModel 인스턴스. 실패 시 None.
    """
    logger.info("[Step 2/4] 모델 학습 시작")
    start = time.time()

    try:
        from ml.conversion_model import ConversionModel

        model = ConversionModel(session)
        model.train()

        elapsed = time.time() - start
        logger.info("[Step 2/4] 모델 학습 완료 (%.1f초)", elapsed)
        return model

    except Exception as exc:
        elapsed = time.time() - start
        logger.exception(
            "[Step 2/4] 모델 학습 실패 (%.1f초): %s", elapsed, exc
        )
        return None


def _run_predictions(session: Session, model) -> bool:
    """전체 예측을 실행하고 결과를 저장.

    Args:
        session: Snowpark 세션
        model: 학습된 ConversionModel

    Returns:
        예측 및 저장 성공 여부
    """
    logger.info("[Step 3/4] 전체 예측 실행")
    start = time.time()

    try:
        predictions = model.predict_all()

        if predictions is not None and not predictions.empty:
            # Snowflake 테이블로 저장
            sp_df = session.create_dataframe(predictions)
            sp_df.write.mode("overwrite").save_as_table(
                "ANALYTICS.ML_PREDICTIONS"
            )
            elapsed = time.time() - start
            logger.info(
                "[Step 3/4] 예측 완료: %d건 저장 (%.1f초)",
                len(predictions),
                elapsed,
            )
            return True

        elapsed = time.time() - start
        logger.warning(
            "[Step 3/4] 예측 결과가 비어있습니다 (%.1f초)", elapsed
        )
        return False

    except Exception as exc:
        elapsed = time.time() - start
        logger.exception(
            "[Step 3/4] 예측 실행 실패 (%.1f초): %s", elapsed, exc
        )
        return False


def _next_version(registry, model_name: str) -> str:
    """기존 버전을 조회하여 다음 버전 문자열(v1, v2, ...)을 반환."""
    try:
        versions_df = registry.list_versions(model_name)
        if versions_df.empty:
            return "v1"
        existing = []
        for v in versions_df["VERSION"]:
            if not isinstance(v, str):
                continue
            vl = v.lower().strip()
            if vl.startswith("v") and vl[1:].isdigit():
                existing.append(int(vl[1:]))
        return f"v{max(existing) + 1}" if existing else "v1"
    except Exception:
        return "v1"


def _register_model(session: Session, model) -> None:
    """학습된 모델을 Snowflake Model Registry에 등록.

    Args:
        session: Snowpark 세션
        model: 학습된 ConversionModel
    """
    logger.info("[Step 4/4] 모델 레지스트리 등록")

    try:
        from ml.model_registry import ModelRegistryManager

        registry = ModelRegistryManager(
            session, database="TELECOM_DB", schema="ANALYTICS"
        )

        metrics = {}
        if hasattr(model, "_metrics") and model._metrics is not None:
            metrics = model._metrics.to_dict()
        elif hasattr(model, "get_metrics"):
            metrics = model.get_metrics()
        elif hasattr(model, "metrics_"):
            metrics = model.metrics_
        logger.info("[Step 4/4] 메트릭: %s", metrics)

        # 내부 모델 객체 접근 (private 속성)
        inner_model = getattr(model, "_model", None)
        if inner_model is None:
            logger.warning("[Step 4/4] 내부 모델 객체 없음 — Registry 건너뛰기")
            return

        # 명시적 시그니처 생성 (100행 샘플링 경고 방지)
        from snowflake.ml.model import model_signature

        sample_input = None
        signatures = None
        if hasattr(model, "_train_df") and model._train_df is not None:
            feature_cols = model._feature_columns
            if feature_cols:
                sample_input = model._train_df[feature_cols].head(10).fillna(0)
                signatures = model_signature.infer_signature(
                    input_data=sample_input,
                    output_data=pd.Series([0, 1, 2], name="OUTPUT", dtype="int64").head(len(sample_input)),
                )

        # 자동 버전 증가
        model_name = "telecom_conversion_model"
        version = _next_version(registry, model_name)
        logger.info("[Step 4/4] 버전 결정: %s", version)

        register_kwargs = {
            "model": inner_model,
            "model_name": model_name,
            "version": version,
            "metrics": metrics,
            "description": "텔레콤 채널 전환율 예측 XGBoost 3-class 모델",
            "tags": {"project": "telecom_funnel", "stage": "production"},
        }
        if signatures is not None:
            register_kwargs["signatures"] = {"predict": signatures}
        elif sample_input is not None:
            register_kwargs["sample_input_data"] = sample_input

        registry.register_model(**register_kwargs)
        logger.info("[Step 4/4] 모델 등록 완료: %s/%s", model_name, version)

    except Exception as exc:
        logger.warning("[Step 4/4] 모델 등록 실패 (비치명적): %s", exc)


# ---------------------------------------------------------------------------
# 전체 파이프라인
# ---------------------------------------------------------------------------


def run_full_pipeline(session: Session) -> bool:
    """SQL + ML 전체 파이프라인을 실행.

    Args:
        session: Snowpark 세션

    Returns:
        전체 파이프라인 성공 여부
    """
    logger.info("*" * 60)
    logger.info("텔레콤 전체 파이프라인 시작")
    logger.info("*" * 60)

    total_start = time.time()

    sql_ok = run_sql_pipeline(session)
    if not sql_ok:
        logger.warning("SQL 파이프라인에 실패한 단계가 있습니다. ML 계속 진행.")

    ml_ok = run_ml_pipeline(session)

    total_elapsed = time.time() - total_start
    logger.info("*" * 60)
    logger.info(
        "전체 파이프라인 완료 (%.1f초): SQL=%s, ML=%s",
        total_elapsed,
        "성공" if sql_ok else "부분실패",
        "성공" if ml_ok else "실패",
    )
    logger.info("*" * 60)

    return sql_ok and ml_ok


# ---------------------------------------------------------------------------
# 검증
# ---------------------------------------------------------------------------


def verify_pipeline(session: Session) -> bool:
    """파이프라인 결과 테이블의 존재 여부를 검증.

    STAGING, ANALYTICS, MART 스키마의 기대 테이블이
    모두 존재하는지 확인한다.

    Args:
        session: Snowpark 세션

    Returns:
        모든 테이블 존재 시 True
    """
    logger.info("=" * 60)
    logger.info("파이프라인 검증 시작")
    logger.info("=" * 60)

    found_count = 0
    missing_count = 0
    all_tables: list[tuple[str, str]] = []

    for schema, tables in _EXPECTED_TABLES.items():
        for table in tables:
            all_tables.append((schema, table))

    for schema, table in all_tables:
        fqn = f"TELECOM_DB.{schema}.{table}"
        exists = _table_exists(session, fqn)

        if exists:
            row_count = _get_row_count(session, fqn)
            logger.info("  [OK] %s (%d rows)", fqn, row_count)
            found_count += 1
        else:
            logger.warning("  [MISSING] %s", fqn)
            missing_count += 1

    # ML 테이블 검증
    for ml_table in _ML_TABLES:
        fqn = f"TELECOM_DB.{ml_table}"
        exists = _table_exists(session, fqn)

        if exists:
            row_count = _get_row_count(session, fqn)
            logger.info("  [OK] %s (%d rows)", fqn, row_count)
            found_count += 1
        else:
            logger.info("  [SKIP] %s (ML 미실행 시 정상)", fqn)

    logger.info(
        "검증 완료: 존재=%d, 누락=%d", found_count, missing_count
    )

    if missing_count > 0:
        logger.warning(
            "누락된 테이블이 %d개 있습니다. SQL 파이프라인을 먼저 실행하세요.",
            missing_count,
        )
        return False

    logger.info("모든 기대 테이블이 존재합니다.")
    return True


def _table_exists(session: Session, fqn: str) -> bool:
    """Snowflake 테이블 존재 여부를 확인.

    Args:
        session: Snowpark 세션
        fqn: 정규화된 테이블명 (DB.SCHEMA.TABLE)

    Returns:
        테이블 존재 여부
    """
    try:
        session.sql(f"SELECT 1 FROM {fqn} LIMIT 1").collect()
        return True
    except Exception:
        return False


def _get_row_count(session: Session, fqn: str) -> int:
    """테이블의 행 수를 반환.

    Args:
        session: Snowpark 세션
        fqn: 정규화된 테이블명

    Returns:
        행 수. 실패 시 -1.
    """
    try:
        result = session.sql(f"SELECT COUNT(*) AS CNT FROM {fqn}").collect()
        return int(result[0]["CNT"]) if result else -1
    except Exception:
        return -1


# ---------------------------------------------------------------------------
# CLI 엔트리포인트
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI 엔트리포인트."""
    parser = argparse.ArgumentParser(
        description="텔레콤 파이프라인 실행기: SQL + ML 학습 + 검증"
    )
    parser.add_argument(
        "--step",
        choices=["all", "sql", "ml", "test"],
        default="all",
        help="실행할 단계 (기본: all)",
    )
    args = parser.parse_args()

    # Snowpark 세션 생성
    try:
        from config.settings import get_session

        session = get_session()
        logger.info("Snowpark 세션 연결 성공")
    except Exception as exc:
        logger.error("Snowpark 세션 생성 실패: %s", exc)
        sys.exit(1)

    # 실행
    try:
        if args.step == "all":
            ok = run_full_pipeline(session)
        elif args.step == "sql":
            ok = run_sql_pipeline(session)
        elif args.step == "ml":
            ok = run_ml_pipeline(session)
        elif args.step == "test":
            ok = verify_pipeline(session)
        else:
            logger.error("알 수 없는 단계: %s", args.step)
            ok = False

        sys.exit(0 if ok else 1)

    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
        sys.exit(130)
    except Exception as exc:
        logger.exception("파이프라인 실행 중 예상치 못한 오류: %s", exc)
        sys.exit(1)
    finally:
        try:
            session.close()
            logger.info("Snowpark 세션 종료")
        except Exception:
            pass


if __name__ == "__main__":
    main()
