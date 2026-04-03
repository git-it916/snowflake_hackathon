"""Snowpark 기반 ML Feature Engineering.

STG_CHANNEL 원본 데이터로부터 lag, rolling, interaction 피처를 생성하여
ANALYTICS.ML_FEATURE_STORE 테이블에 저장한다.

Snowpark DataFrame API(F.lag, F.avg, Window)를 사용하여
모든 피처 연산을 Snowflake 엔진에서 수행한다.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from snowflake.snowpark import Session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------
_SOURCE_TABLE = "STAGING.STG_CHANNEL"
_FEATURE_STORE_TABLE = "ANALYTICS.ML_FEATURE_STORE"
_TRAIN_CUTOFF = "2026-01-01"

_BASE_COLUMNS = [
    "YEAR_MONTH",
    "CATEGORY",
    "CHANNEL",
    "PAYEND_CVR",
    "OPEN_CVR",
    "CONTRACT_COUNT",
    "AVG_NET_SALES",
]

_LAG_SPECS: list[tuple[str, int]] = [
    ("PAYEND_CVR", 1),
    ("PAYEND_CVR", 2),
    ("PAYEND_CVR", 3),
    ("CONTRACT_COUNT", 1),
    ("CONTRACT_COUNT", 2),
    ("AVG_NET_SALES", 1),
    ("OPEN_CVR", 1),
]

_ROLLING_WINDOWS = [3, 6]

_PERCENTILE_LOW = 0.33
_PERCENTILE_HIGH = 0.67


class FeatureEngineer:
    """Snowpark DataFrame API로 ML 피처를 빌드하는 클래스.

    STG_CHANNEL 테이블의 채널별 월간 성과 데이터로부터
    lag/rolling/interaction 피처를 생성하고,
    다음 달 PAYEND_CVR 기반 3-class 타겟(HIGH/MEDIUM/LOW)을 계산한다.
    """

    def __init__(self, session: Session) -> None:
        """FeatureEngineer 초기화.

        Args:
            session: Snowpark 세션 (TELECOM_DB에 연결)
        """
        self._session = session
        logger.info("FeatureEngineer 초기화 완료")

    # ------------------------------------------------------------------
    # 피처 스토어 빌드
    # ------------------------------------------------------------------

    def build_features(self, city_codes: list[str] | None = None) -> None:
        """ML_FEATURE_STORE 테이블을 Snowpark DataFrame API로 빌드.

        STG_CHANNEL로부터 lag, rolling, interaction 피처를 생성하고
        LEAD(1)로 다음 달 전환율을 가져와 percentile 기반 타겟 변수를 만든다.

        Args:
            city_codes: 필터링할 도시 코드 (None이면 전체)
        """
        from snowflake.snowpark import functions as F
        from snowflake.snowpark import Window

        logger.info("피처 스토어 빌드 시작")

        # 1. 원본 데이터 로드
        df = self._session.table(_SOURCE_TABLE)

        if city_codes is not None and len(city_codes) > 0:
            df = df.filter(F.col("CITY_CODE").isin(city_codes))

        # 2. 채널 x 카테고리 파티션 윈도우 정의
        partition_window = Window.partition_by(
            F.col("CHANNEL"), F.col("CATEGORY")
        ).order_by(F.col("YEAR_MONTH"))

        # 3. Lag 피처 생성
        for col_name, lag_n in _LAG_SPECS:
            alias = f"{col_name}_LAG{lag_n}"
            df = df.with_column(
                alias,
                F.lag(F.col(col_name), lag_n).over(partition_window),
            )

        # 4. Rolling 피처 생성 (MA, STD)
        for window_size in _ROLLING_WINDOWS:
            rolling_window = (
                Window.partition_by(F.col("CHANNEL"), F.col("CATEGORY"))
                .order_by(F.col("YEAR_MONTH"))
                .rows_between(-window_size + 1, 0)
            )

            df = df.with_column(
                f"PAYEND_CVR_MA{window_size}",
                F.avg(F.col("PAYEND_CVR")).over(rolling_window),
            )

        # 3개월 표준편차
        std_window = (
            Window.partition_by(F.col("CHANNEL"), F.col("CATEGORY"))
            .order_by(F.col("YEAR_MONTH"))
            .rows_between(-2, 0)
        )
        df = df.with_column(
            "PAYEND_CVR_STD3",
            F.stddev(F.col("PAYEND_CVR")).over(std_window),
        )

        # 5. Historical / Category 평균
        channel_hist_window = Window.partition_by(
            F.col("CHANNEL"), F.col("CATEGORY")
        ).order_by(F.col("YEAR_MONTH")).rows_between(
            Window.UNBOUNDED_PRECEDING, -1
        )
        df = df.with_column(
            "CHANNEL_HISTORICAL_CVR",
            F.avg(F.col("PAYEND_CVR")).over(channel_hist_window),
        )

        category_hist_window = Window.partition_by(
            F.col("CATEGORY")
        ).order_by(F.col("YEAR_MONTH")).rows_between(
            Window.UNBOUNDED_PRECEDING, -1
        )
        df = df.with_column(
            "CATEGORY_AVG_CVR",
            F.avg(F.col("PAYEND_CVR")).over(category_hist_window),
        )

        # 6. 시간 피처 (YEAR_MONTH은 이미 DATE 타입)
        df = df.with_column(
            "MONTH_OF_YEAR",
            F.month(F.col("YEAR_MONTH")),
        )
        df = df.with_column(
            "QUARTER",
            F.quarter(F.col("YEAR_MONTH")),
        )

        # 7. 인코딩 피처
        df = df.with_column(
            "CATEGORY_ENCODED",
            F.dense_rank().over(
                Window.order_by(F.col("CATEGORY"))
            ) - F.lit(1),
        )
        df = df.with_column(
            "CHANNEL_ENCODED",
            F.dense_rank().over(
                Window.order_by(F.col("CHANNEL"))
            ) - F.lit(1),
        )

        # 8. 타겟 변수: LEAD(1) PAYEND_CVR의 percentile 기반 분류
        df = df.with_column(
            "NEXT_CVR",
            F.lead(F.col("PAYEND_CVR"), 1).over(partition_window),
        )

        # percentile 기반 bins 계산
        pct_window = Window.partition_by(F.lit(1))
        df = df.with_column(
            "P33",
            F.percentile_cont(F.lit(_PERCENTILE_LOW)).within_group(
                F.col("NEXT_CVR").asc()
            ).over(pct_window),
        )
        df = df.with_column(
            "P67",
            F.percentile_cont(F.lit(_PERCENTILE_HIGH)).within_group(
                F.col("NEXT_CVR").asc()
            ).over(pct_window),
        )

        df = df.with_column(
            "TARGET_CLASS",
            F.when(F.col("NEXT_CVR").is_null(), None)
            .when(F.col("NEXT_CVR") <= F.col("P33"), F.lit("LOW"))
            .when(F.col("NEXT_CVR") <= F.col("P67"), F.lit("MEDIUM"))
            .otherwise(F.lit("HIGH")),
        )

        # 보조 컬럼 제거
        df = df.drop("NEXT_CVR", "P33", "P67")

        # 9. NULL lag 행 필터 (최소 LAG3이 존재하는 행만)
        df = df.filter(F.col("PAYEND_CVR_LAG3").is_not_null())
        df = df.filter(F.col("TARGET_CLASS").is_not_null())

        # 10. 피처 스토어에 저장
        df.write.mode("overwrite").save_as_table(_FEATURE_STORE_TABLE)

        row_count = self._session.table(_FEATURE_STORE_TABLE).count()
        logger.info(
            "피처 스토어 빌드 완료: %s (%d rows)",
            _FEATURE_STORE_TABLE,
            row_count,
        )

    # ------------------------------------------------------------------
    # 학습 / 테스트 데이터 로드
    # ------------------------------------------------------------------

    def get_training_data(self) -> pd.DataFrame:
        """학습 데이터를 반환 (YEAR_MONTH < 2026-01-01).

        Returns:
            학습용 pandas DataFrame
        """
        return self._load_split(is_train=True)

    def get_test_data(self) -> pd.DataFrame:
        """테스트 데이터를 반환 (YEAR_MONTH >= 2026-01-01).

        Returns:
            테스트용 pandas DataFrame
        """
        return self._load_split(is_train=False)

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _load_split(self, *, is_train: bool) -> pd.DataFrame:
        """피처 스토어에서 train/test 분할 데이터를 로드.

        ML_FEATURE_STORE가 없으면 자동으로 build_features()를 실행한다.

        Args:
            is_train: True면 학습 데이터, False면 테스트 데이터

        Returns:
            pandas DataFrame
        """
        from snowflake.snowpark import functions as F

        if not self._table_exists(_FEATURE_STORE_TABLE):
            logger.warning(
                "ML_FEATURE_STORE 없음 → build_features() 자동 실행"
            )
            self.build_features()

        df = self._session.table(_FEATURE_STORE_TABLE)

        if is_train:
            df = df.filter(F.col("YEAR_MONTH") < F.lit(_TRAIN_CUTOFF))
        else:
            df = df.filter(F.col("YEAR_MONTH") >= F.lit(_TRAIN_CUTOFF))

        result = df.to_pandas()
        split_name = "학습" if is_train else "테스트"
        logger.info("%s 데이터 로드 완료: %d rows", split_name, len(result))
        return result

    def _table_exists(self, full_name: str) -> bool:
        """Snowflake 테이블 존재 여부를 확인.

        Args:
            full_name: SCHEMA.TABLE 형식의 테이블명

        Returns:
            테이블 존재 여부
        """
        try:
            self._session.table(full_name).limit(1).collect()
            return True
        except Exception:
            return False
