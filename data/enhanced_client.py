"""확장 데이터 클라이언트: ML 및 Agent 시스템용."""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from data.snowflake_client import SnowflakeClient, _ANALYTICS, _MART

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cortex 기본 설정
# ---------------------------------------------------------------------------
_DEFAULT_MODEL = "llama3.1-405b"


class EnhancedSnowflakeClient(SnowflakeClient):
    """ML 학습/예측 및 Agent 연동을 위한 확장 클라이언트.

    SnowflakeClient의 모든 기능을 상속하며,
    ML 데이터 파이프라인과 Cortex COMPLETE 유틸리티를 추가합니다.
    """

    # ------------------------------------------------------------------
    # ML 데이터 로드
    # ------------------------------------------------------------------

    def load_ml_train(self) -> pd.DataFrame:
        """ANALYTICS.V_ML_TRAIN — ML 학습용 데이터셋.

        Returns:
            학습 데이터 DataFrame. 빈 테이블이면 빈 DataFrame.
        """
        return self._load_table(_ANALYTICS, "V_ML_TRAIN")

    def load_ml_test(self) -> pd.DataFrame:
        """ANALYTICS.V_ML_TEST — ML 테스트용 데이터셋.

        Returns:
            테스트 데이터 DataFrame. 빈 테이블이면 빈 DataFrame.
        """
        return self._load_table(_ANALYTICS, "V_ML_TEST")

    def save_predictions(self, df: pd.DataFrame) -> None:
        """예측 결과를 MART.FORECAST_OUTPUT에 저장.

        Args:
            df: 저장할 예측 결과 DataFrame. 비어 있으면 아무 작업도 하지 않음.

        Raises:
            Exception: Snowpark write 실패 시 로깅 후 재발생.
        """
        if df.empty:
            logger.warning("저장할 예측 데이터가 비어 있습니다.")
            return

        try:
            snowpark_df = self._session.create_dataframe(df)
            snowpark_df.write.mode("append").save_as_table(
                f"TELECOM_DB.{_MART}.FORECAST_OUTPUT"
            )
            logger.info("예측 결과 %d건 저장 완료.", len(df))
        except Exception:
            logger.exception("예측 결과 저장 실패")
            raise

    # ------------------------------------------------------------------
    # Agent 시스템용 요약
    # ------------------------------------------------------------------

    def get_category_summary(self, category: str) -> str:
        """특정 카테고리의 핵심 지표를 사람이 읽기 쉬운 텍스트로 요약.

        Args:
            category: 분석 대상 상품 카테고리 (예: '인터넷').

        Returns:
            포맷팅된 요약 문자열. 데이터 부재 시 안내 메시지.
        """
        funnel = self.load_funnel_timeseries(category=category)
        channel = self.load_channel_performance(category=category)

        if funnel.empty and channel.empty:
            return f"[{category}] 카테고리의 데이터가 없습니다."

        lines: list[str] = [f"=== {category} 카테고리 요약 ===", ""]

        if not funnel.empty:
            latest = funnel.sort_values("YEAR_MONTH", ascending=False).head(1)
            lines.append("[최신 퍼널 데이터]")
            for col in latest.columns:
                val = latest.iloc[0][col]
                lines.append(f"  {col}: {val}")
            lines.append("")

        if not channel.empty:
            top_channels = (
                channel.sort_values("TOTAL_NET_SALES", ascending=False).head(5)
                if "TOTAL_NET_SALES" in channel.columns
                else channel.head(5)
            )
            lines.append(f"[상위 채널 (최대 5개, 총 {len(channel)}개)]")
            for _, row in top_channels.iterrows():
                ch_name = row.get("CHANNEL", row.get("INFLOW_PATH", "N/A"))
                sales = row.get("TOTAL_NET_SALES", "N/A")
                cvr = row.get("PAYEND_CVR", "N/A")
                lines.append(f"  {ch_name}: 매출={sales}, 납입전환율={cvr}")
            lines.append("")

        return "\n".join(lines)

    def get_cross_category_comparison(self) -> pd.DataFrame:
        """전체 카테고리 간 핵심 KPI 비교 DataFrame.

        Returns:
            카테고리별 최신 월 KPI 요약. 데이터 부재 시 빈 DataFrame.
        """
        funnel = self.load_funnel_timeseries()
        if funnel.empty:
            return pd.DataFrame()

        if "YEAR_MONTH" not in funnel.columns or "CATEGORY" not in funnel.columns:
            logger.warning("V_FUNNEL_TIMESERIES에 YEAR_MONTH 또는 CATEGORY 컬럼 없음.")
            return pd.DataFrame()

        latest_month = funnel["YEAR_MONTH"].max()
        latest = funnel[funnel["YEAR_MONTH"] == latest_month].copy()

        agg_cols: dict[str, str] = {}
        for col in [
            "TOTAL_COUNT",
            "CONSULT_REQUEST_COUNT",
            "SUBSCRIPTION_COUNT",
            "OPEN_COUNT",
            "PAYEND_COUNT",
            "OVERALL_CVR",
        ]:
            if col in latest.columns:
                agg_cols[col] = "sum" if col != "OVERALL_CVR" else "mean"

        if not agg_cols:
            return latest

        result = (
            latest.groupby("CATEGORY", as_index=False)
            .agg(agg_cols)
        )
        return result

    # ------------------------------------------------------------------
    # Cortex COMPLETE 범용 래퍼
    # ------------------------------------------------------------------

    def cortex_complete(
        self,
        system_prompt: str,
        user_message: str,
        model: str = _DEFAULT_MODEL,
    ) -> str:
        """Cortex COMPLETE 범용 호출 래퍼.

        SnowflakeClient._cortex_complete 와 동일하지만 public API로 노출합니다.

        Args:
            system_prompt: 시스템 프롬프트.
            user_message: 사용자 메시지.
            model: 사용할 모델 이름 (기본: llama3.1-405b).

        Returns:
            생성된 텍스트. 실패 시 에러 메시지 문자열.
        """
        return self._cortex_complete(system_prompt, user_message, model=model)
