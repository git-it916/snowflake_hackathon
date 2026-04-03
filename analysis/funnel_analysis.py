"""퍼널 병목 탐지 분석 모듈.

상담 요청 → 가입 신청 → 접수 → 개통 → 납입 완료 퍼널의
단계별 전환율, 이탈률, 트렌드를 분석합니다.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config.constants import FUNNEL_DROP_THRESHOLD, FUNNEL_STAGES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 퍼널 스테이지 → 건수 컬럼 매핑
# ---------------------------------------------------------------------------
_STAGE_COUNT_COLS: dict[str, str] = {
    "CONSULT_REQUEST": "CONSULT_REQUEST_COUNT",
    "SUBSCRIPTION": "SUBSCRIPTION_COUNT",
    "REGISTEND": "REGISTEND_COUNT",
    "OPEN": "OPEN_COUNT",
    "PAYEND": "PAYEND_COUNT",
}


def compute_stage_drops(df: pd.DataFrame) -> pd.DataFrame:
    """퍼널 단계별 전환율 및 이탈률 계산.

    각 인접 스테이지 쌍에 대해 전환율(CVR)과 이탈률(DROP_RATE)을 계산합니다.
    원본 DataFrame을 변경하지 않습니다.

    Args:
        df: STG_FUNNEL 또는 V_FUNNEL_TIMESERIES 데이터.
            YEAR_MONTH, CATEGORY, *_COUNT 컬럼 필요.

    Returns:
        YEAR_MONTH, CATEGORY, FROM_STAGE, TO_STAGE, FROM_COUNT, TO_COUNT,
        CVR, DROP_RATE 컬럼을 가진 DataFrame.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "YEAR_MONTH", "CATEGORY", "FROM_STAGE", "TO_STAGE",
                "FROM_COUNT", "TO_COUNT", "CVR", "DROP_RATE",
            ]
        )

    # TOTAL_COUNT가 있으면 첫 스테이지 이전 단계로 사용
    count_cols = ["TOTAL_COUNT"] if "TOTAL_COUNT" in df.columns else []
    count_cols += [
        _STAGE_COUNT_COLS[s] for s in FUNNEL_STAGES if _STAGE_COUNT_COLS[s] in df.columns
    ]

    if len(count_cols) < 2:
        logger.warning("전환율 계산에 필요한 컬럼 부족: %s", count_cols)
        return pd.DataFrame()

    group_cols = [c for c in ["YEAR_MONTH", "CATEGORY"] if c in df.columns]
    records: list[dict] = []

    for _, row in df.iterrows():
        base = {col: row[col] for col in group_cols if col in row.index}

        for i in range(len(count_cols) - 1):
            from_col = count_cols[i]
            to_col = count_cols[i + 1]
            from_count = row.get(from_col, 0)
            to_count = row.get(to_col, 0)

            from_stage = from_col.replace("_COUNT", "")
            to_stage = to_col.replace("_COUNT", "")

            cvr = to_count / from_count if from_count > 0 else 0.0
            drop_rate = 1.0 - cvr

            records.append(
                {
                    **base,
                    "FROM_STAGE": from_stage,
                    "TO_STAGE": to_stage,
                    "FROM_COUNT": from_count,
                    "TO_COUNT": to_count,
                    "CVR": round(cvr, 4),
                    "DROP_RATE": round(drop_rate, 4),
                }
            )

    return pd.DataFrame(records)


def detect_bottlenecks(
    df: pd.DataFrame,
    threshold: float = FUNNEL_DROP_THRESHOLD,
) -> list[dict]:
    """이탈률이 임계값을 초과하는 병목 스테이지 탐지.

    Args:
        df: compute_stage_drops() 결과 또는 동일 스키마 DataFrame.
            FROM_STAGE, TO_STAGE, DROP_RATE 컬럼 필요.
        threshold: 병목 판단 이탈률 임계값 (기본 0.15 = 15%).

    Returns:
        병목 정보 딕셔너리 리스트. 각 항목:
        - from_stage, to_stage, drop_rate, category, year_month (있으면)
    """
    if df.empty or "DROP_RATE" not in df.columns:
        return []

    stage_drops = compute_stage_drops(df) if "FROM_STAGE" not in df.columns else df

    if stage_drops.empty:
        return []

    bottlenecks_mask = stage_drops["DROP_RATE"] > threshold
    bottleneck_rows = stage_drops[bottlenecks_mask]

    result: list[dict] = []
    for _, row in bottleneck_rows.iterrows():
        entry = {
            "from_stage": row["FROM_STAGE"],
            "to_stage": row["TO_STAGE"],
            "drop_rate": row["DROP_RATE"],
        }
        if "CATEGORY" in row.index:
            entry["category"] = row["CATEGORY"]
        if "YEAR_MONTH" in row.index:
            entry["year_month"] = row["YEAR_MONTH"]
        result.append(entry)

    # 이탈률 내림차순 정렬
    result.sort(key=lambda x: x["drop_rate"], reverse=True)
    return result


def funnel_trend_analysis(
    df: pd.DataFrame,
    months: int = 6,
) -> pd.DataFrame:
    """퍼널 스테이지별 MoM(전월 대비) 트렌드 분석.

    최근 N개월 데이터에 대해 각 스테이지의 전월 대비 변화율을 계산합니다.

    Args:
        df: YEAR_MONTH, CATEGORY, *_COUNT 컬럼을 포함한 DataFrame.
        months: 분석 기간 (최근 N개월, 기본 6).

    Returns:
        YEAR_MONTH, CATEGORY, STAGE, COUNT, MOM_CHANGE, MOM_PCT 컬럼 DataFrame.
    """
    if df.empty:
        return pd.DataFrame(
            columns=["YEAR_MONTH", "CATEGORY", "STAGE", "COUNT", "MOM_CHANGE", "MOM_PCT"]
        )

    # 최근 N개월 필터링
    if "YEAR_MONTH" in df.columns:
        sorted_months = sorted(df["YEAR_MONTH"].unique(), reverse=True)
        recent_months = sorted_months[:months]
        filtered = df[df["YEAR_MONTH"].isin(recent_months)].copy()
    else:
        filtered = df.copy()

    if filtered.empty:
        return pd.DataFrame()

    # 스테이지별 long format 변환
    available_stages = [
        (stage, _STAGE_COUNT_COLS[stage])
        for stage in FUNNEL_STAGES
        if _STAGE_COUNT_COLS[stage] in filtered.columns
    ]

    if not available_stages:
        return pd.DataFrame()

    group_cols = [c for c in ["YEAR_MONTH", "CATEGORY"] if c in filtered.columns]
    records: list[dict] = []

    categories = filtered["CATEGORY"].unique() if "CATEGORY" in filtered.columns else [None]

    for cat in categories:
        cat_data = (
            filtered[filtered["CATEGORY"] == cat] if cat is not None else filtered
        )
        cat_data = cat_data.sort_values("YEAR_MONTH") if "YEAR_MONTH" in cat_data.columns else cat_data

        for stage_name, col_name in available_stages:
            prev_count: Optional[float] = None
            for _, row in cat_data.iterrows():
                count = row.get(col_name, 0)
                mom_change = count - prev_count if prev_count is not None else np.nan
                mom_pct = (
                    mom_change / prev_count if prev_count is not None and prev_count > 0
                    else np.nan
                )

                entry: dict = {"STAGE": stage_name, "COUNT": count}
                if "YEAR_MONTH" in row.index:
                    entry["YEAR_MONTH"] = row["YEAR_MONTH"]
                if cat is not None:
                    entry["CATEGORY"] = cat
                entry["MOM_CHANGE"] = round(mom_change, 2) if not np.isnan(mom_change) else np.nan
                entry["MOM_PCT"] = round(mom_pct, 4) if not np.isnan(mom_pct) else np.nan

                records.append(entry)
                prev_count = count

    return pd.DataFrame(records)


def compare_categories(df: pd.DataFrame) -> pd.DataFrame:
    """전체 카테고리 간 퍼널 전환율 비교.

    각 카테고리의 최신 월 데이터를 기준으로 스테이지별 전환율을 비교합니다.

    Args:
        df: YEAR_MONTH, CATEGORY, *_COUNT 컬럼을 포함한 DataFrame.

    Returns:
        CATEGORY, STAGE_PAIR, CVR 컬럼으로 피벗된 비교 DataFrame.
    """
    if df.empty or "CATEGORY" not in df.columns:
        return pd.DataFrame()

    # 최신 월 데이터만 사용
    if "YEAR_MONTH" in df.columns:
        latest_month = df["YEAR_MONTH"].max()
        latest = df[df["YEAR_MONTH"] == latest_month].copy()
    else:
        latest = df.copy()

    stage_drops = compute_stage_drops(latest)
    if stage_drops.empty:
        return pd.DataFrame()

    # CATEGORY x STAGE_PAIR 피벗
    stage_drops = stage_drops.assign(
        STAGE_PAIR=stage_drops["FROM_STAGE"] + " → " + stage_drops["TO_STAGE"]
    )

    if "CATEGORY" not in stage_drops.columns:
        return stage_drops

    pivot = stage_drops.pivot_table(
        index="CATEGORY",
        columns="STAGE_PAIR",
        values="CVR",
        aggfunc="mean",
    ).reset_index()

    return pivot
