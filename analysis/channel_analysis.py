"""채널 효율성 및 ROI 분석 모듈.

채널별 전환율, 매출 효율, 집중도(HHI), 성장 추세를 분석합니다.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config.constants import CHANNEL_MIN_VOLUME, DEFAULT_WEIGHTS

logger = logging.getLogger(__name__)


def compute_channel_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """채널별 효율성 점수 계산.

    가중 합산 방식으로 납입전환율, 평균매출, 계약건수를 종합합니다.
    원본 DataFrame을 변경하지 않습니다.

    Args:
        df: STG_CHANNEL 또는 V_CHANNEL_PERFORMANCE 데이터.
            CONTRACT_COUNT / REGISTEND_COUNT, OPEN_CVR, PAYEND_CVR,
            AVG_NET_SALES 컬럼 필요.

    Returns:
        원본 컬럼 + EFFICIENCY_SCORE, EFFICIENCY_RANK 가 추가된 DataFrame.
    """
    if df.empty:
        return df.copy()

    result = df.copy()

    # 정규화 대상 컬럼 확인
    numeric_cols = {
        "payend_cvr": "PAYEND_CVR",
        "avg_net_sales": "AVG_NET_SALES",
        "contract_volume": "CONTRACT_COUNT",
    }

    available: dict[str, str] = {}
    for key, col in numeric_cols.items():
        if col in result.columns:
            available[key] = col
        elif key == "contract_volume" and "REGISTEND_COUNT" in result.columns:
            available[key] = "REGISTEND_COUNT"

    if not available:
        logger.warning("효율성 계산에 필요한 수치 컬럼이 없습니다.")
        return result

    # Min-Max 정규화
    for key, col in available.items():
        col_min = result[col].min()
        col_max = result[col].max()
        col_range = col_max - col_min
        norm_col = f"_NORM_{key}"
        result[norm_col] = (
            (result[col] - col_min) / col_range if col_range > 0 else 0.0
        )

    # 가중 합산
    score = pd.Series(0.0, index=result.index)
    total_weight = 0.0
    for key in available:
        weight = DEFAULT_WEIGHTS.get(key, 0.0)
        score = score + result[f"_NORM_{key}"] * weight
        total_weight += weight

    result["EFFICIENCY_SCORE"] = (
        round(score / total_weight, 4) if total_weight > 0 else 0.0
    )

    # 정규화 임시 컬럼 제거
    norm_cols = [c for c in result.columns if c.startswith("_NORM_")]
    result = result.drop(columns=norm_cols)

    # 순위 (내림차순)
    result["EFFICIENCY_RANK"] = (
        result["EFFICIENCY_SCORE"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    return result


def compute_channel_hhi(df: pd.DataFrame) -> pd.DataFrame:
    """채널 집중도 HHI(Herfindahl-Hirschman Index) 계산.

    카테고리 + 월 별로 각 채널이 전체 계약에서 차지하는 점유율의
    제곱합(HHI)을 산출합니다. HHI가 높을수록 특정 채널에 집중.

    Args:
        df: YEAR_MONTH, CATEGORY, CHANNEL (또는 INFLOW_PATH),
            CONTRACT_COUNT 컬럼 필요.

    Returns:
        YEAR_MONTH, CATEGORY, HHI, TOP_CHANNEL, TOP_SHARE 컬럼 DataFrame.
    """
    if df.empty:
        return pd.DataFrame(
            columns=["YEAR_MONTH", "CATEGORY", "HHI", "TOP_CHANNEL", "TOP_SHARE"]
        )

    channel_col = _resolve_channel_col(df)
    count_col = _resolve_count_col(df)

    if channel_col is None or count_col is None:
        logger.warning("HHI 계산에 필요한 컬럼 없음.")
        return pd.DataFrame()

    group_cols = [c for c in ["YEAR_MONTH", "CATEGORY"] if c in df.columns]
    if not group_cols:
        group_cols = ["CATEGORY"] if "CATEGORY" in df.columns else []

    records: list[dict] = []

    groups = df.groupby(group_cols) if group_cols else [(None, df)]
    for group_key, group_df in groups:
        total = group_df[count_col].sum()
        if total == 0:
            continue

        shares = group_df[count_col] / total
        hhi = round((shares**2).sum(), 6)

        top_idx = group_df[count_col].idxmax()
        top_channel = group_df.loc[top_idx, channel_col]
        top_share = round(group_df.loc[top_idx, count_col] / total, 4)

        entry: dict = {"HHI": hhi, "TOP_CHANNEL": top_channel, "TOP_SHARE": top_share}
        if isinstance(group_key, tuple):
            for col, val in zip(group_cols, group_key):
                entry[col] = val
        elif group_key is not None:
            entry[group_cols[0]] = group_key

        records.append(entry)

    return pd.DataFrame(records)


def classify_channel_growth(
    df: pd.DataFrame,
    window: int = 6,
) -> pd.DataFrame:
    """채널별 성장/안정/감소 분류.

    최근 window 개월간의 계약 건수 추이를 선형 회귀로 판별합니다.

    Args:
        df: YEAR_MONTH, CHANNEL/INFLOW_PATH, CONTRACT_COUNT 컬럼 필요.
        window: 분석 기간 (개월 수, 기본 6).

    Returns:
        CHANNEL, CATEGORY(있으면), GROWTH_RATE, TREND_CLASS 컬럼 DataFrame.
        TREND_CLASS: 'GROWTH' / 'STABLE' / 'DECLINE'.
    """
    if df.empty:
        return pd.DataFrame(
            columns=["CHANNEL", "GROWTH_RATE", "TREND_CLASS"]
        )

    channel_col = _resolve_channel_col(df)
    count_col = _resolve_count_col(df)

    if channel_col is None or count_col is None:
        return pd.DataFrame()

    # 최근 window개월 필터
    if "YEAR_MONTH" in df.columns:
        sorted_months = sorted(df["YEAR_MONTH"].unique(), reverse=True)
        recent_months = sorted_months[:window]
        filtered = df[df["YEAR_MONTH"].isin(recent_months)].copy()
    else:
        filtered = df.copy()

    group_keys = [channel_col]
    if "CATEGORY" in filtered.columns:
        group_keys = ["CATEGORY", channel_col]

    records: list[dict] = []

    for key, group in filtered.groupby(group_keys):
        if len(group) < 2:
            continue

        group_sorted = group.sort_values("YEAR_MONTH") if "YEAR_MONTH" in group.columns else group
        counts = group_sorted[count_col].values.astype(float)

        # 간단한 선형 추세 (numpy polyfit degree 1)
        x = np.arange(len(counts))
        if counts.sum() == 0:
            growth_rate = 0.0
        else:
            try:
                slope, _ = np.polyfit(x, counts, 1)
                avg_count = counts.mean()
                growth_rate = slope / avg_count if avg_count > 0 else 0.0
            except (np.linalg.LinAlgError, ValueError):
                growth_rate = 0.0

        # 분류
        if growth_rate > 0.05:
            trend_class = "GROWTH"
        elif growth_rate < -0.05:
            trend_class = "DECLINE"
        else:
            trend_class = "STABLE"

        entry: dict = {
            "CHANNEL": key[-1] if isinstance(key, tuple) else key,
            "GROWTH_RATE": round(growth_rate, 4),
            "TREND_CLASS": trend_class,
        }
        if "CATEGORY" in filtered.columns and isinstance(key, tuple):
            entry["CATEGORY"] = key[0]

        records.append(entry)

    return pd.DataFrame(records)


def rank_channels(
    df: pd.DataFrame,
    category: Optional[str] = None,
    top_n: int = 15,
) -> pd.DataFrame:
    """효율성 점수 기반 채널 순위.

    Args:
        df: STG_CHANNEL 또는 V_CHANNEL_PERFORMANCE 데이터.
        category: 필터할 카테고리. None이면 전체.
        top_n: 반환할 상위 채널 수 (기본 15).

    Returns:
        효율성 점수 기준 상위 top_n 채널 DataFrame.
    """
    if df.empty:
        return pd.DataFrame()

    filtered = df.copy()
    if category is not None and "CATEGORY" in filtered.columns:
        filtered = filtered[filtered["CATEGORY"] == category].copy()

    if filtered.empty:
        return pd.DataFrame()

    # 최소 계약 건수 필터
    count_col = _resolve_count_col(filtered)
    if count_col is not None and count_col in filtered.columns:
        filtered = filtered[filtered[count_col] >= CHANNEL_MIN_VOLUME].copy()

    scored = compute_channel_efficiency(filtered)

    if "EFFICIENCY_SCORE" not in scored.columns:
        return scored.head(top_n)

    return (
        scored.sort_values("EFFICIENCY_SCORE", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------

def _resolve_channel_col(df: pd.DataFrame) -> Optional[str]:
    """채널 이름 컬럼 해소."""
    for col in ("CHANNEL", "INFLOW_PATH", "RECEIVE_PATH_NAME"):
        if col in df.columns:
            return col
    return None


def _resolve_count_col(df: pd.DataFrame) -> Optional[str]:
    """건수 컬럼 해소."""
    for col in ("CONTRACT_COUNT", "REGISTEND_COUNT", "OPEN_COUNT", "PAYEND_COUNT"):
        if col in df.columns:
            return col
    return None
