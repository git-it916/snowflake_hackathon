"""지역 수요 분석 및 클러스터링 모듈.

시/도, 시/군/구 단위의 수요 점수, 성장 지역 탐지,
결합 비율, K-Means 클러스터링을 수행합니다.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config.constants import GROWTH_THRESHOLD

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Z-score 수요 점수 대상 지표
# ---------------------------------------------------------------------------
_DEMAND_METRICS: list[str] = [
    "CONTRACT_COUNT",
    "CONSULT_REQUEST_COUNT",
    "PAYEND_COUNT",
    "AVG_NET_SALES",
    "TOTAL_NET_SALES",
]


def compute_demand_score(df: pd.DataFrame) -> pd.DataFrame:
    """도시별 Z-score 기반 복합 수요 점수 계산.

    사용 가능한 수요 지표들의 Z-score를 평균하여 DEMAND_SCORE를 산출합니다.
    원본 DataFrame을 변경하지 않습니다.

    Args:
        df: STG_REGIONAL 또는 REGIONAL_DEMAND_SCORE 데이터.
            STATE, CITY, 수요 지표 컬럼 필요.

    Returns:
        원본 컬럼 + Z_* (지표별 Z-score), DEMAND_SCORE, DEMAND_RANK DataFrame.
    """
    if df.empty:
        return df.copy()

    result = df.copy()

    available_metrics = [m for m in _DEMAND_METRICS if m in result.columns]
    if not available_metrics:
        logger.warning("수요 점수 계산에 필요한 지표 컬럼이 없습니다.")
        return result

    # 지표별 Z-score 계산
    z_cols: list[str] = []
    for metric in available_metrics:
        z_col = f"Z_{metric}"
        mean_val = result[metric].mean()
        std_val = result[metric].std()
        result[z_col] = (
            (result[metric] - mean_val) / std_val if std_val > 0 else 0.0
        )
        z_cols.append(z_col)

    # 복합 수요 점수 = Z-score 평균
    result["DEMAND_SCORE"] = round(result[z_cols].mean(axis=1), 4)

    # 순위 (내림차순)
    result["DEMAND_RANK"] = (
        result["DEMAND_SCORE"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    return result


def detect_growth_regions(
    df: pd.DataFrame,
    threshold: float = GROWTH_THRESHOLD,
) -> pd.DataFrame:
    """전월 대비 성장률이 임계값을 초과하는 도시 탐지.

    Args:
        df: YEAR_MONTH, STATE, CITY, CONTRACT_COUNT (또는 PAYEND_COUNT) 필요.
        threshold: 성장 판단 임계값 (기본 0.10 = 10%).

    Returns:
        STATE, CITY, YEAR_MONTH, PREV_COUNT, CURR_COUNT,
        MOM_GROWTH, IS_GROWTH 컬럼 DataFrame.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "STATE", "CITY", "YEAR_MONTH",
                "PREV_COUNT", "CURR_COUNT", "MOM_GROWTH", "IS_GROWTH",
            ]
        )

    count_col = _resolve_count_col(df)
    if count_col is None:
        logger.warning("성장 지역 탐지에 필요한 건수 컬럼 없음.")
        return pd.DataFrame()

    location_cols = _resolve_location_cols(df)
    if not location_cols:
        logger.warning("STATE/CITY 컬럼 없음.")
        return pd.DataFrame()

    if "YEAR_MONTH" not in df.columns:
        logger.warning("YEAR_MONTH 컬럼 없음.")
        return pd.DataFrame()

    sorted_df = df.sort_values(location_cols + ["YEAR_MONTH"]).copy()

    records: list[dict] = []

    for key, group in sorted_df.groupby(location_cols):
        group_sorted = group.sort_values("YEAR_MONTH")
        counts = group_sorted[count_col].values
        months = group_sorted["YEAR_MONTH"].values

        for i in range(1, len(counts)):
            prev_count = counts[i - 1]
            curr_count = counts[i]
            mom_growth = (
                (curr_count - prev_count) / prev_count
                if prev_count > 0
                else np.nan
            )

            entry: dict = {
                "YEAR_MONTH": months[i],
                "PREV_COUNT": prev_count,
                "CURR_COUNT": curr_count,
                "MOM_GROWTH": round(mom_growth, 4) if not np.isnan(mom_growth) else np.nan,
                "IS_GROWTH": mom_growth > threshold if not np.isnan(mom_growth) else False,
            }
            if isinstance(key, tuple):
                for col, val in zip(location_cols, key):
                    entry[col] = val
            else:
                entry[location_cols[0]] = key

            records.append(entry)

    result = pd.DataFrame(records)
    if result.empty:
        return result

    # IS_GROWTH == True 인 것만 반환
    growth_only = result[result["IS_GROWTH"]].copy()
    return growth_only.sort_values("MOM_GROWTH", ascending=False).reset_index(drop=True)


def compute_bundle_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """도시별 결합 상품 비율 계산.

    BUNDLE_COUNT / (BUNDLE_COUNT + STANDALONE_COUNT) 비율을 산출합니다.

    Args:
        df: STATE, CITY, BUNDLE_COUNT, STANDALONE_COUNT 컬럼 필요.

    Returns:
        원본 그룹 컬럼 + BUNDLE_RATIO, TOTAL_PRODUCTS DataFrame.
    """
    if df.empty:
        return pd.DataFrame()

    if "BUNDLE_COUNT" not in df.columns or "STANDALONE_COUNT" not in df.columns:
        logger.warning("BUNDLE_COUNT / STANDALONE_COUNT 컬럼 없음.")
        return df.copy()

    result = df.copy()
    total = result["BUNDLE_COUNT"] + result["STANDALONE_COUNT"]
    result["TOTAL_PRODUCTS"] = total
    result["BUNDLE_RATIO"] = np.where(
        total > 0,
        np.round(result["BUNDLE_COUNT"] / total, 4),
        0.0,
    )

    return result


def cluster_regions(
    df: pd.DataFrame,
    n_clusters: int = 5,
) -> pd.DataFrame:
    """K-Means 기반 지역 클러스터링.

    수요 지표를 기반으로 도시들을 n_clusters 그룹으로 분류합니다.
    sklearn이 없는 환경에서는 수요 점수 기반 분위수 분류로 대체합니다.

    Args:
        df: STATE, CITY, 수요 지표 컬럼 필요.
        n_clusters: 클러스터 수 (기본 5).

    Returns:
        원본 컬럼 + CLUSTER, CLUSTER_LABEL DataFrame.
    """
    if df.empty:
        return df.copy()

    available_metrics = [m for m in _DEMAND_METRICS if m in df.columns]
    if not available_metrics:
        logger.warning("클러스터링에 필요한 수요 지표 컬럼이 없습니다.")
        return df.copy()

    result = df.copy()

    # 수치 피처 추출
    features = result[available_metrics].fillna(0).values

    if len(features) < n_clusters:
        logger.warning(
            "데이터 행(%d)이 클러스터 수(%d)보다 적어 단일 클러스터 할당.",
            len(features),
            n_clusters,
        )
        result["CLUSTER"] = 0
        result["CLUSTER_LABEL"] = "CLUSTER_0"
        return result

    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans

        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled)

        result["CLUSTER"] = labels

    except ImportError:
        logger.info("sklearn 미설치. 분위수 기반 분류로 대체합니다.")
        scored = compute_demand_score(result)
        if "DEMAND_SCORE" in scored.columns:
            result["CLUSTER"] = pd.qcut(
                scored["DEMAND_SCORE"],
                q=min(n_clusters, len(scored)),
                labels=False,
                duplicates="drop",
            )
        else:
            result["CLUSTER"] = 0

    # 클러스터 라벨 할당
    result["CLUSTER_LABEL"] = "CLUSTER_" + result["CLUSTER"].astype(str)

    # 클러스터별 대표 통계 로깅
    for cluster_id in sorted(result["CLUSTER"].unique()):
        cluster_data = result[result["CLUSTER"] == cluster_id]
        size = len(cluster_data)
        avg_metrics = {
            m: round(cluster_data[m].mean(), 2)
            for m in available_metrics
            if m in cluster_data.columns
        }
        logger.info("Cluster %d: %d 도시, 평균=%s", cluster_id, size, avg_metrics)

    return result


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------

def _resolve_count_col(df: pd.DataFrame) -> Optional[str]:
    """건수 컬럼 해소."""
    for col in ("CONTRACT_COUNT", "PAYEND_COUNT", "CONSULT_REQUEST_COUNT"):
        if col in df.columns:
            return col
    return None


def _resolve_location_cols(df: pd.DataFrame) -> list[str]:
    """위치 컬럼 해소."""
    cols: list[str] = []
    if "STATE" in df.columns:
        cols.append("STATE")
    if "CITY" in df.columns:
        cols.append("CITY")
    return cols
