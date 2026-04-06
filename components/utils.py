"""공통 Streamlit 유틸리티.

모든 페이지에서 공유하는 데이터 처리, 에러 핸들링, 캐싱 함수.
코드 중복을 제거하고 일관된 UX를 보장한다.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 상수 (페이지별 중복 제거 → 여기서 통합)
# ---------------------------------------------------------------------------
STAGE_LABELS: dict[str, str] = {
    "CONSULT_REQUEST": "상담요청",
    "SUBSCRIPTION": "가입신청",
    "REGISTEND": "접수완료",
    "OPEN": "개통",
    "PAYEND": "납입완료",
}

STAGE_COLORS: dict[str, str] = {
    "상담요청": "#22d3ee",
    "가입신청": "#818cf8",
    "접수완료": "#c084fc",
    "개통": "#f472b6",
    "납입완료": "#34d399",
}

MAJOR_CATEGORIES: list[str] = [
    "인터넷",
    "렌탈",
    "모바일",
    "알뜰 요금제",
    "유심만",
]


# ---------------------------------------------------------------------------
# 데이터 처리 헬퍼
# ---------------------------------------------------------------------------


def drop_incomplete_month(
    df: pd.DataFrame,
    col: str = "YEAR_MONTH",
) -> pd.DataFrame:
    """미완성 월(최신 월) 데이터를 제거.

    Args:
        df: 입력 DataFrame
        col: 년월 컬럼명

    Returns:
        최신 월이 제거된 DataFrame (불변 — 새 DataFrame 반환)
    """
    if df.empty or col not in df.columns:
        return df
    return df[df[col] < df[col].max()].copy()


def filter_major_categories(
    df: pd.DataFrame,
    col: str = "MAIN_CATEGORY_NAME",
) -> pd.DataFrame:
    """주요 카테고리만 필터링.

    Args:
        df: 입력 DataFrame
        col: 카테고리 컬럼명

    Returns:
        주요 카테고리만 포함된 DataFrame
    """
    if df.empty or col not in df.columns:
        return df
    return df[df[col].isin(MAJOR_CATEGORIES)].copy()


def validate_columns(
    df: pd.DataFrame,
    required: list[str],
    context: str = "",
) -> bool:
    """DataFrame에 필수 컬럼이 있는지 검증.

    Args:
        df: 검증할 DataFrame
        required: 필수 컬럼 리스트
        context: 에러 메시지에 포함할 컨텍스트

    Returns:
        True면 모든 필수 컬럼 존재
    """
    if df.empty:
        return False
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning(
            "%s — 필수 컬럼 누락: %s (실제: %s)",
            context,
            missing,
            list(df.columns),
        )
        return False
    return True


# ---------------------------------------------------------------------------
# 에러 핸들링 래퍼
# ---------------------------------------------------------------------------


def safe_data_load(
    loader: Callable[[], pd.DataFrame],
    error_msg: str = "데이터를 불러올 수 없습니다.",
    show_warning: bool = True,
) -> pd.DataFrame:
    """데이터 로드를 안전하게 실행하고, 실패 시 빈 DataFrame 반환.

    기존의 bare `try/except: pass` 패턴을 대체하여
    사용자에게 에러를 명확히 표시한다.

    Args:
        loader: DataFrame을 반환하는 호출 가능 객체
        error_msg: 실패 시 표시할 메시지
        show_warning: True면 st.warning으로 사용자에게 표시

    Returns:
        로드된 DataFrame 또는 빈 DataFrame
    """
    try:
        return loader()
    except Exception as exc:
        logger.exception("데이터 로드 실패: %s", error_msg)
        if show_warning:
            st.warning(f"{error_msg} ({type(exc).__name__})", icon="⚠️")
        return pd.DataFrame()


def safe_render(
    render_fn: Callable[[], None],
    fallback_msg: str = "이 섹션을 렌더링할 수 없습니다.",
) -> None:
    """UI 렌더링을 안전하게 실행.

    차트 생성이나 복잡한 UI 렌더링이 실패해도
    전체 페이지가 죽지 않도록 보호한다.

    Args:
        render_fn: UI를 렌더링하는 호출 가능 객체
        fallback_msg: 실패 시 표시할 메시지
    """
    try:
        render_fn()
    except Exception as exc:
        logger.exception("렌더링 실패: %s", fallback_msg)
        st.error(f"{fallback_msg} ({type(exc).__name__}: {exc})", icon="❌")


# ---------------------------------------------------------------------------
# Snowflake 클라이언트 캐싱
# ---------------------------------------------------------------------------


@st.cache_resource
def get_cached_client():
    """Snowflake 클라이언트 싱글턴 (전 페이지 공유).

    Returns:
        SnowflakeClient 인스턴스 또는 None
    """
    try:
        from data.snowflake_client import SnowflakeClient
        return SnowflakeClient()
    except Exception as exc:
        logger.error("SnowflakeClient 초기화 실패: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Plotly 다크 테마 기본 레이아웃
# ---------------------------------------------------------------------------

PLOTLY_DARK_LAYOUT: dict = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "rgba(255,255,255,0.85)", "size": 12},
    "margin": {"l": 40, "r": 20, "t": 40, "b": 40},
    "xaxis": {
        "gridcolor": "rgba(255,255,255,0.06)",
        "zerolinecolor": "rgba(255,255,255,0.06)",
    },
    "yaxis": {
        "gridcolor": "rgba(255,255,255,0.06)",
        "zerolinecolor": "rgba(255,255,255,0.06)",
    },
    "legend": {"bgcolor": "rgba(0,0,0,0)"},
}
