"""전역 사이드바: 필터 + 상태 + 품질 + 리니지.

모든 페이지에서 `render_sidebar()`를 호출하면
전역 카테고리/기간 필터 + Snowflake 상태 + 품질 요약이 표시된다.
필터 값은 st.session_state에 저장되어 전 페이지에서 공유된다.
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------
_CATEGORIES = ["전체", "인터넷", "렌탈", "모바일", "알뜰 요금제", "유심만"]

_TECH_STACK = [
    ("Cortex COMPLETE", "llama3.1-405b"),
    ("Cortex FORECAST", "시도별 계약수 3개월"),
    ("Cortex ANOMALY", "계약수 급변 탐지"),
    ("Markov Chain", "흡수 마르코프 전이 분석"),
    ("STL Decomposition", "계절성/추세 분해"),
    ("XGBoost", "전환율 3-class 분류"),
    ("Multi-Agent", "Analyst → Strategy → Synth"),
    ("DMF", "데이터 품질 자동 검증"),
    ("Lineage", "테이블 의존성 추적"),
]


@st.cache_resource
def _sidebar_client():
    """사이드바 전용 SnowflakeClient 싱글턴."""
    try:
        from data.snowflake_client import SnowflakeClient
        return SnowflakeClient()
    except Exception:
        return None


def render_sidebar() -> dict:
    """사이드바를 렌더링하고 필터 값을 반환.

    Returns:
        {"category": str | None}
    """
    with st.sidebar:
        # --- 연결 상태 ---
        st.markdown("**Cortex Analytics**")
        _show_connection_status()

        st.divider()

        # --- 전역 필터 ---
        st.markdown("**필터**")
        category = st.selectbox(
            "카테고리",
            _CATEGORIES,
            index=_CATEGORIES.index(st.session_state.get("global_category", "전체")),
            key="sidebar_category",
        )
        st.session_state["global_category"] = category

        st.divider()

        # --- 데이터 품질 ---
        st.markdown("**데이터 품질**")
        _show_data_quality()

        st.divider()

        # --- 리니지 ---
        st.markdown("**파이프라인 리니지**")
        _show_lineage()

        st.divider()

        # --- 기술 스택 ---
        with st.expander("사용 기술", expanded=False):
            for name, desc in _TECH_STACK:
                st.caption(f"**{name}** — {desc}")

    cat_value = None if category == "전체" else category
    return {"category": cat_value}


def _show_connection_status() -> None:
    """Snowflake 연결 상태 표시."""
    try:
        client = _sidebar_client()
        if client is None:
            st.error("Snowflake 미연결", icon="🔴")
            return
        result = client._query("SELECT CURRENT_ACCOUNT() AS ACCT, CURRENT_WAREHOUSE() AS WH")
        if not result.empty:
            acct = result["ACCT"].iloc[0]
            wh = result["WH"].iloc[0]
            st.success(f"Connected", icon="🟢")
            st.caption(f"Account: {acct}")
            st.caption(f"Warehouse: {wh}")
        else:
            st.warning("연결 확인 실패")
    except Exception:
        st.error("Snowflake 미연결", icon="🔴")


def _show_data_quality() -> None:
    """데이터 품질 요약 표시."""
    try:
        client = _sidebar_client()
        dq_df = client.load_data_quality()
        if not dq_df.empty:
            status_col = "QUALITY_STATUS" if "QUALITY_STATUS" in dq_df.columns else None
            if status_col is None:
                st.caption("DMF 미설정")
                return
            pass_n = len(dq_df[dq_df[status_col] == "PASS"])
            warn_n = len(dq_df[dq_df[status_col] == "WARNING"])
            crit_n = len(dq_df[dq_df[status_col] == "CRITICAL"])

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("PASS", pass_n)
            with c2:
                st.metric("WARN", warn_n)
            with c3:
                st.metric("CRIT", crit_n)

            if crit_n > 0:
                st.error(f"{crit_n}건 품질 위반")
            elif warn_n > 0:
                st.warning(f"{warn_n}건 주의")
            else:
                st.caption("모든 검사 통과")
        else:
            st.caption("DMF 미실행")
    except Exception:
        st.caption("DMF 미설정")


def _show_lineage() -> None:
    """파이프라인 리니지 요약 표시."""
    try:
        client = _sidebar_client()
        ln_df = client.load_lineage_summary()
        if not ln_df.empty:
            for _, row in ln_df.iterrows():
                up = row.get("UPSTREAM_LAYER", "?")
                down = row.get("DOWNSTREAM_LAYER", "?")
                cnt = row.get("DEPENDENCY_COUNT", 0)
                st.caption(f"{up} → {down}  ({cnt})")
        else:
            st.caption("리니지 미생성")
    except Exception:
        st.caption("리니지 미설정")
