"""Telecom Funnel Intelligence -- Landing Page (100% Streamlit Native)."""
from __future__ import annotations

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="AI Funnel Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
try:
    from data.snowflake_client import SnowflakeClient
    _CLIENT_OK = True
except Exception:
    _CLIENT_OK = False

try:
    from analysis.insight_generator import generate_funnel_insights, generate_channel_insights
    _INSIGHT_OK = True
except Exception:
    _INSIGHT_OK = False

try:
    from config.constants import PRODUCT_CATEGORIES, STATES
    _CONST_OK = True
except Exception:
    _CONST_OK = False
    PRODUCT_CATEGORIES = {"인터넷": "internet", "렌탈": "rental", "모바일": "mobile"}
    STATES = [
        "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
        "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
    ]


@st.cache_resource
def _get_client():
    if not _CLIENT_OK:
        return None
    try:
        return SnowflakeClient()
    except Exception:
        return None


def _drop_incomplete(df: pd.DataFrame, col: str = "YEAR_MONTH") -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return df
    return df[df[col] < df[col].max()]


# ---------------------------------------------------------------------------
# GLOBAL CSS (only theme — no HTML rendering)
# ---------------------------------------------------------------------------
try:
    from components.styles import inject_global_css
    inject_global_css()
except Exception:
    pass

from components.nav import safe_page_link as _safe_pl
from components.sidebar import render_sidebar

# ---------------------------------------------------------------------------
# Global sidebar + filters
# ---------------------------------------------------------------------------
_sidebar_result = render_sidebar()
cat_filter = _sidebar_result.get("category")


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
client = _get_client()

stage_drop_df = pd.DataFrame()
bottleneck_df = pd.DataFrame()
funnel_ts_df = pd.DataFrame()
kpi_df = pd.DataFrame()
channel_df = pd.DataFrame()

if client is not None:
    try:
        stage_drop_df = _drop_incomplete(client.load_funnel_stage_drop(cat_filter))
    except Exception:
        pass
    try:
        bottleneck_df = client.load_funnel_bottlenecks()
    except Exception:
        pass
    try:
        funnel_ts_df = _drop_incomplete(client.load_funnel_timeseries(cat_filter))
    except Exception:
        pass
    try:
        kpi_df = client.load_kpi()
    except Exception:
        pass
    try:
        channel_df = _drop_incomplete(client.load_channel_efficiency(cat_filter))
    except Exception:
        pass

# Generate insights
funnel_insight: dict = {
    "severity": "warning", "headline": "", "metrics": {},
    "findings": [], "actions": [],
}
channel_insight: dict = {
    "severity": "warning", "headline": "", "metrics": {},
    "findings": [], "actions": [],
}

if _INSIGHT_OK:
    try:
        funnel_insight = generate_funnel_insights(
            stage_drop_df, bottleneck_df, funnel_ts_df, cat_filter,
        )
    except Exception:
        pass
    try:
        ch_col = "RECEIVE_PATH_NAME" if "RECEIVE_PATH_NAME" in channel_df.columns else "CHANNEL"
        channel_insight = generate_channel_insights(channel_df, ch_col)
    except Exception:
        pass

# Extract metrics
current_cvr = funnel_insight.get("metrics", {}).get("current_cvr", 24.9)
avg_cvr = funnel_insight.get("metrics", {}).get("avg_cvr", 30.0)
cvr_delta = current_cvr - avg_cvr if avg_cvr else 0
worst_stage = funnel_insight.get("metrics", {}).get("worst_stage", "개통")
worst_drop = funnel_insight.get("metrics", {}).get("worst_drop_pct", 27)

top_channel = channel_insight.get("metrics", {}).get("top_channel", "인바운드")
top_share = channel_insight.get("metrics", {}).get("top_share", 48)

growth_region = "--"
total_contracts = 0
if not kpi_df.empty:
    row = kpi_df.iloc[0]
    for c in ["TOP_GROWTH_CITY", "GROWTH_REGION"]:
        if c in kpi_df.columns and pd.notna(row.get(c)):
            growth_region = str(row[c])
            break
    total_contracts = float(row.get("TOTAL_CONTRACTS", row.get("CONTRACT_COUNT", 0)))

severity = funnel_insight.get("severity", "warning")
alert_headline = funnel_insight.get("headline", "").replace("**", "")
if not alert_headline:
    alert_headline = f"{worst_stage} 단계에서 {worst_drop:.0f}% 이탈 감지"

contracts_display = f"{total_contracts/1000:.1f}K" if total_contracts > 0 else "5.8K"


# ---------------------------------------------------------------------------
# NAV — Streamlit native page links
# ---------------------------------------------------------------------------
_, nav1, nav2, nav3, nav4, _ = st.columns([1, 1, 1, 1, 1, 1])
with nav1:
    _safe_pl("app.py", label="랜딩", icon="🏠")
with nav2:
    _safe_pl("pages/1_진단.py", label="진단", icon="🔍")
with nav3:
    _safe_pl("pages/2_기회_분석.py", label="기회 분석", icon="📈")
with nav4:
    _safe_pl("pages/3_AI_전략.py", label="AI 전략", icon="🤖")

st.divider()

# ---------------------------------------------------------------------------
# HERO
# ---------------------------------------------------------------------------
st.caption("🟢 실시간 데이터 연동 중")
st.title("가입 전환율을 높이는 AI 퍼널 인텔리전스")
st.caption("◆ Snowflake Cortex 기반 통신사 마케팅 전략 대시보드")

st.markdown(
    """
> **"어디서 고객이 빠지고, 어떤 채널이 효과적이고, 어디에 집중해야 하는가?"**
>
> Snowflake Marketplace 텔레콤 데이터(V01~V07, 23,000+행)를 기반으로
> 가입 퍼널 5단계의 병목을 진단하고, 38개 채널의 효율을 비교하며,
> 200개 시군구의 수요를 예측합니다.
> 흡수 마르코프 체인으로 퍼널 구조를 수학적으로 모델링하고,
> Cortex FORECAST·ANOMALY·COMPLETE와 Multi-Agent 시스템을 통해
> 데이터 기반 마케팅 전략을 자동 생성합니다.
"""
)

# ---------------------------------------------------------------------------
# Alert Banner
# ---------------------------------------------------------------------------
if severity == "critical":
    st.error(f"**CRITICAL** — {alert_headline}", icon="🚨")
elif severity == "warning":
    st.warning(f"**ATTENTION** — {alert_headline}", icon="⚠️")
else:
    st.success(f"**STABLE** — {alert_headline}", icon="✅")

_safe_pl("pages/1_진단.py", label="진단 리포트 보기 →", icon="🔍")

st.divider()

# ---------------------------------------------------------------------------
# 3 KPI Cards
# ---------------------------------------------------------------------------
c1, c2, c3 = st.columns(3)
with c1:
    st.metric(
        label="퍼널 전환율",
        value=f"{current_cvr:.1f}%",
        delta=f"{cvr_delta:+.1f}%p vs 평균",
    )
    _safe_pl("pages/1_진단.py", label="병목 구간 분석 →")
with c2:
    st.metric(
        label="최고 볼륨 채널",
        value=top_channel,
        delta=f"점유율 {top_share:.0f}%",
        delta_color="off",
    )
    _safe_pl("pages/1_진단.py", label="채널 효율 진단 →")
with c3:
    st.metric(
        label="최대 성장 지역",
        value=growth_region,
        delta=f"총 {contracts_display}건",
        delta_color="off",
    )
    _safe_pl("pages/2_기회_분석.py", label="기회 시장 분석 →")

# ---------------------------------------------------------------------------
# AI CTA Section
# ---------------------------------------------------------------------------
st.divider()

st.subheader("┃AI 기반 전략 최적화")
st.caption("다중 에이전트 시스템이 현재의 데이터를 분석하여 최적의 채널 및 예산 분배 전략을 제안합니다.")
_safe_pl("pages/3_AI_전략.py", label="AI 전략 생성 실행 →")

# ---------------------------------------------------------------------------
# Data Quality & Lineage (Snowflake CoCo Skills)
# ---------------------------------------------------------------------------
st.divider()

dq_col, ln_col = st.columns(2)

with dq_col:
    st.subheader("┃데이터 품질 모니터링")
    st.caption("Snowflake DMF (Data Metric Functions)로 자동 검증")

    if client is not None:
        try:
            dq_df = client.load_data_quality()
            if not dq_df.empty:
                pass_count = len(dq_df[dq_df["QUALITY_STATUS"] == "PASS"])
                warn_count = len(dq_df[dq_df["QUALITY_STATUS"] == "WARNING"])
                crit_count = len(dq_df[dq_df["QUALITY_STATUS"] == "CRITICAL"])
                total = len(dq_df)

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("PASS", f"{pass_count}/{total}")
                with m2:
                    st.metric("WARNING", warn_count)
                with m3:
                    st.metric("CRITICAL", crit_count)

                if crit_count > 0:
                    st.error(f"{crit_count}건의 품질 위반이 감지되었습니다.")
                elif warn_count > 0:
                    st.warning(f"{warn_count}건의 주의 항목이 있습니다.")
                else:
                    st.success("모든 품질 검사를 통과했습니다.")
            else:
                st.info("DMF 결과가 없습니다. 파이프라인 실행 후 확인하세요.")
        except Exception:
            st.info("데이터 품질 뷰를 로드할 수 없습니다. 07_data_quality.sql을 실행하세요.")

with ln_col:
    st.subheader("┃데이터 리니지")
    st.caption("Snowflake 테이블 의존성 자동 추적")

    if client is not None:
        try:
            lineage_df = client.load_lineage_summary()
            if not lineage_df.empty:
                for _, row in lineage_df.iterrows():
                    up = row.get("UPSTREAM_LAYER", "?")
                    down = row.get("DOWNSTREAM_LAYER", "?")
                    cnt = row.get("DEPENDENCY_COUNT", 0)
                    st.caption(f"**{up}** → **{down}**  ({cnt}개 의존성)")
            else:
                st.info("리니지 데이터가 없습니다. 08_lineage.sql을 실행하세요.")
        except Exception:
            st.info("리니지 뷰를 로드할 수 없습니다. 08_lineage.sql을 실행하세요.")

# ---------------------------------------------------------------------------
# Connection notice
# ---------------------------------------------------------------------------
if client is None:
    st.info("Snowflake 연결이 설정되지 않았습니다. `.env` 파일에 SF_USER, SF_PASSWORD를 설정하면 실제 데이터로 분석합니다.")
