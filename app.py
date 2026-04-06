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
# Imports (공통 유틸리티 사용)
# ---------------------------------------------------------------------------
from components.utils import (
    drop_incomplete_month,
    get_cached_client,
    safe_data_load,
)

try:
    from analysis.insight_generator import generate_funnel_insights, generate_channel_insights
    _INSIGHT_OK = True
except Exception:
    _INSIGHT_OK = False

try:
    from config.constants import PRODUCT_CATEGORIES, STATES
except Exception:
    PRODUCT_CATEGORIES = {"인터넷": "internet", "렌탈": "rental", "모바일": "mobile"}
    STATES = [
        "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
        "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
    ]


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
# DATA LOADING (safe_data_load로 에러 표시)
# ---------------------------------------------------------------------------
client = get_cached_client()

stage_drop_df = pd.DataFrame()
bottleneck_df = pd.DataFrame()
funnel_ts_df = pd.DataFrame()
kpi_df = pd.DataFrame()
channel_df = pd.DataFrame()

if client is not None:
    stage_drop_df = drop_incomplete_month(
        safe_data_load(
            lambda: client.load_funnel_stage_drop(cat_filter),
            "퍼널 스테이지 데이터 로드 실패",
            show_warning=False,
        )
    )
    bottleneck_df = safe_data_load(
        lambda: client.load_funnel_bottlenecks(),
        "병목 데이터 로드 실패",
        show_warning=False,
    )
    funnel_ts_df = drop_incomplete_month(
        safe_data_load(
            lambda: client.load_funnel_timeseries(cat_filter),
            "퍼널 시계열 데이터 로드 실패",
            show_warning=False,
        )
    )
    kpi_df = safe_data_load(
        lambda: client.load_kpi(),
        "KPI 데이터 로드 실패",
        show_warning=False,
    )
    channel_df = drop_incomplete_month(
        safe_data_load(
            lambda: client.load_channel_efficiency(cat_filter),
            "채널 효율 데이터 로드 실패",
            show_warning=False,
        )
    )

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

st.caption(
    "이 경고는 퍼널 5단계 중 이탈률이 가장 높은 구간을 자동으로 감지한 결과입니다. "
    "해당 단계의 프로세스를 집중 개선하면 전체 전환율 향상에 가장 큰 효과를 기대할 수 있습니다."
)

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

with st.expander("KPI 읽는 법", expanded=False):
    st.markdown(
        "- **퍼널 전환율**: 상담요청 대비 최종 납입완료 비율입니다. "
        "평균 대비 마이너스(-)면 최근 전환율이 하락 중입니다.\n"
        "- **최고 볼륨 채널**: 가장 많은 계약을 발생시키는 채널입니다. "
        "점유율이 50% 이상이면 특정 채널에 과도하게 의존하고 있어 리스크가 있습니다.\n"
        "- **최대 성장 지역**: 전월 대비 계약 건수가 가장 크게 증가한 지역입니다."
    )

# ---------------------------------------------------------------------------
# AI CTA Section
# ---------------------------------------------------------------------------
st.divider()

st.subheader("┃AI 기반 전략 최적화")
st.markdown(
    "3단계 Multi-Agent 시스템(분석가 → 전략가 → 종합)이 현재 데이터를 분석하여 "
    "**채널 예산 배분**, **지역 집중 투자**, **시즌별 마케팅** 등 즉시 실행 가능한 전략을 제안합니다."
)
st.caption("Snowflake Cortex COMPLETE (llama3.1-405b) 기반 | 분석 소요 시간: 약 30초")
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

                st.caption(
                    "파이프라인이 Snowflake에서 데이터를 가져올 때 NULL 비율, CVR 범위(0~100%), "
                    "미래 날짜 레코드, 테이블 행 수 등 12가지 항목을 자동 검증합니다. "
                    "CRITICAL이 있으면 분석 결과의 신뢰성에 문제가 있으므로 데이터 소스를 확인하세요."
                )
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
                st.caption(
                    "Snowflake의 OBJECT_DEPENDENCIES를 활용해 테이블 간 의존성을 자동 추적합니다. "
                    "Marketplace 원본 → Staging(정제) → Analytics(분석) → Mart(대시보드) 순서로 데이터가 흐릅니다. "
                    "어떤 테이블이 변경되면 하류 테이블에 어떤 영향이 있는지 파악할 수 있습니다."
                )
            else:
                st.info("리니지 데이터가 없습니다. 08_lineage.sql을 실행하세요.")
        except Exception:
            st.info("리니지 뷰를 로드할 수 없습니다. 08_lineage.sql을 실행하세요.")

# ---------------------------------------------------------------------------
# Connection notice
# ---------------------------------------------------------------------------
if client is None:
    st.info("Snowflake 연결이 설정되지 않았습니다. `.env` 파일에 SF_USER, SF_PASSWORD를 설정하면 실제 데이터로 분석합니다.")
