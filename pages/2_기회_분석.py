"""Page 2: Opportunity -- Regional demand + What-if simulation.

- State demand horizontal bars (Plotly)
- Growth Top 5 cities (Plotly)
- Cortex FORECAST chart
- Scenario simulation with preset buttons + comparison
- Monte Carlo fan chart
"""

from __future__ import annotations

import logging
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

logger = logging.getLogger(__name__)

st.set_page_config(page_title="기회 분석", layout="wide")

# ---------------------------------------------------------------------------
# Global CSS (CSS-only, no HTML layout)
# ---------------------------------------------------------------------------
try:
    from components.styles import inject_global_css
    inject_global_css()
except Exception:
    pass

from components.nav import safe_page_link as _safe_pl
from components.sidebar import render_sidebar

# ---------------------------------------------------------------------------
# Navigation (native _safe_pl)
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
# Dependency imports (all guarded)
# ---------------------------------------------------------------------------
from components.utils import get_cached_client, safe_data_load, validate_columns, PLOTLY_DARK_LAYOUT

try:
    from data.snowflake_client import SnowflakeClient
    CLIENT_AVAILABLE = True
except Exception:
    CLIENT_AVAILABLE = False

try:
    from analysis.insight_generator import generate_regional_insights
    INSIGHT_AVAILABLE = True
except Exception:
    INSIGHT_AVAILABLE = False

try:
    from analysis.advanced_analytics import FunnelMarkovChain
    MARKOV_AVAILABLE = True
except Exception:
    MARKOV_AVAILABLE = False


# ---------------------------------------------------------------------------
# Plotly dark layout
# ---------------------------------------------------------------------------
_DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Pretendard, sans-serif", color="#e0e0e0"),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
)


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def _load_markov_baseline(category: str | None = None):
    """마르코프 체인 기준 데이터를 캐싱하여 로드."""
    cl = get_cached_client()
    if cl is None or not MARKOV_AVAILABLE:
        return None, None
    try:
        stage_drop_df = cl.load_funnel_stage_drop(category)
        markov = FunnelMarkovChain()
        tm = markov.compute_transition_matrix(stage_drop_df, category=category)
        if tm is None or tm.empty:
            return None, None
        return markov, tm
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------
def _build_state_demand_bar(state_agg: pd.DataFrame) -> go.Figure:
    if not validate_columns(
        state_agg,
        ["DEMAND_SCORE", "INSTALL_STATE"],
        context="지역 수요 바 차트",
    ):
        return go.Figure()
    sorted_df = state_agg.sort_values("DEMAND_SCORE", ascending=True)

    q1 = sorted_df["DEMAND_SCORE"].quantile(0.25)
    q3 = sorted_df["DEMAND_SCORE"].quantile(0.75)
    iqr = q3 - q1
    clip_upper = q3 + 2.0 * iqr

    clipped = sorted_df["DEMAND_SCORE"].clip(upper=clip_upper)

    # Gradient: cyan->indigo for positive, muted for negative
    colors = [
        "rgba(0,229,255,0.8)" if v > 0 else "rgba(255,77,77,0.6)"
        for v in sorted_df["DEMAND_SCORE"]
    ]

    fig = go.Figure(
        go.Bar(
            x=clipped,
            y=sorted_df["INSTALL_STATE"],
            orientation="h",
            marker_color=colors,
            text=sorted_df["DEMAND_SCORE"].round(2).astype(str),
            textposition="outside",
            textfont=dict(color="#e0e0e0", size=11),
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#636363", annotation_text="평균")
    layout_kwargs = {k: v for k, v in _DARK_LAYOUT.items() if k != "yaxis"}
    fig.update_layout(
        **layout_kwargs,
        title="지역별 수요 점수 (Z-score 합산)",
        xaxis_title="수요 점수",
        height=max(400, 30 * len(sorted_df)),
        yaxis=dict(categoryorder="total ascending", gridcolor="rgba(255,255,255,0.05)"),
    )
    return fig


def _build_growth_cities_bar(heatmap_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    if not validate_columns(
        heatmap_df,
        ["GROWTH_3M", "INSTALL_CITY", "INSTALL_STATE", "CONTRACT_COUNT"],
        context="성장 도시 바 차트",
    ):
        return go.Figure()

    # 최소 200건 이상 도시만 (소규모 기저효과 제거)
    valid = heatmap_df[
        heatmap_df["GROWTH_3M"].notna()
        & (heatmap_df["GROWTH_3M"] > 0)
        & (heatmap_df["CONTRACT_COUNT"] >= 200)
    ]
    if valid.empty:
        return go.Figure()

    top = valid.nlargest(top_n, "GROWTH_3M").copy()
    top["label"] = top["INSTALL_STATE"] + " " + top["INSTALL_CITY"]
    top = top.sort_values("GROWTH_3M", ascending=True)

    text_vals = top["GROWTH_3M"].apply(
        lambda v: f"{v:.1%}" if abs(v) < 10 else f"{v:.1f}"
    )

    fig = go.Figure(
        go.Bar(
            x=top["GROWTH_3M"],
            y=top["label"],
            orientation="h",
            marker_color="#4DFF91",
            text=text_vals,
            textposition="outside",
            textfont=dict(color="#e0e0e0", size=11),
        )
    )
    layout = {**_DARK_LAYOUT, "margin": dict(l=120, r=40, t=50, b=40)}
    fig.update_layout(
        **layout,
        title=f"성장률 Top {top_n} 도시 (3개월 MoM)",
        xaxis_title="성장률",
        height=max(280, 50 * top_n),
    )
    return fig


def _build_forecast_chart(
    forecast_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    state: str,
) -> go.Figure | None:
    region_fc = forecast_df.copy()
    if "TARGET_METRIC" in region_fc.columns:
        region_fc = region_fc[region_fc["TARGET_METRIC"] == "CONTRACT_COUNT"]
    if "SERIES_TYPE" in region_fc.columns:
        region_fc = region_fc[region_fc["SERIES_TYPE"] == "STATE"]

    series_col = next(
        (c for c in ["SERIES_KEY", "SERIES"] if c in region_fc.columns), None
    )
    ts_col = next((c for c in ["TS", "YEAR_MONTH"] if c in region_fc.columns), None)

    if not series_col or not ts_col or "FORECAST" not in region_fc.columns:
        return None

    fc_data = region_fc[region_fc[series_col] == state].sort_values(ts_col).copy()
    if fc_data.empty:
        return None

    fc_data["FORECAST"] = fc_data["FORECAST"].clip(lower=0)
    for col in ["LOWER", "LOWER_BOUND"]:
        if col in fc_data.columns:
            fc_data[col] = fc_data[col].clip(lower=0)
    for col in ["UPPER", "UPPER_BOUND"]:
        if col in fc_data.columns:
            fc_data[col] = fc_data[col].clip(lower=0)

    fig = go.Figure()

    # 실적: 최근 12개월만 표시
    if not demand_df.empty and "INSTALL_STATE" in demand_df.columns:
        hist = (
            demand_df[demand_df["INSTALL_STATE"] == state]
            .groupby("YEAR_MONTH")
            .agg(CONTRACT_COUNT=("CONTRACT_COUNT", "sum"))
            .reset_index()
            .sort_values("YEAR_MONTH")
        )
        if not hist.empty:
            # 미완성월 제거 + 최근 12개월
            if len(hist) > 1:
                hist = hist[hist["YEAR_MONTH"] < hist["YEAR_MONTH"].max()]
            hist = hist.tail(12)
            fig.add_trace(
                go.Scatter(
                    x=hist["YEAR_MONTH"],
                    y=hist["CONTRACT_COUNT"],
                    name="실적",
                    line=dict(color="#3B82F6", width=2.5),
                    mode="lines+markers",
                    marker=dict(size=5),
                )
            )

    # 실적 마지막 점 → 예측 첫 점 연결선 (bridge)
    last_actual_x = None
    last_actual_y = None
    if not demand_df.empty and "INSTALL_STATE" in demand_df.columns:
        h = (
            demand_df[demand_df["INSTALL_STATE"] == state]
            .groupby("YEAR_MONTH")
            .agg(CONTRACT_COUNT=("CONTRACT_COUNT", "sum"))
            .reset_index()
            .sort_values("YEAR_MONTH")
        )
        if not h.empty:
            last_actual_x = h["YEAR_MONTH"].iloc[-1]
            last_actual_y = float(h["CONTRACT_COUNT"].iloc[-1])

    fc_x = list(fc_data[ts_col])
    fc_y = list(fc_data["FORECAST"])

    # 연결선: 실적 마지막 → 예측 첫 점
    if last_actual_x is not None and len(fc_x) > 0:
        bridge_x = [last_actual_x] + fc_x
        bridge_y = [last_actual_y] + fc_y
    else:
        bridge_x = fc_x
        bridge_y = fc_y

    fig.add_trace(
        go.Scatter(
            x=bridge_x,
            y=bridge_y,
            name="예측",
            line=dict(color="#00E5FF", width=2.5, dash="dash"),
            mode="lines+markers+text",
            marker=dict(size=7, symbol="diamond"),
            text=[""] + [f"{v:,.0f}" for v in fc_y] if last_actual_x else [f"{v:,.0f}" for v in fc_y],
            textposition="top center",
            textfont=dict(color="#00E5FF", size=11),
        )
    )

    # 신뢰구간: 실적 마지막 점에서 부채꼴로 펼쳐짐
    lo_col_name, hi_col_name = None, None
    for lo_c, hi_c in [("LOWER", "UPPER"), ("LOWER_BOUND", "UPPER_BOUND")]:
        if lo_c in fc_data.columns and hi_c in fc_data.columns:
            lo_col_name, hi_col_name = lo_c, hi_c
            break

    if lo_col_name and hi_col_name:
        ci_upper = list(fc_data[hi_col_name].clip(lower=0))
        ci_lower = list(fc_data[lo_col_name].clip(lower=0))

        # 실적 마지막 점에서 시작 (부채꼴)
        if last_actual_y is not None:
            upper_pts = [last_actual_y] + ci_upper
            lower_pts = [last_actual_y] + ci_lower
            ci_x = [last_actual_x] + fc_x
        else:
            upper_pts = ci_upper
            lower_pts = ci_lower
            ci_x = fc_x

        # +1 시그마 (진한 음영)
        mid_upper = [(u + f) / 2 for u, f in zip(upper_pts, [last_actual_y] + fc_y if last_actual_y is not None else fc_y)]
        mid_lower = [(l + f) / 2 for l, f in zip(lower_pts, [last_actual_y] + fc_y if last_actual_y is not None else fc_y)]

        fig.add_trace(
            go.Scatter(
                x=ci_x + ci_x[::-1],
                y=mid_upper + mid_lower[::-1],
                fill="toself",
                fillcolor="rgba(0,229,255,0.2)",
                line=dict(color="rgba(0,229,255,0)"),
                name="+1σ",
                showlegend=True,
            )
        )

        # +2 시그마 (연한 음영)
        fig.add_trace(
            go.Scatter(
                x=ci_x + ci_x[::-1],
                y=upper_pts + lower_pts[::-1],
                fill="toself",
                fillcolor="rgba(0,229,255,0.08)",
                line=dict(color="rgba(0,229,255,0)"),
                name="+2σ",
                showlegend=True,
            )
        )

    # 예측 구간 배경
    if len(fc_data) > 0:
        fc_start = fc_data[ts_col].min()
        fc_end = fc_data[ts_col].max()
        fig.add_vrect(
            x0=fc_start, x1=fc_end,
            fillcolor="rgba(0,229,255,0.03)",
            line_width=0,
            annotation_text="예측 구간",
            annotation_position="top left",
            annotation_font=dict(color="rgba(0,229,255,0.5)", size=10),
        )

    layout_base = {k: v for k, v in _DARK_LAYOUT.items() if k not in ("yaxis", "margin")}
    fig.update_layout(
        **layout_base,
        title=f"{state} 계약 예측 (Cortex FORECAST)",
        xaxis_title="년월",
        yaxis_title="계약 건수",
        yaxis=dict(rangemode="tozero", gridcolor="rgba(255,255,255,0.05)"),
        height=420,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.08,
                    xanchor="right", x=1, font=dict(size=11)),
        margin=dict(l=50, r=20, t=80, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Markov simulation helpers
# ---------------------------------------------------------------------------
_STAGE_LABELS_KR = {
    "CONSULT_REQUEST": "상담요청",
    "SUBSCRIPTION": "가입신청",
    "REGISTEND": "접수완료",
    "OPEN": "개통",
    "PAYEND": "납입완료",
}

_TRANSITION_PAIRS = [
    ("CONSULT_REQUEST", "SUBSCRIPTION"),
    ("SUBSCRIPTION", "REGISTEND"),
    ("REGISTEND", "OPEN"),
    ("OPEN", "PAYEND"),
]

_PRESET_SCENARIOS: dict[str, dict[str, float]] = {
    "접수→개통 집중": {("REGISTEND", "OPEN"): 10},
    "가입 초기 개선": {("CONSULT_REQUEST", "SUBSCRIPTION"): 8, ("SUBSCRIPTION", "REGISTEND"): 5},
    "전 단계 균등 5%p": {p: 5 for p in _TRANSITION_PAIRS},
    "개통→납입 집중": {("OPEN", "PAYEND"): 15},
}


# =========================================================================
# Global sidebar + filters
# =========================================================================
_sidebar_result = render_sidebar()
category_filter = _sidebar_result.get("category")
start_ym, end_ym = "202301", "202603"

client = get_cached_client()
if client is None:
    st.error("Snowflake 연결 실패. `.env` 파일을 확인하세요.")
    st.stop()

with st.spinner("지역 데이터 로딩 중..."):
    heatmap_df = safe_data_load(
        lambda: client.load_regional_heatmap(),
        "지역 히트맵 데이터 로드 실패",
    )
    demand_df = safe_data_load(
        lambda: client.load_regional_demand(None),
        "지역 수요 데이터 로드 실패",
    )
    forecast_df = safe_data_load(
        lambda: client.load_forecast(),
        "Cortex FORECAST 데이터 로드 실패",
    )

for df in [heatmap_df, demand_df]:
    if not df.empty and "YEAR_MONTH" in df.columns:
        df["YEAR_MONTH"] = pd.to_datetime(df["YEAR_MONTH"])

# 미완성월 제외 — demand_df만 적용 (heatmap은 SQL 뷰에서 이미 최신월 스냅샷)
if not demand_df.empty and "YEAR_MONTH" in demand_df.columns:
    demand_df = demand_df[demand_df["YEAR_MONTH"] < demand_df["YEAR_MONTH"].max()]

# Save insights for cross-page use
if INSIGHT_AVAILABLE and not heatmap_df.empty:
    try:
        _page2_insights = generate_regional_insights(heatmap_df)
        st.session_state["page2_insights"] = {
            "top_growth_city": (
                _page2_insights.get("metrics", {}).get("growth_cities", [{}])[0].get("city", "")
                if _page2_insights.get("metrics", {}).get("growth_cities")
                else ""
            ),
            "top_state": _page2_insights.get("metrics", {}).get("top_state", ""),
        }
    except Exception:
        pass


# =========================================================================
# PAGE LAYOUT -- 100% Streamlit native components
# =========================================================================

# -- Page header --
st.caption("PAGE 2: 기회 분석")
st.title("지역 수요 예측 & 시나리오 시뮬레이션")

st.markdown(
    """
> **분석 방법**: 200개 시군구의 계약·전환율·매출 데이터를 Z-score로 정규화하여 수요 점수를 산출하고,
> **Cortex FORECAST**로 시도별 계약 건수를 3개월 예측합니다.
> **Cortex ANOMALY**로 계약 건수 급변 시도를 자동 탐지하며,
> **마르코프 체인 시뮬레이션**으로 퍼널 전이 확률 변동에 따른 최종 전환율 변화를 수학적으로 계산합니다.
> 500회 **몬테카를로 시뮬레이션**으로 전환율의 불확실성 범위와 리스크를 추정합니다.
>
> **기대효과**: 고수요 지역 선점 + 이상 징후 조기 감지 + 전이 확률 개선 시 정량적 효과 예측
"""
)

# -- KPI summary cards --
if not heatmap_df.empty and "INSTALL_STATE" in heatmap_df.columns:
    _kpi_n_states = heatmap_df["INSTALL_STATE"].nunique()
    _kpi_top_state = (
        heatmap_df.groupby("INSTALL_STATE")["DEMAND_SCORE"]
        .mean()
        .idxmax()
    ) if "DEMAND_SCORE" in heatmap_df.columns else "-"
    _kpi_avg_demand = (
        round(heatmap_df.groupby("INSTALL_STATE")["DEMAND_SCORE"].mean().mean(), 2)
    ) if "DEMAND_SCORE" in heatmap_df.columns else "-"
    _kpi_top_growth_city = "-"
    if "GROWTH_3M" in heatmap_df.columns and "INSTALL_CITY" in heatmap_df.columns:
        _growth_valid = heatmap_df[heatmap_df["GROWTH_3M"].notna() & (heatmap_df["GROWTH_3M"] > 0)]
        if not _growth_valid.empty:
            _kpi_top_growth_city = _growth_valid.loc[_growth_valid["GROWTH_3M"].idxmax(), "INSTALL_CITY"]

    _kpi_cards = [
        ("분석 지역 수", f"{_kpi_n_states}", "#00E5FF"),
        ("최고 수요 지역", str(_kpi_top_state), "#4DFF91"),
        ("평균 수요 점수", str(_kpi_avg_demand), "#a855f7"),
        ("최고 성장 도시", str(_kpi_top_growth_city), "#00E5FF"),
    ]

    kpi_cols = st.columns(4)
    _kpi_help = {
        "분석 지역 수": "데이터에 포함된 시/도 수",
        "최고 수요 지역": "Z-score 합산 기준 가장 수요가 높은 시/도",
        "평균 수요 점수": "전체 시/도 수요 점수의 평균. 0 이상이면 양호",
        "최고 성장 도시": "최근 3개월 계약 증가율이 가장 높은 도시",
    }
    for kpi_col, (kpi_label, kpi_value, kpi_color) in zip(kpi_cols, _kpi_cards):
        with kpi_col:
            st.metric(
                label=kpi_label,
                value=kpi_value,
                help=_kpi_help.get(kpi_label, ""),
            )

    st.caption(
        "**읽는 법**: 수요 점수는 계약 건수·전환율·매출의 Z-score 합산입니다. "
        "0 이상이면 전국 평균 초과, 높을수록 해당 지역의 수요가 강합니다. "
        "성장률은 3개월 전 대비 계약 증가율이며, 높은 지역이 신규 마케팅 투자의 우선 후보입니다."
    )

# -- State filter --
_state_options = (
    ["전체"] + sorted(heatmap_df["INSTALL_STATE"].unique().tolist())
    if not heatmap_df.empty and "INSTALL_STATE" in heatmap_df.columns
    else ["전체"]
)
selected_state = st.selectbox(
    "시도 선택 (아래 예측·이상탐지 차트에 적용)",
    _state_options,
    index=0,
)


# ---------------------------------------------------------------------------
# SECTION 1: Two panels -- State demand (2/3) + Growth cities (1/3)
# ---------------------------------------------------------------------------
if not heatmap_df.empty and "INSTALL_STATE" in heatmap_df.columns:
    state_agg = (
        heatmap_df.groupby("INSTALL_STATE")
        .agg(
            DEMAND_SCORE=("DEMAND_SCORE", "mean"),
            CONTRACT_COUNT=("CONTRACT_COUNT", "sum"),
            PAYEND_CVR=("PAYEND_CVR", "mean"),
            BUNDLE_RATIO=("BUNDLE_RATIO", "mean"),
        )
        .reset_index()
        .sort_values("DEMAND_SCORE", ascending=False)
    )

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("┃지역별 수요 점수")
        st.caption(
            "각 시도의 수요 점수를 수평 바차트로 비교합니다. "
            "0 기준선 우측이 전국 평균 이상이며, 시안(파랑) 바가 길수록 수요가 강합니다. "
            "상위 지역에 마케팅 예산을 집중하면 ROI가 높습니다."
        )
        with st.spinner("수요 차트 생성 중..."):
            fig_demand = _build_state_demand_bar(state_agg)
        st.plotly_chart(fig_demand, use_container_width=True)

    with col_right:
        st.subheader("┃성장률 상위 도시")
        st.caption(
            "최근 3개월 계약 건수가 가장 빠르게 증가하고 있는 도시 TOP 10입니다. "
            "소규모 기저효과를 제거하기 위해 최소 200건 이상인 도시만 포함합니다. "
            "이 도시들은 아직 경쟁이 낮은 '선점 기회' 시장입니다."
        )
        with st.spinner("성장 도시 차트 생성 중..."):
            fig_growth = _build_growth_cities_bar(heatmap_df)
        if fig_growth.data:
            st.plotly_chart(fig_growth, use_container_width=True)
        else:
            st.info("성장 도시 데이터가 없습니다.")
else:
    st.info("지역 히트맵 데이터가 없습니다.")


# ---------------------------------------------------------------------------
# SECTION 2: Cortex FORECAST (full-width)
# ---------------------------------------------------------------------------
st.divider()

st.subheader("┃Cortex FORECAST 예측")
st.caption(
    "Snowflake Cortex FORECAST로 시도별 계약 건수를 3개월 예측합니다. "
    "파란 실선은 과거 실적, 시안 점선은 예측값, 밝은 파란색 음영 영역은 95% 신뢰구간입니다. "
    "음영이 좁을수록 예측의 확실성이 높으며, 넓으면 해당 지역의 변동성이 크다는 의미입니다."
)

if not forecast_df.empty:
    display_state = selected_state if selected_state != "전체" else None
    series_col = next(
        (c for c in ["SERIES_KEY", "SERIES"] if c in forecast_df.columns), None
    )
    if series_col:
        available_states = sorted(forecast_df[series_col].unique().tolist())

        if display_state and display_state in available_states:
            forecast_state = display_state
        elif available_states:
            forecast_state = st.selectbox(
                "예측할 시도 선택",
                options=available_states,
                index=0,
            )
        else:
            forecast_state = None

        if forecast_state:
            fig_fc = _build_forecast_chart(forecast_df, demand_df, forecast_state)
            if fig_fc is not None:
                st.plotly_chart(fig_fc, use_container_width=True)
            else:
                st.info(f"{forecast_state}의 예측 데이터가 없습니다.")
        else:
            st.info("예측 가능한 시도가 없습니다.")
    else:
        st.info("예측 데이터 시리즈 컬럼을 찾을 수 없습니다.")
else:
    st.info("Cortex FORECAST를 아직 실행하지 않았습니다.")


# ---------------------------------------------------------------------------
# SECTION 2.5: Anomaly Detection
# ---------------------------------------------------------------------------
st.divider()

st.subheader("┃이상 탐지")
st.caption(
    "Cortex ANOMALY가 시도별 계약 건수에서 95% 신뢰구간을 벗어난 급변 시점을 자동 탐지합니다. "
    "실측이 기대보다 크면 긍정적 이상(특수 프로모션 효과 등), 작으면 부정적 이상(서비스 장애 등)입니다."
)
st.markdown(
    "**대응 가이드**: 이상 여부가 **True**인 시도는 "
    "① 해당 월의 프로모션/장애 이력 확인 → ② 원인 분석 후 반복 가능성 평가 → "
    "③ 긍정적 이상은 성공 패턴 복제, 부정적 이상은 재발 방지 대책 수립"
)

try:
    anomaly_df = client.load_anomalies()
except Exception:
    anomaly_df = pd.DataFrame()

if not anomaly_df.empty:
    _anom_display = anomaly_df.copy()

    # Filter to CONTRACT_COUNT if TARGET_METRIC column exists
    if "TARGET_METRIC" in _anom_display.columns:
        _anom_display = _anom_display[_anom_display["TARGET_METRIC"] == "CONTRACT_COUNT"]

    # Sort by anomaly score if available
    _sort_col = next(
        (c for c in ["ANOMALY_SCORE", "PERCENTILE"] if c in _anom_display.columns), None
    )
    if _sort_col:
        _anom_display = _anom_display.sort_values(_sort_col, ascending=False)

    # Pick columns to display
    _show_cols = []
    for c in ["SERIES_KEY", "TS", "OBSERVED", "EXPECTED", "IS_ANOMALY"]:
        if c in _anom_display.columns:
            _show_cols.append(c)
    if not _show_cols:
        _show_cols = list(_anom_display.columns)

    _anom_view = _anom_display[_show_cols].head(30).copy()

    _col_rename = {
        "SERIES_KEY": "시도",
        "TS": "시점",
        "OBSERVED": "실측",
        "EXPECTED": "기대",
        "IS_ANOMALY": "이상 여부",
    }
    _anom_view = _anom_view.rename(columns={k: v for k, v in _col_rename.items() if k in _anom_view.columns})

    # Format numbers: remove excessive decimals
    for _num_col in ["실측", "기대"]:
        if _num_col in _anom_view.columns:
            _anom_view[_num_col] = _anom_view[_num_col].apply(
                lambda v: f"{v:,.0f}" if pd.notna(v) else ""
            )

    # Style: highlight anomaly rows in red
    def _highlight_anomaly(row: pd.Series) -> list[str]:
        is_anom = row.get("이상 여부", False)
        if is_anom:
            return ["background-color: rgba(255,77,77,0.15); color: #FF4D4D"] * len(row)
        return [""] * len(row)

    styled = _anom_view.style.apply(_highlight_anomaly, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)
else:
    st.info("이상 탐지 결과가 없습니다. Cortex ANOMALY 파이프라인을 실행하세요.")


# ---------------------------------------------------------------------------
# SECTION 3: Markov Transition Simulation
# ---------------------------------------------------------------------------
st.divider()

st.subheader("┃퍼널 전이 확률 시뮬레이션")
st.caption(
    "퍼널 각 단계의 전이 확률(예: 접수→개통 84%)을 슬라이더로 조절하면, "
    "마르코프 체인이 변경된 전이 행렬로 새로운 최종 전환율을 수학적으로 재계산합니다. "
    "XGBoost 같은 ML 모델이 아닌 확률론적 모델이라 인과관계가 명확합니다. "
    "예: '접수→개통을 10%p 개선하면 최종 전환율이 +X%p 오르고, 월 +Y건 추가 전환'"
)

_markov_obj, _baseline_tm = _load_markov_baseline()

if _markov_obj is not None and _baseline_tm is not None:
    _baseline_ss = _markov_obj.compute_steady_state(_baseline_tm)
    _baseline_payend = _baseline_ss.get("PAYEND", 0.0)

    # --- Preset buttons ---
    st.caption("프리셋 시나리오")
    preset_cols = st.columns(len(_PRESET_SCENARIOS))
    for idx, (sc_name, sc_adj) in enumerate(_PRESET_SCENARIOS.items()):
        with preset_cols[idx]:
            if st.button(sc_name, key=f"mk_preset_{idx}", use_container_width=True):
                st.session_state["mk_scenario"] = sc_name
                st.session_state["mk_adjustments"] = dict(sc_adj)

    col_sliders, col_result = st.columns(2)

    with col_sliders:
        st.caption("전이 확률 조정 (%p)")
        current_mk = st.session_state.get("mk_adjustments", {})
        mk_slider_values: dict[tuple[str, str], int] = {}

        for from_s, to_s in _TRANSITION_PAIRS:
            current_prob = float(_baseline_tm.loc[from_s, to_s])
            label_from = _STAGE_LABELS_KR.get(from_s, from_s)
            label_to = _STAGE_LABELS_KR.get(to_s, to_s)
            max_improve = int(min(50, (1.0 - current_prob) * 100))

            default_val = int(current_mk.get((from_s, to_s), 0))
            default_val = max(-20, min(default_val, max_improve))

            mk_slider_values[(from_s, to_s)] = st.slider(
                f"{label_from} → {label_to} (현재 {current_prob*100:.0f}%)",
                min_value=-20, max_value=max_improve,
                value=default_val,
                key=f"mk_slider_{from_s}_{to_s}",
                format="%+d%%p",
            )

        if st.button("시뮬레이션 실행", key="mk_run", type="primary", use_container_width=True):
            st.session_state["mk_scenario"] = "커스텀"
            st.session_state["mk_adjustments"] = {
                k: v for k, v in mk_slider_values.items() if v != 0
            }

    with col_result:
        st.subheader("┃시뮬레이션 결과")

        mk_adj = st.session_state.get("mk_adjustments")
        if mk_adj:
            scenario_label = st.session_state.get("mk_scenario", "커스텀")

            # Apply adjustments to transition matrix
            modified_tm = _baseline_tm.copy()
            for (fs, ts), delta_pct in mk_adj.items():
                if delta_pct == 0:
                    continue
                old_val = float(modified_tm.loc[fs, ts])
                new_val = min(max(old_val + delta_pct / 100.0, 0.0), 1.0)
                actual_delta = new_val - old_val
                modified_tm.loc[fs, ts] = new_val
                modified_tm.loc[fs, "DROP"] = max(
                    float(modified_tm.loc[fs, "DROP"]) - actual_delta, 0.0
                )

            modified_ss = _markov_obj.compute_steady_state(modified_tm)
            modified_payend = modified_ss.get("PAYEND", 0.0)
            delta_payend = modified_payend - _baseline_payend
            monthly_entries = 10000
            additional = int(round(delta_payend * monthly_entries))

            # KPI cards
            st.markdown(f'''
<div style="display:flex;gap:10px;margin-bottom:16px;">
    <div style="flex:1;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);
                border-radius:12px;padding:14px;text-align:center;">
        <div style="font-size:0.8rem;color:rgba(255,255,255,0.5);text-transform:uppercase;">현재 전환율</div>
        <div style="font-size:1.4rem;font-weight:700;color:#e0e0e0;margin-top:4px;">{_baseline_payend*100:.1f}%</div>
    </div>
    <div style="flex:1;background:rgba(0,229,255,0.08);border:1px solid rgba(0,229,255,0.2);
                border-radius:12px;padding:14px;text-align:center;">
        <div style="font-size:0.8rem;color:rgba(255,255,255,0.5);text-transform:uppercase;">시뮬레이션</div>
        <div style="font-size:1.4rem;font-weight:700;color:#00E5FF;margin-top:4px;">{modified_payend*100:.1f}%</div>
    </div>
    <div style="flex:1;background:rgba({'77,255,145' if delta_payend >= 0 else '255,77,77'},0.08);
                border:1px solid rgba({'77,255,145' if delta_payend >= 0 else '255,77,77'},0.2);
                border-radius:12px;padding:14px;text-align:center;">
        <div style="font-size:0.8rem;color:rgba(255,255,255,0.5);text-transform:uppercase;">변화량</div>
        <div style="font-size:1.4rem;font-weight:700;color:{'#4DFF91' if delta_payend >= 0 else '#FF4D4D'};margin-top:4px;">{delta_payend*100:+.2f}%p</div>
    </div>
    <div style="flex:1;background:rgba({'77,255,145' if additional >= 0 else '255,77,77'},0.08);
                border:1px solid rgba({'77,255,145' if additional >= 0 else '255,77,77'},0.2);
                border-radius:12px;padding:14px;text-align:center;">
        <div style="font-size:0.8rem;color:rgba(255,255,255,0.5);text-transform:uppercase;">추가 전환/월</div>
        <div style="font-size:1.4rem;font-weight:700;color:{'#4DFF91' if additional >= 0 else '#FF4D4D'};margin-top:4px;">{additional:+,}건</div>
    </div>
</div>
''', unsafe_allow_html=True)

            st.caption(
                "현재 전환율은 기존 전이 확률 기반 Steady State이고, "
                "시뮬레이션은 슬라이더로 조절된 전이 확률 기반입니다. "
                "변화량이 양수면 해당 개선이 효과적이라는 의미이며, "
                "추가 전환/월은 월 10,000명 진입 기준 추가 납입완료 고객 수입니다."
            )

            # Stage-by-stage comparison bar chart
            import numpy as np
            from config.constants import FUNNEL_STAGES
            stage_labels = [_STAGE_LABELS_KR.get(s, s) for s in FUNNEL_STAGES]

            baseline_probs = [_baseline_ss.get(s, 0.0) * 100 for s in FUNNEL_STAGES]
            modified_probs = [modified_ss.get(s, 0.0) * 100 for s in FUNNEL_STAGES]

            fig_compare = go.Figure()
            fig_compare.add_trace(go.Bar(
                x=stage_labels, y=baseline_probs,
                name="현재", marker_color="rgba(255,255,255,0.12)",
                text=[f"{v:.1f}%" for v in baseline_probs],
                textposition="outside", textfont=dict(color="#999", size=10),
            ))
            fig_compare.add_trace(go.Bar(
                x=stage_labels, y=modified_probs,
                name="시뮬레이션", marker_color="#00E5FF",
                text=[f"{v:.1f}%" for v in modified_probs],
                textposition="outside", textfont=dict(color="#00E5FF", size=10),
            ))
            _compare_layout = {k: v for k, v in _DARK_LAYOUT.items() if k not in ("margin",)}
            fig_compare.update_layout(
                **_compare_layout, height=320, barmode="group",
                title=f"단계별 도달 확률 비교 ({scenario_label})",
                yaxis_title="도달 확률 (%)",
                margin=dict(l=40, r=20, t=80, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.08,
                            xanchor="right", x=1, font=dict(size=11)),
            )
            st.plotly_chart(fig_compare, use_container_width=True)
        else:
            st.info(
                "위의 프리셋 버튼을 클릭하거나 슬라이더를 조절하면 즉시 시뮬레이션 결과가 여기에 표시됩니다. "
                "시뮬레이션은 마르코프 체인을 기반으로 전이 확률 변경에 따른 최종 전환율 변화를 수학적으로 계산합니다."
            )

    # --- Monte Carlo (Markov-based) ---
    st.divider()

    st.subheader("┃몬테카를로 시뮬레이션")
    st.caption(
        "각 전이 확률에 ±3%p의 랜덤 변동을 500회 부여하여 최종 전환율의 불확실성을 추정합니다. "
        "히스토그램은 500회 시뮬레이션의 전환율 분포이며, 보라색 점선이 평균, 음영이 95% 신뢰구간입니다. "
        "변동성(σ)이 낮으면 퍼널이 안정적이고, 높으면 외부 요인에 취약합니다."
    )

    st.markdown(
        "**활용법**: 변동성이 **높으면** → 채널 다변화, 리스크 헤지 필요 | "
        "변동성이 **낮으면** → 집중 투자 가능, 예측 신뢰도 높음"
    )

    _MC_N = 500
    st.caption(f"약 3~5초 소요됩니다 (500회 시뮬레이션)")
    if st.button(f"Monte Carlo {_MC_N}회 실행", key="mc_run", type="primary"):
        import numpy as np
        rng = np.random.default_rng(42)
        mc_rates: list[float] = []

        with st.spinner(f"Monte Carlo {_MC_N}회 시뮬레이션 중..."):
            for _ in range(_MC_N):
                perturbed = _baseline_tm.copy()
                for fs, ts in _TRANSITION_PAIRS:
                    noise = float(rng.normal(0, 0.03))
                    old = float(perturbed.loc[fs, ts])
                    new_val = min(max(old + noise, 0.01), 0.99)
                    delta = new_val - old
                    perturbed.loc[fs, ts] = new_val
                    perturbed.loc[fs, "DROP"] = max(float(perturbed.loc[fs, "DROP"]) - delta, 0.0)

                ss = _markov_obj.compute_steady_state(perturbed)
                mc_rates.append(ss.get("PAYEND", 0.0))

        st.session_state["mc_markov"] = mc_rates

    mc_rates = st.session_state.get("mc_markov")
    if mc_rates:
        import numpy as np
        arr = np.array(mc_rates) * 100
        mean_v, std_v = float(np.mean(arr)), float(np.std(arr))
        ci_lo, ci_hi = float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
        worst, best = float(np.min(arr)), float(np.max(arr))

        # Summary cards
        _mc_cards = [
            ("평균", f"{mean_v:.1f}%", "#a855f7"),
            ("95% CI 하한", f"{ci_lo:.1f}%", "#FF4D4D"),
            ("95% CI 상한", f"{ci_hi:.1f}%", "#4DFF91"),
            ("최악", f"{worst:.1f}%", "#FF4D4D"),
            ("최선", f"{best:.1f}%", "#4DFF91"),
        ]
        mc_cols = st.columns(5)
        for mc_col, (mc_l, mc_v, mc_c) in zip(mc_cols, _mc_cards):
            with mc_col:
                st.markdown(f'''
<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);
            border-radius:12px;padding:12px;text-align:center;">
    <div style="font-size:0.8rem;color:rgba(255,255,255,0.5);text-transform:uppercase;letter-spacing:1px;">{mc_l}</div>
    <div style="font-size:1.5rem;font-weight:700;color:{mc_c};margin-top:4px;">{mc_v}</div>
</div>
''', unsafe_allow_html=True)

        mc_left, mc_right = st.columns([3, 2])
        with mc_left:
            fig_mc = go.Figure(go.Histogram(
                x=arr, nbinsx=30,
                marker_color="rgba(167,139,250,0.5)", opacity=0.8,
            ))
            fig_mc.add_vline(x=mean_v, line_dash="dash", line_color="#A78BFA",
                            annotation_text=f"평균 {mean_v:.1f}%")
            fig_mc.add_vrect(x0=ci_lo, x1=ci_hi,
                            fillcolor="rgba(167,139,250,0.08)", line_width=0,
                            annotation_text="95% CI", annotation_position="top left",
                            annotation_font=dict(color="rgba(167,139,250,0.6)", size=10))
            fig_mc.update_layout(
                **_DARK_LAYOUT, height=300, showlegend=False,
                title=f"최종 전환율 분포 ({_MC_N}회)",
                xaxis_title="전환율 (%)", yaxis_title="빈도",
            )
            st.plotly_chart(fig_mc, use_container_width=True)

        with mc_right:
            fig_box = go.Figure(go.Box(
                y=arr, name="전환율",
                marker_color="#a855f7", boxmean="sd",
                fillcolor="rgba(167,139,250,0.15)",
                line=dict(color="#a855f7"),
            ))
            fig_box.update_layout(
                **_DARK_LAYOUT, height=300, showlegend=False,
                title="전환율 분포 (Box Plot)", yaxis_title="전환율 (%)",
            )
            st.plotly_chart(fig_box, use_container_width=True)

        risk_level = "낮음" if std_v < 1.5 else "보통" if std_v < 3.0 else "높음"
        risk_color = "#4DFF91" if risk_level == "낮음" else "#FFB84D" if risk_level == "보통" else "#FF4D4D"
        st.markdown(f'''
<div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);
            border-radius:8px;padding:12px;display:flex;justify-content:space-between;align-items:center;">
    <span style="font-size:0.85rem;color:rgba(255,255,255,0.6);">변동성 (σ): {std_v:.2f}%p | 현재 Steady State 대비 ±{ci_hi - ci_lo:.1f}%p 범위</span>
    <span style="font-size:0.85rem;font-weight:600;color:{risk_color};">리스크 수준: {risk_level}</span>
</div>
''', unsafe_allow_html=True)
else:
    st.info("마르코프 체인 데이터를 로드할 수 없습니다. SQL 파이프라인을 먼저 실행하세요.")


# ---------------------------------------------------------------------------
# Cross-page link
# ---------------------------------------------------------------------------
st.divider()

_safe_pl("pages/3_AI_전략.py", label="AI 전략 분석 ->")
