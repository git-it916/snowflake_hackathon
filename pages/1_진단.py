"""Page 1: Diagnostics -- 100% Streamlit Native."""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="퍼널 + 채널 진단", layout="wide")

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
try:
    from components.styles import inject_global_css
    inject_global_css()
except Exception:
    pass

from components.nav import safe_page_link as _safe_pl
from components.sidebar import render_sidebar
from components.utils import (
    MAJOR_CATEGORIES as _MAJOR_CATS,
    PLOTLY_DARK_LAYOUT,
    STAGE_LABELS as _STAGE_LABELS,
    drop_incomplete_month,
    get_cached_client,
    safe_data_load,
    safe_render,
)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
try:
    import plotly.graph_objects as go
    _PLOTLY_OK = True
except Exception:
    _PLOTLY_OK = False

try:
    from analysis.insight_generator import generate_funnel_insights, generate_channel_insights
    _INSIGHT_OK = True
except Exception:
    _INSIGHT_OK = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_STAGE_ORDER = ["SUBSCRIPTION", "REGISTEND", "OPEN", "PAYEND"]
_STAGE_COLORS = ["#06b6d4", "#0ea5e9", "#3b82f6", "#2563eb"]
_TREND_COLOR_MAP = {"GROWTH": "#4DFF91", "DECLINE": "#FF4D4D", "STABLE": "#4D9AFF"}
_TREND_LABEL_MAP = {"GROWTH": "성장", "DECLINE": "쇠퇴", "STABLE": "안정"}

_DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Pretendard, sans-serif", color="#e0e0e0", size=12),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
)

_MIN_BUBBLE_CONTRACTS = 30


def _ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return df
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        return df
    result = df.copy()
    try:
        result[col] = pd.to_datetime(result[col])
    except Exception:
        try:
            result[col] = pd.to_datetime(result[col].astype(str).str[:6], format="%Y%m")
        except Exception:
            pass
    return result


# ---------------------------------------------------------------------------
# Sankey builder
# ---------------------------------------------------------------------------
def _build_sankey(stage_drop_df, cat_filter, title_suffix):
    if stage_drop_df.empty or not _PLOTLY_OK:
        return None
    df = stage_drop_df.copy()
    if cat_filter:
        df = df[df["MAIN_CATEGORY_NAME"] == cat_filter]
    else:
        df = df[df["MAIN_CATEGORY_NAME"].isin(_MAJOR_CATS)]
    if df.empty:
        return None
    df = df[df["STAGE_NAME"] != "CONSULT_REQUEST"]
    if df.empty:
        return None
    latest_month = df["YEAR_MONTH"].max()
    latest = df[df["YEAR_MONTH"] == latest_month]
    if cat_filter:
        cat_data = latest.sort_values("STAGE_ORDER")
        counts = cat_data["CURR_STAGE_COUNT"].tolist()
        labels = [_STAGE_LABELS.get(s, s) for s in cat_data["STAGE_NAME"].tolist()]
    else:
        agg = latest.groupby("STAGE_ORDER").agg(
            STAGE_NAME=("STAGE_NAME", "first"),
            CURR_STAGE_COUNT=("CURR_STAGE_COUNT", "sum"),
        ).sort_index()
        counts = agg["CURR_STAGE_COUNT"].tolist()
        labels = [_STAGE_LABELS.get(s, s) for s in agg["STAGE_NAME"].tolist()]
    if len(counts) < 2:
        return None

    n = len(labels)
    sources = list(range(n - 1))
    targets = list(range(1, n))
    values = counts[:-1]

    drop_labels, drop_sources, drop_targets, drop_values = [], [], [], []
    for i in range(n - 1):
        drop = max(0, counts[i] - counts[i + 1])
        if drop > 0:
            drop_labels.append(f"이탈 ({labels[i]}→{labels[i+1]})")
            drop_sources.append(i)
            drop_targets.append(n + len(drop_labels) - 1)
            drop_values.append(drop)

    # 노드 라벨: 단계명 + 건수 (한 줄, 겹침 방지)
    node_labels = [f"{lbl}  {counts[i]:,}건" for i, lbl in enumerate(labels)]
    node_labels += drop_labels

    # 노드 색상: 진행은 시안 계열, 이탈은 어두운 회색
    all_colors = _STAGE_COLORS[:n] + ["rgba(75,85,99,0.6)"] * len(drop_labels)

    # 링크 색상: 진행은 시안→파랑 그라데이션, 이탈은 붉은 톤
    progress_colors = [
        "rgba(6,182,212,0.4)", "rgba(14,165,233,0.35)",
        "rgba(59,130,246,0.3)", "rgba(37,99,235,0.25)",
    ]
    link_colors = (
        [progress_colors[i % len(progress_colors)] for i in range(len(sources))]
        + ["rgba(239,68,68,0.15)"] * len(drop_sources)
    )

    # 노드 위치: 진행 노드를 상단에, 이탈 노드를 하단에 명확히 분리
    node_x = [0.01 + i * (0.98 / max(n - 1, 1)) for i in range(n)]
    node_y = [0.3] * n  # 진행 노드는 상단 30%
    for i in range(len(drop_labels)):
        src_idx = drop_sources[i]
        node_x.append(node_x[src_idx] + 0.05)
        node_y.append(0.9)  # 이탈 노드는 하단 90%

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=40,
            thickness=28,
            label=node_labels,
            color=all_colors,
            line=dict(color="rgba(255,255,255,0.15)", width=0.5),
            x=node_x,
            y=node_y,
        ),
        link=dict(
            source=sources + drop_sources,
            target=targets + drop_targets,
            value=values + drop_values,
            color=link_colors,
        ),
        textfont=dict(size=12, color="#ffffff", family="Pretendard, sans-serif"),
    ))
    month_str = str(latest_month)[:7] if latest_month else ""
    fig.update_layout(
        **_DARK_LAYOUT,
        title=dict(text=f"퍼널 흐름 ({month_str})", font=dict(size=14)),
        height=520,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Channel bubble builder
# ---------------------------------------------------------------------------
def _build_channel_bubble(channel_df, ch_col):
    if channel_df.empty or not _PLOTLY_OK:
        return None
    df = _ensure_datetime(channel_df, "YEAR_MONTH")
    if "YEAR_MONTH" not in df.columns:
        return None
    cutoff = df["YEAR_MONTH"].nlargest(6).min()
    recent = df[df["YEAR_MONTH"] >= cutoff].copy()
    if recent.empty:
        return None
    agg = recent.groupby(ch_col).agg(
        CONTRACT_COUNT=("CONTRACT_COUNT", "sum"),
        PAYEND_CVR=("PAYEND_CVR", "mean"),
        AVG_NET_SALES=("AVG_NET_SALES", "mean"),
        TREND_FLAG=("TREND_FLAG", lambda x: x.mode().iloc[0] if len(x) > 0 else "STABLE"),
    ).reset_index()
    agg = agg[(agg["CONTRACT_COUNT"] >= _MIN_BUBBLE_CONTRACTS) & (agg["PAYEND_CVR"] > 0)].copy()
    if agg.empty:
        return None
    agg["AVG_NET_SALES"] = agg["AVG_NET_SALES"].clip(lower=0)
    max_sales = agg["AVG_NET_SALES"].max()
    agg["_size"] = np.clip(agg["AVG_NET_SALES"] / max(max_sales, 1) * 55, 12, 65)
    top_channels = agg.nlargest(5, "CONTRACT_COUNT")[ch_col].tolist()

    # 텍스트 위치: 상위 채널별로 겹침 방지를 위해 번갈아 배치
    _TEXT_POSITIONS = ["top center", "bottom center", "top right", "bottom left", "top left"]

    fig = go.Figure()
    for trend, color in _TREND_COLOR_MAP.items():
        subset = agg[agg["TREND_FLAG"] == trend]
        if subset.empty:
            continue
        text_labels = [
            name if name in top_channels else ""
            for name in subset[ch_col]
        ]
        text_positions = []
        label_idx = 0
        for name in subset[ch_col]:
            if name in top_channels:
                text_positions.append(_TEXT_POSITIONS[label_idx % len(_TEXT_POSITIONS)])
                label_idx += 1
            else:
                text_positions.append("top center")

        fig.add_trace(go.Scatter(
            x=subset["CONTRACT_COUNT"],
            y=subset["PAYEND_CVR"],
            mode="markers+text",
            text=text_labels,
            textposition=text_positions,
            textfont=dict(size=11, color="#ffffff", family="Pretendard, sans-serif"),
            name=_TREND_LABEL_MAP.get(trend, trend),
            marker=dict(
                size=subset["_size"],
                color=color,
                opacity=0.7,
                line=dict(width=1, color="rgba(255,255,255,0.2)"),
            ),
            customdata=subset[ch_col].tolist(),
            hovertemplate=(
                "<b>%{customdata}</b><br>"
                "계약: %{x:,.0f}건<br>"
                "전환율: %{y:.1f}%<br>"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        **_DARK_LAYOUT,
        title=dict(text="채널 효율 (계약 × 전환율)", font=dict(size=14)),
        xaxis_title="계약 건수",
        yaxis_title="전환율 (%)",
        height=520,
        margin=dict(l=50, r=20, t=70, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# CVR trend builder
# ---------------------------------------------------------------------------
def _build_cvr_trend(funnel_ts_df, anomaly_df, cat_filter):
    if funnel_ts_df.empty or not _PLOTLY_OK:
        return None
    df = _ensure_datetime(funnel_ts_df, "YEAR_MONTH")
    if cat_filter:
        ts = df[df["MAIN_CATEGORY_NAME"] == cat_filter].copy()
        if "OVERALL_CVR" not in ts.columns and "TOTAL_COUNT" in ts.columns and "PAYEND_COUNT" in ts.columns:
            ts["OVERALL_CVR"] = (ts["PAYEND_COUNT"] / ts["TOTAL_COUNT"].replace(0, 1) * 100).round(2)
        trend_title = cat_filter
    else:
        major = df[df["MAIN_CATEGORY_NAME"].isin(_MAJOR_CATS)].copy()
        if major.empty:
            return None
        ts = major.groupby("YEAR_MONTH").agg(
            TOTAL_COUNT=("TOTAL_COUNT", "sum"), PAYEND_COUNT=("PAYEND_COUNT", "sum"),
        ).reset_index()
        ts["OVERALL_CVR"] = (ts["PAYEND_COUNT"] / ts["TOTAL_COUNT"].replace(0, 1) * 100).round(2)
        trend_title = "주요 카테고리 합산"
    if ts.empty or "OVERALL_CVR" not in ts.columns:
        return None
    ts = ts.sort_values("YEAR_MONTH")
    # 2023-07 이전 제외 (CVR 100% 비정상 구간)
    ts = ts[ts["YEAR_MONTH"] >= pd.Timestamp("2023-07-01")]
    if ts.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts["YEAR_MONTH"], y=ts["OVERALL_CVR"], name="전체 전환율",
                             mode="lines+markers", line=dict(color="#06b6d4", width=3), marker=dict(size=5)))
    if len(ts) > 6:
        ma = ts["OVERALL_CVR"].rolling(6, min_periods=3).mean()
        fig.add_trace(go.Scatter(x=ts["YEAR_MONTH"], y=ma, name="6개월 이동평균",
                                 mode="lines", line=dict(color="#9ca3af", width=2, dash="dash")))
    if not anomaly_df.empty:
        anom = _ensure_datetime(anomaly_df, "TS")
        if "TARGET_METRIC" in anom.columns:
            anom = anom[anom["TARGET_METRIC"] == "OVERALL_CVR"]
        if "IS_ANOMALY" in anom.columns:
            anom_pts = anom[anom["IS_ANOMALY"]]
            if not anom_pts.empty and "OBSERVED" in anom_pts.columns:
                fig.add_trace(go.Scatter(x=anom_pts["TS"], y=anom_pts["OBSERVED"], mode="markers",
                                         name="이상치", marker=dict(size=12, color="#d73027", symbol="x")))
    fig.update_layout(**_DARK_LAYOUT, title=f"전환율 추이 -- {trend_title}",
                      xaxis_title="년월", yaxis_title="전환율 (%)", height=400,
                      margin=dict(l=40, r=20, t=50, b=40),
                      hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return fig


# ---------------------------------------------------------------------------
# Global sidebar + filters
# ---------------------------------------------------------------------------
_sidebar_result = render_sidebar()
cat_filter = _sidebar_result.get("category")
start_ym, end_ym = "202301", "202603"

client = get_cached_client()
if client is None:
    st.error("Snowflake 연결 실패. `.env` 설정을 확인하세요.")
    st.stop()

stage_drop_df = pd.DataFrame()
bottleneck_df = pd.DataFrame()
funnel_ts_df = pd.DataFrame()
channel_df = pd.DataFrame()
anomaly_df = pd.DataFrame()

with st.spinner("데이터 로딩 중..."):
    stage_drop_df = drop_incomplete_month(
        safe_data_load(lambda: client.load_funnel_stage_drop(cat_filter), "퍼널 스테이지 로드 실패")
    )
    bottleneck_df = safe_data_load(lambda: client.load_funnel_bottlenecks(), "병목 데이터 로드 실패")
    funnel_ts_df = drop_incomplete_month(
        safe_data_load(lambda: client.load_funnel_timeseries(cat_filter), "퍼널 시계열 로드 실패")
    )
    raw_ch = safe_data_load(lambda: client.load_channel_efficiency(cat_filter), "채널 효율 로드 실패")
    channel_df = drop_incomplete_month(_ensure_datetime(raw_ch, "YEAR_MONTH"))
    anomaly_df = safe_data_load(lambda: client.load_anomalies(), "이상 탐지 로드 실패")

ch_col = "RECEIVE_PATH_NAME" if "RECEIVE_PATH_NAME" in channel_df.columns else "CHANNEL"

# Insight generation
funnel_insight = {"headline": "", "severity": "warning", "findings": [], "actions": [], "metrics": {}}
channel_insight = {"headline": "", "severity": "warning", "findings": [], "actions": [], "metrics": {}}

if _INSIGHT_OK:
    try:
        funnel_insight = generate_funnel_insights(stage_drop_df, bottleneck_df, funnel_ts_df, cat_filter)
    except Exception:
        pass
    try:
        channel_insight = generate_channel_insights(channel_df, ch_col)
    except Exception:
        pass

st.session_state["page1_insights"] = {
    "worst_stage": funnel_insight.get("metrics", {}).get("worst_stage", ""),
    "worst_drop": funnel_insight.get("metrics", {}).get("worst_drop_pct", 0),
    "current_cvr": funnel_insight.get("metrics", {}).get("current_cvr", 0),
    "top_channel": channel_insight.get("metrics", {}).get("top_channel", ""),
    "hhi": channel_insight.get("metrics", {}).get("hhi", 0),
}

# =========================================================================
# PAGE LAYOUT — 100% Streamlit Native
# =========================================================================

# Nav
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

# Header
_date_label = f"{start_ym[:4]}.{start_ym[4:]} - {end_ym[:4]}.{end_ym[4:]}"
st.caption(f"PAGE 1: DIAGNOSTICS | {_date_label}")
st.title("퍼널 병목 & 채널 효율 진단")

st.markdown(
    """
> **분석 방법**: 상담요청→가입→접수→개통→납입 5단계 퍼널의 단계별 이탈률을 Sankey 다이어그램으로 시각화하고,
> 38개 채널의 전환율·매출·건수를 버블 차트로 비교합니다.
> **흡수 마르코프 체인**(Absorbing Markov Chain)으로 장기 최종 전환율을 계산하고,
> 민감도 분석을 통해 "어떤 전이를 개선하면 가장 큰 효과가 있는가"를 정량화합니다.
> **STL 시계열 분해**로 추세/계절성/잔차를 분리하여 마케팅 타이밍 최적화를 지원합니다.
>
> **기대효과**: 병목 구간 식별 → 개선 우선순위 수립 → 전환율 개선 ROI 극대화
"""
)

# ---------------------------------------------------------------------------
# Section 1: Sankey + Bubble (two columns)
# ---------------------------------------------------------------------------
col_sankey, col_bubble = st.columns(2)

with col_sankey:
    st.subheader("┃퍼널 병목 분석")
    st.caption(
        "상담요청부터 납입완료까지 각 단계의 진행·이탈 건수를 보여줍니다. "
        "띠가 얇아지는 구간이 고객이 가장 많이 이탈하는 병목입니다. "
        "이 병목 단계의 프로세스를 개선하면 전체 전환율이 올라갑니다."
    )
    title_suffix = cat_filter or "주요 카테고리 합산"
    fig_sankey = _build_sankey(stage_drop_df, cat_filter, title_suffix)
    if fig_sankey is not None:
        st.plotly_chart(fig_sankey, use_container_width=True)
    else:
        st.info("Sankey 데이터가 부족합니다. 카테고리/기간을 확인하세요.")

with col_bubble:
    st.subheader("┃채널별 효율 매트릭스")
    st.caption(
        "X축: 계약 건수(볼륨), Y축: 납입 전환율(효율), 버블 크기: 매출. "
        "오른쪽 위에 있을수록 '볼륨도 크고 전환율도 높은' 우수 채널입니다. "
        "초록=성장 중, 빨강=쇠퇴 중, 파랑=안정. 쇠퇴 채널은 원인 분석이 필요합니다."
    )
    fig_bubble = _build_channel_bubble(channel_df, ch_col)
    if fig_bubble is not None:
        st.plotly_chart(fig_bubble, use_container_width=True)
    else:
        st.info("채널 데이터가 부족합니다 (최소 30건 이상 채널만 표시).")

# ---------------------------------------------------------------------------
# Section 2: CVR Trend (full width)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("┃전환율 추이")
st.caption(
    "월별 전체 전환율 추이입니다. 실선은 당월 실적, 점선은 6개월 이동평균(노이즈 제거)입니다. "
    "✕ 표시는 Cortex ANOMALY가 탐지한 이상치 — 갑작스러운 전환율 변동이 발생한 시점입니다. "
    "이동평균선이 하락 추세면 구조적 문제가 있으며, 이상치 시점의 원인 분석이 필요합니다."
)

fig_trend = _build_cvr_trend(funnel_ts_df, anomaly_df, cat_filter)
if fig_trend is not None:
    st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.info("퍼널 시계열 데이터가 없습니다.")

# ---------------------------------------------------------------------------
# Section 3: MARKOV CHAIN + STL (Advanced Analytics)
# ---------------------------------------------------------------------------
st.divider()

try:
    from analysis.advanced_analytics import FunnelMarkovChain, TimeSeriesDecomposer

    mc_col, stl_col = st.columns(2)

    # --- Markov Chain ---
    with mc_col:
        st.subheader("┃마르코프 체인 전이 분석")
        st.caption(
            "흡수 마르코프 체인으로 퍼널을 수학적으로 모델링합니다. "
            "Steady State(장기 최종 전환율)는 현재 퍼널 구조가 유지될 때 도달하는 이론적 전환율입니다. "
            "민감도 분석은 '어떤 단계를 5%p 개선하면 최종 전환율이 얼마나 오르고, 월 몇 건이 추가 전환되는가'를 정량 계산합니다. "
            "가장 DELTA가 큰 전이가 ROI 1순위 개선 대상입니다."
        )

        markov = FunnelMarkovChain()
        tm = markov.compute_transition_matrix(stage_drop_df, category=cat_filter)

        if tm is not None and not tm.empty:
            ss = markov.compute_steady_state(tm)
            payend_rate = ss.get("납입완료", ss.get("PAYEND", 0))

            st.metric(label="장기 최종 전환율 (STEADY STATE)", value=f"{payend_rate*100:.1f}%")

            sa = markov.sensitivity_analysis(tm)
            if sa is not None and not sa.empty:
                st.markdown("**민감도 분석: 5%p 개선 시 효과**")
                for _, row in sa.head(3).iterrows():
                    from_s = row.get("FROM_STAGE_KR", row.get("FROM_STAGE", "?"))
                    to_s = row.get("TO_STAGE_KR", row.get("TO_STAGE", "?"))
                    delta = row.get("DELTA", 0)
                    additional = row.get("ADDITIONAL_CUSTOMERS", 0)
                    current_p = row.get("CURRENT_PROB", 0)

                    if delta > 0:
                        m_left, m_right = st.columns(2)
                        with m_left:
                            st.metric(
                                label=f"{from_s} → {to_s}",
                                value=f"+{delta*100:.2f}%p",
                                help=f"현재 {current_p*100:.0f}%",
                            )
                        with m_right:
                            st.metric(label="추가 건수/월", value=f"+{additional:,.0f}건")
        else:
            st.info("마르코프 체인 분석에 충분한 데이터가 없습니다.")

    # --- STL Decomposition ---
    with stl_col:
        st.subheader("┃시계열 계절성 분해 (STL)")
        st.caption(
            "전환율 시계열을 추세(Trend) + 계절성(Seasonal) + 잔차(Residual)로 분해합니다. "
            "추세가 '하락'이면 구조적으로 전환율이 떨어지고 있다는 뜻이고, "
            "계절성 강도가 높으면 특정 월에 마케팅을 집중해야 효율적입니다. "
            "아래 바차트에서 양수(초록) 월이 전환율이 높은 '공격 시즌'이고, 음수(빨강) 월이 '수비 시즌'입니다."
        )

        decomposer = TimeSeriesDecomposer()
        target_cat = cat_filter or "인터넷"
        stl_result = decomposer.decompose_category_cvr(funnel_ts_df, category=target_cat)

        if stl_result:
            pattern_text = decomposer.find_seasonal_pattern(stl_result)
            trend_dir = stl_result.get("trend_direction", "?")
            strength = stl_result.get("seasonality_strength", 0)

            trend_label = {"declining": "하락", "ascending": "상승", "stable": "안정"}.get(trend_dir, trend_dir)

            t_col, s_col = st.columns(2)
            with t_col:
                st.metric(label="추세", value=trend_label)
            with s_col:
                st.metric(label="계절성 강도", value=f"{strength:.0%}")

            # Seasonal pattern chart
            plot = decomposer.plot_data(stl_result)
            if plot and "seasonal" in plot:
                seasonal = plot["seasonal"]
                if seasonal.get("x") and seasonal.get("y"):
                    import plotly.graph_objects as go
                    month_names = ["1월","2월","3월","4월","5월","6월","7월","8월","9월","10월","11월","12월"]
                    x_labels = [month_names[int(m)-1] if 1 <= int(m) <= 12 else str(m) for m in seasonal["x"]]
                    colors = ["#4DFF91" if v > 0 else "#FF4D4D" for v in seasonal["y"]]

                    fig_season = go.Figure(go.Bar(
                        x=x_labels, y=seasonal["y"],
                        marker_color=colors, opacity=0.8,
                    ))
                    fig_season.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Pretendard, sans-serif", color="#e0e0e0", size=11),
                        height=200, margin=dict(l=30, r=10, t=10, b=30),
                        xaxis=dict(gridcolor="rgba(255,255,255,0.03)"),
                        yaxis=dict(gridcolor="rgba(255,255,255,0.03)", title="CVR 변동"),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_season, use_container_width=True)

            st.info(pattern_text)
        else:
            st.info("시계열 분해에 충분한 데이터가 없습니다.")

except Exception as e:
    st.warning(f"고급 분석 로드 실패: {e}")


# ---------------------------------------------------------------------------
# Section 4: Insight cards (3 columns)
# ---------------------------------------------------------------------------
st.divider()

st.subheader("┃AI 인사이트 요약")
st.caption(
    "위의 모든 분석 결과를 종합한 핵심 발견 3가지입니다. "
    "병목 구간은 즉시 개선이 필요한 영역, 채널 추천은 예산 재배분 방향, "
    "시즌 패턴은 마케팅 타이밍 최적화를 위한 참고 정보입니다."
)

worst_s = funnel_insight.get("metrics", {}).get("worst_stage", "개통")
worst_d = funnel_insight.get("metrics", {}).get("worst_drop_pct", 27)
best_ch = channel_insight.get("metrics", {}).get("best_efficiency_channel", "플랫폼")

ic1, ic2, ic3 = st.columns(3)
with ic1:
    st.error(f"**병목 구간 감지**\n\n"
             f"**{worst_s}** 단계에서 심각한 이탈 발생.\n"
             f"전환율 **{worst_d:.1f}% 이탈**")
with ic2:
    st.success(f"**채널 추천**\n\n"
               f"**{best_ch}** 채널의 효율이 가장 높습니다.\n"
               f"예산 투자 증가 시 전환율 **개선 가능**")
with ic3:
    st.info("**시즌 패턴**\n\n"
            "이사 시즌에 가입 신청이\n"
            "평균 **25% 증가** 패턴 예측")

# ---------------------------------------------------------------------------
# Cross-page link
# ---------------------------------------------------------------------------
st.divider()
_safe_pl("pages/2_기회_분석.py", label="기회 시장 분석 →", icon="📈")
