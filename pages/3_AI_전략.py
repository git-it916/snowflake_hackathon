"""Page 3: AI Strategy -- Full analysis + conversational AI.

Two-panel layout:
  - Left (60%): Analysis tabs (data / channel strategy / executive summary)
  - Right (40%): Chat interface with quick action pills
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

st.set_page_config(page_title="AI 전략", layout="wide")

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

# ---------------------------------------------------------------------------
# Navigation (replaces render_top_nav)
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
try:
    from data.snowflake_client import SnowflakeClient
    CLIENT_AVAILABLE = True
except Exception:
    CLIENT_AVAILABLE = False

try:
    from agents.orchestrator import AgentOrchestrator
    ORCHESTRATOR_AVAILABLE = True
except Exception:
    ORCHESTRATOR_AVAILABLE = False

try:
    from analysis.advanced_analytics import FunnelMarkovChain
    _MARKOV_OK = True
except Exception:
    _MARKOV_OK = False


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------
@st.cache_resource
def _get_client() -> "SnowflakeClient | None":
    if not CLIENT_AVAILABLE:
        return None
    try:
        return SnowflakeClient()
    except Exception:
        return None


@st.cache_resource
def _get_orchestrator() -> "AgentOrchestrator | None":
    if not ORCHESTRATOR_AVAILABLE:
        return None
    try:
        client = _get_client()
        if client is None:
            return None
        return AgentOrchestrator(client._session)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Confidence label helper
# ---------------------------------------------------------------------------
_CONFIDENCE_LABELS: dict[str, str] = {
    "high": "HIGH",
    "medium": "MEDIUM",
    "low": "LOW",
}

_CONFIDENCE_COLORS: dict[str, str] = {
    "high": "#4DFF91",
    "medium": "#FFB84D",
    "low": "#FF4D4D",
}


def _confidence_text(level: str) -> str:
    """Return a plain text confidence label."""
    return _CONFIDENCE_LABELS.get(level, level.upper())


def _confidence_color(level: str) -> str:
    """Return accent color for the confidence level."""
    return _CONFIDENCE_COLORS.get(level, "#FFB84D")


# ---------------------------------------------------------------------------
# Styled section header helper
# ---------------------------------------------------------------------------
def _section_header(icon: str, title: str, subtitle: str = "", color: str = "#00E5FF") -> None:
    """Render section header using native Streamlit."""
    st.subheader(f"┃{title}")
    if subtitle:
        st.caption(subtitle)


def _glass_card(content: str, border_color: str = "rgba(0,229,255,0.15)") -> None:
    """Render content as native Streamlit info box."""
    st.info(content)


# ---------------------------------------------------------------------------
# Full analysis runner with fallback
# ---------------------------------------------------------------------------
def _run_full_analysis(
    orchestrator: "AgentOrchestrator | None",
    client: "SnowflakeClient | None",
    category: str,
    user_query: str | None = None,
) -> dict[str, Any]:
    default_query = (
        f"'{category}' 카테고리의 가입 퍼널 전환율, 채널 효율, "
        "지역 수요를 종합 분석하고 내일 바로 실행할 전략을 추천해주세요."
    )
    query = user_query or default_query

    if orchestrator is not None:
        try:
            return orchestrator.full_analysis(category, query)
        except Exception as exc:
            logger.warning("Orchestrator full_analysis failed: %s", exc)

    fallback_summary = "[분석 에이전트 연결 실패]"
    if client is not None:
        try:
            fallback_summary = client.get_ai_insight(category)
        except Exception:
            fallback_summary = "[AI 인사이트 생성 실패. 잠시 후 다시 시도해주세요.]"

    return {
        "executive_summary": fallback_summary,
        "analyst_report": {
            "analysis": "에이전트 연결 실패로 상세 분석을 제공할 수 없습니다.",
        },
        "strategy_report": {
            "strategy": "에이전트 연결 실패로 전략 추천을 제공할 수 없습니다.",
        },
        "recommended_actions": [
            "에이전트 시스템 연결 상태 확인",
            "Snowflake Cortex 설정 점검",
        ],
        "confidence_level": "low",
    }


# ---------------------------------------------------------------------------
# Chat answer with fallback
# ---------------------------------------------------------------------------
def _get_chat_answer(
    orchestrator: "AgentOrchestrator | None",
    client: "SnowflakeClient | None",
    question: str,
    category: str | None = None,
) -> str:
    if orchestrator is not None:
        try:
            return orchestrator.quick_answer(question, category)
        except Exception as exc:
            logger.warning("Orchestrator quick_answer failed: %s", exc)

    if client is not None:
        try:
            return client.ask_ai(question)
        except Exception:
            pass

    return "AI 응답을 생성할 수 없습니다. Snowflake 연결을 확인해주세요."


# ---------------------------------------------------------------------------
# Markov chain stage label helper
# ---------------------------------------------------------------------------
_STAGE_LABELS_KR: dict[str, str] = {
    "CONSULT_REQUEST": "상담요청",
    "SUBSCRIPTION": "가입신청",
    "REGISTEND": "접수완료",
    "OPEN": "개통",
    "PAYEND": "납입완료",
    "DROP": "이탈",
}


def _stage_kr(stage: str) -> str:
    return _STAGE_LABELS_KR.get(stage, stage)


# =========================================================================
# Initialization + Global sidebar
# =========================================================================
_sidebar_result = render_sidebar()

client = _get_client()
orchestrator = _get_orchestrator()

if client is None and orchestrator is None:
    st.error("Snowflake 연결 실패. `.env` 파일을 확인하세요.")
    st.stop()


# =========================================================================
# PAGE HEADER
# =========================================================================
st.caption("PAGE 3: AI STRATEGY")
st.title("AI 전략 에이전트")

st.markdown(
    """
> **분석 방법**: 3단계 Multi-Agent 오케스트레이션(Analyst → Strategist → Synthesizer)을 통해
> 퍼널 데이터·채널 실적·지역 수요를 종합 분석하고 실행 가능한 전략을 자동 생성합니다.
> **Cortex COMPLETE**(llama3.1-405b)가 각 에이전트를 구동하며,
> 분석 전에도 **마르코프 체인 Steady State**(장기 최종 전환율)와 민감도 분석 결과를 즉시 제공합니다.
> 실시간 Q&A 인터페이스로 데이터 기반 후속 질문이 가능합니다.
>
> **기대효과**: 분석가 없이도 경영진 수준의 전략 보고서 자동 생성 + 데이터 기반 의사결정 지원
"""
)

# --- Cross-page context bar ---
p1 = st.session_state.get("page1_insights", {})
p2 = st.session_state.get("page2_insights", {})

context_parts: list[str] = []
if p1.get("worst_stage"):
    context_parts.append(f"병목: {p1['worst_stage']} ({p1.get('worst_drop', 0):.0f}% 이탈)")
if p1.get("top_channel"):
    context_parts.append(f"상위 채널: {p1['top_channel']}")
if p2.get("top_growth_city"):
    context_parts.append(f"성장 지역: {p2['top_growth_city']}")

if context_parts:
    st.info("이전 분석 요약: " + " | ".join(context_parts))


# ---------------------------------------------------------------------------
# Risk tolerance state
# ---------------------------------------------------------------------------
if "risk_tolerance" not in st.session_state:
    st.session_state["risk_tolerance"] = "중립"

_RISK_OPTIONS = ["보수적", "중립", "공격적"]

# ---------------------------------------------------------------------------
# Control bar (카테고리는 사이드바 전역 필터 사용)
# ---------------------------------------------------------------------------
category_filter = st.session_state.get("global_category", "전체")

ctrl_left, ctrl_right = st.columns([1, 2])

with ctrl_left:
    _section_header("📂", "분석 카테고리", color="#a855f7")
    st.markdown(f"**{category_filter}**")
    st.caption("왼쪽 사이드바에서 변경")

with ctrl_right:
    _section_header("⚡", "리스크 허용도", color="#FFB84D")
    risk_cols = st.columns(3)
    for idx, opt in enumerate(_RISK_OPTIONS):
        with risk_cols[idx]:
            if st.button(
                opt,
                key=f"risk_{opt}",
                type="primary" if st.session_state["risk_tolerance"] == opt else "secondary",
            ):
                st.session_state["risk_tolerance"] = opt
                st.rerun()

current_risk = st.session_state["risk_tolerance"]

st.caption(f"카테고리: {category_filter}  |  리스크: {current_risk}")

st.divider()

# --- Analysis run button ---
run_analysis = st.button(
    "전체 분석 실행",
    type="primary",
    width="stretch",
)

if run_analysis:
    with st.status("AI 에이전트가 분석 중입니다...", expanded=True) as status:
        st.write("Phase 1: 퍼널/지역 데이터 분석 중...")
        st.write("Phase 2: 채널 전략 수립 중...")
        st.write("Phase 3: 통합 경영진 요약 생성 중...")

        result = _run_full_analysis(orchestrator, client, category_filter)
        st.session_state["analysis_result"] = result

        confidence = result.get("confidence_level", "unknown")
        badge_text = _CONFIDENCE_LABELS.get(confidence, confidence.upper())
        status.update(
            label=f"분석 완료 -- 신뢰도: [{badge_text}]",
            state="complete",
            expanded=False,
        )

analysis_result = st.session_state.get("analysis_result", None)


# =========================================================================
# Two-panel layout: Analysis (60%) + Chat (40%)
# =========================================================================
col_analysis, col_chat = st.columns([6, 4])


# =========================================================================
# LEFT PANEL: Analysis tabs
# =========================================================================
with col_analysis:
    _section_header("📊", "분석 결과", color="#00E5FF")

    tab_data, tab_strategy, tab_exec = st.tabs([
        "데이터 분석", "채널 전략", "경영진 요약",
    ])

    # -------------------------------------------------------------------
    # Tab 1: Data Analysis (Analyst Agent) -- consolidated logic
    # -------------------------------------------------------------------
    with tab_data:
        findings: list[str] = []
        confidence = "medium"
        analysis_text = ""

        if analysis_result is not None:
            analyst_report = analysis_result.get("analyst_report", {})
            if isinstance(analyst_report, dict):
                analysis_text = analyst_report.get("analysis", "")
                findings = analyst_report.get("key_findings", [])
                confidence = analyst_report.get("confidence", "medium")

        st.markdown(f"**분석가 Agent 핵심 발견** — 신뢰도: {_confidence_text(confidence)}")

        if findings:
            _FINDING_ICONS = ["📈", "⚠️", "ℹ️", "⭐"]
            for idx, finding in enumerate(findings[:4]):
                icon = _FINDING_ICONS[idx % len(_FINDING_ICONS)]
                with st.expander(f"{icon} 발견 {idx + 1}"):
                    st.markdown(finding)
        else:
            # Markov chain insights as smart defaults when no analysis has run
            _markov_shown = False
            if _MARKOV_OK and client is not None:
                try:
                    markov = FunnelMarkovChain()
                    cat_param = category_filter if category_filter != "전체" else None
                    stage_drop_df = client.load_funnel_stage_drop(cat_param)
                    tm = markov.compute_transition_matrix(stage_drop_df)
                    if tm is not None and not tm.empty:
                        ss = markov.compute_steady_state(tm)
                        payend_rate = ss.get("PAYEND", 0)

                        st.metric(
                            label="마르코프 체인 장기 최종 전환율 (Steady State)",
                            value=f"{payend_rate * 100:.1f}%",
                        )

                        sa = markov.sensitivity_analysis(tm)
                        if sa is not None and not sa.empty:
                            st.markdown("**민감도 분석: 5%p 개선 시 효과 TOP 3**")
                            for _, row in sa.head(3).iterrows():
                                from_s = _stage_kr(str(row.get("FROM_STAGE", "?")))
                                to_s = _stage_kr(str(row.get("TO_STAGE", "?")))
                                delta = row.get("DELTA", 0)
                                additional = row.get("ADDITIONAL_CUSTOMERS", 0)
                                current_p = row.get("CURRENT_PROB", 0)

                                if delta > 0:
                                    m_left, m_right = st.columns(2)
                                    with m_left:
                                        st.metric(
                                            label=f"{from_s} → {to_s}",
                                            value=f"+{delta * 100:.2f}%p",
                                            help=f"현재 {current_p * 100:.0f}%",
                                        )
                                    with m_right:
                                        st.metric(
                                            label="추가 건수/월",
                                            value=f"+{additional:,.0f}건",
                                        )
                    _markov_shown = True
                except Exception:
                    logger.debug("Markov chain fallback failed", exc_info=True)

            if not _markov_shown:
                _DEFAULT_FINDINGS = [
                    ("전환율 추이", "분석을 실행하면 퍼널 단계별 전환율 트렌드를 확인할 수 있습니다."),
                    ("채널 효율", "채널별 ROI 및 가입 전환 효율 분석이 표시됩니다."),
                    ("지역 수요", "지역별 수요 패턴 및 성장 잠재력을 파악합니다."),
                    ("이상 탐지", "데이터 이상치 및 주의가 필요한 구간을 식별합니다."),
                ]
                _FINDING_ICONS_DEFAULT = ["📈", "⚠️", "ℹ️", "⭐"]
                for idx, (title, desc) in enumerate(_DEFAULT_FINDINGS):
                    icon = _FINDING_ICONS_DEFAULT[idx]
                    with st.expander(f"{icon} {title}"):
                        st.markdown(desc)

        if analysis_text:
            st.divider()
            _glass_card(analysis_text)

    # -------------------------------------------------------------------
    # Tab 2: Channel Strategy (Strategy Agent) -- real data
    # -------------------------------------------------------------------
    with tab_strategy:
        strategy_text = ""
        action_items: list[str] = []
        risk_level = "unknown"
        strat_confidence = "unknown"

        if analysis_result is not None:
            strategy_report = analysis_result.get("strategy_report", {})
            if isinstance(strategy_report, dict):
                strategy_text = strategy_report.get("strategy", "")
                action_items = strategy_report.get("action_items", [])
                risk_level = strategy_report.get("risk_level", "unknown")
                strat_confidence = strategy_report.get("confidence", "unknown")

        conf_suffix = ""
        if strat_confidence != "unknown":
            conf_suffix = f" — 신뢰도: {_confidence_text(strat_confidence)}"
        st.markdown(f"**전략가 Agent 추천 믹스**{conf_suffix}")

        # --- Real channel performance from Snowflake ---
        _channel_shown = False
        if client is not None:
            try:
                cat_param = category_filter if category_filter != "전체" else None
                ch_df = client.load_channel_efficiency(cat_param)
                if not ch_df.empty:
                    agg_cols: dict[str, tuple[str, str]] = {}
                    if "CONTRACT_COUNT" in ch_df.columns:
                        agg_cols["계약건수"] = ("CONTRACT_COUNT", "sum")
                    if "PAYEND_CVR" in ch_df.columns:
                        agg_cols["전환율(%)"] = ("PAYEND_CVR", "mean")
                    if "AVG_NET_SALES" in ch_df.columns:
                        agg_cols["평균매출"] = ("AVG_NET_SALES", "mean")

                    if agg_cols and "RECEIVE_PATH_NAME" in ch_df.columns:
                        top_channels = (
                            ch_df.groupby("RECEIVE_PATH_NAME")
                            .agg(**agg_cols)
                            .reset_index()
                        )
                        # Sort by first numeric column
                        sort_col = list(agg_cols.keys())[0]
                        top_channels = (
                            top_channels.sort_values(sort_col, ascending=False)
                            .head(8)
                        )
                        top_channels = top_channels.rename(
                            columns={"RECEIVE_PATH_NAME": "채널"}
                        )
                        if "전환율(%)" in top_channels.columns:
                            top_channels["전환율(%)"] = top_channels["전환율(%)"].round(1)
                        if "평균매출" in top_channels.columns:
                            top_channels["평균매출"] = (
                                top_channels["평균매출"].round(0).astype(int)
                            )

                        st.markdown("**채널별 실적 (실 데이터)**")
                        st.dataframe(top_channels, width="stretch", hide_index=True)
                        _channel_shown = True
            except Exception:
                logger.debug("Channel data load failed", exc_info=True)

        if not _channel_shown:
            st.info("채널 데이터를 불러올 수 없습니다. Snowflake 연결을 확인하세요.")

        # Action item checklist (from AI analysis)
        if action_items:
            st.divider()
            _section_header("✅", "액션 아이템", color="#4DFF91")
            for idx, item in enumerate(action_items):
                st.markdown(f"{idx + 1}. {item}")

        # Risk + confidence display (only if analysis was run)
        if analysis_result is not None:
            st.divider()
            r_col, c_col = st.columns(2)
            with r_col:
                st.metric(label="리스크 수준", value=risk_level)
            with c_col:
                st.metric(label="전략 신뢰도", value=_confidence_text(strat_confidence))

        if strategy_text:
            st.divider()
            _glass_card(strategy_text, border_color="rgba(168,85,247,0.15)")

        if analysis_result is None and not _channel_shown:
            st.info('"전체 분석 실행" 후 구체적인 전략 추천이 표시됩니다.')

    # -------------------------------------------------------------------
    # Tab 3: Executive Summary (Orchestrator)
    # -------------------------------------------------------------------
    with tab_exec:
        if analysis_result is not None:
            exec_summary = analysis_result.get("executive_summary", "")
            recommended_actions = analysis_result.get("recommended_actions", [])
            overall_confidence = analysis_result.get("confidence_level", "unknown")

            _section_header(
                "🎯", "오케스트레이터 종합 결과",
                subtitle="분석가 + 전략가 에이전트의 통합 인사이트",
                color="#00E5FF",
            )

            # Executive summary in glass card
            if exec_summary:
                st.caption("EXECUTIVE SUMMARY")
                _glass_card(exec_summary)

            st.divider()

            # Confidence gauge + overall assessment
            gauge_col, assess_col = st.columns(2)

            risk_pct = {"high": 85, "medium": 55, "low": 25}.get(overall_confidence, 50)

            with gauge_col:
                st.metric(label="신뢰도 게이지", value=f"{risk_pct}%")

            with assess_col:
                st.metric(label="종합 평가", value=_confidence_text(overall_confidence))

            # Priority action list
            if recommended_actions:
                st.divider()
                _section_header("🚀", "우선순위 액션", color="#4DFF91")

                for idx, action in enumerate(recommended_actions):
                    st.markdown(f"**{idx + 1}.** {action}")

                # Progress tracker
                total = len(recommended_actions)
                completed = sum(
                    1 for i in range(total)
                    if st.session_state.get(f"action_check_{i}", False)
                )
                pct = int((completed / total) * 100) if total > 0 else 0

                st.divider()
                st.caption(f"실행 진행률: {completed}/{total}")
                st.progress(pct / 100)

                # Actual checkboxes for tracking
                for idx, action in enumerate(recommended_actions):
                    key = f"action_check_{idx}"
                    st.checkbox(action, value=st.session_state.get(key, False), key=key)

            else:
                st.info("추천 액션이 없습니다.")

        else:
            _section_header(
                "🎯", "오케스트레이터 종합 결과",
                subtitle="분석가 + 전략가 에이전트의 통합 인사이트",
                color="#00E5FF",
            )
            st.info(
                '위의 "전체 분석 실행" 버튼을 클릭하면 '
                "AI 에이전트가 종합 분석을 시작합니다."
            )


# =========================================================================
# RIGHT PANEL: Chat interface
# =========================================================================
with col_chat:
    _section_header("🤖", "AI 에이전트 Q&A", color="#00E5FF")
    st.success("Online", icon="🟢")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": (
                    "안녕하세요! 현재 대시보드의 데이터 분석 및 전략 추천 결과에 "
                    "대해 궁금한 점이 있으신가요?"
                ),
            }
        ]

    # Display chat messages
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Quick action pills
    st.caption("빠른 질문")

    _QUICK_ACTIONS = [
        (
            "퍼널 병목 개선 방안",
            "현재 퍼널에서 가장 큰 병목 구간은 어디이고, 어떻게 개선할 수 있을까?",
        ),
        (
            "채널별 ROI 비교",
            "계약건수 대비 전환율이 높은 채널 Top 5와 투자 우선순위를 분석해줘",
        ),
        (
            "시즌별 마케팅 전략",
            "월별 계절성 패턴을 고려한 분기별 마케팅 집중 시기를 추천해줘",
        ),
    ]

    quick_q = None
    qcols = st.columns(3)
    for i, (label, question) in enumerate(_QUICK_ACTIONS):
        with qcols[i]:
            if st.button(label, key=f"quick_{i}"):
                quick_q = question

    # Chat input
    user_input = st.chat_input("질문을 입력하세요...")
    question = quick_q or user_input

    if question:
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("AI 분석 중..."):
                answer = _get_chat_answer(
                    orchestrator, client, question, category_filter,
                )
            st.markdown(answer)
        st.session_state["messages"].append({"role": "assistant", "content": answer})
