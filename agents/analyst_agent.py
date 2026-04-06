"""퍼널/지역 데이터 분석 에이전트.

Snowflake Cortex COMPLETE를 통해 텔레콤 퍼널 전환율, 병목 구간,
지역별 수요 패턴을 분석하는 전문 에이전트.

키워드 라우팅으로 적절한 도구를 선택하고,
수집된 데이터를 Cortex LLM에 전달하여 분석 결과를 생성한다.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Optional

from agents.tools import (
    query_anomalies,
    query_channel_efficiency,
    query_channel_performance,
    query_forecast,
    query_funnel_bottlenecks,
    query_funnel_data,
    query_marketing,
    query_regional_demand,
    query_regional_growth,
)
from agents.cortex_caller import call_cortex_complete
from config.agent_config import ANALYST_SYSTEM_PROMPT

if TYPE_CHECKING:
    from snowflake.snowpark import Session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 키워드 라우팅 매핑
# ---------------------------------------------------------------------------

_FUNNEL_KEYWORDS: frozenset[str] = frozenset(
    ["퍼널", "전환", "전환율", "병목", "이탈", "CVR", "cvr", "funnel", "bottleneck"]
)
_REGIONAL_KEYWORDS: frozenset[str] = frozenset(
    ["지역", "수요", "도시", "시도", "성장", "regional", "demand", "state"]
)
_CHANNEL_KEYWORDS: frozenset[str] = frozenset(
    ["채널", "효율", "channel", "경로", "인바운드", "아웃바운드", "플랫폼"]
)
_MARKETING_KEYWORDS: frozenset[str] = frozenset(
    ["마케팅", "UTM", "GA4", "marketing", "어트리뷰션", "세션", "소스"]
)


def _detect_topics(query: str) -> list[str]:
    """쿼리 문자열에서 관련 토픽을 감지.

    Args:
        query: 사용자 질문 문자열

    Returns:
        감지된 토픽 리스트 (예: ["funnel", "regional"])
    """
    topics: list[str] = []

    if any(kw in query for kw in _FUNNEL_KEYWORDS):
        topics.append("funnel")
    if any(kw in query for kw in _REGIONAL_KEYWORDS):
        topics.append("regional")
    if any(kw in query for kw in _CHANNEL_KEYWORDS):
        topics.append("channel")
    if any(kw in query for kw in _MARKETING_KEYWORDS):
        topics.append("marketing")

    # 토픽이 감지되지 않으면 전체 분석
    if not topics:
        topics = ["funnel", "channel"]

    return topics


class AnalystAgent:
    """텔레콤 퍼널 및 지역 데이터 분석 에이전트.

    Cortex COMPLETE를 사용하여 데이터 기반 분석을 수행한다.
    키워드 라우팅으로 적절한 도구를 선택하고,
    수집된 데이터를 LLM에 전달하여 한국어 분석 결과를 생성한다.
    """

    def __init__(self, session: Session) -> None:
        """AnalystAgent 초기화.

        Args:
            session: Snowpark 세션 (TELECOM_DB에 연결)
        """
        self._session = session
        logger.info("AnalystAgent 초기화 완료")

    # ------------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------------

    def analyze(
        self,
        query: str,
        category: Optional[str] = None,
    ) -> dict:
        """주어진 질문에 대해 데이터 기반 분석을 수행.

        1. 키워드 라우팅으로 관련 도구를 선택
        2. 도구를 통해 데이터를 수집
        3. Cortex COMPLETE에 데이터 + 질문을 전달
        4. 구조화된 분석 결과를 반환

        Args:
            query: 분석 질문 (한국어 또는 영어)
            category: 상품 카테고리 필터 (None이면 전체)

        Returns:
            분석 결과 딕셔너리:
                - analysis: LLM 생성 분석 텍스트
                - data_used: 사용된 데이터 소스 목록
                - confidence: 신뢰도 (high/medium/low)
                - key_findings: 주요 발견사항 리스트
        """
        try:
            topics = _detect_topics(query)
            data_context, data_sources = self._gather_data(topics, category)

            if not data_context:
                return {
                    "analysis": "분석에 필요한 데이터를 수집할 수 없습니다.",
                    "data_used": [],
                    "confidence": "low",
                    "key_findings": [],
                }

            user_message = self._build_user_message(
                query, category, data_context
            )
            raw_response = self._call_cortex(ANALYST_SYSTEM_PROMPT, user_message)

            key_findings = self._extract_findings(raw_response)
            confidence = self._assess_confidence(data_sources, raw_response)

            return {
                "analysis": raw_response,
                "data_used": data_sources,
                "confidence": confidence,
                "key_findings": key_findings,
            }

        except Exception as exc:
            logger.exception("AnalystAgent.analyze 실패: query=%s", query)
            return {
                "analysis": f"분석 중 오류가 발생했습니다: {exc}",
                "data_used": [],
                "confidence": "low",
                "key_findings": [],
            }

    # ------------------------------------------------------------------
    # 데이터 수집
    # ------------------------------------------------------------------

    def _gather_data(
        self,
        topics: list[str],
        category: Optional[str],
    ) -> tuple[str, list[str]]:
        """토픽에 따라 관련 데이터를 수집.

        Args:
            topics: 분석 토픽 리스트
            category: 상품 카테고리 필터

        Returns:
            (결합된 데이터 문자열, 데이터 소스 이름 리스트) 튜플
        """
        sections: list[str] = []
        sources: list[str] = []

        if "funnel" in topics:
            self._collect_funnel(category, sections, sources)

        if "channel" in topics:
            self._collect_channel(category, sections, sources)

        if "regional" in topics:
            self._collect_regional(sections, sources)

        if "marketing" in topics:
            self._collect_marketing(sections, sources)

        return "\n\n".join(sections), sources

    def _collect_funnel(
        self,
        category: Optional[str],
        sections: list[str],
        sources: list[str],
    ) -> None:
        """퍼널 관련 데이터를 수집하여 sections에 추가."""
        if category:
            funnel_data = query_funnel_data(self._session, category)
            sections.append(funnel_data)
            sources.append("퍼널 시계열")

        bottleneck_data = query_funnel_bottlenecks(self._session, category)
        sections.append(bottleneck_data)
        sources.append("퍼널 병목")

        # Cortex FORECAST (CONTRACT_COUNT) / ANOMALY 데이터 추가
        forecast_data = query_forecast(self._session)
        sections.append(forecast_data)
        sources.append("Cortex FORECAST (계약건수)")

        anomaly_data = query_anomalies(self._session)
        sections.append(anomaly_data)
        sources.append("Cortex ANOMALY")

    def _collect_channel(
        self,
        category: Optional[str],
        sections: list[str],
        sources: list[str],
    ) -> None:
        """채널 관련 데이터를 수집하여 sections에 추가."""
        if category:
            perf_data = query_channel_performance(self._session, category)
            sections.append(perf_data)
            sources.append("채널 성과")

            eff_data = query_channel_efficiency(self._session, category)
            sections.append(eff_data)
            sources.append("채널 효율성")

    def _collect_regional(
        self,
        sections: list[str],
        sources: list[str],
    ) -> None:
        """지역 관련 데이터를 수집하여 sections에 추가."""
        demand_data = query_regional_demand(self._session)
        sections.append(demand_data)
        sources.append("지역 수요")

        growth_data = query_regional_growth(self._session)
        sections.append(growth_data)
        sources.append("지역 성장")

    def _collect_marketing(
        self,
        sections: list[str],
        sources: list[str],
    ) -> None:
        """마케팅 관련 데이터를 수집하여 sections에 추가."""
        mkt_data = query_marketing(self._session)
        sections.append(mkt_data)
        sources.append("GA4 마케팅")

    # ------------------------------------------------------------------
    # 메시지 빌드
    # ------------------------------------------------------------------

    def _build_user_message(
        self,
        query: str,
        category: Optional[str],
        data_context: str,
    ) -> str:
        """Cortex COMPLETE에 전달할 사용자 메시지를 구성.

        Args:
            query: 사용자 질문
            category: 상품 카테고리
            data_context: 수집된 데이터 텍스트

        Returns:
            구조화된 사용자 메시지 문자열
        """
        parts: list[str] = [
            "## 분석 요청",
            f"질문: {query}",
        ]

        if category:
            parts.append(f"대상 카테고리: {category}")

        parts.extend([
            "",
            "## 수집된 데이터",
            data_context,
            "",
            "## 응답 형식",
            "다음 구조로 분석 결과를 작성해주세요:",
            "1. **핵심 요약**: 2-3문장으로 핵심 발견사항 요약",
            "2. **상세 분석**: 데이터 근거를 포함한 상세 분석",
            "3. **주요 발견사항**: 불릿 포인트로 3-5개 핵심 인사이트",
            "4. **데이터 품질 참고**: 이상치나 한계점 언급",
        ])

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Cortex COMPLETE 호출
    # ------------------------------------------------------------------

    def _call_cortex(self, system_prompt: str, user_message: str) -> str:
        """Snowflake Cortex COMPLETE를 안전하게 호출 (공용 유틸리티 사용)."""
        return call_cortex_complete(self._session, system_prompt, user_message)

    # ------------------------------------------------------------------
    # 결과 후처리
    # ------------------------------------------------------------------

    def _extract_findings(self, analysis_text: str) -> list[str]:
        """분석 텍스트에서 주요 발견사항을 추출.

        불릿 포인트(-, *, 1.)로 시작하는 라인을 핵심 인사이트로 추출한다.

        Args:
            analysis_text: LLM 생성 분석 텍스트

        Returns:
            주요 발견사항 문자열 리스트
        """
        findings: list[str] = []

        for line in analysis_text.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            # 불릿 포인트 또는 번호 매기기로 시작하는 라인
            if stripped.startswith(("-", "*", "•")):
                clean = stripped.lstrip("-*• ").strip()
                if len(clean) > 10:
                    findings.append(clean)
            elif len(stripped) > 2 and stripped[0].isdigit() and stripped[1] in ".)" :
                clean = stripped[2:].strip().lstrip(". ").strip()
                if len(clean) > 10:
                    findings.append(clean)

        return findings[:5]

    def _assess_confidence(
        self,
        data_sources: list[str],
        analysis_text: str,
    ) -> str:
        """데이터 소스 수와 분석 품질에 기반한 신뢰도 평가.

        Args:
            data_sources: 사용된 데이터 소스 리스트
            analysis_text: LLM 생성 분석 텍스트

        Returns:
            신뢰도 등급 ("high", "medium", "low")
        """
        source_count = len(data_sources)
        text_length = len(analysis_text)
        has_error = "오류" in analysis_text or "실패" in analysis_text

        if has_error:
            return "low"
        if source_count >= 3 and text_length > 500:
            return "high"
        if source_count >= 2 and text_length > 200:
            return "medium"
        return "low"
