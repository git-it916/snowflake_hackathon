"""에이전트 오케스트레이터.

AnalystAgent와 StrategyAgent를 조율하여 3-Phase 분석을 수행하고,
Cortex COMPLETE를 통해 최종 경영진 요약 보고서를 생성하는 오케스트레이터.

Phase 1: 퍼널/지역 데이터 분석 (AnalystAgent)
Phase 2: 채널 전략 수립 (StrategyAgent)
Phase 3: 통합 요약 (Cortex COMPLETE 합성)
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Optional

from agents.analyst_agent import AnalystAgent
from agents.strategy_agent import StrategyAgent
from config.agent_config import (
    CORTEX_MAX_TOKENS,
    CORTEX_MODEL,
    CORTEX_TEMPERATURE,
    SYNTHESIZER_SYSTEM_PROMPT,
)

if TYPE_CHECKING:
    from snowflake.snowpark import Session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 쿼리 분류 키워드
# ---------------------------------------------------------------------------

_ANALYST_KEYWORDS: frozenset[str] = frozenset([
    "퍼널", "전환", "병목", "이탈", "CVR", "지역", "수요", "성장",
    "분석", "트렌드", "추이", "현황", "통계",
])
_STRATEGY_KEYWORDS: frozenset[str] = frozenset([
    "전략", "채널", "최적화", "시뮬레이션", "예측", "추천", "제안",
    "포트폴리오", "배분", "예산", "시나리오", "what-if",
])

_CHAT_SYSTEM_PROMPT: str = (
    "당신은 한국 통신사 가입 퍼널 분석 대시보드의 AI 어시스턴트입니다. "
    "사용자가 인사하거나 일상적인 대화를 하면 자연스럽고 친근하게 응답하세요. "
    "분석이나 데이터에 대한 질문이 아닌 경우, 짧고 자연스러운 대화를 나누되 "
    "'퍼널 분석, 채널 전략, 지역 수요 등에 대해 궁금한 점이 있으시면 질문해주세요'라고 안내하세요. "
    "한국어로 답변하세요."
)


class AgentOrchestrator:
    """Multi-Agent 오케스트레이터.

    AnalystAgent(퍼널/지역 분석)와 StrategyAgent(채널 전략)를 조율하여
    종합 분석 보고서를 생성한다.
    quick_answer로 단일 에이전트 라우팅도 지원한다.
    """

    def __init__(self, session: Session) -> None:
        """AgentOrchestrator 초기화.

        Args:
            session: Snowpark 세션 (TELECOM_DB에 연결)
        """
        self._session = session
        self._analyst = AnalystAgent(session)
        self._strategist = StrategyAgent(session)
        logger.info("AgentOrchestrator 초기화 완료")

    # ------------------------------------------------------------------
    # 공개 API: 전체 분석
    # ------------------------------------------------------------------

    def full_analysis(
        self,
        category: str,
        user_query: Optional[str] = None,
    ) -> dict:
        """3-Phase 전체 분석을 실행.

        Phase 1: AnalystAgent로 퍼널/지역 데이터 분석
        Phase 2: StrategyAgent로 채널 전략 수립
        Phase 3: Cortex COMPLETE로 두 결과를 합성한 경영진 요약 생성

        Args:
            category: 상품 카테고리명
            user_query: 사용자 질문 (None이면 기본 분석 질문 사용)

        Returns:
            종합 분석 결과 딕셔너리:
                - executive_summary: 경영진 요약 텍스트
                - analyst_report: AnalystAgent 분석 결과
                - strategy_report: StrategyAgent 전략 결과
                - recommended_actions: 통합 액션 아이템 리스트
                - confidence_level: 종합 신뢰도
        """
        logger.info(
            "전체 분석 시작: category=%s, query=%s",
            category,
            user_query,
        )

        # Phase 1: 분석
        analyst_result = self._run_analyst(category, user_query)

        # Phase 2: 전략 — 분석가 에이전트의 결과를 컨텍스트로 전달
        strategy_result = self._run_strategist(
            category, analyst_context=analyst_result
        )

        # Phase 3: 합성
        executive_summary = self._synthesize(analyst_result, strategy_result)

        # 통합 액션 아이템
        actions = self._merge_actions(analyst_result, strategy_result)

        # 종합 신뢰도
        confidence = self._overall_confidence(analyst_result, strategy_result)

        result = {
            "executive_summary": executive_summary,
            "analyst_report": analyst_result,
            "strategy_report": strategy_result,
            "recommended_actions": actions,
            "confidence_level": confidence,
        }

        logger.info("전체 분석 완료: confidence=%s", confidence)
        return result

    # ------------------------------------------------------------------
    # 공개 API: 빠른 답변
    # ------------------------------------------------------------------

    def quick_answer(
        self,
        question: str,
        category: Optional[str] = None,
    ) -> str:
        """질문을 적절한 에이전트로 라우팅하여 빠른 답변을 반환.

        _classify_query로 질문 유형을 판별하고,
        해당 에이전트의 핵심 메서드를 호출한다.

        Args:
            question: 사용자 질문 (한국어 또는 영어)
            category: 상품 카테고리 (None이면 전체)

        Returns:
            에이전트 응답 텍스트 문자열
        """
        try:
            query_type = self._classify_query(question)

            if query_type == "chat":
                return self._call_cortex(_CHAT_SYSTEM_PROMPT, question)

            if query_type == "strategy" and category:
                result = self._strategist.recommend(category)
                return result.get("strategy", "전략 생성 실패")

            if query_type == "analyst":
                result = self._analyst.analyze(question, category)
                return result.get("analysis", "분석 생성 실패")

            # both: 분석 에이전트
            result = self._analyst.analyze(question, category)
            return result.get("analysis", "응답 생성 실패")

        except Exception as exc:
            logger.exception("quick_answer 실패: question=%s", question)
            return f"답변 생성 중 오류가 발생했습니다: {exc}"

    # ------------------------------------------------------------------
    # 쿼리 분류
    # ------------------------------------------------------------------

    def _classify_query(self, query: str) -> str:
        """질문을 키워드 기반으로 분류.

        Args:
            query: 사용자 질문 문자열

        Returns:
            분류 결과 ("analyst", "strategy", "both", "chat")
        """
        analyst_score = sum(1 for kw in _ANALYST_KEYWORDS if kw in query)
        strategy_score = sum(1 for kw in _STRATEGY_KEYWORDS if kw in query)

        if analyst_score == 0 and strategy_score == 0:
            return "chat"
        if analyst_score > 0 and strategy_score > 0:
            return "both"
        if strategy_score > analyst_score:
            return "strategy"
        return "analyst"

    # ------------------------------------------------------------------
    # Phase 실행
    # ------------------------------------------------------------------

    def _run_analyst(
        self,
        category: str,
        user_query: Optional[str],
    ) -> dict:
        """Phase 1: AnalystAgent 실행.

        Args:
            category: 상품 카테고리명
            user_query: 사용자 질문

        Returns:
            AnalystAgent 분석 결과 딕셔너리
        """
        default_query = (
            f"'{category}' 카테고리의 가입 퍼널 전환율 현황, "
            "주요 병목 구간, 지역별 수요 패턴을 종합적으로 분석해주세요."
        )
        query = user_query if user_query else default_query

        try:
            return self._analyst.analyze(query, category)
        except Exception as exc:
            logger.exception("Phase 1 (AnalystAgent) 실패")
            return {
                "analysis": f"분석 에이전트 실행 실패: {exc}",
                "data_used": [],
                "confidence": "low",
                "key_findings": [],
            }

    def _run_strategist(
        self,
        category: str,
        analyst_context: dict | None = None,
    ) -> dict:
        """Phase 2: StrategyAgent 실행.

        Args:
            category: 상품 카테고리명
            analyst_context: Phase 1 AnalystAgent 분석 결과 (전략 에이전트에 전달)

        Returns:
            StrategyAgent 전략 결과 딕셔너리
        """
        try:
            return self._strategist.recommend(
                category, analyst_context=analyst_context
            )
        except Exception as exc:
            logger.exception("Phase 2 (StrategyAgent) 실패")
            return {
                "strategy": f"전략 에이전트 실행 실패: {exc}",
                "scenarios": {},
                "confidence": "low",
                "action_items": [],
                "risk_level": "unknown",
            }

    # ------------------------------------------------------------------
    # Phase 3: 합성
    # ------------------------------------------------------------------

    def _synthesize(
        self,
        analyst_result: dict,
        strategy_result: dict,
    ) -> str:
        """두 에이전트의 결과를 Cortex COMPLETE로 합성.

        분석 결과와 전략 결과를 통합하여 경영진 요약 보고서를 생성한다.
        상충하는 인사이트가 있으면 명시적으로 해결한다.

        Args:
            analyst_result: AnalystAgent 분석 결과
            strategy_result: StrategyAgent 전략 결과

        Returns:
            경영진 요약 텍스트 문자열
        """
        analyst_text = analyst_result.get("analysis", "분석 결과 없음")
        analyst_findings = analyst_result.get("key_findings", [])
        analyst_confidence = analyst_result.get("confidence", "unknown")

        strategy_text = strategy_result.get("strategy", "전략 결과 없음")
        action_items = strategy_result.get("action_items", [])
        risk_level = strategy_result.get("risk_level", "unknown")
        strategy_confidence = strategy_result.get("confidence", "unknown")

        findings_text = "\n".join(
            f"- {f}" for f in analyst_findings
        ) if analyst_findings else "없음"

        actions_text = "\n".join(
            f"- {a}" for a in action_items
        ) if action_items else "없음"

        user_message = (
            "## 에이전트 분석 결과 합성 요청\n\n"
            "아래 두 에이전트의 결과를 통합하여 경영진 요약 보고서를 작성해주세요.\n\n"
            f"### 1. 데이터 분석 (신뢰도: {analyst_confidence})\n"
            f"{analyst_text}\n\n"
            f"#### 주요 발견사항\n{findings_text}\n\n"
            f"### 2. 채널 전략 (신뢰도: {strategy_confidence}, 리스크: {risk_level})\n"
            f"{strategy_text}\n\n"
            f"#### 제안 액션\n{actions_text}\n\n"
            "## 응답 형식\n"
            "경영진이 3분 안에 핵심을 파악할 수 있도록 작성하세요:\n"
            "1. **현황 요약** (2-3문장): 현재 퍼널/채널 상태의 핵심\n"
            "2. **주요 인사이트** (3-5개): 분석과 전략에서 도출된 핵심 인사이트\n"
            "3. **추천 전략**: 리스크를 고려한 최적 전략 방향\n"
            "4. **즉시 실행 항목** (3개 이내): 가장 우선순위 높은 액션\n"
            "5. **주의 사항**: 데이터 한계 또는 상충점\n\n"
            "두 에이전트의 인사이트가 상충하는 경우, 어떤 관점이 더 신뢰할 수 있는지 "
            "데이터 근거와 함께 설명하세요."
        )

        return self._call_cortex(SYNTHESIZER_SYSTEM_PROMPT, user_message)

    # ------------------------------------------------------------------
    # 결과 병합
    # ------------------------------------------------------------------

    def _merge_actions(
        self,
        analyst_result: dict,
        strategy_result: dict,
    ) -> list[str]:
        """두 에이전트의 액션 아이템을 통합하고 중복을 제거.

        Args:
            analyst_result: AnalystAgent 결과
            strategy_result: StrategyAgent 결과

        Returns:
            통합된 액션 아이템 리스트 (최대 7개)
        """
        actions: list[str] = []

        # 전략 에이전트의 액션 우선
        strategy_actions = strategy_result.get("action_items", [])
        actions.extend(strategy_actions)

        # 분석 에이전트의 발견사항 중 실행 가능한 항목 추가
        analyst_findings = analyst_result.get("key_findings", [])
        action_keywords = {"개선", "강화", "최적화", "확대", "축소", "전환", "집중"}

        for finding in analyst_findings:
            if any(kw in finding for kw in action_keywords):
                if finding not in actions:
                    actions.append(finding)

        return actions[:7]

    def _overall_confidence(
        self,
        analyst_result: dict,
        strategy_result: dict,
    ) -> str:
        """두 에이전트의 신뢰도를 종합 평가.

        Args:
            analyst_result: AnalystAgent 결과
            strategy_result: StrategyAgent 결과

        Returns:
            종합 신뢰도 등급 ("high", "medium", "low")
        """
        level_scores = {"high": 3, "medium": 2, "low": 1}

        analyst_level = analyst_result.get("confidence", "low")
        strategy_level = strategy_result.get("confidence", "low")

        total = (
            level_scores.get(analyst_level, 1)
            + level_scores.get(strategy_level, 1)
        )

        if total >= 5:
            return "high"
        if total >= 3:
            return "medium"
        return "low"

    # ------------------------------------------------------------------
    # Cortex COMPLETE 호출
    # ------------------------------------------------------------------

    def _call_cortex(self, system_prompt: str, user_message: str) -> str:
        """Snowflake Cortex COMPLETE를 호출.

        Args:
            system_prompt: 시스템 프롬프트
            user_message: 사용자 메시지

        Returns:
            LLM 응답 텍스트. 실패 시 에러 메시지.
        """
        sys_escaped = system_prompt.replace("'", "''")
        user_escaped = user_message.replace("'", "''")

        query = f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            '{CORTEX_MODEL}',
            [{{'role': 'system', 'content': '{sys_escaped}'}},
             {{'role': 'user', 'content': '{user_escaped}'}}],
            {{'temperature': {CORTEX_TEMPERATURE}, 'max_tokens': {CORTEX_MAX_TOKENS}}}
        ) AS RESPONSE
        """

        try:
            result = self._session.sql(query).collect()

            if not result:
                return "Cortex COMPLETE 응답이 비어 있습니다."

            raw = str(result[0]["RESPONSE"])

            try:
                parsed = json.loads(raw)
                choices = parsed.get("choices", [])
                if choices:
                    message = choices[0].get("messages", "")
                    if not message:
                        message = choices[0].get("message", "")
                    if isinstance(message, dict):
                        return message.get("content", raw)
                    return message if message else raw
                return raw
            except (json.JSONDecodeError, IndexError, KeyError):
                return raw

        except Exception as exc:
            logger.exception("Cortex COMPLETE 호출 실패 (Synthesizer)")
            return f"경영진 요약 생성 실패: {exc}"
