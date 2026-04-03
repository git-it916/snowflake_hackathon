"""채널 전략 및 시뮬레이션 에이전트.

Snowflake Cortex COMPLETE와 ML 모델을 결합하여
채널 포트폴리오 최적화 전략을 수립하는 에이전트.

ML 예측, What-if 시뮬레이션, 피처 중요도 분석을 기반으로
실행 가능한 채널 전략을 제안한다.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Optional

from agents.tools import (
    get_feature_importance,
    get_ml_prediction,
    query_channel_efficiency,
    query_channel_performance,
    run_what_if,
)
from config.agent_config import (
    CORTEX_MAX_TOKENS,
    CORTEX_MODEL,
    CORTEX_TEMPERATURE,
    RISK_THRESHOLDS,
    SCENARIO_PRESETS,
    STRATEGIST_SYSTEM_PROMPT,
)

if TYPE_CHECKING:
    from snowflake.snowpark import Session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 리스크 수준 매핑
# ---------------------------------------------------------------------------

_RISK_LEVELS = {
    "conservative": "보수적 (리스크 최소화 우선)",
    "moderate": "균형적 (리스크-수익 균형)",
    "aggressive": "공격적 (수익 극대화 우선)",
}


class StrategyAgent:
    """텔레콤 채널 전략 어드바이저 에이전트.

    ML 모델 예측과 시나리오 시뮬레이션을 기반으로
    채널 최적화 전략을 수립하고 Cortex COMPLETE로 전략 보고서를 생성한다.
    """

    def __init__(self, session: Session) -> None:
        """StrategyAgent 초기화.

        Args:
            session: Snowpark 세션 (TELECOM_DB에 연결)
        """
        self._session = session
        logger.info("StrategyAgent 초기화 완료")

    # ------------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------------

    def recommend(
        self,
        category: str,
        risk_tolerance: str = "moderate",
        analyst_context: dict | None = None,
    ) -> dict:
        """카테고리에 대한 채널 최적화 전략을 수립.

        1. 현재 채널 성과 데이터 수집
        2. ML 모델로 전환율 예측
        3. 3개 시나리오 프리셋으로 시뮬레이션
        4. 피처 중요도 분석
        5. Cortex COMPLETE로 전략 보고서 생성

        Args:
            category: 상품 카테고리명
            risk_tolerance: 리스크 허용 수준
                ("conservative", "moderate", "aggressive")
            analyst_context: Phase 1 AnalystAgent 분석 결과 딕셔너리 (선택)

        Returns:
            전략 결과 딕셔너리:
                - strategy: LLM 생성 전략 텍스트
                - scenarios: 시나리오별 시뮬레이션 결과
                - confidence: 전략 신뢰도 (high/medium/low)
                - action_items: 실행 액션 리스트
                - risk_level: 평가된 리스크 수준
        """
        try:
            # Phase 1: 데이터 수집
            channel_data = self._gather_channel_data(category)

            # Phase 2: ML 예측
            prediction_data = self._get_predictions(category)

            # Phase 3: 시나리오 시뮬레이션
            scenario_results = self._run_scenarios(category)

            # Phase 4: 피처 중요도
            importance_data = get_feature_importance(self._session)

            # Phase 5: 전략 생성
            user_message = self._build_strategy_message(
                category,
                risk_tolerance,
                channel_data,
                prediction_data,
                scenario_results,
                importance_data,
                analyst_context=analyst_context,
            )
            raw_response = self._call_cortex(
                STRATEGIST_SYSTEM_PROMPT, user_message
            )

            action_items = self._extract_actions(raw_response)
            risk_level = self._assess_risk(scenario_results, risk_tolerance)
            confidence = self._assess_confidence(
                channel_data, prediction_data, scenario_results
            )

            return {
                "strategy": raw_response,
                "scenarios": scenario_results,
                "confidence": confidence,
                "action_items": action_items,
                "risk_level": risk_level,
            }

        except Exception as exc:
            logger.exception(
                "StrategyAgent.recommend 실패: category=%s", category
            )
            return {
                "strategy": f"전략 수립 중 오류가 발생했습니다: {exc}",
                "scenarios": {},
                "confidence": "low",
                "action_items": [],
                "risk_level": "unknown",
            }

    # ------------------------------------------------------------------
    # 데이터 수집
    # ------------------------------------------------------------------

    def _gather_channel_data(self, category: str) -> str:
        """채널 관련 데이터를 수집하여 결합.

        Args:
            category: 상품 카테고리명

        Returns:
            결합된 채널 데이터 문자열
        """
        sections: list[str] = []

        perf = query_channel_performance(self._session, category)
        sections.append(perf)

        eff = query_channel_efficiency(self._session, category)
        sections.append(eff)

        return "\n\n".join(sections)

    def _get_predictions(self, category: str) -> str:
        """ML 모델 예측을 수행.

        Args:
            category: 상품 카테고리명

        Returns:
            ML 예측 결과 문자열
        """
        return get_ml_prediction(self._session, category)

    def _run_scenarios(self, category: str) -> dict[str, str]:
        """사전 정의된 시나리오 프리셋으로 시뮬레이션을 실행.

        SCENARIO_PRESETS에서 최대 3개 시나리오를 선택하여 실행한다.

        Args:
            category: 상품 카테고리명

        Returns:
            시나리오 이름 -> 시뮬레이션 결과 문자열 딕셔너리
        """
        results: dict[str, str] = {}
        scenario_names = list(SCENARIO_PRESETS.keys())[:3]

        for name in scenario_names:
            preset = SCENARIO_PRESETS[name]
            channel_weights = preset.get("channel_weights", {})
            result = run_what_if(self._session, category, channel_weights)
            results[name] = result

        return results

    # ------------------------------------------------------------------
    # 메시지 빌드
    # ------------------------------------------------------------------

    def _build_strategy_message(
        self,
        category: str,
        risk_tolerance: str,
        channel_data: str,
        prediction_data: str,
        scenario_results: dict[str, str],
        importance_data: str,
        analyst_context: dict | None = None,
    ) -> str:
        """Cortex COMPLETE에 전달할 전략 요청 메시지를 구성.

        Args:
            category: 상품 카테고리명
            risk_tolerance: 리스크 허용 수준
            channel_data: 채널 성과/효율성 데이터
            prediction_data: ML 예측 결과
            scenario_results: 시나리오 시뮬레이션 결과
            importance_data: 피처 중요도 데이터
            analyst_context: Phase 1 AnalystAgent 분석 결과 (선택)

        Returns:
            구조화된 전략 요청 메시지 문자열
        """
        risk_desc = _RISK_LEVELS.get(risk_tolerance, _RISK_LEVELS["moderate"])

        scenario_text = ""
        for name, result in scenario_results.items():
            preset = SCENARIO_PRESETS.get(name, {})
            desc = preset.get("description", name)
            scenario_text += f"\n### 시나리오: {name} ({desc})\n{result}\n"

        parts: list[str] = [
            "## 전략 수립 요청",
            f"대상 카테고리: {category}",
            f"리스크 허용 수준: {risk_desc}",
            "",
            "## 현재 채널 성과",
            channel_data,
            "",
            "## ML 모델 예측",
            prediction_data,
            "",
            "## 시나리오 시뮬레이션 결과",
            scenario_text,
            "",
            "## 피처 중요도 (모델이 학습한 핵심 요인)",
            importance_data,
        ]

        # 분석가 에이전트 진단 결과 주입
        if analyst_context:
            analyst_summary = analyst_context.get("analysis", "")
            analyst_findings = analyst_context.get("key_findings", [])
            parts.append("")
            parts.append("## 분석가 에이전트의 진단 결과")
            if analyst_summary:
                parts.append(analyst_summary)
            if analyst_findings:
                parts.append("### 주요 발견:")
                parts.extend(f"- {f}" for f in analyst_findings)

        parts.extend([
            "",
            "## 응답 형식",
            "다음 구조로 전략 보고서를 작성해주세요:",
            "1. **전략 요약**: 핵심 전략 방향 2-3문장",
            "2. **시나리오 비교 분석**: 각 시나리오의 장단점 비교",
            "3. **최적 시나리오 추천**: 리스크 허용 수준 고려한 추천",
            "4. **액션 아이템**: 구체적 실행 항목 5개 이내",
            "5. **리스크 요인**: 주의 사항 및 모니터링 포인트",
            f"6. **리스크 임계값 참고**: CVR 하락 경고={RISK_THRESHOLDS['cvr_drop_warning']:.0%}, "
            f"CVR 하락 크리티컬={RISK_THRESHOLDS['cvr_drop_critical']:.0%}, "
            f"HHI 집중={RISK_THRESHOLDS['hhi_concentrated']:.2f}",
        ])

        return "\n".join(parts)

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
            logger.exception("Cortex COMPLETE 호출 실패")
            return f"AI 전략 생성 실패: {exc}"

    # ------------------------------------------------------------------
    # 결과 후처리
    # ------------------------------------------------------------------

    def _extract_actions(self, strategy_text: str) -> list[str]:
        """전략 텍스트에서 액션 아이템을 추출.

        불릿 포인트 또는 번호 매기기로 시작하는 라인을 추출한다.

        Args:
            strategy_text: LLM 생성 전략 텍스트

        Returns:
            액션 아이템 문자열 리스트
        """
        actions: list[str] = []
        in_action_section = False

        for line in strategy_text.split("\n"):
            stripped = line.strip()

            # 액션 섹션 감지
            if "액션" in stripped or "실행" in stripped or "action" in stripped.lower():
                in_action_section = True
                continue

            # 다른 섹션 시작 시 종료
            if stripped.startswith("##") or stripped.startswith("**"):
                if in_action_section and actions:
                    in_action_section = False
                    continue

            if in_action_section and stripped:
                if stripped.startswith(("-", "*", "•")):
                    clean = stripped.lstrip("-*• ").strip()
                    if len(clean) > 5:
                        actions.append(clean)
                elif (
                    len(stripped) > 2
                    and stripped[0].isdigit()
                    and stripped[1] in ".)"
                ):
                    clean = stripped[2:].strip().lstrip(". ").strip()
                    if len(clean) > 5:
                        actions.append(clean)

        return actions[:5]

    def _assess_risk(
        self,
        scenario_results: dict[str, str],
        risk_tolerance: str,
    ) -> str:
        """시나리오 결과와 리스크 허용 수준에 기반한 리스크 평가.

        Args:
            scenario_results: 시나리오 시뮬레이션 결과
            risk_tolerance: 사용자 리스크 허용 수준

        Returns:
            리스크 수준 문자열 ("low", "medium", "high", "critical")
        """
        error_count = sum(
            1 for v in scenario_results.values()
            if "오류" in v or "실패" in v
        )

        if error_count >= 2:
            return "high"
        if error_count == 1:
            return "medium"

        if risk_tolerance == "aggressive":
            return "medium"
        if risk_tolerance == "conservative":
            return "low"
        return "low"

    def _assess_confidence(
        self,
        channel_data: str,
        prediction_data: str,
        scenario_results: dict[str, str],
    ) -> str:
        """전략 제안의 신뢰도를 평가.

        Args:
            channel_data: 채널 성과 데이터
            prediction_data: ML 예측 결과
            scenario_results: 시나리오 시뮬레이션 결과

        Returns:
            신뢰도 등급 ("high", "medium", "low")
        """
        has_channel = "데이터 없음" not in channel_data and "오류" not in channel_data
        has_prediction = "실패" not in prediction_data and "오류" not in prediction_data
        has_scenarios = any(
            "오류" not in v and "실패" not in v
            for v in scenario_results.values()
        )

        score = sum([has_channel, has_prediction, has_scenarios])

        if score >= 3:
            return "high"
        if score >= 2:
            return "medium"
        return "low"
