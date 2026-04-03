"""텔레콤 가입 퍼널 Multi-Agent 시스템.

Snowflake Cortex COMPLETE 기반 3-Agent 아키텍처:
    - AnalystAgent: 퍼널/지역 데이터 분석 에이전트
    - StrategyAgent: 채널 전략 및 시뮬레이션 에이전트
    - AgentOrchestrator: 분석 + 전략을 통합하는 오케스트레이터

모든 LLM 호출은 Snowflake 네이티브 Cortex COMPLETE를 통해 수행됩니다.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from snowflake.snowpark import Session

logger = logging.getLogger(__name__)

_IMPORT_ERRORS: dict[str, str] = {}

try:
    from agents.analyst_agent import AnalystAgent
except Exception as exc:
    _IMPORT_ERRORS["AnalystAgent"] = str(exc)
    AnalystAgent = None  # type: ignore[assignment, misc]

try:
    from agents.strategy_agent import StrategyAgent
except Exception as exc:
    _IMPORT_ERRORS["StrategyAgent"] = str(exc)
    StrategyAgent = None  # type: ignore[assignment, misc]

try:
    from agents.orchestrator import AgentOrchestrator
except Exception as exc:
    _IMPORT_ERRORS["AgentOrchestrator"] = str(exc)
    AgentOrchestrator = None  # type: ignore[assignment, misc]

if _IMPORT_ERRORS:
    for component, error in _IMPORT_ERRORS.items():
        logger.warning("에이전트 컴포넌트 '%s' 임포트 실패: %s", component, error)

__all__ = [
    "AgentOrchestrator",
    "AnalystAgent",
    "StrategyAgent",
    "create_agents",
]


def create_agents(session: Session) -> dict[str, Any]:
    """에이전트 시스템 전체 컴포넌트를 초기화하여 반환.

    Args:
        session: Snowpark 세션 (TELECOM_DB에 연결)

    Returns:
        에이전트 컴포넌트 딕셔너리. 초기화 실패 시 해당 항목은 None.
    """
    agents: dict[str, Any] = {
        "analyst": None,
        "strategist": None,
        "orchestrator": None,
    }

    if AnalystAgent is not None:
        try:
            agents["analyst"] = AnalystAgent(session)
        except Exception as exc:
            logger.warning("AnalystAgent 초기화 실패: %s", exc)

    if StrategyAgent is not None:
        try:
            agents["strategist"] = StrategyAgent(session)
        except Exception as exc:
            logger.warning("StrategyAgent 초기화 실패: %s", exc)

    if AgentOrchestrator is not None:
        try:
            agents["orchestrator"] = AgentOrchestrator(session)
        except Exception as exc:
            logger.warning("AgentOrchestrator 초기화 실패: %s", exc)

    return agents
