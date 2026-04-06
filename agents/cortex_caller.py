"""Cortex COMPLETE 공통 호출 유틸리티.

모든 에이전트가 공유하는 안전한 Cortex COMPLETE 호출 로직.
SQL 문자열 보간 대신 JSON 직렬화 + 다중 이스케이프를 사용한다.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from config.agent_config import CORTEX_MAX_TOKENS, CORTEX_MODEL, CORTEX_TEMPERATURE

if TYPE_CHECKING:
    from snowflake.snowpark import Session

logger = logging.getLogger(__name__)


def call_cortex_complete(
    session: Session,
    system_prompt: str,
    user_message: str,
    model: str = CORTEX_MODEL,
    temperature: float = CORTEX_TEMPERATURE,
    max_tokens: int = CORTEX_MAX_TOKENS,
) -> str:
    """Snowflake Cortex COMPLETE를 안전하게 호출.

    1차: Snowpark params 바인딩 시도 (SQL 인젝션 원천 차단)
    2차: 안전한 이스케이프 폴백 (SiS 등 params 미지원 환경)

    Args:
        session: Snowpark 세션
        system_prompt: 시스템 프롬프트
        user_message: 사용자 메시지
        model: Cortex 모델명
        temperature: 생성 온도
        max_tokens: 최대 토큰 수

    Returns:
        LLM 응답 텍스트. 실패 시 에러 메시지 문자열.
    """
    try:
        messages = json.dumps([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ], ensure_ascii=False)
        options = json.dumps({
            "temperature": temperature,
            "max_tokens": max_tokens,
        })

        result = session.sql(
            "SELECT SNOWFLAKE.CORTEX.COMPLETE(?, PARSE_JSON(?), PARSE_JSON(?)) AS RESPONSE",
            params=[model, messages, options],
        ).collect()

        if result and len(result) > 0:
            return _parse_response(str(result[0]["RESPONSE"]))
        return "[Cortex 응답 없음]"
    except TypeError:
        return _call_cortex_escaped(
            session, system_prompt, user_message, model, temperature, max_tokens
        )
    except Exception as exc:
        logger.exception("Cortex COMPLETE 호출 실패")
        return f"[AI 응답 생성 실패: {exc}]"


def _call_cortex_escaped(
    session: Session,
    system_prompt: str,
    user_message: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Cortex COMPLETE 폴백: 안전한 이스케이프 방식."""

    def _esc(text: str) -> str:
        return (
            text.replace("\\", "\\\\")
            .replace("'", "''")
            .replace("\n", "\\n")
            .replace("\r", "")
            .replace("\x00", "")
        )

    sql = f"""
    SELECT SNOWFLAKE.CORTEX.COMPLETE(
        '{_esc(model)}',
        [{{'role': 'system', 'content': '{_esc(system_prompt)}'}},
         {{'role': 'user', 'content': '{_esc(user_message)}'}}],
        {{'temperature': {float(temperature)}, 'max_tokens': {int(max_tokens)}}}
    ) AS RESPONSE
    """

    try:
        result = session.sql(sql).collect()
        if result and len(result) > 0:
            return _parse_response(str(result[0]["RESPONSE"]))
        return "[Cortex 응답 없음]"
    except Exception as exc:
        logger.exception("Cortex COMPLETE 폴백 호출 실패")
        return f"[AI 응답 생성 실패: {exc}]"


def _parse_response(raw: str) -> str:
    """Cortex COMPLETE JSON 응답에서 텍스트를 추출."""
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
