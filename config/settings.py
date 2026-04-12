# config/settings.py
"""Snowflake 연결 설정 및 Snowpark Session 빌더."""
import logging
import os

from snowflake.snowpark import Session

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass  # SiS 환경에서는 dotenv 불필요

# ---------------------------------------------------------------------------
# 연결 파라미터 (환경 변수에서 로드)
# ---------------------------------------------------------------------------
_REQUIRED_ENV_VARS = ("SF_ACCOUNT", "SF_USER", "SF_PASSWORD")


def _get_connection_params() -> dict:
    """환경 변수에서 Snowflake 연결 파라미터를 로드.

    Raises:
        EnvironmentError: 필수 환경 변수가 누락된 경우.
    """
    missing = [v for v in _REQUIRED_ENV_VARS if not os.getenv(v)]
    if missing:
        raise EnvironmentError(
            f"Snowflake 연결에 필요한 환경 변수가 설정되지 않았습니다: {', '.join(missing)}. "
            ".env 파일을 확인하세요."
        )
    return {
        "account": os.environ["SF_ACCOUNT"],
        "user": os.environ["SF_USER"],
        "password": os.environ["SF_PASSWORD"],
        "role": os.getenv("SF_ROLE", "ACCOUNTADMIN"),
        "warehouse": os.getenv("SF_WAREHOUSE", "COMPUTE_WH"),
        "database": os.getenv("SF_DATABASE", "TELECOM_DB"),
        "schema": os.getenv("SF_SCHEMA", "ANALYTICS"),
    }


def get_database() -> str:
    """현재 설정된 데이터베이스 이름 반환."""
    return os.getenv("SF_DATABASE", "TELECOM_DB")


def get_session() -> Session:
    """Snowpark Session 생성.

    Raises:
        EnvironmentError: 필수 환경 변수 누락 시.
    """
    return Session.builder.configs(_get_connection_params()).create()


def get_streamlit_session() -> Session:
    """Streamlit in Snowflake 환경에서는 내장 세션 사용."""
    try:
        from snowflake.snowpark.context import get_active_session
        return get_active_session()
    except Exception:
        # 로컬 개발 환경 fallback
        return get_session()
