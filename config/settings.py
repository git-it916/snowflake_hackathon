# config/settings.py
"""Snowflake 연결 설정 및 Snowpark Session 빌더."""
import os
from snowflake.snowpark import Session

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # SiS 환경에서는 dotenv 불필요

# Snowflake 연결 파라미터
CONNECTION_PARAMS = {
    "account": os.getenv("SF_ACCOUNT"),
    "user": os.getenv("SF_USER"),
    "password": os.getenv("SF_PASSWORD"),
    "role": os.getenv("SF_ROLE", "ACCOUNTADMIN"),
    "warehouse": os.getenv("SF_WAREHOUSE", "COMPUTE_WH"),
    "database": "TELECOM_DB",
    "schema": "ANALYTICS",
}


def get_session() -> Session:
    """Snowpark Session 생성. 캐시된 세션이 없으면 새로 생성."""
    return Session.builder.configs(CONNECTION_PARAMS).create()


def get_streamlit_session() -> Session:
    """Streamlit in Snowflake 환경에서는 내장 세션 사용."""
    try:
        from snowflake.snowpark.context import get_active_session
        return get_active_session()
    except Exception:
        # 로컬 개발 환경 fallback
        return get_session()
