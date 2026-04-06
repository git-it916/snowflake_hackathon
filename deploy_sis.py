"""Streamlit in Snowflake (SiS) 배포 스크립트.

사용법:
    conda activate snowflake_hackathon
    python deploy_sis.py
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sis_deploy")

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
_APP_NAME = "TELECOM_FUNNEL_INTELLIGENCE"
_STAGE_NAME = "TELECOM_DB.ANALYTICS.STREAMLIT_STAGE"
_DATABASE = "TELECOM_DB"
_SCHEMA = "ANALYTICS"
_WAREHOUSE = "COMPUTE_WH"
_PROJECT_ROOT = Path(__file__).resolve().parent

# 배포할 디렉토리/파일 목록
_DEPLOY_DIRS = [
    "pages",
    "components",
    "analysis",
    "data",
    "ml",
    "agents",
    "config",
    "semantic_model",
]

_DEPLOY_FILES = [
    "app.py",           # main entry
    "environment.yml",  # 패키지 명세
]

# 제외할 파일/디렉토리
_EXCLUDE_PATTERNS = {
    "__pycache__",
    ".pyc",
    "exploration",
    "run_enhanced_pipeline.py",
    "run_pipeline.py",
    "run_phase1b.py",
    "run_analysis_local.py",
    "debug_staging.py",
    "deploy_sis.py",
    ".env",
}


def _should_exclude(path: Path) -> bool:
    for part in path.parts:
        if part in _EXCLUDE_PATTERNS:
            return True
    if path.suffix in (".pyc",):
        return True
    return False


def main() -> None:
    from config.settings import get_session

    session = get_session()
    logger.info("Snowflake 연결 성공")

    # 1. 스테이지 생성
    session.sql(f"USE DATABASE {_DATABASE}").collect()
    session.sql(f"USE SCHEMA {_SCHEMA}").collect()
    session.sql(f"USE WAREHOUSE {_WAREHOUSE}").collect()
    session.sql(
        f"CREATE STAGE IF NOT EXISTS {_STAGE_NAME} "
        f"DIRECTORY = (ENABLE = TRUE) "
        f"COMMENT = 'Streamlit in Snowflake app files'"
    ).collect()
    logger.info("스테이지 생성/확인 완료: %s", _STAGE_NAME)

    # 2. environment.yml 생성
    env_yml = _PROJECT_ROOT / "environment.yml"
    env_yml.write_text(
        "name: sf_env\n"
        "channels:\n"
        "  - snowflake\n"
        "dependencies:\n"
        "  - plotly\n"
        "  - scipy\n"
        "  - statsmodels\n"
        "  - scikit-learn\n"
        "  - xgboost\n"
        "  - shap\n"
        "  - snowflake-ml-python\n",
        encoding="utf-8",
    )
    logger.info("environment.yml 생성 완료")

    # 3. 파일 업로드
    uploaded = 0

    # 단일 파일 업로드
    for fname in _DEPLOY_FILES:
        local_path = _PROJECT_ROOT / fname
        if local_path.exists():
            _upload_file(session, local_path, f"@{_STAGE_NAME}")
            uploaded += 1

    # 디렉토리 업로드
    for dir_name in _DEPLOY_DIRS:
        dir_path = _PROJECT_ROOT / dir_name
        if not dir_path.exists():
            logger.warning("디렉토리 없음, 건너뜀: %s", dir_name)
            continue

        for file_path in dir_path.rglob("*"):
            if file_path.is_dir():
                continue
            if _should_exclude(file_path):
                continue

            rel = file_path.relative_to(_PROJECT_ROOT)
            stage_dir = f"@{_STAGE_NAME}/{rel.parent.as_posix()}"
            _upload_file(session, file_path, stage_dir)
            uploaded += 1

    logger.info("파일 업로드 완료: %d개", uploaded)

    # 4. .streamlit/config.toml 업로드 (테마 설정)
    config_toml = _PROJECT_ROOT / ".streamlit" / "config.toml"
    if config_toml.exists():
        _upload_file(session, config_toml, f"@{_STAGE_NAME}/.streamlit")
        logger.info(".streamlit/config.toml 업로드 완료")

    # 5. Streamlit 앱 생성
    create_sql = f"""
        CREATE OR REPLACE STREAMLIT {_DATABASE}.{_SCHEMA}.{_APP_NAME}
            ROOT_LOCATION = '@{_STAGE_NAME}'
            MAIN_FILE = 'app.py'
            QUERY_WAREHOUSE = '{_WAREHOUSE}'
            COMMENT = '텔레콤 가입 퍼널 AI 인텔리전스 대시보드'
    """
    session.sql(create_sql).collect()
    logger.info("Streamlit 앱 생성 완료: %s.%s.%s", _DATABASE, _SCHEMA, _APP_NAME)

    # 6. 결과 확인
    try:
        result = session.sql(
            f"SHOW STREAMLITS LIKE '{_APP_NAME}' IN SCHEMA {_DATABASE}.{_SCHEMA}"
        ).collect()
        if result:
            logger.info("배포 성공!")
        else:
            logger.info("배포 확인 — Snowsight에서 직접 확인하세요")
    except Exception:
        logger.info("배포 완료 — Snowsight에서 확인하세요")

    logger.info(
        "Snowsight 접속: Projects → Streamlit → %s", _APP_NAME
    )

    session.close()
    logger.info("세션 종료")


def _upload_file(session, local_path: Path, stage_path: str) -> None:
    """PUT으로 파일을 스테이지에 업로드.

    경로 검증: 프로젝트 루트 밖의 파일은 업로드하지 않는다 (path traversal 방지).
    """
    resolved = local_path.resolve()
    if not str(resolved).startswith(str(_PROJECT_ROOT.resolve())):
        logger.error("보안 위반: 프로젝트 루트 외부 경로 업로드 거부 — %s", resolved)
        return

    local_str = str(resolved).replace("\\", "/")
    put_sql = f"PUT 'file://{local_str}' '{stage_path}' AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
    try:
        session.sql(put_sql).collect()
        rel = local_path.relative_to(_PROJECT_ROOT)
        logger.info("  업로드: %s", rel)
    except Exception as exc:
        logger.warning("  업로드 실패: %s → %s", local_path.name, exc)


if __name__ == "__main__":
    main()
