"""Snowpark Session 기반 텔레콤 데이터 로드 헬퍼."""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

import pandas as pd

from config.settings import get_database, get_streamlit_session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema 상수
# ---------------------------------------------------------------------------
_STAGING = "STAGING"
_ANALYTICS = "ANALYTICS"
_MART = "MART"

# ---------------------------------------------------------------------------
# SQL 안전 식별자 허용 패턴
# ---------------------------------------------------------------------------
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,254}$")

# ---------------------------------------------------------------------------
# Cortex 기본 설정
# ---------------------------------------------------------------------------
_DEFAULT_MODEL = "llama3.1-405b"
_DEFAULT_TEMP = 0.3
_SYSTEM_PROMPT_KR = (
    "당신은 한국 통신사 가입 퍼널 데이터를 분석하는 시니어 데이터 분석가입니다. "
    "상담 요청 → 가입 신청 → 접수 → 개통 → 납입 완료까지의 전환 퍼널을 깊이 이해하고, "
    "채널별 효율, 지역별 수요, 마케팅 성과를 종합적으로 분석합니다. "
    "한국어로 답변하되, 데이터 근거를 명확히 제시하세요. "
    "단, 일반적인 인사나 잡담에는 자연스럽게 대화하세요. "
    "데이터 분석은 사용자가 명시적으로 분석을 요청할 때만 수행하세요. "
    "예: '안녕' → '안녕하세요! 채널 전략이나 퍼널 분석에 대해 궁금한 점이 있으시면 질문해주세요.'"
)


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------
class DataLoadError(Exception):
    """데이터 로드 실패 시 발생하는 기본 예외."""


class ConnectionError(DataLoadError):
    """Snowflake 연결 실패."""


class QueryExecutionError(DataLoadError):
    """SQL 쿼리 실행 실패."""


class SchemaValidationError(DataLoadError):
    """스키마/식별자 검증 실패."""


# ---------------------------------------------------------------------------
# SQL 안전 헬퍼
# ---------------------------------------------------------------------------
def _validate_identifier(name: str) -> str:
    """SQL 식별자(테이블명, 컬럼명)가 안전한지 검증.

    허용: 영문, 숫자, 밑줄만. 최대 255자.

    Raises:
        SchemaValidationError: 유효하지 않은 식별자.
    """
    if not _SAFE_IDENTIFIER_RE.match(name):
        raise SchemaValidationError(
            f"유효하지 않은 SQL 식별자: {name!r}"
        )
    return name


def _qualified(schema: str, table: str) -> str:
    """<DATABASE>.<schema>.<table> 형태의 정규화 이름 반환.

    데이터베이스 이름은 환경변수 SF_DATABASE에서 로드 (기본값: TELECOM_DB).

    Raises:
        SchemaValidationError: schema/table 이름이 유효하지 않을 때.
    """
    _validate_identifier(schema)
    _validate_identifier(table)
    db = get_database()
    return f"{db}.{schema}.{table}"


class SnowflakeClient:
    """Snowpark Session 기반 텔레콤 데이터 로더.

    모든 메서드는 원본 데이터를 변경하지 않고 새 DataFrame을 반환합니다.
    """

    def __init__(self, session=None) -> None:
        """클라이언트 초기화.

        Args:
            session: Snowpark Session. None이면 get_streamlit_session() 사용.
        """
        self._session = session if session is not None else get_streamlit_session()

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _query(self, sql: str) -> pd.DataFrame:
        """SQL 실행 후 Pandas DataFrame 반환.

        연결 에러와 쿼리 에러를 구분하여 로깅하며,
        UI에서 원인을 파악할 수 있도록 에러 유형을 명시합니다.

        Raises:
            ConnectionError: Snowflake 세션이 끊어졌거나 인증 실패 시.

        Returns:
            쿼리 결과 DataFrame. 쿼리 실행 에러 시 빈 DataFrame 반환.
        """
        try:
            return self._session.sql(sql).to_pandas()
        except Exception as exc:
            exc_name = type(exc).__name__
            exc_msg = str(exc).lower()
            # 연결/인증 관련 에러는 상위로 전파 (재시도 불가)
            is_connection_error = any(
                keyword in exc_msg
                for keyword in (
                    "authentication", "connection", "could not connect",
                    "session", "timeout", "network", "refused",
                    "ssl", "socket", "expired",
                )
            )
            if is_connection_error:
                logger.error(
                    "Snowflake 연결 에러 [%s]: %s — SQL: %s",
                    exc_name, exc, sql[:100],
                )
                raise ConnectionError(
                    f"Snowflake 연결 실패 ({exc_name}): {exc}"
                ) from exc
            # 쿼리 에러 (테이블 없음, 권한 부족 등)는 빈 DataFrame 반환
            logger.warning(
                "SQL 실행 실패 [%s]: %s — SQL: %s",
                exc_name, exc, sql[:200],
            )
            return pd.DataFrame()

    def _query_with_filter(
        self,
        fqn: str,
        filters: dict[str, str],
    ) -> pd.DataFrame:
        """Snowpark DataFrame API를 사용한 안전한 필터 쿼리.

        SQL 문자열 보간 대신 Snowpark의 col().equal() 체인을 사용하여
        SQL 인젝션을 원천 차단한다.

        Args:
            fqn: 정규화된 테이블 이름 (TELECOM_DB.SCHEMA.TABLE)
            filters: {컬럼명: 값} 딕셔너리 (빈 값은 무시)
        """
        try:
            from snowflake.snowpark.functions import col

            df = self._session.table(fqn)
            for col_name, value in filters.items():
                if value is not None:
                    _validate_identifier(col_name)
                    df = df.filter(col(col_name) == value)
            return df.to_pandas()
        except SchemaValidationError:
            raise
        except Exception:
            logger.exception("테이블 로드 실패: %s", fqn)
            return pd.DataFrame()

    def _load_table(
        self,
        schema: str,
        table: str,
        category: Optional[str] = None,
        state: Optional[str] = None,
        category_col: str = "MAIN_CATEGORY_NAME",
        state_col: str = "INSTALL_STATE",
    ) -> pd.DataFrame:
        """테이블 로드 + 선택적 필터 (Snowpark API로 SQL 인젝션 방지).

        Args:
            category_col: 카테고리 필터 컬럼명 (기본: MAIN_CATEGORY_NAME)
            state_col: 지역 필터 컬럼명 (기본: INSTALL_STATE)
        """
        fqn = _qualified(schema, table)
        filters = {}
        if category is not None:
            filters[category_col] = category
        if state is not None:
            filters[state_col] = state

        return self._query_with_filter(fqn, filters)

    # ------------------------------------------------------------------
    # MART 테이블 로드
    # ------------------------------------------------------------------

    def load_kpi(self) -> pd.DataFrame:
        """MART.DT_KPI — 핵심 KPI 대시보드 데이터."""
        return self._load_table(_MART, "DT_KPI")

    def load_funnel_timeseries(
        self, category: Optional[str] = None
    ) -> pd.DataFrame:
        """MART.V_FUNNEL_TIMESERIES — 퍼널 시계열 데이터.

        Args:
            category: 상품 카테고리 필터. None이면 전체 반환.
        """
        return self._load_table(_MART, "V_FUNNEL_TIMESERIES", category=category)

    def load_channel_performance(
        self, category: Optional[str] = None
    ) -> pd.DataFrame:
        """MART.V_CHANNEL_PERFORMANCE — 채널 성과 뷰."""
        return self._load_table(_MART, "V_CHANNEL_PERFORMANCE", category=category)

    def load_regional_heatmap(self) -> pd.DataFrame:
        """MART.V_REGIONAL_HEATMAP — 지역 히트맵 뷰."""
        return self._load_table(_MART, "V_REGIONAL_HEATMAP")

    def load_forecast(self, metric: str = "CONTRACT_COUNT") -> pd.DataFrame:
        """MART.FORECAST_OUTPUT — 예측 결과.

        Args:
            metric: 필터할 TARGET_METRIC 값 (기본: CONTRACT_COUNT).
                    None이면 전체 반환.
        """
        df = self._load_table(_MART, "FORECAST_OUTPUT")
        if metric and not df.empty and "TARGET_METRIC" in df.columns:
            filtered = df[df["TARGET_METRIC"] == metric]
            return filtered
        return df

    def load_anomalies(self) -> pd.DataFrame:
        """MART.ANOMALY_OUTPUT — 이상 탐지 결과."""
        return self._load_table(_MART, "ANOMALY_OUTPUT")

    def load_data_quality(self) -> pd.DataFrame:
        """MART.DATA_QUALITY_RESULTS — 데이터 품질 검증 결과."""
        db = get_database()
        return self._query(
            f"SELECT * FROM {db}.MART.DATA_QUALITY_RESULTS ORDER BY CHECK_TIME DESC"
        )

    def load_lineage(self) -> pd.DataFrame:
        """MART.V_TABLE_LINEAGE — 테이블 의존성 계보."""
        db = get_database()
        return self._query(f"SELECT * FROM {db}.MART.V_TABLE_LINEAGE")

    def load_lineage_summary(self) -> pd.DataFrame:
        """MART.V_LINEAGE_SUMMARY — 파이프라인 계보 요약."""
        db = get_database()
        return self._query(f"SELECT * FROM {db}.MART.V_LINEAGE_SUMMARY")

    def load_channel_ai_insight(self) -> pd.DataFrame:
        """MART.CHANNEL_AI_INSIGHT — Cortex AI 채널 분석."""
        return self._load_table(_MART, "CHANNEL_AI_INSIGHT")

    def load_regional_ai_insight(self) -> pd.DataFrame:
        """MART.REGIONAL_AI_INSIGHT — Cortex AI 지역 분석."""
        return self._load_table(_MART, "REGIONAL_AI_INSIGHT")

    # ------------------------------------------------------------------
    # Dynamic Tables (실시간 갱신 테이블)
    # ------------------------------------------------------------------

    def load_funnel_live(
        self, category: Optional[str] = None
    ) -> pd.DataFrame:
        """ANALYTICS.DT_FUNNEL_LIVE — 실시간 퍼널 데이터 (1시간 새로고침).

        Dynamic Table로 소스 데이터 변경 시 자동 갱신됩니다.
        MoM 변화량과 3개월 이동평균이 포함됩니다.

        Args:
            category: 상품 카테고리 필터. None이면 전체 반환.
        """
        return self._load_table(
            _ANALYTICS, "DT_FUNNEL_LIVE",
            category=category, category_col="CATEGORY",
        )

    def load_channel_live(
        self, category: Optional[str] = None
    ) -> pd.DataFrame:
        """ANALYTICS.DT_CHANNEL_LIVE — 실시간 채널 성과 (1시간 새로고침).

        Dynamic Table로 소스 데이터 변경 시 자동 갱신됩니다.
        채널별 계약건수, 전환율, 매출 집계가 포함됩니다.

        Args:
            category: 상품 카테고리 필터. None이면 전체 반환.
        """
        return self._load_table(
            _ANALYTICS, "DT_CHANNEL_LIVE",
            category=category, category_col="CATEGORY",
        )

    # ------------------------------------------------------------------
    # ANALYTICS 테이블 로드
    # ------------------------------------------------------------------

    def load_funnel_bottlenecks(self) -> pd.DataFrame:
        """ANALYTICS.FUNNEL_BOTTLENECKS — 퍼널 병목 분석 결과."""
        return self._load_table(_ANALYTICS, "FUNNEL_BOTTLENECKS")

    def load_funnel_stage_drop(
        self, category: Optional[str] = None
    ) -> pd.DataFrame:
        """ANALYTICS.FUNNEL_STAGE_DROP — 퍼널 스테이지 이탈 데이터.

        Args:
            category: 상품 카테고리 필터. None이면 전체 반환.
        """
        return self._load_table(_ANALYTICS, "FUNNEL_STAGE_DROP", category=category)

    def load_channel_efficiency(
        self, category: Optional[str] = None
    ) -> pd.DataFrame:
        """ANALYTICS.CHANNEL_EFFICIENCY — 채널 효율성 데이터."""
        return self._load_table(_ANALYTICS, "CHANNEL_EFFICIENCY", category=category)

    def load_regional_demand(
        self, state: Optional[str] = None
    ) -> pd.DataFrame:
        """ANALYTICS.REGIONAL_DEMAND_SCORE — 지역 수요 점수.

        Args:
            state: 시/도 필터. None이면 전체 반환.
        """
        return self._load_table(_ANALYTICS, "REGIONAL_DEMAND_SCORE", state=state)

    def load_feature_store(self) -> pd.DataFrame:
        """ANALYTICS.ML_FEATURE_STORE — ML 피처 스토어."""
        return self._load_table(_ANALYTICS, "ML_FEATURE_STORE")

    # ------------------------------------------------------------------
    # STAGING 테이블 로드
    # ------------------------------------------------------------------

    def load_marketing(self) -> pd.DataFrame:
        """STAGING.STG_MARKETING — 마케팅 UTM 데이터."""
        return self._load_table(_STAGING, "STG_MARKETING")

    # ------------------------------------------------------------------
    # Stored Procedure 호출
    # ------------------------------------------------------------------

    def _call_stored_procedure(self, sp_fqn: str, param: str) -> dict:
        """저장 프로시저를 Snowpark call_builtin 방식으로 안전하게 호출.

        Args:
            sp_fqn: 프로시저 정규화 이름
            param: 프로시저 파라미터 값

        Returns:
            프로시저 결과 딕셔너리. 실패 시 에러 정보 딕셔너리.
        """
        result = None
        try:
            from snowflake.snowpark.functions import lit
            result = self._session.call(sp_fqn, lit(param))
            if result:
                raw = str(result)
                return json.loads(raw)
            return {"status": "empty", "param": param}
        except json.JSONDecodeError:
            logger.warning("%s 결과 JSON 파싱 실패: %s", sp_fqn, param)
            return {"status": "parse_error", "raw": str(result) if result else ""}
        except Exception:
            logger.exception("%s 호출 실패: %s", sp_fqn, param)
            return {"status": "error", "param": param}

    def run_funnel_analysis(self, category: str) -> dict:
        """퍼널 분석 저장 프로시저 호출.

        Args:
            category: 분석 대상 상품 카테고리.

        Returns:
            프로시저 결과를 딕셔너리로 반환. 실패 시 에러 정보 딕셔너리.
        """
        db = get_database()
        return self._call_stored_procedure(
            f"{db}.ANALYTICS.SP_FUNNEL_ANALYSIS", category
        )

    def run_channel_analysis(self, category: str) -> dict:
        """채널 분석 저장 프로시저 호출.

        Args:
            category: 분석 대상 상품 카테고리.

        Returns:
            프로시저 결과를 딕셔너리로 반환. 실패 시 에러 정보 딕셔너리.
        """
        db = get_database()
        return self._call_stored_procedure(
            f"{db}.ANALYTICS.SP_CHANNEL_ANALYSIS", category
        )

    # ------------------------------------------------------------------
    # Cortex Analyst (자연어 → SQL)
    # ------------------------------------------------------------------

    def ask_analyst(self, question: str) -> dict:
        """Cortex Analyst를 통한 자연어 SQL 질의.

        Snowflake SiS 환경에서는 _snowflake API를 사용하고,
        로컬 환경에서는 Cortex COMPLETE 기반 폴백으로 SQL을 생성합니다.

        Args:
            question: 자연어 분석 질문 (한국어).

        Returns:
            dict 형태의 응답:
            - sql: 생성된 SQL 문 (없으면 None)
            - text: 자연어 설명 텍스트
            - source: 'analyst' | 'complete_fallback'
            - error: 에러 메시지 (정상이면 None)
        """
        # --- SiS 환경: Cortex Analyst REST API ---
        try:
            import _snowflake  # noqa: F811 — SiS 전용 모듈

            payload = json.dumps({
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": question}],
                    }
                ],
                "semantic_model_file": (
                    f"@{get_database()}.PUBLIC.CORTEX_STAGE/telecom_semantic.yaml"
                ),
            })

            resp = _snowflake.send_snow_api_request(
                "POST",
                "/api/v2/cortex/analyst/message",
                {},
                {},
                payload,
                {},
                30000,
            )

            raw_content = resp.get("content", "{}")
            parsed = (
                json.loads(raw_content)
                if isinstance(raw_content, str)
                else raw_content
            )

            return self._extract_analyst_response(parsed)

        except ImportError:
            logger.info(
                "SiS 환경이 아닙니다. Cortex COMPLETE 폴백을 사용합니다."
            )
            return self._analyst_fallback(question)
        except Exception:
            logger.exception("Cortex Analyst API 호출 실패")
            return {
                "sql": None,
                "text": "[Cortex Analyst 호출 실패. 잠시 후 다시 시도해주세요.]",
                "source": "analyst",
                "error": "API 호출 중 오류가 발생했습니다.",
            }

    def _extract_analyst_response(self, parsed: dict) -> dict:
        """Cortex Analyst API 응답에서 SQL과 텍스트를 추출.

        Args:
            parsed: API 응답 JSON (파싱 완료).

        Returns:
            sql, text, source, error 키를 포함한 딕셔너리.
        """
        sql_text = None
        explanation = ""

        # 응답 메시지 구조 탐색
        message = parsed.get("message", parsed)
        content_items = message.get("content", [])

        if isinstance(content_items, str):
            return {
                "sql": None,
                "text": content_items,
                "source": "analyst",
                "error": None,
            }

        for item in content_items:
            item_type = item.get("type", "")
            if item_type == "sql":
                sql_text = item.get("statement", item.get("text", ""))
            elif item_type == "text":
                explanation += item.get("text", "")

        return {
            "sql": sql_text,
            "text": explanation.strip() or "[응답 텍스트 없음]",
            "source": "analyst",
            "error": None,
        }

    def _analyst_fallback(self, question: str) -> dict:
        """로컬 환경 폴백: Cortex COMPLETE로 SQL 생성.

        테이블 스키마 정보를 프롬프트에 포함하여
        자연어 질문을 SQL로 변환합니다.

        Args:
            question: 사용자의 자연어 질문.

        Returns:
            sql, text, source, error 키를 포함한 딕셔너리.
        """
        db = get_database()
        schema_context = (
            f"아래는 {db}의 주요 테이블 스키마입니다.\n\n"
            f"1. {db}.STAGING.STG_FUNNEL (가입 퍼널 전환 데이터, 250행):\n"
            "   - YEAR_MONTH (VARCHAR): 년월\n"
            "   - MAIN_CATEGORY_NAME (VARCHAR): 상품 대분류\n"
            "   - CATEGORY (VARCHAR): 상품 세부분류\n"
            "   - TOTAL_COUNT (NUMBER): 총유입 건수\n"
            "   - CONSULT_REQUEST_COUNT (NUMBER): 상담요청 건수\n"
            "   - SUBSCRIPTION_COUNT (NUMBER): 가입신청 건수\n"
            "   - REGISTEND_COUNT (NUMBER): 접수 건수\n"
            "   - OPEN_COUNT (NUMBER): 개통 건수\n"
            "   - PAYEND_COUNT (NUMBER): 납입완료 건수\n"
            "   - CVR_CONSULT_REQUEST ~ CVR_PAYEND (NUMBER): 각 단계 전환율\n"
            "   - OVERALL_CVR (NUMBER): 전체 전환율\n\n"
            f"2. {db}.STAGING.STG_CHANNEL (채널별 성과 데이터, 7911행):\n"
            "   - YEAR_MONTH, MAIN_CATEGORY_NAME, CATEGORY\n"
            "   - RECEIVE_PATH_NAME (VARCHAR): 접수경로\n"
            "   - CHANNEL (VARCHAR): 유입채널\n"
            "   - INFLOW_PATH_NAME (VARCHAR): 세부유입경로\n"
            "   - CONTRACT_COUNT (NUMBER): 계약건수\n"
            "   - REGISTEND_COUNT, OPEN_COUNT, PAYEND_COUNT (NUMBER)\n"
            "   - OPEN_CVR, PAYEND_CVR (NUMBER): 전환율\n"
            "   - AVG_NET_SALES (NUMBER): 건당 평균매출\n"
            "   - TOTAL_NET_SALES (NUMBER): 총매출\n\n"
            f"3. {db}.STAGING.STG_REGIONAL (지역별 수요 데이터, 23555행):\n"
            "   - YEAR_MONTH, INSTALL_STATE (VARCHAR): 시/도\n"
            "   - INSTALL_CITY (VARCHAR): 시/군/구\n"
            "   - MAIN_CATEGORY_NAME (VARCHAR): 상품 대분류\n"
            "   - CONTRACT_COUNT, PAYEND_COUNT (NUMBER)\n"
            "   - PAYEND_CVR (NUMBER): 납입전환율\n"
            "   - AVG_NET_SALES (NUMBER): 건당 평균매출\n"
            "   - BUNDLE_COUNT (NUMBER): 번들 건수\n"
            "   - STANDALONE_COUNT (NUMBER): 단독 건수\n"
        )

        system_prompt = (
            "당신은 Snowflake SQL 전문가입니다. "
            "사용자의 자연어 질문을 유효한 Snowflake SQL로 변환하세요.\n\n"
            f"{schema_context}\n"
            "규칙:\n"
            "1. 반드시 유효한 Snowflake SQL만 생성하세요.\n"
            f"2. 테이블은 {db}.STAGING.테이블명 형식으로 참조하세요.\n"
            "3. 응답은 JSON 형식으로: "
            '{"sql": "SELECT ...", "explanation": "설명..."}\n'
            "4. SQL이 불필요한 질문이면 sql을 null로 하고 "
            "explanation만 작성하세요.\n"
            "5. 한국어로 설명하세요."
        )

        raw_response = self._cortex_complete(system_prompt, question)

        try:
            result = json.loads(raw_response)
            return {
                "sql": result.get("sql"),
                "text": result.get("explanation", raw_response),
                "source": "complete_fallback",
                "error": None,
            }
        except (json.JSONDecodeError, TypeError):
            return {
                "sql": None,
                "text": raw_response,
                "source": "complete_fallback",
                "error": None,
            }

    # ------------------------------------------------------------------
    # Cortex COMPLETE
    # ------------------------------------------------------------------

    def get_ai_insight(self, category: str) -> str:
        """Cortex COMPLETE 기반 텔레콤 퍼널 인사이트 생성.

        Args:
            category: 분석 대상 상품 카테고리.

        Returns:
            AI 생성 인사이트 문자열. 실패 시 에러 메시지.
        """
        user_msg = (
            f"'{category}' 카테고리의 가입 퍼널을 분석해주세요. "
            "상담요청→가입신청→접수→개통→납입 각 단계의 전환율 패턴, "
            "주요 병목 구간, 그리고 개선 방향을 제시해주세요."
        )
        return self._cortex_complete(_SYSTEM_PROMPT_KR, user_msg)

    def ask_ai(self, question: str, context: Optional[str] = None) -> str:
        """Cortex COMPLETE 기반 자유 질의응답.

        Args:
            question: 사용자 질문.
            context: 추가 컨텍스트 (데이터 요약 등). None이면 기본 시스템 프롬프트만 사용.

        Returns:
            AI 응답 문자열. 실패 시 에러 메시지.
        """
        system = _SYSTEM_PROMPT_KR
        if context is not None:
            system = f"{system}\n\n참고 데이터:\n{context}"
        return self._cortex_complete(system, question)

    def _cortex_complete(
        self,
        system_prompt: str,
        user_message: str,
        model: str = _DEFAULT_MODEL,
        temperature: float = _DEFAULT_TEMP,
    ) -> str:
        """Cortex COMPLETE 내부 호출 (Snowpark API로 SQL 인젝션 방지).

        Snowpark의 call_builtin + lit()을 사용하여 사용자 입력을
        SQL 문자열에 직접 보간하지 않는다.

        Args:
            system_prompt: 시스템 프롬프트.
            user_message: 사용자 메시지.
            model: 사용할 모델 이름.
            temperature: 생성 온도 (0.0 ~ 1.0).

        Returns:
            생성된 텍스트. 실패 시 에러 메시지 문자열.
        """
        try:
            from snowflake.snowpark.functions import call_builtin, lit, parse_json

            messages = json.dumps([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ], ensure_ascii=False)
            options = json.dumps({
                "temperature": temperature,
                "max_tokens": 2048,
            })

            result_df = self._session.sql(
                "SELECT SNOWFLAKE.CORTEX.COMPLETE(?, PARSE_JSON(?), PARSE_JSON(?)) AS RESPONSE",
                params=[model, messages, options],
            ).collect()

            if result_df and len(result_df) > 0:
                raw = str(result_df[0]["RESPONSE"])
                return self._parse_cortex_response(raw)
            return "[응답 없음]"
        except TypeError:
            # Snowpark 버전에 따라 params 미지원 시 안전한 이스케이프 폴백
            return self._cortex_complete_escaped(
                system_prompt, user_message, model, temperature
            )
        except Exception:
            logger.exception("Cortex COMPLETE 호출 실패")
            return "[AI 인사이트 생성 실패. 잠시 후 다시 시도해주세요.]"

    def _cortex_complete_escaped(
        self,
        system_prompt: str,
        user_message: str,
        model: str,
        temperature: float,
    ) -> str:
        """Cortex COMPLETE 폴백: 안전한 이스케이프 방식.

        Snowpark params 미지원 환경(SiS 등)을 위한 폴백.
        모든 사용자 입력에 대해 다중 이스케이프를 적용한다.
        """
        def _escape_for_sql(text: str) -> str:
            return (
                text.replace("\\", "\\\\")
                .replace("'", "''")
                .replace("\n", "\\n")
                .replace("\r", "")
                .replace("\x00", "")
            )

        safe_system = _escape_for_sql(system_prompt)
        safe_user = _escape_for_sql(user_message)
        safe_model = _escape_for_sql(model)

        sql = f"""
            SELECT SNOWFLAKE.CORTEX.COMPLETE(
                '{safe_model}',
                [
                    {{'role': 'system', 'content': '{safe_system}'}},
                    {{'role': 'user', 'content': '{safe_user}'}}
                ],
                {{
                    'temperature': {float(temperature)},
                    'max_tokens': 2048
                }}
            ) AS RESPONSE
        """
        try:
            result = self._session.sql(sql).collect()
            if result and len(result) > 0:
                raw = str(result[0]["RESPONSE"])
                return self._parse_cortex_response(raw)
            return "[응답 없음]"
        except Exception:
            logger.exception("Cortex COMPLETE 폴백 호출 실패")
            return "[AI 인사이트 생성 실패. 잠시 후 다시 시도해주세요.]"

    @staticmethod
    def _parse_cortex_response(raw: str) -> str:
        """Cortex COMPLETE 응답 JSON을 파싱하여 텍스트 추출.

        두 가지 응답 형식을 모두 처리:
        1) {"messages": "text"}  — messages가 문자열
        2) {"message": {"content": "text"}}  — message가 객체
        """
        try:
            parsed = json.loads(raw)
            choice = parsed.get("choices", [{}])[0]
            msg = choice.get("messages") or choice.get("message")
            if isinstance(msg, str):
                return msg
            if isinstance(msg, dict):
                return msg.get("content", raw)
            return raw
        except (json.JSONDecodeError, IndexError, KeyError):
            return raw
