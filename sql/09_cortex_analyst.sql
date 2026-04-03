-- 09_cortex_analyst.sql
-- Cortex Analyst: 시맨틱 모델 업로드 및 자연어 SQL 서비스 설정
-- Snowflake CoCo Skill: cortex_analyst

USE DATABASE TELECOM_DB;
USE WAREHOUSE COMPUTE_WH;

-- =========================================================================
-- 1. 시맨틱 모델 스테이지 확인 및 생성
--    CORTEX_STAGE가 없으면 새로 생성합니다.
-- =========================================================================
CREATE STAGE IF NOT EXISTS TELECOM_DB.PUBLIC.CORTEX_STAGE
    DIRECTORY = (ENABLE = TRUE)
    COMMENT = 'Cortex Analyst 시맨틱 모델 저장소';

-- =========================================================================
-- 2. 시맨틱 모델 YAML 업로드
--    로컬 파일을 내부 스테이지에 업로드합니다.
--    SnowSQL 또는 Snowsight에서 실행하세요.
-- =========================================================================
PUT file://semantic_model/telecom_semantic.yaml
    @TELECOM_DB.PUBLIC.CORTEX_STAGE
    AUTO_COMPRESS = FALSE
    OVERWRITE = TRUE;

-- =========================================================================
-- 3. 업로드 확인
--    스테이지에 파일이 정상적으로 업로드되었는지 확인합니다.
-- =========================================================================
LIST @TELECOM_DB.PUBLIC.CORTEX_STAGE/telecom_semantic.yaml;

-- =========================================================================
-- 4. 시맨틱 모델 테스트 쿼리
--    Cortex Analyst REST API 호출 예시입니다.
--    실제 호출은 Python SDK 또는 Streamlit 앱에서 수행합니다.
--
--    REST API 엔드포인트:
--      POST /api/v2/cortex/analyst/message
--
--    요청 본문:
--    {
--      "messages": [
--        {
--          "role": "user",
--          "content": [
--            {"type": "text", "text": "월별 전체 전환율 추이를 보여줘"}
--          ]
--        }
--      ],
--      "semantic_model_file": "@TELECOM_DB.PUBLIC.CORTEX_STAGE/telecom_semantic.yaml"
--    }
-- =========================================================================

-- =========================================================================
-- 5. 권한 설정
--    Cortex Analyst 사용에 필요한 권한을 부여합니다.
-- =========================================================================
-- 스테이지 읽기 권한 (앱 사용자 역할에 부여)
-- GRANT READ ON STAGE TELECOM_DB.PUBLIC.CORTEX_STAGE TO ROLE <APP_ROLE>;

-- 테이블 SELECT 권한 (Cortex Analyst가 쿼리를 실행하려면 필요)
-- GRANT SELECT ON ALL TABLES IN SCHEMA TELECOM_DB.STAGING TO ROLE <APP_ROLE>;
-- GRANT SELECT ON ALL TABLES IN SCHEMA TELECOM_DB.ANALYTICS TO ROLE <APP_ROLE>;
-- GRANT SELECT ON ALL TABLES IN SCHEMA TELECOM_DB.MART TO ROLE <APP_ROLE>;

-- =========================================================================
-- 6. Cortex Analyst 사용 가능 모델 확인
-- =========================================================================
-- SELECT SNOWFLAKE.CORTEX.LIST_MODELS();
