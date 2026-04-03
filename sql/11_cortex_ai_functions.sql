-- 11_cortex_ai_functions.sql
-- Cortex AI Functions: CLASSIFY_TEXT, SUMMARIZE, SENTIMENT, TRANSLATE
-- VIEW 대신 TABLE(CTAS)로 생성하여 AI 함수 비용을 1회로 제한

USE DATABASE TELECOM_DB;
USE WAREHOUSE COMPUTE_WH;

-- =========================================================================
-- 1. 채널 AI 인사이트 테이블
--    CLASSIFY_TEXT  : 채널 성과 등급 분류
--    SUMMARIZE      : 한 줄 요약 생성
-- =========================================================================
CREATE OR REPLACE TABLE MART.CHANNEL_AI_INSIGHT AS
SELECT
    CATEGORY,
    CHANNEL,
    CONTRACT_COUNT,
    PAYEND_CVR,
    AVG_NET_SALES,
    -- AI: 채널 성과 등급 분류
    SNOWFLAKE.CORTEX.CLASSIFY_TEXT(
        CHANNEL || ': CVR ' || ROUND(PAYEND_CVR, 1) || '%, 계약 ' || CONTRACT_COUNT || '건, 매출 ' || ROUND(AVG_NET_SALES) || '원',
        ARRAY_CONSTRUCT('고효율_채널', '중간_채널', '저효율_채널')
    ):"label"::STRING AS AI_TIER,
    -- AI: 한 줄 요약
    SNOWFLAKE.CORTEX.SUMMARIZE(
        '채널 ' || CHANNEL || '의 ' || CATEGORY || ' 카테고리 성과: '
        || '전환율 ' || ROUND(PAYEND_CVR, 1) || '%, '
        || '계약 ' || CONTRACT_COUNT || '건, '
        || '평균매출 ' || ROUND(AVG_NET_SALES) || '원. '
        || '전국 평균 대비 비교 필요.'
    ) AS AI_SUMMARY,
    CURRENT_TIMESTAMP() AS GENERATED_AT
FROM (
    SELECT
        CATEGORY,
        CHANNEL,
        SUM(CONTRACT_COUNT)   AS CONTRACT_COUNT,
        AVG(PAYEND_CVR)       AS PAYEND_CVR,
        AVG(AVG_NET_SALES)    AS AVG_NET_SALES
    FROM STAGING.STG_CHANNEL
    WHERE YEAR_MONTH >= DATEADD(MONTH, -3, (SELECT MAX(YEAR_MONTH) FROM STAGING.STG_CHANNEL))
      AND CATEGORY IN ('인터넷', '렌탈', '모바일')
    GROUP BY CATEGORY, CHANNEL
    HAVING SUM(CONTRACT_COUNT) >= 100
);

-- =========================================================================
-- 2. 지역 AI 인사이트 테이블
--    SENTIMENT  : 지역 성과 감성 점수 (-1 ~ 1)
--    TRANSLATE  : 지역 요약 영문 번역
-- =========================================================================
CREATE OR REPLACE TABLE MART.REGIONAL_AI_INSIGHT AS
SELECT
    INSTALL_STATE,
    TOTAL_CONTRACTS,
    AVG_CVR,
    -- AI: 지역 성과 감성 점수
    SNOWFLAKE.CORTEX.SENTIMENT(
        INSTALL_STATE || ' 지역: 계약 ' || TOTAL_CONTRACTS || '건, 전환율 ' || ROUND(AVG_CVR, 1) || '%'
    ) AS PERFORMANCE_SENTIMENT,
    -- AI: 영문 번역
    SNOWFLAKE.CORTEX.TRANSLATE(
        INSTALL_STATE || ': 계약 ' || TOTAL_CONTRACTS || '건, CVR ' || ROUND(AVG_CVR, 1) || '%',
        'ko', 'en'
    ) AS SUMMARY_EN,
    CURRENT_TIMESTAMP() AS GENERATED_AT
FROM (
    SELECT
        INSTALL_STATE,
        SUM(CONTRACT_COUNT) AS TOTAL_CONTRACTS,
        AVG(PAYEND_CVR)     AS AVG_CVR
    FROM STAGING.STG_REGIONAL
    WHERE YEAR_MONTH >= DATEADD(MONTH, -3, (SELECT MAX(YEAR_MONTH) FROM STAGING.STG_REGIONAL))
    GROUP BY INSTALL_STATE
);
