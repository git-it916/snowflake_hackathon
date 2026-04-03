-- 03_mart.sql
-- ANALYTICS → MART 뷰/테이블 (Streamlit 대시보드용)
-- 실행 순서: 02_analytics.sql → 03_mart.sql

USE DATABASE TELECOM_DB;
USE WAREHOUSE COMPUTE_WH;

-- =========================================================================
-- 1. DT_KPI: 최신 월 핵심 KPI 테이블
-- =========================================================================
CREATE OR REPLACE TABLE MART.DT_KPI AS
WITH latest_month AS (
    -- 미완성 현재월 제외: 두 번째로 큰 YEAR_MONTH 사용
    SELECT MAX(YEAR_MONTH) AS MAX_YM FROM STAGING.STG_FUNNEL
    WHERE YEAR_MONTH < (SELECT MAX(YEAR_MONTH) FROM STAGING.STG_FUNNEL)
),
-- 총 계약 건수
total_contracts AS (
    SELECT SUM(TOTAL_COUNT) AS TOTAL_CONTRACTS
    FROM STAGING.STG_FUNNEL
    WHERE YEAR_MONTH = (SELECT MAX_YM FROM latest_month)
),
-- 평균 전체 CVR
avg_cvr AS (
    SELECT ROUND(AVG(OVERALL_CVR), 2) AS AVG_OVERALL_CVR
    FROM STAGING.STG_FUNNEL
    WHERE YEAR_MONTH = (SELECT MAX_YM FROM latest_month)
      AND OVERALL_CVR > 0
),
-- 최고 성과 채널 (납입완료 기준)
top_channel AS (
    SELECT RECEIVE_PATH_NAME AS TOP_CHANNEL
    FROM STAGING.STG_CHANNEL
    WHERE YEAR_MONTH = (SELECT MAX_YM FROM latest_month)
    GROUP BY RECEIVE_PATH_NAME
    ORDER BY SUM(PAYEND_COUNT) DESC
    LIMIT 1
),
-- 최고 성장 도시
top_growth_city AS (
    SELECT INSTALL_CITY AS TOP_GROWTH_CITY
    FROM ANALYTICS.REGIONAL_DEMAND_SCORE
    WHERE YEAR_MONTH = (SELECT MAX_YM FROM latest_month)
      AND GROWTH_FLAG = TRUE
    ORDER BY DEMAND_SCORE DESC
    LIMIT 1
)
SELECT
    (SELECT MAX_YM FROM latest_month)            AS REPORT_MONTH,
    (SELECT TOTAL_CONTRACTS FROM total_contracts) AS TOTAL_CONTRACTS,
    (SELECT AVG_OVERALL_CVR FROM avg_cvr)        AS AVG_OVERALL_CVR,
    (SELECT TOP_CHANNEL FROM top_channel)        AS TOP_CHANNEL,
    (SELECT TOP_GROWTH_CITY FROM top_growth_city) AS TOP_GROWTH_CITY;

-- =========================================================================
-- 2. V_FUNNEL_TIMESERIES: 월별 카테고리별 퍼널 CVR 추이 (라인 차트용)
-- =========================================================================
CREATE OR REPLACE VIEW MART.V_FUNNEL_TIMESERIES AS
SELECT
    YEAR_MONTH,
    MAIN_CATEGORY_NAME,
    CVR_CONSULT_REQUEST,
    CVR_SUBSCRIPTION,
    CVR_REGISTEND,
    CVR_OPEN,
    CVR_PAYEND,
    OVERALL_CVR,
    TOTAL_COUNT,
    PAYEND_COUNT
FROM STAGING.STG_FUNNEL
ORDER BY MAIN_CATEGORY_NAME, YEAR_MONTH;

-- =========================================================================
-- 3. V_CHANNEL_PERFORMANCE: 채널 순위 + 트렌드 지표 (채널 랭킹 테이블용)
-- =========================================================================
CREATE OR REPLACE VIEW MART.V_CHANNEL_PERFORMANCE AS
WITH latest_month AS (
    SELECT MAX(YEAR_MONTH) AS MAX_YM FROM ANALYTICS.CHANNEL_EFFICIENCY
)
SELECT
    ce.YEAR_MONTH,
    ce.MAIN_CATEGORY_NAME,
    ce.RECEIVE_PATH_NAME,
    ce.CONTRACT_COUNT,
    ce.PAYEND_COUNT,
    ce.PAYEND_CVR,
    ce.AVG_NET_SALES,
    ce.EFFICIENCY_SCORE,
    ce.CHANNEL_SHARE,
    ce.MA6_EFFICIENCY,
    ce.TREND_FLAG,
    ce.CHANNEL_HHI,
    -- 카테고리 내 채널 순위 (최신 월 기준)
    RANK() OVER (
        PARTITION BY ce.MAIN_CATEGORY_NAME
        ORDER BY ce.EFFICIENCY_SCORE DESC
    ) AS EFFICIENCY_RANK
FROM ANALYTICS.CHANNEL_EFFICIENCY ce
WHERE ce.YEAR_MONTH = (SELECT MAX_YM FROM latest_month);

-- =========================================================================
-- 4. V_REGIONAL_HEATMAP: 도시별 수요 점수 (히트맵용)
-- =========================================================================
CREATE OR REPLACE VIEW MART.V_REGIONAL_HEATMAP AS
WITH latest_month AS (
    SELECT MAX(YEAR_MONTH) AS MAX_YM FROM ANALYTICS.REGIONAL_DEMAND_SCORE
)
SELECT
    rd.YEAR_MONTH,
    rd.INSTALL_STATE,
    rd.INSTALL_CITY,
    rd.CONTRACT_COUNT,
    rd.PAYEND_CVR,
    rd.AVG_NET_SALES,
    rd.DEMAND_SCORE,
    rd.BUNDLE_RATIO,
    rd.GROWTH_3M,
    rd.GROWTH_FLAG,
    -- 수요 등급 (5분위)
    NTILE(5) OVER (ORDER BY rd.DEMAND_SCORE DESC) AS DEMAND_QUINTILE
FROM ANALYTICS.REGIONAL_DEMAND_SCORE rd
WHERE rd.YEAR_MONTH = (SELECT MAX_YM FROM latest_month);

-- =========================================================================
-- 5. FORECAST_OUTPUT: Cortex FORECAST 결과 저장 테이블
-- =========================================================================
CREATE TABLE IF NOT EXISTS MART.FORECAST_OUTPUT (
    SERIES_KEY    VARCHAR(200)   COMMENT '시리즈 식별자 (카테고리 또는 지역)',
    SERIES_TYPE   VARCHAR(50)    COMMENT '시리즈 유형: CATEGORY | STATE',
    TARGET_METRIC VARCHAR(100)   COMMENT '예측 대상 지표명',
    TS            DATE           COMMENT '예측 시점',
    FORECAST      FLOAT          COMMENT '예측 값',
    LOWER         FLOAT          COMMENT '예측 하한 (95% CI)',
    UPPER         FLOAT          COMMENT '예측 상한 (95% CI)',
    CREATED_AT    TIMESTAMP_NTZ  DEFAULT CURRENT_TIMESTAMP()
                                 COMMENT '생성 시각'
);

-- =========================================================================
-- 6. ANOMALY_OUTPUT: Cortex ANOMALY 결과 저장 테이블
-- =========================================================================
CREATE TABLE IF NOT EXISTS MART.ANOMALY_OUTPUT (
    SERIES_KEY      VARCHAR(200)   COMMENT '시리즈 식별자',
    TARGET_METRIC   VARCHAR(100)   COMMENT '탐지 대상 지표명',
    TS              DATE           COMMENT '이상 탐지 시점',
    OBSERVED        FLOAT          COMMENT '실측 값',
    EXPECTED        FLOAT          COMMENT '기대 값',
    LOWER           FLOAT          COMMENT '정상 범위 하한',
    UPPER           FLOAT          COMMENT '정상 범위 상한',
    IS_ANOMALY      BOOLEAN        COMMENT '이상 여부',
    ANOMALY_SCORE   FLOAT          COMMENT '이상 점수 (0~1)',
    CREATED_AT      TIMESTAMP_NTZ  DEFAULT CURRENT_TIMESTAMP()
                                   COMMENT '생성 시각'
);
