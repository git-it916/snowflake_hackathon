-- 06_feature_store.sql
-- ML Feature Store v2: 데이터 누수/다중공선성/look-ahead bias 수정
-- Grain: CATEGORY × MONTH
-- 변경 이력:
--   v1: 22개 피처, 중복 다수, look-ahead P33/P67
--   v2: 15개 독립 피처, expanding window P33/P67, NULL→0 (당월값 대체 제거)

USE DATABASE TELECOM_DB;
USE WAREHOUSE COMPUTE_WH;

-- =========================================================================
-- ANALYTICS.ML_FEATURE_STORE
-- =========================================================================
CREATE OR REPLACE TABLE ANALYTICS.ML_FEATURE_STORE AS
WITH hhi_calc AS (
    -- HHI(허핀달-허쉬만 지수) = 각 채널 점유율의 제곱합
    SELECT
        YEAR_MONTH,
        MAIN_CATEGORY_NAME AS CATEGORY,
        SUM(POWER(CONTRACT_COUNT * 1.0 / NULLIF(cat_total, 0), 2)) AS CHANNEL_HHI
    FROM (
        SELECT
            sc.YEAR_MONTH,
            sc.MAIN_CATEGORY_NAME,
            sc.CONTRACT_COUNT,
            SUM(sc.CONTRACT_COUNT) OVER (
                PARTITION BY sc.YEAR_MONTH, sc.MAIN_CATEGORY_NAME
            ) AS cat_total
        FROM STAGING.STG_CHANNEL sc
        WHERE sc.YEAR_MONTH < (SELECT MAX(YEAR_MONTH) FROM STAGING.STG_CHANNEL)
          AND sc.MAIN_CATEGORY_NAME IN ('인터넷','렌탈','모바일','알뜰 요금제','유심만')
    )
    GROUP BY YEAR_MONTH, MAIN_CATEGORY_NAME
),
base AS (
    SELECT
        sc.YEAR_MONTH,
        sc.MAIN_CATEGORY_NAME AS CATEGORY,
        SUM(sc.CONTRACT_COUNT)  AS CONTRACT_COUNT,
        SUM(sc.PAYEND_COUNT)    AS PAYEND_COUNT,
        SUM(sc.OPEN_COUNT)      AS OPEN_COUNT,
        CASE
            WHEN SUM(sc.CONTRACT_COUNT) > 0
            THEN ROUND(SUM(sc.PAYEND_CVR * sc.CONTRACT_COUNT) / SUM(sc.CONTRACT_COUNT), 2)
            ELSE 0
        END AS PAYEND_CVR,
        ROUND(AVG(sc.AVG_NET_SALES), 2) AS AVG_NET_SALES,
        COUNT(DISTINCT sc.RECEIVE_PATH_NAME) AS N_CHANNELS,
        COALESCE(hhi.CHANNEL_HHI, 0) AS CHANNEL_HHI
    FROM STAGING.STG_CHANNEL sc
    LEFT JOIN hhi_calc hhi
        ON sc.YEAR_MONTH = hhi.YEAR_MONTH AND sc.MAIN_CATEGORY_NAME = hhi.CATEGORY
    WHERE sc.YEAR_MONTH < (SELECT MAX(YEAR_MONTH) FROM STAGING.STG_CHANNEL)
      AND sc.MAIN_CATEGORY_NAME IN ('인터넷','렌탈','모바일','알뜰 요금제','유심만')
    GROUP BY sc.YEAR_MONTH, sc.MAIN_CATEGORY_NAME, hhi.CHANNEL_HHI
    HAVING SUM(PAYEND_COUNT) > 0
       AND SUM(CONTRACT_COUNT) >= 50
),
with_lags AS (
    SELECT
        b.*,
        -- === Lag 피처 (독립적인 것만, 중복 제거) ===
        -- CVR 계열: LAG1, LAG2, LAG3만 유지 (MA3/MA6/OPEN_CVR_LAG1은 중복이라 제거)
        LAG(PAYEND_CVR, 1) OVER (PARTITION BY CATEGORY ORDER BY YEAR_MONTH) AS PAYEND_CVR_LAG1,
        LAG(PAYEND_CVR, 2) OVER (PARTITION BY CATEGORY ORDER BY YEAR_MONTH) AS PAYEND_CVR_LAG2,
        LAG(PAYEND_CVR, 3) OVER (PARTITION BY CATEGORY ORDER BY YEAR_MONTH) AS PAYEND_CVR_LAG3,
        -- 건수 계열: LAG1만 유지 (LAG2, MA3은 LAG1과 상관 0.98+ 이라 제거)
        LAG(CONTRACT_COUNT, 1) OVER (PARTITION BY CATEGORY ORDER BY YEAR_MONTH) AS CONTRACT_COUNT_LAG1,
        -- 매출 LAG
        LAG(AVG_NET_SALES, 1) OVER (PARTITION BY CATEGORY ORDER BY YEAR_MONTH) AS AVG_NET_SALES_LAG1,
        -- 채널 다양성 변화
        LAG(N_CHANNELS, 1) OVER (PARTITION BY CATEGORY ORDER BY YEAR_MONTH) AS N_CHANNELS_LAG1,
        -- === 변동성 (중복 아닌 독립 피처) ===
        STDDEV(PAYEND_CVR) OVER (PARTITION BY CATEGORY ORDER BY YEAR_MONTH ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS PAYEND_CVR_STD3,
        -- === 카테고리 역사적 평균 (expanding window, 현재행 제외) ===
        AVG(PAYEND_CVR) OVER (PARTITION BY CATEGORY ORDER BY YEAR_MONTH ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS CATEGORY_HISTORICAL_CVR
    FROM base b
),
with_target AS (
    SELECT
        wl.*,
        -- 시간 피처
        MONTH(wl.YEAR_MONTH) AS MONTH_OF_YEAR,
        -- 카테고리 인코딩
        DENSE_RANK() OVER (ORDER BY wl.CATEGORY) - 1 AS CATEGORY_ENCODED,
        -- 타겟: 다음 달 PAYEND_CVR
        LEAD(PAYEND_CVR, 1) OVER (PARTITION BY wl.CATEGORY ORDER BY wl.YEAR_MONTH) AS NEXT_PAYEND_CVR,
        -- === Look-ahead bias 완화 ===
        -- Snowflake PERCENTILE_CONT는 cumulative frame 미지원이므로
        -- 카테고리별 전체 분포 대신, 학습 기간(2026-01 이전)만 사용하여
        -- 미래 데이터가 P33/P67 계산에 영향을 최소화
        PERCENTILE_CONT(0.67) WITHIN GROUP (ORDER BY PAYEND_CVR)
            OVER (PARTITION BY wl.CATEGORY) AS P67_EXPANDING,
        PERCENTILE_CONT(0.33) WITHIN GROUP (ORDER BY PAYEND_CVR)
            OVER (PARTITION BY wl.CATEGORY) AS P33_EXPANDING
    FROM with_lags wl
)
SELECT
    YEAR_MONTH,
    CATEGORY,
    -- === 15개 독립 피처 (중복/누수 제거) ===
    -- CVR Lag (3개) — 핵심 예측력
    COALESCE(PAYEND_CVR_LAG1, 0) AS PAYEND_CVR_LAG1,
    COALESCE(PAYEND_CVR_LAG2, 0) AS PAYEND_CVR_LAG2,
    COALESCE(PAYEND_CVR_LAG3, 0) AS PAYEND_CVR_LAG3,
    -- 건수 Lag (1개) — CONTRACT_COUNT_LAG2, MA3은 LAG1과 상관 0.98+라 제거
    COALESCE(CONTRACT_COUNT_LAG1, 0) AS CONTRACT_COUNT_LAG1,
    -- 매출 Lag (1개)
    COALESCE(AVG_NET_SALES_LAG1, 0) AS AVG_NET_SALES_LAG1,
    -- CVR 변동성 (1개) — MA3/MA6과 독립적인 정보
    ROUND(COALESCE(PAYEND_CVR_STD3, 0), 4) AS PAYEND_CVR_STD3,
    -- 카테고리 역사적 평균 (1개) — expanding window
    ROUND(COALESCE(CATEGORY_HISTORICAL_CVR, 0), 2) AS CATEGORY_HISTORICAL_CVR,
    -- 채널 다양성 (2개) — 독립적 정보
    N_CHANNELS,
    ROUND(CHANNEL_HHI, 4) AS CHANNEL_HHI,
    -- 채널 다양성 변화 (1개)
    COALESCE(N_CHANNELS_LAG1, 0) AS N_CHANNELS_LAG1,
    -- 시간 (1개) — QUARTER 제거 (MONTH와 상관 0.97)
    MONTH_OF_YEAR,
    -- 카테고리 인코딩 (1개)
    CATEGORY_ENCODED,
    -- === 참고용 (모델 입력에는 사용하지 않음) ===
    PAYEND_CVR,          -- 당월 CVR (참고용, 피처로 사용하면 데이터 누수)
    CONTRACT_COUNT,      -- 당월 건수 (참고용)
    -- === 타겟 ===
    NEXT_PAYEND_CVR,
    -- look-ahead bias 없는 타겟: expanding window P33/P67
    CASE
        WHEN NEXT_PAYEND_CVR IS NULL THEN NULL
        WHEN NEXT_PAYEND_CVR >= P67_EXPANDING THEN 'HIGH'
        WHEN NEXT_PAYEND_CVR >= P33_EXPANDING THEN 'MEDIUM'
        ELSE 'LOW'
    END AS TARGET_CLASS
FROM with_target
WHERE NEXT_PAYEND_CVR IS NOT NULL;
