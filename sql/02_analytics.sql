-- 02_analytics.sql
-- STAGING → ANALYTICS 분석 테이블 생성
-- 실행 순서: 01_staging.sql → 02_analytics.sql

USE DATABASE TELECOM_DB;
USE WAREHOUSE COMPUTE_WH;

-- =========================================================================
-- 1. FUNNEL_STAGE_DROP: 카테고리 × 월별 퍼널 단계 간 이탈률
-- =========================================================================
CREATE OR REPLACE TABLE ANALYTICS.FUNNEL_STAGE_DROP AS
WITH stage_unpivot AS (
    -- 퍼널 5단계를 행으로 전개
    SELECT
        YEAR_MONTH,
        MAIN_CATEGORY_NAME,
        1 AS STAGE_ORDER, 'CONSULT_REQUEST' AS STAGE_NAME,
        CONSULT_REQUEST_COUNT AS STAGE_COUNT
    FROM STAGING.STG_FUNNEL
    UNION ALL
    SELECT YEAR_MONTH, MAIN_CATEGORY_NAME,
        2, 'SUBSCRIPTION', SUBSCRIPTION_COUNT
    FROM STAGING.STG_FUNNEL
    UNION ALL
    SELECT YEAR_MONTH, MAIN_CATEGORY_NAME,
        3, 'REGISTEND', REGISTEND_COUNT
    FROM STAGING.STG_FUNNEL
    UNION ALL
    SELECT YEAR_MONTH, MAIN_CATEGORY_NAME,
        4, 'OPEN', OPEN_COUNT
    FROM STAGING.STG_FUNNEL
    UNION ALL
    SELECT YEAR_MONTH, MAIN_CATEGORY_NAME,
        5, 'PAYEND', PAYEND_COUNT
    FROM STAGING.STG_FUNNEL
),
with_prev AS (
    SELECT
        s.*,
        LAG(STAGE_COUNT) OVER (
            PARTITION BY YEAR_MONTH, MAIN_CATEGORY_NAME
            ORDER BY STAGE_ORDER
        ) AS PREV_STAGE_COUNT
    FROM stage_unpivot s
)
SELECT
    YEAR_MONTH,
    MAIN_CATEGORY_NAME,
    STAGE_ORDER,
    STAGE_NAME,
    PREV_STAGE_COUNT,
    STAGE_COUNT AS CURR_STAGE_COUNT,
    -- 이탈률: 이전 단계 대비 감소 비율 (첫 단계는 NULL)
    CASE
        WHEN PREV_STAGE_COUNT IS NULL OR PREV_STAGE_COUNT = 0 THEN NULL
        ELSE ROUND(
            1.0 - (STAGE_COUNT::FLOAT / PREV_STAGE_COUNT::FLOAT), 4
        )
    END AS DROP_RATE,
    -- 병목 플래그: 이탈률 15% 초과
    CASE
        WHEN PREV_STAGE_COUNT IS NOT NULL
             AND PREV_STAGE_COUNT > 0
             AND (1.0 - (STAGE_COUNT::FLOAT / PREV_STAGE_COUNT::FLOAT)) > 0.15
        THEN TRUE
        ELSE FALSE
    END AS BOTTLENECK_FLAG
FROM with_prev;

-- =========================================================================
-- 2. FUNNEL_BOTTLENECKS: 카테고리별 최근 6개월 최악 병목 구간
-- =========================================================================
CREATE OR REPLACE TABLE ANALYTICS.FUNNEL_BOTTLENECKS AS
WITH recent_6m AS (
    SELECT *
    FROM ANALYTICS.FUNNEL_STAGE_DROP
    WHERE YEAR_MONTH >= DATEADD(MONTH, -6, CURRENT_DATE())
      AND DROP_RATE IS NOT NULL
),
ranked AS (
    SELECT
        MAIN_CATEGORY_NAME,
        STAGE_NAME,
        AVG(DROP_RATE)  AS AVG_DROP_RATE,
        MAX(DROP_RATE)  AS MAX_DROP_RATE,
        COUNT(*)        AS MONTH_COUNT,
        ROW_NUMBER() OVER (
            PARTITION BY MAIN_CATEGORY_NAME
            ORDER BY AVG(DROP_RATE) DESC
        ) AS RNK
    FROM recent_6m
    GROUP BY MAIN_CATEGORY_NAME, STAGE_NAME
)
SELECT
    MAIN_CATEGORY_NAME,
    STAGE_NAME        AS WORST_BOTTLENECK_STAGE,
    ROUND(AVG_DROP_RATE, 4) AS AVG_DROP_RATE,
    ROUND(MAX_DROP_RATE, 4) AS MAX_DROP_RATE,
    MONTH_COUNT
FROM ranked
WHERE RNK = 1;

-- =========================================================================
-- 3. CHANNEL_EFFICIENCY: 채널 × 카테고리 × 월별 효율 점수
-- =========================================================================
CREATE OR REPLACE TABLE ANALYTICS.CHANNEL_EFFICIENCY AS
WITH median_sales AS (
    -- 카테고리 × 월별 AVG_NET_SALES 중앙값
    SELECT
        YEAR_MONTH,
        MAIN_CATEGORY_NAME,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY AVG_NET_SALES)
            AS MEDIAN_NET_SALES
    FROM STAGING.STG_CHANNEL
    WHERE CONTRACT_COUNT > 0
    GROUP BY YEAR_MONTH, MAIN_CATEGORY_NAME
),
base AS (
    SELECT
        c.YEAR_MONTH,
        c.MAIN_CATEGORY_NAME,
        c.RECEIVE_PATH_NAME,
        c.CONTRACT_COUNT,
        c.PAYEND_COUNT,
        c.PAYEND_CVR,
        c.AVG_NET_SALES,
        c.TOTAL_NET_SALES,
        m.MEDIAN_NET_SALES,
        -- 효율 점수 = PAYEND_CVR × (AVG_NET_SALES / 중앙값)
        CASE
            WHEN m.MEDIAN_NET_SALES > 0
            THEN ROUND(c.PAYEND_CVR * (c.AVG_NET_SALES / m.MEDIAN_NET_SALES), 4)
            ELSE 0
        END AS EFFICIENCY_SCORE,
        -- 채널 점유율
        c.CONTRACT_COUNT::FLOAT /
            NULLIF(SUM(c.CONTRACT_COUNT) OVER (
                PARTITION BY c.YEAR_MONTH, c.MAIN_CATEGORY_NAME
            ), 0) AS CHANNEL_SHARE
    FROM STAGING.STG_CHANNEL c
    LEFT JOIN median_sales m
        ON c.YEAR_MONTH = m.YEAR_MONTH
        AND c.MAIN_CATEGORY_NAME = m.MAIN_CATEGORY_NAME
),
with_ma AS (
    SELECT
        b.*,
        -- 6개월 이동평균 (성장/쇠퇴 판별)
        AVG(EFFICIENCY_SCORE) OVER (
            PARTITION BY MAIN_CATEGORY_NAME, RECEIVE_PATH_NAME
            ORDER BY YEAR_MONTH
            ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
        ) AS MA6_EFFICIENCY,
        -- 직전 월 효율 점수
        LAG(EFFICIENCY_SCORE, 1) OVER (
            PARTITION BY MAIN_CATEGORY_NAME, RECEIVE_PATH_NAME
            ORDER BY YEAR_MONTH
        ) AS PREV_EFFICIENCY
    FROM base b
)
SELECT
    YEAR_MONTH,
    MAIN_CATEGORY_NAME,
    RECEIVE_PATH_NAME,
    CONTRACT_COUNT,
    PAYEND_COUNT,
    PAYEND_CVR,
    AVG_NET_SALES,
    TOTAL_NET_SALES,
    MEDIAN_NET_SALES,
    ROUND(EFFICIENCY_SCORE, 4)  AS EFFICIENCY_SCORE,
    ROUND(CHANNEL_SHARE, 4)     AS CHANNEL_SHARE,
    ROUND(MA6_EFFICIENCY, 4)    AS MA6_EFFICIENCY,
    -- 성장/쇠퇴 플래그
    CASE
        WHEN PREV_EFFICIENCY IS NOT NULL AND PREV_EFFICIENCY > 0
             AND (EFFICIENCY_SCORE - PREV_EFFICIENCY) / PREV_EFFICIENCY > 0.10
        THEN 'GROWTH'
        WHEN PREV_EFFICIENCY IS NOT NULL AND PREV_EFFICIENCY > 0
             AND (EFFICIENCY_SCORE - PREV_EFFICIENCY) / PREV_EFFICIENCY < -0.10
        THEN 'DECLINE'
        ELSE 'STABLE'
    END AS TREND_FLAG,
    -- HHI (카테고리 × 월별): SUM(share^2)
    SUM(POWER(CHANNEL_SHARE, 2)) OVER (
        PARTITION BY YEAR_MONTH, MAIN_CATEGORY_NAME
    ) AS CHANNEL_HHI
FROM with_ma;

-- =========================================================================
-- 4. REGIONAL_DEMAND_SCORE: 도시 × 월별 수요 Z-score 종합
-- =========================================================================
CREATE OR REPLACE TABLE ANALYTICS.REGIONAL_DEMAND_SCORE AS
WITH stats AS (
    -- 월별 전체 평균/표준편차
    SELECT
        YEAR_MONTH,
        AVG(CONTRACT_COUNT)  AS AVG_CONTRACT,
        STDDEV(CONTRACT_COUNT) AS STD_CONTRACT,
        AVG(PAYEND_CVR)      AS AVG_PAYEND_CVR,
        STDDEV(PAYEND_CVR)   AS STD_PAYEND_CVR,
        AVG(AVG_NET_SALES)   AS AVG_SALES,
        STDDEV(AVG_NET_SALES) AS STD_SALES
    FROM STAGING.STG_REGIONAL
    GROUP BY YEAR_MONTH
),
z_scores AS (
    SELECT
        r.YEAR_MONTH,
        r.INSTALL_STATE,
        r.INSTALL_CITY,
        r.CONTRACT_COUNT,
        r.PAYEND_CVR,
        r.AVG_NET_SALES,
        r.BUNDLE_COUNT,
        r.STANDALONE_COUNT,
        -- 개별 Z-score
        CASE WHEN s.STD_CONTRACT > 0
            THEN (r.CONTRACT_COUNT - s.AVG_CONTRACT) / s.STD_CONTRACT
            ELSE 0 END AS Z_CONTRACT,
        CASE WHEN s.STD_PAYEND_CVR > 0
            THEN (r.PAYEND_CVR - s.AVG_PAYEND_CVR) / s.STD_PAYEND_CVR
            ELSE 0 END AS Z_PAYEND_CVR,
        CASE WHEN s.STD_SALES > 0
            THEN (r.AVG_NET_SALES - s.AVG_SALES) / s.STD_SALES
            ELSE 0 END AS Z_SALES
    FROM STAGING.STG_REGIONAL r
    JOIN stats s ON r.YEAR_MONTH = s.YEAR_MONTH
),
with_lag AS (
    SELECT
        z.*,
        -- 종합 수요 점수: 3개 Z-score 합산
        ROUND(Z_CONTRACT + Z_PAYEND_CVR + Z_SALES, 4) AS DEMAND_SCORE,
        -- 결합 비율
        CASE
            WHEN (BUNDLE_COUNT + STANDALONE_COUNT) > 0
            THEN ROUND(
                BUNDLE_COUNT::FLOAT / (BUNDLE_COUNT + STANDALONE_COUNT), 4
            )
            ELSE NULL
        END AS BUNDLE_RATIO,
        -- 3개월 전 계약 건수 (MoM 성장률 계산용)
        LAG(CONTRACT_COUNT, 3) OVER (
            PARTITION BY INSTALL_STATE, INSTALL_CITY
            ORDER BY YEAR_MONTH
        ) AS CONTRACT_3M_AGO
    FROM z_scores z
)
SELECT
    YEAR_MONTH,
    INSTALL_STATE,
    INSTALL_CITY,
    CONTRACT_COUNT,
    PAYEND_CVR,
    AVG_NET_SALES,
    BUNDLE_COUNT,
    STANDALONE_COUNT,
    ROUND(Z_CONTRACT, 4)    AS Z_CONTRACT,
    ROUND(Z_PAYEND_CVR, 4)  AS Z_PAYEND_CVR,
    ROUND(Z_SALES, 4)       AS Z_SALES,
    DEMAND_SCORE,
    BUNDLE_RATIO,
    -- 3개월 MoM 성장률
    CASE
        WHEN CONTRACT_3M_AGO IS NOT NULL AND CONTRACT_3M_AGO > 0
        THEN ROUND(
            (CONTRACT_COUNT - CONTRACT_3M_AGO)::FLOAT / CONTRACT_3M_AGO, 4
        )
        ELSE NULL
    END AS GROWTH_3M,
    -- 성장 플래그: 3개월 MoM > 10%
    CASE
        WHEN CONTRACT_3M_AGO IS NOT NULL AND CONTRACT_3M_AGO > 0
             AND (CONTRACT_COUNT - CONTRACT_3M_AGO)::FLOAT / CONTRACT_3M_AGO > 0.10
        THEN TRUE
        ELSE FALSE
    END AS GROWTH_FLAG
FROM with_lag;
