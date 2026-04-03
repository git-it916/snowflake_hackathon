-- 05_stored_procedures.sql
-- 분석용 저장 프로시저
-- 실행 순서: 02_analytics.sql 이후 (테이블 의존)

USE DATABASE TELECOM_DB;
USE WAREHOUSE COMPUTE_WH;

-- =========================================================================
-- SP_FUNNEL_ANALYSIS: 특정 카테고리의 퍼널 분석
--   - 단계별 이탈률
--   - 최악 병목 구간
--   - MoM 추이
-- =========================================================================
CREATE OR REPLACE PROCEDURE ANALYTICS.SP_FUNNEL_ANALYSIS(
    P_CATEGORY VARCHAR
)
RETURNS TABLE (
    YEAR_MONTH            DATE,
    STAGE_NAME            VARCHAR,
    STAGE_ORDER           NUMBER,
    PREV_STAGE_COUNT      NUMBER,
    CURR_STAGE_COUNT      NUMBER,
    DROP_RATE             FLOAT,
    BOTTLENECK_FLAG       BOOLEAN,
    MOM_DROP_CHANGE       FLOAT,
    WORST_BOTTLENECK      VARCHAR
)
LANGUAGE SQL
AS
$$
    -- 퍼널 단계별 이탈률 + MoM 변화 + 최악 병목 표시
    WITH funnel_data AS (
        SELECT
            YEAR_MONTH,
            STAGE_NAME,
            STAGE_ORDER,
            PREV_STAGE_COUNT,
            CURR_STAGE_COUNT,
            DROP_RATE,
            BOTTLENECK_FLAG,
            -- 전월 대비 이탈률 변화 (이탈률이 악화되면 양수)
            DROP_RATE - LAG(DROP_RATE) OVER (
                PARTITION BY STAGE_NAME
                ORDER BY YEAR_MONTH
            ) AS MOM_DROP_CHANGE
        FROM ANALYTICS.FUNNEL_STAGE_DROP
        WHERE MAIN_CATEGORY_NAME = P_CATEGORY
    ),
    worst AS (
        -- 해당 카테고리의 최근 6개월 최악 병목
        SELECT WORST_BOTTLENECK_STAGE
        FROM ANALYTICS.FUNNEL_BOTTLENECKS
        WHERE MAIN_CATEGORY_NAME = P_CATEGORY
        LIMIT 1
    )
    SELECT
        f.YEAR_MONTH,
        f.STAGE_NAME,
        f.STAGE_ORDER,
        f.PREV_STAGE_COUNT,
        f.CURR_STAGE_COUNT,
        f.DROP_RATE,
        f.BOTTLENECK_FLAG,
        ROUND(f.MOM_DROP_CHANGE, 4) AS MOM_DROP_CHANGE,
        w.WORST_BOTTLENECK_STAGE    AS WORST_BOTTLENECK
    FROM funnel_data f
    CROSS JOIN worst w
    ORDER BY f.YEAR_MONTH DESC, f.STAGE_ORDER
$$;

-- =========================================================================
-- SP_CHANNEL_ANALYSIS: 특정 카테고리의 채널 분석
--   - 채널 순위 (효율 점수 기준)
--   - HHI 집중도
--   - 지정 기간 내 트렌드
-- =========================================================================
CREATE OR REPLACE PROCEDURE ANALYTICS.SP_CHANNEL_ANALYSIS(
    P_CATEGORY VARCHAR,
    P_MONTHS   INT DEFAULT 6
)
RETURNS TABLE (
    RECEIVE_PATH_NAME  VARCHAR,
    TOTAL_CONTRACTS    NUMBER,
    AVG_PAYEND_CVR     FLOAT,
    AVG_EFFICIENCY     FLOAT,
    AVG_CHANNEL_SHARE  FLOAT,
    TREND_FLAG         VARCHAR,
    EFFICIENCY_RANK    NUMBER,
    PERIOD_HHI         FLOAT,
    HHI_GRADE          VARCHAR
)
LANGUAGE SQL
AS
$$
    WITH period_data AS (
        SELECT *
        FROM ANALYTICS.CHANNEL_EFFICIENCY
        WHERE MAIN_CATEGORY_NAME = P_CATEGORY
          AND YEAR_MONTH >= DATEADD(MONTH, -P_MONTHS, CURRENT_DATE())
    ),
    channel_agg AS (
        -- 채널별 기간 집계
        SELECT
            RECEIVE_PATH_NAME,
            SUM(CONTRACT_COUNT)           AS TOTAL_CONTRACTS,
            ROUND(AVG(PAYEND_CVR), 2)     AS AVG_PAYEND_CVR,
            ROUND(AVG(EFFICIENCY_SCORE), 4) AS AVG_EFFICIENCY,
            ROUND(AVG(CHANNEL_SHARE), 4)  AS AVG_CHANNEL_SHARE,
            -- 최빈 트렌드 플래그 (직전 월 기준)
            MAX_BY(TREND_FLAG, YEAR_MONTH) AS TREND_FLAG
        FROM period_data
        WHERE CONTRACT_COUNT > 0
        GROUP BY RECEIVE_PATH_NAME
    ),
    ranked AS (
        SELECT
            ca.*,
            RANK() OVER (ORDER BY AVG_EFFICIENCY DESC) AS EFFICIENCY_RANK
        FROM channel_agg ca
    ),
    period_hhi AS (
        -- 기간 평균 HHI
        SELECT ROUND(AVG(CHANNEL_HHI), 4) AS PERIOD_HHI
        FROM (
            SELECT DISTINCT YEAR_MONTH, CHANNEL_HHI
            FROM period_data
        )
    )
    SELECT
        r.RECEIVE_PATH_NAME,
        r.TOTAL_CONTRACTS,
        r.AVG_PAYEND_CVR,
        r.AVG_EFFICIENCY,
        r.AVG_CHANNEL_SHARE,
        r.TREND_FLAG,
        r.EFFICIENCY_RANK,
        h.PERIOD_HHI,
        -- HHI 등급
        CASE
            WHEN h.PERIOD_HHI >= 0.25 THEN 'CONCENTRATED'
            WHEN h.PERIOD_HHI >= 0.15 THEN 'MODERATE'
            ELSE 'DIVERSIFIED'
        END AS HHI_GRADE
    FROM ranked r
    CROSS JOIN period_hhi h
    ORDER BY r.EFFICIENCY_RANK
$$;
