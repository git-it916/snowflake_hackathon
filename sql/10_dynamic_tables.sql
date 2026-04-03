-- 10_dynamic_tables.sql
-- Dynamic Tables + Tasks + Alerts
-- Snowflake 자동 새로고침 & 스케줄링 파이프라인

USE DATABASE TELECOM_DB;
USE WAREHOUSE COMPUTE_WH;

-- =========================================================================
-- 1. Dynamic Tables (자동 새로고침)
-- =========================================================================

-- 1-1. 퍼널 라이브 테이블: 소스 변경 시 1시간 내 자동 갱신
CREATE OR REPLACE DYNAMIC TABLE ANALYTICS.DT_FUNNEL_LIVE
    TARGET_LAG = '1 hour'
    WAREHOUSE = COMPUTE_WH
AS
SELECT
    f.YEAR_MONTH,
    f.MAIN_CATEGORY_NAME AS CATEGORY,
    f.TOTAL_COUNT,
    f.PAYEND_COUNT,
    f.OVERALL_CVR,
    -- MoM 변화량
    f.OVERALL_CVR - LAG(f.OVERALL_CVR) OVER (
        PARTITION BY f.MAIN_CATEGORY_NAME ORDER BY f.YEAR_MONTH
    ) AS CVR_MOM_CHANGE,
    -- 3개월 이동 평균
    AVG(f.OVERALL_CVR) OVER (
        PARTITION BY f.MAIN_CATEGORY_NAME
        ORDER BY f.YEAR_MONTH
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS CVR_MA3
FROM STAGING.STG_FUNNEL f
WHERE f.YEAR_MONTH <= CURRENT_DATE()
  AND f.MAIN_CATEGORY_NAME IN ('인터넷', '렌탈', '모바일', '알뜰 요금제', '유심만');

-- 1-2. 채널 성과 라이브 테이블: 소스 변경 시 1시간 내 자동 갱신
CREATE OR REPLACE DYNAMIC TABLE ANALYTICS.DT_CHANNEL_LIVE
    TARGET_LAG = '1 hour'
    WAREHOUSE = COMPUTE_WH
AS
SELECT
    YEAR_MONTH,
    CATEGORY,
    CHANNEL,
    SUM(CONTRACT_COUNT)   AS CONTRACT_COUNT,
    AVG(PAYEND_CVR)       AS AVG_PAYEND_CVR,
    AVG(AVG_NET_SALES)    AS AVG_NET_SALES,
    SUM(TOTAL_NET_SALES)  AS TOTAL_NET_SALES
FROM STAGING.STG_CHANNEL
WHERE YEAR_MONTH <= CURRENT_DATE()
GROUP BY YEAR_MONTH, CATEGORY, CHANNEL;

-- =========================================================================
-- 2. Stored Procedure: 데이터 품질 검증 (Task에서 호출)
-- =========================================================================
-- Snowflake Task는 단일 SQL 문만 지원하므로,
-- 다중 INSERT를 프로시저로 래핑하여 Task에서 CALL 합니다.

CREATE OR REPLACE PROCEDURE ANALYTICS.SP_DAILY_QUALITY_CHECK()
    RETURNS VARCHAR
    LANGUAGE SQL
    EXECUTE AS CALLER
AS
BEGIN
    -- 기존 결과 삭제
    DELETE FROM MART.DATA_QUALITY_RESULTS;

    -- NULL 검사: STG_FUNNEL.OVERALL_CVR
    INSERT INTO MART.DATA_QUALITY_RESULTS
        (TABLE_NAME, COLUMN_NAME, CHECK_TYPE,
         VIOLATION_COUNT, TOTAL_COUNT, VIOLATION_PCT,
         QUALITY_STATUS, DETAIL)
    SELECT 'STG_FUNNEL', 'OVERALL_CVR', 'NULL_CHECK',
        COUNT_IF(OVERALL_CVR IS NULL), COUNT(*),
        ROUND(COUNT_IF(OVERALL_CVR IS NULL) * 100.0 / NULLIF(COUNT(*), 0), 2),
        CASE
            WHEN COUNT_IF(OVERALL_CVR IS NULL) = 0 THEN 'PASS'
            WHEN COUNT_IF(OVERALL_CVR IS NULL) * 100.0 / COUNT(*) < 10 THEN 'WARNING'
            ELSE 'CRITICAL'
        END,
        'CVR NULL 검사'
    FROM STAGING.STG_FUNNEL;

    -- NULL 검사: STG_CHANNEL.AVG_NET_SALES
    INSERT INTO MART.DATA_QUALITY_RESULTS
        (TABLE_NAME, COLUMN_NAME, CHECK_TYPE,
         VIOLATION_COUNT, TOTAL_COUNT, VIOLATION_PCT,
         QUALITY_STATUS, DETAIL)
    SELECT 'STG_CHANNEL', 'AVG_NET_SALES', 'NULL_CHECK',
        COUNT_IF(AVG_NET_SALES IS NULL), COUNT(*),
        ROUND(COUNT_IF(AVG_NET_SALES IS NULL) * 100.0 / NULLIF(COUNT(*), 0), 2),
        CASE
            WHEN COUNT_IF(AVG_NET_SALES IS NULL) = 0 THEN 'PASS'
            WHEN COUNT_IF(AVG_NET_SALES IS NULL) * 100.0 / COUNT(*) < 10 THEN 'WARNING'
            ELSE 'CRITICAL'
        END,
        '평균매출 NULL 검사'
    FROM STAGING.STG_CHANNEL;

    -- CVR 범위 검사 (0~100%)
    INSERT INTO MART.DATA_QUALITY_RESULTS
        (TABLE_NAME, COLUMN_NAME, CHECK_TYPE,
         VIOLATION_COUNT, TOTAL_COUNT, VIOLATION_PCT,
         QUALITY_STATUS, DETAIL)
    SELECT 'STG_FUNNEL', 'OVERALL_CVR', 'RANGE_CHECK',
        COUNT_IF(OVERALL_CVR < 0 OR OVERALL_CVR > 100), COUNT(*),
        ROUND(COUNT_IF(OVERALL_CVR < 0 OR OVERALL_CVR > 100) * 100.0 / NULLIF(COUNT(*), 0), 2),
        CASE
            WHEN COUNT_IF(OVERALL_CVR < 0 OR OVERALL_CVR > 100) = 0 THEN 'PASS'
            ELSE 'CRITICAL'
        END,
        'CVR 0~100% 범위 검사'
    FROM STAGING.STG_FUNNEL;

    -- 행 수 검사: STG_FUNNEL
    INSERT INTO MART.DATA_QUALITY_RESULTS
        (TABLE_NAME, COLUMN_NAME, CHECK_TYPE,
         VIOLATION_COUNT, TOTAL_COUNT, VIOLATION_PCT,
         QUALITY_STATUS, DETAIL)
    SELECT 'STG_FUNNEL', '*', 'ROW_COUNT', 0, COUNT(*), 0,
        CASE WHEN COUNT(*) > 0 THEN 'PASS' ELSE 'CRITICAL' END,
        '테이블 행 수: ' || COUNT(*) || '행'
    FROM STAGING.STG_FUNNEL;

    -- 행 수 검사: STG_CHANNEL
    INSERT INTO MART.DATA_QUALITY_RESULTS
        (TABLE_NAME, COLUMN_NAME, CHECK_TYPE,
         VIOLATION_COUNT, TOTAL_COUNT, VIOLATION_PCT,
         QUALITY_STATUS, DETAIL)
    SELECT 'STG_CHANNEL', '*', 'ROW_COUNT', 0, COUNT(*), 0,
        CASE WHEN COUNT(*) > 0 THEN 'PASS' ELSE 'CRITICAL' END,
        '테이블 행 수: ' || COUNT(*) || '행'
    FROM STAGING.STG_CHANNEL;

    -- 행 수 검사: STG_REGIONAL
    INSERT INTO MART.DATA_QUALITY_RESULTS
        (TABLE_NAME, COLUMN_NAME, CHECK_TYPE,
         VIOLATION_COUNT, TOTAL_COUNT, VIOLATION_PCT,
         QUALITY_STATUS, DETAIL)
    SELECT 'STG_REGIONAL', '*', 'ROW_COUNT', 0, COUNT(*), 0,
        CASE WHEN COUNT(*) > 0 THEN 'PASS' ELSE 'CRITICAL' END,
        '테이블 행 수: ' || COUNT(*) || '행'
    FROM STAGING.STG_REGIONAL;

    -- 미래 날짜 검사: STG_FUNNEL
    INSERT INTO MART.DATA_QUALITY_RESULTS
        (TABLE_NAME, COLUMN_NAME, CHECK_TYPE,
         VIOLATION_COUNT, TOTAL_COUNT, VIOLATION_PCT,
         QUALITY_STATUS, DETAIL)
    SELECT 'STG_FUNNEL', 'YEAR_MONTH', 'FUTURE_DATE',
        COUNT_IF(YEAR_MONTH > CURRENT_DATE()), COUNT(*), 0,
        CASE WHEN COUNT_IF(YEAR_MONTH > CURRENT_DATE()) = 0 THEN 'PASS' ELSE 'CRITICAL' END,
        '미래 날짜 레코드 검사'
    FROM STAGING.STG_FUNNEL;

    RETURN 'Quality check completed at ' || CURRENT_TIMESTAMP()::VARCHAR;
END;

-- =========================================================================
-- 3. Scheduled Task: 매일 오전 6시(KST) 품질 검증 실행
-- =========================================================================
CREATE OR REPLACE TASK ANALYTICS.TASK_DAILY_QUALITY_CHECK
    WAREHOUSE = COMPUTE_WH
    SCHEDULE = 'USING CRON 0 6 * * * Asia/Seoul'
    COMMENT = '매일 06:00 KST에 데이터 품질 검증 프로시저 실행'
AS
    CALL ANALYTICS.SP_DAILY_QUALITY_CHECK();

-- Task 활성화 (생성 후 SUSPENDED 상태이므로 RESUME 필요)
ALTER TASK ANALYTICS.TASK_DAILY_QUALITY_CHECK RESUME;

-- =========================================================================
-- 4. Alert: 품질 CRITICAL 감지 시 로그 기록
-- =========================================================================
-- SYSTEM$SEND_EMAIL은 별도 notification integration 설정이 필요하므로
-- SYSTEM$LOG_INFO 로 감사 로그에 기록합니다.
CREATE OR REPLACE ALERT ANALYTICS.ALERT_DATA_QUALITY
    WAREHOUSE = COMPUTE_WH
    SCHEDULE = 'USING CRON 0 7 * * * Asia/Seoul'
    COMMENT = '매일 07:00 KST에 CRITICAL 품질 이슈 탐지 후 알림'
    IF (EXISTS (
        SELECT 1
        FROM MART.DATA_QUALITY_RESULTS
        WHERE QUALITY_STATUS = 'CRITICAL'
    ))
    THEN
        CALL SYSTEM$LOG_INFO(
            'ALERT_DATA_QUALITY: Critical data quality violation detected in TELECOM_DB. '
            || 'Check MART.DATA_QUALITY_RESULTS for details.'
        );

-- Alert 활성화
ALTER ALERT ANALYTICS.ALERT_DATA_QUALITY RESUME;
