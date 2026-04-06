-- 00_setup.sql
-- Telecom Funnel Intelligence DB 초기화
-- Pipeline: Marketplace → STAGING → ANALYTICS → MART

CREATE DATABASE IF NOT EXISTS TELECOM_DB
    COMMENT = '텔레콤 가입 퍼널 AI 인텔리전스 — Snowflake Hackathon 2026';

-- Schema 계층: STAGING(정제) → ANALYTICS(분석) → MART(대시보드)
CREATE SCHEMA IF NOT EXISTS TELECOM_DB.STAGING
    COMMENT = '마켓플레이스 원본 정제 (CTAS, NULL 보정, CVR 클램핑)';
CREATE SCHEMA IF NOT EXISTS TELECOM_DB.ANALYTICS
    COMMENT = '분석 테이블 (퍼널 병목, 채널 효율, 지역 수요, ML 피처)';
CREATE SCHEMA IF NOT EXISTS TELECOM_DB.MART
    COMMENT = '대시보드 뷰 + KPI + 예측 + AI 인사이트';

CREATE WAREHOUSE IF NOT EXISTS COMPUTE_WH
    WITH WAREHOUSE_SIZE = 'X-SMALL'
    AUTO_SUSPEND = 60
    AUTO_RESUME = TRUE
    COMMENT = '기본 컴퓨트 웨어하우스 (Auto-suspend 60s)';

USE DATABASE TELECOM_DB;
USE WAREHOUSE COMPUTE_WH;

-- Cortex Analyst 시맨틱 모델 스테이지
CREATE OR REPLACE STAGE TELECOM_DB.PUBLIC.CORTEX_STAGE
    DIRECTORY = (ENABLE = TRUE)
    COMMENT = 'Cortex Analyst YAML 시맨틱 모델 저장소';

-- Stored Procedure 스테이지
CREATE OR REPLACE STAGE TELECOM_DB.ANALYTICS.SPROC_STAGE
    COMMENT = '분석 저장 프로시저 코드 스테이지';

-- ---------------------------------------------------------------------------
-- 접근 제어 (RBAC)
-- ---------------------------------------------------------------------------
-- GRANT USAGE ON DATABASE TELECOM_DB TO ROLE SYSADMIN;
-- GRANT USAGE ON ALL SCHEMAS IN DATABASE TELECOM_DB TO ROLE SYSADMIN;
-- GRANT SELECT ON ALL TABLES IN SCHEMA TELECOM_DB.MART TO ROLE PUBLIC;
