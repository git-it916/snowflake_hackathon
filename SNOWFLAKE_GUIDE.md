# Snowflake 완전 초보 가이드

> 버튼 하나하나 따라하면 프로젝트 전체가 실행됩니다.
> 예상 소요 시간: 약 40분

---

## STEP 0: Snowflake 계정 로그인

1. 브라우저에서 Snowflake 계정 URL 접속
   - 형식: `https://xxxxxxx.snowflakecomputing.com`
   - `.env` 파일의 `SF_ACCOUNT` 값이 계정 ID입니다
2. **Username** 입력 (`.env`의 `SF_USER`)
3. **Password** 입력 (`.env`의 `SF_PASSWORD`)
4. **Sign in** 클릭

> 로그인하면 **Snowsight** 라는 웹 UI가 나옵니다. 이게 Snowflake의 전부입니다.

---

## STEP 1: Marketplace 데이터 가져오기

이 프로젝트의 원본 데이터를 Snowflake에 추가해야 합니다.

1. 왼쪽 사이드바에서 **Data Products** 클릭
2. **Marketplace** 클릭
3. 검색창에 `South Korea Telecom Subscription` 입력
4. **South Korea Telecom Subscription Analytics** 카드 클릭
5. 오른쪽 상단 **Get** 버튼 클릭
6. 팝업에서:
   - Database name: 그대로 둡니다 (매우 긴 이름, 바꾸지 마세요)
   - Roles: `ACCOUNTADMIN` 선택
7. **Get** 클릭

> 약 10초 후 "Successfully created database" 메시지가 나옵니다.
> 이제 왼쪽 **Data** > **Databases** 에서 긴 이름의 DB가 보입니다.

---

## STEP 2: SQL 워크시트 만들기

1. 왼쪽 사이드바에서 **Projects** 클릭
2. **Worksheets** 클릭
3. 오른쪽 상단 파란색 **+ 버튼** 클릭
4. **SQL Worksheet** 선택

> 빈 워크시트가 열립니다. 여기에 SQL을 붙여넣고 실행합니다.

---

## STEP 3: 데이터베이스 + 스키마 생성 (00_setup.sql)

1. 프로젝트의 `sql/00_setup.sql` 파일을 메모장이나 VS Code로 엽니다
2. **전체 내용을 복사** (Ctrl+A → Ctrl+C)
3. Snowflake 워크시트에 **붙여넣기** (Ctrl+V)
4. 오른쪽 상단 파란색 **▶ Run** 버튼 클릭
   - 또는 키보드 **Ctrl+Enter**

> 아래에 결과가 나옵니다. "Statement executed successfully" 가 여러 번 보이면 성공입니다.
> 에러가 나면 빨간 글씨로 표시됩니다.

**확인 방법:**
- 왼쪽 **Data** > **Databases** 클릭
- `TELECOM_DB` 가 보이면 성공
- 클릭해서 열면 `STAGING`, `ANALYTICS`, `MART` 3개 스키마가 보입니다

---

## STEP 4: 스테이징 테이블 생성 (01_staging.sql)

1. 워크시트 내용을 **전부 지우고** (Ctrl+A → Delete)
2. `sql/01_staging.sql` 내용을 복사 → 붙여넣기
3. **▶ Run** 클릭

> 4개 테이블이 생성됩니다: STG_FUNNEL, STG_CHANNEL, STG_REGIONAL, STG_MARKETING
> 약 10~30초 소요됩니다.

**확인 방법:**
- 왼쪽 **Data** > **Databases** > `TELECOM_DB` > `STAGING` > **Tables**
- 4개 테이블이 보이면 성공
- 아무 테이블 클릭 → **Data Preview** 탭에서 실제 데이터를 볼 수 있습니다

---

## STEP 5: 분석 테이블 생성 (02_analytics.sql)

1. 워크시트 내용 전부 지우기
2. `sql/02_analytics.sql` 복사 → 붙여넣기
3. **▶ Run** 클릭

---

## STEP 6: 마트 뷰 생성 (03_mart.sql)

1. 워크시트 내용 전부 지우기
2. `sql/03_mart.sql` 복사 → 붙여넣기
3. **▶ Run** 클릭

> 여기까지 하면 기본 파이프라인(Staging → Analytics → Mart)이 완성됩니다.

---

## STEP 7: Cortex FORECAST + ANOMALY (04_cortex_ml.sql)

1. 워크시트 내용 전부 지우기
2. `sql/04_cortex_ml.sql` 복사 → 붙여넣기
3. **▶ Run** 클릭

> FORECAST 모델 학습에 1~3분 소요될 수 있습니다.
> "FORECAST_OUTPUT", "ANOMALY_OUTPUT" 테이블이 MART에 생성됩니다.

**라이브 데모용 확인:**
```sql
SELECT * FROM TELECOM_DB.MART.FORECAST_OUTPUT LIMIT 10;
```
이걸 워크시트에 붙여넣고 실행하면 예측 결과가 보입니다.

---

## STEP 8: Feature Store (06_feature_store.sql)

1. 워크시트 내용 전부 지우기
2. `sql/06_feature_store.sql` 복사 → 붙여넣기
3. **▶ Run** 클릭

---

## STEP 9: 데이터 품질 검증 (07_data_quality.sql)

1. 워크시트 내용 전부 지우기
2. `sql/07_data_quality.sql` 복사 → 붙여넣기
3. **▶ Run** 클릭

**확인:**
```sql
SELECT * FROM TELECOM_DB.MART.DATA_QUALITY_RESULTS ORDER BY CHECK_TIME DESC;
```
12건의 품질 검사 결과가 PASS/WARNING/CRITICAL로 표시됩니다.

---

## STEP 10: 데이터 리니지 (08_lineage.sql)

1. 워크시트 내용 전부 지우기
2. `sql/08_lineage.sql` 복사 → 붙여넣기
3. **▶ Run** 클릭

> 주의: `SNOWFLAKE.ACCOUNT_USAGE` 뷰는 ACCOUNTADMIN 권한이 필요합니다.
> 권한 에러가 나면:
> - 워크시트 상단에서 Role을 **ACCOUNTADMIN**으로 변경
> - 방법: 워크시트 왼쪽 상단에 현재 역할이 표시됩니다 → 클릭 → ACCOUNTADMIN 선택

---

## STEP 11: Cortex Analyst 시맨틱 모델 (09_cortex_analyst.sql)

이건 2단계입니다:

### 11-1: 시맨틱 모델 YAML 파일 업로드

1. 워크시트에 아래 SQL 실행:
```sql
USE DATABASE TELECOM_DB;
CREATE STAGE IF NOT EXISTS PUBLIC.CORTEX_STAGE DIRECTORY = (ENABLE = TRUE);
```

2. 왼쪽 **Data** > **Databases** > `TELECOM_DB` > `PUBLIC` > **Stages** 클릭
3. `CORTEX_STAGE` 클릭
4. 오른쪽 상단 **+ Files** 버튼 클릭
5. 프로젝트의 `semantic_model/telecom_semantic.yaml` 파일을 드래그하거나 선택
6. **Upload** 클릭

### 11-2: SQL 실행

1. 워크시트에서 `sql/09_cortex_analyst.sql` 실행

---

## STEP 12: Dynamic Tables + Tasks + Alerts (10_dynamic_tables.sql)

1. 워크시트 내용 전부 지우기
2. `sql/10_dynamic_tables.sql` 복사 → 붙여넣기
3. **▶ Run** 클릭

> Dynamic Tables는 생성 후 자동으로 데이터를 갱신합니다.
> Tasks는 RESUME 해야 작동합니다 (SQL에 포함되어 있음).

**확인:**
- 왼쪽 **Data** > **Databases** > `TELECOM_DB` > `ANALYTICS` 에서 Dynamic Tables 확인
- 왼쪽 **Monitoring** > **Task History** 에서 Task 실행 이력 확인

---

## STEP 13: Cortex AI Functions (11_cortex_ai_functions.sql)

1. 워크시트 내용 전부 지우기
2. `sql/11_cortex_ai_functions.sql` 복사 → 붙여넣기
3. **▶ Run** 클릭

> SENTIMENT, CLASSIFY_TEXT, SUMMARIZE, TRANSLATE 결과가 테이블로 저장됩니다.
> 약 1~2분 소요.

**발표 때 라이브 데모용 — 이것만 따로 실행하면 임팩트 큼:**
```sql
-- 감성 분석 라이브 데모
SELECT SNOWFLAKE.CORTEX.SENTIMENT('서울 지역: 계약 5000건, 전환율 32%');

-- 채널 분류 라이브 데모
SELECT SNOWFLAKE.CORTEX.CLASSIFY_TEXT(
    '네이버: CVR 45.2%, 계약 1200건',
    ['고효율_채널', '중간_채널', '저효율_채널']
);

-- 한영 번역 라이브 데모
SELECT SNOWFLAKE.CORTEX.TRANSLATE('서울 지역 계약 5000건', 'ko', 'en');

-- 요약 라이브 데모
SELECT SNOWFLAKE.CORTEX.SUMMARIZE('채널 네이버의 인터넷 카테고리 성과: 전환율 45.2%, 계약 1200건, 평균매출 35000원. 전월 대비 12% 성장.');
```

---

## STEP 14: Streamlit 대시보드 배포

### 방법 A: 로컬에서 실행 (간단)

1. 터미널(명령 프롬프트)을 엽니다
2. 프로젝트 폴더로 이동:
```bash
cd C:\Users\sam_s\OneDrive\문서\VS_sidePJ\snowflake_hackathon
```
3. 실행:
```bash
streamlit run app.py
```
4. 브라우저에서 `http://localhost:8501` 이 자동으로 열립니다

### 방법 B: Snowflake 안에서 실행 (SiS)

1. 터미널에서:
```bash
cd C:\Users\sam_s\OneDrive\문서\VS_sidePJ\snowflake_hackathon
python deploy_sis.py
```
2. Snowsight로 돌아가서
3. 왼쪽 사이드바 **Projects** > **Streamlit** 클릭
4. `TELECOM_FUNNEL_INTELLIGENCE` 앱이 보입니다
5. 클릭하면 Snowflake 안에서 대시보드가 열립니다

> 발표 때는 **방법 A (로컬)**이 더 빠르고 안정적입니다.
> "Snowflake 안에서도 실행 가능하다"는 걸 보여주려면 방법 B도 미리 배포해두세요.

---

## STEP 15: Cortex COMPLETE (AI Agent) 테스트

대시보드의 AI 전략 페이지에서 자동으로 실행되지만, 워크시트에서도 직접 테스트 가능합니다:

```sql
SELECT SNOWFLAKE.CORTEX.COMPLETE(
    'llama3.1-405b',
    [
        {'role': 'system', 'content': '당신은 한국 통신사 데이터 분석가입니다. 한국어로 답변하세요.'},
        {'role': 'user', 'content': '인터넷 카테고리의 가입 퍼널에서 개통 단계 이탈이 높은 이유는 무엇일까요?'}
    ],
    {'temperature': 0.3, 'max_tokens': 512}
) AS RESPONSE;
```

> 약 5~10초 후 AI 응답이 나옵니다. 발표 때 이걸 라이브로 보여주면 인상적입니다.

---

## 전체 실행 순서 요약

```
00_setup.sql          ← DB 생성 (10초)
01_staging.sql        ← 원본 데이터 정제 (30초)
02_analytics.sql      ← 분석 테이블 (30초)
03_mart.sql           ← 대시보드 뷰 (30초)
04_cortex_ml.sql      ← FORECAST + ANOMALY (1~3분)
06_feature_store.sql  ← ML 피처 (30초)
07_data_quality.sql   ← 품질 검증 (10초)
08_lineage.sql        ← 리니지 (10초)
09_cortex_analyst.sql ← 시맨틱 모델 (30초, YAML 업로드 필요)
10_dynamic_tables.sql ← 자동화 (30초)
11_cortex_ai_functions.sql ← AI 함수 (1~2분)
```

**총 약 10~15분**이면 전체 파이프라인이 완성됩니다.

---

## 자주 발생하는 에러와 해결법

### "Object does not exist"
- 원인: 이전 단계의 SQL을 안 실행했음
- 해결: 위 순서대로 처음부터 다시 실행

### "Insufficient privileges"
- 원인: 역할(Role) 권한 부족
- 해결: 워크시트 왼쪽 상단 Role → **ACCOUNTADMIN** 선택

### "Warehouse is suspended"
- 원인: 웨어하우스가 자동 중지됨
- 해결: 워크시트 상단 Warehouse 선택 → **COMPUTE_WH** 선택하면 자동 재개

### "Function CORTEX.COMPLETE not found"
- 원인: 리전에서 Cortex 미지원
- 해결: 계정이 AWS US East, US West, EU Central 중 하나여야 합니다.
  Cortex 지원 리전: https://docs.snowflake.com/en/user-guide/snowflake-cortex/llm-functions#availability

### "Credit quota exceeded"
- 원인: 무료 크레딧 소진
- 해결: Cortex 함수는 크레딧을 소모합니다. FORECAST, COMPLETE가 가장 비쌈.
  데모 전에 크레딧 잔량 확인: **Admin** > **Cost Management** > **Usage**

---

## 발표 전 체크리스트

- [ ] 위 SQL 00~11번 전부 실행 완료
- [ ] `streamlit run app.py` 로 대시보드 정상 표시 확인
- [ ] 라이브 데모용 SQL 4개 (SENTIMENT, CLASSIFY_TEXT, TRANSLATE, COMPLETE) 워크시트에 미리 준비
- [ ] 브라우저 탭 2개 준비: (1) Snowsight 워크시트 (2) Streamlit 대시보드
- [ ] Snowsight에서 Role이 ACCOUNTADMIN인지 확인
- [ ] Warehouse가 COMPUTE_WH로 설정되어 있는지 확인
