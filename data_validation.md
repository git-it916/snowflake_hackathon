# Data Validation Report

TELECOM_DB 전체 데이터 감사 결과 (2026-04-02)

---

## 1. FUNNEL_STAGE_DROP - DROP_RATE 완전 깨짐

### 문제

SUBSCRIPTION 단계의 DROP_RATE가 비정상:
- MIN = -3,473.0
- 250개 중 99개가 음수
- AVG = -53.67

### 원인

SUBSCRIPTION_COUNT > CONSULT_REQUEST_COUNT인 경우가 빈번함.
예: 상담요청 226 -> 가입신청 771 (상담 없이 바로 가입하는 경로 존재)

DROP_RATE = (PREV - CURR) / PREV 공식이 적용되어 있으나,
이 데이터에서는 퍼널이 순차적이지 않음 (직접 유입 경로 존재).

### 실제 데이터 예시 (인터넷, 2026-04)

| 단계 | 건수 | DROP_RATE (DB) | 실제 이탈률 (건수 기반) |
|------|------|---------------|---------------------|
| CONSULT_REQUEST | 8,002 | NULL | - |
| SUBSCRIPTION | 7,066 | 0.117 | 11.7% |
| REGISTEND | 3,819 | 0.460 | 45.9% |
| OPEN | 283 | 0.926 | 92.6% (최대 병목) |
| PAYEND | 210 | 0.258 | 25.8% |

### 수정 필요

- DROP_RATE 컬럼 사용 금지 (음수, 비정상값 포함)
- 건수 기반 직접 계산: (PREV_COUNT - CURR_COUNT) / PREV_COUNT, clip(0, 1)
- insight_generator.py의 generate_funnel_insights() 수정 완료 (건수 기반 계산)
- SUBSCRIPTION > CONSULT_REQUEST인 경우 이탈률 0으로 처리

---

## 2. CVR 값이 % 단위 (0~100)

### 문제

```
STG_FUNNEL:
  CVR_CONSULT_REQUEST = 49.94 (%)
  CVR_SUBSCRIPTION = 100.00 (%)
  OVERALL_CVR = 32.03 (%)

STG_CHANNEL:
  PAYEND_CVR = 18.79 (%)
  OPEN_CVR = 47.82 (%)
```

**이미 % 단위 (0~100)**. 코드에서 `* 100` 하면 4,994%가 됨.

### 영향 범위

- analysis/insight_generator.py - 전환율 표시
- analysis/channel_analysis.py - 채널 효율 계산
- analysis/funnel_analysis.py - 퍼널 전환율
- ml/conversion_model.py - XGBoost 타겟 변수
- 모든 차트에서 Y축 단위

### 수정 필요

- CVR 값은 그대로 사용 (이미 %)
- `* 100` 하는 곳 모두 제거
- 차트 Y축 라벨: "전환율 (%)" 확인

---

## 3. CHANNEL_EFFICIENCY - 음수 매출

### 문제

```
MIN_SALES = -720,000
MAX_SALES = 2,158,750
```

AVG_NET_SALES에 음수값 존재. 환불/취소 건이 포함된 것으로 추정.

### 영향

- 채널 버블 차트에서 버블 크기가 음수 -> 비정상 렌더링
- 채널 효율 점수 계산 왜곡

### 수정 필요

- 버블 차트: `AVG_NET_SALES.clip(lower=0)` 처리
- 효율 점수: 음수 매출 채널은 별도 플래그 표시

---

## 4. FORECAST_OUTPUT - 음수 예측값

### 문제

```
FORECAST = -14.12
LOWER = -49.78
```

전환율(CVR)이 음수가 될 수 없는데 Cortex FORECAST가 음수 예측.

### 원인

시계열 데이터의 변동성이 크고, 선형 추세 외삽 시 음수 가능.

### 수정 필요

- `FORECAST.clip(lower=0)` 처리
- `LOWER.clip(lower=0)` 처리
- 차트에서 Y축 최소값 0 고정

---

## 5. 최신월(2026-04) 데이터 미완성

### 문제

```
인터넷 2026-04: 상담 8,002 / 납입 210
인터넷 2026-03: 상담 139,642 / 납입 71,746
```

2026-04는 월중 데이터 (수집 진행 중)로 건수가 1/10 수준.
이 데이터를 포함하면 모든 KPI, 전환율, 추이 차트가 왜곡됨.

### 영향

- 랜딩 KPI: 전환율 급락으로 표시
- 퍼널 Sankey: 최신월 기준이라 건수가 매우 적음
- 추이 차트: 마지막 달에 급락 패턴

### 수정 필요

- 최신 불완전월 제외: `WHERE YEAR_MONTH < (SELECT MAX(YEAR_MONTH) FROM ...)`
- 또는 현재 월 제외: `WHERE YEAR_MONTH < DATE_TRUNC('month', CURRENT_DATE())`
- `_drop_incomplete_month()` 함수가 있으나 모든 곳에서 적용되는지 확인 필요

---

## 6. SUBSCRIPTION > CONSULT_REQUEST 현상

### 문제

여러 카테고리/월에서 가입신청 건수가 상담요청 건수보다 많음.

```
렌탈 2025-10: 상담 17,938 / 가입 24,178
인터넷 2025-12: 상담 60,352 / 가입 98,317
```

### 원인

상담 없이 온라인에서 직접 가입하는 경로 존재.
퍼널이 선형(순차적)이 아니라 다중 진입점이 있는 구조.

### 영향

- Sankey 다이어그램: 상담->가입 흐름이 "증가"로 보임 (역방향 흐름)
- DROP_RATE 음수
- "이탈률" 개념 자체가 성립하지 않는 구간

### 수정 필요

- Sankey에서 "이탈" 대신 "직접 유입" 노드 추가 고려
- 또는 CONSULT_REQUEST를 퍼널 시작점에서 제외하고 SUBSCRIPTION부터 시작
- CVR_CONSULT_REQUEST 대신 OVERALL_CVR(전체 대비 최종 전환) 사용 권장

---

## 7. DT_KPI - 단일 행

### 문제

```
REPORT_MONTH: 2026-04-01 (미완성월)
TOTAL_CONTRACTS: 22,059
AVG_OVERALL_CVR: 4.79
TOP_CHANNEL: 인바운드
TOP_GROWTH_CITY: 용인시
```

KPI가 미완성 월(2026-04) 기준으로 산출됨.
AVG_OVERALL_CVR = 4.79%는 비정상적으로 낮음 (정상: 25~35%).

### 수정 필요

- KPI 산출 시 완성된 최신월(2026-03) 기준으로 변경
- 또는 최근 3개월 평균 사용

---

## 8. Marketplace 원본 - 모든 뷰가 None rows

### 문제

```
V01_MONTHLY_REGIONAL_CONTRACT_STATS (None rows)
V03_CONTRACT_FUNNEL_CONVERSION (None rows)
V04_CHANNEL_CONTRACT_PERFORMANCE (None rows)
...
```

Marketplace 뷰들이 row count = None. 뷰이기 때문에 row count가 표시 안 되는 것으로 추정.
실제 데이터는 STAGING 테이블로 이미 복사되어 있으므로 문제 없음.

---

## 수정 우선순위

| 순위 | 이슈 | 영향도 | 수정 위치 |
|------|------|--------|----------|
| 1 | 최신 미완성월 제외 | 모든 KPI/차트 왜곡 | sql/01_staging.sql, data/snowflake_client.py |
| 2 | DROP_RATE 대신 건수 기반 계산 | Alert 메시지 오류 | analysis/insight_generator.py (수정 완료) |
| 3 | CVR 이미 % 단위 확인 | 수치 x100 오류 | 전체 코드 검색 |
| 4 | FORECAST 음수 clip | 차트 비정상 | data/snowflake_client.py, 차트 코드 |
| 5 | 음수 매출 처리 | 버블 차트 깨짐 | pages/1_진단.py |
| 6 | SUBSCRIPTION > CONSULT 처리 | Sankey 비정상 | pages/1_진단.py |
| 7 | KPI 산출 월 변경 | 랜딩 KPI 왜곡 | sql/03_mart.sql |
