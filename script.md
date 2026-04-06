# 발표 스크립트: Telecom Funnel Intelligence

> 총 10분 | 데모 포함 | Snowflake Hackathon 2026

---

## 1. 오프닝 — 문제 정의 (30초)

안녕하세요, 저희 프로젝트 **Telecom Funnel Intelligence**를 소개하겠습니다.

한국 통신사의 가입 퍼널은 5단계를 거칩니다.

**상담요청 → 가입신청 → 접수 → 개통 → 납입완료**

각 단계에서 다음 단계로 넘어가는 비율을 **CVR(Conversion Rate, 전환율)**이라고 합니다. 예를 들어 상담요청 100명 중 80명이 가입신청하면 CVR은 80%입니다. 최종 지표인 **OVERALL_CVR**은 맨 처음 상담요청 대비 최종 납입완료까지의 비율입니다.

이 과정에서 세 가지 질문이 생깁니다.

- **어디서** 고객이 빠지는가?
- **어떤 채널**이 효과적인가?
- **어디에** 마케팅을 집중해야 하는가?

38개 유입 채널, 200개 시군구, 5단계 퍼널 — 이 세 차원을 **데이터로 진단**하고, **수학적 모델로 정량화**하고, **AI Agent로 전략을 자동 생성**하는 것이 저희 프로젝트의 목표입니다.

데이터 소스는 **Snowflake Marketplace**의 `SOUTH_KOREA_TELECOM_SUBSCRIPTION_ANALYTICS`이고, V01부터 V07까지 실제 텔레콤 데이터 23,000행 이상을 사용합니다.

다만 Marketplace 원본 데이터에는 **품질 이슈가 상당히 많았습니다.** 각 SQL 파일에서 어떻게 처리했는지 아키텍처와 함께 설명하겠습니다.

---

## 2. 아키텍처 + 개발 워크플로우 (2분 30초)

저희는 Snowflake 기능을 **19개** 사용했습니다.

개발은 **로컬에서 테스트 → Snowflake에 배포**하는 방식으로 진행했습니다. SQL 파이프라인과 Cortex 함수는 **Snowflake 워크시트에서 직접 실행**하고, Streamlit 대시보드는 **로컬에서 개발·테스트**한 뒤 `deploy_sis.py`로 **Streamlit in Snowflake(SiS)에 배포**합니다. SiS 환경에서는 Snowflake 내부에서 데이터에 직접 접근하기 때문에 외부 연결 없이 모든 기능이 작동합니다.

크게 네 가지 레이어로 구성됩니다.

### 레이어 1: 데이터 파이프라인 — Snowflake SQL 워크시트에서 실행

Marketplace 원본 데이터를 **CTAS 파이프라인**으로 3단계 정제합니다. 원본 데이터 품질이 좋지 않아서 Staging 단계에서 상당한 전처리가 필요했습니다.

**`01_staging.sql`** — Snowflake 워크시트에서 실행하여 4개 Staging 테이블을 생성합니다:

- **STG_FUNNEL** (V03, 퍼널 전환율): CVR이 100%를 초과하는 행이 있었습니다. 최대 347,400%까지 나옵니다. `LEAST(COALESCE(CVR, 0), 100.0)`으로 NULL 보정 + 100% 상한 클램핑을 동시에 적용했습니다. CVR_SUBSCRIPTION은 22%가 NULL이었는데, `COALESCE`로 0 대체했습니다. 미래 날짜(2027-07 등)도 `WHERE YEAR_MONTH <= CURRENT_DATE()`로 제외합니다.

- **STG_CHANNEL** (V04, 채널 성과): 동일하게 PAYEND_CVR과 OPEN_CVR에 `LEAST(COALESCE(...), 100.0)` 클램핑을 적용하고, AVG_NET_SALES의 NULL도 0으로 보정합니다. 여기서 음수 매출(환불 건)이 존재하는데, 이건 대시보드 차트 렌더링 시 `clip(lower=0)`으로 처리합니다.

- **STG_REGIONAL** (V01+V05, 지역별 계약): 도명이 '전북특별자치도'와 '전북'으로 불일치해서 `REPLACE`로 통일했고, 빈 문자열 지역명은 `WHERE INSTALL_STATE != ''`로 제외합니다. V01(계약 통계)과 V05(신규 설치)를 **LEFT JOIN**으로 결합하여 번들/단독 설치 지표도 추가합니다.

- **STG_MARKETING** (V07, GA4 어트리뷰션): 마찬가지로 NULL 보정 + CVR 클램핑 + 미래 날짜 제외를 적용합니다.

추가로, **최신 월(현재 월)이 미완성 데이터**라는 문제가 있습니다. 예를 들어 4월 데이터가 3월의 1/10 수준밖에 안 됩니다. 이걸 포함하면 모든 KPI와 추이 차트가 급락으로 왜곡되기 때문에, **대시보드에서 최신 불완전월을 자동 제외**합니다.

또 한 가지, 퍼널이 선형이 아닌 경우가 있었습니다. **가입신청 건수가 상담요청보다 많은** 카테고리가 존재합니다 — 상담 없이 온라인에서 직접 가입하는 경로 때문입니다. 이 경우 원본의 DROP_RATE가 음수가 되는데, **`02_analytics.sql`**에서 건수 기반으로 이탈률을 재계산하고, 분석 코드에서 음수 이탈률은 0으로 처리합니다.

**`02_analytics.sql`** — Snowflake에서 실행. 전처리된 Staging 위에 분석 테이블 4개를 생성합니다. 퍼널 병목, 채널 효율, 지역 수요(Z-score), 채널 집중도(HHI) 계산이 여기서 수행됩니다.

**`03_mart.sql`** — Snowflake에서 실행. 대시보드 뷰 6개를 생성합니다. KPI, 퍼널 시계열, 채널 성과, 지역 히트맵 등입니다.

**`10_dynamic_tables.sql`** — Snowflake에서 실행. **Dynamic Tables**로 1시간 주기 자동 갱신을 설정하고, **Snowflake Tasks**로 매일 06시에 데이터 품질 검사를 자동 실행합니다.

**`07_data_quality.sql`** — Snowflake에서 실행. NULL 3건, CVR 범위 2건, 미래날짜 3건, 행수 4건, 중복 1건 — **총 13건**을 체크합니다. CRITICAL이 감지되면 **Snowflake Alerts**가 발동합니다.

**`08_lineage.sql`** — Snowflake에서 실행. `OBJECT_DEPENDENCIES` 시스템 뷰로 테이블 의존성을 자동 추적합니다.

### 레이어 2: ML — 로컬에서 학습, Snowflake에 모델 등록

ML 모델이 하는 일을 쉽게 말하면, **"다음 달에 이 채널의 전환율이 높을까, 보통일까, 낮을까?"**를 예측하는 겁니다.

학습 데이터는 이렇습니다. STG_CHANNEL의 7,900행 채널 성과 데이터에서 카테고리×월 단위로 **"과거 3개월 전환율 추이, 계약 건수 변화, 매출 트렌드, 채널 집중도"** 같은 30개 피처를 만듭니다. 그리고 "그 다음 달 전환율이 상위 33%면 HIGH, 하위 33%면 LOW, 중간이면 MEDIUM"이라는 정답 라벨을 붙입니다.

**`06_feature_store.sql`** — Snowflake에서 실행. 이 30개 피처를 SQL ���도우 함수(LAG, AVG OVER, STDDEV OVER 등)로 계산해서 **Feature Store** 테이블에 저장합니다. 예를 들어:
- `PAYEND_CVR_LAG1` = 지난달 전환율
- `PAYEND_CVR_MA3` = 최근 3개월 전환율 평균
- `CHANNEL_HHI` = 채널 집중도 (특정 채널에 너무 의존하는지)
- `FUNNEL_OVERALL_CVR` = 퍼널 전체 전환율

**`ml/conversion_model.py`** — 로컬에서 실행. Snowpark으로 Feature Store 데이터를 가져와서 **XGBoost**(트리 기반 분류 모델)로 학습합니다. "이 피처 조합이면 다음 달 전환율이 HIGH/MEDIUM/LOW 중 뭘까?"를 학습하는 겁니다. 학습 완료 후 **Model Registry**에 모델과 성능 지표(정확도, F1 점수 등)를 자동 등록합니다. Model Registry는 **Snowflake 서버에 모델을 저장**해서 버전 관리가 됩니다.

**`ml/explainer.py`** — 로컬에서 실행. **SHAP**이라는 해석 도구로 "왜 이 채널이 HIGH로 예측되었는가?"를 설명합니다. 예를 들어 "지난달 전환율이 높았고, 채널 집중도가 낮아서"처럼 예측 근거를 피처별로 보여줍니다.

### 레이어 3: Cortex AI — 전부 Snowflake 서버에서 실행

Cortex 함수는 **모두 Snowflake 서버 내부에서 실행**됩니다. 데이터가 외부로 나가지 않습니다.

**`04_cortex_ml.sql`** — Snowflake 워크시트에서 실행. **Cortex FORECAST**로 시도별 계약수 3개월 예측(95% 신뢰구간 포함), **Cortex ANOMALY**로 시도별 계약수 급변 자동 탐지. 예측 결과는 MART.FORECAST_OUTPUT, 이상 탐지는 MART.ANOMALY_OUTPUT에 저장됩니다.

**`11_cortex_ai_functions.sql`** — Snowflake 워크시트에서 실행. **SENTIMENT**로 지역 성과 감성 점수, **CLASSIFY_TEXT**로 채널 효율 등급(고효율/중간/저효율) 자동 분류, **SUMMARIZE**로 채널별 한 줄 요약, **TRANSLATE**로 한영 번역. 결과는 MART.CHANNEL_AI_INSIGHT, MART.REGIONAL_AI_INSIGHT에 저장됩니다.

**`agents/orchestrator.py`** — SiS 또는 로컬에서 실행. **Cortex COMPLETE**(llama3.1-405b)로 Multi-Agent 시스템을 구동합니다. 대시보드에서 사용자가 "전체 분석 실행"을 클릭하면, Cortex COMPLETE가 Snowflake 서버에서 LLM 추론을 수행하고 결과를 반환합니다.

**`09_cortex_analyst.sql`** + **`semantic_model/telecom_semantic.yaml`** — Snowflake에서 실행. 시맨틱 모델을 Stage에 업로드하면 **Cortex Analyst**가 자연어 질문을 SQL로 변환해줍니다.

### 레이어 4: 거버넌스 — Snowflake 내장 기능

**Data Quality** — Task로 매일 자동 실행, Alerts로 이상 감지 시 알림.
**Data Lineage** — Snowflake 시스템 뷰 기반 자동 추적.
**Cortex Analyst** — 비기술자도 자연어로 데이터 조회 가능.

### 대시보드 배포: 로컬 → SiS

**`app.py` + `pages/`** — 로컬에서 `streamlit run app.py`로 개발·테스트합니다. Snowpark으로 Snowflake 데이터를 가져와서 Plotly 차트로 시각화하고, 마르코프 체인·Monte Carlo 시뮬레이션 등은 Python으로 계산합니다.

개발 완료 후 **`deploy_sis.py`**를 실행하면 전체 코드가 **Streamlit in Snowflake(SiS)**에 배포됩니다. SiS 환경에서는 별도의 `.env` 설정 없이 Snowflake 내부 세션을 자동으로 사용하므로, 보안과 성능이 모두 개선됩니다.

---

## 3. CoCo 스킬 활용 (30초)

CoCo 스킬은 다섯 가지 영역을 커버했습니다.

첫째, **cortex-ai-functions** — Cortex 7종(COMPLETE, FORECAST, ANOMALY, SENTIMENT, CLASSIFY_TEXT, SUMMARIZE, TRANSLATE)을 모두 Snowflake 서버에서 실행합니다.

둘째, **machine-learning** — "다음 달 채널 전환율 예측" XGBoost 모델을 학습하고, Model Registry로 버전 관리하고, Feature Store에 30개 피처를 중앙 관리합니다. SHAP으로 "왜 이렇게 예측했는지"까지 해석합니다.

셋째, **data-quality** — SQL 기반 13건 품질 검증. Snowflake Task로 매일 자동 실행됩니다.

넷째, **lineage** — `OBJECT_DEPENDENCIES` 시스템 뷰로 Marketplace → Staging → Analytics → Mart 전체 흐름을 Snowflake에서 자동 추적합니다.

다섯째, **cost-intelligence** — Dynamic Tables의 `TARGET_LAG`을 1시간으로 설정해서, 불필요한 컴퓨트 비용을 절감합니다.

---

## 4. 데모 — 랜딩 페이지 (30초)

*(SiS 대시보드 화면 전환 — Snowsight > Projects > Streamlit > TELECOM_FUNNEL_INTELLIGENCE)*

지금 보시는 화면은 **Streamlit in Snowflake**에서 실행 중인 대시보드입니다. Snowflake 안에서 직접 돌아가고 있어서, 데이터 연결이나 인증 설정이 필요 없습니다.

상단에 **3개 KPI 카드**가 보입니다. 현재 퍼널 전환율, 최고 볼륨 채널, 최대 성장 지역을 한눈에 볼 수 있습니다.

경고 배너가 자동으로 표시됩니다 — 퍼널에서 이탈률이 가장 높은 구간을 감지해서 보여줍니다.

아래에는 **데이터 품질 모니터링**과 **파이프라인 리니지**가 있습니다. Snowflake Task가 매일 자동 실행한 13건 품질 검사 결과와, `OBJECT_DEPENDENCIES`로 추적한 테이블 의존성을 확인할 수 있습니다.

---

## 5. 데모 — 진단 페이지 (1분 30초)

진단 페이지로 이동하겠습니다.

왼쪽의 **Sankey 다이어그램**은 퍼널 5단계의 흐름을 보여줍니다. 데이터는 Snowflake의 STAGING.STG_FUNNEL에서 실시간으로 가져옵니다. 띠가 갑자기 얇아지는 구간이 병목이고, 빨간 띠는 이탈 고객의 흐름입니���.

오른쪽의 **버블 차트**는 ANALYTICS.CHANNEL_EFFICIENCY에서 38개 채널을 비교합니다. X축이 계약 건수, Y축이 전환율, 버블 크기가 매출입니다. 초록색은 성장 중, 빨간색은 쇠퇴 중입니다.

아래로 내려가면 **전환율 추이 차트**가 있습니다. 여기서 Snowflake에서 실행한 **Cortex ANOMALY**가 탐지한 이상치를 X 마커로 표시합니다. MART.ANOMALY_OUTPUT에 저장된 결과를 그대로 시각화하는 겁니다.

그 다음이 저희 프로젝트의 핵심 분석인 **흡수 마르코프 체인**입니다. 이건 Python으로 계산합니다.

퍼널 5단계와 이탈을 6x6 전이 행렬로 모델링하고, 기본 행렬의 역행렬 `N = (I - Q)^{-1}`, 흡수 확률 `B = N × R`을 계산합니다.

결과로 **Steady State 장기 최종 전환율**을 보여주고, 민감도 분석으로 "어떤 전이를 5%p 개선하면 최종 전환율이 얼마나 오르고, 월 몇 건이 추가 전환되는가"를 정량적으로 계산합니다.

예를 들어, 접수→개통 전이를 5%p 개선하면 최종 전환율이 +4.5%p 올라가고, 월 약 450건이 추가 전환됩니다. 이것이 ROI 1순위 개선 대상입니다.

오른쪽에는 **STL 시계열 분해**가 있습니다. Python의 statsmodels로 추세/계절성/잔차를 분리해서 마케팅 타이밍 최적화를 지원합니다.

---

## 6. 데모 — 기회 분석 페이지 (1분 30초)

기회 분석 페이지입니다.

상단에 **지역별 수요 점수** 바차트가 있습니다. `02_analytics.sql`에서 Z-score로 정규화한 REGIONAL_DEMAND_SCORE 테이블을 Snowflake에서 가져와 시각화합니다.

옆에는 **성장률 상위 도시 TOP 10**이 있습니다. 3개월 전 대비 계약이 가장 빠르게 증가하고 있는 도시들입니다.

아래로 가면 **Cortex FORECAST** 차트입니다. `04_cortex_ml.sql`에서 Snowflake 서버가 학습한 FORECAST 모델의 결과(MART.FORECAST_OUTPUT)를 시각화합니다.

```sql
-- 이 SQL이 Snowflake에서 실행되어 예측 모델이 생성됨
SNOWFLAKE.ML.FORECAST(
    FORECASTING_PERIODS => 3,
    CONFIG_OBJECT => {'prediction_interval': 0.95}
)
```

시도별 계약 건수를 3개월 예측하고, 95% 신뢰구간을 음영으로 표시합니다.

그 다음은 **Cortex ANOMALY 탐지 테이블**입니다. 역시 `04_cortex_ml.sql`에서 Snowflake가 탐지한 결과(MART.ANOMALY_OUTPUT)를 그대로 보여줍니다.

그리고 **마르코프 전이 확률 시뮬레이션**입니다. 슬라이더로 각 단계의 전이 확률을 조정하면, Python이 마르코프 체인을 즉시 재계산해서 최종 전환율 변화를 수학적으로 보여줍니다.

마지막으로 **Monte Carlo 500회 시뮬레이션**입니다. Python으로 각 전이 확률에 ±3%p 랜덤 변동을 500번 주어서 전환율의 불확실성 범위를 추정합니다.

---

## 7. 데모 — AI 전략 페이지 (1분 30초)

마지막 AI 전략 페이지입니다.

여기서 **Multi-Agent 오케스트레이션**이 작동합니다. 각 Agent가 **Snowflake Cortex COMPLETE(llama3.1-405b)**를 호출하는데, 이 LLM 추론은 전부 **Snowflake 서버 내부에서** 실행됩니다. 데이터가 외부로 나가지 않습니다.

"전체 분석 실행" 버튼을 누르면 3단계가 순차 실행됩니다.

**Phase 1** — 분석가 Agent가 Snowflake에서 퍼널, 채널, 지역, 마케팅 데이터를 조회하고, Cortex COMPLETE에 전달하여 핵심 발견사항을 추출합니다.

**Phase 2** — 전략가 Agent가 Phase 1의 결과를 받아서, XGBoost 모델의 "다음 달 전환율 예측"과 3개 시나리오(인바운드 집중, 온라인 전환, 채널 다각화) 시뮬레이션을 기반으로 채널 최적화 전략을 수립합니다.

**Phase 3** — 종합 Agent가 Cortex COMPLETE로 두 Agent의 결과를 합성하고, 경영진 요약 보고서를 생성합니다.

탭을 전환하면 데이터 분석 결과, 채널 전략, 경영진 요약을 각각 볼 수 있습니다.

오른쪽에는 **AI Q&A 챗**이 있습니다. 사용자의 자유 질문을 **Cortex COMPLETE가 Snowflake 안에서** 실시간으로 답변합니다.

분석 전에도 마르코프 체인 Steady State와 민감도 분석 결과는 즉시 표시되어, AI 없이도 데이터 기반 인사이트를 바로 확인할 수 있습니다.

---

## 8. 가장 인상적이었던 Snowflake 기능 (1분)

19개 기능 중에서 가장 인상적이었던 건 **Cortex COMPLETE + Multi-Agent 조합**이었습니다.

llama3.1-405b 모델을 **Snowflake 안에서 바로 호출**할 수 있다는 것이 큰 장점이었습니다. 외부 API 없이, 데이터가 Snowflake 밖으로 나가지 않으면서 LLM을 사용할 수 있습니다. 저희는 이걸로 3개의 전문가 Agent를 만들었고, 각 Agent가 **Snowflake 안의 데이터에 직접 접근**해서 분석하고, 결과를 LLM이 합성하는 구조입니다.

두 번째로 좋았던 건 **Cortex FORECAST**입니다. SQL 한 줄로 시도별 시계열 예측과 95% 신뢰구간을 자동 생성해줍니다. **별도의 모델 학습이나 인프라 관리 없이**, Snowflake 워크시트에서 바로 production-grade 예측이 가능한 점이 인상적이었습니다.

세 번째는 **Streamlit in Snowflake**입니다. 로컬에서 개발한 4페이지 대시보드를 `deploy_sis.py` 한 번으로 Snowflake 안에 배포할 수 있었습니다. SiS에서는 `.env` 인증 설정이 필요 없고, Snowflake 내부 세션을 자동으로 사용하기 때문에 **보안과 배포가 동시에 해결**됩니다.

---

## 9. 마무리 — 심사 기준 매핑 (30초)

마무리하겠습니다.

| 기준 | 저희 프로젝트 |
|------|-------------|
| **창의성 25%** | 마르코프 체인 퍼널 모델링, Monte Carlo 500회, STL 분해, What-if 시뮬레이션 |
| **Snowflake 전문성 25%** | 19개 기능: Cortex 7종 + Dynamic Tables + Tasks/Alerts + Model Registry + Feature Store + SiS + DQ + Lineage |
| **AI 전문성 25%** | Multi-Agent 3-Phase + XGBoost + SHAP + Markov + STL + Cortex Analyst |
| **현실성 15%** | 실제 Marketplace 데이터 + 정량적 민감도 분석(월 +450건) + 자동 품질 검증 |
| **발표 10%** | SiS 배포된 4페이지 대시보드 + 라이브 AI Q&A |

핵심 메시지는 이것입니다:

> **"로컬에서 개발하고, Snowflake에서 실행하고, SiS로 배포한다. 데이터로 진단하고, 수학으로 정량화하고, AI Agent로 전략을 자동 생성한다."**

감사합니다.

---

## 타임라인 요약

| 구간 | 시간 | 누적 |
|------|------|------|
| 오프닝 — 문제 정의 | 0:30 | 0:30 |
| 아키텍처 + 개발 워크플로우 + 전처리 | 2:30 | 3:00 |
| CoCo 스킬 활용 | 0:30 | 3:30 |
| 데모 — 랜딩 (SiS) | 0:30 | 4:00 |
| 데모 — 진단 (Sankey + Markov) | 1:30 | 5:30 |
| 데모 — 기회 분석 (Forecast + Monte Carlo) | 1:30 | 7:00 |
| 데모 — AI 전략 (Multi-Agent + Chat) | 1:30 | 8:30 |
| 가장 좋았던 Snowflake 기능 | 1:00 | 9:30 |
| 마무리 — 심사 기준 매핑 | 0:30 | 10:00 |

---

## 발표 팁

- 데모는 **SiS에서** 보여주세요. "Snowflake 안에서 돌아가는 대시보드"라는 것 자체가 어필됩니다.
- Sankey 다이어그램, Monte Carlo 히스토그램, AI Q&A 챗 — 이 세 장면이 **시각적 임팩트**가 가장 큽니다.
- 마르코프 체인 설명 시 "접수→개통 5%p 개선 = 월 +450건"처럼 **구체적 숫자**로 말하세요.
- "Snowflake 19개 기능" 숫자를 강조하되, 나열하지 말고 **4개 레이어**(파이프라인/ML/Cortex AI/거버넌스)로 묶어 설명하세요.
- 각 기능이 **어디서 실행되는지**(Snowflake 서버 / 로컬 Python / SiS)를 명확히 하면 이해도가 높아집니다.
- AI Q&A 데모는 "채널별 ROI 비교해줘"처럼 **미리 준비한 질문**으로 하세요. 자유 질문은 리스크가 큽니다.

---

## 실행 위치 요약 (Q&A 대비)

| 실행 위치 | 기능 | 비고 |
|----------|------|------|
| **Snowflake 워크시트** | SQL 파이프라인 (00~11), Cortex FORECAST/ANOMALY, Cortex AI Functions, Feature Store, Dynamic Tables, Tasks, Alerts, Data Quality, Lineage | SQL만으로 실행 |
| **Snowflake 서버** (내부) | Cortex COMPLETE (LLM 추론), Cortex SENTIMENT/CLASSIFY/SUMMARIZE/TRANSLATE, Model Registry (모델 저장) | 데이터가 외부로 나가지 않음 |
| **로컬 Python** | XGBoost 학습, SHAP 해석, 마르코프 체인, STL 분해, Monte Carlo, Streamlit 개발/테스트 | Snowpark으로 SF 데이터 접근 |
| **Streamlit in Snowflake** | 4페이지 대시보드 배포 | deploy_sis.py로 업로드 |

---

## Q&A 대비 — 데이터 전처리 상세 (질문 받으면 사용)

> "데이터 품질 이슈를 어떻게 처리했나요?"

| # | 이슈 | 심각도 | 처리 방법 | 적용 위치 |
|---|------|--------|----------|----------|
| 1 | CVR > 100% (최대 347,400%) | 높음 | `LEAST(CVR, 100.0)` 상한 클램핑 | `01_staging.sql` (Snowflake) |
| 2 | 미래 날짜 (2027-07 등) | 높음 | `WHERE YEAR_MONTH <= CURRENT_DATE()` | `01_staging.sql` (Snowflake) |
| 3 | 최신 미완성월 (4월 데이터 1/10 수준) | 높음 | 시계열 분석 시 최신월 자동 제외 | `components/utils.py` (Python) |
| 4 | CVR_SUBSCRIPTION 22% NULL | 중간 | `COALESCE(CVR_SUBSCRIPTION, 0)` | `01_staging.sql` (Snowflake) |
| 5 | 도명 불일치 (전북특별자치도) | 중간 | `REPLACE('전북특별자치도', '전북')` | `01_staging.sql` (Snowflake) |
| 6 | 가입 > 상담 (비순차 퍼널) | 중간 | 건수 기반 이탈률 재계산, 음수→0 처리 | `analysis/insight_generator.py` (Python) |
| 7 | 음수 매출 (환불 포함) | 낮음 | `AVG_NET_SALES.clip(lower=0)` | `pages/1_진단.py` (Python) |
| 8 | FORECAST 음수 예측 | 낮음 | 차트 Y축 최소값 0 고정 | `pages/2_기회_분석.py` (Python) |

> "왜 퍼널이 선형이 아닌가요?"

통신사 가입은 온라인 직접 가입(상담 스킵), 제휴 채널 가입 등 **다중 진입점**이 존재합니다. 그래서 가입신청이 상담요청보다 많을 수 있습니다. 저희는 이 점을 고려해서 `OVERALL_CVR`(전체 대비 최종 전환)을 주요 지표로 사용하고, 단계별 이탈률은 건수 기반으로 재계산했습니다.

> "13건 품질 검증은 구체적으로 뭔가요?"

| 검증 항목 | 대상 | 건수 | 기준 |
|----------|------|------|------|
| NULL 비율 | OVERALL_CVR, AVG_NET_SALES, PAYEND_CVR | 3건 | WARNING: 10% 초과, CRITICAL: 30% 초과 |
| CVR 범위 | OVERALL_CVR, PAYEND_CVR | 2건 | 0~100% 범위 이탈 시 CRITICAL |
| 미래 날짜 | STG_FUNNEL, STG_CHANNEL, STG_REGIONAL | 3건 | 존재 시 CRITICAL |
| 행 수 | STG_FUNNEL, STG_CHANNEL, STG_REGIONAL, ML_FEATURE_STORE | 4건 | 빈 테이블 시 CRITICAL |
| 중복 키 | STG_FUNNEL의 YEAR_MONTH + CATEGORY | 1건 | 중복 시 WARNING |
| | | **총 13건** | |

이 13건이 매일 06시 **Snowflake Task**로 자동 실행되고, CRITICAL이면 **Alert**가 발동합니다.

> "로컬과 Snowflake의 관계가 뭔가요?"

로컬에서 Python 코드를 개발하고, **Snowpark**이라는 Python SDK로 Snowflake에 연결합니다. SQL 파이프라인과 Cortex AI 함수는 Snowflake 서버에서 직접 실행되고, ML 학습과 시각화는 로컬 Python에서 수행합니다. 완성된 대시보드는 `deploy_sis.py`로 **Streamlit in Snowflake**에 배포하여, Snowflake 안에서 인증 없이 바로 실행됩니다.

[snowflake 직접들어가서]
발표 대본 (3분)
오프닝 — 대시보드 소개 (20초)
안녕하세요. 지금부터 Snowflake 통신 구독 분석 대시보드를 소개하겠습니다. 이 대시보드는 Snowflake Marketplace의 아정당 실제 운영 데이터를 기반으로 구축했습니다. 원본 데이터를 Staging → Analytics → Mart 3계층 파이프라인으로 정제했고, Streamlit in Snowflake로 시각화했습니다. 상단 KPI 카드에서 기준월의 총 계약 39만건, 평균 CVR 25.58%, 최고 채널 인바운드, 최고 성장 도시 수원시를 한눈에 확인할 수 있습니다.

탭 1 — 퍼널 분석 (35초)
첫 번째 탭은 퍼널 분석입니다. STG_FUNNEL 테이블 250건을 기반으로, 상담신청부터 납입완료까지 5단계 전환율을 추적합니다. 불완전한 마지막 달은 자동으로 제외됩니다.

상단 KPI 카드는 전월 대비 계약 건수와 CVR 변화를 delta로 보여주고, 최악 병목 구간도 표시합니다. 가운데 라인차트는 카테고리간 전체 CVR을 비교할 수 있고, 그 아래에서 선택한 카테고리의 단계별 전환율 추이를 볼 수 있습니다. 하단 좌측은 최근월 퍼널 건수 바차트, 우측은 단계별 이탈률인데, 병목 구간은 빨간색으로 하이라이트됩니다. 이 병목 데이터는 FUNNEL_BOTTLENECKS 테이블에서 가져온 것입니다.

탭 2 — 채널 분석 (30초)
두 번째 탭은 채널 분석입니다. STG_CHANNEL 7,936건을 기반으로 인바운드, 플랫폼, 카톡 등 유입 채널별 성과를 비교합니다.

좌측은 채널별 계약 건수 Top 15, 우측은 납입완료 CVR을 나란히 배치해서, 건수는 많지만 CVR이 낮은 채널을 바로 식별할 수 있습니다. 하단 라인차트는 상위 5개 채널의 월간 계약 추이로, 인바운드가 2025년 중반 급성장한 패턴을 확인할 수 있습니다.

탭 3 — 지역 분석 (25초)
세 번째 탭은 지역 분석입니다. STG_REGIONAL 23,617건과 V05 신규설치 데이터를 LEFT JOIN하여 구축했습니다.

시도별 계약 건수와 CVR을 가로 바차트로 비교하고, 시도를 선택하면 해당 지역 내 상위 20개 도시로 드릴다운됩니다. 바의 색상은 CVR을 나타내서, 계약 건수와 전환 품질을 동시에 볼 수 있습니다.

탭 4 — 예측 · 이상탐지 (40초)
네 번째 탭이 가장 핵심인 예측과 이상탐지입니다. 여기서는 Snowflake Cortex ML을 직접 활용했습니다.

상단 예측 차트는 FORECAST_REGIONAL_MODEL이 시도별 계약 건수를 향후 3개월 예측한 결과입니다. 음영이 95% 신뢰구간이고, 시도를 선택해서 비교할 수 있습니다.

하단 이상탐지는 ANOMALY_CONTRACT_MODEL이 과거 패턴 대비 실측값이 정상 범위를 벗어난 시도를 탐지한 결과입니다. 불완전월 오탐은 자동 제외되고, 시도별 이상 건수와 편차를 바차트로 보여줍니다. 시도를 선택하면 실측 vs 예상 범위 시계열 차트가 나오고, 빨간 다이아몬드가 이상 포인트입니다. 하단에는 3회 이상 탐지된 시도에 대한 즉시 점검 권고 등 비즈니스 액션 추천도 포함했습니다.

탭 5 — 데이터 품질 + 클로징 (30초)
마지막은 데이터 품질 탭입니다. NULL, 범위 초과, 미래 날짜, 행 수, 중복 등 5가지 관점에서 STAGING 테이블을 자동 검증합니다. 현재 전체 PASS 상태로 데이터 신뢰도가 확보되어 있습니다.

정리하면, 이 대시보드는 Marketplace 데이터 수집부터 3계층 파이프라인, Cortex ML 예측·이상탐지, Streamlit 시각화까지 Snowflake 플랫폼 안에서 End-to-End로 완성한 프로젝트입니다. 감사합니다.