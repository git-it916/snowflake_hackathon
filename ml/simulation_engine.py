"""What-if 채널 예산 시뮬레이션 엔진.

채널별 계약 건수(CONTRACT_COUNT) 변동에 따른
전환율(PAYEND_CVR) 변화를 시뮬레이션한다.
Monte Carlo 분석을 통해 불확실성 범위도 제공한다.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------
_FEATURE_STORE_TABLE = "ANALYTICS.ML_FEATURE_STORE"
_DEFAULT_N_MONTHS = 3
_DEFAULT_N_SIMULATIONS = 300
_MONTE_CARLO_SEED = 42


class SimulationEngine:
    """What-if 시뮬레이션을 통한 채널 예산 배분 분석 엔진.

    ConversionModel의 예측 결과를 기반으로
    채널별 계약 건수 변동 시나리오를 평가한다.
    """

    def __init__(
        self,
        session: Any,
        model: Any | None = None,
    ) -> None:
        """SimulationEngine 초기화.

        Args:
            session: Snowpark 세션
            model: ConversionModel 인스턴스.
                None이면 첫 호출 시 lazy 초기화.
        """
        self._session = session
        self._model = model
        logger.info("SimulationEngine 초기화 완료")

    # ------------------------------------------------------------------
    # 시나리오 실행
    # ------------------------------------------------------------------

    def run_scenario(
        self,
        category: str,
        channel_changes: dict[str, float],
        n_months: int = _DEFAULT_N_MONTHS,
    ) -> pd.DataFrame:
        """채널별 계약 건수 변동 시나리오를 시뮬레이션.

        각 채널의 CONTRACT_COUNT를 지정된 비율(%)만큼 변경한 후
        ConversionModel로 n_months개월간 전환율을 재예측한다.

        Args:
            category: 대상 카테고리
            channel_changes: 채널별 변동 비율 딕셔너리
                예: {"인바운드": 30, "플랫폼": -10}
                양수면 증가, 음수면 감소 (퍼센트)
            n_months: 시뮬레이션할 미래 월 수

        Returns:
            시뮬레이션 결과 DataFrame:
                MONTH, CHANNEL, ORIGINAL_CVR, MODIFIED_CVR,
                ORIGINAL_CONTRACTS, MODIFIED_CONTRACTS
        """
        model = self._ensure_model()
        baseline = self._load_baseline(category)

        if baseline.empty:
            logger.warning("카테고리 '%s' 기준 데이터 없음", category)
            return pd.DataFrame()

        results: list[dict[str, Any]] = []

        for month_offset in range(1, n_months + 1):
            for _, row in baseline.iterrows():
                channel = str(row["CHANNEL"])
                original_contracts = float(row.get("CONTRACT_COUNT", 0))
                change_pct = channel_changes.get(channel, 0.0)
                modified_contracts = original_contracts * (1.0 + change_pct / 100.0)

                # 원본 예측
                original_pred = model.predict(category, channel)
                original_cvr = original_pred.get("prob_high", 0.0)

                # 변동된 피처로 예측
                modified_features = self._build_modified_features(
                    row, modified_contracts, month_offset
                )
                modified_pred = model.predict(
                    category, channel, features=modified_features
                )
                modified_cvr = modified_pred.get("prob_high", 0.0)

                results.append({
                    "MONTH": month_offset,
                    "CHANNEL": channel,
                    "ORIGINAL_CVR": round(original_cvr, 4),
                    "MODIFIED_CVR": round(modified_cvr, 4),
                    "ORIGINAL_CONTRACTS": round(original_contracts, 0),
                    "MODIFIED_CONTRACTS": round(modified_contracts, 0),
                    "CVR_CHANGE": round(modified_cvr - original_cvr, 4),
                })

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # 시나리오 비교
    # ------------------------------------------------------------------

    def compare_scenarios(
        self,
        category: str,
        scenarios: dict[str, dict[str, float]],
    ) -> pd.DataFrame:
        """여러 시나리오를 비교 분석.

        Args:
            category: 대상 카테고리
            scenarios: 시나리오별 채널 변동 딕셔너리
                예: {
                    "인바운드집중": {"인바운드": 30, "온라인": -10},
                    "온라인전환": {"인바운드": -20, "온라인": 40},
                }

        Returns:
            시나리오 비교 DataFrame:
                SCENARIO, CHANNEL, MONTH, ORIGINAL_CVR, MODIFIED_CVR, ...
        """
        all_results: list[pd.DataFrame] = []

        for scenario_name, channel_changes in scenarios.items():
            scenario_df = self.run_scenario(category, channel_changes)
            if not scenario_df.empty:
                scenario_df = scenario_df.copy()
                scenario_df.insert(0, "SCENARIO", scenario_name)
                all_results.append(scenario_df)

        if not all_results:
            return pd.DataFrame()

        comparison = pd.concat(all_results, ignore_index=True)

        # 시나리오별 요약 통계 추가
        summary_rows: list[dict[str, Any]] = []
        for scenario_name in scenarios:
            scenario_data = comparison[comparison["SCENARIO"] == scenario_name]
            if scenario_data.empty:
                continue

            summary_rows.append({
                "SCENARIO": scenario_name,
                "AVG_CVR_CHANGE": round(
                    float(scenario_data["CVR_CHANGE"].mean()), 4
                ),
                "MAX_CVR_CHANGE": round(
                    float(scenario_data["CVR_CHANGE"].max()), 4
                ),
                "TOTAL_MODIFIED_CONTRACTS": round(
                    float(scenario_data["MODIFIED_CONTRACTS"].sum()), 0
                ),
            })

        logger.info(
            "시나리오 비교 완료: %d개 시나리오",
            len(summary_rows),
        )
        return comparison

    # ------------------------------------------------------------------
    # Monte Carlo 시뮬레이션
    # ------------------------------------------------------------------

    def run_monte_carlo(
        self,
        category: str,
        n_simulations: int = _DEFAULT_N_SIMULATIONS,
        n_months: int = _DEFAULT_N_MONTHS,
    ) -> dict[str, Any]:
        """Monte Carlo 시뮬레이션으로 불확실성 범위를 추정.

        과거 데이터의 채널별 변동 분포에서 랜덤 샘플링하여
        다수의 시나리오를 생성하고 전환율 분포를 추정한다.

        Args:
            category: 대상 카테고리
            n_simulations: 시뮬레이션 반복 횟수
            n_months: 시뮬레이션 기간 (월)

        Returns:
            시뮬레이션 결과:
                mean_cvr: 평균 전환율
                ci_95_upper: 95% 상한
                ci_95_lower: 95% 하한
                best_case: 최고 시나리오
                worst_case: 최악 시나리오
                simulations: 개별 시뮬레이션 결과 리스트
        """
        model = self._ensure_model()
        baseline = self._load_baseline(category)

        if baseline.empty:
            logger.warning("Monte Carlo: 카테고리 '%s' 데이터 없음", category)
            return {
                "mean_cvr": 0.0,
                "ci_95_upper": 0.0,
                "ci_95_lower": 0.0,
                "best_case": 0.0,
                "worst_case": 0.0,
                "simulations": [],
            }

        channels = baseline["CHANNEL"].unique().tolist()
        historical_std = self._compute_historical_volatility(category, channels)

        rng = np.random.default_rng(_MONTE_CARLO_SEED)
        simulation_cvrs: list[float] = []

        # baseline을 미리 캐싱하여 루프 내 Snowflake 쿼리 방지
        self._cached_baseline = baseline

        for sim_idx in range(n_simulations):
            # 채널별 랜덤 변동 생성 (정규분포, 과거 변동성 기반)
            channel_changes = {}
            for ch in channels:
                std = historical_std.get(ch, 10.0)
                change = float(rng.normal(0, std))
                channel_changes[ch] = change

            # 시나리오 실행 (캐시된 baseline 사용)
            scenario_df = self.run_scenario(category, channel_changes, n_months)
            if not scenario_df.empty:
                avg_cvr = float(scenario_df["MODIFIED_CVR"].mean())
                simulation_cvrs.append(avg_cvr)

            if (sim_idx + 1) % 100 == 0:
                logger.info(
                    "Monte Carlo 진행: %d/%d", sim_idx + 1, n_simulations
                )

        if not simulation_cvrs:
            return {
                "mean_cvr": 0.0,
                "ci_95_upper": 0.0,
                "ci_95_lower": 0.0,
                "best_case": 0.0,
                "worst_case": 0.0,
                "simulations": [],
            }

        arr = np.array(simulation_cvrs)
        result = {
            "mean_cvr": round(float(np.mean(arr)), 4),
            "ci_95_upper": round(float(np.percentile(arr, 97.5)), 4),
            "ci_95_lower": round(float(np.percentile(arr, 2.5)), 4),
            "best_case": round(float(np.max(arr)), 4),
            "worst_case": round(float(np.min(arr)), 4),
            "simulations": [round(float(v), 4) for v in arr],
        }

        logger.info(
            "Monte Carlo 완료: mean=%.4f, CI=[%.4f, %.4f]",
            result["mean_cvr"],
            result["ci_95_lower"],
            result["ci_95_upper"],
        )
        return result

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _ensure_model(self) -> Any:
        """ConversionModel이 초기화되어 있는지 확인하고 반환."""
        if self._model is None:
            from ml.conversion_model import ConversionModel
            self._model = ConversionModel(self._session)
            self._model.train()
            logger.info("ConversionModel lazy 초기화 및 학습 완료")
        return self._model

    def _load_baseline(self, category: str) -> pd.DataFrame:
        """카테고리의 최신 월 기준 채널별 베이스라인 데이터를 로드.

        Monte Carlo에서 반복 호출 방지를 위해 _cached_baseline을 우선 사용.
        """
        if hasattr(self, "_cached_baseline") and self._cached_baseline is not None:
            cached = self._cached_baseline
            if not cached.empty:
                return cached

        try:
            df = self._session.table(_FEATURE_STORE_TABLE).to_pandas()
            if df.empty:
                return pd.DataFrame()

            # "전체" 또는 None이면 가장 큰 카테고리(인터넷) 사용
            if category and category != "전체":
                cat_df = df[df["CATEGORY"] == category]
            else:
                cat_df = df[df["CATEGORY"] == "인터넷"] if "인터넷" in df["CATEGORY"].values else df
            if cat_df.empty:
                return pd.DataFrame()

            latest_ym = cat_df["YEAR_MONTH"].max()
            return cat_df[cat_df["YEAR_MONTH"] == latest_ym].reset_index(drop=True)

        except Exception as exc:
            logger.warning("베이스라인 데이터 로드 실패: %s", exc)
            return pd.DataFrame()

    def _build_modified_features(
        self,
        baseline_row: pd.Series,
        modified_contracts: float,
        month_offset: int,
    ) -> dict[str, float]:
        """베이스라인 행에 변동된 계약 건수를 반영한 피처 딕셔너리 생성."""
        features: dict[str, float] = {}

        for col in baseline_row.index:
            try:
                features[col] = float(baseline_row[col])
            except (ValueError, TypeError):
                continue

        # 계약 건수 변경 반영
        original_contracts = features.get("CONTRACT_COUNT_LAG1", 0.0)
        original_lag1 = features.get("CONTRACT_COUNT_LAG1", 0.0)
        features["CONTRACT_COUNT_LAG1"] = modified_contracts

        # 이전 LAG1 값을 LAG2로 전파
        features["CONTRACT_COUNT_LAG2"] = original_lag1

        # 포화 효과: 계약 건수 증가 시 CVR 소폭 감소
        contract_ratio = modified_contracts / max(original_contracts, 1)
        if contract_ratio > 1.2:
            saturation = 1.0 - (contract_ratio - 1.0) * 0.05
            for cvr_key in ["PAYEND_CVR_LAG1", "PAYEND_CVR_MA3", "PAYEND_CVR_MA6"]:
                if cvr_key in features:
                    features[cvr_key] = features[cvr_key] * saturation

        # 시간 피처 반영
        base_month = features.get("MONTH_OF_YEAR", 1)
        new_month = ((int(base_month) - 1 + month_offset) % 12) + 1
        features["MONTH_OF_YEAR"] = float(new_month)
        features["QUARTER"] = float((new_month - 1) // 3 + 1)

        return features

    def _compute_historical_volatility(
        self, category: str, channels: list[str]
    ) -> dict[str, float]:
        """채널별 계약 건수의 과거 변동성(표준편차 %)을 계산."""
        try:
            df = self._session.table(_FEATURE_STORE_TABLE).to_pandas()
            cat_df = df[df["CATEGORY"] == category]

            volatility: dict[str, float] = {}
            for ch in channels:
                ch_data = cat_df[cat_df["CHANNEL"] == ch]["CONTRACT_COUNT"]
                if len(ch_data) > 1:
                    mean_val = ch_data.mean()
                    if mean_val > 0:
                        volatility[ch] = float(ch_data.std() / mean_val * 100)
                    else:
                        volatility[ch] = 10.0
                else:
                    volatility[ch] = 10.0

            return volatility

        except Exception:
            return {ch: 10.0 for ch in channels}
