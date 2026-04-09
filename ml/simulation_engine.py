"""What-if 채널 예산 시뮬레이션 엔진.

채널 가중치 변동에 따른 전환율(PAYEND_CVR) 변화를 시뮬레이션한다.
Feature Store가 카테고리×월 단위이므로, 카테고리 레벨에서
채널 가중치를 반영한 피처 변동을 시뮬레이션한다.
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
    채널 가중치 변동 시나리오를 평가한다.
    Feature Store가 카테고리×월 단위이므로 채널별 반복 대신
    카테고리 레벨에서 가중치를 반영한다.
    """

    def __init__(
        self,
        session: Any,
        model: Any | None = None,
    ) -> None:
        self._session = session
        self._model = model
        self._cached_baseline: pd.DataFrame | None = None
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
        """채널 가중치 변동 시나리오를 시뮬레이션.

        Feature Store가 카테고리×월 단위이므로, channel_changes의
        가중치를 종합하여 카테고리 레벨 피처를 변동시킨다.

        Args:
            category: 대상 카테고리
            channel_changes: 채널별 가중치 딕셔너리
                예: {"인바운드": 0.30, "네이버": 0.15}
                또는 변동 비율: {"인바운드": 30, "플랫폼": -10}
            n_months: 시뮬레이션할 미래 월 수

        Returns:
            시뮬레이션 결과 DataFrame:
                MONTH, SCENARIO, ORIGINAL_CVR, MODIFIED_CVR,
                ORIGINAL_CONTRACTS, MODIFIED_CONTRACTS, CVR_CHANGE
        """
        model = self._ensure_model()
        baseline = self._load_baseline(category)

        if baseline.empty:
            logger.warning("카테고리 '%s' 기준 데이터 없음", category)
            return pd.DataFrame()

        # 채널 가중치를 종합 변동률로 변환
        total_weight = sum(abs(v) for v in channel_changes.values())
        if total_weight == 0:
            total_weight = 1.0

        # 가중치가 비율(0~1)인지 퍼센트(>1)인지 판단
        max_val = max(abs(v) for v in channel_changes.values()) if channel_changes else 0
        is_percentage = max_val > 1.0

        # 종합 변동률 계산: 각 채널 가중치의 가중평균
        if is_percentage:
            aggregate_change_pct = sum(channel_changes.values()) / max(len(channel_changes), 1)
        else:
            aggregate_change_pct = (total_weight - 1.0) * 100 if total_weight > 0 else 0

        results: list[dict[str, Any]] = []
        row = baseline.iloc[0]

        original_contracts = float(row.get("CONTRACT_COUNT_LAG1", row.get("CONTRACT_MA3", 0)))
        original_pred = model.predict(category)
        original_cvr = original_pred.get("prob_high", 0.0)

        for month_offset in range(1, n_months + 1):
            modified_contracts = original_contracts * (1.0 + aggregate_change_pct / 100.0)

            modified_features = self._build_modified_features(
                row, modified_contracts, month_offset, aggregate_change_pct
            )
            modified_pred = model.predict(category, features=modified_features)
            modified_cvr = modified_pred.get("prob_high", 0.0)

            scenario_label = ", ".join(
                f"{ch}:{v:.0f}%" if is_percentage else f"{ch}:{v:.0%}"
                for ch, v in list(channel_changes.items())[:3]
            )

            results.append({
                "MONTH": month_offset,
                "SCENARIO": scenario_label,
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
        """여러 시나리오를 비교 분석."""
        all_results: list[pd.DataFrame] = []

        for scenario_name, channel_changes in scenarios.items():
            scenario_df = self.run_scenario(category, channel_changes)
            if not scenario_df.empty:
                scenario_df = scenario_df.copy()
                scenario_df.insert(0, "SCENARIO_NAME", scenario_name)
                all_results.append(scenario_df)

        if not all_results:
            return pd.DataFrame()

        return pd.concat(all_results, ignore_index=True)

    # ------------------------------------------------------------------
    # Monte Carlo 시뮬레이션
    # ------------------------------------------------------------------

    def run_monte_carlo(
        self,
        category: str,
        n_simulations: int = _DEFAULT_N_SIMULATIONS,
        n_months: int = _DEFAULT_N_MONTHS,
    ) -> dict[str, Any]:
        """Monte Carlo 시뮬레이션으로 불확실성 범위를 추정."""
        model = self._ensure_model()
        baseline = self._load_baseline(category)

        if baseline.empty:
            logger.warning("Monte Carlo: 카테고리 '%s' 데이터 없음", category)
            return self._empty_monte_carlo()

        rng = np.random.default_rng(_MONTE_CARLO_SEED)
        simulation_cvrs: list[float] = []

        historical_std = self._compute_historical_volatility(category)

        for sim_idx in range(n_simulations):
            change_pct = float(rng.normal(0, historical_std))
            scenario_df = self.run_scenario(
                category, {"aggregate": change_pct}, n_months
            )
            if not scenario_df.empty:
                avg_cvr = float(scenario_df["MODIFIED_CVR"].mean())
                simulation_cvrs.append(avg_cvr)

            if (sim_idx + 1) % 100 == 0:
                logger.info("Monte Carlo 진행: %d/%d", sim_idx + 1, n_simulations)

        if not simulation_cvrs:
            return self._empty_monte_carlo()

        arr = np.array(simulation_cvrs)
        return {
            "mean_cvr": round(float(np.mean(arr)), 4),
            "ci_95_upper": round(float(np.percentile(arr, 97.5)), 4),
            "ci_95_lower": round(float(np.percentile(arr, 2.5)), 4),
            "best_case": round(float(np.max(arr)), 4),
            "worst_case": round(float(np.min(arr)), 4),
            "simulations": [round(float(v), 4) for v in arr],
        }

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
        """카테고리의 최신 월 기준 베이스라인 데이터를 로드."""
        if self._cached_baseline is not None and not self._cached_baseline.empty:
            return self._cached_baseline

        try:
            df = self._session.table(_FEATURE_STORE_TABLE).to_pandas()
            if df.empty:
                return pd.DataFrame()

            if category and category != "전체" and "CATEGORY" in df.columns:
                cat_df = df[df["CATEGORY"] == category]
            else:
                if "CATEGORY" in df.columns and "인터넷" in df["CATEGORY"].values:
                    cat_df = df[df["CATEGORY"] == "인터넷"]
                else:
                    cat_df = df

            if cat_df.empty:
                return pd.DataFrame()

            latest_ym = cat_df["YEAR_MONTH"].max()
            result = cat_df[cat_df["YEAR_MONTH"] == latest_ym].reset_index(drop=True)
            self._cached_baseline = result
            return result

        except Exception as exc:
            logger.warning("베이스라인 데이터 로드 실패: %s", exc)
            return pd.DataFrame()

    def _build_modified_features(
        self,
        baseline_row: pd.Series,
        modified_contracts: float,
        month_offset: int,
        change_pct: float = 0.0,
    ) -> dict[str, float]:
        """베이스라인 행에 변동된 계약 건수를 반영한 피처 딕셔너리 생성."""
        features: dict[str, float] = {}

        for col in baseline_row.index:
            try:
                features[col] = float(baseline_row[col])
            except (ValueError, TypeError):
                continue

        original_contracts = features.get("CONTRACT_COUNT_LAG1", 0.0)
        original_lag1 = features.get("CONTRACT_COUNT_LAG1", 0.0)
        features["CONTRACT_COUNT_LAG1"] = modified_contracts
        features["CONTRACT_COUNT_LAG2"] = original_lag1

        # 채널 집중도 변동 반영
        if "CHANNEL_HHI" in features and abs(change_pct) > 10:
            hhi_delta = change_pct / 1000
            features["CHANNEL_HHI"] = max(0.01, features["CHANNEL_HHI"] + hhi_delta)

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

    def _compute_historical_volatility(self, category: str) -> float:
        """카테고리의 계약 건수 과거 변동성(표준편차 %)을 계산."""
        try:
            df = self._session.table(_FEATURE_STORE_TABLE).to_pandas()
            if "CATEGORY" in df.columns:
                cat_df = df[df["CATEGORY"] == category]
            else:
                cat_df = df

            col = "CONTRACT_COUNT_LAG1" if "CONTRACT_COUNT_LAG1" in cat_df.columns else "CONTRACT_MA3"
            if col not in cat_df.columns or len(cat_df) < 2:
                return 10.0

            values = cat_df[col].dropna()
            if len(values) < 2 or values.mean() == 0:
                return 10.0

            return float(values.std() / values.mean() * 100)

        except Exception:
            return 10.0

    @staticmethod
    def _empty_monte_carlo() -> dict[str, Any]:
        return {
            "mean_cvr": 0.0,
            "ci_95_upper": 0.0,
            "ci_95_lower": 0.0,
            "best_case": 0.0,
            "worst_case": 0.0,
            "simulations": [],
        }
