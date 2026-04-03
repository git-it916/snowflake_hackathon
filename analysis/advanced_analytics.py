"""마르코프 체인 퍼널 분석 및 STL 시계열 분해 모듈.

Method 1 — FunnelMarkovChain:
    퍼널 5단계(상담요청→가입신청→접수→개통→납입) + 이탈 상태를
    흡수 마르코프 체인으로 모델링하고, 민감도 분석과 몬테카를로 시뮬레이션을 수행.

Method 2 — TimeSeriesDecomposer:
    STL(Seasonal-Trend decomposition using Loess)을 적용하여
    CVR·계약건수의 추세/계절성/잔차를 분리하고 계절 패턴을 요약.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config.constants import FUNNEL_STAGES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 공통 상수
# ---------------------------------------------------------------------------
_MAJOR_CATEGORIES: list[str] = [
    "인터넷",
    "렌탈",
    "모바일",
    "알뜰 요금제",
    "유심만",
]

_MONTH_NAMES_KR: dict[int, str] = {
    1: "1월", 2: "2월", 3: "3월", 4: "4월",
    5: "5월", 6: "6월", 7: "7월", 8: "8월",
    9: "9월", 10: "10월", 11: "11월", 12: "12월",
}


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  METHOD 1: Markov Chain Transition Matrix                             ║
# ╚═════════════════════════════════════════════════════════════════════════╝


class FunnelMarkovChain:
    """퍼널 5단계를 마르코프 체인 전이 행렬로 모델링.

    각 단계 간 전이 확률을 계산하고, 민감도 분석을 통해
    "어떤 전이를 개선하면 최종 전환율이 가장 많이 올라가는가?"를 정량화.
    """

    STAGES: list[str] = FUNNEL_STAGES + ["DROP"]

    STAGE_LABELS: dict[str, str] = {
        "CONSULT_REQUEST": "상담요청",
        "SUBSCRIPTION": "가입신청",
        "REGISTEND": "접수완료",
        "OPEN": "개통",
        "PAYEND": "납입완료",
        "DROP": "이탈",
    }

    # -----------------------------------------------------------------
    # 전이 행렬 계산
    # -----------------------------------------------------------------

    def compute_transition_matrix(
        self,
        stage_drop_df: pd.DataFrame,
        category: Optional[str] = None,
    ) -> pd.DataFrame:
        """카테고리별 전이 확률 행렬 계산.

        Args:
            stage_drop_df: FUNNEL_STAGE_DROP 데이터.
                컬럼: YEAR_MONTH, MAIN_CATEGORY_NAME, STAGE_ORDER,
                       STAGE_NAME, PREV_STAGE_COUNT, CURR_STAGE_COUNT,
                       DROP_RATE, BOTTLENECK_FLAG
            category: None이면 주요 카테고리 합산.

        Returns:
            6x6 DataFrame (from_stage x to_stage), 값은 확률 (0~1).
            행 합 = 1.0.
        """
        if stage_drop_df.empty:
            return self._empty_matrix()

        filtered = self._filter_data(stage_drop_df, category)
        if filtered.empty:
            return self._empty_matrix()

        transition_probs = self._extract_transition_probs(filtered)
        return self._build_matrix(transition_probs)

    # -----------------------------------------------------------------
    # 정상 상태 분포 (흡수 확률)
    # -----------------------------------------------------------------

    def compute_steady_state(
        self,
        transition_matrix: pd.DataFrame,
    ) -> dict[str, float]:
        """흡수 마르코프 체인의 흡수 확률 계산 (장기 전환율).

        전이 행렬에서 과도 상태(transient)의 기본 행렬 N = (I - Q)^{-1}을
        구하고, 흡수 확률 B = N @ R 을 계산하여 CONSULT_REQUEST에서 출발한
        고객이 최종적으로 PAYEND에 도달할 확률을 반환.

        Args:
            transition_matrix: compute_transition_matrix() 결과.

        Returns:
            {"PAYEND": 0.38, "DROP": 0.62, ...} — 장기 전환/이탈 비율.
        """
        if transition_matrix.empty:
            return {s: 0.0 for s in self.STAGES}

        transient = list(FUNNEL_STAGES[:-1])  # PAYEND 제외
        absorbing = ["PAYEND", "DROP"]

        q_matrix = transition_matrix.loc[transient, transient].values.astype(float)
        r_matrix = transition_matrix.loc[transient, absorbing].values.astype(float)

        identity = np.eye(len(transient))

        try:
            fundamental = np.linalg.inv(identity - q_matrix)
        except np.linalg.LinAlgError:
            logger.warning("기본 행렬 역행렬 계산 실패. 정규화된 의사역행렬 사용.")
            fundamental = np.linalg.pinv(identity - q_matrix)

        absorption_probs = fundamental @ r_matrix  # shape: (4, 2)

        # CONSULT_REQUEST 행(인덱스 0)의 흡수 확률
        entry_probs = absorption_probs[0]

        result: dict[str, float] = {}
        for stage in self.STAGES:
            if stage in absorbing:
                idx = absorbing.index(stage)
                result[stage] = round(float(entry_probs[idx]), 6)
            elif stage == "CONSULT_REQUEST":
                result[stage] = 0.0  # 출발점이므로 정상 상태에서 0
            else:
                # 과도 상태의 기대 방문 횟수 (정규화)
                t_idx = transient.index(stage)
                result[stage] = round(float(fundamental[0, t_idx]), 6)

        return result

    # -----------------------------------------------------------------
    # 민감도 분석
    # -----------------------------------------------------------------

    def sensitivity_analysis(
        self,
        transition_matrix: pd.DataFrame,
        improvement_pct: float = 0.05,
        monthly_entries: int = 10000,
    ) -> pd.DataFrame:
        """민감도 분석: 각 전이 확률을 5%p 개선 시 최종 전환율 변화.

        "접수→개통 전이를 5%p 올리면 최종 납입 비율이 몇 %p 올라가는가?"

        Args:
            transition_matrix: 현재 전이 행렬.
            improvement_pct: 개선 폭 (기본 0.05 = 5%p).
            monthly_entries: 월평균 진입 고객 수 (추가 전환 고객 수 계산용).

        Returns:
            DataFrame — FROM_STAGE, TO_STAGE, CURRENT_PROB, IMPROVED_PROB,
            CURRENT_PAYEND_RATE, IMPROVED_PAYEND_RATE, DELTA,
            ADDITIONAL_CUSTOMERS.
        """
        if transition_matrix.empty:
            return pd.DataFrame(columns=[
                "FROM_STAGE", "TO_STAGE", "CURRENT_PROB", "IMPROVED_PROB",
                "CURRENT_PAYEND_RATE", "IMPROVED_PAYEND_RATE", "DELTA",
                "ADDITIONAL_CUSTOMERS",
            ])

        baseline_state = self.compute_steady_state(transition_matrix)
        baseline_payend = baseline_state.get("PAYEND", 0.0)

        records: list[dict[str, object]] = []

        # 개선 대상: 각 과도 상태 → 다음 상태 전이
        for i in range(len(FUNNEL_STAGES) - 1):
            from_stage = FUNNEL_STAGES[i]
            to_stage = FUNNEL_STAGES[i + 1]

            current_prob = float(transition_matrix.loc[from_stage, to_stage])

            improved_prob = min(current_prob + improvement_pct, 1.0)
            actual_improvement = improved_prob - current_prob

            if actual_improvement <= 0:
                records.append({
                    "FROM_STAGE": from_stage,
                    "TO_STAGE": to_stage,
                    "CURRENT_PROB": round(current_prob, 4),
                    "IMPROVED_PROB": round(improved_prob, 4),
                    "CURRENT_PAYEND_RATE": round(baseline_payend, 6),
                    "IMPROVED_PAYEND_RATE": round(baseline_payend, 6),
                    "DELTA": 0.0,
                    "ADDITIONAL_CUSTOMERS": 0,
                })
                continue

            # 변경된 전이 행렬 생성 (불변성: 복사본 사용)
            modified = transition_matrix.copy()
            modified.loc[from_stage, to_stage] = improved_prob
            modified.loc[from_stage, "DROP"] = max(
                float(modified.loc[from_stage, "DROP"]) - actual_improvement, 0.0,
            )

            improved_state = self.compute_steady_state(modified)
            improved_payend = improved_state.get("PAYEND", 0.0)
            delta = improved_payend - baseline_payend

            records.append({
                "FROM_STAGE": from_stage,
                "TO_STAGE": to_stage,
                "CURRENT_PROB": round(current_prob, 4),
                "IMPROVED_PROB": round(improved_prob, 4),
                "CURRENT_PAYEND_RATE": round(baseline_payend, 6),
                "IMPROVED_PAYEND_RATE": round(improved_payend, 6),
                "DELTA": round(delta, 6),
                "ADDITIONAL_CUSTOMERS": int(round(delta * monthly_entries)),
            })

        result = pd.DataFrame(records)
        return result.sort_values("DELTA", ascending=False).reset_index(drop=True)

    # -----------------------------------------------------------------
    # 몬테카를로 시뮬레이션
    # -----------------------------------------------------------------

    def simulate_path(
        self,
        transition_matrix: pd.DataFrame,
        n_customers: int = 10000,
        seed: int = 42,
    ) -> dict[str, object]:
        """Monte Carlo 시뮬레이션으로 고객 경로 추적.

        n_customers명의 가상 고객을 CONSULT_REQUEST에서 시작시켜
        전이 확률에 따라 이동시키고 최종 결과 분포를 반환.

        Args:
            transition_matrix: compute_transition_matrix() 결과.
            n_customers: 시뮬레이션 고객 수 (기본 10,000).
            seed: 난수 시드 (재현성).

        Returns:
            {
                "completed": 3800,
                "dropped_at": {"SUBSCRIPTION": 2000, "REGISTEND": 1500, ...},
                "completion_rate": 0.38,
                "stage_reached": {"CONSULT_REQUEST": 10000, ...},
            }
        """
        if transition_matrix.empty:
            return {
                "completed": 0,
                "dropped_at": {},
                "completion_rate": 0.0,
                "stage_reached": {},
            }

        rng = np.random.default_rng(seed)

        completed = 0
        dropped_at: dict[str, int] = {}
        stage_reached: dict[str, int] = {s: 0 for s in FUNNEL_STAGES}

        stage_indices = {s: i for i, s in enumerate(self.STAGES)}
        matrix_values = transition_matrix.values.astype(float)

        for _ in range(n_customers):
            current = "CONSULT_REQUEST"
            stage_reached[current] = stage_reached.get(current, 0) + 1

            while current not in ("PAYEND", "DROP"):
                row_idx = stage_indices[current]
                probs = matrix_values[row_idx]

                # 확률 정규화 (부동소수점 보정)
                prob_sum = probs.sum()
                if prob_sum > 0:
                    normalized_probs = probs / prob_sum
                else:
                    break

                next_idx = rng.choice(len(self.STAGES), p=normalized_probs)
                current = self.STAGES[next_idx]

                if current in FUNNEL_STAGES:
                    stage_reached[current] = stage_reached.get(current, 0) + 1

            if current == "PAYEND":
                completed += 1
            elif current == "DROP":
                # 마지막으로 방문한 과도 상태를 이탈 지점으로 기록
                # 이탈 직전 상태 = 가장 마지막으로 reached된 비-흡수 상태
                last_transient = self._find_last_transient(stage_reached, completed)
                dropped_at[last_transient] = dropped_at.get(last_transient, 0) + 1

        # 이탈 지점 분포 보정: 시뮬레이션 후 역산
        dropped_at = self._compute_drop_distribution(stage_reached, completed)

        return {
            "completed": completed,
            "dropped_at": dropped_at,
            "completion_rate": round(completed / max(n_customers, 1), 4),
            "stage_reached": stage_reached,
        }

    # -----------------------------------------------------------------
    # 카테고리별 전체 분석
    # -----------------------------------------------------------------

    def analyze_all_categories(
        self,
        stage_drop_df: pd.DataFrame,
    ) -> dict[str, dict[str, object]]:
        """주요 카테고리별 전이 행렬·흡수 확률·민감도를 일괄 계산.

        Args:
            stage_drop_df: FUNNEL_STAGE_DROP 전체 데이터.

        Returns:
            {
                "인터넷": {
                    "transition_matrix": pd.DataFrame,
                    "absorption_probs": dict,
                    "sensitivity": pd.DataFrame,
                },
                ...
            }
        """
        if stage_drop_df.empty:
            return {}

        available_cats = (
            stage_drop_df["MAIN_CATEGORY_NAME"].unique()
            if "MAIN_CATEGORY_NAME" in stage_drop_df.columns
            else []
        )

        target_cats = [c for c in _MAJOR_CATEGORIES if c in available_cats]
        results: dict[str, dict[str, object]] = {}

        for cat in target_cats:
            matrix = self.compute_transition_matrix(stage_drop_df, category=cat)
            if matrix.empty:
                continue

            absorption = self.compute_steady_state(matrix)
            sensitivity = self.sensitivity_analysis(matrix)

            results[cat] = {
                "transition_matrix": matrix,
                "absorption_probs": absorption,
                "sensitivity": sensitivity,
            }

        return results

    # -----------------------------------------------------------------
    # 내부 헬퍼
    # -----------------------------------------------------------------

    def _empty_matrix(self) -> pd.DataFrame:
        """빈 6x6 전이 행렬 반환."""
        return pd.DataFrame(
            0.0,
            index=self.STAGES,
            columns=self.STAGES,
        )

    def _filter_data(
        self,
        df: pd.DataFrame,
        category: Optional[str],
    ) -> pd.DataFrame:
        """카테고리 필터 + 주요 카테고리 제한 + 불완전 월 제외."""
        filtered = df.copy()

        if "MAIN_CATEGORY_NAME" in filtered.columns:
            if category is not None:
                filtered = filtered[
                    filtered["MAIN_CATEGORY_NAME"] == category
                ]
            else:
                filtered = filtered[
                    filtered["MAIN_CATEGORY_NAME"].isin(_MAJOR_CATEGORIES)
                ]

        # 불완전 월(최신 월) 제외
        if "YEAR_MONTH" in filtered.columns and not filtered.empty:
            latest = filtered["YEAR_MONTH"].max()
            filtered = filtered[filtered["YEAR_MONTH"] != latest]

        return filtered

    def _extract_transition_probs(
        self,
        df: pd.DataFrame,
    ) -> dict[str, float]:
        """FUNNEL_STAGE_DROP에서 각 스테이지 전이 확률 추출.

        STAGE_ORDER 순으로 정렬하여 인접 단계 전이 확률을 계산.
        PREV_STAGE_COUNT / CURR_STAGE_COUNT 기반.

        Returns:
            {"CONSULT_REQUEST→SUBSCRIPTION": 0.85, ...}
        """
        probs: dict[str, float] = {}

        if "STAGE_ORDER" in df.columns:
            sorted_df = df.sort_values("STAGE_ORDER")
        else:
            sorted_df = df

        # 스테이지별 집계 (여러 월/카테고리 평균)
        stage_stats: dict[str, dict[str, float]] = {}

        for stage_name in FUNNEL_STAGES:
            stage_rows = sorted_df[sorted_df["STAGE_NAME"] == stage_name]
            if stage_rows.empty:
                continue

            prev_total = stage_rows["PREV_STAGE_COUNT"].sum()
            curr_total = stage_rows["CURR_STAGE_COUNT"].sum()
            stage_stats[stage_name] = {
                "prev": float(prev_total),
                "curr": float(curr_total),
            }

        # 전이 확률 계산
        for i in range(len(FUNNEL_STAGES) - 1):
            curr_stage = FUNNEL_STAGES[i]
            next_stage = FUNNEL_STAGES[i + 1]

            if next_stage in stage_stats:
                stats = stage_stats[next_stage]
                prev_count = stats["prev"]
                curr_count = stats["curr"]

                if prev_count > 0:
                    prob = np.clip(curr_count / prev_count, 0.0, 1.0)
                else:
                    prob = 0.0
            else:
                prob = 0.0

            key = f"{curr_stage}→{next_stage}"
            probs[key] = float(prob)

        return probs

    def _build_matrix(
        self,
        transition_probs: dict[str, float],
    ) -> pd.DataFrame:
        """전이 확률 딕셔너리로 6x6 행렬 구성.

        규칙:
        - P(stage_i → stage_{i+1}) = 전이 확률
        - P(stage_i → DROP) = 1 - P(stage_i → stage_{i+1})
        - PAYEND는 흡수 상태: P(PAYEND → PAYEND) = 1.0
        - DROP은 흡수 상태: P(DROP → DROP) = 1.0
        """
        matrix = pd.DataFrame(
            0.0,
            index=self.STAGES,
            columns=self.STAGES,
        )

        for i in range(len(FUNNEL_STAGES) - 1):
            from_s = FUNNEL_STAGES[i]
            to_s = FUNNEL_STAGES[i + 1]
            key = f"{from_s}→{to_s}"

            prob = transition_probs.get(key, 0.0)
            matrix.loc[from_s, to_s] = prob
            matrix.loc[from_s, "DROP"] = 1.0 - prob

        # 흡수 상태
        matrix.loc["PAYEND", "PAYEND"] = 1.0
        matrix.loc["DROP", "DROP"] = 1.0

        return matrix

    @staticmethod
    def _find_last_transient(
        stage_reached: dict[str, int],
        completed: int,
    ) -> str:
        """가장 마지막으로 도달한 과도 상태를 추정."""
        for stage in reversed(FUNNEL_STAGES[:-1]):
            if stage_reached.get(stage, 0) > 0:
                return stage
        return FUNNEL_STAGES[0]

    @staticmethod
    def _compute_drop_distribution(
        stage_reached: dict[str, int],
        completed: int,
    ) -> dict[str, int]:
        """스테이지 도달 수와 완료 수에서 이탈 분포를 역산.

        각 단계에 도달한 수 - 다음 단계에 도달한 수 = 해당 단계 이탈 수.
        """
        drops: dict[str, int] = {}

        for i in range(len(FUNNEL_STAGES) - 1):
            curr = FUNNEL_STAGES[i]
            nxt = FUNNEL_STAGES[i + 1]
            reached_curr = stage_reached.get(curr, 0)
            reached_next = stage_reached.get(nxt, 0)
            drop_count = max(reached_curr - reached_next, 0)
            if drop_count > 0:
                drops[curr] = drop_count

        return drops


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  METHOD 2: STL Time Series Decomposition                             ║
# ╚═════════════════════════════════════════════════════════════════════════╝


class TimeSeriesDecomposer:
    """시계열 분해: Trend + Seasonal + Residual.

    STL (Seasonal and Trend decomposition using Loess) 또는
    클래식 분해를 적용하여 계절성 패턴을 추출.
    """

    # -----------------------------------------------------------------
    # CVR 시계열 분해
    # -----------------------------------------------------------------

    def decompose_category_cvr(
        self,
        funnel_ts_df: pd.DataFrame,
        category: str = "인터넷",
    ) -> dict[str, object]:
        """카테고리 CVR 시계열 분해.

        Args:
            funnel_ts_df: V_FUNNEL_TIMESERIES 데이터.
                컬럼: YEAR_MONTH, MAIN_CATEGORY_NAME, OVERALL_CVR,
                       TOTAL_COUNT, PAYEND_COUNT
            category: 분석 대상 카테고리 (기본 "인터넷").

        Returns:
            {
                "trend": pd.Series,
                "seasonal": pd.Series,
                "residual": pd.Series,
                "original": pd.Series,
                "seasonal_peaks": [2, 8],
                "seasonal_troughs": [0, 7],
                "trend_direction": "declining",
                "seasonality_strength": 0.35,
                "category": "인터넷",
            }
        """
        if funnel_ts_df.empty:
            return self._empty_decomposition(category)

        ts = self._prepare_cvr_series(funnel_ts_df, category)
        if ts is None or len(ts) < 4:
            logger.warning(
                "카테고리 '%s'의 CVR 시계열 길이 부족 (%d). 최소 4 필요.",
                category,
                0 if ts is None else len(ts),
            )
            return self._empty_decomposition(category)

        return self._decompose(ts, category=category)

    # -----------------------------------------------------------------
    # 지역 계약수 시계열 분해
    # -----------------------------------------------------------------

    def decompose_regional_contracts(
        self,
        regional_df: pd.DataFrame,
        state: str = "경기",
    ) -> dict[str, object]:
        """지역 계약수 시계열 분해.

        계약 건수의 계절성을 분석하여 마케팅 타이밍 최적화 근거 제공.

        Args:
            regional_df: STG_REGIONAL 또는 REGIONAL_DEMAND_SCORE 데이터.
            state: 분석 대상 시/도 (기본 "경기").

        Returns:
            decompose_category_cvr와 동일 구조의 딕셔너리.
        """
        if regional_df.empty:
            return self._empty_decomposition(state)

        ts = self._prepare_regional_series(regional_df, state)
        if ts is None or len(ts) < 4:
            logger.warning(
                "지역 '%s'의 계약수 시계열 길이 부족 (%d). 최소 4 필요.",
                state,
                0 if ts is None else len(ts),
            )
            return self._empty_decomposition(state)

        return self._decompose(ts, category=state)

    # -----------------------------------------------------------------
    # 계절 패턴 요약
    # -----------------------------------------------------------------

    def find_seasonal_pattern(
        self,
        decompose_result: dict[str, object],
    ) -> str:
        """계절 패턴을 한국어로 요약.

        Args:
            decompose_result: decompose_category_cvr() 또는
                              decompose_regional_contracts() 결과.

        Returns:
            "인터넷 가입은 3월과 9월에 피크(+12.3%), 1월과 8월에 바닥(-8.7%)"
        """
        seasonal = decompose_result.get("seasonal")
        category = decompose_result.get("category", "")
        peaks = decompose_result.get("seasonal_peaks", [])
        troughs = decompose_result.get("seasonal_troughs", [])

        if seasonal is None or (not peaks and not troughs):
            return f"{category}: 계절 패턴을 식별할 수 없습니다 (데이터 부족)."

        # 월별 계절 효과 평균
        monthly_effects = self._monthly_seasonal_effects(seasonal)

        peak_parts: list[str] = []
        for month_idx in peaks[:3]:
            month_label = _MONTH_NAMES_KR.get(month_idx + 1, f"{month_idx + 1}월")
            effect = monthly_effects.get(month_idx + 1, 0.0)
            peak_parts.append(f"{month_label}(+{abs(effect):.1%})")

        trough_parts: list[str] = []
        for month_idx in troughs[:3]:
            month_label = _MONTH_NAMES_KR.get(month_idx + 1, f"{month_idx + 1}월")
            effect = monthly_effects.get(month_idx + 1, 0.0)
            trough_parts.append(f"{month_label}(-{abs(effect):.1%})")

        peak_str = "과 ".join(peak_parts) if peak_parts else "없음"
        trough_str = "과 ".join(trough_parts) if trough_parts else "없음"

        direction = decompose_result.get("trend_direction", "stable")
        direction_kr = {"ascending": "상승", "declining": "하락", "stable": "안정"}.get(
            direction, "안정"
        )

        return (
            f"{category}: 피크 {peak_str}, "
            f"바닥 {trough_str}. "
            f"장기 추세: {direction_kr}."
        )

    # -----------------------------------------------------------------
    # Plotly 차트용 데이터
    # -----------------------------------------------------------------

    def plot_data(
        self,
        decompose_result: dict[str, object],
    ) -> dict[str, dict[str, list]]:
        """Plotly 차트용 데이터 반환.

        Args:
            decompose_result: decompose_category_cvr() 등의 결과.

        Returns:
            {
                "original": {"x": dates, "y": values},
                "trend": {"x": dates, "y": values},
                "seasonal": {"x": [1..12], "y": monthly_pattern},
                "residual": {"x": dates, "y": values},
            }
        """
        original = decompose_result.get("original")
        trend = decompose_result.get("trend")
        seasonal = decompose_result.get("seasonal")
        residual = decompose_result.get("residual")

        result: dict[str, dict[str, list]] = {}

        if original is not None and isinstance(original, pd.Series):
            result["original"] = {
                "x": [str(d) for d in original.index],
                "y": original.values.tolist(),
            }

        if trend is not None and isinstance(trend, pd.Series):
            result["trend"] = {
                "x": [str(d) for d in trend.index],
                "y": trend.values.tolist(),
            }

        if seasonal is not None and isinstance(seasonal, pd.Series):
            monthly = self._monthly_seasonal_effects(seasonal)
            result["seasonal"] = {
                "x": list(range(1, 13)),
                "y": [monthly.get(m, 0.0) for m in range(1, 13)],
            }

        if residual is not None and isinstance(residual, pd.Series):
            result["residual"] = {
                "x": [str(d) for d in residual.index],
                "y": residual.values.tolist(),
            }

        return result

    # -----------------------------------------------------------------
    # 카테고리별 전체 분석
    # -----------------------------------------------------------------

    def analyze_all_categories(
        self,
        funnel_ts_df: pd.DataFrame,
    ) -> dict[str, dict[str, object]]:
        """주요 카테고리별 CVR 시계열 분해를 일괄 수행.

        Args:
            funnel_ts_df: V_FUNNEL_TIMESERIES 전체 데이터.

        Returns:
            {"인터넷": decompose_result, "렌탈": decompose_result, ...}
        """
        if funnel_ts_df.empty:
            return {}

        available_cats = (
            funnel_ts_df["MAIN_CATEGORY_NAME"].unique()
            if "MAIN_CATEGORY_NAME" in funnel_ts_df.columns
            else []
        )

        target_cats = [c for c in _MAJOR_CATEGORIES if c in available_cats]
        results: dict[str, dict[str, object]] = {}

        for cat in target_cats:
            decomposed = self.decompose_category_cvr(funnel_ts_df, category=cat)
            if decomposed.get("trend") is not None:
                results[cat] = decomposed

        return results

    # -----------------------------------------------------------------
    # 내부 헬퍼
    # -----------------------------------------------------------------

    def _prepare_cvr_series(
        self,
        df: pd.DataFrame,
        category: str,
    ) -> Optional[pd.Series]:
        """V_FUNNEL_TIMESERIES에서 카테고리 CVR 시계열 추출."""
        filtered = df.copy()

        if "MAIN_CATEGORY_NAME" in filtered.columns:
            filtered = filtered[filtered["MAIN_CATEGORY_NAME"] == category]

        if filtered.empty or "OVERALL_CVR" not in filtered.columns:
            return None

        if "YEAR_MONTH" not in filtered.columns:
            return None

        # 불완전 월 제외
        latest = filtered["YEAR_MONTH"].max()
        filtered = filtered[filtered["YEAR_MONTH"] != latest]

        if filtered.empty:
            return None

        # 월별 집계 (동일 월 여러 행이 있을 경우)
        monthly = (
            filtered.groupby("YEAR_MONTH")["OVERALL_CVR"]
            .mean()
            .sort_index()
        )

        return self._to_datetime_series(monthly)

    def _prepare_regional_series(
        self,
        df: pd.DataFrame,
        state: str,
    ) -> Optional[pd.Series]:
        """REGIONAL_DEMAND_SCORE에서 지역 계약수 시계열 추출."""
        filtered = df.copy()

        state_col = self._find_state_column(filtered)
        if state_col is None:
            return None

        filtered = filtered[filtered[state_col] == state]
        if filtered.empty:
            return None

        if "YEAR_MONTH" not in filtered.columns:
            return None

        # 불완전 월 제외
        latest = filtered["YEAR_MONTH"].max()
        filtered = filtered[filtered["YEAR_MONTH"] != latest]

        if filtered.empty:
            return None

        # 건수 컬럼 결정
        count_col = None
        for col in ("CONTRACT_COUNT", "PAYEND_COUNT", "CONSULT_REQUEST_COUNT"):
            if col in filtered.columns:
                count_col = col
                break

        if count_col is None:
            return None

        monthly = (
            filtered.groupby("YEAR_MONTH")[count_col]
            .sum()
            .sort_index()
        )

        return self._to_datetime_series(monthly)

    def _decompose(
        self,
        ts: pd.Series,
        category: str = "",
    ) -> dict[str, object]:
        """STL 또는 클래식 분해 수행.

        STL 사용 가능 시 STL을 우선 적용하고,
        데이터 길이가 부족하면 이동평균 기반 간이 분해로 대체.
        """
        period = min(12, max(2, len(ts) // 2))

        result = None
        if len(ts) >= period * 2:
            result = self._stl_decompose(ts, period)

        # STL 실패 또는 데이터 부족 시 간이 분해로 대체
        if result is None:
            result = self._simple_decompose(ts, period)

        if result is None:
            return self._empty_decomposition(category)

        trend, seasonal, residual = result

        # 피크/바닥 월 추출
        peaks, troughs = self._find_peaks_troughs(seasonal)

        # 추세 방향 판정
        direction = self._determine_trend_direction(trend)

        # 계절성 강도
        strength = self._compute_seasonality_strength(ts, trend, residual)

        return {
            "trend": trend,
            "seasonal": seasonal,
            "residual": residual,
            "original": ts,
            "seasonal_peaks": peaks,
            "seasonal_troughs": troughs,
            "trend_direction": direction,
            "seasonality_strength": round(strength, 4),
            "category": category,
        }

    @staticmethod
    def _stl_decompose(
        ts: pd.Series,
        period: int,
    ) -> Optional[tuple[pd.Series, pd.Series, pd.Series]]:
        """statsmodels STL 분해."""
        try:
            from statsmodels.tsa.seasonal import STL

            stl = STL(ts, period=period, robust=True)
            fit = stl.fit()
            return fit.trend, fit.seasonal, fit.resid
        except ImportError:
            logger.warning("statsmodels 미설치. 간이 분해로 대체합니다.")
            return None
        except Exception:
            logger.exception("STL 분해 실패.")
            return None

    @staticmethod
    def _simple_decompose(
        ts: pd.Series,
        period: int,
    ) -> Optional[tuple[pd.Series, pd.Series, pd.Series]]:
        """이동평균 기반 간이 분해 (statsmodels 없을 때 대체).

        trend = 이동평균, seasonal = 원본 - trend의 월별 평균, residual = 나머지.
        """
        if len(ts) < 3:
            return None

        window = min(period, len(ts))
        trend = ts.rolling(window=window, center=True, min_periods=1).mean()

        detrended = ts - trend

        # 월별 계절 효과 (DatetimeIndex가 있는 경우)
        if hasattr(ts.index, "month"):
            monthly_effect = detrended.groupby(ts.index.month).mean()
            seasonal = detrended.copy()
            for idx in seasonal.index:
                month = idx.month
                if month in monthly_effect.index:
                    seasonal[idx] = monthly_effect[month]
        else:
            # 주기적 평균
            seasonal_pattern = []
            for i in range(len(detrended)):
                pos = i % period
                same_pos = detrended.iloc[pos::period]
                seasonal_pattern.append(same_pos.mean())
            seasonal = pd.Series(
                [seasonal_pattern[i % period] for i in range(len(ts))],
                index=ts.index,
            )

        residual = ts - trend - seasonal
        return trend, seasonal, residual

    @staticmethod
    def _find_peaks_troughs(
        seasonal: pd.Series,
    ) -> tuple[list[int], list[int]]:
        """계절 컴포넌트에서 피크/바닥 월 추출 (0-indexed month).

        월별 평균 계절 효과를 계산하여 상위/하위 월을 식별.
        """
        if seasonal is None or seasonal.empty:
            return [], []

        # 월별 평균 계절 효과
        if hasattr(seasonal.index, "month"):
            monthly = seasonal.groupby(seasonal.index.month).mean()
        else:
            # 인덱스에 월 정보가 없으면 주기(12) 기반 그룹핑
            period = min(12, len(seasonal))
            monthly_vals: dict[int, list[float]] = {}
            for i, val in enumerate(seasonal.values):
                month = i % period
                monthly_vals.setdefault(month, []).append(val)
            monthly = pd.Series(
                {m: np.mean(vals) for m, vals in monthly_vals.items()}
            )

        if monthly.empty:
            return [], []

        mean_effect = monthly.mean()
        std_effect = monthly.std()

        if std_effect == 0:
            return [], []

        # 피크: 평균 이상 중 상위 (0-indexed로 변환)
        peaks_raw = monthly[monthly > mean_effect].sort_values(ascending=False)
        troughs_raw = monthly[monthly < mean_effect].sort_values(ascending=True)

        # 월 인덱스를 0-indexed로 변환 (1-based month → 0-based)
        if hasattr(seasonal.index, "month"):
            peaks = [int(m) - 1 for m in peaks_raw.index[:3]]
            troughs = [int(m) - 1 for m in troughs_raw.index[:3]]
        else:
            peaks = [int(m) for m in peaks_raw.index[:3]]
            troughs = [int(m) for m in troughs_raw.index[:3]]

        return peaks, troughs

    @staticmethod
    def _determine_trend_direction(trend: pd.Series) -> str:
        """추세 방향 판정: ascending / declining / stable.

        선형 회귀 기울기와 전반/후반 평균 비교를 종합.
        """
        if trend is None or len(trend) < 2:
            return "stable"

        clean = trend.dropna()
        if len(clean) < 2:
            return "stable"

        # 방법 1: 선형 회귀 기울기
        x = np.arange(len(clean), dtype=float)
        y = clean.values.astype(float)
        mean_y = np.mean(y)

        if mean_y == 0:
            return "stable"

        # 최소자승법 기울기
        x_centered = x - np.mean(x)
        slope = np.sum(x_centered * (y - mean_y)) / max(np.sum(x_centered ** 2), 1e-10)

        # 기울기를 평균 대비 정규화 (전체 기간 변화율)
        total_change = slope * len(clean) / abs(mean_y)

        # 방법 2: 전반/후반 평균 비교 (보조)
        midpoint = len(clean) // 2
        first_half_mean = clean.iloc[:midpoint].mean()
        second_half_mean = clean.iloc[midpoint:].mean()
        half_change = (
            (second_half_mean - first_half_mean) / abs(first_half_mean)
            if first_half_mean != 0
            else 0.0
        )

        # 두 방법 모두 같은 방향을 가리키거나, 기울기가 충분히 클 때 판정
        if total_change > 0.03 or half_change > 0.03:
            return "ascending"
        elif total_change < -0.03 or half_change < -0.03:
            return "declining"
        return "stable"

    @staticmethod
    def _compute_seasonality_strength(
        original: pd.Series,
        trend: pd.Series,
        residual: pd.Series,
    ) -> float:
        """계절성 강도 계산: 1 - Var(residual) / Var(detrended).

        값이 1에 가까울수록 계절성이 강함. 0이면 계절성 없음.
        """
        detrended = original - trend
        var_detrended = detrended.var()
        var_residual = residual.var()

        if var_detrended == 0 or np.isnan(var_detrended):
            return 0.0

        strength = 1.0 - (var_residual / var_detrended)
        return float(np.clip(strength, 0.0, 1.0))

    @staticmethod
    def _to_datetime_series(series: pd.Series) -> pd.Series:
        """인덱스를 DatetimeIndex(월초 빈도)로 변환."""
        try:
            dt_index = pd.to_datetime(series.index)
            result = series.copy()
            result.index = dt_index
            result = result.sort_index()
            result = result.asfreq("MS")
            if result is not None:
                result = result.ffill()
            return result
        except Exception:
            logger.warning("DatetimeIndex 변환 실패. 원본 인덱스 유지.")
            return series

    @staticmethod
    def _find_state_column(df: pd.DataFrame) -> Optional[str]:
        """시/도 컬럼명 해소."""
        for col in ("INSTALL_STATE", "STATE"):
            if col in df.columns:
                return col
        return None

    @staticmethod
    def _monthly_seasonal_effects(
        seasonal: pd.Series,
    ) -> dict[int, float]:
        """월별 계절 효과 평균 반환 (1~12월 키)."""
        if seasonal is None or seasonal.empty:
            return {}

        if hasattr(seasonal.index, "month"):
            grouped = seasonal.groupby(seasonal.index.month).mean()
            return {int(m): round(float(v), 4) for m, v in grouped.items()}

        # fallback: 주기 12 기반
        period = min(12, len(seasonal))
        effects: dict[int, list[float]] = {}
        for i, val in enumerate(seasonal.values):
            month = (i % period) + 1
            effects.setdefault(month, []).append(val)

        return {m: round(float(np.mean(vals)), 4) for m, vals in effects.items()}

    @staticmethod
    def _empty_decomposition(category: str = "") -> dict[str, object]:
        """빈 분해 결과 반환."""
        return {
            "trend": None,
            "seasonal": None,
            "residual": None,
            "original": None,
            "seasonal_peaks": [],
            "seasonal_troughs": [],
            "trend_direction": "stable",
            "seasonality_strength": 0.0,
            "category": category,
        }
