"""SHAP 기반 모델 해석 모듈.

XGBoost 모델의 예측 결과를 SHAP으로 분석하여
피처별 기여도, 한국어 설명, 시각화 데이터를 제공한다.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ml.conversion_model import ConversionModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 피처 한국어 라벨
# ---------------------------------------------------------------------------
_FEATURE_LABELS: dict[str, str] = {
    "PAYEND_CVR_LAG1": "전월 결제 전환율",
    "PAYEND_CVR_LAG2": "2개월 전 결제 전환율",
    "PAYEND_CVR_LAG3": "3개월 전 결제 전환율",
    "PAYEND_CVR_MA3": "3개월 평균 전환율",
    "PAYEND_CVR_MA6": "6개월 평균 전환율",
    "PAYEND_CVR_STD3": "3개월 전환율 변동성",
    "CONTRACT_COUNT_LAG1": "전월 계약 건수",
    "CONTRACT_COUNT_LAG2": "2개월 전 계약 건수",
    "CHANNEL_HISTORICAL_CVR": "채널 역대 전환율",
    "CATEGORY_AVG_CVR": "카테고리 평균 전환율",
    "AVG_NET_SALES_LAG1": "전월 객단가",
    "OPEN_CVR_LAG1": "전월 개통 전환율",
    "MONTH_OF_YEAR": "월(계절성)",
    "QUARTER": "분기",
    "CATEGORY_ENCODED": "카테고리 코드",
    "CHANNEL_ENCODED": "채널 코드",
}

_LABEL_MAP_INV: dict[int, str] = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

_CLASS_LABELS_KR: dict[str, str] = {
    "LOW": "낮음",
    "MEDIUM": "보통",
    "HIGH": "높음",
}

# SHAP 해석 방향 (양수 = 전환율 상승 요인)
_DIRECTION_POSITIVE = "상승"
_DIRECTION_NEGATIVE = "하락"


class ModelExplainer:
    """SHAP 기반 XGBoost 모델 해석기.

    ConversionModel의 예측 결과에 대해 피처별 기여도를 분석하고
    한국어 설명 텍스트 및 시각화 데이터를 생성한다.
    """

    def __init__(self, model: ConversionModel) -> None:
        """ModelExplainer 초기화.

        Args:
            model: 학습 완료된 ConversionModel 인스턴스
        """
        self._model = model
        self._shap_explainer: Any | None = None
        self._background_data: pd.DataFrame | None = None
        logger.info("ModelExplainer 초기화 완료")

    # ------------------------------------------------------------------
    # 예측 해석
    # ------------------------------------------------------------------

    def explain_prediction(
        self,
        category: str,
        channel: str | None = None,
    ) -> dict[str, Any]:
        """특정 예측에 대한 SHAP 값을 계산.

        Args:
            category: 상품 카테고리
            channel: 채널명 (None이면 카테고리 대표)

        Returns:
            SHAP 분석 결과:
                shap_values: {feature: [shap_low, shap_med, shap_high]}
                base_values: [base_low, base_med, base_high]
                prediction: 예측 결과 딕셔너리
                top_features: 영향력 상위 피처 리스트
        """
        explainer = self._ensure_explainer()
        if explainer is None:
            return self._empty_explanation(category, channel)

        feature_row = self._model._latest_features(category, channel)
        if feature_row is None or feature_row.empty:
            return self._empty_explanation(category, channel)

        x_input = feature_row[self._model._feature_columns].fillna(0)

        try:
            shap_values = explainer.shap_values(x_input)
        except Exception as exc:
            logger.warning("SHAP values 계산 실패: %s", exc)
            return self._empty_explanation(category, channel)

        prediction = self._model.predict(category, channel)

        # SHAP values를 피처별 딕셔너리로 변환
        feature_shap: dict[str, list[float]] = {}
        for idx, col in enumerate(self._model._feature_columns):
            label = _FEATURE_LABELS.get(col, col)
            if isinstance(shap_values, list):
                # multi-class: shap_values[class][sample, feature]
                feature_shap[label] = [
                    round(float(shap_values[c][0, idx]), 6)
                    for c in range(len(shap_values))
                ]
            else:
                # single output
                feature_shap[label] = [round(float(shap_values[0, idx]), 6)]

        # 기저값
        base_values = self._extract_base_values(explainer)

        # 영향력 상위 피처
        top_features = self._rank_features_by_impact(feature_shap, prediction)

        return {
            "shap_values": feature_shap,
            "base_values": base_values,
            "prediction": prediction,
            "top_features": top_features,
        }

    # ------------------------------------------------------------------
    # 피처 중요도
    # ------------------------------------------------------------------

    def feature_importance(self) -> pd.DataFrame:
        """전체 학습 데이터에 대한 SHAP 기반 피처 중요도를 반환.

        Returns:
            DataFrame: [FEATURE, FEATURE_KR, IMPORTANCE, RANK]
        """
        explainer = self._ensure_explainer()
        if explainer is None:
            return pd.DataFrame(
                columns=["FEATURE", "FEATURE_KR", "IMPORTANCE", "RANK"]
            )

        background = self._get_background_data()
        if background.empty:
            return pd.DataFrame(
                columns=["FEATURE", "FEATURE_KR", "IMPORTANCE", "RANK"]
            )

        try:
            shap_values = explainer.shap_values(background)
        except Exception as exc:
            logger.warning("SHAP feature importance 계산 실패: %s", exc)
            return self._fallback_feature_importance()

        # multi-class면 모든 클래스의 절대값 평균
        if isinstance(shap_values, list):
            abs_mean = np.mean(
                [np.abs(sv).mean(axis=0) for sv in shap_values], axis=0
            )
        else:
            abs_mean = np.abs(shap_values).mean(axis=0)

        records: list[dict[str, Any]] = []
        for idx, col in enumerate(self._model._feature_columns):
            records.append({
                "FEATURE": col,
                "FEATURE_KR": _FEATURE_LABELS.get(col, col),
                "IMPORTANCE": round(float(abs_mean[idx]), 6),
            })

        df = pd.DataFrame(records)
        df = df.sort_values("IMPORTANCE", ascending=False).reset_index(drop=True)
        df["RANK"] = df.index + 1
        return df

    # ------------------------------------------------------------------
    # 한국어 설명 텍스트 생성
    # ------------------------------------------------------------------

    def generate_explanation_text(
        self,
        category: str,
        channel: str | None = None,
    ) -> str:
        """예측 결과에 대한 한국어 자연어 설명을 생성.

        Args:
            category: 상품 카테고리
            channel: 채널명

        Returns:
            한국어 설명 문자열
        """
        explanation = self.explain_prediction(category, channel)
        prediction = explanation.get("prediction", {})
        top_features = explanation.get("top_features", [])

        predicted_class = prediction.get("predicted_class", "UNKNOWN")
        confidence = prediction.get("confidence", 0.0)
        channel_display = channel or "전체"
        class_kr = _CLASS_LABELS_KR.get(predicted_class, predicted_class)

        lines: list[str] = [
            f"[{category}] {channel_display} 채널 전환율 예측 분석",
            f"",
            f"예측 결과: 다음 달 전환율이 '{class_kr}'일 확률이 "
            f"{confidence * 100:.1f}%입니다.",
            f"",
        ]

        if top_features:
            lines.append("주요 영향 요인:")
            for rank, feat_info in enumerate(top_features[:5], 1):
                name = feat_info["feature"]
                direction = feat_info["direction"]
                impact = feat_info["impact"]
                lines.append(
                    f"  {rank}. {name}: 전환율 {direction} 요인 "
                    f"(영향도 {impact:.4f})"
                )

        if predicted_class == "HIGH":
            lines.extend([
                "",
                "권장사항: 현재 채널 성과가 양호합니다. "
                "기존 전략을 유지하되, 계약 건수 확대를 검토하세요.",
            ])
        elif predicted_class == "LOW":
            lines.extend([
                "",
                "권장사항: 전환율 하락이 예상됩니다. "
                "채널 운영 효율화 또는 예산 재배분을 검토하세요.",
            ])
        else:
            lines.extend([
                "",
                "권장사항: 전환율이 보통 수준으로 예측됩니다. "
                "상위 영향 요인을 중심으로 개선 기회를 탐색하세요.",
            ])

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 시각화 데이터
    # ------------------------------------------------------------------

    def plot_data(
        self,
        category: str,
        channel: str | None = None,
    ) -> dict[str, Any]:
        """Plotly 시각화용 waterfall 및 bar 데이터를 반환.

        Args:
            category: 상품 카테고리
            channel: 채널명

        Returns:
            시각화 데이터:
                waterfall_data: Plotly waterfall chart용 데이터
                bar_data: Plotly bar chart용 데이터
        """
        explanation = self.explain_prediction(category, channel)
        shap_dict = explanation.get("shap_values", {})
        prediction = explanation.get("prediction", {})
        predicted_class = prediction.get("predicted_class", "MEDIUM")

        # 예측 클래스에 해당하는 SHAP값 인덱스
        class_idx = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}.get(predicted_class, 1)

        # waterfall 데이터: 각 피처의 기여도
        waterfall_items: list[dict[str, Any]] = []
        for feature_name, shap_vals in shap_dict.items():
            val = shap_vals[class_idx] if len(shap_vals) > class_idx else 0.0
            waterfall_items.append({
                "feature": feature_name,
                "shap_value": round(val, 6),
            })

        # 절대값 기준 정렬
        waterfall_items.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        waterfall_data = {
            "features": [item["feature"] for item in waterfall_items],
            "values": [item["shap_value"] for item in waterfall_items],
            "base_value": (
                explanation.get("base_values", [0.0, 0.0, 0.0])[class_idx]
                if len(explanation.get("base_values", [])) > class_idx
                else 0.0
            ),
            "predicted_class": predicted_class,
        }

        # bar 데이터: 전체 피처 중요도
        importance_df = self.feature_importance()
        bar_data = {
            "features": importance_df["FEATURE_KR"].tolist()
            if not importance_df.empty else [],
            "importances": importance_df["IMPORTANCE"].tolist()
            if not importance_df.empty else [],
        }

        return {
            "waterfall_data": waterfall_data,
            "bar_data": bar_data,
        }

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _ensure_explainer(self) -> Any | None:
        """SHAP TreeExplainer가 초기화되어 있는지 확인."""
        if self._shap_explainer is not None:
            return self._shap_explainer

        underlying_model = self._get_underlying_model()
        if underlying_model is None:
            logger.warning("SHAP 해석기 생성 실패: 학습된 모델 없음")
            return None

        try:
            import shap
            background = self._get_background_data()
            if background.empty:
                self._shap_explainer = shap.TreeExplainer(underlying_model)
            else:
                self._shap_explainer = shap.TreeExplainer(
                    underlying_model, data=background
                )
            logger.info("SHAP TreeExplainer 초기화 완료")
            return self._shap_explainer

        except ImportError:
            logger.error("shap 패키지가 설치되지 않았습니다")
            return None
        except Exception as exc:
            logger.warning("SHAP 해석기 초기화 실패: %s", exc)
            return None

    def _get_underlying_model(self) -> Any | None:
        """ConversionModel에서 실제 XGBoost 모델 객체를 추출.

        Snowpark ML 래퍼 모델의 다양한 내부 구조에 대응하며,
        모든 추출 경로가 실패하면 경량 sklearn 모델을 학습하여 반환한다.
        """
        if self._model is None or not hasattr(self._model, "_model"):
            return None
        if self._model._model is None:
            return self._train_shap_fallback()

        inner = self._model._model

        # sklearn 호환 모델이면 바로 반환
        if hasattr(inner, "feature_importances_") and hasattr(inner, "predict"):
            return inner

        if self._model._is_snowpark_model:
            # Snowpark ML 모델에서 underlying estimator 추출
            try:
                to_sklearn = getattr(inner, "to_sklearn", None)
                if to_sklearn is not None:
                    return to_sklearn()
            except Exception:
                pass

            sklearn_obj = getattr(inner, "_sklearn_object", None)
            if sklearn_obj is not None:
                return sklearn_obj

            # 추가 추출 경로 탐색
            for attr in ("_sklearn_estimator", "estimator_", "_model", "model_"):
                candidate = getattr(inner, attr, None)
                if candidate is not None and hasattr(candidate, "predict"):
                    return candidate

            logger.warning(
                "Snowpark ML 모델에서 underlying estimator 추출 실패 — 폴백 학습 시도"
            )
            return self._train_shap_fallback()

        return inner

    def _train_shap_fallback(self) -> Any | None:
        """SHAP 분석 전용 경량 sklearn XGBClassifier를 학습하여 반환.

        원본 Snowpark ML 모델에서 estimator를 추출할 수 없을 때
        학습 데이터를 사용하여 동일 피처 구조의 경량 모델을 학습한다.
        """
        try:
            from xgboost import XGBClassifier

            train_df = getattr(self._model, "_train_df", None)
            if train_df is None or train_df.empty:
                logger.warning("SHAP 폴백: 학습 데이터 없음")
                return None

            feature_cols = getattr(self._model, "_feature_columns", [])
            target_col = "TARGET_CLASS"

            if target_col not in train_df.columns:
                logger.warning("SHAP 폴백: %s 컬럼 없음", target_col)
                return None

            available_cols = [c for c in feature_cols if c in train_df.columns]
            if not available_cols:
                logger.warning("SHAP 폴백: 사용 가능한 피처 컬럼 없음")
                return None

            X = train_df[available_cols].fillna(0)
            label_map = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
            y = train_df[target_col].map(label_map).fillna(1).astype(int)

            model = XGBClassifier(
                max_depth=3, n_estimators=50, random_state=42
            )
            model.fit(X, y, verbose=False)
            logger.info("SHAP 폴백: sklearn XGBClassifier 학습 완료")
            return model
        except ImportError:
            logger.warning("SHAP 폴백: xgboost 패키지 미설치")
            return None
        except Exception as exc:
            logger.warning("SHAP 폴백 학습 실패: %s", exc)
            return None

    def _get_background_data(self) -> pd.DataFrame:
        """SHAP 배경 데이터(학습 데이터 샘플)를 반환."""
        if self._background_data is not None:
            return self._background_data

        if self._model._train_df is not None:
            cols = [
                c for c in self._model._feature_columns
                if c in self._model._train_df.columns
            ]
            bg = self._model._train_df[cols].fillna(0)
            # SHAP 계산 속도를 위해 최대 200행 샘플링
            if len(bg) > 200:
                bg = bg.sample(n=200, random_state=42)
            self._background_data = bg
            return bg

        return pd.DataFrame()

    def _extract_base_values(self, explainer: Any) -> list[float]:
        """SHAP explainer에서 base values(기저값)를 추출."""
        try:
            base = explainer.expected_value
            if isinstance(base, (list, np.ndarray)):
                return [round(float(v), 6) for v in base]
            return [round(float(base), 6)]
        except Exception:
            return [0.0, 0.0, 0.0]

    def _rank_features_by_impact(
        self,
        feature_shap: dict[str, list[float]],
        prediction: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """SHAP 값을 기반으로 영향력 상위 피처를 정렬."""
        predicted_class = prediction.get("predicted_class", "MEDIUM")
        class_idx = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}.get(predicted_class, 1)

        items: list[dict[str, Any]] = []
        for feature_name, shap_vals in feature_shap.items():
            val = shap_vals[class_idx] if len(shap_vals) > class_idx else 0.0
            items.append({
                "feature": feature_name,
                "shap_value": val,
                "impact": abs(val),
                "direction": _DIRECTION_POSITIVE if val >= 0 else _DIRECTION_NEGATIVE,
            })

        items.sort(key=lambda x: x["impact"], reverse=True)
        return items

    def _fallback_feature_importance(self) -> pd.DataFrame:
        """SHAP 계산 실패 시 모델 자체 피처 중요도를 fallback으로 반환."""
        raw_importance = self._model._extract_feature_importance()
        if not raw_importance:
            return pd.DataFrame(
                columns=["FEATURE", "FEATURE_KR", "IMPORTANCE", "RANK"]
            )

        records = [
            {
                "FEATURE": feat,
                "FEATURE_KR": _FEATURE_LABELS.get(feat, feat),
                "IMPORTANCE": imp,
            }
            for feat, imp in raw_importance.items()
        ]

        df = pd.DataFrame(records)
        df = df.sort_values("IMPORTANCE", ascending=False).reset_index(drop=True)
        df["RANK"] = df.index + 1
        return df

    @staticmethod
    def _empty_explanation(
        category: str, channel: str | None
    ) -> dict[str, Any]:
        """데이터 부족 시 빈 설명 결과를 반환."""
        return {
            "shap_values": {},
            "base_values": [0.0, 0.0, 0.0],
            "prediction": {
                "category": category,
                "channel": channel or "전체",
                "predicted_class": "UNKNOWN",
                "confidence": 0.0,
            },
            "top_features": [],
        }
