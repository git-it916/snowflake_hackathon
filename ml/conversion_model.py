"""XGBoost 기반 채널 전환율 예측 모델.

V04 채널 성과 데이터(7,845행)로부터 채널별 다음 달 PAYEND_CVR을
HIGH/MEDIUM/LOW 3-class로 예측한다.

Snowpark ML XGBClassifier를 우선 시도하고,
실패 시 sklearn XGBClassifier로 자동 fallback한다.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------
_FEATURE_STORE_TABLE = "ANALYTICS.ML_FEATURE_STORE"
_STG_CHANNEL_TABLE = "STAGING.STG_CHANNEL"
_TRAIN_CUTOFF = "2026-01-01"

_LABEL_MAP: dict[str, int] = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
_LABEL_MAP_INV: dict[int, str] = {v: k for k, v in _LABEL_MAP.items()}

_FEATURE_COLUMNS: list[str] = [
    # Lag 피처
    "PAYEND_CVR_LAG1",
    "PAYEND_CVR_LAG2",
    "PAYEND_CVR_LAG3",
    "CONTRACT_COUNT_LAG1",
    "CONTRACT_COUNT_LAG2",
    "AVG_NET_SALES_LAG1",
    "OPEN_CVR_LAG1",
    # 이동평균 / 변동성
    "PAYEND_CVR_MA3",
    "PAYEND_CVR_MA6",
    "CONTRACT_MA3",
    "PAYEND_CVR_STD3",
    "CATEGORY_HISTORICAL_CVR",
    # 채널 다양성 지표 (NEW)
    "N_CHANNELS",
    "CHANNEL_HHI",
    "TOP_CHANNEL_SHARE",
    "N_CHANNELS_LAG1",
    "HHI_LAG1",
    # 퍼널 교차 피처 (NEW)
    "FUNNEL_OVERALL_CVR",
    "FUNNEL_TOTAL_COUNT",
    # 시간 피처
    "MONTH_OF_YEAR",
    "QUARTER",
    # 인코딩
    "CATEGORY_ENCODED",
]

_TARGET_COLUMN = "TARGET_CLASS"

# XGBoost 하이퍼파라미터 (Snowpark ML과 sklearn 공용)
_XGB_PARAMS: dict[str, Any] = {
    "max_depth": 4,
    "n_estimators": 200,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 2.0,
    "min_child_weight": 5,
    "random_state": 42,
}

# Snowpark ML에서 제외할 파라미터 키
_SNOWPARK_EXCLUDED_PARAMS = {"num_class", "objective", "eval_metric", "use_label_encoder"}

_CV_SPLITS = 3


class ConversionModel:
    """채널 전환율(PAYEND_CVR) 3-class 예측 모델.

    ML_FEATURE_STORE 또는 STG_CHANNEL(fallback)로부터 피처를 로드하여
    XGBoost classifier를 학습한다.
    """

    def __init__(self, session: "Any | None" = None) -> None:
        """ConversionModel 초기화.

        Args:
            session: Snowpark 세션. None이면 get_streamlit_session() 사용.
        """
        if session is None:
            from config.settings import get_streamlit_session
            session = get_streamlit_session()

        self._session = session
        self._model: Any | None = None
        self._feature_columns: list[str] = list(_FEATURE_COLUMNS)
        self._is_snowpark_model: bool = False
        self._train_df: pd.DataFrame | None = None
        logger.info("ConversionModel 초기화 완료")

    # ------------------------------------------------------------------
    # 학습
    # ------------------------------------------------------------------

    def train(self, train_df: pd.DataFrame | None = None) -> dict[str, Any]:
        """XGBoost 3-class 모델을 학습하고 평가 결과를 반환.

        Snowpark ML XGBClassifier를 우선 시도하며,
        실패 시 sklearn XGBClassifier로 자동 fallback한다.

        Args:
            train_df: 학습 데이터. None이면 ML_FEATURE_STORE에서 자동 로드.

        Returns:
            학습 결과 딕셔너리:
                accuracy: 정확도
                f1_macro: F1 macro 점수
                classification_report: 분류 리포트 문자열
                feature_importance: {feature: importance} 딕셔너리
        """
        logger.info("모델 학습 시작")

        # 1. 데이터 로드
        df = train_df if train_df is not None else self._load_training_data()
        if df.empty:
            logger.error("학습 데이터가 비어 있습니다")
            return {
                "accuracy": 0.0,
                "f1_macro": 0.0,
                "classification_report": "학습 데이터 없음",
                "feature_importance": {},
            }

        self._train_df = df.copy()

        # 2. 피처 / 타겟 분리
        available_features = [c for c in self._feature_columns if c in df.columns]
        if len(available_features) < 3:
            logger.error(
                "사용 가능한 피처가 부족합니다: %d개", len(available_features)
            )
            return {
                "accuracy": 0.0,
                "f1_macro": 0.0,
                "classification_report": "피처 부족",
                "feature_importance": {},
            }

        self._feature_columns = available_features
        x_data = df[available_features].copy()
        y_data = df[_TARGET_COLUMN].map(_LABEL_MAP).astype(int)

        # NaN 처리
        x_data = x_data.fillna(0)

        # 데이터가 너무 적으면 CV 생략
        n_samples = len(x_data)
        effective_splits = min(_CV_SPLITS, max(2, n_samples // 10))

        # 3. TimeSeriesSplit CV
        tscv = TimeSeriesSplit(n_splits=effective_splits)
        cv_scores: list[float] = []

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(x_data)):
            x_train_fold = x_data.iloc[train_idx]
            y_train_fold = y_data.iloc[train_idx]
            x_val_fold = x_data.iloc[val_idx]
            y_val_fold = y_data.iloc[val_idx]

            fold_model = self._fit_model(x_train_fold, y_train_fold)
            preds = self._predict_with_model(fold_model, x_val_fold)
            fold_f1 = f1_score(y_val_fold, preds, average="macro", zero_division=0)
            cv_scores.append(fold_f1)
            logger.info("Fold %d/%d F1: %.4f", fold_idx + 1, effective_splits, fold_f1)

        # 4. 전체 데이터로 최종 학습
        self._model = self._fit_model(x_data, y_data)

        # 5. 전체 데이터 평가 (in-sample, CV 결과와 함께 보고)
        final_preds = self._predict_with_model(self._model, x_data)
        acc = accuracy_score(y_data, final_preds)
        f1 = float(np.mean(cv_scores)) if cv_scores else f1_score(
            y_data, final_preds, average="macro", zero_division=0
        )

        target_names = [_LABEL_MAP_INV[i] for i in sorted(_LABEL_MAP.values())]
        report = classification_report(
            y_data,
            final_preds,
            target_names=target_names,
            zero_division=0,
        )

        # 6. Feature importance
        importance = self._extract_feature_importance()

        result = {
            "accuracy": round(acc, 4),
            "f1_macro": round(f1, 4),
            "cv_scores": [round(s, 4) for s in cv_scores],
            "classification_report": report,
            "feature_importance": importance,
        }

        logger.info(
            "모델 학습 완료: accuracy=%.4f, f1_macro(CV)=%.4f",
            acc,
            f1,
        )
        return result

    # ------------------------------------------------------------------
    # 예측
    # ------------------------------------------------------------------

    def predict(
        self,
        category: str,
        channel: str | None = None,
        features: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """단일 채널/카테고리에 대한 전환율 클래스를 예측.

        Args:
            category: 상품 카테고리
            channel: 채널명 (None이면 카테고리 전체 대표 예측)
            features: 직접 입력할 피처 딕셔너리 (None이면 최근 데이터로 추출)

        Returns:
            예측 결과:
                category, channel, ym,
                prob_high, prob_medium, prob_low,
                predicted_class, confidence
        """
        if self._model is None:
            logger.warning("모델이 학습되지 않았습니다. train() 자동 실행")
            self.train()

        if self._model is None:
            return self._empty_prediction(category, channel)

        feature_row = (
            self._features_from_dict(features)
            if features is not None
            else self._latest_features(category, channel)
        )

        if feature_row is None or feature_row.empty:
            return self._empty_prediction(category, channel)

        x_input = feature_row[self._feature_columns].fillna(0)
        probas = self._predict_proba_with_model(self._model, x_input)
        pred_label = int(np.argmax(probas[0]))

        return {
            "category": category,
            "channel": channel or "전체",
            "ym": self._get_latest_ym(),
            "prob_high": round(float(probas[0][2]), 4),
            "prob_medium": round(float(probas[0][1]), 4),
            "prob_low": round(float(probas[0][0]), 4),
            "predicted_class": _LABEL_MAP_INV[pred_label],
            "confidence": round(float(np.max(probas[0])), 4),
        }

    def predict_all(self) -> pd.DataFrame:
        """주요 채널에 대해 일괄 예측을 수행.

        Returns:
            채널별 예측 결과 DataFrame
        """
        if self._model is None:
            logger.warning("모델이 학습되지 않았습니다. train() 자동 실행")
            self.train()

        df = self._load_latest_features_all()
        if df.empty:
            return pd.DataFrame()

        results: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            category = str(row.get("CATEGORY", ""))
            channel = str(row.get("CHANNEL", ""))
            pred = self.predict(category, channel)
            results.append(pred)

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Fallback 피처 엔지니어링
    # ------------------------------------------------------------------

    def _build_fallback_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """STG_CHANNEL 원본 데이터로부터 수동 피처 엔지니어링.

        ML_FEATURE_STORE가 없을 때 사용하는 fallback 경로.
        pandas 기반으로 lag, rolling, interaction 피처를 계산한다.

        Args:
            df: STG_CHANNEL에서 로드한 원본 DataFrame

        Returns:
            피처가 추가된 DataFrame
        """
        logger.info("Fallback 피처 엔지니어링 시작 (%d rows)", len(df))

        # 카테고리 레벨로 합산 (채널 제거)
        if "CATEGORY" in df.columns and "CHANNEL" in df.columns:
            df = df.groupby(["CATEGORY", "YEAR_MONTH"]).agg(
                PAYEND_CVR=("PAYEND_CVR", "mean"),
                OPEN_CVR=("OPEN_CVR", "mean"),
                CONTRACT_COUNT=("CONTRACT_COUNT", "sum"),
                AVG_NET_SALES=("AVG_NET_SALES", "mean"),
            ).reset_index()
        df = df[df["PAYEND_CVR"] > 0].copy()  # CVR=0 제거
        df = df.sort_values(["CATEGORY", "YEAR_MONTH"]).copy()
        group_cols = ["CATEGORY"]

        # Lag 피처
        for col_name, lag_n in [
            ("PAYEND_CVR", 1), ("PAYEND_CVR", 2), ("PAYEND_CVR", 3),
            ("CONTRACT_COUNT", 1), ("CONTRACT_COUNT", 2),
            ("AVG_NET_SALES", 1), ("OPEN_CVR", 1),
        ]:
            if col_name in df.columns:
                df[f"{col_name}_LAG{lag_n}"] = (
                    df.groupby(group_cols)[col_name].shift(lag_n)
                )

        # Rolling 피처
        if "PAYEND_CVR" in df.columns:
            df["PAYEND_CVR_MA3"] = (
                df.groupby(group_cols)["PAYEND_CVR"]
                .transform(lambda s: s.rolling(3, min_periods=1).mean())
            )
            df["PAYEND_CVR_MA6"] = (
                df.groupby(group_cols)["PAYEND_CVR"]
                .transform(lambda s: s.rolling(6, min_periods=1).mean())
            )
            df["PAYEND_CVR_STD3"] = (
                df.groupby(group_cols)["PAYEND_CVR"]
                .transform(lambda s: s.rolling(3, min_periods=1).std())
            )

        # Historical / Category 평균
        if "PAYEND_CVR" in df.columns:
            df["CATEGORY_HISTORICAL_CVR"] = (
                df.groupby(group_cols)["PAYEND_CVR"]
                .transform(lambda s: s.expanding().mean().shift(1))
            )

        # 시간 피처
        if "YEAR_MONTH" in df.columns:
            ym_dt = pd.to_datetime(df["YEAR_MONTH"], format="%Y-%m", errors="coerce")
            df["MONTH_OF_YEAR"] = ym_dt.dt.month
            df["QUARTER"] = ym_dt.dt.quarter

        # 인코딩
        if "CATEGORY" in df.columns:
            df["CATEGORY_ENCODED"] = df["CATEGORY"].astype("category").cat.codes
        # CHANNEL_ENCODED 제거 (카테고리 레벨 합산이므로 불필요)

        # 타겟 변수 — 채널-카테고리별 퍼센타일 기준 구간화
        if "PAYEND_CVR" in df.columns:
            df["NEXT_CVR"] = df.groupby(group_cols)["PAYEND_CVR"].shift(-1)

            df["TARGET_CLASS"] = df.groupby(group_cols)["NEXT_CVR"].transform(
                lambda s: pd.cut(
                    s,
                    bins=[-np.inf, s.quantile(0.33), s.quantile(0.67), np.inf],
                    labels=["LOW", "MEDIUM", "HIGH"],
                )
            ).astype(str)

            df = df.drop(columns=["NEXT_CVR"])

        # NULL lag 필터
        df = df.dropna(subset=["PAYEND_CVR_LAG3", "TARGET_CLASS"])
        df = df[df["TARGET_CLASS"] != "nan"]

        logger.info("Fallback 피처 엔지니어링 완료: %d rows", len(df))
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 내부 헬퍼 — 데이터 로드
    # ------------------------------------------------------------------

    def _load_training_data(self) -> pd.DataFrame:
        """ML_FEATURE_STORE 또는 STG_CHANNEL(fallback)에서 학습 데이터를 로드."""
        # ML_FEATURE_STORE 시도
        try:
            df = (
                self._session.table(_FEATURE_STORE_TABLE)
                .filter(f"YEAR_MONTH < '{_TRAIN_CUTOFF}'")
                .to_pandas()
            )
            if len(df) > 0:
                logger.info("ML_FEATURE_STORE에서 학습 데이터 로드: %d rows", len(df))
                return df
        except Exception as exc:
            logger.warning("ML_FEATURE_STORE 로드 실패: %s", exc)

        # Fallback: STG_CHANNEL에서 직접 로드 후 수동 피처 생성
        logger.info("STG_CHANNEL fallback 경로로 데이터 로드")
        try:
            raw_df = self._session.table(_STG_CHANNEL_TABLE).to_pandas()
            if raw_df.empty:
                logger.error("STG_CHANNEL 테이블이 비어 있습니다")
                return pd.DataFrame()

            full_df = self._build_fallback_features(raw_df)
            return full_df[full_df["YEAR_MONTH"] < _TRAIN_CUTOFF].reset_index(drop=True)
        except Exception as exc:
            logger.error("STG_CHANNEL 로드 실패: %s", exc)
            return pd.DataFrame()

    def _load_latest_features_all(self) -> pd.DataFrame:
        """가장 최근 월의 채널별 피처를 로드."""
        try:
            df = self._session.table(_FEATURE_STORE_TABLE).to_pandas()
            if df.empty:
                return pd.DataFrame()

            latest_ym = df["YEAR_MONTH"].max()
            return df[df["YEAR_MONTH"] == latest_ym].reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def _latest_features(
        self, category: str, channel: str | None
    ) -> pd.DataFrame | None:
        """특정 카테고리/채널의 최신 피처 행을 반환."""
        try:
            df = self._session.table(_FEATURE_STORE_TABLE).to_pandas()
            if df.empty:
                return None

            mask = df["CATEGORY"] == category
            if channel is not None:
                mask = mask & (df["CHANNEL"] == channel)

            filtered = df[mask]
            if filtered.empty:
                return None

            latest_ym = filtered["YEAR_MONTH"].max()
            row = filtered[filtered["YEAR_MONTH"] == latest_ym].head(1)
            return row
        except Exception:
            return None

    def _features_from_dict(self, features: dict[str, float]) -> pd.DataFrame:
        """딕셔너리로부터 피처 DataFrame을 생성."""
        row = {col: features.get(col, 0.0) for col in self._feature_columns}
        return pd.DataFrame([row])

    def _get_latest_ym(self) -> str:
        """학습 데이터의 가장 최근 YEAR_MONTH를 반환."""
        if self._train_df is not None and "YEAR_MONTH" in self._train_df.columns:
            return str(self._train_df["YEAR_MONTH"].max())
        return "unknown"

    # ------------------------------------------------------------------
    # 내부 헬퍼 — 모델 학습/예측
    # ------------------------------------------------------------------

    def _fit_model(self, x_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Snowpark ML XGBClassifier 우선 시도, 실패 시 sklearn fallback.

        Args:
            x_train: 피처 DataFrame
            y_train: 정수 인코딩된 타겟 Series

        Returns:
            학습된 모델 객체
        """
        # Snowpark ML 시도
        model = self._try_snowpark_ml(x_train, y_train)
        if model is not None:
            self._is_snowpark_model = True
            return model

        # sklearn fallback
        logger.info("sklearn XGBClassifier fallback 사용")
        return self._fit_sklearn(x_train, y_train)

    def _try_snowpark_ml(
        self, x_train: pd.DataFrame, y_train: pd.Series
    ) -> Any | None:
        """Snowpark ML XGBClassifier 학습을 시도.

        Snowpark ML은 정수 라벨이 필요하며,
        num_class, objective, eval_metric, use_label_encoder 파라미터를 제외해야 한다.

        Returns:
            학습된 Snowpark ML 모델 또는 None (실패 시)
        """
        try:
            from snowflake.ml.modeling.xgboost import XGBClassifier as SpXGBClassifier

            # Snowpark ML 용 파라미터 (제외 키 필터링)
            sp_params = {
                k: v for k, v in _XGB_PARAMS.items()
                if k not in _SNOWPARK_EXCLUDED_PARAMS
            }

            # Snowpark ML은 Snowpark DataFrame 또는 pandas를 받을 수 있음
            # 컬럼명 지정 필수
            train_data = x_train.copy()
            train_data[_TARGET_COLUMN] = y_train.values

            sp_model = SpXGBClassifier(
                input_cols=self._feature_columns,
                label_cols=[_TARGET_COLUMN],
                output_cols=["PREDICTED_CLASS"],
                **sp_params,
            )
            sp_model.fit(train_data)
            logger.info("Snowpark ML XGBClassifier 학습 성공")
            return sp_model

        except ImportError:
            logger.info("snowflake.ml.modeling.xgboost 미설치 → sklearn fallback")
            return None
        except Exception as exc:
            logger.warning(
                "Snowpark ML XGBClassifier 학습 실패: %s → sklearn fallback", exc
            )
            return None

    def _fit_sklearn(self, x_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """sklearn XGBClassifier로 학습."""
        from xgboost import XGBClassifier

        params = {
            **_XGB_PARAMS,
            "objective": "multi:softprob",
            "num_class": len(_LABEL_MAP),
            "eval_metric": "mlogloss",
        }

        model = XGBClassifier(**params)
        model.fit(x_train, y_train)
        self._is_snowpark_model = False
        logger.info("sklearn XGBClassifier 학습 성공")
        return model

    def _predict_with_model(self, model: Any, x_data: pd.DataFrame) -> np.ndarray:
        """모델 타입에 따라 예측을 수행하여 정수 라벨 배열 반환."""
        if self._is_snowpark_model:
            try:
                input_df = x_data.copy()
                result = model.predict(input_df)
                return result["PREDICTED_CLASS"].values.astype(int)
            except Exception as exc:
                logger.warning("Snowpark ML predict 실패: %s", exc)

        return model.predict(x_data)

    def _predict_proba_with_model(
        self, model: Any, x_data: pd.DataFrame
    ) -> np.ndarray:
        """모델 타입에 따라 확률 예측을 수행."""
        if self._is_snowpark_model:
            try:
                input_df = x_data.copy()
                result = model.predict_proba(input_df)
                # Snowpark ML은 컬럼별로 확률을 반환할 수 있음
                proba_cols = [c for c in result.columns if "PROB" in c.upper()]
                if proba_cols:
                    return result[proba_cols].values
            except Exception as exc:
                logger.warning("Snowpark ML predict_proba 실패: %s", exc)

        # sklearn fallback 또는 직접 predict_proba
        try:
            return model.predict_proba(x_data)
        except AttributeError:
            # predict_proba 미지원 시 원핫 fallback
            preds = self._predict_with_model(model, x_data)
            n_classes = len(_LABEL_MAP)
            probas = np.zeros((len(preds), n_classes))
            for i, p in enumerate(preds):
                probas[i, int(p)] = 1.0
            return probas

    def _extract_feature_importance(self) -> dict[str, float]:
        """학습된 모델로부터 피처 중요도를 추출."""
        if self._model is None:
            return {}

        try:
            if self._is_snowpark_model:
                # Snowpark ML XGBClassifier에서 피처 중요도 추출
                # 방법 1: feature_importances_ 속성
                imp = getattr(self._model, "feature_importances_", None)
                if imp is not None:
                    return {
                        col: round(float(v), 6)
                        for col, v in zip(self._feature_columns, imp)
                    }
                # 방법 2: 학습 데이터로 predict 후 균등 분배 (폴백)
                logger.info("Snowpark ML 피처 중요도 속성 없음 — 균등 분배 사용")
                n = len(self._feature_columns)
                return {col: round(1.0 / n, 6) for col in self._feature_columns}
            else:
                importances = self._model.feature_importances_
                return {
                    col: round(float(imp), 6)
                    for col, imp in zip(self._feature_columns, importances)
                }
        except Exception as exc:
            logger.warning("피처 중요도 추출 실패: %s", exc)

        return {}

    # ------------------------------------------------------------------
    # 헬퍼 — 빈 예측
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_prediction(category: str, channel: str | None) -> dict[str, Any]:
        """데이터 부족 시 반환할 빈 예측 결과."""
        return {
            "category": category,
            "channel": channel or "전체",
            "ym": "unknown",
            "prob_high": 0.0,
            "prob_medium": 0.0,
            "prob_low": 0.0,
            "predicted_class": "UNKNOWN",
            "confidence": 0.0,
        }
