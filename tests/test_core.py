"""Core unit tests for the Snowflake hackathon project.

All tests run offline -- no Snowflake connection required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agents.schemas import TransitionMatrixValidation
from components.utils import drop_incomplete_month, validate_columns
from config.constants import FUNNEL_STAGES, PRODUCT_CATEGORIES
from ml.conversion_model import _FEATURE_COLUMNS, _LABEL_MAP
from ml.model_validation import (
    FeatureValidationResult,
    ModelMetrics,
    compute_metrics,
    validate_features,
)


# -----------------------------------------------------------------------
# 1. Markov transition matrix -- row sums
# -----------------------------------------------------------------------

class TestMarkovTransitionMatrix:
    """Tests for transition-matrix properties."""

    def test_markov_transition_matrix_row_sums(self) -> None:
        """A well-formed 3x3 transition matrix must have rows summing to 1.0."""
        labels = ["A", "B", "C"]
        matrix = pd.DataFrame(
            [[0.7, 0.2, 0.1],
             [0.0, 0.5, 0.5],
             [0.1, 0.3, 0.6]],
            index=labels,
            columns=labels,
        )

        row_sums = matrix.values.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-8)

    def test_markov_empty_data(self) -> None:
        """compute_transition_matrix on empty DataFrame returns empty matrix."""
        from analysis.advanced_analytics import FunnelMarkovChain

        chain = FunnelMarkovChain()
        result = chain.compute_transition_matrix(pd.DataFrame())

        assert isinstance(result, pd.DataFrame)
        # Empty matrix should either be truly empty or have all-zero rows
        assert result.empty or result.values.sum() == 0.0


# -----------------------------------------------------------------------
# 2. TransitionMatrixValidation schema
# -----------------------------------------------------------------------

class TestTransitionMatrixValidation:
    """Tests for the TransitionMatrixValidation.validate() classmethod."""

    def test_transition_matrix_validation_valid(self) -> None:
        """Valid stochastic matrix passes validation."""
        labels = ["X", "Y", "Z"]
        matrix = pd.DataFrame(
            [[0.5, 0.3, 0.2],
             [0.1, 0.6, 0.3],
             [0.2, 0.2, 0.6]],
            index=labels,
            columns=labels,
        )

        result = TransitionMatrixValidation.validate(matrix)

        assert result.is_valid is True
        assert result.row_sums_ok is True
        assert result.non_negative is True
        assert result.max_row_sum_error < 1e-4

    def test_transition_matrix_validation_invalid(self) -> None:
        """Matrix with rows not summing to 1.0 fails validation."""
        labels = ["X", "Y"]
        matrix = pd.DataFrame(
            [[0.5, 0.3],   # sum = 0.8, not 1.0
             [0.1, 0.6]],  # sum = 0.7, not 1.0
            index=labels,
            columns=labels,
        )

        result = TransitionMatrixValidation.validate(matrix)

        assert result.is_valid is False
        assert result.row_sums_ok is False
        assert result.max_row_sum_error > 0.1

    def test_transition_matrix_validation_empty(self) -> None:
        """Empty matrix fails validation."""
        result = TransitionMatrixValidation.validate(pd.DataFrame())

        assert result.is_valid is False
        assert len(result.warnings) > 0


# -----------------------------------------------------------------------
# 3. Feature validation (model_validation module)
# -----------------------------------------------------------------------

class TestFeatureValidation:
    """Tests for validate_features()."""

    def test_feature_validation_missing(self) -> None:
        """Missing columns are reported correctly."""
        df = pd.DataFrame({
            "PAYEND_CVR_LAG1": [0.1, 0.2],
            "PAYEND_CVR_LAG2": [0.3, 0.4],
            "TARGET_CLASS": ["LOW", "HIGH"],
        })
        expected = ["PAYEND_CVR_LAG1", "PAYEND_CVR_LAG2", "TOTALLY_MISSING"]

        result = validate_features(df, expected, "TARGET_CLASS")

        assert isinstance(result, FeatureValidationResult)
        assert result.available_features == 2
        assert "TOTALLY_MISSING" in result.missing_features

    def test_feature_validation_all_present(
        self, sample_feature_store_df: pd.DataFrame,
    ) -> None:
        """When all expected columns exist, none are missing."""
        result = validate_features(
            sample_feature_store_df, _FEATURE_COLUMNS, "TARGET_CLASS",
        )

        assert result.available_features == len(_FEATURE_COLUMNS)
        assert len(result.missing_features) == 0
        assert result.is_valid is True


# -----------------------------------------------------------------------
# 4. compute_metrics
# -----------------------------------------------------------------------

class TestComputeMetrics:
    """Tests for compute_metrics()."""

    def test_compute_metrics(self) -> None:
        """Basic metric fields are populated from simple y_true/y_pred."""
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 2, 1, 0, 1, 2])

        metrics = compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            cv_scores=[0.6, 0.7],
            n_features=5,
            feature_columns=["a", "b", "c", "d", "e"],
        )

        assert isinstance(metrics, ModelMetrics)
        assert 0.0 <= metrics.accuracy <= 1.0
        assert 0.0 <= metrics.f1_macro <= 1.0
        assert 0.0 <= metrics.precision_macro <= 1.0
        assert 0.0 <= metrics.recall_macro <= 1.0
        assert metrics.n_train_samples == len(y_true)
        assert metrics.n_features == 5
        assert metrics.cv_mean == pytest.approx(0.65, abs=1e-6)
        assert len(metrics.confusion_matrix) > 0
        assert metrics.classification_report_text != ""

        # to_dict round-trip check
        d = metrics.to_dict()
        assert "accuracy" in d
        assert "f1_macro" in d


# -----------------------------------------------------------------------
# 5. drop_incomplete_month
# -----------------------------------------------------------------------

class TestDropIncompleteMonth:
    """Tests for drop_incomplete_month()."""

    def test_drop_incomplete_month(self) -> None:
        """The latest YEAR_MONTH is removed from the DataFrame."""
        df = pd.DataFrame({
            "YEAR_MONTH": ["2025-09", "2025-10", "2025-11", "2025-11"],
            "VALUE": [1, 2, 3, 4],
        })

        result = drop_incomplete_month(df)

        assert "2025-11" not in result["YEAR_MONTH"].values
        assert len(result) == 2
        assert set(result["YEAR_MONTH"]) == {"2025-09", "2025-10"}

    def test_drop_incomplete_month_empty(self) -> None:
        """Empty DataFrame passes through unchanged."""
        df = pd.DataFrame()
        result = drop_incomplete_month(df)
        assert result.empty


# -----------------------------------------------------------------------
# 6. validate_columns
# -----------------------------------------------------------------------

class TestValidateColumns:
    """Tests for validate_columns()."""

    def test_validate_columns_present(self) -> None:
        """Returns True when all required columns exist."""
        df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})

        assert validate_columns(df, ["A", "B"]) is True

    def test_validate_columns_missing(self) -> None:
        """Returns False when required columns are missing."""
        df = pd.DataFrame({"A": [1], "B": [2]})

        assert validate_columns(df, ["A", "B", "MISSING"]) is False

    def test_validate_columns_empty_df(self) -> None:
        """Returns False for an empty DataFrame."""
        df = pd.DataFrame()
        assert validate_columns(df, ["A"]) is False


# -----------------------------------------------------------------------
# 7. FUNNEL_STAGES ordering
# -----------------------------------------------------------------------

class TestFunnelStages:
    """Tests for FUNNEL_STAGES constant."""

    def test_funnel_stages_order(self) -> None:
        """FUNNEL_STAGES has exactly 5 stages in the correct order."""
        expected = [
            "CONSULT_REQUEST",
            "SUBSCRIPTION",
            "REGISTEND",
            "OPEN",
            "PAYEND",
        ]

        assert FUNNEL_STAGES == expected
        assert len(FUNNEL_STAGES) == 5

    def test_product_categories_non_empty(self) -> None:
        """PRODUCT_CATEGORIES contains at least the five major categories."""
        assert len(PRODUCT_CATEGORIES) >= 5
        assert "인터넷" in PRODUCT_CATEGORIES
        assert "렌탈" in PRODUCT_CATEGORIES
