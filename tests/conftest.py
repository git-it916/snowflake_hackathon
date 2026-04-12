"""Shared pytest fixtures for the Snowflake hackathon test suite.

All fixtures produce offline-friendly DataFrames that mirror
the schemas used in production without requiring a Snowflake connection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.constants import FUNNEL_STAGES
from ml.conversion_model import _FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Fixture: sample stage_drop DataFrame (FUNNEL_STAGE_DROP schema)
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_stage_drop_df() -> pd.DataFrame:
    """Create a minimal FUNNEL_STAGE_DROP DataFrame.

    Schema mirrors the view used by FunnelMarkovChain.compute_transition_matrix:
      YEAR_MONTH, MAIN_CATEGORY_NAME, STAGE_ORDER, STAGE_NAME,
      PREV_STAGE_COUNT, CURR_STAGE_COUNT, DROP_RATE, BOTTLENECK_FLAG
    """
    records: list[dict] = []
    counts = [1000, 800, 700, 500, 400]  # progressive funnel counts

    for month in ["2025-10", "2025-11"]:
        for i, stage in enumerate(FUNNEL_STAGES):
            prev_count = counts[i - 1] if i > 0 else counts[i]
            curr_count = counts[i]
            drop_rate = round(1.0 - curr_count / prev_count, 4) if i > 0 else 0.0
            records.append({
                "YEAR_MONTH": month,
                "MAIN_CATEGORY_NAME": "인터넷",
                "STAGE_ORDER": i + 1,
                "STAGE_NAME": stage,
                "PREV_STAGE_COUNT": prev_count,
                "CURR_STAGE_COUNT": curr_count,
                "DROP_RATE": drop_rate,
                "BOTTLENECK_FLAG": drop_rate > 0.15,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Fixture: sample feature store DataFrame (ML_FEATURE_STORE schema)
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_feature_store_df() -> pd.DataFrame:
    """Create a small ML_FEATURE_STORE DataFrame with all expected columns.

    Includes _FEATURE_COLUMNS plus TARGET_CLASS, YEAR_MONTH, CATEGORY.
    """
    rng = np.random.default_rng(42)
    n_rows = 30

    data: dict[str, object] = {
        "YEAR_MONTH": [f"2025-{m:02d}" for m in rng.choice(range(1, 13), size=n_rows)],
        "CATEGORY": rng.choice(["인터넷", "렌탈", "모바일"], size=n_rows).tolist(),
        "TARGET_CLASS": rng.choice(["LOW", "MEDIUM", "HIGH"], size=n_rows).tolist(),
    }

    for col in _FEATURE_COLUMNS:
        data[col] = rng.random(n_rows).tolist()

    return pd.DataFrame(data)
