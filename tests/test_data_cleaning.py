"""
Unit tests for climate data cleaning module.

Validates deterministic transformations, defensive copying, explicit missing value policies,
and file output protections.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from climate_analyzer.data_cleaning import (
    clean_monthly_gistemp,
    save_processed_data,
)
from climate_analyzer.data_validation import SchemaValidationError


@pytest.fixture
def sample_monthly_df() -> pd.DataFrame:
    """Synthetic fixture: 3 years (2020-2022) with known anomaly values."""
    dates = pd.date_range("2020-01-01", "2022-12-01", freq="MS")
    anomalies = [1.0 + 0.02 * i for i in range(len(dates))]
    return pd.DataFrame(
        {
            "year": dates.year,
            "month": dates.month,
            "date": dates,
            "anomaly": anomalies,
        }
    )


@pytest.fixture
def df_with_missing_and_trailing() -> pd.DataFrame:
    """Synthetic fixture: 2025 (complete) and 2026 (incomplete, with 3 trailing NaNs)."""
    dates_2025 = pd.date_range("2025-01-01", "2025-12-01", freq="MS")
    dates_2026 = pd.date_range("2026-01-01", "2026-12-01", freq="MS")
    dates = dates_2025.append(dates_2026)

    anomalies_2025 = [1.2] * 12
    anomalies_2026 = [1.3] * 9 + [np.nan, np.nan, np.nan]
    anomalies = anomalies_2025 + anomalies_2026

    return pd.DataFrame(
        {
            "year": dates.year,
            "month": dates.month,
            "date": dates,
            "anomaly": anomalies,
        }
    )


class TestDataCleaning:
    """Tests for conservative cleaning and transformation operations."""

    def test_raw_input_not_mutated(self, sample_monthly_df):
        """Verify defensive copy: original DataFrame is untouched."""
        original_copy = sample_monthly_df.copy()
        cleaned = clean_monthly_gistemp(sample_monthly_df, start_year=2021)

        pd.testing.assert_frame_equal(sample_monthly_df, original_copy)
        assert len(cleaned) == 24
        assert len(sample_monthly_df) == 36

    def test_deterministic_cleaning(self, sample_monthly_df):
        """Verify deterministic reproducibility: multiple runs produce identical output."""
        res1 = clean_monthly_gistemp(sample_monthly_df, start_year=2021, end_year=2021)
        res2 = clean_monthly_gistemp(sample_monthly_df, start_year=2021, end_year=2021)
        pd.testing.assert_frame_equal(res1, res2)

    def test_missing_values_preserved_by_default(self, df_with_missing_and_trailing):
        """Verify policy: missing anomalies are NEVER converted to 0 and preserved with NaN."""
        cleaned = clean_monthly_gistemp(df_with_missing_and_trailing, drop_na=False)
        assert cleaned["anomaly"].isna().sum() == 3
        # Check that NaN was NOT converted to 0
        assert 0.0 not in cleaned["anomaly"].values

    def test_opt_in_drop_na(self, df_with_missing_and_trailing):
        """Verify explicit opt-in drop_na removes NaNs."""
        cleaned = clean_monthly_gistemp(df_with_missing_and_trailing, drop_na=True)
        assert cleaned["anomaly"].isna().sum() == 0
        assert len(cleaned) == 21

    def test_trim_trailing_incomplete_year(self, df_with_missing_and_trailing):
        """Verify trimming removes only unobserved months in the final incomplete year."""
        cleaned = clean_monthly_gistemp(
            df_with_missing_and_trailing, trim_trailing_incomplete_year=True
        )
        assert len(cleaned) == 21
        assert cleaned["date"].max() == pd.Timestamp("2026-09-01")
        # All remaining rows are valid
        assert cleaned["anomaly"].isna().sum() == 0

    def test_year_range_filtering(self, sample_monthly_df):
        cleaned = clean_monthly_gistemp(sample_monthly_df, start_year=2021, end_year=2021)
        assert len(cleaned) == 12
        assert set(cleaned["year"].unique()) == {2021}

    def test_invalid_year_range_raises_error(self, sample_monthly_df):
        with pytest.raises(ValueError) as exc:
            clean_monthly_gistemp(sample_monthly_df, start_year=2025, end_year=2020)
        assert "cannot be greater than" in str(exc.value)

    def test_filtering_resulting_in_empty_df_raises_error(self, sample_monthly_df):
        with pytest.raises(ValueError) as exc:
            clean_monthly_gistemp(sample_monthly_df, start_year=1900, end_year=1910)
        assert "empty DataFrame" in str(exc.value)

    def test_extreme_anomalies_not_silently_removed(self, sample_monthly_df):
        """Verify policy: valid extreme observations are preserved."""
        df = sample_monthly_df.copy()
        df.loc[0, "anomaly"] = 2.9  # High observation within quality limits
        cleaned = clean_monthly_gistemp(df)
        assert np.isclose(cleaned.loc[0, "anomaly"], 2.9)


class TestProcessedDataStorage:
    """Tests for saving cleaned outputs and security guardrails."""

    def test_save_processed_csv_successful(self, sample_monthly_df, tmp_path):
        out_file = tmp_path / "processed" / "test_clean.csv"
        saved_path = save_processed_data(sample_monthly_df, output_path=out_file)

        assert saved_path.is_file()
        loaded = pd.read_csv(saved_path)
        assert len(loaded) == len(sample_monthly_df)

    def test_overwriting_data_raw_is_forbidden(self, sample_monthly_df):
        """Security guardrail: attempting to save into data/raw/ raises ValueError."""
        raw_dest = Path("data/raw/should_fail.csv")
        with pytest.raises(ValueError) as exc:
            save_processed_data(sample_monthly_df, output_path=raw_dest)
        assert "Security Violation" in str(exc.value)
