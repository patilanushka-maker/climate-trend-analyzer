"""
Unit tests for climate feature engineering module.

All test fixtures use synthetic, clearly labeled test values and do NOT
depend on a live internet connection.
"""

import numpy as np
import pandas as pd
import pytest

from climate_analyzer.data_validation import (
    SchemaValidationError,
    TemporalIntegrityError,
)
from climate_analyzer.feature_engineering import (
    add_rolling_features,
    calculate_annual_summary,
    calculate_baseline_anomalies,
    calculate_year_over_year_change,
)


@pytest.fixture
def synthetic_monthly_series() -> pd.DataFrame:
    """Synthetic monthly fixture: 5 years (2015-2019, 60 months) with known step anomalies."""
    dates = pd.date_range("2015-01-01", "2019-12-01", freq="MS")
    # Step pattern: anomaly = 1.0 for 2015, 1.2 for 2016, 1.4 for 2017, 1.6 for 2018, 1.8 for 2019
    anomalies = []
    for d in dates:
        step = 1.0 + (d.year - 2015) * 0.2
        anomalies.append(round(step, 2))
    return pd.DataFrame(
        {
            "year": dates.year,
            "month": dates.month,
            "date": dates,
            "anomaly": anomalies,
        }
    )


@pytest.fixture
def series_with_incomplete_current_year() -> pd.DataFrame:
    """Synthetic fixture: 2025 (12 complete months) and 2026 (9 observed months, 3 pending NaNs)."""
    dates_2025 = pd.date_range("2025-01-01", "2025-12-01", freq="MS")
    dates_2026 = pd.date_range("2026-01-01", "2026-12-01", freq="MS")
    dates = dates_2025.append(dates_2026)

    # 2025 anomalies = 1.2; 2026 anomalies = 1.4 for first 9 months, then 3 NaNs
    anomalies = [1.2] * 12 + [1.4] * 9 + [np.nan, np.nan, np.nan]
    return pd.DataFrame(
        {
            "year": dates.year,
            "month": dates.month,
            "date": dates,
            "anomaly": anomalies,
        }
    )


# ---------------------------------------------------------------------------
# Rolling Features Tests
# ---------------------------------------------------------------------------


class TestRollingFeatures:
    """Tests for trailing and centered rolling window calculations."""

    def test_trailing_rolling_known_fixture(self, synthetic_monthly_series):
        """Verify trailing 12-month rolling mean on known values."""
        df = add_rolling_features(synthetic_monthly_series, windows=[12])
        assert "anomaly_rolling_12m" in df.columns

        # Month 12 (index 11: 2015-12) should have trailing 12m mean of exactly 1.0 (all 2015 months = 1.0)
        assert np.isclose(df.loc[11, "anomaly_rolling_12m"], 1.0)

        # Month 24 (index 23: 2016-12) should have trailing 12m mean of exactly 1.2
        assert np.isclose(df.loc[23, "anomaly_rolling_12m"], 1.2)

    def test_min_periods_handling_at_start(self, synthetic_monthly_series):
        """Verify minimum-period thresholding yields NaN when observation count is insufficient."""
        # For window 12, default min_periods = ceil(12 * 0.75) = 9
        df = add_rolling_features(synthetic_monthly_series, windows=[12])
        # Indices 0 to 7 (first 8 months) should be NaN
        assert df.loc[:7, "anomaly_rolling_12m"].isna().all()
        # Index 8 (9th month) meets threshold 9 and should have a valid float
        assert pd.notna(df.loc[8, "anomaly_rolling_12m"])

    def test_rolling_warmup_exact_positions(self):
        """Verify exact warmup lengths: 60m (min=45) produces first value at index 44 (44 NaNs);
        120m (min=90) produces first value at index 89 (89 NaNs)."""
        dates = pd.date_range("1880-01-01", periods=150, freq="MS")
        df = pd.DataFrame(
            {"year": dates.year, "month": dates.month, "date": dates, "anomaly": [0.5] * 150}
        )
        df_roll = add_rolling_features(df, windows=[60, 120])

        # 60m: min_periods = ceil(60 * 0.75) = 45 -> index 44 is 45th observation
        assert df_roll.loc[:43, "anomaly_rolling_60m"].isna().all()
        assert pd.notna(df_roll.loc[44, "anomaly_rolling_60m"])
        assert df_roll["anomaly_rolling_60m"].first_valid_index() == 44

        # 120m: min_periods = ceil(120 * 0.75) = 90 -> index 89 is 90th observation
        assert df_roll.loc[:88, "anomaly_rolling_120m"].isna().all()
        assert pd.notna(df_roll.loc[89, "anomaly_rolling_120m"])
        assert df_roll["anomaly_rolling_120m"].first_valid_index() == 89

    def test_missing_values_do_not_become_zero(self, series_with_incomplete_current_year):
        """Verify policy: missing anomalies remain NaN and are not filled with 0."""
        df = add_rolling_features(series_with_incomplete_current_year, windows=[12])
        # Trailing months with NaNs should not have 0.0 values injected
        assert 0.0 not in df["anomaly"].values
        # The original NaNs are preserved
        assert df["anomaly"].isna().sum() == 3

    def test_caller_dataframe_not_mutated(self, synthetic_monthly_series):
        """Verify defensive copying: input DataFrame is unmodified."""
        orig_cols = list(synthetic_monthly_series.columns)
        df_out = add_rolling_features(synthetic_monthly_series, windows=[12, 60])
        assert list(synthetic_monthly_series.columns) == orig_cols
        assert "anomaly_rolling_12m" not in synthetic_monthly_series.columns
        assert "anomaly_rolling_12m" in df_out.columns

    def test_centered_rolling_label_and_future_data_usage(self, synthetic_monthly_series):
        """Verify centered rolling mean creates explicitly labeled column."""
        df = add_rolling_features(synthetic_monthly_series, windows=[12], center=True)
        assert "anomaly_centered_12m" in df.columns
        assert "anomaly_rolling_12m" not in df.columns

    def test_duplicate_dates_and_non_monotonic_raise_error(self, synthetic_monthly_series):
        """Verify temporal validation fires before computing features."""
        # Duplicate date
        dup_df = pd.concat([synthetic_monthly_series, synthetic_monthly_series.iloc[[0]]])
        with pytest.raises(TemporalIntegrityError):
            add_rolling_features(dup_df)

        # Non-monotonic
        rev_df = synthetic_monthly_series.iloc[::-1].reset_index(drop=True)
        with pytest.raises(TemporalIntegrityError):
            add_rolling_features(rev_df)

    def test_invalid_window_size_raises_error(self, synthetic_monthly_series):
        with pytest.raises(ValueError):
            add_rolling_features(synthetic_monthly_series, windows=[-5])


# ---------------------------------------------------------------------------
# Annual Summary Tests
# ---------------------------------------------------------------------------


class TestAnnualSummary:
    """Tests for annual calendar aggregations and completeness accounting."""

    def test_complete_year_annual_summary(self, synthetic_monthly_series):
        """Verify complete years report correct mean and is_complete=True."""
        annual = calculate_annual_summary(synthetic_monthly_series)
        assert len(annual) == 5
        assert list(annual["year"]) == [2015, 2016, 2017, 2018, 2019]
        assert (annual["valid_months"] == 12).all()
        assert (annual["is_complete"] == True).all()
        assert (annual["completeness_pct"] == 100.0).all()
        assert np.isclose(annual.loc[annual["year"] == 2015, "annual_mean_anomaly"].values[0], 1.0)
        assert np.isclose(annual.loc[annual["year"] == 2016, "annual_mean_anomaly"].values[0], 1.2)

    def test_incomplete_year_not_labeled_as_complete(self, series_with_incomplete_current_year):
        """Verify policy: incomplete year has is_complete=False and annual_mean_anomaly=NaN."""
        annual = calculate_annual_summary(series_with_incomplete_current_year, min_valid_months=12)
        assert len(annual) == 2

        row_2025 = annual[annual["year"] == 2025].iloc[0]
        assert bool(row_2025["is_complete"]) is True
        assert np.isclose(row_2025["annual_mean_anomaly"], 1.2)

        row_2026 = annual[annual["year"] == 2026].iloc[0]
        assert row_2026["valid_months"] == 9
        assert bool(row_2026["is_complete"]) is False
        assert np.isnan(row_2026["annual_mean_anomaly"])
        # Partial diagnostic mean is preserved
        assert np.isclose(row_2026["partial_year_mean"], 1.4)

    def test_configurable_completeness_threshold(self, series_with_incomplete_current_year):
        """Verify configurable min_valid_months (e.g. min_valid_months=9 permits 9-month year)."""
        annual = calculate_annual_summary(series_with_incomplete_current_year, min_valid_months=9)
        row_2026 = annual[annual["year"] == 2026].iloc[0]
        assert bool(row_2026["is_complete"]) is True
        assert np.isclose(row_2026["annual_mean_anomaly"], 1.4)

    def test_invalid_min_valid_months_rejected(self, synthetic_monthly_series):
        with pytest.raises(ValueError):
            calculate_annual_summary(synthetic_monthly_series, min_valid_months=15)
        with pytest.raises(ValueError):
            calculate_annual_summary(synthetic_monthly_series, min_valid_months=0)


# ---------------------------------------------------------------------------
# Baseline Climatology Re-centering Tests
# ---------------------------------------------------------------------------


class TestBaselineAnomalies:
    """Tests for multi-baseline reference re-centering."""

    def test_baseline_anomalies_separate_monthly_means(self):
        """Verify baseline calculates a separate mean for each of the 12 calendar months."""
        dates = pd.date_range("2000-01-01", "2020-12-01", freq="MS")
        # Pattern: January is always 0.5, February is always 0.8, etc.
        anomalies = [0.1 * d.month for d in dates]
        df = pd.DataFrame({"year": dates.year, "month": dates.month, "date": dates, "anomaly": anomalies})

        # Base period 2000-2015 (16 years)
        re_df = calculate_baseline_anomalies(df, baseline_start=2000, baseline_end=2015, min_baseline_years=10)
        col = "anomaly_base_2000_2015"
        assert col in re_df.columns
        # Since every month was identical to its baseline mean, re-centered anomalies should all be 0.0
        assert np.allclose(re_df[col], 0.0)

    def test_insufficient_baseline_samples_raises_error(self, synthetic_monthly_series):
        """Verify that attempting a baseline with fewer observations than threshold raises ValueError."""
        with pytest.raises(ValueError) as exc:
            # Baseline span is only 2 years (2015-2016), but min_baseline_years requires 15
            calculate_baseline_anomalies(synthetic_monthly_series, baseline_start=2015, baseline_end=2016, min_baseline_years=15)
        assert "Insufficient baseline observations" in str(exc.value)

    def test_invalid_baseline_period_raises_error(self, synthetic_monthly_series):
        # start > end
        with pytest.raises(ValueError):
            calculate_baseline_anomalies(synthetic_monthly_series, baseline_start=2018, baseline_end=2015)
        # Period outside data
        with pytest.raises(ValueError):
            calculate_baseline_anomalies(synthetic_monthly_series, baseline_start=1950, baseline_end=1960)


# ---------------------------------------------------------------------------
# Year-over-Year Change Tests
# ---------------------------------------------------------------------------


class TestYearOverYearChange:
    """Tests for annual rate-of-change differences."""

    def test_yoy_change_consecutive_complete_years(self, synthetic_monthly_series):
        """Verify YoY change computes difference in degrees Celsius."""
        annual = calculate_annual_summary(synthetic_monthly_series)
        yoy = calculate_year_over_year_change(annual, require_complete=True)

        assert "yoy_change_deg_c" in yoy.columns
        # First year has no predecessor
        assert np.isnan(yoy.loc[0, "yoy_change_deg_c"])
        # From 2015 (1.0) to 2016 (1.2) is +0.2°C
        assert np.isclose(yoy.loc[1, "yoy_change_deg_c"], 0.2)
        # From 2016 (1.2) to 2017 (1.4) is +0.2°C
        assert np.isclose(yoy.loc[2, "yoy_change_deg_c"], 0.2)

    def test_yoy_change_incomplete_year_handling(self, series_with_incomplete_current_year):
        """Verify policy: YoY change is NaN when comparing complete year with incomplete year."""
        annual = calculate_annual_summary(series_with_incomplete_current_year, min_valid_months=12)
        yoy = calculate_year_over_year_change(annual, require_complete=True)

        # Year 2026 is incomplete (9 months), so its YoY comparison against 2025 is NaN
        row_2026 = yoy[yoy["year"] == 2026].iloc[0]
        assert np.isnan(row_2026["yoy_change_deg_c"])

    def test_yoy_non_consecutive_years_yields_nan(self):
        """Verify non-consecutive years (gap in year sequence) yields NaN."""
        df = pd.DataFrame(
            {
                "year": [2010, 2015],  # 5-year gap
                "annual_mean_anomaly": [0.7, 0.9],
                "is_complete": [True, True],
            }
        )
        yoy = calculate_year_over_year_change(df)
        assert np.isnan(yoy.loc[1, "yoy_change_deg_c"])
