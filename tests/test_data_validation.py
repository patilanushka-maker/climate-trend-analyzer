"""
Unit tests for climate data validation module.

All test fixtures use synthetic, clearly labeled test values and do NOT
depend on a live internet connection.
"""

import numpy as np
import pandas as pd
import pytest

from climate_analyzer.data_validation import (
    DEFAULT_HARD_PHYSICAL_BOUNDS,
    DEFAULT_WARNING_BOUNDS,
    DataValidationError,
    PhysicalBoundError,
    SchemaValidationError,
    TemporalIntegrityError,
    ValidationReport,
    validate_monthly_series,
)


@pytest.fixture
def valid_monthly_df() -> pd.DataFrame:
    """Synthetic fixture: 2 complete years (2020-2021) of valid monthly anomalies."""
    dates = pd.date_range("2020-01-01", "2021-12-01", freq="MS")
    # Synthetic anomalies around 1.0°C (NOT real climate observations)
    anomalies = [1.0 + 0.05 * (i % 6) for i in range(len(dates))]
    return pd.DataFrame(
        {
            "year": dates.year,
            "month": dates.month,
            "date": dates,
            "anomaly": anomalies,
        }
    )


@pytest.fixture
def monthly_df_with_pending_tail() -> pd.DataFrame:
    """Synthetic fixture: year 2026 with trailing pending unobserved months."""
    dates = pd.date_range("2026-01-01", "2026-12-01", freq="MS")
    # First 9 months observed, last 3 months NaN
    anomalies = [1.2] * 9 + [np.nan, np.nan, np.nan]
    return pd.DataFrame(
        {
            "year": dates.year,
            "month": dates.month,
            "date": dates,
            "anomaly": anomalies,
        }
    )


class TestDataValidationSuccess:
    """Test cases for datasets that conform to valid climate series standards."""

    def test_valid_monthly_series_passes(self, valid_monthly_df):
        report = validate_monthly_series(valid_monthly_df)
        assert report.is_valid is True
        assert report.total_records == 24
        assert report.valid_anomalies == 24
        assert report.missing_anomalies == 0
        assert report.completeness_pct == 100.0
        assert report.start_date == pd.Timestamp("2020-01-01")
        assert report.end_date == pd.Timestamp("2021-12-01")
        assert len(report.errors) == 0
        assert len(report.warnings) == 0
        assert "PASSED" in report.summary()

    def test_pending_tail_months_not_flagged_as_corrupted(self, monthly_df_with_pending_tail):
        """Verify policy: expected future months in incomplete year are warnings, not errors."""
        report = validate_monthly_series(monthly_df_with_pending_tail, raise_on_error=True)
        assert report.is_valid is True
        assert report.start_date == pd.Timestamp("2026-01-01")
        assert report.end_date == pd.Timestamp("2026-12-01")
        assert report.observed_start_date == pd.Timestamp("2026-01-01")
        assert report.observed_end_date == pd.Timestamp("2026-09-01")
        assert report.missing_anomalies == 3
        assert len(report.trailing_unobserved) == 3
        assert len(report.internal_gaps) == 0
        assert any("pending months" in w for w in report.warnings)

    def test_internal_gaps_detected(self, valid_monthly_df):
        """Missing observations inside the historical series are reported as internal gaps."""
        df = valid_monthly_df.copy()
        # Set month 6 (June 2020) to NaN
        df.loc[5, "anomaly"] = np.nan
        report = validate_monthly_series(df, raise_on_error=True)
        assert report.is_valid is True
        assert len(report.internal_gaps) == 1
        assert report.internal_gaps[0] == pd.Timestamp("2020-06-01")
        assert any("internal missing" in w for w in report.warnings)

    def test_custom_sliced_index_gap_and_tail_detection(self):
        """Regression test: non-standard index handles internal gaps and trailing unobserved properly."""
        dates = pd.date_range("2026-01-01", "2026-12-01", freq="MS")
        anomalies = [1.0, 1.1, np.nan, 1.2, 1.3, np.nan, 1.4, 1.5, 1.6, np.nan, np.nan, np.nan]
        df = pd.DataFrame(
            {"year": dates.year, "month": dates.month, "date": dates, "anomaly": anomalies}
        )
        # Set custom non-consecutive index to test index-independence
        df.index = [i * 10 for i in range(len(df))]
        report = validate_monthly_series(df, raise_on_error=True)
        assert len(report.trailing_unobserved) == 3
        assert report.trailing_unobserved == [
            pd.Timestamp("2026-10-01"),
            pd.Timestamp("2026-11-01"),
            pd.Timestamp("2026-12-01"),
        ]
        assert len(report.internal_gaps) == 2
        assert report.internal_gaps == [
            pd.Timestamp("2026-03-01"),
            pd.Timestamp("2026-06-01"),
        ]


class TestDataValidationFailures:
    """Test cases for integrity failures that must raise informative exceptions."""

    def test_non_dataframe_input_rejected(self):
        with pytest.raises(SchemaValidationError) as exc:
            validate_monthly_series([{"year": 2020}])
        assert "DataFrame" in str(exc.value)

    def test_empty_dataframe_rejected(self):
        with pytest.raises(SchemaValidationError) as exc:
            validate_monthly_series(pd.DataFrame())
        assert "empty" in str(exc.value)

    def test_missing_column_rejected(self, valid_monthly_df):
        df = valid_monthly_df.drop(columns=["anomaly"])
        with pytest.raises(SchemaValidationError) as exc:
            validate_monthly_series(df)
        assert "anomaly" in str(exc.value)

    def test_invalid_data_types_rejected(self, valid_monthly_df):
        df = valid_monthly_df.copy()
        df["anomaly"] = df["anomaly"].astype(str)
        with pytest.raises(SchemaValidationError) as exc:
            validate_monthly_series(df)
        assert "float64" in str(exc.value)

    def test_invalid_month_index_rejected(self, valid_monthly_df):
        df = valid_monthly_df.copy()
        df.loc[0, "month"] = 13
        with pytest.raises(TemporalIntegrityError) as exc:
            validate_monthly_series(df)
        assert "outside [1, 12]" in str(exc.value)

    def test_calendar_inconsistency_rejected(self, valid_monthly_df):
        """Mismatch between timestamp date and (year, month) columns."""
        df = valid_monthly_df.copy()
        df.loc[0, "date"] = pd.Timestamp("2020-02-01")  # year=2020, month=1 but date=Feb
        with pytest.raises(TemporalIntegrityError) as exc:
            validate_monthly_series(df)
        assert "Inconsistency between 'date'" in str(exc.value)

    def test_duplicate_dates_rejected(self, valid_monthly_df):
        df = pd.concat([valid_monthly_df, valid_monthly_df.iloc[[0]]], ignore_index=True)
        with pytest.raises(TemporalIntegrityError) as exc:
            validate_monthly_series(df)
        assert "Duplicate" in str(exc.value)

    def test_non_monotonic_dates_rejected(self, valid_monthly_df):
        df = valid_monthly_df.iloc[::-1].reset_index(drop=True)
        with pytest.raises(TemporalIntegrityError) as exc:
            validate_monthly_series(df)
        assert "monotonically strictly increasing" in str(exc.value)

    def test_hard_physical_bound_violation_rejected(self, valid_monthly_df):
        """Values exceeding +/-10°C indicate gross corruption and must fail."""
        df = valid_monthly_df.copy()
        df.loc[0, "anomaly"] = 99.9  # Gross sensor corruption
        with pytest.raises(PhysicalBoundError) as exc:
            validate_monthly_series(df)
        assert "hard physical sanity limits" in str(exc.value)


class TestQualityWarningThresholds:
    """Test soft quality warning bounds behavior."""

    def test_soft_warning_triggered_without_raising_fatal_error(self, valid_monthly_df):
        """Anomalies in [3.0°C, 10.0°C] trigger soft warnings, preserving valid extreme data."""
        df = valid_monthly_df.copy()
        df.loc[0, "anomaly"] = 3.5  # Unprecedented extreme observation, not gross corruption
        report = validate_monthly_series(df, warning_bounds=(-3.0, 3.0), raise_on_error=True)

        assert report.is_valid is True
        assert len(report.out_of_bounds_records) == 1
        assert report.out_of_bounds_records[0]["anomaly"] == 3.5
        assert any("soft quality warning bounds" in w for w in report.warnings)

    def test_raise_on_error_false_captures_all_errors(self, valid_monthly_df):
        df = valid_monthly_df.copy()
        df.loc[0, "anomaly"] = -99.0  # Physical bound error
        df.loc[1, "month"] = 99  # Month bounds error
        report = validate_monthly_series(df, raise_on_error=False)

        assert report.is_valid is False
        assert len(report.errors) >= 2
        assert "FAILED" in report.summary()
