"""
Automated unit tests for the NASA GISTEMP data loader module.

All test fixtures use synthetic, clearly labeled test values and do NOT
depend on a live internet connection or external network requests.
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from climate_analyzer.data_loader import (
    DataLoaderError,
    InvalidFormatError,
    MissingDataFileError,
    ValidationError,
    load_gistemp_annual,
    load_gistemp_monthly,
    load_gistemp_raw,
)

# ---------------------------------------------------------------------------
# Synthetic Test Fixtures (Clearly labeled: NOT real climate observations)
# ---------------------------------------------------------------------------

SYNTHETIC_VALID_GISTEMP_CSV = """Land-Ocean: Global Means
Year,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec,J-D,D-N,DJF,MAM,JJA,SON
2020,1.18,1.23,1.18,1.11,.99,.91,.89,.86,.96,.87,1.08,.80,1.01,1.03,1.17,1.09,.89,.97
2021,.81,.64,.89,.76,.79,.85,.92,.81,.92,.98,.92,.88,.85,.84,.75,.81,.86,.94
2022,.91,.89,1.04,.83,.84,.92,.94,.94,.88,.97,.73,.79,.89,.90,.89,.90,.93,.86
"""

SYNTHETIC_WITH_MISSING_MARKERS_CSV = """Land-Ocean: Global Means
Year,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec,J-D,D-N,DJF,MAM,JJA,SON
1880,-.19,-.25,-.09,-.16,-.10,-.21,-.19,-.10,-.15,-.23,-.22,-.18,-.17,***,***,-.12,-.17,-.20
2026,1.08,1.23,1.30,1.16,1.13,1.17,1.24,1.42,1.38,***,***,***,***,***,1.12,1.20,1.28,***
"""

SYNTHETIC_DUPLICATE_YEARS_CSV = """Land-Ocean: Global Means
Year,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec,J-D,D-N,DJF,MAM,JJA,SON
2020,1.18,1.23,1.18,1.11,.99,.91,.89,.86,.96,.87,1.08,.80,1.01,1.03,1.17,1.09,.89,.97
2020,.81,.64,.89,.76,.79,.85,.92,.81,.92,.98,.92,.88,.85,.84,.75,.81,.86,.94
"""

SYNTHETIC_NON_INTEGER_YEAR_CSV = """Land-Ocean: Global Means
Year,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec,J-D,D-N,DJF,MAM,JJA,SON
Year2020,1.18,1.23,1.18,1.11,.99,.91,.89,.86,.96,.87,1.08,.80,1.01,1.03,1.17,1.09,.89,.97
"""

SYNTHETIC_OUTLIER_YEAR_CSV = """Land-Ocean: Global Means
Year,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec,J-D,D-N,DJF,MAM,JJA,SON
1492,1.18,1.23,1.18,1.11,.99,.91,.89,.86,.96,.87,1.08,.80,1.01,1.03,1.17,1.09,.89,.97
"""

SYNTHETIC_MISSING_COLUMNS_CSV = """Land-Ocean: Global Means
Year,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov
2020,1.18,1.23,1.18,1.11,.99,.91,.89,.86,.96,.87,1.08
"""

SYNTHETIC_CORRUPTED_ANOMALY_CSV = """Land-Ocean: Global Means
Year,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec,J-D,D-N,DJF,MAM,JJA,SON
2020,1.18,INVALID_ANOMALY,1.18,1.11,.99,.91,.89,.86,.96,.87,1.08,.80,1.01,1.03,1.17,1.09,.89,.97
"""


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


class TestDataLoaderRaw:
    """Tests for raw NASA GISTEMP data loading and validation."""

    def test_load_valid_in_memory_fixture(self):
        """Verify successful loading, row count, column structure, and types."""
        buf = io.StringIO(SYNTHETIC_VALID_GISTEMP_CSV)
        df = load_gistemp_raw(buf)

        assert len(df) == 3
        assert list(df["Year"]) == [2020, 2021, 2022]
        assert df["Year"].dtype == int
        assert "Jan" in df.columns and "SON" in df.columns
        assert df["Jan"].dtype == np.float64
        assert df["J-D"].dtype == np.float64
        assert np.isclose(df.loc[df["Year"] == 2020, "Jan"].values[0], 1.18)

    def test_missing_value_marker_parsing(self):
        """Verify that '***' missing value markers are converted to NaN."""
        buf = io.StringIO(SYNTHETIC_WITH_MISSING_MARKERS_CSV)
        df = load_gistemp_raw(buf)

        assert len(df) == 2
        # Check 1880 missing aggregates
        row_1880 = df[df["Year"] == 1880].iloc[0]
        assert np.isnan(row_1880["D-N"])
        assert np.isnan(row_1880["DJF"])
        assert np.isclose(row_1880["Jan"], -0.19)

        # Check 2026 future months
        row_2026 = df[df["Year"] == 2026].iloc[0]
        assert np.isnan(row_2026["Oct"])
        assert np.isnan(row_2026["Nov"])
        assert np.isnan(row_2026["Dec"])
        assert np.isclose(row_2026["Sep"], 1.38)

    def test_missing_file_raises_error(self, tmp_path):
        """Verify that pointing to a nonexistent path raises MissingDataFileError."""
        fake_path = tmp_path / "nonexistent_file.csv"
        with pytest.raises(MissingDataFileError) as exc_info:
            load_gistemp_raw(fake_path)
        assert "Climate dataset file not found" in str(exc_info.value)

    def test_empty_content_raises_error(self):
        """Verify that empty inputs raise InvalidFormatError."""
        with pytest.raises(InvalidFormatError) as exc_info:
            load_gistemp_raw(io.StringIO(""))
        assert "empty" in str(exc_info.value).lower()

    def test_missing_header_raises_error(self):
        """Verify that inputs lacking the 'Year' header raise InvalidFormatError."""
        content = "Some random text banner\n1,2,3,4,5\n"
        with pytest.raises(InvalidFormatError) as exc_info:
            load_gistemp_raw(io.StringIO(content))
        assert "header" in str(exc_info.value).lower()

    def test_missing_required_columns_raises_error(self):
        """Verify that truncated or incomplete schemas raise InvalidFormatError."""
        buf = io.StringIO(SYNTHETIC_MISSING_COLUMNS_CSV)
        with pytest.raises(InvalidFormatError) as exc_info:
            load_gistemp_raw(buf)
        assert "missing required columns" in str(exc_info.value).lower()

    def test_duplicate_years_rejected(self):
        """Verify explicit policy: ambiguous duplicate years raise ValidationError."""
        buf = io.StringIO(SYNTHETIC_DUPLICATE_YEARS_CSV)
        with pytest.raises(ValidationError) as exc_info:
            load_gistemp_raw(buf)
        assert "Duplicate year records detected" in str(exc_info.value)

    def test_non_integer_year_rejected(self):
        """Verify that malformed year strings raise ValidationError."""
        buf = io.StringIO(SYNTHETIC_NON_INTEGER_YEAR_CSV)
        with pytest.raises(ValidationError) as exc_info:
            load_gistemp_raw(buf)
        assert "non-integer" in str(exc_info.value).lower()

    def test_outlier_year_rejected(self):
        """Verify that year values outside reasonable bounds raise ValidationError."""
        buf = io.StringIO(SYNTHETIC_OUTLIER_YEAR_CSV)
        with pytest.raises(ValidationError) as exc_info:
            load_gistemp_raw(buf)
        assert "Unreasonable year" in str(exc_info.value)

    def test_corrupted_anomaly_value_rejected(self):
        """Verify that non-numeric anomaly values raise ValidationError."""
        buf = io.StringIO(SYNTHETIC_CORRUPTED_ANOMALY_CSV)
        with pytest.raises(ValidationError) as exc_info:
            load_gistemp_raw(buf)
        assert "Unparseable non-numeric anomaly value" in str(exc_info.value)


class TestDataLoaderMonthly:
    """Tests for tidy monthly time-series transformation."""

    def test_load_monthly_structure_and_types(self):
        """Verify monthly long-format columns, length, and timestamp generation."""
        buf = io.StringIO(SYNTHETIC_VALID_GISTEMP_CSV)
        monthly_df = load_gistemp_monthly(buf, drop_na=False)

        # 3 years * 12 months = 36 monthly rows
        assert len(monthly_df) == 36
        assert list(monthly_df.columns) == ["year", "month", "date", "anomaly"]
        assert monthly_df["year"].dtype == int
        assert monthly_df["month"].dtype == int
        assert ptypes_is_datetime(monthly_df["date"])
        assert monthly_df["anomaly"].dtype == np.float64

        # Check first and last timestamp
        assert monthly_df.iloc[0]["date"] == pd.Timestamp("2020-01-01")
        assert monthly_df.iloc[-1]["date"] == pd.Timestamp("2022-12-01")

        # Check anomaly alignment
        assert np.isclose(monthly_df.iloc[0]["anomaly"], 1.18)

    def test_load_monthly_drop_na_toggle(self):
        """Verify that drop_na=True removes NaNs, while drop_na=False preserves them."""
        # 2 years = 24 total months; 2026 has Oct, Nov, Dec missing (3 missing)
        buf_keep = io.StringIO(SYNTHETIC_WITH_MISSING_MARKERS_CSV)
        df_keep = load_gistemp_monthly(buf_keep, drop_na=False)
        assert len(df_keep) == 24
        assert df_keep["anomaly"].isna().sum() == 3

        buf_drop = io.StringIO(SYNTHETIC_WITH_MISSING_MARKERS_CSV)
        df_drop = load_gistemp_monthly(buf_drop, drop_na=True)
        assert len(df_drop) == 21
        assert df_drop["anomaly"].isna().sum() == 0

    def test_load_monthly_strictly_chronological(self):
        """Verify that the resulting series is strictly monotonic increasing."""
        buf = io.StringIO(SYNTHETIC_VALID_GISTEMP_CSV)
        monthly_df = load_gistemp_monthly(buf)
        assert monthly_df["date"].is_monotonic_increasing


class TestDataLoaderAnnual:
    """Tests for annual and seasonal summary loader."""

    def test_load_annual_structure(self):
        """Verify columns and alignment of annual aggregates."""
        buf = io.StringIO(SYNTHETIC_VALID_GISTEMP_CSV)
        annual_df = load_gistemp_annual(buf)

        expected_cols = [
            "year",
            "calendar_year_anomaly",
            "meteorological_year_anomaly",
            "djf_anomaly",
            "mam_anomaly",
            "jja_anomaly",
            "son_anomaly",
        ]
        assert list(annual_df.columns) == expected_cols
        assert len(annual_df) == 3
        assert np.isclose(annual_df.iloc[0]["calendar_year_anomaly"], 1.01)


class TestActualRawFileIntegration:
    """Optional validation against the downloaded local raw dataset if present."""

    def test_real_raw_file_if_exists(self):
        """If data/raw/GLB.Ts+dSST.csv exists, verify it loads and passes all validation."""
        raw_path = Path("data/raw/GLB.Ts+dSST.csv")
        if not raw_path.is_file():
            pytest.skip("Local raw dataset not present; skipping local file test.")

        raw_df = load_gistemp_raw(raw_path)
        assert len(raw_df) >= 140
        assert raw_df["Year"].iloc[0] == 1880
        assert raw_df["Year"].is_monotonic_increasing

        monthly_df = load_gistemp_monthly(raw_path, drop_na=False)
        assert len(monthly_df) >= 140 * 12
        assert monthly_df["date"].is_monotonic_increasing


def ptypes_is_datetime(series: pd.Series) -> bool:
    """Helper to assert series is datetime64."""
    return pd.api.types.is_datetime64_any_dtype(series)
