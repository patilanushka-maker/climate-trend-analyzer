"""
Automated unit and deterministic tests for descriptive climate analysis.

Covers:
- calculate_descriptive_statistics: moments, percentiles, incomplete-year exclusion,
  custom percentile sequences, validation of inputs.
- calculate_decadal_summary: decade grouping conventions (e.g. 1880-1889),
  completeness flags, minimum year thresholds, defensive copies.
- calculate_warmth_percentiles: reference distribution subsetting, ranking methods,
  tie handling, insufficient reference samples, defensive behavior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from climate_analyzer.analysis.descriptive_analysis import (
    DescriptiveStatisticsResult,
    calculate_decadal_summary,
    calculate_descriptive_statistics,
    calculate_warmth_percentiles,
)
from climate_analyzer.data_validation import SchemaValidationError


# ---------------------------------------------------------------------------
# Fixtures and Synthetic Generators
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_annual_df() -> pd.DataFrame:
    """Fixture providing a clean 10-year annual summary (1990-1999)."""
    years = list(range(1990, 2000))
    # Synthetic anomalies with known properties: mean = 0.50
    anomalies = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    return pd.DataFrame(
        {
            "year": years,
            "annual_mean": anomalies,
            "complete_year": True,
            "observed_months": 12,
        }
    )


@pytest.fixture
def annual_df_with_incomplete() -> pd.DataFrame:
    """Fixture with 10 complete years and 1 incomplete trailing year."""
    years = list(range(2010, 2021))  # 2010 to 2020 (11 years)
    anomalies = [0.70, 0.60, 0.65, 0.68, 0.75, 0.90, 1.02, 0.93, 0.85, 0.98, 1.25]
    complete = [True] * 10 + [False]
    months = [12] * 10 + [8]
    return pd.DataFrame(
        {
            "year": years,
            "annual_mean": anomalies,
            "complete_year": complete,
            "observed_months": months,
        }
    )


# ---------------------------------------------------------------------------
# Descriptive Statistics Tests
# ---------------------------------------------------------------------------


def test_descriptive_statistics_on_known_series():
    """Verify exact mathematical moments on a simple synthetic dataset."""
    # Data: [1.0, 2.0, 3.0, 4.0, 5.0]
    # N = 5, Mean = 3.0, Std (sample, ddof=1) = sqrt(2.5) ≈ 1.5811388
    # Min = 1.0, Max = 5.0, Median = 3.0, IQR = 4.0 - 2.0 = 2.0
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    res = calculate_descriptive_statistics(series)

    assert isinstance(res, DescriptiveStatisticsResult)
    assert res.count == 5
    assert np.isclose(res.mean, 3.0)
    assert np.isclose(res.std, np.sqrt(2.5))
    assert np.isclose(res.min, 1.0)
    assert np.isclose(res.max, 5.0)
    assert np.isclose(res.range, 4.0)
    assert np.isclose(res.median, 3.0)
    assert np.isclose(res.iqr, 2.0)
    assert np.isclose(res.skewness, 0.0)  # perfectly symmetric
    assert 50 in res.percentiles
    assert np.isclose(res.percentiles[50], 3.0)

    # Dictionary conversion
    d = res.to_dict()
    assert d["count"] == 5
    assert d["mean"] == 3.0


def test_descriptive_statistics_dataframe_excludes_incomplete(annual_df_with_incomplete):
    """Verify that incomplete years are excluded by default when passing a DataFrame."""
    # When exclude_incomplete=True (default), 2020 (anomaly=1.25, complete=False) must be ignored.
    res_default = calculate_descriptive_statistics(annual_df_with_incomplete)
    assert res_default.count == 10
    assert res_default.max == 1.02  # 2016 anomaly; 1.25 was excluded

    # When exclude_incomplete=False, all 11 years are included
    res_all = calculate_descriptive_statistics(
        annual_df_with_incomplete, exclude_incomplete=False
    )
    assert res_all.count == 11
    assert res_all.max == 1.25


def test_descriptive_statistics_custom_percentiles():
    """Verify custom percentile thresholds and linear interpolation."""
    s = pd.Series(np.linspace(0.0, 100.0, 101))  # 0 to 100
    res = calculate_descriptive_statistics(s, percentiles=[10, 33, 67, 90])
    assert set(res.percentiles.keys()) == {10, 33, 67, 90}
    assert np.isclose(res.percentiles[10], 10.0)
    assert np.isclose(res.percentiles[33], 33.0)
    assert np.isclose(res.percentiles[90], 90.0)


def test_descriptive_statistics_invalid_inputs():
    """Verify descriptive statistics fails clearly on empty or invalid inputs."""
    # Empty series
    with pytest.raises(ValueError, match="contains no valid"):
        calculate_descriptive_statistics(pd.Series([], dtype=float))

    # All NaNs
    with pytest.raises(ValueError, match="contains no valid"):
        calculate_descriptive_statistics(pd.Series([np.nan, np.nan]))

    # Invalid DataFrame missing 'annual_mean'
    with pytest.raises(SchemaValidationError, match="not found in DataFrame"):
        calculate_descriptive_statistics(pd.DataFrame({"year": [1990, 1991]}))

    # Percentiles out of bounds
    with pytest.raises(ValueError, match="[Pp]ercentile.*must be in the range"):
        calculate_descriptive_statistics(pd.Series([1.0, 2.0]), percentiles=[-5, 50])


# ---------------------------------------------------------------------------
# Decadal Summary Tests
# ---------------------------------------------------------------------------


def test_decadal_summary_grouping_boundaries():
    """Verify calendar decade boundaries: 1980-1989, 1990-1999, etc."""
    years = list(range(1980, 2000))  # exactly 2 decades (1980s, 1990s)
    # 1980s: 0.1 each (mean=0.1), 1990s: 0.3 each (mean=0.3)
    anomalies = [0.1] * 10 + [0.3] * 10
    df = pd.DataFrame(
        {
            "year": years,
            "annual_mean": anomalies,
            "complete_year": True,
            "observed_months": 12,
        }
    )

    decadal = calculate_decadal_summary(df)
    assert len(decadal) == 2
    assert list(decadal["decade"]) == [1980, 1990]
    assert list(decadal["decade_label"]) == ["1980-1989", "1990-1999"]
    assert list(decadal["valid_years_count"]) == [10, 10]
    assert list(decadal["is_complete_decade"]) == [True, True]
    assert np.isclose(decadal.loc[0, "mean_anomaly"], 0.1)
    assert np.isclose(decadal.loc[1, "mean_anomaly"], 0.3)


def test_decadal_summary_incomplete_decade_and_threshold():
    """Verify that decades with missing years or incomplete years are flagged."""
    # Decade 2010-2019: 10 complete years -> complete
    # Decade 2020-2029: 2020 complete, 2021 incomplete -> only 1 complete year -> incomplete
    years = list(range(2010, 2022))
    anomalies = [0.5] * 10 + [0.8, 0.9]
    complete = [True] * 11 + [False]  # 2021 is incomplete
    df = pd.DataFrame(
        {
            "year": years,
            "annual_mean": anomalies,
            "complete_year": complete,
            "observed_months": [12] * 11 + [6],
        }
    )

    decadal = calculate_decadal_summary(df)
    assert len(decadal) == 2
    assert decadal.loc[0, "is_complete_decade"] == True
    assert decadal.loc[0, "valid_years_count"] == 10

    # 2020s has only 1 complete year (2020)
    assert decadal.loc[1, "is_complete_decade"] == False
    assert decadal.loc[1, "valid_years_count"] == 1
    assert decadal.loc[1, "missing_years_count"] == 9

    # Test min_years_threshold: if threshold=3, 2020s must be dropped
    filtered = calculate_decadal_summary(df, min_years_threshold=3)
    assert len(filtered) == 1
    assert filtered.loc[0, "decade"] == 2010


def test_decadal_summary_input_validation():
    """Verify schema and threshold validation in calculate_decadal_summary."""
    with pytest.raises(SchemaValidationError, match="Missing required columns"):
        calculate_decadal_summary(pd.DataFrame({"year": [1990]}))

    with pytest.raises(ValueError, match="min_years_threshold must be an integer between 1 and 10"):
        calculate_decadal_summary(
            pd.DataFrame({"year": [1990], "annual_mean": [0.1]}), min_years_threshold=12
        )


# ---------------------------------------------------------------------------
# Warmth Percentiles and Rankings Tests
# ---------------------------------------------------------------------------


def test_warmth_percentiles_ranking_and_reference_period():
    """Verify percentile calculation against explicit reference period."""
    # 5 years: 1980 (0.10), 1981 (0.20), 1982 (0.30), 1983 (0.40), 1984 (0.50)
    years = [1980, 1981, 1982, 1983, 1984]
    anomalies = [0.10, 0.20, 0.30, 0.40, 0.50]
    df = pd.DataFrame(
        {
            "year": years,
            "annual_mean": anomalies,
            "complete_year": True,
            "observed_months": 12,
        }
    )

    # Calculate against reference period 1980-1984 (the entire set)
    ranked = calculate_warmth_percentiles(
        df, reference_period=(1980, 1984), ranking_order="warmest_first"
    )

    # 1984 is warmest -> rank 1
    # 1980 is coolest -> rank 5
    assert ranked.loc[ranked["year"] == 1984, "rank"].values[0] == 1
    assert ranked.loc[ranked["year"] == 1980, "rank"].values[0] == 5

    # Check warmth percentile against reference distribution
    # Warmest value (0.50): all 5 reference values <= 0.50 -> 100%
    assert ranked.loc[ranked["year"] == 1984, "percentile"].values[0] == 100.0
    # Coolest value (0.10): 1 of 5 reference values <= 0.10 -> 20%
    assert ranked.loc[ranked["year"] == 1980, "percentile"].values[0] == 20.0


def test_warmth_percentiles_ties_and_sub_period():
    """Verify handling of tied anomalies and separate reference period."""
    # Reference period: 1951-1955 with values [0.0, 0.0, 0.1, 0.2, 0.2]
    # Evaluated period: 1951-1956 where 1956 is 0.25 (warmer than all reference)
    years = [1951, 1952, 1953, 1954, 1955, 1956]
    anomalies = [0.0, 0.0, 0.1, 0.2, 0.2, 0.25]
    df = pd.DataFrame(
        {
            "year": years,
            "annual_mean": anomalies,
            "complete_year": True,
            "observed_months": 12,
        }
    )

    ranked = calculate_warmth_percentiles(
        df, reference_period=(1951, 1955), ties_method="min"
    )

    # 1956 has anomaly 0.25, which exceeds all 5 reference values -> 100%
    assert ranked.loc[ranked["year"] == 1956, "percentile"].values[0] == 100.0

    # Tied values 0.0: 2 reference values <= 0.0 -> 2/5 = 40.0%
    assert ranked.loc[ranked["year"] == 1951, "percentile"].values[0] == 40.0
    assert ranked.loc[ranked["year"] == 1952, "percentile"].values[0] == 40.0


def test_warmth_percentiles_insufficient_reference():
    """Verify clear error when reference period contains fewer than 3 samples."""
    df = pd.DataFrame(
        {
            "year": [1980, 1981, 1982],
            "annual_mean": [0.1, 0.2, 0.3],
            "complete_year": True,
        }
    )
    # Reference period 1980-1981 has only 2 samples
    with pytest.raises(ValueError, match="at least 3 valid observations"):
        calculate_warmth_percentiles(df, reference_period=(1980, 1981))


def test_warmth_percentiles_defensive_copies():
    """Ensure original input DataFrame is never mutated by analysis."""
    df = pd.DataFrame(
        {
            "year": [1990, 1991, 1992, 1993],
            "annual_mean": [0.2, 0.3, 0.4, 0.5],
            "complete_year": True,
        }
    )
    cols_before = list(df.columns)
    _ = calculate_warmth_percentiles(df, reference_period=(1990, 1993))
    assert list(df.columns) == cols_before


# ---------------------------------------------------------------------------
# Numerical Audit & Boundary Tests (Phase 6 Audit)
# ---------------------------------------------------------------------------


def test_warmth_percentiles_manual_verification_and_parameter_styles():
    """Verify warmth percentiles with manually calculated ranks and both parameter styles."""
    # Data: 6 years
    # 2000: 0.10
    # 2001: 0.20
    # 2002: 0.30
    # 2003: 0.40
    # 2004: 0.50 (outside reference period)
    # 1999: 0.05 (outside reference period)
    years = [1999, 2000, 2001, 2002, 2003, 2004]
    anomalies = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    df = pd.DataFrame(
        {"year": years, "annual_mean": anomalies, "complete_year": True}
    )

    # Reference distribution: 2000-2003 (values: 0.10, 0.20, 0.30, 0.40; N=4)
    # Style A: reference_period=(2000, 2003)
    res_a = calculate_warmth_percentiles(
        df, reference_period=(2000, 2003), ranking_order="warmest_first"
    )
    # Style B: reference_start=2000, reference_end=2003
    res_b = calculate_warmth_percentiles(
        df, reference_start=2000, reference_end=2003, ranking_order="warmest_first"
    )

    # Check identical behavior across both parameter styles
    pd.testing.assert_frame_equal(res_a, res_b)

    # 1999 (0.05): strictly below all 4 reference values -> 0 of 4 <= 0.05 -> 0.0%
    row_1999 = res_a[res_a["year"] == 1999].iloc[0]
    assert np.isclose(row_1999["percentile"], 0.0)
    assert row_1999["warmth_rank"] == 6  # lowest of 6 evaluated years

    # 2004 (0.50): strictly above all 4 reference values -> 4 of 4 <= 0.50 -> 100.0%
    row_2004 = res_a[res_a["year"] == 2004].iloc[0]
    assert np.isclose(row_2004["percentile"], 100.0)
    assert row_2004["warmth_rank"] == 1  # highest of 6 evaluated years

    # 2001 (0.20): 2 of 4 reference values (0.10, 0.20) are <= 0.20 -> 50.0%
    row_2001 = res_a[res_a["year"] == 2001].iloc[0]
    assert np.isclose(row_2001["percentile"], 50.0)

    # Check coldest_first ranking order
    res_cold = calculate_warmth_percentiles(
        df, reference_period=(2000, 2003), ranking_order="coldest_first"
    )
    # In coldest_first, 1999 (0.05) must get rank 1
    assert res_cold[res_cold["year"] == 1999].iloc[0]["warmth_rank"] == 1
    # 2004 (0.50) must get rank 6
    assert res_cold[res_cold["year"] == 2004].iloc[0]["warmth_rank"] == 6


def test_decadal_summary_boundaries_and_missing_years():
    """Verify decadal boundaries, missing year handling, and partial decade flags."""
    # Create decade 1980-1989 with:
    # 1980 (boundary start), 1985 (middle), 1989 (boundary end)
    # Plus decade 1990-1999 completely missing
    # Plus decade 2020-2029 with 6 valid years (2020 to 2025)
    years = [1980, 1985, 1989] + list(range(2020, 2026))
    anomalies = [0.10, 0.20, 0.30] + [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    df = pd.DataFrame(
        {"year": years, "annual_mean": anomalies, "complete_year": True}
    )

    decadal = calculate_decadal_summary(df, min_valid_years=10)

    # Should have 2 decades in summary: 1980 and 2020
    assert len(decadal) == 2
    row_1980 = decadal[decadal["decade"] == 1980].iloc[0]
    row_2020 = decadal[decadal["decade"] == 2020].iloc[0]

    # Decade 1980-1989 checks:
    assert row_1980["decade_start"] == 1980
    assert row_1980["decade_end"] == 1989
    assert row_1980["valid_years"] == 3
    assert row_1980["missing_years_count"] == 7
    assert row_1980["is_complete"] == False
    assert np.isnan(row_1980["complete_decade_mean"])
    # Partial mean is arithmetic average of [0.10, 0.20, 0.30] = 0.20
    assert np.isclose(row_1980["partial_decade_mean"], 0.20)

    # Decade 2020-2029 checks (partial final decade):
    assert row_2020["decade_start"] == 2020
    assert row_2020["decade_end"] == 2029
    assert row_2020["valid_years"] == 6
    assert row_2020["missing_years_count"] == 4
    assert row_2020["is_complete"] == False
    assert np.isnan(row_2020["complete_decade_mean"])
    # Partial mean is average of 1.0 to 1.5 = 1.25
    assert np.isclose(row_2020["partial_decade_mean"], 1.25)
