"""
Deterministic unit and regression tests for the Streamlit application module.
"""

from __future__ import annotations

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from app.streamlit_app import (
    DEFAULT_RAW_DATA_PATH,
    build_processed_pipeline,
    compute_trend_comparison,
    extract_overview_kpis,
    filter_temporal_data,
    format_decadal_table_for_display,
    load_and_validate_raw_data,
    resolve_seasonal_reference_period,
)
from climate_analyzer.data_validation import SchemaValidationError
from climate_analyzer.analysis.descriptive_analysis import calculate_monthly_climatology


@pytest.fixture
def sample_monthly_clean_df() -> pd.DataFrame:
    """Fixture containing 2 complete years of clean monthly observations."""
    records = []
    for yr in [2020, 2021]:
        for m in range(1, 13):
            records.append({
                "year": yr,
                "month": m,
                "date": pd.Timestamp(year=yr, month=m, day=1),
                "anomaly": 0.5 + 0.01 * m,
            })
    return pd.DataFrame(records)


@pytest.fixture
def sample_annual_summary_df() -> pd.DataFrame:
    """Fixture containing complete and incomplete annual records."""
    records = []
    # 2018-2024: 7 complete years
    for yr in range(2018, 2025):
        val = 0.8 + (yr - 2018) * 0.05
        records.append({
            "year": yr,
            "annual_mean_anomaly": val,
            "valid_months": 12,
            "total_months": 12,
            "completeness_pct": 100.0,
            "is_complete": True,
            "partial_year_mean": val,
        })
    # 2025: complete year (warmest)
    records.append({
        "year": 2025,
        "annual_mean_anomaly": 1.35,
        "valid_months": 12,
        "total_months": 12,
        "completeness_pct": 100.0,
        "is_complete": True,
        "partial_year_mean": 1.35,
    })
    # 2026: incomplete ongoing year (9 months, partial mean 1.2344)
    records.append({
        "year": 2026,
        "annual_mean_anomaly": np.nan,
        "valid_months": 9,
        "total_months": 12,
        "completeness_pct": 75.0,
        "is_complete": False,
        "partial_year_mean": 1.2344444444444445,
    })
    return pd.DataFrame(records)


# ===========================================================================
# 1. Module Import & Pipeline Smoke Tests
# ===========================================================================


def test_streamlit_module_imports_cleanly():
    """Verify app.streamlit_app can be imported without side-effects."""
    import app.streamlit_app as st_app
    assert hasattr(st_app, "main")
    assert hasattr(st_app, "load_and_validate_raw_data")
    assert hasattr(st_app, "build_processed_pipeline")
    assert hasattr(st_app, "filter_temporal_data")
    assert hasattr(st_app, "extract_overview_kpis")
    assert hasattr(st_app, "compute_trend_comparison")
    assert hasattr(st_app, "resolve_seasonal_reference_period")
    assert hasattr(st_app, "format_decadal_table_for_display")


def test_load_and_validate_raw_data_missing_file_raises():
    """Verify helpful FileNotFoundError is raised on non-existent path."""
    with pytest.raises(FileNotFoundError, match="Dataset file not found at"):
        load_and_validate_raw_data("data/raw/non_existent_climate_file.csv")


def test_build_processed_pipeline_outputs_and_immutability(sample_monthly_clean_df):
    """Verify build_processed_pipeline produces the 3 expected pipeline DataFrames."""
    original = sample_monthly_clean_df.copy(deep=True)
    feat_df, annual_df, decadal_df = build_processed_pipeline(sample_monthly_clean_df)

    # Output structure assertions
    assert isinstance(feat_df, pd.DataFrame)
    assert isinstance(annual_df, pd.DataFrame)
    assert isinstance(decadal_df, pd.DataFrame)

    assert "anomaly_rolling_12m" in feat_df.columns
    assert "anomaly_rolling_60m" in feat_df.columns
    assert "annual_mean_anomaly" in annual_df.columns
    assert "complete_decade_mean" in decadal_df.columns or "mean_anomaly" in decadal_df.columns

    # Verify input DataFrame is not mutated
    pd.testing.assert_frame_equal(sample_monthly_clean_df, original)


# ===========================================================================
# 2. Temporal Filtering Tests
# ===========================================================================


def test_filter_temporal_data_valid_range(sample_monthly_clean_df, sample_annual_summary_df):
    """Verify valid year ranges slice both monthly and annual series correctly."""
    feat_sub, annual_sub = filter_temporal_data(
        sample_monthly_clean_df, sample_annual_summary_df, 2020, 2021
    )
    assert feat_sub["year"].min() == 2020
    assert feat_sub["year"].max() == 2021
    assert annual_sub["year"].min() == 2020
    assert annual_sub["year"].max() == 2021


def test_filter_temporal_data_reversed_range_raises(sample_monthly_clean_df, sample_annual_summary_df):
    """Verify start_year > end_year raises ValueError."""
    with pytest.raises(ValueError, match="cannot exceed end_year"):
        filter_temporal_data(sample_monthly_clean_df, sample_annual_summary_df, 2025, 2020)


def test_filter_temporal_data_missing_year_column_raises(sample_monthly_clean_df, sample_annual_summary_df):
    """Verify missing 'year' column raises SchemaValidationError."""
    bad_feat = sample_monthly_clean_df.drop(columns=["year"])
    with pytest.raises(SchemaValidationError, match="feat_df must contain 'year' column"):
        filter_temporal_data(bad_feat, sample_annual_summary_df, 2020, 2021)

    bad_annual = sample_annual_summary_df.drop(columns=["year"])
    with pytest.raises(SchemaValidationError, match="annual_df must contain 'year' column"):
        filter_temporal_data(sample_monthly_clean_df, bad_annual, 2020, 2021)


# ===========================================================================
# 3. Overview KPI Extraction & Partial-Year Resolution Tests
# ===========================================================================


def test_extract_overview_kpis_normal(sample_annual_summary_df):
    """Verify KPI extraction separates complete from incomplete years."""
    kpis = extract_overview_kpis(sample_annual_summary_df)

    assert kpis["latest_complete_year"] == 2025
    assert kpis["latest_complete_val"] == pytest.approx(1.35)
    assert kpis["latest_incomplete_year"] == 2026
    assert kpis["latest_incomplete_val"] == pytest.approx(1.2344, abs=0.001)
    assert kpis["latest_incomplete_months"] == 9
    assert kpis["warmest_complete_year"] == 2025
    assert kpis["warmest_complete_val"] == pytest.approx(1.35)
    assert kpis["complete_years_count"] == 8


def test_extract_overview_kpis_empty_df():
    """Verify empty DataFrame returns safe default KPI dictionary."""
    empty_df = pd.DataFrame()
    kpis = extract_overview_kpis(empty_df)

    assert kpis["latest_complete_year"] is None
    assert kpis["latest_complete_val"] is None
    assert kpis["latest_incomplete_year"] is None
    assert kpis["latest_incomplete_val"] is None
    assert kpis["latest_incomplete_months"] is None
    assert kpis["warmest_complete_year"] is None
    assert kpis["warmest_complete_val"] is None
    assert kpis["complete_years_count"] == 0


def test_extract_overview_kpis_only_incomplete_years():
    """Verify handling when no complete years exist in the filtered range."""
    inc_df = pd.DataFrame([
        {
            "year": 2026,
            "annual_mean_anomaly": np.nan,
            "valid_months": 9,
            "total_months": 12,
            "completeness_pct": 75.0,
            "is_complete": False,
            "partial_year_mean": 1.2344,
        }
    ])
    kpis = extract_overview_kpis(inc_df)

    assert kpis["latest_complete_year"] is None
    assert kpis["latest_incomplete_year"] == 2026
    assert kpis["latest_incomplete_val"] == pytest.approx(1.2344)
    assert kpis["warmest_complete_year"] is None
    assert kpis["complete_years_count"] == 0


# ===========================================================================
# 4. Statistical Trend Comparison Tests
# ===========================================================================


def test_compute_trend_comparison_sufficient_data(sample_annual_summary_df):
    """Verify trend comparison calculates OLS, MK, and Sen's slope on complete years."""
    models = compute_trend_comparison(sample_annual_summary_df, min_years=5)
    assert models is not None
    assert "ols" in models
    assert "mann_kendall" in models
    assert "sens" in models

    # Regression performed strictly on 8 complete years (2018-2025)
    assert models["ols"].sample_count == 8
    assert models["ols"].start_year == 2018
    assert models["ols"].end_year == 2025
    assert models["mann_kendall"].sample_count == 8
    assert models["sens"].sample_count == 8


def test_compute_trend_comparison_insufficient_samples(sample_annual_summary_df):
    """Verify None is returned when fewer than min_years complete observations exist."""
    few_df = sample_annual_summary_df.head(4)  # 4 complete years
    assert compute_trend_comparison(few_df, min_years=5) is None
    assert compute_trend_comparison(pd.DataFrame(), min_years=5) is None


# ===========================================================================
# 5. Seasonal Reference Period Semantics Tests
# ===========================================================================


def test_resolve_seasonal_reference_period_presets():
    """Verify preset mappings to exact calendar year boundaries."""
    assert resolve_seasonal_reference_period("1951–1980 Baseline (Official NASA Base)") == (1951, 1980)
    assert resolve_seasonal_reference_period("Recent 30 Years (1995–2024)") == (1995, 2024)
    assert resolve_seasonal_reference_period("Full Available Record (1880–Present)") == (None, None)


def test_resolve_seasonal_reference_period_custom():
    """Verify custom window mappings and input validation."""
    assert resolve_seasonal_reference_period("Custom Reference Window", custom_range=(1970, 2000)) == (1970, 2000)

    with pytest.raises(ValueError, match="custom_range tuple must be provided"):
        resolve_seasonal_reference_period("Custom Reference Window", custom_range=None)

    with pytest.raises(ValueError, match="cannot exceed end year"):
        resolve_seasonal_reference_period("Custom Reference Window", custom_range=(2010, 1990))


def test_calculate_monthly_climatology_edge_cases(sample_monthly_clean_df):
    """Verify monthly climatology raises expected errors on empty or invalid windows."""
    # Start > end
    with pytest.raises(ValueError, match="reference_start.*>.*reference_end"):
        calculate_monthly_climatology(sample_monthly_clean_df, reference_start=2021, reference_end=2020)

    # Window outside dataset
    with pytest.raises(ValueError, match="No observations available in specified reference window"):
        calculate_monthly_climatology(sample_monthly_clean_df, reference_start=1950, reference_end=1960)


# ===========================================================================
# 6. Decadal Table Display Formatting Tests
# ===========================================================================


def test_format_decadal_table_for_display():
    """Verify decadal display formatter maps completeness and column headers."""
    dec_df = pd.DataFrame([
        {
            "decade_label": "2010-2019",
            "is_complete": True,
            "valid_years": 10,
            "complete_decade_mean": 0.85,
            "partial_decade_mean": 0.85,
        },
        {
            "decade_label": "2020-2029",
            "is_complete": False,
            "valid_years": 6,
            "complete_decade_mean": np.nan,
            "partial_decade_mean": 1.06,
        },
    ])

    formatted = format_decadal_table_for_display(dec_df)
    assert not formatted.empty
    assert "Decade Window" in formatted.columns
    assert "Completeness Status" in formatted.columns
    assert formatted.loc[0, "Completeness Status"] == "Complete (10y)"
    assert formatted.loc[1, "Completeness Status"] == "Partial (Diagnostic)"


def test_format_decadal_table_for_display_empty():
    """Verify empty DataFrame returns empty DataFrame."""
    assert format_decadal_table_for_display(pd.DataFrame()).empty


# ===========================================================================
# 7. Real Data Integration & Partial-Year 2026 Regression Test
# ===========================================================================


@pytest.mark.skipif(not DEFAULT_RAW_DATA_PATH.exists(), reason="Real raw dataset not present")
def test_real_data_partial_year_2026_exact_derivation():
    """Regression test: verify 2026 partial-year mean is dynamically derived as +1.2344°C from 9 observed months."""
    df_raw = load_and_validate_raw_data(str(DEFAULT_RAW_DATA_PATH))
    feat_df, annual_df, decadal_df = build_processed_pipeline(df_raw)

    # 1. Inspect raw 2026 observations
    raw_2026 = df_raw[df_raw["year"] == 2026]
    valid_months_2026 = raw_2026["anomaly"].dropna()
    assert len(valid_months_2026) == 9
    expected_sum = 11.11
    expected_mean = expected_sum / 9.0  # 1.234444...

    assert valid_months_2026.sum() == pytest.approx(expected_sum, abs=0.001)
    assert valid_months_2026.mean() == pytest.approx(expected_mean, abs=0.0001)

    # 2. Check annual summary row for 2026
    row_2026 = annual_df[annual_df["year"] == 2026].iloc[0]
    assert row_2026["is_complete"] == False
    assert np.isnan(row_2026["annual_mean_anomaly"])  # Complete mean MUST be NaN
    assert row_2026["valid_months"] == 9
    assert row_2026["partial_year_mean"] == pytest.approx(expected_mean, abs=0.0001)

    # 3. Check KPI extraction produces matching diagnostic metric
    kpis = extract_overview_kpis(annual_df)
    assert kpis["latest_incomplete_year"] == 2026
    assert kpis["latest_incomplete_months"] == 9
    assert kpis["latest_incomplete_val"] == pytest.approx(expected_mean, abs=0.0001)

    # 4. Check that 2026 is NEVER used in complete-year trend estimation
    complete_years = annual_df[annual_df["is_complete"] == True]
    assert 2026 not in complete_years["year"].values
    assert complete_years["year"].max() == 2025

    models = compute_trend_comparison(complete_years)
    assert models is not None
    assert models["ols"].end_year == 2025
