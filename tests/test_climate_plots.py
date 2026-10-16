"""
Deterministic test suite for climate visualization functions.

Tests verify:
- Valid Matplotlib Figure objects returned.
- Non-interactive backend compatibility ('Agg').
- Axis labels, titles, and reference baselines (1951–1980).
- Preservation of missing observations (no zero-imputation or interpolation).
- Complete vs. incomplete annual mean distinction.
- Complete vs. partial decade distinction.
- Seasonal climatology calculation and reference period filtering.
- Input DataFrame immutability (defensive copy enforcement).
- Correct exception handling on invalid/empty inputs.
- Output file generation when explicit output_path is provided.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for tests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from climate_analyzer.analysis.descriptive_analysis import (
    calculate_decadal_summary,
    calculate_monthly_climatology,
)
from climate_analyzer.data_validation import SchemaValidationError
from climate_analyzer.visualization.climate_plots import (
    plot_annual_anomalies,
    plot_decadal_summary,
    plot_monthly_anomalies,
    plot_seasonal_patterns,
)


@pytest.fixture(autouse=True)
def close_figures():
    """Ensure all figures are cleanly closed after each test to prevent memory leaks."""
    yield
    plt.close("all")


@pytest.fixture
def sample_monthly_df() -> pd.DataFrame:
    """Deterministic 3-year monthly time series (2020-2022) with known gaps."""
    records = []
    # 2020: Complete (12 months), mean ~ 0.5
    for m in range(1, 13):
        records.append({
            "year": 2020,
            "month": m,
            "date": pd.Timestamp(year=2020, month=m, day=1),
            "anomaly": 0.4 + (m * 0.02),
        })
    # 2021: Has a missing month (month 6 is NaN)
    for m in range(1, 13):
        records.append({
            "year": 2021,
            "month": m,
            "date": pd.Timestamp(year=2021, month=m, day=1),
            "anomaly": np.nan if m == 6 else 0.6 + (m * 0.01),
        })
    # 2022: Complete (12 months), mean ~ 0.8
    for m in range(1, 13):
        records.append({
            "year": 2022,
            "month": m,
            "date": pd.Timestamp(year=2022, month=m, day=1),
            "anomaly": 0.7 + (m * 0.015),
        })
    return pd.DataFrame(records)


@pytest.fixture
def sample_annual_df() -> pd.DataFrame:
    """Deterministic annual summary (2015-2026) with complete years and one incomplete year."""
    records = []
    for yr in range(2015, 2026):
        # 2015-2025: 11 complete years with an increasing linear anomaly
        val = -0.2 + (yr - 2015) * 0.1
        records.append({
            "year": yr,
            "annual_mean_anomaly": val,
            "valid_months": 12,
            "total_months": 12,
            "completeness_pct": 100.0,
            "is_complete": True,
            "partial_year_mean": val,
        })
    # 2026: Incomplete ongoing year (9 months observed, annual_mean_anomaly is NaN)
    records.append({
        "year": 2026,
        "annual_mean_anomaly": np.nan,
        "valid_months": 9,
        "total_months": 12,
        "completeness_pct": 75.0,
        "is_complete": False,
        "partial_year_mean": 1.15,
    })
    return pd.DataFrame(records)


# ===========================================================================
# 1. Monthly Anomalies Tests
# ===========================================================================


class TestPlotMonthlyAnomalies:
    """Test suite for plot_monthly_anomalies."""

    def test_returns_valid_figure_and_axes(self, sample_monthly_df):
        fig = plot_monthly_anomalies(sample_monthly_df)
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert "Temperature Anomaly (°C)" in ax.get_ylabel()
        assert "Observation Date" in ax.get_xlabel()
        assert "1951–1980" in ax.get_title()

    def test_missing_values_preserved_as_nan_breaks(self, sample_monthly_df):
        """Verify NaNs are NOT converted to zero or dropped from the x-axis grid."""
        fig = plot_monthly_anomalies(sample_monthly_df)
        ax = fig.axes[0]
        lines = ax.get_lines()
        # Primary series is the first line
        y_data = lines[0].get_ydata()
        assert np.isnan(y_data).any(), "NaN gap in monthly anomaly series must be preserved!"
        assert not (y_data == 0.0).all()

    def test_rolling_average_overlay(self, sample_monthly_df):
        df_with_rolling = sample_monthly_df.copy()
        df_with_rolling["anomaly_rolling_12m"] = df_with_rolling["anomaly"].rolling(12, min_periods=9).mean()
        fig = plot_monthly_anomalies(df_with_rolling, rolling_col="anomaly_rolling_12m")
        ax = fig.axes[0]
        legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert any("Rolling Average" in label for label in legend_labels)

    def test_defensive_copy_no_mutation(self, sample_monthly_df):
        original = sample_monthly_df.copy(deep=True)
        plot_monthly_anomalies(sample_monthly_df, start_year=2021)
        pd.testing.assert_frame_equal(sample_monthly_df, original)

    def test_invalid_inputs_raise_errors(self, sample_monthly_df):
        with pytest.raises(TypeError, match="Expected pandas DataFrame"):
            plot_monthly_anomalies("not_a_df")  # type: ignore

        with pytest.raises(ValueError, match="empty DataFrame"):
            plot_monthly_anomalies(pd.DataFrame())

        with pytest.raises(SchemaValidationError, match="Required anomaly column"):
            plot_monthly_anomalies(pd.DataFrame({"date": [pd.Timestamp("2020-01-01")]}))

        with pytest.raises(ValueError, match="cannot exceed end_year"):
            plot_monthly_anomalies(sample_monthly_df, start_year=2025, end_year=2020)

        # All-NaN anomaly series
        all_nan_df = sample_monthly_df.copy()
        all_nan_df["anomaly"] = np.nan
        with pytest.raises(ValueError, match="contains no valid observations"):
            plot_monthly_anomalies(all_nan_df)

    def test_save_figure_to_output_path(self, sample_monthly_df, tmp_path):
        out_file = tmp_path / "subfolder" / "monthly_plot.png"
        assert not out_file.exists()
        fig = plot_monthly_anomalies(sample_monthly_df, output_path=out_file)
        assert out_file.exists()
        assert out_file.stat().st_size > 0
        # Ensure returned figure is still open
        assert plt.fignum_exists(fig.number)


# ===========================================================================
# 2. Annual Anomalies Tests
# ===========================================================================


class TestPlotAnnualAnomalies:
    """Test suite for plot_annual_anomalies."""

    def test_returns_valid_figure(self, sample_annual_df):
        fig = plot_annual_anomalies(sample_annual_df)
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert "Temperature Anomaly (°C)" in ax.get_ylabel()
        assert "Calendar Year" in ax.get_xlabel()
        assert "1951–1980" in ax.get_title()

    def test_complete_vs_incomplete_distinction(self, sample_annual_df):
        """Incomplete year (2026) must have distinct visual representation and legend label."""
        fig = plot_annual_anomalies(sample_annual_df)
        ax = fig.axes[0]
        legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "Complete Annual Mean (°C)" in legend_labels
        assert "Incomplete Year (Diagnostic Partial Mean)" in legend_labels

        # Total bars must equal total rows (11 complete + 1 incomplete = 12 bars)
        patches = ax.patches
        assert len(patches) == len(sample_annual_df)

        # The last patch (2026) must be hatched
        last_patch = patches[-1]
        assert last_patch.get_hatch() == "//", "Incomplete year bar must be hatched!"

    def test_ols_trend_overlay_on_complete_years_only(self, sample_annual_df):
        """OLS trend line must be fit strictly on complete years."""
        fig = plot_annual_anomalies(sample_annual_df, show_ols_trend=True)
        ax = fig.axes[0]
        legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert any("OLS Trend:" in label for label in legend_labels)
        assert any("°C/decade" in label for label in legend_labels)

        # Check disclaimer text in title
        title_text = ax.get_title()
        assert "Conventional 95% CI assumes independent residuals and is not autocorrelation-adjusted" in title_text
        assert "2015–2025" in title_text, "OLS trend should cover 2015-2025, excluding 2026!"

    def test_defensive_copy(self, sample_annual_df):
        original = sample_annual_df.copy(deep=True)
        plot_annual_anomalies(sample_annual_df, start_year=2018)
        pd.testing.assert_frame_equal(sample_annual_df, original)

    def test_invalid_inputs(self, sample_annual_df):
        with pytest.raises(TypeError):
            plot_annual_anomalies(None)  # type: ignore

        with pytest.raises(ValueError, match="empty DataFrame"):
            plot_annual_anomalies(pd.DataFrame())

        with pytest.raises(SchemaValidationError, match="Required column 'year'"):
            plot_annual_anomalies(pd.DataFrame({"annual_mean_anomaly": [0.1]}))

        with pytest.raises(ValueError, match="at least 5 complete annual observations"):
            # Only 3 complete years
            plot_annual_anomalies(sample_annual_df.iloc[:3], show_ols_trend=True)

    def test_save_figure(self, sample_annual_df, tmp_path):
        out_file = tmp_path / "annual_plot.png"
        fig = plot_annual_anomalies(sample_annual_df, output_path=out_file)
        assert out_file.exists()
        assert out_file.stat().st_size > 0
        assert plt.fignum_exists(fig.number)


# ===========================================================================
# 3. Decadal Summary Tests
# ===========================================================================


class TestPlotDecadalSummary:
    """Test suite for plot_decadal_summary."""

    def test_from_annual_dataframe_computes_and_plots(self, sample_annual_df):
        fig = plot_decadal_summary(sample_annual_df)
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert "Decadal Mean Anomaly (°C)" in ax.get_ylabel()
        assert "Calendar Decade" in ax.get_xlabel()
        legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
        # 2015-2019 is in 2010s (5 complete years -> partial), 2020-2025 is in 2020s (6 complete years -> partial)
        assert any("Partial Decade" in label for label in legend_labels)

    def test_from_precomputed_decadal_summary(self):
        # 1 complete decade, 1 partial decade
        decadal_data = pd.DataFrame([
            {
                "decade": 2010,
                "decade_label": "2010-2019",
                "decade_start": 2010,
                "decade_end": 2019,
                "mean_anomaly": 0.75,
                "complete_decade_mean": 0.75,
                "partial_decade_mean": 0.75,
                "valid_years": 10,
                "is_complete": True,
            },
            {
                "decade": 2020,
                "decade_label": "2020-2029",
                "decade_start": 2020,
                "decade_end": 2029,
                "mean_anomaly": np.nan,
                "complete_decade_mean": np.nan,
                "partial_decade_mean": 1.10,
                "valid_years": 6,
                "is_complete": False,
            },
        ])
        fig = plot_decadal_summary(decadal_data)
        ax = fig.axes[0]
        legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "Complete Decade Mean (10 years)" in legend_labels
        assert any("Partial Decade Mean (Diagnostic" in label for label in legend_labels)

        # 2 bars
        assert len(ax.patches) == 2
        # Bar 0 is complete (not hatched)
        assert ax.patches[0].get_hatch() is None or ax.patches[0].get_hatch() == ""
        # Bar 1 is partial (hatched)
        assert ax.patches[1].get_hatch() == "//"

    def test_defensive_copy(self, sample_annual_df):
        original = sample_annual_df.copy(deep=True)
        plot_decadal_summary(sample_annual_df)
        pd.testing.assert_frame_equal(sample_annual_df, original)

    def test_invalid_inputs(self):
        with pytest.raises(TypeError):
            plot_decadal_summary(12345)  # type: ignore

        with pytest.raises(ValueError, match="empty DataFrame"):
            plot_decadal_summary(pd.DataFrame())

        with pytest.raises(SchemaValidationError, match="must contain either 'decade'"):
            plot_decadal_summary(pd.DataFrame({"foo": [1, 2]}))


# ===========================================================================
# 4. Monthly Climatology & Seasonal Patterns Tests
# ===========================================================================


class TestMonthlyClimatologyCalculation:
    """Test suite for calculate_monthly_climatology."""

    def test_calculation_on_known_values(self):
        # Create 2 years where Jan = 0.2 and 0.4 (mean 0.3), Feb = 0.6 and NaN (mean 0.6)
        data = pd.DataFrame([
            {"year": 2000, "month": 1, "anomaly": 0.2},
            {"year": 2001, "month": 1, "anomaly": 0.4},
            {"year": 2000, "month": 2, "anomaly": 0.6},
            {"year": 2001, "month": 2, "anomaly": np.nan},
        ])
        for m in range(3, 13):
            data = pd.concat([data, pd.DataFrame([{"year": 2000, "month": m, "anomaly": 0.0}])], ignore_index=True)

        res = calculate_monthly_climatology(data)
        assert len(res) == 12
        jan = res[res["month"] == 1].iloc[0]
        assert jan["count"] == 2
        assert pytest.approx(jan["mean_anomaly"], rel=1e-5) == 0.3
        assert pytest.approx(jan["std"], rel=1e-5) == np.std([0.2, 0.4], ddof=1)

        feb = res[res["month"] == 2].iloc[0]
        assert feb["count"] == 1
        assert pytest.approx(feb["mean_anomaly"], rel=1e-5) == 0.6

    def test_missing_values_not_filled_with_zero(self):
        """Missing values must never be imputed with 0.0."""
        data = pd.DataFrame([
            {"year": 2000, "month": 1, "anomaly": 1.0},
            {"year": 2001, "month": 1, "anomaly": np.nan},
            {"year": 2002, "month": 1, "anomaly": np.nan},
        ])
        # If NaNs were replaced with zero, mean would be 1.0 / 3 = 0.333
        # With correct conservative handling, mean is 1.0
        res = calculate_monthly_climatology(data, min_years_per_month=1)
        jan = res[res["month"] == 1].iloc[0]
        assert jan["count"] == 1
        assert pytest.approx(jan["mean_anomaly"], rel=1e-5) == 1.0

    def test_reference_period_filtering(self):
        data = pd.DataFrame([
            {"year": 1950, "month": 1, "anomaly": -0.5},
            {"year": 1960, "month": 1, "anomaly": 0.0},
            {"year": 1970, "month": 1, "anomaly": 0.2},
            {"year": 1990, "month": 1, "anomaly": 0.8},
        ])
        res = calculate_monthly_climatology(data, reference_start=1955, reference_end=1975)
        jan = res[res["month"] == 1].iloc[0]
        assert jan["count"] == 2
        assert pytest.approx(jan["mean_anomaly"], rel=1e-5) == 0.1  # mean of 0.0 and 0.2


class TestPlotSeasonalPatterns:
    """Test suite for plot_seasonal_patterns."""

    def test_returns_valid_figure_from_monthly_df(self, sample_monthly_df):
        fig = plot_seasonal_patterns(sample_monthly_df)
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert "Temperature Anomaly (°C)" in ax.get_ylabel()
        assert "Calendar Month" in ax.get_xlabel()
        assert "1951–1980" in ax.get_title()
        assert "No Zero-Imputation" in ax.get_title()

        # Check that 12 ticks exist for Jan-Dec
        ticks = ax.get_xticks()
        assert len(ticks) == 12

    def test_returns_valid_figure_with_reference_period(self, sample_monthly_df):
        fig = plot_seasonal_patterns(sample_monthly_df, reference_start=2020, reference_end=2021)
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert "2020–2021" in ax.get_title()

    def test_defensive_copy(self, sample_monthly_df):
        original = sample_monthly_df.copy(deep=True)
        plot_seasonal_patterns(sample_monthly_df)
        pd.testing.assert_frame_equal(sample_monthly_df, original)

    def test_invalid_inputs(self, sample_monthly_df):
        with pytest.raises(TypeError):
            plot_seasonal_patterns("invalid")  # type: ignore

        with pytest.raises(ValueError, match="empty DataFrame"):
            plot_seasonal_patterns(pd.DataFrame())

        with pytest.raises(SchemaValidationError, match="must contain 'month'"):
            plot_seasonal_patterns(pd.DataFrame({"year": [2020], "anomaly": [0.5]}))

        with pytest.raises(ValueError, match="reference_start .* > reference_end"):
            plot_seasonal_patterns(sample_monthly_df, reference_start=2025, reference_end=2020)

    def test_save_figure(self, sample_monthly_df, tmp_path):
        out_file = tmp_path / "seasonal_plot.png"
        fig = plot_seasonal_patterns(sample_monthly_df, output_path=out_file)
        assert out_file.exists()
        assert out_file.stat().st_size > 0
        assert plt.fignum_exists(fig.number)
