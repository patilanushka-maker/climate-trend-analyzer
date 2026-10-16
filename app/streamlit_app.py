"""
Climate Trend Analyzer — Streamlit Interactive Exploration Dashboard.

This application provides an interactive presentation layer for the NASA GISTEMP v4
global land-ocean temperature anomaly series (1951–1980 baseline).

SCIENTIFIC INTEGRITY & DESIGN PRINCIPLES:
- Ingestion and transformations reuse the core `climate_analyzer` package modules.
- Raw datasets are never mutated, modified in-place, or downloaded automatically.
- Missing values are strictly preserved; zero-imputation is prohibited.
- Incomplete years (such as 2026) are never labeled or analyzed as complete annual means.
- Partial decades (e.g. 2020–2029) are explicitly labeled as ongoing diagnostic averages.
- Conventional OLS confidence intervals and Mann-Kendall p-values explicitly document
  their sensitivity to serial autocorrelation ($r_1 \\approx +0.80$, Durbin-Watson $d \\approx 0.33$).
- Statistical trends describe historical observations and do not imply causal attribution.
- Matplotlib figures are closed immediately after rendering to prevent memory leaks.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Headless backend for server execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from climate_analyzer.analysis.descriptive_analysis import (
    calculate_decadal_summary,
    calculate_descriptive_statistics,
    calculate_monthly_climatology,
    calculate_warmth_percentiles,
)
from climate_analyzer.analysis.trend_analysis import (
    estimate_ols_trend,
    estimate_sens_slope,
    test_mann_kendall_trend,
)
from climate_analyzer.data_cleaning import clean_monthly_gistemp
from climate_analyzer.data_loader import (
    DataLoaderError,
    load_gistemp_monthly,
)
from climate_analyzer.data_validation import (
    SchemaValidationError,
    validate_monthly_series,
)
from climate_analyzer.feature_engineering import (
    add_rolling_features,
    calculate_annual_summary,
)
from climate_analyzer.visualization.climate_plots import (
    plot_annual_anomalies,
    plot_decadal_summary,
    plot_monthly_anomalies,
    plot_seasonal_patterns,
)


DEFAULT_RAW_DATA_PATH = Path("data/raw/GLB.Ts+dSST.csv")


# ---------------------------------------------------------------------------
# Cached Data Pipeline Routines
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_and_validate_raw_data(filepath_str: str) -> pd.DataFrame:
    """Load raw NASA GISTEMP CSV and validate monthly structure."""
    path = Path(filepath_str)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path.resolve()}")
    df = load_gistemp_monthly(path)
    validate_monthly_series(df, raise_on_error=True)
    return df


@st.cache_data(show_spinner=False)
def build_processed_pipeline(monthly_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute conservative cleaning, feature engineering, and aggregations."""
    clean_df = clean_monthly_gistemp(monthly_raw)
    feat_df = add_rolling_features(clean_df, windows=(12, 60, 120))
    annual_df = calculate_annual_summary(feat_df)
    decadal_df = calculate_decadal_summary(annual_df)
    return feat_df, annual_df, decadal_df


# ---------------------------------------------------------------------------
# Business Logic & Data Filtering Helpers (Pure Functions for Easy Testing)
# ---------------------------------------------------------------------------


def filter_temporal_data(
    feat_df: pd.DataFrame,
    annual_df: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter monthly feature and annual DataFrames to [start_year, end_year] inclusive."""
    if start_year > end_year:
        raise ValueError(f"start_year ({start_year}) cannot exceed end_year ({end_year}).")
    if "year" not in feat_df.columns:
        raise SchemaValidationError("feat_df must contain 'year' column.")
    if "year" not in annual_df.columns:
        raise SchemaValidationError("annual_df must contain 'year' column.")

    feat_filtered = feat_df[(feat_df["year"] >= start_year) & (feat_df["year"] <= end_year)].copy()
    annual_filtered = annual_df[(annual_df["year"] >= start_year) & (annual_df["year"] <= end_year)].copy()
    return feat_filtered, annual_filtered


def extract_overview_kpis(annual_df: pd.DataFrame) -> dict:
    """Extract summary KPIs from an annual DataFrame, segregating complete from incomplete years."""
    if annual_df.empty:
        return {
            "latest_complete_year": None,
            "latest_complete_val": None,
            "latest_incomplete_year": None,
            "latest_incomplete_val": None,
            "latest_incomplete_months": None,
            "warmest_complete_year": None,
            "warmest_complete_val": None,
            "complete_years_count": 0,
        }

    complete_df = annual_df[annual_df["is_complete"] == True]
    incomplete_df = annual_df[annual_df["is_complete"] == False]

    latest_complete = complete_df.iloc[-1] if not complete_df.empty else None
    latest_incomplete = incomplete_df.iloc[-1] if not incomplete_df.empty else None

    warmest_row = (
        complete_df.loc[complete_df["annual_mean_anomaly"].idxmax()]
        if not complete_df.empty and complete_df["annual_mean_anomaly"].notna().any()
        else None
    )

    return {
        "latest_complete_year": int(latest_complete["year"]) if latest_complete is not None else None,
        "latest_complete_val": float(latest_complete["annual_mean_anomaly"]) if latest_complete is not None else None,
        "latest_incomplete_year": int(latest_incomplete["year"]) if latest_incomplete is not None else None,
        "latest_incomplete_val": float(latest_incomplete["partial_year_mean"]) if latest_incomplete is not None else None,
        "latest_incomplete_months": int(latest_incomplete["valid_months"]) if latest_incomplete is not None else None,
        "warmest_complete_year": int(warmest_row["year"]) if warmest_row is not None else None,
        "warmest_complete_val": float(warmest_row["annual_mean_anomaly"]) if warmest_row is not None else None,
        "complete_years_count": len(complete_df),
    }


def compute_trend_comparison(
    complete_years_df: pd.DataFrame,
    min_years: int = 5,
) -> Optional[dict]:
    """Compute OLS, Mann-Kendall, and Sen's slope comparison across complete years."""
    if complete_years_df.empty:
        return None

    if "is_complete" in complete_years_df.columns:
        valid_complete = complete_years_df[
            (complete_years_df["is_complete"] == True) & complete_years_df["annual_mean_anomaly"].notna()
        ]
    else:
        valid_complete = complete_years_df[complete_years_df["annual_mean_anomaly"].notna()]

    if len(valid_complete) < min_years:
        return None

    ols_res = estimate_ols_trend(valid_complete, exclude_incomplete=True)
    mk_res = test_mann_kendall_trend(valid_complete, exclude_incomplete=True)
    sens_res = estimate_sens_slope(valid_complete, exclude_incomplete=True)

    return {
        "ols": ols_res,
        "mann_kendall": mk_res,
        "sens": sens_res,
    }


def resolve_seasonal_reference_period(
    period_preset: str,
    custom_range: Optional[Tuple[int, int]] = None,
) -> Tuple[Optional[int], Optional[int]]:
    """Map UI preset selection to (reference_start, reference_end) calendar years.

    Note: This specifies the summarization window across the 1951-1980 NASA anomalies.
    It does not rebaseline or re-center anomalies.
    """
    if "1951–1980" in period_preset or "1951-1980" in period_preset:
        return 1951, 1980
    elif "Recent 30 Years" in period_preset:
        return 1995, 2024
    elif "Custom" in period_preset:
        if custom_range is None:
            raise ValueError("custom_range tuple must be provided when Custom window is selected.")
        if custom_range[0] > custom_range[1]:
            raise ValueError(f"Custom start year ({custom_range[0]}) cannot exceed end year ({custom_range[1]}).")
        return custom_range[0], custom_range[1]
    elif "Full" in period_preset:
        return None, None
    else:
        return None, None


def format_decadal_table_for_display(decadal_df: pd.DataFrame) -> pd.DataFrame:
    """Format decadal summary table with human-readable columns and completeness status."""
    if decadal_df.empty:
        return pd.DataFrame()

    display_dec = decadal_df[[
        "decade_label", "is_complete", "valid_years", "complete_decade_mean", "partial_decade_mean"
    ]].copy()
    display_dec["Status"] = display_dec["is_complete"].apply(
        lambda x: "Complete (10y)" if x else "Partial (Diagnostic)"
    )
    display_dec = display_dec[[
        "decade_label", "Status", "valid_years", "complete_decade_mean", "partial_decade_mean"
    ]]
    display_dec.columns = [
        "Decade Window", "Completeness Status", "Valid Years", "Complete Mean (°C)", "Partial Mean (°C)"
    ]
    return display_dec


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------


def main():
    st.set_page_config(
        page_title="Climate Trend Analyzer",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Header and scientific scope banner
    st.title("🌍 Climate Trend Analyzer")
    st.markdown(
        "**Historical Surface Temperature Anomaly Analysis** — "
        "NASA GISTEMP v4 Land-Ocean Temperature Index (L-OTI) relative to the **1951–1980 baseline**."
    )

    # Sidebar: Data Source & Global Filters
    st.sidebar.header("📁 Data Source & Configuration")
    data_file_input = st.sidebar.text_input(
        "Dataset Path",
        value=str(DEFAULT_RAW_DATA_PATH),
        help="Local relative or absolute path to the official NASA GISTEMP GLB.Ts+dSST.csv file.",
    )

    data_path = Path(data_file_input)
    if not data_path.exists():
        st.error(
            f"⚠️ **Dataset File Missing:** Could not locate climate dataset at `{data_path}`.\n\n"
            "**Recovery Instructions:**\n"
            "1. Download the official CSV from NASA GISS: "
            "`https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv`\n"
            f"2. Place the file at `{DEFAULT_RAW_DATA_PATH}` or provide the custom path in the sidebar.\n"
            "3. The dashboard will automatically detect and load the dataset."
        )
        st.stop()

    try:
        monthly_raw = load_and_validate_raw_data(str(data_path))
        feat_df, annual_df, decadal_df = build_processed_pipeline(monthly_raw)
    except Exception as exc:
        st.error(f"❌ **Data Pipeline Error:** Failed to load or validate dataset: {exc}")
        st.stop()

    # Determine timeline boundaries
    min_year_avail = int(annual_df["year"].min())
    max_year_avail = int(annual_df["year"].max())

    st.sidebar.subheader("📅 Temporal Filter")
    selected_range = st.sidebar.slider(
        "Select Calendar Year Window",
        min_value=min_year_avail,
        max_value=max_year_avail,
        value=(min_year_avail, max_year_avail),
        step=1,
        help="Filters the analytical time series across all dashboard views.",
    )
    start_year, end_year = selected_range

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "### ℹ️ Scientific Context\n"
        "- **Base Period:** 1951–1980\n"
        "- **Metric:** Temperature Anomaly (°C)\n"
        "- **Zero Imputation:** Prohibited\n"
        "- **Serial Autocorrelation:** Noted on all statistical tests\n"
    )

    # Filtered subsets
    feat_filtered, annual_filtered = filter_temporal_data(feat_df, annual_df, start_year, end_year)

    # Main Tabs Layout
    tabs = st.tabs([
        "📊 Overview",
        "📈 Historical Trends",
        "🍂 Seasonal Patterns",
        "📅 Decadal Comparison",
        "🔬 Data Quality & Methodology",
    ])

    # =======================================================================
    # Tab 1: Overview
    # =======================================================================
    with tabs[0]:
        st.subheader("Global Climate Anomaly Overview")
        st.markdown(
            "Historical temperature departures from the 1951–1980 climatological baseline. "
            "Data is sourced directly from NASA GISS surface temperature observations."
        )

        # High-level metric KPI cards
        kpis = extract_overview_kpis(annual_filtered)
        complete_years_df = annual_filtered[annual_filtered["is_complete"] == True]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if kpis["latest_complete_year"] is not None:
                st.metric(
                    label=f"Latest Complete Year ({kpis['latest_complete_year']})",
                    value=f"{kpis['latest_complete_val']:+.2f} °C",
                    help="Mean anomaly across 12 complete monthly observations.",
                )
            else:
                st.metric(label="Latest Complete Year", value="N/A")

        with col2:
            if kpis["latest_incomplete_year"] is not None:
                st.metric(
                    label=f"Ongoing Year ({kpis['latest_incomplete_year']}, Diagnostic)",
                    value=f"{kpis['latest_incomplete_val']:+.2f} °C",
                    delta=f"{kpis['latest_incomplete_months']}/12 months observed",
                    help="Diagnostic arithmetic mean of observed months. NOT a complete annual mean.",
                )
            else:
                st.metric(label="Ongoing Incomplete Year", value="None in range")

        with col3:
            trend_comp = compute_trend_comparison(complete_years_df, min_years=5)
            if trend_comp is not None:
                ols_overview = trend_comp["ols"]
                st.metric(
                    label="OLS Trend (Complete Years)",
                    value=f"{ols_overview.slope_per_decade:+.3f} °C/dec",
                    delta=f"R² = {ols_overview.r_squared:.2f}",
                    help="Rate of warming per decade evaluated strictly on complete years.",
                )
            else:
                st.metric(label="OLS Trend", value="< 5 complete years")

        with col4:
            if kpis["warmest_complete_year"] is not None:
                st.metric(
                    label=f"Warmest Complete Year ({kpis['warmest_complete_year']})",
                    value=f"{kpis['warmest_complete_val']:+.2f} °C",
                    help="Highest annual mean anomaly recorded among complete calendar years.",
                )
            else:
                st.metric(label="Warmest Complete Year", value="N/A")

        st.markdown("---")

        # Main Monthly Anomaly Plot
        st.subheader("Monthly Global Land-Ocean Temperature Anomalies")
        rolling_choice = st.selectbox(
            "Overlay Rolling Average Smoothing",
            options=["None", "12-Month Trailing (1 Year)", "60-Month Trailing (5 Years)", "120-Month Trailing (10 Years)"],
            index=2,
            help="Trailing rolling averages evaluate only past observations without future data leakage.",
        )

        rolling_col_map = {
            "None": None,
            "12-Month Trailing (1 Year)": "anomaly_rolling_12m",
            "60-Month Trailing (5 Years)": "anomaly_rolling_60m",
            "120-Month Trailing (10 Years)": "anomaly_rolling_120m",
        }
        selected_rolling = rolling_col_map[rolling_choice]

        fig_monthly = plot_monthly_anomalies(
            feat_filtered,
            rolling_col=selected_rolling,
            start_year=start_year,
            end_year=end_year,
            figsize=(12, 5.2),
        )
        st.pyplot(fig_monthly, use_container_width=True)
        plt.close(fig_monthly)

        st.caption(
            "Note: Missing observations are preserved as line breaks without zero-imputation or synthetic interpolation. "
            "Horizontal dashed line represents the 1951–1980 baseline reference (0.0 °C)."
        )

    # =======================================================================
    # Tab 2: Historical Trends
    # =======================================================================
    with tabs[1]:
        st.subheader("Annual Mean Anomalies & Statistical Trend Estimation")
        st.markdown(
            "Annual global temperature departures with strict separation of complete from incomplete years. "
            "Statistical trend estimation is performed strictly on complete years."
        )

        show_trend = st.checkbox(
            "Overlay OLS Linear Trend Line",
            value=True,
            help="Fits Ordinary Least Squares linear regression strictly on complete years in the selected window.",
        )

        fig_annual = plot_annual_anomalies(
            annual_filtered,
            show_ols_trend=show_trend,
            start_year=start_year,
            end_year=end_year,
            figsize=(12, 5.2),
        )
        st.pyplot(fig_annual, use_container_width=True)
        plt.close(fig_annual)

        st.markdown("### 📐 Statistical Trend Comparison (Complete Years Only)")

        trend_models = compute_trend_comparison(complete_years_df, min_years=5)
        if trend_models is not None:
            ols_res = trend_models["ols"]
            mk_res = trend_models["mann_kendall"]
            sens_res = trend_models["sens"]

            trend_col1, trend_col2, trend_col3 = st.columns(3)

            with trend_col1:
                st.markdown("#### Ordinary Least Squares (OLS)")
                st.write(f"**Slope (Decadal):** `{ols_res.slope_per_decade:+.4f} °C/decade`")
                st.write(f"**Slope (Annual):** `{ols_res.slope_per_year:+.6f} °C/year`")
                st.write(f"**Conventional 95% CI:** `[{ols_res.ci_lower_per_decade:+.4f}, {ols_res.ci_upper_per_decade:+.4f}]`")
                st.write(f"**R² Statistic:** `{ols_res.r_squared:.4f}`")
                st.write(f"**Sample Count:** `{ols_res.sample_count} complete years`")
                st.write(f"**Conventional p-value:** `{ols_res.p_value:.4e}`")
                if ols_res.residual_autocorr_lag1 is not None and ols_res.durbin_watson is not None:
                    st.write(f"**Residual Autocorr (r₁):** `{ols_res.residual_autocorr_lag1:+.4f}`")
                    st.write(f"**Durbin-Watson d:** `{ols_res.durbin_watson:.4f}`")

            with trend_col2:
                st.markdown("#### Mann-Kendall Monotonic Test")
                st.write(f"**Direction:** `{mk_res.trend_direction.upper()}`")
                st.write(f"**Kendall S:** `{mk_res.s_statistic:+.0f}`")
                st.write(f"**Standardized Z:** `{mk_res.z_statistic:+.4f}`")
                st.write(f"**p-value:** `{mk_res.p_value:.4e}`")
                if mk_res.log10_p_value is not None:
                    st.write(f"**log₁₀(p-value):** `{mk_res.log10_p_value:.2f}`")
                st.write(f"**Tie-Correction:** Exact Kendall (1975)")

            with trend_col3:
                st.markdown("#### Sen's Robust Slope Estimator")
                st.write(f"**Slope (Decadal):** `{sens_res.slope_per_decade:+.4f} °C/decade`")
                st.write(f"**Slope (Annual):** `{sens_res.slope_per_year:+.6f} °C/year`")
                st.write(f"**Rank-Based 95% CI:** `[{sens_res.ci_lower_per_decade:+.4f}, {sens_res.ci_upper_per_decade:+.4f}]`")
                st.write(f"**Pairwise Slopes:** `{sens_res.pairwise_slopes_count:,}`")
                st.write(f"**Robustness:** Resistant to non-normal errors & outliers")

            # Scientific Caveats Box
            st.warning(
                "⚠️ **Statistical Caveats & Serial Autocorrelation Notice:**\n\n"
                "1. **Serial Dependence:** Annual climate anomalies exhibit substantial persistence and memory "
                f"(Residual Lag-1 Autocorrelation $r_1 = {ols_res.residual_autocorr_lag1:+.4f}$, Durbin-Watson $d = {ols_res.durbin_watson:.4f}$). "
                "Because conventional OLS standard errors and standard Mann-Kendall p-values assume independent errors, "
                "positive autocorrelation deflates standard errors and yields confidence intervals that are narrower than true physical uncertainty.\n"
                "2. **No Causal Attribution:** A statistically significant slope describes historical observations; "
                "it does NOT demonstrate human vs. natural causation or substitute for physics-based General Circulation Models (GCMs)."
            )
        else:
            st.info("At least 5 complete annual observations are required to compute statistical trends.")

    # =======================================================================
    # Tab 3: Seasonal Patterns
    # =======================================================================
    with tabs[2]:
        st.subheader("Seasonal Climatology: Monthly Departure Distribution")
        st.markdown(
            "Distribution of global land-ocean temperature departures by calendar month (Jan through Dec). "
            "Global anomalies in NASA GISTEMP represent departures relative to the 1951–1980 monthly cycle."
        )

        col_s1, col_s2 = st.columns([1, 2])
        with col_s1:
            period_preset = st.radio(
                "Seasonal Climatology Evaluation Window (1951–1980 Base Anomalies)",
                options=[
                    "1951–1980 Baseline (Official NASA Base)",
                    "Full Available Record (1880–Present)",
                    "Recent 30 Years (1995–2024)",
                    "Custom Reference Window",
                ],
                index=0,
                help="Selects the evaluation time window across which 1951–1980 anomalies are summarized. Does NOT rebaseline.",
            )

        custom_range_selected = None
        if period_preset == "Custom Reference Window":
            with col_s2:
                custom_range_selected = st.slider(
                    "Select Custom Reference Years",
                    min_value=min_year_avail,
                    max_value=max_year_avail,
                    value=(1980, 2010),
                )

        ref_start_input, ref_end_input = resolve_seasonal_reference_period(
            period_preset, custom_range=custom_range_selected
        )

        fig_seasonal = plot_seasonal_patterns(
            feat_df,
            reference_start=ref_start_input,
            reference_end=ref_end_input,
            figsize=(10, 5.0),
        )
        st.pyplot(fig_seasonal, use_container_width=True)
        plt.close(fig_seasonal)

        # Monthly Climatology Table
        climatology_summary = calculate_monthly_climatology(
            feat_df,
            reference_start=ref_start_input,
            reference_end=ref_end_input,
        )

        display_clim = climatology_summary[[
            "month_name", "count", "mean_anomaly", "std", "median", "q25", "q75", "min", "max"
        ]].copy()
        display_clim.columns = [
            "Month", "Observations", "Mean Departure (°C)", "Std Dev (°C)",
            "Median (°C)", "25th Pct (°C)", "75th Pct (°C)", "Min (°C)", "Max (°C)"
        ]

        st.dataframe(
            display_clim.style.format({
                "Mean Departure (°C)": "{:+.3f}",
                "Std Dev (°C)": "{:.3f}",
                "Median (°C)": "{:+.3f}",
                "25th Pct (°C)": "{:+.3f}",
                "75th Pct (°C)": "{:+.3f}",
                "Min (°C)": "{:+.3f}",
                "Max (°C)": "{:+.3f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.info(
            "ℹ️ **Scientific Note on Reference Baseline Semantics:**\n\n"
            "All input anomalies in NASA GISTEMP represent departures relative to the official **1951–1980** baseline. "
            "Selecting an evaluation window above computes the monthly central tendencies across the chosen years; "
            "it does **not** re-center or rebaseline the anomalies. "
            "To re-center anomalies against a different baseline period by subtracting calendar-month means, use "
            "`calculate_baseline_anomalies()` in `climate_analyzer.feature_engineering`."
        )

        st.caption(
            "Note: Missing observations are excluded strictly using .dropna(). "
            "Zero-imputation is prohibited because 0.0 °C represents the exact 1951–1980 baseline."
        )

    # =======================================================================
    # Tab 4: Decadal Comparison
    # =======================================================================
    with tabs[3]:
        st.subheader("Decadal Mean Temperature Anomalies")
        st.markdown(
            "Calendar decades grouped according to standard astronomical/calendar boundaries "
            "(1880–1889, 1890–1899, ..., 2010–2019, 2020–2029). "
            "Decades requiring 10 valid complete years are distinguished from partial diagnostic decades."
        )

        fig_decadal = plot_decadal_summary(
            annual_filtered,
            figsize=(11, 5.2),
        )
        st.pyplot(fig_decadal, use_container_width=True)
        plt.close(fig_decadal)

        st.markdown("### 📋 Decadal Summary Statistics")
        decadal_table = calculate_decadal_summary(annual_filtered)
        display_dec = format_decadal_table_for_display(decadal_table)

        st.dataframe(
            display_dec.style.format({
                "Complete Mean (°C)": lambda x: f"{x:+.3f}" if pd.notna(x) else "—",
                "Partial Mean (°C)": lambda x: f"{x:+.3f}" if pd.notna(x) else "—",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.warning(
            "⚠️ **Partial Decade Notice (2020–2029):**\n"
            "The current decade (2020–2029) contains only 6 complete calendar years (2020–2025). "
            "Its arithmetic mean (+1.06 °C) is a diagnostic figure and must NOT be cited as a completed "
            "ten-year decadal climatological mean."
        )

    # =======================================================================
    # Tab 5: Data Quality & Methodology
    # =======================================================================
    with tabs[4]:
        st.subheader("Dataset Provenance, Quality Assurance & Methodology")

        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.markdown("#### 1. Official Dataset Specifications")
            st.write("**Dataset:** NASA GISTEMP Surface Temperature Analysis v4 (L-OTI)")
            st.write("**Provider:** NASA Goddard Institute for Space Studies (GISS)")
            st.write("**Reference Baseline:** 1951–1980 Climatological Base Period")
            st.write("**Physical Units:** Degrees Celsius (°C) temperature departure")
            st.write(f"**Total Ingested Records:** {len(monthly_raw):,} monthly records ({min_year_avail}–{max_year_avail})")
            st.write(f"**Complete Calendar Years:** {int(annual_df['is_complete'].sum())} years")
            st.write(f"**Incomplete Years:** {int((~annual_df['is_complete']).sum())} year (2026: 9/12 months)")

            st.markdown("#### 2. Automated Quality Validations")
            st.write("✅ **Physical Boundary Checks:** Verified all anomalies fall within [-10.0 °C, +10.0 °C].")
            st.write("✅ **Month Bounds:** Verified month index strictly within [1, 12].")
            st.write("✅ **Calendar Monotonicity:** Verified strictly increasing dates without duplicate timestamps.")
            st.write("✅ **Missing Value Encoding:** Parsed NASA markers (`***`, `****`) to IEEE 754 NaN.")

        with col_m2:
            st.markdown("#### 3. Core Scientific Safeguards")
            st.markdown(
                "- **No Zero-Imputation:** Missing monthly anomalies are preserved as NaN. Imputing 0.0 °C "
                "would falsely inject 1951–1980 baseline values into unobserved periods.\n"
                "- **Incomplete Year Segregation:** Incomplete calendar years are excluded from annual summaries "
                "and statistical regressions by default to avoid partial-year seasonal bias.\n"
                "- **Centered Regression Coordinates:** OLS regression uses centered time coordinates "
                "($t^* = t - \\bar{t}$) to eliminate parameter collinearity and preserve numerical precision.\n"
                "- **Underflow-Resistant Inference:** Mann-Kendall test evaluates tail probabilities using normal "
                "survival functions, preventing premature floating-point underflow ($p \\approx 10^{-39}$).\n"
                "- **Serial Autocorrelation Disclaimers:** All linear regressions compute lag-1 residual autocorrelation "
                "and Durbin-Watson test statistics to document the impact of serial persistence."
            )

            st.markdown("#### 4. Scope Limitations")
            st.markdown(
                "- **Observational Scope:** Statistical trends describe past historical observations; they do not "
                "substitute for physics-based General Circulation Models (GCMs).\n"
                "- **No Forecasting:** The dashboard does not extrapolate future warming pathways.\n"
                "- **No Attribution:** Trend detection does not isolate anthropogenic forcing from natural climate variability."
            )


if __name__ == "__main__":
    main()
