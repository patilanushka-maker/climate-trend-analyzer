"""
Reusable, scientifically responsible climate visualization routines.

This module provides plotting functions built on Matplotlib for climate time series:
1. `plot_monthly_anomalies`: High-resolution monthly anomaly series with gap preservation.
2. `plot_annual_anomalies`: Annual mean anomalies distinguishing complete vs. incomplete years,
   with optional OLS linear trend overlay and explicit autocorrelation disclaimers.
3. `plot_decadal_summary`: Decadal averages distinguishing complete 10-year decades from
   partial diagnostic decades.
4. `plot_seasonal_patterns`: Monthly climatological anomaly distribution (Jan–Dec)
   across a specified reference period without zero-imputation.

SCIENTIFIC INTEGRITY PRINCIPLES:
- Figures return open Matplotlib `Figure` objects; closing is left to the caller.
- Missing observations are NEVER imputed with zero or interpolated.
- Incomplete years (such as the ongoing observation year) are visually and semantically
  isolated from complete annual means and are never fed into trend estimations.
- Confidence intervals on OLS trend lines explicitly document that conventional standard
  errors assume independent residuals and are not autocorrelation-adjusted.
- Caller DataFrames are never mutated (defensive copies are enforced).
- Output files are written only when an explicit `output_path` is provided.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from climate_analyzer.analysis.descriptive_analysis import (
    CALENDAR_MONTH_NAMES,
    calculate_decadal_summary,
    calculate_monthly_climatology,
)
from climate_analyzer.analysis.trend_analysis import estimate_ols_trend
from climate_analyzer.data_validation import SchemaValidationError


# ---------------------------------------------------------------------------
# Plotting Style & Palette Constants
# ---------------------------------------------------------------------------

COLOR_WARM = "#d73027"       # Positive temperature anomaly (crimson)
COLOR_COOL = "#4575b4"       # Negative temperature anomaly (ocean blue)
COLOR_LINE_MAIN = "#2b5c8f"  # Primary time series line
COLOR_ROLLING = "#e66101"    # Trailing/centered rolling average line
COLOR_TREND = "#1a1a1a"      # OLS trend line (dark charcoal)
COLOR_PARTIAL = "#fee08b"    # Incomplete/diagnostic year fill
COLOR_PARTIAL_EDGE = "#d73027"
COLOR_GRID = "#e0e0e0"       # Subtle gridlines


def _save_figure(fig: plt.Figure, output_path: Union[str, Path], dpi: int = 300) -> Path:
    """Save a Matplotlib figure to disk without closing it.

    Parameters
    ----------
    fig : plt.Figure
        Active figure to save.
    output_path : str or Path
        Target file path. Parent directories are created if missing.
    dpi : int, default 300
        Resolution in dots per inch.

    Returns
    -------
    Path
        Resolved Path to the saved file.
    """
    out_path = Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return out_path


# ---------------------------------------------------------------------------
# 1. Monthly Anomalies Plot
# ---------------------------------------------------------------------------


def plot_monthly_anomalies(
    df: pd.DataFrame,
    value_col: str = "anomaly",
    date_col: str = "date",
    rolling_col: Optional[str] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    figsize: Tuple[float, float] = (12, 5.5),
    title: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot monthly global temperature anomalies against calendar dates.

    Preserves NaN gaps so missing observations remain visible breaks rather
    than being silently interpolated or filled with zero.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing monthly observations with date_col and value_col.
    value_col : str, default 'anomaly'
        Column name containing monthly anomalies in °C.
    date_col : str, default 'date'
        Column name containing datetime timestamps or convertible dates.
    rolling_col : str, optional
        Optional column name containing pre-computed rolling means (e.g. 'anomaly_rolling_60m').
    start_year : int, optional
        Filter series starting from this year (inclusive).
    end_year : int, optional
        Filter series ending at this year (inclusive).
    figsize : Tuple[float, float], default (12, 5.5)
        Figure dimensions in inches.
    title : str, optional
        Custom title. If None, an informative climatological title is generated.
    output_path : str or Path, optional
        If provided, the figure is saved to this location.

    Returns
    -------
    plt.Figure
        Open Matplotlib Figure object.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame.
    SchemaValidationError
        If required columns are missing.
    ValueError
        If date filtering produces an empty DataFrame or series has no valid observations.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}.")

    if df.empty:
        raise ValueError("Cannot plot monthly anomalies from an empty DataFrame.")

    # Defensive copy
    work_df = df.copy()

    # Column verification
    if date_col not in work_df.columns:
        if "year" in work_df.columns and "month" in work_df.columns:
            work_df[date_col] = pd.to_datetime(
                work_df["year"].astype(str) + "-" + work_df["month"].astype(str).str.zfill(2) + "-01"
            )
        else:
            raise SchemaValidationError(
                f"Required date column '{date_col}' missing from DataFrame. Available: {list(work_df.columns)}"
            )

    if value_col not in work_df.columns:
        raise SchemaValidationError(
            f"Required anomaly column '{value_col}' missing from DataFrame. Available: {list(work_df.columns)}"
        )

    work_df[date_col] = pd.to_datetime(work_df[date_col])
    work_df = work_df.sort_values(date_col).reset_index(drop=True)

    # Optional year filtering
    if start_year is not None and end_year is not None and start_year > end_year:
        raise ValueError(f"start_year ({start_year}) cannot exceed end_year ({end_year}).")

    if "year" not in work_df.columns:
        work_df["year"] = work_df[date_col].dt.year

    if start_year is not None:
        work_df = work_df[work_df["year"] >= start_year]
    if end_year is not None:
        work_df = work_df[work_df["year"] <= end_year]

    if work_df.empty:
        raise ValueError(
            f"No observations remain after filtering date range [{start_year}, {end_year}]."
        )

    valid_anomalies = work_df[value_col].dropna()
    if valid_anomalies.empty:
        raise ValueError(f"Series in column '{value_col}' contains no valid observations.")

    # Determine observation date boundaries based on actual valid observations
    valid_dates = work_df.loc[work_df[value_col].notna(), date_col]
    min_date = valid_dates.min().strftime("%b %Y")
    max_date = valid_dates.max().strftime("%b %Y")

    fig, ax = plt.subplots(figsize=figsize)


    # Plot monthly anomaly series (NaNs are preserved as line breaks)
    ax.plot(
        work_df[date_col],
        work_df[value_col],
        color=COLOR_LINE_MAIN,
        linewidth=0.8,
        alpha=0.75,
        label="Monthly Anomaly",
    )

    # Optional rolling mean overlay
    if rolling_col is not None:
        if rolling_col not in work_df.columns:
            raise SchemaValidationError(
                f"Specified rolling column '{rolling_col}' not found in DataFrame."
            )
        ax.plot(
            work_df[date_col],
            work_df[rolling_col],
            color=COLOR_ROLLING,
            linewidth=2.0,
            label=f"Rolling Average ({rolling_col.replace('anomaly_', '').replace('_', ' ')})",
        )

    # Baseline zero line
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.9, alpha=0.8, label="1951–1980 Baseline (0.0 °C)")

    # Styling
    ax.set_ylabel("Temperature Anomaly (°C)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Observation Date", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6, color=COLOR_GRID)

    plot_title = title or "Global Land-Ocean Surface Temperature Anomalies (Monthly)"
    ax.set_title(
        f"{plot_title}\n"
        f"Reference Baseline: 1951–1980 | Observation Period: {min_date} – {max_date}",
        fontsize=12,
        pad=10,
    )
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)

    fig.tight_layout()

    if output_path is not None:
        _save_figure(fig, output_path)

    return fig


# ---------------------------------------------------------------------------
# 2. Annual Anomalies Plot
# ---------------------------------------------------------------------------


def plot_annual_anomalies(
    annual_df: pd.DataFrame,
    value_col: str = "annual_mean_anomaly",
    partial_col: str = "partial_year_mean",
    year_col: str = "year",
    show_ols_trend: bool = False,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    figsize: Tuple[float, float] = (12, 5.5),
    title: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot annual mean temperature anomalies, visually distinguishing complete from incomplete years.

    SCIENTIFIC INTEGRITY CONSTRAINTS:
    - Complete years (12 valid monthly observations) are clearly distinguished from
      incomplete years (such as the ongoing observation year).
    - Incomplete years are NEVER presented as complete annual means.
    - If an OLS linear trend line is overlaid, it is estimated strictly on complete years.
    - The slope is reported in °C per decade with an explicit disclaimer that conventional
      confidence intervals assume independent errors and are not autocorrelation-adjusted.

    Parameters
    ----------
    annual_df : pd.DataFrame
        DataFrame from calculate_annual_summary() containing year_col and value_col.
    value_col : str, default 'annual_mean_anomaly'
        Column containing complete annual mean anomalies.
    partial_col : str, default 'partial_year_mean'
        Column containing diagnostic means for incomplete years.
    year_col : str, default 'year'
        Column containing integer calendar years.
    show_ols_trend : bool, default False
        If True, estimates and overlays an OLS linear trend line on complete years.
    start_year : int, optional
        Filter starting year (inclusive).
    end_year : int, optional
        Filter ending year (inclusive).
    figsize : Tuple[float, float], default (12, 5.5)
        Figure dimensions in inches.
    title : str, optional
        Custom title.
    output_path : str or Path, optional
        Target file path to save figure.

    Returns
    -------
    plt.Figure
        Open Matplotlib Figure object.

    Raises
    ------
    TypeError
        If annual_df is not a pandas DataFrame.
    SchemaValidationError
        If required columns are missing.
    ValueError
        If no valid observations exist in the specified range.
    """
    if not isinstance(annual_df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(annual_df).__name__}.")

    if annual_df.empty:
        raise ValueError("Cannot plot annual anomalies from an empty DataFrame.")

    work_df = annual_df.copy()

    if year_col not in work_df.columns:
        raise SchemaValidationError(f"Required column '{year_col}' missing from annual DataFrame.")

    actual_val_col = value_col
    if actual_val_col not in work_df.columns:
        if "annual_mean" in work_df.columns:
            actual_val_col = "annual_mean"
        else:
            raise SchemaValidationError(
                f"Required column '{value_col}' missing from annual DataFrame. Available: {list(work_df.columns)}"
            )

    # Optional year filtering
    if start_year is not None and end_year is not None and start_year > end_year:
        raise ValueError(f"start_year ({start_year}) cannot exceed end_year ({end_year}).")

    if start_year is not None:
        work_df = work_df[work_df[year_col] >= start_year]
    if end_year is not None:
        work_df = work_df[work_df[year_col] <= end_year]

    if work_df.empty:
        raise ValueError(f"No annual observations remain in range [{start_year}, {end_year}].")

    work_df = work_df.sort_values(year_col).reset_index(drop=True)

    # Identify complete vs. incomplete years
    if "is_complete" in work_df.columns:
        complete_mask = work_df["is_complete"].astype(bool) & work_df[actual_val_col].notna()
    else:
        complete_mask = work_df[actual_val_col].notna()

    has_partial_col = partial_col in work_df.columns
    if has_partial_col:
        incomplete_mask = (~complete_mask) & work_df[partial_col].notna()
    else:
        incomplete_mask = pd.Series(False, index=work_df.index)

    if not complete_mask.any() and not incomplete_mask.any():
        raise ValueError("No valid complete or partial annual anomaly observations to plot.")

    fig, ax = plt.subplots(figsize=figsize)

    # Plot complete annual anomaly bars (warm red for >=0, cool blue for <0)
    complete_years = work_df.loc[complete_mask, year_col].to_numpy()
    complete_vals = work_df.loc[complete_mask, actual_val_col].to_numpy()

    if len(complete_years) > 0:
        bar_colors = [COLOR_WARM if v >= 0 else COLOR_COOL for v in complete_vals]
        ax.bar(
            complete_years,
            complete_vals,
            color=bar_colors,
            width=0.8,
            alpha=0.85,
            edgecolor="none",
            label="Complete Annual Mean (°C)",
        )

    # Plot incomplete diagnostic years with distinct hatched bars and outline
    incomplete_years = work_df.loc[incomplete_mask, year_col].to_numpy()
    if len(incomplete_years) > 0 and has_partial_col:
        incomplete_vals = work_df.loc[incomplete_mask, partial_col].to_numpy()
        ax.bar(
            incomplete_years,
            incomplete_vals,
            color=COLOR_PARTIAL,
            edgecolor=COLOR_PARTIAL_EDGE,
            hatch="//",
            linewidth=1.2,
            width=0.8,
            alpha=0.9,
            label="Incomplete Year (Diagnostic Partial Mean)",
        )

    # Baseline zero line
    ax.axhline(0.0, color="black", linestyle="-", linewidth=0.8, alpha=0.7)

    # Optional OLS Linear Trend Overlay (computed exclusively on complete years)
    trend_disclaimer = ""
    if show_ols_trend:
        if len(complete_years) < 5:
            raise ValueError(
                f"OLS trend estimation requires at least 5 complete annual observations, found {len(complete_years)}."
            )

        ols_res = estimate_ols_trend(
            work_df,
            value_col=actual_val_col,
            exclude_incomplete=True,
        )

        trend_line = ols_res.intercept + ols_res.slope_per_year * (complete_years - ols_res.center_year)
        ax.plot(
            complete_years,
            trend_line,
            color=COLOR_TREND,
            linestyle="--",
            linewidth=2.2,
            label=(
                f"OLS Trend: {ols_res.slope_per_decade:+.3f} °C/decade "
                f"(95% CI: [{ols_res.ci_lower_per_decade:+.3f}, {ols_res.ci_upper_per_decade:+.3f}])"
            ),
        )
        trend_disclaimer = (
            f"\nOLS Linear Trend ({ols_res.start_year}–{ols_res.end_year}): "
            f"{ols_res.slope_per_decade:+.3f} °C/decade. "
            "Note: Conventional 95% CI assumes independent residuals and is not autocorrelation-adjusted."
        )

    min_yr = work_df[year_col].min()
    max_yr = work_df[year_col].max()
    plot_title = title or "Global Annual Mean Surface Temperature Anomalies"

    ax.set_title(
        f"{plot_title}\n"
        f"Reference Baseline: 1951–1980 | Observation Period: {min_yr}–{max_yr}"
        f"{trend_disclaimer}",
        fontsize=11,
        pad=10,
    )
    ax.set_ylabel("Temperature Anomaly (°C)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Calendar Year", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6, color=COLOR_GRID)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)

    fig.tight_layout()

    if output_path is not None:
        _save_figure(fig, output_path)

    return fig


# ---------------------------------------------------------------------------
# 3. Decadal Summary Plot
# ---------------------------------------------------------------------------


def plot_decadal_summary(
    decadal_or_annual_df: pd.DataFrame,
    value_col: str = "annual_mean_anomaly",
    figsize: Tuple[float, float] = (11, 5.5),
    title: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot decadal temperature anomalies, strictly distinguishing complete from partial decades.

    SCIENTIFIC INTEGRITY CONSTRAINTS:
    - Complete decades (10 complete observation years) are shown with solid bars.
    - Partial decades (such as 2020–2029 with 6 years observed) are shown with hatched,
      distinctive styling and labeled as diagnostic partial means.
    - Partial decades are NEVER presented as equivalent to complete decades.

    Parameters
    ----------
    decadal_or_annual_df : pd.DataFrame
        DataFrame from calculate_decadal_summary() or an annual summary DataFrame
        from calculate_annual_summary().
    value_col : str, default 'annual_mean_anomaly'
        Column name to aggregate if an annual summary is provided.
    figsize : Tuple[float, float], default (11, 5.5)
        Figure dimensions in inches.
    title : str, optional
        Custom title.
    output_path : str or Path, optional
        Target file path to save figure.

    Returns
    -------
    plt.Figure
        Open Matplotlib Figure object.

    Raises
    ------
    TypeError
        If input is not a pandas DataFrame.
    SchemaValidationError
        If required columns are missing.
    ValueError
        If DataFrame is empty or contains no decadal records.
    """
    if not isinstance(decadal_or_annual_df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(decadal_or_annual_df).__name__}.")

    if decadal_or_annual_df.empty:
        raise ValueError("Cannot plot decadal summary from an empty DataFrame.")

    work_df = decadal_or_annual_df.copy()

    # Determine if input is already decadal summary or annual summary
    if "decade" in work_df.columns and ("complete_decade_mean" in work_df.columns or "mean_anomaly" in work_df.columns):
        decadal_df = work_df.copy()
    elif "year" in work_df.columns:
        decadal_df = calculate_decadal_summary(work_df, value_col=value_col)
    else:
        raise SchemaValidationError(
            "DataFrame must contain either 'decade' (decadal summary) or 'year' (annual summary)."
        )

    if decadal_df.empty:
        raise ValueError("Decadal summary contains no records to plot.")

    # Standardize column references
    mean_col = "complete_decade_mean" if "complete_decade_mean" in decadal_df.columns else "mean_anomaly"
    partial_col = "partial_decade_mean" if "partial_decade_mean" in decadal_df.columns else mean_col
    complete_flag = "is_complete" if "is_complete" in decadal_df.columns else "is_complete_decade"

    fig, ax = plt.subplots(figsize=figsize)

    x_indices = np.arange(len(decadal_df))
    labels = decadal_df["decade_label"].tolist() if "decade_label" in decadal_df.columns else decadal_df["decade"].astype(str).tolist()

    has_complete = False
    has_partial = False

    for idx, (_, row) in enumerate(decadal_df.iterrows()):
        is_comp = bool(row.get(complete_flag, pd.notna(row[mean_col])))
        val = row[mean_col] if is_comp else row[partial_col]

        if pd.isna(val):
            continue

        if is_comp:
            label = "Complete Decade Mean (10 years)" if not has_complete else "_nolegend_"
            has_complete = True
            color = COLOR_WARM if val >= 0 else COLOR_COOL
            ax.bar(
                idx,
                val,
                color=color,
                edgecolor="none",
                width=0.7,
                alpha=0.85,
                label=label,
            )
            # Label bar value
            va = "bottom" if val >= 0 else "top"
            offset = 0.02 if val >= 0 else -0.04
            ax.text(idx, val + offset, f"{val:+.2f}", ha="center", va=va, fontsize=8.5, fontweight="bold")
        else:
            valid_yrs = int(row.get("valid_years", row.get("valid_years_count", "?")))
            label = f"Partial Decade Mean (Diagnostic, {valid_yrs}/10 yrs)" if not has_partial else "_nolegend_"
            has_partial = True
            ax.bar(
                idx,
                val,
                color=COLOR_PARTIAL,
                edgecolor=COLOR_PARTIAL_EDGE,
                hatch="//",
                linewidth=1.2,
                width=0.7,
                alpha=0.9,
                label=label,
            )

            va = "bottom" if val >= 0 else "top"
            offset = 0.02 if val >= 0 else -0.04
            ax.text(idx, val + offset, f"{val:+.2f}*\n({valid_yrs}y)", ha="center", va=va, fontsize=8, color="#a50f15", fontweight="bold")

    ax.set_xticks(x_indices)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9.5)
    ax.axhline(0.0, color="black", linestyle="-", linewidth=0.8, alpha=0.7)

    # Provide headroom for text annotations above and below bars
    current_ymin, current_ymax = ax.get_ylim()
    y_margin_bottom = (current_ymax - current_ymin) * 0.12
    y_margin_top = (current_ymax - current_ymin) * 0.14
    ax.set_ylim(current_ymin - y_margin_bottom, current_ymax + y_margin_top)

    ax.set_ylabel("Decadal Mean Anomaly (°C)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Calendar Decade", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6, color=COLOR_GRID)


    plot_title = title or "Global Decadal Mean Temperature Anomalies"
    ax.set_title(
        f"{plot_title}\n"
        "Reference Baseline: 1951–1980 | Solid: Complete Decades (10y) | Hatched (*): Partial Diagnostic Decades",
        fontsize=11,
        pad=10,
    )

    # Deduplicate legend items while preserving explicit labels
    handles, legend_labels = ax.get_legend_handles_labels()
    by_label = dict(zip(legend_labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc="upper left", frameon=True, framealpha=0.9)

    fig.tight_layout()

    if output_path is not None:
        _save_figure(fig, output_path)

    return fig


# ---------------------------------------------------------------------------
# 4. Seasonal Patterns Plot
# ---------------------------------------------------------------------------


def plot_seasonal_patterns(
    monthly_or_climatology_df: pd.DataFrame,
    value_col: str = "anomaly",
    reference_start: Optional[int] = None,
    reference_end: Optional[int] = None,
    figsize: Tuple[float, float] = (10, 5.5),
    title: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot monthly climatological anomaly distributions (Jan through Dec).

    SCIENTIFIC INTEGRITY CONSTRAINTS:
    - Evaluates calendar month departure statistics (mean, std, IQR, extremes)
      across the specified reference period.
    - Missing observations are NEVER imputed with zero or interpolated.
    - Global anomalies from NASA GISTEMP represent departures from the 1951–1980
      seasonal cycle.

    Parameters
    ----------
    monthly_or_climatology_df : pd.DataFrame
        Monthly observation DataFrame (with 'month' and value_col) or pre-computed
        climatology DataFrame from calculate_monthly_climatology().
    value_col : str, default 'anomaly'
        Column name containing monthly anomalies in °C.
    reference_start : int, optional
        First year of analysis window (inclusive).
    reference_end : int, optional
        Last year of analysis window (inclusive).
    figsize : Tuple[float, float], default (10, 5.5)
        Figure dimensions in inches.
    title : str, optional
        Custom title.
    output_path : str or Path, optional
        Target file path to save figure.

    Returns
    -------
    plt.Figure
        Open Matplotlib Figure object.

    Raises
    ------
    TypeError
        If input is not a pandas DataFrame.
    SchemaValidationError
        If required columns are missing.
    ValueError
        If no valid observations exist in the reference window.
    """
    if not isinstance(monthly_or_climatology_df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(monthly_or_climatology_df).__name__}.")

    if monthly_or_climatology_df.empty:
        raise ValueError("Cannot plot seasonal patterns from an empty DataFrame.")

    work_df = monthly_or_climatology_df.copy()

    # Determine if input is already climatology or raw monthly
    if "mean_anomaly" in work_df.columns and "month" in work_df.columns:
        climatology_df = work_df.copy()
    elif "month" in work_df.columns:
        climatology_df = calculate_monthly_climatology(
            work_df,
            value_col=value_col,
            reference_start=reference_start,
            reference_end=reference_end,
        )
    else:
        raise SchemaValidationError("DataFrame must contain 'month' column.")

    fig, ax = plt.subplots(figsize=figsize)

    months = climatology_df["month"].to_numpy()
    month_labels = climatology_df["month_name"].tolist() if "month_name" in climatology_df.columns else [CALENDAR_MONTH_NAMES[m - 1] for m in months]
    means = climatology_df["mean_anomaly"].to_numpy()
    stds = climatology_df["std"].to_numpy()
    medians = climatology_df["median"].to_numpy()

    # Mask valid months
    valid_mask = ~np.isnan(means)
    if not valid_mask.any():
        raise ValueError("No valid monthly climatology means available to plot.")

    valid_months = months[valid_mask]
    valid_means = means[valid_mask]
    valid_stds = stds[valid_mask]
    valid_medians = medians[valid_mask]

    # Plot mean monthly anomaly line with markers
    ax.plot(
        valid_months,
        valid_means,
        color=COLOR_LINE_MAIN,
        marker="o",
        linewidth=2.0,
        markersize=6,
        label="Monthly Mean Anomaly (°C)",
    )

    # Plot median line
    ax.plot(
        valid_months,
        valid_medians,
        color=COLOR_ROLLING,
        linestyle="--",
        linewidth=1.5,
        marker="s",
        markersize=4,
        label="Monthly Median Anomaly (°C)",
    )

    # Uncertainty / variability envelope (mean ± 1 std)
    lower_bound = valid_means - valid_stds
    upper_bound = valid_means + valid_stds
    ax.fill_between(
        valid_months,
        lower_bound,
        upper_bound,
        color=COLOR_LINE_MAIN,
        alpha=0.18,
        label="±1 Sample Std Deviation Band",
    )

    # Baseline zero line
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7, label="1951–1980 Baseline (0.0 °C)")

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(CALENDAR_MONTH_NAMES, fontsize=10)
    ax.set_ylabel("Temperature Anomaly (°C)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Calendar Month", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6, color=COLOR_GRID)

    # Construct period subtitle
    ref_start = climatology_df["reference_start"].iloc[0] if "reference_start" in climatology_df.columns else reference_start
    ref_end = climatology_df["reference_end"].iloc[0] if "reference_end" in climatology_df.columns else reference_end
    if ref_start is not None and ref_end is not None:
        period_str = f"Analyzed Period: {ref_start}–{ref_end}"
    else:
        period_str = "Full Available Record"

    plot_title = title or "Seasonal Climatology: Monthly Departure Distribution"
    ax.set_title(
        f"{plot_title}\n"
        f"Reference Baseline: 1951–1980 | {period_str} | Missing Observations Excluded (No Zero-Imputation)",
        fontsize=11,
        pad=10,
    )
    ax.legend(loc="best", frameon=True, framealpha=0.9)

    fig.tight_layout()

    if output_path is not None:
        _save_figure(fig, output_path)

    return fig
