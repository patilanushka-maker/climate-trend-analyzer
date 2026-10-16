"""
Feature engineering for historical climate time series.

This module provides scientifically defensible feature-engineering transformations
for monthly and annual climate anomaly records:
- Trailing and retrospective centered rolling means with explicit minimum-observation rules.
- Calendar-year aggregation with strict completeness accounting to prevent partial years
  from being mislabeled as complete annual averages.
- Multi-baseline climatological reference re-centering (e.g. 1880-1900 early-instrumental proxy).
- Consecutive year-over-year rate-of-change metrics in degrees Celsius.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from climate_analyzer.data_validation import (
    REQUIRED_MONTHLY_COLUMNS,
    SchemaValidationError,
    TemporalIntegrityError,
    validate_monthly_series,
)


# ---------------------------------------------------------------------------
# Default Constants & Rationales
# ---------------------------------------------------------------------------

DEFAULT_ROLLING_WINDOWS = (12, 60, 120)  # 1-year, 5-year, and 10-year monthly windows

DEFAULT_MIN_BASELINE_YEARS = 15  # Minimum valid monthly samples required across baseline


# ---------------------------------------------------------------------------
# Feature Engineering Functions
# ---------------------------------------------------------------------------


def add_rolling_features(
    df: pd.DataFrame,
    windows: Sequence[int] = DEFAULT_ROLLING_WINDOWS,
    min_periods: Optional[Union[int, Mapping[int, int]]] = None,
    center: bool = False,
    validate_input: bool = True,
) -> pd.DataFrame:
    """Calculate rolling window means over monthly climate anomalies.

    By default, trailing (backward-looking) rolling means are calculated. Trailing
    windows evaluate only observations at or prior to each timestamp and do NOT
    introduce future data leakage.

    If `center=True` is explicitly requested, retrospective centered rolling means
    are calculated. Centered rolling means use observations after the timestamp as
    well as before it; they are retrospective descriptive features and are
    unsuitable for causal prediction at the timestamp.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame conforming to monthly schema (year, month, date, anomaly).
    windows : Sequence[int], default (12, 60, 120)
        Window sizes in months.
    min_periods : int or Mapping[int, int], optional
        Minimum number of valid (non-null) observations required in each window to
        compute a value. If None, defaults to ceil(0.75 * window) for each window
        (e.g., 9 months for 12m, 45 for 60m, 90 for 120m). If an integer, applied to
        all windows. If a dict, specifies per-window thresholds.
    center : bool, default False
        If False, computes trailing rolling means.
        If True, computes retrospective centered rolling means.
    validate_input : bool, default True
        If True, runs structural validation on input before processing.

    Returns
    -------
    pd.DataFrame
        New DataFrame containing original columns plus generated rolling feature columns.
        Input DataFrame is never mutated.

    Raises
    ------
    SchemaValidationError
        If input DataFrame lacks required columns or has invalid types.
    TemporalIntegrityError
        If timestamps are not monotonically strictly increasing or contain duplicates.
    ValueError
        If any window size is non-positive.
    """
    if validate_input:
        validate_monthly_series(df, raise_on_error=True)

    result_df = df.copy()

    for w in windows:
        if not isinstance(w, int) or w <= 0:
            raise ValueError(f"Window size must be a positive integer, got: {w}")

        # Determine minimum observations required for this window
        if min_periods is None:
            # Scientifically conservative rule: require >= 75% valid monthly observations
            threshold = int(np.ceil(w * 0.75))
        elif isinstance(min_periods, int):
            if min_periods <= 0 or min_periods > w:
                raise ValueError(
                    f"min_periods ({min_periods}) must be between 1 and window size ({w})."
                )
            threshold = min_periods
        elif isinstance(min_periods, (dict, Mapping)):
            threshold = min_periods.get(w, int(np.ceil(w * 0.75)))
            if threshold <= 0 or threshold > w:
                raise ValueError(
                    f"min_periods for window {w} ({threshold}) must be between 1 and {w}."
                )
        else:
            raise TypeError(
                f"min_periods must be None, an integer, or a dict, got: {type(min_periods).__name__}"
            )

        col_name = f"anomaly_centered_{w}m" if center else f"anomaly_rolling_{w}m"

        # Trailing / Centered rolling mean on the anomaly column
        rolling_series = result_df["anomaly"].rolling(
            window=w,
            min_periods=threshold,
            center=center,
        ).mean()

        result_df[col_name] = rolling_series.astype("float64")

    return result_df


def calculate_annual_summary(
    df: pd.DataFrame,
    min_valid_months: int = 12,
    validate_input: bool = True,
) -> pd.DataFrame:
    """Calculate annual summary statistics from monthly observations with strict completeness criteria.

    IMPORTANT SCIENTIFIC INTEGRITY DISTINCTIONS:
    - `annual_mean_anomaly` is populated ONLY when valid observations in that year
      meet or exceed `min_valid_months` (default 12). If an incomplete year is
      encountered (such as the current in-progress year), `annual_mean_anomaly` is NaN
      to prevent partial-year averages from being cited as complete annual climate figures.
    - `partial_year_mean` contains the diagnostic arithmetic mean across whatever valid
      months exist in that year, preserving partial-year diagnostics for researchers.
    - This annual summary is calculated directly from the monthly series and is
      distinct from NASA's pre-computed J-D (calendar) and D-N (meteorological) fields.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame conforming to monthly schema (year, month, date, anomaly).
    min_valid_months : int, default 12
        Minimum number of valid monthly observations required to consider an annual
        mean complete (1 to 12).
    validate_input : bool, default True
        If True, runs structural validation on input before processing.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'year' (int): Calendar year.
        - 'annual_mean_anomaly' (float): Complete annual mean (°C), or NaN if incomplete.
        - 'valid_months' (int): Number of months with valid anomaly observations.
        - 'total_months' (int): Total monthly rows present for that year.
        - 'completeness_pct' (float): Percentage of year observed (valid_months / 12 * 100).
        - 'is_complete' (bool): True if valid_months >= min_valid_months.
        - 'partial_year_mean' (float): Diagnostic mean across available valid months.

    Raises
    ------
    ValueError
        If min_valid_months is outside [1, 12].
    """
    if min_valid_months < 1 or min_valid_months > 12:
        raise ValueError(
            f"min_valid_months must be between 1 and 12, got: {min_valid_months}"
        )

    if validate_input:
        validate_monthly_series(df, raise_on_error=True)

    grouped = df.groupby("year")

    records = []
    for year, group in grouped:
        total_months = len(group)
        valid_anomalies = group["anomaly"].dropna()
        valid_count = len(valid_anomalies)
        completeness = (valid_count / 12.0) * 100.0
        is_complete = valid_count >= min_valid_months

        partial_mean = float(valid_anomalies.mean()) if valid_count > 0 else np.nan
        annual_mean = partial_mean if is_complete else np.nan

        records.append(
            {
                "year": int(year),
                "annual_mean_anomaly": annual_mean,
                "valid_months": int(valid_count),
                "total_months": int(total_months),
                "completeness_pct": float(completeness),
                "is_complete": bool(is_complete),
                "partial_year_mean": partial_mean,
            }
        )

    summary_df = pd.DataFrame(records).sort_values("year").reset_index(drop=True)
    return summary_df


def calculate_baseline_anomalies(
    df: pd.DataFrame,
    baseline_start: int,
    baseline_end: int,
    min_baseline_years: int = DEFAULT_MIN_BASELINE_YEARS,
    validate_input: bool = True,
) -> pd.DataFrame:
    """Re-center climate anomalies relative to an alternative multi-year climatological baseline.

    For each calendar month (Jan through Dec), this function computes the reference
    mean anomaly over the specified baseline window [baseline_start, baseline_end] and
    subtracts that monthly reference mean from all observations for that calendar month.

    SCIENTIFIC NOTE ON BASELINES:
    The original NASA GISTEMP series is centered on the 1951-1980 base period.
    Users may re-center relative to early instrumental periods (e.g. 1880-1900), but
    such periods must be documented as early-instrumental proxies rather than definitive
    pre-industrial baselines (IPCC assessments typically reference 1850-1900).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame conforming to monthly schema (year, month, date, anomaly).
    baseline_start : int
        First year of the baseline period (inclusive).
    baseline_end : int
        Last year of the baseline period (inclusive).
    min_baseline_years : int, default 15
        Minimum number of valid observations required for EACH calendar month
        within the baseline window.
    validate_input : bool, default True
        If True, runs structural validation on input before processing.

    Returns
    -------
    pd.DataFrame
        New DataFrame containing original columns plus a new column:
        f"anomaly_base_{baseline_start}_{baseline_end}". Original 'anomaly' is untouched.

    Raises
    ------
    ValueError
        If baseline_start > baseline_end, if baseline range has no overlap with data,
        or if any calendar month has fewer than min_baseline_years valid samples.
    """
    if baseline_start > baseline_end:
        raise ValueError(
            f"baseline_start ({baseline_start}) cannot be greater than baseline_end ({baseline_end})."
        )

    if validate_input:
        validate_monthly_series(df, raise_on_error=True)

    result_df = df.copy()

    # Filter baseline subset
    baseline_subset = result_df[
        (result_df["year"] >= baseline_start) & (result_df["year"] <= baseline_end)
    ]

    if baseline_subset.empty:
        raise ValueError(
            f"No data available in baseline window [{baseline_start}, {baseline_end}]. "
            f"Dataset spans {result_df['year'].min()} to {result_df['year'].max()}."
        )

    # Calculate monthly climatological reference means for the baseline
    monthly_refs: Dict[int, float] = {}
    for m in range(1, 13):
        m_subset = baseline_subset[baseline_subset["month"] == m]["anomaly"].dropna()
        if len(m_subset) < min_baseline_years:
            raise ValueError(
                f"Insufficient baseline observations for month {m} in [{baseline_start}, {baseline_end}]: "
                f"found {len(m_subset)} valid years, required {min_baseline_years}."
            )
        monthly_refs[m] = float(m_subset.mean())

    new_col = f"anomaly_base_{baseline_start}_{baseline_end}"

    # Map month to monthly reference mean and subtract from anomaly
    ref_series = result_df["month"].map(monthly_refs)
    result_df[new_col] = (result_df["anomaly"] - ref_series).astype("float64")

    return result_df


def calculate_year_over_year_change(
    annual_df: pd.DataFrame,
    require_complete: bool = True,
) -> pd.DataFrame:
    """Calculate consecutive annual rate of change in degrees Celsius (°C).

    Computes the absolute difference in annual mean anomaly between consecutive years:
        yoy_change_deg_c = annual_mean_t - annual_mean_{t-1}

    SCIENTIFIC NOTE:
    Changes are expressed as absolute differences in degrees Celsius (°C).
    Percentage changes are avoided because temperature anomalies oscillate around 0°C,
    making percentage changes mathematically volatile and physically misleading.

    Parameters
    ----------
    annual_df : pd.DataFrame
        DataFrame from calculate_annual_summary() containing 'year', 'annual_mean_anomaly',
        and 'is_complete'.
    require_complete : bool, default True
        If True, computes YoY change only when BOTH year t and year t-1 are marked
        is_complete=True and are strictly consecutive (year_t == year_{t-1} + 1).
        If False, computes diff on whatever annual_mean_anomaly values exist.

    Returns
    -------
    pd.DataFrame
        New DataFrame with an added 'yoy_change_deg_c' column.
        Input DataFrame is never mutated.

    Raises
    ------
    SchemaValidationError
        If annual_df lacks 'year' or 'annual_mean_anomaly' columns.
    """
    required = ["year", "annual_mean_anomaly"]
    missing = [c for c in required if c not in annual_df.columns]
    if missing:
        raise SchemaValidationError(
            f"Missing required columns in annual_df: {missing}. Expected at least: {required}"
        )

    result_df = annual_df.copy().sort_values("year").reset_index(drop=True)

    changes = []
    for i in range(len(result_df)):
        if i == 0:
            changes.append(np.nan)
            continue

        curr = result_df.iloc[i]
        prev = result_df.iloc[i - 1]

        # Check consecutive calendar years
        if curr["year"] != prev["year"] + 1:
            changes.append(np.nan)
            continue

        if require_complete:
            # Check completeness flag if present
            curr_complete = curr.get("is_complete", pd.notna(curr["annual_mean_anomaly"]))
            prev_complete = prev.get("is_complete", pd.notna(prev["annual_mean_anomaly"]))
            if not (curr_complete and prev_complete):
                changes.append(np.nan)
                continue

        diff = curr["annual_mean_anomaly"] - prev["annual_mean_anomaly"]
        changes.append(diff)

    result_df["yoy_change_deg_c"] = pd.Series(changes, dtype="float64")
    return result_df
