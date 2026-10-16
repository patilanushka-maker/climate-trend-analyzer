"""
Descriptive climatological analysis and statistical distributions.

This module provides functions for descriptive summary statistics, decadal aggregations,
and warmth rankings/percentiles on annual climate anomalies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats

from climate_analyzer.data_validation import SchemaValidationError


# ---------------------------------------------------------------------------
# Result Structures
# ---------------------------------------------------------------------------


@dataclass
class DescriptiveStatisticsResult:
    """Statistical summary moments and quantile metrics for climate anomalies.

    Attributes
    ----------
    count : int
        Number of valid (non-null) observation records.
    mean : float
        Sample arithmetic mean anomaly (°C).
    std : float
        Unbiased sample standard deviation (°C, ddof=1).
    min : float
        Minimum observed anomaly (°C).
    max : float
        Maximum observed anomaly (°C).
    range : float
        Absolute range (max - min in °C).
    median : float
        50th percentile / sample median anomaly (°C).
    iqr : float
        Interquartile range (75th percentile - 25th percentile in °C).
    skewness : float
        Sample Fisher-Pearson coefficient of skewness.
    kurtosis : float
        Sample excess kurtosis (Fisher definition; normal distribution = 0.0).
    percentiles : Dict[int, float]
        Dictionary mapping percentile thresholds to anomaly values in °C.
    interpolation_method : str
        Method used for quantile interpolation (default 'linear').
    """

    count: int
    mean: float
    std: float
    min: float
    max: float
    range: float
    median: float
    iqr: float
    skewness: float
    kurtosis: float
    percentiles: Dict[int, float]
    interpolation_method: str = "linear"

    def summary(self) -> str:
        """Return a formatted text summary of descriptive statistics."""
        lines = [
            "Descriptive Climatology Statistics",
            f"  Observations:  {self.count}",
            f"  Mean Anomaly:  {self.mean:+.4f} °C",
            f"  Std Deviation: {self.std:.4f} °C",
            f"  Median (p50):  {self.median:+.4f} °C",
            f"  Min / Max:     {self.min:+.4f} °C / {self.max:+.4f} °C (Range: {self.range:.4f} °C)",
            f"  IQR (p75-p25): {self.iqr:.4f} °C",
            f"  Skewness:      {self.skewness:+.4f}",
            f"  Kurtosis:      {self.kurtosis:+.4f} (excess)",
            "  Percentiles:   " + ", ".join(f"p{k}={v:+.3f}°C" for k, v in sorted(self.percentiles.items())),
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert results to a standard dictionary."""
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "range": self.range,
            "median": self.median,
            "iqr": self.iqr,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "percentiles": self.percentiles,
            "interpolation_method": self.interpolation_method,
        }


# ---------------------------------------------------------------------------
# Core Analytical Functions
# ---------------------------------------------------------------------------


def calculate_descriptive_statistics(
    series_or_df: Union[pd.Series, pd.DataFrame],
    value_col: str = "annual_mean_anomaly",
    percentiles: Sequence[int] = (10, 25, 50, 75, 90, 95, 99),
    interpolation: str = "linear",
    exclude_incomplete: bool = True,
) -> DescriptiveStatisticsResult:
    """Compute central tendency, variability, moments, and quantiles on climate anomalies.

    Parameters
    ----------
    series_or_df : pd.Series or pd.DataFrame
        Input data. If DataFrame, extracts `value_col`. If `is_complete` column
        is present and `exclude_incomplete=True`, filters out incomplete years.
    value_col : str, default 'annual_mean_anomaly'
        Column name to extract from DataFrame.
    percentiles : Sequence[int], default (10, 25, 50, 75, 90, 95, 99)
        Percentile thresholds to calculate (integers between 0 and 100).
    interpolation : str, default 'linear'
        Quantile interpolation method passed to numpy.percentile
        ('linear', 'lower', 'higher', 'midpoint', 'nearest').
    exclude_incomplete : bool, default True
        If True and DataFrame contains 'is_complete', only rows with is_complete=True
        are evaluated. Incomplete years are strictly excluded from complete climatological stats.

    Returns
    -------
    DescriptiveStatisticsResult
        Dataclass containing sample moments, extremes, and quantiles.

    Raises
    ------
    SchemaValidationError
        If value_col is missing from DataFrame.
    ValueError
        If no valid observations remain or percentiles are invalid.
    """
    if isinstance(series_or_df, pd.DataFrame):
        actual_col = value_col
        if actual_col not in series_or_df.columns:
            if actual_col == "annual_mean_anomaly" and "annual_mean" in series_or_df.columns:
                actual_col = "annual_mean"
            elif actual_col == "annual_mean_anomaly" and "anomaly" in series_or_df.columns:
                actual_col = "anomaly"
            else:
                raise SchemaValidationError(
                    f"Column '{value_col}' not found in DataFrame. Available columns: {list(series_or_df.columns)}"
                )
        working_df = series_or_df.copy()
        complete_col = None
        if "is_complete" in working_df.columns:
            complete_col = "is_complete"
        elif "complete_year" in working_df.columns:
            complete_col = "complete_year"

        if exclude_incomplete and complete_col is not None:
            working_df = working_df[working_df[complete_col] == True]

        valid_vals = working_df[actual_col].dropna().to_numpy(dtype=float)
    elif isinstance(series_or_df, pd.Series):
        valid_vals = series_or_df.dropna().to_numpy(dtype=float)
    else:
        raise TypeError(
            f"Expected pandas Series or DataFrame, got: {type(series_or_df).__name__}"
        )

    n_samples = len(valid_vals)
    if n_samples < 2:
        raise ValueError(
            f"Descriptive statistics require at least 2 valid observations, found {n_samples} (contains no valid observations or insufficient data)."
        )

    for p in percentiles:
        if not (0 <= p <= 100):
            raise ValueError(f"Percentile thresholds must be in the range [0, 100], got: {p}")

    mean_val = float(np.mean(valid_vals))
    std_val = float(np.std(valid_vals, ddof=1))
    min_val = float(np.min(valid_vals))
    max_val = float(np.max(valid_vals))
    range_val = max_val - min_val

    # Percentile calculations
    pct_dict = {
        int(p): float(np.percentile(valid_vals, p, method=interpolation))
        for p in percentiles
    }

    median_val = pct_dict.get(50, float(np.median(valid_vals)))
    p25 = pct_dict.get(25, float(np.percentile(valid_vals, 25, method=interpolation)))
    p75 = pct_dict.get(75, float(np.percentile(valid_vals, 75, method=interpolation)))
    iqr_val = p75 - p25

    skew_val = float(stats.skew(valid_vals, bias=False))
    kurt_val = float(stats.kurtosis(valid_vals, bias=False))

    return DescriptiveStatisticsResult(
        count=n_samples,
        mean=mean_val,
        std=std_val,
        min=min_val,
        max=max_val,
        range=range_val,
        median=median_val,
        iqr=iqr_val,
        skewness=skew_val,
        kurtosis=kurt_val,
        percentiles=pct_dict,
        interpolation_method=interpolation,
    )


def calculate_decadal_summary(
    df: pd.DataFrame,
    min_valid_years: int = 10,
    min_years_threshold: Optional[int] = None,
    exclude_incomplete_years: bool = True,
    value_col: str = "annual_mean_anomaly",
) -> pd.DataFrame:
    """Group annual observations into standard calendar decades with completeness tracking.

    Calendar Decade Convention:
    Decades follow standard astronomical/calendar boundaries:
    - 1880–1889, 1890–1899, ..., 2010–2019, 2020–2029.
    Decade start year is `(year // 10) * 10`; decade end year is `decade_start + 9`.

    Completeness & Missing Value Policy:
    - Decades with fewer than `effective_min` complete observations (default 10)
      have `complete_decade_mean = NaN` and `is_complete = False`.
    - Partial decades (such as 2020–2029, which currently only has observed years 2020–2025)
      are NEVER labeled complete. Their arithmetic average is reported exclusively as
      `partial_decade_mean` to prevent partial decadal figures from being cited as full decadal means.
    - Missing years within a decade are NEVER filled or interpolated.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from calculate_annual_summary() containing 'year' and value_col.
    min_valid_years : int, default 10
        Minimum number of valid complete annual anomalies required for a complete decade (1 to 10).
    min_years_threshold : int, optional
        Filter threshold: if provided, decades with fewer than `min_years_threshold`
        valid observations are dropped from the returned summary.
    exclude_incomplete_years : bool, default True
        If True, only years with is_complete=True (or non-null annual_mean_anomaly)
        are aggregated into decadal means.
    value_col : str, default 'annual_mean_anomaly'
        Annual anomaly column to aggregate.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'decade' (int): Starting year e.g. 1980.
        - 'decade_label' (str): Label e.g. '1980-1989'.
        - 'decade_start' (int): Starting calendar year.
        - 'decade_end' (int): Ending calendar year.
        - 'mean_anomaly' (float): Mean anomaly (°C) of complete decade, else NaN.
        - 'complete_decade_mean' (float): Alias for mean_anomaly.
        - 'partial_decade_mean' (float): Diagnostic mean of available valid years.
        - 'valid_years' (int): Count of valid complete annual observations.
        - 'valid_years_count' (int): Alias for valid_years.
        - 'missing_years_count' (int): 10 - valid_years.
        - 'expected_years' (int): 10.
        - 'completeness_pct' (float): valid_years / 10 * 100.
        - 'is_complete' (bool): True if valid_years >= min_valid_years.
        - 'is_complete_decade' (bool): Alias for is_complete.

    Raises
    ------
    SchemaValidationError
        If df lacks required 'year' or value_col.
    ValueError
        If min_valid_years or min_years_threshold is outside [1, 10].
    """
    effective_threshold = (
        min_years_threshold if min_years_threshold is not None else min_valid_years
    )
    if effective_threshold < 1 or effective_threshold > 10:
        raise ValueError(
            f"min_years_threshold must be an integer between 1 and 10, got: {effective_threshold}"
        )

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}.")

    if "year" not in df.columns:
        raise SchemaValidationError("Required column 'year' missing from DataFrame.")

    actual_col = value_col
    if actual_col not in df.columns:
        if actual_col == "annual_mean_anomaly" and "annual_mean" in df.columns:
            actual_col = "annual_mean"
        elif actual_col == "annual_mean_anomaly" and "anomaly" in df.columns:
            actual_col = "anomaly"
        else:
            raise SchemaValidationError(f"Missing required columns in DataFrame: ['{value_col}']")

    work_df = df.copy()

    complete_col = None
    if "is_complete" in work_df.columns:
        complete_col = "is_complete"
    elif "complete_year" in work_df.columns:
        complete_col = "complete_year"

    # Determine validity of each annual observation
    if exclude_incomplete_years and complete_col is not None:
        work_df["_valid_anomaly"] = work_df.apply(
            lambda r: r[actual_col] if r[complete_col] else np.nan, axis=1
        )
    else:
        work_df["_valid_anomaly"] = work_df[actual_col]

    work_df["decade_start"] = (work_df["year"] // 10) * 10
    work_df["decade_end"] = work_df["decade_start"] + 9

    records = []
    for (d_start, d_end), group in work_df.groupby(["decade_start", "decade_end"]):
        valid_vals = group["_valid_anomaly"].dropna()
        valid_count = len(valid_vals)
        completeness = (valid_count / 10.0) * 100.0
        is_complete = valid_count >= effective_threshold

        partial_mean = float(valid_vals.mean()) if valid_count > 0 else np.nan
        complete_mean = partial_mean if is_complete else np.nan

        records.append(
            {
                "decade": int(d_start),
                "decade_label": f"{d_start}-{d_end}",
                "decade_start": int(d_start),
                "decade_end": int(d_end),
                "mean_anomaly": complete_mean,
                "complete_decade_mean": complete_mean,
                "partial_decade_mean": partial_mean,
                "valid_years": int(valid_count),
                "valid_years_count": int(valid_count),
                "missing_years_count": int(10 - valid_count),
                "expected_years": 10,
                "completeness_pct": float(completeness),
                "is_complete": bool(is_complete),
                "is_complete_decade": bool(is_complete),
            }
        )

    decadal_df = pd.DataFrame(records).sort_values("decade_start").reset_index(drop=True)

    if min_years_threshold is not None:
        decadal_df = (
            decadal_df[decadal_df["valid_years"] >= min_years_threshold]
            .reset_index(drop=True)
        )

    return decadal_df


def calculate_warmth_percentiles(
    annual_df: pd.DataFrame,
    reference_period: Optional[Union[Tuple[int, int], Sequence[int]]] = None,
    reference_start: Optional[int] = None,
    reference_end: Optional[int] = None,
    value_col: str = "annual_mean_anomaly",
    exclude_incomplete: bool = True,
    min_reference_years: int = 3,
    ranking_order: str = "warmest_first",
    ties_method: str = "min",
) -> pd.DataFrame:
    """Compute warmth rankings and empirical percentile scores against a specified reference period.

    Methodology:
    - Compares each annual anomaly against a defined reference distribution [reference_start, reference_end].
      If no bounds are provided, the reference distribution encompasses all valid historical years.
    - Percentile score (0.0 to 100.0) represents the empirical percentage of reference years
      that were cooler than or equal to that year (scipy percentileofscore, kind='weak').
    - Warmth Rank (1 = warmest if ranking_order='warmest_first') is computed across evaluated years.

    Scientific Qualification:
    Percentile scores are strictly conditional on the reference period specified.
    Do not cite a percentile score without reporting its reference window.

    Parameters
    ----------
    annual_df : pd.DataFrame
        DataFrame from calculate_annual_summary().
    reference_period : tuple of (int, int), optional
        (start_year, end_year) of the baseline reference distribution.
    reference_start : int, optional
        Starting calendar year of the reference baseline period.
    reference_end : int, optional
        Ending calendar year of the reference baseline period.
    value_col : str, default 'annual_mean_anomaly'
        Column name containing annual mean anomalies.
    exclude_incomplete : bool, default True
        If True, only complete years are scored and used in the reference distribution.
    min_reference_years : int, default 3
        Minimum valid reference years required in the baseline distribution.
    ranking_order : str, default 'warmest_first'
        'warmest_first' assigns rank 1 to the highest anomaly; 'coolest_first' assigns rank 1 to lowest.
    ties_method : str, default 'min'
        Method used for rank tie handling ('min', 'average', 'max', 'dense').

    Returns
    -------
    pd.DataFrame
        DataFrame with added columns:
        - 'warmth_rank' / 'rank': Integer rank.
        - 'percentile_score' / 'percentile': Empirical percentile score (0.0 to 100.0).
        - 'reference_period': Label describing the reference period used.

    Raises
    ------
    SchemaValidationError
        If value_col is missing.
    ValueError
        If reference period is invalid or contains fewer than min_reference_years.
    """
    if not isinstance(annual_df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(annual_df).__name__}.")

    actual_col = value_col
    if actual_col not in annual_df.columns:
        if actual_col == "annual_mean_anomaly" and "annual_mean" in annual_df.columns:
            actual_col = "annual_mean"
        elif actual_col == "annual_mean_anomaly" and "anomaly" in annual_df.columns:
            actual_col = "anomaly"
        else:
            raise SchemaValidationError(f"Missing required column '{value_col}' in annual_df.")

    work_df = annual_df.copy().sort_values("year").reset_index(drop=True)

    complete_col = None
    if "is_complete" in work_df.columns:
        complete_col = "is_complete"
    elif "complete_year" in work_df.columns:
        complete_col = "complete_year"

    # Filter eligible rows
    if exclude_incomplete and complete_col is not None:
        eligible_mask = work_df[complete_col] == True
    else:
        eligible_mask = work_df[actual_col].notna()

    # Determine reference distribution
    if reference_period is not None:
        reference_start = int(reference_period[0])
        reference_end = int(reference_period[1])

    ref_df = work_df[eligible_mask].copy()
    if reference_start is not None and reference_end is not None:
        if reference_start > reference_end:
            raise ValueError(
                f"reference_start ({reference_start}) > reference_end ({reference_end})."
            )
        ref_df = ref_df[(ref_df["year"] >= reference_start) & (ref_df["year"] <= reference_end)]
        ref_label = f"{reference_start}-{reference_end}"
    elif reference_start is not None or reference_end is not None:
        raise ValueError("Must provide both reference_start and reference_end, or neither.")
    else:
        ref_label = f"Full-Record ({ref_df['year'].min()}-{ref_df['year'].max()})"

    ref_values = ref_df[actual_col].dropna().to_numpy(dtype=float)
    if len(ref_values) < min_reference_years:
        raise ValueError(
            f"Reference distribution requires at least {min_reference_years} valid observations, "
            f"found {len(ref_values)} in [{reference_start}, {reference_end}]."
        )

    # Compute percentiles and ranks
    percentile_scores = []
    warmth_ranks = []

    # Dense warmth rank among eligible years
    valid_series = work_df.loc[eligible_mask, actual_col]
    ascending = ranking_order != "warmest_first"
    rank_series = valid_series.rank(ascending=ascending, method=ties_method).astype(int)

    for idx, row in work_df.iterrows():
        if not eligible_mask.loc[idx] or pd.isna(row[actual_col]):
            percentile_scores.append(np.nan)
            warmth_ranks.append(np.nan)
        else:
            val = float(row[actual_col])
            # kind='weak' includes tied values in the percentile score
            pct = stats.percentileofscore(ref_values, val, kind="weak")
            percentile_scores.append(float(pct))
            warmth_ranks.append(rank_series.loc[idx])

    work_df["warmth_rank"] = pd.Series(warmth_ranks, dtype="Int64")
    work_df["rank"] = work_df["warmth_rank"]
    work_df["percentile_score"] = pd.Series(percentile_scores, dtype="float64")
    work_df["percentile"] = work_df["percentile_score"]
    work_df["reference_period"] = ref_label

    return work_df


# ---------------------------------------------------------------------------
# Monthly Climatology / Seasonal Distribution
# ---------------------------------------------------------------------------

CALENDAR_MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def calculate_monthly_climatology(
    df: pd.DataFrame,
    value_col: str = "anomaly",
    reference_start: Optional[int] = None,
    reference_end: Optional[int] = None,
    min_years_per_month: int = 1,
) -> pd.DataFrame:
    """Compute monthly climatological distributions and central tendencies across calendar months.

    For each calendar month (Jan through Dec), this function computes the sample mean,
    standard deviation, median, interquartile range (25th-75th percentiles), and extremes
    of the anomaly series over the specified evaluation window [reference_start, reference_end].

    SCIENTIFIC INTEGRITY & BASELINE DISTINCTION:
    - This function SUMMARIZES the input anomaly series across the specified period.
    - It does NOT re-center or rebaseline the anomalies by subtracting calendar-month means.
      The input anomalies retain their original reference baseline (e.g. 1951–1980 for NASA GISTEMP).
    - To re-center anomalies relative to an alternative multi-year baseline, use
      `calculate_baseline_anomalies()` in `climate_analyzer.feature_engineering`.
    - Missing observations are NEVER filled with zero or interpolated; only non-null values are evaluated.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame conforming to the monthly schema containing 'month' and value_col.
        If 'year' is present, enables filtering by evaluation window [reference_start, reference_end].
    value_col : str, default 'anomaly'
        Column name containing monthly anomalies in °C.
    reference_start : int, optional
        First year of the evaluation summarization window (inclusive).
    reference_end : int, optional
        Last year of the evaluation summarization window (inclusive).
    min_years_per_month : int, default 1
        Minimum number of valid monthly observations required for a month.


    Returns
    -------
    pd.DataFrame
        DataFrame with 12 rows (months 1 to 12) containing:
        - 'month' (int): Month index (1 to 12).
        - 'month_name' (str): Three-letter abbreviation ('Jan' to 'Dec').
        - 'count' (int): Number of valid observations analyzed.
        - 'mean_anomaly' (float): Arithmetic mean anomaly (°C).
        - 'std' (float): Unbiased sample standard deviation (°C, ddof=1).
        - 'median' (float): Median anomaly (°C).
        - 'q25' (float): 25th percentile anomaly (°C).
        - 'q75' (float): 75th percentile anomaly (°C).
        - 'min' (float): Minimum observed anomaly (°C).
        - 'max' (float): Maximum observed anomaly (°C).
        - 'reference_start' (int or None): Reference start year.
        - 'reference_end' (int or None): Reference end year.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame.
    SchemaValidationError
        If df lacks required 'month' or value_col.
    ValueError
        If reference_start > reference_end, or if no valid observations exist in the window.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got: {type(df).__name__}")

    if "month" not in df.columns:
        raise SchemaValidationError("Required column 'month' missing from DataFrame.")

    actual_col = value_col
    if actual_col not in df.columns:
        if actual_col == "anomaly" and "monthly_anomaly" in df.columns:
            actual_col = "monthly_anomaly"
        else:
            raise SchemaValidationError(
                f"Column '{value_col}' not found in DataFrame. Available: {list(df.columns)}"
            )

    work_df = df.copy()

    if reference_start is not None and reference_end is not None:
        if reference_start > reference_end:
            raise ValueError(
                f"reference_start ({reference_start}) > reference_end ({reference_end})."
            )
        if "year" not in work_df.columns:
            raise SchemaValidationError("Filtering by reference years requires 'year' column.")
        work_df = work_df[(work_df["year"] >= reference_start) & (work_df["year"] <= reference_end)]
    elif reference_start is not None or reference_end is not None:
        raise ValueError("Must provide both reference_start and reference_end, or neither.")

    if work_df.empty:
        raise ValueError(
            f"No observations available in specified reference window [{reference_start}, {reference_end}]."
        )

    # Check if there are any non-null anomaly observations at all
    valid_total = work_df[actual_col].dropna()
    if valid_total.empty:
        raise ValueError(
            f"No valid non-null observations found in column '{actual_col}'."
        )

    records = []
    for m in range(1, 13):
        m_subset = work_df[work_df["month"] == m]
        valid_vals = m_subset[actual_col].dropna().to_numpy(dtype=float)
        cnt = len(valid_vals)

        if cnt < min_years_per_month:
            records.append({
                "month": m,
                "month_name": CALENDAR_MONTH_NAMES[m - 1],
                "count": cnt,
                "mean_anomaly": np.nan,
                "std": np.nan,
                "median": np.nan,
                "q25": np.nan,
                "q75": np.nan,
                "min": np.nan,
                "max": np.nan,
                "reference_start": reference_start,
                "reference_end": reference_end,
            })
        else:
            mean_val = float(np.mean(valid_vals))
            std_val = float(np.std(valid_vals, ddof=1)) if cnt >= 2 else 0.0
            median_val = float(np.median(valid_vals))
            q25_val = float(np.percentile(valid_vals, 25))
            q75_val = float(np.percentile(valid_vals, 75))
            min_val = float(np.min(valid_vals))
            max_val = float(np.max(valid_vals))

            records.append({
                "month": m,
                "month_name": CALENDAR_MONTH_NAMES[m - 1],
                "count": cnt,
                "mean_anomaly": mean_val,
                "std": std_val,
                "median": median_val,
                "q25": q25_val,
                "q75": q75_val,
                "min": min_val,
                "max": max_val,
                "reference_start": reference_start,
                "reference_end": reference_end,
            })

    return pd.DataFrame(records)

