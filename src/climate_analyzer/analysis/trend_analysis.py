"""
Statistical trend estimation and non-parametric monotonic trend testing for climate series.

This module provides three robust methods for trend estimation on annual climate records:
1. Ordinary Least Squares (OLS) parametric linear regression with centered time coordinates.
2. Mann-Kendall non-parametric monotonic trend test with full tie correction.
3. Sen's non-parametric robust slope estimator with rank-based confidence bounds.

SCIENTIFIC INTEGRITY & STATISTICAL LIMITATIONS:
- Annual global temperature anomalies frequently exhibit serial autocorrelation (e.g. from ENSO cycles
  or ocean heat inertia).
- Conventional OLS standard errors and standard Mann-Kendall p-values assume independent errors;
  positive serial autocorrelation can underestimate uncertainty and artificially inflate significance.
- A statistically significant trend describes observational slope in the analyzed window and
  does NOT establish physical attribution, human vs. natural causation, or unconstrained future projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from climate_analyzer.data_validation import SchemaValidationError, TemporalIntegrityError


# ---------------------------------------------------------------------------
# Result Data Structures
# ---------------------------------------------------------------------------


@dataclass
class OLSTrendResult:
    """Results from Ordinary Least Squares (OLS) linear trend estimation.

    Attributes
    ----------
    slope_per_year : float
        Rate of temperature change in °C per calendar year.
    slope_per_decade : float
        Rate of temperature change in °C per decade (slope_per_year * 10).
    intercept : float
        Model intercept at the centered year coordinate (t* = year - center_year).
    intercept_raw : float
        Extrapolated intercept at calendar year 0 (for mathematical completeness).
    center_year : float
        Center year coordinate used to maintain numeric stability and parameter orthogonality.
    std_err_per_decade : float
        Standard error of the decadal slope (°C/decade).
    p_value : float
        Two-sided p-value testing H0: slope == 0 under conventional OLS assumptions.
    ci_lower_per_decade : float
        Lower bound of the 95% confidence interval for decadal slope (°C/decade).
    ci_upper_per_decade : float
        Upper bound of the 95% confidence interval for decadal slope (°C/decade).
    r_squared : float
        Coefficient of determination (R^2).
    sample_count : int
        Number of valid annual observations analyzed.
    start_year : int
        Earliest year analyzed in the regression.
    end_year : int
        Latest year analyzed in the regression.
    assumptions_note : str
        Documentation of inference assumptions and serial autocorrelation limitations.
    residual_autocorr_lag1 : float, optional
        Empirical lag-1 autocorrelation coefficient (r1) of regression residuals.
    durbin_watson : float, optional
        Durbin-Watson d test statistic for first-order serial correlation in residuals.
    """

    slope_per_year: float
    slope_per_decade: float
    intercept: float
    intercept_raw: float
    center_year: float
    std_err_per_decade: float
    p_value: float
    ci_lower_per_decade: float
    ci_upper_per_decade: float
    r_squared: float
    sample_count: int
    start_year: int
    end_year: int
    assumptions_note: str
    residual_autocorr_lag1: Optional[float] = None
    durbin_watson: Optional[float] = None

    def summary(self) -> str:
        """Formatted text summary of OLS linear trend estimation."""
        lines = [
            "OLS Linear Trend Estimation",
            f"  Analyzed Period:     {self.start_year} to {self.end_year} ({self.sample_count} years)",
            f"  Decadal Trend:       {self.slope_per_decade:+.4f} °C/decade (95% CI: [{self.ci_lower_per_decade:+.4f}, {self.ci_upper_per_decade:+.4f}])",
            f"  Annual Trend:        {self.slope_per_year:+.6f} °C/year",
            f"  Model Intercept:     {self.intercept:+.4f} °C (at center year {self.center_year:.1f})",
            f"  R-Squared:           {self.r_squared:.4f}",
            f"  Std Error:           {self.std_err_per_decade:.4f} °C/decade",
            f"  p-value:             {self.p_value:.4e} (conventional two-sided t-test)",
        ]
        if self.residual_autocorr_lag1 is not None and self.durbin_watson is not None:
            lines.append(
                f"  Residual Autocorr:   r1 = {self.residual_autocorr_lag1:+.4f} (Durbin-Watson d = {self.durbin_watson:.4f})"
            )
        lines.append(f"  Limitations:         {self.assumptions_note}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert results to a standard dictionary."""
        return {
            "slope_per_year": self.slope_per_year,
            "slope_per_decade": self.slope_per_decade,
            "intercept": self.intercept,
            "intercept_raw": self.intercept_raw,
            "center_year": self.center_year,
            "std_err_per_decade": self.std_err_per_decade,
            "p_value": self.p_value,
            "ci_lower_per_decade": self.ci_lower_per_decade,
            "ci_upper_per_decade": self.ci_upper_per_decade,
            "r_squared": self.r_squared,
            "sample_count": self.sample_count,
            "start_year": self.start_year,
            "end_year": self.end_year,
            "assumptions_note": self.assumptions_note,
            "residual_autocorr_lag1": self.residual_autocorr_lag1,
            "durbin_watson": self.durbin_watson,
        }


@dataclass
class MannKendallResult:
    """Results from the Mann-Kendall non-parametric monotonic trend test.

    Attributes
    ----------
    s_statistic : float
        Kendall S score (sum of sign differences across all observation pairs).
    var_s : float
        Variance of S with exact tie correction.
    z_statistic : float
        Standard normal test statistic with continuity correction.
    p_value : float
        Two-sided p-value testing H0: no monotonic trend against H1: monotonic trend.
    trend_direction : str
        'increasing', 'decreasing', or 'no_trend'.
    sample_count : int
        Number of valid observations analyzed.
    start_year : int
        Earliest year analyzed.
    end_year : int
        Latest year analyzed.
    method_variant : str
        Specification of test variant and tie correction method.
    assumptions_note : str
        Documentation of assumptions and autocorrelation limitations.
    log10_p_value : float, default 0.0
        Base-10 logarithm of the two-sided p-value for numerical stability when p << 1e-15.
    """

    s_statistic: float
    var_s: float
    z_statistic: float
    p_value: float
    trend_direction: str
    sample_count: int
    start_year: int
    end_year: int
    method_variant: str
    assumptions_note: str
    log10_p_value: float = 0.0

    def summary(self) -> str:
        """Formatted text summary of Mann-Kendall trend test."""
        if self.p_value > 0.0:
            pval_str = f"{self.p_value:.4e} (log10(p) = {self.log10_p_value:.2f})"
        else:
            pval_str = f"< 1e-300 (log10(p) ≈ {self.log10_p_value:.1f})"

        return (
            "Mann-Kendall Monotonic Trend Test\n"
            f"  Analyzed Period:     {self.start_year} to {self.end_year} ({self.sample_count} years)\n"
            f"  Trend Direction:     {self.trend_direction.upper()}\n"
            f"  Test Statistic (S):  {self.s_statistic:+.0f}\n"
            f"  Variance of S:       {self.var_s:.2f}\n"
            f"  Standardized Z:      {self.z_statistic:+.4f}\n"
            f"  Two-Sided p-value:   {pval_str}\n"
            f"  Method:              {self.method_variant}\n"
            f"  Limitations:         {self.assumptions_note}"
        )

    def to_dict(self) -> dict:
        """Convert results to a standard dictionary."""
        return {
            "s_statistic": self.s_statistic,
            "var_s": self.var_s,
            "z_statistic": self.z_statistic,
            "p_value": self.p_value,
            "trend_direction": self.trend_direction,
            "sample_count": self.sample_count,
            "start_year": self.start_year,
            "end_year": self.end_year,
            "method_variant": self.method_variant,
            "assumptions_note": self.assumptions_note,
            "log10_p_value": self.log10_p_value,
        }


@dataclass
class SensSlopeResult:
    """Results from Sen's robust non-parametric slope estimation.

    Attributes
    ----------
    slope_per_year : float
        Median pairwise slope in °C per calendar year.
    slope_per_decade : float
        Median pairwise slope in °C per decade (slope_per_year * 10).
    ci_lower_per_decade : float
        Non-parametric 95% confidence interval lower bound for decadal slope (°C/decade).
    ci_upper_per_decade : float
        Non-parametric 95% confidence interval upper bound for decadal slope (°C/decade).
    pairwise_slopes_count : int
        Total number of valid pairwise slopes evaluated (n * (n - 1) / 2).
    sample_count : int
        Number of valid annual observations analyzed.
    start_year : int
        Earliest year analyzed.
    end_year : int
        Latest year analyzed.
    assumptions_note : str
        Documentation of assumptions and properties.
    """

    slope_per_year: float
    slope_per_decade: float
    ci_lower_per_decade: float
    ci_upper_per_decade: float
    pairwise_slopes_count: int
    sample_count: int
    start_year: int
    end_year: int
    assumptions_note: str

    def summary(self) -> str:
        """Formatted text summary of Sen's slope estimation."""
        return (
            "Sen's Robust Slope Estimation\n"
            f"  Analyzed Period:     {self.start_year} to {self.end_year} ({self.sample_count} years)\n"
            f"  Decadal Sen's Slope: {self.slope_per_decade:+.4f} °C/decade (95% CI: [{self.ci_lower_per_decade:+.4f}, {self.ci_upper_per_decade:+.4f}])\n"
            f"  Annual Sen's Slope:  {self.slope_per_year:+.6f} °C/year\n"
            f"  Pairwise Slopes:     {self.pairwise_slopes_count} pairs evaluated\n"
            f"  Limitations:         {self.assumptions_note}"
        )

    def to_dict(self) -> dict:
        """Convert results to a standard dictionary."""
        return {
            "slope_per_year": self.slope_per_year,
            "slope_per_decade": self.slope_per_decade,
            "ci_lower_per_decade": self.ci_lower_per_decade,
            "ci_upper_per_decade": self.ci_upper_per_decade,
            "pairwise_slopes_count": self.pairwise_slopes_count,
            "sample_count": self.sample_count,
            "start_year": self.start_year,
            "end_year": self.end_year,
            "assumptions_note": self.assumptions_note,
        }


# ---------------------------------------------------------------------------
# Validation & Preparation Helper
# ---------------------------------------------------------------------------


def _prepare_annual_series(
    df: pd.DataFrame,
    value_col: str = "annual_mean_anomaly",
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    exclude_incomplete: bool = True,
    min_required_years: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and extract clean (years, anomalies) arrays from annual input.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from calculate_annual_summary().
    value_col : str, default 'annual_mean_anomaly'
        Column name containing annual mean anomalies.
    start_year : int, optional
        Starting calendar year of analysis window (inclusive).
    end_year : int, optional
        Ending calendar year of analysis window (inclusive).
    exclude_incomplete : bool, default True
        If True, excludes rows where is_complete=False or value is NaN.
    min_required_years : int, default 5
        Minimum valid annual observations required for statistical estimation.

    Returns
    -------
    years : np.ndarray (int64)
        Sorted array of valid calendar years.
    anomalies : np.ndarray (float64)
        Array of corresponding annual mean anomalies in °C.

    Raises
    ------
    SchemaValidationError
        If required columns are missing.
    TemporalIntegrityError
        If years contain duplicates.
    ValueError
        If fewer than min_required_years valid observations are available.
    """
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

    # Filter by completeness
    complete_col = None
    if "is_complete" in work_df.columns:
        complete_col = "is_complete"
    elif "complete_year" in work_df.columns:
        complete_col = "complete_year"

    if exclude_incomplete and complete_col is not None:
        work_df = work_df[work_df[complete_col] == True]

    # Drop null values in actual_col
    work_df = work_df.dropna(subset=[actual_col, "year"])

    # Optional year bounds
    if start_year is not None:
        work_df = work_df[work_df["year"] >= start_year]
    if end_year is not None:
        work_df = work_df[work_df["year"] <= end_year]

    if work_df.empty:
        raise ValueError(
            f"No valid annual observations available in specified range [{start_year}, {end_year}]."
        )

    # Validate duplicates
    if work_df["year"].duplicated().any():
        dups = work_df["year"][work_df["year"].duplicated()].tolist()
        raise TemporalIntegrityError(f"Duplicate years detected in annual series: {dups}")

    # Sort strictly chronologically
    work_df = work_df.sort_values("year").reset_index(drop=True)

    years = work_df["year"].to_numpy(dtype=int)
    anomalies = work_df[actual_col].to_numpy(dtype=float)

    if len(years) < min_required_years:
        raise ValueError(
            f"Trend estimation requires at least {min_required_years} valid annual observations, "
            f"found {len(years)} in [{start_year}, {end_year}]."
        )

    return years, anomalies


# ---------------------------------------------------------------------------
# Core Trend Estimation Methods
# ---------------------------------------------------------------------------


def estimate_ols_trend(
    df: pd.DataFrame,
    value_col: str = "annual_mean_anomaly",
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    exclude_incomplete: bool = True,
    confidence_level: float = 0.95,
) -> OLSTrendResult:
    """Estimate parametric Ordinary Least Squares (OLS) linear trend on annual climate series.

    Time Coordinate & Numerical Stability:
    A centered time coordinate is used:
        t* = year - center_year, where center_year = mean(years)
    The fitted linear model is:
        anomaly_t = intercept + slope * t*
    The slope is mathematically identical to regression on raw calendar years, but centering
    eliminates high correlation between intercept and slope and preserves numeric stability.
    The intercept represents the estimated anomaly at the midpoint center_year.

    Statistical Limitations:
    Conventional OLS inference assumes independent and identically distributed (i.i.d.) normal errors.
    In climate anomaly series, positive serial autocorrelation (e.g. from El Niño-Southern Oscillation
    or ocean heat inertia) deflates standard errors, causing p-values to appear artificially significant.
    Conventional p-values must therefore be interpreted conditionally.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from calculate_annual_summary().
    value_col : str, default 'annual_mean_anomaly'
        Column name containing annual mean anomalies.
    start_year : int, optional
        First year to include in analysis (inclusive).
    end_year : int, optional
        Last year to include in analysis (inclusive).
    exclude_incomplete : bool, default True
        If True, excludes incomplete years (such as an ongoing year).
    confidence_level : float, default 0.95
        Confidence level for slope confidence interval (e.g. 0.95 for 95% CI).

    Returns
    -------
    OLSTrendResult
        Structured results dataclass.
    """
    years, anomalies = _prepare_annual_series(
        df,
        value_col=value_col,
        start_year=start_year,
        end_year=end_year,
        exclude_incomplete=exclude_incomplete,
        min_required_years=3,
    )

    n = len(years)
    center_year = float(np.mean(years))
    t_centered = years.astype(float) - center_year

    # Scipy linregress on centered coordinates
    reg = stats.linregress(t_centered, anomalies)

    slope_per_year = float(reg.slope)
    slope_per_decade = slope_per_year * 10.0
    intercept = float(reg.intercept)
    # Raw intercept at year 0: intercept_raw = intercept - slope * center_year
    intercept_raw = intercept - (slope_per_year * center_year)

    std_err_year = float(reg.stderr)
    std_err_decade = std_err_year * 10.0

    # Student's t critical value for df = n - 2
    dof = n - 2
    alpha = 1.0 - confidence_level
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=dof))

    ci_lower_decade = slope_per_decade - (t_crit * std_err_decade)
    ci_upper_decade = slope_per_decade + (t_crit * std_err_decade)

    # Residual serial autocorrelation diagnostics
    fitted_vals = intercept + (slope_per_year * t_centered)
    residuals = anomalies - fitted_vals
    ss_res = float(np.sum(residuals**2))
    if ss_res > 0:
        dw = float(np.sum(np.diff(residuals)**2) / ss_res)
        r1 = float(np.sum(residuals[1:] * residuals[:-1]) / ss_res)
    else:
        dw = 2.0
        r1 = 0.0

    assumptions_note = (
        "Conventional OLS assumes independent, homoscedastic errors. Serial autocorrelation in climate "
        "series may underestimate standard errors and overstate statistical significance. Results should "
        "not be taken as causal attribution."
    )

    return OLSTrendResult(
        slope_per_year=slope_per_year,
        slope_per_decade=slope_per_decade,
        intercept=intercept,
        intercept_raw=intercept_raw,
        center_year=center_year,
        std_err_per_decade=std_err_decade,
        p_value=float(reg.pvalue),
        ci_lower_per_decade=ci_lower_decade,
        ci_upper_per_decade=ci_upper_decade,
        r_squared=float(reg.rvalue**2),
        sample_count=n,
        start_year=int(years[0]),
        end_year=int(years[-1]),
        assumptions_note=assumptions_note,
        residual_autocorr_lag1=r1,
        durbin_watson=dw,
    )


def test_mann_kendall_trend(
    df: pd.DataFrame,
    value_col: str = "annual_mean_anomaly",
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    exclude_incomplete: bool = True,
    alpha: float = 0.05,
) -> MannKendallResult:
    """Perform the Mann-Kendall non-parametric monotonic trend test with full tie correction.

    Methodology:
    Kendall's S score measures monotonic rank ordering across all pairwise time points:
        S = sum_{k=1}^{n-1} sum_{j=k+1}^n sgn(x_j - x_k)
    Variance of S incorporates exact tie correction:
        Var(S) = (1/18) * [ n(n-1)(2n+5) - sum_{p=1}^g t_p(t_p - 1)(2t_p + 5) ]
    where g is the count of tied groups and t_p is the group size of tie p.
    Standardized normal test statistic Z is continuity-corrected:
        Z = (S - 1) / sqrt(Var(S)) if S > 0
        Z = 0                      if S == 0
        Z = (S + 1) / sqrt(Var(S)) if S < 0
    Two-sided p-value is computed using the survival function (2 * norm.sf(|Z|)) to prevent
    premature floating-point underflow to exact 0.0 for 8.5 < |Z| < 37.5, and log10(p) is
    reported via logsf to preserve exact scientific order of magnitude.

    Statistical Limitations:
    The standard Mann-Kendall test assumes independent observations. Positive serial correlation
    increases the variance of S and inflates Type I error rates (false positives).
    No pre-whitening or variance correction is applied in this standard test.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from calculate_annual_summary().
    value_col : str, default 'annual_mean_anomaly'
        Column name containing annual mean anomalies.
    start_year : int, optional
        First year of analysis (inclusive).
    end_year : int, optional
        Last year of analysis (inclusive).
    exclude_incomplete : bool, default True
        If True, excludes incomplete annual records.
    alpha : float, default 0.05
        Significance level threshold to report 'increasing' or 'decreasing' vs 'no_trend'.

    Returns
    -------
    MannKendallResult
        Structured results dataclass.
    """
    years, anomalies = _prepare_annual_series(
        df,
        value_col=value_col,
        start_year=start_year,
        end_year=end_year,
        exclude_incomplete=exclude_incomplete,
        min_required_years=3,
    )

    n = len(anomalies)

    # Compute Kendall's S statistic
    s = 0
    for k in range(n - 1):
        diffs = anomalies[k + 1 :] - anomalies[k]
        s += int(np.sum(np.sign(diffs)))

    # Compute tie correction for Var(S)
    unique_vals, counts = np.unique(anomalies, return_counts=True)
    tie_groups = counts[counts > 1]
    tie_term = np.sum(tie_groups * (tie_groups - 1) * (2 * tie_groups + 5))

    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0

    # Continuity-corrected Z
    if var_s <= 0 or np.isclose(var_s, 0.0):
        z = 0.0
    elif s > 0:
        z = (s - 1.0) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1.0) / np.sqrt(var_s)
    else:
        z = 0.0

    if z == 0.0:
        p_val = 1.0
        log10_p = 0.0
    else:
        abs_z = abs(z)
        p_val = float(2.0 * stats.norm.sf(abs_z))
        log10_p = float((np.log(2.0) + stats.norm.logsf(abs_z)) / np.log(10.0))

    if p_val > alpha:
        direction = "no_trend"
    elif s > 0:
        direction = "increasing"
    elif s < 0:
        direction = "decreasing"
    else:
        direction = "no_trend"

    assumptions_note = (
        "Standard Mann-Kendall assumes independent observations. In the presence of serial autocorrelation, "
        "false rejection of the null hypothesis increases. Does not establish physical causation."
    )

    return MannKendallResult(
        s_statistic=float(s),
        var_s=float(var_s),
        z_statistic=float(z),
        p_value=p_val,
        trend_direction=direction,
        sample_count=n,
        start_year=int(years[0]),
        end_year=int(years[-1]),
        method_variant="Mann-Kendall (standard normal with tie correction)",
        assumptions_note=assumptions_note,
        log10_p_value=log10_p,
    )


# Prevent pytest from mistaking the analytical test_mann_kendall_trend function for a test suite
test_mann_kendall_trend.__test__ = False


def estimate_sens_slope(
    df: pd.DataFrame,
    value_col: str = "annual_mean_anomaly",
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    exclude_incomplete: bool = True,
    confidence_level: float = 0.95,
) -> SensSlopeResult:
    """Estimate Sen's robust non-parametric slope and confidence intervals on annual series.

    Methodology:
    Calculates the median of all pairwise slopes across calendar years:
        S_{ij} = (anomaly_j - anomaly_i) / (year_j - year_i), for all j > i
    Using actual calendar year differences explicitly handles nonconsecutive years rigorously.
    Non-parametric confidence intervals for Sen's slope are computed using Kendall's rank distribution
    (Gilbert 1987 / Sen 1968 method).

    Statistical Limitations:
    While Sen's slope is resistant to gross outliers and non-normal residuals, its rank-based
    confidence bounds still assume independent errors. Positive autocorrelation can cause
    confidence intervals to be narrower than nominal coverage.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from calculate_annual_summary().
    value_col : str, default 'annual_mean_anomaly'
        Column name containing annual mean anomalies.
    start_year : int, optional
        First year to include in analysis (inclusive).
    end_year : int, optional
        Last year to include in analysis (inclusive).
    exclude_incomplete : bool, default True
        If True, excludes incomplete annual records.
    confidence_level : float, default 0.95
        Confidence level for non-parametric rank interval (e.g. 0.95 for 95% CI).

    Returns
    -------
    SensSlopeResult
        Structured results dataclass.
    """
    years, anomalies = _prepare_annual_series(
        df,
        value_col=value_col,
        start_year=start_year,
        end_year=end_year,
        exclude_incomplete=exclude_incomplete,
        min_required_years=3,
    )

    n = len(years)

    # Compute all pairwise slopes with exact calendar time difference
    pairwise_slopes = []
    for i in range(n - 1):
        dy = anomalies[i + 1 :] - anomalies[i]
        dt = (years[i + 1 :] - years[i]).astype(float)
        # dt is guaranteed > 0 because years are strictly sorted and duplicates are rejected
        pairwise_slopes.extend(dy / dt)

    pairwise_slopes = np.sort(np.asarray(pairwise_slopes, dtype=float))
    n_slopes = len(pairwise_slopes)

    sen_slope_per_year = float(np.median(pairwise_slopes))
    sen_slope_per_decade = sen_slope_per_year * 10.0

    # Non-parametric rank-based confidence interval (Gilbert 1987 / Sen 1968)
    # Variance of Kendall's S with tie correction
    unique_vals, counts = np.unique(anomalies, return_counts=True)
    tie_groups = counts[counts > 1]
    tie_term = np.sum(tie_groups * (tie_groups - 1) * (2 * tie_groups + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0

    alpha = 1.0 - confidence_level
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))

    # Rank indices for confidence interval
    # In 1-indexed order statistics: M1 = round((N' - C_alpha) / 2), M2 = round((N' + C_alpha) / 2) + 1
    # In 0-based Python indexing:
    # Lower limit is rank M1 -> index M1 - 1
    # Upper limit is rank M2 + 1 -> index (M2 + 1) - 1 = M2
    c_alpha = z_crit * np.sqrt(var_s)
    idx_lower = max(0, min(n_slopes - 1, int(np.round((n_slopes - c_alpha) / 2.0)) - 1))
    idx_upper = max(0, min(n_slopes - 1, int(np.round((n_slopes + c_alpha) / 2.0))))

    ci_lower_decade = float(pairwise_slopes[idx_lower] * 10.0)
    ci_upper_decade = float(pairwise_slopes[idx_upper] * 10.0)

    assumptions_note = (
        "Sen's slope is robust to non-normal distributions and monotonic outliers. However, rank-based "
        "confidence intervals assume independent errors; serial autocorrelation can narrow confidence "
        "intervals below nominal coverage."
    )

    return SensSlopeResult(
        slope_per_year=sen_slope_per_year,
        slope_per_decade=sen_slope_per_decade,
        ci_lower_per_decade=ci_lower_decade,
        ci_upper_per_decade=ci_upper_decade,
        pairwise_slopes_count=n_slopes,
        sample_count=n,
        start_year=int(years[0]),
        end_year=int(years[-1]),
        assumptions_note=assumptions_note,
    )
