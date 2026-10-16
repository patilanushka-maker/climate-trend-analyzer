"""
Automated unit and numerical correctness tests for statistical trend analysis.

Covers:
- OLS linear trend estimation (centered time coordinates, decadal scaling, CIs, p-values).
- Mann-Kendall monotonic trend test (S-score, exact tie correction, continuity correction).
- Sen's robust slope estimator (median pairwise slopes, dt calendar intervals, rank CIs).
- Handling of incomplete years, nonconsecutive years, duplicate years, and edge cases.
- Enforcement of statistical integrity caveats (serial autocorrelation notes).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from climate_analyzer.analysis.trend_analysis import (
    MannKendallResult,
    OLSTrendResult,
    SensSlopeResult,
    estimate_ols_trend,
    estimate_sens_slope,
    test_mann_kendall_trend,
)
from climate_analyzer.data_validation import (
    SchemaValidationError,
    TemporalIntegrityError,
)


# ---------------------------------------------------------------------------
# Synthetic Test Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def perfect_linear_increasing() -> pd.DataFrame:
    """Fixture with a known exact slope of +0.02 °C/year (+0.20 °C/decade)."""
    years = list(range(1970, 2020))  # 50 years: 1970 to 2019
    # y = 0.02 * (year - 1970)
    anomalies = [0.02 * (yr - 1970) for yr in years]
    return pd.DataFrame(
        {
            "year": years,
            "annual_mean": anomalies,
            "complete_year": True,
            "observed_months": 12,
        }
    )


@pytest.fixture
def perfect_linear_decreasing() -> pd.DataFrame:
    """Fixture with a known exact slope of -0.015 °C/year (-0.150 °C/decade)."""
    years = list(range(1960, 2000))  # 40 years
    anomalies = [-0.015 * (yr - 1960) for yr in years]
    return pd.DataFrame(
        {
            "year": years,
            "annual_mean": anomalies,
            "complete_year": True,
            "observed_months": 12,
        }
    )


# ---------------------------------------------------------------------------
# OLS Linear Trend Tests
# ---------------------------------------------------------------------------


def test_ols_trend_perfect_linear(perfect_linear_increasing):
    """Verify OLS reproduces exact known synthetic linear trend."""
    res = estimate_ols_trend(perfect_linear_increasing)

    assert isinstance(res, OLSTrendResult)
    assert res.sample_count == 50
    assert res.start_year == 1970
    assert res.end_year == 2019

    # Known slope: 0.02 °C/year -> 0.20 °C/decade
    assert np.isclose(res.slope_per_year, 0.02, atol=1e-10)
    assert np.isclose(res.slope_per_decade, 0.20, atol=1e-10)

    # R^2 should be 1.0 on a noiseless line
    assert np.isclose(res.r_squared, 1.0, atol=1e-10)

    # p-value should be extremely small (< 1e-15)
    assert res.p_value < 1e-15

    # Center year is mean(1970..2019) = 1994.5
    assert np.isclose(res.center_year, 1994.5)

    # Check that decadal conversion is exact: slope_per_decade == slope_per_year * 10
    assert np.isclose(res.slope_per_decade, res.slope_per_year * 10.0)

    # Check assumptions note is populated
    assert "serial autocorrelation" in res.assumptions_note.lower()


def test_ols_trend_decreasing(perfect_linear_decreasing):
    """Verify OLS correctly estimates a negative slope."""
    res = estimate_ols_trend(perfect_linear_decreasing)
    assert np.isclose(res.slope_per_year, -0.015, atol=1e-10)
    assert np.isclose(res.slope_per_decade, -0.150, atol=1e-10)
    assert res.slope_per_year < 0.0


def test_ols_trend_constant_data():
    """Verify OLS handles horizontal / constant series with zero slope."""
    years = list(range(1980, 2000))
    df = pd.DataFrame(
        {"year": years, "annual_mean": [0.50] * len(years), "complete_year": True}
    )
    res = estimate_ols_trend(df)
    assert np.isclose(res.slope_per_year, 0.0, atol=1e-10)
    assert np.isclose(res.slope_per_decade, 0.0, atol=1e-10)
    assert np.isclose(res.r_squared, 0.0, atol=1e-10)


def test_ols_trend_excludes_incomplete_by_default():
    """Verify that incomplete years are excluded by default from OLS fitting."""
    years = [2000, 2001, 2002, 2003, 2004]
    # Complete years slope = 0.1/yr; 2004 is incomplete with an extreme outlier (10.0)
    anomalies = [0.1, 0.2, 0.3, 0.4, 10.0]
    complete = [True, True, True, True, False]
    df = pd.DataFrame(
        {"year": years, "annual_mean": anomalies, "complete_year": complete}
    )

    res_default = estimate_ols_trend(df)
    assert res_default.sample_count == 4
    assert res_default.end_year == 2003
    assert np.isclose(res_default.slope_per_year, 0.1, atol=1e-10)

    # When explicitly included, sample_count = 5 and slope is distorted
    res_included = estimate_ols_trend(df, exclude_incomplete=False)
    assert res_included.sample_count == 5
    assert res_included.end_year == 2004
    assert res_included.slope_per_year > 1.0


# ---------------------------------------------------------------------------
# Mann-Kendall Trend Test Tests
# ---------------------------------------------------------------------------


def test_mann_kendall_perfect_monotonic(perfect_linear_increasing):
    """Verify Mann-Kendall test on strictly monotonically increasing series."""
    res = test_mann_kendall_trend(perfect_linear_increasing)

    assert isinstance(res, MannKendallResult)
    assert res.trend_direction == "increasing"
    assert res.sample_count == 50
    # For n=50 strictly increasing, S = n*(n-1)/2 = 50*49/2 = 1225
    assert np.isclose(res.s_statistic, 1225.0)
    assert res.z_statistic > 0
    assert res.p_value < 1e-10
    assert "serial autocorrelation" in res.assumptions_note.lower()


def test_mann_kendall_decreasing(perfect_linear_decreasing):
    """Verify Mann-Kendall test on strictly monotonically decreasing series."""
    res = test_mann_kendall_trend(perfect_linear_decreasing)
    assert res.trend_direction == "decreasing"
    # For n=40 strictly decreasing, S = -40*39/2 = -780
    assert np.isclose(res.s_statistic, -780.0)
    assert res.z_statistic < 0
    assert res.p_value < 1e-10


def test_mann_kendall_constant_and_ties():
    """Verify Mann-Kendall tie correction on tied and constant data."""
    years = list(range(1990, 2000))
    # All identical values: S = 0, p_value = 1.0
    df_const = pd.DataFrame(
        {"year": years, "annual_mean": [0.42] * 10, "complete_year": True}
    )
    res_const = test_mann_kendall_trend(df_const)
    assert res_const.s_statistic == 0.0
    assert res_const.trend_direction == "no_trend"
    assert np.isclose(res_const.p_value, 1.0)

    # Series with partial ties: [0.1, 0.1, 0.2, 0.2, 0.3, 0.3]
    # S should still be positive and tie correction should reduce Var(S)
    df_ties = pd.DataFrame(
        {"year": list(range(2000, 2006)), "annual_mean": [0.1, 0.1, 0.2, 0.2, 0.3, 0.3], "complete_year": True}
    )
    res_ties = test_mann_kendall_trend(df_ties)
    assert res_ties.s_statistic > 0
    assert res_ties.trend_direction == "increasing"
    assert "tie correction" in res_ties.method_variant.lower()


def test_mann_kendall_alternating_no_trend():
    """Verify Mann-Kendall detects no monotonic trend in an oscillating series."""
    years = list(range(1980, 2010))  # 30 years
    # Alternating +0.2, -0.2
    anomalies = [0.2 if i % 2 == 0 else -0.2 for i in range(len(years))]
    df = pd.DataFrame({"year": years, "annual_mean": anomalies, "complete_year": True})
    res = test_mann_kendall_trend(df)
    assert res.trend_direction == "no_trend"
    assert res.p_value > 0.05


# ---------------------------------------------------------------------------
# Sen's Slope Estimator Tests
# ---------------------------------------------------------------------------


def test_sens_slope_agreement_with_ols(perfect_linear_increasing):
    """Verify Sen's slope matches OLS on clean synthetic linear series."""
    ols_res = estimate_ols_trend(perfect_linear_increasing)
    sen_res = estimate_sens_slope(perfect_linear_increasing)

    assert isinstance(sen_res, SensSlopeResult)
    assert np.isclose(sen_res.slope_per_year, ols_res.slope_per_year, atol=1e-10)
    assert np.isclose(sen_res.slope_per_decade, ols_res.slope_per_decade, atol=1e-10)
    # Total pairwise slopes: 50 * 49 / 2 = 1225
    assert sen_res.pairwise_slopes_count == 1225


def test_sens_slope_robustness_to_outliers():
    """Verify Sen's slope is robust to extreme outliers that distort OLS."""
    years = list(range(1990, 2010))  # 20 years
    # Baseline slope: 0.05/year
    anomalies = [0.05 * (yr - 1990) for yr in years]
    # Add a huge single outlier at the endpoint (maximum leverage)
    anomalies[-1] = 50.0  # extreme spurious value
    df = pd.DataFrame({"year": years, "annual_mean": anomalies, "complete_year": True})

    sen_res = estimate_sens_slope(df)
    ols_res = estimate_ols_trend(df)

    # Sen's median slope should remain near 0.05
    assert np.isclose(sen_res.slope_per_year, 0.05, atol=0.01)
    # OLS is severely distorted
    assert ols_res.slope_per_year > 0.15


def test_sens_slope_nonconsecutive_years():
    """Verify Sen's slope correctly computes dt for nonconsecutive years."""
    # Data points: (1980, 0.0), (1990, 0.20), (2010, 0.60)
    # Slopes:
    # (1990 - 1980): dt = 10, dy = 0.20 -> 0.02
    # (2010 - 1990): dt = 20, dy = 0.40 -> 0.02
    # (2010 - 1980): dt = 30, dy = 0.60 -> 0.02
    # All pairwise slopes are exactly 0.02 °C/year -> 0.20 °C/decade
    df = pd.DataFrame(
        {
            "year": [1980, 1990, 2010],
            "annual_mean": [0.0, 0.20, 0.60],
            "complete_year": True,
        }
    )
    res = estimate_sens_slope(df)
    assert res.sample_count == 3
    assert np.isclose(res.slope_per_year, 0.02, atol=1e-10)
    assert np.isclose(res.slope_per_decade, 0.20, atol=1e-10)


# ---------------------------------------------------------------------------
# Input Validation and Safeguard Tests
# ---------------------------------------------------------------------------


def test_trend_methods_reject_duplicate_years():
    """Verify that duplicate years are rejected with a clear ValueError."""
    df_dup = pd.DataFrame(
        {
            "year": [1990, 1991, 1991, 1992],
            "annual_mean": [0.1, 0.2, 0.25, 0.3],
            "complete_year": True,
        }
    )
    with pytest.raises((TemporalIntegrityError, ValueError), match="[Dd]uplicate years"):
        estimate_ols_trend(df_dup)

    with pytest.raises((TemporalIntegrityError, ValueError), match="[Dd]uplicate years"):
        test_mann_kendall_trend(df_dup)

    with pytest.raises((TemporalIntegrityError, ValueError), match="[Dd]uplicate years"):
        estimate_sens_slope(df_dup)


def test_trend_methods_reject_insufficient_samples():
    """Verify that fewer than 3 valid samples raise a descriptive ValueError."""
    df_small = pd.DataFrame(
        {"year": [1990, 1991], "annual_mean": [0.1, 0.2], "complete_year": True}
    )
    with pytest.raises(ValueError, match="at least 3 valid"):
        estimate_ols_trend(df_small)

    with pytest.raises(ValueError, match="at least 3 valid"):
        test_mann_kendall_trend(df_small)

    with pytest.raises(ValueError, match="at least 3 valid"):
        estimate_sens_slope(df_small)


def test_trend_methods_missing_columns():
    """Verify SchemaValidationError when required columns are absent."""
    df_bad = pd.DataFrame({"temperature": [0.1, 0.2, 0.3]})
    with pytest.raises(SchemaValidationError, match="Required column"):
        estimate_ols_trend(df_bad)


def test_trend_methods_defensive_copies():
    """Verify that input DataFrames are not mutated or reordered externally."""
    years = [2005, 1995, 2000, 2010]  # unordered
    anomalies = [0.5, 0.3, 0.4, 0.6]
    df = pd.DataFrame({"year": years, "annual_mean": anomalies, "complete_year": True})
    orig_years = list(df["year"])

    _ = estimate_ols_trend(df)
    assert list(df["year"]) == orig_years


# ---------------------------------------------------------------------------
# Numerical Audit & Stability Tests (Phase 6 Audit)
# ---------------------------------------------------------------------------


def test_mann_kendall_p_value_large_z_no_underflow():
    """Verify that large absolute Z values compute accurate p-values without false zero underflow."""
    # 1. Zero Z-score: p-value must be exactly 1.0, log10(p) = 0.0
    df_const = pd.DataFrame(
        {"year": list(range(2000, 2010)), "annual_mean": [0.5] * 10, "complete_year": True}
    )
    res_zero = test_mann_kendall_trend(df_const)
    assert np.isclose(res_zero.z_statistic, 0.0)
    assert np.isclose(res_zero.p_value, 1.0)
    assert np.isclose(res_zero.log10_p_value, 0.0)

    # 2. Moderate Z-score: 50 observations with known moderate trend
    # Z should be positive, p_value in normal float range
    years_50 = list(range(1970, 2020))
    df_50 = pd.DataFrame(
        {"year": years_50, "annual_mean": [0.02 * yr for yr in range(50)], "complete_year": True}
    )
    res_50 = test_mann_kendall_trend(df_50)
    # Z is around ~10, p-value is extremely small but > 0.0
    assert res_50.z_statistic > 8.0
    assert res_50.p_value > 0.0
    assert not np.isnan(res_50.p_value)
    assert res_50.log10_p_value < -15.0

    # 3. Large Z-score (representing the actual 146-year GISTEMP record where Z ~ 13.02)
    # The previous buggy formula (1.0 - cdf) evaluated to literal 0.0.
    # The accurate survival function formulation must evaluate to ~9e-39.
    years_146 = list(range(1880, 2026))
    df_146 = pd.DataFrame(
        {"year": years_146, "annual_mean": [0.01 * yr for yr in range(146)], "complete_year": True}
    )
    res_146 = test_mann_kendall_trend(df_146)
    assert res_146.z_statistic > 12.0
    # Must NOT be literal 0.0
    assert res_146.p_value > 0.0
    assert res_146.p_value < 1e-30
    assert res_146.log10_p_value < -30.0
    # Check that summary text does not display a misleading '0.0000e+00'
    assert "0.0000e+00" not in res_146.summary()

    # 4. Extreme Z-score test (Z > 38 where 64-bit float reaches its representation limit)
    # Even if p_value reaches IEEE float limit, log10_p_value and summary must remain valid
    years_300 = list(range(1700, 2000))
    df_300 = pd.DataFrame(
        {"year": years_300, "annual_mean": [0.1 * yr for yr in range(300)], "complete_year": True}
    )
    res_300 = test_mann_kendall_trend(df_300)
    assert res_300.z_statistic > 20.0
    assert res_300.log10_p_value < -80.0
    assert not np.isnan(res_300.log10_p_value)
    assert not np.isinf(res_300.log10_p_value)


def test_sens_slope_manual_derivation_and_confidence_levels():
    """Verify Sen's slope and rank-based confidence bounds against manual derivation and SciPy."""
    from scipy import stats

    # Small 5-point dataset:
    # Points: (2000, 0.1), (2001, 0.3), (2002, 0.4), (2003, 0.8), (2004, 0.9)
    # 10 pairwise slopes:
    # (0.3-0.1)/1 = 0.20
    # (0.4-0.1)/2 = 0.15
    # (0.8-0.1)/3 = 0.23333333
    # (0.9-0.1)/4 = 0.20
    # (0.4-0.3)/1 = 0.10
    # (0.8-0.3)/2 = 0.25
    # (0.9-0.3)/3 = 0.20
    # (0.8-0.4)/1 = 0.40
    # (0.9-0.4)/2 = 0.25
    # (0.9-0.8)/1 = 0.10
    # Sorted 10 slopes: [0.10, 0.10, 0.15, 0.20, 0.20, 0.20, 0.23333333, 0.25, 0.25, 0.40]
    # Median is average of 5th and 6th = 0.20
    years = [2000, 2001, 2002, 2003, 2004]
    anomalies = [0.1, 0.3, 0.4, 0.8, 0.9]
    df = pd.DataFrame({"year": years, "annual_mean": anomalies, "complete_year": True})

    # Test multiple confidence levels: 90%, 95%, 99%
    for conf in [0.90, 0.95, 0.99]:
        res_ca = estimate_sens_slope(df, confidence_level=conf)
        res_scipy = stats.theilslopes(anomalies, years, alpha=conf)

        # Median slope must be exactly 0.20 °C/year -> 2.0 °C/decade
        assert np.isclose(res_ca.slope_per_year, 0.20, atol=1e-10)
        assert np.isclose(res_ca.slope_per_decade, 2.0, atol=1e-10)
        assert res_ca.pairwise_slopes_count == 10

        # Exact agreement with SciPy Theil-Sen reference implementation
        assert np.isclose(res_ca.slope_per_year, res_scipy.slope, atol=1e-10)
        assert np.isclose(res_ca.ci_lower_per_decade / 10.0, res_scipy.low_slope, atol=1e-10)
        assert np.isclose(res_ca.ci_upper_per_decade / 10.0, res_scipy.high_slope, atol=1e-10)

        # Confidence bounds must be monotonic: lower <= slope <= upper
        assert res_ca.ci_lower_per_decade <= res_ca.slope_per_decade <= res_ca.ci_upper_per_decade


def test_ols_residual_serial_dependence_diagnostics():
    """Verify that OLS computes residual serial dependence diagnostics (Durbin-Watson and r1)."""
    # 1. Noiseless linear series: residuals are identical to 0
    years = list(range(1980, 2000))
    df_clean = pd.DataFrame(
        {"year": years, "annual_mean": [0.05 * yr for yr in range(len(years))], "complete_year": True}
    )
    res_clean = estimate_ols_trend(df_clean)
    assert res_clean.durbin_watson is not None
    assert res_clean.residual_autocorr_lag1 is not None

    # 2. Synthetic series with strongly autocorrelated residuals:
    # y_t = 0.05 * t + AR(1) error where e_t = 0.8 * e_{t-1} + noise
    np.random.seed(42)
    n = 100
    noise = np.random.normal(0, 0.1, size=n)
    ar_error = np.zeros(n)
    for t in range(1, n):
        ar_error[t] = 0.85 * ar_error[t - 1] + noise[t]

    y_auto = 0.05 * np.arange(n) + ar_error
    df_auto = pd.DataFrame(
        {"year": list(range(1900, 1900 + n)), "annual_mean": y_auto, "complete_year": True}
    )
    res_auto = estimate_ols_trend(df_auto)

    # Autocorrelation must be strongly positive (r1 > 0.6) and Durbin-Watson must be low (d < 1.0)
    assert res_auto.residual_autocorr_lag1 > 0.60
    assert res_auto.durbin_watson < 1.0
    # Summary must report the diagnostics
    summary = res_auto.summary()
    assert "Residual Autocorr:" in summary
    assert "Durbin-Watson" in summary
