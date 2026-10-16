"""
Analytical subroutines: descriptive distributions and statistical trend estimation.
"""

from climate_analyzer.analysis.descriptive_analysis import (
    DescriptiveStatisticsResult,
    calculate_decadal_summary,
    calculate_descriptive_statistics,
    calculate_monthly_climatology,
    calculate_warmth_percentiles,
)
from climate_analyzer.analysis.trend_analysis import (
    MannKendallResult,
    OLSTrendResult,
    SensSlopeResult,
    estimate_ols_trend,
    estimate_sens_slope,
    test_mann_kendall_trend,
)

__all__ = [
    "DescriptiveStatisticsResult",
    "calculate_descriptive_statistics",
    "calculate_decadal_summary",
    "calculate_warmth_percentiles",
    "calculate_monthly_climatology",
    "OLSTrendResult",
    "MannKendallResult",
    "SensSlopeResult",
    "estimate_ols_trend",
    "test_mann_kendall_trend",
    "estimate_sens_slope",
]

