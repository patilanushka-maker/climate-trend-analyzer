# Root package initialization for climate_analyzer
"""
Climate Trend Analyzer: Core package for data loading, processing,
analysis, and visualization of historical climate series.
"""

from climate_analyzer.data_cleaning import (
    clean_monthly_gistemp,
    save_processed_data,
)
from climate_analyzer.data_loader import (
    DataLoaderError,
    InvalidFormatError,
    MissingDataFileError,
    ValidationError,
    load_gistemp_annual,
    load_gistemp_monthly,
    load_gistemp_raw,
)
from climate_analyzer.data_validation import (
    DEFAULT_HARD_PHYSICAL_BOUNDS,
    DEFAULT_WARNING_BOUNDS,
    DataValidationError,
    PhysicalBoundError,
    SchemaValidationError,
    TemporalIntegrityError,
    ValidationReport,
    validate_monthly_series,
)
from climate_analyzer.analysis import (
    DescriptiveStatisticsResult,
    MannKendallResult,
    OLSTrendResult,
    SensSlopeResult,
    calculate_decadal_summary,
    calculate_descriptive_statistics,
    calculate_monthly_climatology,
    calculate_warmth_percentiles,
    estimate_ols_trend,
    estimate_sens_slope,
    test_mann_kendall_trend,
)
from climate_analyzer.feature_engineering import (
    DEFAULT_ROLLING_WINDOWS,
    add_rolling_features,
    calculate_annual_summary,
    calculate_baseline_anomalies,
    calculate_year_over_year_change,
)
from climate_analyzer.visualization import (
    plot_annual_anomalies,
    plot_decadal_summary,
    plot_monthly_anomalies,
    plot_seasonal_patterns,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # Loader
    "DataLoaderError",
    "MissingDataFileError",
    "InvalidFormatError",
    "ValidationError",
    "load_gistemp_raw",
    "load_gistemp_monthly",
    "load_gistemp_annual",
    # Validation
    "DEFAULT_HARD_PHYSICAL_BOUNDS",
    "DEFAULT_WARNING_BOUNDS",
    "DataValidationError",
    "SchemaValidationError",
    "TemporalIntegrityError",
    "PhysicalBoundError",
    "ValidationReport",
    "validate_monthly_series",
    # Cleaning
    "clean_monthly_gistemp",
    "save_processed_data",
    # Feature Engineering
    "DEFAULT_ROLLING_WINDOWS",
    "add_rolling_features",
    "calculate_annual_summary",
    "calculate_baseline_anomalies",
    "calculate_year_over_year_change",
    # Analysis
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
    # Visualization
    "plot_monthly_anomalies",
    "plot_annual_anomalies",
    "plot_decadal_summary",
    "plot_seasonal_patterns",
]

