"""
Data validation layer for historical climate observations.

This module provides validation routines for historical climate time series,
checking schema integrity, temporal continuity, calendar consistency, and
distinguishing genuine data-integrity failures from observational quality warnings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Default Constants & Rationales
# ---------------------------------------------------------------------------

REQUIRED_MONTHLY_COLUMNS = ["year", "month", "date", "anomaly"]

# RATIONALE FOR HARD PHYSICAL BOUNDS (-10.0°C to +10.0°C):
# The dataset records global monthly mean surface temperature anomalies relative
# to the 1951-1980 baseline. Globally averaged monthly temperatures in the
# instrumental era (1880-present) fluctuate within approximately [-1.5°C, +2.5°C].
# A global mean monthly anomaly exceeding +/-10.0°C represents gross corruption
# (such as an unparsed sentinel value, bit flip, or column offset), as Earth's
# global mean surface climate cannot experience a 10°C shift in a single month.
DEFAULT_HARD_PHYSICAL_BOUNDS: tuple[float, float] = (-10.0, 10.0)

# RATIONALE FOR SOFT QUALITY WARNING BOUNDS (-3.0°C to +3.0°C):
# Monthly global anomalies outside [-3.0°C, +3.0°C] are historically unprecedented
# in the 1880-present GISTEMP record and warrant inspection, but do not
# necessarily indicate unrecoverable file corruption.
DEFAULT_WARNING_BOUNDS: tuple[float, float] = (-3.0, 3.0)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DataValidationError(Exception):
    """Base exception for climate data validation failures."""


class SchemaValidationError(DataValidationError, ValueError):
    """Raised when columns or data types do not conform to the expected schema."""


class TemporalIntegrityError(DataValidationError, ValueError):
    """Raised when timestamps, month indices, or sequence continuity are invalid."""


class PhysicalBoundError(DataValidationError, ValueError):
    """Raised when anomaly values violate physical sanity limits for global climate."""


# ---------------------------------------------------------------------------
# Structured Validation Report
# ---------------------------------------------------------------------------


@dataclass
class ValidationReport:
    """Structured report summarizing data validation results and quality metrics.

    Attributes
    ----------
    is_valid : bool
        True if all structural and integrity checks passed.
    total_records : int
        Total number of rows in the series.
    valid_anomalies : int
        Count of valid (non-null) numeric anomaly values.
    missing_anomalies : int
        Count of missing (NaN) anomaly values.
    completeness_pct : float
        Percentage of rows with valid observations (0.0 to 100.0).
    start_date : pd.Timestamp or None
        Earliest observation date in the series.
    end_date : pd.Timestamp or None
        Latest observation date in the series.
    internal_gaps : list[pd.Timestamp]
        List of missing observation dates that occur between valid records.
    trailing_unobserved : list[pd.Timestamp]
        List of expected future months at the end of the current incomplete year.
    warnings : list[str]
        List of non-fatal quality warnings (e.g. unusual values, missing months).
    errors : list[str]
        List of fatal data integrity errors detected.
    out_of_bounds_records : list[dict]
        Details of observations exceeding quality warning thresholds.
    """

    is_valid: bool = True
    total_records: int = 0
    valid_anomalies: int = 0
    missing_anomalies: int = 0
    completeness_pct: float = 0.0
    start_date: pd.Timestamp | None = None
    end_date: pd.Timestamp | None = None
    observed_start_date: pd.Timestamp | None = None
    observed_end_date: pd.Timestamp | None = None
    internal_gaps: list[pd.Timestamp] = field(default_factory=list)
    trailing_unobserved: list[pd.Timestamp] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    out_of_bounds_records: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        """Produce a human-readable text summary of the validation report."""
        status = "PASSED" if self.is_valid else "FAILED"
        lines = [
            f"Validation Status: {status}",
            f"Records: {self.total_records} total | {self.valid_anomalies} valid | {self.missing_anomalies} missing ({self.completeness_pct:.2f}% complete)",
            f"Calendar Span: {self.start_date.strftime('%Y-%m') if self.start_date else 'N/A'} to {self.end_date.strftime('%Y-%m') if self.end_date else 'N/A'}",
            f"Observed Span: {self.observed_start_date.strftime('%Y-%m') if self.observed_start_date else 'N/A'} to {self.observed_end_date.strftime('%Y-%m') if self.observed_end_date else 'N/A'}",
            f"Internal Missing Gaps: {len(self.internal_gaps)}",
            f"Trailing Pending Months: {len(self.trailing_unobserved)}",
            f"Quality Warnings: {len(self.warnings)}",
            f"Errors: {len(self.errors)}",
        ]
        if self.errors:
            lines.append("Errors Encountered:")
            for err in self.errors:
                lines.append(f"  - [ERROR] {err}")
        if self.warnings:
            lines.append("Warnings Encountered:")
            for warn in self.warnings:
                lines.append(f"  - [WARNING] {warn}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation Functions
# ---------------------------------------------------------------------------


def validate_monthly_series(
    df: pd.DataFrame,
    warning_bounds: tuple[float, float] = DEFAULT_WARNING_BOUNDS,
    hard_bounds: tuple[float, float] = DEFAULT_HARD_PHYSICAL_BOUNDS,
    raise_on_error: bool = True,
) -> ValidationReport:
    """Validate a tidy monthly climate series DataFrame.

    Checks:
    1. Schema and columns (year, month, date, anomaly) and datatypes.
    2. Temporal integrity: month range [1, 12], consistency between date and (year, month).
    3. Duplicate observations: duplicate timestamps or (year, month) pairs.
    4. Chronological monotonicity: strictly increasing date order.
    5. Sequence completeness: detects internal missing gaps vs. trailing unobserved months.
    6. Physical sanity bounds: raises PhysicalBoundError for values outside hard_bounds.
    7. Quality warning thresholds: records warnings for values outside warning_bounds.

    Parameters
    ----------
    df : pd.DataFrame
        Tidy monthly DataFrame, typically from load_gistemp_monthly().
    warning_bounds : tuple[float, float], default (-3.0, 3.0)
        (min, max) expected anomaly range in °C. Observations outside trigger warnings.
    hard_bounds : tuple[float, float], default (-10.0, 10.0)
        (min, max) physical sanity bounds in °C. Observations outside trigger errors.
    raise_on_error : bool, default True
        If True, raises a DataValidationError subclass immediately upon the first fatal error.
        If False, records errors in the report and sets report.is_valid = False.

    Returns
    -------
    ValidationReport
        Comprehensive report containing metrics, warnings, and error diagnostics.

    Raises
    ------
    SchemaValidationError
        If required columns or data types are invalid.
    TemporalIntegrityError
        If timestamps are duplicated, non-monotonic, or inconsistent with calendar months.
    PhysicalBoundError
        If anomaly values violate hard physical limits.
    """
    report = ValidationReport()

    # --- 1. Basic Type & DataFrame Check ---
    if not isinstance(df, pd.DataFrame):
        msg = f"Expected pandas DataFrame, got {type(df).__name__}."
        report.errors.append(msg)
        report.is_valid = False
        if raise_on_error:
            raise SchemaValidationError(msg)
        return report

    report.total_records = len(df)
    if df.empty:
        msg = "The provided climate dataset is empty (0 records)."
        report.errors.append(msg)
        report.is_valid = False
        if raise_on_error:
            raise SchemaValidationError(msg)
        return report

    # --- 2. Schema and Column Presence ---
    missing_cols = [c for c in REQUIRED_MONTHLY_COLUMNS if c not in df.columns]
    if missing_cols:
        msg = f"Missing required columns: {missing_cols}. Expected: {REQUIRED_MONTHLY_COLUMNS}"
        report.errors.append(msg)
        report.is_valid = False
        if raise_on_error:
            raise SchemaValidationError(msg)
        return report

    # --- 3. Data Types Check ---
    if not pd.api.types.is_integer_dtype(df["year"]):
        msg = f"Column 'year' must be integer, got {df['year'].dtype}."
        report.errors.append(msg)
        report.is_valid = False
        if raise_on_error:
            raise SchemaValidationError(msg)

    if not pd.api.types.is_integer_dtype(df["month"]):
        msg = f"Column 'month' must be integer, got {df['month'].dtype}."
        report.errors.append(msg)
        report.is_valid = False
        if raise_on_error:
            raise SchemaValidationError(msg)

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        msg = f"Column 'date' must be datetime64, got {df['date'].dtype}."
        report.errors.append(msg)
        report.is_valid = False
        if raise_on_error:
            raise SchemaValidationError(msg)

    if not pd.api.types.is_float_dtype(df["anomaly"]):
        msg = f"Column 'anomaly' must be float64, got {df['anomaly'].dtype}."
        report.errors.append(msg)
        report.is_valid = False
        if raise_on_error:
            raise SchemaValidationError(msg)

    # --- 4. Month Values & Calendar Consistency ---
    invalid_months = df[(df["month"] < 1) | (df["month"] > 12)]["month"].tolist()
    if invalid_months:
        msg = f"Invalid month values detected outside [1, 12]: {invalid_months}."
        report.errors.append(msg)
        report.is_valid = False
        if raise_on_error:
            raise TemporalIntegrityError(msg)

    # Consistency: date.year == year and date.month == month and date.day == 1
    mismatched_dates = df[
        (df["date"].dt.year != df["year"])
        | (df["date"].dt.month != df["month"])
        | (df["date"].dt.day != 1)
    ]
    if not mismatched_dates.empty:
        sample_bad = mismatched_dates[["year", "month", "date"]].head(3).to_dict("records")
        msg = f"Inconsistency between 'date' timestamp and (year, month) columns. Sample: {sample_bad}."
        report.errors.append(msg)
        report.is_valid = False
        if raise_on_error:
            raise TemporalIntegrityError(msg)

    # --- 5. Duplicate Observations ---
    dup_mask = df.duplicated(subset=["date"], keep=False)
    if dup_mask.any():
        dup_dates = df.loc[dup_mask, "date"].unique().tolist()
        msg = f"Duplicate timestamp records detected: {len(dup_dates)} duplicates found ({dup_dates[:3]}...)."
        report.errors.append(msg)
        report.is_valid = False
        if raise_on_error:
            raise TemporalIntegrityError(msg)

    # --- 6. Chronological Monotonicity ---
    if not df["date"].is_monotonic_increasing:
        msg = "Dates are not monotonically strictly increasing. Series must be chronologically ordered."
        report.errors.append(msg)
        report.is_valid = False
        if raise_on_error:
            raise TemporalIntegrityError(msg)

    # --- 7. Temporal Bounds & Continuity ---
    report.start_date = df["date"].min()
    report.end_date = df["date"].max()

    # Expected full monthly grid between start and end
    expected_range = pd.date_range(start=report.start_date, end=report.end_date, freq="MS")
    if len(expected_range) != len(df):
        missing_in_grid = set(expected_range) - set(df["date"])
        if missing_in_grid:
            msg = f"Missing timestamps in calendar grid: {len(missing_in_grid)} monthly dates omitted from series."
            report.warnings.append(msg)

    # --- 8. Missing Anomaly Values & Trailing Unobserved Distinction ---
    nan_mask = df["anomaly"].isna()
    report.missing_anomalies = int(nan_mask.sum())
    report.valid_anomalies = report.total_records - report.missing_anomalies
    report.completeness_pct = (report.valid_anomalies / report.total_records) * 100.0

    valid_dates = df.loc[~nan_mask, "date"]
    if not valid_dates.empty:
        report.observed_start_date = valid_dates.min()
        report.observed_end_date = valid_dates.max()
    else:
        report.observed_start_date = None
        report.observed_end_date = None

    if report.missing_anomalies > 0:
        # Determine last valid observation position
        valid_positions = np.where(~nan_mask.values)[0]
        if len(valid_positions) > 0 and valid_positions[-1] < len(df) - 1:
            last_pos = valid_positions[-1]
            report.trailing_unobserved = df["date"].iloc[last_pos + 1 :].tolist()
            # Any missing dates prior to last_pos are internal gaps
            internal_mask = nan_mask.values.copy()
            internal_mask[last_pos + 1 :] = False
            report.internal_gaps = df["date"].iloc[internal_mask].tolist()
        else:
            report.internal_gaps = df.loc[nan_mask, "date"].tolist()
            report.trailing_unobserved = []

        if report.internal_gaps:
            report.warnings.append(
                f"Detected {len(report.internal_gaps)} internal missing anomaly observations within historical record."
            )
        if report.trailing_unobserved:
            report.warnings.append(
                f"Detected {len(report.trailing_unobserved)} expected pending months at the end of the time series."
            )

    # --- 9. Physical Sanity Bounds (Hard Limits) ---
    valid_anomalies = df.loc[~nan_mask, "anomaly"]
    hard_min, hard_max = hard_bounds
    extreme_hard_mask = (valid_anomalies < hard_min) | (valid_anomalies > hard_max)
    if extreme_hard_mask.any():
        violations = df.loc[extreme_hard_mask.index[extreme_hard_mask], ["year", "month", "anomaly"]].to_dict("records")
        msg = (
            f"Anomalies outside hard physical sanity limits [{hard_min}°C, {hard_max}°C] detected. "
            f"Violations: {violations}. "
            "Rationale: Global mean surface anomalies cannot shift >10°C on a monthly basis without data corruption."
        )
        report.errors.append(msg)
        report.is_valid = False
        if raise_on_error:
            raise PhysicalBoundError(msg)

    # --- 10. Quality Warning Bounds (Soft Limits) ---
    warn_min, warn_max = warning_bounds
    outside_warn_mask = (valid_anomalies < warn_min) | (valid_anomalies > warn_max)
    if outside_warn_mask.any():
        warn_records = df.loc[outside_warn_mask.index[outside_warn_mask], ["year", "month", "anomaly"]].to_dict("records")
        report.out_of_bounds_records = warn_records
        report.warnings.append(
            f"{len(warn_records)} observations exceed soft quality warning bounds [{warn_min}°C, {warn_max}°C]. "
            "These observations are preserved as valid historical data but flagged for user inspection."
        )

    return report
