"""
Conservative data cleaning for historical climate time series.

This module provides deterministic cleaning and standardization transformations
for climate observations. It adheres strictly to conservative data science
principles:
- Raw inputs are never modified in-place.
- Missing values are NEVER filled with zero or interpolated without user consent.
- Valid extreme observations are NEVER silently removed.
- Row filtering or trimming of trailing incomplete years is strictly explicit and opt-in.
- Processed outputs can be written to data/processed/ without risking raw source files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from climate_analyzer.data_validation import (
    REQUIRED_MONTHLY_COLUMNS,
    SchemaValidationError,
    validate_monthly_series,
)


def clean_monthly_gistemp(
    df: pd.DataFrame,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    trim_trailing_incomplete_year: bool = False,
    drop_na: bool = False,
    validate_input: bool = True,
) -> pd.DataFrame:
    """Clean and standardize a monthly climate anomaly DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame conforming to the monthly schema (year, month, date, anomaly).
    start_year : int, optional
        If specified, filters the series to start from this year (inclusive).
    end_year : int, optional
        If specified, filters the series to end at this year (inclusive).
    trim_trailing_incomplete_year : bool, default False
        If True, trims trailing unobserved (NaN) months at the tail of the series
        belonging to an incomplete final year. Does not discard historical years.
    drop_na : bool, default False
        Explicit opt-in: if True, removes rows where anomaly is NaN.
        If False (default), preserves missing observations with NaN to maintain temporal continuity.
    validate_input : bool, default True
        If True, validates the input with validate_monthly_series before transforming.

    Returns
    -------
    pd.DataFrame
        Cleaned, defensively copied DataFrame sorted strictly by date with columns:
        ['year', 'month', 'date', 'anomaly'].

    Raises
    ------
    SchemaValidationError
        If input DataFrame fails required column or structure checks.
    ValueError
        If start_year > end_year or year filtering produces an empty DataFrame.
    """
    if validate_input:
        validate_monthly_series(df, raise_on_error=True)

    # 1. Ensure defensive copy to guarantee raw input is never mutated
    clean_df = df.copy()

    # 2. Enforce clean types
    clean_df["year"] = clean_df["year"].astype(int)
    clean_df["month"] = clean_df["month"].astype(int)
    clean_df["date"] = pd.to_datetime(clean_df["date"])
    clean_df["anomaly"] = clean_df["anomaly"].astype("float64")

    # 3. Ensure strictly sorted chronologically
    clean_df = clean_df.sort_values("date").reset_index(drop=True)

    # 4. Optional Year Range Filtering
    if start_year is not None and end_year is not None and start_year > end_year:
        raise ValueError(
            f"start_year ({start_year}) cannot be greater than end_year ({end_year})."
        )

    if start_year is not None:
        clean_df = clean_df[clean_df["year"] >= start_year]
    if end_year is not None:
        clean_df = clean_df[clean_df["year"] <= end_year]

    if clean_df.empty:
        raise ValueError(
            f"Year filtering [{start_year}, {end_year}] resulted in an empty DataFrame."
        )

    clean_df = clean_df.reset_index(drop=True)

    # 5. Optional Trailing Incomplete Year Trimming
    if trim_trailing_incomplete_year and not clean_df.empty:
        latest_year = clean_df["year"].max()
        latest_year_mask = clean_df["year"] == latest_year
        # If the latest year has trailing NaNs, find the last valid observation index
        last_valid_idx = clean_df["anomaly"].last_valid_index()
        if last_valid_idx is not None and last_valid_idx < len(clean_df) - 1:
            # Only trim trailing NaNs that belong to the latest year
            trailing_nan_in_latest = clean_df.index[
                (clean_df.index > last_valid_idx) & latest_year_mask
            ]
            if not trailing_nan_in_latest.empty:
                clean_df = clean_df.drop(index=trailing_nan_in_latest).reset_index(drop=True)

    # 6. Optional Missing Value Dropping (Explicit Opt-In)
    if drop_na:
        clean_df = clean_df.dropna(subset=["anomaly"]).reset_index(drop=True)

    return clean_df[REQUIRED_MONTHLY_COLUMNS].copy()


def save_processed_data(
    df: pd.DataFrame,
    output_path: Union[str, Path] = "data/processed/gistemp_monthly_clean.csv",
    format: str = "csv",
) -> Path:
    """Save cleaned climate data to processed storage with directory and raw overwrite safeguards.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned climate DataFrame.
    output_path : str or Path, default 'data/processed/gistemp_monthly_clean.csv'
        Destination file path.
    format : str, default 'csv'
        Serialization format ('csv' or 'parquet').

    Returns
    -------
    Path
        Resolved path to the saved processed file.

    Raises
    ------
    ValueError
        If output_path points to data/raw/ (protects raw files from overwrite),
        or if format is unsupported.
    """
    dest = Path(output_path).resolve()

    # Safeguard: Prevent writing into raw directory
    dest_str_normalized = str(dest).replace("\\", "/").lower()
    if "/data/raw/" in dest_str_normalized or dest_str_normalized.endswith("/data/raw"):
        raise ValueError(
            f"Security Violation: Overwriting or saving processed data directly into 'data/raw/' is forbidden: {dest}"
        )

    # Ensure parent directory exists
    dest.parent.mkdir(parents=True, exist_ok=True)

    if format.lower() == "csv":
        df.to_csv(dest, index=False)
    elif format.lower() == "parquet":
        df.to_parquet(dest, index=False)
    else:
        raise ValueError(f"Unsupported storage format '{format}'. Supported formats: 'csv', 'parquet'.")

    return dest
