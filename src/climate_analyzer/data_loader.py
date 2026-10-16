"""
Data loading and initial schema validation for NASA GISTEMP climate records.

This module provides functions to ingest the official NASA GISTEMP Land-Ocean
Temperature Index (L-OTI) tabular datasets, parse missing-value markers,
validate required fields, check temporal integrity, and produce standardized
pandas DataFrames for downstream processing.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TextIO, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants & Specifications
# ---------------------------------------------------------------------------

GISTEMP_TITLE_MARKER = "Land-Ocean: Global Means"

MONTH_NAMES = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

AGGREGATE_COLUMNS = ["J-D", "D-N", "DJF", "MAM", "JJA", "SON"]

EXPECTED_RAW_COLUMNS = ["Year"] + MONTH_NAMES + AGGREGATE_COLUMNS

MONTH_NAME_TO_INT = {name: i + 1 for i, name in enumerate(MONTH_NAMES)}

MISSING_VALUE_MARKERS = ["***", "****", "NaN", "nan", ""]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DataLoaderError(Exception):
    """Base exception for all data loader failures."""


class MissingDataFileError(DataLoaderError, FileNotFoundError):
    """Raised when the specified climate data file does not exist."""


class InvalidFormatError(DataLoaderError, ValueError):
    """Raised when the file structure, header, or columns violate expected schema."""


class ValidationError(DataLoaderError, ValueError):
    """Raised when data values, types, or temporal consistency checks fail."""


# ---------------------------------------------------------------------------
# Core Loading Subroutines
# ---------------------------------------------------------------------------


def _find_header_row(lines: list[str]) -> int:
    """Locate the 0-indexed line containing the column header (Year,Jan,...).

    Parameters
    ----------
    lines : list[str]
        List of text lines from the data source.

    Returns
    -------
    int
        The line index of the header row.

    Raises
    ------
    InvalidFormatError
        If no recognizable header row is found.
    """
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("Year") and "Jan" in stripped:
            return idx
    raise InvalidFormatError(
        "Could not locate the CSV column header row starting with 'Year'."
    )


def load_gistemp_raw(
    filepath_or_buffer: Union[str, Path, TextIO],
) -> pd.DataFrame:
    """Load and perform structural validation on the raw NASA GISTEMP CSV.

    Parameters
    ----------
    filepath_or_buffer : str, Path, or TextIO
        Filepath or file-like buffer pointing to the GISTEMP CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Year', 'Jan', ..., 'Dec', 'J-D', 'D-N', 'DJF', 'MAM', 'JJA', 'SON'],
        where 'Year' is integer, missing values are represented as NaN, and all anomaly
        columns are float64.

    Raises
    ------
    MissingDataFileError
        If a provided filepath does not exist.
    InvalidFormatError
        If the file has missing required columns or malformed header structure.
    ValidationError
        If year records are duplicated, invalid, or numeric conversion fails.
    """
    # 1. Resolve and read raw text lines
    if isinstance(filepath_or_buffer, (str, Path)):
        path = Path(filepath_or_buffer)
        if not path.is_file():
            raise MissingDataFileError(
                f"Climate dataset file not found at: {path.resolve()}"
            )
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as exc:
            raise DataLoaderError(f"Failed to read file {path}: {exc}") from exc
    elif hasattr(filepath_or_buffer, "read"):
        content = filepath_or_buffer.read()
    else:
        raise InvalidFormatError(
            f"Unsupported input type for filepath_or_buffer: {type(filepath_or_buffer)}"
        )

    lines = [line for line in content.splitlines() if line.strip()]
    if not lines:
        raise InvalidFormatError("The provided climate data source is empty.")

    # 2. Determine header location
    header_idx = _find_header_row(lines)

    # 3. Parse with pandas
    data_buffer = io.StringIO("\n".join(lines[header_idx:]))
    try:
        df = pd.read_csv(
            data_buffer,
            na_values=MISSING_VALUE_MARKERS,
            keep_default_na=True,
            dtype=str,  # Load as string initially for deterministic validation
        )
    except Exception as exc:
        raise InvalidFormatError(f"CSV parsing error: {exc}") from exc

    # Clean column names
    df.columns = df.columns.str.strip()

    # 4. Validate presence of required fields
    missing_cols = [c for c in EXPECTED_RAW_COLUMNS if c not in df.columns]
    if missing_cols:
        raise InvalidFormatError(
            f"Dataset is missing required columns: {missing_cols}. "
            f"Expected: {EXPECTED_RAW_COLUMNS}"
        )

    # Filter down to the expected 19 columns
    df = df[EXPECTED_RAW_COLUMNS].copy()

    # Drop any trailing repeated header rows that NASA occasionally embeds in text tables
    df = df[df["Year"].str.strip() != "Year"].copy()

    if df.empty:
        raise ValidationError("No data records found in dataset after header parsing.")

    # 5. Validate and convert 'Year'
    df["Year"] = df["Year"].str.strip()
    try:
        df["Year"] = pd.to_numeric(df["Year"], errors="raise").astype(int)
    except Exception as exc:
        raise ValidationError(
            f"Invalid non-integer year values detected in 'Year' column: {exc}"
        ) from exc

    # Policy: Duplicate years are strictly rejected to prevent ambiguous observations
    duplicate_years = df[df["Year"].duplicated(keep=False)]["Year"].unique().tolist()
    if duplicate_years:
        raise ValidationError(
            f"Duplicate year records detected for years: {duplicate_years}. "
            "Policy rejects ambiguous duplicate records."
        )

    # Check reasonable temporal range
    if (df["Year"] < 1700).any() or (df["Year"] > 2200).any():
        outlier_years = df[
            (df["Year"] < 1700) | (df["Year"] > 2200)
        ]["Year"].tolist()
        raise ValidationError(
            f"Unreasonable year values detected outside [1700, 2200]: {outlier_years}"
        )

    # 6. Validate and convert numerical anomaly columns
    anomaly_cols = MONTH_NAMES + AGGREGATE_COLUMNS
    for col in anomaly_cols:
        col_series = df[col].astype(str).str.strip()
        is_missing = col_series.isin(MISSING_VALUE_MARKERS)
        col_series = col_series.mask(is_missing, np.nan)
        try:
            df[col] = pd.to_numeric(col_series, errors="raise").astype("float64")
        except Exception as exc:
            raise ValidationError(
                f"Unparseable non-numeric anomaly value in column '{col}': {exc}"
            ) from exc

    df = df.sort_values("Year").reset_index(drop=True)
    return df


def load_gistemp_monthly(
    filepath_or_buffer: Union[str, Path, TextIO],
    drop_na: bool = False,
) -> pd.DataFrame:
    """Load NASA GISTEMP data transformed into a tidy, chronological monthly series.

    Parameters
    ----------
    filepath_or_buffer : str, Path, or TextIO
        Filepath or file-like buffer pointing to the GISTEMP CSV.
    drop_na : bool, default False
        If True, drops months with missing (NaN) anomaly values.
        If False, preserves all calendar months to maintain temporal continuity.

    Returns
    -------
    pd.DataFrame
        Tidy DataFrame with columns:
        - 'year' (int): Calendar year (e.g. 1880, 2026)
        - 'month' (int): Month index (1 to 12)
        - 'date' (pd.Timestamp): First day of month (YYYY-MM-01)
        - 'anomaly' (float): Land-Ocean temperature anomaly in °C vs 1951-1980 baseline.

    Raises
    ------
    DataLoaderError
        If reading, formatting, or validation fails.
    """
    raw_df = load_gistemp_raw(filepath_or_buffer)

    records = []
    for _, row in raw_df.iterrows():
        year = int(row["Year"])
        for month_name in MONTH_NAMES:
            month_num = MONTH_NAME_TO_INT[month_name]
            anomaly = row[month_name]
            # Construct first day of the month timestamp
            timestamp = pd.Timestamp(year=year, month=month_num, day=1)
            records.append(
                {
                    "year": year,
                    "month": month_num,
                    "date": timestamp,
                    "anomaly": anomaly,
                }
            )

    monthly_df = pd.DataFrame(records)
    monthly_df["year"] = monthly_df["year"].astype(int)
    monthly_df["month"] = monthly_df["month"].astype(int)
    monthly_df["anomaly"] = monthly_df["anomaly"].astype("float64")

    # Sort strictly chronologically
    monthly_df = monthly_df.sort_values("date").reset_index(drop=True)

    if drop_na:
        monthly_df = monthly_df.dropna(subset=["anomaly"]).reset_index(drop=True)

    return monthly_df


def load_gistemp_annual(
    filepath_or_buffer: Union[str, Path, TextIO],
) -> pd.DataFrame:
    """Load NASA GISTEMP data focused on annual and seasonal summary aggregates.

    Parameters
    ----------
    filepath_or_buffer : str, Path, or TextIO
        Filepath or file-like buffer pointing to the GISTEMP CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame containing:
        - 'year' (int): Calendar year
        - 'calendar_year_anomaly' (float): J-D mean anomaly
        - 'meteorological_year_anomaly' (float): D-N mean anomaly
        - 'djf_anomaly' (float): Dec-Jan-Feb seasonal anomaly
        - 'mam_anomaly' (float): Mar-Apr-May seasonal anomaly
        - 'jja_anomaly' (float): Jun-Jul-Aug seasonal anomaly
        - 'son_anomaly' (float): Sep-Oct-Nov seasonal anomaly
    """
    raw_df = load_gistemp_raw(filepath_or_buffer)

    annual_df = pd.DataFrame(
        {
            "year": raw_df["Year"].astype(int),
            "calendar_year_anomaly": raw_df["J-D"].astype("float64"),
            "meteorological_year_anomaly": raw_df["D-N"].astype("float64"),
            "djf_anomaly": raw_df["DJF"].astype("float64"),
            "mam_anomaly": raw_df["MAM"].astype("float64"),
            "jja_anomaly": raw_df["JJA"].astype("float64"),
            "son_anomaly": raw_df["SON"].astype("float64"),
        }
    )
    return annual_df.sort_values("year").reset_index(drop=True)
