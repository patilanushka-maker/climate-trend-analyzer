# Data Governance & Storage Policy

This directory manages all input, intermediate, and processed datasets for the **Climate Trend Analyzer** project.

---

## 1. Directory Structure & Classification

* **`raw/` (`data/raw/`)**:
  * Original, unaltered data downloaded or extracted directly from authoritative sources.
  * **Immutability Policy**: Raw data files are strictly read-only once saved. No cleaning, column renaming, or value adjustments are performed directly on raw files.
* **`processed/` (`data/processed/`)**:
  * Cleaned, transformed, standardized, and feature-engineered datasets ready for analysis and visualization.
  * Every processed dataset must be 100% reproducible via executable data pipelines in `src/climate_analyzer/`.
* **`external/` (`data/external/`)**:
  * Ancillary or third-party reference data, such as regional boundary metadata, elevation lookups, or station coordinate cross-references.

---

## 2. Primary Dataset Specification: NASA GISTEMP v4

### 2.1 Metadata & Verified Provenance
* **Dataset Name**: GISS Surface Temperature Analysis (GISTEMP v4) — Land-Ocean Temperature Index (L-OTI)
* **Publishing Organization**: NASA Goddard Institute for Space Studies (GISS)
* **Official Webpage**: [https://data.giss.nasa.gov/gistemp/](https://data.giss.nasa.gov/gistemp/)
* **Direct Download URL**: [https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv](https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv)
* **Local Raw Storage Path**: `data/raw/GLB.Ts+dSST.csv`
* **File Size**: `12,886 bytes` (149 rows, 1880–2026)
* **SHA-256 Checksum**: `4c64eda288bce32b8d76926278f346ecea1e53506df8ee72a3d5accb0657c416`
* **Retrieval & Verification Date**: 2026-10-16 (verified 100% byte-for-byte identical to direct official download)
* **Input Data Sources**: NOAA GHCN v4 (meteorological land stations) and ERSST v5 (ocean areas, 1200 km smoothing).

### 2.2 Physical Units & Baseline
* **Physical Units**: Degrees Celsius (°C) temperature anomalies.
* **Baseline Period**: **1951–1980 base period** (deviations from the corresponding 1951–1980 climatological monthly means).
* **Observation Frequency**: Monthly, along with seasonal and annual summary aggregates.
* **Temporal Coverage**: January 1880 through September 2026 (updated monthly by NASA GISS).

### 2.3 Raw Schema & Missing-Value Conventions
* **Header Structure**:
  * Line 1: Dataset title banner (`Land-Ocean: Global Means`).
  * Line 2: Column headers (`Year,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec,J-D,D-N,DJF,MAM,JJA,SON`).
* **Columns (19 total)**:
  * `Year`: Integer calendar year (1880–present).
  * `Jan` through `Dec`: Monthly mean global surface temperature anomalies (°C).
  * `J-D`: Calendar year mean anomaly (January through December).
  * `D-N`: Meteorological year mean anomaly (December of previous year through November).
  * `DJF`, `MAM`, `JJA`, `SON`: Seasonal mean anomalies (Dec–Feb, Mar–May, Jun–Aug, Sep–Nov).
* **Missing Value Convention**:
  * Represented as `***` (triple asterisk) when observations are missing or pending (e.g. incomplete future months in the current year, or seasonal aggregates for years lacking predecessor December observations like 1880).

### 2.4 Licensing & Attribution
* **License**: Public Domain (NASA Open Data Policy / US Government Work).
* **Citations**:
  * GISTEMP Team, 2026: *GISS Surface Temperature Analysis (GISTEMP), version 4*. NASA Goddard Institute for Space Studies. Dataset accessed 2026-10-16 at [https://data.giss.nasa.gov/gistemp/](https://data.giss.nasa.gov/gistemp/).
  * Lenssen, N., G.A. Schmidt, M. Hendrickson, P. Jacobs, M. Menne, and R. Ruedy, 2024: *A GISTEMPv4 observational uncertainty ensemble*. J. Geophys. Res. Atmos., 129, no. 17, e2023JD040179, doi:10.1029/2023JD040179.

---

## 3. Data Validation & Conservative Cleaning Rules

### 3.1 Validation Rules (`src/climate_analyzer/data_validation.py`)
* **Hard Physical Bounds (`[-10.0°C, 10.0°C]`)**: Fatal error (`PhysicalBoundError`). Rationale: Global mean surface temperature cannot shift >10°C on a monthly basis without data corruption.
* **Soft Quality Warning Bounds (`[-3.0°C, 3.0°C]`)**: Non-fatal warning (`ValidationReport.warnings`). Observations outside this range are preserved as genuine extreme observations but flagged.
* **Temporal Integrity**: Asserts valid months (1–12), consistent `date.year == year` and `date.month == month`, strict chronological monotonicity, and detects internal gaps vs. expected trailing unobserved months.

### 3.2 Cleaning Rules (`src/climate_analyzer/data_cleaning.py`)
* **Immutability**: Raw DataFrame is defensively copied; raw data files are never modified.
* **No Zero-Filling**: Missing values are NEVER converted to zero or artificially interpolated.
* **No Silent Truncation**: Valid extreme observations are never removed solely because of high/low values.
* **Opt-In Operations**: Trimming trailing incomplete years (`trim_trailing_incomplete_year=True`) and dropping missing rows (`drop_na=True`) are strictly opt-in.
* **Storage Guardrail**: `save_processed_data` strictly refuses to write into `data/raw/`.
