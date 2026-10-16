# Implementation & Execution Log

This log tracks every implementation step, technical decision, and validation procedure across the project lifecycle.

---

## Log Entry: 2026-10-16 — Phase 1: Project Scaffolding & Repository Governance Setup

### 1. Workspace Inspection
* **Project Root:** `D:\Anushka\IIT\DS\Climate Trend Analyzer`
* **Python Runtime:** Python 3.10.8 (Windows)
* **Git Repository Status:** No `.git/` repository was initialized prior to this step.
* **Pre-existing Artifacts:** Found prior documentation structure in [`docs/`](file:///d:/Anushka/IIT/DS/Climate%20Trend%20Analyzer/docs) (`README.md`, `architecture.md`, `data_pipeline.md`, `model_details.md`, `execution_log.md`). All preserved.

### 2. Implementation Summary
* **Directory Scaffolding Established:**
  * `data/raw/`, `data/processed/`, `data/external/`
  * `notebooks/`
  * `src/climate_analyzer/`, `src/climate_analyzer/analysis/`, `src/climate_analyzer/visualization/`
  * `app/components/`
  * `tests/`
  * `outputs/figures/`, `outputs/tables/`, `outputs/summaries/`
  * `reports/`
  * `images/`
  * `scripts/`
* **Package Initialization:**
  * Added `__init__.py` to `src/climate_analyzer/`, `src/climate_analyzer/analysis/`, `src/climate_analyzer/visualization/`, and `app/components/`.
* **Root Configuration & Governance Files:**
  * [`README.md`](file:///d:/Anushka/IIT/DS/Climate%20Trend%20Analyzer/README.md): High-level overview, architecture overview, setup guide, testing protocol, and planned module roadmap.
  * [`data/README.md`](file:///d:/Anushka/IIT/DS/Climate%20Trend%20Analyzer/data/README.md): Immutability rules, metadata attribution requirements, and storage policies.
  * [`.gitignore`](file:///d:/Anushka/IIT/DS/Climate%20Trend%20Analyzer/.gitignore): Exclusions for Python bytecode, virtual environments, local secrets, Jupyter checkpoints, test artifacts, OS files, and large data/outputs while preserving directory tracking with `.gitkeep`.
  * [`requirements.txt`](file:///d:/Anushka/IIT/DS/Climate%20Trend%20Analyzer/requirements.txt): Runtime dependencies (`pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `plotly`, `scikit-learn`, `statsmodels`, `streamlit`, `jupyter`).
  * [`requirements-dev.txt`](file:///d:/Anushka/IIT/DS/Climate%20Trend%20Analyzer/requirements-dev.txt): Development/testing stack (`pytest`, `pytest-cov`, `black`, `flake8`).
  * [`pyproject.toml`](file:///d:/Anushka/IIT/DS/Climate%20Trend%20Analyzer/pyproject.toml): Build system metadata, package auto-discovery in `src/`, and pytest configuration.
  * [`.env.example`](file:///d:/Anushka/IIT/DS/Climate%20Trend%20Analyzer/.env.example): Environment variable template.

---

## Log Entry: 2026-10-16 — Phase 2: Repository Audit & Isolated Packaging Verification

### 1. Audit & Verification Summary
* **Isolated Environment Creation:** Created project-local virtual environment `.venv/` using Python `3.10.8`.
* **Dependency Installation:** Installed all packages from `requirements-dev.txt` (which inherits `requirements.txt`) into `.venv/`.
  * Pip Version: `pip 22.2.2` (upgradable to `26.2.1`).
  * All 14 core dependencies and transitive packages installed with status code 0.
* **Editable Package Installation:** Installed `climate-trend-analyzer` via `pip install -e .`.
  * Build backend: `setuptools.build_meta` via `pyproject.toml`.
  * Package discovered: `src/climate_analyzer` and subpackages (`analysis`, `visualization`).
  * Installation status: Successful (Status code 0).
* **Package Import Validation:** Executed live Python validation script:
  * Successfully imported `climate_analyzer` (`__version__ = '0.1.0'`).
  * Origin verified at `D:\Anushka\IIT\DS\Climate Trend Analyzer\src\climate_analyzer\__init__.py`.
  * Subpackages `climate_analyzer.analysis` and `climate_analyzer.visualization` confirmed importable.
* **Testing Runner Validation:** Executed `pytest` via `.venv\Scripts\pytest.exe`. Pytest discovered configuration from `pyproject.toml`, test paths set to `tests/`, and correctly completed collection (0 test items collected, as planned analytical tests have not yet been written).
* **Git Ignore Audit:**
  * Confirmed that `git check-ignore` cannot be run natively because Git has not been initialized (`fatal: not a git repository`).
  * Audited `.gitignore` rules manually: confirmed comprehensive exclusion of `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`, `.env`, and data/output files while tracking empty directory structures via `!.gitkeep`.
  * Verified no unintentional exclusion of `src/`, `docs/`, `tests/`, or config files.
* **Documentation Synchronization:** Refined [`docs/model_details.md`](file:///d:/Anushka/IIT/DS/Climate%20Trend%20Analyzer/docs/model_details.md) to align with actual project scope (trend analysis, STL decomposition, anomaly Z-scores/IQR) and clarified that observational statistical trends do not substitute for GCMs.

---

## Log Entry: 2026-10-16 — Phase 3: Dataset Ingestion & Automated Test Suite Implementation

### 1. Git Initialization & Verification
* Initialized empty Git repository (`git init`) and renamed initial branch to `main`.
* Verified Git ignore exclusions via `git check-ignore -v` on representative targets:
  * Excluded: `.venv`, `.env`, `data/raw/GLB.Ts+dSST.csv`, `outputs/figures/test.png`, `reports/test.pdf`, `.pytest_cache/`.
  * Preserved/Trackable: `.env.example`, `data/raw/.gitkeep`, `data/processed/.gitkeep`, `outputs/figures/.gitkeep`.

### 2. Dataset Selection & Official Verification
* **Source:** NASA GISS Surface Temperature Analysis (GISTEMP v4).
* **Official Webpage:** `https://data.giss.nasa.gov/gistemp/`
* **Direct File:** `https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv`
* **Schema Confirmed:** Title banner on line 1, header on line 2, 19 columns: `Year`, 12 months (`Jan`-`Dec`), 2 annual means (`J-D`, `D-N`), 4 seasonal averages (`DJF`, `MAM`, `JJA`, `SON`).
* **Baseline Period:** 1951–1980 base period.
* **Units:** Degrees Celsius (°C) anomaly.
* **Missing Value Marker:** `***` (e.g., upcoming months in current year).
* Saved raw dataset to `data/raw/GLB.Ts+dSST.csv` (1880–2026, 147 years). Confirmed ignored by Git.

### 3. Data Loader Implementation (`src/climate_analyzer/data_loader.py`)
* Implemented `load_gistemp_raw(filepath_or_buffer)`: Loads raw table, strips banner, parses `***` as `np.nan`, validates integer years, strictly checks against duplicate years, and ensures numeric float64 anomaly types.
* Implemented `load_gistemp_monthly(filepath_or_buffer, drop_na=False)`: Reshapes monthly columns into tidy chronological time series with columns `[year, month, date, anomaly]`.
* Implemented `load_gistemp_annual(filepath_or_buffer)`: Extracts annual and seasonal mean anomalies.
* Exported API in `src/climate_analyzer/__init__.py`.

---

## Log Entry: 2026-10-16 — Phase 4: Raw-Data Verification, Data Validation & Conservative Cleaning

### 1. Raw-Data Integrity & Checksum Verification
* Downloaded official CSV directly from `https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv` into a temporary staging file.
* Byte-for-byte comparison against `data/raw/GLB.Ts+dSST.csv`:
  * Size: `12,886 bytes`
  * SHA-256 Checksum: `4c64eda288bce32b8d76926278f346ecea1e53506df8ee72a3d5accb0657c416`
  * Result: **100% identical exact match**. Raw data confirmed unaltered.
* Removed temporary staging download.

### 2. Data Validation Layer (`src/climate_analyzer/data_validation.py`)
* Implemented `validate_monthly_series()` returning a structured `ValidationReport`.
* Distinguishes fatal data integrity failures (`SchemaValidationError`, `TemporalIntegrityError`, `PhysicalBoundError`) from quality warnings:
  * Hard physical sanity limit: `[-10.0°C, 10.0°C]`. Rationale: Global mean surface temperature cannot shift >10°C in a single month without gross data corruption.
  * Soft quality warning bounds: `[-3.0°C, 3.0°C]`. Historical extremes are flagged for inspection but never removed.
  * Temporal integrity: checks month index `[1, 12]`, checks that `date` matches `(year, month)` and `day=1`, asserts strict chronological monotonicity, and checks for duplicate dates.
  * Missing observation classification: distinguishes internal sequence gaps from expected trailing unobserved months of an incomplete current year.
  * Updated position-based index masking to support custom and sliced non-consecutive indexes robustly.

### 3. Conservative Data Cleaning Layer (`src/climate_analyzer/data_cleaning.py`)
* Implemented `clean_monthly_gistemp()`:
  * Raw inputs are defensively copied and never mutated.
  * Enforces clean types and strict chronological sorting.
  * Supports deterministic `start_year` and `end_year` filtering.
  * Missing observations are NEVER converted to zero or interpolated.
  * Trimming trailing incomplete years (`trim_trailing_incomplete_year=True`) and dropping missing rows (`drop_na=True`) are strictly opt-in.
* Implemented `save_processed_data()`:
  * Saves cleaned DataFrames to `data/processed/`.
  * Security guardrail: strictly raises `ValueError` if output path targets `data/raw/`.

---

## Log Entry: 2026-10-16 — Phase 5: Feature Engineering Implementation & Verification

### 1. Feature Engineering Layer (`src/climate_analyzer/feature_engineering.py`)
* **`add_rolling_features(df, windows=(12, 60, 120), min_periods=None, center=False)`:**
  * Trailing rolling means default (`center=False`) avoiding future data leakage.
  * Conservatively requires $\ge 75\%$ valid observations by default.
  * If `center=True`, retrospective centered feature is labeled `anomaly_centered_{w}m` and documented as unsuitable for causal prediction.
* **`calculate_annual_summary(df, min_valid_months=12)`:**
  * Computes annual statistics from monthly series.
  * Strict completeness accounting: `annual_mean_anomaly` evaluates strictly to `NaN` if `valid_months < min_valid_months` (default 12), ensuring incomplete years (e.g. 2026) are never reported as full annual averages.
  * Preserves `partial_year_mean` as an explicit diagnostic for researchers.
* **`calculate_baseline_anomalies(df, baseline_start, baseline_end, min_baseline_years=15)`:**
  * Calculates separate reference means for each of the 12 calendar months over `[baseline_start, baseline_end]` and subtracts them to produce re-centered anomalies.
  * Protects against ungrounded "pre-industrial" labeling without qualification.
* **`calculate_year_over_year_change(annual_df, require_complete=True)`:**
  * Computes rate of change in degrees Celsius ($\Delta Y_t = Y_t - Y_{t-1}$).
  * Avoids percentage changes (which are volatile and physically misleading for anomaly metrics crossing 0°C).
  * Requires consecutive complete calendar years; incomplete comparisons evaluate to `NaN`.

### 2. Testing & Coverage Execution
* Implemented 17 new unit tests in `tests/test_feature_engineering.py` + 1 regression test in `tests/test_data_validation.py`.
* Pytest Suite: **58 passed in 0.91s (0 warnings, 0 failures)**.
* Coverage: **90% package statement coverage** (`pytest --cov=climate_analyzer`).
* End-to-end dataset verification: Tested on `data/raw/GLB.Ts+dSST.csv` (1,764 records), confirming complete 146 years (1880–2025) and incomplete 2026 year handling.

---

## Log Entry: 2026-10-16 — Phase 6: Descriptive Analysis & Statistical Trend Estimation

### 1. Audit of Phase 5 & Defect Corrections
* **Rolling-Window Warm-up Verification:**
  * For $W=60$ months with `min_periods=45`, the first valid rolling value occurs at index 44 (observation #45; exactly 44 leading NaNs).
  * For $W=120$ months with `min_periods=90`, the first valid rolling value occurs at index 89 (observation #90; exactly 89 leading NaNs).
  * Added deterministic regression test `test_rolling_warmup_exact_positions` in `tests/test_feature_engineering.py`.
* **Calendar vs Observed Temporal Coverage:**
  * Dataset spans 1,764 calendar rows (1880-01 to 2026-12), but valid observations span 1880-01 to 2026-09 (1,761 valid months; 3 trailing unobserved NaNs).
  * Extended `ValidationReport` in `src/climate_analyzer/data_validation.py` to record `observed_start_date` and `observed_end_date` alongside calendar boundaries.
* **Pre-Industrial Baseline Qualification:**
  * Preserved NASA anomalies (1951–1980 base period) unaltered; documented 1880–1900 re-centering strictly as an early-instrumental proxy.

### 2. Descriptive Analysis Layer (`src/climate_analyzer/analysis/descriptive_analysis.py`)
* Implemented `calculate_descriptive_statistics()` returning `DescriptiveStatisticsResult` with count, mean, std (ddof=1), min, max, median, IQR, Fisher skewness, Fisher kurtosis, and configurable percentiles with linear quantile interpolation.
* Implemented `calculate_decadal_summary()` grouping valid complete years into calendar decades (1880–1889, ..., 2020–2029). Partial decades report `partial_decade_mean` and `is_complete=False`.
* Implemented `calculate_warmth_percentiles()` computing empirical warmth rankings (Rank 1 = warmest) and percentile scores relative to explicit reference distributions (e.g. 1951–1980).

### 3. Statistical Trend Estimation Layer (`src/climate_analyzer/analysis/trend_analysis.py`)
* Implemented `estimate_ols_trend()` with centered year coordinates ($t^* = \text{year} - \bar{t}$), converting to °C/decade, computing standard errors, 95% Student's t confidence intervals, $R^2$, and two-sided p-values.
* Implemented `test_mann_kendall_trend()` with Kendall's $S$, exact Kendall (1975) tie-corrected $\text{Var}(S)$, continuity-corrected normal score $Z$, and two-sided p-value. Conditioned trend direction on significance threshold ($\alpha = 0.05$).
* Implemented `estimate_sens_slope()` evaluating all pairwise slopes with exact calendar time intervals ($S_{ij} = (y_j - y_i) / (t_j - t_i)$), reporting median slope in °C/decade and non-parametric 95% rank confidence bounds (Gilbert 1987 / Sen 1968).
* Enforced statistical caveats across all outputs documenting that positive serial autocorrelation in climate series can inflate significance, and that historical trends describe observations without implying causal attribution.

### 4. Automated Testing & Verification
* Created `tests/test_descriptive_analysis.py` (10 tests) and `tests/test_trend_analysis.py` (16 tests).
* Pytest Suite: **85 passed in 1.67s (0 failures, 0 warnings)**.
* Test Coverage: **89% total statement coverage** across the package (`climate_analyzer.analysis` at 86–91%).
* Live End-to-End Execution on Local Raw GISTEMP Dataset (`data/raw/GLB.Ts+dSST.csv`):
  * Analyzed complete period: **1880 to 2025 (146 complete calendar years)**.
  * Year 2026 successfully excluded from complete annual stats (9/12 months observed = 75%, `annual_mean_anomaly` = NaN).
  * OLS Linear Trend: **+0.0826 °C/decade** (95% CI: [+0.0751, +0.0902] °C/decade, $R^2 = 0.7646$, $p = 4.41 \times 10^{-47}$).
  * Mann-Kendall Test: $S = +7698$, $\text{Var}(S) = 349,297.33$, $Z = +13.0234$, $p < 10^{-15}$, Direction = **INCREASING**.
  * Sen's Robust Slope: **+0.0804 °C/decade** (95% CI: [+0.0720, +0.0892] °C/decade, 10,585 pairwise slopes).
  * Top 5 Warmest Complete Years: 2024 (+1.28 °C), 2025 (+1.18 °C), 2023 (+1.17 °C), 2016 (+1.01 °C), 2020 (+1.01 °C).

---

## Log Entry: 2026-10-16 — Phase 6: Focused Scientific & Numerical Audit

### 1. Confirmed Defects and Target Fixes
1. **Mann-Kendall p-value Underflow Elimination:**
   * *Investigation:* The real-data run reported $p = 0.0000\text{e}+00$ at $Z = +13.0234$.
   * *Root Cause:* Evaluated via `2.0 * (1.0 - stats.norm.cdf(abs(z)))`. Because standard 64-bit float mantissa has ~16 digits, $1.0 - \text{cdf}(Z)$ rounds to $0.0$ for all $|Z| > 8.5$.
   * *Fix:* Switched to the direct survival function `float(2.0 * stats.norm.sf(abs(z)))`, which provides exact tail probabilities down to $|Z| \approx 37.5$. For $Z = +13.0234$, $p = 9.0088 \times 10^{-39}$. Added `log10_p_value` via `stats.norm.logsf` to preserve exact scientific magnitude without underflow across all extreme domains.
2. **Sen's Slope Rank-Based Confidence Interval 0-Based Indexing:**
   * *Investigation:* Compared against manual slope derivations and SciPy's `theilslopes` reference implementation.
   * *Root Cause:* Gilbert (1987) / Sen (1968) define rank $M_1 = \frac{N' - C_\alpha}{2}$ in 1-based indexing. In 0-based Python indexing, rank $M_1$ corresponds to index $M_1 - 1$. The previous implementation omitted the $-1$ offset, shifting the lower bound one rank higher than nominal coverage on small samples.
   * *Fix:* Implemented `idx_lower = max(0, min(n_slopes - 1, int(np.round((n_slopes - c_alpha) / 2.0)) - 1))` and `idx_upper = max(0, min(n_slopes - 1, int(np.round((n_slopes + c_alpha) / 2.0))))`. Verified 100% agreement with `scipy.stats.theilslopes` across 90%, 95%, and 99% confidence levels.
3. **OLS Residual Autocorrelation Diagnostic Added:**
   * Computed Durbin-Watson $d = 0.3346$ and lag-1 autocorrelation $r_1 = +0.7987$ on the 1880–2025 OLS fit. Exposed both in `OLSTrendResult` to provide quantitative proof of serial dependence invalidating conventional independence assumptions.
4. **Decadal Completeness Threshold Consistency:**
   * Unified `is_complete = valid_count >= effective_threshold` in `calculate_decadal_summary()`.

### 2. Audit Verification & Test Execution
* Total Tests: **90 passed in 1.66s** (5 new deterministic audit tests added).
* Package Statement Coverage: **89%**.
* Recalculated Real-Data GISTEMP Metrics:
  * Complete Years: 146 (1880–2025)
  * OLS Slope: $+0.0826$ °C/decade (95% CI: $[+0.0751, +0.0902]$ °C/decade, $p = 4.41 \times 10^{-47}$)
  * OLS Residual Autocorrelation: $r_1 = +0.7987$, Durbin-Watson $d = 0.3346$
  * Mann-Kendall: $S = +7698$, $Z = +13.0234$, $p = 9.0088 \times 10^{-39}$ ($\log_{10}(p) = -38.05$)
  * Sen's Slope: $+0.0804$ °C/decade (95% CI: $[+0.0720, +0.0892]$ °C/decade, 10,585 pairs)

---

## Log Entry: 2026-10-16 — Phase 7: Reusable, Scientifically Responsible Climate Visualizations

### 1. Scope & Implementation Summary
Implemented publication-ready, scientifically responsible Matplotlib plotting routines in `src/climate_analyzer/visualization/climate_plots.py` with strict defensive guarantees:

1. **`plot_monthly_anomalies(df, ...)`**:
   * Continuous high-resolution monthly timeline with explicit 1951–1980 baseline reference line.
   * NaN gaps are strictly preserved as line breaks (no silent bridging or zero-imputation).
   * Optional rolling average overlay (e.g. 5-year trailing mean).
   * Y-axis labeled in `Temperature Anomaly (°C)` and observation period clearly displayed.

2. **`plot_annual_anomalies(annual_df, ...)`**:
   * Visual separation of complete annual observations (solid bars) from incomplete years (hatched/outlined bars).
   * Incomplete years (such as 2026 with 9 observed months) are never presented as complete annual averages.
   * Optional OLS linear trend overlay fitted strictly on complete years.
   * Slope labeled in `°C/decade` with mandatory disclaimer explaining that conventional 95% confidence intervals assume independent residuals and are not autocorrelation-adjusted.

3. **`plot_decadal_summary(decadal_or_annual_df, ...)`**:
   * Reuses `calculate_decadal_summary()` API.
   * Complete 10-year decades rendered with solid bars and anomaly values.
   * Partial decades (e.g. 2020–2029 with 6 complete years) rendered with distinct hatched styling and observation count annotation (e.g., `+1.10* (6y)`).
   * Prevents partial decades from being cited as full decadal averages.

4. **`plot_seasonal_patterns(monthly_or_climatology_df, ...)`**:
   * Implemented explicit `calculate_monthly_climatology()` in `descriptive_analysis.py` to evaluate calendar month departure distributions without altering feature-engineering schemas.
   * Missing observations are strictly excluded with `.dropna()` (no zero-imputation).
   * Renders 12 calendar months (Jan–Dec) with mean departure curve, median markers, and $\pm 1$ standard deviation envelope.
   * Explicitly documents the evaluated reference period and 1951–1980 baseline context.

### 2. Plotting Safeguards Enforced
* All plotting functions return open `matplotlib.figure.Figure` objects; closing is delegated to callers.
* Caller DataFrames are never mutated (`df.copy()` defensive copies enforced on ingestion).
* Figures are written to disk only when caller explicitly passes `output_path`; parent directories are created automatically.
* Compatible with non-interactive backend (`Agg`) for headless testing and CLI execution.

### 3. Verification & Real-Data Execution
* Automated Test Suite (`tests/test_climate_plots.py`):
  * **24 new deterministic tests** covering figure types, axis labels, missing-data handling, incomplete-year separation, partial-decade hatching, OLS trend disclaimers, defensive copies, error handling, and file saving.
  * Entire Test Suite: **114 passed in 5.11s (0 failures, 0 warnings)**.
  * Package Statement Coverage: **90% total** (`climate_plots.py` at 91%).
* Real-Data Execution (`data/raw/GLB.Ts+dSST.csv`):
  * Successfully generated and verified 4 publication-quality figures in `outputs/figures/`:
    * `gistemp_monthly_anomalies.png` (499,821 bytes)
    * `gistemp_annual_anomalies.png` (274,144 bytes)
    * `gistemp_decadal_summary.png` (289,504 bytes)
    * `gistemp_seasonal_patterns.png` (309,531 bytes)

---

## Log Entry: 2026-10-16 — Phase 8: Interactive Streamlit Exploration Dashboard

### 1. Phase 7 Verification Findings & Pre-Dashboard Corrections
* **Visual Audit of Generated Figures:**
  * In `plot_monthly_anomalies()`, observation boundary detection originally used `work_df[date_col].max()`, which included trailing unobserved `NaN` months and printed `Dec 2026` rather than the latest observed month (`Sep 2026`). Corrected to compute boundaries from `work_df.loc[work_df[value_col].notna(), date_col]`.
  * In `plot_decadal_summary()`, text value annotations for extreme decades (`-0.33` for 1910–1919 and `+1.06*` for 2020–2029) touched the top and bottom axis boundaries. Added 12–14% vertical headroom to `ax.set_ylim()` to guarantee readable, unclipped annotations.
  * Verified all labels: departures are explicitly named `Temperature Anomaly (°C)`, reference baseline is cited as 1951–1980, shaded envelope is labeled `±1 Sample Std Deviation Band`, and incomplete years are segregated.

### 2. Streamlit Dashboard Implementation (`app/streamlit_app.py`)
Implemented full multi-tabbed Streamlit application organized into 5 core exploratory modules:
1. **Overview:**
   * High-level KPI metric cards: Latest Complete Year (`2025: +1.18 °C`), Ongoing Year (`2026: +1.23 °C, 9/12 mos`), Warmest Complete Year on Record (`2024: +1.28 °C`), and OLS Rate of Warming (`+0.083 °C/decade`).
   * High-resolution monthly anomaly timeline with interactive trailing rolling average options (12m, 60m, 120m).
2. **Historical Trends:**
   * Annual anomaly plot with complete vs. incomplete year styling and optional OLS linear trend overlay.
   * Statistical comparison panel displaying OLS, Mann-Kendall, and Sen's slope parameters.
   * Prominent callout on serial autocorrelation ($r_1 \approx +0.80$, Durbin-Watson $d \approx 0.33$) explaining why conventional standard errors are deflated and confidence intervals are not autocorrelation-adjusted.
3. **Seasonal Patterns:**
   * Calendar-month anomaly departure distributions (Jan through Dec).
   * Reference period controls: 1951–1980 baseline, full record, recent 30 years, or custom year slider.
   * Detailed monthly statistics table (mean departure, std dev, median, IQR, extremes).
   * Missing-data note explaining zero-imputation prohibition.
4. **Decadal Comparison:**
   * Decadal mean anomaly plot distinguishing complete 10-year decades from partial diagnostic decades.
   * Decadal data table listing completeness status, valid year count, complete mean, and diagnostic partial mean.
   * Explicit notice that 2020–2029 has only 6 complete years and is not a finished decadal mean.
5. **Data Quality & Methodology:**
   * Official NASA GISTEMP v4 dataset provenance and 1951–1980 baseline definition.
   * Automated validation checklist (physical boundaries, calendar continuity, missing-value markers).
   * Methodological overview (centered OLS coordinates, Kendall tie corrections, Sen's rank-based intervals).
   * Clear statement of non-causal, non-forecasting scope.

### 3. Architecture & Safety Guarantees
* **Data Flow:** Reuses existing modules (`load_gistemp_monthly`, `clean_monthly_gistemp`, `add_rolling_features`, `calculate_annual_summary`, `calculate_decadal_summary`).
* **Caching:** Applied `@st.cache_data` to data loading and preprocessing. Matplotlib `Figure` objects are created cleanly during render and closed immediately with `plt.close(fig)` to avoid shared state and memory leaks.
* **Error Handling:** If the dataset is absent or invalid, displays an actionable Streamlit error message with the expected local path and official NASA download link without crashing.

### 4. Verification & Testing
* Created `tests/test_streamlit_app.py` (4 unit and pipeline smoke tests).
* Total Tests: **118 passed in 6.66s (0 failures, 0 warnings)**.
* Core package coverage: **90%**.
* Headless Startup Check: Verified Streamlit server starts up cleanly on `--server.headless=true`, binds port, and returns HTTP 200 OK on health check (`/_stcore/health`).


