# Global Climate Surface Temperature Trend Analysis (1880–2026)
**Technical Portfolio Report — NASA GISTEMP v4 Land-Ocean Temperature Index**

---

## Executive Summary

This report documents the statistical analysis and historical trend estimation of global surface temperature anomalies conducted with the **Climate Trend Analyzer** system. Utilizing the authoritative NASA GISS Surface Temperature Analysis version 4 (GISTEMP v4) Land-Ocean Temperature Index (L-OTI), this study examines monthly and annual temperature departures relative to the official **1951–1980 climatological baseline**.

### Key Verified Findings (1880–2025 Complete Observation Record):
* **Observed Record Length:** 146 complete calendar years (1880 through 2025) comprising 1,761 valid monthly observations through September 2026.
* **Long-Term Linear Trend (OLS):** $+0.0826 \pm 0.0076$ °C/decade ($+0.008262$ °C/year, $R^2 = 0.7646$).
* **Robust Non-Parametric Trend (Sen's Slope):** $+0.0804$ °C/decade ($+0.008038$ °C/year, 95% rank-based CI: $[+0.0720, +0.0892]$ °C/decade across 10,585 pairwise slopes).
* **Monotonic Trend Significance (Mann–Kendall):** Statistically significant upward monotonic trend ($Z = +13.0234$, Kendall $S = +7,698$, $p = 9.01 \times 10^{-39}$, $\log_{10}(p) = -38.05$, exact Kendall 1975 tie correction).
* **Warmest Complete Calendar Year on Record:** **2024** ($+1.2817$ °C anomaly relative to 1951–1980).
* **Latest Complete Calendar Year:** **2025** ($+1.1850$ °C anomaly).
* **Ongoing Incomplete Year (2026 Diagnostic):** $+1.2344$ °C arithmetic mean across 9 observed months (Jan–Sep 2026). In accordance with data integrity rules, its annual mean is strictly treated as `NaN` and excluded from all completed-year regressions.
* **Ongoing Decadal Diagnostic (2020–2029):** Arithmetic average of $+1.063$ °C across 6 complete years (2020–2025). Categorized strictly as an incomplete diagnostic decade.

---

## 1. Dataset Provenance & Baseline Specifications

* **Data Source:** NASA Goddard Institute for Space Studies (GISS).
* **Dataset Identifier:** GISTEMP Surface Temperature Analysis v4 (Land-Ocean Temperature Index, L-OTI).
* **Official URL:** [https://data.giss.nasa.gov/gistemp/](https://data.giss.nasa.gov/gistemp/)
* **Direct File Path:** `data/raw/GLB.Ts+dSST.csv` (SHA-256: `4c64eda288bce32b8d76926278f346ecea1e53506df8ee72a3d5accb0657c416`).
* **Underlying Observations:** Integrates meteorological station observations from NOAA GHCN v4 (land) with ocean surface temperature observations from ERSST v5 (sea surface), applying a 1,200 km spatial smoothing radius.
* **Official Baseline Period:** **1951–1980**. All anomalies represent deviations from the 1951–1980 calendar-month means.
* **Physical Units:** Degrees Celsius (°C).

---

## 2. Data Governance, Validation & Cleaning Safeguards

To prevent artificial bias and preserve scientific fidelity, data handling adheres to strict rules:

1. **Physical Boundary Enforcement:**
   * Values must fall within $[-10.0 \text{ °C}, +10.0 \text{ °C}]$ (fatal error if violated).
   * Soft warning threshold at $[-3.0 \text{ °C}, +3.0 \text{ °C}]$ flags potential anomalies for inspection without altering the underlying data.
2. **Zero-Imputation Prohibition:**
   * Missing observations (NASA `***` markers) are parsed to IEEE 754 `NaN`.
   * Under no circumstances are missing values filled with `0.0`, as $0.0\text{ °C}$ represents the exact 1951–1980 climatological baseline.
3. **Temporal Integrity:**
   * Strict verification of calendar month bounds ($1 \le m \le 12$), strict date monotonicity, and verification that $date.year == year$ and $date.month == month$.
4. **Immutability:**
   * Raw CSV files in `data/raw/` are treated as immutable and never modified. Transformations operate strictly on defensive copies.

---

## 3. Analytical Methodology

### 3.1 Completeness Classification
* **Complete Calendar Years:** Defined as calendar years possessing exactly 12 valid monthly observations ($100\%$ completeness). Only complete years are admitted into annual summary statistics and long-term trend regressions.
* **Incomplete Years:** Years with fewer than 12 valid months (such as the ongoing observation year 2026 with 9 months) have their `annual_mean_anomaly` set strictly to `NaN`. A separate `partial_year_mean` is reported for diagnostic tracking.
* **Decadal Completeness:** Decades follow calendar boundaries (`1880–1889`, `1890–1899`, ..., `2020–2029`). Exactly 10 complete annual means are required to compute a `complete_decade_mean`. Decades with fewer contributing complete years report only a `partial_decade_mean`.

### 3.2 Trend Estimation Models
Three complementary statistical techniques are evaluated:

1. **Ordinary Least Squares (OLS) Regression:**
   * Linear model fitted on centered time coordinates $t^* = t - \bar{t}$ to eliminate parameter collinearity and ensure numerical precision:
     $$\hat{y}(t) = \hat{\beta}_0 + \hat{\beta}_1 (t - \bar{t})$$
   * Slopes are converted to decadal warming rates ($\hat{\beta}_{\text{decadal}} = 10 \times \hat{\beta}_1$).
2. **Mann–Kendall Monotonic Trend Test:**
   * Non-parametric rank-based test evaluating the Kendall $S$ score across all pairs $(y_j - y_i)$ for $j > i$.
   * Variance calculation applies exact Kendall (1975) tie correction. Standard normal $Z$ score is evaluated using the normal distribution survival function (`2 * norm.sf(|Z|)`) to prevent floating-point underflow.
3. **Sen's Robust Slope Estimator:**
   * Computes the median of all $N(N-1)/2 = 10,585$ pairwise slopes:
     $$\text{Slope} = \text{median}\left(\frac{y_j - y_i}{t_j - t_i}\right) \quad \text{for } 1 \le i < j \le N$$
   * Computes non-parametric rank-based 95% confidence intervals (Gilbert 1987). Robust against non-normal errors, extreme values, and outliers.

---

## 4. Empirical Results & Findings

### 4.1 Statistical Trend Comparison (1880–2025 Complete Record)

| Statistical Metric | Ordinary Least Squares (OLS) | Sen's Robust Slope | Mann–Kendall Test |
| :--- | :--- | :--- | :--- |
| **Analyzed Sample** | 146 complete calendar years | 146 complete calendar years | 146 complete calendar years |
| **Observation Window** | 1880 to 2025 | 1880 to 2025 | 1880 to 2025 |
| **Warming Rate (Decadal)** | **$+0.0826$ °C/decade** | **$+0.0804$ °C/decade** | — |
| **Warming Rate (Annual)** | $+0.008262$ °C/year | $+0.008038$ °C/year | — |
| **95% Confidence Interval** | $[+0.0751, +0.0902]$ °C/dec* | $[+0.0720, +0.0892]$ °C/dec | — |
| **Goodness-of-Fit / Score** | $R^2 = 0.7646$ | 10,585 pairwise slopes | $S = +7,698$, $Z = +13.0234$ |
| **p-value** | $4.41 \times 10^{-47}$* | Non-parametric rank CI | $9.01 \times 10^{-39}$ ($\log_{10} p = -38.05$) |
| **Trend Direction** | Upward / Warming | Upward / Warming | Monotonic Increasing |

*\*Note: Conventional OLS uncertainty bounds assume serially independent residuals and are unadjusted for autocorrelation.*

### 4.2 Decadal Climatology Summary

| Calendar Decade | Status | Valid Complete Years | Decadal Mean Anomaly (°C) | Diagnostic Note |
| :---: | :---: | :---: | :---: | :--- |
| **1880–1889** | Complete | 10 | $-0.212$ | Early instrumental baseline |
| **1890–1899** | Complete | 10 | $-0.237$ | |
| **1900–1909** | Complete | 10 | $-0.313$ | |
| **1910–1919** | Complete | 10 | $-0.327$ | Coolest decade in instrumental record |
| **1920–1929** | Complete | 10 | $-0.236$ | |
| **1930–1939** | Complete | 10 | $-0.122$ | |
| **1940–1949** | Complete | 10 | $+0.043$ | Early mid-century warm period |
| **1950–1959** | Complete | 10 | $-0.048$ | Base period component |
| **1960–1969** | Complete | 10 | $-0.030$ | Base period component |
| **1970–1979** | Complete | 10 | $+0.034$ | Base period component |
| **1980–1989** | Complete | 10 | $+0.247$ | Onset of accelerated warming |
| **1990–1999** | Complete | 10 | $+0.386$ | |
| **2000–2009** | Complete | 10 | $+0.589$ | |
| **2010–2019** | Complete | 10 | $+0.804$ | |
| **2020–2029** | **Partial** | **6** | **$+1.063$** | **Diagnostic mean (2020–2025). NOT a complete decade.** |

### 4.3 Seasonal Climatology Patterns
Across the 12 calendar months (Jan through Dec), monthly anomalies exhibit distinct variance structures:
* Climatological summaries evaluate existing 1951–1980 baseline anomalies across the chosen time window (e.g. 1951–1980 baseline vs. recent 1995–2024 window).
* Selecting an alternative evaluation window summarizes the sample distribution of 1951–1980 anomalies over those years; it does not re-zero or rebaseline anomalies.
* Northern Hemisphere winter months (Dec–Feb) display greater anomaly dispersion and higher standard deviation than Northern Hemisphere summer months (Jun–Aug), consistent with higher continental land-mass thermal inertia variability in the high latitudes.

---

## 5. Statistical Caveats & Methodological Limitations

1. **Serial Autocorrelation Impact:**
   * Residuals from the OLS linear model exhibit strong positive first-order autocorrelation:
     $$r_1 = +0.7987, \quad \text{Durbin-Watson } d = 0.3346$$
   * Because conventional OLS standard errors and standard Mann–Kendall test formulations assume independent and identically distributed (i.i.d.) errors, positive autocorrelation underestimates the true standard error.
   * Consequently, nominal confidence intervals ($[+0.0751, +0.0902]$ °C/decade) and extremely small p-values are unadjusted for autocorrelation and narrower than the true physical uncertainty envelope.
2. **Observational Scope vs. Physical Causality:**
   * The statistical trends estimated in this study represent historical observational fits.
   * They do **not** isolate human anthropogenic forcing from natural climate drivers (solar irradiance, volcanic aerosols, internal variability such as ENSO or AMO). Causal attribution requires coupled ocean-atmosphere General Circulation Models (GCMs).
3. **Absence of Future Climate Projections:**
   * Linear and monotonic trends cannot be extrapolated into the future as unconstrained forecasts.
   * Future warming trajectories depend on greenhouse gas emissions scenarios, socioeconomic pathways (SSPs), carbon cycle feedbacks, and climate sensitivity parameters assessed in IPCC assessment reports.

---

## 6. Software Architecture & Verification Status

* **Package:** `climate_analyzer` (Python $\ge 3.10$, modular `src` layout).
* **Automated Tests:** **131 passing unit, regression, and integration tests** in `tests/`.
* **Statement Coverage:** **80% overall project coverage** (core packages 85%–94%; Streamlit application 36%).
* **Headless Dashboard Verification:** Verified clean headless execution via subprocess (`streamlit run app/streamlit_app.py --server.headless=true`) returning `HTTP 200 OK` on health checks without memory leaks or unclosed Matplotlib figures.

---

## 7. Conclusions

The historical climate observation record from NASA GISTEMP v4 confirms a pronounced, statistically robust warming trajectory of approximately $+0.08$ °C/decade across the 1880–2025 complete instrumental period. Both parametric (OLS: $+0.0826$ °C/decade) and non-parametric (Sen's slope: $+0.0804$ °C/decade) models demonstrate consistent rate agreement. Decadal averages have risen continuously from $-0.327$ °C (1910–1919) to $+0.804$ °C (2010–2019), with the ongoing partial 2020–2029 period diagnostic average reaching $+1.063$ °C across its first six complete years (2020–2025). All metrics, completeness boundaries, and uncertainty limits are documented under strict data science governance standards.
