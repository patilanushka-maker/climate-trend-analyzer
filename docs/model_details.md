# Analytical Methods & Statistical Specifications

## 1. Scope & Analytical Objectives
The analytical engine focuses on empirical statistical methods for historical climate series:
* **Descriptive Climatology**: Central tendency, variability, quantile distributions, and historical baseline normal periods.
* **Long-Term Trend Analysis**: Quantifying rates of change over multi-decadal observation records.
* **Seasonal Pattern Decomposition**: Disentangling annual periodicity and cyclic variation from underlying signals.
* **Climatological Anomaly Detection**: Identifying extreme events, record-breaking temperatures, and sustained departures from baseline climatology.

> [!NOTE]
> Advanced predictive forecasting and machine learning models are optional future extensions. Any future data-driven forecasts are statistical approximations of historical observations and are **not equivalent to physics-based global climate models (GCMs)** like CMIP6.

---

## 2. Statistical Formulations

### 2.1 Trend Estimation
* **Parametric Linear & Polynomial Regression (OLS)**: Fits rate of change ($\beta_1$) with standard error and confidence intervals:
  $$Y_t = \beta_0 + \beta_1 t + \epsilon_t$$
* **Non-Parametric Mann-Kendall Test**: Evaluates monotonic upward or downward trend significance without assuming normality:
  $$S = \sum_{k=1}^{n-1}\sum_{j=k+1}^n \text{sgn}(x_j - x_k)$$
* **Sen's Slope Estimator**: Robust non-parametric slope calculation using the median of pairwise slopes:
  $$\beta = \text{median}\left(\frac{x_j - x_k}{j - k}\right), \quad \forall j > k$$

### 2.2 Seasonal Decomposition
* **Additive Decomposition**: Appropriate when seasonal fluctuations are constant over time:
  $$Y_t = T_t + S_t + R_t$$
* **Multiplicative Decomposition**: Appropriate when seasonal amplitude scales with trend magnitude:
  $$Y_t = T_t \times S_t \times R_t$$
* **STL (Seasonal and Trend decomposition using Loess)**: Non-linear filtering capable of handling changing seasonal patterns and robust to outliers.

### 2.3 Anomaly Detection
* **Baseline Normal Reference**: Standard baseline (e.g., 1951–1980 or 1961–1990) monthly means ($\mu_m$) and standard deviations ($\sigma_m$).
* **Z-Score Anomaly**:
  $$Z_t = \frac{x_{t,m} - \mu_m}{\sigma_m}$$
* **Interquartile Range (IQR) Thresholding**: Non-parametric detection of extreme temperature/precipitation events.

---

## 3. Validation & Statistical Diagnostics
* **Autocorrelation & Stationarity**: Augmented Dickey-Fuller (ADF) test and Durbin-Watson diagnostic for residual autocorrelation.
* **Goodness of Fit & Significance**: $R^2$, adjusted $R^2$, and $p$-values ($\alpha = 0.05$).
* **Error Metrics**: Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).
