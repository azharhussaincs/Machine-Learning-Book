# Time Series Forecasting

## Introduction

A **time series** is data recorded **in time order** — daily sales, hourly temperatures,
monthly revenue, stock prices, website traffic, ECG signals. **Time series forecasting** is
predicting *future* values from *past* ones. It's everywhere in business and science:
demand planning, financial forecasting, weather prediction, capacity planning, and anomaly
detection in monitoring.

Time series is special because **order matters** and observations are **not
independent** — today depends on yesterday. This breaks a core assumption of the models in
Part IV and requires its own techniques and a different way of splitting data.

::: keyidea
In time series, **the past predicts the future**, and **you must never shuffle the data**.
Yesterday influences today, so we keep time order, engineer features from the past (lags,
rolling averages), and always split **chronologically** — train on the past, test on the
future. Shuffling would let the model "see the future", a fatal leak.
:::

By the end of this chapter you will be able to:

- Identify the **components** of a time series (trend, seasonality, noise).
- Use **classic** (moving average, exponential smoothing, ARIMA) and **ML/DL** methods.
- Engineer **lag and rolling features** and split data **chronologically**.
- Forecast a series and evaluate it properly.

## Components of a time series

![A time series decomposed into its components: a long-term trend, a repeating seasonal pattern, and random noise. Real series combine these; understanding them guides the forecasting method.](assets/images/ch42_components.png)

- **Trend** — the long-term direction (sales growing over years).
- **Seasonality** — a repeating pattern at fixed periods (ice-cream sales peak each summer;
  traffic peaks each rush hour).
- **Cyclical** — longer, irregular ups and downs (economic cycles), not fixed-period.
- **Noise (residual)** — random fluctuation left over.

**Decomposing** a series into trend + seasonality + noise helps you understand it and choose
a method.

### Stationarity

Many classic methods assume the series is **stationary** — its statistical properties (mean,
variance) don't change over time. Real series often have trends/seasonality (non-stationary)
and are made stationary by **differencing** (modelling the *change* from one step to the
next) before applying such models.

## Forecasting methods

### Classic statistical methods

- **Moving average** — predict the average of the last *k* values; smooths noise.
- **Exponential smoothing** — weighted average giving more weight to recent values; handles
  trend and seasonality (Holt-Winters).
- **ARIMA** (AutoRegressive Integrated Moving Average) — the classic workhorse: combines
  autoregression (past values), differencing (for stationarity), and moving average of past
  errors. **SARIMA** adds seasonality. (Library: `statsmodels`.)

### Machine-learning approach (feature engineering + regression)

A powerful, flexible approach: turn the time series into a **supervised problem** by
creating **features from the past**, then use any regressor (Part IV):

- **Lag features** — previous values (`lag1` = yesterday, `lag12` = same month last year).
- **Rolling features** — moving averages/std over recent windows.
- **Calendar features** — month, day-of-week, holiday flags (Chapter 12).

Then predict the next value with linear regression, random forest, or gradient boosting.

### Deep learning

**LSTMs/GRUs** (Chapter 35) and increasingly **Transformers** (Chapter 37) handle complex,
long sequences and multiple related series, at the cost of more data and compute.

## The cardinal rule: split by time

![Time-series train/test split: always split chronologically — train on earlier data, test on later data. Never shuffle, or the model "sees the future" (data leakage).](assets/images/ch42_split.png)

::: warning
**Never use a random train/test split for time series.** You must split **chronologically**
— train on the past, test on the future — and never shuffle. Random splitting (or using
future lags) leaks future information into training, giving falsely great results that
collapse in real forecasting. Use `TimeSeriesSplit` for cross-validation.
:::

## Practical: forecasting with lag features

Let's forecast a synthetic series (trend + yearly seasonality) using lag features and
linear regression, and compare to a naive baseline.

```python
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
np.random.seed(0)

t = np.arange(120)                                       # 120 months
series = 0.5 * t + 10 * np.sin(2 * np.pi * t / 12) + 50 + np.random.normal(0, 3, 120)
df = pd.DataFrame({"y": series})

for lag in [1, 2, 3, 12]:                                # past values as features
    df[f"lag{lag}"] = df["y"].shift(lag)                 # lag12 captures yearly seasonality
df = df.dropna()

split = int(len(df) * 0.8)                               # CHRONOLOGICAL split (no shuffle!)
feats = ["lag1", "lag2", "lag3", "lag12"]
X_tr, X_te = df[feats].iloc[:split], df[feats].iloc[split:]
y_tr, y_te = df["y"].iloc[:split], df["y"].iloc[split:]

model = LinearRegression().fit(X_tr, y_tr)
pred = model.predict(X_te)
print("lag-feature regression MAE:", round(mean_absolute_error(y_te, pred), 2))
print("naive baseline MAE (predict last month):",
      round(mean_absolute_error(y_te, X_te["lag1"]), 2))
```

**Output:**
```text
lag-feature regression MAE: 4.13
naive baseline MAE (predict last month): 5.30
```

### Explanation

- We turned the series into a supervised problem with **lag features** — crucially including
  **`lag12`** (the value 12 months ago) to capture the **yearly seasonality**.
- We split **chronologically** (first 80% train, last 20% test) — never shuffling.
- The lag-feature model (**MAE 4.13**) beat the **naive "predict last month" baseline
  (5.30)** — proving the model learned the trend and seasonal pattern, not just persistence.

::: keyidea
Notice the workflow: **feature-engineer the past (lags, seasonal lags) → use a standard
regressor → evaluate on the future.** This lets you bring the entire power of Part IV
(random forests, gradient boosting) to forecasting — often beating classical methods on
complex, multi-feature problems. The keys are the *seasonal lag* and the *chronological
split*.
:::

## Evaluation

- **MAE / RMSE** — average error in the target's units.
- **MAPE (Mean Absolute Percentage Error)** — error as a percentage (comparable across
  scales), though it misbehaves near zero.
- **Always compare to a naive baseline** (predict the last value, or last season's value) —
  a model that can't beat "predict yesterday" isn't useful.

::: tip
**Practical & debugging tips:** (1) **Always include seasonal lags** (e.g. lag 12 for
monthly, lag 7 for daily-weekly) and calendar features. (2) **Split chronologically**; use
`sklearn`'s `TimeSeriesSplit` for CV. (3) **Beat a naive baseline** or your model adds
nothing. (4) For multi-step forecasts, either predict recursively or train separate models
per horizon. (5) Watch for **non-stationarity** — difference the series for ARIMA. (6) Tools:
`statsmodels` (ARIMA), `Prophet` (`pip install prophet`, easy trend+seasonality),
`sktime`/`darts` for ML/DL time series.
:::

## Advantages, disadvantages, and use cases

| Approach | Strengths | Weaknesses |
|---|---|---|
| Moving average / smoothing | Simple, fast, good baseline | Limited; lags trends |
| ARIMA/SARIMA | Strong for stationary/seasonal data | Assumes structure; tuning needed |
| ML (lags + trees) | Flexible; many features; non-linear | Needs feature engineering |
| Deep learning (LSTM/Transformer) | Complex/long patterns, multi-series | Data/compute hungry |

**Use cases:** demand & sales forecasting, finance (prices, risk), energy load, weather,
inventory/capacity planning, web traffic, IoT/sensor monitoring, and anomaly detection.

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Random train/test split (or shuffling).** This is the cardinal sin of time
series — it leaks future data and gives fake accuracy. Always split chronologically.
:::

- **Mistake 2 — No seasonal lag feature**, missing repeating patterns.
- **Mistake 3 — Not comparing to a naive baseline** (so you can't tell if the model helps).
- **Mistake 4 — Ignoring non-stationarity** (trends) for methods that assume stationarity.
- **Mistake 5 — Using future information** in features (look-ahead bias / leakage).
- **Mistake 6 — Treating time series as independent rows** and applying Part IV blindly.

## Best practices

- **Split chronologically; never shuffle.** Use `TimeSeriesSplit`.
- **Engineer lag, rolling, seasonal, and calendar features.**
- **Always beat a naive baseline.**
- **Decompose** to understand trend/seasonality; difference for stationarity if needed.
- **Choose the method by complexity:** smoothing/ARIMA for simple, ML for feature-rich, DL
  for complex/long/multi-series.
- **Evaluate with MAE/RMSE/MAPE** on the future hold-out.

## Chapter Summary

- A **time series** is time-ordered, dependent data; forecasting predicts future from past.
  **Order matters** and you must **never shuffle**.
- Series have **trend, seasonality, cyclical, and noise** components; **stationarity** (via
  differencing) matters for classic methods.
- Methods: **moving average / exponential smoothing**, **ARIMA/SARIMA**, **ML with lag &
  rolling features** (then any regressor), and **deep learning (LSTM/Transformer)**.
- **Split chronologically** (train past, test future); a lag-feature regression (MAE 4.13)
  beat the naive baseline (5.30) by capturing trend and seasonality (`lag12`).
- Always **compare to a naive baseline** and evaluate with **MAE/RMSE/MAPE**; avoid the
  cardinal sin of random splitting/leakage.

---

::: {.qband}
Practice Zone — Chapter 42
:::

## Multiple-Choice Questions (MCQs)

**Q1.** A defining feature of time series data is that:
a) Rows are independent  b) Order/time matters and observations are dependent  c) It has no
trend  d) It must be shuffled

**Q2.** A repeating pattern at fixed periods is called:
a) Trend  b) Seasonality  c) Noise  d) Drift

**Q3.** For time series, the train/test split must be:
a) Random  b) Chronological (past→train, future→test)  c) Shuffled  d) By class

**Q4.** ARIMA is mainly used for:
a) Images  b) Classic time-series forecasting  c) Clustering  d) Text

**Q5.** A "lag1" feature is:
a) The next value  b) The previous time step's value  c) The average  d) The label

**Q6.** Making a series stationary often involves:
a) Shuffling  b) Differencing  c) One-hot encoding  d) Pooling

**Q7.** You should always compare a forecast to a:
a) Random model  b) Naive baseline (e.g. predict last value)  c) Classifier  d) Cluster

**Q8.** Randomly shuffling time series before splitting causes:
a) Better accuracy  b) Data leakage (seeing the future)  c) Faster training  d) Nothing

### MCQ Answers
**1:** b. **2:** b. **3:** b. **4:** b. **5:** b. **6:** b. **7:** b. **8:** b.

## Interview Questions (with answers)

**Q1. Why can't you use a normal random train/test split for time series?**
*Answer:* Because observations are time-dependent; random splitting (or shuffling) would put
future data in training and past data in testing, leaking future information ("look-ahead
bias"). You must split chronologically — train on earlier data, test on later data — to
honestly simulate forecasting.

**Q2. What are the components of a time series?**
*Answer:* Trend (long-term direction), seasonality (repeating pattern at fixed periods),
cyclical (longer irregular fluctuations), and noise (random residual). Decomposing into these
helps understand the data and pick a forecasting method.

**Q3. How can you use standard ML regressors for forecasting?**
*Answer:* By converting the series into a supervised problem with engineered features from
the past — lag features (previous values, including seasonal lags), rolling statistics, and
calendar features — then training any regressor (linear, random forest, gradient boosting) to
predict the next value, evaluated on a chronological hold-out.

**Q4. What is ARIMA?**
*Answer:* AutoRegressive Integrated Moving Average — a classic model combining autoregression
on past values (AR), differencing to achieve stationarity (I), and a moving average of past
forecast errors (MA). SARIMA adds seasonal terms. It's a strong baseline for univariate,
reasonably structured series.

**Q5. Why compare against a naive baseline?**
*Answer:* Because a forecast is only valuable if it beats trivial predictions like "tomorrow
equals today" or "this season equals last season". The naive baseline sets the bar; a complex
model that can't beat it isn't worth deploying.

## Scenario-Based Questions (with answers)

**Q1.** *Your sales-forecasting model has amazing test accuracy but fails badly in production.
You used a random train/test split. What went wrong?*
*Answer:* Data leakage from random splitting — the model trained on future data and tested on
past, inflating accuracy unrealistically. In production it only has the past. Fix by splitting
chronologically (and using `TimeSeriesSplit` for CV).

**Q2.** *Monthly demand has a clear yearly pattern, but your model misses it. Which feature
should you add?*
*Answer:* A seasonal lag — for monthly data, `lag12` (the value 12 months ago) — plus calendar
features like month. This lets the model capture the yearly seasonality it was missing.

**Q3.** *Your forecasting model barely beats predicting last month's value. Is it useful?*
*Answer:* Only marginally — and you should question whether the added complexity is justified.
Investigate better features (seasonal lags, rolling stats, external regressors) or simpler
robust methods; if nothing meaningfully beats the naive baseline, the series may be largely
unpredictable at that horizon.

## Logic-Based Questions (with answers)

**Q1.** Why does including `lag12` help a monthly series with yearly seasonality?
*Answer:* Because the value 12 months ago is from the same season; if the pattern repeats
yearly, last year's same-month value is highly predictive of this month's, directly encoding
seasonality as a feature.

**Q2.** Why is shuffling time series data a form of cheating?
*Answer:* It mixes future observations into the training set, so the model effectively learns
from data it wouldn't have at prediction time, producing optimistic results that can't be
achieved in real forecasting.

**Q3.** Why might a moving-average forecast lag behind a rising trend?
*Answer:* Because it averages recent past values, which are lower than the current rising
level; the average trails the trend, systematically under-predicting during sustained
increases.

## Practical Questions (with answers)

**Q1.** Write code to create a lag-1 feature in a pandas DataFrame column `y`.
*Answer:* `df["lag1"] = df["y"].shift(1)`.

**Q2.** Which scikit-learn tool gives time-series cross-validation?
*Answer:* `TimeSeriesSplit` (from `sklearn.model_selection`), which produces train/test folds
that respect time order.

**Q3.** Name two libraries for time-series forecasting.
*Answer:* `statsmodels` (ARIMA/SARIMA) and `Prophet` (or `sktime`/`darts` for ML/DL time
series).

## Long Questions (with answers)

**Q1. Explain how to forecast a time series with machine learning, including feature
engineering, splitting, and evaluation, and why time series needs special handling.**

*Answer:* Time series data is **time-ordered and dependent** — each value relates to previous
ones — which violates the independence assumptions of standard ML and demands special care.
The ML approach converts forecasting into a **supervised problem** through **feature
engineering from the past**: **lag features** (previous values, e.g. `lag1` = last step), and
crucially **seasonal lags** (e.g. `lag12` for monthly yearly seasonality); **rolling
statistics** (moving averages/standard deviations over recent windows); and **calendar
features** (month, day-of-week, holidays). A regressor (linear regression, random forest,
gradient boosting) then predicts the next value from these features. **Splitting must be
chronological** — train on earlier data, test on later — and you must **never shuffle**, or
future information leaks into training (look-ahead bias), producing fake accuracy that
collapses in production; `TimeSeriesSplit` provides proper cross-validation. **Evaluation**
uses MAE/RMSE/MAPE on the future hold-out, and you must **compare against a naive baseline**
(predict the last value or last season's value) — a model that can't beat persistence is not
useful. In the chapter's example, a lag-feature regression (MAE 4.13), helped by the seasonal
`lag12`, beat the naive baseline (5.30), demonstrating it learned the trend and seasonality.
This approach lets the full power of Part IV's algorithms apply to forecasting, provided the
temporal structure and chronological splitting are respected.

**Q2. Compare classical statistical methods (ARIMA) with machine-learning and deep-learning
approaches for time-series forecasting.**

*Answer:* **Classical methods** like moving averages, exponential smoothing (Holt-Winters),
and **ARIMA/SARIMA** model the series' statistical structure directly: ARIMA combines
autoregression on past values, differencing for stationarity, and a moving average of past
errors, with SARIMA adding seasonal terms. They are well-understood, work well on
**univariate, reasonably structured** series, need relatively little data, and provide
interpretable parameters and confidence intervals — but they assume specific structure (often
stationarity), require careful order selection, and struggle with many external features or
highly non-linear patterns. **Machine-learning approaches** instead engineer lag, rolling,
seasonal, and calendar **features** and apply flexible regressors (random forests, gradient
boosting); they handle **multiple input features and non-linearities** easily, often
outperforming ARIMA on complex, feature-rich problems, at the cost of manual feature
engineering and careful chronological validation. **Deep-learning approaches** (LSTMs, GRUs,
and increasingly Transformers) can model **complex, long-range, and multivariate** patterns
and many related series jointly, but are **data- and compute-hungry**, harder to tune, and
overkill for simple problems. The practical guidance: start with a naive baseline and a
classical or simple ML model; escalate to feature-rich ML (gradient boosting) for complex,
multi-feature problems; and reserve deep learning for large-scale, complex, or multi-series
forecasting — always validating chronologically and beating the baseline.

## Exercises

1. Identify trend, seasonality, and noise in a real example (e.g. monthly retail sales).
2. Explain why you must not shuffle time-series data before splitting.
3. For daily data with weekly patterns, which lag feature captures the weekly cycle?
4. Why always compare a forecast to a naive baseline?
5. Name two classic and two ML/DL forecasting methods.

## Mini-Project

**Project: Forecast a real time series.**

1. Get a time series (e.g. airline passengers, retail sales, or your own daily data).
2. Plot it and decompose into trend/seasonality/noise (`statsmodels seasonal_decompose`).
3. Engineer lag (including seasonal), rolling, and calendar features; train a regressor with a
   **chronological** split.
4. Compare MAE to a naive baseline and to a simple ARIMA/exponential-smoothing model.
5. Plot forecasts vs actuals and write a short report. Save in `my-ml-journey/`.

## Assignments

1. **Coding:** Build a forecast with `TimeSeriesSplit` cross-validation and report mean MAE
   across folds.
2. **Coding (stretch):** Fit an ARIMA model with `statsmodels` (or `Prophet`) and compare to
   your lag-feature ML model.
3. **Conceptual:** Write one page on why standard ML assumptions break for time series and
   how chronological splitting and lag features address this.

::: tip
You can now forecast across time. Chapter 43, **Generative AI**, ties together the generative
models (Ch 36), Transformers/LLMs (Ch 37/39), and diffusion to survey the technology
generating text, images, audio, and video — the most transformative AI of the moment.
:::
