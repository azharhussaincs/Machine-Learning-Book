# Linear Regression

## Introduction

Meet your first real algorithm in depth. **Linear Regression** is the "Hello, World!"
of Machine Learning — the simplest, most fundamental model for predicting a **number**.
Despite its simplicity, it is used everywhere: predicting house prices, sales,
temperatures, medical measurements, and more. Master it deeply and you'll understand
ideas (loss, training, coefficients, evaluation) that carry into *every* later
algorithm, including neural networks.

The core idea is one you've seen since school: **fit a straight line through points.**
Linear regression finds the line that best captures the relationship between inputs and
a numeric output, then uses that line to predict.

::: keyidea
Linear regression assumes the output is (approximately) a **weighted sum of the
inputs**. Its job is to find the best weights. Simple, fast, and interpretable — and
the foundation for logistic regression, neural networks, and much more.
:::

By the end of this chapter you will be able to:

- Explain the **intuition** and **mathematics** of linear regression.
- Understand its **cost function** (MSE) and two ways to train it (**normal equation**
  and **gradient descent**).
- Evaluate regression with **MAE, MSE, RMSE, and R²**.
- Implement it **from scratch** and with **scikit-learn**.
- Interpret coefficients and know the model's **assumptions, pros, and cons**.

## Intuition: the best-fit line

Imagine plotting house *size* (x) against *price* (y). The points roughly follow an
upward trend. Linear regression draws the single straight line that sits "best" among
them — closest to all points on average.

![Linear regression fits the line that minimises the total squared vertical distance (the red "residuals") between each point and the line. That line is the model.](assets/images/ch17_best_fit.png)

The vertical distance from each point to the line is the **residual** (the error for
that point). The "best" line is the one that makes these errors as small as possible —
specifically, the smallest **sum of squared residuals**.

## The mathematics

### Simple linear regression (one feature)

With one input, the model is the equation of a line:

<div class="equation"><img class="eq" src="assets/images/eq_ch17_simple.png" alt="simple linear regression"></div>

- **w** (the **weight** or *slope*) — how much `y` changes when `x` increases by 1.
- **b** (the **bias** or *intercept*) — the predicted value when `x = 0`.

### Multiple linear regression (many features)

With several inputs, it becomes a weighted sum (a *plane* or *hyperplane*):

<div class="equation"><img class="eq" src="assets/images/eq_ch17_multiple.png" alt="multiple linear regression"></div>

This is exactly the **dot product** from Chapter 5: `ŷ = w·x + b`. Each weight says how
strongly its feature pushes the prediction up or down.

### The cost function: Mean Squared Error

To find the best weights, we need to measure "how wrong" a line is. Linear regression
uses **Mean Squared Error (MSE)** — the average squared residual (Chapter 5):

<div class="equation"><img class="eq" src="assets/images/eq_ch05_mse.png" alt="mean squared error"></div>

Training = finding the `w` and `b` that **minimise** this cost. There are two ways.

### Training method 1 — The Normal Equation (exact solution)

For linear regression, calculus gives a *direct, exact* formula for the best weights —
no iteration needed:

<div class="equation"><img class="eq" src="assets/images/eq_ch17_normal_eq.png" alt="normal equation"></div>

This uses the matrix operations (transpose `Xᵀ`, inverse `⁻¹`, multiplication) from
Chapter 5. It's exact and fast for modest numbers of features, but inverting the matrix
becomes slow/unstable when there are very many features.

### Training method 2 — Gradient Descent (iterative)

For large datasets or many features, we use **gradient descent** (Chapter 5): start
with random weights, compute the gradient of MSE, step downhill, repeat. You already
implemented this in Chapter 5 — that *was* linear regression training.

::: note
**Two roads, same destination.** The normal equation jumps straight to the answer with
linear algebra; gradient descent walks there step by step. Small problems → normal
equation. Large problems (or neural networks, where no formula exists) → gradient
descent.
:::

## Evaluating a regression model

For regression we measure *how far off* predictions are. Four standard metrics:

- **MAE (Mean Absolute Error)** — average absolute error; easy to interpret, robust to
  outliers.

<div class="equation"><img class="eq" src="assets/images/eq_ch17_mae.png" alt="MAE"></div>

- **MSE (Mean Squared Error)** — average squared error; punishes big errors more.
- **RMSE (Root Mean Squared Error)** — square root of MSE, back in original units (most
  popular).

<div class="equation"><img class="eq" src="assets/images/eq_ch17_rmse.png" alt="RMSE"></div>

- **R² (R-squared, coefficient of determination)** — the *fraction of variance
  explained* by the model; ranges from 1 (perfect) down through 0 (no better than
  predicting the mean) and can go negative (worse than the mean).

<div class="equation"><img class="eq" src="assets/images/eq_ch17_r2.png" alt="R-squared"></div>

::: tip
**Which metric?** Use **RMSE** when large errors are especially bad and you want units;
use **MAE** when you want a robust, equally-weighted average error; use **R²** to
communicate "how much of the variation we explain" (e.g. R²=0.45 means we explain 45%
of the variance). Report at least one error metric *and* R².
:::

## Implementation from scratch (the normal equation)

Let's solve linear regression directly with the normal equation on tiny data.

```python
import numpy as np

X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 5, 4, 5], dtype=float)

# Add a column of 1s so the bias b is learned alongside the weight w
X_b = np.c_[np.ones(len(X)), X]          # design matrix: [1, x] per row

# Normal equation: w = (XᵀX)⁻¹ Xᵀ y
w = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
print("Learned [b, w]:", np.round(w, 3).tolist())
```

**Output:**
```text
Learned [b, w]: [2.2, 0.6]
```

The model learned `b = 2.2`, `w = 0.6`, i.e. `ŷ = 0.6·x + 2.2`. The matrix algebra from
Chapter 5 solved the whole problem in one line — no iteration.

## Implementation with scikit-learn (real data)

Now the practical way, on the real **diabetes** dataset (10 health features predicting
disease progression).

```python
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

X, y = load_diabetes(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression().fit(X_tr, y_tr)   # trains via the normal equation
pred = model.predict(X_te)

print("R2:  ", round(r2_score(y_te, pred), 3))
print("RMSE:", round(np.sqrt(mean_squared_error(y_te, pred)), 1))
print("MAE: ", round(mean_absolute_error(y_te, pred), 1))
print("intercept:", round(model.intercept_, 1))
print("n_coefs:", len(model.coef_))
```

**Output:**
```text
R2:   0.453
RMSE: 53.9
MAE:  42.8
intercept: 151.3
n_coefs: 10
```

### Explanation

- **`LinearRegression().fit(...)`** found the 10 weights + intercept that minimise MSE.
- **R² = 0.453** — the model explains about **45%** of the variance in disease
  progression. Modest, but meaningful for this hard medical problem.
- **RMSE = 53.9** — predictions are off by ~54 units on average (in the target's
  units). **MAE = 42.8** — the typical absolute error.
- **intercept = 151.3** is the predicted value when all (standardised) features are at
  their mean; the **10 coefficients** weight each health feature.

::: keyidea
Linear regression gave us not just predictions but **interpretation**: each coefficient
tells us how each health factor relates to disease progression, and R² tells us how
much we explain. This transparency — knowing *why* the model predicts what it does — is
linear regression's superpower and why it remains a first choice in medicine, finance,
and science.
:::

::: tip
**Practical & debugging tips:** (1) **Scale features** if you'll interpret/compare
coefficients or use gradient descent (Chapter 11). (2) Check **R² on the test set**, not
train — a high train R² with low test R² means overfitting. (3) For non-linear
relationships, add **polynomial features** (Chapter 12) — "polynomial regression" is
just linear regression on `x, x², x³…`. (4) If features are highly correlated
(multicollinearity), coefficients become unstable — use **Ridge/Lasso** (Chapter 26).
(5) Plot residuals; a pattern in them means the linear assumption is violated.
:::

## Assumptions of linear regression

Linear regression works best when these hold (violations don't always break it, but
they hurt):

1. **Linearity** — the relationship between features and target is roughly linear.
2. **Independence** — observations are independent of each other.
3. **Homoscedasticity** — the error spread is roughly constant across predictions.
4. **Normality of residuals** — errors are roughly normally distributed.
5. **Little multicollinearity** — features aren't strongly correlated with each other.

## Advantages, disadvantages, and use cases

| Advantages | Disadvantages |
|---|---|
| Simple, fast to train | Assumes a linear relationship |
| Highly interpretable (coefficients) | Sensitive to outliers |
| No hyperparameters (basic form) | Underfits complex/non-linear data |
| Works with little data | Hurt by multicollinearity |
| Strong baseline | Limited flexibility |

**Use cases:** house/sales/price prediction, demand forecasting, risk and trend
analysis, medical and scientific modelling, and as a **baseline** for any regression
problem.

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Forcing linear regression on clearly non-linear data.** If a scatter plot
shows a curve, either add polynomial features or use a non-linear model. A straight line
will underfit (high bias).
:::

- **Mistake 2 — Interpreting coefficients without scaling**, making them
  incomparable.
- **Mistake 3 — Ignoring outliers**, which can drastically tilt the line (MSE squares
  their effect).
- **Mistake 4 — Trusting a high training R²** that doesn't hold on test data.
- **Mistake 5 — Confusing correlation/coefficients with causation** (Chapter 6).
- **Mistake 6 — Multicollinearity** making individual coefficients meaningless.

## Best practices

- **Plot the data and residuals** to check the linearity assumption.
- **Scale features** when interpreting or comparing coefficients.
- **Use RMSE/MAE and R² together**, on the test set.
- **Add polynomial features** for gentle non-linearity; **regularize** (Ridge/Lasso)
  when features are many or correlated.
- **Treat linear regression as your baseline** before trying complex models.

## Chapter Summary

- **Linear regression** predicts a number as a **weighted sum of features** (`ŷ = w·x +
  b`); training finds the weights that minimise **MSE**.
- Two training methods: the **normal equation** (exact, via matrix algebra — best for
  modest features) and **gradient descent** (iterative — best for large data).
- Evaluate with **MAE**, **MSE**, **RMSE** (units, popular), and **R²** (variance
  explained).
- On the diabetes data, the model explained **45% of variance (R²=0.453)** with
  RMSE≈53.9 — and its **coefficients are interpretable**.
- It's simple, fast, and interpretable, but assumes linearity and is sensitive to
  outliers and multicollinearity — a perfect **baseline** and the foundation for later
  algorithms.

---

::: {.qband}
Practice Zone — Chapter 17
:::

## Multiple-Choice Questions (MCQs)

**Q1.** Linear regression predicts:
a) A category  b) A continuous number  c) A cluster  d) A probability only

**Q2.** In `ŷ = wx + b`, `b` is the:
a) Slope  b) Intercept (bias)  c) Error  d) Feature

**Q3.** The cost function minimised by ordinary linear regression is:
a) Accuracy  b) Mean Squared Error  c) Cross-entropy  d) R²

**Q4.** The normal equation is:
a) `w = Xy`  b) `w = (XᵀX)⁻¹Xᵀy`  c) `w = X⁻¹y`  d) `w = XᵀX`

**Q5.** An R² of 0.45 means the model:
a) Is 45% accurate  b) Explains 45% of the variance  c) Has 45% error  d) Is overfit

**Q6.** Which metric is in the same units as the target and popular for reporting?
a) MSE  b) R²  c) RMSE  d) Accuracy

**Q7.** Linear regression on clearly curved data will:
a) Overfit  b) Underfit (high bias)  c) Be perfect  d) Crash

**Q8.** "Polynomial regression" is:
a) A different algorithm  b) Linear regression on x, x², x³, …  c) Classification
d) Clustering

### MCQ Answers
**1:** b. **2:** b. **3:** b. **4:** b. **5:** b. **6:** c. **7:** b. **8:** b.

## Interview Questions (with answers)

**Q1. Explain how linear regression works and how it's trained.**
*Answer:* It models the target as a weighted sum of features plus a bias (`ŷ = w·x +
b`) and finds the weights that minimise the Mean Squared Error between predictions and
true values. Training uses either the normal equation `(XᵀX)⁻¹Xᵀy` for an exact
solution, or gradient descent for large data, iteratively stepping downhill on the MSE.

**Q2. What is R² and what does a negative R² mean?**
*Answer:* R² is the fraction of the target's variance explained by the model; 1 is
perfect and 0 means no better than predicting the mean. A negative R² means the model
performs *worse* than simply predicting the mean — a sign of a poor or mis-specified
model.

**Q3. When would you use the normal equation vs gradient descent?**
*Answer:* The normal equation gives an exact solution and is convenient for modest
numbers of features, but matrix inversion is O(n³) and unstable with many features.
Gradient descent scales to large datasets and high dimensions and is necessary when no
closed-form solution exists (e.g. neural networks).

**Q4. What are the key assumptions of linear regression?**
*Answer:* Linearity (linear feature–target relationship), independence of observations,
homoscedasticity (constant error variance), approximately normal residuals, and low
multicollinearity among features. Violations reduce reliability of predictions and
coefficient interpretation.

**Q5. Why is linear regression sensitive to outliers?**
*Answer:* It minimises *squared* errors, so a far-off point contributes a very large
squared residual, pulling the fitted line toward it disproportionately. Robust methods
or outlier handling (Chapter 10) mitigate this.

## Scenario-Based Questions (with answers)

**Q1.** *Your linear model has high training R² but low test R². What's happening and
what do you do?*
*Answer:* Overfitting (often from too many/collinear features or polynomial terms of
high degree). Reduce complexity, apply regularization (Ridge/Lasso), remove redundant
features, or gather more data, and re-evaluate on the test set.

**Q2.** *A scatter plot of your data shows a clear curve, but linear regression
underfits. How do you keep using a linear model?*
*Answer:* Add polynomial/interaction features (e.g. x², Chapter 12), turning it into
polynomial regression — still linear in the parameters but able to fit curves. Validate
the degree to avoid overfitting.

**Q3.** *Two of your features are 0.95 correlated, and the coefficients look
nonsensical (huge, opposite signs). What's the issue and fix?*
*Answer:* Multicollinearity makes individual coefficients unstable and hard to
interpret. Fix by removing one of the correlated features (Chapter 13), combining them,
or using Ridge regression, which stabilises coefficients under collinearity.

## Logic-Based Questions (with answers)

**Q1.** Why does the normal equation add a column of ones to X?
*Answer:* To learn the bias/intercept `b` as just another weight. The column of ones
multiplies `b`, so the single matrix formula yields both the slope weights and the
intercept together.

**Q2.** If a model's RMSE is larger than its MAE, what does that imply about the
errors?
*Answer:* RMSE ≥ MAE always, and a notably larger RMSE implies some large errors
(outliers), because squaring inflates big residuals more than small ones. A big gap
signals a few large mistakes.

**Q3.** A model predicting the mean for every input gets R² = 0. Why?
*Answer:* R² compares the model's squared error to the variance around the mean. If the
model *is* the mean, its error equals that variance, so the ratio is 1 and R² = 1 − 1 =
0 — explaining none of the variation.

## Practical Questions (with answers)

**Q1.** Write code to compute RMSE from true `y` and predictions `p`.
*Answer:* `np.sqrt(mean_squared_error(y, p))` (from `sklearn.metrics`).

**Q2.** How do you read off a fitted scikit-learn model's slope and intercept?
*Answer:* `model.coef_` gives the weights (slopes) and `model.intercept_` gives the
bias/intercept.

**Q3.** How would you turn linear regression into polynomial regression of degree 2 in
scikit-learn?
*Answer:* Use a pipeline: `make_pipeline(PolynomialFeatures(degree=2),
LinearRegression())`.

## Long Questions (with answers)

**Q1. Explain linear regression end to end: the model, the cost function, the two
training methods, and how to evaluate the result.**

*Answer:* **The model** assumes the target is a weighted sum of the features plus a
bias: ŷ = w₁x₁ + … + wₙxₙ + b, equivalently the dot product w·x + b; geometrically it
fits a line (one feature) or hyperplane (many features). **The cost function** is Mean
Squared Error, the average of squared residuals (differences between predictions and
true values); squaring keeps errors positive and penalises large mistakes more, and the
best parameters are those that minimise it. **Training** can be done two ways: the
**normal equation**, w = (XᵀX)⁻¹Xᵀy, gives the exact minimiser directly via linear
algebra and is convenient for modest feature counts, but matrix inversion scales poorly
(≈O(n³)) and is unstable with many or collinear features; **gradient descent** instead
starts from initial weights and iteratively steps opposite the gradient of MSE
(Chapter 5), scaling to large, high-dimensional data and generalising to models with no
closed form. **Evaluation** uses error metrics — MAE (robust average absolute error),
MSE, and RMSE (in target units, the popular default) — together with R², the fraction
of variance explained (1 perfect, 0 equals predicting the mean, negative is worse). One
always evaluates on a held-out test set to measure generalisation, and inspects
residuals to validate the linearity assumption.

**Q2. Discuss the strengths, weaknesses, and assumptions of linear regression, and when
it is (and isn't) the right tool.**

*Answer:* **Strengths:** linear regression is simple, fast to train, needs little data,
and is highly **interpretable** — each coefficient quantifies how a feature relates to
the target, which is invaluable in medicine, finance, and science where explanations
matter. It has essentially no hyperparameters in basic form and makes an excellent
**baseline**. **Weaknesses:** it assumes a linear relationship and therefore *underfits*
genuinely non-linear data (high bias); it is **sensitive to outliers** because it
minimises squared errors; and it suffers from **multicollinearity**, where strongly
correlated features make coefficients unstable and uninterpretable. **Assumptions:**
linearity, independence of observations, homoscedasticity (constant error variance),
approximately normal residuals, and low multicollinearity; violations degrade both
predictions and coefficient reliability. It is the **right tool** when relationships are
roughly linear, interpretability is valued, data is limited, or you need a quick,
trustworthy baseline. It is the **wrong tool** when relationships are strongly
non-linear (use trees, kernel methods, or neural networks, or add polynomial features),
when outliers dominate (clean them or use robust methods), or when features are highly
collinear (use Ridge/Lasso or feature selection). Knowing these boundaries lets you
deploy linear regression where it shines and reach for richer models when it doesn't.

## Exercises

1. Write the equation of simple linear regression and label every symbol.
2. For predictions vs truth `(pred, true)`: (3,2), (5,5), (4,6) — compute MAE and MSE by
   hand.
3. Explain in your own words what R² = 0.8 means.
4. List the five assumptions of linear regression.
5. When would you prefer the normal equation over gradient descent, and vice versa?

## Mini-Project

**Project: Predict house prices.**

1. Load a housing dataset (e.g. scikit-learn's California housing, or any CSV with a
   numeric target).
2. Do quick EDA (Chapter 15), handle missing values, and scale features.
3. Train a `LinearRegression` model; report R², RMSE, and MAE on the test set.
4. Print and interpret the top 3 coefficients (which features matter most and in which
   direction?).
5. Add polynomial features (degree 2) and compare — did it help or overfit? Save in
   `my-ml-journey/`.

## Assignments

1. **Coding:** Implement linear regression from scratch using **both** the normal
   equation and gradient descent on the same data, and confirm they give similar
   weights.
2. **Coding:** On the diabetes dataset, scale the features, refit, and compare which
   features have the largest (absolute) coefficients. Interpret the top two.
3. **Conceptual:** Write one page comparing MAE, RMSE, and R², including when each is
   most appropriate and how outliers affect them.

::: tip
You've mastered predicting *numbers*. Chapter 18, **Logistic Regression**, makes one
clever change — squashing the linear output through a sigmoid — to predict
*probabilities* and *categories*, opening the door to classification.
:::
