# Data Preprocessing

## Introduction

In Chapter 10 we **cleaned** the data — fixed errors, missing values, and outliers.
But clean data is still not always *model-ready*. Models are mathematical machines:
they only understand **numbers**, and many of them are sensitive to the **scale** of
those numbers. **Data preprocessing** is the step that transforms clean data into
the precise numerical form a model needs.

Imagine comparing two people by "age" (range 20–60) and "salary" (range 30,000–
200,000). To a distance-based model, salary would completely dominate age simply
because its numbers are bigger — not because it's more important. Preprocessing fixes
exactly this kind of problem.

By the end of this chapter you will be able to:

- Understand **why** and **when** feature scaling matters.
- Apply **normalization (Min-Max)** and **standardization (Z-score)** correctly.
- **Encode categorical variables** with label, ordinal, and one-hot encoding.
- Split data into **train and test sets** the right way (and why it must come
  first).
- Understand **pipelines** that bundle all preprocessing into one reproducible
  object.

::: keyidea
**Clean (Ch 10) → Preprocess (Ch 11) → Model.** Cleaning fixes *wrong* data;
preprocessing reshapes *correct* data into the numerical format and scale the
algorithm expects. Skipping preprocessing silently cripples many models.
:::

## Feature scaling: putting features on a level playing field

**Feature scaling** transforms numerical features so they share a comparable range.
It does **not** change the shape of the data — only its scale.

### Why scaling matters (and when)

- **Distance-based models** (KNN, K-Means, SVM) measure distances between points. A
  large-range feature dominates the distance, drowning out others.
- **Gradient-based models** (linear/logistic regression, neural networks) train
  faster and more stably when features are on similar scales (the loss "bowl" from
  Chapter 5 becomes rounder, so gradient descent converges quicker).
- **Tree-based models** (Decision Trees, Random Forests, XGBoost) **do not need
  scaling** — they split on thresholds, so the scale is irrelevant.

::: warning
A frequent beginner question: *"Do I always scale?"* No. Scale for distance-based
and gradient-based models; **skip it for tree-based models**. Knowing this saves you
needless work and is a common interview question.
:::

### Method 1 — Normalization (Min-Max scaling)

**Normalization** squeezes values into a fixed range, usually **[0, 1]**:

<div class="equation"><img class="eq" src="assets/images/eq_ch11_minmax.png" alt="min-max normalization"></div>

The minimum becomes 0, the maximum becomes 1, everything else falls in between. Good
when you need bounded values (e.g. image pixels, neural network inputs) and the data
has no extreme outliers (because the min/max are sensitive to them).

### Method 2 — Standardization (Z-score scaling)

**Standardization** rescales data to have **mean 0 and standard deviation 1** (recall
the z-score from Chapter 6):

<div class="equation"><img class="eq" src="assets/images/eq_ch11_standardize.png" alt="standardization"></div>

The result is *unbounded* but centred. It is **more robust to outliers** than Min-Max
and is the **most common default** for general ML. Use `StandardScaler`.

![Feature scaling does not change the shape of the data, only its axis range. Standardization centres the data at 0 with spread 1; normalization squeezes it into [0, 1].](assets/images/ch11_scaling.png)

| | Min-Max (Normalization) | Z-score (Standardization) |
|---|---|---|
| Output range | [0, 1] (bounded) | mean 0, std 1 (unbounded) |
| Sensitive to outliers? | Yes (uses min/max) | Less so (uses mean/std) |
| Good for | pixels, bounded inputs, neural nets | general ML default |
| scikit-learn | `MinMaxScaler` | `StandardScaler` |

::: warning
**The cardinal rule of scaling:** fit the scaler on the **training data only**, then
*apply* it to the test data. Fitting on all data leaks test information into
training (Chapter 25). In code: `scaler.fit(X_train)` then
`scaler.transform(X_test)`.
:::

## Encoding categorical variables

Models need numbers, but real data has text categories ("Lahore", "red", "Yes").
**Encoding** turns categories into numbers — and choosing the right encoding matters.

### Label / Ordinal encoding — for ordered categories

Assign each category an integer: `low=0, medium=1, high=2`. This is correct **only
when order is meaningful** (ordinal data, Chapter 9), because the model will treat
2 as "greater than" 1.

```python
# ordinal: order matters, so integers make sense
size_map = {"small": 0, "medium": 1, "large": 2}
```

### One-hot encoding — for unordered categories

For **nominal** data (no order), label encoding is *wrong* — it would falsely tell
the model "Karachi (1) > Lahore (0)". Instead, **one-hot encoding** creates a
separate 0/1 column for each category.

![One-hot encoding turns one nominal column into several 0/1 columns — one per category — so the model never sees a false ordering between categories.](assets/images/ch11_onehot.png)

```python
import pandas as pd
df = pd.DataFrame({"city": ["Lahore", "Karachi", "Lahore", "Karachi", "Multan"]})
one_hot = pd.get_dummies(df["city"], prefix="city").astype(int)
print(one_hot.columns.tolist())
print(one_hot.values.tolist())
```

**Output:**
```text
['city_Karachi', 'city_Lahore', 'city_Multan']
[[0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]]
```

Each row now has a 1 in exactly one column. "Lahore" → `[0,1,0]`, "Karachi" →
`[1,0,0]`. No false ordering is implied.

::: warning
**One-hot encoding and high cardinality.** If a column has thousands of categories
(e.g. zip codes), one-hot creates thousands of columns — the "curse of
dimensionality." For such cases prefer **target encoding** or **frequency encoding**
(Chapter 12), or group rare categories together.
:::

## Splitting data into train and test sets

Before training, you must hold out a **test set** the model never sees, so you can
fairly measure generalization (Chapter 2). Do this **early** — ideally right after
loading and cleaning, and *before* fitting any scaler or encoder, to avoid leakage.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
```

- **`test_size=0.2`** — keep 20% for testing (common splits: 70/30, 80/20).
- **`random_state`** — reproducible split.
- **`stratify=y`** — keep the same class proportions in train and test (important for
  imbalanced classification — e.g. ensures both sets have the rare class).

## Practical: a complete preprocessing flow

Let's preprocess a small mixed dataset (numbers + a category), the way you will in
real projects.

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.DataFrame({
    "age":    [25, 32, 41, 28, 38],
    "salary": [50000, 85000, 62000, 90000, 58000],
    "city":   ["Lahore", "Karachi", "Lahore", "Karachi", "Multan"],
})

# --- 1) Normalize age to [0, 1] ---
age_norm = MinMaxScaler().fit_transform(df[["age"]]).ravel()
print("MinMax age:", np.round(age_norm, 3).tolist())

# --- 2) Standardize salary (mean 0, std 1) ---
sal_std = StandardScaler().fit_transform(df[["salary"]]).ravel()
print("Standardized salary:", np.round(sal_std, 3).tolist())

# --- 3) One-hot encode the city ---
one_hot = pd.get_dummies(df["city"], prefix="city").astype(int)
print("One-hot columns:", one_hot.columns.tolist())

# --- 4) Train/test split (do this early in real projects) ---
X, y = df[["age", "salary"]], [0, 1, 0, 1, 0]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42)
print("train rows:", len(X_train), "test rows:", len(X_test))
```

**Output:**
```text
MinMax age: [0.0, 0.438, 1.0, 0.188, 0.812]
Standardized salary: [-1.212, 1.021, -0.447, 1.34, -0.702]
One-hot columns: ['city_Karachi', 'city_Lahore', 'city_Multan']
train rows: 3 test rows: 2
```

### Explanation

- **(1) Min-Max** mapped the youngest age (25) to `0.0`, the oldest (41) to `1.0`,
  and the rest in between — now bounded in [0, 1].
- **(2) Standardization** centred salary at 0: negative values are below-average
  salaries, positive are above. The big-earner (90,000) has the highest z-score
  (1.34).
- **(3) One-hot** turned the single `city` text column into three 0/1 columns — no
  false ordering.
- **(4)** We split into 3 training and 2 test rows. In a real project, you'd fit the
  scalers/encoders on `X_train` only.

::: keyidea
Notice age and salary started on *wildly* different scales (tens vs tens-of-
thousands). After scaling, both contribute fairly to any distance- or gradient-based
model. This single step often makes the difference between a model that works and one
that doesn't.
:::

## Pipelines: bundling preprocessing with the model

Doing scaling, encoding, and modelling as separate steps is error-prone — especially
keeping train and test handling identical. scikit-learn's **`Pipeline`** and
**`ColumnTransformer`** bundle all steps into one object that does the right thing
automatically (and prevents leakage).

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

# Apply different preprocessing to numeric vs categorical columns
preprocess = ColumnTransformer([
    ("num", StandardScaler(), ["age", "salary"]),
    ("cat", OneHotEncoder(),  ["city"]),
])

# Chain preprocessing + model into ONE object
model = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression()),
])
# model.fit(X_train, y_train)  # scales, encodes, and trains — all correctly
```

::: tip
**Why pipelines are a best practice:** (1) They apply the *exact same* transformations
to training, test, and future production data. (2) They fit transformers on training
data only, preventing leakage automatically. (3) They make your work reproducible and
deployable (Part VIII). Get comfortable with pipelines early — professionals use them
everywhere.
:::

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Fitting the scaler on the whole dataset (including test).** This leaks
test information into training. Always `fit` on train only.
:::

- **Mistake 2 — Label-encoding nominal data.** Turning "red/green/blue" into 0/1/2
  invents a false ordering. Use one-hot for nominal data.
- **Mistake 3 — Scaling tree-based models** unnecessarily (it doesn't help them).
- **Mistake 4 — One-hot encoding very high-cardinality columns**, exploding the
  feature count.
- **Mistake 5 — Splitting *after* preprocessing.** Split first (or use a pipeline) so
  test data stays truly unseen.
- **Mistake 6 — Forgetting to apply the *same* preprocessing to new/production data.**

## Best practices

- **Split first**, then fit preprocessing on training data only.
- **Standardize by default**; use Min-Max for bounded inputs like pixels.
- **One-hot for nominal, ordinal/label encoding only for truly ordered categories.**
- **Skip scaling for tree-based models.**
- **Use `Pipeline` + `ColumnTransformer`** to bundle everything reproducibly.
- **Use `stratify`** for imbalanced classification splits.

## Chapter Summary

- **Preprocessing** reshapes clean data into the numeric form and scale models need.
- **Feature scaling:** **normalization (Min-Max → [0,1])** for bounded inputs;
  **standardization (Z-score → mean 0, std 1)** as the robust default. Needed for
  **distance-based** and **gradient-based** models; **not** for **tree-based** models.
- **Encoding:** **label/ordinal** encoding for ordered categories; **one-hot**
  encoding for unordered (nominal) categories to avoid false ordering; watch out for
  high cardinality.
- **Split into train/test early** (`train_test_split`, with `stratify` for imbalanced
  classes), and **fit all transformers on training data only** to prevent leakage.
- **Pipelines** (`Pipeline` + `ColumnTransformer`) bundle preprocessing and modelling
  into one reproducible, leak-proof object.

---

::: {.qband}
Practice Zone — Chapter 11
:::

## Multiple-Choice Questions (MCQs)

**Q1.** Standardization rescales data to have:
a) Range [0, 1]  b) Mean 0 and std 1  c) Sum 1  d) Max 100

**Q2.** Which model type does NOT require feature scaling?
a) KNN  b) SVM  c) Neural network  d) Decision Tree

**Q3.** Nominal categories like city names should be encoded with:
a) Label encoding  b) One-hot encoding  c) Standardization  d) Min-Max scaling

**Q4.** Min-Max scaling maps the minimum value to:
a) 1  b) 0  c) −1  d) the mean

**Q5.** You must fit a scaler on:
a) The test set  b) The training set only  c) All data  d) Random data

**Q6.** `stratify=y` in `train_test_split` ensures:
a) Faster training  b) Same class proportions in train and test  c) No scaling
d) Random shuffling off

**Q7.** One-hot encoding a column with 5,000 unique values causes:
a) Better accuracy always  b) Thousands of new columns (curse of dimensionality)
c) Faster training  d) Nothing

**Q8.** A `ColumnTransformer` is used to:
a) Split data  b) Apply different preprocessing to different columns  c) Train models
d) Plot data

### MCQ Answers
**1:** b. **2:** d. **3:** b. **4:** b. **5:** b. **6:** b. **7:** b. **8:** b.

## Interview Questions (with answers)

**Q1. What is the difference between normalization and standardization, and when do
you use each?**
*Answer:* Normalization (Min-Max) rescales to a fixed range like [0,1] and is
sensitive to outliers; use it for bounded inputs such as image pixels or neural-net
inputs. Standardization (Z-score) rescales to mean 0, std 1, is more robust to
outliers, and is the common default for general ML, especially distance- and
gradient-based models.

**Q2. Which algorithms need feature scaling and which don't?**
*Answer:* Distance-based (KNN, K-Means, SVM) and gradient-based (linear/logistic
regression, neural networks) models need scaling. Tree-based models (Decision Trees,
Random Forests, gradient boosting) do not, because they split on thresholds and are
invariant to monotonic scaling.

**Q3. Why is one-hot encoding preferred over label encoding for nominal data?**
*Answer:* Label encoding assigns integers that imply an order/magnitude (e.g.
blue=2 > red=0), which is false for unordered categories and misleads the model.
One-hot encoding creates independent 0/1 columns with no implied ordering.

**Q4. What is data leakage in preprocessing, and how do you prevent it?**
*Answer:* Leakage is when information from the test set influences training — e.g.
fitting a scaler or imputer on the whole dataset. Prevent it by splitting first and
fitting all transformers on training data only (then transforming test data), ideally
via a `Pipeline`.

**Q5. Why use a scikit-learn Pipeline?**
*Answer:* It chains preprocessing and modelling into one object that applies identical
transformations to train, test, and production data, fits transformers on training
data only (preventing leakage), and is reproducible and easy to deploy.

## Scenario-Based Questions (with answers)

**Q1.** *Your KNN model performs terribly. You notice features include age (20–60)
and income (20,000–500,000), unscaled. What's likely wrong and how do you fix it?*
*Answer:* KNN uses distances, so the large-range income dominates the distance,
making age irrelevant. Standardize (or normalize) the features so each contributes
fairly, then retrain — accuracy should improve substantially.

**Q2.** *A colleague label-encodes "color" as red=0, green=1, blue=2 for a linear
model and gets odd results. Why?*
*Answer:* The model interprets the codes as quantities, so it thinks blue (2) is
"twice" green (1) and ordered above red — a meaningless relationship for nominal
colors. Use one-hot encoding instead so no false order is implied.

**Q3.** *Your model scores 95% in testing but 70% in production. You scaled features
by fitting the scaler on the entire dataset before splitting. Could that be the
cause?*
*Answer:* Yes — that's data leakage. The scaler "saw" the test data's statistics, so
test performance was optimistic. In production, only training-derived statistics
exist. Fix by fitting the scaler on training data only (use a pipeline).

## Logic-Based Questions (with answers)

**Q1.** After Min-Max scaling, what are the values of the original minimum and maximum
of a feature?
*Answer:* The minimum becomes 0 and the maximum becomes 1, by definition of the
Min-Max formula.

**Q2.** Why does scaling not affect a Decision Tree's splits?
*Answer:* A tree splits on thresholds (e.g. "age > 30"). Any monotonic rescaling just
moves the threshold correspondingly, producing the same partition of the data, so the
tree's decisions are unchanged.

**Q3.** One-hot encoding a 3-category column adds how many columns, and why might you
drop one?
*Answer:* It adds 3 columns (one per category). You may drop one ("drop first")
because the dropped category is implied when all others are 0, avoiding redundancy
(useful for linear models to prevent multicollinearity).

## Practical Questions (with answers)

**Q1.** Write code to standardize a training set and apply the same transform to a
test set without leakage.
*Answer:*
```python
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)   # uses TRAIN statistics
```

**Q2.** In the practical, why did standardized salary contain negative numbers?
*Answer:* Standardization subtracts the mean, so salaries below the average become
negative and those above become positive; the magnitude is in standard-deviation
units.

**Q3.** Write one line to one-hot encode a `"color"` column with Pandas, as integers.
*Answer:* `pd.get_dummies(df["color"], prefix="color").astype(int)`.

## Long Questions (with answers)

**Q1. Explain feature scaling: the two main methods, the formulas, when each is
appropriate, which models need it, and the leakage rule.**

*Answer:* Feature scaling puts numerical features on comparable ranges without
changing their shape. **Normalization (Min-Max)** uses x' = (x − min)/(max − min) to
map values into [0, 1]; it is bounded and intuitive but sensitive to outliers (which
set the min/max), so it suits bounded inputs like image pixels and neural-network
inputs. **Standardization (Z-score)** uses x' = (x − μ)/σ to produce mean 0 and
standard deviation 1; it is unbounded but more robust to outliers and is the common
default for general ML. Scaling matters for **distance-based** models (KNN, K-Means,
SVM), where a large-range feature would dominate the distance, and for
**gradient-based** models (linear/logistic regression, neural networks), where it
speeds and stabilizes convergence by making the loss surface rounder. **Tree-based**
models (Decision Trees, Random Forests, gradient boosting) do **not** need scaling,
because they split on thresholds and are invariant to monotonic rescaling. Crucially,
to avoid **data leakage**, fit the scaler on the **training data only** and apply the
learned parameters to validation, test, and production data; fitting on all data lets
test statistics influence training and produces optimistic, unreliable results.

**Q2. Explain categorical encoding: the methods, when each is correct, and the
problems that arise from choosing wrongly.**

*Answer:* Models require numbers, so categorical (text) features must be encoded.
**Label/ordinal encoding** assigns an integer to each category (low=0, medium=1,
high=2); this is correct **only for ordinal data** where the order is meaningful,
because the model treats the integers as ordered magnitudes. **One-hot encoding**
creates a separate 0/1 column per category and is correct for **nominal data** (no
order), since it implies no ranking — "Karachi" becomes [1,0,0] and "Lahore" [0,1,0],
which the model cannot misread as one being greater than the other. Choosing wrongly
causes real harm: label-encoding nominal data invents a false ordering (the model may
conclude blue > red), distorting linear and distance-based models; while one-hot
encoding a very high-cardinality column (thousands of unique values like zip codes)
explodes the feature count, causing the curse of dimensionality, slow training, and
overfitting. For high cardinality, alternatives like target encoding, frequency
encoding, or grouping rare categories are preferred. The rule of thumb: ordinal →
ordinal/label encoding; low-cardinality nominal → one-hot; high-cardinality nominal →
target/frequency encoding.

## Exercises

1. For each, choose the scaling method and justify: pixel values (0–255), a salary
   column with extreme outliers, neural-network inputs.
2. Decide the encoding for: T-shirt size (S/M/L), country, satisfaction
   (low/med/high), favourite colour.
3. Explain in two sentences why fitting a scaler on the test set is wrong.
4. A column "grade" has values A, B, C, D, F. Which encoding preserves meaning, and
   what integer order would you use?
5. Why don't Random Forests need feature scaling?

## Mini-Project

**Project: Preprocess a real mixed dataset.**

1. Take a dataset with both numeric and categorical columns (e.g. Titanic).
2. Split into train/test *first*.
3. Build a `ColumnTransformer` that standardizes the numeric columns and one-hot
   encodes the categorical ones.
4. Fit it on the training set, transform both sets, and print the resulting shapes.
5. Write 4–5 sentences explaining each choice and how the pipeline prevents leakage.

## Assignments

1. **Coding:** Take age `[20, 25, 30, 35, 40]`. Apply both Min-Max and Standard
   scaling by hand (using the formulas) and verify with scikit-learn.
2. **Coding:** Build a full `Pipeline` (preprocessing + `LogisticRegression`) on any
   classification dataset, fit on train, and report test accuracy. Show that the same
   pipeline transforms new data correctly.
3. **Conceptual:** Write one page on data leakage in preprocessing: what it is, three
   ways it happens, and how pipelines prevent it.

::: tip
Cleaning (Ch 10) and preprocessing (Ch 11) prepare *existing* features. Next, in
Chapter 12, **feature engineering** *creates new, more powerful features* — often the
single biggest lever on model performance.
:::
