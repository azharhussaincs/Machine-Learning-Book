# Data Cleaning

## Introduction

Here is a hard truth every professional learns: **real-world data is messy.** It has
missing values, duplicate rows, typos, inconsistent spellings, impossible numbers
(an age of 200!), wrong data types, and outliers. Raw data is almost never ready for
a model.

**Data cleaning** is the process of fixing these problems. It is unglamorous, it is
where ML practitioners spend most of their time, and it is *absolutely critical*. The
golden rule of computing applies in full force here:

::: keyidea
**Garbage In, Garbage Out (GIGO).** No algorithm — however advanced — can produce
good results from bad data. A simple model on clean data beats a fancy model on dirty
data, every time. Cleaning is not a chore you rush through; it *is* the work.
:::

By the end of this chapter you will be able to:

- Recognise the common kinds of "dirty" data.
- **Detect and handle missing values** with the right strategy for the situation.
- Find and remove **duplicate** records.
- Detect and treat **outliers** using the IQR and z-score methods.
- Fix **inconsistent text** and **wrong data types**.
- Clean a realistically messy dataset end to end.

## The common kinds of dirty data

| Problem | Example | Why it's harmful |
|---|---|---|
| **Missing values** | empty cells, `NaN` | Break calculations and many models |
| **Duplicates** | the same record twice | Bias results, inflate counts |
| **Outliers** | age = 200, salary = −5 | Distort averages and models |
| **Inconsistent text** | "Lahore", "lahore", "Lahore " | Treated as different categories |
| **Wrong data types** | numbers stored as text | Block maths and modelling |
| **Structural errors** | wrong column, shifted rows | Corrupt the whole analysis |

## Handling missing values

Missing values (`NaN`, `None`, blanks) are the most common data problem. They occur
because of data-entry errors, optional survey fields, sensor failures, merging
mismatched sources, and more.

### Step 1 — Detect them

```python
df.isnull().sum()      # count of missing values per column
df.info()              # shows non-null counts per column
```

### Step 2 — Decide on a strategy

There are two broad strategies — **remove** or **impute** (fill in) — and the right
choice depends on *how much* is missing and *why*.

![Strategies for missing data. If little is missing, dropping rows is fine; otherwise impute (fill) with a statistic or a model. Drop a whole column only if most of it is missing.](assets/images/ch10_missing_strategies.png)

**A) Remove (drop):**

- **Drop rows** (`df.dropna()`) — fine when only a *small* fraction of rows have
  missing values. You lose data, so use sparingly.
- **Drop a column** (`df.drop(columns=...)`) — only if *most* of the column is
  missing (e.g. >50–70%) and it's not crucial.

**B) Impute (fill in) with `fillna()`:**

- **Mean** — for roughly symmetric numerical data.
- **Median** — for skewed numerical data or when outliers exist (safer default).
- **Mode** (most frequent) — for categorical data.
- **Forward/backward fill** — for time series (carry the last value forward).
- **A constant** (e.g. `0` or `"Unknown"`) — when missingness itself is meaningful.
- **Model-based** (e.g. `KNNImputer`) — predict the missing value from other columns
  (advanced).

::: warning
**Don't blindly drop or fill.** Ask *why* the value is missing. Sometimes
"missing" carries information (e.g. a blank "income" field might mean "unemployed").
And imputing with the mean on skewed data, or dropping 40% of your rows, can do more
harm than the missingness itself.
:::

## Handling duplicates

Duplicate rows bias your analysis — they over-count whatever they represent.

```python
df.duplicated().sum()        # how many duplicate rows?
df = df.drop_duplicates()    # remove them (keeps the first occurrence)
```

Sometimes duplicates are only partial (same person, different ID). You can check
specific columns: `df.drop_duplicates(subset=["name", "email"])`.

## Handling outliers

An **outlier** is a value far outside the normal range. Some outliers are genuine
(a real billionaire); others are errors (age = 200). Either way, they can wreck
averages and many models, so you must find and handle them deliberately.

### Detecting outliers

**Method 1 — The IQR rule (robust, recommended).** Recall the IQR from Chapter 6.
Anything below `Q1 − 1.5×IQR` or above `Q3 + 1.5×IQR` is flagged as an outlier. This
is exactly what a **box plot** shows.

![A box plot showing the IQR method. The box spans Q1 to Q3; the "whiskers" reach 1.5×IQR; points beyond the whiskers (like the dot on the right) are outliers.](assets/images/ch10_outliers.png)

**Method 2 — The z-score rule.** Recall z-scores from Chapter 6. A value with
`|z| > 3` (more than 3 standard deviations from the mean) is often treated as an
outlier. Best for roughly normal data.

### Treating outliers

- **Remove** them (if they are clearly errors).
- **Cap / clip** them to a maximum/minimum reasonable value (called "winsorising").
- **Transform** the feature (e.g. a log transform, Chapter 12) to reduce their pull.
- **Keep** them (if they are genuine and important — e.g. in fraud detection the
  outliers *are* the target!).

## Fixing inconsistent text and wrong types

- **Whitespace and case:** `" Sara"`, `"sara"`, and `"SARA"` look different to a
  computer. Standardise with `.str.strip()` (remove spaces) and `.str.title()` /
  `.str.lower()` (fix case).
- **Inconsistent categories:** "USA", "U.S.A.", "United States" should be merged
  with `.replace(...)` or a mapping.
- **Wrong data types:** numbers stored as text block maths. Convert with
  `pd.to_numeric(df["col"], errors="coerce")` (invalid entries become `NaN`) or
  `df["date"] = pd.to_datetime(df["date"])`.

## Practical: clean a messy dataset end to end

Let's clean a deliberately messy employee table containing *every* problem above.

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "name":   ["Ali", " Sara", "OMAR", "Ali", "Lina", None],
    "age":    [25, 32, np.nan, 25, 200, 29],          # missing + an impossible 200
    "city":   ["Lahore", "karachi", "Lahore", "Lahore", "Karachi ", "Multan"],
    "salary": [50000, 85000, 62000, 50000, 90000, np.nan],
})

# --- 1) Detect problems ---
print("Missing per column:", df.isnull().sum().tolist())   # name, age, city, salary
print("Duplicate rows:", df.duplicated().sum())
```

**Output:**
```text
Missing per column: [1, 1, 0, 1]
Duplicate rows: 1
```

We have one missing `name`, one missing `age`, one missing `salary`, and one fully
duplicated row (the second "Ali").

```python
# --- 2) Remove duplicate rows ---
clean = df.drop_duplicates().copy()      # .copy() avoids the SettingWithCopyWarning

# --- 3) Standardise text: strip spaces and fix capitalisation ---
clean["name"] = clean["name"].str.strip().str.title()
clean["city"] = clean["city"].str.strip().str.title()
print("Unique cities now:", sorted(clean["city"].dropna().unique().tolist()))
```

**Output:**
```text
Unique cities now: ['Karachi', 'Lahore', 'Multan']
```

Before cleaning, "karachi" and "Karachi " were treated as *different* cities. After
stripping spaces and fixing case, they correctly merge into one "Karachi".

```python
# --- 4) Impute missing numbers with the MEDIAN (robust to the 200 outlier) ---
clean["age"]    = clean["age"].fillna(clean["age"].median())
clean["salary"] = clean["salary"].fillna(clean["salary"].median())

# --- 5) Detect outliers in age using the IQR rule ---
q1, q3 = clean["age"].quantile([0.25, 0.75])
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f"Age IQR bounds: low={lo}, high={hi}")
print("Outliers:", clean[(clean["age"] < lo) | (clean["age"] > hi)]["age"].tolist())

# --- 6) Treat the outlier: cap it to the upper bound (winsorise) ---
clean["age"] = clean["age"].clip(lower=lo, upper=hi)
print("Final shape:", clean.shape)
```

**Output:**
```text
Age IQR bounds: low=24.5, high=36.5
Outliers: [200.0]
Final shape: (5, 4)
```

### Explanation

- **(1)** We detected the missing values and the duplicate before changing anything —
  always diagnose first.
- **(2)** `drop_duplicates()` removed the repeated "Ali" row (6 rows → 5).
- **(3)** Text standardisation merged "karachi"/"Karachi " into "Karachi" — without
  this, a model would treat them as different categories.
- **(4)** We filled missing `age` and `salary` with the **median** (chosen over the
  mean because the age column has that extreme 200).
- **(5)** The IQR rule flagged **age = 200** as an outlier (bounds were 24.5 to
  36.5).
- **(6)** We **capped** it to the upper bound rather than deleting the whole row,
  keeping the rest of that person's valid data.

::: keyidea
Look at how *deliberate* every step was. We didn't just call one magic "clean"
function — we diagnosed each problem and chose a strategy with a reason. That
judgement is the real skill of data cleaning, and it directly determines how good
your final model can be.
:::

::: tip
**Practical tips:** (1) Always clean on a `.copy()` and keep the raw data untouched
so you can re-run. (2) Print before/after at each step to confirm the change did what
you intended. (3) For categorical imputation use the mode:
`df["col"].fillna(df["col"].mode()[0])`. (4) Build cleaning into a reusable function
so the same steps apply to new data later (vital for deployment, Part VIII). (5) The
order matters: dedupe and fix types *before* imputing, so statistics aren't computed
on dirty values.
:::

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Imputing with the mean on skewed/outlier-ridden data.** The 200-age
would have dragged a mean-based fill upward. Prefer the median when in doubt.
:::

- **Mistake 2 — Dropping too many rows.** Losing 30% of your data to remove a few
  `NaN`s is usually worse than imputing.
- **Mistake 3 — Cleaning the test set using statistics from itself.** Compute
  imputation values (means/medians) on the *training* data only, then apply them to
  test/new data — otherwise you leak information (Chapter 25).
- **Mistake 4 — Deleting all outliers blindly.** In fraud or anomaly detection, the
  outliers are exactly what you want to keep.
- **Mistake 5 — Ignoring inconsistent text**, so "Lahore" and "lahore" become two
  categories.
- **Mistake 6 — Not investigating *why* data is missing** — the reason can be
  informative.

## Best practices

- **Diagnose before you fix:** count missing values, duplicates, and check ranges
  first.
- **Keep the raw data;** clean into a copy, with each step documented.
- **Choose imputation by data type and shape** (median for skewed, mode for
  categorical).
- **Compute cleaning statistics on training data only** to avoid leakage.
- **Make cleaning a reusable function/pipeline** so it can run on future data.
- **Investigate outliers** — decide case by case whether to remove, cap, transform,
  or keep.

## Chapter Summary

- Real data is dirty: **missing values, duplicates, outliers, inconsistent text,
  wrong types, structural errors.** **Garbage in → garbage out.**
- **Missing values:** detect with `isnull().sum()`; then either **drop** (rows if
  few; a column if mostly empty) or **impute** (mean for symmetric, **median** for
  skewed, **mode** for categorical, forward-fill for time series, or model-based).
- **Duplicates:** detect with `duplicated()`, remove with `drop_duplicates()`.
- **Outliers:** detect with the **IQR rule** (1.5×IQR beyond the quartiles) or
  **z-score** (|z|>3); treat by removing, **capping**, transforming, or keeping.
- **Inconsistent text:** standardise with `.str.strip()`, `.str.title()`,
  `.replace()`. **Wrong types:** convert with `pd.to_numeric` / `pd.to_datetime`.
- Cleaning is **deliberate, documented, and reusable** — and it largely determines
  your model's ceiling.

---

::: {.qband}
Practice Zone — Chapter 10
:::

## Multiple-Choice Questions (MCQs)

**Q1.** "Garbage In, Garbage Out" means:
a) Delete all data  b) Bad input data leads to bad model output  c) Models are
garbage  d) Outliers are always errors

**Q2.** The safest single statistic to impute a *skewed* numeric column is the:
a) Mean  b) Median  c) Maximum  d) Mode

**Q3.** For a *categorical* column, missing values are best filled with the:
a) Mean  b) Median  c) Mode  d) Zero

**Q4.** The IQR outlier rule flags values beyond:
a) Q1 − IQR and Q3 + IQR  b) Q1 − 1.5×IQR and Q3 + 1.5×IQR
c) mean ± 1 std  d) min and max

**Q5.** `df.drop_duplicates()` by default keeps:
a) The last occurrence  b) The first occurrence  c) None  d) A random one

**Q6.** "Lahore" and "lahore " being treated as two categories is a problem of:
a) Outliers  b) Missing values  c) Inconsistent text  d) Wrong type

**Q7.** In fraud detection, outliers should usually be:
a) Always deleted  b) Kept — they may be the target  c) Replaced with the mean
d) Ignored

**Q8.** Imputation statistics (e.g. the median) should be computed on:
a) The test set  b) The training set only  c) All data combined  d) Random data

### MCQ Answers
**1:** b. **2:** b. **3:** c. **4:** b. **5:** b. **6:** c. **7:** b. **8:** b.

## Interview Questions (with answers)

**Q1. What strategies exist for handling missing data, and how do you choose?**
*Answer:* Drop rows (if few are missing), drop a column (if mostly missing), or
impute with a statistic (mean for symmetric data, median for skewed/outlier data,
mode for categorical), forward/backward fill for time series, a meaningful constant,
or a model (e.g. KNN imputer). Choose based on how much is missing, why it's missing,
the data type, and the cost of losing rows.

**Q2. How do you detect and treat outliers?**
*Answer:* Detect with the IQR rule (beyond Q1−1.5·IQR or Q3+1.5·IQR, visualised by a
box plot) or z-scores (|z|>3 for roughly normal data). Treat by removing genuine
errors, capping/winsorising extreme values, transforming the feature (e.g. log), or
keeping them when they are the signal of interest (fraud, anomalies).

**Q3. Why should imputation values be computed on the training set only?**
*Answer:* To prevent data leakage. If you compute the mean/median using the test set
(or all data), information from the test set leaks into training, giving overly
optimistic performance that won't hold on truly new data.

**Q4. Why is median often preferred over mean for imputation?**
*Answer:* The median is robust to outliers and skew, so it represents the "typical"
value better when the data has extreme values. The mean can be dragged far from the
bulk of the data by a few extremes.

## Scenario-Based Questions (with answers)

**Q1.** *A column "income" is missing for 5% of rows. A teammate suggests dropping
all those rows; another suggests filling with the mean. The income data is highly
right-skewed. What do you recommend?*
*Answer:* Dropping 5% may be acceptable, but imputing keeps more data. Given the
right-skew, fill with the **median** (not the mean, which the skew inflates). Also
investigate *why* income is missing — if "missing" tends to mean "no income", a
constant like 0 or a "missing" flag may be more truthful.

**Q2.** *Your dataset has ages of 25, 30, 28, 35, and 999. The 999 came from a "no
answer = 999" coding. How should you handle it?*
*Answer:* 999 is a coded missing value, not a real age. Replace 999 with `NaN` first
(`df["age"].replace(999, np.nan)`), then impute properly (e.g. median). Treating 999
as a real number would badly distort the mean and the model.

**Q3.** *After cleaning, your model performs great in testing but poorly in
production. You discover you imputed missing values using the mean of the entire
dataset including test data. What went wrong?*
*Answer:* Data leakage. The imputation used information from the test set, so test
performance was optimistic. In production, no such information exists. Fix: fit
imputation on training data only and apply the stored statistics to new data.

## Logic-Based Questions (with answers)

**Q1.** A dataset has 1000 rows; 950 have a missing "secondary_phone" value. Should
you impute or drop the column? Why?
*Answer:* Drop the column. With 95% missing, there's almost no signal to impute from,
and filling it would mostly fabricate data. Removing the column is cleaner.

**Q2.** Why can a single duplicated row meaningfully bias the average of a small
dataset but barely affect a huge one?
*Answer:* In a small dataset the duplicate is a large fraction of the data, so it
shifts the average noticeably. In a huge dataset one extra row is a tiny fraction, so
its effect on the average is negligible.

**Q3.** If capping an outlier changes the dataset's mean but not its median, what
does that reveal about each statistic?
*Answer:* It confirms the mean is sensitive to extreme values (capping the extreme
moved it), while the median is robust (the middle value didn't change). This is why
the median is preferred for skewed/outlier data.

## Practical Questions (with answers)

**Q1.** Write one line to fill missing values in a categorical column `"city"` with
its most frequent value.
*Answer:* `df["city"] = df["city"].fillna(df["city"].mode()[0])`.

**Q2.** In the practical, why did we use `.copy()` after `drop_duplicates()`?
*Answer:* To create an independent DataFrame so subsequent column assignments don't
trigger a `SettingWithCopyWarning` or accidentally modify a view of the original.

**Q3.** Write code to flag rows where `salary` is an outlier by the IQR rule.
*Answer:*
```python
q1, q3 = df["salary"].quantile([0.25, 0.75]); iqr = q3 - q1
mask = (df["salary"] < q1 - 1.5*iqr) | (df["salary"] > q3 + 1.5*iqr)
```

## Long Questions (with answers)

**Q1. Explain the full process of handling missing data: how to detect it, the
available strategies, and how to choose among them, including the leakage pitfall.**

*Answer:* First **detect** missing values with `df.isnull().sum()` (count per column)
and `df.info()`, and investigate *why* they're missing, since the reason can be
informative. Then choose a **strategy**. **Removal**: drop rows when only a small
fraction are affected (`dropna`), or drop a column when most of it is missing and it's
not essential. **Imputation** (filling): use the **mean** for roughly symmetric
numeric data, the **median** for skewed data or data with outliers, the **mode** for
categorical data, **forward/backward fill** for time series, a **meaningful constant**
(like 0 or "Unknown") when missingness itself carries meaning, or a **model-based**
method like KNN imputation that predicts the value from other columns. Choose based on
the percentage missing, the reason, the data type and shape, and the cost of losing
rows. Critically, compute any imputation statistics (means, medians, modes) on the
**training data only**, store them, and apply the stored values to validation, test,
and production data — otherwise information from those sets leaks into training and
inflates measured performance, a mistake that looks fine in testing but fails in the
real world.

**Q2. Discuss outliers: what they are, how to detect them, how to decide whether to
remove, cap, transform, or keep them, with examples.**

*Answer:* An **outlier** is a value far outside the typical range of a variable. They
arise from data-entry errors (age = 200), measurement glitches, coded missing values
(999), or genuine rare events (a billionaire's net worth). **Detection** uses the
**IQR rule** — flag values beyond Q1 − 1.5·IQR or Q3 + 1.5·IQR, visualised by a box
plot's whiskers — or the **z-score rule** (|z| > 3) for roughly normal data; plotting
the distribution also reveals them. **Deciding what to do** depends on cause and
context: **remove** clear errors (the age of 200), but verify they're truly errors
first; **cap/winsorise** to a sensible bound when you want to limit influence without
discarding the row's other valid data; **transform** the feature (e.g. a log
transform on income or price) to compress a long tail so extremes pull less; and
**keep** outliers when they are the very thing you care about — in fraud detection,
network-intrusion detection, or rare-disease screening, the outliers *are* the target,
and deleting them would destroy the task. The guiding principle is to treat outliers
deliberately and contextually, never by reflex.

## Exercises

1. List five sources of missing data you can think of from real life.
2. Given ages `[22, 25, 27, 24, 300]`, compute the IQR bounds by hand and identify
   the outlier.
3. Explain when you would impute with the mode rather than the median.
4. Why is dropping 40% of your rows usually a bad idea? What would you do instead?
5. Give one example each where an outlier should be removed and where it should be
   kept.

## Mini-Project

**Project: Build a reusable cleaning function.**

1. Take a messy dataset (download one, or extend the chapter's example with more
   problems: extra duplicates, mixed-case categories, a coded missing value like
   999).
2. Write a function `clean_data(df)` that: removes duplicates, standardises text
   columns, replaces coded missing values with `NaN`, imputes numerics with the
   median and categoricals with the mode, and caps outliers via the IQR rule.
3. Print a before/after summary (shape, missing counts, unique categories).
4. Write 4–5 sentences justifying each cleaning choice. Save in `my-ml-journey/`.

## Assignments

1. **Coding:** Recreate the chapter's messy DataFrame and add a `salary` outlier of
   `-5000` (an impossible negative salary). Detect and handle it, explaining your
   choice.
2. **Coding:** Demonstrate the leakage pitfall: split data into train/test, then
   impute (a) using the whole dataset's median and (b) using only the training
   median. Explain why (b) is correct.
3. **Conceptual:** Write one page on "why data cleaning determines the ceiling of
   model performance," with at least three concrete examples of how dirty data
   harms models.

::: tip
With clean data in hand, Chapter 11 transforms it into the *form* models need —
scaling numbers and encoding categories. Clean first (Ch 10), then preprocess
(Ch 11): the order matters.
:::
