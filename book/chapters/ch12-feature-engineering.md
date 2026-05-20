# Feature Engineering

## Introduction

If there is one chapter in Part III that can single-handedly transform a mediocre
model into a great one, this is it. There's a famous saying among practitioners:

> "Applied Machine Learning is basically feature engineering." — Andrew Ng

**Feature engineering** is the art and science of creating new, more informative
input features from your existing data. While Chapters 10–11 *fixed* and *reshaped*
data, this chapter *creates* signal — turning raw columns into features that make the
patterns obvious to the model.

A simple example: knowing a person's `height` and `weight` separately is okay, but
combining them into **BMI** (`weight / height²`) gives the model a single, powerful
health indicator it would struggle to discover on its own.

::: keyidea
**Better features usually beat better algorithms.** A simple model with brilliant
features will routinely outperform a complex model with raw features. This is where
your *domain knowledge* — what you understand about the problem — becomes a
superpower that no algorithm can replace.
:::

By the end of this chapter you will be able to:

- Explain *why* feature engineering is so powerful.
- Create new features through **combinations, ratios, and domain knowledge**.
- Extract features from **dates/times** and **bin** continuous values.
- Build **polynomial and interaction** features.
- Tame skewed data with **log/power transforms**.
- Handle **high-cardinality** categories with target/frequency encoding.

## What is a feature, and what is feature engineering?

A **feature** is an input column the model learns from. **Feature engineering** is
creating, transforming, or combining features so the model can learn better. It
includes:

- **Creating** new features (BMI from height and weight; "price per square metre").
- **Transforming** features (log of a skewed column).
- **Extracting** hidden features (day-of-week from a date).
- **Combining** features (interactions, ratios).
- **Encoding** features cleverly (target encoding for many categories).

![Feature engineering sits between raw data and the model: it converts raw columns into informative features. Strong features make the learning problem dramatically easier.](assets/images/ch12_fe_overview.png)

## Creating features from domain knowledge

The most valuable features often come from *understanding the problem*, not from any
algorithm.

- **Ratios:** `price / area` (price per m²), `debt / income` (a key credit feature),
  `wins / games` (win rate).
- **Differences:** `current_value − previous_value` (change), `today − signup_date`
  (account age).
- **Combinations:** `weight / height²` (BMI), `clicks / impressions` (click-through
  rate).
- **Counts / flags:** "number of previous purchases", "is_first_time_buyer" (0/1).

```python
import pandas as pd
df = pd.DataFrame({
    "height_m":  [1.70, 1.80, 1.60, 1.75],
    "weight_kg": [65, 90, 55, 80],
})
df["bmi"] = (df["weight_kg"] / df["height_m"] ** 2).round(1)
print("BMI:", df["bmi"].tolist())
```

**Output:**
```text
BMI: [22.5, 27.8, 21.5, 26.1]
```

One new column (`bmi`) captures a relationship the model would otherwise have to
infer from two raw columns. That's feature engineering in miniature.

## Extracting features from dates and times

A raw date like `2024-06-20` is nearly useless to a model as-is. But it *hides* many
useful features.

```python
df = pd.DataFrame({"signup": pd.to_datetime(
    ["2024-01-15", "2024-06-20", "2024-11-02", "2024-03-30"])})
df["signup_month"]   = df["signup"].dt.month
df["signup_weekday"] = df["signup"].dt.day_name()
print("months:", df["signup_month"].tolist())
print("weekdays:", df["signup_weekday"].tolist())
```

**Output:**
```text
months: [1, 6, 11, 3]
weekdays: ['Monday', 'Thursday', 'Saturday', 'Saturday']
```

From one date column we extracted **month** and **day-of-week**. You can also derive
year, quarter, hour, "is_weekend", "is_holiday", "days_since_event", and more — all
of which can reveal seasonal or behavioural patterns (vital for the time-series work
in Chapter 42).

## Binning (discretization): turning numbers into categories

**Binning** groups a continuous variable into ranges (bins). This can capture
non-linear effects and reduce the impact of small fluctuations and outliers.

```python
df = pd.DataFrame({"bmi": [22.5, 27.8, 21.5, 26.1]})
df["bmi_cat"] = pd.cut(df["bmi"],
                       bins=[0, 18.5, 25, 30, 100],
                       labels=["under", "normal", "over", "obese"])
print("bmi_cat:", df["bmi_cat"].tolist())
```

**Output:**
```text
bmi_cat: ['normal', 'over', 'normal', 'over']
```

![Binning groups a continuous range into labelled buckets. A continuous BMI becomes the medically meaningful categories underweight / normal / overweight / obese.](assets/images/ch12_binning.png)

We turned exact BMI numbers into the medically meaningful categories doctors actually
use. Binning trades precision for robustness and interpretability — useful when the
*range* matters more than the exact value.

## Transforming skewed features (log / power transforms)

Recall from Chapter 6 that skewed data (income, prices, populations) has a long tail
that distorts many models. A **log transform** compresses that tail, making the
distribution more symmetric and the relationship more linear.

```python
import numpy as np
df = pd.DataFrame({"income": [30000, 120000, 45000, 800000]})  # very right-skewed
df["log_income"] = np.log1p(df["income"]).round(3)             # log(1 + x)
print("log_income:", df["log_income"].tolist())
```

**Output:**
```text
log_income: [10.309, 11.695, 10.714, 13.592]
```

Notice how the huge gap between 800,000 and the rest shrinks dramatically after the
log — the giant value (13.592) is now only modestly larger than the others. We use
`log1p` (which is `log(1 + x)`) so that zeros are handled safely. Other power
transforms include **Box-Cox** and **Yeo-Johnson** (available in scikit-learn).

::: warning
**When NOT to log-transform:** only apply log to **positive, right-skewed** data.
Don't log-transform features that are already symmetric, contain negatives (use
Yeo-Johnson instead), or are categorical. And remember to *reverse* the transform
when interpreting predictions in the original units.
:::

## Polynomial and interaction features

Sometimes the relationship between features and the target is **non-linear** or
depends on **combinations** of features. **Polynomial features** add powers and
products of existing features, letting even a linear model capture curves and
interactions.

```python
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
df = pd.DataFrame({"h": [1.70, 1.80, 1.60, 1.75],
                   "w": [65, 90, 55, 80]})
pf = PolynomialFeatures(degree=2, include_bias=False)
out = pf.fit_transform(df[["h", "w"]])
print("poly feature names:", pf.get_feature_names_out().tolist())
print("poly shape:", out.shape)
```

**Output:**
```text
poly feature names: ['h', 'w', 'h^2', 'h w', 'w^2']
poly shape: (4, 5)
```

From 2 features we generated 5: the originals plus `h²`, `w²`, and the **interaction
term `h w`** (height × weight). The interaction lets the model learn effects that
only appear when two features combine.

::: warning
**Polynomial features explode quickly.** Degree-2 on 100 features creates thousands
of new columns, risking overfitting and slow training. Use low degrees (2–3), apply
them selectively, and combine with feature selection (Chapter 13).
:::

## Encoding high-cardinality categories

In Chapter 11 we one-hot encoded low-cardinality categories. But for columns with
*many* categories (e.g. 10,000 product IDs), one-hot creates too many columns. Two
better options:

- **Frequency encoding** — replace each category with how often it appears. Common
  categories get high numbers; rare ones get low numbers.
- **Target encoding** — replace each category with the *average target value* for
  that category (e.g. average purchase amount per city). Powerful, but must be done
  carefully (using cross-validation) to avoid leakage.

## Practical: engineering features end to end

Putting it together on a customer table:

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({
    "height_m":  [1.70, 1.80, 1.60, 1.75],
    "weight_kg": [65, 90, 55, 80],
    "signup":    pd.to_datetime(["2024-01-15", "2024-06-20",
                                 "2024-11-02", "2024-03-30"]),
    "income":    [30000, 120000, 45000, 800000],
})

df["bmi"]            = (df["weight_kg"] / df["height_m"] ** 2).round(1)   # ratio feature
df["signup_month"]   = df["signup"].dt.month                              # date feature
df["signup_weekday"] = df["signup"].dt.day_name()                         # date feature
df["bmi_cat"]        = pd.cut(df["bmi"], bins=[0, 18.5, 25, 30, 100],     # binning
                              labels=["under", "normal", "over", "obese"])
df["log_income"]     = np.log1p(df["income"]).round(3)                     # skew fix

print(df[["bmi", "signup_month", "signup_weekday", "bmi_cat", "log_income"]].to_string(index=False))
```

**Output:**
```text
 bmi  signup_month signup_weekday bmi_cat  log_income
22.5             1         Monday  normal      10.309
27.8             6       Thursday    over      11.695
21.5            11       Saturday  normal      10.714
26.1             3       Saturday    over      13.592
```

### Explanation

From 4 raw columns we created 5 powerful new features: a health ratio (**BMI**), two
**calendar features**, a meaningful **category** (BMI band), and a **de-skewed
income**. Each new feature exposes signal the model can use directly — without it
having to discover these relationships from scratch.

::: keyidea
This is the practitioner's edge. Two people can use the *exact same algorithm* on the
*exact same raw data* — the one who engineers smarter features will win. Feature
engineering is where your creativity and domain understanding directly translate into
model performance.
:::

::: tip
**Practical & debugging tips:** (1) Engineer features using **training data
statistics only** (target/frequency encoding especially), then apply to test data — or
leakage creeps in. (2) After creating features, check correlations and feature
importance (Chapter 13) to keep the useful ones. (3) Always be able to explain *why*
a feature should help — random features add noise and overfitting. (4) Keep a record
of every feature and its definition, so you can reproduce it in production.
:::

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Target leakage via engineered features.** Creating a feature that
secretly contains future or target information (e.g. "total_spent" when predicting
whether they'll spend) gives unrealistically high scores that collapse in production.
:::

- **Mistake 2 — Adding features with no rationale.** Random combinations add noise
  and overfitting; every feature should have a *why*.
- **Mistake 3 — Log-transforming inappropriate data** (negatives, already-symmetric,
  categorical).
- **Mistake 4 — Polynomial explosion** creating thousands of columns.
- **Mistake 5 — One-hot encoding high-cardinality columns** instead of using
  frequency/target encoding.
- **Mistake 6 — Engineering on the full dataset** (including test) — compute
  data-dependent features on training data only.

## Best practices

- **Use domain knowledge first** — the best features come from understanding the
  problem.
- **Create features with a clear hypothesis** for why they help.
- **Fit data-dependent transforms on training data only** (prevent leakage).
- **Keep polynomial degrees low** and combine with feature selection.
- **Transform skewed positive features** with log/Box-Cox.
- **Document every engineered feature** for reproducibility and deployment.

## Chapter Summary

- **Feature engineering** creates informative features from raw data — often the
  single biggest lever on performance. **Better features beat better algorithms.**
- Techniques: **ratios/differences/combinations** (BMI, price-per-m²), **date/time
  extraction** (month, weekday, is_weekend), **binning** continuous values into
  categories, **log/power transforms** for skew, and **polynomial/interaction**
  features for non-linearity.
- For **high-cardinality** categories, use **frequency** or **target encoding**
  instead of one-hot.
- Engineer **data-dependent** features using **training data only** to avoid leakage,
  give every feature a rationale, and document them for production.

---

::: {.qband}
Practice Zone — Chapter 12
:::

## Multiple-Choice Questions (MCQs)

**Q1.** Combining height and weight into BMI is an example of:
a) Scaling  b) Feature engineering (a ratio feature)  c) Encoding  d) Cleaning

**Q2.** Extracting "day of week" from a date is:
a) Binning  b) A datetime feature  c) A polynomial feature  d) Normalization

**Q3.** A log transform is most appropriate for:
a) Categorical data  b) Symmetric data  c) Positive right-skewed data  d) Negative
data

**Q4.** `pd.cut` is used for:
a) Splitting data into train/test  b) Binning a continuous variable  c) Removing
outliers  d) One-hot encoding

**Q5.** For a column with 10,000 unique categories, the best encoding is usually:
a) One-hot  b) Target/frequency encoding  c) Standardization  d) Min-Max

**Q6.** `PolynomialFeatures(degree=2)` on features `h, w` produces an interaction
term:
a) `h + w`  b) `h w`  c) `h / w`  d) `h - w`

**Q7.** "Better features beat better algorithms" emphasises the importance of:
a) GPUs  b) Feature engineering and domain knowledge  c) Bigger models  d) More
epochs

**Q8.** Creating a feature that secretly contains the target is called:
a) Binning  b) Target leakage  c) Scaling  d) Encoding

### MCQ Answers
**1:** b. **2:** b. **3:** c. **4:** b. **5:** b. **6:** b. **7:** b. **8:** b.

## Interview Questions (with answers)

**Q1. What is feature engineering and why is it important?**
*Answer:* It's the process of creating, transforming, and combining input features to
make patterns easier for a model to learn. It's important because informative
features often improve performance more than switching algorithms — domain-driven
features encode knowledge the model can't easily discover on its own.

**Q2. Give examples of useful engineered features.**
*Answer:* Ratios (price per m², debt-to-income), date parts (month, weekday,
is_weekend), age from a birthdate, aggregations (average purchase per customer),
binned categories, log-transformed skewed values, and interaction/polynomial terms.

**Q3. When and why would you log-transform a feature?**
*Answer:* For positive, right-skewed features (income, prices, counts). The log
compresses the long tail, reducing skew and the influence of extreme values, often
making relationships more linear and models better-behaved.

**Q4. What is target encoding and what's its main risk?**
*Answer:* Target encoding replaces each category with the mean target value for that
category. It's powerful for high-cardinality features but risks target leakage if
computed on all data; it must be done with cross-validation/holdout so a row's own
target doesn't leak into its encoding.

**Q5. How do polynomial features help, and what's the downside?**
*Answer:* They add powers and products of features, letting linear models capture
non-linear and interaction effects. The downside is combinatorial explosion of
columns at higher degrees, leading to overfitting and heavy computation, so degrees
are kept low and paired with feature selection.

## Scenario-Based Questions (with answers)

**Q1.** *You're predicting house prices and have "total_price" and "area" columns.
What engineered feature would likely help, and why?*
*Answer:* "Price per square metre" (`total_price / area`). It normalises price by size,
capturing value density that's more comparable across houses than raw price, often a
strong predictor.

**Q2.** *Your fraud model gets 99.9% accuracy in testing but fails in production. You
engineered a feature "is_flagged_by_investigator." What happened?*
*Answer:* Target leakage. That feature is only known *after* fraud is suspected/
confirmed, so it encodes the answer. The model "cheated" using future information not
available at prediction time. Remove such features and use only data available before
the prediction moment.

**Q3.** *A "city" column has 5,000 unique values. One-hot encoding makes training
crawl and overfit. What do you do?*
*Answer:* Use frequency or target encoding to represent each city as a single
informative number, or group rare cities into an "other" bucket. This drastically
reduces dimensionality while keeping signal.

## Logic-Based Questions (with answers)

**Q1.** Why can a *linear* model fit a curved relationship after adding polynomial
features?
*Answer:* The model is still linear in its parameters, but the added features (x², x³,
interactions) are non-linear functions of the inputs. The linear combination of these
non-linear features can represent curves.

**Q2.** After a log transform, the gap between 800,000 and 30,000 shrinks far more
than the gap between 45,000 and 30,000. Why?
*Answer:* The log function grows slower for larger inputs, so it compresses large
values much more than small ones. This is exactly why it reduces right-skew and the
dominance of extreme values.

**Q3.** You add 50 random-noise features to your data. What is the likely effect on a
flexible model, and why?
*Answer:* Overfitting — the model may latch onto coincidental patterns in the noise
that don't generalise, lowering test performance. Features should have a rationale,
not be random.

## Practical Questions (with answers)

**Q1.** Write one line to create an "account_age_days" feature from a `signup` date
and a reference date `today`.
*Answer:* `df["account_age_days"] = (today - df["signup"]).dt.days`.

**Q2.** Write code to bin ages into "child" (<18), "adult" (18–64), "senior" (65+).
*Answer:*
```python
df["age_group"] = pd.cut(df["age"], bins=[0, 18, 65, 200],
                         labels=["child", "adult", "senior"], right=False)
```

**Q3.** Why use `np.log1p(x)` instead of `np.log(x)`?
*Answer:* `log1p` computes `log(1 + x)`, which safely handles `x = 0` (log of 0 is
undefined) and is more numerically accurate for small `x`.

## Long Questions (with answers)

**Q1. Explain why feature engineering is often more impactful than algorithm choice,
and describe at least five feature-engineering techniques with examples.**

*Answer:* Algorithms learn relationships *present in the features they're given*. If
the right signal isn't expressed in the features, even a powerful model struggles to
find it; conversely, a well-crafted feature can hand the model the answer directly. So
investing in features frequently yields larger gains than swapping algorithms,
especially because good features encode human domain knowledge that no general
algorithm possesses. Key techniques: **(1) Ratios/combinations** — e.g. BMI =
weight/height², or price per square metre, which capture relationships between columns;
**(2) Date/time extraction** — pulling month, weekday, hour, or "is_weekend" from a
timestamp to expose seasonal and behavioural patterns; **(3) Binning** — grouping a
continuous value (BMI) into meaningful categories (under/normal/over/obese) to capture
range effects and add robustness; **(4) Transformations** — applying a log to positive
right-skewed features (income) to reduce skew and the dominance of extremes; **(5)
Polynomial/interaction features** — adding x², products like height×weight to let
linear models capture curves and combined effects. Additional techniques include
aggregation features (average purchase per customer) and smart encodings
(target/frequency) for high-cardinality categories. Applied with domain insight and
guarded against leakage, these techniques routinely turn a weak model into a strong
one.

**Q2. What is data/target leakage in the context of feature engineering, how does it
arise, and how do you prevent it?**

*Answer:* Leakage occurs when a feature contains information that would not be
available at the moment of prediction — typically information about the target or the
future — causing the model to "cheat" and score unrealistically well in testing while
failing in production. In feature engineering it arises in several ways: creating a
feature derived from the target itself (e.g. "total_amount_spent" when predicting
whether a customer will spend), using future data relative to the prediction time
(e.g. "was_refunded" when predicting a sale), or computing data-dependent transforms
(target encoding, scaling statistics, imputation values) on the entire dataset
including the test set, so test information bleeds into training. Prevention: only use
information available *before* the prediction moment; split data first and fit all
data-dependent transformations on the **training set only**, applying the stored
parameters to validation/test/production; use cross-validation when target-encoding;
and critically question every high-importance feature — if it seems "too good," check
whether it secretly encodes the answer. Pipelines (Chapter 11) help enforce these
boundaries automatically.

## Exercises

1. List five engineered features you could create for predicting house prices, each
   with a one-line rationale.
2. From a `birthdate` column, write the features you'd extract for a marketing model.
3. Decide whether to log-transform: income, temperature in °C, number of website
   visits, customer satisfaction (1–5). Justify each.
4. Explain, with an example, how an engineered feature could cause target leakage.
5. Why does binning sometimes help and sometimes hurt? Give one case of each.

## Mini-Project

**Project: Feature-engineer a real dataset.**

1. Take a dataset with at least one date column and some numeric columns (e.g. a
   sales or e-commerce dataset).
2. Engineer at least six new features: a ratio, a difference, two date features, one
   binned feature, and one log-transformed feature.
3. For each feature, write a one-sentence hypothesis for why it should help.
4. Train a simple model (e.g. `LogisticRegression` or `LinearRegression`) before and
   after adding your features, and compare performance.
5. Write a short report on which features helped most. Save in `my-ml-journey/`.

## Assignments

1. **Coding:** Build the "customer" DataFrame from this chapter and add three more
   engineered features of your own (e.g. account age in days, income bracket via
   binning, a height×weight interaction). Print the resulting table.
2. **Coding:** Demonstrate the danger of target encoding leakage: target-encode a
   category using all data vs using only training data, and discuss the difference.
3. **Conceptual:** Write one page on "feature engineering as encoded domain
   knowledge," with three real examples from a field you know.

::: tip
You can now *create* powerful features. But more features isn't always better — too
many can cause overfitting and slow training. Chapter 13, **Feature Selection**, shows
how to keep only the features that truly matter.
:::
