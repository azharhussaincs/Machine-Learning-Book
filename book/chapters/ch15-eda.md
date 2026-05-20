# Exploratory Data Analysis (EDA)

## Introduction

This chapter is the **capstone of Part III**. Everything you learned — analysis (Ch
9), cleaning (Ch 10), preprocessing (Ch 11), feature engineering (Ch 12), selection
(Ch 13), and visualization (Ch 14) — comes together into one disciplined process:
**Exploratory Data Analysis (EDA)**.

EDA, a term coined by the statistician John Tukey, is the **detective work** you do
*before* modelling. You interrogate the data: What's in it? What's its shape? What's
missing? What relationships exist? What surprises lurk? The goal is to deeply
understand your data and form **hypotheses** — so that when you build a model, you do
it with insight, not blind hope.

::: keyidea
**EDA is where you "meet" your data.** Skipping it is like marrying a stranger.
Spend real time here: most modelling mistakes and breakthroughs both trace back to
how well you understood the data first. Strong practitioners are obsessive about EDA.
:::

By the end of this chapter you will be able to:

- Follow a **systematic EDA workflow** on any new dataset.
- Inspect structure, analyse distributions, and uncover relationships.
- Diagnose data-quality problems (missing values, outliers, imbalance).
- Analyse the **target variable** and find its strongest predictors.
- Turn observations into **actionable insights and hypotheses**.

## The EDA workflow

EDA is exploratory, not rigid — but a checklist keeps you thorough.

![The EDA workflow: understand structure → analyse single variables → explore relationships → check data quality → analyse the target → form insights. It loops as new questions arise.](assets/images/ch15_eda_workflow.png)

1. **Understand the structure** — shape, column types, a peek at rows.
2. **Univariate analysis** — each variable's distribution (Ch 14 histograms, value
   counts).
3. **Bivariate / multivariate analysis** — relationships and correlations.
4. **Data-quality check** — missing values, duplicates, outliers (Ch 10).
5. **Target analysis** — for supervised problems, study the target and what predicts
   it.
6. **Insights & hypotheses** — write down what you learned and what to try.

## Practical: a complete EDA on the Titanic dataset

We'll run a full EDA on the famous **Titanic** dataset (passengers of the 1912
disaster, with who survived). Our eventual goal would be predicting survival — but
first, we *understand*.

### Step 1 — Understand the structure

```python
import seaborn as sns
df = sns.load_dataset("titanic")
print("shape:", df.shape)            # rows, columns
# df.head(); df.info(); df.describe()  # always run these too
```

**Output:**
```text
shape: (891, 15)
```

891 passengers, 15 columns. (`df.info()` would show types and `df.describe()` the
numeric summaries — always run them.)

### Step 2 — Check data quality (missing values)

```python
print(df.isnull().sum().sort_values(ascending=False).head(4).to_dict())
```

**Output:**
```text
{'deck': 688, 'age': 177, 'embarked': 2, 'embark_town': 2}
```

Immediately we learn: **`deck` is missing for 688 of 891** passengers (~77%) — likely
to be dropped (Ch 10 rule: drop mostly-empty columns). **`age` is missing for 177** —
worth imputing (median). `embarked` has only 2 missing — trivial to fill. This single
check shapes our whole cleaning plan.

### Step 3 — Analyse the target variable

For a supervised problem, always study the target first.

```python
print("Overall survival rate:", round(df["survived"].mean(), 3))
```

**Output:**
```text
Overall survival rate: 0.384
```

About **38%** survived. This is our **baseline**: a model that always predicts "died"
would be ~62% accurate. Any real model must beat that. (Note the mild class imbalance —
relevant for metrics in Chapter 25.)

### Step 4 — Bivariate analysis: what predicts survival?

Now the detective work — which factors relate to survival?

```python
print("By sex:  ", df.groupby("sex")["survived"].mean().round(3).to_dict())
print("By class:", df.groupby("pclass")["survived"].mean().round(3).to_dict())
```

**Output:**
```text
By sex:   {'female': 0.742, 'male': 0.189}
By class: {1: 0.63, 2: 0.473, 3: 0.242}
```

These are *striking* insights:

- **Sex is hugely predictive:** **74%** of women survived vs only **19%** of men —
  the "women and children first" policy in the data.
- **Class matters strongly:** 1st class **63%**, 2nd **47%**, 3rd only **24%** survival
  — wealth and deck location affected access to lifeboats.

These two features alone will likely power a strong survival model.

### Step 5 — Multivariate: relationships among features

```python
print("Mean age by class:", df.groupby("pclass")["age"].mean().round(1).to_dict())
```

**Output:**
```text
Mean age by class: {1: 38.2, 2: 29.9, 3: 25.1}
```

First-class passengers were older on average (38) than third-class (25) — wealthier,
older travellers. This is the kind of relationship a **correlation heatmap** and
**grouped plots** (Chapter 14) reveal at scale.

### Step 6 — Insights and hypotheses

From this short EDA we can already write down concrete, actionable conclusions:

- **Drop `deck`** (77% missing); **impute `age`** with the median (perhaps per class,
  since age varies by class); fill the 2 missing `embarked` with the mode.
- **`sex` and `pclass` are the strongest predictors** — keep them, encode them
  properly (Ch 11).
- The target is **mildly imbalanced** (38% positive) — use appropriate metrics
  (Ch 25), not just accuracy.
- **Hypothesis to test:** a model using sex, class, and age should substantially beat
  the 62% baseline.
- **Feature ideas (Ch 12):** family size (`sibsp + parch`), a "is_child" flag, title
  extracted from name.

::: keyidea
Look how much we learned in *six steps and a handful of lines* — before training a
single model. We now know what to clean, what to keep, what to engineer, which metric
to use, and roughly how well we should expect to do. **This is the payoff of EDA: you
enter modelling with a map, not a blindfold.**
:::

::: tip
**EDA tips & tools:** (1) Automated tools like **`ydata-profiling`** (formerly
pandas-profiling) generate a full EDA report in one line — great for a first pass. (2)
Always pair numbers with plots (histograms, box plots, a correlation heatmap, grouped
bar charts). (3) Do EDA on the **training data** to avoid peeking at the test set. (4)
Keep a running notes file of every insight — it becomes your modelling to-do list. (5)
Question surprises: a "too good" predictor may be leakage (Ch 12).
:::

## Univariate, bivariate, multivariate — the EDA lens

EDA layers these three views (from Chapter 9), now with visualization:

- **Univariate:** `value_counts()` and histograms/box plots for each variable.
- **Bivariate:** `groupby` comparisons, scatter plots, grouped bar charts (target vs
  one feature).
- **Multivariate:** correlation heatmaps, pair plots (`sns.pairplot`), and grouped
  analysis across several variables.

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Skipping EDA and jumping to modelling.** You'll miss data problems,
leakage, and the obvious strong predictors, wasting far more time later.
:::

- **Mistake 2 — Doing EDA on the whole dataset including test**, which risks peeking
  and biasing decisions.
- **Mistake 3 — Only looking at numbers, never plotting** (Anscombe's Quartet, Ch 14).
- **Mistake 4 — Ignoring the target variable's distribution** (missing class
  imbalance).
- **Mistake 5 — Not recording insights**, so the EDA work doesn't guide modelling.
- **Mistake 6 — Treating a strong predictor as a win without checking for leakage.**

## Best practices

- **Follow a checklist** but stay curious — chase surprises.
- **Always run** `head`, `info`, `describe`, `isnull().sum()` first.
- **Study the target** and establish a **baseline** before modelling.
- **Pair every statistic with a plot.**
- **Do EDA on training data**, write down insights and a modelling plan.
- **Let EDA drive** your cleaning, feature engineering, and metric choices.

## Chapter Summary

- **EDA** is the systematic detective work of understanding data *before* modelling —
  the capstone that unites all of Part III.
- The workflow: **structure → univariate → bivariate/multivariate → data quality →
  target analysis → insights**.
- On Titanic, a short EDA revealed: `deck` is 77% missing (drop), `age` needs imputing,
  the **38% survival baseline**, and that **sex (74% vs 19%)** and **class (63% → 24%)**
  are powerful predictors.
- EDA outputs a concrete plan: what to clean, keep, engineer, and which metric to use —
  so you **enter modelling with a map**.
- Always pair statistics with plots, study the target, watch for leakage, and record
  insights.

---

::: {.qband}
Practice Zone — Chapter 15
:::

## Multiple-Choice Questions (MCQs)

**Q1.** EDA stands for:
a) Extended Data Algorithm  b) Exploratory Data Analysis  c) Easy Data Access
d) Empirical Data Adjustment

**Q2.** The first thing to check on a new dataset is usually its:
a) Model accuracy  b) Structure (shape, types, head)  c) Deployment  d) Learning rate

**Q3.** On Titanic, the overall survival rate (~0.384) serves as a:
a) Final model  b) Baseline to beat  c) Loss function  d) Hyperparameter

**Q4.** A column missing 77% of its values should usually be:
a) Imputed with the mean  b) Dropped  c) One-hot encoded  d) Scaled

**Q5.** Studying which features relate to the target is:
a) Univariate analysis  b) Bivariate/target analysis  c) Deployment  d) Scaling

**Q6.** EDA should be performed on:
a) The test set  b) The training data  c) Random data  d) The deployed model

**Q7.** A predictor that seems "too good to be true" might indicate:
a) A great model  b) Data leakage  c) Underfitting  d) Scaling error

**Q8.** Which tool generates an automated EDA report in one line?
a) NumPy  b) ydata-profiling  c) Flask  d) PolynomialFeatures

### MCQ Answers
**1:** b. **2:** b. **3:** b. **4:** b. **5:** b. **6:** b. **7:** b. **8:** b.

## Interview Questions (with answers)

**Q1. What is EDA and why is it important?**
*Answer:* Exploratory Data Analysis is the systematic process of understanding a
dataset before modelling — examining structure, distributions, relationships, data
quality, and the target. It's important because it reveals data problems, leakage,
class imbalance, and the strongest predictors, guiding cleaning, feature engineering,
and metric choice, and preventing costly mistakes later.

**Q2. Walk me through your EDA process on a new dataset.**
*Answer:* First understand structure (`shape`, `info`, `head`, `describe`). Then
univariate analysis (distributions via histograms/value_counts). Then bivariate/
multivariate analysis (relationships, correlation heatmap, grouped comparisons). Then
a data-quality check (missing values, duplicates, outliers). For supervised problems,
analyse the target and establish a baseline, and identify strong predictors. Finally,
record insights and a concrete modelling plan.

**Q3. Why establish a baseline during EDA?**
*Answer:* A baseline (e.g. always predicting the majority class, ~62% on Titanic) sets
the minimum any real model must beat. It prevents being fooled by a high accuracy that
merely reflects class imbalance, and it frames realistic expectations.

**Q4. How does EDA connect to feature engineering and cleaning?**
*Answer:* EDA reveals exactly what to do: which columns to drop (mostly missing), which
to impute and how, which features are strong (keep/encode), which are redundant
(drop), and what new features to engineer (e.g. family size from sibsp+parch). It turns
guesswork into a directed plan.

## Scenario-Based Questions (with answers)

**Q1.** *You're given a customer-churn dataset and one week before modelling is due.
Your manager says "skip EDA, just train a model." How do you respond?*
*Answer:* I'd push back briefly: a few hours of EDA typically *saves* days by revealing
missing data, leakage, imbalance, and the strongest predictors up front. Without it,
models often fail in ways that are expensive to debug later. I'd propose a quick,
time-boxed EDA (even an automated profiling report) as a fast compromise.

**Q2.** *During EDA you find a feature "account_closed_date" that perfectly predicts
churn. Should you celebrate?*
*Answer:* No — this is almost certainly leakage. The closing date is known only *after*
churn happens, so it can't be used to predict it in advance. Drop it (and audit for
similar future-information features) before modelling.

**Q3.** *Your target is 95% class A and 5% class B. What does EDA tell you to do
differently?*
*Answer:* The data is highly imbalanced. EDA flags that accuracy is misleading (always
predicting A scores 95%), so you should use precision/recall/F1 or AUC (Chapter 25),
consider resampling or class weights, and set a sensible baseline (95%) that the model
must meaningfully beat on the minority class.

## Logic-Based Questions (with answers)

**Q1.** If 38% of passengers survived, what accuracy would a model that always predicts
"did not survive" achieve, and why is that important?
*Answer:* About 62% (since 62% did not survive). It's important as the baseline: a
"useful" model must beat 62%, or it's no better than a trivial constant guess.

**Q2.** Survival is 74% for women and 19% for men. Logically, why will "sex" be a
powerful feature for a survival model?
*Answer:* Because it strongly separates the classes — knowing sex alone shifts the
survival probability dramatically (from 19% to 74%), giving the model a lot of
predictive signal.

**Q3.** Why should EDA be done on training data only, not the full dataset?
*Answer:* To avoid peeking at the test set. Insights and decisions (cleaning,
features, thresholds) influenced by the test data leak information, producing
over-optimistic estimates that won't hold on truly unseen data.

## Practical Questions (with answers)

**Q1.** Write one line to get the survival rate grouped by passenger class.
*Answer:* `df.groupby("pclass")["survived"].mean()`.

**Q2.** Write code to list the columns with the most missing values, descending.
*Answer:* `df.isnull().sum().sort_values(ascending=False)`.

**Q3.** Which two single lines give a quick numeric and structural overview of a
DataFrame?
*Answer:* `df.describe()` (numeric summary) and `df.info()` (types and non-null
counts).

## Long Questions (with answers)

**Q1. Describe the full EDA workflow step by step, explaining the purpose of each
step, and illustrate with the Titanic dataset.**

*Answer:* EDA proceeds through six connected steps. **(1) Understand structure** — check
`shape`, `info`, `head`, and `describe` to learn the number of rows/columns, data
types, and basic ranges; on Titanic this reveals 891 passengers and 15 mixed columns.
**(2) Univariate analysis** — examine each variable's distribution with histograms and
value counts to see shapes, skew, and category frequencies. **(3) Bivariate/
multivariate analysis** — explore relationships using grouped comparisons, scatter
plots, and a correlation heatmap; on Titanic, grouping survival by sex (74% vs 19%) and
by class (63%→24%) exposes the strongest predictors. **(4) Data-quality check** — count
missing values, duplicates, and outliers; Titanic shows `deck` 77% missing (drop) and
`age` 20% missing (impute). **(5) Target analysis** — for supervised tasks, study the
target's distribution and set a baseline; Titanic's 38% survival implies a 62% majority
baseline and mild imbalance, which dictates using appropriate metrics. **(6) Insights &
hypotheses** — record concrete conclusions and a plan: drop `deck`, impute `age`, keep
and encode sex and class, engineer family-size and title features, expect to beat 62%.
The purpose throughout is to enter modelling with a thorough, evidence-based plan rather
than guesswork, which is why EDA is the single highest-leverage habit in applied ML.

**Q2. Explain why EDA so often prevents serious modelling failures, giving at least
three concrete categories of problems it catches.**

*Answer:* EDA prevents failures because most model problems originate in
misunderstood data, and EDA surfaces them early. First, it catches **data-quality
problems**: missing values, duplicates, impossible values (an age of 200), and
inconsistent categories that would otherwise silently corrupt training; spotting `deck`
as 77% missing or `age` as 20% missing tells you exactly how to clean. Second, it
catches **target and evaluation problems**: discovering class imbalance (e.g. 95%/5%)
warns that accuracy is misleading and that precision/recall or resampling are needed,
and establishing a baseline (62% on Titanic) sets honest expectations. Third, it
catches **leakage and spurious predictors**: a feature that predicts the target
suspiciously well (like a post-event date) is exposed during bivariate analysis as
something unavailable at prediction time, so it can be removed before it inflates test
scores and then fails in production. EDA additionally reveals the **strongest genuine
predictors** (sex and class on Titanic) and **feature-engineering opportunities**
(family size, titles), focusing effort where it pays off. In short, a few hours of EDA
routinely saves days of confused debugging and prevents models that look good in
testing but fail in the real world.

## Exercises

1. List the six steps of the EDA workflow and one action you'd take in each.
2. For a dataset you choose, compute the target's baseline (majority-class) accuracy.
3. Explain why a column missing 80% of its values is usually dropped, not imputed.
4. Give two examples of insights from EDA that would change your cleaning plan.
5. Why is establishing a baseline before modelling so valuable?

## Mini-Project

**Project: Full EDA report on a real dataset.**

1. Load the Titanic dataset (`sns.load_dataset("titanic")`) or any classification
   dataset.
2. Run the complete six-step EDA workflow: structure, univariate (with plots),
   bivariate/multivariate (grouped stats + correlation heatmap), data-quality check,
   target analysis with a baseline.
3. Produce at least five charts and five written insights.
4. End with a concrete **modelling plan**: what to drop, impute, encode, engineer, and
   which metric to use.
5. Save the report (notebook + charts) in `my-ml-journey/` — this is portfolio-quality
   work.

## Assignments

1. **Coding:** Perform EDA on the Titanic dataset and engineer a `family_size` feature
   (`sibsp + parch + 1`). Show survival rate by family size and write your finding.
2. **Coding:** Use an automated profiling tool (e.g. `ydata-profiling`) on a dataset
   and compare its report to your manual EDA. What did each catch that the other
   missed?
3. **Conceptual:** Write one page on "EDA as the foundation of trustworthy ML," citing
   at least three problems EDA prevents.

::: tip
**Part III complete!** You can now take raw, messy data and turn it into clean,
well-understood, model-ready features — the skill that consumes most of real ML work.
**Part IV** finally begins the modelling you've been building toward: **Supervised
Learning**, starting with a complete overview and then every major algorithm, one by
one.
:::
