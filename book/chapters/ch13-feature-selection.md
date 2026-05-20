# Feature Selection

## Introduction

In Chapter 12 you learned to *create* features. This chapter teaches the opposite,
equally important skill: **choosing which features to keep**. More features is **not**
always better. Irrelevant or redundant features add noise, slow down training, make
models harder to understand, and — most dangerously — cause **overfitting**.

**Feature selection** is the process of keeping only the features that genuinely help
the model, and discarding the rest. Think of it like packing for a trip: taking
*everything you own* makes the journey harder, not easier. You want the *right* few
items.

::: keyidea
**Quality over quantity.** A model with 10 carefully chosen features often beats one
with 100 noisy features — it trains faster, generalises better, and is easier to
explain. Feature selection is how you find those 10.
:::

By the end of this chapter you will be able to:

- Explain the **curse of dimensionality** and why too many features hurt.
- Apply the three families of selection methods: **filter, wrapper, embedded**.
- Use correlation, statistical tests, recursive elimination, and model-based
  importance to pick features.
- Choose the right method for your situation.

## Why select features? The curse of dimensionality

As the number of features (dimensions) grows, the data becomes increasingly
**sparse** — points spread out in a vast empty space, and the amount of data needed to
learn reliably grows explosively. This is the **curse of dimensionality**.

![The curse of dimensionality: as features increase, model performance often rises then falls. Too few features underfit; too many add noise and overfitting. There is a "sweet spot".](assets/images/ch13_curse.png)

**Benefits of fewer, better features:**

- **Less overfitting** — fewer chances to learn noise.
- **Faster training and prediction** — less data to crunch.
- **Easier interpretation** — you can explain a 10-feature model, not a 1000-feature
  one.
- **Lower cost** — fewer features to collect, store, and maintain in production.
- **Often better accuracy** — removing noise can *improve* performance.

## The three families of feature selection

![The three families of feature selection. Filter methods score features independently of any model; wrapper methods search using a model's performance; embedded methods select during model training.](assets/images/ch13_methods.png)

### 1. Filter methods — score features independently

Filter methods rank features using statistics, **before and independent of** any
model. They are fast and simple.

- **Variance threshold** — drop features that barely change (near-constant columns
  carry no information).
- **Correlation** — drop features highly correlated with each other (redundant); keep
  those correlated with the target.
- **Statistical tests** — ANOVA F-test (`f_classif`), chi-square (for categorical),
  **mutual information** (captures non-linear relationships). Use `SelectKBest` to
  keep the top-k scorers.

**Pros:** very fast, model-agnostic. **Cons:** ignores feature *interactions* and the
specific model.

### 2. Wrapper methods — search using a model

Wrapper methods *try* different subsets of features, train a model on each, and keep
the subset that performs best. They "wrap" the selection around a model.

- **Recursive Feature Elimination (RFE)** — train a model, remove the least important
  feature, repeat until the desired number remains.
- **Forward selection** — start empty, add the most helpful feature one at a time.
- **Backward elimination** — start with all, remove the least helpful one at a time.

**Pros:** consider interactions and the actual model; often higher accuracy. **Cons:**
slow (train many models), risk of overfitting the selection.

### 3. Embedded methods — select during training

Embedded methods perform selection *as part of* training the model.

- **Lasso (L1 regularization)** — shrinks unimportant feature weights to *exactly
  zero*, effectively removing them (Chapter 26).
- **Tree-based feature importance** — Random Forests and gradient boosting naturally
  rank features by how much they improve splits.

**Pros:** efficient (one training run), model-aware. **Cons:** tied to that specific
model type.

| Family | How it works | Speed | Considers model? |
|---|---|---|---|
| **Filter** | Statistical scores, pre-model | Fast | No |
| **Wrapper** | Search subsets via model performance | Slow | Yes |
| **Embedded** | Selection during model training | Medium | Yes |

## Practical: three ways to select features on the Iris dataset

Let's apply one method from each family to the Iris dataset (4 features) and see if
they agree on which features matter most.

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

X, y = load_iris(return_X_y=True)
names = load_iris().feature_names
print("features:", names)

# --- FILTER: ANOVA F-test, keep top 2 ---
skb = SelectKBest(f_classif, k=2).fit(X, y)
print("F-scores:", np.round(skb.scores_, 1).tolist())
print("Top 2 (filter):", [names[i] for i in np.argsort(skb.scores_)[-2:][::-1]])

# --- WRAPPER: Recursive Feature Elimination, keep 2 ---
rfe = RFE(LogisticRegression(max_iter=500), n_features_to_select=2).fit(X, y)
print("RFE selected:", [names[i] for i in range(len(names)) if rfe.support_[i]])

# --- EMBEDDED: Random Forest feature importance ---
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
print("RF importances:", np.round(rf.feature_importances_, 3).tolist())
print("RF top feature:", names[int(np.argmax(rf.feature_importances_))])
```

**Output:**
```text
features: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
F-scores: [119.3, 49.2, 1180.2, 960.0]
Top 2 (filter): ['petal length (cm)', 'petal width (cm)']
RFE selected: ['petal length (cm)', 'petal width (cm)']
RF importances: [0.106, 0.022, 0.436, 0.436]
RF top feature: petal length (cm)
```

### Explanation

- **Filter (F-test):** the petal measurements scored *vastly* higher (1180, 960) than
  the sepal measurements (119, 49). The top 2 are **petal length** and **petal
  width**.
- **Wrapper (RFE):** independently arrived at the *same* two petal features.
- **Embedded (Random Forest):** importance scores confirm it — the two petal features
  carry ~87% of the importance (0.436 + 0.436), while sepal width is nearly useless
  (0.022).

::: keyidea
All three methods, using completely different logic, **agreed**: the petal features
matter most for classifying iris species. When multiple methods agree, you can be
confident. This is a great real-world habit — cross-check selection methods rather
than trusting one blindly.
:::

::: tip
**Practical tips:** (1) Always do selection **inside cross-validation / on training
data only** — selecting features using the whole dataset (including test) leaks
information and inflates scores. (2) Start with cheap filter methods to drop obvious
junk, then use embedded/wrapper methods to refine. (3) For correlated features, keep
one representative rather than all. (4) `RandomForest.feature_importances_` is a quick,
powerful first look at what matters. (5) Don't over-prune — removing a feature that
*looks* weak alone but is strong *in combination* can hurt.
:::

## How to choose a method

- **Many features, need speed?** Start with **filter** methods (variance, correlation,
  `SelectKBest`).
- **Want best accuracy and can afford compute?** Use **wrapper** methods (RFE).
- **Using linear models?** **Lasso (L1)** does selection for free.
- **Using trees/boosting?** Read off **feature importances** (embedded).
- **In practice:** combine them — filter to remove obvious noise, then embedded/
  wrapper to fine-tune.

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Selecting features using the whole dataset (including test).** This
leaks information and gives over-optimistic results. Do selection within
cross-validation or on training data only.
:::

- **Mistake 2 — Removing a feature that's weak alone but strong in combination.**
  Wrapper/embedded methods help catch these interactions.
- **Mistake 3 — Keeping highly correlated (redundant) features**, which adds noise and
  multicollinearity.
- **Mistake 4 — Trusting a single method blindly.** Cross-check across families.
- **Mistake 5 — Over-pruning** to the point of underfitting (too few features).
- **Mistake 6 — Confusing correlation-with-target (useful) and correlation-between-
  features (redundant).**

## Best practices

- **Do selection on training data / within CV** to avoid leakage.
- **Start cheap (filter), then refine (embedded/wrapper).**
- **Remove redundant (highly inter-correlated) features.**
- **Cross-check methods**; trust features multiple methods agree on.
- **Balance** the number of features against performance and interpretability.
- **Re-evaluate** after feature engineering (Chapter 12) — new features may make old
  ones redundant.

## Chapter Summary

- **Feature selection** keeps only the useful features; more features is **not**
  better — extras cause overfitting, slowness, and complexity (**curse of
  dimensionality**).
- Three families: **filter** (fast statistical scores — variance, correlation,
  `SelectKBest`/F-test/mutual information), **wrapper** (search subsets via model
  performance — **RFE**, forward/backward), and **embedded** (selection during
  training — **Lasso L1**, **tree feature importance**).
- On Iris, all three methods agreed that **petal length and width** are the key
  features — cross-checking builds confidence.
- Always select **on training data / within cross-validation** to avoid leakage, and
  combine cheap and accurate methods in practice.

---

::: {.qband}
Practice Zone — Chapter 13
:::

## Multiple-Choice Questions (MCQs)

**Q1.** The "curse of dimensionality" refers to problems caused by:
a) Too little data  b) Too many features  c) Slow CPUs  d) Missing values

**Q2.** Which is a *filter* method?
a) RFE  b) Lasso  c) SelectKBest (F-test)  d) Forward selection

**Q3.** Recursive Feature Elimination (RFE) is a:
a) Filter method  b) Wrapper method  c) Embedded method  d) Scaling method

**Q4.** Lasso (L1 regularization) performs feature selection by:
a) Counting features  b) Shrinking some weights to exactly zero  c) Scaling features
d) One-hot encoding

**Q5.** Tree-based feature importance is an example of:
a) Filter  b) Wrapper  c) Embedded  d) Normalization

**Q6.** Which is a benefit of feature selection?
a) More overfitting  b) Slower training  c) Better interpretability  d) More noise

**Q7.** Two features with correlation 0.98 to each other are:
a) Both essential  b) Likely redundant (keep one)  c) Outliers  d) The target

**Q8.** Feature selection should be performed on:
a) The whole dataset including test  b) Training data / within CV  c) The test set
d) Random data

### MCQ Answers
**1:** b. **2:** c. **3:** b. **4:** b. **5:** c. **6:** c. **7:** b. **8:** b.

## Interview Questions (with answers)

**Q1. Why is feature selection important?**
*Answer:* It reduces overfitting by removing noisy/irrelevant features, speeds up
training and prediction, improves interpretability, lowers data-collection cost, and
can even raise accuracy. It combats the curse of dimensionality, where too many
features make learning harder.

**Q2. Explain the three families of feature selection.**
*Answer:* Filter methods score features with statistics independently of any model
(fast, model-agnostic, e.g. correlation, F-test). Wrapper methods search feature
subsets by training a model and measuring performance (e.g. RFE — accurate but slow).
Embedded methods select during model training (e.g. Lasso's L1 zeroing weights, tree
feature importances — efficient and model-aware).

**Q3. What is the difference between filter and wrapper methods?**
*Answer:* Filter methods evaluate features using statistical measures before/
independent of a model, so they're fast but ignore feature interactions and the
specific model. Wrapper methods evaluate subsets using a model's actual performance,
capturing interactions but at much higher computational cost.

**Q4. How does Lasso perform feature selection?**
*Answer:* Lasso adds an L1 penalty proportional to the absolute size of the
coefficients. This penalty drives the weights of unimportant features to exactly zero,
effectively dropping them from the model — selection happens automatically during
training.

**Q5. Why must feature selection be done within cross-validation?**
*Answer:* If you select features using the entire dataset (including test/validation),
information about the targets leaks into the selection, producing over-optimistic
performance estimates. Selecting inside CV (on each training fold only) gives an
honest estimate of how the pipeline generalises.

## Scenario-Based Questions (with answers)

**Q1.** *Your dataset has 2,000 features but only 500 rows. The model overfits badly.
What's happening and what do you do?*
*Answer:* This is the curse of dimensionality — far more features than samples, so the
model memorises noise. Apply aggressive feature selection (filter to drop low-variance
and redundant features, then embedded/wrapper to keep the most predictive few) and/or
dimensionality reduction (Chapter 28), and consider regularization.

**Q2.** *A filter method ranks "feature A" as useless, but you know from domain
expertise it's only useful combined with "feature B." How do you proceed?*
*Answer:* Filter methods miss interactions. Use a wrapper (RFE) or embedded method that
evaluates features in combination, or engineer an explicit interaction feature (A×B,
Chapter 12), then re-evaluate. Don't drop A based on the filter alone.

**Q3.** *Two methods disagree: the F-test ranks feature X highly, but Random Forest
importance ranks it low. How do you interpret this?*
*Answer:* The F-test measures linear/univariate association with the target, while RF
importance reflects usefulness within the model including interactions and redundancy.
Disagreement often means X is linearly related to the target but redundant given other
features the forest uses. Investigate correlations and test model performance with and
without X.

## Logic-Based Questions (with answers)

**Q1.** A feature has nearly zero variance (almost the same value for every row). Why
is it safe to drop?
*Answer:* A near-constant feature provides almost no information to distinguish rows or
predict the target, so removing it loses essentially no signal while reducing
dimensionality.

**Q2.** Why can adding more features sometimes *decrease* test accuracy even though
training accuracy increases?
*Answer:* Extra features give the model more ways to fit noise in the training data
(overfitting), raising training accuracy but harming generalisation, so test accuracy
drops — a classic curse-of-dimensionality symptom.

**Q3.** On Iris, filter, wrapper, and embedded methods all chose the two petal
features. What does this agreement tell you?
*Answer:* That the petal features are robustly the most predictive, since three
methods with different logic independently agree. Such consensus gives high confidence
in the selection.

## Practical Questions (with answers)

**Q1.** Write code to keep the top 3 features by ANOVA F-test.
*Answer:*
```python
from sklearn.feature_selection import SelectKBest, f_classif
X_new = SelectKBest(f_classif, k=3).fit_transform(X, y)
```

**Q2.** How would you read off feature importances from a trained Random Forest?
*Answer:* `rf.feature_importances_` returns an array of importance scores aligned with
the feature columns; sort it to rank features (e.g. `np.argsort(rf.feature_importances_)
[::-1]`).

**Q3.** Write code to drop features whose variance is below 0.01.
*Answer:*
```python
from sklearn.feature_selection import VarianceThreshold
X_new = VarianceThreshold(threshold=0.01).fit_transform(X)
```

## Long Questions (with answers)

**Q1. Explain the three families of feature selection (filter, wrapper, embedded) in
detail, with their mechanisms, pros, cons, and an example method for each.**

*Answer:* **Filter methods** score each feature using statistical measures of its
relationship with the target, independently of any model — examples include variance
thresholding, correlation with the target, the ANOVA F-test (`SelectKBest`),
chi-square, and mutual information. They are very fast and model-agnostic, making them
ideal for an initial cull of obviously useless features, but they ignore feature
interactions and aren't tuned to the specific model you'll use. **Wrapper methods**
treat selection as a search: they repeatedly train a model on different feature subsets
and keep the subset that performs best — Recursive Feature Elimination (RFE) trains a
model, drops the least important feature, and repeats; forward selection adds features
one at a time, backward elimination removes them one at a time. Wrappers account for
interactions and the actual model, often yielding the best accuracy, but they are
computationally expensive (many model trainings) and can overfit the selection.
**Embedded methods** perform selection as a natural part of model training — Lasso
(L1 regularization) shrinks unimportant coefficients to exactly zero, and tree-based
models (Random Forests, gradient boosting) produce feature importances. They are
efficient (one training run) and model-aware, but the selection is tied to that model
type. In practice, a strong workflow combines them: filter to remove obvious noise
cheaply, then use embedded importances or a wrapper to refine — always done within
cross-validation to avoid leakage.

**Q2. What is the curse of dimensionality, and how does feature selection help address
it? Discuss the trade-offs of removing features.**

*Answer:* The **curse of dimensionality** is the set of problems that arise as the
number of features grows: data points become sparse in the high-dimensional space, the
volume to "cover" grows exponentially, distances between points become less meaningful,
and the amount of data needed to learn reliably explodes. With too many features
relative to samples, models easily fit noise, overfit, train slowly, and become hard to
interpret. **Feature selection helps** by reducing the dimensionality to the most
informative features, which lowers overfitting, speeds training and inference, improves
interpretability, cuts data-collection and storage costs, and can even raise accuracy by
removing noise. The **trade-offs** of removing features are real: prune too aggressively
and you may discard features that are weak alone but valuable in combination, causing
underfitting; rely on a single (especially filter) method and you may miss interactions;
and selecting on the full dataset leaks information. The art is to remove genuine noise
and redundancy while preserving combined signal — achieved by cross-checking multiple
selection methods, performing selection within cross-validation, and balancing the
number of features against measured performance and the need for interpretability.

## Exercises

1. List four benefits of using fewer, well-chosen features.
2. Classify each as filter, wrapper, or embedded: RFE, Lasso, chi-square test,
   Random Forest importance, forward selection.
3. Explain why a near-zero-variance feature can be dropped.
4. Give an example of two features that are individually weak but jointly strong.
5. Why must feature selection be done on training data only?

## Mini-Project

**Project: Select features three ways and compare.**

1. Take a dataset with at least 10 features (e.g. scikit-learn's breast cancer
   dataset).
2. Apply a filter method (`SelectKBest`), a wrapper method (`RFE`), and an embedded
   method (Random Forest importance) to pick the top features.
3. Compare the selected sets — which features do all methods agree on?
4. Train a model using all features vs only the selected features and compare test
   accuracy and training time.
5. Write a short report on what you found. Save in `my-ml-journey/`.

## Assignments

1. **Coding:** On the Iris dataset, train a model using only the two petal features vs
   all four features. Compare test accuracy. Did dropping the sepal features hurt?
2. **Coding:** Demonstrate the leakage pitfall: select features using the whole
   dataset vs within cross-validation, and discuss why the second is correct.
3. **Conceptual:** Write one page explaining the three families of feature selection
   with a real-world analogy for each.

::: tip
You can now create (Ch 12) and select (Ch 13) features. Chapter 14 teaches you to
**see** your data through visualisation — and Chapter 15 ties Part III together into
the full **Exploratory Data Analysis** workflow.
:::
