# Ensemble Learning: Random Forest & Bagging

## Introduction

In Chapter 21 we saw that a single decision tree is interpretable but **unstable** and
prone to overfitting. What if, instead of trusting one tree, we asked **hundreds of
trees** and let them vote? That's the idea behind **ensemble learning** — and the
**Random Forest**, one of the most popular, reliable, and accurate algorithms for
tabular data.

The principle is the **wisdom of the crowd**: a large group of diverse, independent
opinions, averaged together, is usually wiser than any single expert. A roomful of
people guessing the number of jellybeans in a jar will, on average, be remarkably close
— even if each individual is off.

::: keyidea
A single decision tree is a weak, high-variance "expert." A **Random Forest** trains
many *different* trees on *different* random slices of the data and features, then
**averages their votes**. The individual errors largely cancel out, dramatically
reducing variance and producing a strong, stable model — usually with almost no tuning.
:::

By the end of this chapter you will be able to:

- Understand **ensemble learning** and the two big families: **bagging** and
  **boosting**.
- Explain how **bagging (Bootstrap Aggregating)** reduces variance.
- Understand how a **Random Forest** adds *random feature selection* on top of bagging.
- Use **out-of-bag (OOB)** error and **feature importances**.
- Tune and apply Random Forests, knowing their pros, cons, and use cases.

## Ensemble learning: many models, one prediction

An **ensemble** combines several models into one stronger model. There are two main
strategies:

- **Bagging (parallel)** — train many models *independently* on different random
  samples of the data, then **average/vote**. Reduces **variance** (overfitting). →
  Random Forest.
- **Boosting (sequential)** — train models *one after another*, each fixing the
  previous one's mistakes. Reduces **bias**. → Gradient Boosting / XGBoost (Chapter 24).

![Bagging vs boosting. Bagging trains many models in parallel on random data samples and averages them (reduces variance). Boosting trains models sequentially, each correcting the last (reduces bias).](assets/images/ch23_ensemble.png)

## Bagging: Bootstrap Aggregating

**Bagging** has two steps:

1. **Bootstrap:** create many new training sets by sampling the original data **with
   replacement** (each new set is the same size, but some rows repeat and some are left
   out).
2. **Aggregate:** train one model on each bootstrap sample, then combine their
   predictions — **majority vote** (classification) or **average** (regression).

Because each model sees slightly different data, they make *different* errors. Averaging
cancels out much of that random error, **reducing variance** without increasing bias.
Bagging works best with high-variance models — like deep decision trees.

## Random Forest = Bagging + random features

A **Random Forest** is bagging applied to decision trees, plus one clever extra source
of randomness:

- **Bagging:** each tree trains on a different bootstrap sample of the rows.
- **Random feature selection:** at *each split*, each tree considers only a **random
  subset of features** (not all of them).

This second trick **decorrelates** the trees — without it, a few strong features would
dominate every tree and make them all similar. Diverse trees → better averaging →
better generalisation.

![A Random Forest: many decision trees, each trained on a random sample of rows and features, vote on the final prediction. Their diverse errors cancel, yielding a robust, accurate model.](assets/images/ch23_random_forest.png)

To predict: each tree votes; the forest outputs the **majority class** (classification)
or the **mean** (regression).

## Out-of-bag (OOB) error: free validation

Here's an elegant bonus. Since each tree's bootstrap sample leaves out ~37% of the rows
(the "out-of-bag" samples), each row can be tested on all the trees that *didn't* see
it. Averaging these gives the **OOB score** — a built-in, free estimate of
generalisation, with no separate validation set needed.

## Practical: Random Forest vs a single tree

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
names = load_breast_cancer().feature_names
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

tree = DecisionTreeClassifier(random_state=42).fit(X_tr, y_tr)
forest = RandomForestClassifier(
    n_estimators=200, oob_score=True, random_state=42).fit(X_tr, y_tr)

print("single tree test acc:  ", round(accuracy_score(y_te, tree.predict(X_te)), 3))
print("random forest test acc:", round(accuracy_score(y_te, forest.predict(X_te)), 3))
print("OOB score:", round(forest.oob_score_, 3))

top = np.argsort(forest.feature_importances_)[::-1][:3]
print("top 3 features:", [(names[i], round(forest.feature_importances_[i], 3)) for i in top])
```

**Output:**
```text
single tree test acc:   0.918
random forest test acc: 0.942
OOB score: 0.967
top 3 features: [('worst perimeter', 0.141), ('worst area', 0.132), ('worst concave points', 0.122)]
```

### Explanation

- The **forest (0.942)** beat the **single tree (0.918)** — averaging many trees reduced
  variance and improved generalisation.
- The **OOB score (0.967)** gave a free internal estimate of performance, no extra
  validation split needed.
- **Feature importances** (averaged over all trees) are more *reliable* than a single
  tree's — here "worst perimeter", "worst area", and "worst concave points" are the top
  diagnostic features.

### How many trees?

```python
for n in [1, 10, 50, 200]:
    m = RandomForestClassifier(n_estimators=n, random_state=42).fit(X_tr, y_tr)
    print(f"n_estimators={n:3d}: {accuracy_score(y_te, m.predict(X_te)):.3f}")
```

**Output:**
```text
n_estimators=  1: 0.912
n_estimators= 10: 0.930
n_estimators= 50: 0.924
n_estimators=200: 0.942
```

More trees generally helps (and **never overfits** from adding trees — it just plateaus),
at the cost of more computation. A few hundred trees is typical.

::: keyidea
Notice the forest needed *no scaling, no careful tuning* and still beat the tree out of
the box, while throwing in a free validation score (OOB) and reliable feature
importances. This "great results with little effort" is exactly why Random Forest is one
of the most-used algorithms in the world and a superb default for tabular data.
:::

::: tip
**Practical & debugging tips:** (1) Like all trees, RF needs **no feature scaling**.
(2) **More trees** (`n_estimators`) only helps then plateaus — set a few hundred and
move on. (3) Key tuning knobs: `max_depth`, `max_features` (the random-feature count),
`min_samples_leaf`. (4) Use `oob_score=True` for free validation. (5) RF can be **slow
and memory-heavy** with thousands of deep trees; balance accuracy vs cost. (6) Use
`RandomForestRegressor` for regression. (7) For top accuracy on tabular data, also try
gradient boosting (Chapter 24).
:::

## Advantages, disadvantages, and use cases

| Advantages | Disadvantages |
|---|---|
| High accuracy, robust, low variance | Less interpretable than one tree |
| Little tuning needed; great default | Slower, more memory than a single tree |
| No feature scaling required | Large models can be heavy to deploy |
| Handles non-linearity & interactions | Can be biased toward high-cardinality features |
| Free OOB validation & feature importances | Boosting often slightly more accurate |

**Use cases:** the **go-to baseline for almost any tabular problem** — credit scoring,
churn, fraud, medical diagnosis, demand forecasting, feature-importance analysis, and
Kaggle competitions (alongside boosting).

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Thinking more trees cause overfitting.** They don't — adding trees only
reduces variance and then plateaus. (Overfitting in RF comes from trees that are too
deep or too few features per split, not from too many trees.)
:::

- **Mistake 2 — Scaling features for Random Forest** (unnecessary — it's tree-based).
- **Mistake 3 — Over-trusting feature importances** for correlated features (importance
  can split between them).
- **Mistake 4 — Ignoring the OOB score**, then wasting data on a separate validation
  set.
- **Mistake 5 — Expecting interpretability** like a single tree — a forest is a
  near-black box.
- **Mistake 6 — Using tiny `n_estimators`** (e.g. 5) and getting unstable results.

## Best practices

- **Use Random Forest as a strong default** for tabular data.
- **Set a few hundred trees**; tune `max_features`, `max_depth`, `min_samples_leaf`.
- **Don't scale** features.
- **Use `oob_score=True`** for free validation.
- **Read feature importances** (with care for correlated features).
- **Compare against gradient boosting** (Chapter 24) when chasing top accuracy.

## Chapter Summary

- **Ensemble learning** combines many models; the two families are **bagging**
  (parallel, reduces variance) and **boosting** (sequential, reduces bias).
- **Bagging (Bootstrap Aggregating)** trains models on bootstrap samples and
  **averages/votes**, cancelling random errors — ideal for high-variance models like
  deep trees.
- A **Random Forest** = bagging of decision trees **plus random feature selection at
  each split**, which **decorrelates** the trees for better averaging.
- It offers **free OOB validation** and reliable **feature importances**; on
  breast-cancer data it beat a single tree (0.942 vs 0.918) with OOB 0.967.
- Random Forest is **accurate, robust, low-maintenance, and needs no scaling** — a
  premier default for tabular ML, though less interpretable and sometimes edged out by
  boosting.

---

::: {.qband}
Practice Zone — Chapter 23
:::

## Multiple-Choice Questions (MCQs)

**Q1.** Bagging primarily reduces:
a) Bias  b) Variance  c) The number of features  d) Training data

**Q2.** A Random Forest combines predictions by:
a) Picking one tree  b) Majority vote / averaging across trees  c) Gradient descent
d) The deepest tree

**Q3.** "Bootstrap" sampling means sampling:
a) Without replacement  b) With replacement  c) Only the test set  d) Only features

**Q4.** Beyond bagging, Random Forest adds randomness by:
a) Scaling features  b) Selecting a random subset of features at each split  c) Using
gradient descent  d) Pruning

**Q5.** The OOB score is:
a) A test set  b) A free internal validation estimate from left-out samples  c) The
training accuracy  d) A hyperparameter

**Q6.** Adding more trees to a Random Forest:
a) Causes overfitting  b) Helps then plateaus (doesn't overfit)  c) Reduces accuracy
d) Requires scaling

**Q7.** Random Forests require feature scaling:
a) Always  b) Never (tree-based)  c) Only for regression  d) Only with OOB

**Q8.** Compared to a single tree, a Random Forest is:
a) More interpretable  b) More accurate and stable, less interpretable  c) Faster to
train  d) Always worse

### MCQ Answers
**1:** b. **2:** b. **3:** b. **4:** b. **5:** b. **6:** b. **7:** b. **8:** b.

## Interview Questions (with answers)

**Q1. How does a Random Forest work?**
*Answer:* It trains many decision trees, each on a bootstrap sample of the rows and, at
each split, on a random subset of features. To predict, the trees vote (classification)
or average (regression). The randomness makes the trees diverse, so their errors cancel
when combined, reducing variance and improving generalisation.

**Q2. What is the difference between bagging and boosting?**
*Answer:* Bagging trains models independently in parallel on random samples and averages
them, mainly reducing variance (Random Forest). Boosting trains models sequentially,
each correcting the previous one's errors, mainly reducing bias (Gradient Boosting/
XGBoost). Bagging combats overfitting; boosting combats underfitting.

**Q3. Why does Random Forest select a random subset of features at each split?**
*Answer:* To decorrelate the trees. If all features were available, a few strong
features would dominate every tree, making them similar and reducing the benefit of
averaging. Random feature subsets force diversity, so averaging cancels more error.

**Q4. What is the out-of-bag (OOB) error?**
*Answer:* Each bootstrap sample leaves out ~37% of rows; those out-of-bag rows can be
predicted by the trees that didn't train on them. Averaging these predictions gives the
OOB score — a built-in validation estimate without a separate hold-out set.

**Q5. Does adding more trees cause overfitting?**
*Answer:* No. More trees reduce variance and the score plateaus; they don't cause
overfitting. Overfitting in RF stems from individual trees being too deep or too few
features per split, not from the number of trees.

## Scenario-Based Questions (with answers)

**Q1.** *You need a strong, reliable model for a tabular dataset with minimal tuning and
no time to scale features. What do you pick and why?*
*Answer:* Random Forest. It delivers high accuracy out of the box, needs no feature
scaling, requires little tuning, resists overfitting, and provides OOB validation and
feature importances — an ideal low-effort default for tabular data.

**Q2.** *Your single decision tree's accuracy swings wildly when you change the random
seed. How does Random Forest help?*
*Answer:* That instability is high variance. Random Forest averages many diverse trees,
which cancels out individual trees' idiosyncrasies, producing far more stable and
accurate predictions across seeds.

**Q3.** *A stakeholder wants to know exactly why the model made each decision. Is Random
Forest the best choice?*
*Answer:* Not for full transparency — a forest of hundreds of trees is effectively a
black box. If exact, auditable rules are required, a single (pruned) decision tree or a
linear model is better; RF can still offer feature importances and tools like SHAP for
partial explanations.

## Logic-Based Questions (with answers)

**Q1.** Why does averaging many high-variance trees reduce overall variance?
*Answer:* If trees make independent random errors, averaging them causes those errors to
partially cancel (the variance of an average of n roughly-independent estimates is much
smaller than that of one), yielding a more stable prediction — the statistical basis of
bagging.

**Q2.** Why does ~37% of data end up "out-of-bag" for each tree?
*Answer:* Sampling n items with replacement from n, the probability a given row is never
picked is (1−1/n)ⁿ → 1/e ≈ 0.368 for large n, so on average ~37% are left out per
bootstrap sample.

**Q3.** Two features are highly correlated and both predictive. Why might each show only
moderate importance in a Random Forest?
*Answer:* Trees randomly use one or the other at splits, so the total importance is
*split* between them; individually each looks moderately important even though together
they're very predictive.

## Practical Questions (with answers)

**Q1.** Write code to train a Random Forest with 300 trees and OOB scoring.
*Answer:* `RandomForestClassifier(n_estimators=300, oob_score=True).fit(X_train,
y_train)`.

**Q2.** How do you read the forest's feature importances?
*Answer:* `model.feature_importances_` (an array aligned with the feature columns); sort
it to rank features.

**Q3.** Do you need to scale features before a Random Forest? Why or why not?
*Answer:* No. Trees split on thresholds and are scale-invariant, so scaling has no
effect — it's unnecessary work.

## Long Questions (with answers)

**Q1. Explain how a Random Forest is built and why it outperforms a single decision
tree, covering bagging, random feature selection, and OOB error.**

*Answer:* A Random Forest builds many decision trees and combines them. First it applies
**bagging (Bootstrap Aggregating)**: it creates many training sets by sampling the
original data **with replacement**, so each tree trains on a slightly different
bootstrap sample. Second, it adds **random feature selection**: at each split, a tree
considers only a random subset of the features rather than all of them. To predict, the
trees **vote** (classification) or **average** (regression). It outperforms a single
tree because a deep tree is a low-bias but **high-variance** model — small data changes
produce very different trees. Training many *diverse* trees (diversity coming from both
the bootstrap rows and the random features) means their individual errors are largely
**independent**, so averaging cancels much of that random error, sharply reducing
variance while keeping bias low — the wisdom-of-the-crowd effect. The random feature
selection is essential to *decorrelate* the trees; without it, dominant features would
make all trees alike and averaging would help little. A bonus is **out-of-bag (OOB)
error**: since each bootstrap leaves out ~37% of rows, each row can be scored by the
trees that didn't see it, giving a free, reliable generalisation estimate without a
separate validation set. The net result is a model that is accurate, robust, needs no
scaling, requires little tuning, and resists overfitting — explaining its popularity.

**Q2. Compare bagging and boosting as ensemble strategies, including what each reduces,
how they train, and example algorithms.**

*Answer:* Bagging and boosting both combine many "weak" models into a strong one, but
differently. **Bagging (parallel)** trains models **independently** on different
bootstrap samples of the data and then **averages or votes** their predictions. Because
the models are diverse and their errors roughly independent, averaging mainly reduces
**variance**, making bagging ideal for high-variance, low-bias base learners like deep
decision trees; the canonical example is the **Random Forest** (bagging + random feature
selection). **Boosting (sequential)** trains models **one after another**, where each new
model focuses on the examples the previous models got wrong (by reweighting data or
fitting residuals) and the models are combined as a weighted sum. This sequential
error-correction mainly reduces **bias**, turning weak learners (often shallow trees)
into a highly accurate committee; examples include **AdaBoost**, **Gradient Boosting**,
and **XGBoost/LightGBM** (Chapter 24). The trade-offs: bagging is easy, parallelisable,
robust, and hard to overfit by adding models; boosting usually achieves higher accuracy
but is sequential (slower to train), more sensitive to noise and hyperparameters, and
can overfit if over-boosted. In practice, both are top performers on tabular data, and
practitioners commonly try a Random Forest first for a robust baseline and gradient
boosting when squeezing out maximum accuracy.

## Exercises

1. Explain "bootstrap sampling" and why it makes trees different from each other.
2. In one sentence each, state what bagging and boosting reduce.
3. Why does Random Forest pick random features at each split?
4. What is the OOB score and why is it useful?
5. Explain why adding more trees doesn't cause overfitting.

## Mini-Project

**Project: Forest vs tree, and feature insight.**

1. On a tabular dataset (e.g. breast cancer, wine, or Titanic after cleaning), train a
   single decision tree and a Random Forest; compare test accuracy.
2. Enable `oob_score=True` and compare the OOB score to the test accuracy.
3. Plot the top 10 feature importances (Chapter 14) and interpret them.
4. Vary `n_estimators` over [1, 10, 50, 100, 300] and plot accuracy — find where it
   plateaus.
5. Write a short report on why the forest beats the tree. Save in `my-ml-journey/`.

## Assignments

1. **Coding:** Build a `BaggingClassifier` of decision trees manually and compare it to
   `RandomForestClassifier` — does the extra feature randomness help?
2. **Coding:** Train a `RandomForestRegressor` on a regression dataset; report R² and the
   top feature importances.
3. **Conceptual:** Write one page explaining the wisdom-of-the-crowd intuition behind
   bagging and why decorrelating the trees matters.

::: tip
Random Forest builds trees *in parallel* and averages them. Chapter 24, **Boosting**,
builds trees *sequentially* — each one fixing the last's mistakes — to reach the highest
accuracy on tabular data, the technique behind XGBoost and countless competition wins.
:::
