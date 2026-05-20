# Supervised Learning Overview

## Introduction

Welcome to **Part IV** — the part you've been building toward since Chapter 1. With
your foundations (maths, stats, Python) and data skills (cleaning, features, EDA) in
place, you are finally ready to **build models that learn from labelled data**.

**Supervised learning** is the most widely used type of Machine Learning in industry.
This chapter is the *map* of Part IV: it explains how supervised learning works as a
whole, the deep idea of the **bias–variance trade-off**, how models form **decision
boundaries**, and how to *choose* among algorithms. Then Chapters 17–24 teach each
major algorithm in depth, and Chapters 25–26 teach how to evaluate and tune them.

::: keyidea
Every supervised algorithm — from simple linear regression to giant neural networks —
follows the same recipe: **learn a function that maps inputs (X) to outputs (y) from
labelled examples, so it can predict y for new X.** The algorithms differ only in
*what kind of function* they learn and *how* they learn it.
:::

By the end of this chapter you will be able to:

- Describe the supervised learning workflow precisely.
- Distinguish **classification** and **regression** and their algorithms.
- Understand **decision boundaries** and model **complexity**.
- Explain the **bias–variance trade-off** — the central idea of all of ML.
- Understand the **"No Free Lunch"** theorem and how to choose an algorithm.
- Run a multi-algorithm **bake-off** and compare results.

## What supervised learning learns

Recall from Chapter 4: supervised learning uses **labelled** data — inputs `X` paired
with correct answers `y`. The algorithm learns a function `f` such that `f(X) ≈ y`,
then uses `f` to predict `y` for new, unseen `X`.

- **Classification** — `y` is a category (spam/not-spam, disease/healthy, digit 0–9).
- **Regression** — `y` is a number (price, temperature, age).

![The supervised learning workflow: labelled training data teaches the model a mapping from features to target; the trained model then predicts the target for new, unseen data, and we evaluate it against known answers.](assets/images/ch16_supervised_flow.png)

## The algorithms of Part IV (a roadmap)

Each algorithm has strengths, weaknesses, and ideal use cases. Here is your map.

| Chapter | Algorithm | Type | One-line idea |
|---|---|---|---|
| 17 | Linear Regression | Regression | Fit a straight line/plane |
| 18 | Logistic Regression | Classification | Linear model + sigmoid → probability |
| 19 | K-Nearest Neighbors | Both | Predict like your closest neighbours |
| 20 | Naive Bayes | Classification | Apply Bayes' theorem with independence |
| 21 | Decision Trees | Both | Ask a series of yes/no questions |
| 22 | Support Vector Machines | Both | Find the widest separating margin |
| 23 | Random Forest | Both | Vote across many decision trees |
| 24 | Boosting / XGBoost | Both | Many weak models, each fixing the last |

## Decision boundaries: how classifiers "decide"

A classifier divides the feature space into regions, one per class. The line (or
curve, or surface) separating these regions is the **decision boundary**. Different
algorithms produce different boundary *shapes*:

- **Linear models** (logistic regression, linear SVM) draw **straight** boundaries.
- **Tree-based models** draw **box-like, axis-aligned** boundaries.
- **KNN and kernel SVM** can draw **complex, curvy** boundaries.

![Decision boundaries of different model families on the same data. Linear models draw straight boundaries; trees draw rectangular ones; KNN draws flexible, wiggly ones. More flexibility can fit better — or overfit.](assets/images/ch16_decision_boundaries.png)

The *shape* a model can draw determines what patterns it can learn — and how easily it
can overfit.

## The bias–variance trade-off (the heart of ML)

We met overfitting and underfitting in Chapter 2. Now we name the underlying forces
precisely. A model's error comes from two sources:

- **Bias** — error from wrong *assumptions*; the model is **too simple** to capture
  the pattern. High bias → **underfitting**. (Example: fitting a straight line to
  curved data.)
- **Variance** — error from being **too sensitive** to the training data; the model
  is **too complex** and learns noise. High variance → **overfitting**. (Example: a
  wiggly curve through every point.)

![The bias–variance trade-off. As model complexity grows, bias falls but variance rises. Total error is lowest at an intermediate "sweet spot" — neither too simple nor too complex.](assets/images/ch16_bias_variance.png)

::: keyidea
**You cannot minimise bias and variance independently — reducing one tends to raise
the other.** The art of ML is finding the **sweet spot** of complexity that minimises
*total* error on unseen data. Regularization (Chapter 26), more data, and ensembles
(Chapters 23–24) are the main tools for managing this balance.
:::

| | High Bias (underfit) | High Variance (overfit) |
|---|---|---|
| Symptom | Poor on train *and* test | Great on train, poor on test |
| Cause | Model too simple | Model too complex |
| Fix | More complex model, more features | More data, simpler model, regularization |

## The "No Free Lunch" theorem

A famous result states: **no single algorithm is best for every problem.** An
algorithm that excels on one dataset may be mediocre on another. There is no universal
"best" model.

::: keyidea
The practical consequence: **always try several algorithms and compare them** on your
data. Don't assume the fanciest model wins — let evidence decide. This is why we run
"bake-offs" (below) and why Part IV teaches you a *toolbox*, not one tool.
:::

## How to choose an algorithm (practical guidance)

| If you want… | Consider |
|---|---|
| A simple, interpretable baseline | Linear/Logistic Regression |
| Strong performance on tabular data | Random Forest, XGBoost |
| Probabilistic outputs | Logistic Regression, Naive Bayes |
| To handle non-linear patterns | SVM (kernel), trees, neural nets |
| Fast training on huge text data | Naive Bayes, linear models |
| Maximum accuracy (competitions) | Gradient boosting (XGBoost/LightGBM) |
| Images/text/audio (unstructured) | Deep learning (Part VI) |

Also weigh: **interpretability** (can you explain it?), **training speed**, **dataset
size**, and **deployment constraints** (Part VIII).

## Practical: a multi-algorithm bake-off

Let's prove "No Free Lunch" and practise comparison by training five different
classifiers on the same dataset (breast-cancer diagnosis) and comparing accuracy.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features (needed for distance/gradient models, not for trees)
sc = StandardScaler().fit(X_tr)
X_tr_s, X_te_s = sc.transform(X_tr), sc.transform(X_te)

# (model, needs_scaling)
models = {
    "Logistic Regression": (LogisticRegression(max_iter=5000), True),
    "KNN":                 (KNeighborsClassifier(),            True),
    "Decision Tree":       (DecisionTreeClassifier(random_state=42), False),
    "Random Forest":       (RandomForestClassifier(random_state=42), False),
    "SVM":                 (SVC(),                             True),
}

for name, (model, scale) in models.items():
    if scale:
        model.fit(X_tr_s, y_tr); acc = accuracy_score(y_te, model.predict(X_te_s))
    else:
        model.fit(X_tr, y_tr);   acc = accuracy_score(y_te, model.predict(X_te))
    print(f"{name:22s}: {acc:.3f}")
```

**Output:**
```text
Logistic Regression   : 0.982
KNN                   : 0.956
Decision Tree         : 0.912
Random Forest         : 0.956
SVM                   : 0.982
```

### Explanation

- Five very different algorithms, the *same* data — yet results vary from **0.912 to
  0.982**. That spread *is* the No Free Lunch theorem in action.
- On this dataset, the **linear models (Logistic Regression, SVM)** did best (0.982),
  while the single **Decision Tree** was weakest (0.912) — a reminder that fancier
  isn't always better.
- Note we **scaled** features for the distance/gradient models (LogReg, KNN, SVM) but
  **not** for the tree-based ones (Chapter 11) — applying the right preprocessing per
  model.

::: keyidea
This bake-off is the professional workflow in miniature: prepare data once, try
several algorithms, compare fairly on a held-out set, and let the evidence pick the
winner. In the coming chapters you'll learn *why* each of these models behaves as it
does — but the habit of *comparing* is timeless.
:::

::: tip
**Better comparison practices (previewing Chapter 25):** (1) Accuracy alone can
mislead on imbalanced data — also compare precision, recall, F1, and AUC. (2) Use
**cross-validation** instead of a single split for a more reliable estimate. (3)
Compare on the *same* split and seed for fairness. (4) Consider training time and
interpretability, not just accuracy. (5) Wrap each model in a `Pipeline` (Chapter 11)
so scaling is handled correctly and reproducibly.
:::

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Assuming the most complex model is best.** Here, simple logistic
regression tied for the top. Always compare; respect No Free Lunch.
:::

- **Mistake 2 — Judging on training accuracy** instead of held-out test/CV
  performance.
- **Mistake 3 — Forgetting to scale** for distance/gradient models (or needlessly
  scaling trees).
- **Mistake 4 — Using accuracy on imbalanced data** (Chapter 25).
- **Mistake 5 — Trying only one algorithm** and stopping.
- **Mistake 6 — Confusing bias and variance** — high bias = underfit (too simple),
  high variance = overfit (too complex).

## Best practices

- **Start with a simple baseline** (logistic/linear regression), then try more complex
  models.
- **Always compare several algorithms** on the same fair split.
- **Manage the bias–variance trade-off** toward the sweet spot.
- **Use cross-validation** and multiple metrics (Chapter 25).
- **Match preprocessing to the model** (scale for distance/gradient, not trees).
- **Weigh interpretability, speed, and deployment**, not just accuracy.

## Chapter Summary

- **Supervised learning** learns a mapping `f(X) ≈ y` from labelled data; it splits
  into **classification** (categories) and **regression** (numbers).
- Models carve the feature space with **decision boundaries** whose shape (straight,
  box-like, curvy) depends on the algorithm family.
- The **bias–variance trade-off** is central: **bias** = too-simple → underfitting;
  **variance** = too-complex → overfitting. Minimise *total* error at the complexity
  **sweet spot**.
- **No Free Lunch:** no algorithm is best everywhere — **always compare several** on
  your data.
- A **bake-off** on breast-cancer data showed accuracies from 0.912 to 0.982, with the
  simple linear models tying for best — evidence over assumptions.

---

::: {.qband}
Practice Zone — Chapter 16
:::

## Multiple-Choice Questions (MCQs)

**Q1.** Supervised learning requires:
a) No data  b) Labelled data (X with known y)  c) Only images  d) A reward signal

**Q2.** High bias typically leads to:
a) Overfitting  b) Underfitting  c) Perfect models  d) Data leakage

**Q3.** High variance typically leads to:
a) Underfitting  b) Overfitting  c) High bias  d) Faster training

**Q4.** The "No Free Lunch" theorem implies you should:
a) Always use deep learning  b) Try and compare several algorithms  c) Never use
linear models  d) Use the most complex model

**Q5.** Tree-based models produce decision boundaries that are:
a) Always straight lines  b) Box-like / axis-aligned  c) Always circular  d) Random

**Q6.** As model complexity increases, bias generally ___ and variance generally ___.
a) rises, rises  b) falls, rises  c) falls, falls  d) rises, falls

**Q7.** A model scoring 99% on train and 70% on test most likely has:
a) High bias  b) High variance (overfitting)  c) Low variance  d) A data error only

**Q8.** Which preprocessing applies to KNN/SVM but NOT to a Decision Tree?
a) Removing duplicates  b) Feature scaling  c) Handling missing values  d) Splitting
data

### MCQ Answers
**1:** b. **2:** b. **3:** b. **4:** b. **5:** b. **6:** b. **7:** b. **8:** b.

## Interview Questions (with answers)

**Q1. Explain the bias–variance trade-off.**
*Answer:* Total prediction error decomposes into bias (error from overly simple
assumptions — underfitting) and variance (error from over-sensitivity to training data
— overfitting). Increasing model complexity lowers bias but raises variance, and vice
versa. The goal is the complexity that minimises total error on unseen data; tools
include regularization, more data, and ensembles.

**Q2. What is the No Free Lunch theorem and its practical implication?**
*Answer:* It states no single algorithm is optimal across all possible problems —
performance depends on the dataset. Practically, you should try multiple algorithms and
select based on validated performance rather than assuming one is universally best.

**Q3. What is a decision boundary?**
*Answer:* The surface in feature space that separates the regions a classifier assigns
to different classes. Its shape depends on the algorithm: linear models produce straight
boundaries, trees produce axis-aligned (box) boundaries, and KNN/kernel SVM can produce
complex curved ones.

**Q4. How do you decide which algorithm to use?**
*Answer:* Consider the problem type (classification/regression), data size and type
(tabular vs unstructured), need for interpretability, training/inference speed, and
deployment constraints — then empirically compare a few candidates with
cross-validation and appropriate metrics, starting from a simple baseline.

**Q5. Why start with a simple baseline model?**
*Answer:* A simple model (e.g. logistic/linear regression) is fast, interpretable, and
sets a performance floor. It tells you whether more complex models are actually adding
value, and sometimes it's already good enough — avoiding needless complexity.

## Scenario-Based Questions (with answers)

**Q1.** *Your complex model gets 100% training accuracy but 65% on test. A simpler
model gets 80% on both. Which is better and why?*
*Answer:* The simpler model. The complex one is badly overfitting (high variance) — its
real-world performance is 65%, worse than the simpler model's 80%. We care about
generalisation to unseen data, not training accuracy.

**Q2.** *A teammate insists deep learning will beat everything on a 2,000-row tabular
dataset. What's your evidence-based response?*
*Answer:* On small/medium tabular data, tree ensembles (Random Forest, XGBoost) and
even linear models often beat deep learning, which is data-hungry and prone to overfit
small data. By No Free Lunch, we should run a bake-off and let validated results decide
rather than assume.

**Q3.** *Your model underfits (poor on train and test). List three things to try.*
*Answer:* (1) Use a more complex/flexible model. (2) Add more informative features or
engineer interactions/polynomials (Chapter 12). (3) Reduce regularization, or train
longer — all of which lower bias.

## Logic-Based Questions (with answers)

**Q1.** Why can't you simultaneously drive both bias and variance to zero with a fixed
amount of data?
*Answer:* Reducing bias requires more model flexibility, which increases sensitivity to
the training sample (variance); reducing variance requires simpler/constrained models,
which increases bias. With limited data they trade off, so you optimise total error
rather than eliminate both.

**Q2.** If five different algorithms give noticeably different accuracies on the same
data, which theorem does this illustrate, and what should you do?
*Answer:* The No Free Lunch theorem. You should compare them fairly (same split,
cross-validation, appropriate metrics) and select the best performer for this problem.

**Q3.** A linear model underfits curved data. Logically, is this a bias or a variance
problem, and what's a fix?
*Answer:* A bias problem — the model is too simple to represent the curve. Fixes:
add polynomial/interaction features or use a more flexible model (tree, kernel SVM,
neural net).

## Practical Questions (with answers)

**Q1.** In the bake-off, why did we scale features for SVM/KNN/LogReg but not the
trees?
*Answer:* SVM, KNN, and logistic regression are distance/gradient-based and sensitive
to feature scale; trees split on thresholds and are invariant to monotonic scaling, so
scaling them is unnecessary.

**Q2.** How would you make the comparison more reliable than a single train/test split?
*Answer:* Use cross-validation (e.g. `cross_val_score`) to average performance over
multiple folds, reducing the luck of one particular split, and report multiple metrics.

**Q3.** Write one line to compute accuracy given true labels `y_te` and predictions
`preds`.
*Answer:* `accuracy_score(y_te, preds)` (from `sklearn.metrics`).

## Long Questions (with answers)

**Q1. Explain the bias–variance trade-off in depth: define each term, describe their
symptoms and causes, how they relate to model complexity, and how to manage them.**

*Answer:* A model's expected error on unseen data can be decomposed into **bias**,
**variance**, and irreducible noise. **Bias** is error from erroneous simplifying
assumptions — a model too simple to capture the true pattern (e.g. a straight line for
curved data); its symptom is poor performance on *both* training and test sets
(**underfitting**). **Variance** is error from excessive sensitivity to the particular
training sample — a model so flexible it fits noise; its symptom is excellent training
performance but poor test performance (**overfitting**), seen as a large train-test gap.
These relate to **complexity**: as a model grows more complex, bias falls (it can
represent more patterns) but variance rises (it fits more noise); as it grows simpler,
the reverse happens. Total error is therefore U-shaped in complexity, minimised at an
intermediate **sweet spot**. To **manage** the trade-off: combat high bias with more
flexible models, more/better features, and less regularization; combat high variance
with more training data, simpler models, **regularization** (Chapter 26), early
stopping, and **ensembles** (bagging reduces variance, Chapter 23; boosting reduces
bias, Chapter 24). Cross-validation is used to locate the sweet spot empirically. This
balance is the single most important conceptual tool in supervised learning.

**Q2. Describe the supervised-learning model-selection process end to end, from problem
framing to choosing a final algorithm, referencing the No Free Lunch theorem.**

*Answer:* Selection begins with **framing**: identify whether it's classification or
regression, define the target and the metric that matches the business goal (Chapter
25), and split off a test set early (Chapter 11). Next, **prepare data** once
(cleaning, features, appropriate scaling per model) and establish a **simple baseline**
(e.g. logistic/linear regression) to set a performance floor. Then, guided by the **No
Free Lunch theorem** — which says no algorithm is universally best — run a **bake-off**
of several candidate algorithms suited to the data (e.g. trees and boosting for tabular
data, linear/Naive Bayes for high-dimensional text), comparing them fairly using
**cross-validation** and multiple metrics on the same splits. Evaluate not only
accuracy but also **interpretability**, **training/inference speed**, **data
requirements**, and **deployment constraints**. Diagnose each model's bias/variance to
decide whether to add complexity or regularize. Finally, select the algorithm with the
best validated trade-off for the specific problem, tune its hyperparameters (Chapter
26), and confirm on the held-out test set. The throughline is empiricism: let validated
evidence on *your* data — not assumptions about which model is "best" — drive the
choice.

## Exercises

1. For each, state whether it's classification or regression and suggest two suitable
   algorithms: predicting house price; detecting spam; forecasting temperature;
   diagnosing disease.
2. Explain bias and variance in one sentence each, with a fresh example.
3. Sketch the bias–variance curve and mark underfitting, overfitting, and the sweet
   spot.
4. State the No Free Lunch theorem in your own words and its practical implication.
5. List four factors besides accuracy that influence algorithm choice.

## Mini-Project

**Project: Your own bake-off.**

1. Pick a classification dataset (e.g. breast cancer, wine, or Titanic after Part III
   cleaning).
2. Train at least five different algorithms with correct per-model preprocessing
   (scale where needed).
3. Compare them using cross-validation (`cross_val_score`) and at least two metrics.
4. Make a bar chart of the results (Chapter 14) and identify the winner.
5. Write 4–5 sentences interpreting the results in light of bias/variance and No Free
   Lunch. Save in `my-ml-journey/`.

## Assignments

1. **Coding:** Reproduce the chapter's bake-off, then add cross-validation and report
   mean ± std accuracy per model. Did the ranking change?
2. **Coding:** Take one model and deliberately make it overfit (e.g. a very deep
   Decision Tree), then fix it (limit depth). Show train vs test accuracy before and
   after.
3. **Conceptual:** Write one page explaining the bias–variance trade-off with diagrams,
   including how ensembles and regularization help.

::: tip
You now have the map of supervised learning. Next, Chapter 17 begins the journey
through individual algorithms with **Linear Regression** — the simplest, most
fundamental model, and the perfect place to deeply understand training, loss, and
gradient descent in action.
:::
