# Hyperparameter Tuning & Regularization

## Introduction

You can now build and evaluate many models. This final chapter of Part IV teaches the
two skills that turn a *good* model into the *best* model: **hyperparameter tuning**
(systematically finding the best settings) and **regularization** (mathematically
controlling overfitting). Together they are how professionals squeeze out maximum,
reliable performance.

Recall from Chapter 2: **parameters** are learned by the model; **hyperparameters** are
set by *you* before training (like a tree's `max_depth`, KNN's `k`, or SVM's `C`).
Picking good hyperparameters can be the difference between a mediocre and an excellent
model — and we should choose them *systematically*, not by guesswork.

::: keyidea
**Tuning** searches the space of hyperparameters for the combination that generalises
best (measured by cross-validation). **Regularization** adds a penalty to the loss that
discourages overly complex models. Both fight overfitting and improve generalisation —
the central goal of all Machine Learning.
:::

By the end of this chapter you will be able to:

- Tune hyperparameters with **grid search** and **random search** (and know about
  Bayesian methods).
- Understand **L1 (Lasso)** and **L2 (Ridge)** regularization and **Elastic Net**.
- Know how the **regularization strength** controls the bias–variance trade-off.
- Apply `GridSearchCV` and regularized models in scikit-learn.

# Part A — Hyperparameter Tuning

## The methods

![Grid search tries every combination on a regular grid; random search samples random combinations. For the same budget, random search often finds good values faster, especially when only a few hyperparameters truly matter.](assets/images/ch26_search.png)

- **Manual / trial-and-error** — change values by hand. Fine for learning, but slow and
  unsystematic.
- **Grid Search** — try **every combination** of a predefined set of values. Thorough
  but **expensive** (combinations multiply: 3 values × 3 values × 3 values = 27 fits).
- **Random Search** — sample **random combinations**. Surprisingly effective: with the
  same budget it often finds better values, because usually only a few hyperparameters
  really matter and random search explores them more widely.
- **Bayesian optimisation** (e.g. Optuna, Hyperopt) — uses past results to *intelligently*
  pick the next combination to try. The most efficient for expensive models.

**Crucial:** tuning is always evaluated with **cross-validation** on the *training* data
— never the test set (Chapter 25).

## Practical: Grid Search with cross-validation

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
sc = StandardScaler().fit(X_tr)

grid = GridSearchCV(
    SVC(),
    param_grid={"C": [0.1, 1, 10], "gamma": [0.001, 0.01, 0.1]},
    cv=5)
grid.fit(sc.transform(X_tr), y_tr)

print("best params:", grid.best_params_)
print("best CV score:", round(grid.best_score_, 3))
```

**Output:**
```text
best params: {'C': 10, 'gamma': 0.001}
best CV score: 0.98
```

### Explanation

- `GridSearchCV` tried all **3 × 3 = 9** combinations of `C` and `gamma`, each with 5-fold
  cross-validation (45 fits total), and reported the best.
- The winning combination (`C=10, gamma=0.001`) achieved **0.98** cross-validated
  accuracy — found automatically, no manual guessing.
- `grid.best_estimator_` is the refitted best model, ready to evaluate **once** on the
  test set.

# Part B — Regularization

## Why regularize?

Recall the bias–variance trade-off (Chapter 16). A model with large, extreme weights can
fit training noise (overfit). **Regularization** adds a **penalty for large weights** to
the loss function, encouraging simpler, smaller-weight models that generalise better. It
directly trades a little training fit for a lot of test stability.

## L2 regularization (Ridge)

**Ridge** adds the **sum of squared weights** to the loss:

<div class="equation"><img class="eq" src="assets/images/eq_ch26_ridge.png" alt="ridge loss"></div>

This **shrinks** all weights toward zero (but rarely exactly zero), spreading influence
across features and stabilising the model — especially helpful with multicollinearity
(Chapter 17).

## L1 regularization (Lasso)

**Lasso** adds the **sum of absolute weights**:

<div class="equation"><img class="eq" src="assets/images/eq_ch26_lasso.png" alt="lasso loss"></div>

L1 has a special property: it drives some weights to **exactly zero**, performing
automatic **feature selection** (Chapter 13). Use Lasso when you suspect many features
are irrelevant and want a sparse, interpretable model.

![L1 (Lasso) vs L2 (Ridge) regularization. L1's diamond-shaped constraint tends to hit corners, zeroing some weights (sparsity/feature selection); L2's circular constraint shrinks weights smoothly toward zero without eliminating them.](assets/images/ch26_l1_l2.png)

- **Elastic Net** combines L1 and L2 — sparsity *and* stability.
- The **regularization strength** (`λ`, called `alpha` in scikit-learn for Ridge/Lasso, or
  inversely `C` for SVM/logistic regression) controls how strong the penalty is: more
  regularization → simpler model (more bias, less variance).

## Practical: Ridge vs Lasso

```python
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score

X, y = load_diabetes(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
sc = StandardScaler().fit(X_tr)
X_tr_s, X_te_s = sc.transform(X_tr), sc.transform(X_te)

for name, model in [("Ridge", Ridge(alpha=1.0)), ("Lasso", Lasso(alpha=1.0))]:
    model.fit(X_tr_s, y_tr)
    nonzero = np.sum(np.abs(model.coef_) > 1e-6)
    print(f"{name}: R2={r2_score(y_te, model.predict(X_te_s)):.3f}, "
          f"nonzero coefs={nonzero}/{len(model.coef_)}")
```

**Output:**
```text
Ridge: R2=0.478, nonzero coefs=10/10
Lasso: R2=0.484, nonzero coefs=9/10
```

### Explanation

- **Ridge** kept **all 10** features (it shrinks but doesn't zero them) with R²=0.478.
- **Lasso** zeroed **one** feature (9/10 nonzero) — automatic feature selection — while
  matching/slightly beating Ridge (R²=0.484). Increasing Lasso's `alpha` would zero out
  *more* features, producing a sparser model.

::: keyidea
Regularization gives you a *dial* (`alpha`/`C`) on model complexity. Turn it up to combat
overfitting (more bias, less variance); turn it down to combat underfitting. **L2 (Ridge)
shrinks; L1 (Lasso) selects.** This dial, tuned by cross-validation, is one of the most
powerful and universal tools in your kit — it appears again in neural networks (weight
decay, dropout) in Part VI.
:::

## Other regularization techniques

- **Early stopping** — stop training when validation performance stops improving (used in
  boosting and neural networks).
- **Dropout** — randomly "switch off" neurons during training (neural networks,
  Chapter 33).
- **Data augmentation** — create more training data (images/text) to reduce overfitting.
- **Reducing model complexity** — fewer features, shallower trees, smaller networks.

::: tip
**Practical & debugging tips:** (1) Always tune with **cross-validation on training
data**, then test **once**. (2) Prefer **`RandomizedSearchCV`** over `GridSearchCV` when
the search space is large. (3) For SVM/logistic regression, remember **small `C` = strong
regularization** (the opposite direction from `alpha`). (4) **Scale features** before
Ridge/Lasso (penalties depend on weight size). (5) Use **Lasso** for feature selection,
**Ridge** for multicollinearity, **Elastic Net** for both. (6) Don't over-tune — chasing
tiny CV gains often just fits noise.
:::

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Tuning hyperparameters on the test set.** This leaks information and
inflates results. Tune with CV on training data; test once at the end.
:::

- **Mistake 2 — Forgetting to scale** before Ridge/Lasso (the penalty is scale-sensitive).
- **Mistake 3 — Confusing `C` and `alpha`** directions (small `C` = strong
  regularization; large `alpha` = strong regularization).
- **Mistake 4 — Grid-searching a huge space** when random/Bayesian search is far cheaper.
- **Mistake 5 — Over-tuning** to squeeze tiny gains, overfitting the validation folds.
- **Mistake 6 — Ignoring regularization entirely** and then wondering why the model
  overfits.

## Best practices

- **Tune systematically** (grid/random/Bayesian) with **cross-validation**.
- **Use random or Bayesian search** for large spaces.
- **Regularize** to control overfitting; pick L1/L2/Elastic Net by your goal.
- **Scale features** before penalised linear models.
- **Tune the regularization strength** as a key hyperparameter.
- **Evaluate the final tuned model once** on the held-out test set.

## Chapter Summary

- **Hyperparameters** are set before training; **tuning** searches for the best
  combination using **cross-validation** — via **grid search** (all combinations),
  **random search** (random samples, often better per budget), or **Bayesian
  optimisation** (smart search).
- **Regularization** adds a penalty for large weights to fight overfitting: **L2 (Ridge)**
  shrinks weights smoothly; **L1 (Lasso)** drives some to **exactly zero** (feature
  selection); **Elastic Net** combines both.
- The **regularization strength** (`alpha`, or inversely `C`) is a complexity dial tuned by
  cross-validation; other regularizers include **early stopping**, **dropout**, and
  **data augmentation**.
- In practice, grid search found SVM `C=10, gamma=0.001` (CV 0.98), and Lasso zeroed a
  feature that Ridge kept — illustrating tuning and L1 sparsity.

---

::: {.qband}
Practice Zone — Chapter 26
:::

## Multiple-Choice Questions (MCQs)

**Q1.** Grid search finds hyperparameters by:
a) Random sampling  b) Trying all combinations on a grid  c) Gradient descent  d) Guessing

**Q2.** Which regularization can drive weights to exactly zero (feature selection)?
a) L2 (Ridge)  b) L1 (Lasso)  c) Neither  d) Both equally

**Q3.** L2 (Ridge) regularization penalises the:
a) Sum of absolute weights  b) Sum of squared weights  c) Number of features  d) Accuracy

**Q4.** Increasing regularization strength generally:
a) Increases variance  b) Increases bias / simplifies the model  c) Causes overfitting
d) Removes the need for data

**Q5.** Hyperparameter tuning should be evaluated using:
a) The test set  b) Cross-validation on training data  c) Training accuracy  d) Random
data

**Q6.** For SVM/logistic regression, a **small** `C` means:
a) Weak regularization  b) Strong regularization  c) More features  d) Faster training

**Q7.** Random search is often preferred over grid search because:
a) It's always exact  b) It explores important hyperparameters more efficiently per
budget  c) It needs no CV  d) It can't overfit

**Q8.** Elastic Net combines:
a) Grid and random search  b) L1 and L2 regularization  c) Bagging and boosting  d) Two
datasets

### MCQ Answers
**1:** b. **2:** b. **3:** b. **4:** b. **5:** b. **6:** b. **7:** b. **8:** b.

## Interview Questions (with answers)

**Q1. What is the difference between grid search and random search?**
*Answer:* Grid search exhaustively tries every combination of specified hyperparameter
values; random search samples random combinations from the space. For the same compute
budget, random search often finds better settings because typically only a few
hyperparameters matter, and it explores those more widely without wasting fits on a rigid
grid.

**Q2. Explain L1 vs L2 regularization.**
*Answer:* L2 (Ridge) adds the sum of squared weights to the loss, shrinking all weights
smoothly toward zero (rarely exactly zero) and stabilising the model under
multicollinearity. L1 (Lasso) adds the sum of absolute weights, which drives some weights
to exactly zero, performing automatic feature selection and yielding sparse, interpretable
models. Elastic Net combines both.

**Q3. How does regularization combat overfitting?**
*Answer:* It adds a penalty for large/complex weights to the loss, so the optimiser
balances fitting the data against keeping weights small. This discourages the model from
contorting itself to fit training noise, increasing bias slightly but reducing variance,
which usually improves generalisation.

**Q4. Why must tuning use cross-validation rather than the test set?**
*Answer:* Repeatedly evaluating and selecting hyperparameters on the test set leaks its
information into model choice, producing optimistic, dishonest performance. Cross-validation
on the training data estimates generalisation fairly, and the untouched test set gives an
unbiased final estimate.

**Q5. What does the regularization strength control and how do you choose it?**
*Answer:* It controls how heavily large weights are penalised — i.e. model complexity and
the bias–variance balance. Higher strength → simpler model (more bias, less variance).
Choose it as a hyperparameter via cross-validation, scanning values (often on a log scale)
and picking the best validated score.

## Scenario-Based Questions (with answers)

**Q1.** *Your model overfits (great train, poor test). Name two regularization actions and
one tuning action.*
*Answer:* Regularization: increase the L2/L1 penalty (`alpha`) or reduce model complexity
(e.g. shallower trees, fewer features); for neural nets, add dropout/early stopping.
Tuning: use cross-validated grid/random search to find a simpler, better-generalising
hyperparameter setting.

**Q2.** *You have 500 features but suspect most are irrelevant and want an interpretable
model. Which regularization and why?*
*Answer:* L1 (Lasso). It drives the weights of irrelevant features to exactly zero,
performing feature selection and yielding a sparse, interpretable model with only the
useful features retained.

**Q3.** *A grid search over 6 hyperparameters with 5 values each is too slow. What do you
do?*
*Answer:* That's 5⁶ = 15,625 combinations — infeasible. Switch to RandomizedSearchCV (or
Bayesian optimisation like Optuna) to sample a manageable number of promising
combinations, optionally narrowing the range after an initial coarse search.

## Logic-Based Questions (with answers)

**Q1.** Why does L1's diamond-shaped constraint produce zero weights while L2's circle
does not?
*Answer:* Optimisation tends to meet the constraint boundary at its extreme points; the
diamond (L1) has sharp corners on the axes, where some weights are exactly zero, so
solutions often land there. The circle (L2) is smooth with no corners, so it shrinks
weights but rarely makes them exactly zero.

**Q2.** If increasing `alpha` lowers both training and test accuracy, what does that
indicate?
*Answer:* Over-regularisation causing underfitting — the penalty is now too strong, making
the model too simple to capture the pattern. Reduce `alpha`.

**Q3.** Why is random search often as good as grid search with far fewer trials?
*Answer:* Because performance usually depends strongly on only a few hyperparameters;
random search samples many distinct values of those important ones, whereas grid search
wastes many trials varying unimportant hyperparameters at fixed (possibly poor) values of
the important ones.

## Practical Questions (with answers)

**Q1.** Write code to grid-search a Random Forest's `n_estimators` and `max_depth` with
5-fold CV.
*Answer:*
```python
GridSearchCV(RandomForestClassifier(),
             {"n_estimators": [100, 300], "max_depth": [3, 5, None]}, cv=5).fit(X, y)
```

**Q2.** In scikit-learn, which parameter controls regularization strength for `Ridge` and
which for `SVC`?
*Answer:* `alpha` for `Ridge`/`Lasso` (higher = stronger), and `C` for `SVC`/
`LogisticRegression` (lower = stronger — it's inverse).

**Q3.** After `GridSearchCV`, how do you get the best model and its settings?
*Answer:* `grid.best_estimator_` (the refitted best model) and `grid.best_params_` (the
winning hyperparameters); `grid.best_score_` gives its cross-validated score.

## Long Questions (with answers)

**Q1. Explain hyperparameter tuning: why it matters, the main search strategies, and how
to do it without data leakage.**

*Answer:* Hyperparameters — settings chosen before training such as a tree's depth, KNN's
k, SVM's C and gamma, or a regularization strength — strongly affect a model's
generalisation, so finding good values can be the difference between a mediocre and an
excellent model. **Search strategies:** *manual* tuning is simple but slow and
unsystematic; *grid search* exhaustively evaluates every combination of specified values,
which is thorough but expensive because combinations multiply; *random search* samples
random combinations and, for a given budget, often finds better settings because typically
only a few hyperparameters matter and it explores those more broadly; *Bayesian
optimisation* (e.g. Optuna) uses past results to choose the next, most promising
combination, the most efficient approach for expensive models. **Avoiding leakage:** every
candidate is evaluated by **cross-validation on the training data only**, never on the
test set — repeatedly tuning against the test set would leak its information into model
selection and yield optimistic, untrustworthy results. After the best hyperparameters are
selected, the final model is refit and evaluated **once** on the untouched test set for an
honest performance estimate. In scikit-learn this workflow is captured by `GridSearchCV`/
`RandomizedSearchCV`, which return the best parameters and a refitted best estimator.

**Q2. Explain regularization, comparing L1 and L2, including their mathematical form,
effects, and when to use each.**

*Answer:* **Regularization** combats overfitting by adding a penalty for large weights to
the loss, so the optimiser balances fitting the data against keeping the model simple;
this raises bias slightly but reduces variance, usually improving generalisation. **L2
(Ridge)** adds λΣwⱼ² (the sum of squared weights) to the loss; its smooth, circular
constraint **shrinks** all weights toward zero without (generally) eliminating them,
distributing influence across correlated features and stabilising the model — ideal under
**multicollinearity** and as a safe default. **L1 (Lasso)** adds λΣ|wⱼ| (the sum of
absolute weights); its diamond-shaped constraint has corners on the axes, so the optimum
often lands where some weights are **exactly zero**, performing automatic **feature
selection** and yielding a sparse, interpretable model — ideal when you suspect many
features are irrelevant. **Elastic Net** combines both penalties to get sparsity and
stability together. In all cases the **regularization strength** (λ, i.e. `alpha` in
scikit-learn, or inversely `C` for SVM/logistic regression) is a complexity dial tuned by
cross-validation: too little leaves overfitting, too much causes underfitting. Features
should be **scaled** first, since these penalties depend on weight magnitudes. The same
principles extend to other models as **early stopping**, **dropout**, and **data
augmentation** in deep learning.

## Exercises

1. Explain the difference between a parameter and a hyperparameter with an example.
2. For a grid of `C ∈ {0.1,1,10}` and `gamma ∈ {0.01,0.1}`, how many model fits does
   5-fold grid search perform?
3. State which regularization (L1/L2) you'd use for feature selection and which for
   multicollinearity.
4. Explain what happens as the regularization strength goes from 0 to very large.
5. Why must hyperparameter tuning use cross-validation, not the test set?

## Mini-Project

**Project: Tune and regularize a model.**

1. On a dataset, build a model and use `GridSearchCV` (then `RandomizedSearchCV`) to tune
   2–3 hyperparameters with 5-fold CV; report the best settings and CV score.
2. Compare grid vs random search: how many fits did each use, and how close were the best
   scores?
3. For a linear model, train Ridge and Lasso across several `alpha` values; plot R² and the
   number of nonzero coefficients vs `alpha`.
4. Evaluate the final tuned model **once** on the test set.
5. Write a short report on what tuning and regularization achieved. Save in
   `my-ml-journey/`.

## Assignments

1. **Coding:** Implement a manual cross-validated search over one hyperparameter (a loop +
   `cross_val_score`) and verify it matches `GridSearchCV`'s choice.
2. **Coding:** Show Lasso's feature selection: increase `alpha` and print how many
   coefficients become zero at each level.
3. **Conceptual:** Write one page explaining how tuning and regularization both serve the
   bias–variance trade-off, with diagrams.

::: tip
**Part IV complete!** You can now build, evaluate, tune, and regularize the full toolbox of
supervised algorithms. **Part V** explores learning *without labels* — clustering,
dimensionality reduction, and reinforcement learning — opening up a whole new world of what
machines can discover on their own.
:::
