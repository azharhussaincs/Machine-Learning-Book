# Decision Trees

## Introduction

A **Decision Tree** is the most *human* of all algorithms — it makes decisions exactly
the way you might, by asking a series of yes/no questions. "Is the petal narrow? If
yes, it's species A. If no, is it long? …" The result is a flowchart you can read,
explain, and trust.

Decision trees are **highly interpretable** (you can literally see every decision),
handle both classification and regression, need little data preparation (no scaling!),
and — crucially — are the **building blocks** of the most powerful tabular models in
the world: Random Forests (Chapter 23) and Gradient Boosting / XGBoost (Chapter 24).

::: keyidea
A decision tree splits the data into smaller and smaller groups by asking the single
**most informative question** at each step, until each group is as "pure" (single-
class) as possible. Predicting is then just following the questions down to a leaf. No
maths to interpret — just a readable flowchart.
:::

By the end of this chapter you will be able to:

- Understand the anatomy of a tree (root, nodes, branches, leaves).
- Explain how trees choose splits using **Gini impurity** and **entropy / information
  gain**.
- Control overfitting with **depth and pruning** hyperparameters.
- Read **feature importances** from a tree.
- Build, visualise, and tune a tree with scikit-learn.

## Anatomy of a decision tree

![A decision tree for iris classification. The root node asks the most informative question; each branch leads to further questions or to a leaf (a final prediction). To predict, follow the answers from root to leaf.](assets/images/ch21_tree.png)

- **Root node** — the top; the first (most informative) question.
- **Internal nodes** — further questions, each splitting the data.
- **Branches** — the yes/no outcomes of a question.
- **Leaf nodes** — the ends; each gives a final prediction (a class or a value).

To predict, you start at the root and follow the answers down to a leaf.

## How a tree chooses its questions: impurity

At each node, the tree tries *every* possible split and picks the one that makes the
resulting groups **purest** (most dominated by one class). "Purity" is measured by
**Gini impurity** or **entropy**.

### Gini impurity

<div class="equation"><img class="eq" src="assets/images/eq_ch21_gini.png" alt="Gini impurity"></div>

Here `pₖ` is the fraction of class `k` in the node. **Gini = 0** means perfectly pure
(all one class); **Gini = 0.5** (for two classes) means maximally mixed.

```python
def gini(ps): return 1 - sum(p*p for p in ps)
print("gini([0.5, 0.5]):", gini([0.5, 0.5]))   # maximally mixed
print("gini([1, 0]):    ", gini([1, 0]))        # perfectly pure
```

**Output:**
```text
gini([0.5, 0.5]): 0.5
gini([1, 0]):     0
```

### Entropy and information gain

**Entropy** is an alternative impurity measure from information theory:

<div class="equation"><img class="eq" src="assets/images/eq_ch21_entropy.png" alt="entropy"></div>

The tree chooses the split with the highest **information gain** — the reduction in
impurity from parent to children:

<div class="equation"><img class="eq" src="assets/images/eq_ch21_infogain.png" alt="information gain"></div>

::: note
**Gini vs entropy:** they usually produce very similar trees. Gini is slightly faster
to compute (no logarithm) and is scikit-learn's default. Both reward splits that
separate the classes cleanly. You rarely need to worry about which to pick.
:::

The tree is built **greedily and recursively**: pick the best split for the root,
then repeat for each child group, and so on — until a stopping rule is hit.

## Controlling overfitting: depth and pruning

Left unchecked, a tree will keep splitting until every leaf is perfectly pure —
**memorising the training data** (classic overfitting). We control this with
hyperparameters:

- **`max_depth`** — the maximum number of question-levels (the most important dial).
- **`min_samples_split`** — don't split a node with fewer than this many samples.
- **`min_samples_leaf`** — each leaf must keep at least this many samples.
- **`max_leaf_nodes`** — cap the total number of leaves.

Limiting these is a form of **pre-pruning**. (Post-pruning grows a full tree then trims
it, e.g. via `ccp_alpha` cost-complexity pruning.)

### Seeing overfitting with depth

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

for d in [1, 2, 3, 5, None]:
    m = DecisionTreeClassifier(max_depth=d, random_state=1).fit(X_tr, y_tr)
    tr = accuracy_score(y_tr, m.predict(X_tr))
    te = accuracy_score(y_te, m.predict(X_te))
    print(f"depth={str(d):4s}: train={tr:.3f} test={te:.3f}")
```

**Output:**
```text
depth=1   : train=0.667 test=0.667
depth=2   : train=0.952 test=0.956
depth=3   : train=0.952 test=0.978
depth=5   : train=0.990 test=0.978
depth=None: train=1.000 test=0.978
```

![As a decision tree grows deeper, training accuracy keeps rising toward 100% (memorising), but test accuracy plateaus — the gap is overfitting. A moderate depth generalises best.](assets/images/ch21_depth_overfit.png)

### Explanation

- **depth=1** *underfits*: one question isn't enough (66.7% on both).
- **depth=3** is the **sweet spot**: 0.978 on test with no overfitting gap.
- **depth=None** (unlimited) reaches **100% on training** but the same 0.978 on test —
  the train-test gap signals **overfitting**. The deeper tree memorised training noise
  without improving generalisation.

::: keyidea
This is the bias–variance trade-off (Chapter 16) made vividly concrete. A single dial
(`max_depth`) moves the tree from underfitting (too shallow) to overfitting (too deep).
Tuning it — ideally with cross-validation — is the core skill of using trees.
:::

## Reading the tree's rules and feature importances

Trees are transparent — you can print their exact rules:

```python
from sklearn.tree import export_text
m = DecisionTreeClassifier(max_depth=2, random_state=1).fit(X_tr, y_tr)
print(export_text(m, feature_names=load_iris().feature_names))
```

**Output:**
```text
|--- petal width (cm) <= 0.75
|   |--- class: 0
|--- petal width (cm) >  0.75
|   |--- petal length (cm) <= 4.75
|   |   |--- class: 1
|   |--- petal length (cm) >  4.75
|   |   |--- class: 2
```

You can read this aloud: *"If petal width ≤ 0.75 → setosa; otherwise, if petal length ≤
4.75 → versicolor, else virginica."* No black box — every decision is visible.

Trees also rank features by how much they reduce impurity (**feature importance**):

```text
feature importances: {'sepal length': 0.0, 'sepal width': 0.0,
                      'petal length': 0.41, 'petal width': 0.59}
```

Just like in Chapter 13, the **petal** features carry all the importance; the sepal
features were never even used. Feature importance is a major practical benefit of
trees.

::: tip
**Practical & debugging tips:** (1) Trees need **no feature scaling** (they split on
thresholds) — don't waste time scaling for them. (2) Always set `max_depth` (or other
limits) and tune with cross-validation, or trees overfit. (3) `random_state` makes
trees reproducible (split ties are broken randomly). (4) Use `plot_tree` /
`export_text` to *show* stakeholders the logic — a huge selling point. (5) Single trees
are unstable (small data changes → different tree); ensembles (Chapters 23–24) fix this.
:::

## Decision trees for regression

Trees also do regression: instead of a class vote, each leaf predicts the **average
target value** of its samples, and splits are chosen to reduce **variance / MSE** rather
than Gini. The result is a step-like prediction surface.

## Advantages, disadvantages, and use cases

| Advantages | Disadvantages |
|---|---|
| Highly interpretable (a flowchart) | Prone to overfitting (deep trees) |
| No feature scaling needed | Unstable (small changes → different tree) |
| Handles numeric & categorical features | Greedy splitting → not globally optimal |
| Captures non-linear patterns & interactions | Can be biased toward many-valued features |
| Gives feature importances | A single tree is rarely the most accurate |

**Use cases:** credit approval and risk rules, medical decision support, churn analysis,
any setting needing **explainable** decisions — and, above all, as the **base learner**
for Random Forests and Gradient Boosting.

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Letting trees grow unlimited.** An unpruned tree memorises the training
data. Always limit depth/leaves and validate.
:::

- **Mistake 2 — Scaling features for trees** (unnecessary; they're scale-invariant).
- **Mistake 3 — Trusting a single tree's stability** — tiny data changes can reshape it;
  use ensembles.
- **Mistake 4 — Reading too much into exact split thresholds** as if they're causal.
- **Mistake 5 — Expecting a single tree to win on accuracy** — it's a building block,
  not usually the final model.
- **Mistake 6 — Ignoring class imbalance**, which biases splits (use `class_weight`).

## Best practices

- **Tune `max_depth` (and `min_samples_leaf`)** with cross-validation.
- **Don't scale** features for trees.
- **Visualise** the tree (`plot_tree`/`export_text`) for insight and communication.
- **Use feature importances** to understand and select features.
- **Prefer ensembles** (Random Forest, XGBoost) when you want top accuracy and
  stability.

## Chapter Summary

- A **decision tree** classifies/predicts by asking a series of yes/no questions from a
  **root** down to a **leaf** — a readable flowchart.
- It chooses splits that maximise purity, measured by **Gini impurity** or **entropy /
  information gain**; it builds greedily and recursively.
- Unchecked, trees **overfit** (depth=None hit 100% train, 0.978 test on iris);
  control with **`max_depth`**, **`min_samples_leaf`**, and **pruning**, tuned by
  cross-validation.
- Trees are **interpretable**, need **no scaling**, handle non-linearities, and yield
  **feature importances** (petal features dominated iris).
- A single tree is **unstable** and rarely the most accurate — but it's the **building
  block** of Random Forests and Boosting.

---

::: {.qband}
Practice Zone — Chapter 21
:::

## Multiple-Choice Questions (MCQs)

**Q1.** The topmost node of a decision tree is the:
a) Leaf  b) Branch  c) Root node  d) Stump

**Q2.** Gini impurity of a perfectly pure node (all one class) is:
a) 1  b) 0.5  c) 0  d) −1

**Q3.** Which hyperparameter most directly controls a tree's overfitting?
a) learning_rate  b) max_depth  c) n_neighbors  d) alpha

**Q4.** Decision trees require feature scaling:
a) Always  b) Never (they're scale-invariant)  c) Only for regression  d) Only for
classification

**Q5.** Trees choose splits to maximise:
a) Distance  b) Purity / information gain  c) The learning rate  d) The number of leaves

**Q6.** A tree with `max_depth=None` on training data tends to:
a) Underfit  b) Overfit (memorise)  c) Ignore the data  d) Scale features

**Q7.** A leaf node in a classification tree outputs:
a) A question  b) A predicted class  c) A distance  d) A gradient

**Q8.** Single decision trees are described as:
a) Very stable  b) Unstable (sensitive to data changes)  c) Always optimal  d) Needing
scaling

### MCQ Answers
**1:** c. **2:** c. **3:** b. **4:** b. **5:** b. **6:** b. **7:** b. **8:** b.

## Interview Questions (with answers)

**Q1. How does a decision tree decide where to split?**
*Answer:* At each node it evaluates candidate splits and picks the one that most reduces
impurity in the resulting children — measured by Gini impurity or entropy
(maximising information gain). It builds greedily and recursively until a stopping
criterion (e.g. max depth, min samples) is met.

**Q2. What is the difference between Gini impurity and entropy?**
*Answer:* Both measure node impurity (how mixed the classes are). Gini = 1 − Σpₖ²;
entropy = −Σpₖlog₂pₖ. They usually yield very similar trees; Gini is slightly faster
(no log) and is scikit-learn's default, while entropy/information gain comes from
information theory.

**Q3. Why do decision trees overfit, and how do you prevent it?**
*Answer:* Unconstrained, a tree keeps splitting until leaves are pure, memorising noise.
Prevent it by limiting `max_depth`, `min_samples_split`, `min_samples_leaf`, or
`max_leaf_nodes` (pre-pruning), or by post-pruning (`ccp_alpha`), tuned with
cross-validation; and by using ensembles.

**Q4. Why don't decision trees need feature scaling?**
*Answer:* They split on thresholds (e.g. "petal width ≤ 0.75"). Any monotonic rescaling
of a feature just shifts the threshold correspondingly, producing the same partition, so
scaling has no effect.

**Q5. What is a key weakness of a single decision tree, and how is it addressed?**
*Answer:* Instability — small changes in the data can produce a very different tree —
and limited accuracy. Ensembles address this: Random Forests average many trees to
reduce variance, and boosting combines trees to reduce bias (Chapters 23–24).

## Scenario-Based Questions (with answers)

**Q1.** *A bank needs a loan-approval model that auditors can fully explain to
regulators. Which algorithm fits and why?*
*Answer:* A decision tree (or a shallow, pruned one). Its decisions are an explicit set
of if/then rules that can be printed and audited, satisfying the explainability and
compliance requirement that black-box models struggle with.

**Q2.** *Your decision tree scores 100% on training but 78% on test. What's wrong and
what do you change?*
*Answer:* Overfitting from an unconstrained (too deep) tree. Limit `max_depth`, increase
`min_samples_leaf`, or prune, choosing values via cross-validation; consider switching to
a Random Forest for stability and accuracy.

**Q3.** *A colleague carefully standardises all features before training a decision
tree and is surprised it changes nothing. Why?*
*Answer:* Trees are scale-invariant — they split on thresholds, so monotonic scaling
doesn't alter the splits or predictions. The standardisation was unnecessary effort for
a tree (though it would matter for KNN/SVM/logistic regression).

## Logic-Based Questions (with answers)

**Q1.** Why does a tree's training accuracy approach 100% as depth increases without
limit?
*Answer:* With enough depth, the tree can keep splitting until each leaf contains a
single (or pure) group of training points, perfectly fitting them — but this memorises
noise and doesn't improve test accuracy.

**Q2.** If two features are equally predictive but one is used at the root, why might
the other show low importance?
*Answer:* Once the root split uses one feature, much of the class separation is already
achieved, leaving little impurity for the correlated feature to reduce — so it gets low
importance even though it was equally informative in isolation.

**Q3.** A two-class node has class proportions [0.5, 0.5]. Why is its Gini exactly the
maximum (0.5)?
*Answer:* Gini = 1 − (0.5² + 0.5²) = 1 − 0.5 = 0.5. Equal proportions mean maximum
mixing/uncertainty, which is the highest impurity for two classes.

## Practical Questions (with answers)

**Q1.** Write code to train a decision tree with maximum depth 4.
*Answer:* `DecisionTreeClassifier(max_depth=4).fit(X_train, y_train)`.

**Q2.** How do you print a fitted tree's human-readable rules in scikit-learn?
*Answer:* `from sklearn.tree import export_text; print(export_text(model,
feature_names=feature_names))` (or use `plot_tree` for a visual).

**Q3.** Which attribute gives a fitted tree's feature importances?
*Answer:* `model.feature_importances_`.

## Long Questions (with answers)

**Q1. Explain how a decision tree is built and how it makes predictions, including the
role of impurity measures and stopping criteria.**

*Answer:* A decision tree is built **greedily and recursively** from the top down.
Starting at the **root** with all the training data, the algorithm evaluates every
possible split (a feature and a threshold) and selects the one that most reduces
**impurity** in the resulting child nodes. Impurity is measured by **Gini** (1 − Σpₖ²)
or **entropy** (−Σpₖlog₂pₖ); the best split maximises **information gain**, the impurity
of the parent minus the weighted impurity of the children. The data is partitioned by
that split, and the process repeats on each child, growing the tree until a **stopping
criterion** is met — such as reaching `max_depth`, having too few samples to split
(`min_samples_split`), pure leaves, or a leaf-count cap. To **predict**, a new sample
starts at the root and follows the branch matching each question's answer until it
reaches a **leaf**, which outputs the majority class (classification) or the average
target (regression) of the training samples that landed there. Stopping criteria and
pruning are essential because, without them, the tree keeps splitting until it
memorises the training data, overfitting; constraining depth and leaf size keeps it at
the bias–variance sweet spot.

**Q2. Discuss the strengths and weaknesses of decision trees and explain why they are
the foundation of ensemble methods.**

*Answer:* **Strengths:** decision trees are exceptionally **interpretable** — their
decisions form an explicit, auditable flowchart of if/then rules; they require **no
feature scaling** (being threshold-based and scale-invariant); they handle numeric and
categorical features and naturally capture **non-linear relationships and
interactions**; and they provide **feature importances**. **Weaknesses:** a single
unconstrained tree **overfits** by memorising training data; trees are **unstable** —
small data changes can yield a very different tree; their **greedy** construction finds
locally, not globally, optimal splits; they can be **biased toward features with many
levels**; and individually they are rarely the most accurate model. These very
weaknesses are why trees are the perfect **base learner for ensembles**: because a
single tree is high-variance but low-bias when grown deep, **bagging** many trees on
bootstrap samples and averaging them (Random Forest, Chapter 23) cancels out the
variance and instability, yielding a robust, accurate model; and because shallow trees
are weak but fast, **boosting** can combine many of them sequentially, each correcting
the last, to reduce bias and achieve state-of-the-art accuracy (Gradient Boosting/
XGBoost, Chapter 24). Thus the humble, interpretable tree becomes the cornerstone of
the most powerful tabular-data models in practice.

## Exercises

1. Draw a small decision tree for "should I take an umbrella?" using two features
   (cloudy?, rain forecast?).
2. Compute Gini impurity for a node with class proportions [0.7, 0.3].
3. Explain why a tree of depth 1 underfits the 3-class iris problem.
4. List four hyperparameters that control tree overfitting.
5. Explain in one sentence why trees don't need feature scaling.

## Mini-Project

**Project: Tune and visualise a decision tree.**

1. Load a dataset (iris, wine, or Titanic after Part III cleaning).
2. Loop `max_depth` from 1 to 10, recording train and test accuracy; plot both curves
   (Chapter 14) and identify where overfitting begins.
3. Train the best-depth tree and **visualise** it with `plot_tree`.
4. Print and interpret the feature importances and the top rules (`export_text`).
5. Write a short report on the depth–overfitting relationship. Save in `my-ml-journey/`.

## Assignments

1. **Coding:** Train a decision tree on a dataset, then prune it with `ccp_alpha`
   (cost-complexity pruning). Compare accuracy and tree size before and after.
2. **Coding:** Build a **regression** tree (`DecisionTreeRegressor`) on a numeric target
   and visualise its step-like predictions on one feature.
3. **Conceptual:** Write one page explaining Gini vs entropy and why a single tree is
   unstable, motivating the need for ensembles.

::: tip
A single tree is interpretable but unstable. Chapter 23, **Random Forest**, combines
*many* trees to dramatically boost accuracy and stability — but first, Chapter 22 covers
**Support Vector Machines**, a powerful and elegant approach that finds the widest
possible margin between classes.
:::
