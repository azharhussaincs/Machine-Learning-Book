# K-Nearest Neighbors (KNN)

## Introduction

So far our models *learned* equations (weights) from data. **K-Nearest Neighbors
(KNN)** does something refreshingly different and intuitive: it **remembers all the
training data** and, to classify a new point, simply looks at its **closest
neighbours** and lets them vote.

The whole philosophy is captured by an old saying: *"You are the average of the five
people you spend the most time with."* KNN predicts that a new point is like the points
nearest to it.

::: keyidea
KNN is a **lazy, instance-based** learner: it does almost no work during "training"
(it just stores the data) and does *all* its work at prediction time (finding the
nearest neighbours). There's no equation to learn — the data *is* the model.
:::

By the end of this chapter you will be able to:

- Explain how KNN classifies and predicts via nearest neighbours.
- Understand **distance metrics** (Euclidean, Manhattan).
- Choose **k** wisely and understand its effect on the bias–variance trade-off.
- Know why **feature scaling is critical** for KNN.
- Implement KNN with scikit-learn and understand its pros, cons, and use cases.

## How KNN works

To predict for a new point:

1. **Compute the distance** from the new point to *every* training point.
2. **Find the k closest** training points (the "k nearest neighbours").
3. **Vote** (classification) or **average** (regression):
   - *Classification:* the new point gets the **majority class** among its k neighbours.
   - *Regression:* the new point gets the **average value** of its k neighbours.

![KNN classifies a new point (the star) by the majority vote of its k nearest neighbours. With k=3 the three closest points decide the class; changing k can change the answer.](assets/images/ch19_knn_vote.png)

There is **no training phase** in the usual sense — KNN just stores the data and does
the work at prediction time.

## Distance metrics

"Nearest" requires a definition of distance. The most common is **Euclidean distance**
— the straight-line distance you learned in geometry (the Pythagorean theorem,
Chapter 5):

<div class="equation"><img class="eq" src="assets/images/eq_ch19_euclidean.png" alt="Euclidean distance"></div>

For example, the distance from `(0,0)` to `(3,4)` is `√(3² + 4²) = √25 = 5`.

Another common metric is **Manhattan distance** (sum of absolute differences — like
walking city blocks):

<div class="equation"><img class="eq" src="assets/images/eq_ch19_manhattan.png" alt="Manhattan distance"></div>

Both are special cases of the general **Minkowski distance**. Euclidean is the default
and works well for most problems.

## Choosing k: the key hyperparameter

`k` (the number of neighbours) controls everything — and it's the bias–variance
trade-off (Chapter 16) in action:

- **Small k (e.g. 1)** — very flexible, follows every point; **low bias, high
  variance** → can **overfit** and be fooled by noise.
- **Large k** — very smooth, averages over many points; **high bias, low variance** →
  can **underfit** and blur class boundaries.

![The effect of k on KNN's decision boundary. Small k (left) gives a jagged, overfit boundary; large k (right) gives a smooth, possibly underfit one. A middle value generalises best.](assets/images/ch19_k_effect.png)

::: tip
**Choosing k in practice:** (1) Try a range of k values with cross-validation and pick
the best (we did a mini version in Chapter 2). (2) Use an **odd k** for binary
classification to avoid tie votes. (3) A common rule of thumb is k ≈ √n, but always
validate. (4) Plot accuracy vs k to see the sweet spot.
:::

## Why feature scaling is CRITICAL for KNN

Because KNN relies entirely on **distances**, a feature with a large range will
**dominate** the distance and drown out other features. This makes scaling (Chapter 11)
not optional but essential.

```python
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X, y = load_wine(return_X_y=True)        # features on very different scales
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
sc = StandardScaler().fit(X_tr)
X_tr_s, X_te_s = sc.transform(X_tr), sc.transform(X_te)

acc_unscaled = accuracy_score(
    y_te, KNeighborsClassifier(5).fit(X_tr,   y_tr).predict(X_te))
acc_scaled = accuracy_score(
    y_te, KNeighborsClassifier(5).fit(X_tr_s, y_tr).predict(X_te_s))
print("k=5 UNSCALED:", round(acc_unscaled, 3))
print("k=5 SCALED:  ", round(acc_scaled, 3))
```

**Output:**
```text
k=5 UNSCALED: 0.722
k=5 SCALED:   0.944
```

::: keyidea
Scaling jumped accuracy from **72% to 94%** — a *massive* difference from one
preprocessing step! On the wine data, one feature (`proline`) ranges in the hundreds
while others are near 1, so unscaled it dominated every distance. **For KNN, always
scale your features.** This single lesson is worth the whole chapter.
:::

## Choosing k: the effect on accuracy

```python
for k in [1, 3, 5, 7, 15]:
    m = KNeighborsClassifier(k).fit(X_tr_s, y_tr)
    print(f"k={k:2d}: {accuracy_score(y_te, m.predict(X_te_s)):.3f}")
```

**Output:**
```text
k= 1: 0.963
k= 3: 0.944
k= 5: 0.944
k= 7: 0.944
k=15: 0.981
```

Accuracy varies with k — here k=15 happened to do best on this split. In practice you'd
use cross-validation (not a single split) to choose k reliably.

## The curse of dimensionality (again)

KNN struggles badly in **high dimensions** (many features). As dimensions grow, all
points become roughly **equidistant**, so "nearest" loses meaning (the curse of
dimensionality, Chapter 13). KNN works best with relatively few, well-chosen, scaled
features — pair it with feature selection (Chapter 13) or dimensionality reduction
(Chapter 28) when you have many features.

## Advantages, disadvantages, and use cases

| Advantages | Disadvantages |
|---|---|
| Simple, intuitive, no training time | **Slow at prediction** (compares to all data) |
| No assumptions about data shape | Needs lots of memory (stores all data) |
| Naturally handles multiclass | **Very sensitive to feature scale** |
| Can model complex boundaries | Suffers in high dimensions |
| Works for classification & regression | Sensitive to noisy data and the choice of k |

**Use cases:** recommendation systems ("users like you also liked…"), simple image
classification, anomaly detection, filling missing values (`KNNImputer`), and as an
easy, strong baseline on small, low-dimensional, scaled datasets.

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Forgetting to scale features.** The single biggest KNN error. As we saw,
it can cost you 20+ points of accuracy.
:::

- **Mistake 2 — Using a single train/test split to pick k** instead of cross-validation.
- **Mistake 3 — Using an even k** for binary classification (causes tie votes).
- **Mistake 4 — Using KNN on very high-dimensional data** without reducing dimensions.
- **Mistake 5 — Expecting fast predictions** on large datasets (KNN is slow at predict
  time).
- **Mistake 6 — Thinking KNN "trains"** — it mostly just stores data; the work is at
  prediction.

## Best practices

- **Always scale features** before KNN.
- **Choose k by cross-validation**; prefer odd k for binary problems.
- **Reduce dimensions / select features** for high-dimensional data.
- **Use efficient structures** (KD-trees/Ball-trees, which scikit-learn uses) for
  faster neighbour search.
- **Treat KNN as a strong baseline** on small, clean, low-dimensional datasets.

## Chapter Summary

- **KNN** is a **lazy, instance-based** algorithm: it stores the training data and, for
  a new point, finds its **k nearest neighbours** and **votes** (classification) or
  **averages** (regression).
- "Nearest" uses a **distance metric** — usually **Euclidean** (straight-line) or
  **Manhattan**.
- **k** is the key hyperparameter: small k → overfit (high variance), large k →
  underfit (high bias); choose it with **cross-validation**.
- **Feature scaling is critical** — on wine data it raised accuracy from **0.72 to
  0.94**.
- KNN is simple and flexible but **slow at prediction**, **memory-hungry**, and weak in
  **high dimensions** — best on small, scaled, low-dimensional data.

---

::: {.qband}
Practice Zone — Chapter 19
:::

## Multiple-Choice Questions (MCQs)

**Q1.** KNN is described as a ___ learner.
a) Eager  b) Lazy / instance-based  c) Probabilistic  d) Linear

**Q2.** To classify a new point, KNN uses the:
a) Average of all data  b) Majority vote of its k nearest neighbours  c) A learned
equation  d) Random guess

**Q3.** The most common distance metric in KNN is:
a) Cosine  b) Euclidean  c) Hamming  d) Jaccard

**Q4.** A very small k (e.g. 1) tends to:
a) Underfit  b) Overfit (high variance)  c) Be unbiased  d) Ignore the data

**Q5.** Which preprocessing step is most critical for KNN?
a) One-hot encoding  b) Feature scaling  c) Removing the target  d) Adding features

**Q6.** KNN's main weakness at prediction time is:
a) It can't classify  b) It's slow (compares to all data)  c) It overfits always
d) It needs labels at predict time

**Q7.** For binary classification, k is often chosen to be:
a) Even  b) Odd (to avoid ties)  c) Exactly 2  d) As large as possible

**Q8.** KNN performs poorly when there are:
a) Few features  b) Many features (high dimensions)  c) Scaled features  d) Two classes

### MCQ Answers
**1:** b. **2:** b. **3:** b. **4:** b. **5:** b. **6:** b. **7:** b. **8:** b.

## Interview Questions (with answers)

**Q1. How does the KNN algorithm work?**
*Answer:* It stores all training data. To predict for a new point, it computes the
distance to every training point, selects the k closest, and outputs the majority class
(classification) or the average value (regression) among them. There is no parametric
training phase.

**Q2. Why is KNN called a "lazy" learner?**
*Answer:* Because it does no real work at training time — it just memorises the data —
and defers all computation (distance calculations and neighbour search) to prediction
time, making training trivial but prediction expensive.

**Q3. How does the choice of k affect the model?**
*Answer:* Small k makes the model flexible and sensitive to noise (low bias, high
variance → overfitting); large k makes it smooth and general (high bias, low variance →
underfitting). k is chosen via cross-validation to balance the trade-off; odd k avoids
ties in binary classification.

**Q4. Why must features be scaled for KNN?**
*Answer:* KNN decisions depend entirely on distances. A feature with a large numeric
range dominates the distance calculation, so unscaled features can make the model
ignore others. Scaling (e.g. standardisation) puts features on comparable ranges so all
contribute fairly — often a large accuracy difference.

**Q5. What are the main limitations of KNN?**
*Answer:* Slow, memory-heavy predictions (it compares to all stored data), strong
sensitivity to feature scale and noisy points, the need to choose k, and poor
performance in high dimensions due to the curse of dimensionality.

## Scenario-Based Questions (with answers)

**Q1.** *Your KNN model gives 72% accuracy. A colleague gets 94% on the same data and
k. What did they likely do differently?*
*Answer:* They almost certainly **scaled the features** (e.g. StandardScaler). KNN is
distance-based, so unscaled features with large ranges dominate; scaling typically
gives a big accuracy boost, exactly as in this chapter's wine example.

**Q2.** *KNN predictions are too slow for your real-time app serving millions of
requests. What can you do?*
*Answer:* Use efficient neighbour-search structures (KD-trees/Ball-trees), reduce the
dataset (prototype selection), reduce dimensions (PCA), reduce k, or switch to a model
with fast inference (e.g. logistic regression or a trained tree/forest). KNN's
prediction cost grows with data size.

**Q3.** *With k=1 your model is perfect on training data but poor on test data. Why,
and what do you change?*
*Answer:* k=1 means each point's nearest neighbour is itself in training (perfect
recall) but the model overfits noise, hurting generalisation. Increase k (and use
cross-validation to pick it) to smooth the decision boundary and reduce variance.

## Logic-Based Questions (with answers)

**Q1.** Why does k=1 give 100% training accuracy?
*Answer:* For any training point, its single nearest neighbour is itself (distance 0),
so it always "votes" its own correct label — perfect on training, but this doesn't
reflect generalisation.

**Q2.** In two dimensions, the distance from (0,0) to (3,4) is 5. Show why.
*Answer:* Euclidean distance = √((3−0)² + (4−0)²) = √(9 + 16) = √25 = 5 — the
Pythagorean theorem.

**Q3.** Why does KNN degrade as the number of features grows very large?
*Answer:* In high dimensions, distances between points become nearly equal (the curse of
dimensionality), so "nearest" neighbours are barely nearer than far ones, destroying the
signal KNN relies on.

## Practical Questions (with answers)

**Q1.** Write code to train a KNN classifier with 7 neighbours.
*Answer:* `KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)`.

**Q2.** Why did scaling raise wine accuracy from 0.72 to 0.94?
*Answer:* The wine features have very different ranges (e.g. proline in the hundreds vs
others near 1). Unscaled, the large-range feature dominated the distance; scaling let
all features contribute, dramatically improving neighbour quality.

**Q3.** How would you choose the best k reliably?
*Answer:* Use cross-validation over a range of k values (e.g. `GridSearchCV` or
`cross_val_score` in a loop) and pick the k with the best mean validation score, rather
than relying on one train/test split.

## Long Questions (with answers)

**Q1. Explain the KNN algorithm in full: how it predicts, the role of distance and k,
and why scaling matters, with the bias–variance perspective.**

*Answer:* KNN is a lazy, instance-based learner that stores the entire training set and
predicts at query time. To classify a new point, it (1) computes the distance — usually
**Euclidean**, √Σ(pᵢ−qᵢ)² — from that point to every training point, (2) selects the
**k nearest** of them, and (3) takes the **majority class** (for classification) or the
**average target** (for regression) of those neighbours. The hyperparameter **k**
governs the bias–variance trade-off: a small k (e.g. 1) makes a highly flexible model
that hugs individual points, giving low bias but high variance and a tendency to overfit
noise; a large k averages over many points, giving high bias but low variance and a
smoother, possibly underfit boundary. k is therefore chosen by cross-validation, often
odd for binary tasks to avoid ties. Because every decision is based on distances,
**feature scaling is essential**: a feature with a large numeric range dominates the
distance and drowns out the rest, so standardising features (Chapter 11) can transform
performance — as shown on the wine dataset, where scaling raised accuracy from 0.72 to
0.94. KNN makes no assumptions about the data's functional form and can model complex
boundaries, but it is slow and memory-heavy at prediction and degrades in high
dimensions, where distances lose meaning.

**Q2. Compare KNN with logistic regression, discussing how they learn, their
assumptions, costs, and when to prefer each.**

*Answer:* **How they learn:** Logistic regression is an *eager, parametric* learner —
it learns a fixed set of weights during training by minimising log-loss, then discards
the data and predicts cheaply via a single dot product and sigmoid. KNN is a *lazy,
non-parametric* learner — it stores all training data and computes distances to
neighbours at prediction time, with essentially no training. **Assumptions:** Logistic
regression assumes a roughly linear decision boundary (in the feature space), so it
underfits strongly non-linear data unless features are engineered; KNN assumes only that
nearby points share labels, letting it fit complex, non-linear boundaries directly.
**Costs:** Logistic regression has slow-ish training but very fast, memory-light
prediction and scales to high dimensions and large data; KNN has trivial training but
slow, memory-heavy prediction and degrades badly in high dimensions. **Interpretability
& output:** Logistic regression gives interpretable coefficients and calibrated
probabilities; KNN gives no global model and only crude probability estimates from
neighbour proportions. **When to prefer each:** choose logistic regression for
high-dimensional, large, or latency-sensitive problems, when you need probabilities or
interpretability, or as a fast baseline; choose KNN for small, low-dimensional, well-
scaled datasets with complex boundaries, for recommendation-style "similar items"
tasks, or when a simple, assumption-free method suffices. In practice both are quick to
try, and (per No Free Lunch, Chapter 16) you compare them empirically.

## Exercises

1. Compute the Euclidean distance between (1, 2) and (4, 6) by hand.
2. Explain how the prediction changes for the same point when k goes from 1 to 51.
3. Why should k usually be odd for binary classification?
4. Give two real applications where KNN is a natural fit.
5. Explain in your own words why scaling is essential for KNN but not for decision
   trees.

## Mini-Project

**Project: KNN tuning and scaling study.**

1. Load the wine (or breast cancer) dataset and split into train/test.
2. Train KNN with and without scaling at k=5 and compare accuracy — confirm scaling
   helps.
3. With scaling on, loop k over [1, 3, 5, …, 25] and plot accuracy vs k (Chapter 14).
4. Identify the best k via cross-validation (`cross_val_score`).
5. Write a short report on the scaling effect and the best k. Save in `my-ml-journey/`.

## Assignments

1. **Coding:** Implement KNN classification from scratch (compute distances, find k
   nearest, majority vote) and verify it matches scikit-learn on a small dataset.
2. **Coding:** Compare Euclidean vs Manhattan distance (`metric="manhattan"`) on a
   dataset — does it change accuracy?
3. **Conceptual:** Write one page explaining why KNN is "lazy", how the curse of
   dimensionality affects it, and three ways to mitigate it.

::: tip
KNN decides by *similarity*. Chapter 20, **Naive Bayes**, decides by *probability* —
applying Bayes' theorem (Chapter 6) with a clever simplifying assumption to build a
fast, surprisingly effective classifier, especially for text.
:::
