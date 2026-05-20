# Support Vector Machines (SVM)

## Introduction

**Support Vector Machines (SVM)** are among the most elegant and powerful
classification algorithms ever invented. They dominated machine learning in the
1990s–2000s (Chapter 3) and remain excellent today, especially on **small-to-medium,
high-dimensional datasets** like text and gene data.

The core idea is geometric and beautiful: of all the possible lines that could
separate two classes, SVM finds the one with the **widest possible "street"** between
them. A wider street means a safer, more confident, more generalisable boundary.

::: keyidea
SVM doesn't just find *a* separating boundary — it finds the **maximum-margin**
boundary: the one as far as possible from the nearest points of each class. Those
nearest points are the **support vectors**; they alone define the boundary. And with
the **kernel trick**, SVM can draw curved boundaries to separate data that no straight
line ever could.
:::

By the end of this chapter you will be able to:

- Explain the **margin**, the **support vectors**, and why "maximum margin" is good.
- Understand **hard vs soft margins** and the **C** hyperparameter.
- Understand the **kernel trick** (linear, polynomial, RBF) for non-linear data.
- Tune SVM (**C**, **gamma**, **kernel**) and know its pros, cons, and use cases.

## The maximum-margin idea

Imagine two classes of points. Many lines could separate them — but which is best? SVM
picks the line that maximises the **margin**: the distance to the nearest point on each
side.

![SVM finds the separating line (hyperplane) with the widest margin — the largest gap to the nearest points of each class. Those nearest points, which touch the margin, are the support vectors and alone determine the boundary.](assets/images/ch22_margin.png)

- The **hyperplane** is the decision boundary (a line in 2-D, a plane in 3-D, a
  hyperplane in higher dimensions): `w·x + b = 0`.

<div class="equation"><img class="eq" src="assets/images/eq_ch22_hyperplane.png" alt="hyperplane"></div>

- The **margin** is the width of the "street". SVM maximises it; mathematically the
  margin equals `2 / ‖w‖`, so maximising the margin means minimising `‖w‖`:

<div class="equation"><img class="eq" src="assets/images/eq_ch22_margin.png" alt="margin"></div>

- The **support vectors** are the points sitting *on* the edges of the street. Remove
  any other point and the boundary is unchanged; move a support vector and it shifts.
  This is why SVM is **memory-efficient** — only the support vectors matter.

A wider margin generalises better (more robust to new points), which is the deep reason
SVM works so well.

## Hard margin vs soft margin: the C parameter

Real data is rarely perfectly separable — there's noise and overlap. A **hard margin**
(no mistakes allowed) would fail or overfit. So SVM uses a **soft margin** that allows
some points to be inside the margin or misclassified, controlled by **C**:

- **Large C** — punishes mistakes heavily → narrow margin, fits training data tightly
  → risk of **overfitting** (low bias, high variance).
- **Small C** — tolerates more mistakes → wider margin, simpler boundary → risk of
  **underfitting** (high bias, low variance).

`C` is the SVM's version of the bias–variance dial (Chapter 16), tuned by
cross-validation.

## The kernel trick: separating the unseparable

Here is SVM's most brilliant idea. What if no straight line can separate the classes —
like one class forming a *circle* inside another? The **kernel trick** projects the
data into a higher dimension where it *becomes* linearly separable, then finds a flat
boundary there — which looks **curved** back in the original space. The magic is that
SVM does this *without ever computing the high-dimensional coordinates* (it uses kernel
functions), so it's efficient.

![The kernel trick: data that can't be split by a straight line in 2-D (left) becomes linearly separable when lifted into a higher dimension (right). The flat boundary up there appears curved back in the original space.](assets/images/ch22_kernel.png)

Common **kernels**:

- **Linear** — a straight boundary; fast; great for high-dimensional data (text).
- **Polynomial (poly)** — curved boundaries of a chosen degree.
- **RBF (Radial Basis Function / Gaussian)** — flexible, smooth, curved boundaries; the
  **default** and usually best for non-linear data. Its **gamma** parameter controls how
  wiggly the boundary is (high gamma → very flexible → risk of overfitting).

### Seeing the kernel trick in action

```python
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# One class is a CIRCLE inside the other — not linearly separable
X, y = make_circles(n_samples=300, noise=0.1, factor=0.4, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

lin = SVC(kernel="linear").fit(X_tr, y_tr)
rbf = SVC(kernel="rbf").fit(X_tr, y_tr)
print("circles, linear kernel:", round(accuracy_score(y_te, lin.predict(X_te)), 3))
print("circles, RBF kernel:   ", round(accuracy_score(y_te, rbf.predict(X_te)), 3))
```

**Output:**
```text
circles, linear kernel: 0.411
circles, RBF kernel:    1.0
```

::: keyidea
The linear kernel scored a hopeless **0.411** (worse than guessing) — no straight line
can separate a circle inside a circle. The **RBF kernel scored a perfect 1.0** by
implicitly lifting the data into a space where the classes *are* separable. This is the
kernel trick: turning impossible problems into easy ones. It's why SVM was so
revolutionary.
:::

## Practical: SVM on real data with different kernels

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
sc = StandardScaler().fit(X_tr)             # SVM is distance-based -> ALWAYS scale
X_tr_s, X_te_s = sc.transform(X_tr), sc.transform(X_te)

for k in ["linear", "rbf", "poly"]:
    m = SVC(kernel=k).fit(X_tr_s, y_tr)
    print(f"kernel={k:7s}: {accuracy_score(y_te, m.predict(X_te_s)):.3f}")
```

**Output:**
```text
kernel=linear : 0.982
kernel=rbf    : 0.977
kernel=poly   : 0.895
```

### Explanation

- We **scaled** first — SVM is distance/margin-based and *very* sensitive to feature
  scale (like KNN, Chapter 19). Always scale for SVM.
- On this dataset the **linear** kernel won (0.982) — the classes are nearly linearly
  separable in this high-dimensional space, so the simplest kernel was best (No Free
  Lunch, Chapter 16).
- The **poly** kernel underperformed (0.895) here — more flexibility isn't always
  better. Always compare kernels with cross-validation.

::: tip
**Practical & debugging tips:** (1) **Always scale features** for SVM. (2) Start with
the **RBF kernel** for non-linear problems, **linear** for high-dimensional/text data.
(3) Tune **C** and **gamma** together with `GridSearchCV` — they interact strongly. (4)
SVM doesn't output probabilities by default; use `SVC(probability=True)` (slower) if you
need them. (5) SVM **scales poorly to very large datasets** (training is roughly
O(n²–n³)); for big data prefer `LinearSVC`, SGD, or tree ensembles. (6) Use
`SVR` for regression.
:::

## Advantages, disadvantages, and use cases

| Advantages | Disadvantages |
|---|---|
| Effective in high dimensions | **Slow / memory-heavy on large datasets** |
| Powerful non-linear boundaries (kernels) | Needs careful tuning (C, gamma, kernel) |
| Robust via maximum margin | Sensitive to feature scaling |
| Memory-efficient (only support vectors) | No native probabilities (extra cost) |
| Works with few samples, many features | Hard to interpret (especially with kernels) |

**Use cases:** text and document classification, image classification (classic),
bioinformatics/gene expression (many features, few samples), handwriting recognition,
and any small-to-medium high-dimensional problem.

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Forgetting to scale features.** SVM relies on distances/margins; unscaled
features wreck performance, just like KNN.
:::

- **Mistake 2 — Using SVM on huge datasets** where its O(n²–n³) training is too slow
  (use LinearSVC/SGD/ensembles).
- **Mistake 3 — Not tuning C and gamma** (or tuning them separately) — they interact.
- **Mistake 4 — Assuming RBF is always best** — for high-dimensional text, linear often
  wins.
- **Mistake 5 — Expecting probabilities by default** — enable `probability=True` at a
  cost.
- **Mistake 6 — Very high gamma** causing the RBF boundary to overfit each point.

## Best practices

- **Always scale features** before SVM.
- **Choose the kernel by data**: linear for high-dimensional/text, RBF for non-linear.
- **Grid-search C and gamma together** with cross-validation.
- **Prefer LinearSVC / ensembles** for large datasets.
- **Use SVR** for regression and `probability=True` only when you truly need
  probabilities.

## Chapter Summary

- **SVM** finds the **maximum-margin** boundary — the separating hyperplane with the
  widest "street" to the nearest points; those points are the **support vectors** and
  alone define the boundary.
- The **C** parameter sets the soft-margin trade-off: large C → narrow margin/overfit,
  small C → wide margin/underfit (the bias–variance dial).
- The **kernel trick** (linear, polynomial, **RBF**) draws non-linear boundaries by
  implicitly lifting data to higher dimensions — RBF turned an impossible circles
  problem from **0.41 to 1.0** accuracy.
- SVM is powerful in **high dimensions** and with **few samples**, but **scales poorly
  to large data**, **must be scaled**, and **needs C/gamma tuning**.

---

::: {.qband}
Practice Zone — Chapter 22
:::

## Multiple-Choice Questions (MCQs)

**Q1.** SVM finds the boundary that maximises the:
a) Number of support vectors  b) Margin between classes  c) Training accuracy  d) Depth

**Q2.** Support vectors are:
a) All training points  b) The points nearest the boundary (on the margin)  c) The
features  d) The test points

**Q3.** The kernel trick allows SVM to:
a) Train faster always  b) Separate non-linearly separable data  c) Avoid scaling
d) Output probabilities

**Q4.** A very large C tends to cause:
a) Underfitting  b) Overfitting (narrow margin)  c) Wider margin  d) Faster training

**Q5.** The default, flexible kernel for non-linear data is:
a) linear  b) poly  c) RBF  d) sigmoid

**Q6.** Before training an SVM you should:
a) Remove the target  b) Scale the features  c) One-hot the labels  d) Nothing

**Q7.** SVM's main weakness is:
a) Can't do non-linear  b) Slow/memory-heavy on large datasets  c) Needs no tuning
d) Only binary

**Q8.** In the circles example, why did the linear kernel fail?
a) Bad scaling  b) No straight line can separate a circle inside a circle  c) Too much
data  d) Wrong metric

### MCQ Answers
**1:** b. **2:** b. **3:** b. **4:** b. **5:** c. **6:** b. **7:** b. **8:** b.

## Interview Questions (with answers)

**Q1. What is a Support Vector Machine and what does it optimise?**
*Answer:* A maximum-margin classifier: it finds the separating hyperplane that maximises
the margin (distance) to the nearest points of each class. Maximising the margin
(equivalently minimising ‖w‖) yields a robust boundary that generalises well; the
nearest points defining it are the support vectors.

**Q2. What are support vectors?**
*Answer:* The training points lying on (or within) the margin that determine the
position of the decision boundary. Only they matter — removing other points doesn't
change the model — which makes SVM memory-efficient.

**Q3. Explain the kernel trick.**
*Answer:* It lets SVM learn non-linear boundaries by implicitly mapping data into a
higher-dimensional space where it becomes linearly separable, using kernel functions
(linear, polynomial, RBF) that compute inner products in that space without ever
constructing the coordinates — so it's efficient.

**Q4. What do C and gamma control?**
*Answer:* C is the soft-margin penalty: large C fits training data tightly (narrow
margin, overfit risk), small C allows more violations (wider margin, underfit risk).
Gamma (for RBF) sets how far a single point's influence reaches: high gamma → very
wiggly, local boundary (overfit), low gamma → smoother boundary.

**Q5. When would you not use SVM?**
*Answer:* On very large datasets, where its O(n²–n³) training is too slow (prefer
LinearSVC, SGD, or tree ensembles), or when you need fast native probabilities or strong
interpretability. SVM shines on small-to-medium, high-dimensional data.

## Scenario-Based Questions (with answers)

**Q1.** *Your data has one class forming a ring around another. A linear SVM scores ~40%.
What do you change?*
*Answer:* Switch to a non-linear kernel, typically RBF, which can separate the ring via
the kernel trick — as in this chapter, accuracy jumps to ~100%. Tune C and gamma with
cross-validation.

**Q2.** *Your SVM trains for hours on a 2-million-row dataset. What's the issue and
what are alternatives?*
*Answer:* Kernel SVM scales roughly O(n²–n³), so it's impractical at that size. Use
`LinearSVC` or SGD-based linear models for large data, or switch to tree ensembles
(Random Forest/XGBoost) which handle large tabular data efficiently.

**Q3.** *An SVM performs poorly until a colleague applies StandardScaler, after which it
excels. Why?*
*Answer:* SVM is margin/distance-based, so features with large ranges dominate the
geometry. Scaling puts features on comparable ranges, letting the margin be computed
fairly — often a dramatic improvement, as with KNN.

## Logic-Based Questions (with answers)

**Q1.** Why does maximising the margin tend to improve generalisation?
*Answer:* A wider margin means the boundary is as far as possible from both classes, so
small perturbations or new nearby points are less likely to cross it — making the
classifier more robust and less likely to overfit.

**Q2.** If you delete a non-support-vector point and retrain, why is the boundary
unchanged?
*Answer:* Only support vectors (the points on the margin) define the boundary;
non-support points lie safely beyond the margin and impose no active constraint, so
removing them doesn't move the optimal hyperplane.

**Q3.** Why does the linear kernel sometimes beat RBF on text data?
*Answer:* Text is extremely high-dimensional, where data is often already (nearly)
linearly separable; a linear kernel then suffices and avoids the extra flexibility (and
overfitting risk) of RBF, also training much faster.

## Practical Questions (with answers)

**Q1.** Write code to train an RBF-kernel SVM.
*Answer:* `SVC(kernel="rbf").fit(X_train, y_train)`.

**Q2.** Why must you scale features before SVM?
*Answer:* Because SVM's margin depends on distances; unscaled large-range features
dominate, distorting the boundary. Standardising features lets all contribute fairly.

**Q3.** How do you get probability estimates from an SVM in scikit-learn?
*Answer:* Set `SVC(probability=True)` (it uses Platt scaling internally; slower), then
call `predict_proba`.

## Long Questions (with answers)

**Q1. Explain how SVM works: the margin, support vectors, the soft-margin C parameter,
and why maximum margin generalises well.**

*Answer:* An SVM is a **maximum-margin classifier**. Among all hyperplanes (w·x + b = 0)
that separate two classes, it selects the one whose **margin** — the perpendicular
distance to the nearest points of each class — is largest. Geometrically it finds the
widest possible "street" between the classes; the margin equals 2/‖w‖, so maximising it
means minimising ‖w‖ subject to the points being on the correct side. The points that
lie exactly on the edges of this street are the **support vectors**, and they alone
determine the boundary: other points can be removed without effect, making SVM
memory-efficient and robust. Because real data isn't perfectly separable, SVM uses a
**soft margin** governed by **C**, which penalises margin violations: a large C enforces
few violations (narrow margin, tight fit, overfitting risk), while a small C tolerates
more violations (wider margin, simpler boundary, underfitting risk) — the bias–variance
dial, tuned by cross-validation. Maximising the margin generalises well because a
boundary placed as far as possible from both classes is the least sensitive to noise
and to new points: there is the largest possible "buffer" before a point would be
misclassified, which is a principled form of regularisation.

**Q2. Explain the kernel trick and its importance, comparing the common kernels and when
to use each.**

*Answer:* The **kernel trick** enables SVMs to learn **non-linear** decision boundaries
without explicitly transforming the data into a high-dimensional space. The insight is
that the SVM optimisation depends on the data only through **inner products** between
points; a **kernel function** computes the inner product *as if* the points had been
mapped into a higher-dimensional feature space, without ever constructing those
coordinates. So data that is not linearly separable in the original space (e.g. a circle
inside a circle) can become separable in the implicit higher-dimensional space, and the
flat boundary there appears curved when viewed in the original space — exactly why the
RBF kernel turned a hopeless circles problem from 0.41 to 1.0 accuracy. The common
kernels: the **linear** kernel draws straight boundaries, is fast, and excels on
high-dimensional data such as text where classes are often already linearly separable;
the **polynomial** kernel produces curved boundaries of a chosen degree, useful for some
structured non-linearities but prone to instability at high degrees; and the **RBF
(Gaussian)** kernel produces smooth, flexible, local boundaries and is the default for
general non-linear problems, with a **gamma** parameter controlling flexibility (high
gamma → very wiggly/overfit, low gamma → smooth). In practice, start with linear for
high-dimensional/text data and RBF for non-linear problems, always scale features, and
tune C (and gamma for RBF) jointly with cross-validation. The kernel trick's importance
is historical and practical: it gave SVMs state-of-the-art non-linear power with
mathematical elegance and efficiency, making them dominant before the deep-learning era
and still strong on small-to-medium high-dimensional data today.

## Exercises

1. In your own words, explain "margin" and "support vector".
2. Describe what happens to the boundary as C goes from very small to very large.
3. Why can an RBF SVM separate a circle-inside-a-circle but a linear SVM cannot?
4. List two situations where you would prefer the linear kernel.
5. Why must features be scaled for SVM?

## Mini-Project

**Project: Kernel and hyperparameter study.**

1. Take a non-linear 2-D dataset (`make_moons` or `make_circles`).
2. Train SVMs with linear, poly, and RBF kernels; plot each decision boundary
   (Chapter 14/16 style) and compare accuracy.
3. For the RBF kernel, grid-search C and gamma with `GridSearchCV` and report the best
   combination.
4. On a real high-dimensional dataset (e.g. breast cancer or a text dataset), compare
   linear vs RBF.
5. Write a short report on which kernel/settings worked where and why. Save in
   `my-ml-journey/`.

## Assignments

1. **Coding:** Reproduce the circles example and visualise the linear vs RBF decision
   boundaries to *see* why linear fails.
2. **Coding:** On a scaled dataset, `GridSearchCV` over `C ∈ {0.1, 1, 10}` and
   `gamma ∈ {0.01, 0.1, 1}` for an RBF SVM; report the best parameters and test accuracy.
3. **Conceptual:** Write one page explaining the kernel trick to a beginner, including
   why it's efficient (no explicit high-dimensional mapping).

::: tip
SVM finds one optimal boundary. Next we harness the power of *many* models together:
Chapter 23, **Random Forest**, averages hundreds of decision trees to build one of the
most reliable, accurate, and popular algorithms for tabular data.
:::
