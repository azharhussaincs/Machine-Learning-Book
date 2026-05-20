# Dimensionality Reduction (PCA, t-SNE, UMAP)

## Introduction

Real datasets can have **hundreds or thousands of features** — a photo has thousands of
pixels, a gene dataset thousands of genes. This causes the **curse of dimensionality**
(Chapter 13): models slow down, overfit, and "nearest" loses meaning. **Dimensionality
reduction** is the unsupervised art of **squeezing many features into a few** while
keeping the important information.

Think of it like a shadow: a 3-D object casts a 2-D shadow that still captures its
essential shape. Dimensionality reduction finds the most informative "shadow" of your
high-dimensional data.

::: keyidea
Dimensionality reduction serves three big goals: **(1) visualisation** — plot
high-dimensional data in 2-D/3-D; **(2) speed & storage** — fewer features train faster;
**(3) denoising & anti-overfitting** — drop noisy, redundant dimensions. The two
workhorses are **PCA** (fast, linear, for compression) and **t-SNE/UMAP** (for beautiful
visualisations).
:::

By the end of this chapter you will be able to:

- Understand **PCA** — principal components, explained variance, and when to use it.
- Use **t-SNE** and **UMAP** for visualisation.
- Choose the right method and avoid common pitfalls.

## Principal Component Analysis (PCA)

**PCA** is the most widely used dimensionality-reduction technique. It finds new axes
(called **principal components**) that capture the **maximum variance** in the data, then
keeps only the top few. The first component is the direction of greatest spread; the
second is the next-greatest direction perpendicular to it; and so on.

![PCA finds new axes (principal components) along the directions of greatest variance. Projecting the data onto the first component (PC1) keeps most of the spread while reducing two dimensions to one.](assets/images/ch28_pca.png)

PCA seeks the direction `w` (unit length) that maximises the variance of the projected
data:

<div class="equation"><img class="eq" src="assets/images/eq_ch28_variance.png" alt="PCA variance objective"></div>

Mathematically, the principal components are the **eigenvectors** of the data's
covariance matrix, ordered by their **eigenvalues** (the amount of variance each
captures). You don't compute this by hand — scikit-learn does — but the *idea* is:
**rotate the axes to point along the data's biggest spread, then drop the
least-informative axes.**

### Explained variance: how many components to keep

Each component captures some fraction of the total variance. The **explained variance
ratio** tells you how much information you keep. You choose enough components to retain,
say, 90–95% of the variance.

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X, y = load_digits(return_X_y=True)     # 8x8 handwritten digits = 64 features
print("original shape:", X.shape)

X_s = StandardScaler().fit_transform(X)
pca = PCA().fit(X_s)
cum = np.cumsum(pca.explained_variance_ratio_)

print("variance explained by first 2 PCs:", round(cum[1], 3))
print("components for 90% variance:", int(np.argmax(cum >= 0.90)) + 1, "/ 64")
print("components for 95% variance:", int(np.argmax(cum >= 0.95)) + 1, "/ 64")
```

**Output:**
```text
original shape: (1797, 64)
variance explained by first 2 PCs: 0.216
components for 90% variance: 31 / 64
components for 95% variance: 40 / 64
```

![The cumulative explained-variance curve. Choose the number of components where the curve reaches your target (e.g. 90–95%). Here ~31 of 64 components retain 90% of the information.](assets/images/ch28_explained_variance.png)

### Explanation

- The digits have **64 features** (8×8 pixels). The **first 2 components capture only
  21.6%** of the variance — enough for a rough 2-D *picture*, but not for full accuracy.
- To keep **90%** of the information we need **31 components**, and **95%** needs **40** —
  so we can roughly **halve** the dimensions with little information loss, speeding up any
  downstream model.

::: warning
**Scale before PCA.** PCA is variance-based, so a large-range feature would dominate the
components purely because of its scale. Standardise first (Chapter 11). Also, PCA is
**linear** — it can't capture curved structure (that's where t-SNE/UMAP come in), and its
components are combinations of original features, so they lose direct interpretability.
:::

## t-SNE: beautiful visualisations

**t-SNE** (t-distributed Stochastic Neighbor Embedding) is built for one job:
**visualising high-dimensional data in 2-D/3-D**. Unlike PCA, it's **non-linear** and
focuses on preserving **local structure** — keeping points that are neighbours in high
dimensions close together in the 2-D plot. The result is often stunning, well-separated
clusters.

![t-SNE projects the 64-dimensional digit images into 2-D, revealing well-separated clusters for each digit (0–9) — structure that PCA's linear projection blurs together. t-SNE is for *seeing*, not for feeding into models.](assets/images/ch28_tsne.png)

::: warning
**t-SNE is for visualisation only — not for preprocessing before a model.** It's slow,
non-deterministic (different runs differ), and the *distances between clusters* in the
plot are not meaningful (only the grouping is). Never feed t-SNE output into a classifier;
use PCA for that.
:::

## UMAP: faster, preserves global structure

**UMAP** (Uniform Manifold Approximation and Projection) is a newer technique that, like
t-SNE, produces excellent visualisations — but it's **much faster**, scales to larger
data, and better preserves the **global** structure (the relationships *between*
clusters, not just within them). It's increasingly the default for visualising big,
high-dimensional datasets. (It's a separate install: `pip install umap-learn`.)

## Choosing a method

| Method | Type | Best for | Note |
|---|---|---|---|
| **PCA** | Linear | Compression, speed, denoising, preprocessing | Fast, interpretable variance; feed into models |
| **t-SNE** | Non-linear | Visualisation (local structure) | Slow; viz only; distances not meaningful |
| **UMAP** | Non-linear | Visualisation (local + global), large data | Fast; viz; some preprocessing use |

::: tip
**Practical & debugging tips:** (1) **Scale features** before PCA. (2) Use the explained-
variance curve to choose the number of components (target 90–95%). (3) Use **PCA to
reduce to ~50 dimensions first, then t-SNE/UMAP** for visualisation — it speeds them up
and reduces noise. (4) Use t-SNE/UMAP **only for plots**, PCA for model preprocessing. (5)
PCA components aren't interpretable as original features — don't over-explain them. (6)
Fix `random_state` for reproducible t-SNE/UMAP layouts.
:::

## Advantages, disadvantages, and use cases

| Advantages | Disadvantages |
|---|---|
| Fights the curse of dimensionality | PCA loses interpretability of features |
| Speeds up training, saves memory | PCA is linear (misses curved structure) |
| Enables 2-D/3-D visualisation | t-SNE is slow & viz-only |
| Removes noise & redundancy | Some information is always lost |
| Can improve model performance | Choosing #components needs judgement |

**Use cases:** visualising high-dimensional data (digits, embeddings, gene data),
compressing images, speeding up models, denoising, and preprocessing before clustering or
classification (PCA).

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Forgetting to scale before PCA.** Unscaled features let large-range
variables dominate the components for the wrong reason.
:::

- **Mistake 2 — Feeding t-SNE/UMAP output into a model** — they're for visualisation, not
  features.
- **Mistake 3 — Over-interpreting t-SNE cluster distances** (only grouping is meaningful).
- **Mistake 4 — Reducing too aggressively** and losing important information (check
  explained variance).
- **Mistake 5 — Treating PCA components as original features** — they're linear
  combinations.
- **Mistake 6 — Applying PCA to the test set separately** — fit on train, transform both.

## Best practices

- **Scale features** before PCA.
- **Choose components by the explained-variance curve** (90–95% is common).
- **Use PCA for compression/preprocessing**, t-SNE/UMAP for visualisation.
- **Reduce with PCA first, then t-SNE/UMAP** for big data.
- **Fit on training data, transform test data** with the same fitted reducer.
- **Remember some information is always lost** — verify downstream performance.

## Chapter Summary

- **Dimensionality reduction** compresses many features into a few, fighting the **curse
  of dimensionality** and enabling visualisation, speed, and denoising.
- **PCA** finds **principal components** — the directions of maximum variance — and keeps
  the top few; use the **explained-variance ratio** to choose how many (digits: ~31 of 64
  components keep 90%). It's fast, linear, and good for preprocessing, but loses feature
  interpretability and can't capture curved structure. **Scale first.**
- **t-SNE** gives beautiful non-linear 2-D **visualisations** (local structure) but is
  slow and **viz-only**; **UMAP** is faster and preserves more global structure.
- Use **PCA for compression/models**, **t-SNE/UMAP for plots**; never feed t-SNE output
  into a classifier, and don't over-interpret its distances.

---

::: {.qband}
Practice Zone — Chapter 28
:::

## Multiple-Choice Questions (MCQs)

**Q1.** PCA finds new axes that maximise:
a) Accuracy  b) Variance  c) The number of features  d) Distance to the mean

**Q2.** The principal components are the ___ of the covariance matrix.
a) rows  b) eigenvectors  c) means  d) inverses

**Q3.** Before PCA you should:
a) Add labels  b) Scale the features  c) Train a model  d) Nothing

**Q4.** t-SNE is primarily used for:
a) Compression for models  b) Visualisation  c) Classification  d) Regression

**Q5.** Which preserves global structure better and is faster than t-SNE?
a) PCA  b) UMAP  c) K-Means  d) Lasso

**Q6.** The explained-variance ratio tells you:
a) The accuracy  b) How much information each component keeps  c) The number of clusters
d) The learning rate

**Q7.** A key limitation of PCA is that it is:
a) Too slow  b) Linear (misses curved structure)  c) Supervised  d) Only for images

**Q8.** You should NOT feed t-SNE output into a classifier because:
a) It's too fast  b) It's viz-only, non-deterministic, distances not meaningful  c) It
needs labels  d) It scales features

### MCQ Answers
**1:** b. **2:** b. **3:** b. **4:** b. **5:** b. **6:** b. **7:** b. **8:** b.

## Interview Questions (with answers)

**Q1. What is PCA and how does it work?**
*Answer:* Principal Component Analysis is a linear dimensionality-reduction method that
finds new orthogonal axes (principal components) along the directions of maximum variance
in the data — the eigenvectors of the covariance matrix ordered by eigenvalue. Projecting
onto the top few components reduces dimensions while retaining most of the variance
(information).

**Q2. How do you decide how many principal components to keep?**
*Answer:* Use the cumulative explained-variance ratio: plot it against the number of
components and keep enough to retain a target (commonly 90–95%) of the total variance,
balancing information retention against dimensionality.

**Q3. What is the difference between PCA and t-SNE?**
*Answer:* PCA is linear, fast, deterministic, preserves global variance, and is suitable
for compression and preprocessing for models. t-SNE is non-linear, slow, non-deterministic,
focuses on local neighbourhood structure, and is meant only for 2-D/3-D visualisation —
its inter-cluster distances aren't meaningful and its output shouldn't feed a model.

**Q4. Why must features be scaled before PCA?**
*Answer:* PCA maximises variance, so a feature with a large numeric range would dominate
the components simply because of its scale, not its importance. Standardising features
ensures each contributes fairly to the variance computation.

**Q5. Why is dimensionality reduction useful?**
*Answer:* It combats the curse of dimensionality by reducing features, which speeds up
training, lowers memory use, removes noise/redundancy (reducing overfitting), and enables
visualisation of high-dimensional data in 2-D/3-D.

## Scenario-Based Questions (with answers)

**Q1.** *You have 5,000 features and a model that's slow and overfits. How can
dimensionality reduction help, and which method?*
*Answer:* Apply PCA (after scaling) to compress the 5,000 features into a few dozen
components retaining ~95% variance. This speeds training, reduces memory, and removes
noisy/redundant dimensions, often reducing overfitting — then feed the components into the
model.

**Q2.** *You want to show a stakeholder that your 64-dimensional digit data forms clear
groups. Which technique and why?*
*Answer:* t-SNE (or UMAP) for visualisation, optionally after a PCA pre-reduction to ~50
dims for speed. It projects to 2-D and reveals well-separated clusters per digit — far
clearer than PCA's linear projection — though for *modelling* you'd use PCA, not the t-SNE
output.

**Q3.** *Your PCA results look strange — one feature seems to dominate everything. What did
you likely forget?*
*Answer:* To scale the features. PCA is variance-based, so an unscaled large-range feature
dominates the principal components. Standardise the data and recompute.

## Logic-Based Questions (with answers)

**Q1.** Why is the first principal component the direction of greatest variance?
*Answer:* By construction, PCA chooses axes to maximise the variance of the projected
data; the first component is defined as the single direction along which the data is most
spread out, capturing the most information in one dimension.

**Q2.** If the first 2 of 64 components capture only 21.6% of variance, what does that
imply about visualising the data in 2-D?
*Answer:* A 2-D PCA plot shows only ~22% of the information, so it gives a rough picture
that may blur groups; non-linear methods (t-SNE/UMAP) or more components are needed to
reveal the full structure.

**Q3.** Why is some information always lost in dimensionality reduction?
*Answer:* Reducing dimensions discards directions of variance (or distorts structure), so
unless the data truly lies in a lower-dimensional subspace, the compressed representation
cannot perfectly reconstruct the original — a deliberate trade-off for simplicity and
speed.

## Practical Questions (with answers)

**Q1.** Write code to reduce data to 2 principal components.
*Answer:* `PCA(n_components=2).fit_transform(X_scaled)`.

**Q2.** How do you find how much variance each PCA component explains?
*Answer:* `pca.explained_variance_ratio_` (and `np.cumsum(...)` for the cumulative total).

**Q3.** Should you fit PCA on the whole dataset or just training data? Why?
*Answer:* Fit on the **training** data only, then transform both train and test with that
fitted PCA — fitting on all data leaks test information (Chapter 11/25).

## Long Questions (with answers)

**Q1. Explain PCA in detail: the intuition, what principal components are, how to choose
the number to keep, and PCA's strengths and limitations.**

*Answer:* **PCA** reduces dimensionality by finding a new set of axes, the **principal
components**, that capture the most variance in the data. Intuitively, it rotates the
coordinate system so the first axis points along the direction of greatest spread, the
second along the next-greatest direction orthogonal to the first, and so on; projecting the
data onto the top few axes keeps most of the variation while discarding the least-
informative directions. Mathematically the components are the **eigenvectors of the
covariance matrix**, ordered by their **eigenvalues**, which quantify how much variance
each captures. To **choose how many to keep**, compute the cumulative **explained-variance
ratio** and retain enough components to reach a target such as 90–95% (for the 64-feature
digits, ~31 components retain 90%). **Strengths:** PCA is fast, deterministic, reduces
noise and redundancy, speeds up and can improve models, and enables visualisation; the
explained-variance ratio makes information loss measurable. **Limitations:** it is
**linear**, so it cannot capture curved/non-linear structure (t-SNE/UMAP do that better
for visualisation); its components are linear combinations of original features and so
**lose interpretability**; it is sensitive to feature scale (so you must standardise
first); and reducing too aggressively discards useful information. Used well — scale, fit
on training data, keep enough variance — PCA is a powerful, general-purpose preprocessing
and compression tool.

**Q2. Compare PCA, t-SNE, and UMAP, explaining when to use each and the dangers of misusing
visualisation methods.**

*Answer:* **PCA** is a **linear** method that preserves global variance, is fast and
deterministic, and produces components usable both for **visualisation** and, importantly,
as **compressed features for models** and preprocessing; its limitation is that linear
projections can blur non-linear structure. **t-SNE** is a **non-linear** method designed
purely for **visualisation**: it preserves **local** neighbourhood structure, often
revealing crisp, well-separated clusters (e.g. the ten digit groups) that PCA blurs — but
it is **slow**, **non-deterministic** (different runs and `random_state`s give different
layouts), and the **distances between clusters in its plots are not meaningful**, only the
groupings. **UMAP** is a newer non-linear technique that, like t-SNE, gives excellent
visualisations but is **much faster**, scales to larger datasets, and better preserves
**global** structure (relationships between clusters). The key **danger** is misusing the
visualisation methods: t-SNE and UMAP outputs should **not** be fed into classifiers as
features, because they are optimised for display rather than faithful geometry, are
non-deterministic, and distort distances; for modelling and compression, use **PCA**. A
common best practice is to **pre-reduce with PCA to ~50 dimensions and then apply t-SNE/
UMAP** for plotting, combining PCA's speed and denoising with the non-linear methods'
visual clarity.

## Exercises

1. In your own words, explain what a "principal component" is.
2. Why must you scale features before PCA?
3. Given an explained-variance curve, describe how you'd choose the number of components.
4. State two differences between PCA and t-SNE.
5. Why should t-SNE output not be used as model features?

## Mini-Project

**Project: Compress and visualise the digits.**

1. Load the digits dataset (64 features). Scale it.
2. Fit PCA; plot the cumulative explained-variance curve and find the components needed for
   90% and 95%.
3. Reduce to 2-D with PCA and with t-SNE; scatter-plot both, coloured by digit, and compare
   how well the digits separate.
4. Train a classifier on (a) all 64 features and (b) the PCA-reduced features; compare
   accuracy and training time.
5. Write a short report on the trade-offs. Save in `my-ml-journey/`.

## Assignments

1. **Coding:** Apply PCA to a high-dimensional dataset, reduce to enough components for
   95% variance, train a model on the reduced data, and compare accuracy/speed to the full
   data.
2. **Coding:** (Optional `pip install umap-learn`) Compare t-SNE and UMAP visualisations of
   the same dataset; note speed and cluster separation.
3. **Conceptual:** Write one page on the curse of dimensionality and how dimensionality
   reduction addresses it.

::: tip
PCA and clustering handle the two big unsupervised tasks. Chapter 29, **Association Rule
Learning**, covers a third — discovering "items that go together" (market-basket analysis)
— before we move to the special paradigms of semi-supervised and reinforcement learning.
:::
