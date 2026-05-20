# Unsupervised Learning & Clustering

## Introduction

Welcome to **Part V**, where we leave the world of labelled data behind. In supervised
learning (Part IV) we always had the "answers" (`y`). Now we enter **unsupervised
learning**: the data has **no labels**, and the machine must discover hidden structure
*on its own*.

The most important unsupervised task is **clustering** — automatically grouping similar
items together. Imagine handing a child a box of mixed toys; without being told the
categories, they'd naturally group the cars, the dolls, and the blocks. Clustering does
the same for data.

::: keyidea
Clustering finds **natural groups** in unlabelled data based on similarity. *You* don't
tell it the groups — it discovers them. The catch: there's no "correct answer" to check
against, so evaluating and interpreting clusters takes judgement.
:::

By the end of this chapter you will be able to:

- Understand **K-Means** and how to choose the number of clusters (**elbow** &
  **silhouette**).
- Understand **Hierarchical** clustering and read a **dendrogram**.
- Understand **DBSCAN** and when it beats K-Means.
- Evaluate clusters and know each method's pros, cons, and use cases.

## K-Means clustering

**K-Means** is the most popular clustering algorithm. You tell it how many clusters
(`k`) you want, and it finds `k` cluster **centres (centroids)** and assigns each point
to its nearest centre.

![K-Means in action: (1) place k random centroids, (2) assign each point to its nearest centroid, (3) move each centroid to the mean of its points, (4) repeat until centroids stop moving.](assets/images/ch27_kmeans.png)

**The algorithm:**

1. **Initialise** `k` centroids randomly.
2. **Assign** each point to the nearest centroid (forming clusters).
3. **Update** each centroid to the *mean* of the points assigned to it.
4. **Repeat** steps 2–3 until the centroids stop moving (convergence).

K-Means minimises the total within-cluster squared distance, called **inertia**:

<div class="equation"><img class="eq" src="assets/images/eq_ch27_inertia.png" alt="K-Means inertia"></div>

(Each point's squared distance to its assigned centroid, summed up.) Lower inertia means
tighter clusters.

### Choosing k: the elbow method

K-Means needs you to pick `k` in advance. The **elbow method** plots inertia against `k`:
inertia always drops as `k` rises, but at the "right" `k` the improvement sharply slows,
forming an **elbow**.

```python
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

X, _ = load_iris(return_X_y=True)
for k in [1, 2, 3, 4, 5]:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    print(f"k={k}: inertia={km.inertia_:.1f}")
```

**Output:**
```text
k=1: inertia=681.4
k=2: inertia=152.3
k=3: inertia=78.9
k=4: inertia=57.2
k=5: inertia=46.5
```

![The elbow method: inertia falls sharply then levels off. The "elbow" (here around k=3) suggests a good number of clusters — adding more barely helps.](assets/images/ch27_elbow.png)

The big drops are from k=1→2→3, then it flattens — the **elbow is around k=3**, which
matches iris's three real species (a satisfying check, though clustering didn't know the
species).

### Choosing k: the silhouette score

The **silhouette score** (−1 to +1) measures how well-separated the clusters are
(higher = better). It's a more objective check than the elbow.

```python
from sklearn.metrics import silhouette_score
for k in [2, 3, 4]:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    print(f"k={k}: silhouette={silhouette_score(X, km.labels_):.3f}")
```

**Output:**
```text
k=2: silhouette=0.681
k=3: silhouette=0.553
k=4: silhouette=0.498
```

Here k=2 scores highest (0.681) because two of the iris species overlap and merge
naturally. This shows clustering is *interpretive* — the elbow suggested 3, the
silhouette favours 2, and both are defensible. **You** decide based on domain knowledge.

::: warning
**K-Means assumes round, similar-sized clusters** and needs `k` chosen in advance. It's
sensitive to the initial centroid placement (use `n_init` to try several) and to feature
scale (**scale your features first**, Chapter 11). It struggles with elongated or oddly
shaped clusters — which is where DBSCAN shines.
:::

## Hierarchical clustering

**Hierarchical (agglomerative) clustering** builds a tree of clusters without needing `k`
upfront:

1. Start with every point as its own cluster.
2. Repeatedly **merge** the two closest clusters.
3. Continue until everything is one cluster.

The result is a **dendrogram** — a tree showing how clusters merge. You "cut" it at a
chosen height to get any number of clusters.

![A dendrogram from hierarchical clustering. Points merge into clusters bottom-up; cutting the tree at a chosen height (dashed line) yields that many clusters. The height of a merge shows how dissimilar the merged groups are.](assets/images/ch27_dendrogram.png)

Hierarchical clustering is great for **understanding structure at multiple levels** and
doesn't require choosing `k` first, but it's **slow on large datasets** (O(n²) or worse).

## DBSCAN: density-based clustering

**DBSCAN** (Density-Based Spatial Clustering) groups points that are **densely packed
together**, marking lonely points as **noise/outliers**. Its superpowers:

- **Finds clusters of any shape** (not just round blobs).
- **Doesn't need `k`** — it discovers the number of clusters itself.
- **Automatically detects outliers** (great for anomaly detection).

It has two parameters: `eps` (neighbourhood radius) and `min_samples` (minimum points to
form a dense region).

### K-Means vs DBSCAN on tricky shapes

```python
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

X, _ = make_moons(n_samples=300, noise=0.06, random_state=0)  # two interleaving crescents
km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
db = DBSCAN(eps=0.2, min_samples=5).fit(X)

print("KMeans silhouette:", round(silhouette_score(X, km.labels_), 3))
n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
print("DBSCAN clusters found:", n_clusters, "| noise points:", list(db.labels_).count(-1))
```

**Output:**
```text
KMeans silhouette: 0.488
DBSCAN clusters found: 2 | noise points: 0
```

![K-Means vs DBSCAN on two crescent-shaped clusters. K-Means (which assumes round clusters) splits them wrongly down the middle; DBSCAN correctly follows the density to separate the two crescents.](assets/images/ch27_dbscan.png)

### Explanation

The "two moons" form crescent shapes that **interleave**. **K-Means fails** — it assumes
round clusters, so it slices straight down the middle (low silhouette 0.488). **DBSCAN
succeeds** — it follows the *density* of each crescent and correctly finds the **2
clusters** with no points wrongly grouped. This is the classic demonstration of why no
single clustering algorithm is best (No Free Lunch again).

::: keyidea
**Match the algorithm to the cluster shape.** K-Means: fast, simple, round/similar-sized
clusters, known k. Hierarchical: multi-level structure, no k needed, small data. DBSCAN:
arbitrary shapes, automatic k, outlier detection, but sensitive to its `eps`/`min_samples`
settings. Always *visualise* your clusters to judge them.
:::

::: tip
**Practical & debugging tips:** (1) **Scale features** before K-Means/DBSCAN (distance-
based). (2) For K-Means, set `n_init=10`+ to avoid bad random starts. (3) Use elbow *and*
silhouette to choose k, plus domain knowledge. (4) DBSCAN's `eps` is finicky — use a
k-distance plot to choose it. (5) Cluster labels are arbitrary IDs (Chapter 4) — judge by
*which points group together*, not the numbers. (6) Always plot clusters (reduce to 2-D
with PCA, Chapter 28, if needed).
:::

## Evaluating clusters

Without labels, evaluation is harder. Common approaches:

- **Internal metrics:** silhouette score (separation), inertia (compactness), Davies-
  Bouldin index.
- **Visual inspection:** plot the clusters (use PCA/t-SNE for high dimensions, Chapter 28).
- **External validation:** if you *do* have some labels, compare with the Adjusted Rand
  Index.
- **Domain judgement:** do the clusters make business sense?

## Advantages, disadvantages, and use cases

| Method | Best for | Watch out for |
|---|---|---|
| **K-Means** | Fast, round, similar-sized clusters | Needs k; assumes round; scale-sensitive |
| **Hierarchical** | Multi-level structure; no k needed | Slow on big data (O(n²)) |
| **DBSCAN** | Arbitrary shapes; outliers; auto-k | Sensitive to eps/min_samples; varying density |

**Use cases:** customer segmentation, market research, document/topic grouping, image
compression (K-Means), anomaly/fraud detection (DBSCAN), gene-expression analysis, and
exploratory data analysis to *discover* groups you didn't know existed.

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Using K-Means on non-round clusters** (like the moons). Use DBSCAN or
spectral clustering for arbitrary shapes.
:::

- **Mistake 2 — Forgetting to scale** features before distance-based clustering.
- **Mistake 3 — Picking k from the elbow alone** — combine with silhouette and judgement.
- **Mistake 4 — Treating cluster IDs as meaningful labels** — they're arbitrary; you must
  interpret each cluster.
- **Mistake 5 — Not visualising** the clusters before trusting them.
- **Mistake 6 — Expecting a single "correct" clustering** — different methods/parameters
  give different valid groupings.

## Best practices

- **Scale features** before clustering.
- **Choose k with elbow + silhouette + domain knowledge.**
- **Match the method to the expected cluster shape.**
- **Visualise** clusters (via PCA/t-SNE for high dimensions).
- **Interpret** each cluster — what does it represent?
- **Try multiple algorithms/parameters** and compare.

## Chapter Summary

- **Unsupervised learning** finds structure in **unlabelled** data; the key task is
  **clustering** (grouping similar items) — with no ground-truth answer to check against.
- **K-Means** assigns points to `k` nearest centroids and iterates; choose `k` with the
  **elbow** (inertia) and **silhouette** methods. It's fast but assumes round, similar-
  sized clusters and needs scaling.
- **Hierarchical** clustering builds a **dendrogram** (no `k` needed, multi-level) but is
  slow on big data.
- **DBSCAN** finds **arbitrary-shaped** clusters by density, auto-detects the number of
  clusters and **outliers** — it solved the "two moons" that K-Means couldn't.
- Evaluate with **silhouette/inertia**, **visualisation**, and **domain judgement**; match
  the algorithm to the data, and always interpret the clusters.

---

::: {.qband}
Practice Zone — Chapter 27
:::

## Multiple-Choice Questions (MCQs)

**Q1.** Clustering is a type of:
a) Supervised learning  b) Unsupervised learning  c) Reinforcement learning  d) Regression

**Q2.** K-Means requires you to specify:
a) The labels  b) The number of clusters k  c) The test set  d) The learning rate

**Q3.** The elbow method plots inertia against:
a) Accuracy  b) k (number of clusters)  c) Epochs  d) Features

**Q4.** Which algorithm can find arbitrarily shaped clusters and detect outliers?
a) K-Means  b) Linear regression  c) DBSCAN  d) Naive Bayes

**Q5.** A dendrogram is produced by:
a) K-Means  b) Hierarchical clustering  c) DBSCAN  d) PCA

**Q6.** K-Means struggles with the "two moons" data because it assumes:
a) Labels exist  b) Round, similar-sized clusters  c) Scaled data  d) Few features

**Q7.** The silhouette score ranges from:
a) 0 to 1  b) −1 to +1  c) 0 to 100  d) −∞ to ∞

**Q8.** Before distance-based clustering you should:
a) Add labels  b) Scale the features  c) Train a classifier  d) Nothing

### MCQ Answers
**1:** b. **2:** b. **3:** b. **4:** c. **5:** b. **6:** b. **7:** b. **8:** b.

## Interview Questions (with answers)

**Q1. How does the K-Means algorithm work?**
*Answer:* Choose k; initialise k centroids; assign each point to its nearest centroid;
move each centroid to the mean of its assigned points; repeat assign/update until
centroids stop moving. It minimises within-cluster squared distance (inertia).

**Q2. How do you choose the number of clusters k?**
*Answer:* Use the elbow method (plot inertia vs k and look for where the decrease sharply
slows), the silhouette score (pick k with the highest separation), and domain knowledge.
These can disagree, so judgement is required.

**Q3. When would you use DBSCAN over K-Means?**
*Answer:* When clusters have arbitrary (non-round) shapes, when you don't know k in
advance, or when you need to detect outliers/noise. DBSCAN groups by density and handles
the "two moons" case that K-Means gets wrong.

**Q4. What are the limitations of K-Means?**
*Answer:* It needs k specified upfront, assumes round and similar-sized clusters, is
sensitive to initialisation (mitigated by n_init) and to feature scale, and can converge
to poor local optima. It also struggles with clusters of varying density or non-convex
shape.

**Q5. How do you evaluate clustering without labels?**
*Answer:* Use internal metrics (silhouette, inertia, Davies-Bouldin), visualise the
clusters (reducing dimensions with PCA/t-SNE if needed), apply external metrics (e.g.
Adjusted Rand Index) if some labels exist, and judge whether the clusters make domain
sense.

## Scenario-Based Questions (with answers)

**Q1.** *A retailer wants to segment customers into groups for marketing but has no
predefined segments. Which technique and how do you choose the number of groups?*
*Answer:* Clustering (e.g. K-Means on scaled behavioural features). Choose the number of
segments using the elbow method and silhouette score, then validate that the segments are
meaningful and actionable for marketing (domain judgement), adjusting k if needed.

**Q2.** *Your K-Means clusters look wrong: it splits two clearly crescent-shaped groups
down the middle. What's the fix?*
*Answer:* K-Means assumes round clusters and can't follow crescent shapes. Switch to
DBSCAN (density-based) or spectral clustering, which can capture arbitrary shapes — DBSCAN
correctly separates such "moons".

**Q3.** *You need to detect unusual transactions (outliers) without labels. Which
clustering approach helps and why?*
*Answer:* DBSCAN, because it labels points in low-density regions as noise/outliers
automatically. Those flagged points are candidate anomalies — useful for fraud or
fault detection.

## Logic-Based Questions (with answers)

**Q1.** Why does inertia always decrease as k increases, and why doesn't that mean "more
clusters is always better"?
*Answer:* More centroids means points are closer to some centroid, so within-cluster
distance (inertia) always drops — at k = n, inertia is zero. But that just memorises the
data; the goal is meaningful structure, so we look for the elbow where extra clusters stop
adding real value.

**Q2.** Cluster labels come out as [1,1,0,0] one run and [0,0,1,1] another. Did the
clustering change?
*Answer:* No — only the arbitrary cluster IDs swapped; the same points are grouped
together. Cluster numbers carry no inherent meaning.

**Q3.** Why must features be scaled before K-Means?
*Answer:* K-Means uses distances; a large-range feature would dominate the distance and
the cluster assignments, just as in KNN. Scaling lets all features contribute fairly.

## Practical Questions (with answers)

**Q1.** Write code to fit K-Means with 3 clusters.
*Answer:* `KMeans(n_clusters=3, n_init=10, random_state=42).fit(X)`.

**Q2.** How do you get the silhouette score of a clustering?
*Answer:* `silhouette_score(X, model.labels_)` from `sklearn.metrics`.

**Q3.** In DBSCAN's output, what does a label of −1 mean?
*Answer:* That the point is classified as **noise** (an outlier) — not assigned to any
dense cluster.

## Long Questions (with answers)

**Q1. Explain K-Means in full: the algorithm, how to choose k, and its strengths and
limitations.**

*Answer:* **K-Means** partitions data into k clusters by alternating two steps until
convergence. After choosing k and initialising k centroids (often randomly, or smartly via
k-means++), it (1) **assigns** each point to its nearest centroid by distance, then (2)
**updates** each centroid to the mean of the points assigned to it; these steps repeat
until centroids stop moving. This minimises **inertia**, the total within-cluster squared
distance. To **choose k**, use the **elbow method** — plot inertia against k and pick the
point where the decrease sharply flattens (the "elbow") — together with the **silhouette
score**, which measures how well-separated clusters are (higher is better), and **domain
knowledge**, since these can disagree (on iris the elbow suggested 3, the silhouette
favoured 2). **Strengths:** it's simple, fast, scalable, and works well when clusters are
roughly round and similar-sized. **Limitations:** it requires k in advance; assumes
convex, similar-sized clusters and so fails on elongated or non-convex shapes (e.g. the
two moons); is sensitive to centroid initialisation (mitigated by running several inits)
and to feature scale (features must be scaled); and can settle in poor local optima.
These limits motivate hierarchical clustering (no k, multi-level) and DBSCAN (arbitrary
shapes, outliers, automatic cluster count).

**Q2. Compare K-Means, hierarchical clustering, and DBSCAN across how they work, their key
parameters, strengths, and ideal use cases.**

*Answer:* **K-Means** assigns points to the nearest of k centroids and iteratively updates
centroids to minimise within-cluster distance; its key parameter is **k** (plus
initialisation settings). It is fast and scalable and best for large datasets with
roughly round, similar-sized clusters — e.g. customer segmentation and image compression —
but needs k specified, assumes convex clusters, and is scale-sensitive. **Hierarchical
(agglomerative)** clustering starts with each point as its own cluster and repeatedly
merges the two closest clusters, producing a **dendrogram** that can be cut at any height;
its key choices are the **linkage** and **distance** metrics. It needs no k in advance,
reveals structure at multiple levels, and suits smaller datasets and exploratory analysis,
but is computationally expensive (≈O(n²)) and so impractical for very large data.
**DBSCAN** groups points in dense regions and labels sparse points as noise; its
parameters are **eps** (neighbourhood radius) and **min_samples**. It finds **arbitrary-
shaped** clusters, determines the **number of clusters automatically**, and **detects
outliers**, making it ideal for non-convex clusters and anomaly detection (it solved the
two-moons case K-Means failed), but it is sensitive to its parameters and struggles when
clusters have very different densities. In practice, choose K-Means for speed and round
clusters, hierarchical for multi-level insight on smaller data, and DBSCAN for irregular
shapes and outlier-aware clustering — and, since clustering has no single correct answer,
try several, visualise the results, and apply domain judgement.

## Exercises

1. List the four steps of the K-Means algorithm.
2. Explain the elbow method and what an "elbow" indicates.
3. Give one situation each where you'd choose K-Means, hierarchical, and DBSCAN.
4. Why are cluster labels (0, 1, 2…) arbitrary?
5. Explain why scaling matters for clustering.

## Mini-Project

**Project: Customer (or iris) segmentation.**

1. Take a dataset (iris, or a customer dataset). Scale the features.
2. Run K-Means for k = 1…8, plot the elbow curve and silhouette scores, and choose k.
3. Visualise the clusters in 2-D (use PCA, Chapter 28, if more than 2 features).
4. Run DBSCAN and hierarchical clustering on the same data and compare the groupings.
5. Interpret each cluster in plain language and write a short report. Save in
   `my-ml-journey/`.

## Assignments

1. **Coding:** Implement one iteration of K-Means by hand (assign points to nearest of 2
   given centroids, then recompute the centroids) and verify against scikit-learn.
2. **Coding:** On `make_moons`, show K-Means failing and DBSCAN succeeding by plotting both
   clusterings.
3. **Conceptual:** Write one page on how you would evaluate and validate clusters when no
   labels exist.

::: tip
Clustering groups points. Chapter 28, **Dimensionality Reduction**, tackles the other big
unsupervised task: compressing many features into a few — to visualise data, speed up
models, and fight the curse of dimensionality — using PCA, t-SNE, and UMAP.
:::
