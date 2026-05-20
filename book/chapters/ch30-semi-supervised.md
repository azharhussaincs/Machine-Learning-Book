# Semi-Supervised Learning

## Introduction

In Chapter 4 you met a frustrating reality: **supervised learning needs labelled data,
but labelling is expensive.** Imagine paying radiologists to label 100,000 X-rays, or
manually tagging a million product photos. Meanwhile, *unlabelled* data is everywhere and
nearly free. **Semi-supervised learning** is the clever middle ground: learn from a
**small amount of labelled data plus a large amount of unlabelled data** — getting much
of the benefit of supervision without the full labelling cost.

::: keyidea
**Semi-supervised learning = a few labels + lots of unlabelled data.** The unlabelled
data reveals the *shape* and *structure* of the feature space (like clustering), and the
few labels tell the model what those structures *mean*. Combined, they beat using the few
labels alone.
:::

By the end of this chapter you will be able to:

- Explain *why* and *when* semi-supervised learning is used.
- Understand the key techniques: **self-training (pseudo-labelling)** and **label
  propagation**.
- Know the **assumptions** that make it work.
- Apply self-training and see it beat a supervised-only baseline.

## Why semi-supervised learning?

The motivation is the **label-cost problem**:

- **Labelled data is scarce and expensive** — it needs human experts, time, and money.
- **Unlabelled data is abundant and cheap** — every photo, document, and sensor reading.

Semi-supervised learning extracts value from the cheap, plentiful unlabelled data to
improve a model trained on the scarce labelled data. Real examples: Google Photos
recognising a face you tagged just once, speech recognisers trained on vast untranscribed
audio, and web-page classification.

![Semi-supervised learning uses a few labelled points (coloured) to anchor the meaning of the many unlabelled points (grey). The unlabelled data reveals the cluster structure; the labels assign meaning to each cluster.](assets/images/ch30_semi_supervised.png)

## The key assumptions

Semi-supervised learning only helps if the unlabelled data is informative. It relies on
one or more assumptions:

- **Smoothness:** points close together likely share a label.
- **Cluster assumption:** points in the same cluster likely share a label (so decision
  boundaries should lie in *low-density* regions, between clusters).
- **Manifold assumption:** high-dimensional data lies on a lower-dimensional surface;
  nearby points on that surface share labels.

If these hold, the structure revealed by unlabelled data genuinely guides the few labels.
If they don't, unlabelled data may not help (and can even hurt).

## Technique 1 — Self-training (pseudo-labelling)

The simplest, most popular approach. The model **teaches itself** using its own confident
predictions:

![Self-training loop: train on the labelled data, predict the unlabelled data, add the most-confident predictions as new "pseudo-labels", and retrain — repeating until done.](assets/images/ch30_self_training.png)

1. Train a model on the **labelled** data.
2. Use it to **predict** the unlabelled data.
3. Add the **most confident** predictions to the training set as **pseudo-labels**.
4. **Retrain** on the enlarged set. Repeat until no more confident predictions remain.

The risk: if the model confidently mislabels something, that error gets baked in. So
self-training uses a **confidence threshold** and works best when the initial model is
reasonably good.

## Technique 2 — Label propagation / spreading

These **graph-based** methods connect all points (labelled and unlabelled) into a graph
where edges link similar points. Labels then **"flow"** from labelled points to nearby
unlabelled ones along the edges, until every point has a (soft) label. This directly uses
the smoothness and cluster assumptions. scikit-learn provides `LabelPropagation` and
`LabelSpreading`.

## Practical: self-training beats supervised-only

We'll hide **90%** of the labels on the digits dataset and show that using the unlabelled
data (self-training) beats using only the few remaining labels.

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import accuracy_score

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# Hide 90% of training labels (mark them -1 = "unlabelled")
rng = np.random.RandomState(42)
mask = rng.rand(len(y_tr)) < 0.90
y_semi = y_tr.copy(); y_semi[mask] = -1
print(f"labelled training points: {(y_semi != -1).sum()} of {len(y_tr)}")

# Baseline: supervised model using ONLY the few labelled points
base = SVC(probability=True, gamma=0.001).fit(X_tr[~mask], y_tr[~mask])
print("supervised (few labels only):", round(accuracy_score(y_te, base.predict(X_te)), 3))

# Self-training: uses the unlabelled points too
st = SelfTrainingClassifier(SVC(probability=True, gamma=0.001)).fit(X_tr, y_semi)
print("self-training (uses unlabelled):", round(accuracy_score(y_te, st.predict(X_te)), 3))
```

**Output:**
```text
labelled training points: 131 of 1257
supervised (few labels only): 0.917
self-training (uses unlabelled): 0.95
```

### Explanation

- With only **131 labelled** points (10%), the supervised baseline reached **0.917**.
- **Self-training**, by leveraging the **1,126 unlabelled** points via confident
  pseudo-labels, improved to **0.95** — a meaningful gain *for free*, using data we
  already had but hadn't labelled.

::: keyidea
That jump from 0.917 to 0.95 came *purely* from using unlabelled data we'd otherwise
ignore. When labels are scarce and expensive but raw data is plentiful — the common real-
world situation — semi-supervised learning turns "wasted" unlabelled data into real
performance. This same insight, taken to the extreme, powers the self-supervised training
of Large Language Models (Chapter 39).
:::

::: tip
**Practical & debugging tips:** (1) Mark unlabelled points as **-1** for scikit-learn's
semi-supervised classifiers. (2) Self-training needs a **decent base model** — if it's
poor, it will propagate its own errors. (3) Use a **confidence threshold** so only
trustworthy pseudo-labels are added. (4) Check that semi-supervised actually *helps* on a
validation set — if the assumptions don't hold, it may not. (5) `LabelSpreading` is more
robust to noise than `LabelPropagation`. (6) Scale features for distance/graph-based
methods.
:::

## Self-supervised learning (a glimpse)

Closely related is **self-supervised learning**, where the data **creates its own labels**
automatically (no human at all) — e.g. hide a word in a sentence and predict it, or hide
part of an image and reconstruct it. This removes the label bottleneck entirely and is the
engine behind modern LLMs and foundation models. We cover it properly in Chapter 39.

## Advantages, disadvantages, and use cases

| Advantages | Disadvantages |
|---|---|
| Uses cheap, abundant unlabelled data | Can hurt if assumptions don't hold |
| Big savings on labelling cost | Risk of propagating confident errors |
| Often beats supervised-only with few labels | Needs a reasonable base model |
| Bridges supervised & unsupervised | More complex to set up and validate |

**Use cases:** medical imaging (few expert labels), speech recognition, text/document
classification, fraud detection (few confirmed cases), and any domain where unlabelled
data is plentiful but labelling is costly.

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Assuming unlabelled data always helps.** It helps only when the smoothness/
cluster/manifold assumptions hold. Validate that it actually improves performance.
:::

- **Mistake 2 — Self-training from a weak base model**, which then propagates its errors.
- **Mistake 3 — Adding low-confidence pseudo-labels** (use a threshold).
- **Mistake 4 — Forgetting to mark unlabelled data correctly** (e.g. -1 in scikit-learn).
- **Mistake 5 — Confusing semi-supervised with self-supervised** (the latter auto-generates
  labels from the data itself).

## Best practices

- **Use semi-supervised when labels are scarce but unlabelled data is plentiful.**
- **Start from a reasonable base model** and use a confidence threshold for pseudo-labels.
- **Validate** that unlabelled data genuinely improves results.
- **Prefer `LabelSpreading`** for noisy data; scale features for graph methods.
- **Consider self-supervised pre-training** (Part VI/VII) for images and text.

## Chapter Summary

- **Semi-supervised learning** combines a **small labelled set** with a **large unlabelled
  set**, addressing the high cost of labelling while exploiting cheap, abundant unlabelled
  data.
- It works when the **smoothness, cluster, or manifold** assumptions hold — unlabelled data
  reveals structure that the few labels give meaning to.
- Key techniques: **self-training (pseudo-labelling)** — the model adds its confident
  predictions to the training set and retrains — and **label propagation/spreading** —
  labels flow through a similarity graph.
- On digits with only **131 labels**, self-training improved accuracy from **0.917 to
  0.95** by using unlabelled data — a free gain.
- It can backfire if assumptions fail or the base model propagates errors; validate that it
  helps. **Self-supervised** learning (auto-generated labels) extends the idea and powers
  modern LLMs.

---

::: {.qband}
Practice Zone — Chapter 30
:::

## Multiple-Choice Questions (MCQs)

**Q1.** Semi-supervised learning uses:
a) Only labelled data  b) A few labels + lots of unlabelled data  c) Only unlabelled data
d) Rewards

**Q2.** The main motivation for semi-supervised learning is:
a) Faster GPUs  b) The high cost of labelling data  c) Smaller models  d) More features

**Q3.** Self-training adds which predictions to the training set?
a) Random ones  b) The most confident ones (pseudo-labels)  c) The least confident ones
d) None

**Q4.** In scikit-learn semi-supervised classifiers, unlabelled points are marked as:
a) 0  b) -1  c) NaN  d) "unknown"

**Q5.** The cluster assumption says points in the same cluster:
a) Have different labels  b) Likely share a label  c) Are outliers  d) Are noise

**Q6.** Which method propagates labels through a similarity graph?
a) Self-training  b) Label propagation  c) PCA  d) Apriori

**Q7.** A risk of self-training is:
a) Too few features  b) Propagating confident errors  c) Needing labels at test time
d) Overfitting the test set

**Q8.** Semi-supervised learning may NOT help when:
a) Assumptions hold  b) The cluster/smoothness assumptions fail  c) Data is abundant
d) The base model is good

### MCQ Answers
**1:** b. **2:** b. **3:** b. **4:** b. **5:** b. **6:** b. **7:** b. **8:** b.

## Interview Questions (with answers)

**Q1. What is semi-supervised learning and when is it useful?**
*Answer:* It trains on a small labelled set together with a large unlabelled set. It's
useful when labelling is expensive or slow but unlabelled data is abundant — common in
medical imaging, speech, and text — because the unlabelled data reveals structure that
improves the model beyond using the few labels alone.

**Q2. How does self-training work?**
*Answer:* Train a model on the labelled data; predict the unlabelled data; add the most
confident predictions as pseudo-labels to the training set; retrain; repeat. A confidence
threshold limits error propagation. It effectively lets the model teach itself using
unlabelled data.

**Q3. What assumptions does semi-supervised learning rely on?**
*Answer:* Smoothness (nearby points share labels), the cluster assumption (points in the
same cluster share labels, so boundaries lie in low-density regions), and the manifold
assumption (data lies on a lower-dimensional surface where nearby points share labels). If
these hold, unlabelled data is informative.

**Q4. What's the difference between semi-supervised and self-supervised learning?**
*Answer:* Semi-supervised uses a few human labels plus unlabelled data. Self-supervised
uses *no* human labels — it generates labels automatically from the data itself (e.g.
predicting a hidden word), which is how LLMs are pre-trained.

**Q5. What are the risks of semi-supervised learning?**
*Answer:* If the assumptions don't hold, unlabelled data may not help or can hurt. Self-
training can amplify a weak model's confident mistakes. So one should use a good base
model, confidence thresholds, and validate that performance actually improves.

## Scenario-Based Questions (with answers)

**Q1.** *You have 1,000,000 product images but budget to label only 5,000. How do you build
a good classifier?*
*Answer:* Use semi-supervised (or self-supervised pre-training): label the 5,000, then
leverage the unlabelled 995,000 via self-training/label propagation (or pre-train a model
self-supervised on all images then fine-tune on the 5,000). This extracts value from the
unlabelled majority, beating training on 5,000 alone.

**Q2.** *Your self-training model got worse than the supervised baseline. What might be
happening?*
*Answer:* Likely the assumptions don't hold for this data, or the base model was weak and
propagated confident errors via pseudo-labels. Try a stronger base model, a stricter
confidence threshold, label spreading instead, or conclude that unlabelled data isn't
helpful here.

**Q3.** *A teammate marks unlabelled rows as 0 (a real class) instead of -1 for
scikit-learn's self-training. What goes wrong?*
*Answer:* The classifier will treat all those points as genuinely belonging to class 0,
corrupting training. Unlabelled points must be marked -1 so the algorithm knows to predict
(pseudo-label) rather than trust them.

## Logic-Based Questions (with answers)

**Q1.** Why can unlabelled data improve a classifier even though it has no labels?
*Answer:* Because it reveals the **structure** of the feature space — clusters, density,
and manifolds — which (under the cluster/smoothness assumptions) indicates where decision
boundaries should lie. The few labels then assign meaning to that structure, sharpening the
boundary.

**Q2.** Why does self-training require a confidence threshold?
*Answer:* To avoid adding wrong pseudo-labels. Low-confidence predictions are likely
errors; baking them into training would propagate mistakes and degrade the model. Only
high-confidence predictions are trustworthy enough to add.

**Q3.** In the example, accuracy rose from 0.917 to 0.95 by adding unlabelled data. What
does this demonstrate about the unlabelled points?
*Answer:* That they carried useful structural information consistent with the labels (the
assumptions held), so leveraging them improved generalisation beyond the 131 labelled
points alone.

## Practical Questions (with answers)

**Q1.** How do you mark a point as unlabelled for scikit-learn's `SelfTrainingClassifier`?
*Answer:* Set its label to **-1** in the target array.

**Q2.** Write code to wrap an SVM in a self-training classifier.
*Answer:* `SelfTrainingClassifier(SVC(probability=True)).fit(X, y_semi)` where `y_semi`
has -1 for unlabelled points.

**Q3.** Which scikit-learn classes implement graph-based label propagation?
*Answer:* `LabelPropagation` and `LabelSpreading` (in `sklearn.semi_supervised`).

## Long Questions (with answers)

**Q1. Explain semi-supervised learning: its motivation, the assumptions it relies on, and
the main techniques, with the example from this chapter.**

*Answer:* Semi-supervised learning is motivated by the **label-cost problem**: labelled
data requires expensive human effort, while unlabelled data is cheap and plentiful, so we
want to extract value from the unlabelled majority. It combines a **small labelled set**
with a **large unlabelled set** and works only when the unlabelled data is informative,
which is captured by three **assumptions**: *smoothness* (nearby points share labels),
the *cluster assumption* (points in the same cluster share a label, so boundaries belong
in low-density gaps between clusters), and the *manifold assumption* (data lies on a
lower-dimensional surface where neighbours share labels). The main **techniques** are
**self-training (pseudo-labelling)** — train on the labels, predict the unlabelled data,
add the most confident predictions as new labels, and retrain, repeating with a confidence
threshold to limit error propagation — and **label propagation/spreading**, graph-based
methods where labels flow from labelled to similar unlabelled points along edges of a
similarity graph. The chapter's example hid 90% of the digit labels, leaving only 131
labelled points; a supervised model on those alone scored 0.917, while self-training,
which also used the 1,126 unlabelled points, reached 0.95 — a real gain achieved purely by
using data we already had. The caveat is that if the assumptions fail or the base model is
weak, semi-supervised learning may not help, so one must validate it.

**Q2. Compare supervised, unsupervised, semi-supervised, and self-supervised learning,
explaining how each uses labels and where each fits.**

*Answer:* These four paradigms differ in how they use labels. **Supervised learning** uses
a fully labelled dataset (every input paired with a correct output) to learn a mapping; it
is powerful but limited by the cost of obtaining labels. **Unsupervised learning** uses no
labels at all, instead discovering structure such as clusters or low-dimensional
representations; it's cheap on labels but produces no direct predictions of a target.
**Semi-supervised learning** sits between them: it uses a **small amount of labelled data
plus a large amount of unlabelled data**, letting the unlabelled data's structure improve a
model that the few labels make meaningful — ideal when labelling is expensive but raw data
is abundant (medical imaging, speech, text). **Self-supervised learning** uses **no human
labels** but isn't unsupervised either: it **automatically generates labels from the data
itself** (e.g. masking a word and predicting it, or a patch of an image and reconstructing
it), turning unlabelled data into a supervised-style task; this removes the label
bottleneck entirely and, at scale, is how Large Language Models and modern foundation
models are pre-trained before being fine-tuned (often with supervised or reinforcement
methods). In short: supervised needs all labels, unsupervised needs none and predicts no
target, semi-supervised needs a few labels and exploits the rest, and self-supervised
fabricates its own labels — a spectrum of decreasing reliance on costly human annotation.

## Exercises

1. Explain the label-cost problem and how semi-supervised learning addresses it.
2. State the three assumptions semi-supervised learning relies on.
3. Describe the self-training loop in four steps.
4. Why must unlabelled points be marked correctly (e.g. -1) for the algorithm?
5. Give two real domains where semi-supervised learning is valuable and why.

## Mini-Project

**Project: How many labels do you really need?**

1. Take a labelled dataset (e.g. digits). Progressively hide labels: keep 5%, 10%, 25%,
   50%.
2. At each level, compare a supervised model (few labels only) vs self-training (using
   unlabelled too).
3. Plot accuracy vs the fraction of labels kept for both approaches (Chapter 14).
4. Identify how few labels you can use before semi-supervised stops helping.
5. Write a short report on the labelling-cost vs accuracy trade-off. Save in
   `my-ml-journey/`.

## Assignments

1. **Coding:** Reproduce the chapter's experiment, then try `LabelSpreading` and compare it
   to self-training at the same label fraction.
2. **Coding:** Show a failure case: construct data where the cluster assumption is violated
   and demonstrate semi-supervised learning not helping.
3. **Conceptual:** Write one page distinguishing semi-supervised from self-supervised
   learning, with an example of each.

::: tip
We've now covered learning with full labels, no labels, and a few labels. The final
paradigm of Part V is completely different: Chapter 31, **Reinforcement Learning**, where
an agent learns by **trial, reward, and error** — the technology behind game-playing AIs
and robotics.
:::
