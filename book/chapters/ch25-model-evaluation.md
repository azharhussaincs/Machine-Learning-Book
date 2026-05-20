# Model Evaluation, Validation & Metrics

## Introduction

You've built models in Chapters 17–24. But a crucial question remains: **how do you
know if a model is actually good?** "It's 95% accurate!" sounds great — until you learn
that 95% of the data was one class, so a model that *always guesses that class* would
also score 95%. Choosing the wrong metric is one of the most common and costly mistakes
in Machine Learning.

This chapter teaches you to **evaluate models honestly**: how to split and validate data
properly, and which metric to trust for each situation. These skills separate
practitioners who *think* their model works from those who *know*.

::: keyidea
The right metric depends on the problem. **Accuracy can lie**, especially on imbalanced
data. The **confusion matrix** and metrics derived from it (precision, recall, F1, AUC)
tell the real story — and which one matters depends on the *cost of each kind of
mistake*.
:::

By the end of this chapter you will be able to:

- Split data correctly (train/validation/test) and use **cross-validation**.
- Read a **confusion matrix** and compute **accuracy, precision, recall, F1**.
- Understand **ROC curves and AUC**.
- Recognise the **accuracy paradox** on imbalanced data and pick the right metric.

## Proper validation: train, validation, and test

- **Training set** — the model learns from this.
- **Validation set** — used to tune hyperparameters and choose models (Chapter 26).
- **Test set** — touched **only once**, at the very end, to estimate real-world
  performance.

::: warning
**Never tune on the test set.** If you repeatedly check the test set and adjust your
model, you indirectly "learn" it, and your reported performance becomes optimistic and
dishonest. Keep the test set in a vault until the end.
:::

### Cross-validation: using data efficiently

A single train/validation split wastes data and depends on luck. **k-fold
cross-validation** splits the training data into *k* parts ("folds"), trains on k−1 and
validates on the remaining one, and rotates so every fold is validated once. The k
scores are averaged for a robust estimate.

![5-fold cross-validation: the data is split into 5 folds; the model trains on 4 and validates on the 5th, rotating through all folds. Averaging the 5 scores gives a reliable performance estimate.](assets/images/ch25_crossval.png)

```python
from sklearn.model_selection import cross_val_score
cv = cross_val_score(model, X_train, y_train, cv=5)
print("5-fold CV mean±std:", round(cv.mean(), 3), "±", round(cv.std(), 3))
```

**Output:**
```text
5-fold CV mean±std: 0.98 ± 0.015
```

The ± tells you how *stable* the performance is across folds — a small spread means a
reliable estimate. Use **stratified** k-fold for classification to keep class
proportions in each fold.

## The confusion matrix

For classification, the **confusion matrix** is the foundation of all metrics. It cross-
tabulates predictions vs reality into four cells:

![The confusion matrix. Rows are the true class, columns the predicted class. The four cells — True Positives, False Positives, True Negatives, False Negatives — are the basis of every classification metric.](assets/images/ch25_confusion.png)

- **True Positive (TP)** — predicted positive, *was* positive. ✓
- **True Negative (TN)** — predicted negative, *was* negative. ✓
- **False Positive (FP)** — predicted positive, *was* negative ("false alarm", Type I
  error).
- **False Negative (FN)** — predicted negative, *was* positive ("miss", Type II error).

```python
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, pred))
```

**Output:**
```text
[[ 63   1]
 [  1 106]]
```

Reading it: 63 TN, 106 TP (the diagonal — correct), and just 1 FP and 1 FN (off-diagonal
— errors). An almost-perfect classifier.

## The core metrics

From the four cells we derive the key metrics:

**Accuracy** — fraction of all predictions that were correct:

<div class="equation"><img class="eq" src="assets/images/eq_ch25_accuracy.png" alt="accuracy"></div>

**Precision** — of those *predicted positive*, how many really were? (Punishes false
alarms.)

<div class="equation"><img class="eq" src="assets/images/eq_ch25_precision.png" alt="precision"></div>

**Recall (Sensitivity)** — of the *actual positives*, how many did we catch? (Punishes
misses.)

<div class="equation"><img class="eq" src="assets/images/eq_ch25_recall.png" alt="recall"></div>

**F1-score** — the harmonic mean of precision and recall (a single balanced number):

<div class="equation"><img class="eq" src="assets/images/eq_ch25_f1.png" alt="F1 score"></div>

::: keyidea
**Precision vs recall is a trade-off.** A spam filter with high precision rarely flags
good email as spam (few false alarms) but may let some spam through; high recall catches
all spam but may flag good email. Which matters more depends entirely on the **cost of
each error** — and you tune the decision threshold (Chapter 18) to balance them.
:::

### The full classification report

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, pred, digits=3))
```

**Output:**
```text
              precision    recall  f1-score   support

           0      0.984     0.984     0.984        64
           1      0.991     0.991     0.991       107

    accuracy                          0.988       171
   macro avg      0.988     0.988     0.988       171
weighted avg      0.988     0.988     0.988       171
```

One call gives precision, recall, and F1 *per class*, plus accuracy and averages. This is
your go-to summary for any classifier.

## ROC curve and AUC

The **ROC curve** plots the **True Positive Rate (recall)** against the **False Positive
Rate** as you vary the decision threshold. The **AUC** (Area Under the Curve) summarises
it in one number: **1.0 = perfect**, **0.5 = random guessing**.

![An ROC curve plots true-positive rate vs false-positive rate across all thresholds. A model hugging the top-left corner (high AUC) is excellent; the diagonal (AUC 0.5) is random guessing.](assets/images/ch25_roc.png)

```python
from sklearn.metrics import roc_auc_score
print("ROC AUC:", round(roc_auc_score(y_test, proba), 3))
```

**Output:**
```text
ROC AUC: 0.998
```

AUC is **threshold-independent** and works well even on imbalanced data, making it a
favourite for comparing classifiers. (For very imbalanced data, the **precision-recall
curve** is often more informative.)

## The accuracy paradox (imbalanced data)

Suppose 99% of transactions are legitimate and 1% are fraud. A lazy model that predicts
"legitimate" for *everything* scores **99% accuracy** — yet catches **zero fraud**! This
is the **accuracy paradox**.

::: warning
**On imbalanced data, accuracy is misleading.** Always look at **precision, recall, F1,
and AUC** for the minority class — and consider techniques like resampling, class
weights (`class_weight="balanced"`), or threshold tuning. The whole point (catching
fraud) is in the rare class that accuracy ignores.
:::

## Regression metrics (recap)

For regression (Chapter 17), use **MAE**, **MSE/RMSE** (in target units), and **R²**
(variance explained). There's no confusion matrix — you measure how far predictions are
from true values.

## Choosing the right metric

| Situation | Prefer |
|---|---|
| Balanced classes, equal error costs | Accuracy |
| Imbalanced classes | Precision, Recall, F1, AUC |
| False alarms costly (e.g. spam → inbox) | **Precision** |
| Misses costly (e.g. cancer screening) | **Recall** |
| Need one balanced number | **F1** |
| Comparing classifiers across thresholds | **AUC** |
| Regression | RMSE/MAE + R² |

::: tip
**Practical & debugging tips:** (1) Always start with the **confusion matrix** — it
reveals *which* errors happen. (2) Use **stratified k-fold CV** for reliable, fair
estimates. (3) Report the metric that matches the **business cost**, not just accuracy.
(4) For imbalance, set `class_weight="balanced"` and tune the threshold via the
precision-recall curve. (5) Keep the **test set untouched** until the very end.
:::

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Trusting accuracy on imbalanced data.** The accuracy paradox makes a
useless model look great. Use precision/recall/F1/AUC.
:::

- **Mistake 2 — Tuning on the test set** (data leakage; optimistic results).
- **Mistake 3 — Using a single train/test split** instead of cross-validation for model
  selection.
- **Mistake 4 — Optimising the wrong metric** for the problem's error costs.
- **Mistake 5 — Ignoring the precision-recall trade-off** and the decision threshold.
- **Mistake 6 — Not stratifying** folds/splits for imbalanced classification.

## Best practices

- **Split into train/validation/test; guard the test set** until the end.
- **Use (stratified) cross-validation** for robust estimates.
- **Read the confusion matrix first**, then choose metrics by error cost.
- **Prefer F1/AUC/precision/recall** on imbalanced data.
- **Tune the threshold** to balance precision and recall for your use case.
- **Report mean ± std** across folds, not a single lucky number.

## Chapter Summary

- Evaluate honestly: **train** (learn), **validation** (tune), **test** (final, untouched
  estimate); use **k-fold (stratified) cross-validation** for robust scores.
- The **confusion matrix** (TP, FP, TN, FN) is the basis of classification metrics:
  **accuracy**, **precision** (few false alarms), **recall** (few misses), and **F1**
  (their balance).
- **ROC/AUC** summarise performance across thresholds (1.0 perfect, 0.5 random) and suit
  imbalanced data; the model here scored **AUC 0.998**, **F1 ~0.99**, CV **0.98±0.015**.
- Beware the **accuracy paradox**: on imbalanced data, accuracy is misleading — use
  precision/recall/F1/AUC and consider class weights/threshold tuning.
- **Choose the metric by the cost of each error**, and never tune on the test set.

---

::: {.qband}
Practice Zone — Chapter 25
:::

## Multiple-Choice Questions (MCQs)

**Q1.** A False Positive is when the model predicts positive but the truth is:
a) Positive  b) Negative  c) Missing  d) Unknown

**Q2.** Precision is:
a) TP/(TP+FN)  b) TP/(TP+FP)  c) (TP+TN)/all  d) TN/(TN+FP)

**Q3.** Recall is:
a) TP/(TP+FP)  b) TP/(TP+FN)  c) TN/all  d) FP/(FP+TN)

**Q4.** The F1-score is the:
a) Sum of precision and recall  b) Harmonic mean of precision and recall  c) Accuracy
d) AUC

**Q5.** An AUC of 0.5 means the model is:
a) Perfect  b) Random guessing  c) Overfit  d) Underfit

**Q6.** On data that is 99% one class, high accuracy:
a) Proves a great model  b) Can be misleading (accuracy paradox)  c) Means high recall
d) Means high AUC

**Q7.** For cancer screening where missing a case is dangerous, optimise:
a) Precision  b) Recall  c) Accuracy  d) Training speed

**Q8.** k-fold cross-validation is used to:
a) Train faster  b) Get a robust performance estimate  c) Replace the test set forever
d) Scale features

### MCQ Answers
**1:** b. **2:** b. **3:** b. **4:** b. **5:** b. **6:** b. **7:** b. **8:** b.

## Interview Questions (with answers)

**Q1. Why can accuracy be a poor metric?**
*Answer:* On imbalanced data, predicting the majority class for everything yields high
accuracy while failing entirely on the (often important) minority class — the accuracy
paradox. Accuracy ignores *which* errors occur, so precision, recall, F1, and AUC are
preferred when classes are imbalanced or error costs differ.

**Q2. Explain precision vs recall and the trade-off.**
*Answer:* Precision = TP/(TP+FP), the fraction of positive predictions that are correct
(penalises false alarms). Recall = TP/(TP+FN), the fraction of actual positives caught
(penalises misses). Raising the threshold typically increases precision but lowers
recall, and vice versa; the right balance depends on the cost of each error type.

**Q3. What is the confusion matrix and why is it useful?**
*Answer:* A table cross-tabulating predicted vs actual classes into TP, FP, TN, FN. It
reveals exactly which kinds of errors a classifier makes, and all standard metrics
(accuracy, precision, recall, F1) are derived from its cells — making it the starting
point of evaluation.

**Q4. What is cross-validation and why use it?**
*Answer:* It splits training data into k folds, trains on k−1 and validates on the
remaining fold, rotating so each fold is validated once, then averages the scores. It
gives a more robust, less luck-dependent estimate than a single split and uses data
efficiently; stratified k-fold preserves class proportions.

**Q5. What is AUC and what does it measure?**
*Answer:* The Area Under the ROC Curve summarises classification performance across all
thresholds: it's the probability the model ranks a random positive above a random
negative. AUC of 1.0 is perfect, 0.5 is random; it's threshold-independent and useful for
comparing models, including on imbalanced data.

## Scenario-Based Questions (with answers)

**Q1.** *A fraud model reports 99.5% accuracy, and management is thrilled. Fraud is 0.5%
of cases. Why are you skeptical, and what do you check?*
*Answer:* A model predicting "not fraud" for everything scores 99.5% yet catches no
fraud (accuracy paradox). I'd check the confusion matrix and the **recall and precision
for the fraud class** and the AUC/PR curve — if recall is near zero, the model is useless
despite the headline accuracy.

**Q2.** *A spam filter is flagging important emails as spam, annoying users. Which metric
should you prioritise and how do you adjust?*
*Answer:* Prioritise **precision** for the spam class (minimise false positives — good
mail wrongly flagged). Raise the decision threshold so the model only flags spam when
very confident, accepting that some spam slips through (lower recall).

**Q3.** *Two models have the same accuracy, but you must choose one for medical
diagnosis. What do you compare?*
*Answer:* Compare **recall** (to avoid missing sick patients), precision, F1, and AUC,
and inspect the confusion matrices. In medicine, missing a true case (false negative) is
usually far costlier, so the higher-recall model is typically preferred.

## Logic-Based Questions (with answers)

**Q1.** A model predicts the majority class for everything on 90/10 data. What are its
accuracy and its recall for the minority class?
*Answer:* Accuracy ≈ 90% (it gets all majority cases right), but minority-class recall =
0% (it catches none of them) — illustrating why accuracy alone is misleading.

**Q2.** Why is F1 the harmonic (not arithmetic) mean of precision and recall?
*Answer:* The harmonic mean punishes imbalance: if either precision or recall is very
low, F1 is low. An arithmetic mean could stay high when one is near zero, hiding a serious
weakness, so the harmonic mean better reflects needing *both* to be good.

**Q3.** If raising the threshold increases precision but decreases recall, why does that
happen?
*Answer:* A higher threshold makes the model predict positive only when very confident,
so fewer false positives (higher precision) but also more missed true positives (lower
recall) — the precision-recall trade-off.

## Practical Questions (with answers)

**Q1.** Write code to print a full classification report.
*Answer:* `from sklearn.metrics import classification_report;
print(classification_report(y_test, pred))`.

**Q2.** Which scikit-learn function gives 5-fold cross-validation scores?
*Answer:* `cross_val_score(model, X, y, cv=5)`.

**Q3.** How do you compute AUC, and what input does it need?
*Answer:* `roc_auc_score(y_true, y_scores)` where `y_scores` are predicted probabilities
or decision scores for the positive class (e.g. `model.predict_proba(X)[:, 1]`), not
hard class labels.

## Long Questions (with answers)

**Q1. Explain the confusion matrix and the metrics derived from it (accuracy, precision,
recall, F1), including when each metric is the right choice.**

*Answer:* The **confusion matrix** cross-tabulates predictions against truth into four
cells: **True Positives (TP)** and **True Negatives (TN)** are correct predictions, while
**False Positives (FP)** (predicted positive but actually negative — a false alarm) and
**False Negatives (FN)** (predicted negative but actually positive — a miss) are the two
error types. From these: **Accuracy** = (TP+TN)/all, the overall fraction correct — fine
when classes are balanced and error costs equal, but misleading on imbalanced data.
**Precision** = TP/(TP+FP), the fraction of positive predictions that are correct — the
right focus when **false positives are costly** (e.g. flagging good email as spam, or
convicting the innocent). **Recall (sensitivity)** = TP/(TP+FN), the fraction of actual
positives caught — the right focus when **false negatives are costly** (e.g. missing a
cancer or a fraud). Because precision and recall trade off against each other as the
decision threshold moves, the **F1-score**, their harmonic mean, gives a single balanced
number that is high only when both are high. Choosing the right metric means asking which
error is more expensive in the real problem and optimising accordingly, rather than
defaulting to accuracy.

**Q2. Describe a complete, honest evaluation procedure for a classification model, from
data splitting to final reporting, and explain how it avoids common pitfalls.**

*Answer:* A sound procedure begins by **splitting** the data into a training set, a
validation set (or using cross-validation), and a **test set that is locked away** until
the very end. During development, use **stratified k-fold cross-validation** on the
training data to estimate performance robustly and to **tune hyperparameters and select
models** — never touching the test set, because tuning on it leaks information and yields
optimistic, dishonest results. For each candidate model, examine the **confusion matrix**
to see *which* errors occur, then compute the metrics that match the **cost of those
errors**: accuracy only if classes are balanced, otherwise **precision, recall, F1, and
AUC**, focusing on the minority class for imbalanced problems. Tune the **decision
threshold** using a precision-recall (or ROC) analysis to balance false alarms against
misses for the use case, and handle imbalance with class weights or resampling. Report
results as **mean ± standard deviation across folds** to convey stability, not a single
lucky number. Only after the model and hyperparameters are finalised do you evaluate
**once** on the held-out test set to estimate real-world performance. This procedure
avoids the key pitfalls — the accuracy paradox, test-set leakage, luck-dependent single
splits, and optimising an irrelevant metric — producing an honest, trustworthy
assessment.

## Exercises

1. Given a confusion matrix [[TN=80, FP=20],[FN=10, TP=90]], compute accuracy, precision,
   and recall.
2. For each problem, name the metric to prioritise: spam filter, cancer screening,
   recommendation relevance, credit-fraud detection.
3. Explain the accuracy paradox with your own example.
4. Why is the test set kept untouched until the end?
5. Explain precision vs recall to a non-technical friend.

## Mini-Project

**Project: Evaluate honestly on imbalanced data.**

1. Take or create an imbalanced classification dataset (e.g. fraud, or down-sample one
   class).
2. Train a classifier and print the confusion matrix and full classification report.
3. Compute AUC and plot the ROC and precision-recall curves (Chapter 14).
4. Show the accuracy paradox: compare your model's accuracy to a "predict majority"
   baseline, then compare their recall on the minority class.
5. Tune the decision threshold to improve recall and discuss the precision cost. Save in
   `my-ml-journey/`.

## Assignments

1. **Coding:** On a dataset, run 5-fold and 10-fold stratified cross-validation and
   report mean ± std. Discuss the stability.
2. **Coding:** Train a classifier, then sweep the decision threshold from 0.1 to 0.9 and
   plot precision and recall vs threshold. Identify a good operating point for a chosen
   use case.
3. **Conceptual:** Write one page on "why accuracy is not enough", with two real examples
   where the wrong metric would lead to a harmful decision.

::: tip
You can now evaluate models honestly. The final piece of Part IV, Chapter 26, is
**Hyperparameter Tuning & Regularization** — how to systematically *improve* models and
control overfitting, using the validation techniques you just learned.
:::
