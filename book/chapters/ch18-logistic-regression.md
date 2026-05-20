# Logistic Regression

## Introduction

Despite its slightly confusing name, **Logistic Regression** is a **classification**
algorithm, not a regression one. It is the workhorse of binary classification — "yes
or no", "spam or not", "disease or healthy", "fraud or legitimate" — and it's
typically the *first* classifier you should try on any problem.

The brilliant idea is small: take linear regression's weighted sum, then **squash it
through an S-shaped "sigmoid" function** so the output becomes a **probability between
0 and 1**. That one change turns a number-predictor into a probability-predictor, and a
probability into a class.

::: keyidea
**Logistic regression = linear regression + sigmoid.** It computes `z = w·x + b` (just
like Chapter 17), then maps `z` to a probability with the sigmoid, then thresholds the
probability into a class. Understand this and you understand the building block of
every neural network neuron.
:::

By the end of this chapter you will be able to:

- Explain the **sigmoid function** and why it's used.
- Understand the **log-loss (cross-entropy)** cost and why MSE isn't used here.
- Set and interpret the **decision threshold**.
- Handle **binary and multiclass** classification.
- Implement logistic regression with **scikit-learn**, reading **probabilities** and
  **precision/recall**.
- Interpret coefficients as **log-odds**, and know the pros, cons, and use cases.

## From linear to logistic: the sigmoid

Linear regression outputs any number (−∞ to +∞). But a probability must live in
**[0, 1]**. The **sigmoid function** (also called the logistic function) squashes any
number into that range:

<div class="equation"><img class="eq" src="assets/images/eq_ch18_sigmoid.png" alt="sigmoid and z"></div>

![The sigmoid function squashes any input z into a probability between 0 and 1. Large positive z → near 1, large negative z → near 0, and z = 0 → exactly 0.5 (the default decision threshold).](assets/images/ch18_sigmoid.png)

```python
import numpy as np
def sigmoid(z): return 1 / (1 + np.exp(-z))
print("sigmoid(-2, 0, 2):", [round(float(sigmoid(v)), 3) for v in [-2, 0, 2]])
```

**Output:**
```text
sigmoid(-2, 0, 2): [0.119, 0.5, 0.881]
```

Notice: `z = 0` gives exactly **0.5**; negative `z` gives below 0.5; positive `z` gives
above. The further from zero, the closer to 0 or 1.

### From probability to class: the decision threshold

The model outputs a probability `p`. We convert it to a class with a **threshold**,
usually **0.5**:

- if `p ≥ 0.5` → predict class **1** (positive)
- if `p < 0.5` → predict class **0** (negative)

You can *move* the threshold to trade off the error types (Chapter 25): a lower
threshold catches more positives (higher recall) at the cost of more false alarms.

## The cost function: log-loss (cross-entropy)

To train, we need a cost that measures how wrong the *probabilities* are. We do **not**
use MSE here (it makes the optimisation non-convex and learning poor). Instead we use
**log-loss** (a.k.a. binary cross-entropy):

<div class="equation"><img class="eq" src="assets/images/eq_ch18_logloss.png" alt="log loss"></div>

The intuition: it **heavily punishes confident wrong answers.** If the true label is 1
and you predict `p = 0.99`, the loss is tiny; if you confidently predict `p = 0.01`, the
loss is huge. This pushes the model toward well-calibrated probabilities. Training
minimises log-loss with **gradient descent** (Chapter 5).

::: warning
**Why not MSE for classification?** With the sigmoid, MSE produces a bumpy
(non-convex) loss surface full of flat regions where gradient descent stalls. Log-loss
is convex for logistic regression, so gradient descent reliably finds the best weights.
Use the right loss for the task.
:::

## Binary vs multiclass classification

- **Binary** (2 classes) — one sigmoid output, as above.
- **Multiclass** (3+ classes) — two common strategies:
  - **One-vs-Rest (OvR):** train one binary classifier per class ("is it this class or
    not?") and pick the highest.
  - **Softmax (multinomial):** generalises the sigmoid to output a probability for each
    class that all sum to 1. (Softmax also powers the output layer of classification
    neural networks — Chapter 32.)

scikit-learn handles both automatically.

## Implementation with scikit-learn

Let's classify tumours as malignant/benign on the breast-cancer dataset.

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

sc = StandardScaler().fit(X_tr)              # logistic reg is gradient-based -> scale
model = LogisticRegression(max_iter=5000).fit(sc.transform(X_tr), y_tr)

X_te_s = sc.transform(X_te)
pred  = model.predict(X_te_s)                # the predicted classes (0/1)
proba = model.predict_proba(X_te_s)[:, 1]    # probability of class 1

print("accuracy: ", round(accuracy_score(y_te, pred), 3))
print("precision:", round(precision_score(y_te, pred), 3))
print("recall:   ", round(recall_score(y_te, pred), 3))
print("first 5 probabilities:", np.round(proba[:5], 3).tolist())
print("first 5 predictions:  ", pred[:5].tolist())
```

**Output:**
```text
accuracy:  0.982
precision: 0.986
recall:    0.986
first 5 probabilities: [0.0, 1.0, 0.006, 0.534, 0.0]
first 5 predictions:   [0, 1, 0, 1, 0]
```

### Explanation

- **`predict`** gives the class; **`predict_proba`** gives the underlying probability —
  a key advantage of logistic regression (you get *confidence*, not just a label).
- The probabilities make sense: 0.0 and 1.0 are confident calls; **0.534** is an
  *uncertain* case just above the 0.5 threshold (predicted 1, but barely).
- **Accuracy 0.982** with **precision and recall both 0.986** — excellent and balanced.
  (Precision = of those predicted positive, how many really were; recall = of the real
  positives, how many we caught — full treatment in Chapter 25.)

::: keyidea
Logistic regression didn't just classify — it gave **calibrated probabilities** and
remained **interpretable** (each coefficient affects the log-odds). For a medical
decision, knowing the model is "53% sure" vs "99% sure" is hugely valuable. This blend
of simplicity, speed, probability output, and interpretability is why it's the default
first classifier.
:::

## Interpreting coefficients (log-odds)

In logistic regression, each coefficient affects the **log-odds** of the positive
class. A positive coefficient means "increasing this feature increases the probability
of class 1"; a negative one decreases it. Exponentiating a coefficient gives an **odds
ratio** (how the odds multiply per unit increase) — interpretable, which is why the
method is loved in medicine and finance.

::: tip
**Practical & debugging tips:** (1) **Scale features** — logistic regression is
gradient-based. (2) If you get a "failed to converge" warning, increase `max_iter` or
scale features. (3) Use `predict_proba` to access probabilities and to **tune the
threshold** for your precision/recall needs (Chapter 25). (4) Use the `C` parameter
(inverse regularization strength) and `penalty` ('l2'/'l1') to control overfitting
(Chapter 26) — smaller `C` = stronger regularization. (5) For imbalanced data, set
`class_weight="balanced"`.
:::

## Advantages, disadvantages, and use cases

| Advantages | Disadvantages |
|---|---|
| Simple, fast, interpretable | Assumes a (roughly) linear decision boundary |
| Outputs **probabilities** | Underfits complex non-linear patterns |
| Strong baseline classifier | Sensitive to outliers and multicollinearity |
| Works well in high dimensions (text) | Needs feature scaling |
| Easy to regularize | Struggles when classes overlap heavily |

**Use cases:** spam detection, medical diagnosis, credit default/fraud scoring,
customer churn, click-through prediction, and any binary decision where you want
probabilities and interpretability.

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Thinking logistic regression does regression.** It's a *classification*
algorithm; the "regression" in its name refers to the underlying linear model.
:::

- **Mistake 2 — Using MSE as the loss** (use log-loss/cross-entropy).
- **Mistake 3 — Forgetting to scale features**, causing slow/failed convergence.
- **Mistake 4 — Always using a 0.5 threshold** even when the costs of the two error
  types differ (move it, Chapter 25).
- **Mistake 5 — Reporting only accuracy on imbalanced data** (use precision/recall/AUC).
- **Mistake 6 — Expecting it to learn highly non-linear boundaries** without engineered
  features.

## Best practices

- **Scale features** and increase `max_iter` if needed.
- **Use it as your first classifier / baseline.**
- **Read `predict_proba`** and tune the threshold to your costs.
- **Evaluate with precision, recall, F1, and AUC**, not just accuracy (Chapter 25).
- **Regularize** (`C`, `penalty`) for many or correlated features.
- **Add polynomial/interaction features** for mild non-linearity, or switch models for
  strong non-linearity.

## Chapter Summary

- **Logistic regression** is a **classification** algorithm: it computes a linear score
  `z = w·x + b`, squashes it with the **sigmoid** into a probability in [0, 1], and
  thresholds (default 0.5) into a class.
- It's trained by minimising **log-loss (cross-entropy)** — which punishes confident
  wrong answers — via gradient descent. **Not MSE.**
- It handles **binary** (one sigmoid) and **multiclass** (one-vs-rest or softmax)
  problems, and outputs **interpretable probabilities** and log-odds coefficients.
- On breast-cancer data it reached **0.982 accuracy** with balanced precision/recall
  (0.986), and revealed an uncertain case (p=0.534).
- It's the **default first classifier**: simple, fast, probabilistic, and
  interpretable — but assumes a roughly linear boundary and needs scaling.

---

::: {.qband}
Practice Zone — Chapter 18
:::

## Multiple-Choice Questions (MCQs)

**Q1.** Logistic regression is used for:
a) Regression  b) Classification  c) Clustering  d) Dimensionality reduction

**Q2.** The sigmoid function outputs values in the range:
a) (−∞, ∞)  b) [0, 1]  c) [−1, 1]  d) {0, 1}

**Q3.** `sigmoid(0)` equals:
a) 0  b) 1  c) 0.5  d) −1

**Q4.** The cost function for logistic regression is:
a) MSE  b) Log-loss (cross-entropy)  c) MAE  d) R²

**Q5.** The default decision threshold for binary logistic regression is:
a) 0  b) 0.5  c) 1  d) 0.1

**Q6.** `predict_proba` returns:
a) The class label  b) The probability of each class  c) The accuracy  d) The
coefficients

**Q7.** For 3+ classes, logistic regression can use:
a) Only binary  b) Softmax or one-vs-rest  c) MSE  d) Clustering

**Q8.** Why not use MSE for logistic regression?
a) It's too fast  b) It makes the loss non-convex / learning poor  c) It needs scaling
d) It's only for trees

### MCQ Answers
**1:** b. **2:** b. **3:** c. **4:** b. **5:** b. **6:** b. **7:** b. **8:** b.

## Interview Questions (with answers)

**Q1. How does logistic regression work?**
*Answer:* It computes a linear combination of features `z = w·x + b`, passes it through
the sigmoid to get a probability between 0 and 1, and thresholds that probability
(default 0.5) to assign a class. It's trained by minimising log-loss via gradient
descent.

**Q2. Why is it called "regression" if it does classification?**
*Answer:* Because it builds on a linear *regression* model of the log-odds; the linear
score is then mapped to a probability. The underlying machinery is regression, but the
output (a class) makes it a classifier.

**Q3. Why use log-loss instead of MSE?**
*Answer:* With the sigmoid, MSE yields a non-convex loss with flat regions where
gradient descent stalls, and it doesn't penalise confident wrong probabilities well.
Log-loss is convex for logistic regression and strongly penalises confident mistakes,
giving reliable training and well-behaved probabilities.

**Q4. How does logistic regression handle multiclass problems?**
*Answer:* Via one-vs-rest (train one binary classifier per class and pick the highest)
or multinomial softmax (output probabilities for all classes that sum to 1). scikit-learn
supports both.

**Q5. How do you interpret the coefficients?**
*Answer:* Each coefficient is the change in the log-odds of the positive class per unit
increase in that feature (with others fixed). Exponentiating gives an odds ratio.
Positive coefficients raise the probability of class 1; negative ones lower it.

## Scenario-Based Questions (with answers)

**Q1.** *You're detecting a rare cancer. Missing a real case (false negative) is far
worse than a false alarm. Default threshold 0.5 misses too many. What do you do?*
*Answer:* Lower the decision threshold (e.g. to 0.3) so more cases are flagged positive,
increasing recall (catching more true cancers) at the cost of more false positives —
acceptable given the asymmetric costs. Use `predict_proba` and choose the threshold via
a precision-recall analysis (Chapter 25).

**Q2.** *Your logistic regression warns "failed to converge." What are the likely
causes and fixes?*
*Answer:* Usually unscaled features or too few iterations. Fix by standardising features
and/or increasing `max_iter`; also check for extreme outliers and consider
regularization.

**Q3.** *Stakeholders want to know not just the predicted class but how confident the
model is. Why is logistic regression a good fit?*
*Answer:* Because it natively outputs calibrated probabilities via `predict_proba`, so
you can report confidence (e.g. "78% likely to churn") and set thresholds appropriate to
the decision's stakes — something pure label-only classifiers don't provide as cleanly.

## Logic-Based Questions (with answers)

**Q1.** If `z = w·x + b = 0`, what probability does the model output and what class
(at threshold 0.5)?
*Answer:* sigmoid(0) = 0.5, exactly the threshold. It's the boundary case; conventionally
predicted as class 1 (≥ 0.5), but it's maximally uncertain.

**Q2.** A model outputs p = 0.534 for a sample. Why might this prediction be considered
"risky"?
*Answer:* Because it's barely above the 0.5 threshold — the model is nearly 50/50, so
small changes could flip the prediction. Such low-confidence cases are where errors
concentrate and may warrant human review or more data.

**Q3.** Confidently predicting p = 0.01 when the true label is 1 yields a huge log-loss.
Why is that desirable behaviour for the loss?
*Answer:* Because it strongly discourages confident wrong answers, pushing the model to
be both accurate *and* honestly calibrated. A loss that didn't punish confident errors
would tolerate dangerous overconfidence.

## Practical Questions (with answers)

**Q1.** Write code to get the probability of the positive class from a fitted model.
*Answer:* `model.predict_proba(X)[:, 1]`.

**Q2.** How would you classify using a threshold of 0.3 instead of 0.5?
*Answer:* `(model.predict_proba(X)[:, 1] >= 0.3).astype(int)`.

**Q3.** Why did we apply `StandardScaler` before logistic regression?
*Answer:* Because logistic regression is gradient-based and sensitive to feature scale;
scaling speeds and stabilises convergence and makes coefficients comparable.

## Long Questions (with answers)

**Q1. Explain how logistic regression transforms a linear model into a probabilistic
classifier, covering the sigmoid, the threshold, the loss function, and training.**

*Answer:* Logistic regression begins exactly like linear regression by computing a
linear score z = w·x + b, a weighted sum of features plus a bias. Because a raw score
can be any real number while a probability must lie in [0, 1], it applies the **sigmoid
function** σ(z) = 1/(1 + e⁻ᶻ), an S-shaped curve mapping any z to a probability: large
positive z → near 1, large negative z → near 0, and z = 0 → exactly 0.5. This
probability is converted to a class using a **decision threshold** (default 0.5):
p ≥ 0.5 → class 1, else class 0; the threshold can be moved to trade precision against
recall. To **train**, we need a loss measuring how wrong the probabilities are: logistic
regression uses **log-loss (binary cross-entropy)**, which heavily penalises confident
wrong predictions and is convex for this model, unlike MSE which would be non-convex
under the sigmoid. **Gradient descent** then adjusts the weights to minimise log-loss.
The result is a fast, interpretable classifier whose coefficients describe each
feature's effect on the log-odds and whose `predict_proba` outputs calibrated
confidence, making it the standard first choice for binary classification and the
conceptual basis of a neural-network neuron.

**Q2. Discuss the strengths, limitations, and appropriate use cases of logistic
regression, including how to extend it to harder problems.**

*Answer:* **Strengths:** logistic regression is simple, fast, and highly
**interpretable** — coefficients map to log-odds and odds ratios, prized in medicine,
finance, and the social sciences — and it natively outputs **probabilities**, enabling
threshold tuning and confidence reporting. It performs well in high-dimensional sparse
settings (like text classification), is easy to **regularize** (L1/L2 via the `C`
parameter), and makes an excellent **baseline**. **Limitations:** it assumes a roughly
**linear decision boundary**, so it underfits strongly non-linear patterns; it is
sensitive to outliers and multicollinearity; it needs **feature scaling** to train
well; and it struggles when classes heavily overlap. **Use cases:** spam detection,
disease diagnosis, credit/fraud scoring, churn, and click-through prediction — anywhere
a probabilistic, explainable binary (or, via softmax/one-vs-rest, multiclass) decision
is needed. To **extend** it to harder problems, add polynomial/interaction features
(Chapter 12) for mild non-linearity, apply regularization for many/correlated features,
adjust the threshold and class weights for imbalanced or asymmetric-cost problems, and —
when the boundary is genuinely complex — move to non-linear models such as kernel SVMs,
tree ensembles, or neural networks while still keeping logistic regression as the
interpretable benchmark.

## Exercises

1. Compute `sigmoid(z)` by hand for z = 0 and explain what it means for classification.
2. Explain in two sentences why logistic regression uses log-loss instead of MSE.
3. A model outputs probabilities [0.92, 0.10, 0.55, 0.49]. Give the predicted classes at
   threshold 0.5, then at threshold 0.4.
4. List three real problems where you'd want probabilities, not just labels.
5. Explain what a positive coefficient means for the predicted probability.

## Mini-Project

**Project: A spam/churn probability classifier.**

1. Take a binary classification dataset (e.g. churn, or breast cancer).
2. Scale features, train logistic regression, and report accuracy, precision, and
   recall on the test set.
3. Print the probabilities for 10 samples and identify the most *uncertain* ones (near
   0.5).
4. Re-classify at thresholds 0.3 and 0.7 and describe how precision and recall change.
5. Interpret the three largest-magnitude coefficients. Save in `my-ml-journey/`.

## Assignments

1. **Coding:** Implement the sigmoid and log-loss from scratch, then verify your
   log-loss matches `sklearn.metrics.log_loss` on some probabilities and labels.
2. **Coding:** On a multiclass dataset (e.g. iris or digits), train multinomial logistic
   regression and report per-class precision/recall.
3. **Conceptual:** Write one page explaining how logistic regression relates to a single
   neuron in a neural network (linear score + activation), previewing Chapter 32.

::: tip
You can now predict numbers (Ch 17) and probabilities/classes (Ch 18) with linear
models. Chapter 19, **K-Nearest Neighbors**, takes a completely different,
non-parametric approach: predict based on your closest examples — no equation, no
training in the usual sense.
:::
