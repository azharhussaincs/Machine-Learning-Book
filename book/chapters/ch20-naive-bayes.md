# Naive Bayes

## Introduction

**Naive Bayes** is a beautifully simple, fast, and surprisingly powerful
classification algorithm built directly on **Bayes' theorem** (Chapter 6). It is the
classic engine behind **spam filters** and is a go-to for **text classification**.

Its secret is a bold simplifying assumption — that all features are **independent** of
each other given the class. This assumption is usually *false* in the real world
(hence "naive"), yet the algorithm works remarkably well anyway, especially on text.

::: keyidea
Naive Bayes asks: *"Given these features, which class is most probable?"* It uses
Bayes' theorem to flip that into something computable: combine the **prior** (how
common each class is) with the **likelihood** of seeing these features in each class.
The "naive" part is assuming features don't interact — which trades a little accuracy
for enormous speed and simplicity.
:::

By the end of this chapter you will be able to:

- Apply **Bayes' theorem** to classification.
- Understand the **naive independence assumption** and why it still works.
- Choose between **Gaussian, Multinomial, and Bernoulli** Naive Bayes.
- Understand **Laplace smoothing**.
- Build a spam classifier and a numeric classifier with scikit-learn.

## From Bayes' theorem to a classifier

Recall Bayes' theorem (Chapter 6): `P(A|B) = P(B|A)·P(A) / P(B)`. For classification,
we want the probability of each **class** `c` given the **features** `x₁…xₙ`:

The "naive" assumption — that features are independent given the class — lets us
multiply individual feature probabilities, giving the **Naive Bayes classifier**:

<div class="equation"><img class="eq" src="assets/images/eq_ch20_nb.png" alt="naive bayes"></div>

In words: **the probability of a class is proportional to its prior `P(c)` times the
product of the likelihoods `P(xᵢ|c)` of each feature.** We compute this for every class
and **pick the class with the highest value.** (We drop the denominator `P(B)` because
it's the same for all classes — we only need to compare.)

![Naive Bayes assumes features are independent given the class, so it multiplies their individual likelihoods. It computes a score for each class and picks the highest. The independence assumption is "naive" but works well in practice.](assets/images/ch20_naive_bayes.png)

### Why "naive"?

The independence assumption says, for example, that in spam detection the word "free"
appearing is independent of "money" appearing — which isn't really true (spam emails
often have both together). Yet despite this wrong assumption, Naive Bayes classifies
very well, because for *picking the most likely class* it usually doesn't matter that
the exact probabilities are off.

## The three flavours of Naive Bayes

The right variant depends on your feature type:

| Variant | Feature type | Typical use |
|---|---|---|
| **Gaussian NB** | Continuous numbers (assumed normal) | Numeric data (e.g. iris) |
| **Multinomial NB** | Counts (e.g. word counts) | **Text classification, spam** |
| **Bernoulli NB** | Binary (present/absent) | Text with word presence flags |

## Laplace smoothing: handling unseen features

Problem: if a word never appeared in spam during training, its likelihood `P(word|spam)`
is **zero** — and since we *multiply* probabilities, one zero makes the whole product
zero, wrongly ruling out the class. **Laplace (additive) smoothing** fixes this by
adding a small count (usually 1) to every feature count, so no probability is ever
exactly zero. scikit-learn does this by default (the `alpha` parameter).

## Practical: a spam classifier with Multinomial NB

This is Naive Bayes' most famous application. We turn text into word counts, then
classify.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

texts = ["win money now", "cheap meds buy now", "hi how are you",
         "let us meet tomorrow", "free prize click here", "call me later",
         "claim your free reward", "see you at lunch"]
labels = [1, 1, 0, 0, 1, 0, 1, 0]          # 1 = spam, 0 = not spam

# Turn text into a matrix of word counts ("bag of words")
cv = CountVectorizer()
X_counts = cv.fit_transform(texts)

# Train Multinomial Naive Bayes
nb = MultinomialNB().fit(X_counts, labels)

# Classify new messages
test = ["free money now", "meet me for lunch"]
print("predictions:", nb.predict(cv.transform(test)).tolist())
print("spam probabilities:", nb.predict_proba(cv.transform(test))[:, 1].round(3).tolist())
```

**Output:**
```text
predictions: [1, 0]
spam probabilities: [0.947, 0.111]
```

### Explanation

- **`CountVectorizer`** converts each message into word counts (a "bag of words") — the
  feature representation Multinomial NB expects.
- **"free money now"** was flagged as **spam (0.947 probability)** — it contains
  "free" and "money", strongly associated with spam in training.
- **"meet me for lunch"** was correctly classified as **not spam (0.111)** — its words
  appeared in the non-spam messages.
- With just 8 tiny training examples, Naive Bayes learned sensible spam patterns — its
  efficiency on text is legendary.

## Practical: Gaussian NB on numeric data

For continuous features, use **Gaussian NB**, which models each feature as a normal
distribution per class.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y)
g = GaussianNB().fit(X_tr, y_tr)
print("GaussianNB iris accuracy:", round(accuracy_score(y_te, g.predict(X_te)), 3))
```

**Output:**
```text
GaussianNB iris accuracy: 0.978
```

97.8% accuracy on iris from an extremely fast, assumption-light model — Naive Bayes
makes a strong, instant baseline.

::: keyidea
Two lines of modelling, two domains (text and numbers), both excellent results. Naive
Bayes' combination of **speed, simplicity, and effectiveness** — especially on
high-dimensional text — is why it remains a first choice for spam, sentiment, and
document classification, and a great baseline everywhere.
:::

::: tip
**Practical tips:** (1) Use **Multinomial NB** for text counts, **Gaussian NB** for
continuous features, **Bernoulli NB** for binary/presence features. (2) Keep Laplace
smoothing on (`alpha=1.0` default); tune `alpha` if needed. (3) Naive Bayes is so fast
it's perfect as a **baseline** and for **huge text datasets**. (4) It gives probabilities
but they're often **poorly calibrated** (too confident) — rank/threshold them with care.
(5) For text, also try TF-IDF features (Chapter 38) instead of raw counts.
:::

## Advantages, disadvantages, and use cases

| Advantages | Disadvantages |
|---|---|
| Extremely fast to train and predict | Independence assumption is usually false |
| Works great on high-dimensional text | Probabilities poorly calibrated |
| Needs little training data | Struggles when features strongly interact |
| Simple, few parameters | Gaussian NB assumes normal features |
| Strong baseline | Zero-frequency problem (fixed by smoothing) |

**Use cases:** spam filtering, sentiment analysis, document/topic classification, news
categorisation, medical diagnosis screening, and real-time classification where speed
matters.

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Using the wrong variant.** Multinomial NB for counts/text, Gaussian NB
for continuous features, Bernoulli NB for binary. Using Gaussian NB on word counts (or
vice versa) hurts performance.
:::

- **Mistake 2 — Forgetting smoothing**, causing zero probabilities to nullify a class.
- **Mistake 3 — Trusting the probability values** as well-calibrated (they're often
  over-confident).
- **Mistake 4 — Expecting it to capture feature interactions** (it assumes
  independence).
- **Mistake 5 — Dismissing it as "too simple"** — on text it often rivals far more
  complex models.

## Best practices

- **Match the variant to the feature type.**
- **Keep Laplace smoothing on.**
- **Use it as a fast baseline**, especially for text.
- **Prefer it for very high-dimensional sparse data** (like bag-of-words).
- **Be cautious interpreting its probabilities**; use ranking/thresholds thoughtfully.

## Chapter Summary

- **Naive Bayes** is a probabilistic classifier based on **Bayes' theorem** plus a
  **naive independence assumption** (features independent given the class).
- It picks the class maximising **`P(c) × ∏ P(xᵢ|c)`** — prior times the product of
  feature likelihoods.
- Variants: **Gaussian** (continuous), **Multinomial** (counts/text), **Bernoulli**
  (binary). **Laplace smoothing** prevents zero probabilities.
- It's **extremely fast**, needs little data, and excels on **high-dimensional text** —
  a spam classifier flagged "free money now" at 0.947 and Gaussian NB hit **0.978** on
  iris.
- Limitations: the false independence assumption, poorly calibrated probabilities, and
  weakness when features strongly interact — but it remains a superb baseline.

---

::: {.qband}
Practice Zone — Chapter 20
:::

## Multiple-Choice Questions (MCQs)

**Q1.** Naive Bayes is based on:
a) Gradient descent  b) Bayes' theorem  c) Distance metrics  d) Decision trees

**Q2.** The "naive" assumption is that features are:
a) Normally distributed  b) Independent given the class  c) Scaled  d) Correlated

**Q3.** For text/word-count features, use:
a) Gaussian NB  b) Multinomial NB  c) Bernoulli NB only  d) Linear NB

**Q4.** Laplace smoothing prevents:
a) Overfitting  b) Zero probabilities from unseen features  c) Scaling issues
d) Slow training

**Q5.** Naive Bayes predicts the class with the highest:
a) Distance  b) `P(c) × ∏ P(xᵢ|c)`  c) MSE  d) Variance

**Q6.** A key strength of Naive Bayes is:
a) Capturing feature interactions  b) Speed on high-dimensional text  c) Calibrated
probabilities  d) Modelling non-linearity

**Q7.** Gaussian NB assumes each feature is:
a) Binary  b) A word count  c) Normally distributed within each class  d) Categorical

**Q8.** Naive Bayes probabilities are often:
a) Perfectly calibrated  b) Poorly calibrated (over-confident)  c) Always 0.5  d) Negative

### MCQ Answers
**1:** b. **2:** b. **3:** b. **4:** b. **5:** b. **6:** b. **7:** c. **8:** b.

## Interview Questions (with answers)

**Q1. How does Naive Bayes classify?**
*Answer:* Using Bayes' theorem, it computes, for each class, the prior probability times
the product of the likelihoods of the observed features given that class
(`P(c)·∏P(xᵢ|c)`), then predicts the class with the highest such value. The
denominator P(features) is dropped since it's constant across classes.

**Q2. What is the "naive" assumption and why does the algorithm still work?**
*Answer:* It assumes all features are conditionally independent given the class, which
is usually false (features interact). It still works well because for choosing the
*most probable* class, the relative ranking is often correct even when the exact
probabilities are inaccurate — especially in high-dimensional text.

**Q3. What are the variants of Naive Bayes and when do you use each?**
*Answer:* Gaussian NB for continuous features (assumes normality), Multinomial NB for
count features like word counts (text), and Bernoulli NB for binary present/absent
features. Match the variant to the data type.

**Q4. What is Laplace smoothing and why is it needed?**
*Answer:* It adds a small constant (e.g. 1) to all feature counts so no likelihood is
zero. Without it, a feature value unseen for a class gives P=0, which zeros the entire
product and wrongly eliminates that class.

**Q5. Why is Naive Bayes popular for text classification?**
*Answer:* Text produces very high-dimensional, sparse count features; Naive Bayes is
extremely fast, needs little data, handles many features gracefully, and the
independence assumption, while wrong, is mild enough that it classifies text (spam,
sentiment, topics) very effectively.

## Scenario-Based Questions (with answers)

**Q1.** *You need a spam filter that trains in milliseconds on millions of emails and
predicts in real time. Which algorithm and variant, and why?*
*Answer:* Multinomial Naive Bayes on bag-of-words/TF-IDF features. It trains and
predicts extremely fast, scales to high-dimensional sparse text, needs little tuning,
and is a proven strong performer for spam — ideal for the speed and scale requirements.

**Q2.** *Your Naive Bayes model assigns probability 0 to a class whenever a new word
appears. What's wrong and how do you fix it?*
*Answer:* The zero-frequency problem: an unseen word has likelihood 0, zeroing the
product. Fix with Laplace smoothing (add-one), which scikit-learn applies by default
via `alpha`; ensure it's not set to 0.

**Q3.** *Two features in your data are strongly correlated, and Naive Bayes
underperforms a logistic regression. Why might that be?*
*Answer:* Naive Bayes assumes feature independence; strongly correlated features
violate this and get "double counted", distorting the probabilities. Logistic
regression can weight correlated features jointly, so it may handle the interaction
better here.

## Logic-Based Questions (with answers)

**Q1.** Why can a single zero likelihood eliminate a class entirely in Naive Bayes?
*Answer:* Because the class score is a *product* of likelihoods; multiplying by zero
makes the whole product zero regardless of the other (possibly strong) evidence — hence
the need for smoothing.

**Q2.** Naive Bayes' independence assumption is usually false, yet it classifies well.
What does this tell you about the relationship between accurate probabilities and
correct decisions?
*Answer:* That you don't need perfectly accurate probabilities to make the right
*decision* — you only need the correct class to have the highest score. Ranking can be
right even when the numbers are off.

**Q3.** Why is Naive Bayes especially suited to high-dimensional data compared to KNN?
*Answer:* Naive Bayes handles many features by simply multiplying per-feature
probabilities (fast, scales linearly), whereas KNN's distance-based approach degrades in
high dimensions (curse of dimensionality). Hence NB thrives on text where KNN struggles.

## Practical Questions (with answers)

**Q1.** Which Naive Bayes variant would you use for word-count features?
*Answer:* `MultinomialNB`.

**Q2.** Write code to convert a list of texts into a bag-of-words matrix.
*Answer:* `CountVectorizer().fit_transform(texts)`.

**Q3.** What does the `alpha` parameter control in scikit-learn's Naive Bayes?
*Answer:* The Laplace/Lidstone smoothing strength (`alpha=1.0` is add-one smoothing);
it prevents zero probabilities for unseen feature values.

## Long Questions (with answers)

**Q1. Explain the Naive Bayes algorithm in full: the underlying theorem, the naive
assumption, how it classifies, and the role of smoothing.**

*Answer:* Naive Bayes applies **Bayes' theorem** to find the most probable class given
the features. Bayes' theorem states P(c|x) = P(x|c)·P(c)/P(x). To classify, we want the
class c maximising P(c|x₁,…,xₙ); since P(x) is the same across classes, we maximise
P(c)·P(x₁,…,xₙ|c). Computing the joint likelihood of all features is hard, so Naive
Bayes makes the **naive assumption** that features are conditionally independent given
the class, which lets the joint likelihood factor into a simple product:
P(c)·∏P(xᵢ|c). The algorithm estimates the **prior** P(c) (class frequencies) and the
per-feature **likelihoods** P(xᵢ|c) from training data, computes the product for each
class, and predicts the class with the highest score. The independence assumption is
usually false, but the method still classifies well because the correct class often
still wins the comparison. A practical issue is the **zero-frequency problem**: a
feature value never seen with a class gives likelihood 0, which zeros the entire product
and wrongly eliminates that class. **Laplace smoothing** fixes this by adding a small
constant to all counts so no probability is exactly zero. Variants (Gaussian,
Multinomial, Bernoulli) adapt the likelihood estimate to continuous, count, or binary
features respectively.

**Q2. Discuss why Naive Bayes is so effective for text classification despite its
simplistic assumptions, and compare it with logistic regression for this task.**

*Answer:* Text classification produces extremely **high-dimensional, sparse** feature
vectors (one dimension per vocabulary word), often with more features than samples.
Naive Bayes thrives here for several reasons: it estimates each word's
class-conditional probability independently, so it scales linearly with vocabulary size
and trains and predicts in milliseconds; it needs little data per feature; and it is
robust to the many irrelevant words because their likelihoods are similar across
classes. Although its **independence assumption** is clearly violated (words co-occur,
e.g. "free" and "money" in spam), this rarely flips the *ranking* of classes, so
classification accuracy stays high even when the probability estimates are inexact —
which is why spam filters historically relied on it. Compared with **logistic
regression**, which is also excellent for text: logistic regression is *discriminative*
(it directly models the decision boundary and can down-weight correlated/irrelevant
features through learned coefficients), often achieving slightly higher accuracy and
better-calibrated probabilities, but it is slower to train and needs more data. Naive
Bayes is *generative*, faster, and works with tiny datasets, making it the better choice
when speed, scale, or limited data dominate, and an ideal baseline. In practice both are
quick to try, and you compare them empirically (No Free Lunch, Chapter 16).

## Exercises

1. State Bayes' theorem and label each part (prior, likelihood, posterior).
2. Explain the naive independence assumption with a spam example.
3. Choose the variant (Gaussian/Multinomial/Bernoulli) for: word counts, sensor
   temperatures, word present/absent flags.
4. Explain why a zero likelihood is dangerous and how smoothing helps.
5. Give three reasons Naive Bayes suits text classification.

## Mini-Project

**Project: Build a spam/sentiment classifier.**

1. Get a small text dataset (e.g. SMS spam, or movie review snippets you label).
2. Vectorise with `CountVectorizer` (then try TF-IDF), train `MultinomialNB`.
3. Report accuracy, precision, and recall on a held-out set.
4. Print the most "spammy" words (highest `P(word|spam)` vs `P(word|ham)`).
5. Compare against logistic regression on the same features. Save in `my-ml-journey/`.

## Assignments

1. **Coding:** Implement Gaussian Naive Bayes from scratch for one feature (estimate
   per-class mean/variance, apply the normal pdf) and verify against scikit-learn.
2. **Coding:** On a text dataset, compare `MultinomialNB` with and without smoothing
   (`alpha=0` vs `alpha=1`) and explain the difference.
3. **Conceptual:** Write one page explaining why the "naive" assumption is wrong yet the
   algorithm works, and when it would fail.

::: tip
Naive Bayes decides by probability under independence. Chapter 21, **Decision Trees**,
takes a totally different, rule-based approach — asking a series of yes/no questions —
that is highly interpretable and forms the building block of the powerful ensembles in
Chapters 23–24.
:::
