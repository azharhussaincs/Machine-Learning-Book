# Probability and Statistics for Machine Learning

## Introduction

If linear algebra and calculus (Chapter 5) are the *engine* of Machine Learning,
then **statistics and probability** are the *eyes and judgement*. They let us
understand our data, measure uncertainty, separate real patterns from random luck,
and reason about what a model's predictions actually *mean*.

Here is a comforting truth: **Machine Learning is, at its core, applied
statistics** running on fast computers. Every time a model says "this email is 92%
likely to be spam," that "92%" is probability. Every time you check whether your
new model is *really* better than the old one, that's a hypothesis test.

This chapter teaches the essential statistical ideas every ML practitioner needs —
in plain English, with pictures and code. We split it into three parts:

1. **Descriptive statistics** — summarising and understanding data.
2. **Probability** — the maths of uncertainty (including Bayes' theorem).
3. **Inferential statistics** — drawing reliable conclusions from samples.

::: keyidea
You will hear "the data is normally distributed," "these features are correlated,"
"the result is statistically significant," and "the model is 80% confident."
By the end of this chapter, every one of those phrases will be clear and usable.
:::

---

# Part A — Descriptive Statistics

## Measures of central tendency: where is the "middle"?

When you have a column of numbers, the first question is "what's a typical value?"
There are three answers.

- **Mean (average)** — add all values, divide by the count.

<div class="equation"><img class="eq" src="assets/images/eq_ch06_mean.png" alt="mean"></div>

- **Median** — the *middle* value when the data is sorted (half are below, half
  above).
- **Mode** — the value that appears *most often*.

::: warning
**The mean can lie.** Imagine salaries: 30k, 32k, 35k, 33k, and 5,000k (a CEO). The
**mean** is over 1,000k — but that describes *nobody*. The **median** (33k) is far
more honest. **Rule:** when data has extreme values (outliers) or is skewed, prefer
the median. This is why "median household income" is reported, not the mean.
:::

### Example

For the data `[2, 4, 4, 6, 9]`:

- Mean = (2+4+4+6+9)/5 = 25/5 = **5**
- Median = middle of sorted list = **4**
- Mode = most frequent = **4**

## Measures of spread: how scattered is the data?

Two datasets can have the same mean but look completely different. Spread tells us
how "tight" or "loose" the data is around the middle.

- **Range** = maximum − minimum (simple, but sensitive to outliers).
- **Variance (σ²)** — the average of the squared distances from the mean:

<div class="equation"><img class="eq" src="assets/images/eq_ch06_variance.png" alt="variance"></div>

- **Standard deviation (σ)** — the square root of variance. We take the square root
  so the spread is back in the *original units* (e.g. dollars, not dollars²). This
  is the **most used** measure of spread.

<div class="equation"><img class="eq" src="assets/images/eq_ch06_std.png" alt="standard deviation"></div>

::: note
**Why squared distances?** Squaring makes all distances positive (so they don't
cancel) and punishes far-away points more. Notice this is the *same idea* as MSE in
Chapter 5 — statistics and ML loss functions are deeply connected.
:::

## Percentiles, quartiles, and the IQR

- A **percentile** tells you the value below which a given percent of data falls.
  The 90th percentile is the value below which 90% of the data lies.
- **Quartiles** split sorted data into four equal parts: Q1 (25%), Q2 (50% = the
  median), Q3 (75%).
- The **Interquartile Range (IQR)** = Q3 − Q1 — the spread of the *middle half* of
  the data. It is **robust to outliers**, which makes it great for detecting them
  (a common rule: anything beyond Q1 − 1.5·IQR or Q3 + 1.5·IQR is an outlier).

## Skewness: is the data lopsided?

![Three shapes of data. Left-skewed (tail on the left), symmetric (balanced, mean ≈ median), and right-skewed (tail on the right). In a right-skewed distribution the mean is pulled above the median.](assets/images/ch06_skewness.png)

- **Symmetric** — balanced; mean ≈ median (e.g. heights).
- **Right-skewed (positive)** — a long tail to the right; mean > median (e.g.
  income, house prices).
- **Left-skewed (negative)** — a long tail to the left; mean < median.

Knowing skew matters because many models and statistics assume roughly symmetric
data, and skew often signals you should transform a feature (Chapter 12).

---

# Part B — Probability

## What is probability?

**Probability** is a number between 0 and 1 measuring how likely an event is.
0 means impossible, 1 means certain, 0.5 means "fifty-fifty."

- P(heads) on a fair coin = 0.5.
- P(rolling a 6) on a fair die = 1/6 ≈ 0.167.

We often write probabilities as percentages (0.92 = 92%). ML models constantly
output probabilities ("this image is 0.87 cat, 0.13 dog").

## The basic rules

- **Range:** every probability is between 0 and 1.
- **Complement:** P(not A) = 1 − P(A). If rain is 30% likely, no-rain is 70%.
- **Addition (for mutually exclusive events):** P(A or B) = P(A) + P(B).
- **Multiplication (for independent events):** P(A and B) = P(A) × P(B). Two coins
  both heads: 0.5 × 0.5 = 0.25.

## Conditional probability: probability *given* something

**Conditional probability** P(A | B) reads "the probability of A *given that* B has
happened." Knowing B can change the odds of A.

<div class="equation"><img class="eq" src="assets/images/eq_ch06_conditional.png" alt="conditional probability"></div>

*Example:* P(disease) might be 1%. But P(disease | positive test) — the probability
*given* a positive test — could be much higher. The new information changes
everything.

## Bayes' Theorem: updating beliefs with evidence

This is one of the most important formulas in all of Machine Learning. **Bayes'
theorem** tells us how to *update* a probability when we get new evidence:

<div class="equation"><img class="eq" src="assets/images/eq_ch06_bayes.png" alt="Bayes theorem"></div>

In words: **posterior = (likelihood × prior) / evidence.**

- **Prior** P(A) — what we believed before seeing evidence.
- **Likelihood** P(B|A) — how likely the evidence is, if A is true.
- **Posterior** P(A|B) — our updated belief after seeing evidence B.

### A famous, eye-opening example

A disease affects **1%** of people. A test is **99% accurate** (correctly flags 99%
of sick people, and is wrong for 1% of healthy people). You test **positive**.
What's the chance you're actually sick? Most people guess 99%. The real answer is
about **50%**. Let's see why with Bayes:

```text
P(sick) = 0.01,  P(healthy) = 0.99
P(positive | sick) = 0.99
P(positive | healthy) = 0.01   (false positive rate)

P(positive) = 0.99×0.01 + 0.01×0.99 = 0.0099 + 0.0099 = 0.0198

P(sick | positive) = (0.99 × 0.01) / 0.0198 = 0.0099 / 0.0198 = 0.5  (50%)
```

::: keyidea
Because the disease is *rare*, even a good test produces many false positives among
the huge healthy population. This is the **base rate** lesson — and it is exactly
how the **Naive Bayes** classifier (Chapter 20) works: combine prior beliefs with
evidence to update probabilities.
:::

## Common probability distributions

A **distribution** describes how likely each value is. A few show up everywhere:

- **Normal (Gaussian)** — the famous "bell curve." Symmetric around the mean. Heights,
  measurement errors, and many natural quantities follow it. Defined by its mean μ
  and standard deviation σ.
- **Uniform** — every value equally likely (a fair die).
- **Binomial** — number of successes in n yes/no trials (e.g. heads in 10 flips).
- **Poisson** — counts of rare events in a fixed period (e.g. emails per hour).

### The normal distribution and the 68–95–99.7 rule

![The normal (bell) curve and the empirical rule: about 68% of data lies within 1 standard deviation of the mean, 95% within 2, and 99.7% within 3.](assets/images/ch06_normal.png)

The normal distribution's probability density is:

<div class="equation"><img class="eq" src="assets/images/eq_ch06_normal_pdf.png" alt="normal pdf"></div>

You won't compute this by hand, but the **68–95–99.7 rule** is essential everyday
knowledge:

- ~**68%** of values fall within **1σ** of the mean.
- ~**95%** within **2σ**.
- ~**99.7%** within **3σ**.

This is why "3-sigma event" means "very rare." It also underlies outlier detection
and how we read model uncertainty.

### The z-score: how unusual is a value?

The **z-score** rescales a value to "how many standard deviations from the mean" it
is — letting you compare apples to oranges:

<div class="equation"><img class="eq" src="assets/images/eq_ch06_zscore.png" alt="z-score"></div>

A z-score of 0 is exactly average; +2 means "2 standard deviations above average"
(unusually high). Standardising features with z-scores (Chapter 11) is one of the
most common preprocessing steps in ML.

### The Central Limit Theorem (CLT)

![The Central Limit Theorem: even when the original data is not normal (left), the distribution of sample *means* becomes bell-shaped as the sample size grows (right). This is why the normal distribution appears everywhere.](assets/images/ch06_clt.png)

The **Central Limit Theorem** says: *if you take many samples and average each one,
those averages form a normal distribution — even if the original data was not
normal.* This is one of the deepest results in statistics and explains why the bell
curve appears so often, and why so many statistical methods work.

---

# Part C — Correlation, Sampling, and Inference

## Correlation: do two things move together?

**Correlation** measures how strongly two variables move together. The **Pearson
correlation coefficient (r)** ranges from −1 to +1:

<div class="equation"><img class="eq" src="assets/images/eq_ch06_correlation.png" alt="correlation coefficient"></div>

- **r = +1** — perfect positive (as one rises, so does the other).
- **r = −1** — perfect negative (as one rises, the other falls).
- **r = 0** — no linear relationship.

![Scatter plots showing strong positive correlation (r ≈ +0.9), no correlation (r ≈ 0), and strong negative correlation (r ≈ −0.9).](assets/images/ch06_correlation.png)

::: warning
**Correlation is NOT causation.** Ice-cream sales and drowning deaths are
correlated — but ice cream doesn't cause drowning. A third factor (hot weather)
drives both. This is one of the most important and most violated rules in all of
data analysis. Always ask: *could a hidden third factor explain this?*
:::

## Population vs sample

- A **population** is *everyone/everything* you care about (e.g. all customers).
- A **sample** is a *subset* you actually measure (e.g. 1,000 surveyed customers).

We almost never have the whole population, so we **infer** facts about the
population from a sample. Good sampling must be **representative** (e.g. **random
sampling**) — a biased sample gives biased conclusions, no matter how big.

## Inferential statistics: drawing conclusions

### Hypothesis testing

Hypothesis testing answers: *"Is this result real, or just random luck?"* It is how
you decide whether a new model, drug, or website really is better.

- The **null hypothesis (H₀)** is the boring default: "there is no real effect /
  no difference."
- The **alternative hypothesis (H₁)** is what you suspect: "there *is* an effect."
- The **p-value** is the probability of seeing your result (or more extreme) *if the
  null hypothesis were true*. A **small p-value (commonly < 0.05)** means the result
  is unlikely to be mere luck, so we **reject the null** and call the result
  **statistically significant**.

::: warning
**The p-value is widely misunderstood.** It is **not** the probability that your
hypothesis is true. It is the probability of your data *assuming the null is true*.
A p-value of 0.03 does **not** mean "97% chance my idea is correct."
:::

### Confidence intervals

A **confidence interval** gives a *range* instead of a single estimate. "The average
height is 170 cm ± 3 cm with 95% confidence" means: if we repeated the study many
times, about 95% of such intervals would contain the true average. Ranges are more
honest than single numbers because they show uncertainty.

### Type I and Type II errors

| | H₀ is actually TRUE | H₀ is actually FALSE |
|---|---|---|
| **We reject H₀** | Type I error (false positive) ✗ | Correct ✓ |
| **We keep H₀** | Correct ✓ | Type II error (false negative) ✗ |

- **Type I error (false positive):** crying wolf — claiming an effect that isn't
  real.
- **Type II error (false negative):** missing a real effect.

These map directly onto ML classification errors (Chapter 25): a spam filter
marking good mail as spam is a false positive; letting spam through is a false
negative.

## Practical: statistics with NumPy, Pandas, and SciPy

Let's compute the core statistics on real-ish data and run a hypothesis test.

```python
import numpy as np
import pandas as pd
from scipy import stats

# --- Some sample exam scores ---
scores = np.array([55, 62, 68, 71, 75, 75, 78, 82, 88, 95])

# --- Descriptive statistics ---
print("Mean   :", np.mean(scores))                 # average
print("Median :", np.median(scores))               # middle value
print("Std dev:", round(np.std(scores), 2))        # spread (population)
print("Q1, Q3 :", np.percentile(scores, [25, 75])) # quartiles
print("IQR    :", stats.iqr(scores))               # Q3 - Q1

# --- z-scores: how unusual is each score? ---
z = (scores - scores.mean()) / scores.std()
print("z of 95:", round(z[-1], 2))                 # the top score, standardised

# --- Correlation between two variables (study hours vs score) ---
hours = np.array([2, 3, 4, 4, 5, 5, 6, 7, 8, 9])
r, p = stats.pearsonr(hours, scores)
print(f"Correlation r = {r:.3f}  (p = {p:.4f})")

# --- Hypothesis test: is the class mean different from 70? ---
t_stat, p_val = stats.ttest_1samp(scores, popmean=70)
print(f"t-test vs 70: p = {p_val:.3f}",
      "-> significant" if p_val < 0.05 else "-> not significant")
```

**Output:**
```text
Mean   : 74.9
Median : 75.0
Std dev: 11.23
Q1, Q3 : [68.75 81.  ]
IQR    : 12.25
z of 95: 1.79
Correlation r = 0.989  (p = 0.0000)
t-test vs 70: p = 0.223 -> not significant
```

### Explanation

- **Mean vs median** are close (74.9 vs 75.0), so the data is fairly symmetric.
- **Std dev ≈ 11.2** means scores typically sit about 11 points from the mean.
- **The z-score of 95 is 1.79** — the top score is about 1.8 standard deviations
  above average: high, but not extreme (within 2σ).
- **Correlation r = 0.989** between study hours and scores is very strong and
  positive — more hours strongly associate with higher scores. The tiny p-value
  says this is very unlikely to be random luck.
- **The t-test** asks "is the true mean different from 70?" The p-value 0.223 is
  *above* 0.05, so we do **not** have enough evidence — the class mean is *not*
  significantly different from 70.

::: keyidea
Notice we did real **inference**: from 10 numbers we measured relationships and
tested a claim, while honestly reporting uncertainty. This mindset — *"is this real
or luck?"* — protects you from fooling yourself, the most important skill in data
work.
:::

::: tip
**Debugging/usage tips:** (1) `np.std` uses the *population* formula (divides by n);
`np.std(x, ddof=1)` gives the *sample* version (divides by n−1) used for samples.
(2) `pearsonr` only detects *linear* correlation — two variables can be strongly
related in a curved way with r ≈ 0. Always plot! (3) Install SciPy with
`pip install scipy` if needed.
:::

## Why each idea matters in Machine Learning

| Statistics idea | Where it powers ML |
|---|---|
| Mean / std / standardisation | Feature scaling (Ch 11), normalisation |
| Median / IQR | Robust stats, outlier detection (Ch 10) |
| Skewness | Deciding when to transform features (Ch 12) |
| Normal distribution | Many model assumptions, anomaly detection |
| Bayes' theorem | Naive Bayes classifier (Ch 20), Bayesian methods |
| Correlation | Feature selection (Ch 13), EDA (Ch 15) |
| Sampling | Train/test splits, cross-validation (Ch 25) |
| Hypothesis testing | Comparing models, A/B testing |
| Type I / II errors | Precision, recall, the confusion matrix (Ch 25) |

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Reporting the mean for skewed data.** Use the median for income,
prices, and anything with a long tail.
:::

- **Mistake 2 — Confusing correlation with causation.** The classic, costly error.
- **Mistake 3 — Misreading the p-value** as "probability my hypothesis is true."
- **Mistake 4 — Trusting a big but biased sample.** Size doesn't fix bias.
- **Mistake 5 — Ignoring the base rate** (the disease-test example) — rare events
  produce many false positives.
- **Mistake 6 — Assuming data is normal** without checking (plot a histogram first).

## Best practices

- **Always look at the distribution** (histogram/box plot) before summarising.
- **Report spread, not just the average** — a mean without a standard deviation is
  half a story.
- **Prefer robust statistics** (median, IQR) when outliers are present.
- **Plot relationships** before trusting a correlation number.
- **State your hypotheses and significance level *before* looking at results** to
  avoid fooling yourself.
- **Quantify uncertainty** with intervals, not just point estimates.

## Chapter Summary

- **Central tendency:** mean (average, sensitive to outliers), median (robust
  middle), mode (most frequent).
- **Spread:** variance and **standard deviation** (most used), range, and the
  outlier-robust **IQR**.
- **Skewness** describes lopsided data; prefer the median when skewed.
- **Probability** measures uncertainty (0–1); key tools are conditional
  probability and **Bayes' theorem** (posterior ∝ likelihood × prior).
- The **normal distribution** and the **68–95–99.7 rule**, **z-scores**, and the
  **Central Limit Theorem** explain why bell curves are everywhere.
- **Correlation (r ∈ [−1, 1])** measures linear association — but **correlation is
  not causation**.
- **Inference:** we generalise from **samples** to **populations** using
  **hypothesis tests** (p-values, significance), **confidence intervals**, and we
  watch out for **Type I/II errors**.

---

::: {.qband}
Practice Zone — Chapter 6
:::

## Multiple-Choice Questions (MCQs)

**Q1.** Which measure of central tendency is most robust to outliers?
a) Mean  b) Median  c) Mode  d) Range

**Q2.** Standard deviation is the:
a) Square of the variance  b) Square root of the variance  c) Mean of the data
d) Middle value

**Q3.** In the 68–95–99.7 rule, roughly what % of data lies within 2 standard
deviations of the mean?
a) 50%  b) 68%  c) 95%  d) 99.7%

**Q4.** A correlation coefficient of −0.9 indicates:
a) No relationship  b) Strong positive  c) Strong negative  d) Causation

**Q5.** Bayes' theorem updates the:
a) Prior into the posterior using evidence  b) Mean into the median
c) Variance into the std  d) Sample into the population

**Q6.** A p-value of 0.03 (with threshold 0.05) means we:
a) Accept the null  b) Reject the null (significant)  c) Prove H₁ is true
d) Made a Type II error

**Q7.** Marking a legitimate email as spam is a:
a) Type I error (false positive)  b) Type II error (false negative)
c) Correct decision  d) Bias

**Q8.** A z-score of +2 means the value is:
a) The mean  b) 2 units above the mean  c) 2 standard deviations above the mean
d) Impossible

### MCQ Answers
**1:** b. **2:** b. **3:** c. **4:** c. **5:** a. **6:** b. **7:** a. **8:** c.

## Interview Questions (with answers)

**Q1. What is the difference between mean and median, and when do you use each?**
*Answer:* The mean is the arithmetic average; the median is the middle value of
sorted data. Use the mean for roughly symmetric data without outliers; use the
median for skewed data or data with outliers (e.g. income, house prices), because
the median is not dragged by extreme values.

**Q2. Explain Bayes' theorem and why it matters in ML.**
*Answer:* Bayes' theorem updates a prior belief into a posterior belief using
evidence: P(A|B) = P(B|A)·P(A) / P(B). It matters because it formalises learning
from evidence and underlies the Naive Bayes classifier and Bayesian methods, and it
explains base-rate effects (why rare conditions yield many false positives).

**Q3. What does a p-value actually mean?**
*Answer:* It's the probability of observing data as extreme as ours *assuming the
null hypothesis is true*. A small p-value means such data would be unlikely under
the null, so we reject the null. It is **not** the probability that the hypothesis
is true.

**Q4. Why is "correlation does not imply causation" important?**
*Answer:* Two variables can move together due to coincidence or a hidden common
cause (confounder), without one causing the other. Acting on correlation as if it
were causation leads to wrong decisions; establishing causation generally requires
controlled experiments.

**Q5. What is the Central Limit Theorem and why is it useful?**
*Answer:* It states that the distribution of sample means approaches a normal
distribution as sample size grows, regardless of the population's distribution. It's
useful because it justifies using normal-based methods (confidence intervals,
many tests) even when the underlying data isn't normal.

## Scenario-Based Questions (with answers)

**Q1.** *A report says "our average customer spends \$500/month" and recommends
big spending on premium features. You find most customers spend \$50 but a few spend
\$20,000. What's wrong, and what would you report?*
*Answer:* The mean is distorted by a few huge spenders (right-skewed data). The
*median* (~\$50) describes the typical customer far better. I'd report the median
plus the distribution, and note the small high-value segment separately.

**Q2.** *Your A/B test shows the new website has a higher conversion rate, p = 0.40.
Marketing wants to roll it out. What do you say?*
*Answer:* p = 0.40 is far above 0.05, so the difference is not statistically
significant — it could easily be random chance. I'd recommend not rolling out yet,
collecting more data, or re-testing, rather than acting on noise.

**Q3.** *A medical test is 99% accurate and a patient tests positive for a disease
that affects 1 in 1000 people. The patient panics, sure they have it. How do you
explain the real risk?*
*Answer:* Using Bayes and the base rate: among 1000 people, ~1 is truly sick (likely
detected), but ~10 healthy people falsely test positive (1% of 999). So a positive
result means only about 1-in-11 (~9%) chance of actually being sick. The rarity of
the disease makes false positives dominate; confirmatory testing is needed.

## Logic-Based Questions (with answers)

**Q1.** Two datasets have the same mean but different standard deviations. What can
you conclude about their shapes?
*Answer:* They have the same centre but different spreads — one is more tightly
clustered around the mean, the other more spread out. The mean alone cannot
distinguish them, which is why spread must always be reported too.

**Q2.** If P(A and B) = P(A)·P(B), what does that tell you about A and B?
*Answer:* That A and B are **independent** — knowing one occurred doesn't change the
probability of the other. (This is exactly the "naive" assumption in Naive Bayes.)

**Q3.** A study with a tiny p-value (0.001) uses a sample that was not random
(only volunteers). Is the conclusion trustworthy?
*Answer:* Not necessarily. A small p-value addresses random chance, but a biased
(non-random) sample can make the whole estimate systematically wrong. Statistical
significance cannot rescue a biased sample.

## Practical Questions (with answers)

**Q1.** In the code, why might `np.std(scores)` differ from a value computed with
`ddof=1`?
*Answer:* `np.std` defaults to the population standard deviation (divides by n).
`ddof=1` gives the sample standard deviation (divides by n−1, Bessel's correction),
which is the unbiased estimate when the data is a sample of a larger population.

**Q2.** The correlation between hours and scores was r = 0.989 with a tiny p-value.
What do both numbers together tell you?
*Answer:* r = 0.989 means a very strong positive linear relationship; the tiny
p-value means this correlation is highly unlikely to have arisen by chance. Together:
study hours and scores move together strongly and reliably (though this is
association, not proof of causation).

**Q3.** Write one line to find the value below which 90% of `scores` fall.
*Answer:* `np.percentile(scores, 90)`.

## Long Questions (with answers)

**Q1. Explain Bayes' theorem fully using the disease-testing example, and state the
general lesson it teaches for interpreting positive results of rare conditions.**

*Answer:* Bayes' theorem is P(A|B) = P(B|A)·P(A) / P(B), updating a prior P(A) into
a posterior P(A|B) using evidence B. In the disease example, let A = "sick" with
prior P(sick) = 0.01, and B = "positive test." The test detects sick people well,
P(positive|sick) = 0.99, but also has a 1% false-positive rate,
P(positive|healthy) = 0.01. The evidence probability is
P(positive) = P(positive|sick)·P(sick) + P(positive|healthy)·P(healthy)
= 0.99·0.01 + 0.01·0.99 = 0.0198. Then
P(sick|positive) = (0.99·0.01)/0.0198 = 0.5, i.e. 50%. Despite a "99% accurate"
test, a positive result means only a 50% chance of being sick. The lesson is the
**base-rate effect**: when a condition is rare, the large healthy population
generates many false positives that can equal or outnumber the true positives, so a
single positive test for a rare condition is far less conclusive than intuition
suggests. This is why doctors confirm with further tests, and why ML models for rare
events must be evaluated with precision/recall, not raw accuracy.

**Q2. Compare descriptive and inferential statistics, giving the purpose, tools, and
a real example of each, and explain how they work together in a Machine Learning
project.**

*Answer:* **Descriptive statistics** summarise and describe the data you actually
have. Their purpose is understanding: tools include the mean, median, mode, standard
deviation, IQR, skewness, histograms, and box plots. Example: computing that your
customers' median monthly spend is \$50 with an IQR of \$30. **Inferential
statistics** use a sample to draw conclusions about a larger population you can't
fully measure. Their purpose is generalisation and decision-making under
uncertainty: tools include sampling, confidence intervals, hypothesis tests
(p-values), and the Central Limit Theorem. Example: testing whether a new feature
significantly increased spend across *all* customers based on a sample. In an ML
project they work together: you begin with **descriptive** statistics and
visualisation during exploratory data analysis (Chapter 15) to understand and clean
the data; you use **inferential** thinking to split data into train/test sets fairly
(sampling), to judge whether one model is *significantly* better than another, and
to reason honestly about uncertainty in your model's performance. Descriptive
statistics tell you what your data *is*; inferential statistics tell you what you can
*reliably conclude* from it.

## Exercises

1. For `[3, 7, 7, 2, 9, 7, 4]`, compute the mean, median, and mode by hand.
2. Two classes have mean 70. Class A has σ = 2, Class B has σ = 20. Describe how
   their score distributions differ.
3. A fair coin is flipped 3 times. What is P(all three heads)? Show your working.
4. Explain in your own words why "correlation ≠ causation," with a fresh example.
5. A value has a z-score of −3. Is it common or rare? Roughly what percentile?

## Mini-Project

**Project: Explore a distribution.**

1. Pick any numeric column from a dataset you like (or generate 200 random salaries
   that are right-skewed).
2. Compute and report the mean, median, std, and IQR. Comment on whether mean ≈
   median and what that says about skew.
3. Plot a histogram and a box plot (matplotlib/seaborn). Mark any outliers using
   the 1.5×IQR rule.
4. Write a short paragraph: which single number best describes a "typical" value
   here, and why?

## Assignments

1. **By hand + code:** Work the disease-test Bayes example for a disease affecting
   5% of people with a 95%-accurate test. Compute P(sick | positive) by hand, then
   verify with Python.
2. **Coding:** Demonstrate the Central Limit Theorem: draw 1000 samples (size 30
   each) from a *non-normal* distribution (e.g. `np.random.exponential`), take each
   sample's mean, and plot a histogram of those means. Confirm it looks bell-shaped.
3. **Conceptual:** Write one page explaining the difference between a Type I and a
   Type II error, with a real example where each would be more costly than the
   other.

::: tip
Statistics is the "lie detector" of data science. Whenever a result seems
surprising, ask the three questions from this chapter: *Is the sample
representative? Could this be random luck? Is a hidden factor driving it?*
:::
