# Association Rule Learning

## Introduction

Have you ever noticed online stores saying *"Customers who bought this also bought…"*?
Or heard the famous (possibly mythical) retail story that men who buy diapers often also
buy beer? That's **association rule learning** — an unsupervised technique for
discovering **items that frequently occur together** in data.

Its classic application is **market-basket analysis**: finding which products are bought
together so a store can place them nearby, bundle them, or recommend them. But the same
idea applies to web-page navigation, medical symptom co-occurrence, and more.

::: keyidea
Association rule learning finds patterns of the form **"if A, then B"** (e.g. *if bread
and butter, then milk*). It's unsupervised — there's no target to predict. The art is
measuring *how meaningful* a rule is, using three metrics: **support, confidence, and
lift**.
:::

By the end of this chapter you will be able to:

- Understand **itemsets** and association rules.
- Compute and interpret **support**, **confidence**, and **lift**.
- Understand the **Apriori** and **FP-Growth** algorithms.
- Apply association mining and avoid common misinterpretations.

## Key concepts and metrics

An **itemset** is a group of items (e.g. {bread, butter}). A **rule** `A → B` says "items
in A tend to appear with items in B". We judge rules with three metrics.

![Market-basket analysis finds rules like {bread, butter} → {milk}. Support measures how often the items appear, confidence how reliable the rule is, and lift whether the items appear together more than chance.](assets/images/ch29_market_basket.png)

**Support** — how *frequent* an itemset is (its popularity):

<div class="equation"><img class="eq" src="assets/images/eq_ch29_support.png" alt="support"></div>

**Confidence** — how *reliable* the rule is: given A, how often does B also appear?

<div class="equation"><img class="eq" src="assets/images/eq_ch29_confidence.png" alt="confidence"></div>

**Lift** — the *most important* metric: how much more often A and B occur together than
if they were **independent**. Lift > 1 means a *positive* association (they attract); = 1
means independent; < 1 means they *avoid* each other.

<div class="equation"><img class="eq" src="assets/images/eq_ch29_lift.png" alt="lift"></div>

::: warning
**Confidence can mislead; lift corrects it.** If 90% of *all* customers buy milk, then a
rule "bread → milk" with 90% confidence is *worthless* — milk is just popular, not
associated with bread. **Lift** accounts for B's overall popularity, so lift > 1 is the
real signal of a meaningful association. Always check lift, not just confidence.
:::

## Practical: computing the metrics

Let's mine a tiny shopping dataset by hand (no special library — just sets and Python).

```python
transactions = [
    {"bread", "butter", "milk"}, {"bread", "butter"}, {"bread", "milk"},
    {"bread", "butter", "milk", "jam"}, {"milk", "jam"}, {"bread", "butter", "jam"},
    {"bread", "butter", "milk"}, {"butter", "milk"},
]
n = len(transactions)

def support(items):       # fraction of transactions containing all of `items`
    return sum(1 for t in transactions if items <= t) / n
def confidence(a, b):     # support(A and B) / support(A)
    return support(a | b) / support(a)
def lift(a, b):           # confidence / support(B)
    return confidence(a, b) / support(b)

print("support(bread):", round(support({"bread"}), 3))
print("support(bread, butter):", round(support({"bread", "butter"}), 3))
print("confidence(bread -> butter):", round(confidence({"bread"}, {"butter"}), 3))
print("lift(bread -> butter):", round(lift({"bread"}, {"butter"}), 3))
print("confidence(jam -> milk):", round(confidence({"jam"}, {"milk"}), 3))
print("lift(jam -> milk):", round(lift({"jam"}, {"milk"}), 3))
```

**Output:**
```text
support(bread): 0.75
support(bread, butter): 0.625
confidence(bread -> butter): 0.833
lift(bread -> butter): 1.111
confidence(jam -> milk): 0.667
lift(jam -> milk): 0.889
```

### Explanation

- **bread → butter:** confidence 0.833 (83% of bread-buyers also buy butter) and **lift
  1.111 (> 1)** → a **genuine positive association**. Place them together!
- **jam → milk:** confidence 0.667 sounds decent, but **lift 0.889 (< 1)** → they actually
  appear together *less* than chance. The confidence was high only because milk is
  common; lift reveals the truth. **This is exactly why lift matters.**

::: keyidea
Notice how lift changed the story. Confidence alone made jam→milk look like a real
pattern; lift exposed it as an illusion driven by milk's popularity. When mining rules,
**rank by lift** (and require a minimum support so rules aren't based on too few
transactions).
:::

## The Apriori algorithm

Checking every possible itemset is explosively expensive (with 1,000 products there are
astronomically many combinations). The **Apriori algorithm** makes it tractable using a
clever principle:

::: note
**The Apriori principle:** *if an itemset is infrequent, all of its supersets are also
infrequent.* So once {bread, eggs} is rare, we don't bother checking {bread, eggs, milk}
— it can't be more frequent. This **pruning** dramatically reduces the search.
:::

Apriori works bottom-up: find frequent single items (above a minimum support), then
frequent pairs, then triples, etc., pruning infrequent branches at each level. From the
frequent itemsets, it generates rules and keeps those above a minimum confidence/lift.

## FP-Growth

**FP-Growth** is a faster modern alternative that builds a compact tree (an "FP-tree")
of the transactions and mines frequent itemsets from it **without generating all
candidates**, making it much more efficient than Apriori on large datasets. (Libraries
like `mlxtend` provide both: `pip install mlxtend`.)

::: tip
**Practical & debugging tips:** (1) Set a sensible **minimum support** to avoid rules
based on a handful of transactions (which look strong but are noise). (2) **Rank rules by
lift**, then confidence; ignore high-confidence/low-lift rules. (3) For real data, use
`mlxtend.frequent_patterns` (`apriori`, `association_rules`) with one-hot-encoded
transactions. (4) Watch the explosion of rules — filter by min support/confidence/lift and
focus on actionable ones. (5) Remember association ≠ causation (Chapter 6).
:::

## Advantages, disadvantages, and use cases

| Advantages | Disadvantages |
|---|---|
| Finds interpretable "if-then" patterns | Can generate a huge number of rules |
| Unsupervised — no labels needed | Computationally expensive (Apriori) |
| Useful, actionable for business | Association ≠ causation |
| Simple, intuitive metrics | Rare-but-important items may be missed |

**Use cases:** market-basket analysis (product placement, bundling, cross-selling),
recommendation systems, website click-path analysis, medical symptom/diagnosis
co-occurrence, and fraud-pattern discovery.

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Using confidence alone.** A high-confidence rule can be meaningless if the
consequent is just popular. Always check **lift**.
:::

- **Mistake 2 — Confusing association with causation** — co-occurrence doesn't mean one
  causes the other.
- **Mistake 3 — Setting min support too low**, producing countless noisy rules from rare
  combinations.
- **Mistake 4 — Drowning in rules** without filtering by support/confidence/lift.
- **Mistake 5 — Ignoring rare but valuable items** (high support bias toward common
  items).

## Best practices

- **Rank by lift**, with minimum support and confidence thresholds.
- **Set thresholds thoughtfully** to balance noise vs missing patterns.
- **Use FP-Growth** for large datasets.
- **Filter to actionable rules** and interpret them in business terms.
- **Remember association is not causation.**

## Chapter Summary

- **Association rule learning** is unsupervised mining of **"items that go together"** —
  classic for **market-basket analysis** (rules like {bread, butter} → {milk}).
- Three metrics: **support** (how frequent), **confidence** (how reliable the rule), and
  **lift** (how much more than chance — **the key metric**; > 1 = positive association).
- **Confidence can mislead** when the consequent is popular; **lift corrects this** (jam→
  milk looked good by confidence but had lift < 1).
- **Apriori** uses the principle that supersets of infrequent itemsets are infrequent to
  prune the search; **FP-Growth** is a faster alternative for big data.
- Rank rules by lift with sensible support thresholds, filter to actionable rules, and
  remember **association ≠ causation**.

---

::: {.qband}
Practice Zone — Chapter 29
:::

## Multiple-Choice Questions (MCQs)

**Q1.** Association rule learning is mainly used for:
a) Classification  b) Finding items that occur together  c) Regression  d) Image
recognition

**Q2.** Support measures an itemset's:
a) Reliability  b) Frequency/popularity  c) Causation  d) Accuracy

**Q3.** Confidence of A → B is:
a) Support(A)  b) Support(A∪B)/Support(A)  c) Support(B)  d) Lift × Support

**Q4.** A lift greater than 1 means A and B:
a) Are independent  b) Are positively associated  c) Avoid each other  d) Cause each other

**Q5.** Which metric corrects for the consequent simply being popular?
a) Support  b) Confidence  c) Lift  d) Accuracy

**Q6.** The Apriori principle states that supersets of an infrequent itemset are:
a) Frequent  b) Also infrequent  c) Unknown  d) The most important

**Q7.** A faster alternative to Apriori is:
a) PCA  b) FP-Growth  c) K-Means  d) DBSCAN

**Q8.** Association implies:
a) Causation  b) Co-occurrence, not causation  c) Prediction  d) Labels

### MCQ Answers
**1:** b. **2:** b. **3:** b. **4:** b. **5:** c. **6:** b. **7:** b. **8:** b.

## Interview Questions (with answers)

**Q1. What are support, confidence, and lift?**
*Answer:* Support is the fraction of transactions containing an itemset (its frequency).
Confidence of A→B is the fraction of transactions with A that also contain B (the rule's
reliability). Lift is confidence divided by the support of B — how much more A and B occur
together than expected if independent; lift > 1 indicates positive association.

**Q2. Why is lift more informative than confidence?**
*Answer:* Confidence can be high simply because the consequent is very common (e.g. milk
bought by most customers), giving false rules. Lift normalises by the consequent's
support, so lift > 1 indicates a genuine association beyond base popularity, > 1 attract,
< 1 avoid, = 1 independent.

**Q3. How does the Apriori algorithm reduce computation?**
*Answer:* It uses the Apriori principle — if an itemset is infrequent, all its supersets
are too — to prune the candidate search. It builds frequent itemsets level by level
(singles, pairs, triples), discarding infrequent branches, avoiding the combinatorial
explosion of checking all itemsets.

**Q4. What is the difference between Apriori and FP-Growth?**
*Answer:* Apriori generates and tests candidate itemsets level by level, scanning the data
repeatedly. FP-Growth compresses the data into an FP-tree and mines frequent itemsets
directly without generating all candidates, making it substantially faster on large
datasets.

## Scenario-Based Questions (with answers)

**Q1.** *A rule "bread → milk" has 95% confidence. A colleague wants to act on it. What do
you check first?*
*Answer:* The **lift**. If 95% of all customers buy milk anyway, the rule is meaningless
(lift ≈ 1). Only if lift > 1 is buying bread genuinely associated with buying milk; act
only on high-lift rules with sufficient support.

**Q2.** *Your association mining returns 50,000 rules. How do you make this useful?*
*Answer:* Filter aggressively: set minimum support (to avoid rare-combination noise),
minimum confidence, and especially minimum lift; then rank by lift and focus on the small
set of high-lift, actionable rules relevant to the business goal.

**Q3.** *You find that buying sunscreen is associated with buying ice cream (lift > 1).
Should the store conclude sunscreen makes people want ice cream?*
*Answer:* No — association is not causation. A confounder (hot weather) likely drives both.
Use the association for placement/promotions, but don't infer a causal mechanism.

## Logic-Based Questions (with answers)

**Q1.** If support({A,B}) = support(A) × support(B), what is the lift of A → B and what
does it mean?
*Answer:* Lift = confidence/support(B) = [support(A∪B)/support(A)]/support(B) =
[support(A)·support(B)/support(A)]/support(B) = 1, meaning A and B are independent — no
association.

**Q2.** Why does a high-confidence rule with lift < 1 actually indicate the items avoid
each other?
*Answer:* Lift < 1 means A and B co-occur *less* than if independent, so although a decent
fraction of A-buyers also buy B (confidence), having A actually *reduces* the chance of B
relative to the baseline — a negative association.

**Q3.** Why would setting the minimum support too low be problematic?
*Answer:* It admits itemsets that appear in only a handful of transactions; such rules can
show high confidence/lift by chance, flooding you with noisy, unreliable patterns.

## Practical Questions (with answers)

**Q1.** Write the formula for the lift of A → B in terms of supports.
*Answer:* Lift(A→B) = Support(A∪B) / (Support(A) × Support(B)).

**Q2.** In the example, why is jam → milk (confidence 0.667) not a useful rule?
*Answer:* Its lift is 0.889 (< 1), so jam and milk co-occur less than chance — the
confidence was inflated by milk's overall popularity, not a real association.

**Q3.** Which Python library provides `apriori` and `association_rules`?
*Answer:* `mlxtend` (`mlxtend.frequent_patterns`).

## Long Questions (with answers)

**Q1. Explain support, confidence, and lift with examples, and explain why lift is the key
metric for judging association rules.**

*Answer:* **Support** measures how frequently an itemset appears: Support(A) = (number of
transactions containing A) / (total transactions); e.g. if bread appears in 6 of 8 baskets,
support(bread) = 0.75. **Confidence** measures a rule's reliability: Confidence(A→B) =
Support(A∪B)/Support(A), the fraction of A-transactions that also contain B; e.g.
confidence(bread→butter) = 0.833 means 83% of bread-buyers also bought butter. **Lift**
measures association strength relative to chance: Lift(A→B) = Confidence(A→B)/Support(B) =
Support(A∪B)/(Support(A)·Support(B)); lift > 1 means A and B occur together more than if
independent (positive association), = 1 means independent, < 1 means they avoid each other.
Lift is the **key metric** because confidence can be deceptively high simply when the
consequent is popular: in the example, jam→milk had a respectable confidence of 0.667, but
its lift was 0.889 (< 1), revealing that jam and milk actually appear together *less* than
chance — the confidence was inflated by milk being common. By normalising for the
consequent's base rate, lift exposes genuine associations and filters out illusions, so
rules should be ranked by lift (subject to a minimum support so they aren't based on too
few transactions).

**Q2. Describe how the Apriori algorithm works and why such an algorithm is necessary,
including the principle it exploits.**

*Answer:* The number of possible itemsets grows combinatorially with the number of
products — with even a few hundred items, checking every itemset's support is
computationally infeasible — so an efficient strategy is essential. **Apriori** exploits
the **Apriori principle**: *if an itemset is infrequent (below the minimum support), then
every superset of it is also infrequent.* This lets it prune vast portions of the search
space. It works **bottom-up**: first it scans the data to find all frequent single items
(those meeting minimum support); then it combines them into candidate pairs, keeps only
the frequent pairs, combines those into candidate triples, and so on, at each level
**discarding any candidate that contains an infrequent subset**. Once all frequent
itemsets are found, it generates candidate rules from them and retains those meeting
minimum confidence (and ideally lift) thresholds. The principle ensures that work is never
wasted exploring supersets of itemsets already known to be rare, turning an intractable
search into a practical one. For very large datasets, **FP-Growth** improves further by
compressing the transactions into an FP-tree and mining frequent itemsets directly without
generating all candidates, avoiding Apriori's repeated data scans.

## Exercises

1. Define support, confidence, and lift in your own words.
2. Given support(A)=0.5, support(B)=0.4, support(A∪B)=0.3, compute confidence(A→B) and
   lift(A→B).
3. Explain why a rule with 90% confidence might still be useless.
4. State the Apriori principle and why it speeds up mining.
5. Give two real applications of association rule learning beyond shopping.

## Mini-Project

**Project: Mine a shopping basket dataset.**

1. Take a transactions dataset (or create 20+ baskets, or use a public groceries dataset).
2. (`pip install mlxtend`) One-hot encode the transactions and run `apriori` with a minimum
   support.
3. Generate association rules and sort them by **lift**.
4. Report the top 5 rules and interpret each in plain business language (e.g. "place X
   near Y").
5. Discuss one high-confidence/low-lift rule you should ignore. Save in `my-ml-journey/`.

## Assignments

1. **Coding:** Implement support/confidence/lift from scratch (as in this chapter) and find
   all rules with lift > 1.1 on a small dataset.
2. **Coding:** Use `mlxtend` to compare Apriori vs FP-Growth on the same data — confirm
   they find the same frequent itemsets.
3. **Conceptual:** Write half a page on why "association is not causation," with a fresh
   market-basket example.

::: tip
We've covered clustering, dimensionality reduction, and association rules — the three
pillars of unsupervised learning. Chapter 30, **Semi-Supervised Learning**, bridges
supervised and unsupervised: how to learn from a *little* labelled data plus a *lot* of
unlabelled data.
:::
