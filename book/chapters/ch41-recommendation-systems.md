# Recommendation Systems

## Introduction

Every time Netflix suggests a show, YouTube queues the next video, Amazon shows "customers
also bought", or Spotify builds your Discover Weekly, a **recommendation system** is at
work. These systems are enormous business drivers — a large share of Netflix viewing and
Amazon sales comes from recommendations. They solve a specific, valuable problem: **out of
millions of items, predict the few that *this particular user* will like.**

::: keyidea
A recommender predicts a user's preference for items they haven't seen, usually using one
of two ideas: **content-based** ("recommend items *similar to what you liked*") or
**collaborative filtering** ("recommend what *similar users* liked"). Most real systems
**combine** both (hybrid).
:::

By the end of this chapter you will be able to:

- Distinguish **content-based**, **collaborative**, and **hybrid** filtering.
- Understand the **user–item matrix**, sparsity, and **matrix factorization**.
- Handle the **cold-start** problem and evaluate recommenders.
- Build a simple collaborative filter from scratch.

## The two main approaches

![Two recommendation strategies. Content-based: recommend items similar to ones the user liked (using item features). Collaborative filtering: recommend items that similar users liked (using the pattern of ratings).](assets/images/ch41_approaches.png)

### Content-based filtering

Recommends items **similar to those the user already liked**, based on item **features**.
If you watched several sci-fi action films, it recommends other sci-fi action films. It
builds a profile of your tastes from item attributes (genre, director, keywords).

- **Pros:** works for new users with some history; no need for other users' data;
  explainable ("because you liked X").
- **Cons:** stays within your existing tastes (limited novelty); needs good item features.

### Collaborative filtering

Recommends items based on the **behaviour of similar users** (or similar items), using the
**ratings/interactions matrix** — *not* item features. The classic insight: "people who
agreed in the past will agree in the future."

- **User-based:** find users similar to you; recommend what they liked.
- **Item-based:** find items similar to ones you liked (by who rated them similarly).
- **Pros:** discovers surprising recommendations beyond your obvious tastes; no item
  features needed.
- **Cons:** the **cold-start** problem (new users/items have no ratings), and **sparsity**
  (most user–item pairs are unrated).

## The user–item matrix and similarity

Collaborative filtering centres on the **user–item matrix**: rows are users, columns are
items, entries are ratings (mostly empty — *sparse*). To find "similar" users or items, we
measure similarity, commonly with **cosine similarity** (the angle between rating vectors):

<div class="equation"><img class="eq" src="assets/images/eq_ch41_cosine.png" alt="cosine similarity"></div>

![The user–item matrix: rows are users, columns are items, cells are ratings (most are empty). Collaborative filtering finds similar users (or items) from this matrix and predicts the missing entries.](assets/images/ch41_matrix.png)

### Practical: user-based collaborative filtering from scratch

```python
import numpy as np

# rows = users, columns = movies; 0 means "not rated yet"
R = np.array([[5, 4, 0, 1, 0],
              [4, 5, 3, 1, 1],
              [1, 0, 5, 4, 4],
              [1, 1, 4, 5, 0],
              [0, 1, 5, 4, 5]], dtype=float)
users = ["Ann", "Bob", "Cara", "Dan", "Eve"]; movies = ["M1", "M2", "M3", "M4", "M5"]

def cosine(a, b):                         # similarity over commonly-rated items
    mask = (a > 0) & (b > 0)
    if mask.sum() == 0: return 0.0
    a, b = a[mask], b[mask]
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

target, item = 0, 4                       # predict Ann's (user 0) rating for M5 (item 4)
sims = sorted([(u, cosine(R[target], R[u])) for u in range(len(users)) if u != target],
              key=lambda x: -x[1])
print("most similar to Ann:", [(users[u], round(s, 3)) for u, s in sims])

num = den = 0                             # similarity-weighted average of others' M5 ratings
for u, s in sims:
    if R[u, item] > 0: num += s * R[u, item]; den += s
print(f"predicted Ann's rating for M5: {num / den:.2f}")
```

**Output:**
```text
most similar to Ann: [('Bob', 0.976), ('Eve', 0.471), ('Cara', 0.428), ('Dan', 0.416)]
predicted Ann's rating for M5: 2.69
```

### Explanation

- **Bob is by far Ann's most similar user (0.976)** — they both love M1/M2 and dislike M4.
- To predict Ann's rating for M5, we take a **similarity-weighted average** of how other
  users rated M5. Because her closest match (Bob) rated M5 low, the prediction is **2.69**
  — so we'd *not* strongly recommend M5 to Ann. This is collaborative filtering in a
  nutshell: *use similar users to fill in the blanks*.

::: keyidea
No item features were used — only the *pattern of ratings*. That's the magic (and the
limitation) of collaborative filtering: it finds taste-mates and borrows their opinions.
It can recommend a film you'd never have guessed, purely because people like you loved it.
:::

## Matrix factorization

For large, sparse matrices, **matrix factorization** (e.g. SVD, the technique that won the
Netflix Prize) is more powerful. It decomposes the user–item matrix into two smaller
matrices of **latent factors** — hidden dimensions like "amount of comedy" or
"action-vs-romance" — learned automatically. Each user and item becomes a short vector;
their dot product predicts the rating. This scales well and captures subtle patterns.

## The cold-start problem

What do you recommend to a **brand-new user** (no history) or a **brand-new item** (no
ratings)? This **cold-start** problem is a central challenge. Solutions: ask new users for
a few preferences, use **content-based** features for new items, recommend popular items as
a fallback, or use side information (demographics, item metadata).

## Evaluating recommenders

- **Rating prediction:** RMSE/MAE between predicted and actual ratings.
- **Top-N recommendation:** **Precision@k** and **Recall@k** (how many of the top-k
  recommendations the user actually liked), and ranking metrics like NDCG.
- **Beyond accuracy:** diversity, novelty, and serendipity matter — recommending only
  obvious items is boring.

::: tip
**Practical & debugging tips:** (1) Real rating matrices are extremely **sparse** — use
libraries like **Surprise** or **implicit** (`pip install scikit-surprise`) and matrix
factorization, not dense loops. (2) **Normalise** for users who rate harshly/generously
(subtract user means). (3) Handle **cold start** explicitly (popularity fallback +
content features). (4) Optimise for the **business metric** (engagement, retention), not
just RMSE. (5) Watch for **feedback loops/filter bubbles** — recommenders shape the very
behaviour they learn from. (6) Most modern systems are **hybrid** + deep learning.
:::

## Advantages, disadvantages, and use cases

| Approach | Strengths | Weaknesses |
|---|---|---|
| Content-based | Works with little user data; explainable; no cold-start for items with features | Limited novelty; needs good features |
| Collaborative | Discovers surprising items; no item features needed | Cold-start; sparsity; popularity bias |
| Matrix factorization | Scales; captures latent patterns | Less interpretable; needs enough data |
| Hybrid | Best of both | More complex to build |

**Use cases:** streaming (Netflix, Spotify, YouTube), e-commerce (Amazon), social feeds,
news, app stores, online learning, and advertising.

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Ignoring the cold-start problem.** A system that only does collaborative
filtering fails for new users/items. Always have a content-based or popularity fallback.
:::

- **Mistake 2 — Optimising only RMSE** while ignoring ranking quality, diversity, and the
  real business goal.
- **Mistake 3 — Not normalising** for harsh vs generous raters.
- **Mistake 4 — Using dense computations** on huge sparse matrices (use proper libraries).
- **Mistake 5 — Creating filter bubbles** by over-optimising for short-term clicks.
- **Mistake 6 — Forgetting popularity bias** (popular items get over-recommended).

## Best practices

- **Use hybrid approaches** (content + collaborative) and matrix factorization.
- **Plan for cold start** with content features and popularity fallbacks.
- **Normalise ratings** per user; handle sparsity with proper libraries.
- **Evaluate with ranking metrics** (Precision@k, NDCG) and the business KPI.
- **Promote diversity/novelty**, and watch for feedback loops and filter bubbles.

## Chapter Summary

- **Recommendation systems** predict which items a user will like, out of many — huge
  business value (streaming, e-commerce, social).
- **Content-based** filtering recommends items *similar to what you liked* (using item
  features); **collaborative filtering** recommends what *similar users/items* liked (using
  the **user–item matrix**), measured by similarity such as **cosine**.
- We built a user-based collaborative filter from scratch: Bob was Ann's closest taste-mate,
  yielding a predicted M5 rating of 2.69 — using *only* rating patterns.
- **Matrix factorization** (latent factors; Netflix Prize) scales to large sparse data.
  Key challenges: **cold start** and **sparsity**; evaluate with RMSE and **Precision@k**,
  plus diversity/novelty.
- Most real systems are **hybrid** and increasingly deep-learning based.

---

::: {.qband}
Practice Zone — Chapter 41
:::

## Multiple-Choice Questions (MCQs)

**Q1.** Recommending items *similar to ones you liked* (using item features) is:
a) Collaborative filtering  b) Content-based filtering  c) Clustering  d) Regression

**Q2.** "Users like you also liked…" describes:
a) Content-based  b) Collaborative filtering  c) PCA  d) Segmentation

**Q3.** The user–item matrix is typically:
a) Dense  b) Sparse (mostly empty)  c) Square always  d) All ones

**Q4.** The cold-start problem refers to:
a) Slow servers  b) New users/items with no history  c) Cold weather  d) Overfitting

**Q5.** Matrix factorization represents users and items as:
a) Images  b) Latent factor vectors  c) Decision trees  d) Pixels

**Q6.** Which technique famously won the Netflix Prize?
a) KNN  b) Matrix factorization (SVD)  c) Naive Bayes  d) PCA

**Q7.** A good top-N recommendation metric is:
a) RMSE only  b) Precision@k  c) Silhouette  d) Inertia

**Q8.** A common fix for cold-start new items is:
a) Delete them  b) Use content-based features  c) Ignore features  d) Train longer

### MCQ Answers
**1:** b. **2:** b. **3:** b. **4:** b. **5:** b. **6:** b. **7:** b. **8:** b.

## Interview Questions (with answers)

**Q1. What is the difference between content-based and collaborative filtering?**
*Answer:* Content-based filtering recommends items similar to those a user liked, using item
features (genre, keywords), building a per-user taste profile. Collaborative filtering uses
the pattern of ratings/interactions across users — recommending what similar users (or
similar items) were rated highly — without needing item features.

**Q2. What is the cold-start problem and how do you address it?**
*Answer:* It's the difficulty of recommending for new users (no history) or new items (no
ratings), where collaborative filtering has nothing to go on. Solutions: ask new users for
initial preferences, use content-based features for new items, fall back to popular items,
and use side information (demographics, metadata).

**Q3. How does matrix factorization work for recommendations?**
*Answer:* It decomposes the sparse user–item matrix into two smaller matrices of latent
factors — short vectors representing hidden traits of users and items. A user's predicted
rating for an item is the dot product of their vectors. It scales to large data and captures
subtle patterns; SVD-based factorization won the Netflix Prize.

**Q4. How do you evaluate a recommendation system?**
*Answer:* For rating prediction, RMSE/MAE between predicted and true ratings. For top-N
recommendations, ranking metrics like Precision@k, Recall@k, and NDCG. Crucially, also
consider business KPIs (engagement, retention) and qualities like diversity and novelty —
not just accuracy.

**Q5. What are the limitations of collaborative filtering?**
*Answer:* Cold start (new users/items), sparsity (most pairs unrated), popularity bias
(popular items over-recommended), scalability on huge matrices, and limited explainability.
Hybrid approaches and matrix factorization mitigate several of these.

## Scenario-Based Questions (with answers)

**Q1.** *A new streaming app has many movies but few user ratings yet. Which approach should
it start with and why?*
*Answer:* Content-based filtering (using movie features like genre, cast) plus
popularity-based recommendations, because collaborative filtering needs substantial rating
data it doesn't yet have. As ratings accumulate, add collaborative/hybrid methods.

**Q2.** *Your recommender has great RMSE but users complain it's boring and repetitive. What's
missing?*
*Answer:* Accuracy alone isn't enough — it lacks diversity, novelty, and serendipity.
Optimise also for these (e.g. re-rank for diversity), and evaluate with ranking and
engagement metrics, not just RMSE, to avoid recommending only obvious items (filter bubble).

**Q3.** *A brand-new user signs up and you must recommend something immediately. What do you
do?*
*Answer:* Handle cold start: show popular/trending items, optionally ask for a few
preferences or genres during onboarding, and use any available side info (demographics,
context) — then switch to personalised collaborative recommendations as history builds.

## Logic-Based Questions (with answers)

**Q1.** In the example, why is Ann's predicted M5 rating low despite some users loving M5?
*Answer:* Because the prediction is weighted by similarity, and Ann's most similar user
(Bob, 0.976) rated M5 low; the users who loved M5 (Cara, Dan, Eve) are much less similar to
Ann, so their high ratings carry little weight.

**Q2.** Why is collaborative filtering able to recommend items unlike a user's past choices,
while content-based cannot?
*Answer:* Collaborative filtering relies on other users' behaviour, so it can surface items
your taste-mates loved even if they don't match your item-feature profile. Content-based only
recommends items similar in features to what you already liked, limiting novelty.

**Q3.** Why does sparsity make collaborative filtering hard?
*Answer:* Most user–item entries are empty, so there's little overlap to compute reliable
similarities or fill in predictions; with few co-rated items, similarity estimates are noisy
and many predictions are unsupported — motivating matrix factorization.

## Practical Questions (with answers)

**Q1.** Write the cosine-similarity formula for two rating vectors.
*Answer:* sim(u,v) = (u·v) / (‖u‖·‖v‖) — the dot product divided by the product of the
vector norms.

**Q2.** In user-based CF, how do you predict a user's rating for an item?
*Answer:* Take a similarity-weighted average of the ratings that *other* users (weighted by
their similarity to the target user) gave that item — as in the chapter's `num/den`
calculation.

**Q3.** Which Python library is commonly used to build recommenders with matrix
factorization?
*Answer:* `scikit-surprise` (Surprise) — or `implicit` for implicit-feedback data.

## Long Questions (with answers)

**Q1. Compare content-based and collaborative filtering in detail, including how each works,
their strengths, weaknesses, and when to use each.**

*Answer:* **Content-based filtering** recommends items similar to those a user has liked,
using **item features** (e.g. genre, director, keywords). It builds a profile of the user's
tastes and scores candidate items by feature similarity. Its **strengths**: it works for a
user as soon as they have a little history, needs no data from other users, handles new
items that have features, and is **explainable** ("recommended because you liked X"). Its
**weaknesses**: it tends to recommend more of the same (limited novelty/serendipity) and
depends on having good item features. **Collaborative filtering** instead uses the
**user–item interaction matrix** — recommending items that **similar users** liked
(user-based) or items similar to ones the user liked based on co-rating patterns
(item-based) — *without* item features. Its **strengths**: it can discover surprising,
cross-domain recommendations because it leverages collective behaviour, and it needs no
feature engineering. Its **weaknesses**: the **cold-start** problem (new users/items lack
ratings), **sparsity** (most pairs unrated, making similarities noisy), **popularity bias**,
and scalability challenges. **When to use each**: content-based suits situations with rich
item features and limited cross-user data or many new items; collaborative suits platforms
with abundant interaction data wanting serendipitous recommendations. In practice, **hybrid**
systems combine both — using content features to address cold start and collaborative
signals for personalised discovery — often with **matrix factorization** or deep learning
underneath.

**Q2. Explain the main challenges in building recommendation systems and how they are
addressed.**

*Answer:* Several challenges recur. **Cold start** — new users or items have no interaction
history, so collaborative methods can't score them; addressed by content-based features,
onboarding preference questions, popularity fallbacks, and side information. **Sparsity** —
the user–item matrix is mostly empty, making similarity estimates unreliable; addressed by
**matrix factorization**, which learns dense low-dimensional latent factors that generalise
across the gaps. **Scalability** — matrices with millions of users and items are huge;
addressed with efficient sparse representations, factorization, approximate nearest
neighbours, and specialised libraries. **Popularity bias and filter bubbles** — systems tend
to over-recommend popular items and narrow users' exposure, reinforcing the behaviour they
learn from; addressed by re-ranking for **diversity and novelty** and monitoring feedback
loops. **Evaluation mismatch** — optimising RMSE doesn't guarantee good user experience;
addressed by using ranking metrics (Precision@k, NDCG) and, ultimately, the **business KPI**
(engagement, retention) via online A/B tests. Finally, **changing tastes and context** mean
recommendations must adapt over time. Modern production systems combine hybrid signals,
matrix factorization or deep models, careful cold-start handling, diversity-aware ranking,
and continuous online evaluation to manage these challenges.

## Exercises

1. Classify each as content-based or collaborative: "because you watched sci-fi", "users
   like you bought this", "similar songs by audio features".
2. Explain the cold-start problem and one solution for new users and one for new items.
3. Why is the user–item matrix sparse, and why does that matter?
4. Compute cosine similarity by hand for u=[5,0,3] and v=[4,0,2] (over rated items).
5. Give two metrics for evaluating top-N recommendations.

## Mini-Project

**Project: Build a movie recommender.**

1. Use a small ratings dataset (e.g. MovieLens-100k, or create a user–item matrix).
2. Implement user-based and item-based collaborative filtering with cosine similarity;
   predict held-out ratings and report RMSE.
3. (`pip install scikit-surprise`) Compare against matrix factorization (SVD).
4. Generate top-5 recommendations for a few users and sanity-check them.
5. Add a popularity fallback for cold-start users. Write a short report. Save in
   `my-ml-journey/`.

## Assignments

1. **Coding:** Extend the chapter's CF to recommend the **top-2 unseen movies** for each
   user (predict all unrated items, rank them).
2. **Coding (stretch):** Implement simple matrix factorization with gradient descent on a
   small rating matrix and compare predictions to the similarity method.
3. **Conceptual:** Write one page on filter bubbles and the ethics of recommendation systems
   (engagement vs well-being), connecting to Chapter 48.

::: tip
Recommenders predict preferences across users and items. Chapter 42, **Time Series
Forecasting**, tackles prediction across *time* — sales, prices, demand, weather — where the
order and trends of past values predict the future.
:::
