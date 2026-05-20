# Types of Machine Learning

## Introduction

In Chapter 2 you met the three main learning styles in one paragraph each. Now we
open the box fully. Choosing the **right type of learning** is the very first
decision in any project — get it wrong and nothing else can save you.

Think of ML types like tools in a toolbox. A hammer, a screwdriver, and a saw each
solve different problems. Using a hammer on a screw is painful and pointless. This
chapter teaches you which "tool" fits which problem, *and why*.

By the end you will be able to:

- Confidently classify any problem as supervised, unsupervised, semi-supervised,
  self-supervised, or reinforcement learning.
- Tell the difference between **classification** and **regression** instantly.
- Understand **clustering**, **dimensionality reduction**, and **anomaly
  detection**.
- Understand the **agent–reward** loop of reinforcement learning.
- Know two other important splits: **batch vs online** and **instance-based vs
  model-based** learning.
- Run small examples of classification, regression, and clustering yourself.

::: keyidea
The single most useful question to ask at the start of any project: **"Do I have
labelled answers in my data?"** If yes → supervised. If no → unsupervised. If a
little → semi-supervised. If learning by reward → reinforcement. This one question
guides everything that follows.
:::

## The big map of Machine Learning types

![A taxonomy of Machine Learning. The main branches are supervised, unsupervised, and reinforcement learning, with semi-/self-supervised sitting between supervised and unsupervised.](assets/images/ch04_ml_taxonomy.png)

We will walk through each branch, from most common to least common in everyday
industry work.

## Supervised Learning

**Supervised Learning** means learning from data that comes **with the answers**
(labels). It is called "supervised" because it is like a student learning with a
teacher who provides the correct answer for every example.

::: note
**The recipe:** you have input features **X** *and* the correct output **y** for
many examples. The model learns the mapping **X → y**, so it can predict **y** for
new **X**.
:::

Supervised learning is by far the **most used** type in industry, because most
business problems are "given this, predict that."

It splits into two sub-types based on *what kind* of answer you predict.

### Classification — predicting a category

In **classification**, the label `y` is a **category** (a class) from a fixed set.

- **Binary classification** — exactly two classes. *Examples:* spam / not spam;
  fraud / legitimate; disease / healthy.
- **Multiclass classification** — more than two, but each item belongs to **one**
  class. *Examples:* a photo is a cat, dog, or horse; a review is 1–5 stars.
- **Multilabel classification** — each item can have **several** labels at once.
  *Example:* a news article tagged both "sports" *and* "politics".

### Regression — predicting a number

In **regression**, the label `y` is a **continuous number**.

*Examples:* predicting house price, tomorrow's temperature, a person's age from a
photo, or expected sales next month.

![Classification predicts a category (which side of the boundary a point falls on); regression predicts a continuous number (a value on a fitted curve).](assets/images/ch04_classification_vs_regression.png)

::: warning
**Classification vs regression** is the most common beginner mix-up.
Ask: *is the answer a label/category or a number on a scale?*
"Will it rain (yes/no)?" → classification.
"How many millimetres of rain?" → regression.
"Which star rating (1–5)?" → usually classification (ordered categories), though it
can be treated as regression.
:::

### Common supervised algorithms (preview)

You will study each of these in depth in Part IV:

| Algorithm | Mainly used for | Chapter |
|---|---|---|
| Linear Regression | Regression | 17 |
| Logistic Regression | Classification | 18 |
| K-Nearest Neighbors | Both | 19 |
| Naive Bayes | Classification | 20 |
| Decision Trees | Both | 21 |
| Support Vector Machines | Both | 22 |
| Random Forest | Both | 23 |
| Gradient Boosting / XGBoost | Both | 24 |

### The cost of labels

Supervised learning needs labelled data — and labelling is often **expensive and
slow**. Imagine paying doctors to label 100,000 X-rays. This cost is exactly why
the next types (semi-supervised, self-supervised) were invented.

## Unsupervised Learning

**Unsupervised Learning** means learning from data with **no labels**. There is no
teacher and no "correct answer." The goal is to discover **hidden structure** in
the data on its own.

There are three main jobs unsupervised learning does.

### Clustering — finding natural groups

**Clustering** groups similar instances together. The algorithm decides the groups;
you did not tell it what they are.

*Examples:* segmenting customers by buying behaviour; grouping news articles by
topic; organising photos by similarity.

![Clustering finds natural groups in unlabelled data. The algorithm is given only the points (left) and discovers the groups itself (right).](assets/images/ch04_clustering.png)

Key algorithms (Chapter 27): **K-Means**, **Hierarchical clustering**, **DBSCAN**.

### Dimensionality reduction — simplifying data

Real data can have hundreds or thousands of features. **Dimensionality reduction**
squeezes them into a few while keeping the important information. This makes data
easier to visualise, faster to process, and less prone to overfitting.

*Example:* compressing 784 pixel values of a digit image into 2 numbers you can
plot. Key algorithms (Chapter 28): **PCA**, **t-SNE**, **UMAP**.

### Association rule learning — finding "goes-together" patterns

**Association rules** find items that frequently occur together.

*Example:* "customers who buy bread and butter often also buy jam" (market-basket
analysis). Key algorithms (Chapter 29): **Apriori**, **FP-Growth**.

### Anomaly detection — finding the odd ones out

**Anomaly (outlier) detection** finds rare, unusual instances.

*Example:* spotting a fraudulent transaction among millions of normal ones, or a
faulty product on a production line.

## Semi-Supervised Learning

**Semi-Supervised Learning** sits between the two: you have a **small amount of
labelled data** and a **large amount of unlabelled data**. The model uses the few
labels plus the structure of the unlabelled data to learn better than it could
from the labels alone.

*Real example:* Google Photos. You label a few faces ("this is Sara"); the system
uses thousands of unlabelled photos to recognise her everywhere.

::: note
Semi-supervised learning exists because of the **label cost** problem. Unlabelled
data is cheap and plentiful; labelled data is expensive. Using both gets the best
of each.
:::

## Self-Supervised Learning

**Self-Supervised Learning** is a clever modern idea: the data **creates its own
labels** automatically, with no humans needed. It is the secret behind modern Large
Language Models (Chapter 39).

*How it works for text:* take a sentence, hide one word, and ask the model to
predict the hidden word. The "label" (the hidden word) comes free from the data
itself. Do this on billions of sentences and the model learns deep patterns of
language.

::: keyidea
Self-supervised learning unlocked the LLM revolution because it removed the label
bottleneck: the entire internet became "free" training data, since the text labels
itself. Remember this when we reach Chapter 39.
:::

## Reinforcement Learning

**Reinforcement Learning (RL)** is completely different. There is no fixed dataset.
Instead, an **agent** learns by **interacting** with an **environment**, taking
**actions**, and receiving **rewards** (positive) or **penalties** (negative). Over
time it learns a **policy** — a strategy for choosing actions that maximise total
reward.

It is how you might train a dog: good behaviour → treat; bad behaviour → no treat.

![The reinforcement learning loop. The agent observes the state, takes an action, and the environment returns a reward and a new state. The agent learns the policy that maximises long-term reward.](assets/images/ch04_rl_loop.png)

**Key vocabulary:**

- **Agent** — the learner/decision-maker (e.g., a game-playing program).
- **Environment** — the world the agent acts in (e.g., the game).
- **State** — the current situation the agent observes.
- **Action** — a choice the agent makes.
- **Reward** — feedback signal (a number) telling the agent how good the action
  was.
- **Policy** — the agent's strategy: state → action.

*Real examples:* AlphaGo (Chapter 3), robots learning to walk, self-driving
decisions, recommendation systems that optimise long-term engagement, and training
chatbots with human feedback (RLHF). RL is covered fully in Chapter 31.

::: warning
RL is powerful but **hard**: rewards can be sparse or delayed, training is slow and
unstable, and a poorly designed reward can teach the agent to "cheat." Do not reach
for RL unless the problem is genuinely about *sequential decisions with feedback*.
:::

## Two other important ways to categorise learning

The supervised/unsupervised/RL split is about *what kind of feedback* the model
gets. Two other splits describe *how* the model is trained and used.

### Batch vs Online learning

- **Batch (offline) learning** — the model is trained **once** on the whole dataset,
  then deployed. To learn from new data you retrain from scratch. Simple and common.
- **Online (incremental) learning** — the model learns **continuously**, one piece
  (or small batch) of data at a time, updating as new data arrives. Good for data
  streams (e.g., stock prices, live sensors) and when data is too big to fit in
  memory.

### Instance-based vs Model-based learning

- **Instance-based** — the model **memorises** the training examples and compares
  new data to them. *Example:* K-Nearest Neighbors (it literally looks up the
  closest stored examples).
- **Model-based** — the model **builds a general formula/model** from the data and
  then throws the data away. *Example:* Linear Regression (it keeps only the learned
  line, not the original points).

## How to choose the right type — a decision guide

| Question | If YES → |
|---|---|
| Do you have labelled answers (`y`)? | Supervised |
| → Is the answer a category? | Classification |
| → Is the answer a number? | Regression |
| No labels, want to find groups/structure? | Unsupervised (clustering / DR) |
| Want to find rare/odd items? | Anomaly detection |
| A few labels + lots of unlabelled data? | Semi-supervised |
| Labels come "free" from the data itself? | Self-supervised |
| Learning by trial, reward, and feedback over time? | Reinforcement |

::: tip
Print this table and keep it beside you for your first ten projects. Correctly
identifying the learning type is half the battle — and a favourite interview
question.
:::

## Practical: classification, regression, and clustering side by side

Let's see three learning jobs in action, each in a few lines. This makes the
differences concrete.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans

# ---------- 1) CLASSIFICATION: predict a CATEGORY (0 or 1) ----------
# Feature: exam score. Label: admitted (1) or not (0).
X_cls = np.array([[35], [45], [50], [60], [65], [80]])
y_cls = np.array([0,    0,    0,    1,    1,    1])      # categories
clf = LogisticRegression().fit(X_cls, y_cls)
print("Classification — score 70 ->", clf.predict([[70]])[0])  # 0 or 1

# ---------- 2) REGRESSION: predict a NUMBER ----------
# Feature: house size (m²). Label: price ($1000s) — a continuous number.
X_reg = np.array([[50], [70], [90], [110], [130]])
y_reg = np.array([100,  140,  180,  220,   260])         # numbers
reg = LinearRegression().fit(X_reg, y_reg)
print(f"Regression — 100 m² -> ${reg.predict([[100]])[0]:.0f}k")

# ---------- 3) CLUSTERING: find GROUPS with NO labels ----------
# Just points; we provide NO answers. KMeans finds 2 groups itself.
X_clu = np.array([[1,1],[1.5,2],[1,1.5],   [8,8],[9,8.5],[8.5,9]])
km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X_clu)
print("Clustering — group labels:", km.labels_)
```

**Output (yours may vary slightly):**
```text
Classification — score 70 -> 1
Regression — 100 m² -> $200k
Clustering — group labels: [0 0 0 1 1 1]
```

### Explanation

- **Classification** (`LogisticRegression`): the label `y_cls` is a **category**
  (0/1). The model learns a boundary and predicts class `1` (admitted) for a score
  of 70. (Try 55 — it sits right on the boundary and predicts `0`; borderline
  inputs are exactly where models are least confident.)
- **Regression** (`LinearRegression`): the label `y_reg` is a **number** (price).
  The model learns a line and predicts ~`$200k` for a 100 m² house.
- **Clustering** (`KMeans`): we gave **no labels** — only points. The algorithm
  discovered two groups on its own (the first three points in group `0`, the last
  three in group `1`). The group numbers (0/1) are arbitrary names, not "correct
  answers."

::: keyidea
Same library, three completely different jobs. The difference is **not** the code
length — it is whether you have labels, and whether those labels are categories or
numbers. That decision *is* the type of learning.
:::

::: tip
**Debugging tip:** `KMeans` can give different group *numbers* on different runs
(group 0 and 1 may swap) — that is normal, because cluster names are arbitrary.
Fix `random_state` for reproducibility, and judge clustering by *which points are
grouped together*, not by the label numbers.
:::

## Real-world applications by type

| Type | Real-world application |
|---|---|
| Classification | Spam detection, disease diagnosis, image recognition, sentiment |
| Regression | Price prediction, demand forecasting, risk scoring |
| Clustering | Customer segmentation, document grouping, image compression |
| Dimensionality reduction | Data visualisation, noise reduction, speeding up models |
| Anomaly detection | Fraud detection, fault detection, network intrusion |
| Semi-supervised | Photo face grouping, medical imaging with few labels |
| Self-supervised | Large Language Models, modern vision pre-training |
| Reinforcement | Game AI, robotics, autonomous control, RLHF for chatbots |

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Treating a regression problem as classification (or vice versa).**
Predicting exact price is regression; predicting a price *range* (cheap/medium/
expensive) is classification. Pick based on what the business actually needs.
:::

- **Mistake 2 — Expecting clustering to give "correct" labels.** It finds groups,
  but *you* must interpret what each group means.
- **Mistake 3 — Reaching for reinforcement learning** for problems that are really
  simple supervised learning. RL is complex; use it only for sequential decisions.
- **Mistake 4 — Forgetting semi-/self-supervised options** when labels are scarce.
- **Mistake 5 — Confusing multiclass and multilabel.** One-class-per-item vs
  many-labels-per-item are different problems with different code.

## Best practices

- **Start by writing down the learning type** and justifying it in one sentence.
- **Prefer the simplest type** that fits; don't over-engineer.
- **For scarce labels,** seriously consider semi-/self-supervised approaches.
- **State your metric (P) per type:** accuracy/F1 for classification, error metrics
  (RMSE/MAE) for regression, and task-specific measures for clustering.
- **Re-examine the type if results are poor** — sometimes the problem was framed as
  the wrong type from the start.

## Chapter Summary

- **Supervised** learning uses **labelled** data (X → y); it splits into
  **classification** (predict a category) and **regression** (predict a number).
- **Unsupervised** learning uses **unlabelled** data to find structure:
  **clustering**, **dimensionality reduction**, **association rules**, and
  **anomaly detection**.
- **Semi-supervised** uses a few labels + lots of unlabelled data; **self-
  supervised** lets the data label itself (the engine behind LLMs).
- **Reinforcement learning** trains an **agent** via **rewards** from an
  **environment** to learn a **policy**.
- Other splits: **batch vs online** (train once vs continuously) and
  **instance-based vs model-based** (memorise vs generalise to a formula).
- Choosing the right type starts with one question: **"Do I have labels?"**

---

::: {.qband}
Practice Zone — Chapter 4
:::

## Multiple-Choice Questions (MCQs)

**Q1.** Predicting whether an email is spam or not is:
a) Regression  b) Binary classification  c) Clustering  d) Reinforcement learning

**Q2.** Predicting the exact price of a house is:
a) Classification  b) Clustering  c) Regression  d) Anomaly detection

**Q3.** Grouping customers by behaviour with no labels is:
a) Supervised  b) Clustering (unsupervised)  c) Regression  d) RL

**Q4.** In reinforcement learning, the feedback signal is called the:
a) Label  b) Feature  c) Reward  d) Cluster

**Q5.** Tagging an article as both "sports" and "politics" is:
a) Binary classification  b) Multiclass classification  c) Multilabel
classification  d) Regression

**Q6.** The learning type where the data creates its own labels is:
a) Supervised  b) Self-supervised  c) Reinforcement  d) Batch

**Q7.** K-Nearest Neighbors, which compares new data to stored examples, is:
a) Model-based  b) Instance-based  c) Online  d) Reinforcement

**Q8.** Reducing 784 features down to 2 for plotting is:
a) Classification  b) Clustering  c) Dimensionality reduction  d) Regression

### MCQ Answers
**1:** b. **2:** c. **3:** b. **4:** c. **5:** c. **6:** b. **7:** b. **8:** c.

## Interview Questions (with answers)

**Q1. What is the difference between supervised and unsupervised learning?**
*Answer:* Supervised learning uses labelled data (inputs paired with known correct
outputs) to learn a mapping X → y. Unsupervised learning uses unlabelled data to
discover hidden structure (groups, patterns, lower-dimensional representations)
without any target answers.

**Q2. Difference between classification and regression?**
*Answer:* Both are supervised. Classification predicts a discrete category (e.g.,
spam/not spam). Regression predicts a continuous number (e.g., price). The output
type determines which one you use.

**Q3. What is semi-supervised learning and why is it useful?**
*Answer:* It uses a small amount of labelled data together with a large amount of
unlabelled data. It is useful because labelling is expensive; combining a few
labels with abundant unlabelled data often beats using the few labels alone.

**Q4. Explain reinforcement learning in simple terms.**
*Answer:* An agent interacts with an environment, takes actions, and receives
rewards or penalties. Over time it learns a policy (a strategy mapping states to
actions) that maximises total long-term reward — like training a pet with treats.

**Q5. What is the difference between batch and online learning?**
*Answer:* Batch learning trains once on the full dataset and must be retrained to
incorporate new data. Online learning updates continuously as new data arrives, one
sample or mini-batch at a time — suited to data streams and very large datasets.

## Scenario-Based Questions (with answers)

**Q1.** *A bank wants to detect fraudulent transactions. Fraud is extremely rare
and mostly unlabelled. What learning type(s) fit?*
*Answer:* Primarily **anomaly detection** (unsupervised), since fraud is rare and
labels are scarce. If some confirmed-fraud labels exist, a **semi-supervised** or
imbalanced **supervised classification** approach can be added. Accuracy is a poor
metric here; use precision/recall.

**Q2.** *A company has millions of product images but only 500 are labelled by
category. They want an image classifier. What approach makes sense?*
*Answer:* Use **self-supervised / transfer learning** to pre-train on the unlabelled
images (or use a pre-trained model), then fine-tune with the 500 labels — a
**semi-supervised** strategy. Labelling all millions would be too costly.

**Q3.** *You're asked to build a system that learns to control a drone to fly
through hoops, improving with practice. Which type, and what's the main risk?*
*Answer:* **Reinforcement learning** — it's sequential decision-making with reward
(passing hoops). Main risks: sparse/delayed rewards, unstable and slow training,
and reward mis-specification causing unintended "cheating" behaviour.

## Logic-Based Questions (with answers)

**Q1.** A dataset has inputs but the target column is completely empty. Which whole
*family* of learning is impossible, and which becomes the natural choice?
*Answer:* Supervised learning is impossible (no labels). Unsupervised learning
(clustering, dimensionality reduction, anomaly detection) becomes the natural
choice.

**Q2.** If clustering gives group labels `[1,1,0,0]` on one run and `[0,0,1,1]` on
the next for the same data, did the result actually change?
*Answer:* No. The *grouping* is identical (the same points are together); only the
arbitrary group *names* swapped. Cluster IDs carry no inherent meaning.

**Q3.** A model that "memorises all training points and compares new inputs to
them" — is it more likely batch or online, instance-based or model-based?
*Answer:* It is **instance-based** (it stores examples). It can be used in either
batch or online settings, but the defining trait here is instance-based vs
model-based.

## Practical Questions (with answers)

**Q1.** In the practical code, how can you tell the classification example from the
regression example just by looking at `y`?
*Answer:* In classification, `y_cls` contains discrete categories (only 0s and 1s).
In regression, `y_reg` contains continuous numbers (100, 140, 180, …). The nature
of `y` defines the task.

**Q2.** Why did we pass `n_clusters=2` to KMeans, and what would happen with
`n_clusters=3`?
*Answer:* We told KMeans to find 2 groups because we designed the data with two
obvious clusters. With `n_clusters=3`, it would force the points into 3 groups,
splitting one natural group artificially — a reminder that *you* must choose a
sensible number of clusters (methods for this are in Chapter 27).

**Q3.** Write one line to predict the cluster of a new point `[8, 8]`.
*Answer:* `print(km.predict([[8, 8]]))` — it should return the group containing the
high-valued points.

## Long Questions (with answers)

**Q1. Compare supervised, unsupervised, and reinforcement learning across: the data
they need, what they learn, example tasks, and main challenges.**

*Answer:* **Supervised learning** needs labelled data (inputs with correct
outputs); it learns a mapping from inputs to outputs; example tasks are spam
detection (classification) and price prediction (regression); its main challenge is
the cost and quality of labels and the risk of overfitting. **Unsupervised
learning** needs only unlabelled data; it learns hidden structure such as clusters
or compressed representations; example tasks are customer segmentation and anomaly
detection; its main challenges are that there is no ground truth to measure against,
making evaluation and interpretation hard. **Reinforcement learning** needs an
interactive environment rather than a fixed dataset; it learns a policy that
maximises long-term reward through trial and error; example tasks are game-playing,
robotics, and control; its main challenges are sparse/delayed rewards, sample
inefficiency, training instability, and reward design. In practice, supervised
learning dominates industry because most problems are "given X, predict y" and
labels, while costly, are obtainable.

**Q2. Explain why self-supervised learning was a turning point for modern AI, and
how it relates to supervised and unsupervised learning.**

*Answer:* Self-supervised learning generates labels automatically from the data
itself — for example, hiding a word in a sentence and training the model to predict
it. This sidesteps the biggest bottleneck of supervised learning: the expensive,
slow, human labelling process. Because the labels come "for free," practically
unlimited raw data (like all the text on the internet) becomes usable training
data. Conceptually it sits between the two classic families: like unsupervised
learning, it needs no human labels; like supervised learning, it trains on a clear
prediction target (the auto-generated label). This combination is what made it
possible to train enormous Large Language Models, which first learn general
patterns via self-supervision on massive text, then are fine-tuned (often with
supervised or reinforcement methods) for specific tasks. In short, self-supervised
learning unlocked scale, and scale unlocked the capabilities we see in modern AI.

## Exercises

1. For each of these, name the learning type and sub-type: (a) predicting
   tomorrow's temperature, (b) grouping songs by sound, (c) flagging unusual logins,
   (d) a robot learning to grasp objects, (e) tagging a photo with multiple objects.
2. Explain the difference between multiclass and multilabel classification with a
   fresh example.
3. Give two real situations where you'd prefer online learning over batch learning.
4. In your own words, why is labelling data expensive, and how do semi-/self-
   supervised learning help?
5. Draw the reinforcement-learning loop from memory and label all six terms.

## Mini-Project

**Project: The "which type?" classifier (for humans).**

1. Collect 12 real problem statements (from news, products, or your own ideas),
   e.g. "predict next month's electricity demand."
2. For each, write down: the learning type, the sub-type, what `X` and `y` would
   be (or "no labels"), and the metric `P` you'd use.
3. For at least three of them, argue why a *different* type would be the wrong
   choice.
4. Present your 12 problems as a neat table. *(This exact skill — framing the
   problem — is what separates strong ML practitioners from beginners.)*

## Assignments

1. **Conceptual:** Write one page comparing classification and regression, with two
   original real-world examples of each and the metric you'd use.
2. **Coding:** Extend the practical code with a **multiclass** classification
   example using three classes (hint: `make_blobs` from `sklearn.datasets` or three
   sets of scores). Print predictions for two new inputs.
3. **Research:** Find one real product that uses **reinforcement learning** and one
   that uses **self-supervised learning**. Describe each in terms of this chapter's
   vocabulary. Cite sources.

::: tip
You now understand the *map* of Machine Learning. Part II next gives you the
*tools* — the maths, statistics, and Python — to start walking the territory for
real.
:::
