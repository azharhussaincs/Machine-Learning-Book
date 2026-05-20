# Real-World ML Projects

## Introduction

Welcome to **Part IX** — turning knowledge into a **career**. Reading about ML is one thing;
**building real projects** is what makes you skilled, confident, and employable. This chapter
gives you a **repeatable project template**, one **complete worked project** that ties the
whole book together, and a **catalog of 18 portfolio projects** (the ones you've seen
referenced throughout) with blueprints to build each.

::: keyidea
**Projects are your portfolio, and your portfolio gets you hired.** Employers and clients
care less about certificates than about *what you've built*. A handful of well-documented,
end-to-end projects — from data to deployed app — proves you can actually *do* ML, not just
talk about it.
:::

By the end of this chapter you will be able to:

- Follow a **standard end-to-end project workflow** for any ML project.
- Build one **complete project** from data to a saved, deployable model.
- Choose and scope **portfolio projects** across domains.

## The end-to-end project workflow

Every real project follows the lifecycle from Chapter 2, now with everything you've learned:

1. **Problem definition** — what are you predicting (T, E, P)? Is ML the right tool?
2. **Data collection & understanding** — gather data; **EDA** (Chapter 15).
3. **Cleaning & preprocessing** — handle missing values, outliers, scaling, encoding (Ch
   10–11).
4. **Feature engineering & selection** — create and choose informative features (Ch 12–13).
5. **Model building** — try several algorithms (the bake-off, Ch 16); a **baseline first**.
6. **Evaluation** — proper metrics + cross-validation (Ch 25).
7. **Tuning** — hyperparameters + regularization (Ch 26).
8. **Deployment** — serve via API/app (Ch 44); **monitor** (Ch 45).
9. **Documentation** — README, model card, and a clear write-up.

## A complete worked project: Customer Churn Prediction

**Problem:** predict which customers will leave (churn) so the business can retain them — a
classic, high-value classification problem. *(We use a clean dataset as a stand-in; the
workflow is identical for real churn data.)*

```python
import joblib, numpy as np
from sklearn.datasets import load_breast_cancer            # proxy for a churn dataset
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, roc_auc_score

X, y = load_breast_cancer(return_X_y=True)                 # 1) load data
X_tr, X_te, y_tr, y_te = train_test_split(                 # 4) split (stratified)
    X, y, test_size=0.2, random_state=42, stratify=y)

# 5) a pipeline: preprocessing + model (prevents leakage, deployable)
pipe = make_pipeline(StandardScaler(),
                     RandomForestClassifier(n_estimators=200, random_state=42))

cv = cross_val_score(pipe, X_tr, y_tr, cv=5)               # 6) cross-validated estimate
pipe.fit(X_tr, y_tr)                                       # train on full training set
proba = pipe.predict_proba(X_te)[:, 1]

print("5-fold CV accuracy: %.3f ± %.3f" % (cv.mean(), cv.std()))
print("Test AUC: %.3f" % roc_auc_score(y_te, proba))      # 6) honest test metric
print(classification_report(y_te, pipe.predict(X_te), digits=3))

joblib.dump(pipe, "churn_model.joblib")                    # 8) save → deploy with FastAPI (Ch 44)
```

**Output (key lines):**
```text
5-fold CV accuracy: 0.958 ± 0.016
Test AUC: 0.993
   ... precision/recall ~0.94–0.97 per class ...
model saved & ready to deploy
```

### Walkthrough

- We **loaded** data, **split** it (stratified), and built a **pipeline** combining
  preprocessing and a Random Forest — so the same transformations apply at inference and no
  leakage occurs.
- We measured a **cross-validated** accuracy (0.958 ± 0.016) for a robust estimate, then an
  honest **test AUC of 0.993** and a per-class report.
- We **saved** the whole pipeline with `joblib`, ready to serve via FastAPI (Chapter 44).

This single, compact project exercises **the entire book**: data → split → pipeline →
cross-validation → metrics → save → deploy. Make this workflow second nature, and you can
tackle any tabular ML project.

::: tip
**Portfolio tips:** (1) For each project, write a clear **README** (problem, data, approach,
results, how to run) and a short **model card** (Chapter 48). (2) Show your **EDA and
reasoning**, not just the final model. (3) **Deploy at least one** project (FastAPI/Streamlit)
— a live demo is worth a thousand words. (4) Put projects on **GitHub**; pin your best three.
(5) Pick projects in domains you care about (or a target employer's domain). (6) Quality over
quantity: 3 polished, deployed projects beat 20 half-finished notebooks.
:::

## A catalog of 18 portfolio projects

Each blueprint lists the **problem**, **data**, **approach** (chapters), and a **deployment**
idea. Build them progressively — start with the beginner tier.

### Beginner

| Project | Problem → Data → Approach | Chapters |
|---|---|---|
| **House Price Prediction** | Predict price → housing dataset → regression + feature engineering | 12, 17 |
| **Spam Detection** | Spam vs not → SMS/email text → TF-IDF + Naive Bayes/LogReg | 20, 38 |
| **Sentiment Analysis** | Pos/neg → reviews → TF-IDF or Transformer | 38, 39 |
| **Iris/Wine Classification** | Species/quality → classic datasets → bake-off | 16, 19 |
| **Customer Churn** | Will leave? → telco data → RF/XGBoost + pipeline | 23, 24, 49 |

### Intermediate

| Project | Problem → Data → Approach | Chapters |
|---|---|---|
| **Fake News Detection** | Real vs fake → news text → NLP + classifier | 38 |
| **Fraud Detection** | Fraud? → imbalanced transactions → anomaly/imbalanced classification | 25, 27 |
| **Recommendation System** | Suggest items → ratings → collaborative filtering / MF | 41 |
| **Resume Screening** | Match resumes → text → NLP similarity/classification | 38, 39 |
| **Stock/Sales Forecasting** | Future values → time series → lag features / LSTM | 35, 42 |
| **Image Classification** | Categorise images → image dataset → CNN / transfer learning | 34, 40 |
| **Medical Diagnosis** | Disease? → clinical/imaging data → classification (careful ethics) | 25, 40, 48 |

### Advanced

| Project | Problem → Data → Approach | Chapters |
|---|---|---|
| **Face Recognition** | Identify faces → face images → CNN embeddings | 34, 40 |
| **Object Detection** | Find objects → annotated images → YOLO/Faster R-CNN | 40 |
| **Chatbot / AI Tutor** | Converse/teach → text → LLM + RAG | 39, 43 |
| **Speech Recognition** | Audio → text → sequence/Transformer models | 35, 37 |
| **Translation System** | Language→language → parallel text → seq2seq/Transformer | 37, 38 |
| **Image Generation / Deepfake Detection** | Create/detect → images → diffusion/GAN; CNN detector | 36, 43 |

::: keyidea
Notice every project maps back to chapters you've already studied. You are **not** starting
from zero on any of them — you have the tools. Pick one, follow the 9-step workflow, deploy
it, and document it. Then repeat. That is how you build both skill and a portfolio.
:::

## Scoping a project well

- **Start small and end-to-end.** A simple model *deployed* beats a fancy model in a
  notebook.
- **Get a working baseline fast**, then improve.
- **Use a real, messy dataset** when you can — handling reality is the skill.
- **Define success up front** (the metric and target).
- **Time-box** exploration so you actually finish.

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Never finishing or deploying.** Endless tweaking in a notebook produces no
portfolio. Ship an end-to-end version first, then iterate.
:::

- **Mistake 2 — Skipping EDA/cleaning** and jumping to modelling (Part III matters most).
- **Mistake 3 — Chasing accuracy without a baseline** or the right metric.
- **Mistake 4 — Data leakage** (preprocessing before split; future info) — use pipelines.
- **Mistake 5 — No documentation** — undocumented projects don't impress anyone.
- **Mistake 6 — Only toy datasets** — include at least one real, messy dataset.

## Best practices

- **Follow the 9-step workflow** every time.
- **Baseline first**, then improve; compare several models.
- **Use pipelines** to prevent leakage and enable deployment.
- **Deploy and document** at least your best projects (GitHub + live demo).
- **Pick meaningful domains**; show your reasoning, not just results.
- **Iterate**: each project teaches you more than the last.

## Chapter Summary

- **Projects build skill and a portfolio** — what actually gets you hired or wins clients.
- Every project follows the **9-step workflow**: define → data/EDA → clean/preprocess →
  feature engineer/select → model (baseline + bake-off) → evaluate → tune → deploy → document.
- The **worked churn project** exercised the whole book — pipeline, cross-validation (0.958),
  test AUC (0.993), and a saved, deployable model.
- A **catalog of 18 projects** (beginner → advanced) maps each to the chapters you've learned;
  pick, build end-to-end, deploy, and document them.
- **Finish and deploy** projects, use **pipelines** (no leakage), and **document** them on
  GitHub with a README and model card.

---

::: {.qband}
Practice Zone — Chapter 49
:::

## Multiple-Choice Questions (MCQs)

**Q1.** The best evidence of ML skill for employers is usually:
a) Certificates  b) A portfolio of built/deployed projects  c) Memorised theory  d) Long CVs

**Q2.** The first modelling step in a project should be:
a) The most complex model  b) A simple baseline  c) Deployment  d) Hyperparameter tuning

**Q3.** Using a pipeline (preprocess + model) primarily helps:
a) Make plots  b) Prevent data leakage & enable deployment  c) Reduce features  d) Add labels

**Q4.** For a spam-detection project you'd combine:
a) CNN + pooling  b) TF-IDF + Naive Bayes/LogReg  c) K-Means + PCA  d) ARIMA

**Q5.** A churn-prediction project is a:
a) Regression task  b) Classification task  c) Clustering task  d) Forecasting task

**Q6.** A common project-killing mistake is:
a) Writing a README  b) Never finishing/deploying  c) Doing EDA  d) Using a baseline

**Q7.** Time-series/sales forecasting projects use:
a) One-hot encoding only  b) Lag features / LSTM  c) K-Means  d) Naive Bayes

**Q8.** For a real portfolio, it's best to have:
a) 20 unfinished notebooks  b) 3 polished, deployed, documented projects  c) Only theory
d) No code

### MCQ Answers
**1:** b. **2:** b. **3:** b. **4:** b. **5:** b. **6:** b. **7:** b. **8:** b.

## Interview Questions (with answers)

**Q1. Walk me through an end-to-end ML project you'd build.**
*Answer:* Define the problem and success metric; collect and explore the data (EDA); clean and
preprocess it (missing values, scaling, encoding); engineer and select features; build a
baseline then compare several models with cross-validation; tune the best with the right
metric; deploy it as an API/app with a saved pipeline; and document it (README + model card)
while monitoring in production.

**Q2. Why is deploying a project important even if it's simple?**
*Answer:* Deployment proves you can take a model from notebook to a usable product — the skill
employers and clients value most. A live demo demonstrates the full lifecycle (data → model →
serving) and stands out far more than an undeployed notebook, however accurate.

**Q3. Why use a pipeline in a project?**
*Answer:* A pipeline bundles preprocessing and the model so the exact same transformations are
applied during training and inference, preventing data leakage and making the artifact
directly deployable and reproducible.

**Q4. How do you choose which projects to build for a portfolio?**
*Answer:* Pick projects that cover varied skills (tabular, NLP, vision, time series), align
with target domains/employers, use at least one real messy dataset, and can be completed and
deployed. Favour a few polished, documented, deployed projects over many unfinished ones.

## Scenario-Based Questions (with answers)

**Q1.** *You have two weeks to build a portfolio project to land an ML role at a fintech. What
do you build and how?*
*Answer:* Build a fintech-relevant, end-to-end project — e.g. credit-fraud or churn prediction
on a real (imbalanced) dataset. Do EDA, build a pipeline with proper metrics (precision/recall/
AUC for imbalance), tune it, deploy it via FastAPI/Streamlit, and document it on GitHub with a
README and fairness/model card — showing the full lifecycle and domain awareness.

**Q2. ** *Your project gets 99% accuracy on imbalanced fraud data. A recruiter asks if it's a
good model. What do you say?*
*Answer:* Accuracy is misleading on imbalanced data (the accuracy paradox); I'd report
precision, recall, F1, and AUC on the fraud class and the confusion matrix, and explain how I
handled imbalance — demonstrating I understand evaluation, not just a headline number.

**Q3.** *You keep tweaking a model for weeks and have nothing to show. What should you change?*
*Answer:* Ship an end-to-end version now: baseline model, deployed and documented, then iterate.
Time-box exploration and prioritise finishing. A complete, imperfect project beats an endlessly
"almost-done" one.

## Logic-Based Questions (with answers)

**Q1.** Why does a deployed simple model often impress more than an undeployed complex one?
*Answer:* Because deployment demonstrates the complete, real-world skill set (serving,
reproducibility, integration), which is what produces value; raw model complexity in a notebook
doesn't show you can deliver a working product.

**Q2.** Why is a baseline model valuable at the start of a project?
*Answer:* It sets a reference performance to beat, validates the data pipeline end to end, and
reveals whether complex models actually add value — preventing wasted effort and overcomplex
solutions.

**Q3.** Why include at least one messy real-world dataset in your portfolio?
*Answer:* Because real ML work is dominated by handling messy data (missing values, outliers,
inconsistencies); demonstrating you can do this proves practical competence that clean toy
datasets never test.

## Practical Questions (with answers)

**Q1.** Which scikit-learn helper bundles preprocessing and a model into one deployable object?
*Answer:* `Pipeline` / `make_pipeline` (with `ColumnTransformer` for mixed column types).

**Q2.** How do you save a trained pipeline for deployment?
*Answer:* `joblib.dump(pipe, "model.joblib")`, then load it in the serving app with
`joblib.load`.

**Q3.** Name the nine steps of the end-to-end project workflow.
*Answer:* Define problem → data/EDA → clean/preprocess → feature engineer/select → model
(baseline + bake-off) → evaluate → tune → deploy → document.

## Long Questions (with answers)

**Q1. Describe the complete workflow for building a real-world ML project, using churn
prediction as the example.**

*Answer:* Start by **defining the problem**: predict which customers will churn (a binary
classification) and choose the success metric — for imbalanced churn, precision/recall/AUC on
the churn class, not just accuracy. **Collect and explore** the data (customer attributes,
usage, contract details) with EDA to understand distributions, missing values, and which
features relate to churn. **Clean and preprocess**: handle missing values, treat outliers,
encode categoricals, and scale numerics — ideally inside a **pipeline** so the same steps apply
at inference. **Engineer and select features** (e.g. tenure buckets, usage ratios), then
**build models**: establish a simple baseline (logistic regression), then run a bake-off
(random forest, gradient boosting) using **cross-validation** for robust estimates. **Evaluate**
on a held-out test set with appropriate metrics and a confusion matrix, and **tune** the best
model's hyperparameters. **Deploy** the saved pipeline as a FastAPI endpoint or Streamlit app so
the business can score customers, and set up **monitoring** for drift and performance.
Finally, **document** everything — a README explaining the problem, data, approach, and results,
plus a model card noting limitations and fairness. The worked example in this chapter performed
exactly this flow, achieving a cross-validated accuracy of 0.958 and a test AUC of 0.993 with a
saved, deployable pipeline — demonstrating the whole book in one project.

**Q2. Explain how to build an effective ML portfolio and why it matters for a career.**

*Answer:* A **portfolio** is a curated set of completed, documented, ideally deployed projects
that demonstrate your ability to do ML end to end — and it matters because employers and clients
trust **demonstrated work** over certificates or claimed knowledge. To build an effective one:
**choose diverse, meaningful projects** spanning tabular, NLP, vision, and time-series skills,
aligned with the domains you want to work in, and including at least one **real, messy dataset**
to prove practical competence. For each project, **follow the full workflow** (EDA → cleaning →
features → modelling with a baseline and bake-off → evaluation with the right metrics → tuning →
deployment), and crucially **finish and deploy** at least your best ones with FastAPI/Streamlit
so there's a live demo. **Document** each clearly — a README (problem, data, approach, results,
how to run) and a model card (limitations, fairness) — and host them on **GitHub**, pinning your
strongest three. Favour **quality over quantity**: three polished, deployed, well-documented
projects communicate far more than twenty abandoned notebooks. Such a portfolio shows you can
turn data into working products, handle real-world messiness, evaluate honestly, and communicate
your work — exactly the competencies that land ML jobs and freelance clients (Chapters 51–52).

## Exercises

1. Write the 9-step workflow from memory.
2. For three catalog projects, name the data type and the main algorithm/approach.
3. Explain why a deployed simple model beats an undeployed complex one for a portfolio.
4. Pick a project and write its problem statement and success metric.
5. List what a good project README should contain.

## Mini-Project

**Project: Build and deploy your first complete project.**

1. Choose a beginner project from the catalog (e.g. churn, house prices, or spam).
2. Run the full 9-step workflow on a real dataset; use a pipeline and proper metrics.
3. Deploy it (FastAPI or Streamlit, Chapter 44) with a working demo.
4. Write a README and a short model card (Chapter 48).
5. Push it to GitHub. This is portfolio project #1 — save everything in `my-ml-journey/`.

## Assignments

1. **Build:** Complete two projects from different tiers (e.g. one beginner, one intermediate),
   each deployed and documented.
2. **Build (stretch):** Take one project to production quality — add input validation, logging,
   and basic monitoring (Chapters 44–45).
3. **Write:** Create a portfolio page (or GitHub profile README) summarising your projects,
   the skills each demonstrates, and links to live demos.

::: tip
You can now build real projects. Chapter 50 shows how the same techniques power **real industry
systems** through case studies — and Chapter 51 prepares you to *talk* about all of this in **ML
interviews**.
:::
