# MLOps & Production ML Systems

## Introduction

Deploying *one* model once (Chapter 44) is hard enough. Running *many* models reliably for
*years* — as data changes, code evolves, and teams grow — is a whole engineering discipline:
**MLOps** (Machine Learning Operations). It applies the lessons of **DevOps** (software
operations) to the unique challenges of ML, where the system depends not just on code but
also on **data** and **models** that change over time.

A famous Google paper noted that the ML model is often a **tiny fraction** of a real
production system; the vast majority is data pipelines, serving infrastructure, monitoring,
and maintenance. MLOps is about all of that.

::: keyidea
**MLOps = the practices and tools to deploy, monitor, and maintain ML systems reliably and
repeatably.** Its defining challenge: unlike normal software, ML systems **decay over time**
because the world (the data) changes — so production ML is never "done"; it must be
continuously monitored and retrained.
:::

By the end of this chapter you will be able to:

- Explain *why* MLOps exists and how ML differs from normal software.
- Describe the **production ML lifecycle** and its components.
- Understand **versioning, CI/CD, experiment tracking, model registries, and feature
  stores**.
- Detect **data/concept drift** and plan **monitoring and retraining**.

## Why ML systems are different (and decay)

Normal software does the same thing forever unless you change the code. ML systems are
different in two big ways:

1. **They depend on data, not just code.** The same code with different data behaves
   differently. You must version and track **data** and **models**, not just code.
2. **They decay.** As the real world changes, the data the model sees in production drifts
   away from its training data, and accuracy silently degrades. This is **model decay**, and
   it's why production ML needs constant attention.

![The MLOps lifecycle is a continuous loop: data → train → validate → deploy → monitor → (drift detected) → retrain → redeploy. Unlike normal software, ML systems must be continuously maintained because data changes over time.](assets/images/ch45_lifecycle.png)

## The two kinds of drift

- **Data drift** — the *input* distribution changes (e.g. a sensor is recalibrated, user
  demographics shift, prices inflate). The model now sees inputs unlike its training data.
- **Concept drift** — the *relationship* between inputs and target changes (e.g. spam tactics
  evolve, customer behaviour shifts after a pandemic). The old rules no longer hold.

Both degrade performance and trigger the need to **retrain**.

### Practical: detecting data drift

A simple, common approach: statistically compare the **production** feature distribution to
the **training** distribution. The Kolmogorov–Smirnov (KS) test flags significant
differences.

```python
import numpy as np
from scipy import stats
np.random.seed(0)

train      = np.random.normal(50, 10, 1000)    # training feature distribution
prod_ok    = np.random.normal(50, 10, 1000)    # production: same distribution
prod_drift = np.random.normal(60, 12, 1000)    # production: SHIFTED (drift)

def check(name, ref, new):
    stat, p = stats.ks_2samp(ref, new)         # compare two distributions
    print(f"{name}: KS p-value={p:.4f} -> "
          f"{'DRIFT detected!' if p < 0.05 else 'no drift'}")

check("batch 1 (similar)", train, prod_ok)
check("batch 2 (shifted)", train, prod_drift)
```

**Output:**
```text
batch 1 (similar): KS p-value=0.2635 -> no drift
batch 2 (shifted): KS p-value=0.0000 -> DRIFT detected!
```

### Explanation

- For the similar batch, the KS test found no significant difference (p=0.26) → **no drift**.
- For the shifted batch (mean moved 50→60), the test fired (p≈0) → **drift detected!** In
  production, this alarm would trigger investigation and likely **retraining**.
- This is the essence of **monitoring**: keep comparing live data to training data (and
  tracking live accuracy where labels are available) so you catch decay *before* users do.

## The components of an MLOps system

| Component | Purpose | Example tools |
|---|---|---|
| **Version control** | Track code, **data**, and **models** | Git, **DVC**, **MLflow** |
| **Experiment tracking** | Record every run's params/metrics | MLflow, Weights & Biases |
| **Model registry** | Store, version, stage models | MLflow Registry |
| **CI/CD pipelines** | Automate test/build/deploy | GitHub Actions, Jenkins |
| **Pipeline orchestration** | Automate data/train workflows | Airflow, Kubeflow |
| **Feature store** | Consistent features for train & serve | Feast |
| **Monitoring** | Track drift, performance, latency | Evidently, Prometheus |

::: keyidea
Three things must be **versioned and reproducible** in ML, not just one: **code, data, and
model**. If you can't reproduce exactly which data + code produced a model, you can't debug,
audit, or roll it back. This "reproducibility" is the heart of MLOps.
:::

## CI/CD for ML

In software, **CI/CD** (Continuous Integration / Continuous Deployment) automates testing and
releasing code. ML adds **continuous training (CT)**: automated pipelines that, when triggered
(by new data or detected drift), **retrain**, **validate** (does the new model beat the old on
held-out data and checks?), and **deploy** the model — with the ability to **roll back** if it
underperforms.

## Monitoring and retraining

Production monitoring tracks:

- **Operational metrics:** latency, throughput, errors, uptime.
- **Data quality:** missing values, schema changes, out-of-range inputs.
- **Drift:** input distribution shifts (as above).
- **Model performance:** accuracy/error *when ground-truth labels arrive* (often delayed).

When monitoring detects decay, you **retrain** — manually, on a schedule, or automatically
(triggered by drift) — then validate and redeploy.

::: tip
**Practical & debugging tips:** (1) **Monitor from day one** — drift is inevitable. (2)
Version **data and models**, not just code (DVC + MLflow). (3) Use a **feature store** or
shared transformation code so training and serving compute features identically
(training/serving skew is a classic bug). (4) Automate **retraining + validation**, but
always **validate the new model beats the old** before deploying. (5) Keep a **rollback**
path. (6) Start simple — even basic logging + a weekly drift check beats nothing.
:::

## Technical debt in ML

ML systems accumulate hidden **technical debt**: tangled data dependencies,
training/serving skew, undeclared consumers of a model's output, glue code, and
"pipeline jungles". Good MLOps practices — reproducibility, testing, monitoring, and clear
interfaces — keep this debt manageable so the system stays maintainable.

## Advantages, disadvantages, and use cases

| Advantages of MLOps | Challenges |
|---|---|
| Reliable, reproducible ML in production | Added engineering complexity |
| Catches model decay (monitoring) | Tooling/infra investment |
| Faster, safer updates (CI/CD/CT) | Cultural change (DS + Eng + Ops) |
| Auditability & rollback | Overkill for tiny one-off projects |

**Use cases:** any ML running in production at scale — fraud detection, recommendations,
forecasting, credit scoring, and enterprise ML platforms.

## Common mistakes & misconceptions

::: warning
**Mistake 1 — "Deploy and forget".** Models decay as data drifts. Without monitoring,
performance silently degrades and you find out from angry users, not dashboards.
:::

- **Mistake 2 — Versioning only code**, not data and models (can't reproduce or roll back).
- **Mistake 3 — Training/serving skew** — features computed differently in training vs
  production.
- **Mistake 4 — Auto-deploying a retrained model without validating** it beats the current
  one.
- **Mistake 5 — No rollback plan** when a new model misbehaves.
- **Mistake 6 — Over-engineering** MLOps for a tiny project (match effort to scale).

## Best practices

- **Monitor in production** (drift, performance, data quality) from day one.
- **Version code, data, and models**; ensure reproducibility.
- **Eliminate training/serving skew** (shared feature code / feature store).
- **Automate** training, validation, and deployment (CI/CD/CT), with **validation gates**.
- **Keep a rollback path** and an audit trail.
- **Match MLOps maturity to the project's scale and risk.**

## Chapter Summary

- **MLOps** brings DevOps discipline to ML, addressing ML's unique trait: systems **depend on
  data** and **decay over time** as the world changes, so production ML is never "done".
- The production lifecycle is a **loop**: data → train → validate → deploy → **monitor** →
  (drift) → **retrain** → redeploy.
- Two decays: **data drift** (input distribution changes) and **concept drift** (input→target
  relationship changes); both are detected by monitoring (e.g. a KS test fired on shifted
  data while passing on similar data).
- Core components: **version control of code/data/models**, **experiment tracking**, **model
  registry**, **CI/CD + continuous training**, **feature stores**, and **monitoring**;
  reproducibility is central.
- Best practice: **monitor from day one**, version everything, avoid training/serving skew,
  automate retraining with **validation gates**, and keep a **rollback** path.

---

::: {.qband}
Practice Zone — Chapter 45
:::

## Multiple-Choice Questions (MCQs)

**Q1.** MLOps is best described as:
a) A model architecture  b) DevOps practices applied to ML systems  c) A dataset  d) A loss
function

**Q2.** ML systems differ from normal software because they:
a) Never change  b) Depend on data and decay over time  c) Have no code  d) Don't need
testing

**Q3.** When the input data distribution changes in production, it's called:
a) Concept drift  b) Data drift  c) Overfitting  d) Underfitting

**Q4.** When the input→output relationship changes, it's called:
a) Data drift  b) Concept drift  c) Leakage  d) Pruning

**Q5.** In ML you must version:
a) Only code  b) Code, data, and models  c) Only data  d) Nothing

**Q6.** A model registry is used to:
a) Plot data  b) Store, version, and stage models  c) Clean data  d) Tune hyperparameters

**Q7.** "Deploy and forget" is risky because:
a) Models get faster  b) Models decay as data drifts  c) Code disappears  d) Nothing

**Q8.** Training/serving skew means:
a) Same features both sides  b) Features computed differently in training vs serving  c) Two
models  d) A drift test

### MCQ Answers
**1:** b. **2:** b. **3:** b. **4:** b. **5:** b. **6:** b. **7:** b. **8:** b.

## Interview Questions (with answers)

**Q1. What is MLOps and why is it needed?**
*Answer:* MLOps is the set of practices and tools for deploying, monitoring, and maintaining
ML systems reliably and reproducibly — DevOps adapted to ML. It's needed because ML systems
depend on data (not just code) and decay over time as data drifts, so they require continuous
monitoring, versioning of code/data/models, automated retraining, and rollback — beyond what
normal software engineering covers.

**Q2. What is model decay and what causes it?**
*Answer:* Model decay is the gradual decline of a deployed model's performance over time. It's
caused by data drift (the input distribution changes) and concept drift (the relationship
between inputs and target changes), so the model's learned patterns no longer match reality.
It's addressed by monitoring and retraining.

**Q3. What's the difference between data drift and concept drift?**
*Answer:* Data drift is a change in the input feature distribution (e.g. new user
demographics), while concept drift is a change in the mapping from inputs to the target (e.g.
fraud tactics evolve so the same features now mean something different). Both degrade
performance and may require retraining.

**Q4. Why must you version data and models, not just code?**
*Answer:* Because an ML system's behaviour is determined by data + code + model together.
Without versioning data and models, you can't reproduce a result, debug a regression, audit a
decision, or roll back to a known-good model — all essential for reliable production ML.

**Q5. How would you detect that a deployed model needs retraining?**
*Answer:* Monitor for data drift (statistical tests comparing production vs training feature
distributions, e.g. KS test), track model performance when labels become available, and watch
data-quality and operational metrics. A drift alarm or performance drop triggers
investigation and retraining (with validation before redeploying).

## Scenario-Based Questions (with answers)

**Q1.** *A fraud model's accuracy quietly dropped over six months and no one noticed until
losses spiked. What MLOps practice was missing?*
*Answer:* Production monitoring (and drift detection). With monitoring of drift and performance
plus alerts, the decay would have been caught early and triggered retraining before causing
losses — the cost of "deploy and forget".

**Q2.** *Your retrained model passed offline tests but performed worse in production. What
should your pipeline have done before deploying?*
*Answer:* Used a validation gate: automatically compare the new model against the current
production model on a fair held-out/online test and only deploy if it's genuinely better,
with a rollback path if it underperforms. Also check for training/serving skew.

**Q3.** *Predictions differ between the data scientist's notebook and production for the same
input. What's a likely cause?*
*Answer:* Training/serving skew — features are computed or preprocessed differently in
production than in training (or a different model/library version). Fix by sharing the exact
preprocessing pipeline/feature code (or a feature store) and pinning versions.

## Logic-Based Questions (with answers)

**Q1.** Why is production ML "never done", unlike a finished piece of normal software?
*Answer:* Because its performance depends on data that keeps changing; even with frozen code,
real-world drift erodes accuracy over time, so the system must be continuously monitored and
periodically retrained — an ongoing process rather than a one-time delivery.

**Q2.** In the drift example, why did the KS test fire for the shifted batch but not the
similar one?
*Answer:* The KS test measures whether two samples come from the same distribution. The
similar batch matched the training distribution (high p-value → no significant difference),
while the shifted batch (mean 50→60) differed significantly (p≈0), signalling drift.

**Q3.** Why can a retrained model that scores higher offline still be worse to deploy?
*Answer:* Offline scores may not reflect production conditions (different live distribution,
training/serving skew, or metric mismatch), and the new model might regress on important
segments. Hence the need for validation gates and the ability to roll back.

## Practical Questions (with answers)

**Q1.** Which statistical test can compare two feature distributions to detect drift?
*Answer:* The Kolmogorov–Smirnov (KS) two-sample test (`scipy.stats.ks_2samp`); others include
Population Stability Index (PSI) and chi-square for categoricals.

**Q2.** Name two tools used for experiment tracking / model versioning.
*Answer:* MLflow and Weights & Biases (DVC for data/model versioning).

**Q3.** What does "continuous training (CT)" add to CI/CD for ML?**
*Answer:* Automated retraining pipelines triggered by new data or drift, which retrain,
validate, and (if better) deploy the model — extending CI/CD's code automation to the
model/data lifecycle.

## Long Questions (with answers)

**Q1. Explain why MLOps is necessary, how production ML differs from normal software, and the
key components of an MLOps system.**

*Answer:* **MLOps is necessary** because deploying and maintaining ML in production involves
challenges normal software engineering doesn't face. **The key difference** is that ML
systems depend on **data and a learned model**, not just code: the same code with different
data behaves differently, and — crucially — ML systems **decay over time** because the real
world changes, causing **data drift** (input distributions shift) and **concept drift**
(input→target relationships change) that silently erode accuracy. So production ML is a
continuous **loop** — data → train → validate → deploy → monitor → retrain — rather than a
one-time release. To manage this, an MLOps system has several **components**: **version
control** of code, **data**, and **models** (Git, DVC, MLflow) for reproducibility;
**experiment tracking** to record parameters and metrics of every run; a **model registry**
to store, version, and stage models; **CI/CD plus continuous training** pipelines that
automate testing, retraining, validation, and deployment with rollback; **pipeline
orchestration** (Airflow/Kubeflow) for data and training workflows; **feature stores** to
ensure training and serving compute features identically (avoiding training/serving skew);
and **monitoring** of drift, data quality, latency, and model performance. Together these
practices make ML systems reliable, reproducible, auditable, and maintainable at scale.

**Q2. Describe model decay and a complete strategy for monitoring and retraining a production
model.**

*Answer:* **Model decay** is the gradual decline in a deployed model's performance as the
world drifts away from its training data — through **data drift** (the input distribution
changes, e.g. a recalibrated sensor or new customer base) and **concept drift** (the
input→target relationship changes, e.g. evolving fraud tactics). A complete strategy starts
with **monitoring from day one**: track **operational metrics** (latency, errors, uptime),
**data quality** (missing values, schema/range changes), **drift** (statistically compare
live feature distributions to training, e.g. a KS test that fires when a feature's mean
shifts), and **model performance** (accuracy/error once ground-truth labels arrive, which is
often delayed). When monitoring signals decay — a drift alarm or a performance drop — the
system triggers **retraining** (manually, on a schedule, or automatically) on fresh, properly
versioned data. Critically, the retrained model passes through a **validation gate**: it must
demonstrably beat the current production model on a fair held-out (or shadow/online) test and
pass quality checks **before** it is promoted, and a **rollback path** remains ready if it
misbehaves. All artifacts — data, code, model — are versioned for reproducibility and audit.
This closed loop of monitor → detect → retrain → validate → deploy → monitor keeps the system
accurate despite a changing world.

## Exercises

1. Explain in your own words why ML systems decay but normal software doesn't.
2. Give an example each of data drift and concept drift.
3. List three artifacts you must version in ML and why.
4. Describe a validation gate and why it matters before deploying a retrained model.
5. Name three components of an MLOps system and their purposes.

## Mini-Project

**Project: A simple monitoring & drift check.**

1. Train a model and save its training feature statistics/distributions.
2. Simulate production batches: some from the same distribution, some shifted.
3. Implement a drift check (KS test or PSI) that flags batches that differ significantly from
   training.
4. Log predictions and (when available) accuracy over the batches; plot performance over time.
5. Write a short "retraining policy": when would you retrain, and how would you validate the
   new model? Save in `my-ml-journey/`.

## Assignments

1. **Coding:** Build a drift-detection function for multiple features and report which
   features drifted on a simulated shifted dataset.
2. **Research:** Pick one MLOps tool (MLflow, DVC, or Evidently) and write half a page on what
   it does and where it fits in the lifecycle.
3. **Conceptual:** Write one page on "why production ML is never finished", covering decay,
   monitoring, and retraining.

::: tip
MLOps keeps models healthy in production. But *where* do they run? Chapter 46, **Cloud ML**,
covers training and serving models on cloud platforms (AWS, GCP, Azure) — the infrastructure
behind most modern ML systems.
:::
