# Industry Case Studies

## Introduction

You've learned the techniques and built projects. Now let's see how the *same* tools you've
studied power **real systems at scale** across industries — and extract the lessons that
separate ML that *works in production* from ML that fails. These case studies show that ML is
not abstract: it diagnoses disease, prevents fraud, recommends what billions of people watch,
and keeps factories running.

::: keyidea
Across every industry, successful ML shares the same pattern: a **clearly defined business
problem**, **good data**, the **right (often simple) model**, **careful evaluation**, solid
**deployment/monitoring**, and **domain expertise**. The algorithm is rarely the hard part —
the data, the framing, and the engineering around it are.
:::

By the end of this chapter you will be able to:

- Describe how ML is applied across major industries.
- Connect each application to the techniques from this book.
- Extract the **cross-cutting lessons** of real-world ML success and failure.

## ML across industries

![Machine Learning across industries: healthcare, finance, retail/e-commerce, transport, entertainment, manufacturing, and agriculture — each using the same core techniques (classification, forecasting, recommendation, vision, NLP) for domain-specific problems.](assets/images/ch50_industries.png)

### Healthcare

- **What:** disease detection from medical images (X-rays, CT, retinal scans), early-warning
  risk scores, drug discovery, hospital demand forecasting.
- **How:** **CNNs / transfer learning** (Ch 34, 40) for imaging; classification/regression
  (Ch 17–24) for risk; generative models for molecules (Ch 36).
- **Impact:** earlier diagnoses, fewer errors, accelerated research — but high stakes demand
  rigorous validation, explainability (Ch 48), and human-in-the-loop oversight.

### Finance

- **What:** fraud detection, credit scoring, algorithmic trading, customer churn,
  anti-money-laundering.
- **How:** imbalanced **classification** and **anomaly detection** (Ch 25, 27) for fraud;
  **gradient boosting** (Ch 24) for credit; **time series** (Ch 42) for markets.
- **Impact:** billions saved from fraud; faster, broader credit access — with strong fairness
  and regulatory requirements (Ch 48).

### Retail & E-commerce

- **What:** product **recommendations**, demand forecasting, dynamic pricing, customer
  segmentation, supply-chain optimisation.
- **How:** **recommender systems** (Ch 41), **clustering** (Ch 27), **time-series
  forecasting** (Ch 42).
- **Impact:** a large share of sales/engagement comes from recommendations (Amazon, etc.);
  better stock and pricing decisions.

### Transportation

- **What:** self-driving perception, ride-hailing ETA and matching, route optimisation,
  predictive maintenance.
- **How:** **CNNs + object detection** (Ch 40), **deep RL** (Ch 31), **time-series** demand
  prediction.
- **Impact:** safer/assisted driving, efficient logistics, less downtime.

### Entertainment & Streaming

- **What:** content recommendation (Netflix, YouTube, Spotify), thumbnail/content
  optimisation, generative content.
- **How:** **recommender systems + deep learning** (Ch 41), **generative AI** (Ch 43).
- **Impact:** huge engagement gains — most viewing/listening is driven by recommendations.

### Manufacturing & Agriculture

- **What:** defect detection (vision), predictive maintenance, yield prediction, crop/disease
  detection from images.
- **How:** **CNNs** (Ch 34, 40), **time-series** (Ch 42), classification.
- **Impact:** higher quality, less waste and downtime, better yields.

## Mini case study: Netflix recommendations

- **Problem:** keep users engaged by surfacing content they'll love from a vast catalog.
- **Approach:** large-scale **collaborative filtering** and deep learning over viewing
  history, combined with content features (hybrid, Ch 41); heavy **A/B testing** and
  monitoring.
- **Lesson:** the famous **Netflix Prize** showed **ensembles and matrix factorization** win,
  but also that **production constraints** (latency, freshness, diversity) matter as much as
  raw accuracy — and that optimising *engagement* requires care to avoid filter bubbles
  (Ch 48).

## Cross-cutting lessons from real deployments

::: keyidea
The most valuable lessons aren't about algorithms:

1. **Data quality beats model complexity.** Most wins come from better data, not fancier
   models.
2. **Start simple.** A logistic regression or gradient-boosting baseline often ships first
   and sometimes wins.
3. **Domain expertise is decisive** — understanding the problem and features matters more than
   ML tricks.
4. **Deployment & monitoring are the hard part** (Chapters 44–45) — many models never reach
   production or silently decay.
5. **Ethics and fairness are real risks** (Chapter 48), not afterthoughts.
6. **Measure the business metric**, not just ML metrics — accuracy that doesn't move the KPI
   is worthless.
:::

## Why ML projects fail (the other side)

Studies repeatedly find most ML projects never deliver business value. Common causes:

- **Poorly defined problem** or wrong success metric.
- **Bad/insufficient data**, or leakage that inflates offline results.
- **No path to deployment** ("notebook to nowhere").
- **No monitoring** → silent model decay.
- **Ignoring stakeholders, domain experts, or ethics.**

Knowing these failure modes — and the success patterns above — is what makes you valuable.

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Believing the algorithm is the hard part.** In real systems, data, problem
framing, deployment, and monitoring dominate. Companies win with good data and engineering,
not exotic models.
:::

- **Mistake 2 — Optimising ML metrics that don't move the business KPI.**
- **Mistake 3 — Underestimating deployment/monitoring effort.**
- **Mistake 4 — Skipping domain experts.**
- **Mistake 5 — Ignoring fairness/regulation** in high-stakes domains.
- **Mistake 6 — Assuming offline accuracy guarantees production success.**

## Best practices (from industry)

- **Frame the problem with stakeholders** and tie success to a **business KPI**.
- **Invest in data quality**; start with a simple, shippable baseline.
- **Bring in domain expertise** for features and validation.
- **Plan deployment and monitoring from the start** (MLOps).
- **A/B test** changes; measure real impact.
- **Address fairness, privacy, and regulation** for high-stakes use.

## Chapter Summary

- ML powers real systems across **healthcare, finance, retail, transport, entertainment,
  manufacturing, and agriculture** — using the same techniques from this book applied to
  domain problems.
- The **same success pattern** recurs: clear problem, good data, the right (often simple)
  model, honest evaluation, solid deployment/monitoring, and **domain expertise**.
- **Netflix** illustrates large-scale hybrid recommendation plus the reality that production
  constraints and engagement ethics matter as much as accuracy.
- The biggest lessons are **non-algorithmic**: data quality beats complexity, start simple,
  domain expertise is decisive, deployment/monitoring is the hard part, and ethics and the
  **business KPI** must drive the work.
- Most ML projects **fail** from poor problem framing, bad data/leakage, no deployment path,
  no monitoring, or ignored stakeholders/ethics — knowing this makes you effective.

---

::: {.qband}
Practice Zone — Chapter 50
:::

## Multiple-Choice Questions (MCQs)

**Q1.** In real industry ML, the hardest part is usually:
a) Choosing an exotic algorithm  b) Data, problem framing, deployment & monitoring  c) Plotting
d) Naming the model

**Q2.** Disease detection from X-rays primarily uses:
a) ARIMA  b) CNNs / transfer learning  c) K-Means  d) Naive Bayes only

**Q3.** Bank fraud detection is characterised by:
a) Balanced data  b) Highly imbalanced data (anomaly/imbalanced classification)  c) No labels
d) Images only

**Q4.** Netflix/YouTube/Spotify rely heavily on:
a) Recommendation systems  b) Edge AI only  c) ARIMA  d) GANs only

**Q5.** A key cross-cutting lesson is:
a) Complexity beats data  b) Data quality beats model complexity  c) Skip baselines  d) Avoid
domain experts

**Q6.** Most ML projects fail because of:
a) Too-simple models  b) Poor problem framing, bad data, no deployment/monitoring  c) Too much
documentation  d) Using pipelines

**Q7.** You should ultimately optimise:
a) Only ML accuracy  b) The business KPI / real impact  c) Code length  d) Number of features

**Q8.** Predictive maintenance in manufacturing uses mainly:
a) Recommenders  b) Time-series + classification  c) Translation  d) Clustering only

### MCQ Answers
**1:** b. **2:** b. **3:** b. **4:** a. **5:** b. **6:** b. **7:** b. **8:** b.

## Interview Questions (with answers)

**Q1. Give an example of ML in a specific industry and the techniques involved.**
*Answer:* In finance, fraud detection uses highly imbalanced classification and anomaly
detection (precision/recall/AUC over accuracy), often with gradient boosting; credit scoring
uses classification with strong fairness/regulatory constraints; markets use time-series
forecasting. The impact is large (billions saved), and ethics/regulation are central.

**Q2. Why do so many ML projects fail to deliver value?**
*Answer:* Common causes include a poorly defined problem or wrong metric, insufficient or
biased data (or leakage inflating offline results), no path to deployment, no monitoring (so
models decay), and ignoring stakeholders, domain experts, or ethics. Success depends far more
on these than on the algorithm.

**Q3. What's the most important lesson from real-world ML deployments?**
*Answer:* That data quality, problem framing, deployment, monitoring, and domain expertise
matter more than model complexity. A simple model on good data, deployed and monitored,
tied to a real business KPI, beats a sophisticated model that never ships or optimises the
wrong metric.

**Q4. Why optimise the business KPI rather than just ML metrics?**
*Answer:* Because the goal is business value, not a number. A model can improve accuracy/AUC
yet not move revenue, retention, or cost; conversely a modest model targeting the right metric
can deliver large value. ML metrics are proxies; the KPI is the objective.

## Scenario-Based Questions (with answers)

**Q1.** *A retailer wants to "use AI" but has no specific goal. How do you start?*
*Answer:* Work with stakeholders to define a concrete, valuable problem and a measurable KPI
(e.g. reduce stockouts, increase recommendation click-through), assess data availability and
quality, then start with a simple baseline tied to that KPI — rather than chasing "AI" for its
own sake.

**Q2.** *A hospital's diagnostic model is accurate but clinicians don't trust it. What's
missing?*
*Answer:* Explainability and integration: provide interpretable explanations (SHAP/
interpretable models), validate across patient subgroups (fairness), keep clinicians in the
loop, and fit the model into their workflow — trust and adoption matter as much as accuracy in
high-stakes domains.

**Q3.** *An ML model performed great offline but added no business value after launch. What
likely happened?*
*Answer:* Possible causes: it optimised an ML metric that didn't move the KPI; offline results
were inflated by leakage; the deployment/serving differed from training; or it wasn't actually
adopted/monitored. Tie work to the KPI, prevent leakage, ensure correct deployment, and measure
real impact (A/B test).

## Logic-Based Questions (with answers)

**Q1.** Why does data quality often matter more than model choice in industry?
*Answer:* Models can only learn what's in the data; better, cleaner, more representative data
improves every model, while a fancier algorithm on poor data still fails ("garbage in, garbage
out"). Most real-world gains therefore come from improving data.

**Q2.** Why is deployment/monitoring considered the "hard part" of industrial ML?
*Answer:* Because turning a model into a reliable, scalable, monitored service — that keeps
working as data drifts — requires substantial engineering and ongoing maintenance, and many
projects stall here ("notebook to nowhere") or decay silently without monitoring.

**Q3.** Why might the same recommendation model that boosts engagement also cause harm?
*Answer:* Optimising purely for short-term engagement can create filter bubbles, promote
addictive or polarising content, and ignore user well-being — so business and ethical metrics
must be balanced (Chapter 48).

## Practical Questions (with answers)

**Q1.** Match the technique to the industry use: CNN, recommender, time-series, anomaly
detection.
*Answer:* CNN → medical imaging / defect detection; recommender → streaming/e-commerce;
time-series → demand/sales/market forecasting; anomaly detection → fraud/fault detection.

**Q2.** What should you tie an industry ML project's success to?
*Answer:* A concrete business KPI (e.g. fraud losses prevented, retention, conversion,
downtime reduced) — not just an ML metric.

**Q3.** Name two reasons an offline-accurate model fails in production.
*Answer:* Data leakage inflating offline results, and data drift / training-serving skew (plus
optimising the wrong metric or lack of adoption).

## Long Questions (with answers)

**Q1. Survey how machine learning is applied across at least four industries, naming the
techniques and impact in each.**

*Answer:* **Healthcare** uses CNNs and transfer learning (Ch 34, 40) to detect disease in
medical images, classification/regression for risk scores, and generative models for drug
discovery, enabling earlier diagnoses and accelerated research — under strict validation,
explainability, and oversight given the high stakes. **Finance** applies imbalanced
classification and anomaly detection (Ch 25, 27) for fraud, gradient boosting (Ch 24) for
credit scoring, and time-series methods (Ch 42) for markets, saving billions and broadening
credit access, with heavy fairness and regulatory requirements. **Retail and e-commerce** rely
on recommender systems (Ch 41), clustering for segmentation (Ch 27), and time-series
forecasting (Ch 42) for demand and pricing, driving a large share of sales through
personalised recommendations and improving inventory decisions. **Entertainment/streaming**
(Netflix, YouTube, Spotify) use large-scale hybrid recommendation and deep learning (Ch 41,
43) to maximise engagement, where most consumption is recommendation-driven. Additional
industries — **transportation** (CNN-based perception and deep RL for self-driving and routing,
Ch 31, 40), **manufacturing** (vision-based defect detection and predictive maintenance), and
**agriculture** (crop/disease detection from images) — all reuse the same core techniques. The
throughline is that identical methods, applied with domain understanding to well-framed
problems and good data, create value across wildly different sectors.

**Q2. What separates successful industrial ML from failed projects? Discuss the key lessons.**

*Answer:* Success and failure in industrial ML hinge far more on **process and data** than on
algorithms. **Successful** projects share a pattern: a **clearly defined problem** tied to a
measurable **business KPI** (not just an ML metric); **good, representative data**, since data
quality beats model complexity; starting with a **simple, shippable baseline** and only adding
complexity if it helps; leveraging **domain expertise** for features and validation; robust
**deployment and monitoring** (MLOps, Ch 44–45) so the model reaches production and doesn't
silently decay; **A/B testing** to measure real impact; and attention to **fairness, privacy,
and regulation** in high-stakes settings (Ch 48). **Failed** projects typically suffer the
opposites: a vague problem or wrong metric; insufficient or biased data, or **leakage** that
inflates offline scores; **no path to deployment** ("notebook to nowhere"); **no monitoring**,
leading to decay; optimising an ML metric that doesn't move the business; and ignoring
stakeholders, domain experts, or ethics. The decisive lesson is that the **algorithm is rarely
the bottleneck** — framing, data, engineering, monitoring, and alignment with real business and
ethical goals determine whether ML delivers value, which is precisely why these "boring"
skills make a practitioner valuable.

## Exercises

1. For three industries, name one ML application and the technique it uses.
2. List four cross-cutting lessons from real ML deployments.
3. Give three common reasons ML projects fail.
4. Explain why "data quality beats model complexity".
5. Why tie a project's success to a business KPI rather than ML accuracy alone?

## Mini-Project

**Project: Analyse a real case study.**

1. Pick a real company/ML system (Netflix, Tesla Autopilot, a bank's fraud system, etc.).
2. Research and document: the problem, the ML approach/techniques, the data, and the impact.
3. Identify which book chapters the techniques come from.
4. Note the challenges (deployment, fairness, scale) and lessons.
5. Write a one-page case-study report. Save in `my-ml-journey/`.

## Assignments

1. **Research:** Write a two-page case study of an ML system in an industry you care about,
   connecting it to this book's techniques and the success/failure lessons.
2. **Analysis:** Find a documented ML *failure* (biased system, failed deployment) and analyse
   what went wrong and how it could have been prevented.
3. **Conceptual:** Write one page on "why most ML value comes from data and deployment, not
   algorithms".

::: tip
You've seen how ML works in industry. Now Chapter 51 prepares you to **get the job**: a
comprehensive **ML interview preparation** guide covering concepts, coding, and the questions
you'll actually be asked.
:::
