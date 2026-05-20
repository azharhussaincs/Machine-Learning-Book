# Cloud ML

## Introduction

Training a modern deep-learning model can require dozens of GPUs running for days — hardware
that costs tens of thousands of dollars and would sit idle most of the time. Serving a
popular model might need to scale from 10 to 10,000 requests per second overnight. Few teams
can own and manage such infrastructure. The solution is the **cloud**: rent exactly the
compute you need, when you need it, and hand the operational burden to a provider.

**Cloud ML** means running your ML workloads — data storage, training, and serving — on
cloud platforms like **AWS**, **Google Cloud (GCP)**, and **Microsoft Azure**, using their
**managed services** built specifically for machine learning.

::: keyidea
The cloud turns expensive, fixed **capital** (buying GPUs) into flexible, pay-as-you-go
**operating** cost. You get **on-demand GPUs/TPUs**, **elastic scaling**, and **managed
services** that handle the undifferentiated heavy lifting (provisioning, scaling,
monitoring) — so you focus on the model, not the machines.
:::

By the end of this chapter you will be able to:

- Explain **why** most production ML runs in the cloud.
- Identify the major providers and their **managed ML platforms**.
- Understand the cloud ML **stack**: compute, storage, managed training, serving, AutoML,
  and pre-trained AI APIs.
- Reason about **cost** and when to use cloud vs local.

## Why the cloud for ML?

- **On-demand compute** — spin up powerful **GPUs/TPUs** for a few hours of training, then
  shut them down (pay only for what you use).
- **Elastic scaling** — serving infrastructure auto-scales up and down with traffic.
- **Managed services** — the provider handles provisioning, scaling, patching, and
  monitoring, so a small team can run production ML.
- **Storage & data** — cheap, durable storage for huge datasets, integrated with compute.
- **Pre-built AI** — ready-made APIs (vision, speech, translation, LLMs) you can call
  without training anything.
- **Global reach & reliability** — deploy close to users with high uptime.

## The cloud ML stack

![The cloud ML stack, from bottom to top: raw compute and storage; managed training; managed deployment (endpoints); AutoML; and ready-to-use pre-trained AI APIs. Higher layers require less ML expertise and infrastructure work.](assets/images/ch46_stack.png)

From lowest-level (most control) to highest-level (least effort):

1. **Compute & storage (IaaS)** — rent virtual machines (with GPUs) and object storage; you
   manage everything else. Most flexible, most work.
2. **Managed training** — submit a training job; the platform provisions machines, runs it,
   and tears them down (e.g. SageMaker Training, Vertex AI Training).
3. **Managed deployment (endpoints)** — deploy a model to a managed, auto-scaling endpoint
   with one command; the platform handles serving infrastructure.
4. **AutoML** — the platform automatically tries models and hyperparameters for your data
   (good for non-experts or strong baselines).
5. **Pre-trained AI APIs** — call ready-made models (image labelling, OCR, speech-to-text,
   translation, LLMs) via an API — no ML expertise needed.

## The major providers

| Provider | ML platform | Notable services |
|---|---|---|
| **AWS** | **SageMaker** | EC2 (GPU VMs), S3 (storage), Bedrock (LLMs) |
| **Google Cloud** | **Vertex AI** | TPUs, BigQuery ML, Vision/Speech APIs, Gemini |
| **Microsoft Azure** | **Azure ML** | Azure OpenAI Service, Cognitive Services |

All three offer the full stack (compute, storage, managed training/serving, AutoML,
pre-trained APIs). The concepts transfer between them; the names differ.

```python
# Illustrative: deploying to a managed endpoint is often just a few lines.
# (AWS SageMaker example — conceptual)
# from sagemaker.sklearn import SKLearnModel
# model = SKLearnModel(model_data="s3://bucket/model.tar.gz", role=ROLE, entry_point="inference.py")
# predictor = model.deploy(instance_type="ml.m5.large", initial_instance_count=1)
# predictor.predict(features)   # auto-scaled, managed serving
```

## Cost: the double-edged sword

The cloud's pay-as-you-go model is powerful but can be expensive if mismanaged:

- **GPUs are costly per hour** — shut them down when idle; use them only for training.
- **Spot/preemptible instances** — much cheaper compute for interruptible jobs (great for
  training).
- **Serverless inference** — pay per request, scales to zero when idle (cheap for sporadic
  traffic).
- **Watch storage and data-egress fees** — moving data out can cost more than expected.

::: warning
**A forgotten running GPU instance can cost thousands.** Always shut down resources you're
not using, set **budget alerts**, and prefer auto-scaling/serverless for variable traffic.
Cloud cost management ("FinOps") is a real and important skill.
:::

## Cloud vs local: when to use which

| Use the cloud when… | Use local when… |
|---|---|
| You need GPUs/TPUs you don't own | The dataset is small and CPU suffices |
| Workloads are large or bursty | You're learning/prototyping |
| You need to scale serving | Data can't leave premises (privacy/regulation) |
| You want managed services & uptime | You want zero ongoing cost |
| You need pre-trained AI APIs | Latency to cloud is unacceptable (use edge, Ch 47) |

::: tip
**Practical & debugging tips:** (1) Start small (free tiers, small instances) while learning;
scale up only when needed. (2) Use **spot/preemptible** instances for training to cut costs.
(3) Use **serverless/auto-scaling** endpoints for variable traffic. (4) **Set budget alerts**
and shut down idle GPUs. (5) Keep data and compute in the **same region** to avoid egress
costs/latency. (6) For quick wins, try **pre-trained AI APIs** or **AutoML** before building
custom models. (7) Containerise (Chapter 44) for portability across clouds.
:::

## Advantages, disadvantages, and use cases

| Advantages | Disadvantages |
|---|---|
| On-demand GPUs/TPUs; no hardware to own | Costs can spiral if unmanaged |
| Elastic scaling; managed services | Vendor lock-in risk |
| Pre-trained AI APIs & AutoML | Data privacy/compliance concerns |
| Global reliability | Requires cloud skills; egress fees |

**Use cases:** training large models, scalable model serving, big-data ML, calling
pre-trained AI APIs, AutoML baselines, and end-to-end managed ML platforms (SageMaker/Vertex/
Azure ML) for production.

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Leaving GPU instances running idle.** The most common (and expensive) cloud
mistake. Shut them down and set budget alerts.
:::

- **Mistake 2 — Ignoring data-egress and storage costs**, which can dominate the bill.
- **Mistake 3 — Over-provisioning** large instances for small jobs.
- **Mistake 4 — Not using spot/serverless** where appropriate.
- **Mistake 5 — Vendor lock-in** by relying on proprietary services without an exit plan
  (containers/open formats help).
- **Mistake 6 — Sending sensitive data to the cloud** without checking privacy/compliance
  rules.

## Best practices

- **Right-size** compute; use **spot** for training and **serverless/auto-scaling** for
  serving.
- **Shut down idle resources**; set **budget alerts**.
- **Keep data + compute co-located**; mind egress.
- **Use managed services and pre-trained APIs** to move fast.
- **Containerise** for portability and to reduce lock-in.
- **Check privacy/compliance** before moving sensitive data.

## Chapter Summary

- **Cloud ML** runs storage, training, and serving on providers (**AWS, GCP, Azure**),
  turning fixed hardware cost into flexible **pay-as-you-go** compute with **on-demand
  GPUs/TPUs** and **elastic scaling**.
- The **cloud ML stack** rises from **compute/storage** → **managed training** → **managed
  endpoints** → **AutoML** → **pre-trained AI APIs**, with higher layers needing less ML/ops
  effort.
- Each provider has a managed platform — **SageMaker (AWS)**, **Vertex AI (GCP)**, **Azure
  ML** — offering the full stack with different names.
- **Cost management is crucial**: use spot/serverless, shut down idle GPUs, set budget
  alerts, and watch egress/storage fees.
- Use the cloud for large/bursty workloads, scaling, and managed services; use **local/edge**
  for small jobs, privacy, or low latency.

---

::: {.qband}
Practice Zone — Chapter 46
:::

## Multiple-Choice Questions (MCQs)

**Q1.** A key advantage of cloud ML is:
a) Free forever  b) On-demand GPUs/TPUs without owning hardware  c) No need for data  d) No
internet needed

**Q2.** AWS's managed ML platform is:
a) Vertex AI  b) SageMaker  c) Azure ML  d) Colab

**Q3.** Google Cloud's managed ML platform is:
a) SageMaker  b) Vertex AI  c) Azure ML  d) Bedrock

**Q4.** Calling a ready-made vision/speech model via an API is using:
a) IaaS  b) Managed training  c) Pre-trained AI APIs  d) Spot instances

**Q5.** The most common expensive cloud mistake is:
a) Too little storage  b) Leaving GPU instances running idle  c) Using serverless  d) Setting
budget alerts

**Q6.** Cheap, interruptible compute for training jobs is called:
a) Reserved  b) Spot/preemptible instances  c) On-demand premium  d) Edge

**Q7.** AutoML automatically:
a) Deploys to edge  b) Tries models/hyperparameters for your data  c) Cleans data only  d)
Writes documentation

**Q8.** You should use local/on-premise instead of cloud when:
a) You need huge GPUs  b) Data can't leave premises for privacy/regulation  c) Traffic is
bursty  d) You want managed services

### MCQ Answers
**1:** b. **2:** b. **3:** b. **4:** c. **5:** b. **6:** b. **7:** b. **8:** b.

## Interview Questions (with answers)

**Q1. Why do most production ML systems run in the cloud?**
*Answer:* Because the cloud provides on-demand GPUs/TPUs without owning hardware, elastic
scaling for variable traffic, managed services that handle infrastructure (so small teams can
run production ML), cheap durable storage for big data, and ready-made AI APIs — all on a
pay-as-you-go model that converts capital cost into flexible operating cost.

**Q2. Describe the layers of the cloud ML stack.**
*Answer:* From low to high: raw compute and storage (IaaS — most control, most work); managed
training (submit a job, the platform provisions/tears down machines); managed deployment
(auto-scaling endpoints); AutoML (automated model/hyperparameter search); and pre-trained AI
APIs (call ready models with no ML expertise). Higher layers need less effort and expertise.

**Q3. How do you control cloud ML costs?**
*Answer:* Right-size instances, use spot/preemptible compute for interruptible training, use
serverless/auto-scaling endpoints that scale to zero for sporadic traffic, shut down idle
GPUs, set budget alerts, keep data and compute co-located to avoid egress fees, and prefer
pre-trained APIs/AutoML for quick wins.

**Q4. What is vendor lock-in and how do you mitigate it?**
*Answer:* Lock-in is over-dependence on a provider's proprietary services, making it hard/
costly to switch. Mitigate by containerising workloads (Docker), using open formats and
portable frameworks, abstracting cloud-specific code, and keeping an exit strategy — while
still benefiting from managed services where worthwhile.

## Scenario-Based Questions (with answers)

**Q1.** *You need to train a large model for a few hours but don't own GPUs. What's the
cost-effective cloud approach?*
*Answer:* Rent GPU instances on-demand (ideally spot/preemptible for big savings), run the
training job (or use managed training), then shut the instances down so you pay only for the
hours used — far cheaper than buying hardware that sits idle.

**Q2.** *Your prediction service gets sporadic, bursty traffic. How do you serve it
cost-effectively?*
*Answer:* Use serverless/auto-scaling inference that scales to zero when idle and up during
bursts, paying per request rather than for always-on servers — matching cost to actual usage.

**Q3.** *A hospital wants ML but cannot send patient data to the cloud due to regulation.
What are the options?*
*Answer:* Keep data and compute on-premise (local servers), use a private cloud, or use edge
deployment (Chapter 47); if cloud is needed, use compliant regions/services with strict
controls, anonymisation, and possibly federated learning so raw data never leaves premises.

## Logic-Based Questions (with answers)

**Q1.** Why does the cloud convert capital cost into operating cost, and why is that valuable?
*Answer:* Instead of buying expensive hardware upfront (capital), you rent it by the hour
(operating). This is valuable because ML compute needs are spiky — heavy during training,
light otherwise — so paying only for actual usage avoids costly idle hardware and lowers the
barrier to entry.

**Q2.** Why might pre-trained AI APIs be the best first choice for a common task?
*Answer:* They deliver strong results immediately with no training, data collection, or ML
expertise, at low effort — so for standard tasks (OCR, translation, image labelling) they're
faster and cheaper than building a custom model, which you'd only do if the API is
insufficient.

**Q3.** Why can keeping data and compute in the same region save money and time?
*Answer:* Moving data across regions or out of the cloud incurs egress fees and added latency;
co-locating storage and compute minimises data transfer cost and speeds up training/serving.

## Practical Questions (with answers)

**Q1.** Name the managed ML platform for each: AWS, GCP, Azure.
*Answer:* AWS → SageMaker; GCP → Vertex AI; Azure → Azure ML.

**Q2.** What kind of cloud instance is cheapest for an interruptible training job?
*Answer:* Spot (AWS) / preemptible (GCP) instances — heavily discounted compute that can be
reclaimed, suitable for fault-tolerant training.

**Q3.** Which serving option is most cost-effective for rare, sporadic requests?
*Answer:* Serverless inference, which scales to zero when idle and charges per request.

## Long Questions (with answers)

**Q1. Explain the cloud ML stack and how a team chooses the right level for their needs.**

*Answer:* The cloud ML stack spans levels of abstraction. At the bottom is **compute and
storage (IaaS)** — raw GPU/CPU virtual machines and object storage — offering maximum control
but requiring you to manage everything; suitable for custom or research workloads. Next,
**managed training** lets you submit a job and have the platform provision machines, run it,
and tear them down, removing infrastructure toil while keeping your own model code. Above
that, **managed deployment (endpoints)** turns a trained model into an auto-scaling,
monitored serving service with minimal effort. Higher still, **AutoML** automatically searches
models and hyperparameters for your data — ideal for non-experts or strong baselines — and at
the top, **pre-trained AI APIs** provide ready-made capabilities (vision, speech, translation,
LLMs) callable with no ML expertise at all. **Choosing a level** depends on expertise,
control, and effort trade-offs: use pre-trained APIs or AutoML for common tasks and speed; use
managed training/endpoints when you have custom models but want the provider to handle
infrastructure; and drop to raw compute only when you need full control. Most teams mix levels
— e.g. pre-trained APIs for some features and managed training/serving for custom models —
optimising for time-to-value, cost, and the skills available.

**Q2. Discuss cloud cost management for ML and the trade-offs of cloud vs local/on-premise.**

*Answer:* The cloud's **pay-as-you-go** model is powerful but can be costly if mismanaged, so
**cost management** is essential. GPUs are expensive per hour, so you should run them only for
training and **shut them down when idle**; use **spot/preemptible** instances for big savings
on interruptible jobs; use **serverless/auto-scaling** endpoints that scale to zero for
sporadic serving; **right-size** instances rather than over-provisioning; keep **data and
compute co-located** to avoid egress fees; and **set budget alerts** to catch runaway costs (a
forgotten GPU can cost thousands). On the **cloud vs local** trade-off: the **cloud** wins when
you need GPUs/TPUs you don't own, when workloads are large or bursty, when you must scale
serving, and when managed services and pre-trained APIs accelerate delivery — but it risks
spiralling costs, vendor lock-in, and data-privacy/compliance concerns. **Local/on-premise**
(or edge, Chapter 47) wins for small jobs where CPU suffices, for learning/prototyping with
zero ongoing cost, when data cannot leave premises for regulatory reasons, or when low latency
to a remote cloud is unacceptable. The pragmatic approach is to start small, use the cloud's
elasticity and managed services where they add value, manage costs deliberately, and keep
workloads portable (via containers) to limit lock-in.

## Exercises

1. List four reasons teams run ML in the cloud.
2. Name the managed ML platform for AWS, GCP, and Azure.
3. Explain the difference between IaaS, managed training, and pre-trained AI APIs.
4. Give three ways to reduce cloud ML costs.
5. State two situations where local/on-premise is preferable to the cloud.

## Mini-Project

**Project: Plan a cloud ML deployment.**

1. Pick a project (e.g. your deployed model from Chapter 44). Choose a cloud provider and
   sketch the architecture (storage, training, serving).
2. Decide which stack level fits each part (raw compute, managed training, managed endpoint,
   pre-trained API, AutoML) and justify.
3. Estimate the cost drivers (compute hours, storage, requests, egress) and list three
   cost-saving choices.
4. (Stretch, if you have a free-tier account) Deploy a small model to a managed endpoint or
   call a pre-trained AI API.
5. Write a one-page architecture + cost plan. Save in `my-ml-journey/`.

## Assignments

1. **Research:** Compare SageMaker, Vertex AI, and Azure ML on one dimension (e.g. managed
   training or AutoML) in half a page.
2. **Conceptual:** Write one page on cloud cost management for ML, including spot instances,
   serverless inference, and budget alerts.
3. **Hands-on (optional):** Use a cloud free tier or Google Colab (free GPUs) to train a model
   and note the experience vs local.

::: tip
The cloud centralises ML in big data centres. But sometimes you need ML to run **on the
device itself** — your phone, a camera, a car — for speed, privacy, or offline use. Chapter
47, **Edge AI**, covers running models at the edge.
:::
