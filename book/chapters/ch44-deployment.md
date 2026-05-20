# Model Deployment

## Introduction

Welcome to **Part VIII** — turning models into **real, working products**. A model sitting
in a Jupyter notebook helps no one. **Deployment** is the process of making your trained
model available to **real users or other software**, so it can make predictions on demand.
This is where many ML projects *fail* — not because the model was bad, but because it never
got reliably into production.

::: keyidea
A model has **two lives**: **training** (done once, offline, on historical data) and
**inference/serving** (done forever, online, on new data). Deployment is about the second:
**save the trained model**, wrap it in a **service** (an API or app), and make it reliable,
fast, and reproducible for real requests.
:::

By the end of this chapter you will be able to:

- **Save and load** trained models (serialization).
- Serve a model as a **REST API** with **FastAPI** (and **Flask**).
- Build an instant interactive demo with **Streamlit**.
- Understand deployment architecture and best practices.

## Step 1: Save the trained model (serialization)

You train once, then **save** the model to a file so the serving code can load it without
retraining. Use **`joblib`** (great for scikit-learn) or `pickle`; deep-learning frameworks
have their own (`torch.save`, Keras `.save`).

```python
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True)
model = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42)).fit(X, y)

joblib.dump(model, "model.joblib")          # SAVE the whole pipeline
loaded = joblib.load("model.joblib")        # LOAD it (as a server would at startup)
print("prediction:", load_iris().target_names[loaded.predict([[5.1, 3.5, 1.4, 0.2]])[0]])
```

**Output:**
```text
prediction: setosa
```

::: keyidea
**Save the entire pipeline, not just the model.** Because we saved the `StandardScaler` +
classifier together, the loaded object applies the *exact same preprocessing* to new data
automatically. Saving only the model and forgetting the preprocessing is a top deployment
bug (the model gets raw, unscaled inputs and fails silently).
:::

## Step 2: Serve it as an API

The standard way to deploy is a **REST API**: your model runs on a server, and clients
(websites, apps, other services) send input data via HTTP and get predictions back as JSON.

![Deployment architecture: a client sends input data to your model API over HTTP; the API loads the saved model, runs inference, and returns a prediction as JSON. The model lives behind the API, reusable by any client.](assets/images/ch44_deployment.png)

### FastAPI (the modern choice)

**FastAPI** is fast, modern, and auto-generates interactive API docs. (`pip install fastapi
uvicorn`.)

```python
# save as app.py, then run:  uvicorn app:app --reload
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("model.joblib")          # load once at startup (not per request!)

class IrisInput(BaseModel):                  # define & validate the input schema
    features: list[float]

@app.post("/predict")
def predict(data: IrisInput):
    pred = model.predict([data.features])[0]
    return {"prediction": int(pred)}         # return JSON
```

A client then POSTs `{"features": [5.1, 3.5, 1.4, 0.2]}` to `/predict` and receives
`{"prediction": 0}`. FastAPI gives you free interactive docs at `/docs`.

### Flask (the classic)

**Flask** is the simple, long-established alternative (`pip install flask`):

```python
from flask import Flask, request, jsonify
import joblib
app = Flask(__name__)
model = joblib.load("model.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    features = request.get_json()["features"]
    return jsonify({"prediction": int(model.predict([features])[0])})
```

## Step 3: Or build an instant demo with Streamlit

**Streamlit** turns a Python script into an interactive **web app** in minutes — perfect for
demos, dashboards, and letting non-programmers try your model (`pip install streamlit`):

```python
# save as app.py, then run:  streamlit run app.py
import streamlit as st
import joblib
model = joblib.load("model.joblib")

st.title("Iris Classifier")
sl = st.slider("sepal length", 4.0, 8.0, 5.1)   # interactive widgets
sw = st.slider("sepal width", 2.0, 4.5, 3.5)
pl = st.slider("petal length", 1.0, 7.0, 1.4)
pw = st.slider("petal width", 0.1, 2.5, 0.2)
if st.button("Predict"):
    pred = model.predict([[sl, sw, pl, pw]])[0]
    st.success(f"Predicted species: {pred}")
```

## Choosing the right tool

| Tool | Best for | Note |
|---|---|---|
| **FastAPI** | Production APIs | Fast, async, auto-docs — the modern default |
| **Flask** | Simple APIs, legacy | Mature, huge ecosystem |
| **Streamlit** | Demos, dashboards, internal apps | Fastest to a UI; not for high-scale APIs |
| **Gradio** | Quick ML demos (Hugging Face) | Great for sharing model demos |

## Packaging and scaling: Docker & beyond

Real deployments add layers for reliability and scale:

- **Docker** — package the app + dependencies into a **container** that runs identically
  anywhere ("works on my machine" solved).
- **Cloud hosting** — deploy the container to AWS/GCP/Azure (Chapter 46).
- **Scaling** — multiple instances behind a **load balancer**; auto-scale with demand.
- **Batch vs real-time** — serve live requests (real-time) or score large datasets on a
  schedule (batch).

## Beyond serving: it's a system

Deployment is more than an API. Production needs **input validation**, **logging**,
**monitoring** (latency, errors, prediction drift), **versioning** (of model and data), and
a **rollback** plan — the concerns of **MLOps** (Chapter 45).

::: tip
**Practical & debugging tips:** (1) **Load the model once at startup**, not on every request
(huge speed difference). (2) **Save the whole pipeline** (preprocessing + model). (3)
**Validate inputs** (FastAPI/Pydantic does this for you). (4) **Pin your library versions** —
a model trained with one scikit-learn version may not load in another. (5) Containerise with
**Docker** for reproducibility. (6) Add **health-check** and **logging** endpoints. (7) Test
the API with sample requests before going live.
:::

## Advantages, disadvantages, and use cases

| Tool/approach | Strengths | Weaknesses |
|---|---|---|
| REST API (FastAPI/Flask) | Standard, language-agnostic, scalable | Needs infra/ops |
| Streamlit/Gradio | Instant UI, great demos | Not for high-scale production APIs |
| Docker | Reproducible, portable | Extra learning curve |
| Batch scoring | Efficient for bulk | Not real-time |

**Use cases:** prediction microservices, ML-powered web/mobile apps, dashboards, internal
tools, and embedding models into larger software systems.

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Forgetting preprocessing at inference.** If you scaled/encoded during training
but feed raw inputs to the deployed model, predictions are wrong. Save and apply the **full
pipeline**.
:::

- **Mistake 2 — Loading the model on every request** (slow) instead of once at startup.
- **Mistake 3 — Library-version mismatch** between training and serving (pin versions).
- **Mistake 4 — No input validation** (garbage in → crashes or silent bad output).
- **Mistake 5 — No monitoring/logging**, so failures and drift go unnoticed (Chapter 45).
- **Mistake 6 — Using Streamlit for a high-traffic production API** (use FastAPI).

## Best practices

- **Serialize the full pipeline**; load it **once** at startup.
- **Use FastAPI** for production APIs (validation + auto-docs); Streamlit/Gradio for demos.
- **Pin dependency versions**; **containerise with Docker**.
- **Validate inputs**, add **logging, monitoring, and health checks**.
- **Version your models and data**; have a **rollback** plan.
- **Test** the endpoint thoroughly before release.

## Chapter Summary

- **Deployment** makes a trained model available to real users/software for inference —
  where many ML projects succeed or fail.
- The flow: **serialize** the trained model/pipeline (e.g. `joblib.dump`), then **serve** it.
- Serve as a **REST API** with **FastAPI** (modern, validated, auto-docs) or **Flask**
  (classic); build instant UIs/demos with **Streamlit** or **Gradio**.
- **Save the whole pipeline** (preprocessing + model), **load once at startup**, **pin
  versions**, and **containerise with Docker** for reproducibility.
- Production is a **system**: add input validation, logging, monitoring, versioning, and
  rollback — the realm of **MLOps** (Chapter 45).

---

::: {.qband}
Practice Zone — Chapter 44
:::

## Multiple-Choice Questions (MCQs)

**Q1.** Deployment means:
a) Training a model  b) Making a trained model available for predictions  c) Cleaning data
d) Tuning hyperparameters

**Q2.** Which library is commonly used to save scikit-learn models?
a) NumPy  b) joblib  c) Matplotlib  d) Flask

**Q3.** The modern, fast Python framework for production ML APIs is:
a) Streamlit  b) FastAPI  c) Pandas  d) NLTK

**Q4.** For a quick interactive demo UI, you'd use:
a) FastAPI  b) Streamlit  c) Docker  d) joblib

**Q5.** You should load the model:
a) On every request  b) Once at server startup  c) Never  d) Only in training

**Q6.** Saving only the model but not the preprocessing leads to:
a) Faster serving  b) Wrong predictions on raw inputs  c) Smaller files only  d) Nothing

**Q7.** Docker is used to:
a) Train models  b) Package the app + dependencies for reproducible deployment  c) Plot data
d) Tune models

**Q8.** A REST API typically returns predictions as:
a) Images  b) JSON  c) CSV files  d) Plots

### MCQ Answers
**1:** b. **2:** b. **3:** b. **4:** b. **5:** b. **6:** b. **7:** b. **8:** b.

## Interview Questions (with answers)

**Q1. Walk me through deploying a trained model.**
*Answer:* Serialize the trained pipeline (e.g. `joblib.dump`), write a serving app (FastAPI/
Flask) that loads the model once at startup and exposes a `/predict` endpoint accepting JSON
input and returning a JSON prediction, validate inputs, containerise with Docker for
reproducibility, deploy to a server/cloud, and add logging and monitoring. Pin library
versions to match training.

**Q2. Why save the whole pipeline rather than just the model?**
*Answer:* Because inference must apply the *same* preprocessing (scaling, encoding) as
training. Saving the full pipeline ensures the deployed model receives correctly transformed
inputs automatically; saving only the estimator and feeding raw data causes silent, wrong
predictions.

**Q3. Why load the model once at startup instead of per request?**
*Answer:* Loading a model from disk is relatively slow; doing it on every request adds heavy
latency and wastes resources. Loading once at startup keeps the model in memory so each
request is fast.

**Q4. When would you use Streamlit vs FastAPI?**
*Answer:* Use Streamlit (or Gradio) for quick interactive demos, dashboards, and internal
tools where you want a UI fast. Use FastAPI for production APIs that other software calls,
needing speed, input validation, scalability, and documentation.

**Q5. What role does Docker play in deployment?**
*Answer:* Docker packages the application, its dependencies, and environment into a portable
container that runs identically across machines and clouds, eliminating "works on my machine"
problems and making deployments reproducible and scalable.

## Scenario-Based Questions (with answers)

**Q1.** *Your model works in the notebook but the deployed API returns nonsense predictions.
You scaled features during training. What likely went wrong?*
*Answer:* The deployment probably applies the model to raw, unscaled inputs because the
scaler wasn't included. Save and serve the entire pipeline (scaler + model) so inference
preprocessing matches training.

**Q2.** *Your API is slow under load and you notice it reads the model file on every request.
How do you fix it?*
*Answer:* Load the model once at application startup (a module-level/global load), so it stays
in memory and each request just runs inference. Also consider multiple worker processes and a
load balancer for scale.

**Q3.** *A model trained months ago won't load in production and throws a version error. What
happened and how do you prevent it?*
*Answer:* A library version mismatch (e.g. different scikit-learn version) broke
deserialization. Prevent it by pinning exact dependency versions (requirements/lockfile) and
containerising with Docker so training and serving use identical environments.

## Logic-Based Questions (with answers)

**Q1.** Why does deployment, not modelling, often determine whether an ML project delivers
value?
*Answer:* Because a model only creates value when it's used on real, new data; without
reliable deployment, even an accurate model never reaches users or systems, so the project's
benefit is never realised.

**Q2.** Why is input validation important in a prediction API?
*Answer:* Real clients send malformed, missing, or wrong-type data; without validation the API
can crash or, worse, return confident but meaningless predictions. Validation rejects bad
input clearly and protects the model from receiving inputs it can't handle.

**Q3.** Why does containerisation improve reproducibility?
*Answer:* A container bundles the exact code, dependencies, and environment, so the app runs
the same way regardless of the host machine — removing differences in installed versions and
OS that otherwise cause inconsistent behaviour.

## Practical Questions (with answers)

**Q1.** Write code to save and load a scikit-learn pipeline with joblib.
*Answer:* `joblib.dump(model, "model.joblib")` to save; `model = joblib.load("model.joblib")`
to load.

**Q2.** What's the minimal FastAPI endpoint to serve a prediction?
*Answer:* A `@app.post("/predict")` function that takes a validated input model, calls
`model.predict([...])`, and returns a JSON dict like `{"prediction": int(pred)}` (with the
model loaded once at startup).

**Q3.** Which command runs a FastAPI app named `app` in `app.py`?
*Answer:* `uvicorn app:app --reload`.

## Long Questions (with answers)

**Q1. Describe the end-to-end process of deploying a machine-learning model as an API, and
the pitfalls to avoid.**

*Answer:* Deployment begins by **serializing the trained pipeline** — saving the full
preprocessing + model object (e.g. via `joblib.dump`) so inference reproduces training's
transformations. Next, build a **serving application**, typically a **REST API** with FastAPI
(or Flask): it **loads the model once at startup**, exposes a `/predict` endpoint that accepts
input as JSON, **validates** that input (Pydantic in FastAPI), runs `model.predict`, and
returns the result as JSON. The app is then **containerised with Docker** (bundling code,
dependencies, and environment with **pinned versions**) for reproducibility, and deployed to a
server or cloud, often behind a **load balancer** with multiple instances for scale. Finally,
production-grade systems add **logging, monitoring** (latency, errors, prediction drift),
**model/data versioning**, and a **rollback** plan. The main **pitfalls**: forgetting to
include preprocessing (raw inputs → wrong predictions), loading the model per request (slow),
library-version mismatches between training and serving (load failures), missing input
validation (crashes or silent bad outputs), and no monitoring (failures and drift go
unnoticed). Avoiding these turns a notebook model into a reliable product.

**Q2. Compare FastAPI, Flask, and Streamlit for deploying ML models, and explain when to use
each.**

*Answer:* **FastAPI** is a modern, high-performance Python framework for building **REST
APIs**; it offers asynchronous support, automatic input validation via Pydantic, and
auto-generated interactive documentation, making it the **default choice for production ML
APIs** that other software (websites, apps, services) will call at scale. **Flask** is the
older, simpler, very mature framework for building APIs and web apps; it has a huge ecosystem
and is fine for **simple or legacy services**, though it lacks FastAPI's built-in validation,
async, and auto-docs. **Streamlit** is different in purpose: it turns a Python script into an
**interactive web app/UI** in minutes with sliders, inputs, and charts — ideal for **demos,
dashboards, internal tools, and letting non-programmers try a model** — but it isn't designed
to be a high-throughput API backend. **When to use each:** choose **FastAPI** for scalable
production prediction services; choose **Flask** for simple APIs or when an existing Flask
stack is in place; choose **Streamlit** (or Gradio) to quickly showcase or share a model
interactively. In practice, teams often expose the model via a FastAPI service for production
and build a separate Streamlit/Gradio demo for stakeholders.

## Exercises

1. List the steps to deploy a trained model as an API.
2. Explain why you must serialize the whole pipeline, not just the estimator.
3. Write a minimal FastAPI `/predict` endpoint (pseudocode is fine).
4. When would you choose Streamlit over FastAPI?
5. What does Docker solve in deployment?

## Mini-Project

**Project: Deploy your model three ways.**

1. Train a model (e.g. the Iris or heart-disease pipeline) and save it with `joblib`.
2. Build a **FastAPI** `/predict` endpoint; test it with a sample JSON request (via `/docs` or
   `curl`).
3. Build a **Streamlit** app with input widgets that calls the model and shows the prediction.
4. (Stretch) Write a `Dockerfile` to containerise the FastAPI app.
5. Document the steps and a sample request/response in `my-ml-journey/`.

## Assignments

1. **Coding:** Build and run a FastAPI service for a model you trained earlier; include input
   validation and a health-check endpoint.
2. **Coding:** Build a Streamlit demo for the same model with sliders/inputs and a prediction
   display.
3. **Conceptual:** Write one page on the difference between a model's "training life" and
   "serving life", and the deployment pitfalls to avoid.

::: tip
Deploying a model once is just the start. Keeping many models reliable, monitored, and
continuously updated in production is **MLOps** — Chapter 45 — the engineering discipline that
makes ML systems sustainable at scale.
:::
