# Neural Networks & Deep Learning Foundations

## Introduction

Welcome to **Part VI** — **Deep Learning**, the technology behind the most spectacular AI
of our era: image recognition, speech, translation, self-driving perception, and the Large
Language Models powering modern chatbots. At its heart is one idea you already met in
Chapter 3: the **artificial neuron**. Stack enough of them in **layers**, and you get a
**neural network** capable of learning almost any pattern.

The inspiration is the human brain, which has ~86 billion neurons connected in a vast
network. Artificial neural networks are a *loose* mathematical caricature of this — but
that simple idea, scaled up with data and compute (the "perfect storm" of Chapter 2), has
revolutionised AI.

::: keyidea
A neural network is just **layers of simple neurons**, each computing a weighted sum
followed by a non-linear "activation". Individually trivial; stacked together they learn
**hierarchies of features** — edges → shapes → objects in vision, letters → words →
meaning in language. Depth is what makes it "deep" learning.
:::

By the end of this chapter you will be able to:

- Understand the **artificial neuron** and **activation functions**.
- Understand **multi-layer networks (MLPs)** and the **forward pass**.
- Explain *why depth* and *why non-linearity* matter.
- Build a neural network in **PyTorch** and know when to choose deep learning.

## From neuron to network

### The artificial neuron

Recall the perceptron (Chapter 3). A neuron takes inputs, multiplies each by a **weight**,
adds a **bias**, and passes the result through an **activation function** φ:

<div class="equation"><img class="eq" src="assets/images/eq_ch32_neuron.png" alt="neuron output"></div>

![A biological neuron (left) loosely inspires the artificial neuron (right): inputs are weighted, summed with a bias, and passed through an activation function to produce the output.](assets/images/ch32_neuron.png)

```python
import numpy as np
def relu(z): return max(0.0, z)
def sigmoid(z): return 1 / (1 + np.exp(-z))

x = np.array([1.0, 2.0]); w = np.array([0.5, 0.2]); b = 0.1
z = float(np.dot(w, x) + b)          # the weighted sum + bias
print("z = w·x + b =", round(z, 3))
print("ReLU(z) =", round(relu(z), 3))
print("sigmoid(z) =", round(sigmoid(z), 3))
```

**Output:**
```text
z = w·x + b = 1.0
ReLU(z) = 1.0
sigmoid(z) = 0.731
```

That weighted-sum-then-activate is *exactly* logistic regression (Chapter 18) when the
activation is the sigmoid. **A single neuron is a tiny linear model.** The power comes from
combining many.

### Activation functions: why non-linearity is essential

Without a non-linear activation, stacking layers would be pointless — a stack of linear
functions is still just a linear function. **Activations inject non-linearity**, letting
networks learn curved, complex patterns (remember XOR from Chapter 3).

![Common activation functions. Sigmoid squashes to (0,1); tanh to (−1,1); ReLU passes positives and zeros negatives (the modern default); softmax turns scores into class probabilities.](assets/images/ch32_activations.png)

| Activation | Formula / behaviour | Use |
|---|---|---|
| **Sigmoid** | squashes to (0, 1) | output layer for binary probability |
| **Tanh** | squashes to (−1, 1) | hidden layers (older) |
| **ReLU** | `max(0, z)` | the modern **default** for hidden layers |
| **Softmax** | turns scores into probabilities summing to 1 | output layer for multiclass |

**ReLU** is so popular because it's simple, fast, and avoids the "vanishing gradient"
problem (Chapter 33) that plagued sigmoid/tanh in deep networks.

<div class="equation"><img class="eq" src="assets/images/eq_ch32_relu.png" alt="ReLU"></div>
<div class="equation"><img class="eq" src="assets/images/eq_ch32_softmax.png" alt="softmax"></div>

## Multi-layer networks (MLPs)

Connect many neurons in **layers** and you get a **Multi-Layer Perceptron (MLP)** — a
*feedforward* neural network:

![A multi-layer perceptron: an input layer (the features), one or more hidden layers of neurons, and an output layer. Each connection has a weight; data flows left to right in the forward pass.](assets/images/ch32_mlp.png)

- **Input layer** — one node per feature (no computation; just the inputs).
- **Hidden layers** — neurons that transform the data; *more/wider hidden layers = more
  capacity*. "Deep" learning means **many** hidden layers.
- **Output layer** — produces the prediction (1 node + sigmoid for binary; N nodes +
  softmax for N-class; 1 linear node for regression).

### The forward pass

Making a prediction is the **forward pass**: feed inputs to the first layer, each layer
computes its neurons' outputs and passes them to the next, until the output layer produces
the answer. It's just repeated matrix multiplications (Chapter 5) and activations — which
is why GPUs (built for matrix math) accelerate deep learning.

## Why depth? Hierarchical feature learning

The magic of deep networks is **representation learning** — they learn *useful features
automatically*, layer by layer, instead of you hand-crafting them (Chapter 12):

- In a face recogniser: early layers learn **edges**, middle layers learn **eyes/noses**,
  later layers learn **whole faces**.
- In language: early layers learn **word patterns**, deeper layers learn **grammar and
  meaning**.

::: keyidea
This automatic, hierarchical feature learning is *the* superpower of deep learning. For
images, text, and audio, networks discover better features than humans could design —
which is why deep learning dominates these "unstructured" domains (while tree ensembles
often still win on tabular data, Chapter 23–24).
:::

## Practical: build a neural network in PyTorch

Let's train a real MLP on the breast-cancer data using **PyTorch**, the leading deep-
learning framework.

```python
import torch, torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
torch.manual_seed(0)

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc = StandardScaler().fit(X_tr)                      # always scale for neural nets
X_tr_t = torch.tensor(sc.transform(X_tr), dtype=torch.float32)
X_te_t = torch.tensor(sc.transform(X_te), dtype=torch.float32)
y_tr_t = torch.tensor(y_tr, dtype=torch.float32).view(-1, 1)

# Define the network: 30 inputs -> 16 hidden (ReLU) -> 1 output (sigmoid)
model = nn.Sequential(
    nn.Linear(30, 16), nn.ReLU(),
    nn.Linear(16, 1),  nn.Sigmoid())

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)   # the optimiser (Ch 33)
loss_fn = nn.BCELoss()                                       # binary cross-entropy (Ch 18)

for epoch in range(100):                # training loop
    optimizer.zero_grad()               # reset gradients
    output = model(X_tr_t)              # forward pass
    loss = loss_fn(output, y_tr_t)      # how wrong are we?
    loss.backward()                     # backprop: compute gradients (Ch 33)
    optimizer.step()                    # update the weights

with torch.no_grad():                                       # evaluate
    pred = (model(X_te_t) > 0.5).float().view(-1).numpy()
print("MLP test accuracy:", round(float((pred == y_te).mean()), 3))
```

**Output:**
```text
MLP test accuracy: 0.947
```

### Line-by-line explanation

- **`nn.Sequential(...)`** stacks layers: a `Linear` layer (30→16) with **ReLU**, then a
  `Linear` (16→1) with **Sigmoid** for a probability — exactly the architecture in the
  diagram.
- **`Adam`** is a smart gradient-descent optimiser (Chapter 33); **`BCELoss`** is the
  binary cross-entropy loss from Chapter 18.
- **The training loop** repeats the universal pattern from Chapter 5: *forward pass →
  compute loss → `backward()` (backprop) → `optimizer.step()` (update weights)*. We'll open
  up `backward()` in Chapter 33.
- The network reached **0.947** accuracy — comparable to the classic models, on a tabular
  problem where they're already strong. Deep learning's real edge appears on images, text,
  and audio (Chapters 34–40).

::: tip
**Practical & debugging tips:** (1) **Always scale features** for neural nets. (2) **ReLU**
for hidden layers, **sigmoid/softmax** for outputs, **linear** for regression outputs. (3)
Start simple (one hidden layer) and grow only if needed. (4) If loss is `nan`, lower the
learning rate (Chapter 5). (5) Set seeds for reproducibility. (6) For tabular data, compare
against Random Forest/XGBoost — they often win; reserve deep nets for unstructured data.
:::

## Deep learning vs classic ML — when to use which

| Use deep learning when… | Use classic ML when… |
|---|---|
| Data is unstructured (images, text, audio) | Data is tabular/structured |
| You have lots of data | Data is small/medium |
| You have GPU compute | Compute is limited |
| Features are hard to hand-craft | Interpretability matters |
| Maximum accuracy on perception tasks | Fast training & simple deployment |

## Advantages, disadvantages, and use cases

| Advantages | Disadvantages |
|---|---|
| Learns features automatically | Needs lots of data & compute |
| State-of-the-art on images/text/audio | "Black box" — hard to interpret |
| Very flexible (any architecture) | Many hyperparameters; slow to train |
| Scales with data and compute | Easy to overfit small data |

**Use cases:** image classification/detection, speech recognition, machine translation,
chatbots/LLMs, recommendation, generative AI — covered across Parts VI–VII.

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Forgetting to scale inputs.** Neural nets are gradient-based and train
poorly on unscaled features.
:::

- **Mistake 2 — No non-linear activation** in hidden layers (collapses to a linear model).
- **Mistake 3 — Using deep learning on small tabular data** where trees win and overfitting
  is severe.
- **Mistake 4 — Wrong output activation** (e.g. sigmoid for multiclass instead of softmax).
- **Mistake 5 — Going too deep/wide too soon**, causing overfitting and slow training.
- **Mistake 6 — Expecting interpretability** from a deep net (it's largely a black box).

## Best practices

- **Scale inputs**; use **ReLU** hidden, task-appropriate output activations.
- **Start small** and add capacity only as needed.
- **Use proven frameworks** (PyTorch/TensorFlow) and set seeds.
- **Reserve deep learning for unstructured data** or large datasets.
- **Watch for overfitting** (we add regularization in Chapter 33).
- **Compare against classic ML** baselines, especially on tabular data.

## Chapter Summary

- A **neural network** is layers of **artificial neurons**, each computing
  `a = φ(w·x + b)` — a weighted sum plus bias through a non-linear **activation**.
- **Activations** (sigmoid, tanh, **ReLU** as the default, **softmax** for multiclass
  output) inject the **non-linearity** that lets stacked layers learn complex patterns.
- An **MLP** has **input, hidden, and output layers**; prediction is the **forward pass**
  (matrix multiplications + activations). **Depth** enables **hierarchical feature
  learning** — the superpower behind vision and language AI.
- We built and trained an MLP in **PyTorch** (0.947 on breast cancer) using the universal
  loop *forward → loss → backward → update*.
- Use **deep learning for unstructured data and large datasets**; classic ML often wins on
  small tabular data. Always scale inputs.

---

::: {.qband}
Practice Zone — Chapter 32
:::

## Multiple-Choice Questions (MCQs)

**Q1.** A single artificial neuron computes:
a) A random number  b) A weighted sum plus bias, through an activation  c) A cluster  d) A
distance

**Q2.** Without a non-linear activation, a deep network is equivalent to:
a) A decision tree  b) A single linear model  c) A clustering algorithm  d) Random guessing

**Q3.** The most common hidden-layer activation today is:
a) Sigmoid  b) Tanh  c) ReLU  d) Softmax

**Q4.** For multiclass classification, the output layer typically uses:
a) ReLU  b) Sigmoid  c) Softmax  d) Linear

**Q5.** "Deep" learning refers to networks with:
a) Large inputs  b) Many hidden layers  c) Big learning rates  d) Many features only

**Q6.** Making a prediction by passing data through the layers is the:
a) Backward pass  b) Forward pass  c) Loss  d) Gradient

**Q7.** Deep learning's key advantage on images/text is:
a) Interpretability  b) Automatic hierarchical feature learning  c) Tiny data needs  d) No
compute

**Q8.** Before training a neural network you should:
a) Add labels to test set  b) Scale the inputs  c) Remove the output layer  d) Nothing

### MCQ Answers
**1:** b. **2:** b. **3:** c. **4:** c. **5:** b. **6:** b. **7:** b. **8:** b.

## Interview Questions (with answers)

**Q1. What is an artificial neuron and how does it relate to logistic regression?**
*Answer:* A neuron computes a weighted sum of its inputs plus a bias, then applies an
activation function: a = φ(w·x + b). With a sigmoid activation, a single neuron is exactly
logistic regression. Neural networks gain power by stacking many such neurons in layers.

**Q2. Why are non-linear activation functions necessary?**
*Answer:* Because composing linear functions yields another linear function, so without
non-linearity a deep network could only represent linear relationships. Activations like
ReLU introduce non-linearity, enabling the network to learn complex, curved patterns (e.g.
XOR) that linear models cannot.

**Q3. What is the difference between sigmoid, ReLU, and softmax, and where is each used?**
*Answer:* Sigmoid squashes a value to (0,1) and is used for binary-probability outputs.
ReLU outputs max(0,z), is the default for hidden layers (fast, avoids vanishing gradients).
Softmax converts a vector of scores into probabilities summing to 1, used for multiclass
output layers.

**Q4. What does "depth" give a neural network?**
*Answer:* Depth enables hierarchical feature learning: successive layers build
increasingly abstract representations (edges → parts → objects in vision). This lets deep
networks automatically discover features that would be hard to hand-engineer, which is why
they excel on unstructured data.

**Q5. When should you prefer deep learning over classic ML?**
*Answer:* For unstructured data (images, text, audio), when you have large datasets and GPU
compute, and when good features are hard to hand-craft. For small/medium tabular data,
classic models (especially gradient boosting) are often more accurate, faster, and more
interpretable.

## Scenario-Based Questions (with answers)

**Q1.** *Your neural network won't learn — the loss barely changes and inputs range from 0
to 100,000. What's the first thing to fix?*
*Answer:* Scale the inputs (e.g. StandardScaler). Neural nets are gradient-based and train
poorly on unscaled, large-range features; standardising usually lets training proceed
properly.

**Q2.** *On a 1,000-row tabular dataset, your deep net overfits and underperforms a Random
Forest. Why, and what do you recommend?*
*Answer:* Deep nets are data-hungry and overfit small tabular data, where tree ensembles
typically win. Recommend using Random Forest/XGBoost here, or if using a net, keep it small
with strong regularization — but classic ML is the better tool for this case.

**Q3.** *You built a network with three Linear layers and no activations between them and
it performs like a linear model. Why?*
*Answer:* Stacked linear layers without non-linear activations collapse mathematically into
a single linear transformation, so the network can only learn linear relationships. Insert
non-linear activations (e.g. ReLU) between layers.

## Logic-Based Questions (with answers)

**Q1.** Why is a single sigmoid neuron equivalent to logistic regression?
*Answer:* Both compute σ(w·x + b) — a weighted sum through a sigmoid to produce a
probability — and are trained by minimising cross-entropy. The neuron is literally the same
model.

**Q2.** In the example, w·x + b = 1.0 and ReLU(1.0)=1.0 but ReLU(−1.0)=0. What property of
ReLU does this show?
*Answer:* ReLU passes positive inputs unchanged and zeros out negative inputs (max(0,z)).
This sparsity and linear-positive behaviour helps gradients flow and is why ReLU is the
default hidden activation.

**Q3.** Why do GPUs accelerate neural networks so much?
*Answer:* The forward and backward passes are dominated by large matrix multiplications,
and GPUs are massively parallel hardware optimised for exactly that bulk linear algebra,
making them far faster than CPUs for these operations.

## Practical Questions (with answers)

**Q1.** Write a PyTorch MLP with one hidden layer of 16 ReLU units for 10 inputs and a
binary output.
*Answer:* `nn.Sequential(nn.Linear(10,16), nn.ReLU(), nn.Linear(16,1), nn.Sigmoid())`.

**Q2.** What are the four steps inside a PyTorch training loop?
*Answer:* `optimizer.zero_grad()` (reset gradients), forward pass (`output = model(X)`),
`loss.backward()` (compute gradients), `optimizer.step()` (update weights).

**Q3.** Which activation would you use on the output layer for a 5-class classifier?
*Answer:* Softmax (5 output units), typically paired with cross-entropy loss.

## Long Questions (with answers)

**Q1. Explain the structure and forward pass of a multi-layer neural network, and why
activation functions and depth are essential.**

*Answer:* A multi-layer perceptron has an **input layer** (one node per feature, no
computation), one or more **hidden layers** of neurons, and an **output layer**. Each neuron
computes a weighted sum of its inputs plus a bias and applies a non-linear **activation**:
a = φ(w·x + b). In the **forward pass**, data flows left to right: the first hidden layer
computes its neurons' activations from the inputs, those become the inputs to the next
layer, and so on until the output layer produces the prediction — mechanically a sequence
of matrix multiplications interleaved with activation functions (which is why GPUs
accelerate it). **Activation functions are essential** because, without non-linearity,
composing linear layers would collapse into a single linear function, limiting the network
to linear relationships; non-linearities like ReLU let the network represent complex,
curved decision boundaries (solving problems like XOR). **Depth is essential** because
stacking layers enables **hierarchical feature learning**: early layers learn simple
patterns (edges, character n-grams), and deeper layers compose them into abstract concepts
(objects, meaning). This automatic discovery of useful representations — rather than
hand-engineered features — is what makes deep networks so powerful on unstructured data like
images, audio, and language.

**Q2. Compare deep learning with classic machine learning: their strengths, weaknesses, and
when to choose each.**

*Answer:* **Classic ML** (linear/logistic regression, SVMs, tree ensembles like Random
Forest and XGBoost) works directly on usually hand-engineered or tabular features; it is
fast to train, needs less data, is often interpretable, and frequently achieves the best
accuracy on **structured/tabular** problems. Its weakness is that it relies on good features
and struggles with raw unstructured data. **Deep learning** uses many-layered neural
networks that **learn features automatically**, achieving state-of-the-art results on
**unstructured** data — images, text, audio — and scaling well as data and compute grow. Its
weaknesses are that it is data-hungry and compute-intensive, easy to overfit on small
datasets, has many hyperparameters, trains slowly, and is largely a **black box** (hard to
interpret). **Choosing:** prefer deep learning when the data is unstructured, abundant, and
GPU compute is available, or when features are hard to hand-craft and maximum perception
accuracy matters; prefer classic ML when the data is tabular and small-to-medium, when
interpretability or fast/simple deployment matters, or as a strong baseline — and in
practice, especially on tabular data, compare both empirically (No Free Lunch, Chapter 16)
since gradient boosting often beats neural nets there.

## Exercises

1. Compute a neuron's output for x=[2,1], w=[0.3,0.4], b=−0.1 with a ReLU activation.
2. Explain why removing all activations makes a deep network linear.
3. State which activation you'd use for: a hidden layer, a binary output, a 4-class output,
   a regression output.
4. Describe, in your own words, hierarchical feature learning in a face recogniser.
5. Give two cases where classic ML beats deep learning.

## Mini-Project

**Project: Your first neural network.**

1. Pick a dataset (tabular like breast cancer, or `make_moons` for a non-linear 2-D
   problem).
2. Build an MLP in PyTorch (or Keras) with one hidden layer; scale the inputs; train it.
3. Compare its accuracy to logistic regression and a Random Forest on the same data.
4. Experiment with the number of hidden units and layers; note the effect on accuracy and
   overfitting.
5. Write a short report on what you observed. Save in `my-ml-journey/`.

## Assignments

1. **Coding:** Implement a 2-layer neural network's **forward pass** from scratch in NumPy
   (matrix multiplications + ReLU + sigmoid) and verify the output shape on sample data.
2. **Coding:** Train MLPs of increasing depth/width on `make_moons` and plot the decision
   boundaries to see capacity grow.
3. **Conceptual:** Write one page explaining why non-linear activations and depth are what
   make neural networks powerful, with examples.

::: tip
You've built a neural network — but *how* does `loss.backward()` actually teach it?
Chapter 33, **Training Deep Networks**, opens the black box: backpropagation, optimisers
(SGD, Adam), and the regularization tricks (dropout, batch norm, early stopping) that make
deep learning work.
:::
