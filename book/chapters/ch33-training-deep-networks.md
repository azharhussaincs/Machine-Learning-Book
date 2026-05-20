# Training Deep Networks

## Introduction

In Chapter 32 you built a neural network and called `loss.backward()` — but what does that
*actually do*? This chapter opens the black box of **how neural networks learn**. The
answer combines three things you've already met: a **loss function** (Chapter 18), the
**chain rule** (Chapter 5), and **gradient descent** (Chapter 5). Together they form
**backpropagation** — the algorithm that powers all of deep learning.

We'll also cover the practical machinery that makes training *actually work*: smart
**optimisers** (like Adam), the right **batch sizes**, and **regularization** tricks
(dropout, batch norm, early stopping) that stop deep networks from overfitting.

::: keyidea
Training = **forward pass** (predict) → **compute loss** (how wrong) → **backward pass**
(backprop: find each weight's gradient via the chain rule) → **update** (gradient descent).
Repeat over many **mini-batches** and **epochs**. Backprop is just the chain rule applied
efficiently, layer by layer, from output back to input.
:::

By the end of this chapter you will be able to:

- Explain **backpropagation** as the chain rule applied through the network.
- Choose **loss functions** and **optimisers** (SGD, Momentum, **Adam**).
- Understand **batch size**, **epochs**, and mini-batch training.
- Diagnose **vanishing/exploding gradients**.
- Apply **regularization**: dropout, batch normalization, early stopping, weight decay.

## Backpropagation: how networks learn

To reduce the loss, we need to know **how each weight affects the loss** — i.e. the
gradient of the loss with respect to every weight. With millions of weights across many
layers, computing this naively is impossible. **Backpropagation** computes them all
efficiently using the **chain rule** (Chapter 5), working **backward** from the output:

![Backpropagation. The forward pass computes the prediction and loss (left to right); the backward pass propagates the error gradient from the output back through each layer (right to left) using the chain rule, giving every weight's gradient.](assets/images/ch33_backprop.png)

1. **Forward pass:** compute the prediction and the loss.
2. **Backward pass:** starting at the loss, compute the gradient at the output layer, then
   use the **chain rule** to propagate it backward through each layer, getting `∂Loss/∂w`
   for every weight.
3. **Update:** each weight steps downhill: `w ← w − η·(∂Loss/∂w)` (gradient descent).

::: note
You rarely implement backprop by hand — frameworks like PyTorch do **automatic
differentiation**: `loss.backward()` computes every gradient for you, and
`optimizer.step()` applies the update. But understanding that it's *just the chain rule
running backward* demystifies the whole process.
:::

## Loss functions

The loss measures "how wrong" the network is (Chapters 5, 18). Pick it by task:

- **Regression** → **MSE** (mean squared error).
- **Binary classification** → **Binary Cross-Entropy** (log-loss).
- **Multiclass classification** → **Categorical Cross-Entropy** (with softmax).

## Optimisers: smarter gradient descent

Plain gradient descent (Chapter 5) works, but smarter **optimisers** converge faster and
more reliably:

- **SGD (Stochastic Gradient Descent)** — updates on small **mini-batches** rather than the
  whole dataset; faster and adds helpful noise.
- **Momentum** — accumulates a "velocity" so updates keep rolling through flat spots and
  small bumps, like a ball rolling downhill.
- **Adam** — combines momentum with per-parameter adaptive learning rates. It's robust,
  fast, and the **default choice** for most deep learning.

### Practical: Adam vs SGD

```python
import torch, torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc = StandardScaler().fit(X_tr)
X_tr_t = torch.tensor(sc.transform(X_tr), dtype=torch.float32)
y_tr_t = torch.tensor(y_tr, dtype=torch.float32).view(-1, 1)

def train(opt_name):
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(30, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())
    opt = (torch.optim.SGD(m.parameters(), lr=0.1) if opt_name == "SGD"
           else torch.optim.Adam(m.parameters(), lr=0.01))
    loss_fn = nn.BCELoss(); losses = []
    for _ in range(50):
        opt.zero_grad(); loss = loss_fn(m(X_tr_t), y_tr_t)
        loss.backward(); opt.step(); losses.append(loss.item())
    return losses

for name in ["SGD", "Adam"]:
    L = train(name)
    print(f"{name}: loss epoch1={L[0]:.3f}, epoch10={L[9]:.3f}, epoch50={L[49]:.3f}")
```

**Output:**
```text
SGD:  loss epoch1=0.745, epoch10=0.521, epoch50=0.177
Adam: loss epoch1=0.745, epoch10=0.251, epoch50=0.051
```

![Training loss over epochs for SGD vs Adam. Both start equal, but Adam drives the loss down far faster and lower — illustrating why adaptive optimisers are the default for deep learning.](assets/images/ch33_optimizers.png)

### Explanation

Both optimisers start at the same loss (0.745), but **Adam** drops to **0.051** by epoch
50 while **SGD** only reaches **0.177** — Adam learned roughly 3× faster here. Its
per-parameter adaptive steps and momentum make it converge quickly with little tuning,
which is why it's the go-to optimiser.

## Batch size and epochs

- **Epoch** — one full pass through the entire training dataset.
- **Batch size** — how many samples are processed before each weight update.
  - **Full-batch** (whole dataset): stable but slow and memory-heavy.
  - **Mini-batch** (e.g. 32–256): the standard — a good balance of speed and stability.
  - **Stochastic** (size 1): noisy, rarely used alone.

You train for many epochs, updating on each mini-batch, until the validation loss stops
improving.

## Vanishing and exploding gradients

In **deep** networks, gradients are multiplied through many layers during backprop. They
can shrink toward zero (**vanishing** — early layers stop learning) or blow up
(**exploding** — training diverges). This plagued early deep networks with sigmoid/tanh
activations.

**Fixes:** **ReLU** activations (don't squash positive gradients), careful **weight
initialisation**, **batch normalization**, **residual connections** (skip connections, as
in ResNets), and **gradient clipping** (for exploding gradients).

## Regularization: stopping overfitting

Deep networks have huge capacity and overfit easily. Key regularizers:

![Dropout randomly "switches off" a fraction of neurons during each training step, forcing the network not to over-rely on any one neuron — a powerful, simple regularizer. At test time all neurons are used.](assets/images/ch33_dropout.png)

- **Dropout** — randomly disables a fraction (e.g. 50%) of neurons each training step,
  forcing redundancy and preventing co-dependence. Off at test time.
- **Batch Normalization** — normalises each layer's inputs per mini-batch; speeds and
  stabilises training and mildly regularizes.
- **Early stopping** — stop training when **validation** loss stops improving (before it
  starts rising from overfitting).
- **Weight decay (L2)** — penalises large weights (Chapter 26), keeping the model simpler.
- **Data augmentation** — create more training data (rotate/flip images, etc., Chapter 40).

![The classic overfitting picture. Training loss keeps falling, but validation loss bottoms out then rises — the gap is overfitting. Early stopping halts at the validation minimum.](assets/images/ch33_train_val.png)

::: keyidea
Watch the **training vs validation loss curves**. When validation loss starts *rising*
while training loss keeps falling, you're overfitting — that's the moment **early
stopping** picks, and where dropout/weight-decay help. This single plot is the most
important diagnostic in deep learning.
:::

::: tip
**Practical & debugging tips:** (1) Use **Adam** (lr ≈ 0.001–0.01) as a default optimiser.
(2) Use **mini-batches** (32–256). (3) **ReLU** + good init avoids vanishing gradients. (4)
Add **dropout/batch-norm** and use **early stopping** if overfitting. (5) If loss is `nan`,
lower the learning rate or clip gradients. (6) `model.train()` vs `model.eval()` matters —
dropout/batch-norm behave differently in each mode. (7) Always plot train *and* validation
loss.
:::

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Not separating `model.train()` and `model.eval()` modes.** Dropout and batch
norm behave differently; forgetting to switch gives wrong results at evaluation time.
:::

- **Mistake 2 — Learning rate too high** (loss explodes/`nan`) or too low (painfully slow).
- **Mistake 3 — Training too long** without early stopping → overfitting.
- **Mistake 4 — Ignoring vanishing gradients** in deep nets (use ReLU, batch norm, skip
  connections).
- **Mistake 5 — Full-batch training** on large data (slow, memory-heavy) — use mini-batches.
- **Mistake 6 — Applying dropout at test time** (it should be disabled in eval mode).

## Best practices

- **Default to Adam**, mini-batches, and ReLU.
- **Plot train and validation loss**; use **early stopping**.
- **Regularize** with dropout, weight decay, and/or batch norm when overfitting.
- **Tune the learning rate** first — it's the most important hyperparameter.
- **Use batch norm / good init / skip connections** for very deep networks.
- **Switch eval mode** before testing.

## Chapter Summary

- Networks learn by **backpropagation** — the **chain rule** applied backward through layers
  to get every weight's gradient — followed by a **gradient-descent update**. Frameworks
  automate this via `loss.backward()` and `optimizer.step()`.
- Pick the **loss** by task (MSE / binary or categorical cross-entropy) and use a smart
  **optimiser**; **Adam** (momentum + adaptive rates) is the default and converged ~3×
  faster than SGD in our test.
- Train over **epochs** using **mini-batches**; beware **vanishing/exploding gradients**
  (fixed by ReLU, batch norm, good init, skip connections).
- **Regularize** to fight overfitting: **dropout**, **batch normalization**, **early
  stopping**, **weight decay**, and **data augmentation** — guided by the **train-vs-
  validation loss** curves.

---

::: {.qband}
Practice Zone — Chapter 33
:::

## Multiple-Choice Questions (MCQs)

**Q1.** Backpropagation is essentially repeated use of the:
a) Dot product  b) Chain rule  c) Matrix inverse  d) Softmax

**Q2.** The default optimiser for most deep learning is:
a) Plain GD  b) SGD only  c) Adam  d) K-Means

**Q3.** One epoch means:
a) One weight update  b) One full pass through the training data  c) One mini-batch  d) One
layer

**Q4.** Dropout helps by:
a) Adding layers  b) Randomly disabling neurons during training  c) Scaling inputs  d)
Removing the loss

**Q5.** Vanishing gradients mainly affect:
a) Output layers  b) Early layers of deep networks  c) The loss function  d) The optimiser

**Q6.** Early stopping halts training when:
a) Training loss rises  b) Validation loss stops improving  c) Accuracy is 100%  d) The
first epoch ends

**Q7.** Which activation helps prevent vanishing gradients?
a) Sigmoid  b) Tanh  c) ReLU  d) Softmax

**Q8.** At test time, dropout should be:
a) Increased  b) Disabled (eval mode)  c) Doubled  d) Randomised

### MCQ Answers
**1:** b. **2:** c. **3:** b. **4:** b. **5:** b. **6:** b. **7:** c. **8:** b.

## Interview Questions (with answers)

**Q1. Explain backpropagation.**
*Answer:* Backpropagation computes the gradient of the loss with respect to every weight by
applying the chain rule backward through the network: after the forward pass computes the
loss, it propagates the error gradient from the output layer back to the input, layer by
layer, giving each weight's ∂Loss/∂w efficiently. Those gradients are then used by gradient
descent to update the weights.

**Q2. What is the difference between SGD, momentum, and Adam?**
*Answer:* SGD updates weights using gradients from mini-batches. Momentum adds a velocity
term that accumulates past gradients, smoothing updates and powering through flat regions.
Adam combines momentum with per-parameter adaptive learning rates, making it fast and
robust with little tuning — the common default.

**Q3. What are vanishing and exploding gradients, and how do you address them?**
*Answer:* In deep networks, backprop multiplies gradients across layers; they can shrink to
near zero (vanishing — early layers stop learning) or blow up (exploding — training
diverges). Fixes include ReLU activations, careful weight initialisation, batch
normalization, residual/skip connections, and gradient clipping (for explosions).

**Q4. How does dropout prevent overfitting?**
*Answer:* During training it randomly switches off a fraction of neurons each step, so the
network can't rely on any single neuron and must learn redundant, robust features. This
acts like training an ensemble of sub-networks. At test time dropout is disabled and all
neurons are used.

**Q5. How do you know when to stop training?**
*Answer:* Monitor the validation loss. While both training and validation loss fall, keep
going; when validation loss bottoms out and starts rising (even as training loss keeps
falling), the model is overfitting — stop at the validation minimum (early stopping).

## Scenario-Based Questions (with answers)

**Q1.** *Your deep network's training loss keeps dropping but validation loss is rising
after epoch 20. What's happening and what do you do?*
*Answer:* Overfitting. Apply early stopping (use the epoch-20 weights), and/or add
regularization (dropout, weight decay), reduce model size, or get more data/augmentation.

**Q2.** *A very deep network's early layers barely change during training. What's the likely
cause and fixes?*
*Answer:* Vanishing gradients — gradients shrink as they propagate back to early layers.
Fixes: use ReLU activations, better weight initialisation, batch normalization, and
residual/skip connections (as in ResNets) so gradients flow.

**Q3.** *Your loss becomes `nan` after a few iterations. What do you check first?*
*Answer:* The learning rate is likely too high (exploding updates). Lower it; also consider
gradient clipping, check for bad inputs (unscaled features, NaNs), and ensure the loss/
activations are appropriate.

## Logic-Based Questions (with answers)

**Q1.** Why is backpropagation more efficient than computing each weight's gradient
independently?
*Answer:* Because it reuses intermediate results: by propagating the gradient backward and
applying the chain rule, the gradient computations shared across weights are computed once
and reused, turning an otherwise explosive computation into one pass proportional to the
forward pass.

**Q2.** Why does Adam often converge faster than plain SGD?
*Answer:* Adam adapts the learning rate per parameter (scaling steps by recent gradient
magnitudes) and uses momentum, so it takes well-sized, smoothed steps in each direction,
navigating the loss surface faster and more stably than a single global learning rate.

**Q3.** Why must dropout be turned off at test time?
*Answer:* Dropout's random neuron removal is a training-time regularizer; at test time we
want the full, deterministic network to make the best prediction. Frameworks scale
activations appropriately and disable dropout in eval mode.

## Practical Questions (with answers)

**Q1.** Which PyTorch calls perform the backward pass and the weight update?
*Answer:* `loss.backward()` computes gradients (backprop); `optimizer.step()` applies the
gradient-descent update.

**Q2.** How do you add 50% dropout after a hidden layer in PyTorch?
*Answer:* Insert `nn.Dropout(0.5)` after the activation in your `nn.Sequential` (or module).

**Q3.** What's the difference between `model.train()` and `model.eval()`?
*Answer:* `train()` enables training behaviour (dropout active, batch-norm uses batch
statistics); `eval()` disables dropout and makes batch-norm use running statistics — use it
for validation/testing.

## Long Questions (with answers)

**Q1. Explain the full training process of a neural network — forward pass, loss,
backpropagation, and the weight update — and how optimisers and batches fit in.**

*Answer:* Training repeats a four-step loop. **(1) Forward pass:** a mini-batch of inputs
flows through the layers (weighted sums + activations) to produce predictions. **(2) Loss:**
a loss function measures how wrong the predictions are versus the targets (MSE for
regression, cross-entropy for classification). **(3) Backward pass (backpropagation):**
starting from the loss, the chain rule is applied backward through the network to compute
the gradient of the loss with respect to every weight efficiently — frameworks do this
automatically via `loss.backward()`. **(4) Update:** an **optimiser** adjusts each weight in
the downhill direction of its gradient — plain SGD uses a fixed learning rate on
mini-batches, momentum adds a velocity term to smooth and accelerate updates, and **Adam**
adapts the step size per parameter, converging fast with little tuning (in our test it
reached far lower loss than SGD in the same epochs). Data is processed in **mini-batches**
(e.g. 32–256 samples) for a balance of speed and stability, and one full pass over the data
is an **epoch**; training runs for many epochs until the validation loss stops improving.
This loop — predict, measure error, backpropagate gradients, update — is the engine of all
deep learning.

**Q2. Discuss the main techniques for preventing overfitting and training instability in
deep networks.**

*Answer:* Deep networks have enormous capacity, so **overfitting** and **instability** are
central concerns, addressed by several techniques. **Dropout** randomly disables a fraction
of neurons during each training step, forcing the network to learn redundant, robust
features and behaving like an ensemble; it is disabled at test time. **Batch normalization**
normalises each layer's inputs per mini-batch, which stabilises and speeds training and
provides mild regularization. **Early stopping** monitors validation loss and halts training
at its minimum, before the model begins memorising noise (the point where validation loss
rises while training loss keeps falling). **Weight decay (L2 regularization)** penalises
large weights, keeping the model simpler (Chapter 26). **Data augmentation** synthetically
expands the training set (e.g. rotating/flipping images), reducing overfitting on perception
tasks. For **training instability**, especially in very deep nets, **vanishing/exploding
gradients** are mitigated by **ReLU** activations (which don't squash positive gradients),
careful **weight initialisation**, **batch normalization**, **residual/skip connections**
that let gradients flow directly, and **gradient clipping** for explosions; choosing a
sensible **learning rate** (and a good optimiser like Adam) is the most important single
factor. Used together and guided by the train-vs-validation loss curves, these techniques
let large networks train stably and generalise well.

## Exercises

1. List the four steps of the training loop in order.
2. Explain in one sentence each: epoch, batch size, learning rate.
3. Why does ReLU help with vanishing gradients?
4. Describe how dropout works during training and at test time.
5. Sketch train vs validation loss curves showing overfitting and mark where early stopping
   acts.

## Mini-Project

**Project: Diagnose and fix overfitting.**

1. Train an MLP on a dataset with a *small* training set to induce overfitting; record train
   and validation loss every epoch.
2. Plot both loss curves (Chapter 14) and identify where overfitting begins.
3. Add dropout and weight decay; retrain and compare the gap between train and validation
   loss.
4. Compare SGD vs Adam convergence on the same problem (plot loss curves).
5. Write a short report on what helped. Save in `my-ml-journey/`.

## Assignments

1. **Coding:** Implement early stopping manually (track validation loss; stop after N epochs
   without improvement) and report the chosen epoch.
2. **Coding:** Compare a network with vs without batch normalization on a deeper MLP; note
   training speed and stability.
3. **Conceptual:** Write one page explaining backpropagation as "the chain rule running
   backward", connecting it to Chapter 5.

::: tip
You can now train deep networks. Next we specialise them for specific data: Chapter 34,
**Convolutional Neural Networks (CNNs)**, are purpose-built for *images* — the architecture
that sparked the deep-learning revolution (AlexNet, 2012).
:::
