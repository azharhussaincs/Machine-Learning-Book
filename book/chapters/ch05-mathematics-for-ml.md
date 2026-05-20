# Mathematics for Machine Learning

## Introduction

Let's address the fear first. Many beginners hear "mathematics" and want to run
away. **Please don't.** You do *not* need to be a mathematician to do Machine
Learning. You need to understand a *small* set of ideas — and understand them
*intuitively*, with pictures, not just symbols.

Think of it like driving a car. You don't need to be a mechanical engineer to
drive well. But knowing *roughly* how the engine, brakes, and steering work makes
you a far better, safer driver. This chapter is the "how the engine works" tour of
ML maths — enough to make you confident, not a PhD.

We focus on the **three pillars** that power almost all of Machine Learning:

1. **Linear Algebra** — the maths of vectors and matrices (how data is stored and
   transformed).
2. **Calculus** — the maths of change and slopes (how models *learn*).
3. **Optimization** — the maths of finding the best answer (the actual *learning*
   step, gradient descent).

::: keyidea
You do not need to memorise proofs. You need to understand: data is stored as
**vectors and matrices**; learning means **reducing a loss**; and we reduce it by
following the **slope (gradient) downhill**. Everything in this chapter serves
those three sentences.
:::

By the end you will understand vectors, matrices, derivatives, gradients, and
gradient descent — and you will have coded gradient descent *from scratch* to make
a model learn before your eyes.

---

# Part A — Linear Algebra

## Why linear algebra? Data is numbers in boxes

Look at any dataset — it is a **table**: rows and columns of numbers. Linear
algebra is simply the maths of working with tables of numbers efficiently. When
you multiply a data table by a list of "weights," you get predictions for every
row at once. That is the whole game.

## Scalars, vectors, matrices, and tensors

These four words describe data of different shapes. They sound fancy; they are not.

| Name | What it is | Example | Shape |
|---|---|---|---|
| **Scalar** | A single number | `7` or `3.14` | 0-D |
| **Vector** | A list of numbers | `[170, 65, 25]` (height, weight, age) | 1-D |
| **Matrix** | A table of numbers (rows × columns) | a spreadsheet of many people | 2-D |
| **Tensor** | A box of numbers with 3+ dimensions | a colour image (height × width × 3) | 3-D+ |

::: note
**Connecting to ML:** one **row** of your data (one person, one house, one email)
is a **vector**. The **whole dataset** is a **matrix**. A batch of colour images is
a **4-D tensor**. The library *TensorFlow* is literally named after tensors.
:::

## Vectors and their operations

A **vector** is an ordered list of numbers. We can also picture a 2-D vector as an
**arrow** from the origin to a point.

![A 2-D vector drawn as an arrow from the origin. Its components (3, 2) are its horizontal and vertical parts; its length (magnitude) is found with the Pythagorean theorem.](assets/images/ch05_vector.png)

### Vector addition and scalar multiplication

- **Addition:** add matching elements. `[1, 2] + [3, 4] = [4, 6]`.
- **Scalar multiplication:** multiply every element by a number.
  `3 × [1, 2] = [3, 6]` (this stretches the arrow to 3× its length).

### The dot product — the most important operation in ML

The **dot product** multiplies two vectors element-by-element and adds the
results, giving a *single number*:

<div class="equation"><img class="eq" src="assets/images/eq_ch05_dot_product.png" alt="dot product"></div>

*Example:* `[2, 3] · [4, 5] = (2×4) + (3×5) = 8 + 15 = 23`.

**Why it matters:** a model's prediction is almost always a dot product of the
**features** and the **weights**, plus a bias:

<div class="equation"><img class="eq" src="assets/images/eq_ch05_linear_combo.png" alt="linear combination"></div>

This single formula is the heart of linear regression, logistic regression, and
every neuron in a neural network. Learn to read it: *"multiply each feature by its
importance (weight), add them up, add a bias."*

### The norm (length) of a vector

The **norm** (written ‖v‖) is the length of the vector's arrow, from the
Pythagorean theorem:

<div class="equation"><img class="eq" src="assets/images/eq_ch05_vector_norm.png" alt="vector norm"></div>

**Why it matters:** norms measure "size" and "distance." The distance between two
points is the norm of their difference — the basis of K-Nearest Neighbors (Chapter
19) and many clustering methods.

## Matrices and their operations

A **matrix** is a rectangular grid of numbers with *rows* and *columns*. We write
its shape as (rows × columns). A 3×2 matrix has 3 rows and 2 columns.

```text
        col1  col2
row1  [  1     2  ]
row2  [  3     4  ]      <- this is a 3 x 2 matrix
row3  [  5     6  ]
```

### Transpose

The **transpose** (written Xᵀ) flips a matrix over its diagonal — rows become
columns. A 3×2 matrix becomes 2×3.

```text
        1  2                  1  3  5
A =     3  4       Aᵀ  =      2  4  6
        5  6
```

### Matrix multiplication

This is the operation that does the heavy lifting. To multiply two matrices, you
take the **dot product of each row of the first with each column of the second**.

![Matrix multiplication: each output cell is the dot product of a row from the left matrix and a column from the right matrix. The inner dimensions must match.](assets/images/ch05_matrix_mult.png)

::: warning
**The shape rule:** to multiply an (a×b) matrix by a (c×d) matrix, the inner
numbers must match: **b must equal c**. The result has shape (a×d). This "shape
mismatch" is the single most common error beginners hit in NumPy and deep
learning. Always check shapes!
:::

**Why it matters:** with one matrix multiplication you compute predictions for your
*entire dataset at once*. If `X` is your data matrix and `w` is your weight vector:

<div class="equation"><img class="eq" src="assets/images/eq_ch05_matrix_vec.png" alt="matrix form of prediction"></div>

This computes a prediction for every row in a single, lightning-fast operation
(GPUs are built to do exactly this). This is *why* ML uses matrices: speed.

### Special matrices

- **Identity matrix (I)** — 1s on the diagonal, 0s elsewhere. Multiplying by it
  changes nothing (like multiplying a number by 1).
- **Inverse (A⁻¹)** — the matrix that "undoes" A, so that A·A⁻¹ = I (like the
  reciprocal of a number). Used in the closed-form solution of linear regression
  (Chapter 17).

---

# Part B — Calculus

## Why calculus? Learning means following a slope

Here is the big picture. A model has knobs (parameters). We measure how *wrong* it
is with a **loss**. **Learning = turning the knobs to make the loss as small as
possible.** Calculus tells us *which way* to turn each knob — it gives us the
**slope** of the loss. That's it. Calculus in ML is mostly about slopes.

## Functions and slope

A **function** is a machine: put a number in, get a number out, e.g. `f(x) = x²`.
The **slope** at a point tells you how steep the function is there — how fast the
output changes when you nudge the input.

- A **positive** slope means the function is going **up** (left to right).
- A **negative** slope means it is going **down**.
- A **zero** slope means it is **flat** — a peak, a valley, or a plateau.

## The derivative

The **derivative**, written f′(x) or df/dx, is the *exact* slope of a function at a
point. Formally it is the limit of "rise over run" as the run shrinks to zero:

<div class="equation"><img class="eq" src="assets/images/eq_ch05_derivative.png" alt="derivative definition"></div>

![The derivative is the slope of the tangent line touching the curve at a point. Where the curve is steep the derivative is large; at the bottom of the valley the slope is zero.](assets/images/ch05_derivative.png)

You don't need to compute limits by hand. Just know a few rules and the *meaning*.

| Function f(x) | Derivative f′(x) | Plain meaning |
|---|---|---|
| constant `c` | `0` | flat line, no slope |
| `x` | `1` | slope is always 1 |
| `x²` | `2x` | steeper as x grows |
| `xⁿ` | `n·xⁿ⁻¹` | the "power rule" |
| `eˣ` | `eˣ` | grows at its own rate |

*Example:* for `f(x) = x²`, the derivative is `2x`. At `x = 3` the slope is `6`
(steep, going up). At `x = 0` the slope is `0` (the bottom of the valley).

## Partial derivatives and the gradient

Real models have *many* knobs (parameters), not one. A **partial derivative** asks:
"if I nudge *only this one* parameter and keep the rest fixed, how does the loss
change?" We write it with a curly ∂:

<div class="equation"><img class="eq" src="assets/images/eq_ch05_partial.png" alt="partial derivative"></div>

The **gradient** (written ∇L, "nabla L") is just the list of *all* the partial
derivatives — one slope per parameter:

<div class="equation"><img class="eq" src="assets/images/eq_ch05_gradient.png" alt="gradient"></div>

::: keyidea
The gradient is a **direction**. It points in the direction where the loss
*increases fastest*. So to *decrease* the loss, we step in the **opposite**
direction — downhill. That single insight is the engine of nearly all ML training.
:::

## The chain rule

The **chain rule** lets us find the slope of functions nested inside functions
(like a function of a function). If `z` depends on `y`, and `y` depends on `x`:

<div class="equation"><img class="eq" src="assets/images/eq_ch05_chain_rule.png" alt="chain rule"></div>

**Why it matters:** neural networks are giant nests of functions. The famous
**backpropagation** algorithm (Chapter 33) is just the chain rule applied
cleverly, layer by layer. Remember this when we get there.

---

# Part C — Optimization

## The loss function: a score for "how wrong"

To *learn*, a model needs a number that says how badly it is doing. That number is
the **loss** (or **cost**). For predicting numbers, the classic loss is **Mean
Squared Error (MSE)** — the average of the squared mistakes:

<div class="equation"><img class="eq" src="assets/images/eq_ch05_mse.png" alt="mean squared error"></div>

Here `yᵢ` is the true value and `ŷᵢ` ("y-hat") is the model's prediction. We square
the errors so that (a) negatives don't cancel positives, and (b) big mistakes are
punished more. **Training = finding the parameters that make this loss smallest.**

## Gradient descent: rolling downhill to the answer

Imagine standing on a foggy hillside, wanting to reach the lowest valley. You can't
see far, but you can feel the **slope** under your feet. A good strategy: take a
small step in the steepest downhill direction, then repeat. Eventually you reach
the bottom. **That is gradient descent.**

![Gradient descent: starting from a random point, we repeatedly step downhill (opposite the gradient) until we reach the minimum of the loss curve.](assets/images/ch05_gradient_descent.png)

The update rule for each parameter `w` is:

<div class="equation"><img class="eq" src="assets/images/eq_ch05_grad_step.png" alt="gradient descent update"></div>

In words: **new weight = old weight − (learning rate × slope).** We subtract
because we want to go *downhill* (against the gradient). The Greek letter η ("eta")
is the **learning rate** — a hyperparameter controlling step size.

### The learning rate: the most important dial

![The effect of the learning rate. Too small: painfully slow. Just right: steady progress to the minimum. Too large: it overshoots and may diverge (bounce away).](assets/images/ch05_learning_rate.png)

- **Too small** → learning is correct but *painfully slow*.
- **Too large** → it *overshoots* the valley, bouncing around or even flying off to
  infinity (diverging).
- **Just right** → steady, efficient progress to the minimum.

Tuning the learning rate is one of the first things you'll do for any model.

### Local vs global minima

A simple bowl shape has one lowest point (the **global minimum**). Complex loss
"landscapes" (especially in deep learning) have many dips. A **local minimum** is a
dip that isn't the deepest. Gradient descent can get stuck in one. Smart variants
(momentum, Adam — Chapter 33) help escape them. For now, just know the danger
exists.

## Practical: code gradient descent from scratch

Time to make all of this real. We will fit a line `y = w·x + b` to data using
**only NumPy and gradient descent** — no ML library. Watch the loss shrink as the
model learns.

```python
import numpy as np

# --- The data: true relationship is y = 2x + 1 (we'll let the model discover it) ---
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([3, 5, 7, 9, 11], dtype=float)   # = 2x + 1 exactly
n = len(X)

# --- Start with random/zero guesses for the parameters ---
w = 0.0      # weight (slope)
b = 0.0      # bias (intercept)
lr = 0.01    # learning rate (η)  -- a hyperparameter

# --- Gradient descent loop ---
for epoch in range(1, 1001):                 # 1000 passes over the data
    y_pred = w * X + b                       # current predictions (vectorised)
    error = y_pred - y                       # how far off we are

    # MSE loss = mean of squared errors
    loss = np.mean(error ** 2)

    # Partial derivatives of MSE w.r.t. w and b (from calculus)
    dw = (2 / n) * np.dot(error, X)          # ∂L/∂w
    db = (2 / n) * np.sum(error)             # ∂L/∂b

    # The gradient-descent UPDATE: step downhill
    w -= lr * dw
    b -= lr * db

    if epoch % 200 == 0:                      # print progress occasionally
        print(f"epoch {epoch:4d} | loss {loss:8.5f} | w {w:.3f} | b {b:.3f}")

print(f"\nLearned model: y = {w:.2f} x + {b:.2f}   (true: y = 2x + 1)")
```

**Output:**
```text
epoch  200 | loss  0.00811 | w 2.058 | b 0.790
epoch  400 | loss  0.00209 | w 2.030 | b 0.893
epoch  600 | loss  0.00054 | w 2.015 | b 0.946
epoch  800 | loss  0.00014 | w 2.008 | b 0.972
epoch 1000 | loss  0.00004 | w 2.004 | b 0.986
Learned model: y = 2.00 x + 0.99   (true: y = 2x + 1)
```

### Line-by-line explanation

- **`y_pred = w * X + b`** — the prediction formula (a linear combination), applied
  to all 5 points at once thanks to NumPy (vectorisation).
- **`error = y_pred - y`** — the vector of mistakes.
- **`loss = np.mean(error ** 2)`** — the MSE: our single "how wrong" number.
- **`dw` and `db`** — the **gradient**: the partial derivatives of MSE with respect
  to `w` and `b`. (These come straight from the calculus rules above; we derive
  them fully in Chapter 17.)
- **`w -= lr * dw`** — the gradient-descent step: nudge `w` downhill. Same for `b`.
- **The result:** after 1000 steps, the model discovered `w ≈ 2.00`, `b ≈ 0.99` —
  almost exactly the true `y = 2x + 1`. It *learned* the relationship purely by
  following slopes downhill. (Notice the loss shrinking toward zero each time we
  print — that is the model getting better.)

::: keyidea
You just witnessed the core loop of nearly all of Machine Learning:
**predict → measure loss → compute gradient → step downhill → repeat.**
Linear regression, logistic regression, and deep neural networks all train with
this exact pattern. Everything else is detail.
:::

::: tip
**Experiments & debugging:** (1) Change `lr` to `0.001` — learning is much slower
(needs more epochs). (2) Change `lr` to `0.3` — watch the loss explode to `nan`
(overshooting). (3) If your loss ever increases or becomes `nan`, your learning
rate is almost always too high. This is the #1 training bug.
:::

## Where these maths show up later in the book

| Maths idea | Where you'll use it |
|---|---|
| Dot product / matrix multiply | Every model's prediction; neural network layers |
| Vector norm / distance | KNN (Ch 19), clustering (Ch 27) |
| Matrix inverse | Linear regression closed form (Ch 17) |
| Derivatives & gradient | Training every model |
| Chain rule | Backpropagation in neural networks (Ch 33) |
| Gradient descent | Training linear/logistic models and deep nets |
| Eigenvectors (briefly) | PCA dimensionality reduction (Ch 28) |

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Shape mismatches in matrix multiplication.** Always confirm the inner
dimensions match. Print `.shape` constantly while debugging.
:::

- **Mistake 2 — Forgetting the minus sign in the update.** We step *against* the
  gradient (downhill). Using `+` makes the loss *grow*.
- **Mistake 3 — A learning rate that's too high.** The most common cause of "my loss
  became nan / exploded."
- **Mistake 4 — Thinking you must memorise formulas.** You must understand
  *meanings*; libraries compute the formulas for you.
- **Mistake 5 — Confusing the gradient's direction.** The gradient points *uphill*;
  we go the opposite way.

## Best practices

- **Always picture it.** Vector = arrow; derivative = slope; gradient descent =
  rolling downhill. Intuition beats memorisation.
- **Check shapes early and often** when coding with matrices.
- **Start with a small learning rate** and increase if learning is too slow.
- **Plot the loss over epochs** — it should steadily decrease. A rising or jagged
  loss signals a problem (usually the learning rate).
- **Trust libraries for the heavy maths**, but understand what they do under the
  hood — that's what this chapter gives you.

## Chapter Summary

- ML data lives in **vectors** (rows), **matrices** (datasets), and **tensors**
  (images/batches).
- The **dot product** powers every prediction: features × weights, summed, plus a
  bias. **Matrix multiplication** does this for the whole dataset at once (fast).
- The **derivative** is the slope of a function; the **gradient** is the list of
  slopes for all parameters and points in the direction of *steepest increase*.
- The **chain rule** handles nested functions and is the basis of backpropagation.
- A **loss function** (e.g. MSE) scores how wrong the model is; **training** means
  minimising it.
- **Gradient descent** minimises the loss by repeatedly stepping *downhill*:
  `w ← w − η·(∂L/∂w)`. The **learning rate** η controls step size and must be tuned.
- You implemented gradient descent from scratch and watched a model learn
  `y = 2x + 1`.

---

::: {.qband}
Practice Zone — Chapter 5
:::

## Multiple-Choice Questions (MCQs)

**Q1.** A single row of a dataset (one example) is best described as a:
a) Scalar  b) Vector  c) Matrix  d) Tensor

**Q2.** The dot product of `[1, 2, 3]` and `[4, 5, 6]` is:
a) `[4, 10, 18]`  b) `32`  c) `15`  d) `21`

**Q3.** To multiply a (2×3) matrix by a (3×4) matrix, the result has shape:
a) (2×4)  b) (3×3)  c) (2×3)  d) Cannot multiply

**Q4.** The derivative of `f(x) = x²` is:
a) `x`  b) `2x`  c) `x²`  d) `2`

**Q5.** The gradient points in the direction of:
a) Steepest decrease  b) Steepest increase  c) Zero slope  d) The data

**Q6.** In `w = w − η·(∂L/∂w)`, the symbol η is the:
a) Loss  b) Gradient  c) Learning rate  d) Weight

**Q7.** If your loss explodes to `nan` during gradient descent, the most likely
cause is:
a) Too few epochs  b) Learning rate too high  c) Too much data  d) A small dataset

**Q8.** Backpropagation in neural networks is essentially repeated use of the:
a) Dot product  b) Matrix inverse  c) Chain rule  d) Norm

### MCQ Answers
**1:** b. **2:** b (4+10+18=32). **3:** a. **4:** b. **5:** b. **6:** c. **7:** b.
**8:** c.

## Interview Questions (with answers)

**Q1. Why does Machine Learning rely so heavily on linear algebra?**
*Answer:* Because data is naturally represented as vectors (examples) and matrices
(datasets), and predictions are computed as dot products / matrix multiplications.
This representation lets us process an entire dataset in one fast, parallelisable
operation — exactly what CPUs and especially GPUs are optimised for.

**Q2. What is a gradient, and why is it central to training?**
*Answer:* The gradient is the vector of partial derivatives of the loss with
respect to every parameter. It points in the direction of steepest increase of the
loss, so stepping in the opposite direction reduces the loss. Training (gradient
descent) repeatedly takes such downhill steps.

**Q3. Explain gradient descent in simple terms.**
*Answer:* It's an iterative method to minimise a loss. Start with random
parameters, compute the slope (gradient) of the loss, take a small step opposite to
the gradient (downhill), and repeat until the loss stops decreasing. The step size
is the learning rate.

**Q4. What does the learning rate control and what happens at extremes?**
*Answer:* It controls the step size of each update. Too small → very slow
convergence. Too large → overshooting, oscillation, or divergence (loss explodes).
The right value gives steady, efficient convergence.

**Q5. What is the difference between a derivative and a partial derivative?**
*Answer:* A derivative measures how a single-variable function changes with its one
input. A partial derivative measures how a multi-variable function changes with
respect to *one* variable while holding the others fixed. The gradient collects all
partial derivatives.

## Scenario-Based Questions (with answers)

**Q1.** *You train a model and the loss keeps increasing every epoch instead of
decreasing. List the two most likely bugs.*
*Answer:* (1) The learning rate is too high (overshooting uphill) — reduce it. (2) A
sign error — you're adding the gradient instead of subtracting it, so you're
climbing the loss rather than descending. Check the update rule.

**Q2.** *Your matrix multiplication code throws a shape error: shapes (100,5) and
(3,1) not aligned. What's wrong and how do you fix it?*
*Answer:* The inner dimensions don't match (5 ≠ 3). For `X·w`, `w` must have 5 rows
to match X's 5 columns, i.e. shape (5,1). Fix the weight vector's shape (or
transpose the appropriate matrix) so the inner dimensions agree.

**Q3.** *Training is correct but extremely slow, taking thousands of epochs to make
tiny progress. What's the simplest first thing to try?*
*Answer:* Increase the learning rate (e.g. from 0.001 to 0.01 or 0.1) and watch the
loss. If it still decreases smoothly, you've sped up training; if it starts
oscillating or exploding, you went too far — back off.

## Logic-Based Questions (with answers)

**Q1.** At the exact minimum of a smooth loss curve, what is the value of the
gradient, and what does gradient descent do there?
*Answer:* The gradient is zero (the slope is flat). The update `w − η·0` leaves `w`
unchanged, so gradient descent naturally stops moving at a minimum.

**Q2.** If doubling every feature value also doubles the prediction (with bias 0),
which operation does that reveal the prediction is built from?
*Answer:* The dot product / linear combination `w·x` — scaling `x` by 2 scales the
dot product by 2. This is the "linear" in linear models.

**Q3.** You square the errors in MSE. Give two distinct reasons squaring is used
instead of just summing the raw errors.
*Answer:* (1) Squaring makes all terms positive, so positive and negative errors
don't cancel out. (2) Squaring penalises large errors much more than small ones,
pushing the model to avoid big mistakes.

## Practical Questions (with answers)

**Q1.** In the gradient-descent code, what does `np.dot(error, X)` compute and why?
*Answer:* It computes the sum of `error × X` across all data points — the core of
the partial derivative ∂L/∂w for MSE. The dot product efficiently multiplies each
error by its corresponding `x` and adds them in one operation.

**Q2.** How would you modify the code to also store and later plot the loss at every
epoch?
*Answer:* Create `losses = []` before the loop and append inside it:
`losses.append(loss)`. After training, `import matplotlib.pyplot as plt;
plt.plot(losses)` shows the loss curve (it should slope steadily downward).

**Q3.** Why are the predictions computed as `w * X + b` able to handle all five data
points without a loop?
*Answer:* Because `X` is a NumPy array, the operation is *vectorised* — NumPy
applies `w * X + b` element-wise to the whole array at once, which is both shorter
and far faster than a Python loop.

## Long Questions (with answers)

**Q1. Explain, end to end, how a model "learns" using a loss function and gradient
descent. Use the line-fitting example to ground your answer.**

*Answer:* Learning is the search for parameter values that make the model's
predictions match reality as closely as possible. First we define a **model** —
here `ŷ = w·x + b` — with parameters `w` and `b` that start at arbitrary values
(e.g. 0). Next we define a **loss function** that scores how wrong the predictions
are; for regression we use **MSE**, the mean of squared errors. The loss depends on
the parameters, so it forms a "landscape" whose lowest point corresponds to the
best parameters. To find that point we use **gradient descent**: we compute the
**gradient** (the partial derivatives ∂L/∂w and ∂L/∂b), which points uphill, and
then step in the *opposite* (downhill) direction by a small amount controlled by
the **learning rate** η, using `w ← w − η·∂L/∂w`. Repeating this many times
(epochs) steadily reduces the loss. In the example, starting from `w=0, b=0`, after
1000 steps the parameters converged to `w≈2.01, b≈0.96`, almost exactly the true
relationship `y = 2x + 1`. The model was never *told* the answer; it discovered it
by repeatedly measuring its error and rolling downhill.

**Q2. Discuss the role of the learning rate in gradient descent, including what
happens when it is too small, too large, and well-chosen, and how you would tune
it in practice.**

*Answer:* The learning rate η sets how big a step gradient descent takes each
update. **Too small:** each step barely moves the parameters, so convergence is
correct but extremely slow, possibly needing far more epochs (and compute) than
practical. **Too large:** steps overshoot the minimum; the loss may oscillate
without settling, or grow without bound and become `nan` (divergence), because the
algorithm keeps leaping past and up the far side of the valley. **Well-chosen:** the
loss decreases steadily and reaches a low value efficiently. In practice you tune η
by trying values on a logarithmic scale (e.g. 0.0001, 0.001, 0.01, 0.1) and
plotting the loss curve: pick the largest rate that still decreases smoothly
without instability. Advanced methods (learning-rate schedules that decay over
time, and adaptive optimisers like Adam in Chapter 33) automate much of this, but
the intuition — balance speed against stability — remains the same.

## Exercises

1. Compute by hand: `[2, -1, 3] · [1, 4, 2]` (dot product). Then its result's sign
   and what it would mean as a prediction.
2. Write the shapes that result from multiplying: (4×3)·(3×2), (2×5)·(5×5),
   (3×3)·(2×3) (one is impossible — say which).
3. For `f(x) = 3x² + 2x`, write the derivative and evaluate the slope at `x = 1`.
4. In one sentence each, explain "gradient," "learning rate," and "loss" to a
   beginner.
5. Sketch a loss curve and mark: a starting point, the gradient-descent steps, and
   the minimum.

## Mini-Project

**Project: Visualise learning.**

1. Take the gradient-descent code from this chapter and record the `loss` at every
   epoch into a list.
2. Plot the loss curve with matplotlib (`plt.plot(losses)`). Confirm it decreases.
3. Re-run with three learning rates — `0.001`, `0.01`, and `0.3` — and plot all
   three loss curves on one chart.
4. Write a short paragraph describing the three behaviours (slow, good, diverging)
   and which learning rate you'd choose, and why.

## Assignments

1. **By hand:** Multiply the matrices `A = [[1,2],[3,4]]` and `B = [[5,6],[7,8]]`.
   Show every dot-product step. Verify with NumPy (`A @ B`).
2. **Coding:** Extend the gradient-descent code to fit data where the true line is
   `y = -3x + 5` (create the data yourself). Confirm the learned `w` and `b` are
   close to −3 and 5.
3. **Conceptual:** In one page, explain *why* matrix multiplication makes ML fast,
   and connect it to why GPUs are used for deep learning. Use your own words and one
   diagram.

::: tip
This chapter is the mathematical foundation for *everything* that follows. If a
later chapter's maths feels confusing, come back here — almost every formula in the
book reduces to dot products, gradients, and gradient descent.
:::
