# Recurrent Networks, LSTM & GRU

## Introduction

CNNs (Chapter 34) conquered images. But much of the world's data is **sequential** —
sentences (word after word), speech (sound over time), stock prices, sensor readings,
music. In sequences, **order matters** and **context carries across time**: to understand
"the movie was not good", you must remember "not" when you reach "good".

Standard MLPs and CNNs have no **memory** and expect fixed-size inputs, so they handle
sequences poorly. **Recurrent Neural Networks (RNNs)** — and their improved versions
**LSTM** and **GRU** — were designed for exactly this: networks with a form of memory that
processes sequences step by step.

::: keyidea
An RNN processes a sequence one element at a time, maintaining a **hidden state** that acts
as a **memory** of what it has seen so far. At each step it combines the new input with its
memory to update that memory and produce an output. This recurrence is what lets it model
**context** and **order**.
:::

By the end of this chapter you will be able to:

- Explain how an **RNN** processes sequences using a hidden state.
- Understand the **vanishing gradient** problem that limits plain RNNs.
- Explain how **LSTM** and **GRU** gates solve long-term memory.
- Build a recurrent model in **PyTorch** for a sequence task.

## How an RNN works

An RNN reads a sequence element by element. At each time step `t`, it updates its **hidden
state** `hₜ` (the memory) from the current input `xₜ` and the previous hidden state `hₜ₋₁`:

<div class="equation"><img class="eq" src="assets/images/eq_ch35_rnn.png" alt="RNN recurrence"></div>

The same weights are reused at every step (parameter sharing again). We often draw the RNN
**unrolled** across time to see the flow:

![An RNN unrolled across time. The same cell processes each element, passing its hidden state (memory) forward. The hidden state carries information from earlier steps to later ones — giving the network memory of the sequence.](assets/images/ch35_rnn_unrolled.png)

This lets RNNs handle **variable-length** sequences and capture **order** — the same word
in different positions, or earlier context affecting later meaning.

## The vanishing gradient problem

Plain RNNs have a serious flaw. To learn long-range dependencies, gradients must flow back
through *many* time steps during backprop (Chapter 33). Across many steps they tend to
**shrink toward zero (vanish)** — so the RNN effectively **forgets** information from far
back, unable to connect, say, the start and end of a long paragraph.

::: warning
**Plain RNNs have short memory.** They handle short sequences but struggle to remember
long-range context (the vanishing-gradient problem). This limitation is exactly what LSTM
and GRU were invented to fix.
:::

## LSTM: Long Short-Term Memory

The **LSTM** (1997) solves the memory problem with a clever design: a separate **cell
state** that runs through the sequence like a conveyor belt, regulated by three **gates**
that learn what to remember, forget, and output.

![An LSTM cell. Three gates control the flow of information: the forget gate decides what to discard from the cell state, the input gate decides what new information to store, and the output gate decides what to output. The cell state carries long-term memory across many steps.](assets/images/ch35_lstm.png)

- **Forget gate** — decides what to **discard** from the cell state.
- **Input gate** — decides what new information to **store**.
- **Output gate** — decides what to **output** from the cell state.

Because the cell state can carry information *unchanged* across many steps (the gates can
choose to leave it alone), gradients flow better and the LSTM can learn **long-term
dependencies** that plain RNNs cannot.

## GRU: Gated Recurrent Unit

The **GRU** (2014) is a simpler, faster alternative to the LSTM with only **two gates**
(reset and update) and no separate cell state. It often performs comparably to the LSTM
with fewer parameters, so it's a popular choice when speed matters.

| | Plain RNN | LSTM | GRU |
|---|---|---|---|
| Memory | Short | Long (cell state + 3 gates) | Long (2 gates) |
| Parameters | Fewest | Most | Medium |
| Long dependencies | Poor | Excellent | Very good |
| Speed | Fast | Slower | Faster than LSTM |

## Bidirectional RNNs

A **bidirectional** RNN runs two RNNs — one forward and one backward through the sequence —
and combines them, so each position has context from **both** past *and* future. This helps
tasks like text understanding where later words clarify earlier ones.

## Practical: an LSTM in PyTorch

Let's train an LSTM on a task that *requires memory*: given a length-10 sequence of numbers,
predict whether their **sum is positive**. The model must remember the whole sequence.

```python
import torch, torch.nn as nn, numpy as np
torch.manual_seed(0); np.random.seed(0)

def make(n):                                    # sequences of 10 numbers; label = sum>0
    X = np.random.randn(n, 10, 1).astype("float32")
    y = (X.sum(axis=1).squeeze() > 0).astype("int64")
    return torch.tensor(X), torch.tensor(y)

X_tr, y_tr = make(2000); X_te, y_te = make(500)

class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)
        self.fc = nn.Linear(16, 2)
    def forward(self, x):
        out, (h, c) = self.lstm(x)              # h[-1] = final hidden state (the memory)
        return self.fc(h[-1])

model = LSTMClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
for epoch in range(40):
    optimizer.zero_grad(); loss = loss_fn(model(X_tr), y_tr)
    loss.backward(); optimizer.step()

with torch.no_grad():
    acc = (model(X_te).argmax(1) == y_te).float().mean().item()
print("LSTM sequence-classification accuracy:", round(acc, 3))
```

**Output:**
```text
LSTM sequence-classification accuracy: 0.948
```

### Explanation

- **`nn.LSTM(1, 16)`** processes each 1-feature element of the sequence, maintaining a
  16-dimensional hidden state (memory). We use the **final hidden state** `h[-1]` — a
  summary of the whole sequence — for classification.
- The task **requires memory**: the label depends on the *sum of all 10 elements*, so the
  network must accumulate information across the sequence. It reached **0.948** — proving
  the LSTM learned to remember and combine the whole sequence, something a memoryless model
  couldn't do.

::: keyidea
This is the essence of recurrent models: they carry a memory across time so that **earlier
inputs influence later outputs**. That's why they (and their successors) handle language,
speech, and time series — domains where context and order are everything.
:::

::: tip
**Practical & debugging tips:** (1) RNN inputs are `(batch, sequence_length, features)` with
`batch_first=True`. (2) Use **LSTM/GRU**, not plain RNN, for anything but very short
sequences. (3) Try **GRU** first if you want speed; LSTM for maximum capacity. (4) Use
**bidirectional** RNNs when the whole sequence is available (e.g. text classification). (5)
**Gradient clipping** helps with exploding gradients. (6) For most modern NLP, **Transformers**
(Chapter 37) have largely replaced RNNs — but RNNs remain useful for time series and
streaming.
:::

## Applications

- **Natural Language Processing:** language modelling, sentiment analysis, named-entity
  recognition (historically RNN/LSTM; now mostly Transformers).
- **Machine translation:** sequence-to-sequence models (encoder-decoder LSTMs).
- **Speech recognition:** audio → text.
- **Time series forecasting:** stock prices, weather, demand (Chapter 42).
- **Music and text generation.**

## The shift to Transformers

RNNs process sequences **step by step**, which is slow (no parallelism) and still struggles
with very long-range dependencies. In 2017 the **Transformer** (Chapter 37) replaced
recurrence with **attention**, processing all positions in parallel and modelling long-range
context far better. Transformers now dominate NLP and beyond — but understanding RNNs
explains *why* attention was such a breakthrough.

## Advantages, disadvantages, and use cases

| Advantages | Disadvantages |
|---|---|
| Handle variable-length sequences | Slow (sequential, no parallelism) |
| Capture order & context (memory) | Plain RNNs forget long-range info |
| Parameter sharing across time | Harder to train than feedforward nets |
| LSTM/GRU model long dependencies | Largely superseded by Transformers for NLP |

**Use cases:** time series forecasting, speech, streaming/online sequence data, and
historically all of NLP (now mostly Transformers).

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Using a plain RNN for long sequences.** It will forget early context
(vanishing gradients). Use LSTM or GRU.
:::

- **Mistake 2 — Wrong input shape** (RNNs need `(batch, seq_len, features)`).
- **Mistake 3 — Ignoring exploding gradients** (use gradient clipping).
- **Mistake 4 — Using a unidirectional RNN** when future context is available and helpful
  (use bidirectional).
- **Mistake 5 — Reaching for RNNs for modern NLP** when Transformers usually outperform them.
- **Mistake 6 — Forgetting that RNNs are sequential** and therefore slow to train on long
  sequences.

## Best practices

- **Use LSTM/GRU** over plain RNNs; **GRU** for speed, **LSTM** for capacity.
- **Shape inputs** as `(batch, seq_len, features)`.
- **Use bidirectional** RNNs when the full sequence is available.
- **Clip gradients** to avoid explosions.
- **Consider Transformers** (Chapter 37) for NLP; reserve RNNs for time series/streaming.

## Chapter Summary

- **RNNs** process sequences one step at a time, maintaining a **hidden state** (memory) via
  the recurrence `hₜ = tanh(Wₕhₜ₋₁ + Wₓxₜ + b)`, capturing **order and context**.
- Plain RNNs suffer **vanishing gradients**, giving them only **short memory**.
- **LSTM** fixes this with a **cell state** and three **gates** (forget, input, output) for
  **long-term memory**; **GRU** is a simpler, faster two-gate variant. **Bidirectional**
  RNNs add future context.
- We trained an LSTM that learned to classify length-10 sequences by their sum (**0.948**),
  demonstrating real sequence memory.
- RNNs power time series, speech, and (historically) NLP, but **Transformers** (Chapter 37)
  have largely replaced them for language by adding **attention** and parallelism.

---

::: {.qband}
Practice Zone — Chapter 35
:::

## Multiple-Choice Questions (MCQs)

**Q1.** RNNs are designed for:
a) Images  b) Sequential data  c) Tabular data only  d) Clustering

**Q2.** The RNN's "memory" is its:
a) Weights only  b) Hidden state  c) Loss  d) Filter

**Q3.** Plain RNNs struggle with long sequences because of:
a) Too many filters  b) Vanishing gradients  c) Pooling  d) Softmax

**Q4.** LSTMs solve long-term memory using:
a) Convolutions  b) Gates and a cell state  c) Pooling  d) Dropout only

**Q5.** How many gates does an LSTM have?
a) 1  b) 2  c) 3 (forget, input, output)  d) 4

**Q6.** A GRU compared to an LSTM is:
a) Slower with more gates  b) Simpler/faster with fewer gates  c) Identical  d) For images

**Q7.** A bidirectional RNN provides each position with context from:
a) Only the past  b) Only the future  c) Both past and future  d) Neither

**Q8.** RNNs have largely been replaced for NLP by:
a) CNNs  b) Transformers  c) SVMs  d) Decision trees

### MCQ Answers
**1:** b. **2:** b. **3:** b. **4:** b. **5:** c. **6:** b. **7:** c. **8:** b.

## Interview Questions (with answers)

**Q1. How does an RNN differ from a feedforward network?**
*Answer:* A feedforward network maps a fixed input to an output with no memory. An RNN
processes a sequence step by step, maintaining a hidden state that carries information from
previous steps, so its output at each step depends on prior context — giving it memory and
the ability to handle variable-length sequences.

**Q2. What is the vanishing gradient problem in RNNs?**
*Answer:* During backpropagation through time, gradients are multiplied across many time
steps and can shrink toward zero, so the network can't learn dependencies between distant
elements — it effectively forgets long-range context. (Gradients can also explode.)

**Q3. How do LSTMs solve the long-term dependency problem?**
*Answer:* LSTMs add a cell state that carries information across many steps largely
unchanged, regulated by three gates — forget (what to discard), input (what to store), and
output (what to emit). The gates let gradients flow over long ranges, enabling learning of
long-term dependencies.

**Q4. What is the difference between an LSTM and a GRU?**
*Answer:* Both are gated RNNs for long dependencies. The LSTM has three gates and a separate
cell state (more parameters, more capacity); the GRU simplifies this to two gates and no
separate cell state (fewer parameters, faster), often with comparable performance.

**Q5. When would you use an RNN/LSTM versus a Transformer?**
*Answer:* Transformers generally outperform RNNs on language tasks due to attention and
parallel training, so they're preferred for most NLP. RNNs/LSTMs remain useful for time
series, streaming/online data, and smaller-scale sequential problems where their
step-by-step processing fits.

## Scenario-Based Questions (with answers)

**Q1.** *Your sentiment model based on a plain RNN does poorly on long reviews, missing
context from the start. What's wrong and what do you change?*
*Answer:* The plain RNN suffers vanishing gradients and forgets early context. Switch to an
LSTM or GRU (and possibly bidirectional), which maintain long-term memory; or, for best
results on text, use a Transformer.

**Q2.** *You need real-time forecasting on a stream of sensor readings. Why might an RNN
suit this better than a Transformer here?*
*Answer:* RNNs process sequentially and maintain a running hidden state, naturally fitting
streaming/online data where you update with each new reading; they can be lighter for
on-device/time-series use, whereas Transformers typically operate on whole sequences and are
heavier.

**Q3.** *Your LSTM trains but throws shape errors. What input format does it expect?*
*Answer:* With `batch_first=True`, inputs must be `(batch_size, sequence_length,
num_features)`. Reshape your data accordingly (e.g. add a feature dimension for univariate
sequences).

## Logic-Based Questions (with answers)

**Q1.** Why does parameter sharing across time steps make sense for sequences?
*Answer:* Because the same kind of pattern can occur at any position in a sequence; reusing
the same weights at every step lets the network detect and process patterns regardless of
where they appear, and keeps the parameter count manageable for variable-length inputs.

**Q2.** In the example, why couldn't a memoryless model solve "is the sum positive"?
*Answer:* The label depends on *all* elements together; a model without memory that sees
elements independently can't accumulate the running sum across the sequence, whereas the
LSTM's hidden state carries that accumulated information to the end.

**Q3.** Why is an LSTM's cell state key to learning long-term dependencies?
*Answer:* The cell state can pass information across many time steps with minimal change
(the gates can choose to preserve it), so gradients don't vanish as quickly, allowing the
network to connect distant parts of the sequence.

## Practical Questions (with answers)

**Q1.** What input shape does a PyTorch LSTM (`batch_first=True`) expect?
*Answer:* `(batch_size, sequence_length, input_features)`.

**Q2.** In the code, what does `h[-1]` represent and why use it for classification?
*Answer:* The final hidden state after processing the whole sequence — a summary/memory of
the entire input — which is a natural fixed-size representation to feed the classifier.

**Q3.** Which would you choose for a quick, lighter recurrent model: LSTM or GRU, and why?
*Answer:* GRU — it has fewer gates and parameters than an LSTM, trains faster, and often
matches LSTM performance.

## Long Questions (with answers)

**Q1. Explain how RNNs process sequences, the vanishing-gradient limitation, and how LSTMs
and GRUs overcome it.**

*Answer:* An **RNN** processes a sequence element by element, maintaining a **hidden state**
that serves as memory. At each step it combines the current input with the previous hidden
state — hₜ = tanh(Wₕhₜ₋₁ + Wₓxₜ + b) — reusing the same weights at every step (parameter
sharing), which lets it handle variable-length sequences and capture order and context. The
limitation is the **vanishing gradient problem**: training uses backpropagation through
time, and gradients multiplied across many steps tend to shrink toward zero, so the network
cannot learn dependencies between distant elements — it forgets long-range context (and
gradients can also explode). **LSTMs** overcome this with a dedicated **cell state** that
flows through the sequence like a conveyor belt, regulated by three **gates**: the forget
gate removes irrelevant information, the input gate adds new relevant information, and the
output gate controls what is emitted. Because the cell state can carry information across
many steps largely unchanged, gradients propagate over long ranges, enabling **long-term
memory**. **GRUs** achieve similar benefits more simply, with two gates (reset and update)
and no separate cell state, giving fewer parameters and faster training with often
comparable accuracy. Bidirectional variants additionally provide future context. Together
these gated architectures made recurrent networks practical for real sequence tasks before
Transformers.

**Q2. Compare RNNs/LSTMs with Transformers for sequence modelling, explaining why
Transformers largely replaced RNNs in NLP.**

*Answer:* **RNNs/LSTMs** process sequences **sequentially**, carrying a hidden state from one
step to the next. This is intuitive and memory-efficient for streaming/time-series data, and
LSTMs/GRUs handle moderate long-range dependencies via gating. However, sequential
processing has two big drawbacks: it **cannot be parallelised** across time steps (each step
depends on the previous), making training slow on long sequences and large datasets; and even
LSTMs struggle to connect *very* distant elements. **Transformers** (Chapter 37) replace
recurrence with an **attention mechanism** that lets every position directly attend to every
other position, so they (1) **model long-range dependencies** far better by giving direct
paths between distant tokens, and (2) **process all positions in parallel**, dramatically
speeding training and enabling the scaling to massive datasets and models that produced
modern LLMs. Because language tasks benefit enormously from long-range context and from
training at scale, Transformers outperformed RNNs across NLP and largely replaced them. RNNs
remain relevant for **time-series forecasting and streaming** scenarios where step-by-step
processing and a running state are natural, but for language understanding and generation,
attention-based Transformers are now dominant.

## Exercises

1. Explain in your own words what an RNN's hidden state does.
2. Why do plain RNNs forget long-range context?
3. Name the three LSTM gates and what each controls.
4. State two differences between an LSTM and a GRU.
5. Give two real applications of recurrent networks.

## Mini-Project

**Project: Sequence modelling with an LSTM.**

1. Choose a sequence task: forecast a sine wave's next value, or classify sequences (e.g.
   trend up/down, sum positive).
2. Build an LSTM (or GRU) in PyTorch; shape inputs correctly; train it.
3. Report accuracy/error and compare LSTM vs GRU vs plain RNN on the same task.
4. (Stretch) Try a bidirectional version and compare.
5. Write a short report on memory and which architecture worked best. Save in
   `my-ml-journey/`.

## Assignments

1. **Coding:** Replace the LSTM in the chapter's code with a plain `nn.RNN` and a `nn.GRU`;
   compare accuracy and discuss why they differ.
2. **Coding:** Build an LSTM that forecasts the next value of a noisy sine wave; plot
   predictions vs truth.
3. **Conceptual:** Write one page explaining the vanishing-gradient problem and how LSTM
   gates address it.

::: tip
RNNs gave networks memory, but their sequential nature and limited long-range reach held them
back. Chapter 36, **Generative Models**, explores networks that *create* data (autoencoders
and GANs) — and then Chapter 37 reveals the **Transformer**, the architecture that changed
everything.
:::
