# Transformers & Attention

## Introduction

This is one of the most important chapters in the book. In 2017, a paper titled
**"Attention Is All You Need"** introduced the **Transformer** — an architecture that
replaced recurrence with a mechanism called **attention**. It was a turning point: the
Transformer powers **ChatGPT and all modern Large Language Models**, Google Translate,
modern speech systems, and even image and protein models. If you understand Transformers,
you understand the engine of the current AI era.

Recall the limits of RNNs (Chapter 35): they process sequences **one step at a time**
(slow, no parallelism) and still struggle with **long-range** dependencies. The
Transformer fixes both with a beautifully simple idea: let every word **directly look at
(attend to) every other word**, all at once.

::: keyidea
**Attention** lets a model, when processing each word, decide **which other words matter
most** and focus on them — directly, regardless of distance. This gives Transformers two
superpowers RNNs lack: they capture **long-range context** and they process the whole
sequence **in parallel** (enabling training on internet-scale data). That combination
unlocked Large Language Models.
:::

By the end of this chapter you will be able to:

- Explain the **attention mechanism** (query, key, value) and **self-attention**.
- Understand **multi-head attention** and **positional encoding**.
- Describe the **Transformer architecture** and why it beat RNNs.
- Distinguish the **encoder (BERT)** and **decoder (GPT)** families.

## The attention mechanism

Consider the sentence *"The animal didn't cross the street because **it** was tired."* What
does "it" refer to — the animal or the street? To understand "it", the model must **attend
to** "animal". Attention computes exactly this: for each word, how much should it focus on
every other word.

Attention uses three vectors per word, all learned:

- **Query (Q)** — "what am I looking for?"
- **Key (K)** — "what do I contain?"
- **Value (V)** — "what information do I carry?"

Each word's query is compared (dot product) with every word's key to get **attention
scores** (how relevant each other word is). These are scaled, softmaxed into weights that
sum to 1, and used to take a weighted sum of the **values**. The formula:

<div class="equation"><img class="eq" src="assets/images/eq_ch37_attention.png" alt="scaled dot-product attention"></div>

(The `√dₖ` scaling keeps the dot products from getting too large.) When Q, K, V all come
from the *same* sequence, it's called **self-attention** — words attending to other words
in the same sentence.

![Self-attention: each word forms a Query, Key, and Value. The query of one word is matched against the keys of all words to produce attention weights, which then mix the values. Here "it" attends strongly to "animal".](assets/images/ch37_attention.png)

### Self-attention from scratch

```python
import numpy as np
def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True)); return e / e.sum(axis=-1, keepdims=True)

np.random.seed(0)
Q = np.random.randn(3, 4)   # 3 tokens, each query a 4-dim vector
K = np.random.randn(3, 4)   # keys
V = np.random.randn(3, 4)   # values
d_k = Q.shape[1]

scores = Q @ K.T / np.sqrt(d_k)     # how relevant is each token to each other
weights = softmax(scores)           # attention weights (each row sums to 1)
output = weights @ V                # weighted sum of values

print("attention weights:")
print(np.round(weights, 3))
print("row sums:", np.round(weights.sum(axis=1), 3).tolist())
print("output shape:", output.shape)
```

**Output:**
```text
attention weights:
[[0.682 0.302 0.015]
 [0.291 0.696 0.013]
 [0.5   0.188 0.312]]
row sums: [1.0, 1.0, 1.0]
output shape: (3, 4)
```

### Explanation

- Each **row** of the weights shows how much one token attends to the three tokens; rows
  **sum to 1** (softmax). Token 0 attends mostly to itself (0.682) and token 1 (0.302),
  barely to token 2 (0.015).
- The **output** is each token's new representation — a blend of all tokens' **values**,
  weighted by relevance. That's the whole mechanism: *figure out what's relevant, then mix
  it in*. Modern models just do this at massive scale with learned Q/K/V projections.

## Multi-head attention

Instead of one attention computation, Transformers use **multiple "heads" in parallel**,
each learning to focus on different relationships (e.g. one head tracks grammar, another
tracks meaning). Their outputs are combined. This **multi-head attention** lets the model
capture many kinds of relationships simultaneously.

## Positional encoding

Attention processes all words **at once**, so — unlike an RNN — it has **no built-in sense
of order**. To fix this, Transformers add **positional encodings** to the word embeddings,
injecting information about each word's **position** in the sequence. Now the model knows
both *what* the words are and *where* they are.

## The Transformer architecture

The full Transformer stacks attention with simple feed-forward layers, plus two stabilising
tricks (residual/skip connections and layer normalisation, Chapters 33–34):

![The Transformer block: input embeddings + positional encoding flow through multi-head self-attention and a feed-forward network, each wrapped with residual connections and layer normalisation. Many such blocks are stacked.](assets/images/ch37_transformer.png)

A block contains: **multi-head self-attention → add & normalise → feed-forward → add &
normalise.** Stacking many blocks builds a deep, powerful model. The original Transformer
had an **encoder** (reads input) and a **decoder** (generates output) — ideal for
translation. Modern models often use just one half.

## Encoder vs decoder families

| Family | Type | Reads/Generates | Examples |
|---|---|---|---|
| **Encoder-only** | Understanding | Reads whole text (bidirectional) | **BERT** (search, classification) |
| **Decoder-only** | Generation | Generates text left-to-right | **GPT** family (ChatGPT) |
| **Encoder-decoder** | Seq-to-seq | Reads then generates | **T5**, translation models |

We dive into these and Large Language Models in Chapter 39.

## Why Transformers won

- **Parallelism** — process the whole sequence at once (not step by step), so they train
  *much* faster on modern hardware, enabling internet-scale training.
- **Long-range context** — any word can attend to any other directly, capturing
  dependencies RNNs miss.
- **Scalability** — performance keeps improving as you add data and parameters ("scaling
  laws"), which is *exactly* how we got from BERT/GPT-2 to today's giant LLMs.

::: keyidea
The Transformer's combination of **attention** (long-range understanding) and
**parallelism** (fast, scalable training) is *why* it displaced RNNs and *why* the AI of the
2020s — LLMs, chatbots, multimodal models — exists. Almost every state-of-the-art model
today is a Transformer.
:::

::: tip
**Practical & debugging tips:** (1) You rarely build Transformers from scratch — use
**Hugging Face Transformers** (`pip install transformers`) to load pretrained models in a
few lines (Chapter 39). (2) Attention cost grows with the **square** of sequence length —
long inputs are expensive (active research area). (3) Always add **positional information**.
(4) Transformers are **data- and compute-hungry**; for small problems classic models or
fine-tuning a pretrained Transformer is far more practical than training one from scratch.
:::

## Beyond text

Transformers now dominate far beyond language: **Vision Transformers (ViT)** for images,
audio models, **AlphaFold** for protein structure, and **multimodal** models that handle
text + images together. Attention turned out to be a remarkably general idea.

## Advantages, disadvantages, and use cases

| Advantages | Disadvantages |
|---|---|
| Captures long-range context | Quadratic cost in sequence length |
| Massively parallel (fast training) | Very data- and compute-hungry |
| Scales with data/params (scaling laws) | Large models are expensive to run |
| General (text, images, audio, more) | Less interpretable; can hallucinate (LLMs) |
| Pretrained models easy to reuse | Training from scratch is impractical for most |

**Use cases:** language models and chatbots, translation, search, summarisation,
question-answering, code generation, image classification (ViT), speech, and multimodal AI.

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Forgetting positional encoding.** Without it, a Transformer is order-blind
("dog bites man" = "man bites dog"). Always inject position information.
:::

- **Mistake 2 — Training a Transformer from scratch** for a small task — fine-tune a
  pretrained one instead.
- **Mistake 3 — Ignoring the quadratic length cost** and feeding extremely long sequences.
- **Mistake 4 — Confusing encoder (BERT, understanding) and decoder (GPT, generation)**
  families.
- **Mistake 5 — Thinking attention "understands" like a human** — it computes
  statistical relevance, not true comprehension.

## Best practices

- **Use pretrained Transformers** (Hugging Face) and fine-tune; don't train from scratch.
- **Always add positional encoding.**
- **Match the family to the task**: encoder for understanding, decoder for generation.
- **Mind sequence length** (quadratic attention cost).
- **Leverage scaling** thoughtfully, but use the smallest model that meets your needs.

## Chapter Summary

- The **Transformer** (2017, "Attention Is All You Need") replaced recurrence with
  **attention** and now powers virtually all state-of-the-art AI, including LLMs.
- **Attention** uses **Query, Key, Value** vectors: each token's query is matched against
  all keys to get weights (softmax of `QKᵀ/√dₖ`), which mix the values — **self-attention**
  when within one sequence. We computed it from scratch (weights summing to 1).
- **Multi-head attention** captures many relationship types in parallel; **positional
  encoding** restores word order; **residuals + layer norm** stabilise deep stacks.
- Transformers won because of **long-range context + parallelism + scalability**; families
  include **encoder-only (BERT)**, **decoder-only (GPT)**, and **encoder-decoder (T5)**.
- They generalise beyond text (vision, audio, proteins, multimodal). Use **pretrained**
  models and fine-tune rather than training from scratch.

---

::: {.qband}
Practice Zone — Chapter 37
:::

## Multiple-Choice Questions (MCQs)

**Q1.** The Transformer architecture is built around:
a) Convolution  b) Recurrence  c) Attention  d) Pooling

**Q2.** Self-attention lets each word:
a) Ignore other words  b) Attend to (focus on) other words in the sequence  c) Become a
filter  d) Skip training

**Q3.** Attention uses which three vectors?
a) Input, Hidden, Output  b) Query, Key, Value  c) Forget, Input, Output  d) Mean, Var, Std

**Q4.** Transformers need positional encoding because attention:
a) Is too slow  b) Has no built-in sense of word order  c) Uses pooling  d) Can't scale

**Q5.** Multi-head attention allows the model to:
a) Use one relationship  b) Capture multiple relationship types in parallel  c) Avoid
training  d) Reduce parameters to zero

**Q6.** Which model family is decoder-only and used for generation?
a) BERT  b) GPT  c) ResNet  d) LSTM

**Q7.** A key reason Transformers beat RNNs is:
a) They're sequential  b) Parallel processing + long-range context  c) Fewer parameters
d) No data needed

**Q8.** Attention's computational cost grows with sequence length as:
a) Linear  b) Quadratic  c) Constant  d) Logarithmic

### MCQ Answers
**1:** c. **2:** b. **3:** b. **4:** b. **5:** b. **6:** b. **7:** b. **8:** b.

## Interview Questions (with answers)

**Q1. What is the attention mechanism?**
*Answer:* Attention lets a model, when processing each element, weigh how much to focus on
every other element. Using Query, Key, and Value vectors, it computes relevance scores
(query·key), softmaxes them into weights, and produces a weighted sum of the values — so
each token's new representation blends in the most relevant other tokens' information.

**Q2. Why did Transformers replace RNNs for NLP?**
*Answer:* RNNs process sequentially (slow, no parallelism) and struggle with long-range
dependencies. Transformers use attention so every token can directly attend to every other
(capturing long-range context) and process the whole sequence in parallel (fast training),
which also enabled scaling to massive data and models — the basis of modern LLMs.

**Q3. What is self-attention and multi-head attention?**
*Answer:* Self-attention is attention applied within a single sequence — words attend to
other words in the same sentence. Multi-head attention runs several attention computations
in parallel, each with its own learned projections, so the model can capture different kinds
of relationships (e.g. syntax and semantics) simultaneously, then combines them.

**Q4. Why do Transformers need positional encoding?**
*Answer:* Because attention processes all tokens simultaneously and is order-agnostic by
itself, it would treat a sentence as a bag of words. Positional encodings add information
about each token's position to its embedding, letting the model use word order.

**Q5. What's the difference between BERT and GPT?**
*Answer:* BERT is encoder-only and bidirectional — it reads the whole text at once and excels
at understanding tasks (classification, search). GPT is decoder-only and generates text
left-to-right — it excels at text generation (chatbots). Both are Transformers; they differ
in architecture and training objective.

## Scenario-Based Questions (with answers)

**Q1.** *You need to classify the sentiment of product reviews with high accuracy and have a
modest dataset. What modern approach do you take?*
*Answer:* Fine-tune a pretrained Transformer (e.g. a BERT-style encoder) using Hugging Face —
it brings powerful pretrained language understanding, so fine-tuning on your modest labelled
data typically beats training any model from scratch.

**Q2.** *Your Transformer model treats "dog bites man" and "man bites dog" identically. What
did you forget?*
*Answer:* Positional encoding. Without it, attention is order-blind, so word order is lost.
Adding positional encodings lets the model distinguish the two sentences.

**Q3.** *You try to feed a Transformer a 100,000-token document and it runs out of memory.
Why, and what are options?*
*Answer:* Attention cost scales quadratically with sequence length, so very long inputs are
expensive. Options: chunk the document, use long-context/efficient-attention variants, or
summarise/retrieve relevant passages (retrieval-augmented approaches, Chapter 39).

## Logic-Based Questions (with answers)

**Q1.** Why do attention weights sum to 1 across the keys for each query?
*Answer:* Because they're produced by a softmax over the relevance scores, which normalises
them into a probability distribution — so the output is a proper weighted average of the
values.

**Q2.** Why does parallel processing give Transformers a training-speed advantage over RNNs?
*Answer:* RNNs must compute step t before t+1 (sequential dependency), preventing
parallelisation across time. Transformers compute all positions' attention simultaneously,
fully utilising parallel hardware (GPUs/TPUs) and drastically speeding training on long
sequences and large datasets.

**Q3.** Why does the attention formula divide by √dₖ?**
*Answer:* For large key/query dimensions, the dot products grow large in magnitude, pushing
softmax into regions with tiny gradients. Dividing by √dₖ scales the scores back to a stable
range, keeping gradients healthy during training.

## Practical Questions (with answers)

**Q1.** In the from-scratch code, what does `softmax(Q @ K.T / sqrt(d_k))` produce?
*Answer:* The attention weight matrix — each row is a softmax-normalised distribution over
the keys, indicating how much each query token attends to every token.

**Q2.** How would you use a pretrained Transformer in practice?
*Answer:* Load it with the Hugging Face `transformers` library (e.g. `pipeline(...)` or
`AutoModel.from_pretrained(...)`) and either use it directly or fine-tune it on your task —
avoiding training from scratch.

**Q3.** What does each attention "head" learn?
*Answer:* A different type of relationship/pattern among tokens (e.g. one head may track
subject–verb agreement, another long-range coreference), and their results are combined for
a richer representation.

## Long Questions (with answers)

**Q1. Explain the attention mechanism and the Transformer architecture, and why they
revolutionised AI.**

*Answer:* **Attention** computes, for each element of a sequence, how much it should focus on
every other element. Each token is projected into a **Query**, **Key**, and **Value** vector;
the token's query is dotted with all keys to produce relevance scores, which are scaled by
√dₖ and passed through a **softmax** to get weights summing to 1, and these weights take a
weighted sum of the **values** — so each token's new representation is a relevance-weighted
blend of all tokens (self-attention when within one sequence). The **Transformer** stacks
this into blocks: **multi-head self-attention** (several attention computations in parallel,
each capturing different relationships) followed by a **feed-forward network**, each wrapped
with **residual connections and layer normalisation** for stable deep training; since
attention is order-agnostic, **positional encodings** are added to inject word order. The
original design had an **encoder** and **decoder**, but modern models often use one half
(BERT = encoder, GPT = decoder). Transformers **revolutionised AI** for three reasons:
**parallelism** (the whole sequence is processed at once, unlike sequential RNNs, enabling
fast training on huge data), **long-range context** (any token can attend directly to any
other, capturing dependencies RNNs miss), and **scalability** (quality keeps improving with
more data and parameters — scaling laws). Together these properties produced the Large
Language Models and multimodal systems that define modern AI.

**Q2. Compare RNNs and Transformers and explain the encoder/decoder model families with their
uses.**

*Answer:* **RNNs** (Chapter 35) process sequences step by step, carrying a hidden state; they
are memory-efficient and natural for streaming but are **sequential** (can't parallelise
across time, so slow to train on long sequences) and struggle with **long-range
dependencies**. **Transformers** replace recurrence with **attention**, letting every token
attend to every other directly — capturing long-range context — and process all positions
**in parallel**, which makes training fast and enables internet-scale models; their main
costs are **quadratic** attention scaling with length and heavy data/compute needs. Within
Transformers, three **families** serve different goals: **encoder-only** models like **BERT**
read the entire input bidirectionally and produce rich representations, excelling at
*understanding* tasks (classification, search, question answering); **decoder-only** models
like the **GPT** family generate text left-to-right by predicting the next token, excelling
at *generation* (chatbots, code, writing) and underpinning modern LLMs; and **encoder-decoder**
models like **T5** read an input then generate an output, ideal for *sequence-to-sequence*
tasks like translation and summarisation. In practice, RNNs remain useful for time-series and
streaming, but for language and most sequence tasks Transformers — typically used via
pretrained models that are fine-tuned — are now dominant.

## Exercises

1. Explain Query, Key, and Value in your own words with the "it/animal" sentence.
2. Why do attention weights sum to 1, and how is that achieved?
3. Why do Transformers need positional encoding but RNNs don't?
4. State two reasons Transformers train faster than RNNs.
5. Give the typical use of an encoder-only vs a decoder-only model.

## Mini-Project

**Project: Attention from scratch + a pretrained Transformer.**

1. Implement scaled dot-product self-attention from scratch (as in this chapter); verify the
   weights sum to 1 and inspect what each token attends to.
2. (`pip install transformers`) Load a small pretrained Transformer with Hugging Face
   `pipeline` and run sentiment analysis or text generation on a few sentences.
3. Visualise an attention weight matrix as a heatmap (Chapter 14).
4. Compare a Transformer's output quality to your Chapter 35 LSTM on a small text task.
5. Write a short report on what attention let the model do. Save in `my-ml-journey/`.

## Assignments

1. **Coding:** Extend the from-scratch attention to **multi-head** (run it for 2 heads with
   different random projections and concatenate the outputs).
2. **Coding:** Use Hugging Face to fine-tune (or zero-shot) a pretrained model on a small text
   classification task and report accuracy.
3. **Conceptual:** Write one page explaining why "attention + parallelism + scale" produced
   the modern LLM era, connecting to Chapters 35 and 39.

::: tip
**Part VI complete!** You now understand the full deep-learning stack — neural nets, training,
CNNs, RNNs, generative models, and Transformers. **Part VII** applies all of this to real
domains: NLP, **Large Language Models**, computer vision, recommenders, time series, and
generative AI.
:::
