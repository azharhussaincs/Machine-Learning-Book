# Large Language Models (LLMs)

## Introduction

In late 2022, an AI chatbot reached 100 million users faster than any product in history.
**Large Language Models (LLMs)** — the technology behind ChatGPT, Claude, Gemini, and
others — can write essays, answer questions, generate and explain code, translate, and
hold conversations. They feel almost magical. This chapter demystifies them: an LLM is, at
its core, a very large **Transformer** (Chapter 37) trained to do one deceptively simple
thing — **predict the next word** — at an enormous scale.

::: keyidea
An LLM is fundamentally a **next-token predictor**: given some text, it predicts the most
likely next token (word/sub-word), then the next, and so on. Trained on a huge fraction of
the internet via **self-supervised learning** (Chapter 30), this simple objective, at
massive scale, produces surprisingly broad abilities — a phenomenon called **emergence**.
:::

By the end of this chapter you will be able to:

- Explain what an LLM is and the **next-token** principle.
- Describe how LLMs are trained: **pretraining → fine-tuning → RLHF**.
- Understand **tokens, context windows, prompting**, and **scaling laws**.
- Know LLM **limitations** (hallucination, knowledge cutoff) and tools like **RAG**.

## The core idea: predicting the next token

At heart, an LLM repeatedly answers: *"given the text so far, what comes next?"* Let's see
this principle with a tiny "language model" that learns which word tends to follow another.

```python
from collections import defaultdict, Counter

text = ("the cat sat on the mat the cat ate the food the dog sat on the rug "
        "the dog ate the bone the cat and the dog sat together").split()

model = defaultdict(Counter)            # for each word, count what follows it
for a, b in zip(text[:-1], text[1:]):
    model[a][b] += 1

def next_word(w):                       # most likely next words and their probabilities
    c = model[w]; total = sum(c.values())
    return {k: round(v / total, 2) for k, v in c.most_common(3)}

print("After 'the':", next_word("the"))
print("After 'sat':", next_word("sat"))

# Generate greedily: always pick the most likely next word
w = "the"; out = [w]
for _ in range(8):
    w = model[w].most_common(1)[0][0]; out.append(w)
print("generated:", " ".join(out))
```

**Output:**
```text
After 'the': {'cat': 0.3, 'dog': 0.3, 'mat': 0.1}
After 'sat': {'on': 0.67, 'together': 0.33}
generated: the cat sat on the cat sat on the
```

### Explanation

- This toy model learned, from text, that "the" is often followed by "cat" or "dog", and
  "sat" by "on". An LLM does *exactly this* — predict the next token from probabilities —
  but with a giant Transformer, billions of parameters, and trillions of words of training.
- Notice the **greedy generation repeats** ("the cat sat on the cat sat on the…"). Real
  LLMs avoid this by **sampling** with a **temperature** (controlling randomness) instead
  of always picking the single most likely token — which is why they produce varied,
  natural text.

::: keyidea
That tiny demo *is* the essence of an LLM. The leap to ChatGPT is **scale and
architecture**: replace bigram counts with a deep Transformer that uses **attention**
(Chapter 37) over thousands of tokens of context, trained on a huge corpus. Scale turns
"predict the next word" into apparent reasoning, knowledge, and conversation.
:::

## How LLMs are trained: three stages

![LLM training stages: (1) self-supervised pretraining on massive text (predict the next token), (2) supervised fine-tuning on instruction examples, (3) RLHF — aligning to human preferences. Each stage shapes a more useful, aligned assistant.](assets/images/ch39_training.png)

1. **Pretraining (self-supervised).** The model learns language by predicting the next
   token across a massive text corpus (much of the internet, books, code). No human labels
   — the text *is* the label (Chapter 30). This is hugely expensive and produces a "base
   model" that knows language and facts but isn't yet a helpful assistant.
2. **Supervised fine-tuning (instruction tuning).** The base model is fine-tuned on
   curated examples of instructions and good responses, teaching it to *follow
   instructions* and be helpful.
3. **RLHF (Reinforcement Learning from Human Feedback).** Humans rank model responses; a
   reward model learns those preferences; then reinforcement learning (Chapter 31) tunes
   the LLM to produce responses humans prefer — making it more helpful, honest, and
   harmless ("alignment").

## Tokens, context windows, and prompting

- **Tokens** — LLMs work in **tokens** (word pieces), not whole words. "unhappiness" might
  be `un + happi + ness`. Pricing and limits are measured in tokens.
- **Context window** — the maximum tokens the model can "see" at once (its short-term
  memory). Bigger windows allow longer documents/conversations.
- **Prompting** — how you instruct the model. Techniques:
  - **Zero-shot** — just ask ("Translate this to French: …").
  - **Few-shot** — give a few examples in the prompt to guide the format.
  - **Chain-of-thought** — ask it to "think step by step", which improves reasoning.

::: tip
**Prompt engineering matters.** Clear instructions, relevant context, examples, and asking
for step-by-step reasoning can dramatically improve outputs from the *same* model. It's a
practical skill worth developing.
:::

## Scaling laws and emergence

A striking discovery: LLM performance improves **predictably** as you increase model size,
data, and compute (**scaling laws**). And beyond certain scales, **new abilities emerge**
that smaller models simply don't have (e.g. multi-step reasoning, in-context learning).
This is *why* the field raced to build ever-larger models — bigger reliably meant better.

## Limitations and risks

::: warning
**LLMs hallucinate.** They generate plausible-sounding text, which can be **confidently
wrong** — they predict likely words, not verified truth. Never trust an LLM's facts without
checking, especially for medical, legal, or factual claims.
:::

- **Hallucination** — fabricating facts, citations, or code that looks right but isn't.
- **Knowledge cutoff** — they only "know" data up to their training date.
- **No true understanding** — they model statistical patterns, not genuine comprehension.
- **Bias** — they reflect biases in their training data.
- **Cost & compute** — large models are expensive to train and run.
- **Misuse** — misinformation, spam, plagiarism, and security concerns.

## RAG: giving LLMs fresh, factual knowledge

**Retrieval-Augmented Generation (RAG)** is a key technique to reduce hallucination and add
up-to-date or private knowledge: before answering, the system **retrieves** relevant
documents (from a database or search) and includes them in the prompt, so the LLM answers
**grounded in real sources** rather than only its memory. RAG powers most enterprise LLM
applications (chatbots over company docs, etc.).

## Using LLMs in practice

Three main ways to adapt an LLM to your needs:

| Approach | What it is | When to use |
|---|---|---|
| **Prompting** | Craft instructions/examples | Most tasks; fastest, no training |
| **RAG** | Retrieve facts into the prompt | Need fresh/private/factual grounding |
| **Fine-tuning** | Further-train on your data | Need a specific style/format/behaviour at scale |

You access LLMs via **APIs** (e.g. provider SDKs) or run **open models** (Llama, Mistral)
locally with libraries like **Hugging Face Transformers** (`pip install transformers`).

::: tip
**Building with LLMs (practical):** start with **prompting**; add **RAG** when you need
factual grounding or private data; **fine-tune** only when prompting/RAG can't achieve the
needed behaviour. Always **validate outputs**, handle hallucinations, and consider cost,
latency, and privacy. Modern Claude/GPT-class models are accessed via simple APIs — a few
lines of code.
:::

## Advantages, disadvantages, and use cases

| Advantages | Disadvantages |
|---|---|
| Broad, general capabilities | Hallucinations (confidently wrong) |
| Few-shot / zero-shot (little data needed) | Expensive to train/run large models |
| Natural language interface | Knowledge cutoff; no true understanding |
| Strong at text, code, reasoning aids | Bias and misuse risks |

**Use cases:** chatbots and virtual assistants, writing and summarisation, code generation
and explanation, customer support (with RAG), translation, data extraction, and as building
blocks of "agentic" AI systems.

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Trusting LLM outputs as facts.** They can hallucinate convincingly. Verify
important claims and use RAG/citations for factual tasks.
:::

- **Mistake 2 — Thinking the LLM "understands"** — it predicts likely tokens, not truth.
- **Mistake 3 — Ignoring the context window** (feeding more than it can attend to).
- **Mistake 4 — Fine-tuning when prompting/RAG would do** (costly and often unnecessary).
- **Mistake 5 — Poor prompts** then blaming the model — prompt quality hugely affects
  output.
- **Mistake 6 — Forgetting the knowledge cutoff** and asking about recent events without
  RAG.

## Best practices

- **Prompt clearly**; use few-shot examples and chain-of-thought for hard tasks.
- **Use RAG** for factual, fresh, or private knowledge to reduce hallucination.
- **Validate and fact-check** outputs, especially in high-stakes domains.
- **Prefer prompting/RAG over fine-tuning** unless truly needed.
- **Mind cost, latency, privacy, and bias**; follow Responsible AI (Chapter 48).

## Chapter Summary

- An **LLM** is a very large **Transformer** trained to **predict the next token**;
  at scale, this simple objective yields broad, **emergent** abilities.
- Training has three stages: **self-supervised pretraining** (next-token on massive text),
  **supervised fine-tuning** (instruction following), and **RLHF** (aligning to human
  preferences).
- LLMs work in **tokens** within a **context window**; **prompting** (zero-/few-shot,
  chain-of-thought) steers them, and **scaling laws** explain why bigger models got better.
- They **hallucinate**, have a **knowledge cutoff**, lack true understanding, and carry
  bias/misuse risks — mitigated by **RAG** (grounding answers in retrieved sources) and
  validation.
- Adapt LLMs via **prompting → RAG → fine-tuning** (in that order of preference); access
  them via APIs or open models — but always verify outputs.

---

::: {.qband}
Practice Zone — Chapter 39
:::

## Multiple-Choice Questions (MCQs)

**Q1.** At its core, an LLM is trained to:
a) Cluster text  b) Predict the next token  c) Translate only  d) Classify images

**Q2.** LLMs are built on which architecture?
a) CNN  b) RNN  c) Transformer  d) Decision tree

**Q3.** Pretraining an LLM is an example of:
a) Supervised learning  b) Self-supervised learning  c) Reinforcement learning only  d)
Clustering

**Q4.** RLHF stands for Reinforcement Learning from:
a) Hidden Features  b) Human Feedback  c) High Frequency  d) Hyperparameter Functions

**Q5.** When an LLM produces confident but false information, it is:
a) Overfitting  b) Hallucinating  c) Underfitting  d) Clustering

**Q6.** RAG reduces hallucination by:
a) Training longer  b) Retrieving relevant documents into the prompt  c) Lowering
temperature  d) Removing tokens

**Q7.** The maximum tokens an LLM can attend to at once is its:
a) Vocabulary  b) Context window  c) Learning rate  d) Batch size

**Q8.** "Chain-of-thought" prompting asks the model to:
a) Answer instantly  b) Reason step by step  c) Use fewer tokens  d) Translate

### MCQ Answers
**1:** b. **2:** c. **3:** b. **4:** b. **5:** b. **6:** b. **7:** b. **8:** b.

## Interview Questions (with answers)

**Q1. What is a Large Language Model and how does it work?**
*Answer:* An LLM is a very large Transformer trained to predict the next token given prior
text. By learning this objective over a massive corpus, it captures language patterns,
facts, and reasoning aids; at inference it generates text token by token from the
probabilities it has learned, using its attention mechanism over the context.

**Q2. Describe the stages of training an LLM.**
*Answer:* (1) Self-supervised **pretraining** on huge text via next-token prediction,
producing a base model; (2) **supervised fine-tuning** on instruction-response examples to
follow instructions; (3) **RLHF**, where human preference rankings train a reward model and
reinforcement learning aligns the LLM to produce preferred (helpful, honest, harmless)
responses.

**Q3. What is hallucination and how do you mitigate it?**
*Answer:* Hallucination is when an LLM generates plausible but false content, because it
predicts likely tokens rather than verified facts. Mitigations: Retrieval-Augmented
Generation (ground answers in retrieved sources), asking for citations, lowering
temperature, constraining the task, and always validating high-stakes outputs.

**Q4. When would you fine-tune an LLM versus use prompting or RAG?**
*Answer:* Use prompting first (fast, no training). Use RAG when you need fresh, private, or
factual grounding. Fine-tune only when you need a consistent specialised style, format, or
behaviour at scale that prompting/RAG can't achieve, and you have suitable data — since
fine-tuning is costlier and less flexible.

**Q5. What are scaling laws and emergence?**
*Answer:* Scaling laws describe how LLM performance improves predictably as model size,
data, and compute increase. Emergence refers to new capabilities (e.g. multi-step
reasoning, in-context learning) that appear only beyond certain scales and aren't present in
smaller models.

## Scenario-Based Questions (with answers)

**Q1.** *You're building a customer-support bot that must answer from your company's latest
policy documents. Which LLM approach and why?*
*Answer:* RAG. Retrieve the relevant policy documents and include them in the prompt so the
LLM answers grounded in current, company-specific sources — reducing hallucination and
keeping answers up to date without retraining the model.

**Q2.** *Your LLM confidently cites a research paper that doesn't exist. What happened and
what do you do?*
*Answer:* It hallucinated — fabricating a plausible citation. Don't trust LLM facts blindly;
add RAG with a real source database, require it to cite retrieved documents, and verify
references before use.

**Q3.** *A simple task works with a good prompt, but a colleague wants to fine-tune a model
for it. What's your advice?*
*Answer:* Avoid unnecessary fine-tuning. If prompting (and optionally RAG) already solves
the task, it's faster, cheaper, and more flexible. Reserve fine-tuning for cases where you
need specific behaviour/format at scale that prompting can't reliably achieve.

## Logic-Based Questions (with answers)

**Q1.** Why does greedy "always pick the most likely next word" generation tend to repeat?
*Answer:* Because it deterministically follows the highest-probability path, it can enter
loops where a sequence's most likely continuation cycles back, producing repetition.
Sampling with temperature introduces controlled randomness to avoid this.

**Q2.** Why is next-token prediction considered self-supervised?
*Answer:* The training labels (the next token) come directly from the text itself — no human
annotation is needed — so the data provides its own supervision, fitting the definition of
self-supervised learning.

**Q3.** Why can an LLM be unaware of last week's news?
*Answer:* Because of its knowledge cutoff — it only learned from data up to its training
date. Without RAG or tools to fetch current information, it cannot know events after that
cutoff.

## Practical Questions (with answers)

**Q1.** What is a "token" in an LLM?
*Answer:* A sub-word unit the model processes (words can split into several tokens, e.g.
"unhappiness" → "un"+"happi"+"ness"). Context limits and pricing are measured in tokens.

**Q2.** Name two prompting techniques that improve LLM outputs.
*Answer:* Few-shot prompting (include examples in the prompt) and chain-of-thought (ask the
model to reason step by step).

**Q3.** What library lets you run open LLMs locally?
*Answer:* Hugging Face `transformers` (`pip install transformers`), which loads pretrained
open models like Llama or Mistral.

## Long Questions (with answers)

**Q1. Explain how an LLM works and how it is trained, from the next-token objective through
to a helpful aligned assistant.**

*Answer:* An LLM is a very large **Transformer** whose fundamental objective is **next-token
prediction**: given the preceding text, output a probability distribution over the next
token, and generate text by repeatedly predicting and appending tokens. Training proceeds in
stages. First, **self-supervised pretraining** runs next-token prediction over a massive
corpus (much of the internet, books, code); because the next token *is* the label, no human
annotation is needed, and the model learns grammar, facts, styles, and reasoning patterns,
yielding a capable but raw **base model**. Second, **supervised fine-tuning (instruction
tuning)** further trains the model on curated instruction-response pairs so it learns to
follow instructions and be helpful rather than merely continue text. Third, **RLHF** aligns
it with human preferences: humans rank candidate responses, a reward model learns those
preferences, and reinforcement learning (Chapter 31) tunes the LLM to generate responses
people prefer — more helpful, honest, and harmless. At inference, the model uses **attention**
over a **context window** of tokens and generates by **sampling** (with a temperature)
rather than always taking the top token, producing varied, natural output. The remarkable
result is that a simple objective — predict the next token — at enormous **scale** produces
broad, **emergent** abilities, which is the foundation of modern conversational AI.

**Q2. Discuss the limitations and risks of LLMs and the techniques used to make them more
reliable and useful.**

*Answer:* Despite their power, LLMs have serious **limitations**. They **hallucinate** —
generating fluent, confident, but false content (fake facts, citations, or code) — because
they predict likely tokens, not verified truth. They have a **knowledge cutoff**, unaware of
events after training. They possess **no genuine understanding**, modelling statistical
patterns rather than meaning, and they can reflect and amplify **biases** in their training
data. They are also **expensive** to train and run, and can be **misused** for
misinformation, spam, or other harms. Several **techniques** improve reliability and
usefulness: **Retrieval-Augmented Generation (RAG)** grounds answers in retrieved, current,
or private documents included in the prompt, sharply reducing hallucination and adding
fresh knowledge; **prompt engineering** (clear instructions, few-shot examples,
chain-of-thought) substantially improves outputs from the same model; **fine-tuning** adapts
behaviour/style when needed; **RLHF** aligns models to be safer and more helpful; and
practical safeguards — output validation, fact-checking, citations, controlling randomness
(temperature), and human oversight in high-stakes settings — manage residual errors. The
recommended adaptation order is **prompting → RAG → fine-tuning**, applying the lightest
sufficient approach, while always validating outputs and following Responsible-AI principles
(Chapter 48).

## Exercises

1. Explain, in one sentence, the single core task an LLM is trained to do.
2. Describe the three training stages of an LLM and what each adds.
3. What is hallucination, and name two ways to reduce it.
4. Explain the difference between zero-shot and few-shot prompting.
5. When would you choose RAG over fine-tuning?

## Mini-Project

**Project: Explore next-token generation and prompting.**

1. Extend the chapter's bigram model into a **trigram** model (predict from the previous two
   words) on a larger text; compare the generated text quality.
2. Add **sampling** (pick the next word randomly weighted by probability) instead of greedy;
   observe how repetition decreases.
3. (`pip install transformers`) Load a small open LLM and try zero-shot, few-shot, and
   chain-of-thought prompts on the same task; compare outputs.
4. Document a case where the model hallucinates and one where RAG-style context fixes it.
5. Write a short report on what scale and prompting changed. Save in `my-ml-journey/`.

## Assignments

1. **Coding:** Build the trigram language model and generate text with temperature-based
   sampling; show how temperature affects creativity vs coherence.
2. **Coding (stretch):** Use a Hugging Face `pipeline` for text generation or
   question-answering and experiment with prompts.
3. **Conceptual:** Write one page explaining why "predict the next token at scale" leads to
   broad capabilities, and the main risks society must manage.

::: tip
LLMs generate language. Chapter 40, **Computer Vision**, applies deep learning to the visual
world — image classification, object detection, and the transfer-learning techniques that
make state-of-the-art vision accessible.
:::
