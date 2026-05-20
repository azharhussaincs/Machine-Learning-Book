# Generative AI

## Introduction

We arrive at the technology defining this AI moment: **Generative AI** — systems that
**create brand-new content**: text, images, audio, video, and code. In the last few years,
tools like ChatGPT, Claude, DALL·E, Midjourney, Stable Diffusion, and Sora have moved AI
from *analysing* the world to *creating* in it. This chapter is the capstone of Part VII,
tying together the generative models (Chapter 36), Transformers (Chapter 37), and LLMs
(Chapter 39) into the bigger picture.

::: keyidea
**Generative AI = AI that produces new content** rather than just predicting a label or
number. Under the hood it's the techniques you've already learned — **LLMs/Transformers**
for text and code, **diffusion models** for images/video, and **GANs/VAEs** — now trained
at massive scale as **foundation models** that can be adapted to countless tasks.
:::

By the end of this chapter you will be able to:

- Explain what generative AI is and the technologies behind each modality.
- Understand **foundation models** and the paradigm shift they caused.
- Control generation (e.g. the **temperature** dial).
- Survey applications and grapple with the ethical issues.

## The technologies behind each modality

Generative AI isn't one technique — it's a family, matched to the data type:

| Modality | Main technology | Examples |
|---|---|---|
| **Text & code** | LLMs (Transformers, Ch 37/39) | ChatGPT, Claude, Copilot |
| **Images** | Diffusion models (Ch 36) | DALL·E, Midjourney, Stable Diffusion |
| **Audio / music / speech** | Transformers + diffusion | voice cloning, music generation |
| **Video** | Diffusion + Transformers | Sora, Runway |
| **Multimodal** | Unified Transformers | models that handle text + images + audio |

![The generative-AI landscape: different modalities (text, image, audio, video) powered by underlying technologies (LLMs/Transformers, diffusion models, GANs/VAEs), increasingly unified into multimodal foundation models.](assets/images/ch43_landscape.png)

### How diffusion models generate images

The dominant image generators are **diffusion models** (Chapter 36). The idea is elegant:

![A diffusion model learns to reverse noise. Training gradually adds noise to images until they're pure static; generation runs this backward — starting from random noise and denoising step by step into a coherent image, guided by a text prompt.](assets/images/ch43_diffusion.png)

1. **Forward process (training):** gradually add noise to real images until they become pure
   static.
2. **Reverse process (generation):** a network learns to **remove** noise step by step,
   turning random noise back into a realistic image — guided by a **text prompt** so you get
   the image you described.

Diffusion models are more **stable to train** than GANs and produce diverse, high-quality
results, which is why they dominate image generation.

## Foundation models: the paradigm shift

The biggest shift generative AI brought is the **foundation model**: a single, enormous
model **pretrained** on vast data (self-supervised, Chapter 30) that can then be **adapted**
to many downstream tasks via prompting, RAG, or fine-tuning (Chapter 39).

::: keyidea
**Old paradigm:** build and train a *separate* model for each task. **New paradigm:**
**pretrain one giant foundation model**, then adapt it to thousands of tasks. This is why a
single LLM can write code, summarise, translate, and answer questions — and why "AI
engineering" increasingly means *building on top of* foundation models rather than training
from scratch.
:::

## Controlling generation: the temperature dial

Generative models produce a probability distribution over what to generate next; **how you
sample** from it controls the output's character. The key knob is **temperature**:

```python
import numpy as np
words = ["cat", "dog", "sun", "sea", "car"]; probs = np.array([0.40, 0.30, 0.15, 0.10, 0.05])

def sample_with_temperature(p, T):
    logits = np.log(p) / T                 # divide log-probs by temperature
    e = np.exp(logits - logits.max()); p2 = e / e.sum()
    return {w: round(float(x), 3) for w, x in zip(words, p2)}

print("T=0.3 (focused):", sample_with_temperature(probs, 0.3))
print("T=1.0 (original):", sample_with_temperature(probs, 1.0))
print("T=2.0 (creative):", sample_with_temperature(probs, 2.0))
```

**Output:**
```text
T=0.3 (focused): {'cat': 0.698, 'dog': 0.268, 'sun': 0.027, 'sea': 0.007, 'car': 0.001}
T=1.0 (original): {'cat': 0.4, 'dog': 0.3, 'sun': 0.15, 'sea': 0.1, 'car': 0.05}
T=2.0 (creative): {'cat': 0.3, 'dog': 0.26, 'sun': 0.184, 'sea': 0.15, 'car': 0.106}
```

### Explanation

- **Low temperature (0.3)** sharpens the distribution toward the most likely option (cat
  0.70) → **focused, deterministic, "safe"** output.
- **High temperature (2.0)** flattens it toward uniform → **diverse, surprising,
  "creative"** (but riskier) output.
- This single dial is *why* the same generative model can be made precise (for code) or
  imaginative (for brainstorming). Most generation APIs expose it directly.

## Applications across industries

- **Content & marketing:** drafting copy, images, video, personalised campaigns.
- **Software:** code generation, completion, debugging, documentation.
- **Design & art:** concept art, product design, music.
- **Education:** tutoring, explanations, practice generation.
- **Healthcare & science:** drug/molecule design, synthetic data, literature synthesis.
- **Business:** chatbots, summarisation, knowledge assistants (with RAG).

## Limitations and ethics

::: warning
**Generative AI raises serious ethical issues** that you must take seriously as a
practitioner: **deepfakes** and misinformation, **copyright** (training data and outputs),
**bias** amplification, **hallucinated** facts presented confidently, **job displacement**,
**plagiarism/academic integrity**, and **privacy**. We dedicate Chapter 48 (Responsible AI)
to these. Build with consent, provenance, transparency, and safeguards.
:::

- **Hallucination** — fabricates plausible but false content (Chapter 39).
- **Deepfakes** — realistic fake media of real people.
- **Copyright & consent** — what data was it trained on; who owns the output?
- **Bias** — reflects and can amplify training-data biases.
- **Misuse** — spam, fraud, manipulation at scale.

## The agentic direction

The frontier is **AI agents** — generative models that don't just produce content but
**take actions**: using tools, calling APIs, browsing, writing and running code, and
chaining steps to accomplish goals. Combined with RAG and tool use, foundation models are
becoming the "reasoning engine" of autonomous systems — a major theme for the future
(Chapter 54).

## Advantages, disadvantages, and use cases

| Advantages | Disadvantages |
|---|---|
| Creates content across modalities | Hallucination / factual errors |
| Foundation models adapt to many tasks | High training cost & compute |
| Boosts productivity & creativity | Deepfakes, copyright, bias risks |
| Natural-language / prompt interface | Hard to control precisely; misuse potential |

**Use cases:** writing assistants, image/video/music creation, code generation, chatbots,
synthetic data, design, education, and AI agents.

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Trusting generated content blindly.** Generative AI hallucinates and can be
biased or wrong; verify facts, check for copyright/consent, and keep humans in the loop for
important decisions.
:::

- **Mistake 2 — Thinking it "understands" or is conscious** — it generates statistically
  likely content (Chapter 39).
- **Mistake 3 — Ignoring temperature/sampling settings** when output is too bland or too
  random.
- **Mistake 4 — Training a foundation model from scratch** — almost always adapt an existing
  one.
- **Mistake 5 — Overlooking ethics, provenance, and bias.**
- **Mistake 6 — Confusing modalities/technologies** (LLMs for text, diffusion for images).

## Best practices

- **Adapt foundation models** (prompt/RAG/fine-tune); don't train from scratch.
- **Tune sampling** (temperature) for the task — low for precision, higher for creativity.
- **Keep humans in the loop** and **verify** generated content.
- **Address ethics:** consent, provenance/watermarking, bias checks, transparency.
- **Match the technology to the modality.**

## Chapter Summary

- **Generative AI** creates new content (text, image, audio, video, code), powered by the
  techniques from earlier chapters: **LLMs/Transformers** (text/code), **diffusion models**
  (images/video), and **GANs/VAEs**.
- **Diffusion models** generate images by learning to **reverse a noising process** —
  denoising random static into a coherent image guided by a prompt.
- The **foundation-model** paradigm — pretrain one giant model, adapt to many tasks — is the
  central shift; building with AI now means building *on top of* these models.
- **Temperature** controls generation: low = focused/precise, high = diverse/creative.
- Generative AI is hugely powerful and broadly applicable, but carries serious risks
  (**hallucination, deepfakes, copyright, bias**) — to be addressed responsibly (Chapter 48),
  and is evolving toward **agentic** systems that take actions.

---

::: {.qband}
Practice Zone — Chapter 43
:::

## Multiple-Choice Questions (MCQs)

**Q1.** Generative AI is distinguished by its ability to:
a) Classify data  b) Create new content  c) Cluster points  d) Reduce dimensions

**Q2.** The dominant technology for text/code generation is:
a) Diffusion models  b) LLMs (Transformers)  c) Decision trees  d) KNN

**Q3.** Modern image generators (DALL·E, Stable Diffusion) are mainly:
a) GANs only  b) Diffusion models  c) RNNs  d) SVMs

**Q4.** A foundation model is:
a) A small task-specific model  b) A large pretrained model adapted to many tasks  c) A
database  d) A loss function

**Q5.** Lowering the generation temperature makes output:
a) More random  b) More focused/deterministic  c) Longer  d) Multimodal

**Q6.** Diffusion models generate images by:
a) Classifying pixels  b) Reversing a noising process (denoising)  c) Clustering  d) Pooling

**Q7.** Realistic fake media of real people is called:
a) Augmentation  b) A deepfake  c) A hallucination  d) Transfer learning

**Q8.** "AI agents" extend generative models by:
a) Only generating text  b) Taking actions (tools, APIs, multi-step tasks)  c) Removing
attention  d) Avoiding prompts

### MCQ Answers
**1:** b. **2:** b. **3:** b. **4:** b. **5:** b. **6:** b. **7:** b. **8:** b.

## Interview Questions (with answers)

**Q1. What is generative AI and what technologies power it?**
*Answer:* Generative AI creates new content (text, images, audio, video, code) rather than
just classifying or predicting. It's powered by LLMs/Transformers (text and code), diffusion
models (images and video), and GANs/VAEs — typically as large foundation models trained at
scale and adapted to specific tasks.

**Q2. How do diffusion models work?**
*Answer:* During training they progressively add noise to images until pure static, learning
to reverse each step. To generate, they start from random noise and iteratively denoise —
often guided by a text prompt — producing a realistic image. They're more stable than GANs
and yield diverse, high-quality outputs.

**Q3. What is a foundation model and why is it significant?**
*Answer:* A foundation model is a large model pretrained (self-supervised) on vast data that
can be adapted to many downstream tasks via prompting, RAG, or fine-tuning. It shifted the
paradigm from training a separate model per task to building on one general model — making AI
broadly capable and far cheaper to apply.

**Q4. What does the temperature parameter control?**
*Answer:* It controls the randomness of sampling from the model's output distribution. Low
temperature sharpens toward the most likely tokens (focused, deterministic), high temperature
flattens the distribution (diverse, creative, riskier). It tunes the precision-vs-creativity
trade-off.

**Q5. What are the main ethical concerns with generative AI?**
*Answer:* Hallucinated/false content, deepfakes and misinformation, copyright and consent
over training data and outputs, bias amplification, job displacement, plagiarism, and privacy.
Responsible practice includes verification, provenance/watermarking, bias checks,
transparency, and human oversight (Chapter 48).

## Scenario-Based Questions (with answers)

**Q1.** *You need a generative model to write precise, deterministic code snippets. How do
you set the temperature and why?*
*Answer:* Use a low temperature (e.g. 0.1–0.3) so the model samples the most likely,
"safest" tokens, producing focused, consistent, correct-leaning output — appropriate for code
where creativity/randomness causes bugs.

**Q2.** *A marketing team wants varied, creative ad slogans. What sampling setting and what
caution?*
*Answer:* Use a higher temperature for diversity/creativity, but caution that output may be
less coherent or factual — review and curate results, and check for bias or unintended
implications before publishing.

**Q3.** *Your company wants to deploy an image generator. What ethical safeguards should you
build in?*
*Answer:* Respect copyright/consent in training and outputs, add provenance/watermarking to
mark AI-generated content, filter harmful or deepfake misuse, mitigate bias, and keep human
review — aligning with Responsible AI (Chapter 48) and relevant regulation.

## Logic-Based Questions (with answers)

**Q1.** Why does dividing logits by a high temperature make output more random?
*Answer:* Dividing by a larger number shrinks the differences between log-probabilities, so
after softmax the probabilities become more equal (closer to uniform), making less-likely
options more probable — increasing randomness.

**Q2.** Why is the foundation-model paradigm more efficient than per-task models?
*Answer:* Because the expensive pretraining (learning general capabilities) is done once and
amortised across many tasks; adapting via prompting/RAG/fine-tuning is far cheaper than
training a new model per task, so total cost and data needs drop dramatically.

**Q3.** Why are diffusion models often preferred over GANs for image generation?
*Answer:* They are more stable to train (no adversarial min-max instability or mode collapse)
and produce diverse, high-quality, controllable outputs, making them more reliable for
large-scale image generation.

## Practical Questions (with answers)

**Q1.** Which technology would you use to generate (a) text, (b) images?
*Answer:* (a) An LLM/Transformer; (b) a diffusion model.

**Q2.** Write the idea of temperature sampling in one line.
*Answer:* Divide the log-probabilities by the temperature, re-softmax, and sample — lower T
sharpens toward the top choice, higher T flattens toward uniform.

**Q3.** Name two ways to adapt a foundation model to your task.
*Answer:* Prompting (zero/few-shot) and fine-tuning (also retrieval-augmented generation,
RAG).

## Long Questions (with answers)

**Q1. Explain what generative AI is, the technologies behind different modalities, and how
diffusion models and LLMs generate content.**

*Answer:* **Generative AI** refers to systems that **create new content** — text, images,
audio, video, and code — rather than only classifying or predicting. Different modalities use
different underlying technologies. **Text and code** are generated by **LLMs**, which are
large **Transformers** (Chapter 37) trained to predict the next token; they generate by
sampling tokens one after another from the probabilities they've learned, with **temperature**
controlling how focused or creative the output is. **Images and video** are generated mainly
by **diffusion models**: during training they progressively add noise to real images until
pure static, learning to reverse each step; to generate, they start from random noise and
**iteratively denoise**, guided by a text prompt, into a coherent image — a process more
stable than GANs and capable of diverse, high-quality results. **GANs and VAEs** (Chapter 36)
also generate images, and **multimodal** models increasingly handle several modalities at
once. Across all of them, the common modern pattern is the **foundation model** — a single
large model pretrained on vast data and then adapted — so the same engine can power many
generative applications.

**Q2. Discuss the opportunities and risks of generative AI, and what responsible deployment
looks like.**

*Answer:* Generative AI offers enormous **opportunities**: it boosts productivity and
creativity across writing, software development, design, education, science, and business; it
makes powerful capabilities accessible through a natural-language interface; and via
**foundation models** a single system can serve countless tasks. But it carries serious
**risks**: it **hallucinates** confident falsehoods; it enables **deepfakes** and
misinformation; it raises **copyright and consent** questions about training data and outputs;
it can **amplify biases**; it threatens some **jobs** and academic integrity; and it can be
misused for fraud or manipulation at scale. **Responsible deployment** therefore requires:
keeping **humans in the loop** and **verifying** generated content for important decisions;
ensuring **provenance** (watermarking/labelling AI-generated media); respecting **copyright
and consent**; testing for and mitigating **bias**; adding **safety filters** against harmful
uses; being **transparent** that content is AI-generated; protecting **privacy**; and aligning
with emerging **regulation** and the Responsible-AI principles of Chapter 48. The goal is to
capture generative AI's benefits while actively minimising its harms — a balance every
practitioner now shares responsibility for.

## Exercises

1. Match each to its technology: generating an essay, generating a photo, generating music.
2. Explain the foundation-model paradigm and why it's efficient.
3. Describe how a diffusion model turns noise into an image.
4. What does raising the temperature do to generated output?
5. List four ethical risks of generative AI.

## Mini-Project

**Project: Generative AI exploration.**

1. Implement temperature sampling (as in this chapter) on a small next-word model; generate
   text at temperatures 0.2, 0.7, and 1.5 and compare creativity vs coherence.
2. (Optional, with access) Use a text-generation API/open model and an image generator;
   experiment with prompts and settings.
3. Document one clear hallucination and one impressive result.
4. Write a half-page on an ethical concern you observed (e.g. bias, copyright, plausibility of
   false content).
5. Save your findings in `my-ml-journey/`.

## Assignments

1. **Coding:** Build a temperature-controlled text generator from a trigram model; show how
   temperature changes the output's style.
2. **Research:** Compare two generative-AI tools (e.g. an LLM and an image generator) — what
   technology powers each, and what are their strengths/limits?
3. **Conceptual:** Write one page on "the foundation-model era" and how it changes what it
   means to build AI applications.

::: tip
**Part VII complete!** You've applied ML across NLP, LLMs, vision, recommenders, time series,
and generative AI. But building a model is only half the job — **Part VIII** covers getting
models into the real world: **deployment, MLOps, cloud, edge, and responsible AI**.
:::
