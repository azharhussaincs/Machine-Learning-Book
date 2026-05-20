# Generative Models (Autoencoders & GANs)

## Introduction

Until now, almost every model we built was **discriminative** — it took an input and
predicted a label or value (cat or dog? spam or not?). **Generative models** do something
far more creative: they **learn to create new data** that resembles the training data. They
can generate realistic faces of people who don't exist, turn text into images, compose
music, and remove noise from photos.

This chapter introduces the two classic families — **Autoencoders** and **GANs** — and
glances at **diffusion models**, the technology behind today's image generators. These ideas
lead directly into Generative AI (Chapter 43).

::: keyidea
**Discriminative models** learn the *boundary* between classes (`P(label | data)`).
**Generative models** learn the *distribution of the data itself* (`P(data)`), so they can
**sample new examples** from it. Learning to *create* is much harder than learning to
*classify* — but it's behind some of the most exciting AI today.
:::

By the end of this chapter you will be able to:

- Explain the difference between discriminative and generative models.
- Understand **autoencoders** (encoder–bottleneck–decoder) and their uses.
- Understand **GANs** (generator vs discriminator) and adversarial training.
- Know what **VAEs** and **diffusion models** are, and the applications and risks.

## Autoencoders

An **autoencoder** is a neural network trained to **reconstruct its own input**. It has two
halves joined by a narrow **bottleneck**:

![An autoencoder: the encoder compresses the input into a small bottleneck (the latent code), and the decoder reconstructs the input from it. Forced through the bottleneck, the network must learn the most essential features.](assets/images/ch36_autoencoder.png)

- **Encoder** — compresses the input into a small **latent code** (the bottleneck).
- **Decoder** — reconstructs the original input from that code.
- It's trained to make the **output match the input** (minimising reconstruction error),
  using *no labels* — it's **self-supervised** (the data is its own target).

Because the bottleneck is small, the network can't just copy — it must learn the **most
essential features** to reconstruct from. Uses include:

- **Dimensionality reduction** (a non-linear cousin of PCA, Chapter 28).
- **Denoising** — train it to reconstruct clean images from noisy ones.
- **Anomaly detection** — anomalies reconstruct poorly (high error), flagging them.
- **Pretraining / feature learning.**

### Practical: an autoencoder in PyTorch

```python
import torch, torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
torch.manual_seed(0)

X, _ = load_digits(return_X_y=True); X = X / 16.0          # 64 pixels, scaled to [0,1]
X_tr, X_te = train_test_split(X, test_size=0.2, random_state=42)
X_tr_t = torch.tensor(X_tr, dtype=torch.float32); X_te_t = torch.tensor(X_te, dtype=torch.float32)

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 8))   # 64 -> 8
        self.decoder = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 64), nn.Sigmoid())
    def forward(self, x):
        return self.decoder(self.encoder(x))               # reconstruct the input

ae = Autoencoder()
optimizer = torch.optim.Adam(ae.parameters(), lr=0.01)
loss_fn = nn.MSELoss()                                     # reconstruction error
for epoch in range(100):
    optimizer.zero_grad(); loss = loss_fn(ae(X_tr_t), X_tr_t); loss.backward(); optimizer.step()

with torch.no_grad():
    print("test reconstruction MSE:", round(loss_fn(ae(X_te_t), X_te_t).item(), 4))
print("compression: 64 -> 8 -> 64  (8x)")
```

**Output:**
```text
test reconstruction MSE: 0.0266
compression: 64 -> 8 -> 64  (8x)
```

The autoencoder compressed each 64-pixel digit into just **8 numbers** and reconstructed it
with low error (MSE 0.027) — learning the essential structure of digits with **no labels**.
That 8-number latent code is a compact, learned representation.

### Variational Autoencoders (VAEs)

A **VAE** is a generative upgrade of the autoencoder. Instead of a fixed code, it learns a
*distribution* in the latent space, so you can **sample** new latent codes and decode them
into **new, realistic data**. VAEs produce somewhat blurry but reliable generations and are
a foundational generative model.

## Generative Adversarial Networks (GANs)

**GANs** (2014, Ian Goodfellow) are a brilliant, game-theoretic approach to generation. Two
networks compete:

- **Generator** — tries to create **fake** data realistic enough to fool the discriminator.
- **Discriminator** — tries to tell **real** data from the generator's **fakes**.

![A GAN pits two networks against each other: the generator turns random noise into fake samples, while the discriminator tries to distinguish real data from fakes. Their competition drives the generator to produce ever more realistic data.](assets/images/ch36_gan.png)

Think of a **forger** (generator) and a **detective** (discriminator). The forger makes
fake paintings; the detective spots fakes. As they compete, the forger gets better and
better until its fakes are indistinguishable from real. This is **adversarial training** —
the two networks improve by trying to beat each other.

::: keyidea
The genius of GANs is the **adversarial game**: there's no fixed "correct output" to copy.
Instead, the discriminator *learns* what "realistic" means and provides that as a training
signal to the generator. This is how GANs learned to generate photorealistic faces of
people who have never existed.
:::

### GAN challenges

GANs are powerful but **notoriously hard to train**:

- **Mode collapse** — the generator produces only a few varieties, ignoring the data's
  diversity.
- **Training instability** — the two networks can fail to reach balance and oscillate or
  diverge.
- **No clear stopping signal** — there's no simple "loss going down = better".

These difficulties are why training GANs from scratch is left as an installable example
rather than a quick demo here:

```python
# GANs need careful tuning; this is a sketch of the structure.
# generator: noise -> fake sample;  discriminator: sample -> real/fake probability
# Train alternately: update D on real+fake, then update G to fool D.
# Libraries: PyTorch, or higher-level frameworks. See assignments.
```

## Diffusion models

The current state of the art for image generation is **diffusion models** (behind DALL·E,
Stable Diffusion, Midjourney). The idea: gradually add noise to images until they're pure
noise, then train a network to **reverse** the process — turning random noise back into a
realistic image, step by step. They're more stable to train than GANs and produce stunning,
diverse results. We discuss them more in Chapter 43.

## Applications and risks

| Applications | Risks |
|---|---|
| Image/art generation (DALL·E, Midjourney) | **Deepfakes** (fake video/audio of real people) |
| Data augmentation (synthetic training data) | Misinformation & fraud |
| Denoising, super-resolution, inpainting | Copyright and consent issues |
| Drug/molecule design | Bias amplification |
| Anomaly detection (autoencoders) | Detection arms race |

::: warning
**Generative AI is dual-use.** The same technology that creates art and augments data also
powers **deepfakes** and misinformation. As a practitioner, build responsibly, consider
consent and provenance, and be aware of detection and watermarking efforts (Chapter 48,
Responsible AI).
:::

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Expecting GANs to train easily.** They're famously unstable (mode collapse,
oscillation). Use proven architectures/tricks, or prefer VAEs/diffusion for stability.
:::

- **Mistake 2 — Confusing discriminative and generative models** (classify vs create).
- **Mistake 3 — Using too small a bottleneck** in an autoencoder (loses too much) or too
  large (just copies, learns little).
- **Mistake 4 — Treating autoencoder reconstruction as "generation"** — plain AEs reconstruct;
  VAEs/GANs/diffusion generate novel samples.
- **Mistake 5 — Ignoring the ethical risks** of generative models (deepfakes, consent).

## Best practices

- **Use autoencoders** for compression, denoising, and anomaly detection.
- **Use VAEs/diffusion** for stable generation; **GANs** for sharp images with careful
  tuning.
- **Choose the bottleneck size** to balance compression vs reconstruction.
- **Monitor for mode collapse** when training GANs.
- **Consider provenance, consent, and misuse** — build generative AI responsibly.

## Chapter Summary

- **Generative models** learn the data distribution to **create new data**, unlike
  discriminative models that only classify/predict.
- **Autoencoders** (encoder → bottleneck → decoder) learn compact representations by
  reconstructing their input (self-supervised); used for **compression, denoising, anomaly
  detection**. Our AE compressed 64 pixels to **8** with low error. **VAEs** extend them to
  generate new samples.
- **GANs** pit a **generator** against a **discriminator** in an **adversarial game**,
  producing highly realistic data — but are **hard to train** (mode collapse, instability).
- **Diffusion models** (denoise random noise into images) are today's state of the art for
  image generation.
- Generative AI is **dual-use**: powerful for art and augmentation, but enables **deepfakes**
  and misinformation — build it responsibly.

---

::: {.qband}
Practice Zone — Chapter 36
:::

## Multiple-Choice Questions (MCQs)

**Q1.** A generative model learns to:
a) Classify inputs  b) Create new data resembling the training data  c) Cluster points
d) Reduce dimensions only

**Q2.** An autoencoder is trained to:
a) Predict labels  b) Reconstruct its own input  c) Maximise margin  d) Sort data

**Q3.** The narrow middle of an autoencoder is the:
a) Filter  b) Bottleneck (latent code)  c) Gate  d) Kernel

**Q4.** In a GAN, the generator's job is to:
a) Classify real vs fake  b) Create fakes that fool the discriminator  c) Compress data
d) Label data

**Q5.** "Mode collapse" is a problem where the GAN generator:
a) Trains too slowly  b) Produces only a few varieties of output  c) Overfits the
discriminator  d) Uses too much memory

**Q6.** Which model type powers DALL·E / Stable Diffusion?
a) Decision trees  b) Diffusion models  c) KNN  d) Autoencoders only

**Q7.** Autoencoders can detect anomalies because anomalies:
a) Train faster  b) Have high reconstruction error  c) Are labelled  d) Are removed

**Q8.** A key ethical risk of generative models is:
a) Slow training  b) Deepfakes / misinformation  c) Too few parameters  d) Scaling

### MCQ Answers
**1:** b. **2:** b. **3:** b. **4:** b. **5:** b. **6:** b. **7:** b. **8:** b.

## Interview Questions (with answers)

**Q1. What's the difference between discriminative and generative models?**
*Answer:* Discriminative models learn the decision boundary between classes — P(label|data)
— to classify or predict (e.g. logistic regression, most classifiers). Generative models
learn the distribution of the data itself — P(data) — so they can generate new samples
resembling the training data (e.g. VAEs, GANs, diffusion models).

**Q2. How does an autoencoder work and what is it used for?**
*Answer:* It's a network with an encoder that compresses input into a small latent code
(bottleneck) and a decoder that reconstructs the input from it, trained to minimise
reconstruction error with no labels. The bottleneck forces it to learn essential features.
Uses: dimensionality reduction, denoising, anomaly detection, and pretraining.

**Q3. Explain how a GAN is trained.**
*Answer:* A GAN trains two networks adversarially: the generator turns random noise into fake
samples, and the discriminator tries to distinguish real from fake. They alternate updates —
the discriminator learns to catch fakes, the generator learns to fool it — and this
competition drives the generator toward producing realistic data.

**Q4. Why are GANs hard to train?**
*Answer:* The two-network game can fail to balance, causing instability and oscillation;
mode collapse can make the generator produce only a few outputs; and there's no
straightforward loss that signals "better", making convergence and stopping hard to judge.

**Q5. What are diffusion models?**
*Answer:* Generative models that learn to reverse a gradual noising process: training adds
noise to data until it's pure noise, and the model learns to denoise step by step, so at
generation time it turns random noise into realistic samples. They're more stable than GANs
and power modern image generators.

## Scenario-Based Questions (with answers)

**Q1.** *You want to detect rare manufacturing defects with very few defect examples. How
can an autoencoder help?*
*Answer:* Train the autoencoder only on normal (defect-free) items so it reconstructs them
well. Defective items, being unlike the training data, will reconstruct poorly (high error);
flagging high-reconstruction-error items detects anomalies without needing labelled defects.

**Q2.** *Your GAN produces almost identical images every time. What's happening and what can
you try?*
*Answer:* Mode collapse — the generator found a few outputs that fool the discriminator and
stopped diversifying. Remedies: architectural/loss tweaks (e.g. Wasserstein GAN, minibatch
discrimination), different learning rates, or switching to a VAE/diffusion model for more
stable, diverse generation.

**Q3.** *A client wants a model to generate realistic synthetic patient images to augment a
small medical dataset. What do you recommend and what cautions apply?*
*Answer:* Use a stable generative model (a VAE or diffusion model) to augment data, validating
that synthetic samples improve downstream performance without leaking real patient identity.
Cautions: privacy/consent, avoiding bias amplification, clearly labelling synthetic data, and
ensuring it doesn't degrade real-world reliability.

## Logic-Based Questions (with answers)

**Q1.** Why must an autoencoder's bottleneck be smaller than the input?
*Answer:* If the bottleneck were as large as (or larger than) the input, the network could
trivially copy the input through, learning nothing useful. A smaller bottleneck forces it to
compress and learn the most essential, informative features.

**Q2.** Why does the adversarial setup let a GAN generate realistic data without being told
exactly what to produce?
*Answer:* The discriminator learns, from real data, what "realistic" looks like and provides
a learned signal to the generator. Instead of copying fixed targets, the generator improves
by trying to fool an ever-improving critic, implicitly learning the data distribution.

**Q3.** Why do anomalies produce high reconstruction error in an autoencoder trained on
normal data?
*Answer:* The autoencoder learned to reconstruct only the normal patterns it was trained on;
anomalous inputs don't match those patterns, so the decoder can't reconstruct them well,
yielding high error — a useful anomaly signal.

## Practical Questions (with answers)

**Q1.** In the autoencoder, what loss is used and why?
*Answer:* Mean Squared Error (MSE) between the reconstruction and the input — it measures how
closely the output matches the original, which is exactly what reconstruction should
minimise.

**Q2.** How would you use a trained autoencoder for anomaly detection?
*Answer:* Pass inputs through it and compute the reconstruction error; flag inputs whose
error exceeds a threshold as anomalies (since they reconstruct poorly compared to normal
data).

**Q3.** What are the two networks in a GAN and their objectives?
*Answer:* The generator (creates fakes to fool the discriminator) and the discriminator
(classifies samples as real or fake). They have opposing objectives and train adversarially.

## Long Questions (with answers)

**Q1. Explain autoencoders and GANs: how each works, how they're trained, and what each is
used for.**

*Answer:* An **autoencoder** is a neural network that learns to reconstruct its own input
through a narrow **bottleneck**. The **encoder** compresses the input into a small latent
code, and the **decoder** reconstructs the input from that code; it is trained, with no
labels, to minimise reconstruction error (e.g. MSE), making it **self-supervised**. The
bottleneck forces the network to capture only the most essential features, which makes
autoencoders useful for **dimensionality reduction**, **denoising** (reconstruct clean from
noisy), **anomaly detection** (anomalies reconstruct poorly), and pretraining; a
**variational** autoencoder extends this to *generate* new samples by learning a latent
distribution. A **GAN** instead trains two competing networks: a **generator** that maps
random noise to fake samples, and a **discriminator** that tries to distinguish real data
from fakes. They are trained **adversarially** in alternation — the discriminator improves at
catching fakes, and the generator improves at fooling it — so the generator gradually learns
to produce highly realistic data without ever copying fixed targets, because the
discriminator supplies a learned notion of "realistic". GANs excel at sharp, photorealistic
image generation but are **hard to train** (mode collapse, instability). Thus autoencoders
are workhorses for representation, compression, and anomaly tasks, while GANs (and VAEs and
diffusion models) are used for generating new images, art, and synthetic data.

**Q2. Discuss generative models' applications and ethical risks, and how a responsible
practitioner should approach them.**

*Answer:* Generative models have transformative **applications**: creating images, art, and
music (DALL·E, Midjourney, diffusion models); **data augmentation** to expand scarce training
sets; **denoising, super-resolution, and inpainting** of images; **drug and molecule
design**; and **anomaly detection** via autoencoders. But they are inherently **dual-use** and
carry serious **risks**: **deepfakes** — realistic fake images, audio, and video of real
people — enable misinformation, fraud, harassment, and political manipulation; generative
models can **amplify biases** present in training data; they raise **copyright and consent**
questions about training data and outputs; and synthetic media fuels a detection "arms race".
A **responsible practitioner** should: obtain proper consent and respect copyright for
training data; clearly **label or watermark** synthetic content for provenance; assess and
mitigate bias; consider misuse potential before release and add safeguards; validate that
synthetic data genuinely helps without leaking private information; and stay aligned with
emerging regulation and the principles of Responsible AI (Chapter 48). The goal is to harness
generative power for benefit while actively minimising harm.

## Exercises

1. State the difference between a discriminative and a generative model with an example of
   each.
2. Describe the three parts of an autoencoder and what the bottleneck forces.
3. Explain the GAN forger–detective analogy in your own words.
4. What is mode collapse, and why is it a problem?
5. Give two beneficial applications and two risks of generative models.

## Mini-Project

**Project: Autoencoder for compression and anomaly detection.**

1. Train an autoencoder on the digits dataset (or images); report reconstruction error and
   the compression ratio.
2. Visualise a few originals vs reconstructions (Chapter 14).
3. Use reconstruction error to flag anomalies: feed in some corrupted/odd images and show
   they have higher error.
4. Try different bottleneck sizes and plot reconstruction error vs bottleneck size.
5. Write a short report on the compression vs quality trade-off. Save in `my-ml-journey/`.

## Assignments

1. **Coding:** Build a **denoising** autoencoder — add noise to inputs but train it to
   reconstruct the clean originals; show it removes noise.
2. **Coding (stretch):** Implement a simple GAN on a small dataset (e.g. 1-D distribution or
   tiny images) and observe training instability/mode collapse.
3. **Conceptual:** Write one page on the benefits and dangers of generative AI, including
   deepfakes and how society might respond.

::: tip
Autoencoders and GANs *create* data. But the architecture that truly transformed modern AI —
powering ChatGPT, translation, and beyond — is next. Chapter 37, **Transformers &
Attention**, reveals how "attention" replaced recurrence and unlocked the era of large
language models.
:::
