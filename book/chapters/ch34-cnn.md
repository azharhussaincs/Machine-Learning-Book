# Convolutional Neural Networks (CNN)

## Introduction

**Convolutional Neural Networks (CNNs)** are the architecture that **sparked the deep-
learning revolution**. When AlexNet, a CNN, crushed the ImageNet competition in 2012
(Chapter 3), the entire field pivoted to deep learning overnight. CNNs are purpose-built
for **images** (and any grid-like data), and they power face recognition, medical imaging,
self-driving perception, and more.

Why a special architecture? An image is huge — even a small 200×200 colour image has
120,000 numbers. A regular MLP (Chapter 32) would need an enormous number of weights and
would ignore the **spatial structure** (that nearby pixels are related). CNNs solve both
problems elegantly.

::: keyidea
A CNN scans an image with small learnable **filters** that detect local patterns (edges,
textures) anywhere in the image. By **sharing** these filters across the whole image, it
uses far fewer parameters than an MLP *and* respects spatial structure. Stacking
convolutional layers builds a **hierarchy**: edges → shapes → objects.
:::

By the end of this chapter you will be able to:

- Explain why MLPs are poor for images and how **convolution** fixes it.
- Understand **filters, feature maps, stride, padding**, and **pooling**.
- Read a CNN architecture and famous models (LeNet, AlexNet, VGG, ResNet).
- Build and train a CNN in **PyTorch**.

## Why not just use an MLP for images?

Two big problems:

1. **Too many parameters.** Flattening a 200×200×3 image gives 120,000 inputs; a single
   hidden layer of 1,000 neurons would need 120 *million* weights — impossible to train
   well.
2. **No spatial awareness.** Flattening destroys the 2-D structure. An MLP treats a pixel
   in the corner the same as its neighbour, ignoring that images have *local* patterns.

CNNs fix both with **convolution** and **parameter sharing**.

## The convolution operation

A **filter (kernel)** is a small grid of weights (e.g. 3×3) that slides across the image.
At each position it computes a dot product with the patch underneath, producing one number.
Sliding it across the whole image produces a **feature map** that highlights *where* the
filter's pattern occurs.

![A 3×3 filter slides over the image; at each location it computes a weighted sum of the underlying patch, producing one value of the output feature map. Different filters detect different patterns (edges, corners, textures).](assets/images/ch34_convolution.png)

The key ideas:

- **Local connectivity:** each output sees only a small patch — capturing local patterns.
- **Parameter sharing:** the *same* filter is used everywhere, so an "edge detector"
  learned in one corner works across the whole image. This slashes the parameter count.
- **Many filters:** each conv layer learns *many* filters, producing many feature maps
  (each detecting a different pattern).

The output size of a convolution depends on the input width `W`, kernel size `K`, padding
`P`, and stride `S`:

<div class="equation"><img class="eq" src="assets/images/eq_ch34_convsize.png" alt="conv output size"></div>

- **Stride (S):** how far the filter jumps each step (larger stride → smaller output).
- **Padding (P):** adding a border of zeros so the output stays the same size (and edges
  aren't lost).

## Pooling: shrinking while keeping the essence

**Pooling** downsamples feature maps, reducing size (and computation) while keeping the
strongest signals. **Max pooling** (the most common) takes the maximum in each small window
(e.g. 2×2), making the network more robust to small shifts in the image.

![Max pooling slides a window (e.g. 2×2) over the feature map and keeps only the maximum value in each window, halving the width and height while retaining the strongest activations.](assets/images/ch34_pooling.png)

## A complete CNN architecture

A typical CNN stacks these building blocks:

![A typical CNN: alternating convolution + ReLU + pooling layers extract increasingly abstract features and shrink the spatial size, then flattened features feed fully-connected layers and a softmax output. Early layers detect edges; deeper layers detect objects.](assets/images/ch34_cnn_arch.png)

`[Conv → ReLU → Pool] × N → Flatten → Fully-Connected → Softmax`

- The **convolution + pooling** stack extracts features and shrinks spatial size.
- **Flatten** turns the final feature maps into a vector.
- **Fully-connected layers + softmax** make the final classification.
- Crucially, the early layers learn **edges**, middle layers learn **shapes/parts**, and
  deep layers learn **whole objects** — automatic hierarchical feature learning (Chapter
  32).

### Famous CNN architectures

| Model | Year | Significance |
|---|---|---|
| **LeNet-5** | 1998 | First successful CNN (digit recognition) |
| **AlexNet** | 2012 | Won ImageNet; ignited deep learning |
| **VGG** | 2014 | Showed depth matters (16–19 layers) |
| **ResNet** | 2015 | "Skip connections" enabled 100+ layers (solved vanishing gradients) |

## Practical: a CNN in PyTorch

Let's build a small CNN to classify the 8×8 digit images.

```python
import torch, torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
torch.manual_seed(0)

X, y = load_digits(return_X_y=True)
X = X.reshape(-1, 1, 8, 8) / 16.0          # shape: (n, channels, height, width); scale to [0,1]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_tr_t = torch.tensor(X_tr, dtype=torch.float32); y_tr_t = torch.tensor(y_tr, dtype=torch.long)
X_te_t = torch.tensor(X_te, dtype=torch.float32)

cnn = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, padding=1),  nn.ReLU(), nn.MaxPool2d(2),  # 8x8 -> 4x4
    nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 4x4 -> 2x2
    nn.Flatten(),
    nn.Linear(16 * 2 * 2, 10))                  # 10 digit classes

optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
for epoch in range(30):
    optimizer.zero_grad(); loss = loss_fn(cnn(X_tr_t), y_tr_t); loss.backward(); optimizer.step()

with torch.no_grad():
    acc = (cnn(X_te_t).argmax(1).numpy() == y_te).mean()
print("CNN test accuracy:", round(float(acc), 3))
print("total parameters:", sum(p.numel() for p in cnn.parameters()))
```

**Output:**
```text
CNN test accuracy: 0.919
total parameters: 1898
```

### Explanation

- **`Conv2d(1, 8, 3, padding=1)`** — 8 filters of size 3×3 on a 1-channel input; `padding=1`
  keeps the size at 8×8. **`MaxPool2d(2)`** halves it to 4×4.
- The second conv/pool block produces 16 feature maps of 2×2, which **Flatten** turns into a
  64-number vector for the final `Linear(64, 10)` classifier.
- **0.919 accuracy with only 1,898 parameters!** Compare that to an MLP, which would need
  *far* more parameters for similar performance. This **parameter efficiency** — thanks to
  filter sharing — is a core CNN advantage.

::: keyidea
The CNN matched strong accuracy with a *tiny* parameter count because each filter is reused
across the whole image, and pooling progressively shrinks the data. On large real images
this efficiency is the difference between feasible and impossible — which is exactly why
CNNs, not MLPs, power computer vision (Chapter 40).
:::

::: tip
**Practical & debugging tips:** (1) Images go in as `(batch, channels, height, width)`;
scale pixels to [0,1]. (2) Use `padding` to control output size; the conv formula
`(W−K+2P)/S+1` tells you the dimensions. (3) For real images, use **data augmentation**
(Chapter 33/40) and **transfer learning** (a pretrained model like ResNet) instead of
training from scratch. (4) Use `CrossEntropyLoss` for multiclass (it includes softmax). (5)
For large images/datasets you'll want a **GPU**. (6) Watch the channel/size bookkeeping —
shape errors are the most common CNN bug.
:::

## Transfer learning (a glimpse)

For real-world images you rarely train a CNN from scratch. Instead you take a model
**pretrained** on millions of images (e.g. ResNet on ImageNet), which already knows generic
features (edges, textures, shapes), and **fine-tune** its last layers on your specific task
with far less data. This **transfer learning** is the standard practice — covered in
Chapter 40.

## Advantages, disadvantages, and use cases

| Advantages | Disadvantages |
|---|---|
| Excellent for images/grid data | Needs lots of data & compute (GPU) |
| Parameter-efficient (filter sharing) | Less interpretable (black box) |
| Captures spatial structure | Many architecture choices |
| Translation-invariant (via pooling) | Sensitive to image size/orientation |
| Learns features automatically | Training from scratch is costly |

**Use cases:** image classification, object detection, face recognition, medical imaging,
satellite imagery, video analysis, and even non-image grid data (audio spectrograms, some
NLP).

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Shape/channel bugs.** Feeding the wrong tensor shape (forgetting the channel
dimension, or miscomputing the flattened size) is the #1 CNN error. Use the conv size
formula and print shapes.
:::

- **Mistake 2 — Using an MLP for images** (parameter explosion, ignores spatial structure).
- **Mistake 3 — Training from scratch** when transfer learning would work with far less data.
- **Mistake 4 — Forgetting to scale pixels** to [0,1] (or normalise).
- **Mistake 5 — No data augmentation**, leading to overfitting on small image sets.
- **Mistake 6 — Ignoring vanishing gradients** in very deep CNNs (use ResNet-style skip
  connections).

## Best practices

- **Use CNNs for images/grid data**; scale pixels and feed correct shapes.
- **Use transfer learning** (pretrained models) for real tasks.
- **Apply data augmentation** to reduce overfitting.
- **Use GPUs** for non-trivial datasets.
- **Use skip connections / batch norm** for very deep networks.
- **Track the spatial dimensions** through the layers with the conv formula.

## Chapter Summary

- **CNNs** are purpose-built for **images**: they use small learnable **filters** that
  **convolve** across the image to produce **feature maps**, with **parameter sharing** and
  **local connectivity** giving efficiency and spatial awareness that MLPs lack.
- Key components: **convolution** (+ stride, padding — output size `(W−K+2P)/S+1`), **ReLU**,
  and **pooling** (max pooling downsamples and adds shift-robustness), then **flatten +
  fully-connected + softmax**.
- Stacking conv/pool blocks learns a **feature hierarchy** (edges → parts → objects); famous
  models include **LeNet, AlexNet, VGG, ResNet** (skip connections for great depth).
- We built a PyTorch CNN reaching **0.919** on digits with only **1,898 parameters** —
  showcasing CNN **parameter efficiency**.
- For real images, use **transfer learning** and **data augmentation** rather than training
  from scratch.

---

::: {.qband}
Practice Zone — Chapter 34
:::

## Multiple-Choice Questions (MCQs)

**Q1.** CNNs are primarily designed for:
a) Tabular data  b) Images / grid-like data  c) Time series only  d) Text only

**Q2.** A convolution filter (kernel):
a) Is a single number  b) Slides over the image computing weighted sums  c) Replaces the
loss  d) Shuffles pixels

**Q3.** "Parameter sharing" in CNNs means:
a) Layers share a loss  b) The same filter is used across the whole image  c) All weights
are equal  d) No weights are learned

**Q4.** Max pooling does what?
a) Adds neurons  b) Downsamples by keeping the max in each window  c) Increases resolution
d) Normalises inputs

**Q5.** Which famous CNN won ImageNet in 2012?
a) LeNet  b) AlexNet  c) ResNet  d) VGG

**Q6.** ResNet's key innovation was:
a) Pooling  b) Skip (residual) connections enabling very deep nets  c) Softmax  d) Dropout

**Q7.** Why are CNNs more parameter-efficient than MLPs for images?
a) They use fewer layers  b) Filter sharing reuses weights across the image  c) They skip
training  d) They ignore pixels

**Q8.** For real-world image tasks with limited data, you should usually:
a) Train from scratch  b) Use transfer learning (pretrained model)  c) Use an MLP  d) Use
KNN

### MCQ Answers
**1:** b. **2:** b. **3:** b. **4:** b. **5:** b. **6:** b. **7:** b. **8:** b.

## Interview Questions (with answers)

**Q1. Why are CNNs better than MLPs for images?**
*Answer:* MLPs flatten images, exploding the parameter count and discarding spatial
structure. CNNs use small filters with local connectivity and parameter sharing, so they
have far fewer parameters and exploit the 2-D spatial relationships (nearby pixels), making
them efficient and effective on images.

**Q2. Explain the convolution operation and feature maps.**
*Answer:* A filter (small weight grid) slides over the image; at each position it computes a
dot product with the underlying patch, producing one output value. Sliding it everywhere
yields a feature map highlighting where the filter's pattern (e.g. an edge) occurs. Each
conv layer learns many filters, producing many feature maps.

**Q3. What is the role of pooling?**
*Answer:* Pooling downsamples feature maps (e.g. max pooling keeps the max in each 2×2
window), reducing spatial size and computation while retaining the strongest activations and
adding robustness to small translations of the input.

**Q4. What does stride and padding control?**
*Answer:* Stride is how far the filter moves each step — larger stride gives a smaller
output. Padding adds a border (often zeros) so the output can keep the same size and edge
information isn't lost. The output size is (W − K + 2P)/S + 1.

**Q5. What is transfer learning and why use it for images?**
*Answer:* Transfer learning takes a model pretrained on a large dataset (e.g. ResNet on
ImageNet), which already learned generic visual features, and fine-tunes its later layers on
your specific task. It achieves strong results with far less data and compute than training
from scratch.

## Scenario-Based Questions (with answers)

**Q1.** *You try to classify 224×224 colour images with a fully-connected MLP and it has
hundreds of millions of parameters and overfits. What architecture should you use and why?*
*Answer:* A CNN. Its filters with parameter sharing drastically reduce parameters while
capturing spatial patterns, making it feasible and far less prone to overfitting on images —
the reason CNNs replaced MLPs for vision.

**Q2.** *You have only 2,000 labelled medical images. Training a CNN from scratch overfits
badly. What do you do?*
*Answer:* Use transfer learning: take a CNN pretrained on ImageNet, freeze early layers, and
fine-tune the last layers on your 2,000 images; also apply data augmentation. This leverages
generic features and needs far less data.

**Q3.** *Your CNN throws a shape mismatch error at the first Linear layer. What's the likely
cause and fix?*
*Answer:* The flattened size doesn't match the Linear layer's input. Recompute the spatial
dimensions through the conv/pool layers (using (W−K+2P)/S+1 and pooling), multiply by the
channel count, and set the Linear layer's in_features accordingly (or print the shape before
it).

## Logic-Based Questions (with answers)

**Q1.** An 8×8 input through a 3×3 conv with padding 1 and stride 1 gives what output size?
*Answer:* (8 − 3 + 2·1)/1 + 1 = 8. Padding 1 with a 3×3 kernel and stride 1 preserves the
8×8 size.

**Q2.** Why does parameter sharing make a learned "edge detector" useful across the whole
image?
*Answer:* Because the same filter weights are applied at every position, a pattern detector
learned for one location automatically detects that pattern anywhere in the image —
providing translation invariance and efficiency.

**Q3.** Why did the CNN achieve good accuracy with so few parameters compared to an MLP?
*Answer:* Filter sharing means a few filter weights are reused across all spatial positions
(instead of a separate weight per pixel-neuron connection), and pooling shrinks the data, so
the CNN captures the needed patterns with far fewer parameters.

## Practical Questions (with answers)

**Q1.** What tensor shape does a PyTorch Conv2d expect for a batch of grayscale 28×28 images?
*Answer:* `(batch_size, 1, 28, 28)` — (batch, channels, height, width).

**Q2.** Write the first layer of a CNN: 16 filters of size 3×3 on a 3-channel (RGB) input,
same-size output.
*Answer:* `nn.Conv2d(3, 16, kernel_size=3, padding=1)`.

**Q3.** Why use `CrossEntropyLoss` (not BCELoss) for the 10-digit classifier?
*Answer:* It's multiclass (10 classes); `CrossEntropyLoss` handles multiclass and applies
softmax internally to the logits, expecting integer class labels.

## Long Questions (with answers)

**Q1. Explain the architecture and operation of a CNN: convolution, pooling, and the overall
pipeline, and why CNNs suit images.**

*Answer:* A CNN processes images through a stack of specialised layers. **Convolution
layers** apply small learnable **filters** that slide across the input; at each position a
filter computes a dot product with the underlying patch, and sweeping it across the image
produces a **feature map** that highlights where the filter's pattern occurs. Each layer
learns many filters, and two properties make this powerful for images: **local connectivity**
(each output depends only on a small region, capturing local patterns) and **parameter
sharing** (the same filter is reused everywhere, slashing parameters and giving translation
invariance). A non-linear **ReLU** follows. **Pooling layers** (typically max pooling)
downsample the feature maps, reducing size and computation while keeping the strongest
activations and adding robustness to small shifts. Stacking `[Conv → ReLU → Pool]` blocks
builds a **feature hierarchy** — early layers detect edges, deeper layers detect shapes and
then whole objects. Finally the feature maps are **flattened** and passed to **fully-
connected layers** with a **softmax** output for classification. CNNs suit images because,
unlike MLPs, they preserve and exploit 2-D spatial structure and use far fewer parameters,
making large-image learning feasible — which is why they dominate computer vision.

**Q2. Discuss the evolution of CNN architectures (LeNet to ResNet) and the role of transfer
learning in modern practice.**

*Answer:* CNNs evolved from small proofs of concept to very deep, powerful models. **LeNet-5**
(1998) was the first successful CNN, recognising handwritten digits and establishing the
conv-pool-FC template. Progress stalled until **AlexNet** (2012), a larger, deeper CNN
trained on GPUs with ReLU and dropout, **won ImageNet by a wide margin and ignited the
deep-learning revolution** (Chapter 3). **VGG** (2014) showed that **depth** matters,
stacking many small 3×3 convolutions into 16–19 layers for stronger features. As networks got
deeper, **vanishing gradients** (Chapter 33) made them hard to train, which **ResNet** (2015)
solved with **residual/skip connections** that let gradients bypass layers, enabling networks
of 100+ layers and superhuman ImageNet accuracy. In **modern practice**, few people train such
networks from scratch; instead they use **transfer learning** — taking a model pretrained on a
massive dataset (e.g. ResNet on ImageNet), which has already learned generic visual features
(edges, textures, shapes), and **fine-tuning** its later layers on a specific task with far
less data and compute. Combined with **data augmentation**, transfer learning makes
state-of-the-art image models accessible even with small datasets, and is the default approach
for real-world computer vision (Chapter 40).

## Exercises

1. Explain in your own words why an MLP is impractical for large images.
2. Compute the output size of a 32×32 input through a 5×5 conv with stride 1, padding 0.
3. Describe what max pooling does and why it helps.
4. Order these by depth in feature hierarchy: object detector, edge detector, shape detector.
5. What is transfer learning and when would you use it?

## Mini-Project

**Project: Train a CNN image classifier.**

1. Use a small image dataset (digits 8×8, or MNIST/Fashion-MNIST if you can download).
2. Build a CNN in PyTorch (conv → ReLU → pool blocks → flatten → FC → softmax); train it.
3. Report test accuracy and the total parameter count; compare to an MLP on the same data.
4. Visualise a few learned first-layer filters or feature maps (Chapter 14).
5. (Stretch) Apply transfer learning with a pretrained model on a larger image set. Save in
   `my-ml-journey/`.

## Assignments

1. **Coding:** Modify the chapter's CNN — add a third conv layer, or change filter counts —
   and report the effect on accuracy and parameter count.
2. **Coding:** Add data augmentation (random flips/rotations) to a small image dataset and
   show it reduces overfitting.
3. **Conceptual:** Write one page explaining convolution, parameter sharing, and pooling, and
   why they make CNNs ideal for images.

::: tip
CNNs conquer *spatial* data (images). But what about *sequential* data — text, speech, time
series — where order matters? Chapter 35, **Recurrent Networks, LSTM & GRU**, introduces
networks with memory, built for sequences.
:::
