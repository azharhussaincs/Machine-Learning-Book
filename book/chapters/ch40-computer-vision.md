# Computer Vision

## Introduction

**Computer Vision (CV)** is the field of teaching machines to **see and understand
images and video**. It powers face unlock on your phone, medical-scan diagnosis,
self-driving car perception, quality inspection in factories, satellite analysis, and
photo apps. CV was the domain where deep learning first proved its dominance (AlexNet,
2012, Chapter 3), and it builds directly on the **CNNs** of Chapter 34.

::: keyidea
To a computer, an **image is just a grid of numbers** (pixel values). Computer Vision is
the art of extracting meaning from those numbers — *what* is in the image, *where* it is,
and *what each pixel belongs to*. **CNNs** (and now Vision Transformers) do this, and
**transfer learning** makes state-of-the-art vision accessible to everyone.
:::

By the end of this chapter you will be able to:

- Understand how images are represented and processed.
- Distinguish the main CV tasks: **classification, detection, segmentation**.
- Apply **transfer learning** — the key practical technique for real vision tasks.
- Use **data augmentation** and know the main tools (OpenCV, torchvision).

## Images are just numbers

An image is a grid of **pixels**. A grayscale image is a 2-D array (one intensity per
pixel); a colour image is a 3-D array (height × width × 3 for Red, Green, Blue channels).
Let's prove it and apply a classic **edge-detection** filter (a convolution, Chapter 34).

```python
import numpy as np
from sklearn.datasets import load_digits
from scipy.signal import convolve2d

d = load_digits()
img = d.images[0]                      # an 8x8 grayscale image of the digit 0
print("image shape:", img.shape, "| label:", d.target[0])
print("pixel value range:", int(img.min()), "to", int(img.max()))

# A Sobel filter detects vertical edges — exactly the kind of filter a CNN learns
sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
edges = convolve2d(img, sobel, mode="same")
print("edge-map shape:", edges.shape, "| max edge response:", round(float(np.abs(edges).max()), 1))
print("top-left 3x3 pixels:\n", img[:3, :3].astype(int))
```

**Output:**
```text
image shape: (8, 8) | label: 0
pixel value range: 0 to 15
edge-map shape: (8, 8)
max edge response: 55.0
top-left 3x3 pixels:
 [[ 0  0  5]
 [ 0  0 13]
 [ 0  3 15]]
```

### Explanation

- The image is literally an **8×8 array of numbers** (0 = black, 15 = bright here). The
  top-left corner shows the dark background (0s) giving way to the bright digit stroke (13,
  15).
- Applying the **Sobel filter** via convolution produced an **edge map** highlighting where
  intensity changes sharply. This is *exactly* what a CNN's first-layer filters learn to do
  automatically (Chapter 34) — the connection between "image processing" and "deep
  learning" made concrete.

## The main computer-vision tasks

![The three core CV tasks. Classification: what is in the image (one label). Object detection: what and where (boxes + labels). Segmentation: which pixels belong to each object (pixel-level mask).](assets/images/ch40_cv_tasks.png)

- **Image classification** — assign one label to the whole image ("cat"). (Chapters 32, 34.)
- **Object detection** — find *and locate* multiple objects with bounding **boxes**
  ("cat at (x,y,w,h), dog at …"). Models: **YOLO**, **Faster R-CNN**, **SSD**.
- **Semantic / instance segmentation** — label *every pixel* ("these pixels are road,
  those are car"). Models: **U-Net**, **Mask R-CNN**. Used in medical imaging and
  self-driving.
- Others: **face recognition**, **pose estimation**, **OCR** (text in images), **image
  generation** (Chapter 36/43).

## Transfer learning: the key to practical CV

Training a deep CNN from scratch needs millions of images and huge compute. The practical
secret of modern CV is **transfer learning**: take a model **pretrained** on a giant
dataset (e.g. ResNet on ImageNet's 14M images), which already learned generic visual
features (edges, textures, shapes), and **adapt** it to your task.

![Transfer learning: a model pretrained on millions of images already knows generic features. You freeze its early layers and fine-tune the last layers on your (much smaller) dataset — achieving strong results with little data and compute.](assets/images/ch40_transfer.png)

Two common modes:

- **Feature extraction** — freeze the pretrained layers, replace and train only the final
  classifier on your data.
- **Fine-tuning** — also unfreeze and slightly retrain some later layers for your task.

```python
# Sketch (needs: pip install torchvision)
# from torchvision import models
# model = models.resnet18(weights="IMAGENET1K_V1")   # pretrained
# for p in model.parameters(): p.requires_grad = False   # freeze
# model.fc = nn.Linear(model.fc.in_features, num_classes) # new head for YOUR classes
# ... train only the new head on your (small) dataset ...
```

::: keyidea
Transfer learning is *why* a small team with a few thousand images can build a great
medical-image classifier. You stand on the shoulders of a model that already learned to
see, and just teach it your specific categories. **For almost any real vision task, start
with a pretrained model — don't train from scratch.**
:::

## Data augmentation

Vision models overfit small datasets. **Data augmentation** creates more training variety
by applying label-preserving transformations: random **flips, rotations, crops, zooms,
brightness/contrast changes, and noise**. A cat is still a cat when flipped — so each
transformed image is a free new training example, improving generalisation.

## Tools of the trade

- **OpenCV** (`pip install opencv-python`) — classic image processing: reading/resizing
  images, filters, edge detection, face detection (Haar cascades), video.
- **torchvision / TensorFlow-Keras** — datasets, pretrained models, augmentation, and CNN
  building blocks.
- **Pillow (PIL)** — basic image loading and manipulation.

## Advantages, disadvantages, and use cases

| Advantages | Disadvantages |
|---|---|
| State-of-the-art on visual tasks | Needs lots of data/compute (eased by transfer learning) |
| Transfer learning needs little data | Sensitive to lighting, angle, occlusion |
| Automatic feature learning (CNNs) | Black-box; hard to interpret |
| Broad applications | Vulnerable to adversarial examples; privacy concerns |

**Use cases:** medical imaging, autonomous vehicles, face recognition, manufacturing
quality control, retail (cashier-less stores), agriculture (crop/disease detection),
security/surveillance, document OCR, and AR/VR.

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Training from scratch when transfer learning would work.** For most real
tasks with limited data, fine-tuning a pretrained model is faster and far more accurate.
:::

- **Mistake 2 — No data augmentation**, leading to overfitting on small image sets.
- **Mistake 3 — Forgetting to preprocess** images to the pretrained model's expected size
  and normalisation.
- **Mistake 4 — Ignoring class imbalance** (e.g. rare disease images).
- **Mistake 5 — Using image classification when you need detection/segmentation** (wrong
  task framing).
- **Mistake 6 — Assuming robustness** — models can fail on different lighting, angles, or
  adversarial inputs.

## Best practices

- **Use transfer learning** (pretrained CNN/ViT) for real tasks.
- **Apply data augmentation** to reduce overfitting.
- **Match preprocessing** (size, normalisation) to the pretrained model.
- **Pick the right task** (classification vs detection vs segmentation).
- **Use GPUs**; evaluate with appropriate metrics (e.g. mAP for detection).
- **Test robustness** across realistic conditions.

## Chapter Summary

- **Computer Vision** extracts meaning from images, which are simply **grids of pixel
  numbers**; convolution filters (e.g. Sobel) detect features like edges — exactly what CNN
  layers learn.
- Core tasks: **classification** (whole-image label), **object detection** (boxes + labels;
  YOLO, R-CNN), and **segmentation** (per-pixel labels; U-Net, Mask R-CNN), plus face
  recognition, OCR, and more.
- **Transfer learning** — fine-tuning a model pretrained on millions of images — is the key
  practical technique, giving strong results with little data and compute. **Data
  augmentation** further fights overfitting.
- Tools: **OpenCV** (image processing), **torchvision/Keras** (pretrained models,
  augmentation). For real tasks, **start from a pretrained model**, not from scratch.

---

::: {.qband}
Practice Zone — Chapter 40
:::

## Multiple-Choice Questions (MCQs)

**Q1.** To a computer, an image is:
a) A sound wave  b) A grid of pixel numbers  c) A text string  d) A single number

**Q2.** Assigning one label to a whole image is:
a) Detection  b) Segmentation  c) Classification  d) Augmentation

**Q3.** Finding objects with bounding boxes is:
a) Classification  b) Object detection  c) Clustering  d) OCR

**Q4.** Labelling every pixel by what it belongs to is:
a) Classification  b) Detection  c) Segmentation  d) Augmentation

**Q5.** Transfer learning means:
a) Training from scratch  b) Adapting a pretrained model to your task  c) Moving data
d) Removing layers

**Q6.** Data augmentation helps by:
a) Reducing the dataset  b) Creating label-preserving variations to reduce overfitting
c) Removing labels  d) Scaling features

**Q7.** A Sobel filter is used to detect:
a) Colours  b) Edges  c) Faces only  d) Text

**Q8.** For a real task with a few thousand images, you should usually:
a) Train a deep CNN from scratch  b) Use transfer learning  c) Use KNN on raw pixels  d)
Avoid CNNs

### MCQ Answers
**1:** b. **2:** c. **3:** b. **4:** c. **5:** b. **6:** b. **7:** b. **8:** b.

## Interview Questions (with answers)

**Q1. How are images represented for computer vision?**
*Answer:* As arrays of pixel values: a grayscale image is a 2-D array (height × width), and
a colour image is 3-D (height × width × 3 RGB channels). Models process these numbers;
preprocessing typically scales/normalises pixels and resizes images to a fixed size.

**Q2. What are the main computer-vision tasks?**
*Answer:* Image classification (one label per image), object detection (locate multiple
objects with bounding boxes), and segmentation (label every pixel — semantic or instance),
plus tasks like face recognition, pose estimation, OCR, and image generation.

**Q3. What is transfer learning and why is it important in CV?**
*Answer:* Transfer learning reuses a model pretrained on a large dataset (e.g. ResNet on
ImageNet), which already learned generic visual features, and adapts it (feature extraction
or fine-tuning) to a new task. It's important because it achieves strong results with far
less data and compute than training from scratch — essential for most real projects.

**Q4. Why use data augmentation?**
*Answer:* To expand and diversify the training set with label-preserving transforms (flips,
rotations, crops, brightness changes), which reduces overfitting and improves
generalisation, especially when labelled images are limited.

**Q5. What's the difference between object detection and segmentation?**
*Answer:* Detection locates objects with bounding boxes and class labels (what and roughly
where). Segmentation classifies every pixel (which pixels belong to which object/class) —
giving precise, pixel-level outlines, useful in medical imaging and autonomous driving.

## Scenario-Based Questions (with answers)

**Q1.** *You have 3,000 labelled X-ray images and need a classifier. Training a CNN from
scratch overfits. What do you do?*
*Answer:* Use transfer learning: take a CNN pretrained on ImageNet, freeze early layers,
replace and train the final classifier head on your X-rays (optionally fine-tune later
layers), and apply data augmentation. This leverages generic features and needs far less
data.

**Q2.** *A self-driving system must know exactly which pixels are road vs pedestrian. Which
CV task and model type?*
*Answer:* Semantic/instance segmentation (e.g. U-Net or Mask R-CNN), which labels every
pixel — necessary for precise drivable-area and obstacle boundaries that bounding boxes
can't provide.

**Q3.** *Your classifier works in the lab but fails on phone photos with different lighting
and angles. What's the issue and fix?*
*Answer:* The model isn't robust to real-world variation it didn't see in training. Fix with
data augmentation covering those conditions, more diverse training data, proper
normalisation, and testing across realistic scenarios.

## Logic-Based Questions (with answers)

**Q1.** Why does a convolution filter like Sobel detect edges?
*Answer:* It computes differences between neighbouring pixels; where intensity changes
sharply (an edge), the weighted difference is large, producing a high response, while flat
regions produce near-zero response.

**Q2.** Why does transfer learning need less data than training from scratch?
*Answer:* The pretrained model already learned generic, reusable features (edges, textures,
shapes) from millions of images; only the task-specific final layers must be learned, which
requires far fewer examples than learning all features from scratch.

**Q3.** Why is a flipped image of a cat still a valid "cat" training example?
*Answer:* The label (cat) is invariant to horizontal flipping — it's still a cat — so the
transformed image is a legitimate new example that teaches the model invariance to
orientation, improving generalisation.

## Practical Questions (with answers)

**Q1.** What shape is a colour image array, and what do the dimensions mean?
*Answer:* `(height, width, 3)` — rows of pixels, columns of pixels, and 3 colour channels
(Red, Green, Blue). (Frameworks may use channels-first: `(3, height, width)`.)

**Q2.** Name two data-augmentation transformations.
*Answer:* Random horizontal flip and random rotation (others: crop, zoom, brightness/
contrast jitter, added noise).

**Q3.** Which library would you use to load a pretrained ResNet in PyTorch?
*Answer:* `torchvision.models` (e.g. `torchvision.models.resnet18(weights=...)`).

## Long Questions (with answers)

**Q1. Explain how computer vision works, from image representation to the main tasks, and
the role of CNNs.**

*Answer:* Computer Vision extracts meaning from images, which are represented as **arrays of
pixel numbers** — 2-D for grayscale (height × width) and 3-D for colour (height × width × 3
RGB channels). Early systems used hand-crafted filters (like the **Sobel** edge detector,
which responds where pixel intensities change sharply), but modern CV uses **CNNs**
(Chapter 34) that *learn* such filters automatically: convolution layers detect local
patterns with parameter sharing, pooling adds shift-robustness, and stacked layers build a
hierarchy from edges to shapes to objects. On top of this, CV addresses several **tasks**:
**classification** assigns one label to the whole image; **object detection** locates
multiple objects with bounding boxes and labels (YOLO, Faster R-CNN); **segmentation** labels
every pixel (U-Net, Mask R-CNN) for precise outlines; and there are specialised tasks like
face recognition, pose estimation, and OCR. The pipeline is: preprocess images (resize,
normalise) → extract features with a CNN → produce the task output. Because CNNs learn
features directly from pixels, they removed the need for manual feature engineering and made
CV the first great success of deep learning.

**Q2. Explain transfer learning and data augmentation, and why they are essential for
practical computer vision.**

*Answer:* Training a deep vision model from scratch requires millions of labelled images and
massive compute — out of reach for most projects — so two techniques make CV practical.
**Transfer learning** reuses a model **pretrained** on a huge dataset (e.g. ResNet on
ImageNet's 14M images), which has already learned **generic visual features** (edges,
textures, shapes) in its early layers. You then adapt it to your task by either **feature
extraction** (freeze the pretrained layers and train only a new classifier head on your
data) or **fine-tuning** (also retrain some later layers); either way, only task-specific
parameters must be learned, so strong results are achievable with just thousands — sometimes
hundreds — of images. **Data augmentation** complements this by synthetically expanding the
training set with **label-preserving transformations** (flips, rotations, crops, zooms,
brightness/contrast changes, noise); since these don't change the label (a flipped cat is
still a cat), each transform is a free new example that teaches invariance and **reduces
overfitting**, which is critical on small datasets. Together they let small teams build
accurate, robust vision models — which is why, for almost any real-world vision task, the
standard approach is to start from a pretrained model and train with augmentation rather
than building and training a network from scratch.

## Exercises

1. Describe the array shape of a grayscale vs a colour image.
2. Explain the difference between classification, detection, and segmentation with an
   example each.
3. In your own words, why does transfer learning need less data?
4. List four data-augmentation transformations and why each preserves the label.
5. Why might a vision model fail on real-world photos despite high lab accuracy?

## Mini-Project

**Project: Transfer learning image classifier.**

1. Pick a small image dataset (e.g. cats vs dogs subset, or Fashion-MNIST).
2. (`pip install torchvision`) Load a pretrained CNN (e.g. ResNet18), freeze its layers, and
   replace the final layer for your classes.
3. Apply data augmentation; train only the new head; report test accuracy.
4. Compare to training a small CNN from scratch on the same data.
5. Write a short report on the accuracy/data/compute trade-offs. Save in `my-ml-journey/`.

## Assignments

1. **Coding:** Load an image, display it as an array, and apply edge-detection and blur
   filters via convolution; visualise the results.
2. **Coding (stretch):** Fine-tune a pretrained model on a small dataset with and without
   data augmentation; compare overfitting.
3. **Conceptual:** Write one page explaining why transfer learning and augmentation are the
   pillars of practical computer vision.

::: tip
Vision lets machines see; NLP lets them read. Chapter 41, **Recommendation Systems**, applies
ML to a different everyday problem — predicting what *you'll* like — the technology behind
Netflix, YouTube, Amazon, and Spotify suggestions.
:::
