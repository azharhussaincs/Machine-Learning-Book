# Edge AI

## Introduction

When your phone unlocks with your face *instantly* and *offline*, when a smart camera
detects motion without sending video to the cloud, when a car's autopilot reacts in
milliseconds — that's **Edge AI**: running ML models **directly on the device** (the
"edge") rather than on a remote cloud server.

The cloud (Chapter 46) is powerful but lives far away. For many applications, sending data
to the cloud and waiting for a response is too **slow**, too **privacy-invasive**, too
**bandwidth-hungry**, or simply impossible **offline**. Edge AI brings the model to where
the data is.

::: keyidea
**Edge AI runs models on local devices** — phones, cameras, sensors, cars, microcontrollers.
The trade-off: edge devices have **limited compute, memory, and power**, so models must be
**shrunk** (via quantization, pruning, distillation) to fit — accepting a small accuracy cost
for big gains in **latency, privacy, offline capability, and bandwidth**.
:::

By the end of this chapter you will be able to:

- Explain **why** and **when** to run AI on the edge vs the cloud.
- Understand the **constraints** of edge devices.
- Apply model-shrinking techniques: **quantization, pruning, knowledge distillation**.
- Know the main **edge tools** (TensorFlow Lite, ONNX, Core ML).

## Why run AI on the edge?

- **Latency** — no round-trip to the cloud; responses in milliseconds (vital for cars,
  AR/VR, robotics).
- **Privacy** — data (your face, your voice, your health) stays on the device; nothing is
  sent away.
- **Offline** — works without an internet connection (remote areas, planes, field devices).
- **Bandwidth & cost** — no need to stream large data (e.g. video) to the cloud
  continuously.
- **Reliability** — no dependence on network/cloud availability.

![Cloud AI vs Edge AI. Cloud: the device sends data to a powerful remote server and waits for a response (flexible, but adds latency and privacy/connectivity concerns). Edge: the model runs on the device itself (fast, private, offline — but constrained by limited hardware).](assets/images/ch47_edge_vs_cloud.png)

## The challenge: limited resources

Edge devices are tiny compared to cloud servers: limited **CPU/GPU**, little **memory**,
constrained **battery/power**, and no room for huge models. A 500 MB deep-learning model
won't fit on a microcontroller with kilobytes of RAM. So the central task of Edge AI is
**making models small and efficient enough** to run on-device — without losing too much
accuracy.

## Techniques to shrink models

![Three model-compression techniques. Quantization: use fewer bits per weight (e.g. 32-bit → 8-bit). Pruning: remove unimportant weights/neurons. Knowledge distillation: train a small "student" model to mimic a large "teacher".](assets/images/ch47_compression.png)

### Quantization

**Quantization** stores weights with **fewer bits** — e.g. converting 32-bit floats to
8-bit integers. This shrinks the model ~4× and speeds up inference, with usually tiny
accuracy loss.

```python
import numpy as np
np.random.seed(0)
weights = np.random.randn(100000).astype(np.float32)   # a model's weights
f32_bytes = weights.nbytes

scale = np.abs(weights).max() / 127                     # quantize to int8 [-127, 127]
q = np.round(weights / scale).astype(np.int8)
deq = q.astype(np.float32) * scale                      # dequantized (for error check)

print(f"float32 size: {f32_bytes:,} bytes")
print(f"int8 size:    {q.nbytes:,} bytes")
print(f"compression:  {f32_bytes / q.nbytes:.0f}x smaller")
print(f"mean abs error: {np.mean(np.abs(weights - deq)):.5f} (tiny)")
```

**Output:**
```text
float32 size: 400,000 bytes
int8 size:    100,000 bytes
compression:  4x smaller
mean abs error: 0.00955 (tiny)
```

The model became **4× smaller** (400 KB → 100 KB) with a **negligible** average error
(0.0096). That's the magic of quantization: huge size/speed gains for almost no accuracy
loss — making it the workhorse of edge deployment.

### Pruning

**Pruning** removes **unimportant weights or neurons** (those near zero contribute little).
A pruned network is smaller and faster, and can often be retrained to recover any lost
accuracy.

### Knowledge distillation

**Knowledge distillation** trains a small "**student**" model to **mimic** a large, accurate
"**teacher**" model. The student learns from the teacher's outputs and ends up far smaller
while keeping much of the teacher's performance — ideal for the edge.

## Edge AI tools

| Tool | Purpose |
|---|---|
| **TensorFlow Lite (LiteRT)** | Run TF models on mobile/embedded; quantization built in |
| **PyTorch Mobile / ExecuTorch** | Run PyTorch models on devices |
| **ONNX / ONNX Runtime** | Open format to convert/run models across frameworks & hardware |
| **Core ML** | Apple devices (iPhone, etc.) |
| **Edge TPU / NPUs** | Specialised low-power AI chips on devices |

## Cloud vs edge: a spectrum

It's not all-or-nothing. Many systems are **hybrid**: do quick, private inference on the
edge, and offload heavy or occasional work to the cloud.

| Prefer edge when… | Prefer cloud when… |
|---|---|
| Low latency is critical | The model is too large for the device |
| Privacy/data locality matters | You need maximum accuracy/scale |
| Offline operation is needed | Compute needs are heavy/bursty |
| Bandwidth is limited/costly | You want easy updates & central control |

::: tip
**Practical & debugging tips:** (1) **Quantize first** — it's the easiest big win (≈4×
smaller, faster). (2) Use **TensorFlow Lite** or **ONNX Runtime** to convert and optimise
models for devices. (3) **Measure on the target device** — latency/memory differ hugely from
your laptop. (4) Combine techniques (prune + quantize + distill) for maximum shrinkage. (5)
Validate that accuracy after compression is still acceptable. (6) Consider a **hybrid**
edge-cloud design when models don't fit.
:::

## Advantages, disadvantages, and use cases

| Advantages | Disadvantages |
|---|---|
| Low latency (real-time) | Limited compute/memory/power |
| Privacy (data stays local) | Smaller models → some accuracy loss |
| Works offline | Harder to update than cloud |
| Saves bandwidth/cost | Device fragmentation/optimisation effort |

**Use cases:** face/fingerprint unlock, voice assistants' wake-word detection, smart
cameras, wearables/health monitors, autonomous vehicles and drones, industrial IoT sensors,
and any real-time, private, or offline application.

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Deploying a full-size cloud model to a tiny device.** It won't fit or will be
too slow. Compress (quantize/prune/distill) and validate accuracy on the target hardware.
:::

- **Mistake 2 — Not measuring on the real device** (laptop performance ≠ phone/MCU).
- **Mistake 3 — Over-compressing** until accuracy collapses — balance size vs accuracy.
- **Mistake 4 — Ignoring power/battery constraints** for always-on devices.
- **Mistake 5 — Forgetting update/maintenance** is harder at the edge than in the cloud.
- **Mistake 6 — Using edge when the cloud is fine** (or vice versa) — match to requirements.

## Best practices

- **Quantize** (and consider pruning + distillation) to fit the device.
- **Benchmark on the target hardware** for latency, memory, and power.
- **Validate post-compression accuracy** is acceptable.
- **Use edge tools** (TF Lite, ONNX Runtime, Core ML) for conversion/optimisation.
- **Consider hybrid edge-cloud** designs for large models.
- **Plan for device updates** and fragmentation.

## Chapter Summary

- **Edge AI** runs models **on local devices** (phones, cameras, IoT, cars) instead of the
  cloud, for **low latency, privacy, offline operation, and bandwidth savings**.
- Edge devices have **limited compute, memory, and power**, so models must be **shrunk**.
- Key compression techniques: **quantization** (fewer bits per weight — ~4× smaller with tiny
  error, as shown), **pruning** (remove unimportant weights), and **knowledge distillation**
  (small student mimics large teacher).
- Tools include **TensorFlow Lite, ONNX Runtime, Core ML**, and specialised **edge TPUs/NPUs**.
- Edge vs cloud is a **spectrum**; choose by latency, privacy, offline needs, model size, and
  accuracy — often a **hybrid** design is best.

---

::: {.qband}
Practice Zone — Chapter 47
:::

## Multiple-Choice Questions (MCQs)

**Q1.** Edge AI means running models:
a) On remote cloud servers  b) On local devices  c) Without any model  d) Only in browsers

**Q2.** A key reason to use edge AI is:
a) Unlimited compute  b) Low latency / privacy / offline  c) Larger models  d) No model needed

**Q3.** Quantization reduces model size by:
a) Removing layers  b) Using fewer bits per weight  c) Adding neurons  d) Scaling inputs

**Q4.** Converting 32-bit floats to 8-bit ints shrinks the model about:
a) 2×  b) 4×  c) 10×  d) 100×

**Q5.** Removing unimportant weights/neurons is called:
a) Quantization  b) Pruning  c) Distillation  d) Augmentation

**Q6.** Training a small model to mimic a large one is:
a) Pruning  b) Knowledge distillation  c) Quantization  d) Bagging

**Q7.** A common edge deployment tool is:
a) TensorFlow Lite  b) Pandas  c) Matplotlib  d) Flask

**Q8.** The main constraint of edge devices is:
a) Too much memory  b) Limited compute/memory/power  c) No data  d) Too fast

### MCQ Answers
**1:** b. **2:** b. **3:** b. **4:** b. **5:** b. **6:** b. **7:** a. **8:** b.

## Interview Questions (with answers)

**Q1. What is Edge AI and why use it instead of the cloud?**
*Answer:* Edge AI runs ML models directly on local devices rather than remote servers. It's
used for low latency (no network round-trip), privacy (data stays on device), offline
operation, and bandwidth/cost savings — critical for real-time, private, or disconnected
applications like face unlock, smart cameras, and autonomous vehicles.

**Q2. What are the main challenges of edge deployment?**
*Answer:* Limited compute, memory, and power on devices; the need to compress models without
losing too much accuracy; harder updates and device fragmentation; and validating performance
on diverse target hardware. These drive the use of model-shrinking techniques.

**Q3. Explain quantization, pruning, and knowledge distillation.**
*Answer:* Quantization stores weights with fewer bits (e.g. 32-bit→8-bit), shrinking the model
~4× and speeding inference with little accuracy loss. Pruning removes unimportant
weights/neurons to reduce size and computation. Knowledge distillation trains a small
"student" model to mimic a large "teacher", retaining much accuracy in a far smaller model.

**Q4. When would you choose edge over cloud and vice versa?**
*Answer:* Choose edge for low latency, privacy/data-locality, offline use, and limited
bandwidth. Choose cloud when the model is too large for the device, when you need maximum
accuracy/scale, when compute is heavy/bursty, or when easy central updates matter. Many
systems are hybrid.

**Q5. Why is quantization the most popular edge technique?**
*Answer:* Because it gives large, reliable gains — roughly 4× smaller and faster — with
minimal accuracy loss and is well-supported by tools (TF Lite, ONNX), making it the easiest,
highest-impact first step for fitting models on devices.

## Scenario-Based Questions (with answers)

**Q1.** *A smart doorbell must recognise faces instantly and keep video private. Cloud or
edge, and why?*
*Answer:* Edge. On-device inference gives instant response (no cloud round-trip) and keeps the
video/face data local for privacy, and it works even if the internet is down — all key for a
security device.

**Q2.** *Your accurate cloud model is 200 MB and won't run on the target phone. What's your
plan?*
*Answer:* Compress it: quantize (≈4× smaller), prune unimportant weights, and/or distill into
a smaller student model; convert with TensorFlow Lite/ONNX; then benchmark latency/memory and
validate accuracy on the actual phone — using a hybrid edge-cloud design if it still doesn't
fit.

**Q3.** *After aggressive compression, your edge model's accuracy dropped too much. What do you
do?*
*Answer:* Back off the compression (less aggressive quantization/pruning), fine-tune/retrain
after pruning to recover accuracy, try distillation instead, or accept a slightly larger model
— balancing size/latency against acceptable accuracy on the target device.

## Logic-Based Questions (with answers)

**Q1.** Why does quantizing float32 to int8 give ~4× compression?
*Answer:* float32 uses 32 bits per weight while int8 uses 8 bits; 32/8 = 4, so each weight
takes a quarter of the memory, shrinking the model about 4× (with a small quantization error).

**Q2.** Why does running a model on the edge improve privacy?
*Answer:* Because the data (image, voice, health signal) is processed locally and never leaves
the device, so it isn't transmitted to or stored on remote servers — eliminating a major
privacy exposure of cloud inference.

**Q3.** Why is there usually an accuracy/size trade-off at the edge?
*Answer:* Compressing a model (fewer bits, fewer weights, smaller student) discards some
information/capacity, which can slightly reduce accuracy; the goal is to shrink enough to fit
and run fast while keeping accuracy within acceptable bounds.

## Practical Questions (with answers)

**Q1.** How many bits does int8 use vs float32, and what compression does that give?
*Answer:* int8 uses 8 bits, float32 uses 32 bits — a 4× reduction in size.

**Q2.** Name two tools for deploying models to edge devices.
*Answer:* TensorFlow Lite (LiteRT) and ONNX Runtime (also Core ML for Apple devices).

**Q3.** Which compression technique uses a "teacher" and a "student" model?
*Answer:* Knowledge distillation.

## Long Questions (with answers)

**Q1. Explain Edge AI: why it's used, its challenges, and the techniques used to fit models on
devices.**

*Answer:* **Edge AI** runs ML models directly on local devices — phones, cameras, sensors,
cars, microcontrollers — rather than on remote cloud servers. It is used because for many
applications the cloud is unsuitable: it adds **latency** (network round-trips), risks
**privacy** (sending personal data away), requires **connectivity** (no offline use), and
consumes **bandwidth/cost** (streaming large data like video). Running on the edge delivers
millisecond responses, keeps data local, works offline, and saves bandwidth — essential for
face unlock, autonomous vehicles, smart cameras, and wearables. The core **challenge** is that
edge devices have far less **compute, memory, and power** than cloud servers, so large models
won't fit or run fast enough. The solution is **model compression**: **quantization** stores
weights with fewer bits (e.g. 32-bit floats → 8-bit integers), shrinking the model ~4× and
speeding inference with minimal accuracy loss (as demonstrated, 400 KB → 100 KB with negligible
error); **pruning** removes weights/neurons that contribute little, reducing size and
computation (often followed by retraining to recover accuracy); and **knowledge distillation**
trains a small "student" model to mimic a large "teacher", keeping much of its accuracy in a
compact form. Tools like **TensorFlow Lite, ONNX Runtime, and Core ML**, plus specialised
**edge chips (TPUs/NPUs)**, convert and optimise models for devices. The recurring trade-off is
**accuracy vs size/latency**, tuned by how aggressively the model is compressed and validated
on the target hardware.

**Q2. Compare edge and cloud deployment, and explain when a hybrid approach is appropriate.**

*Answer:* **Cloud deployment** (Chapter 46) offers virtually unlimited compute, easy scaling,
the ability to run very large/accurate models, and simple central updates — but it adds
**latency** from network round-trips, raises **privacy** concerns (data leaves the device),
requires **connectivity**, and can consume significant **bandwidth and cost** for high-volume
data. **Edge deployment** runs the model on-device for **low latency, privacy, offline
operation, and bandwidth savings**, but is constrained by **limited compute/memory/power**,
forces **model compression** (with potential accuracy loss), and makes **updates and
maintenance** harder across many heterogeneous devices. The choice depends on requirements:
prefer **edge** when real-time response, data locality/privacy, or offline operation are
critical and the model can be made small enough; prefer **cloud** when the model is too large
for devices, when maximum accuracy or scale is needed, or when compute is heavy/bursty. A
**hybrid** approach is appropriate — and common — when you want the best of both: run fast,
private, lightweight inference on the edge (e.g. wake-word detection, initial filtering) and
**offload** heavy, occasional, or higher-accuracy processing to the cloud (e.g. full speech
understanding). This balances latency, privacy, and cost against capability, and is the
practical pattern for many real products like voice assistants and smart cameras.

## Exercises

1. List four reasons to deploy AI on the edge instead of the cloud.
2. Explain why edge devices need compressed models.
3. Describe quantization, pruning, and distillation in one sentence each.
4. Why does float32→int8 give ~4× compression?
5. Give two real edge-AI applications and why they need the edge.

## Mini-Project

**Project: Compress a model for the edge.**

1. Take a trained model (e.g. an MLP or small CNN from Part VI).
2. Apply quantization (manually as in this chapter, or with TensorFlow Lite's converter);
   measure the size before/after and the accuracy change.
3. (Stretch) Prune small-magnitude weights and re-measure size and accuracy.
4. Discuss the size/latency vs accuracy trade-off you observed.
5. Note which edge tool you'd use to deploy it and why. Save in `my-ml-journey/`.

## Assignments

1. **Coding:** Quantize a NumPy weight array to int8 and back; plot the original vs
   dequantized values and report the error and compression ratio.
2. **Research:** Pick one edge tool (TensorFlow Lite, ONNX Runtime, or Core ML) and write half
   a page on what it does and how it optimises models.
3. **Conceptual:** Write one page comparing edge and cloud deployment with a real hybrid
   example (e.g. a voice assistant).

::: tip
Deployment, MLOps, cloud, and edge cover *how* and *where* models run. But just because we
*can* build and deploy AI doesn't mean we always *should* — or that we're doing it fairly.
Chapter 48, **Responsible AI & Ethics**, closes Part VIII with the crucial question of
building AI that is fair, safe, and trustworthy.
:::
