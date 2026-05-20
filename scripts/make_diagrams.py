#!/usr/bin/env python3
"""
make_diagrams.py
================
Generates every diagram/figure used in the book as a PNG inside
assets/images/.  Each figure has its own function so chapters can be added
incrementally.  Run from the project root:

    python3 scripts/make_diagrams.py

The visual style matches the book theme (indigo / sky / violet on white).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed; render straight to file
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

# ---- shared theme ----------------------------------------------------------
INK      = "#1a1f2b"
PRIMARY  = "#4f46e5"   # indigo
SKY      = "#0ea5e9"   # sky
VIOLET   = "#7c3aed"   # violet
GREEN    = "#10b981"
AMBER    = "#f59e0b"
LIGHT    = "#eef2ff"

OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "images")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path)}")


def eq(latex, name, fontsize=22):
    """Render a LaTeX-style equation to a transparent PNG using matplotlib
    mathtext (no system LaTeX needed). Referenced from markdown as
    <img class="eq" src="assets/images/eq_<name>.png">."""
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.text(0.5, 0.5, f"${latex}$", fontsize=fontsize, ha="center",
             va="center", color=INK)
    path = os.path.join(OUT, f"eq_{name}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.06,
                transparent=True)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path)}")


def _box(ax, x, y, w, h, text, fc, ec=None, tc="white", fs=11, bold=True):
    ec = ec or fc
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=tc, fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)


def _arrow(ax, x1, y1, x2, y2, color=INK):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=18, lw=2, color=color))


# ===========================================================================
# CHAPTER 1 — Introduction to AI
# ===========================================================================
def ai_ml_dl_venn():
    """Nested circles showing AI ⊃ ML ⊃ Deep Learning, with Data Science."""
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    # (cx, cy, r, fill, edge, label) — labels sit just inside the top of each ring
    circles = [
        (5.0, 4.7, 4.5, "#e0e7ff", PRIMARY, "Artificial Intelligence"),
        (4.6, 4.2, 3.2, "#bae6fd", SKY,     "Machine Learning"),
        (4.3, 3.8, 1.9, "#ddd6fe", VIOLET,  "Deep Learning"),
    ]
    for cx, cy, r, fc, ec, label in circles:
        ax.add_patch(Circle((cx, cy), r, fc=fc, ec=ec, lw=2.5, alpha=0.95))
        ax.text(cx, cy + r - 0.42, label, ha="center", va="center",
                color=ec, fontsize=12, fontweight="bold")
    ax.text(4.3, 3.5, "Neural\nNetworks", ha="center", va="center",
            color=VIOLET, fontsize=10)
    ax.set_xlim(0, 10); ax.set_ylim(0, 11); ax.axis("off")
    ax.set_title("AI ⊃ Machine Learning ⊃ Deep Learning",
                 color=INK, fontsize=13, fontweight="bold", pad=14)
    save(fig, "ch01_ai_ml_dl_venn.png")


def traditional_vs_ml():
    """Side-by-side flow: classic programming vs machine learning."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    # --- Traditional programming ---
    ax = axes[0]; ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("Traditional Programming", color=INK, fontsize=13, fontweight="bold")
    _box(ax, 1, 7.5, 3.5, 1.4, "Data", SKY)
    _box(ax, 1, 4.8, 3.5, 1.4, "Rules\n(written by\nhuman)", AMBER, fs=10)
    _box(ax, 5.5, 6.1, 3.3, 1.4, "Computer\nProgram", PRIMARY)
    _box(ax, 5.5, 2.6, 3.3, 1.4, "Answers", GREEN)
    _arrow(ax, 4.5, 8.2, 5.5, 7.2)
    _arrow(ax, 4.5, 5.5, 5.5, 6.6)
    _arrow(ax, 7.15, 6.1, 7.15, 4.0)
    # --- Machine Learning ---
    ax = axes[1]; ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("Machine Learning", color=INK, fontsize=13, fontweight="bold")
    _box(ax, 1, 7.5, 3.5, 1.4, "Data", SKY)
    _box(ax, 1, 4.8, 3.5, 1.4, "Answers\n(examples)", GREEN, fs=10)
    _box(ax, 5.5, 6.1, 3.3, 1.4, "ML Algorithm", PRIMARY)
    _box(ax, 5.5, 2.6, 3.3, 1.4, "Rules\n(the model)", VIOLET, fs=10)
    _arrow(ax, 4.5, 8.2, 5.5, 7.2)
    _arrow(ax, 4.5, 5.5, 5.5, 6.6)
    _arrow(ax, 7.15, 6.1, 7.15, 4.0)
    fig.suptitle("Humans write rules  vs.  Machines learn rules from examples",
                 color=VIOLET, fontsize=12, fontweight="bold", y=1.02)
    save(fig, "ch01_traditional_vs_ml.png")


def types_of_ai():
    """Capability-based types of AI: ANI, AGI, ASI."""
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    data = [
        (0.3, "Artificial Narrow\nIntelligence (ANI)",
         "Good at ONE task\n(e.g. spam filter,\nface unlock)", SKY, "Exists today"),
        (4.2, "Artificial General\nIntelligence (AGI)",
         "Human-level at ANY\ntask a person can do", PRIMARY, "Not yet achieved"),
        (8.1, "Artificial Super\nIntelligence (ASI)",
         "Smarter than humans\nat everything", VIOLET, "Hypothetical"),
    ]
    for x, title, desc, color, tag in data:
        _box(ax, x, 2.2, 3.4, 2.6, "", color)
        ax.text(x + 1.7, 4.2, title, ha="center", va="center", color="white",
                fontsize=11, fontweight="bold")
        ax.text(x + 1.7, 3.0, desc, ha="center", va="center", color="white", fontsize=9)
        ax.text(x + 1.7, 1.6, tag, ha="center", va="center", color=color,
                fontsize=9.5, fontweight="bold")
    for x in (3.7, 7.6):
        _arrow(ax, x, 3.5, x + 0.5, 3.5)
    ax.set_title("Three Types of AI by Capability", color=INK,
                 fontsize=13, fontweight="bold", pad=8)
    save(fig, "ch01_types_of_ai.png")


# ===========================================================================
# CHAPTER 2 — Introduction to Machine Learning
# ===========================================================================
def ml_drivers():
    """Three forces (Data, Compute, Algorithms) feeding modern ML."""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")
    _box(ax, 0.4, 4.6, 3.2, 1.7, "BIG DATA\n(billions of\nexamples)", SKY, fs=10)
    _box(ax, 0.4, 0.5, 3.2, 1.7, "COMPUTING\n(GPUs & cloud,\ncheap & fast)", VIOLET, fs=10)
    _box(ax, 4.6, 2.55, 3.0, 1.7, "ALGORITHMS\n(esp. deep\nlearning)", AMBER, fs=10)
    _box(ax, 8.6, 2.4, 3.0, 2.0, "MODERN\nMACHINE\nLEARNING", PRIMARY, fs=12)
    _arrow(ax, 3.6, 5.2, 8.6, 3.7)
    _arrow(ax, 3.6, 1.4, 8.6, 3.1)
    _arrow(ax, 7.6, 3.4, 8.6, 3.4)
    ax.set_title("The “perfect storm”: three forces behind modern ML",
                 color=INK, fontsize=13, fontweight="bold", pad=8)
    save(fig, "ch02_ml_drivers.png")


def ml_workflow():
    """The 10-step ML lifecycle as a flow with a feedback loop."""
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")
    steps = [
        ("1. Define\nproblem", SKY), ("2. Collect\ndata", SKY),
        ("3. Prepare\ndata", VIOLET), ("4. Split\ndata", VIOLET),
        ("5. Choose\nmodel", PRIMARY), ("6. Train", PRIMARY),
        ("7. Evaluate", AMBER), ("8. Tune &\nimprove", AMBER),
        ("9. Deploy", GREEN), ("10. Monitor", GREEN),
    ]
    # arrange in two rows of five (snake order)
    positions = []
    for i in range(5):
        positions.append((0.4 + i * 2.35, 5.6))
    for i in range(5):
        positions.append((0.4 + (4 - i) * 2.35, 2.6))
    w, h = 2.0, 1.5
    for (label, color), (x, y) in zip(steps, positions):
        _box(ax, x, y, w, h, label, color, fs=9.5)
    # arrows along the snake
    for i in range(9):
        x1, y1 = positions[i]; x2, y2 = positions[i + 1]
        if y1 == y2 and x2 > x1:                    # top row, going right
            _arrow(ax, x1 + w, y1 + h / 2, x2, y2 + h / 2)
        elif y1 == y2 and x2 < x1:                  # bottom row, going left
            _arrow(ax, x1, y1 + h / 2, x2 + w, y2 + h / 2)
        else:                                       # the drop between rows
            _arrow(ax, x1 + w / 2, y1, x2 + w / 2, y2 + h)
    # feedback loop: Evaluate/Tune back to Prepare data
    ax.annotate("", xy=(positions[2][0] + w / 2, positions[2][1]),
                xytext=(positions[7][0] + w / 2, positions[7][1] + h),
                arrowprops=dict(arrowstyle="-|>", color="#ef4444", lw=2,
                                connectionstyle="arc3,rad=-0.3", ls="--"))
    ax.text(6.0, 4.25, "iterate / loop back", color="#ef4444",
            fontsize=9.5, style="italic", ha="center")
    ax.set_title("The end-to-end Machine Learning workflow",
                 color=INK, fontsize=13, fontweight="bold", pad=6)
    save(fig, "ch02_ml_workflow.png")


def ml_types_overview():
    """Three main learning styles with one-line descriptions."""
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    _box(ax, 4.6, 4.1, 2.8, 0.8, "MACHINE LEARNING", INK, fs=11)
    cards = [
        (0.3, "SUPERVISED", "Learns from LABELLED data\n(answers provided).\nClassification & Regression.", SKY),
        (4.3, "UNSUPERVISED", "Finds hidden patterns in\nUNLABELLED data.\nClustering & reduction.", VIOLET),
        (8.3, "REINFORCEMENT", "Agent learns by TRIAL,\nREWARD and error.\nGames, robotics, control.", GREEN),
    ]
    for x, title, desc, color in cards:
        _box(ax, x, 0.4, 3.4, 2.7, "", color)
        ax.text(x + 1.7, 2.55, title, ha="center", color="white",
                fontsize=12, fontweight="bold")
        ax.text(x + 1.7, 1.4, desc, ha="center", color="white", fontsize=9.3)
        _arrow(ax, 6.0, 4.1, x + 1.7, 3.1)
    ax.set_title("The three main types of Machine Learning",
                 color=INK, fontsize=13, fontweight="bold", pad=6)
    save(fig, "ch02_ml_types_overview.png")


def overfitting():
    """Underfit vs good fit vs overfit on the same noisy data."""
    rng = np.random.default_rng(7)
    x = np.linspace(0, 1, 22)
    true = np.sin(2 * np.pi * x)
    y = true + rng.normal(0, 0.28, size=x.shape)
    xs = np.linspace(0, 1, 300)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7))
    titles = ["UNDERFITTING\n(too simple)", "GOOD FIT\n(generalises)",
              "OVERFITTING\n(memorises noise)"]
    degrees = [1, 3, 15]
    colors = [AMBER, GREEN, "#ef4444"]
    for ax, deg, title, color in zip(axes, degrees, titles, colors):
        coeffs = np.polyfit(x, y, deg)
        ax.scatter(x, y, s=22, color=INK, zorder=3, label="data")
        ax.plot(xs, np.polyval(coeffs, xs), color=color, lw=2.4, label="model")
        ax.set_title(title, color=color, fontsize=11, fontweight="bold")
        ax.set_ylim(-2, 2); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#cbd5e1")
    fig.suptitle("Underfitting vs Good Fit vs Overfitting",
                 color=INK, fontsize=13, fontweight="bold", y=1.04)
    save(fig, "ch02_overfitting.png")


# ===========================================================================
# CHAPTER 3 — History of Machine Learning
# ===========================================================================
def perceptron_diagram():
    """A single perceptron: inputs -> weights -> sum -> step -> output."""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")
    # input nodes
    inputs = [(1.2, 5.2, "x₁"), (1.2, 3.4, "x₂"), (1.2, 1.6, "1\n(bias)")]
    sum_xy = (6.2, 3.4)
    for ix, iy, lab in inputs:
        ax.add_patch(Circle((ix, iy), 0.55, fc="#bae6fd", ec=SKY, lw=2))
        ax.text(ix, iy, lab, ha="center", va="center", fontsize=11, fontweight="bold", color=INK)
        _arrow(ax, ix + 0.55, iy, sum_xy[0] - 0.7, sum_xy[1], color="#94a3b8")
    ax.text(3.6, 5.0, "w₁", color=PRIMARY, fontsize=11, fontweight="bold")
    ax.text(3.6, 3.6, "w₂", color=PRIMARY, fontsize=11, fontweight="bold")
    ax.text(3.6, 2.0, "b",  color=PRIMARY, fontsize=11, fontweight="bold")
    # sum node
    ax.add_patch(Circle(sum_xy, 0.72, fc="#ddd6fe", ec=VIOLET, lw=2.2))
    ax.text(*sum_xy, "Σ", ha="center", va="center", fontsize=20, fontweight="bold", color=VIOLET)
    # step activation box
    _box(ax, 7.7, 2.7, 2.1, 1.4, "step\nfunction", AMBER, fs=11)
    _arrow(ax, sum_xy[0] + 0.72, sum_xy[1], 7.7, 3.4)
    # output
    ax.add_patch(Circle((10.9, 3.4), 0.6, fc="#bbf7d0", ec=GREEN, lw=2))
    ax.text(10.9, 3.4, "0 / 1", ha="center", va="center", fontsize=11, fontweight="bold", color=INK)
    _arrow(ax, 9.8, 3.4, 10.3, 3.4)
    ax.text(6.2, 0.6, "output = step(w₁x₁ + w₂x₂ + b)", ha="center",
            fontsize=11, style="italic", color=INK)
    ax.set_title("The Perceptron (1957)", color=INK, fontsize=13, fontweight="bold", pad=6)
    save(fig, "ch03_perceptron.png")


def ai_winters():
    """Stylised AI 'hype vs time' curve with two winters and a rising trend."""
    fig, ax = plt.subplots(figsize=(10, 4.4))
    yrs = np.linspace(1956, 2025, 500)
    # base rising trend + two gaussian 'hype bumps' that crash into winters
    trend = (yrs - 1956) / (2025 - 1956)
    bump1 = 0.9 * np.exp(-((yrs - 1965) ** 2) / (2 * 4 ** 2))
    bump2 = 0.8 * np.exp(-((yrs - 1985) ** 2) / (2 * 4 ** 2))
    boom  = 1.4 * (1 / (1 + np.exp(-(yrs - 2014) / 2.5)))
    hype = 0.3 + 0.7 * trend + bump1 + bump2 + boom
    ax.plot(yrs, hype, color=PRIMARY, lw=2.6)
    ax.fill_between(yrs, hype, 0, color=PRIMARY, alpha=0.08)
    # winter shading
    for (a, b, label) in [(1974, 1980, "1st AI\nWinter"), (1987, 1994, "2nd AI\nWinter")]:
        ax.axvspan(a, b, color="#93c5fd", alpha=0.35)
        ax.text((a + b) / 2, 0.35, label, ha="center", fontsize=9, color="#1e3a8a")
    for x, lab in [(1957, "Perceptron"), (1986, "Backprop"), (2012, "AlexNet"),
                   (2017, "Transformer"), (2022, "ChatGPT")]:
        ax.axvline(x, color="#cbd5e1", ls=":", lw=1)
        ax.text(x, hype.max() * 1.02, lab, rotation=45, fontsize=8.5,
                ha="left", va="bottom", color=INK)
    ax.set_xlabel("Year"); ax.set_yticks([])
    ax.set_ylabel("Excitement / funding")
    ax.set_xlim(1956, 2027); ax.set_ylim(0, hype.max() * 1.35)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.set_title("AI hype cycles: two winters, then the modern boom",
                 color=INK, fontsize=13, fontweight="bold", pad=18)
    save(fig, "ch03_ai_winters.png")


def ml_timeline():
    """Horizontal timeline of major ML milestones."""
    fig, ax = plt.subplots(figsize=(11, 4.2))
    events = [
        (1943, "First artificial\nneuron"), (1950, "Turing\nTest"),
        (1956, "AI named\n(Dartmouth)"), (1957, "Perceptron"),
        (1986, "Back-\npropagation"), (1997, "Deep Blue"),
        (2012, "AlexNet"), (2016, "AlphaGo"),
        (2017, "Transformer"), (2022, "ChatGPT"),
    ]
    ax.axhline(0, color=PRIMARY, lw=3, zorder=1)
    for i, (yr, lab) in enumerate(events):
        up = i % 2 == 0
        y = 1.0 if up else -1.0
        ax.plot([yr, yr], [0, y * 0.6], color="#94a3b8", lw=1.2, zorder=1)
        ax.scatter([yr], [0], s=70, color=VIOLET, zorder=3)
        ax.text(yr, y * 0.95, f"{yr}", ha="center",
                va="bottom" if up else "top", fontsize=9.5, fontweight="bold", color=PRIMARY)
        ax.text(yr, y * 0.62, lab, ha="center",
                va="bottom" if up else "top", fontsize=8.6, color=INK)
    ax.set_xlim(1938, 2028); ax.set_ylim(-1.8, 1.8); ax.axis("off")
    ax.set_title("A timeline of Machine Learning (1943 → today)",
                 color=INK, fontsize=13, fontweight="bold", pad=4)
    save(fig, "ch03_timeline.png")


# ===========================================================================
# CHAPTER 4 — Types of Machine Learning
# ===========================================================================
def ml_taxonomy():
    """Tree of ML types and their sub-types."""
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")
    _box(ax, 4.7, 6.7, 2.6, 1.0, "MACHINE\nLEARNING", INK, fs=12)
    # three main branches
    branches = [(1.4, SKY, "SUPERVISED", ["Classification", "Regression"]),
                (4.7, VIOLET, "UNSUPERVISED", ["Clustering", "Dim. reduction", "Anomaly / Assoc."]),
                (8.4, GREEN, "REINFORCEMENT", ["Agent + reward", "Learns a policy"])]
    for bx, color, title, subs in branches:
        _box(ax, bx, 3.9, 2.4, 1.0, title, color, fs=11)
        _arrow(ax, 6.0, 6.7, bx + 1.2, 4.9, color="#94a3b8")
        for j, s in enumerate(subs):
            _box(ax, bx + 0.15, 2.5 - j * 1.0, 2.1, 0.78, s, "white", ec=color, tc=color, fs=9, bold=False)
            _arrow(ax, bx + 1.2, 3.9, bx + 1.2, 3.28 - j * 1.0, color="#cbd5e1")
    # semi/self note between supervised and unsupervised
    ax.text(3.1, 0.4, "Semi-supervised & Self-supervised sit between Supervised and Unsupervised",
            ha="left", fontsize=9.2, style="italic", color="#64748b")
    ax.set_title("A taxonomy of Machine Learning", color=INK,
                 fontsize=13, fontweight="bold", pad=4)
    save(fig, "ch04_ml_taxonomy.png")


def classification_vs_regression():
    """Left: classification boundary. Right: regression line."""
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    # classification
    ax = axes[0]
    a = rng.normal([2, 2], 0.6, size=(20, 2))
    b = rng.normal([4.2, 4.2], 0.6, size=(20, 2))
    ax.scatter(a[:, 0], a[:, 1], color=SKY, s=28, label="Class A")
    ax.scatter(b[:, 0], b[:, 1], color="#ef4444", s=28, label="Class B")
    xs = np.linspace(0.5, 5.8, 10)
    ax.plot(xs, 6.3 - xs, color=INK, lw=2, ls="--")
    ax.text(3.0, 5.7, "decision boundary", fontsize=9, style="italic")
    ax.set_title("CLASSIFICATION\n(predict a category)", color=SKY, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower left"); ax.set_xticks([]); ax.set_yticks([])
    # regression
    ax = axes[1]
    x = np.linspace(1, 9, 25)
    y = 1.4 * x + 3 + rng.normal(0, 1.6, size=x.shape)
    ax.scatter(x, y, color=VIOLET, s=28, label="data")
    coef = np.polyfit(x, y, 1)
    ax.plot(x, np.polyval(coef, x), color=INK, lw=2.2, label="fitted line")
    ax.set_title("REGRESSION\n(predict a number)", color=VIOLET, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left"); ax.set_xticks([]); ax.set_yticks([])
    save(fig, "ch04_classification_vs_regression.png")


def clustering_demo():
    """Unlabelled points (left) -> discovered clusters (right)."""
    rng = np.random.default_rng(4)
    g1 = rng.normal([2, 2], 0.5, size=(25, 2))
    g2 = rng.normal([6, 6], 0.5, size=(25, 2))
    g3 = rng.normal([2.5, 6.5], 0.5, size=(25, 2))
    allp = np.vstack([g1, g2, g3])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    axes[0].scatter(allp[:, 0], allp[:, 1], color="#94a3b8", s=26)
    axes[0].set_title("INPUT: unlabelled points", color=INK, fontsize=11, fontweight="bold")
    for c, g in zip([SKY, VIOLET, GREEN], [g1, g2, g3]):
        axes[1].scatter(g[:, 0], g[:, 1], color=c, s=26)
    axes[1].set_title("OUTPUT: clusters found", color=INK, fontsize=11, fontweight="bold")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    save(fig, "ch04_clustering.png")


def rl_loop():
    """Agent <-> Environment reinforcement-learning loop."""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")
    _box(ax, 1.0, 2.7, 3.2, 1.6, "AGENT\n(the learner)", PRIMARY, fs=12)
    _box(ax, 7.8, 2.7, 3.2, 1.6, "ENVIRONMENT", GREEN, fs=12)
    # action: agent -> env (top)
    ax.annotate("", xy=(7.8, 4.0), xytext=(4.2, 4.0),
                arrowprops=dict(arrowstyle="-|>", color=PRIMARY, lw=2.2,
                                connectionstyle="arc3,rad=-0.35"))
    ax.text(6.0, 5.7, "ACTION", ha="center", color=PRIMARY, fontsize=11, fontweight="bold")
    # reward + state: env -> agent (bottom)
    ax.annotate("", xy=(4.2, 3.0), xytext=(7.8, 3.0),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.2,
                                connectionstyle="arc3,rad=-0.35"))
    ax.text(6.0, 1.0, "REWARD  +  new STATE", ha="center", color=GREEN, fontsize=11, fontweight="bold")
    ax.set_title("The Reinforcement Learning loop", color=INK,
                 fontsize=13, fontweight="bold", pad=6)
    save(fig, "ch04_rl_loop.png")


# ===========================================================================
# CHAPTER 5 — Mathematics for ML
# ===========================================================================
def vector_diagram():
    """A 2-D vector drawn as an arrow with its components."""
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.annotate("", xy=(3, 2), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=PRIMARY, lw=3))
    ax.plot([3, 3], [0, 2], color="#94a3b8", ls="--", lw=1.5)
    ax.plot([0, 3], [0, 0], color="#94a3b8", ls="--", lw=1.5)
    ax.text(1.5, -0.32, "x-component = 3", ha="center", color="#64748b", fontsize=10)
    ax.text(3.15, 1.0, "y-component = 2", ha="left", color="#64748b", fontsize=10, rotation=90)
    ax.text(1.4, 1.25, "v = (3, 2)", color=PRIMARY, fontsize=13, fontweight="bold", rotation=33)
    ax.text(2.0, 2.35, "length = √(3²+2²) ≈ 3.6", color=VIOLET, fontsize=10)
    ax.scatter([0], [0], color=INK, s=30)
    ax.text(-0.25, -0.25, "origin", color=INK, fontsize=9)
    ax.set_xlim(-0.6, 4.2); ax.set_ylim(-0.6, 3.0)
    ax.set_xticks(range(0, 5)); ax.set_yticks(range(0, 4))
    ax.grid(alpha=0.25); ax.set_aspect("equal")
    ax.set_title("A vector as an arrow", color=INK, fontsize=13, fontweight="bold")
    save(fig, "ch05_vector.png")


def matrix_mult_diagram():
    """Illustrate row x column dot product producing one output cell."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")

    def grid(x0, y0, vals, hl_row=None, hl_col=None, color=SKY):
        rows = len(vals); cols = len(vals[0])
        for r in range(rows):
            for c in range(cols):
                fc = "white"
                if hl_row is not None and r == hl_row: fc = "#dbeafe"
                if hl_col is not None and c == hl_col: fc = "#ede9fe"
                ax.add_patch(plt.Rectangle((x0 + c * 0.8, y0 - r * 0.8), 0.8, 0.8,
                                           fc=fc, ec=color, lw=1.4))
                ax.text(x0 + c * 0.8 + 0.4, y0 - r * 0.8 + 0.4, str(vals[r][c]),
                        ha="center", va="center", fontsize=11, color=INK)

    grid(0.6, 5.0, [[1, 2], [3, 4]], hl_row=0, color=SKY)
    ax.text(1.4, 5.6, "A (2×2)", ha="center", fontsize=10, color=SKY, fontweight="bold")
    ax.text(3.0, 4.2, "×", fontsize=20, color=INK)
    grid(3.7, 5.0, [[5, 6], [7, 8]], hl_col=0, color=VIOLET)
    ax.text(4.5, 5.6, "B (2×2)", ha="center", fontsize=10, color=VIOLET, fontweight="bold")
    ax.text(6.1, 4.2, "=", fontsize=20, color=INK)
    grid(6.8, 5.0, [[19, 22], [43, 50]], hl_row=0, hl_col=0, color=GREEN)
    ax.text(7.6, 5.6, "result", ha="center", fontsize=10, color=GREEN, fontweight="bold")
    ax.text(6.0, 1.4, "top-left cell = (row 1 of A) · (col 1 of B) = 1×5 + 2×7 = 19",
            ha="center", fontsize=11, style="italic", color=INK)
    ax.set_title("Matrix multiplication = dot products of rows and columns",
                 color=INK, fontsize=13, fontweight="bold", pad=2)
    save(fig, "ch05_matrix_mult.png")


def derivative_diagram():
    """Curve with a tangent line showing the slope (derivative)."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    x = np.linspace(-3, 3, 200)
    ax.plot(x, x ** 2, color=PRIMARY, lw=2.4, label="f(x) = x²")
    x0 = 1.5; slope = 2 * x0
    tang = slope * (x - x0) + x0 ** 2
    ax.plot(x, tang, color="#ef4444", lw=2, ls="--", label=f"tangent (slope = {slope:.0f})")
    ax.scatter([x0], [x0 ** 2], color=INK, s=50, zorder=3)
    ax.scatter([0], [0], color=GREEN, s=50, zorder=3)
    ax.text(0.1, 0.4, "slope = 0\n(minimum)", color=GREEN, fontsize=9)
    ax.set_ylim(-1, 9); ax.legend(fontsize=9)
    ax.set_title("The derivative is the slope of the tangent",
                 color=INK, fontsize=12, fontweight="bold")
    ax.grid(alpha=0.2)
    save(fig, "ch05_derivative.png")


def gradient_descent_diagram():
    """Parabola with descending gradient-descent steps."""
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    x = np.linspace(-4, 4, 200)
    f = lambda t: t ** 2
    ax.plot(x, f(x), color=PRIMARY, lw=2.4)
    # simulate GD steps
    pos = -3.6; lr = 0.18
    xs = [pos]
    for _ in range(8):
        pos = pos - lr * (2 * pos)
        xs.append(pos)
    xs = np.array(xs)
    ax.plot(xs, f(xs), "o-", color="#ef4444", lw=1.6, ms=7, zorder=3)
    ax.annotate("start", (xs[0], f(xs[0])), textcoords="offset points",
                xytext=(6, 6), color="#ef4444", fontsize=10)
    ax.annotate("minimum", (0, 0), textcoords="offset points",
                xytext=(8, 12), color=GREEN, fontsize=10, fontweight="bold")
    ax.set_xlabel("parameter w"); ax.set_ylabel("loss L(w)")
    ax.set_title("Gradient descent: stepping downhill to the minimum",
                 color=INK, fontsize=12, fontweight="bold")
    ax.grid(alpha=0.2)
    save(fig, "ch05_gradient_descent.png")


def learning_rate_diagram():
    """Three panels: too small, just right, too large."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7))
    x = np.linspace(-4, 4, 200)
    f = lambda t: t ** 2
    configs = [("Too small", 0.02, AMBER, 9), ("Just right", 0.2, GREEN, 8),
               ("Too large", 0.62, "#ef4444", 8)]
    for ax, (title, lr, color, steps) in zip(axes, configs):
        ax.plot(x, f(x), color=PRIMARY, lw=2)
        pos = -3.5; xs = [pos]
        for _ in range(steps):
            pos = pos - lr * (2 * pos); xs.append(pos)
        xs = np.array(xs)
        ax.plot(xs, f(xs), "o-", color=color, lw=1.5, ms=5)
        ax.set_title(title, color=color, fontsize=11, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(-1, 16)
    fig.suptitle("Effect of the learning rate", color=INK,
                 fontsize=13, fontweight="bold", y=1.03)
    save(fig, "ch05_learning_rate.png")


# ===========================================================================
# CHAPTER 6 — Statistics for ML
# ===========================================================================
def _bell(x, mu, sd):
    return np.exp(-((x - mu) ** 2) / (2 * sd ** 2)) / (sd * np.sqrt(2 * np.pi))


def skewness_diagram():
    """Left-skewed, symmetric, right-skewed distributions."""
    from scipy import stats as _st
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    x = np.linspace(0, 1, 400)
    sets = [("Left-skewed\n(mean < median)", _st.beta(5, 2), AMBER),
            ("Symmetric\n(mean ≈ median)", _st.beta(5, 5), GREEN),
            ("Right-skewed\n(mean > median)", _st.beta(2, 5), VIOLET)]
    for ax, (title, dist, color) in zip(axes, sets):
        y = dist.pdf(x)
        ax.plot(x, y, color=color, lw=2.4)
        ax.fill_between(x, y, color=color, alpha=0.15)
        ax.set_title(title, color=color, fontsize=11, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Skewness: the shape of data", color=INK,
                 fontsize=13, fontweight="bold", y=1.05)
    save(fig, "ch06_skewness.png")


def normal_diagram():
    """Bell curve with 68-95-99.7 shaded regions."""
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    x = np.linspace(-4, 4, 500); y = _bell(x, 0, 1)
    ax.plot(x, y, color=PRIMARY, lw=2.6)
    bands = [(-1, 1, "#4f46e5", "68%"), (-2, -1, "#7c9cf5", "95%"),
             (1, 2, "#7c9cf5", None), (-3, -2, "#c7d2fe", "99.7%"), (2, 3, "#c7d2fe", None)]
    for a, b, c, lab in bands:
        xx = np.linspace(a, b, 100)
        ax.fill_between(xx, _bell(xx, 0, 1), color=c, alpha=0.55)
    for sd in range(-3, 4):
        ax.axvline(sd, color="#cbd5e1", lw=0.8, ls=":")
        ax.text(sd, -0.018, f"{sd}σ", ha="center", fontsize=9, color="#475569")
    ax.text(0, 0.18, "68%", ha="center", fontsize=10, color="white", fontweight="bold")
    ax.text(-1.5, 0.06, "95%", ha="center", fontsize=9, color=INK)
    ax.text(2.5, 0.02, "99.7%", ha="center", fontsize=8.5, color=INK)
    ax.set_ylim(-0.03, 0.45); ax.set_yticks([]); ax.set_xticks([])
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.set_title("The normal distribution and the 68–95–99.7 rule",
                 color=INK, fontsize=12, fontweight="bold")
    save(fig, "ch06_normal.png")


def clt_diagram():
    """Skewed source distribution vs near-normal distribution of sample means."""
    rng = np.random.default_rng(0)
    pop = rng.exponential(1.0, size=100000)        # very skewed
    means = [np.mean(rng.choice(pop, 40)) for _ in range(3000)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].hist(pop, bins=50, color=VIOLET, alpha=0.8)
    axes[0].set_title("Original data\n(skewed, NOT normal)", color=VIOLET,
                      fontsize=11, fontweight="bold")
    axes[1].hist(means, bins=40, color=GREEN, alpha=0.8)
    axes[1].set_title("Means of samples\n(becomes bell-shaped!)", color=GREEN,
                      fontsize=11, fontweight="bold")
    for ax in axes:
        ax.set_yticks([]); ax.set_xticks([])
    fig.suptitle("The Central Limit Theorem", color=INK, fontsize=13,
                 fontweight="bold", y=1.04)
    save(fig, "ch06_clt.png")


def correlation_diagram():
    """Positive, none, negative correlation scatter plots."""
    rng = np.random.default_rng(3)
    n = 80
    x = rng.normal(0, 1, n)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    configs = [("Strong positive\nr ≈ +0.9", x + rng.normal(0, 0.45, n), SKY),
               ("No correlation\nr ≈ 0", rng.normal(0, 1, n), "#94a3b8"),
               ("Strong negative\nr ≈ −0.9", -x + rng.normal(0, 0.45, n), "#ef4444")]
    for ax, (title, y, color) in zip(axes, configs):
        ax.scatter(x, y, color=color, s=22, alpha=0.8)
        ax.set_title(title, color=color, fontsize=11, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Correlation (r ranges from −1 to +1)", color=INK,
                 fontsize=13, fontweight="bold", y=1.04)
    save(fig, "ch06_correlation.png")


# ===========================================================================
# CHAPTER 54 — Future of AI  (defined first; registered below)
# ===========================================================================
def frontiers_diagram():
    """Frontiers of AI around a center."""
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.5, 1.5); ax.axis("off")
    ax.add_patch(Circle((0, 0), 0.55, fc=INK, ec="white", lw=2, zorder=3))
    ax.text(0, 0, "Future\nof AI", ha="center", va="center", color="white", fontsize=11, fontweight="bold", zorder=4)
    frontiers = [("Multimodal\nfoundation models", SKY), ("AI agents\n(take actions)", PRIMARY),
                 ("Reasoning &\nplanning", VIOLET), ("Efficient /\nsmall models", GREEN),
                 ("AI for\nscience", AMBER), ("Alignment\n& safety", "#ef4444")]
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, len(frontiers), endpoint=False)
    for (lab, color), a in zip(frontiers, angles):
        x, y = 1.08 * np.cos(a), 1.08 * np.sin(a)
        ax.add_patch(Circle((x, y), 0.42, fc=color, ec="white", lw=2, zorder=3))
        ax.text(x, y, lab, ha="center", va="center", color="white", fontsize=7.8, fontweight="bold", zorder=4)
        ax.plot([0, x], [0, y], color="#cbd5e1", lw=1.2, zorder=1)
    ax.set_title("Frontiers of AI research", color=INK, fontsize=13, fontweight="bold")
    save(fig, "ch54_frontiers.png")


# ===========================================================================
# CHAPTER 53 — Career & Startups  (defined first; registered below)
# ===========================================================================
def ml_roles_diagram():
    """The spectrum of data/ML roles."""
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    roles = [
        (0.3, 4.0, SKY, "Data Analyst", "SQL, stats,\ndashboards"),
        (4.1, 4.0, GREEN, "Data Scientist", "stats + ML +\nstorytelling"),
        (7.9, 4.0, VIOLET, "ML Engineer", "software eng +\ndeploy/scale"),
        (0.3, 1.3, AMBER, "MLOps Engineer", "pipelines,\nmonitoring"),
        (4.1, 1.3, "#0891b2", "Research Scientist", "deep maths,\nnew methods"),
        (7.9, 1.3, "#ef4444", "Data Eng / AI PM", "pipelines /\nproduct"),
    ]
    for x, y, color, title, desc in roles:
        _box(ax, x, y, 3.5, 1.5, "", color)
        ax.text(x + 1.75, y + 1.05, title, ha="center", color="white", fontsize=10.5, fontweight="bold")
        ax.text(x + 1.75, y + 0.45, desc, ha="center", color="white", fontsize=8.5)
    ax.set_title("The spectrum of ML/data career roles", color=INK, fontsize=13,
                 fontweight="bold", pad=2)
    save(fig, "ch53_roles.png")


# ===========================================================================
# CHAPTER 52 — Freelancing  (defined first; registered below)
# ===========================================================================
def freelance_services_diagram():
    """Spectrum of ML freelance services from accessible to advanced."""
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")
    services = [(0.3, SKY, "Data analysis\n& dashboards"), (2.3, GREEN, "Predictive\nmodelling"),
                (4.3, VIOLET, "NLP / Vision\nservices"), (6.3, AMBER, "LLM / chatbot\nintegration"),
                (8.3, "#0891b2", "Model\ndeployment"), (10.3, "#ef4444", "Consulting\n& training")]
    for x, color, lab in services:
        _box(ax, x, 1.6, 1.85, 1.5, lab, color, fs=8.5)
    ax.annotate("", xy=(11.6, 1.0), xytext=(0.4, 1.0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.8))
    ax.text(1.5, 0.5, "accessible", fontsize=9, color=SKY, fontweight="bold")
    ax.text(10.0, 0.5, "advanced", fontsize=9, color="#ef4444", fontweight="bold")
    ax.set_title("ML freelance services: start where you are, then grow",
                 color=INK, fontsize=13, fontweight="bold", pad=2)
    save(fig, "ch52_services.png")


# ===========================================================================
# CHAPTER 51 — Interview Prep  (defined first; registered below)
# ===========================================================================
def interview_types_diagram():
    """Five components of ML interviews around a center."""
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    items = [(0.4, SKY, "ML Concepts", "algorithms,\nmetrics, theory"),
             (2.7, GREEN, "Coding", "Python, Pandas,\nscikit-learn"),
             (5.0, VIOLET, "Maths & Stats", "gradient descent,\nprobability"),
             (7.3, AMBER, "System Design", "end-to-end\nML systems"),
             (9.6, "#ef4444", "Projects &\nBehavioural", "your work,\ncommunication")]
    for x, color, title, desc in items:
        _box(ax, x, 1.6, 2.1, 2.0, "", color)
        ax.text(x + 1.05, 3.0, title, ha="center", color="white", fontsize=9.5, fontweight="bold")
        ax.text(x + 1.05, 2.1, desc, ha="center", color="white", fontsize=8)
    ax.text(6.0, 0.6, "strong candidates prepare for all five", ha="center",
            fontsize=10, style="italic", color=INK)
    ax.set_title("The five components of ML interviews", color=INK, fontsize=13,
                 fontweight="bold", pad=2)
    save(fig, "ch51_interview_types.png")


# ===========================================================================
# CHAPTER 50 — Industry Case Studies  (defined first; registered below)
# ===========================================================================
def industries_diagram():
    """ML applications mapped across industries."""
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    items = [
        (0.3, 4.2, SKY, "Healthcare", "diagnosis (CNNs),\nrisk scores"),
        (4.2, 4.2, GREEN, "Finance", "fraud, credit,\nforecasting"),
        (8.1, 4.2, VIOLET, "Retail/E-com", "recommenders,\ndemand"),
        (0.3, 1.4, AMBER, "Transport", "self-driving,\nETA, routing"),
        (4.2, 1.4, "#0891b2", "Entertainment", "recommendation,\ngenerative"),
        (8.1, 1.4, "#ef4444", "Manufacturing", "defect detect,\npredictive maint."),
    ]
    for x, y, color, title, desc in items:
        _box(ax, x, y, 3.4, 1.5, "", color)
        ax.text(x + 1.7, y + 1.05, title, ha="center", color="white", fontsize=11, fontweight="bold")
        ax.text(x + 1.7, y + 0.45, desc, ha="center", color="white", fontsize=8.5)
    ax.set_title("Machine Learning across industries", color=INK, fontsize=13,
                 fontweight="bold", pad=2)
    save(fig, "ch50_industries.png")


# ===========================================================================
# CHAPTER 48 — Responsible AI  (defined first; registered below)
# ===========================================================================
def responsible_ai_diagram():
    """Six principles of responsible AI around a center."""
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.5, 1.5); ax.axis("off")
    ax.add_patch(Circle((0, 0), 0.5, fc=INK, ec="white", lw=2, zorder=3))
    ax.text(0, 0, "Responsible\nAI", ha="center", va="center", color="white", fontsize=10, fontweight="bold", zorder=4)
    principles = [("Fairness", SKY), ("Transparency", PRIMARY), ("Privacy", VIOLET),
                  ("Safety", GREEN), ("Accountability", AMBER), ("Human\noversight", "#ef4444")]
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, len(principles), endpoint=False)
    for (lab, color), a in zip(principles, angles):
        x, y = 1.05 * np.cos(a), 1.05 * np.sin(a)
        ax.add_patch(Circle((x, y), 0.38, fc=color, ec="white", lw=2, zorder=3))
        ax.text(x, y, lab, ha="center", va="center", color="white", fontsize=8.5, fontweight="bold", zorder=4)
        ax.plot([0, x], [0, y], color="#cbd5e1", lw=1.2, zorder=1)
    ax.set_title("The principles of Responsible AI", color=INK, fontsize=13, fontweight="bold")
    save(fig, "ch48_principles.png")


# ===========================================================================
# CHAPTER 47 — Edge AI  (defined first; registered below)
# ===========================================================================
def edge_vs_cloud_diagram():
    """Cloud (round-trip to server) vs edge (model on device)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]; ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    ax.set_title("CLOUD AI", color=PRIMARY, fontsize=12, fontweight="bold")
    _box(ax, 0.5, 2.0, 2.2, 1.3, "Device", SKY, fs=9.5)
    _box(ax, 6.8, 2.0, 2.6, 1.3, "Cloud server\n(big model)", PRIMARY, fs=9)
    ax.annotate("", xy=(6.8, 2.9), xytext=(2.7, 2.9), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.8))
    ax.annotate("", xy=(2.7, 2.3), xytext=(6.8, 2.3), arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.8))
    ax.text(4.7, 3.2, "send data →", ha="center", fontsize=8, color=INK)
    ax.text(4.7, 1.6, "← result (latency!)", ha="center", fontsize=8, color=GREEN)
    ax = axes[1]; ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    ax.set_title("EDGE AI", color=GREEN, fontsize=12, fontweight="bold")
    _box(ax, 3.0, 1.6, 4.0, 2.0, "Device\n+ small model\n(runs locally)", GREEN, fs=10)
    ax.text(5.0, 0.7, "instant • private • offline", ha="center", fontsize=9.5, style="italic", color=INK)
    fig.suptitle("Cloud AI vs Edge AI", color=INK, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ch47_edge_vs_cloud.png")


def compression_diagram():
    """Quantization, pruning, distillation."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    # quantization
    ax = axes[0]; ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_title("QUANTIZATION", color=SKY, fontsize=11, fontweight="bold")
    _box(ax, 1, 3.6, 8, 1.0, "32-bit float weights", SKY, fs=9.5)
    _box(ax, 2.5, 1.3, 5, 1.0, "8-bit int (4× smaller)", GREEN, fs=9.5)
    _arrow(ax, 5, 3.6, 5, 2.3, color=INK)
    # pruning
    ax = axes[1]; ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_title("PRUNING", color=VIOLET, fontsize=11, fontweight="bold")
    rng = np.random.default_rng(0)
    for i in range(4):
        for j in range(4):
            kept = rng.random() > 0.4
            c = VIOLET if kept else "white"
            ax.add_patch(Circle((2 + i * 2, 1 + j * 1.2), 0.28, fc=c, ec=VIOLET if kept else "#cbd5e1"))
            if not kept: ax.plot(2 + i * 2, 1 + j * 1.2, "x", color="#ef4444", ms=7)
    ax.text(5, 5.3, "remove unimportant\nweights/neurons", ha="center", fontsize=8.5, color=INK)
    # distillation
    ax = axes[2]; ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_title("DISTILLATION", color=AMBER, fontsize=11, fontweight="bold")
    _box(ax, 1, 3.6, 3.5, 1.4, "TEACHER\n(big, accurate)", "#94a3b8", fs=8.5)
    _box(ax, 5.5, 3.6, 3.5, 1.4, "STUDENT\n(small, mimics)", GREEN, fs=8.5)
    _arrow(ax, 4.5, 4.3, 5.5, 4.3, color=INK)
    ax.text(5, 2.2, "small model learns\nfrom the big one", ha="center", fontsize=8.5, color=INK)
    fig.suptitle("Shrinking models for the edge", color=INK, fontsize=13, fontweight="bold", y=1.04)
    fig.tight_layout()
    save(fig, "ch47_compression.png")


# ===========================================================================
# CHAPTER 46 — Cloud ML  (defined first; registered below)
# ===========================================================================
def cloud_stack_diagram():
    """Pyramid of the cloud ML stack: more control at bottom, less effort at top."""
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    layers = [
        (0.5, 10.5, "Compute & Storage (IaaS) — most control, most work", "#475569"),
        (1.6, 9.4, "Managed Training", SKY),
        (2.7, 8.3, "Managed Deployment (endpoints)", PRIMARY),
        (3.8, 7.2, "AutoML", VIOLET),
        (4.9, 6.1, "Pre-trained AI APIs — least effort", GREEN),
    ]
    for i, (x0, x1, lab, color) in enumerate(layers):
        y = 0.7 + i * 1.0
        ax.add_patch(plt.Rectangle((x0, y), x1 - x0, 0.85, fc=color, ec="white", lw=2))
        ax.text((x0 + x1) / 2, y + 0.42, lab, ha="center", va="center",
                color="white", fontsize=9.5, fontweight="bold")
    ax.annotate("less effort\n& expertise", xy=(6, 5.6), xytext=(8.5, 5.2),
                arrowprops=dict(arrowstyle="-|>", color=GREEN), color=GREEN, fontsize=9)
    ax.annotate("more control", xy=(6, 1.1), xytext=(8.5, 1.0),
                arrowprops=dict(arrowstyle="-|>", color="#475569"), color="#475569", fontsize=9)
    ax.set_title("The cloud ML stack", color=INK, fontsize=13, fontweight="bold")
    save(fig, "ch46_stack.png")


# ===========================================================================
# CHAPTER 45 — MLOps  (defined first; registered below)
# ===========================================================================
def mlops_lifecycle_diagram():
    """Circular MLOps lifecycle with a retrain loop."""
    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.axis("off")
    steps = ["Data", "Train", "Validate", "Deploy", "Monitor", "Detect drift"]
    colors = [SKY, PRIMARY, VIOLET, GREEN, AMBER, "#ef4444"]
    n = len(steps)
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, n, endpoint=False)
    pts = [(np.cos(a), np.sin(a)) for a in angles]
    for (x, y), s, c in zip(pts, steps, colors):
        ax.add_patch(Circle((x, y), 0.32, fc=c, ec="white", lw=2, zorder=3))
        ax.text(x, y, s, ha="center", va="center", color="white", fontsize=8.5, fontweight="bold", zorder=4)
    for i in range(n):
        x1, y1 = pts[i]; x2, y2 = pts[(i + 1) % n]
        ax.annotate("", xy=(x2 * 0.78, y2 * 0.78), xytext=(x1 * 0.78, y1 * 0.78),
                    arrowprops=dict(arrowstyle="-|>", color="#94a3b8", lw=1.8,
                                    connectionstyle="arc3,rad=0.25"))
    ax.text(0, 0, "MLOps\nloop", ha="center", va="center", fontsize=12, fontweight="bold", color=INK)
    ax.set_title("The MLOps lifecycle: a continuous loop", color=INK, fontsize=13,
                 fontweight="bold")
    save(fig, "ch45_lifecycle.png")


# ===========================================================================
# CHAPTER 44 — Deployment  (defined first; registered below)
# ===========================================================================
def deployment_diagram():
    """Client -> API (loads saved model) -> prediction JSON."""
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    _box(ax, 0.4, 1.8, 2.3, 1.4, "CLIENT\n(web/app)", SKY, fs=9.5)
    _box(ax, 4.2, 1.8, 2.8, 1.4, "MODEL API\n(FastAPI/Flask)", PRIMARY, fs=9.5)
    _box(ax, 8.5, 2.7, 2.8, 1.0, "saved model\n(joblib)", VIOLET, fs=9)
    _box(ax, 8.5, 0.9, 2.8, 1.0, "prediction\n(JSON)", GREEN, fs=9.5)
    ax.annotate("", xy=(4.2, 2.7), xytext=(2.7, 2.7), arrowprops=dict(arrowstyle="-|>", color=INK, lw=2))
    ax.text(3.45, 3.0, "input\n(JSON)", ha="center", fontsize=8, color=INK)
    ax.annotate("", xy=(2.7, 2.2), xytext=(4.2, 2.2), arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2))
    ax.text(3.45, 1.5, "result", ha="center", fontsize=8, color=GREEN)
    _arrow(ax, 7.0, 2.9, 8.5, 3.1, color="#cbd5e1")
    _arrow(ax, 7.0, 2.4, 8.5, 1.5, color="#cbd5e1")
    ax.text(6.0, 0.3, "model loaded ONCE at startup; serves many requests over HTTP",
            ha="center", fontsize=9.5, style="italic", color=INK)
    ax.set_title("Deploying a model as an API", color=INK, fontsize=13, fontweight="bold", pad=2)
    save(fig, "ch44_deployment.png")


# ===========================================================================
# CHAPTER 43 — Generative AI  (defined first; registered below)
# ===========================================================================
def genai_landscape_diagram():
    """Modalities mapped to underlying technologies."""
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    mods = [(0.5, "Text & Code"), (3.4, "Images"), (6.3, "Audio/Music"), (9.2, "Video")]
    techs = [(0.5, "LLMs /\nTransformers", PRIMARY), (3.4, "Diffusion\nmodels", VIOLET),
             (6.3, "Transformers\n+ diffusion", SKY), (9.2, "Diffusion +\nTransformers", GREEN)]
    for x, m in mods:
        _box(ax, x, 4.2, 2.4, 1.0, m, INK, fs=10)
    for (x, t, color) in techs:
        _box(ax, x, 1.6, 2.4, 1.3, t, color, fs=9)
        _arrow(ax, x + 1.2, 4.2, x + 1.2, 2.9, color="#cbd5e1")
    ax.text(6.0, 0.6, "increasingly unified into MULTIMODAL foundation models",
            ha="center", fontsize=10, style="italic", color=INK)
    ax.set_title("The generative-AI landscape", color=INK, fontsize=13, fontweight="bold", pad=2)
    save(fig, "ch43_landscape.png")


def diffusion_diagram():
    """Forward (add noise) and reverse (denoise) diffusion process."""
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 5, figsize=(11, 3.0))
    base = np.zeros((16, 16))
    base[5:11, 5:11] = 1.0   # a simple "image"
    levels = [0.0, 0.4, 0.8, 1.2, 2.0]
    for ax, lv in zip(axes, levels):
        img = base + rng.normal(0, lv, base.shape)
        ax.imshow(img, cmap="magma"); ax.set_xticks([]); ax.set_yticks([])
    axes[0].set_title("image", fontsize=9, color=GREEN, fontweight="bold")
    axes[-1].set_title("pure noise", fontsize=9, color="#ef4444", fontweight="bold")
    fig.text(0.5, 0.04, "→ forward: add noise   |   ← reverse (generation): denoise step by step ←",
             ha="center", fontsize=10, style="italic", color=INK)
    fig.suptitle("Diffusion: learn to reverse noise into images", color=INK,
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save(fig, "ch43_diffusion.png")


# ===========================================================================
# CHAPTER 42 — Time Series  (defined first; registered below)
# ===========================================================================
def ts_components_diagram():
    """Trend, seasonality, noise, and combined series."""
    t = np.arange(120)
    trend = 0.5 * t + 50
    seasonal = 10 * np.sin(2 * np.pi * t / 12)
    rng = np.random.default_rng(0); noise = rng.normal(0, 3, 120)
    combined = trend + seasonal + noise
    fig, axes = plt.subplots(4, 1, figsize=(8, 6.0), sharex=True)
    for ax, data, title, color in [(axes[0], trend, "Trend", SKY),
                                    (axes[1], seasonal, "Seasonality", VIOLET),
                                    (axes[2], noise, "Noise", "#94a3b8"),
                                    (axes[3], combined, "Combined series", PRIMARY)]:
        ax.plot(t, data, color=color, lw=1.8)
        ax.set_ylabel(title, fontsize=9, color=color, fontweight="bold")
        ax.set_yticks([])
    axes[3].set_xlabel("time (months)")
    fig.suptitle("A time series = trend + seasonality + noise", color=INK,
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout()
    save(fig, "ch42_components.png")


def ts_split_diagram():
    """Chronological train/test split."""
    t = np.arange(120)
    rng = np.random.default_rng(1)
    y = 0.5 * t + 10 * np.sin(2 * np.pi * t / 12) + 50 + rng.normal(0, 3, 120)
    split = 96
    fig, ax = plt.subplots(figsize=(9, 4.0))
    ax.plot(t[:split], y[:split], color=PRIMARY, lw=2, label="TRAIN (past)")
    ax.plot(t[split:], y[split:], color="#ef4444", lw=2, label="TEST (future)")
    ax.axvline(split, color="#94a3b8", ls="--")
    ax.axvspan(0, split, color=PRIMARY, alpha=0.06); ax.axvspan(split, 120, color="#ef4444", alpha=0.06)
    ax.text(48, y.max(), "train on the past", ha="center", color=PRIMARY, fontsize=10, fontweight="bold")
    ax.text(108, y.max(), "test on the future", ha="center", color="#ef4444", fontsize=10, fontweight="bold")
    ax.set_xlabel("time"); ax.set_yticks([]); ax.legend(fontsize=9, loc="lower right")
    ax.set_title("Time-series split: chronological, never shuffled", color=INK,
                 fontsize=12, fontweight="bold")
    save(fig, "ch42_split.png")


# ===========================================================================
# CHAPTER 41 — Recommendation Systems  (defined first; registered below)
# ===========================================================================
def rec_approaches_diagram():
    """Content-based vs collaborative filtering."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    ax = axes[0]; ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_title("CONTENT-BASED", color=SKY, fontsize=12, fontweight="bold")
    _box(ax, 0.5, 3.5, 3.8, 1.4, "You liked:\nsci-fi action films", SKY, fs=9.5)
    _box(ax, 5.5, 3.5, 3.8, 1.4, "Recommend:\nother sci-fi action", GREEN, fs=9.5)
    _arrow(ax, 4.3, 4.2, 5.5, 4.2)
    ax.text(5.0, 1.6, "uses ITEM FEATURES\n(similar items)", ha="center", fontsize=9.5, style="italic", color=INK)
    ax = axes[1]; ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_title("COLLABORATIVE FILTERING", color=VIOLET, fontsize=12, fontweight="bold")
    _box(ax, 0.5, 3.5, 3.8, 1.4, "Users LIKE YOU\nliked item X", VIOLET, fs=9.5)
    _box(ax, 5.5, 3.5, 3.8, 1.4, "Recommend:\nitem X to you", GREEN, fs=9.5)
    _arrow(ax, 4.3, 4.2, 5.5, 4.2)
    ax.text(5.0, 1.6, "uses RATING PATTERNS\n(similar users)", ha="center", fontsize=9.5, style="italic", color=INK)
    fig.suptitle("Two recommendation strategies", color=INK, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ch41_approaches.png")


def user_item_matrix_diagram():
    """User-item rating matrix with empty cells."""
    R = [[5, 4, "", 1, ""], [4, 5, 3, 1, 1], [1, "", 5, 4, 4], [1, 1, 4, 5, ""], ["", 1, 5, 4, 5]]
    users = ["Ann", "Bob", "Cara", "Dan", "Eve"]; movies = ["M1", "M2", "M3", "M4", "M5"]
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.set_xlim(0, 7); ax.set_ylim(0, 7); ax.axis("off")
    cw = 1.0
    for j, m in enumerate(movies):
        ax.text(1.8 + j * cw + cw / 2, 6.2, m, ha="center", fontsize=10, fontweight="bold", color=VIOLET)
    for i, u in enumerate(users):
        ax.text(1.5, 5.4 - i * cw + cw / 2 - 0.5, u, ha="right", fontsize=10, fontweight="bold", color=SKY)
        for j in range(5):
            val = R[i][j]
            fc = "#f1f5f9" if val == "" else "#dbeafe"
            ax.add_patch(plt.Rectangle((1.8 + j * cw, 5.4 - i * cw - 0.5), cw, cw, fc=fc, ec="#94a3b8"))
            ax.text(1.8 + j * cw + cw / 2, 5.4 - i * cw + cw / 2 - 0.5, str(val), ha="center", va="center", fontsize=10, color=INK)
    ax.text(3.8, 0.3, "empty cells = unrated → predict these", ha="center", fontsize=9.5, style="italic", color=INK)
    ax.set_title("The user–item rating matrix (sparse)", color=INK, fontsize=12.5, fontweight="bold")
    save(fig, "ch41_matrix.png")


# ===========================================================================
# CHAPTER 40 — Computer Vision  (defined first; registered below)
# ===========================================================================
def cv_tasks_diagram():
    """Classification vs detection vs segmentation."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    titles = ["CLASSIFICATION\n(one label)", "OBJECT DETECTION\n(boxes + labels)",
              "SEGMENTATION\n(per-pixel)"]
    for ax, title in zip(axes, titles):
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_xticks([]); ax.set_yticks([])
        ax.add_patch(plt.Rectangle((0, 0), 10, 10, fc="#f1f5f9", ec="#cbd5e1"))
        ax.set_title(title, fontsize=10, fontweight="bold", color=INK)
    # classification: a shape + label
    axes[0].add_patch(Circle((5, 5), 2.2, fc=SKY)); axes[0].text(5, 1.2, "“cat”", ha="center", fontsize=11, fontweight="bold", color=INK)
    # detection: boxes
    axes[1].add_patch(Circle((3.5, 6), 1.6, fc=SKY)); axes[1].add_patch(plt.Rectangle((1.6, 4.1, ), 3.8, 3.8, fill=False, ec="#ef4444", lw=2))
    axes[1].add_patch(Circle((7, 3.5), 1.3, fc=GREEN)); axes[1].add_patch(plt.Rectangle((5.5, 2.0), 3.0, 3.0, fill=False, ec="#ef4444", lw=2))
    axes[1].text(3.5, 8.2, "cat", ha="center", fontsize=8, color="#ef4444"); axes[1].text(7, 5.3, "dog", ha="center", fontsize=8, color="#ef4444")
    # segmentation: pixel mask
    seg = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            if (i-6)**2+(j-3)**2 < 8: seg[i,j]=1
            elif (i-3)**2+(j-7)**2 < 5: seg[i,j]=2
    axes[2].imshow(seg, cmap="Set2", extent=[0,10,0,10], origin="lower")
    fig.suptitle("Three core computer-vision tasks", color=INK, fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ch40_cv_tasks.png")


def transfer_learning_diagram():
    """Pretrained frozen layers + new trainable head."""
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")
    _box(ax, 0.3, 1.4, 2.3, 1.6, "Pretrained on\nMILLIONS of\nimages", "#94a3b8", fs=9)
    for i, x in enumerate([3.0, 4.6, 6.2]):
        _box(ax, x, 1.5, 1.4, 1.4, f"frozen\nlayer {i+1}", SKY, fs=8)
        if i < 2: _arrow(ax, x+1.4, 2.2, x+1.6, 2.2, color="#cbd5e1")
    _arrow(ax, 7.6, 2.2, 8.2, 2.2, color=INK)
    _box(ax, 8.3, 1.5, 2.0, 1.4, "NEW head\n(train on YOUR\nsmall data)", GREEN, fs=8.5)
    ax.text(6.0, 0.5, "reuse generic features (edges/shapes); only train the new head",
            ha="center", fontsize=10, style="italic", color=INK)
    ax.set_title("Transfer learning", color=INK, fontsize=13, fontweight="bold", pad=2)
    save(fig, "ch40_transfer.png")


# ===========================================================================
# CHAPTER 39 — LLMs  (defined first; registered below)
# ===========================================================================
def llm_training_diagram():
    """Three LLM training stages."""
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")
    stages = [
        (0.3, SKY, "1. PRETRAINING", "self-supervised\nnext-token on\nmassive text", "base model"),
        (4.2, PRIMARY, "2. FINE-TUNING", "supervised on\ninstruction-response\nexamples", "follows instructions"),
        (8.1, VIOLET, "3. RLHF", "align to human\npreferences via\nreinforcement learning", "helpful assistant"),
    ]
    for i, (x, color, title, desc, tag) in enumerate(stages):
        _box(ax, x, 1.2, 3.4, 2.2, "", color)
        ax.text(x + 1.7, 3.0, title, ha="center", color="white", fontsize=11, fontweight="bold")
        ax.text(x + 1.7, 2.0, desc, ha="center", color="white", fontsize=8.6)
        ax.text(x + 1.7, 0.8, tag, ha="center", color=color, fontsize=9, fontweight="bold")
        if i < 2:
            _arrow(ax, x + 3.4, 2.3, x + 3.9, 2.3, color=INK)
    ax.set_title("How an LLM is trained: three stages", color=INK, fontsize=13,
                 fontweight="bold", pad=2)
    save(fig, "ch39_training.png")


# ===========================================================================
# CHAPTER 38 — NLP  (defined first; registered below)
# ===========================================================================
def nlp_representations_diagram():
    """Evolution of text representations."""
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")
    items = [(0.3, SKY, "Bag of Words", "raw counts\n(no order)"),
             (3.1, GREEN, "TF-IDF", "weighted counts\n(down-weight common)"),
             (5.9, VIOLET, "Word embeddings", "dense vectors\n(similar = close)"),
             (8.7, PRIMARY, "Contextual\n(Transformers)", "meaning depends\non context")]
    for i, (x, color, title, desc) in enumerate(items):
        _box(ax, x, 1.2, 2.6, 1.7, "", color)
        ax.text(x + 1.3, 2.5, title, ha="center", color="white", fontsize=10, fontweight="bold")
        ax.text(x + 1.3, 1.7, desc, ha="center", color="white", fontsize=8.3)
        if i < 3:
            _arrow(ax, x + 2.6, 2.05, x + 2.8, 2.05, color=INK)
    ax.text(6.0, 0.4, "each step captures more meaning →", ha="center", fontsize=10,
            style="italic", color=INK)
    ax.set_title("Evolution of text representations", color=INK, fontsize=13,
                 fontweight="bold", pad=2)
    save(fig, "ch38_representations.png")


def embeddings_diagram():
    """2-D word-embedding space with the king-queen analogy."""
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    words = {"king": (3, 4), "queen": (4.2, 4.3), "man": (2.6, 2.2), "woman": (3.8, 2.5),
             "dog": (7, 1.5), "cat": (7.5, 1.9), "puppy": (6.6, 1.1)}
    for w, (x, y) in words.items():
        color = "#ef4444" if w in ("king", "queen", "man", "woman") else SKY
        ax.scatter(x, y, color=color, s=70, zorder=3)
        ax.text(x + 0.1, y + 0.12, w, fontsize=11, color=INK)
    for a, b in [("man", "king"), ("woman", "queen")]:
        ax.annotate("", xy=words[b], xytext=words[a],
                    arrowprops=dict(arrowstyle="-|>", color=VIOLET, lw=2, ls="--"))
    ax.text(5.0, 4.6, "king − man + woman ≈ queen", color=VIOLET, fontsize=10, fontweight="bold")
    ax.set_xlim(1, 9); ax.set_ylim(0, 5.2); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Word embeddings: meaning as geometry", color=INK, fontsize=12, fontweight="bold")
    save(fig, "ch38_embeddings.png")


def nlp_pipeline_diagram():
    """text -> preprocess -> represent -> model -> output."""
    fig, ax = plt.subplots(figsize=(11, 2.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3); ax.axis("off")
    steps = [(0.3, SKY, "Raw text"), (2.4, PRIMARY, "Preprocess\n(clean, tokenize)"),
             (4.8, VIOLET, "Represent\n(TF-IDF / embed)"), (7.2, AMBER, "Model"),
             (9.4, GREEN, "Task output\n(sentiment, NER…)")]
    w = 2.0
    for i, (x, color, lab) in enumerate(steps):
        _box(ax, x, 0.9, w, 1.2, lab, color, fs=8.6)
        if i < len(steps) - 1:
            _arrow(ax, x + w, 1.5, steps[i + 1][0], 1.5)
    ax.set_title("The NLP pipeline", color=INK, fontsize=13, fontweight="bold", pad=2)
    save(fig, "ch38_pipeline.png")


# ===========================================================================
# CHAPTER 37 — Transformers  (defined first; registered below)
# ===========================================================================
def attention_diagram():
    """The word 'it' attending to other words, weight by line thickness."""
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    words = ["The", "animal", "didn't", "cross", "the", "street", "because", "it"]
    xs = np.linspace(0.6, 11.4, len(words))
    weights = [0.05, 0.55, 0.03, 0.05, 0.03, 0.20, 0.04, 0.05]
    for x, w_, word in zip(xs, weights, words):
        color = "#ef4444" if word == "it" else INK
        ax.text(x, 1.0, word, ha="center", fontsize=11,
                fontweight="bold" if word in ("it", "animal") else "normal", color=color)
        if word not in ("it",):
            ax.plot([xs[-1], x], [1.4, 4.0], color=VIOLET, lw=0.5 + w_ * 10, alpha=0.5)
    ax.text(xs[-1], 4.2, "“it” attends to →", ha="center", color=VIOLET, fontsize=10, fontweight="bold")
    ax.text(xs[1], 4.2, "animal (strongest)", ha="center", color=VIOLET, fontsize=9.5)
    ax.set_title("Self-attention: 'it' focuses on 'animal'", color=INK, fontsize=13,
                 fontweight="bold", pad=2)
    save(fig, "ch37_attention.png")


def transformer_diagram():
    """A single Transformer block."""
    fig, ax = plt.subplots(figsize=(7, 5.6))
    ax.set_xlim(0, 8); ax.set_ylim(0, 11); ax.axis("off")
    blocks = [
        (0.5, SKY, "Input embeddings + positional encoding"),
        (2.3, PRIMARY, "Multi-Head Self-Attention"),
        (4.0, "#94a3b8", "Add & Normalise"),
        (5.7, VIOLET, "Feed-Forward Network"),
        (7.4, "#94a3b8", "Add & Normalise"),
        (9.1, GREEN, "Output → next block"),
    ]
    for y, color, lab in blocks:
        _box(ax, 1.0, y, 6.0, 1.1, lab, color, fs=9.5)
        if y < 9.0:
            _arrow(ax, 4.0, y + 1.1, 4.0, y + 1.7)
    ax.text(4.0, 0.2, "× N stacked blocks", ha="center", fontsize=9.5, style="italic", color=INK)
    ax.set_title("A Transformer block", color=INK, fontsize=13, fontweight="bold", pad=2)
    save(fig, "ch37_transformer.png")


# ===========================================================================
# CHAPTER 36 — Generative Models  (defined first; registered below)
# ===========================================================================
def autoencoder_diagram():
    """Hourglass: encoder -> bottleneck -> decoder."""
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    layer_x = [1.0, 3.0, 5.5, 8.0, 10.0]
    counts = [6, 4, 2, 4, 6]
    colors = [SKY, PRIMARY, VIOLET, PRIMARY, GREEN]
    pos = []
    for x, c in zip(layer_x, counts):
        ys = np.linspace(1.2, 4.8, c)
        pos.append([(x, y) for y in ys])
    for li in range(len(pos) - 1):
        for (x1, y1) in pos[li]:
            for (x2, y2) in pos[li + 1]:
                ax.plot([x1, x2], [y1, y2], color="#e2e8f0", lw=0.5, zorder=1)
    for layer, color in zip(pos, colors):
        for (x, y) in layer:
            ax.add_patch(Circle((x, y), 0.2, fc=color, ec="white", lw=1, zorder=3))
    ax.text(1.0, 5.3, "input", ha="center", fontsize=9.5, color=SKY, fontweight="bold")
    ax.text(2.0, 0.6, "ENCODER", ha="center", fontsize=9.5, color=PRIMARY, fontweight="bold")
    ax.text(5.5, 5.3, "bottleneck\n(latent code)", ha="center", fontsize=9, color=VIOLET, fontweight="bold")
    ax.text(9.0, 0.6, "DECODER", ha="center", fontsize=9.5, color=PRIMARY, fontweight="bold")
    ax.text(10.0, 5.3, "reconstruction", ha="center", fontsize=9.5, color=GREEN, fontweight="bold")
    ax.set_title("Autoencoder: compress then reconstruct", color=INK, fontsize=13, fontweight="bold")
    save(fig, "ch36_autoencoder.png")


def gan_diagram():
    """Generator vs discriminator."""
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    _box(ax, 0.4, 3.4, 2.0, 1.0, "random\nnoise", "#94a3b8", fs=9)
    _box(ax, 3.0, 3.4, 2.4, 1.0, "GENERATOR\n(forger)", VIOLET, fs=9.5)
    _box(ax, 6.0, 3.4, 1.9, 1.0, "fake\nsamples", "#fca5a5", fs=9)
    _box(ax, 6.0, 1.0, 1.9, 1.0, "real\ndata", "#86efac", fs=9)
    _box(ax, 8.6, 2.2, 2.6, 1.2, "DISCRIMINATOR\n(detective)", PRIMARY, fs=9)
    _box(ax, 8.6, 4.4, 2.6, 0.9, "real or fake?", GREEN, fs=9.5)
    _arrow(ax, 2.4, 3.9, 3.0, 3.9); _arrow(ax, 5.4, 3.9, 6.0, 3.9)
    _arrow(ax, 7.9, 3.9, 8.6, 3.1); _arrow(ax, 7.9, 1.5, 8.6, 2.5)
    _arrow(ax, 9.9, 3.4, 9.9, 4.4)
    ax.annotate("feedback trains the generator", xy=(4.2, 3.4), xytext=(6.0, 0.3),
                arrowprops=dict(arrowstyle="-|>", color="#ef4444", lw=1.6, ls="--",
                                connectionstyle="arc3,rad=0.3"), color="#ef4444", fontsize=9, ha="center")
    ax.set_title("A GAN: generator vs discriminator (adversarial training)",
                 color=INK, fontsize=12.5, fontweight="bold", pad=2)
    save(fig, "ch36_gan.png")


# ===========================================================================
# CHAPTER 35 — RNN/LSTM  (defined first; registered below)
# ===========================================================================
def rnn_unrolled_diagram():
    """RNN unrolled across time with hidden-state arrows."""
    fig, ax = plt.subplots(figsize=(11, 4.0))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    xs = [1.0, 4.0, 7.0, 10.0]; labels = ["x₁", "x₂", "x₃", "x₄"]
    for i, (x, lab) in enumerate(zip(xs, labels)):
        _box(ax, x - 0.7, 2.0, 1.5, 1.2, "RNN\ncell", PRIMARY, fs=9)
        ax.add_patch(Circle((x + 0.05, 0.8), 0.32, fc="#bae6fd", ec=SKY, lw=1.5))
        ax.text(x + 0.05, 0.8, lab, ha="center", va="center", fontsize=9, fontweight="bold")
        _arrow(ax, x + 0.05, 1.12, x + 0.05, 2.0, color="#94a3b8")
        ax.add_patch(Circle((x + 0.05, 4.2), 0.32, fc="#bbf7d0", ec=GREEN, lw=1.5))
        ax.text(x + 0.05, 4.2, f"y{i+1}", ha="center", va="center", fontsize=9, fontweight="bold")
        _arrow(ax, x + 0.05, 3.2, x + 0.05, 3.88, color="#94a3b8")
        if i < 3:
            _arrow(ax, x + 0.8, 2.6, xs[i + 1] - 0.7, 2.6, color="#ef4444")
    ax.text(2.5, 2.95, "hidden state", color="#ef4444", fontsize=9, style="italic")
    ax.text(6.0, 0.15, "the same cell processes each step, passing memory (hidden state) forward",
            ha="center", fontsize=9.5, style="italic", color=INK)
    ax.set_title("An RNN unrolled across time", color=INK, fontsize=13, fontweight="bold", pad=2)
    save(fig, "ch35_rnn_unrolled.png")


def lstm_diagram():
    """LSTM cell with three gates and the cell-state conveyor."""
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    ax.add_patch(plt.Rectangle((1.5, 1.0), 9.0, 4.0, fc="#eef2ff", ec=PRIMARY, lw=2))
    # cell state conveyor (top line)
    ax.annotate("", xy=(10.6, 4.4), xytext=(1.4, 4.4),
                arrowprops=dict(arrowstyle="-|>", color=VIOLET, lw=2.6))
    ax.text(6.0, 4.65, "cell state (long-term memory conveyor)", ha="center",
            color=VIOLET, fontsize=9.5, fontweight="bold")
    gates = [(3.0, "#ef4444", "FORGET\ngate"), (5.6, GREEN, "INPUT\ngate"), (8.2, AMBER, "OUTPUT\ngate")]
    for x, color, lab in gates:
        _box(ax, x, 2.0, 1.6, 1.1, lab, color, fs=9)
        _arrow(ax, x + 0.8, 3.1, x + 0.8, 4.4, color="#94a3b8")
    # hidden state in/out
    ax.add_patch(Circle((0.7, 2.5), 0.3, fc="#bae6fd", ec=SKY)); ax.text(0.7, 2.5, "hₜ₋₁", ha="center", va="center", fontsize=8)
    ax.add_patch(Circle((11.3, 2.5), 0.3, fc="#bbf7d0", ec=GREEN)); ax.text(11.3, 2.5, "hₜ", ha="center", va="center", fontsize=8)
    ax.text(6.0, 0.4, "gates learn what to forget, store, and output", ha="center",
            fontsize=10, style="italic", color=INK)
    ax.set_title("An LSTM cell: three gates protect long-term memory",
                 color=INK, fontsize=12.5, fontweight="bold", pad=2)
    save(fig, "ch35_lstm.png")


# ===========================================================================
# CHAPTER 34 — CNNs  (defined first; registered below)
# ===========================================================================
def _grid(ax, x0, y0, vals, cell=0.5, hl=None, ec="#94a3b8", cmap_hi="#fde68a"):
    rows, cols = vals.shape
    for r in range(rows):
        for c in range(cols):
            fc = cmap_hi if (hl and (r, c) in hl) else "white"
            ax.add_patch(plt.Rectangle((x0 + c * cell, y0 - r * cell), cell, cell,
                                       fc=fc, ec=ec, lw=1.1))
            ax.text(x0 + c * cell + cell / 2, y0 - r * cell + cell / 2, str(int(vals[r, c])),
                    ha="center", va="center", fontsize=8, color=INK)


def convolution_diagram():
    """A 3x3 filter over an input producing one feature-map value."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 2, (5, 5))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    hl = {(r, c) for r in range(3) for c in range(3)}
    _grid(ax, 0.5, 5.0, img, hl=hl)
    ax.text(1.75, 5.4, "input image", ha="center", fontsize=9.5, color=INK, fontweight="bold")
    ax.text(4.6, 3.0, "✷ filter\n(3×3)", ha="center", fontsize=10, color=PRIMARY, fontweight="bold")
    _arrow(ax, 3.3, 3.0, 5.6, 3.0, color=INK)
    fmap = rng.integers(0, 5, (3, 3))
    _grid(ax, 6.4, 4.2, fmap, ec=GREEN)
    ax.text(7.15, 4.6, "feature map", ha="center", fontsize=9.5, color=GREEN, fontweight="bold")
    ax.text(6.0, 0.6, "filter slides over the image; each position → one feature-map value",
            ha="center", fontsize=10, style="italic", color=INK)
    ax.set_title("The convolution operation", color=INK, fontsize=13, fontweight="bold", pad=2)
    save(fig, "ch34_convolution.png")


def pooling_diagram():
    """4x4 feature map -> 2x2 via 2x2 max pooling."""
    vals = np.array([[1, 3, 2, 4], [5, 6, 1, 2], [7, 2, 3, 8], [1, 0, 4, 5]])
    pooled = np.array([[6, 4], [7, 8]])
    fig, ax = plt.subplots(figsize=(9, 4.0))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    _grid(ax, 0.5, 5.0, vals, cell=0.7, ec=SKY)
    ax.text(1.9, 5.5, "feature map (4×4)", ha="center", fontsize=9.5, color=SKY, fontweight="bold")
    _arrow(ax, 4.0, 3.3, 6.0, 3.3, color=INK)
    ax.text(5.0, 3.7, "2×2 max pool", ha="center", fontsize=9.5, style="italic", color=INK)
    _grid(ax, 6.7, 4.3, pooled, cell=0.8, ec=GREEN, cmap_hi="white")
    ax.text(7.5, 5.5, "pooled (2×2)", ha="center", fontsize=9.5, color=GREEN, fontweight="bold")
    ax.text(6.0, 0.7, "keeps the maximum of each window → smaller, shift-robust",
            ha="center", fontsize=10, style="italic", color=INK)
    ax.set_title("Max pooling", color=INK, fontsize=13, fontweight="bold", pad=2)
    save(fig, "ch34_pooling.png")


def cnn_arch_diagram():
    """End-to-end CNN pipeline."""
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")
    blocks = [(0.3, SKY, "INPUT\nimage"), (2.0, PRIMARY, "Conv+ReLU\n+ Pool"),
              (4.0, PRIMARY, "Conv+ReLU\n+ Pool"), (6.0, VIOLET, "Flatten"),
              (7.7, VIOLET, "Fully\nconnected"), (9.6, GREEN, "Softmax\n(classes)")]
    w = 1.5
    for i, (x, color, lab) in enumerate(blocks):
        _box(ax, x, 1.4, w, 1.3, lab, color, fs=8.6)
        if i < len(blocks) - 1:
            _arrow(ax, x + w, 2.05, blocks[i + 1][0], 2.05)
    ax.text(3.0, 0.6, "edges → shapes", ha="center", fontsize=8.5, color="#64748b", style="italic")
    ax.text(8.5, 0.6, "objects → class", ha="center", fontsize=8.5, color="#64748b", style="italic")
    ax.set_title("A typical CNN architecture", color=INK, fontsize=13, fontweight="bold", pad=2)
    save(fig, "ch34_cnn_arch.png")


# ===========================================================================
# CHAPTER 33 — Training Deep Networks  (defined first; registered below)
# ===========================================================================
def backprop_diagram():
    """Forward pass (right) and backward gradient flow (left) through layers."""
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    names = ["Input", "Hidden 1", "Hidden 2", "Output", "Loss"]
    colors = [SKY, PRIMARY, PRIMARY, GREEN, "#ef4444"]
    xs = [0.5 + i * 2.4 for i in range(5)]
    for x, name, color in zip(xs, names, colors):
        _box(ax, x, 2.0, 1.9, 1.2, name, color, fs=9.5)
    for i in range(4):
        _arrow(ax, xs[i] + 1.9, 3.0, xs[i + 1], 3.0, color=INK)        # forward
    ax.text(6.0, 3.5, "FORWARD pass →", ha="center", color=INK, fontsize=10, fontweight="bold")
    for i in range(4):
        ax.annotate("", xy=(xs[i] + 1.5, 1.7), xytext=(xs[i + 1] + 0.4, 1.7),
                    arrowprops=dict(arrowstyle="-|>", color="#ef4444", lw=2))
    ax.text(6.0, 0.9, "← BACKWARD pass (gradients via chain rule)", ha="center",
            color="#ef4444", fontsize=10, fontweight="bold")
    ax.set_title("Backpropagation: forward predict, backward gradients",
                 color=INK, fontsize=12.5, fontweight="bold", pad=2)
    save(fig, "ch33_backprop.png")


def optimizers_diagram():
    """Real SGD vs Adam training-loss curves."""
    import torch, torch.nn as nn
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xt = torch.tensor(sc.transform(Xtr), dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32).view(-1, 1)

    def run(opt_name):
        torch.manual_seed(0)
        m = nn.Sequential(nn.Linear(30, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())
        opt = (torch.optim.SGD(m.parameters(), lr=0.1) if opt_name == "SGD"
               else torch.optim.Adam(m.parameters(), lr=0.01))
        lf = nn.BCELoss(); losses = []
        for _ in range(50):
            opt.zero_grad(); loss = lf(m(Xt), yt); loss.backward(); opt.step()
            losses.append(loss.item())
        return losses
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(run("SGD"), color=AMBER, lw=2.4, label="SGD")
    ax.plot(run("Adam"), color=PRIMARY, lw=2.4, label="Adam")
    ax.set_xlabel("epoch"); ax.set_ylabel("training loss"); ax.legend(fontsize=10)
    ax.grid(alpha=0.2)
    ax.set_title("Adam converges faster than SGD", color=INK, fontsize=12, fontweight="bold")
    save(fig, "ch33_optimizers.png")


def dropout_diagram():
    """Full network vs dropout (some neurons switched off)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    rng = np.random.default_rng(0)
    layers = [3, 5, 5, 2]
    for ax, title, drop in [(axes[0], "Full network", False), (axes[1], "With dropout (training)", True)]:
        ax.set_xlim(0, 5); ax.set_ylim(0, 6); ax.axis("off")
        pos = []
        for li, c in enumerate(layers):
            ys = np.linspace(1, 5, c)
            pos.append([(1 + li * 1.1, y) for y in ys])
        dropped = set()
        if drop:
            for li in [1, 2]:
                for ni in range(layers[li]):
                    if rng.random() < 0.4:
                        dropped.add((li, ni))
        for li in range(len(layers) - 1):
            for ai, (x1, y1) in enumerate(pos[li]):
                for bi, (x2, y2) in enumerate(pos[li + 1]):
                    if (li, ai) in dropped or (li + 1, bi) in dropped:
                        continue
                    ax.plot([x1, x2], [y1, y2], color="#e2e8f0", lw=0.6, zorder=1)
        for li, layer in enumerate(pos):
            for ni, (x, y) in enumerate(layer):
                if (li, ni) in dropped:
                    ax.add_patch(Circle((x, y), 0.18, fc="white", ec="#cbd5e1", lw=1, zorder=3))
                    ax.plot(x, y, "x", color="#ef4444", ms=8, zorder=4)
                else:
                    ax.add_patch(Circle((x, y), 0.18, fc=PRIMARY, ec="white", lw=1, zorder=3))
        ax.set_title(title, fontsize=11, fontweight="bold", color=INK)
    fig.suptitle("Dropout randomly switches off neurons during training",
                 color=INK, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ch33_dropout.png")


def train_val_diagram():
    """Classic overfitting curve: train loss down, val loss U-shaped."""
    e = np.linspace(0, 50, 200)
    train = 1.2 * np.exp(-e / 12) + 0.05
    val = 1.2 * np.exp(-e / 12) + 0.12 + 0.012 * np.maximum(0, e - 18)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(e, train, color=PRIMARY, lw=2.4, label="training loss")
    ax.plot(e, val, color="#ef4444", lw=2.4, label="validation loss")
    best = e[np.argmin(val)]
    ax.axvline(best, color=GREEN, ls="--", lw=1.6)
    ax.annotate("early stopping\n(val minimum)", xy=(best, val.min()),
                xytext=(best + 6, val.min() + 0.3),
                arrowprops=dict(arrowstyle="-|>", color=GREEN), color=GREEN, fontsize=9.5)
    ax.fill_between(e, train, val, where=(e > best), color="#fca5a5", alpha=0.25)
    ax.text(38, 0.5, "overfitting gap", color="#ef4444", fontsize=9.5)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(fontsize=10)
    ax.set_yticks([]); ax.grid(alpha=0.2)
    ax.set_title("Train vs validation loss: spotting overfitting", color=INK,
                 fontsize=12, fontweight="bold")
    save(fig, "ch33_train_val.png")


# ===========================================================================
# CHAPTER 32 — Neural Networks  (defined first; registered below)
# ===========================================================================
def neuron_diagram():
    """An artificial neuron: inputs -> weights -> sum -> activation -> output."""
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    ins = [(0.6, 4.6, "x₁"), (0.6, 3.0, "x₂"), (0.6, 1.4, "x₃")]
    sumxy = (5.6, 3.0)
    for ix, iy, lab in ins:
        ax.add_patch(Circle((ix, iy), 0.5, fc="#bae6fd", ec=SKY, lw=1.8))
        ax.text(ix, iy, lab, ha="center", va="center", fontsize=11, fontweight="bold")
        _arrow(ax, ix + 0.5, iy, sumxy[0] - 0.7, sumxy[1], color="#94a3b8")
        ax.text((ix + sumxy[0]) / 2 - 0.3, (iy + sumxy[1]) / 2 + 0.15, "w", color=PRIMARY, fontsize=9)
    ax.add_patch(Circle(sumxy, 0.75, fc="#ddd6fe", ec=VIOLET, lw=2))
    ax.text(*sumxy, "Σ +b", ha="center", va="center", fontsize=12, fontweight="bold", color=VIOLET)
    _box(ax, 7.2, 2.3, 2.0, 1.4, "activation\nφ(z)", AMBER, fs=10)
    _arrow(ax, sumxy[0] + 0.75, sumxy[1], 7.2, sumxy[1])
    ax.add_patch(Circle((10.6, 3.0), 0.6, fc="#bbf7d0", ec=GREEN, lw=2))
    ax.text(10.6, 3.0, "output", ha="center", va="center", fontsize=9, fontweight="bold")
    _arrow(ax, 9.2, 3.0, 10.0, 3.0)
    ax.text(5.6, 0.5, "a = φ(w·x + b)", ha="center", fontsize=12, style="italic", color=INK)
    ax.set_title("The artificial neuron", color=INK, fontsize=13, fontweight="bold", pad=2)
    save(fig, "ch32_neuron.png")


def activations_diagram():
    """Sigmoid, tanh, ReLU curves + softmax bars."""
    z = np.linspace(-5, 5, 200)
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.0))
    axes[0].plot(z, 1 / (1 + np.exp(-z)), color=PRIMARY, lw=2.4); axes[0].set_title("Sigmoid", fontweight="bold", fontsize=10)
    axes[1].plot(z, np.tanh(z), color=SKY, lw=2.4); axes[1].set_title("Tanh", fontweight="bold", fontsize=10)
    axes[2].plot(z, np.maximum(0, z), color=GREEN, lw=2.4); axes[2].set_title("ReLU", fontweight="bold", fontsize=10)
    s = np.exp([1, 2, 0.5]) / np.exp([1, 2, 0.5]).sum()
    axes[3].bar(["A", "B", "C"], s, color=VIOLET); axes[3].set_title("Softmax (→ probs)", fontweight="bold", fontsize=10)
    for ax in axes[:3]:
        ax.axhline(0, color="#cbd5e1", lw=0.8); ax.axvline(0, color="#cbd5e1", lw=0.8)
        ax.set_xticks([]); ax.set_yticks([])
    axes[3].set_yticks([])
    fig.suptitle("Activation functions", color=INK, fontsize=13, fontweight="bold", y=1.05)
    fig.tight_layout()
    save(fig, "ch32_activations.png")


def mlp_diagram():
    """A small multi-layer perceptron architecture."""
    fig, ax = plt.subplots(figsize=(9, 5.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
    layers = [(1.5, 4, "Input"), (4.0, 5, "Hidden 1"), (6.5, 5, "Hidden 2"), (9.0, 2, "Output")]
    positions = []
    for x, count, _ in layers:
        ys = np.linspace(1.5, 6.5, count)
        positions.append([(x, y) for y in ys])
    colors = [SKY, PRIMARY, PRIMARY, GREEN]
    # connections
    for li in range(len(positions) - 1):
        for (x1, y1) in positions[li]:
            for (x2, y2) in positions[li + 1]:
                ax.plot([x1, x2], [y1, y2], color="#e2e8f0", lw=0.7, zorder=1)
    for (x, count, name), pts, color in zip(layers, positions, colors):
        for (px, py) in pts:
            ax.add_patch(Circle((px, py), 0.28, fc=color, ec="white", lw=1.2, zorder=3))
        ax.text(x, 7.2, name, ha="center", fontsize=10, fontweight="bold", color=color)
    ax.text(5, 0.5, "data flows left → right (the forward pass)", ha="center",
            fontsize=10, style="italic", color=INK)
    ax.set_title("A multi-layer perceptron (MLP)", color=INK, fontsize=13, fontweight="bold")
    save(fig, "ch32_mlp.png")


# ===========================================================================
# CHAPTER 31 — Reinforcement Learning  (defined first; registered below)
# ===========================================================================
def rl_loop2_diagram():
    """Detailed agent <-> environment loop with state/action/reward."""
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")
    _box(ax, 1.0, 2.8, 3.2, 1.6, "AGENT\npolicy π(s) → a", PRIMARY, fs=11)
    _box(ax, 7.8, 2.8, 3.2, 1.6, "ENVIRONMENT", GREEN, fs=12)
    ax.annotate("", xy=(7.8, 4.1), xytext=(4.2, 4.1),
                arrowprops=dict(arrowstyle="-|>", color=PRIMARY, lw=2.4,
                                connectionstyle="arc3,rad=-0.35"))
    ax.text(6.0, 5.9, "ACTION  a", ha="center", color=PRIMARY, fontsize=11, fontweight="bold")
    ax.annotate("", xy=(4.2, 3.1), xytext=(7.8, 3.1),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.4,
                                connectionstyle="arc3,rad=-0.35"))
    ax.text(6.0, 0.9, "REWARD  r   +   next STATE  s'", ha="center", color=GREEN,
            fontsize=11, fontweight="bold")
    ax.text(6.0, 6.6, "goal: maximise long-term (discounted) reward",
            ha="center", fontsize=10, style="italic", color=INK)
    ax.set_title("The Reinforcement Learning loop", color=INK, fontsize=13,
                 fontweight="bold", pad=0)
    save(fig, "ch31_rl_loop.png")


def explore_exploit_diagram():
    """Exploration vs exploitation conceptual split."""
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    _box(ax, 0.5, 1.6, 4.6, 2.2, "", SKY)
    ax.text(2.8, 3.2, "EXPLOIT", ha="center", color="white", fontsize=14, fontweight="bold")
    ax.text(2.8, 2.2, "use the best action\nyou already know\n(safe, immediate reward)",
            ha="center", color="white", fontsize=9.5)
    _box(ax, 6.9, 1.6, 4.6, 2.2, "", VIOLET)
    ax.text(9.2, 3.2, "EXPLORE", ha="center", color="white", fontsize=14, fontweight="bold")
    ax.text(9.2, 2.2, "try a new action\nthat MIGHT be better\n(risky, may discover gold)",
            ha="center", color="white", fontsize=9.5)
    ax.text(6.0, 0.6, "ε-greedy: explore with probability ε, otherwise exploit",
            ha="center", fontsize=10, style="italic", color=INK)
    ax.set_title("Exploration vs Exploitation", color=INK, fontsize=13, fontweight="bold", pad=2)
    save(fig, "ch31_explore.png")


# ===========================================================================
# CHAPTER 30 — Semi-Supervised  (defined first; registered below)
# ===========================================================================
def semi_supervised_diagram():
    """Many unlabelled (grey) points + a few labelled (coloured) anchors."""
    rng = np.random.default_rng(1)
    g1 = rng.normal([2, 2], 0.6, (40, 2)); g2 = rng.normal([6, 6], 0.6, (40, 2))
    fig, ax = plt.subplots(figsize=(7, 5.0))
    ax.scatter(g1[:, 0], g1[:, 1], color="#cbd5e1", s=24)
    ax.scatter(g2[:, 0], g2[:, 1], color="#cbd5e1", s=24, label="unlabelled (many)")
    ax.scatter(g1[:2, 0], g1[:2, 1], color="#ef4444", s=120, edgecolor="black", label="class A label")
    ax.scatter(g2[:2, 0], g2[:2, 1], color=SKY, s=120, edgecolor="black", label="class B label")
    ax.text(2, 0.4, "structure from unlabelled\n+ meaning from few labels", ha="center",
            fontsize=9.5, style="italic", color=INK)
    ax.legend(fontsize=8.5, loc="upper left"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Semi-supervised: few labels anchor many unlabelled points",
                 color=INK, fontsize=12, fontweight="bold")
    save(fig, "ch30_semi_supervised.png")


def self_training_diagram():
    """Self-training loop."""
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    _box(ax, 0.5, 3.6, 2.6, 1.2, "1. Train on\nlabelled data", SKY, fs=9.5)
    _box(ax, 4.6, 3.6, 2.6, 1.2, "2. Predict the\nunlabelled data", PRIMARY, fs=9.5)
    _box(ax, 8.7, 3.6, 2.8, 1.2, "3. Add CONFIDENT\npredictions\n(pseudo-labels)", VIOLET, fs=8.8)
    _arrow(ax, 3.1, 4.2, 4.6, 4.2); _arrow(ax, 7.2, 4.2, 8.7, 4.2)
    ax.annotate("4. retrain & repeat", xy=(1.8, 3.6), xytext=(6, 1.4),
                arrowprops=dict(arrowstyle="-|>", color="#ef4444", lw=1.8,
                                connectionstyle="arc3,rad=0.3", ls="--"),
                color="#ef4444", fontsize=10, ha="center")
    ax.set_title("Self-training: the model teaches itself", color=INK,
                 fontsize=13, fontweight="bold", pad=2)
    save(fig, "ch30_self_training.png")


# ===========================================================================
# CHAPTER 29 — Association Rules  (defined first; registered below)
# ===========================================================================
def market_basket_diagram():
    """Baskets -> rule {bread,butter} -> milk with the three metrics."""
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    baskets = ["🛒 bread, butter, milk", "🛒 bread, butter", "🛒 bread, milk",
               "🛒 bread, butter, milk, jam"]
    for i, b in enumerate(baskets):
        ax.text(0.4, 5.0 - i * 0.7, b, fontsize=10, color=INK)
    _box(ax, 4.6, 2.6, 2.8, 1.3, "{bread, butter}\n→ {milk}", PRIMARY, fs=10)
    _arrow(ax, 3.8, 3.3, 4.6, 3.3, color="#94a3b8")
    metrics = [(8.0, SKY, "SUPPORT", "how frequent"),
               (8.0, GREEN, "CONFIDENCE", "how reliable"),
               (8.0, VIOLET, "LIFT", "vs chance (key!)")]
    for j, (x, color, name, desc) in enumerate(metrics):
        y = 4.3 - j * 1.25
        _box(ax, x, y, 3.4, 0.95, f"{name} — {desc}", color, fs=9)
        _arrow(ax, 7.4, 3.25, x, y + 0.5, color="#cbd5e1")
    ax.set_title("Market-basket analysis: rules and their metrics", color=INK,
                 fontsize=13, fontweight="bold", pad=2)
    save(fig, "ch29_market_basket.png")


# ===========================================================================
# CHAPTER 28 — Dimensionality Reduction  (defined first; registered below)
# ===========================================================================
def pca_diagram():
    """Correlated 2-D cloud with principal-component axes drawn."""
    rng = np.random.default_rng(2)
    n = 200
    base = rng.normal(0, 1, n)
    x = base * 2.2 + rng.normal(0, 0.4, n)
    y = base * 1.1 + rng.normal(0, 0.4, n)
    pts = np.c_[x, y]; pts = pts - pts.mean(0)
    cov = np.cov(pts.T); vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]; vals, vecs = vals[order], vecs[:, order]
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    ax.scatter(pts[:, 0], pts[:, 1], color=SKY, s=14, alpha=0.7)
    for i, (color, lab) in enumerate([(("#ef4444"), "PC1 (most variance)"),
                                      ((VIOLET), "PC2")]):
        v = vecs[:, i] * np.sqrt(vals[i]) * 2.5
        ax.annotate("", xy=v, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=3))
        ax.text(v[0] * 1.05, v[1] * 1.05, lab, color=color, fontsize=10, fontweight="bold")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("PCA: new axes along directions of greatest variance",
                 color=INK, fontsize=12, fontweight="bold")
    save(fig, "ch28_pca.png")


def explained_variance_diagram():
    """Cumulative explained variance vs number of components (digits)."""
    from sklearn.datasets import load_digits
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    X, _ = load_digits(return_X_y=True)
    pca = PCA().fit(StandardScaler().fit_transform(X))
    cum = np.cumsum(pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(range(1, len(cum) + 1), cum, color=PRIMARY, lw=2.4)
    ax.axhline(0.9, color="#ef4444", ls="--", lw=1.4)
    n90 = int(np.argmax(cum >= 0.9)) + 1
    ax.axvline(n90, color="#ef4444", ls="--", lw=1.4)
    ax.scatter([n90], [cum[n90 - 1]], color="#ef4444", s=80, zorder=3)
    ax.text(n90 + 1, 0.55, f"{n90} components\n→ 90% variance", color="#ef4444", fontsize=9.5)
    ax.set_xlabel("number of components"); ax.set_ylabel("cumulative explained variance")
    ax.grid(alpha=0.2)
    ax.set_title("How many components to keep?", color=INK, fontsize=12, fontweight="bold")
    save(fig, "ch28_explained_variance.png")


def tsne_diagram():
    """t-SNE 2-D embedding of the digits, coloured by digit."""
    from sklearn.datasets import load_digits
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    X, y = load_digits(return_X_y=True)
    Xp = PCA(n_components=20, random_state=0).fit_transform(X)   # pre-reduce for speed
    emb = TSNE(n_components=2, random_state=0, init="pca", perplexity=30).fit_transform(Xp)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=y, cmap="tab10", s=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(sc, ax=ax, label="digit", ticks=range(10))
    ax.set_title("t-SNE: 64-D digit images projected to 2-D", color=INK,
                 fontsize=12, fontweight="bold")
    save(fig, "ch28_tsne.png")


# ===========================================================================
# CHAPTER 27 — Clustering  (defined first; registered below)
# ===========================================================================
def kmeans_diagram():
    """K-Means result: coloured clusters with centroids."""
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(1)
    pts = np.vstack([rng.normal([2, 2], 0.6, (40, 2)),
                     rng.normal([6, 6], 0.6, (40, 2)),
                     rng.normal([2.5, 6.5], 0.6, (40, 2))])
    km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(pts)
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    colors = [SKY, VIOLET, GREEN]
    for c in range(3):
        m = km.labels_ == c
        ax.scatter(pts[m, 0], pts[m, 1], color=colors[c], s=28)
    ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
               color="black", marker="X", s=220, edgecolor="white", lw=1.5, zorder=5)
    ax.text(km.cluster_centers_[0, 0], km.cluster_centers_[0, 1] + 0.5, "centroids",
            fontsize=9, color=INK)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("K-Means: points grouped around k centroids", color=INK,
                 fontsize=12, fontweight="bold")
    save(fig, "ch27_kmeans.png")


def elbow_diagram():
    """Inertia vs k with the elbow marked."""
    from sklearn.datasets import load_iris
    from sklearn.cluster import KMeans
    X, _ = load_iris(return_X_y=True)
    ks = range(1, 8); inertias = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(X).inertia_ for k in ks]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(list(ks), inertias, "o-", color=PRIMARY, lw=2.4)
    ax.scatter([3], [inertias[2]], color="#ef4444", s=120, zorder=3)
    ax.annotate("elbow (k≈3)", xy=(3, inertias[2]), xytext=(4.2, inertias[2] + 120),
                arrowprops=dict(arrowstyle="-|>", color="#ef4444"), color="#ef4444", fontsize=10)
    ax.set_xlabel("number of clusters k"); ax.set_ylabel("inertia")
    ax.grid(alpha=0.2)
    ax.set_title("The elbow method", color=INK, fontsize=12, fontweight="bold")
    save(fig, "ch27_elbow.png")


def dendrogram_diagram():
    """A hierarchical-clustering dendrogram."""
    from scipy.cluster.hierarchy import dendrogram, linkage
    rng = np.random.default_rng(3)
    pts = np.vstack([rng.normal([0, 0], 0.4, (5, 2)),
                     rng.normal([4, 0], 0.4, (5, 2)),
                     rng.normal([2, 4], 0.4, (5, 2))])
    Z = linkage(pts, method="ward")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    dendrogram(Z, ax=ax, color_threshold=4, above_threshold_color="#94a3b8")
    ax.axhline(4, color="#ef4444", ls="--", lw=1.6)
    ax.text(1, 4.3, "cut here → 3 clusters", color="#ef4444", fontsize=9.5)
    ax.set_xlabel("data points"); ax.set_ylabel("merge distance")
    ax.set_title("Hierarchical clustering: a dendrogram", color=INK, fontsize=12, fontweight="bold")
    save(fig, "ch27_dendrogram.png")


def dbscan_diagram():
    """K-Means vs DBSCAN on two moons."""
    from sklearn.datasets import make_moons
    from sklearn.cluster import KMeans, DBSCAN
    X, _ = make_moons(n_samples=300, noise=0.06, random_state=0)
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
    db = DBSCAN(eps=0.2, min_samples=5).fit(X)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].scatter(X[:, 0], X[:, 1], c=km.labels_, cmap="coolwarm", s=14)
    axes[0].set_title("K-Means (wrong: splits the middle)", color="#ef4444", fontsize=11, fontweight="bold")
    axes[1].scatter(X[:, 0], X[:, 1], c=db.labels_, cmap="coolwarm", s=14)
    axes[1].set_title("DBSCAN (correct: follows density)", color=GREEN, fontsize=11, fontweight="bold")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("K-Means vs DBSCAN on crescent-shaped clusters", color=INK,
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ch27_dbscan.png")


# ===========================================================================
# CHAPTER 26 — Tuning & Regularization  (defined first; registered below)
# ===========================================================================
def search_diagram():
    """Grid search vs random search."""
    rng = np.random.default_rng(2)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    gx, gy = np.meshgrid(np.linspace(0.15, 0.85, 4), np.linspace(0.15, 0.85, 4))
    axes[0].scatter(gx, gy, color=PRIMARY, s=55)
    axes[0].set_title("GRID search\n(every combination)", color=PRIMARY, fontsize=11, fontweight="bold")
    rx, ry = rng.uniform(0.1, 0.9, 16), rng.uniform(0.1, 0.9, 16)
    axes[1].scatter(rx, ry, color=GREEN, s=55)
    axes[1].set_title("RANDOM search\n(random combinations)", color=GREEN, fontsize=11, fontweight="bold")
    for ax in axes:
        ax.set_xlabel("hyperparameter 1"); ax.set_ylabel("hyperparameter 2")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.suptitle("Grid vs random hyperparameter search", color=INK, fontsize=13,
                 fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ch26_search.png")


def l1_l2_diagram():
    """L1 diamond vs L2 circle constraint geometry."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    th = np.linspace(0, 2 * np.pi, 200)
    # contours of the loss (ellipses) centred away from origin
    for ax in axes:
        for r in [0.5, 1.0, 1.5]:
            ax.plot(1.6 + r * np.cos(th), 1.2 + 0.6 * r * np.sin(th), color="#cbd5e1", lw=1)
        ax.scatter([1.6], [1.2], color="#94a3b8", s=20)
        ax.set_xlim(-1.8, 3.2); ax.set_ylim(-1.8, 2.6); ax.set_xticks([]); ax.set_yticks([])
        ax.axhline(0, color="#e2e8f0", lw=0.8); ax.axvline(0, color="#e2e8f0", lw=0.8)
    # L1 diamond
    axes[0].plot([1, 0, -1, 0, 1], [0, 1, 0, -1, 0], color="#ef4444", lw=2.4)
    axes[0].scatter([0], [1], color="#ef4444", s=80, zorder=3)
    axes[0].text(0.05, 1.25, "hits a corner →\nweight = 0 (sparse)", color="#ef4444", fontsize=9)
    axes[0].set_title("L1 (Lasso): diamond → zeros", color="#ef4444", fontsize=11, fontweight="bold")
    # L2 circle
    axes[1].plot(np.cos(th), np.sin(th), color=PRIMARY, lw=2.4)
    axes[1].scatter([0.78], [0.62], color=PRIMARY, s=80, zorder=3)
    axes[1].text(-1.6, 1.4, "touches smoothly →\nweights shrink (not zero)", color=PRIMARY, fontsize=9)
    axes[1].set_title("L2 (Ridge): circle → shrinks", color=PRIMARY, fontsize=11, fontweight="bold")
    fig.suptitle("Why L1 zeros weights and L2 only shrinks them", color=INK,
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ch26_l1_l2.png")


# ===========================================================================
# CHAPTER 25 — Evaluation & Metrics  (defined first; registered below)
# ===========================================================================
def crossval_diagram():
    """5-fold cross-validation: validation fold rotates."""
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    k = 5; w = 2.0
    for row in range(k):
        y = 4.8 - row * 0.9
        for col in range(k):
            x = 1.2 + col * w
            is_val = (col == row)
            fc = AMBER if is_val else "#c7d2fe"
            ax.add_patch(plt.Rectangle((x, y), w - 0.1, 0.7, fc=fc, ec="white", lw=1.5))
            if is_val:
                ax.text(x + w / 2, y + 0.35, "VAL", ha="center", va="center",
                        fontsize=8, fontweight="bold", color="white")
        ax.text(0.9, y + 0.35, f"Fold {row+1}", ha="right", va="center", fontsize=9, color=INK)
    ax.text(6.2, 5.6, "blue = train     orange = validate", ha="center", fontsize=9.5, color=INK)
    ax.text(6.2, 0.2, "average the 5 validation scores → robust estimate",
            ha="center", fontsize=9.5, style="italic", color=INK)
    ax.set_title("5-fold cross-validation", color=INK, fontsize=13, fontweight="bold", pad=2)
    save(fig, "ch25_crossval.png")


def confusion_diagram():
    """Labelled 2x2 confusion matrix."""
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    cells = [(2, 6, "True Positive\n(TP) ✓", GREEN),
             (6, 6, "False Negative\n(FN) ✗ miss", "#ef4444"),
             (2, 2, "False Positive\n(FP) ✗ alarm", "#ef4444"),
             (6, 2, "True Negative\n(TN) ✓", GREEN)]
    for x, y, lab, color in cells:
        ax.add_patch(plt.Rectangle((x, y), 4, 4, fc=color, ec="white", lw=2, alpha=0.8))
        ax.text(x + 2, y + 2, lab, ha="center", va="center", color="white",
                fontsize=11, fontweight="bold")
    ax.text(0.9, 8, "Actual: Positive", rotation=90, va="center", fontsize=9.5, color=INK)
    ax.text(0.9, 4, "Actual: Negative", rotation=90, va="center", fontsize=9.5, color=INK)
    ax.text(4, 10.3, "Predicted: Positive", ha="center", fontsize=9.5, color=INK)
    ax.text(8, 10.3, "Predicted: Negative", ha="center", fontsize=9.5, color=INK)
    ax.set_title("The confusion matrix", color=INK, fontsize=13, fontweight="bold")
    save(fig, "ch25_confusion.png")


def roc_diagram():
    """An ROC curve vs the random diagonal."""
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    fpr = np.linspace(0, 1, 200)
    tpr = fpr ** 0.25                       # a strong classifier hugging top-left
    ax.plot(fpr, tpr, color=PRIMARY, lw=2.8, label="model (AUC ≈ 0.9)")
    ax.fill_between(fpr, tpr, alpha=0.12, color=PRIMARY)
    ax.plot([0, 1], [0, 1], color="#94a3b8", ls="--", lw=1.6, label="random (AUC = 0.5)")
    ax.annotate("better →", xy=(0.12, 0.85), xytext=(0.3, 0.55),
                arrowprops=dict(arrowstyle="-|>", color=GREEN), color=GREEN, fontsize=10)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate (Recall)")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("ROC curve and AUC", color=INK, fontsize=12, fontweight="bold")
    save(fig, "ch25_roc.png")


# ===========================================================================
# CHAPTER 24 — Boosting  (defined first; registered below)
# ===========================================================================
def boosting_diagram():
    """Sequential weak learners each fixing the previous errors."""
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    labels = ["Weak\nlearner 1", "Weak learner 2\n(fix L1 errors)",
              "Weak learner 3\n(fix L1+L2 errors)"]
    xs = [0.4, 4.0, 7.6]
    for i, (x, lab) in enumerate(zip(xs, labels)):
        _box(ax, x, 2.6, 2.9, 1.3, lab, VIOLET, fs=8.8)
        if i < 2:
            _arrow(ax, x + 2.9, 3.25, xs[i + 1], 3.25)
    _box(ax, 10.6, 2.7, 1.2, 1.1, "STRONG\nmodel", GREEN, fs=8.5)
    for x in xs:
        _arrow(ax, x + 1.45, 2.6, 11.0, 2.4, color="#cbd5e1")
    ax.text(6.0, 1.2, "each model concentrates on the examples the previous ones got wrong",
            ha="center", fontsize=9.5, style="italic", color=INK)
    ax.set_title("Boosting: sequential error-correction", color=INK,
                 fontsize=13, fontweight="bold", pad=2)
    save(fig, "ch24_boosting.png")


def gradient_boosting_diagram():
    """Predictions converging to the truth as residual-fitting trees are added."""
    rng = np.random.default_rng(1)
    x = np.linspace(0, 4, 40)
    truth = np.sin(x) + 0.5 * x
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    stages = [("after 1 tree", 0.35), ("after 5 trees", 0.7), ("after 30 trees", 0.97)]
    for ax, (title, frac) in zip(axes, stages):
        pred = frac * truth + (1 - frac) * truth.mean()
        ax.scatter(x, truth, color=INK, s=14, label="true")
        ax.plot(x, pred, color="#ef4444", lw=2.4, label="model")
        ax.set_title(title, fontsize=10.5, fontweight="bold", color=VIOLET)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Gradient boosting: each tree fits the residuals → converges to truth",
                 color=INK, fontsize=12.5, fontweight="bold", y=1.04)
    fig.tight_layout()
    save(fig, "ch24_gradient_boosting.png")


# ===========================================================================
# CHAPTER 23 — Random Forest  (defined first; registered below)
# ===========================================================================
def ensemble_diagram():
    """Bagging (parallel) vs boosting (sequential)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    ax = axes[0]; ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("BAGGING (parallel)", color=SKY, fontsize=12, fontweight="bold")
    _box(ax, 4, 6.6, 2, 0.9, "DATA", INK, fs=10)
    for i, x in enumerate([0.6, 4, 7.4]):
        _box(ax, x, 3.8, 2, 0.9, f"Model {i+1}", SKY, fs=9)
        _arrow(ax, 5, 6.6, x + 1, 4.7, color="#cbd5e1")
        _arrow(ax, x + 1, 3.8, 5, 2.4, color="#cbd5e1")
    _box(ax, 3.7, 1.3, 2.6, 0.95, "AVERAGE / VOTE", GREEN, fs=9.5)
    ax.text(5, 0.6, "reduces variance", ha="center", fontsize=9, style="italic", color=INK)
    ax = axes[1]; ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("BOOSTING (sequential)", color=VIOLET, fontsize=12, fontweight="bold")
    xs = [0.6, 3.7, 6.8]
    for i, x in enumerate(xs):
        _box(ax, x, 4.2, 2.4, 1.1, f"Model {i+1}\n(fix errors)", VIOLET, fs=8.5)
        if i < 2:
            _arrow(ax, x + 2.4, 4.75, xs[i+1], 4.75)
    _box(ax, 3.4, 1.6, 2.8, 0.95, "WEIGHTED SUM", GREEN, fs=9.5)
    for x in xs:
        _arrow(ax, x + 1.2, 4.2, 4.8, 2.55, color="#cbd5e1")
    ax.text(5, 0.8, "reduces bias", ha="center", fontsize=9, style="italic", color=INK)
    fig.suptitle("Two ensemble strategies", color=INK, fontsize=13, fontweight="bold", y=1.04)
    fig.tight_layout()
    save(fig, "ch23_ensemble.png")


def random_forest_diagram():
    """Many trees vote on the final prediction."""
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    _box(ax, 4.7, 5.0, 2.6, 0.9, "Training data", INK, fs=10)
    votes = ["Class A", "Class A", "Class B", "Class A"]
    for i in range(4):
        x = 0.4 + i * 2.9
        _box(ax, x, 2.8, 2.3, 1.2, f"Tree {i+1}\n(random rows\n& features)", GREEN, fs=8.3)
        _arrow(ax, 6.0, 5.0, x + 1.15, 4.05, color="#cbd5e1")
        _box(ax, x + 0.35, 1.5, 1.6, 0.7, votes[i], SKY if votes[i].endswith("A") else "#ef4444", fs=8.5)
        _arrow(ax, x + 1.15, 2.8, x + 1.15, 2.25, color="#cbd5e1")
    ax.text(6.0, 0.55, "Majority vote  →  Class A", ha="center", fontsize=12,
            fontweight="bold", color=VIOLET)
    ax.set_title("Random Forest: diverse trees vote", color=INK, fontsize=13,
                 fontweight="bold", pad=2)
    save(fig, "ch23_random_forest.png")


# ===========================================================================
# CHAPTER 22 — SVM  (defined first; registered below)
# ===========================================================================
def svm_margin_diagram():
    """Max-margin separating line with margins and support vectors."""
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    # two clusters
    blue = np.array([[1.5, 4], [2, 5.2], [1.2, 6], [2.6, 6.4], [1.8, 7]])
    red = np.array([[5, 1.5], [5.6, 2.6], [6.2, 1.2], [6.6, 2.8], [5.2, 3.2]])
    ax.scatter(blue[:, 0], blue[:, 1], color=SKY, s=70, zorder=3)
    ax.scatter(red[:, 0], red[:, 1], color="#ef4444", s=70, zorder=3)
    xs = np.linspace(0, 8, 10)
    ax.plot(xs, -xs + 7.5, color=INK, lw=2.4, label="decision boundary")        # w·x+b=0
    ax.plot(xs, -xs + 8.8, color="#94a3b8", lw=1.4, ls="--")                     # margin
    ax.plot(xs, -xs + 6.2, color="#94a3b8", lw=1.4, ls="--")
    # support vectors (closest to boundary)
    sv = np.array([[2.6, 6.4], [5.2, 3.2], [5, 1.5]])
    for s in sv:
        ax.scatter(*s, s=240, facecolors="none", edgecolors=VIOLET, lw=2.4, zorder=4)
    ax.annotate("support\nvectors", xy=(5.2, 3.2), xytext=(6.6, 5.0),
                arrowprops=dict(arrowstyle="-|>", color=VIOLET), color=VIOLET, fontsize=10)
    ax.annotate("margin", xy=(3.5, 4.3), xytext=(3.8, 6.6),
                arrowprops=dict(arrowstyle="-|>", color="#64748b"), color="#64748b", fontsize=10)
    ax.set_xlim(0, 8.5); ax.set_ylim(0, 8.5); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("SVM: the maximum-margin boundary", color=INK, fontsize=12, fontweight="bold")
    save(fig, "ch22_margin.png")


def kernel_trick_diagram():
    """Circles not linearly separable in 2D -> separable when lifted by radius."""
    rng = np.random.default_rng(0)
    t = rng.uniform(0, 2 * np.pi, 60)
    r_in = rng.uniform(0, 1.2, 60); r_out = rng.uniform(2.2, 3.2, 60)
    inner = np.c_[r_in * np.cos(t), r_in * np.sin(t)]
    outer = np.c_[r_out * np.cos(t), r_out * np.sin(t)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    axes[0].scatter(inner[:, 0], inner[:, 1], color="#ef4444", s=18)
    axes[0].scatter(outer[:, 0], outer[:, 1], color=SKY, s=18)
    axes[0].set_title("2-D: no straight line works", color=INK, fontsize=11, fontweight="bold")
    axes[0].set_xticks([]); axes[0].set_yticks([])
    # lifted: z = radius
    axes[1].scatter(np.hypot(inner[:, 0], inner[:, 1]), rng.uniform(0, 1, 60), color="#ef4444", s=18)
    axes[1].scatter(np.hypot(outer[:, 0], outer[:, 1]), rng.uniform(0, 1, 60), color=SKY, s=18)
    axes[1].axvline(1.8, color=INK, lw=2.2, ls="--")
    axes[1].set_title("Lifted (z = radius): a line separates them!", color=INK, fontsize=11, fontweight="bold")
    axes[1].set_xlabel("distance from centre"); axes[1].set_yticks([])
    fig.suptitle("The kernel trick: lift to a higher dimension to separate",
                 color=INK, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ch22_kernel.png")


# ===========================================================================
# CHAPTER 21 — Decision Trees  (defined first; registered below)
# ===========================================================================
def decision_tree_diagram():
    """A real, rendered decision tree on iris (depth 3)."""
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    iris = load_iris()
    clf = DecisionTreeClassifier(max_depth=3, random_state=1).fit(iris.data, iris.target)
    fig, ax = plt.subplots(figsize=(11, 5.6))
    plot_tree(clf, feature_names=iris.feature_names, class_names=list(iris.target_names),
              filled=True, rounded=True, fontsize=8, ax=ax, impurity=True)
    ax.set_title("A decision tree (iris, max_depth=3) — every decision is visible",
                 color=INK, fontsize=13, fontweight="bold")
    save(fig, "ch21_tree.png")


def depth_overfit_diagram():
    """Train vs test accuracy as tree depth increases."""
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    X, y = load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    depths = range(1, 11); tr, te = [], []
    for d in depths:
        m = DecisionTreeClassifier(max_depth=d, random_state=1).fit(Xtr, ytr)
        tr.append(accuracy_score(ytr, m.predict(Xtr)))
        te.append(accuracy_score(yte, m.predict(Xte)))
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(list(depths), tr, "o-", color="#ef4444", lw=2, label="training accuracy")
    ax.plot(list(depths), te, "s-", color=PRIMARY, lw=2, label="test accuracy")
    ax.fill_between(list(depths), te, tr, color="#fca5a5", alpha=0.25)
    ax.text(7, 0.86, "gap = overfitting", color="#ef4444", fontsize=10)
    ax.set_xlabel("max_depth"); ax.set_ylabel("accuracy"); ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    ax.set_title("Tree depth vs overfitting", color=INK, fontsize=12, fontweight="bold")
    save(fig, "ch21_depth_overfit.png")


# ===========================================================================
# CHAPTER 20 — Naive Bayes  (defined first; registered below)
# ===========================================================================
def naive_bayes_diagram():
    """Concept: features -> per-class score (prior x product of likelihoods) -> pick max."""
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    _box(ax, 0.3, 2.4, 2.4, 1.4, "Email words:\n'free', 'money'", SKY, fs=9.5)
    _box(ax, 3.5, 3.5, 4.2, 1.1, "P(spam) × P(free|spam) × P(money|spam)", VIOLET, fs=8.5)
    _box(ax, 3.5, 1.3, 4.2, 1.1, "P(ham) × P(free|ham) × P(money|ham)", SKY, fs=8.5)
    _box(ax, 8.4, 2.4, 3.3, 1.4, "Pick the class\nwith the HIGHEST\nscore  →  SPAM", GREEN, fs=9.5)
    _arrow(ax, 2.7, 3.4, 3.5, 4.0); _arrow(ax, 2.7, 2.8, 3.5, 1.9)
    _arrow(ax, 7.7, 4.0, 8.4, 3.4); _arrow(ax, 7.7, 1.9, 8.4, 2.6)
    ax.text(6.0, 0.55, "“naive” = assume the words are independent given the class",
            ha="center", fontsize=9.5, style="italic", color=INK)
    ax.set_title("Naive Bayes: prior × product of likelihoods, pick the max",
                 color=INK, fontsize=12.5, fontweight="bold", pad=2)
    save(fig, "ch20_naive_bayes.png")


# ===========================================================================
# CHAPTER 19 — KNN  (defined first; registered below)
# ===========================================================================
def knn_vote_diagram():
    """A new point classified by majority vote of k=3 nearest neighbours."""
    fig, ax = plt.subplots(figsize=(7, 5.4))
    rng = np.random.default_rng(3)
    red = rng.normal([3, 7], 1.0, (8, 2))
    blue = rng.normal([7, 4], 1.0, (8, 2))
    ax.scatter(red[:, 0], red[:, 1], color="#ef4444", s=70, label="Class A")
    ax.scatter(blue[:, 0], blue[:, 1], color=SKY, s=70, label="Class B")
    new = np.array([5.2, 5.5])
    ax.scatter(*new, color=VIOLET, marker="*", s=420, edgecolor="white",
               zorder=5, label="new point")
    # find 3 nearest overall
    allp = np.vstack([red, blue]); labels = ["A"] * 8 + ["B"] * 8
    d = np.linalg.norm(allp - new, axis=1); idx = np.argsort(d)[:3]
    r = d[idx].max()
    ax.add_patch(Circle(new, r, fill=False, ec=VIOLET, ls="--", lw=2))
    for i in idx:
        ax.plot([new[0], allp[i, 0]], [new[1], allp[i, 1]], color="#94a3b8", lw=1.2, zorder=1)
    ax.text(new[0], new[1] + r + 0.4, "k = 3 nearest decide", ha="center",
            color=VIOLET, fontsize=10, fontweight="bold")
    ax.legend(fontsize=9, loc="lower left"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("KNN: classify by majority vote of nearest neighbours",
                 color=INK, fontsize=12, fontweight="bold")
    save(fig, "ch19_knn_vote.png")


def k_effect_diagram():
    """Decision boundary for small vs large k."""
    from sklearn.neighbors import KNeighborsClassifier
    rng = np.random.default_rng(5)
    n = 40
    Xa = rng.normal([2, 2], 1.1, (n, 2)); Xb = rng.normal([4.5, 4.5], 1.1, (n, 2))
    X = np.vstack([Xa, Xb]); y = np.array([0] * n + [1] * n)
    xx, yy = np.meshgrid(np.linspace(-1, 8, 200), np.linspace(-1, 8, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, k, title in [(axes[0], 1, "k = 1  (overfit, jagged)"),
                         (axes[1], 25, "k = 25  (smooth, may underfit)")]:
        m = KNeighborsClassifier(k).fit(X, y)
        Z = m.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
        ax.scatter(Xa[:, 0], Xa[:, 1], c="#2563eb", s=12)
        ax.scatter(Xb[:, 0], Xb[:, 1], c="#ef4444", s=12)
        ax.set_title(title, fontsize=10.5, fontweight="bold", color=INK)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("The effect of k on the decision boundary", color=INK,
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ch19_k_effect.png")


# ===========================================================================
# CHAPTER 18 — Logistic Regression  (defined first; registered below)
# ===========================================================================
def sigmoid_diagram():
    """The sigmoid curve with the 0.5 threshold marked."""
    z = np.linspace(-8, 8, 300)
    p = 1 / (1 + np.exp(-z))
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(z, p, color=PRIMARY, lw=2.8)
    ax.axhline(0.5, color="#94a3b8", ls="--", lw=1.2)
    ax.axvline(0, color="#94a3b8", ls="--", lw=1.2)
    ax.scatter([0], [0.5], color="#ef4444", s=60, zorder=3)
    ax.annotate("threshold = 0.5\n(z = 0)", xy=(0, 0.5), xytext=(1.5, 0.32),
                arrowprops=dict(arrowstyle="-|>", color="#ef4444"), color="#ef4444", fontsize=9.5)
    ax.text(-7, 0.08, "predict 0", color=SKY, fontsize=11, fontweight="bold")
    ax.text(4.5, 0.9, "predict 1", color="#ef4444", fontsize=11, fontweight="bold")
    ax.set_xlabel("z = w·x + b"); ax.set_ylabel("probability  σ(z)")
    ax.set_ylim(-0.05, 1.05)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_title("The sigmoid: turning a score into a probability", color=INK,
                 fontsize=12, fontweight="bold")
    save(fig, "ch18_sigmoid.png")


# ===========================================================================
# CHAPTER 17 — Linear Regression  (defined first; registered below)
# ===========================================================================
def best_fit_diagram():
    """Scatter with best-fit line and residuals drawn."""
    rng = np.random.default_rng(8)
    x = np.linspace(1, 10, 14)
    y = 1.3 * x + 2 + rng.normal(0, 1.6, len(x))
    coef = np.polyfit(x, y, 1); yhat = np.polyval(coef, x)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for xi, yi, yh in zip(x, y, yhat):           # residual lines
        ax.plot([xi, xi], [yi, yh], color="#ef4444", lw=1.3, zorder=1)
    ax.scatter(x, y, color=PRIMARY, s=42, zorder=3, label="data points")
    ax.plot(x, yhat, color=INK, lw=2.4, zorder=2, label="best-fit line  ŷ = wx + b")
    ax.plot([], [], color="#ef4444", label="residuals (errors)")
    ax.legend(fontsize=9); ax.set_xlabel("feature x"); ax.set_ylabel("target y")
    ax.set_title("Linear regression: the best-fit line minimises squared residuals",
                 color=INK, fontsize=12, fontweight="bold")
    ax.grid(alpha=0.2)
    save(fig, "ch17_best_fit.png")


# ===========================================================================
# CHAPTER 16 — Supervised Overview  (defined first; registered below)
# ===========================================================================
def supervised_flow_diagram():
    """Labelled data -> train -> model -> predict -> evaluate."""
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    _box(ax, 0.3, 2.6, 2.5, 1.5, "LABELLED\nTRAINING DATA\n(X, y)", SKY, fs=9)
    _box(ax, 3.4, 2.6, 2.3, 1.5, "TRAIN\nthe model", PRIMARY, fs=10)
    _box(ax, 6.3, 2.6, 2.3, 1.5, "TRAINED\nMODEL f(X)", VIOLET, fs=10)
    _box(ax, 9.2, 2.6, 2.5, 1.5, "PREDICT y\nfor new X", GREEN, fs=10)
    _arrow(ax, 2.8, 3.35, 3.4, 3.35); _arrow(ax, 5.7, 3.35, 6.3, 3.35)
    _arrow(ax, 8.6, 3.35, 9.2, 3.35)
    _box(ax, 6.3, 0.5, 2.3, 1.0, "EVALUATE\n(vs known y)", AMBER, fs=9.5)
    _arrow(ax, 10.4, 2.6, 8.0, 1.5, color="#cbd5e1")
    ax.set_title("The supervised learning workflow", color=INK, fontsize=13,
                 fontweight="bold", pad=2)
    save(fig, "ch16_supervised_flow.png")


def decision_boundaries_diagram():
    """Decision boundaries: linear, tree (boxy), KNN (wiggly)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    rng = np.random.default_rng(6)
    n = 60
    Xa = rng.normal([2, 2], 0.9, (n, 2)); Xb = rng.normal([4, 4], 0.9, (n, 2))
    X = np.vstack([Xa, Xb]); y = np.array([0] * n + [1] * n)
    xx, yy = np.meshgrid(np.linspace(-1, 7, 200), np.linspace(-1, 7, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    models = [("Linear (Logistic)", LogisticRegression()),
              ("Decision Tree (boxy)", DecisionTreeClassifier(max_depth=6, random_state=0)),
              ("KNN (wiggly)", KNeighborsClassifier(5))]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, (title, m) in zip(axes, models):
        m.fit(X, y)
        Z = m.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
        ax.scatter(Xa[:, 0], Xa[:, 1], c="#2563eb", s=10)
        ax.scatter(Xb[:, 0], Xb[:, 1], c="#ef4444", s=10)
        ax.set_title(title, fontsize=10.5, fontweight="bold", color=INK)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Different models, different decision boundaries", color=INK,
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "ch16_decision_boundaries.png")


def bias_variance_diagram():
    """Classic bias-variance vs complexity curves."""
    fig, ax = plt.subplots(figsize=(8, 4.4))
    c = np.linspace(0, 10, 200)
    bias2 = 6 * np.exp(-c / 2.2)
    variance = 0.25 * c ** 1.5 / 3
    total = bias2 + variance + 0.4
    ax.plot(c, bias2, color=AMBER, lw=2.2, label="Bias²  (underfitting)")
    ax.plot(c, variance, color="#ef4444", lw=2.2, label="Variance (overfitting)")
    ax.plot(c, total, color=PRIMARY, lw=2.8, label="Total error")
    opt = c[np.argmin(total)]
    ax.axvline(opt, color=GREEN, ls="--", lw=1.6)
    ax.scatter([opt], [total.min()], color=GREEN, s=70, zorder=3)
    ax.annotate("sweet spot", xy=(opt, total.min()), xytext=(opt + 1, total.min() + 1.5),
                arrowprops=dict(arrowstyle="-|>", color=GREEN), color=GREEN, fontsize=10)
    ax.set_xlabel("model complexity →"); ax.set_ylabel("error")
    ax.set_yticks([]); ax.legend(fontsize=9)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_title("The bias–variance trade-off", color=INK, fontsize=13, fontweight="bold")
    save(fig, "ch16_bias_variance.png")


# ===========================================================================
# CHAPTER 15 — EDA  (defined first; registered below)
# ===========================================================================
def eda_workflow_diagram():
    """Six-step EDA workflow as a flow with a loop-back."""
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    steps = [
        ("1. Understand\nstructure", SKY), ("2. Univariate\nanalysis", SKY),
        ("3. Relationships\n(bi/multivariate)", VIOLET), ("4. Data-quality\ncheck", VIOLET),
        ("5. Target\nanalysis", PRIMARY), ("6. Insights &\nhypotheses", GREEN),
    ]
    w = 1.75
    xs = [0.2 + i * 1.96 for i in range(6)]
    for (label, color), x in zip(steps, xs):
        _box(ax, x, 2.3, w, 1.4, label, color, fs=9)
        if x != xs[-1]:
            _arrow(ax, x + w, 3.0, x + 1.96, 3.0)
    ax.annotate("", xy=(xs[0] + w / 2, 2.3), xytext=(xs[-1] + w / 2, 2.3),
                arrowprops=dict(arrowstyle="-|>", color="#ef4444", lw=1.6, ls="--",
                                connectionstyle="arc3,rad=0.25"))
    ax.text(6.0, 0.9, "loop back as new questions arise", ha="center",
            color="#ef4444", fontsize=9.5, style="italic")
    ax.set_title("The Exploratory Data Analysis workflow", color=INK,
                 fontsize=13, fontweight="bold", pad=2)
    save(fig, "ch15_eda_workflow.png")


# ===========================================================================
# CHAPTER 14 — Data Visualization  (defined first; registered below)
# ===========================================================================
def chart_gallery_diagram():
    """A 2x3 gallery of the six essential charts (real plots)."""
    rng = np.random.default_rng(11)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
    # line
    ax = axes[0, 0]; ax.plot(range(1, 8), [12, 15, 14, 18, 22, 20, 26], marker="o", color=PRIMARY)
    ax.set_title("LINE — trend over time", fontsize=10, fontweight="bold", color=PRIMARY)
    # bar
    ax = axes[0, 1]; ax.bar(["A", "B", "C", "D"], [23, 17, 35, 12], color=SKY)
    ax.set_title("BAR — compare categories", fontsize=10, fontweight="bold", color=SKY)
    # histogram
    ax = axes[0, 2]; ax.hist(rng.normal(0, 1, 500), bins=20, color=VIOLET, edgecolor="white")
    ax.set_title("HISTOGRAM — distribution", fontsize=10, fontweight="bold", color=VIOLET)
    # box
    ax = axes[1, 0]
    bp = ax.boxplot([rng.normal(50, 8, 80), rng.normal(60, 12, 80)], patch_artist=True, labels=["X", "Y"])
    for b in bp["boxes"]:
        b.set(facecolor="#bbf7d0", edgecolor=GREEN)
    ax.set_title("BOX — spread & outliers", fontsize=10, fontweight="bold", color=GREEN)
    # scatter
    ax = axes[1, 1]; xx = rng.normal(0, 1, 80); ax.scatter(xx, xx + rng.normal(0, 0.5, 80), color=AMBER, s=14)
    ax.set_title("SCATTER — relationship", fontsize=10, fontweight="bold", color=AMBER)
    # heatmap
    ax = axes[1, 2]; mat = rng.uniform(-1, 1, (4, 4)); np.fill_diagonal(mat, 1)
    im = ax.imshow(mat, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("HEATMAP — correlations", fontsize=10, fontweight="bold", color="#ef4444")
    ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel():
        if ax is not axes[1, 2]:
            ax.tick_params(labelsize=7)
    fig.suptitle("The six essential charts", color=INK, fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    save(fig, "ch14_gallery.png")


def chart_chooser_diagram():
    """Decision guide: question -> chart type."""
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    _box(ax, 4.7, 5.0, 2.6, 0.9, "What's your question?", INK, fs=10)
    items = [
        (0.2, "Trend over\nTIME?", "LINE", SKY),
        (2.5, "Compare\nCATEGORIES?", "BAR", VIOLET),
        (4.8, "ONE variable's\ndistribution?", "HISTOGRAM", GREEN),
        (7.1, "TWO numeric\nvariables?", "SCATTER", AMBER),
        (9.4, "MATRIX of\nvalues?", "HEATMAP", "#ef4444"),
    ]
    for x, q, chart, color in items:
        _box(ax, x, 2.7, 2.2, 1.3, q, "white", ec=color, tc=color, fs=8.6, bold=False)
        _box(ax, x, 0.9, 2.2, 0.95, chart, color, fs=9.5)
        _arrow(ax, 6.0, 5.0, x + 1.1, 4.05, color="#cbd5e1")
        _arrow(ax, x + 1.1, 2.7, x + 1.1, 1.9, color="#cbd5e1")
    ax.set_title("Which chart should I use?", color=INK, fontsize=13, fontweight="bold", pad=2)
    save(fig, "ch14_chooser.png")


# ===========================================================================
# CHAPTER 13 — Feature Selection  (defined first; registered below)
# ===========================================================================
def curse_diagram():
    """Performance vs number of features: rises then falls (sweet spot)."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    n = np.linspace(1, 30, 300)
    perf = 1 - np.exp(-n / 4) - 0.02 * np.maximum(0, n - 8)
    ax.plot(n, perf, color=PRIMARY, lw=2.6)
    peak = n[np.argmax(perf)]
    ax.axvline(peak, color=GREEN, ls="--", lw=1.6)
    ax.scatter([peak], [perf.max()], color=GREEN, s=70, zorder=3)
    ax.annotate("sweet spot", xy=(peak, perf.max()), xytext=(peak + 4, perf.max() - 0.05),
                arrowprops=dict(arrowstyle="-|>", color=GREEN), color=GREEN, fontsize=10)
    ax.text(2.5, 0.25, "too few:\nunderfit", color=AMBER, fontsize=9.5)
    ax.text(22, 0.45, "too many:\nnoise & overfit", color="#ef4444", fontsize=9.5)
    ax.set_xlabel("number of features"); ax.set_ylabel("model performance")
    ax.set_yticks([])
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_title("The curse of dimensionality", color=INK, fontsize=13, fontweight="bold")
    save(fig, "ch13_curse.png")


def selection_methods_diagram():
    """Three families of feature selection."""
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    cards = [
        (0.3, SKY, "FILTER", "Statistical scores,\nbefore any model.\nFast, model-agnostic.\ne.g. F-test, correlation"),
        (4.1, VIOLET, "WRAPPER", "Search subsets via\nmodel performance.\nAccurate but slow.\ne.g. RFE"),
        (7.9, GREEN, "EMBEDDED", "Selection during\nmodel training.\nEfficient, model-aware.\ne.g. Lasso, tree importance"),
    ]
    for x, color, title, desc in cards:
        _box(ax, x, 0.6, 3.6, 3.6, "", color)
        ax.text(x + 1.8, 3.5, title, ha="center", color="white", fontsize=13, fontweight="bold")
        ax.text(x + 1.8, 2.0, desc, ha="center", color="white", fontsize=9.3)
    ax.set_title("Three families of feature selection", color=INK, fontsize=13,
                 fontweight="bold", pad=2)
    save(fig, "ch13_methods.png")


# ===========================================================================
# CHAPTER 12 — Feature Engineering  (defined first; registered below)
# ===========================================================================
def fe_overview_diagram():
    """Raw data -> feature engineering -> features -> model."""
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    _box(ax, 0.4, 1.7, 2.4, 1.6, "RAW DATA\nheight, weight,\ndate, income", "#94a3b8", fs=9.5)
    _box(ax, 3.7, 1.7, 2.7, 1.6, "FEATURE\nENGINEERING\n(create signal)", PRIMARY, fs=10)
    _box(ax, 7.3, 1.7, 2.5, 1.6, "FEATURES\nBMI, weekday,\nlog-income", GREEN, fs=9.5)
    _box(ax, 10.5, 2.0, 1.3, 1.0, "MODEL", VIOLET, fs=10)
    _arrow(ax, 2.8, 2.5, 3.7, 2.5); _arrow(ax, 6.4, 2.5, 7.3, 2.5)
    _arrow(ax, 9.8, 2.5, 10.5, 2.5)
    ax.text(6.0, 0.7, "“Better features beat better algorithms.”",
            ha="center", fontsize=11, style="italic", color=INK)
    ax.set_title("Feature engineering turns raw data into signal", color=INK,
                 fontsize=13, fontweight="bold", pad=2)
    save(fig, "ch12_fe_overview.png")


def binning_diagram():
    """Continuous BMI axis split into labelled bins."""
    fig, ax = plt.subplots(figsize=(10, 3.2))
    edges = [15, 18.5, 25, 30, 40]
    labels = ["underweight", "normal", "overweight", "obese"]
    colors = [SKY, GREEN, AMBER, "#ef4444"]
    for i in range(len(labels)):
        ax.axvspan(edges[i], edges[i + 1], color=colors[i], alpha=0.35)
        ax.text((edges[i] + edges[i + 1]) / 2, 0.5, labels[i], ha="center",
                va="center", fontsize=10, fontweight="bold", color=INK)
        ax.text(edges[i], -0.25, str(edges[i]), ha="center", fontsize=9, color="#475569")
    ax.text(edges[-1], -0.25, str(edges[-1]), ha="center", fontsize=9, color="#475569")
    # sample points
    pts = [22.5, 27.8, 21.5, 26.1]
    ax.scatter(pts, [0.85] * len(pts), color=INK, s=45, zorder=3)
    ax.set_xlim(15, 40); ax.set_ylim(0, 1.1); ax.set_yticks([])
    ax.set_xlabel("BMI (continuous)")
    ax.set_title("Binning: continuous values → labelled categories", color=INK,
                 fontsize=12, fontweight="bold")
    save(fig, "ch12_binning.png")


# ===========================================================================
# CHAPTER 11 — Data Preprocessing  (defined first; registered below)
# ===========================================================================
def scaling_diagram():
    """Original vs standardized vs normalized — same shape, different axes."""
    rng = np.random.default_rng(2)
    x = rng.normal(35, 7, 60)            # age-like
    y = rng.normal(70000, 12000, 60)     # salary-like (huge scale)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7))
    axes[0].scatter(x, y, color=SKY, s=18)
    axes[0].set_title("ORIGINAL\n(wildly different scales)", color=SKY, fontsize=10, fontweight="bold")
    xs = (x - x.mean()) / x.std(); ys = (y - y.mean()) / y.std()
    axes[1].scatter(xs, ys, color=PRIMARY, s=18)
    axes[1].set_title("STANDARDIZED\n(mean 0, std 1)", color=PRIMARY, fontsize=10, fontweight="bold")
    xn = (x - x.min()) / (x.max() - x.min()); yn = (y - y.min()) / (y.max() - y.min())
    axes[2].scatter(xn, yn, color=GREEN, s=18)
    axes[2].set_title("NORMALIZED\n(range [0, 1])", color=GREEN, fontsize=10, fontweight="bold")
    for ax in axes:
        ax.set_xlabel("age"); ax.set_ylabel("salary")
    fig.suptitle("Scaling changes the axis range, not the shape", color=INK,
                 fontsize=13, fontweight="bold", y=1.05)
    fig.tight_layout()
    save(fig, "ch11_scaling.png")


def onehot_diagram():
    """Show one nominal column becoming several 0/1 columns."""
    fig, ax = plt.subplots(figsize=(10, 4.0))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    cities = ["Lahore", "Karachi", "Multan"]
    # before
    ax.text(1.5, 5.4, "city", ha="center", fontsize=11, fontweight="bold", color=VIOLET)
    for i, c in enumerate(cities):
        ax.add_patch(plt.Rectangle((0.5, 4.4 - i), 2.0, 0.9, fc="white", ec=VIOLET, lw=1.4))
        ax.text(1.5, 4.85 - i, c, ha="center", va="center", fontsize=10, color=INK)
    _arrow(ax, 2.8, 3.0, 4.4, 3.0, color=INK)
    ax.text(3.6, 3.4, "one-hot", ha="center", fontsize=9.5, style="italic", color=INK)
    # after
    cols = ["Karachi", "Lahore", "Multan"]
    rows = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
    x0 = 5.0
    for j, col in enumerate(cols):
        ax.text(x0 + j * 1.7 + 0.85, 5.4, f"city_{col}", ha="center", fontsize=8.5,
                fontweight="bold", color=GREEN, rotation=0)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            fc = "#bbf7d0" if val else "white"
            ax.add_patch(plt.Rectangle((x0 + j * 1.7, 4.4 - i), 1.7, 0.9, fc=fc, ec=GREEN, lw=1.2))
            ax.text(x0 + j * 1.7 + 0.85, 4.85 - i, str(val), ha="center", va="center",
                    fontsize=10, color=INK)
    ax.text(6.0, 0.9, "each category becomes its own 0/1 column — no false ordering",
            ha="center", fontsize=10, style="italic", color=INK)
    ax.set_title("One-hot encoding", color=INK, fontsize=13, fontweight="bold", pad=2)
    save(fig, "ch11_onehot.png")


# ===========================================================================
# CHAPTER 10 — Data Cleaning  (defined first; registered below)
# ===========================================================================
def missing_strategies_diagram():
    """Decision flow for handling missing values."""
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")
    _box(ax, 4.6, 5.7, 2.8, 1.0, "Missing values?", INK, fs=11)
    _box(ax, 0.4, 3.4, 2.9, 1.1, "Most of column\nmissing? -> DROP column", "#ef4444", fs=9)
    _box(ax, 3.7, 3.4, 2.7, 1.1, "Only a few rows?\n-> DROP rows", AMBER, fs=9)
    _box(ax, 6.8, 3.4, 4.6, 1.1, "Otherwise -> IMPUTE (fill in)", GREEN, fs=10)
    _arrow(ax, 5.4, 5.7, 1.8, 4.55, color="#94a3b8")
    _arrow(ax, 5.8, 5.7, 5.0, 4.55, color="#94a3b8")
    _arrow(ax, 6.4, 5.7, 9.0, 4.55, color="#94a3b8")
    imp = [(6.8, "Mean\n(symmetric)", SKY), (8.1, "Median\n(skewed)", VIOLET),
           (9.4, "Mode\n(categorical)", PRIMARY), (10.6, "Model\n(KNN)", "#0891b2")]
    for x, lab, color in imp:
        _box(ax, x, 1.4, 1.25, 1.2, lab, "white", ec=color, tc=color, fs=8, bold=False)
        _arrow(ax, 9.1, 3.4, x + 0.6, 2.65, color="#cbd5e1")
    ax.set_title("Strategies for missing data", color=INK, fontsize=13,
                 fontweight="bold", pad=2)
    save(fig, "ch10_missing_strategies.png")


def outliers_diagram():
    """Box plot illustrating IQR and an outlier."""
    rng = np.random.default_rng(5)
    data = np.concatenate([rng.normal(50, 8, 60), [95]])   # one clear outlier
    fig, ax = plt.subplots(figsize=(9, 3.6))
    bp = ax.boxplot(data, vert=False, widths=0.5, patch_artist=True,
                    flierprops=dict(marker="o", markerfacecolor="#ef4444",
                                    markersize=9, markeredgecolor="#ef4444"))
    bp["boxes"][0].set(facecolor="#c7d2fe", edgecolor=PRIMARY, linewidth=2)
    for med in bp["medians"]:
        med.set(color=VIOLET, linewidth=2.5)
    for w in bp["whiskers"] + bp["caps"]:
        w.set(color=PRIMARY, linewidth=1.6)
    ax.annotate("outlier\n(beyond 1.5×IQR)", xy=(95, 1), xytext=(80, 1.32),
                arrowprops=dict(arrowstyle="-|>", color="#ef4444"), color="#ef4444",
                fontsize=9.5, ha="center")
    ax.annotate("box = Q1 to Q3 (the IQR)", xy=(50, 0.75), xytext=(40, 0.55),
                color=PRIMARY, fontsize=9.5)
    ax.annotate("median", xy=(50, 1.25), xytext=(52, 1.34), color=VIOLET, fontsize=9.5)
    ax.set_yticks([]); ax.set_xlabel("value")
    ax.set_title("Detecting outliers with a box plot (IQR method)", color=INK,
                 fontsize=12, fontweight="bold")
    save(fig, "ch10_outliers.png")


# ===========================================================================
# CHAPTER 9 — Data Analysis  (defined first; registered below)
# ===========================================================================
def data_types_diagram():
    """Tree of data types: numerical/categorical and their sub-types."""
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")
    _box(ax, 4.7, 5.8, 2.6, 0.9, "DATA", INK, fs=12)
    _box(ax, 1.8, 4.0, 2.6, 0.9, "NUMERICAL", SKY, fs=11)
    _box(ax, 7.6, 4.0, 2.6, 0.9, "CATEGORICAL", VIOLET, fs=11)
    _arrow(ax, 5.5, 5.8, 3.1, 4.9, color="#94a3b8")
    _arrow(ax, 6.5, 5.8, 8.9, 4.9, color="#94a3b8")
    leaves = [
        (0.3, SKY, "DISCRETE", "counts:\n# children"),
        (3.3, SKY, "CONTINUOUS", "any value:\nheight, price"),
        (6.3, VIOLET, "NOMINAL", "no order:\ncity, colour"),
        (9.3, VIOLET, "ORDINAL", "ordered:\nlow<med<high"),
    ]
    parents = [3.1, 3.1, 8.9, 8.9]
    for (x, color, title, desc), px in zip(leaves, parents):
        _box(ax, x, 1.9, 2.3, 1.5, "", "white", ec=color, tc=color)
        ax.text(x + 1.15, 2.95, title, ha="center", color=color, fontsize=10, fontweight="bold")
        ax.text(x + 1.15, 2.3, desc, ha="center", color=INK, fontsize=8.6)
        _arrow(ax, px, 4.0, x + 1.15, 3.45, color="#cbd5e1")
    ax.set_title("Types of data", color=INK, fontsize=13, fontweight="bold", pad=2)
    save(fig, "ch09_data_types.png")


def analytics_levels_diagram():
    """Ascending staircase of the four analytics levels."""
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")
    steps = [
        ("DESCRIPTIVE", "What happened?", SKY),
        ("DIAGNOSTIC", "Why did it happen?", GREEN),
        ("PREDICTIVE", "What will happen?\n(Machine Learning)", PRIMARY),
        ("PRESCRIPTIVE", "What should we do?", VIOLET),
    ]
    for i, (title, q, color) in enumerate(steps):
        x = 0.4 + i * 2.9
        y = 0.6 + i * 1.4
        _box(ax, x, y, 2.7, 1.2, "", color)
        ax.text(x + 1.35, y + 0.78, title, ha="center", color="white", fontsize=10.5, fontweight="bold")
        ax.text(x + 1.35, y + 0.32, q, ha="center", color="white", fontsize=8.3)
    ax.annotate("", xy=(11.4, 6.4), xytext=(0.2, 0.5),
                arrowprops=dict(arrowstyle="-|>", color="#cbd5e1", lw=2, ls=":"))
    ax.text(0.3, 6.3, "more value →", color="#64748b", fontsize=10, style="italic")
    ax.set_title("The analytics ladder (ML lives at 'predictive')", color=INK,
                 fontsize=13, fontweight="bold", pad=2)
    save(fig, "ch09_analytics_levels.png")


# ===========================================================================
# CHAPTER 8 — NumPy & Pandas  (defined first; registered below)
# ===========================================================================
def dataframe_diagram():
    """Anatomy of a Pandas DataFrame: index, columns, values."""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")
    cols = ["name", "age", "city", "score"]
    data = [["Ali", 21, "Lahore", 72], ["Sara", 22, "Karachi", 88],
            ["Omar", 20, "Lahore", 56]]
    x0, y0, cw, rh = 2.0, 5.2, 1.9, 0.8
    # column headers
    for c, name in enumerate(cols):
        ax.add_patch(plt.Rectangle((x0 + c * cw, y0), cw, rh, fc="#eef0fb", ec=PRIMARY, lw=1.4))
        ax.text(x0 + c * cw + cw / 2, y0 + rh / 2, name, ha="center", va="center",
                fontsize=10, fontweight="bold", color=PRIMARY)
    # index + values
    for r, row in enumerate(data):
        ax.add_patch(plt.Rectangle((x0 - cw * 0.55, y0 - (r + 1) * rh), cw * 0.55, rh,
                                   fc="#f1f5f9", ec="#94a3b8", lw=1.2))
        ax.text(x0 - cw * 0.27, y0 - (r + 1) * rh + rh / 2, str(r), ha="center",
                va="center", fontsize=10, color="#475569", fontweight="bold")
        for c, val in enumerate(row):
            ax.add_patch(plt.Rectangle((x0 + c * cw, y0 - (r + 1) * rh), cw, rh,
                                       fc="white", ec="#cbd5e1", lw=1.1))
            ax.text(x0 + c * cw + cw / 2, y0 - (r + 1) * rh + rh / 2, str(val),
                    ha="center", va="center", fontsize=9.5, color=INK)
    ax.annotate("columns (named)", xy=(x0 + 2 * cw, y0 + rh), xytext=(x0 + 1.5 * cw, y0 + rh + 0.7),
                arrowprops=dict(arrowstyle="-|>", color=PRIMARY), color=PRIMARY, fontsize=10)
    ax.annotate("index\n(row labels)", xy=(x0 - cw * 0.28, y0 - 1.5 * rh),
                xytext=(0.0, y0 - 2.4 * rh), arrowprops=dict(arrowstyle="-|>", color="#475569"),
                color="#475569", fontsize=9.5)
    ax.text(x0 + 2 * cw, y0 - 3.6 * rh, "one column = a Series", ha="center",
            fontsize=10, style="italic", color=GREEN)
    ax.set_title("Anatomy of a Pandas DataFrame", color=INK, fontsize=13,
                 fontweight="bold", pad=2)
    save(fig, "ch08_dataframe.png")


def groupby_diagram():
    """Split-apply-combine illustration."""
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    _box(ax, 0.3, 2.3, 2.4, 1.4, "ALL ROWS\n(mixed cities)", INK, fs=10)
    _box(ax, 4.0, 3.6, 2.4, 1.1, "Karachi rows", SKY, fs=10)
    _box(ax, 4.0, 1.3, 2.4, 1.1, "Lahore rows", VIOLET, fs=10)
    _box(ax, 7.6, 3.6, 2.0, 1.1, "mean = 89.5", SKY, fs=10)
    _box(ax, 7.6, 1.3, 2.0, 1.1, "mean = 64.0", VIOLET, fs=10)
    _box(ax, 10.0, 2.3, 1.7, 1.4, "COMBINED\nresult", GREEN, fs=9.5)
    _arrow(ax, 2.7, 3.2, 4.0, 4.1); _arrow(ax, 2.7, 2.8, 4.0, 1.9)
    _arrow(ax, 6.4, 4.1, 7.6, 4.1); _arrow(ax, 6.4, 1.9, 7.6, 1.9)
    _arrow(ax, 9.6, 4.1, 10.0, 3.4); _arrow(ax, 9.6, 1.9, 10.0, 2.6)
    ax.text(3.35, 5.2, "SPLIT", ha="center", color=INK, fontsize=10, fontweight="bold")
    ax.text(6.9, 5.2, "APPLY", ha="center", color=INK, fontsize=10, fontweight="bold")
    ax.text(10.3, 5.2, "COMBINE", ha="center", color=INK, fontsize=10, fontweight="bold")
    ax.set_title("GroupBy: split → apply → combine", color=INK, fontsize=13,
                 fontweight="bold", pad=2)
    save(fig, "ch08_groupby.png")


# ===========================================================================
# CHAPTER 7 — Python for ML
# ===========================================================================
def data_structures_diagram():
    """Visual of list, tuple, dict, set."""
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    cards = [
        (0.3, SKY, "LIST", "ordered, changeable", "[85, 90, 78]", "use []"),
        (3.2, VIOLET, "TUPLE", "ordered, FIXED", "(3, 4)", "use ()"),
        (6.1, GREEN, "DICT", "key → value", '{"name": "Ali"}', "use {k:v}"),
        (9.0, AMBER, "SET", "unique, unordered", "{1, 2, 3}", "use {}"),
    ]
    for x, color, title, desc, example, hint in cards:
        _box(ax, x, 3.4, 2.7, 2.2, "", color)
        ax.text(x + 1.35, 5.05, title, ha="center", color="white", fontsize=14, fontweight="bold")
        ax.text(x + 1.35, 4.35, desc, ha="center", color="white", fontsize=9)
        ax.text(x + 1.35, 3.75, hint, ha="center", color="white", fontsize=8.5, style="italic")
        ax.text(x + 1.35, 2.7, example, ha="center", color=color, fontsize=10,
                fontfamily="monospace", fontweight="bold")
    ax.text(6.0, 1.4, "Choosing the right container is a core Python skill",
            ha="center", fontsize=11, style="italic", color=INK)
    ax.set_title("Python's four core data structures", color=INK,
                 fontsize=13, fontweight="bold", pad=2)
    save(fig, "ch07_data_structures.png")


def ecosystem_diagram():
    """The Python ML library ecosystem grouped by job."""
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")
    groups = [
        (0.3, SKY,    "DATA",        ["NumPy", "Pandas"]),
        (2.7, GREEN,  "VISUALISE",   ["Matplotlib", "Seaborn"]),
        (5.1, PRIMARY,"CLASSIC ML",  ["scikit-learn"]),
        (7.5, VIOLET, "DEEP LEARNING",["TensorFlow", "PyTorch"]),
        (9.9, AMBER,  "DEPLOY",      ["FastAPI", "Streamlit"]),
    ]
    for x, color, title, libs in groups:
        _box(ax, x, 4.6, 2.1, 1.0, title, color, fs=10)
        for j, lib in enumerate(libs):
            _box(ax, x + 0.1, 3.3 - j * 0.95, 1.9, 0.72, lib, "white", ec=color, tc=color, fs=9.5, bold=False)
            _arrow(ax, x + 1.05, 4.6, x + 1.05, 4.02 - j * 0.95, color="#cbd5e1")
    # workflow arrow underneath
    ax.annotate("", xy=(11.0, 0.7), xytext=(0.6, 0.7),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=2))
    ax.text(6.0, 0.95, "typical project flow:  data  →  explore  →  model  →  deploy",
            ha="center", fontsize=10, style="italic", color=INK)
    ax.set_title("The Python Machine Learning ecosystem", color=INK,
                 fontsize=13, fontweight="bold", pad=2)
    save(fig, "ch07_ecosystem.png")


# ===========================================================================
REGISTRY = {
    "ch01": [ai_ml_dl_venn, traditional_vs_ml, types_of_ai],
    "ch02": [ml_drivers, ml_workflow, ml_types_overview, overfitting],
    "ch03": [perceptron_diagram, ai_winters, ml_timeline],
    "ch04": [ml_taxonomy, classification_vs_regression, clustering_demo, rl_loop],
    "ch05": [vector_diagram, matrix_mult_diagram, derivative_diagram,
             gradient_descent_diagram, learning_rate_diagram],
    "ch06": [skewness_diagram, normal_diagram, clt_diagram, correlation_diagram],
    "ch07": [data_structures_diagram, ecosystem_diagram],
    "ch08": [dataframe_diagram, groupby_diagram],
    "ch09": [data_types_diagram, analytics_levels_diagram],
    "ch10": [missing_strategies_diagram, outliers_diagram],
    "ch11": [scaling_diagram, onehot_diagram],
    "ch12": [fe_overview_diagram, binning_diagram],
    "ch13": [curse_diagram, selection_methods_diagram],
    "ch14": [chart_gallery_diagram, chart_chooser_diagram],
    "ch15": [eda_workflow_diagram],
    "ch16": [supervised_flow_diagram, decision_boundaries_diagram, bias_variance_diagram],
    "ch17": [best_fit_diagram],
    "ch18": [sigmoid_diagram],
    "ch19": [knn_vote_diagram, k_effect_diagram],
    "ch20": [naive_bayes_diagram],
    "ch21": [decision_tree_diagram, depth_overfit_diagram],
    "ch22": [svm_margin_diagram, kernel_trick_diagram],
    "ch23": [ensemble_diagram, random_forest_diagram],
    "ch24": [boosting_diagram, gradient_boosting_diagram],
    "ch25": [crossval_diagram, confusion_diagram, roc_diagram],
    "ch26": [search_diagram, l1_l2_diagram],
    "ch27": [kmeans_diagram, elbow_diagram, dendrogram_diagram, dbscan_diagram],
    "ch28": [pca_diagram, explained_variance_diagram, tsne_diagram],
    "ch29": [market_basket_diagram],
    "ch30": [semi_supervised_diagram, self_training_diagram],
    "ch31": [rl_loop2_diagram, explore_exploit_diagram],
    "ch32": [neuron_diagram, activations_diagram, mlp_diagram],
    "ch33": [backprop_diagram, optimizers_diagram, dropout_diagram, train_val_diagram],
    "ch34": [convolution_diagram, pooling_diagram, cnn_arch_diagram],
    "ch35": [rnn_unrolled_diagram, lstm_diagram],
    "ch36": [autoencoder_diagram, gan_diagram],
    "ch37": [attention_diagram, transformer_diagram],
    "ch38": [nlp_representations_diagram, embeddings_diagram, nlp_pipeline_diagram],
    "ch39": [llm_training_diagram],
    "ch40": [cv_tasks_diagram, transfer_learning_diagram],
    "ch41": [rec_approaches_diagram, user_item_matrix_diagram],
    "ch42": [ts_components_diagram, ts_split_diagram],
    "ch43": [genai_landscape_diagram, diffusion_diagram],
    "ch44": [deployment_diagram],
    "ch45": [mlops_lifecycle_diagram],
    "ch46": [cloud_stack_diagram],
    "ch47": [edge_vs_cloud_diagram, compression_diagram],
    "ch48": [responsible_ai_diagram],
    "ch50": [industries_diagram],
    "ch51": [interview_types_diagram],
    "ch52": [freelance_services_diagram],
    "ch53": [ml_roles_diagram],
    "ch54": [frontiers_diagram],
}

# ===========================================================================
# EQUATIONS — rendered to images via mathtext (WeasyPrint can't run MathJax).
# In markdown, reference as:  <img class="eq" src="assets/images/eq_NAME.png">
# Keep names unique and prefixed by chapter.
# ===========================================================================
EQUATIONS = {
    # ---- Chapter 5: Mathematics for ML ----
    "ch05_linear_combo":  r"\hat{y} = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b",
    "ch05_dot_product":   r"\mathbf{w}\cdot\mathbf{x} = \sum_{i=1}^{n} w_i x_i",
    "ch05_vector_norm":   r"\|\mathbf{v}\| = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}",
    "ch05_matrix_vec":    r"\hat{\mathbf{y}} = X\mathbf{w} + b",
    "ch05_mse":           r"L(w,b) = \frac{1}{n}\sum_{i=1}^{n}\left(y_i - \hat{y}_i\right)^2",
    "ch05_derivative":    r"f'(x) = \lim_{h \to 0}\frac{f(x+h) - f(x)}{h}",
    "ch05_partial":       r"\frac{\partial L}{\partial w_j}",
    "ch05_gradient":      r"\nabla L = \left[\frac{\partial L}{\partial w_1}, \frac{\partial L}{\partial w_2}, \dots, \frac{\partial L}{\partial w_n}\right]",
    "ch05_grad_step":     r"w_{\mathrm{new}} = w_{\mathrm{old}} - \eta \,\frac{\partial L}{\partial w}",
    "ch05_chain_rule":    r"\frac{dz}{dx} = \frac{dz}{dy}\cdot\frac{dy}{dx}",
    "ch05_sigmoid":       r"\sigma(z) = \frac{1}{1 + e^{-z}}",
    # ---- Chapter 6: Statistics for ML ----
    "ch06_mean":          r"\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i",
    "ch06_variance":      r"\sigma^2 = \frac{1}{n}\sum_{i=1}^{n}\left(x_i - \bar{x}\right)^2",
    "ch06_std":           r"\sigma = \sqrt{\frac{1}{n}\sum_{i=1}^{n}\left(x_i - \bar{x}\right)^2}",
    "ch06_conditional":   r"P(A \mid B) = \frac{P(A \cap B)}{P(B)}",
    "ch06_bayes":         r"P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)}",
    "ch06_normal_pdf":    r"f(x) = \frac{1}{\sigma\sqrt{2\pi}}\;e^{-\frac{(x-\mu)^2}{2\sigma^2}}",
    "ch06_zscore":        r"z = \frac{x - \mu}{\sigma}",
    "ch06_correlation":   r"r = \frac{\mathrm{cov}(x,y)}{\sigma_x\,\sigma_y}",
    # ---- Chapter 11: Data Preprocessing ----
    "ch11_minmax":        r"x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}",
    "ch11_standardize":   r"x' = \frac{x - \mu}{\sigma}",
    # ---- Chapter 17: Linear Regression ----
    "ch17_simple":        r"\hat{y} = w x + b",
    "ch17_multiple":      r"\hat{y} = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b",
    "ch17_normal_eq":     r"\mathbf{w} = (X^{T}X)^{-1}X^{T}\mathbf{y}",
    "ch17_rmse":          r"\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}",
    "ch17_mae":           r"\mathrm{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|",
    "ch17_r2":            r"R^2 = 1 - \frac{\sum_i (y_i-\hat{y}_i)^2}{\sum_i (y_i-\bar{y})^2}",
    # ---- Chapter 18: Logistic Regression ----
    "ch18_sigmoid":       r"p = \sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = \mathbf{w}\cdot\mathbf{x} + b",
    "ch18_logloss":       r"L = -\frac{1}{n}\sum_{i=1}^{n}\left[\,y_i\log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)\,\right]",
    # ---- Chapter 19: KNN ----
    "ch19_euclidean":     r"d(\mathbf{p},\mathbf{q}) = \sqrt{\sum_{i=1}^{n}(p_i - q_i)^2}",
    "ch19_manhattan":     r"d(\mathbf{p},\mathbf{q}) = \sum_{i=1}^{n}|p_i - q_i|",
    # ---- Chapter 20: Naive Bayes ----
    "ch20_nb":            r"P(c \mid x_1,\dots,x_n) \;\propto\; P(c)\prod_{i=1}^{n} P(x_i \mid c)",
    # ---- Chapter 21: Decision Trees ----
    "ch21_gini":          r"G = 1 - \sum_{k=1}^{K} p_k^{\,2}",
    "ch21_entropy":       r"H = -\sum_{k=1}^{K} p_k \log_2 p_k",
    "ch21_infogain":      r"IG = H_{\mathrm{parent}} - \sum_{j} \frac{n_j}{n}\,H_j",
    # ---- Chapter 22: SVM ----
    "ch22_hyperplane":    r"\mathbf{w}\cdot\mathbf{x} + b = 0",
    "ch22_margin":        r"\mathrm{margin} = \frac{2}{\|\mathbf{w}\|}",
    # ---- Chapter 25: Evaluation & Metrics ----
    "ch25_accuracy":      r"\mathrm{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}",
    "ch25_precision":     r"\mathrm{Precision} = \frac{TP}{TP + FP}",
    "ch25_recall":        r"\mathrm{Recall} = \frac{TP}{TP + FN}",
    "ch25_f1":            r"F_1 = 2 \cdot \frac{\mathrm{Precision}\cdot\mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}}",
    # ---- Chapter 26: Tuning & Regularization ----
    "ch26_ridge":         r"L_{\mathrm{Ridge}} = \mathrm{MSE} + \lambda \sum_{j=1}^{n} w_j^{\,2}",
    "ch26_lasso":         r"L_{\mathrm{Lasso}} = \mathrm{MSE} + \lambda \sum_{j=1}^{n} |w_j|",
    # ---- Chapter 27: Clustering ----
    "ch27_inertia":       r"J = \sum_{i=1}^{n}\; \min_{c}\; \|\mathbf{x}_i - \boldsymbol{\mu}_c\|^2",
    # ---- Chapter 28: PCA ----
    "ch28_variance":      r"\mathrm{maximise}\;\; \mathrm{Var}(X\mathbf{w}) \quad \mathrm{s.t.}\;\; \|\mathbf{w}\|=1",
    # ---- Chapter 29: Association Rules ----
    "ch29_support":       r"\mathrm{Support}(A) = \frac{\#\{\text{transactions containing } A\}}{\#\{\text{total transactions}\}}",
    "ch29_confidence":    r"\mathrm{Confidence}(A \to B) = \frac{\mathrm{Support}(A \cup B)}{\mathrm{Support}(A)}",
    "ch29_lift":          r"\mathrm{Lift}(A \to B) = \frac{\mathrm{Confidence}(A \to B)}{\mathrm{Support}(B)}",
    # ---- Chapter 31: Reinforcement Learning ----
    "ch31_return":        r"G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots = \sum_{k=0}^{\infty}\gamma^k r_{t+k}",
    "ch31_qlearning":     r"Q(s,a) \leftarrow Q(s,a) + \alpha\left[\,r + \gamma \max_{a'} Q(s',a') - Q(s,a)\,\right]",
    # ---- Chapter 32: Neural Networks ----
    "ch32_neuron":        r"a = \phi(\mathbf{w}\cdot\mathbf{x} + b)",
    "ch32_relu":          r"\mathrm{ReLU}(z) = \max(0, z)",
    "ch32_softmax":       r"\mathrm{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}",
    # ---- Chapter 34: CNNs ----
    "ch34_convsize":      r"\mathrm{out} = \frac{W - K + 2P}{S} + 1",
    # ---- Chapter 35: RNN/LSTM ----
    "ch35_rnn":           r"h_t = \tanh\!\left(W_h\,h_{t-1} + W_x\,x_t + b\right)",
    # ---- Chapter 37: Transformers ----
    "ch37_attention":     r"\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^{T}}{\sqrt{d_k}}\right)V",
    # ---- Chapter 38: NLP ----
    "ch38_tfidf":         r"\mathrm{tfidf}(t,d) = \mathrm{tf}(t,d)\times\log\frac{N}{\mathrm{df}(t)}",
    # ---- Chapter 41: Recommendation Systems ----
    "ch41_cosine":        r"\mathrm{sim}(\mathbf{u},\mathbf{v}) = \frac{\mathbf{u}\cdot\mathbf{v}}{\|\mathbf{u}\|\,\|\mathbf{v}\|}",
}


def render_equations():
    print("Generating equations...")
    failed = []
    for name, latex in EQUATIONS.items():
        try:
            eq(latex, name)
        except Exception as exc:                       # one bad eq must not stop the rest
            failed.append(name)
            print(f"  !! FAILED to render eq_{name}: {exc}")
    if failed:
        print(f"  !! {len(failed)} equation(s) failed: {failed}")


if __name__ == "__main__":
    print("Generating diagrams...")
    for chapter, funcs in REGISTRY.items():
        for f in funcs:
            f()
    render_equations()
    print("All diagrams and equations generated.")
