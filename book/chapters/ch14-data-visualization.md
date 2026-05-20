# Data Visualization

## Introduction

There is a reason the saying "a picture is worth a thousand words" survives. Humans
are visual creatures. You can stare at a table of 10,000 numbers and see nothing —
but plot them, and a pattern, a trend, or a glaring outlier jumps out instantly.

**Data visualization** is the art of turning numbers into pictures that reveal what
the data is *really* doing. It serves two purposes in Machine Learning:

1. **Exploration (for you)** — to *understand* your data, spot problems, and find
   patterns before modelling.
2. **Communication (for others)** — to *explain* your findings clearly to teammates,
   bosses, and clients.

::: keyidea
A famous dataset called **Anscombe's Quartet** has four groups of points with nearly
*identical* statistics (same mean, variance, correlation) — yet when plotted they look
completely different (a line, a curve, an outlier-driven trend). The lesson:
**always plot your data. Numbers alone can lie; pictures reveal the truth.**
:::

By the end of this chapter you will be able to:

- Use **Matplotlib** and **Seaborn**, the two main Python plotting libraries.
- Choose the **right chart** for each kind of question.
- Read **histograms, box plots, scatter plots, bar charts, line charts, and
  heatmaps**.
- Avoid common ways charts mislead.

## The two main libraries

- **Matplotlib** — the foundational, highly customisable plotting library. Everything
  is possible, but you control every detail. Import as `import matplotlib.pyplot as
  plt`.
- **Seaborn** — built *on top of* Matplotlib; it makes beautiful statistical charts
  with far less code and sensible defaults. Import as `import seaborn as sns`.

Use Seaborn for quick, attractive statistical plots; drop to Matplotlib when you need
fine control.

## A gallery of essential charts

Each chart type answers a different question. This gallery shows the six you'll use
most.

![A gallery of the six essential charts: line (trends over time), bar (compare categories), histogram (distribution of one variable), box plot (spread & outliers), scatter (relationship between two variables), and heatmap (a grid of values such as a correlation matrix).](assets/images/ch14_gallery.png)

### Which chart should I use?

![A chart chooser. Start from your question: showing a trend over time, comparing categories, examining one variable's distribution, relating two numeric variables, or showing a matrix of values — each points to a chart type.](assets/images/ch14_chooser.png)

| Your question | Best chart |
|---|---|
| How does something change over **time**? | **Line chart** |
| How do **categories** compare? | **Bar chart** |
| What is the **distribution** of one numeric variable? | **Histogram** |
| What is the **spread** and are there **outliers**? | **Box plot** |
| Is there a **relationship** between two numeric variables? | **Scatter plot** |
| How do many variables **correlate**? | **Heatmap** |
| What are the **proportions** of a whole? | Pie chart (use sparingly) |

## Making charts with Matplotlib

The basic pattern: create data, call a plotting function, label everything, show or
save.

```python
import matplotlib.pyplot as plt

months = ["Jan", "Feb", "Mar", "Apr", "May"]
sales  = [120, 135, 150, 145, 170]

plt.figure(figsize=(7, 4))          # create a figure of a chosen size
plt.plot(months, sales, marker="o") # line chart with point markers
plt.title("Monthly Sales")          # ALWAYS title your chart
plt.xlabel("Month")                 # ALWAYS label the axes
plt.ylabel("Sales ($1000s)")
plt.grid(alpha=0.3)                 # a light grid aids reading
plt.savefig("sales.png")            # save to a file (or plt.show() to display)
```

This produces a line chart showing sales rising over the months. Every chart you make
should have a **title** and **labelled axes** — an unlabelled chart is almost useless.

### A histogram (distribution of one variable)

```python
import numpy as np
import matplotlib.pyplot as plt

ages = np.random.normal(35, 10, 500)   # 500 ages, mean 35, std 10
plt.hist(ages, bins=20, color="#4f46e5", edgecolor="white")
plt.title("Distribution of Ages"); plt.xlabel("Age"); plt.ylabel("Count")
```

A **histogram** splits the range into bins and counts how many values fall in each —
revealing the shape (symmetric? skewed? Chapter 6) of one variable.

## Making charts with Seaborn

Seaborn produces richer statistical charts in fewer lines and works directly with
DataFrames.

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")    # a built-in example dataset

# Scatter plot coloured by a category, in one line
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="time")
plt.title("Tip vs Total Bill")

# A correlation heatmap (very common in EDA)
corr = tips.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
```

- **`scatterplot(..., hue="time")`** colours points by the `time` category — adding a
  third dimension to a 2-D chart.
- **`heatmap(corr, annot=True)`** displays the correlation matrix as a colour grid
  with numbers — the fastest way to see which variables move together (a staple of
  Chapter 15's EDA).

::: tip
**Reading a correlation heatmap:** values near **+1** (often warm colours) mean strong
positive correlation, near **−1** (cool colours) strong negative, near **0** little
linear relationship. Scan for bright off-diagonal cells — those are your strongly
related variable pairs. (The diagonal is always 1: every variable correlates perfectly
with itself.)
:::

## Principles of good (and honest) visualization

A chart can clarify *or* mislead. Follow these principles:

- **Always title and label axes** (including units).
- **Start bar-chart axes at zero** — truncating the y-axis exaggerates differences
  (a classic way charts deceive).
- **Don't overload** one chart with too much; prefer several clear charts.
- **Use colour meaningfully**, not for decoration; consider colour-blind-friendly
  palettes.
- **Pick the right chart** for the question (use the chooser above).
- **Avoid 3-D effects and pie charts with many slices** — they're hard to read
  accurately.

::: warning
**The truncated-axis trick.** A bar chart of values 98, 99, 100 looks like a *huge*
difference if the y-axis starts at 97, but a tiny one if it starts at 0. Honest bar
charts start at zero. Be alert to this both when *making* and *reading* charts.
:::

## Practical: visualising to find insight

Visualisation isn't decoration — it drives decisions. For example:

- A **histogram** of income shows right-skew → you decide to log-transform (Chapter 12).
- A **box plot** of age reveals an outlier of 200 → you investigate and clean it
  (Chapter 10).
- A **scatter plot** of study-hours vs score shows a clear upward line → you confirm a
  useful predictor.
- A **heatmap** shows two features with 0.97 correlation → you drop one as redundant
  (Chapter 13).

Every plot should answer a question or trigger an action. If a chart tells you nothing,
make a different one.

::: keyidea
Visualisation is the *bridge* between raw data and insight. The best practitioners plot
constantly — before cleaning, before modelling, and after — because the eye catches
what summary statistics miss. Make plotting a reflex, not an afterthought.
:::

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Trusting statistics without plotting** (remember Anscombe's Quartet).
Identical summaries can hide wildly different shapes.
:::

- **Mistake 2 — Misleading axes** (truncated y-axis, inconsistent scales).
- **Mistake 3 — No titles or axis labels**, leaving the reader guessing.
- **Mistake 4 — Wrong chart for the data** (e.g. a line chart for unordered
  categories).
- **Mistake 5 — Overcrowding** one chart with too many series or slices.
- **Mistake 6 — Decorative 3-D and chartjunk** that obscure the data.

## Best practices

- **Plot early and often** — during exploration, cleaning, and after modelling.
- **Match the chart to the question** (use the chooser).
- **Title and label everything**, with units.
- **Start bar axes at zero**; keep scales honest.
- **Prefer clarity over decoration**; one message per chart.
- **Use Seaborn for quick statistical plots**, Matplotlib for fine control.

## Chapter Summary

- **Visualization** turns numbers into pictures, for both **exploration** (understand
  the data) and **communication** (explain findings). Always plot — summaries alone
  can mislead (**Anscombe's Quartet**).
- **Matplotlib** is the customisable foundation; **Seaborn** adds beautiful statistical
  charts with less code.
- Core charts: **line** (trends/time), **bar** (categories), **histogram**
  (distribution), **box plot** (spread/outliers), **scatter** (relationships),
  **heatmap** (correlation matrix).
- Good charts **title and label** everything, use **honest axes** (bars start at zero),
  pick the **right chart** for the question, and avoid clutter and deception.

---

::: {.qband}
Practice Zone — Chapter 14
:::

## Multiple-Choice Questions (MCQs)

**Q1.** To show how a value changes over time, use a:
a) Pie chart  b) Line chart  c) Histogram  d) Heatmap

**Q2.** To see the distribution (shape) of one numeric variable, use a:
a) Scatter plot  b) Bar chart  c) Histogram  d) Line chart

**Q3.** A box plot is especially good for showing:
a) Trends over time  b) Spread and outliers  c) Proportions  d) Correlation matrices

**Q4.** To examine the relationship between two numeric variables, use a:
a) Histogram  b) Bar chart  c) Scatter plot  d) Pie chart

**Q5.** A correlation matrix is best displayed as a:
a) Line chart  b) Heatmap  c) Pie chart  d) Box plot

**Q6.** Anscombe's Quartet teaches that:
a) Pie charts are best  b) Identical statistics can hide very different data — always
plot  c) Heatmaps are useless  d) Bigger data is always better

**Q7.** Starting a bar chart's y-axis above zero can:
a) Improve accuracy  b) Exaggerate differences (mislead)  c) Add labels  d) Fix
outliers

**Q8.** Which library is built on top of Matplotlib for statistical charts?
a) NumPy  b) Pandas  c) Seaborn  d) Flask

### MCQ Answers
**1:** b. **2:** c. **3:** b. **4:** c. **5:** b. **6:** b. **7:** b. **8:** c.

## Interview Questions (with answers)

**Q1. Why is data visualization important in Machine Learning?**
*Answer:* It lets you understand data, spot outliers, missing values, skew, and
relationships before modelling, and it communicates findings to others. Visuals reveal
patterns and problems that summary statistics can hide (Anscombe's Quartet).

**Q2. When would you use a histogram vs a box plot?**
*Answer:* A histogram shows the full *shape* of one variable's distribution (modes,
skew). A box plot compactly summarises spread (quartiles) and flags outliers, and is
great for comparing distributions across categories side by side.

**Q3. What is a correlation heatmap and how do you read it?**
*Answer:* It displays a correlation matrix as a colour grid (often with annotated
numbers). Values near +1 indicate strong positive correlation, near −1 strong negative,
near 0 little linear relationship. Bright off-diagonal cells reveal strongly related
feature pairs (useful for feature selection).

**Q4. Name two ways a chart can mislead.**
*Answer:* Truncating the y-axis (so small differences look huge), and using the wrong
chart type (e.g. a line chart for unordered categories). Others include 3-D effects,
inconsistent scales, and cherry-picked ranges.

## Scenario-Based Questions (with answers)

**Q1.** *Before modelling, you want to quickly check which features are most related to
the target and to each other. What single visualization helps most?*
*Answer:* A correlation heatmap of all numeric features (including the target). It shows
at a glance which features correlate with the target (candidate predictors) and which
correlate with each other (candidate redundancies for feature selection).

**Q2.** *A stakeholder presents a bar chart where sales appear to have doubled, but you
suspect manipulation. What would you check?*
*Answer:* Check whether the y-axis starts at zero. A truncated axis can make a small
real change look enormous. Re-plot with a zero baseline to see the true magnitude.

**Q3.** *Your income feature's summary stats look fine, but the model behaves oddly.
What plot would you make and what might it reveal?*
*Answer:* A histogram (and/or box plot) of income. It likely reveals strong right-skew
and/or outliers that the mean hid — prompting a log transform (Chapter 12) or outlier
handling (Chapter 10).

## Logic-Based Questions (with answers)

**Q1.** Why is a line chart inappropriate for unordered categories like cities?
*Answer:* A line implies a continuous order and connection between points. Cities have
no inherent order, so connecting them with a line suggests a trend that doesn't exist;
a bar chart is correct.

**Q2.** Two datasets have identical mean and standard deviation. Can they look
completely different when plotted? Why?
*Answer:* Yes (as in Anscombe's Quartet). Mean and standard deviation summarise centre
and spread but not shape, so different distributions, clusters, or outlier patterns can
share the same summary statistics — which is exactly why you must plot.

**Q3.** On a correlation heatmap, why is the diagonal always 1?
*Answer:* Each diagonal cell is a variable's correlation with itself, which is always
perfectly 1.

## Practical Questions (with answers)

**Q1.** Write Matplotlib code to make a labelled histogram of an array `data` with 30
bins.
*Answer:*
```python
plt.hist(data, bins=30); plt.title("Distribution"); plt.xlabel("value"); plt.ylabel("count")
```

**Q2.** Write Seaborn code for a scatter plot of `x` vs `y` from DataFrame `df`,
coloured by category `group`.
*Answer:* `sns.scatterplot(data=df, x="x", y="y", hue="group")`.

**Q3.** Which one line gives the correlation matrix of a DataFrame's numeric columns
(to feed a heatmap)?
*Answer:* `df.corr(numeric_only=True)`.

## Long Questions (with answers)

**Q1. Describe the six essential chart types, what question each answers, and give a
real example of when you would use each.**

*Answer:* **(1) Line chart** answers "how does a value change over time/order?" — e.g.
plotting monthly revenue to see a trend. **(2) Bar chart** answers "how do categories
compare?" — e.g. average salary per department; bars should start at zero. **(3)
Histogram** answers "what is the distribution/shape of one numeric variable?" — e.g.
the spread of customer ages, revealing skew or multiple peaks. **(4) Box plot** answers
"what is the spread and are there outliers?" — e.g. comparing exam-score distributions
across classes and spotting extreme values. **(5) Scatter plot** answers "is there a
relationship between two numeric variables?" — e.g. study hours vs exam score, showing
a positive trend; adding colour (hue) encodes a third variable. **(6) Heatmap** answers
"how do many variables relate?" — e.g. a correlation matrix of all features to find
predictors and redundancies. Choosing the right chart for the question is the core skill;
the wrong chart can hide or distort the message.

**Q2. Explain how visualization supports each stage of a Machine Learning project, with
examples, and why plotting is essential rather than optional.**

*Answer:* Visualization supports the whole ML lifecycle. During **data understanding**,
histograms and box plots reveal distributions, skew, and outliers (e.g. an impossible
age of 200), and scatter plots and heatmaps reveal relationships and redundancies,
guiding **cleaning** (Chapter 10), **preprocessing** (e.g. deciding to log-transform a
right-skewed income, Chapter 12), and **feature selection** (dropping a feature that a
heatmap shows is 0.97-correlated with another, Chapter 13). During **modelling and
evaluation**, plots of the loss over epochs reveal training problems, and confusion
matrices and ROC curves (Chapter 25) communicate performance. For **communication**,
clear charts let non-technical stakeholders grasp findings and trust decisions.
Plotting is essential, not optional, because summary statistics can be identical for
very different data (Anscombe's Quartet): the human eye catches clusters, gaps,
outliers, non-linear shapes, and errors that means and correlations miss. A practitioner
who skips visualization routinely misses data problems that quietly ruin models, which
is why "always plot your data" is one of the field's most repeated pieces of advice.

## Exercises

1. For each question, name the best chart: trend of website visits over a year;
   comparing revenue across 4 products; the distribution of house prices; relationship
   between ad spend and sales; correlations among 8 features.
2. Explain why pie charts with 10 slices are hard to read; suggest a better chart.
3. Describe two ways a bar chart can be made misleading.
4. What does a long right "tail" in a histogram tell you, and what might you do about
   it?
5. Why should every chart have a title and labelled axes?

## Mini-Project

**Project: A visual EDA gallery.**

1. Load a dataset (e.g. seaborn's `tips`, `titanic`, or `iris`).
2. Create at least six charts: a histogram, a box plot, a bar chart, a scatter plot
   (with a hue), a line chart (if a time/order column exists), and a correlation
   heatmap.
3. Title and label every chart properly.
4. For each chart, write one sentence stating the insight it reveals.
5. Save all charts and a short write-up in `my-ml-journey/`.

## Assignments

1. **Coding:** Recreate the chart gallery from this chapter using Matplotlib and
   Seaborn on a dataset of your choice. Ensure every chart is titled and labelled.
2. **Coding:** Make a correlation heatmap of a numeric dataset and identify the two
   most strongly correlated feature pairs and the feature most correlated with the
   target.
3. **Conceptual:** Find a misleading chart online (or in news/ads), explain how it
   misleads, and redraw it honestly (sketch or code).

::: tip
Visualization is the eyes of data analysis. Chapter 15 now combines everything from
Part III — cleaning, preprocessing, features, and visualization — into the complete,
professional **Exploratory Data Analysis (EDA)** workflow you'll run at the start of
every project.
:::
