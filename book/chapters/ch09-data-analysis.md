# Data Analysis Fundamentals

## Introduction

There is a famous saying in the field: **"Machine Learning is 80% data and 20%
modelling."** Before any algorithm can learn, *you* must understand the data — what
it contains, what shape it's in, what's missing, and what stories it tells. That
skill is **data analysis**, and it is the focus of Part III of this book.

Think of a doctor. Before prescribing treatment (the "model"), they examine the
patient, run tests, and understand the symptoms (the "data"). A doctor who skips
the examination is dangerous. An ML practitioner who skips data analysis builds
models that fail in surprising, expensive ways.

This chapter lays the foundation: the *types* of data, where data comes from, and
the core operations for analysing it. Chapters 10–15 then go deep on cleaning,
preprocessing, feature engineering, visualisation, and full exploratory analysis.

By the end you will be able to:

- Recognise the different **types of data** and why the type changes how you treat
  it.
- Know the common **data sources and formats**.
- Understand the four levels of analytics: **descriptive, diagnostic, predictive,
  prescriptive**.
- Perform **univariate**, **bivariate**, and **grouped** analysis in Pandas.
- Build **pivot tables** to summarise data by category.

::: keyidea
Every modelling decision later in this book depends on understanding your data
*first*. The practitioners who win are not those with the fanciest algorithms, but
those who understand their data most deeply.
:::

## Types of data

The *type* of a variable decides which analysis, chart, and model are appropriate.
Getting this wrong is a common, costly beginner mistake.

![A map of data types. The first split is numerical vs categorical; numerical splits into discrete and continuous; categorical splits into nominal (no order) and ordinal (ordered).](assets/images/ch09_data_types.png)

### Structured vs unstructured

- **Structured data** — neatly organised in rows and columns (spreadsheets,
  databases). Easy for classic ML. *Example:* a table of customers.
- **Unstructured data** — no fixed format: text, images, audio, video. Needs special
  handling (Parts VI–VII). *Example:* product reviews, photos.

### Numerical data

- **Discrete** — countable whole numbers. *Example:* number of children, number of
  purchases.
- **Continuous** — any value in a range, including decimals. *Example:* height,
  temperature, price.

### Categorical data

- **Nominal** — categories with **no order**. *Example:* city, colour, gender.
- **Ordinal** — categories with a **meaningful order** but unequal/unknown gaps.
  *Example:* education level (school < bachelor < master), ratings (low < medium <
  high).

### Other important types

- **Datetime** — dates and times (need special parsing; Chapter 42 covers time
  series).
- **Text** — free-form language (Chapter 38, NLP).
- **Boolean** — True/False (a special two-category type).

::: warning
**Why the type matters:** you cannot take the "average" of nominal data (the mean
city is meaningless). You must encode categories as numbers before modelling
(Chapter 11), and ordinal data should keep its order. Treating an ordinal rating as
plain text — or a category code like "1, 2, 3" as a real number — leads to wrong
models. Always identify each column's true type first.
:::

## Data sources and formats

Data can come from many places, in many formats. Pandas reads almost all of them.

| Source / Format | Description | Pandas reader |
|---|---|---|
| **CSV** | Comma-separated text; the most common | `pd.read_csv()` |
| **Excel** | `.xlsx` spreadsheets | `pd.read_excel()` |
| **JSON** | Nested key–value data (web APIs) | `pd.read_json()` |
| **SQL database** | Relational tables | `pd.read_sql()` |
| **APIs** | Live data over the web (often JSON) | `requests` + `read_json` |
| **Web scraping** | Extracting from web pages | `pd.read_html()`, BeautifulSoup |
| **Parquet** | Compressed columnar format (big data) | `pd.read_parquet()` |

## The four levels of analytics

Data analysis answers progressively harder questions. This "analytics ladder" shows
where Machine Learning fits.

![The analytics ladder. Each level answers a harder question and adds more value: descriptive (what happened), diagnostic (why), predictive (what will happen — where ML lives), prescriptive (what to do).](assets/images/ch09_analytics_levels.png)

1. **Descriptive analytics — "What happened?"** Summaries of past data (totals,
   averages, charts). *Example:* "Sales fell 10% last quarter."
2. **Diagnostic analytics — "Why did it happen?"** Finding causes and relationships.
   *Example:* "Sales fell because of a stock shortage in May."
3. **Predictive analytics — "What will happen?"** Using patterns to forecast the
   future. **This is where most Machine Learning lives.** *Example:* "We predict a
   15% drop next quarter."
4. **Prescriptive analytics — "What should we do?"** Recommending actions.
   *Example:* "Increase stock by 20% and run a promotion." (Often uses optimisation
   and reinforcement learning.)

::: note
This book takes you up the whole ladder, but its heart is **predictive** analytics —
teaching machines to forecast and classify from data.
:::

## The data analysis process

A reliable, repeatable sequence:

1. **Ask a clear question** ("Which department has the highest salaries?").
2. **Collect / load** the relevant data.
3. **Inspect** structure, types, and quality (`head`, `info`, `describe`).
4. **Clean** errors and missing values (Chapter 10).
5. **Analyse** — univariate, then relationships, then grouped summaries.
6. **Visualise** to reveal patterns (Chapter 14).
7. **Interpret and communicate** findings clearly.

## Univariate, bivariate, and multivariate analysis

- **Univariate** — analysing **one** variable at a time (its distribution, average,
  spread). *Question:* "What's the typical salary?"
- **Bivariate** — analysing the **relationship between two** variables. *Question:*
  "Does salary relate to age?"
- **Multivariate** — analysing **three or more** variables together. *Question:*
  "How do age, department, and experience together affect salary?"

## Practical: analysing an employee dataset

Let's perform real data analysis on a small employee table, touching each technique.

```python
import pandas as pd

df = pd.DataFrame({
    "employee":    ["Ali", "Sara", "Omar", "Lina", "Zed", "Maya", "Bilal", "Nida"],
    "department":  ["Sales", "Eng", "Sales", "Eng", "Sales", "HR", "Eng", "HR"],
    "age":         [25, 32, 41, 28, 38, 45, 29, 35],
    "salary":      [50000, 85000, 62000, 90000, 58000, 55000, 88000, 60000],
    "satisfaction":[3.5, 4.2, 2.8, 4.5, 3.0, 3.8, 4.1, 3.6],
})

# --- Step 1: what TYPES are the columns? ---
print(df.dtypes)
```

**Output:**
```text
employee         object
department       object
age               int64
salary            int64
satisfaction    float64
dtype: object
```

`object` columns are text (categorical here), `int64`/`float64` are numerical.
Identifying types is always step one.

```python
# --- Step 2: univariate analysis of a categorical column ---
print(df["department"].value_counts())
```

**Output:**
```text
department
Sales    3
Eng      3
HR       2
Name: count, dtype: int64
```

`value_counts()` is the go-to tool for a categorical variable — it counts each
category. We have 3 Sales, 3 Eng, 2 HR employees.

```python
# --- Step 3: univariate analysis of a numerical column ---
print(df["salary"].agg(["mean", "median", "min", "max", "std"]).round(1))
```

**Output:**
```text
mean      68500.0
median    61000.0
min       50000.0
max       90000.0
std       16318.3
Name: salary, dtype: float64
```

The mean (68,500) is noticeably higher than the median (61,000) — a sign of
**right-skew** (a few high earners pull the average up), exactly the situation
Chapter 6 warned about.

```python
# --- Step 4: bivariate / grouped analysis ---
print(df.groupby("department")[["salary", "satisfaction"]].mean().round(1))
```

**Output:**
```text
             salary  satisfaction
department
Eng         87666.7           4.3
HR          57500.0           3.7
Sales       56666.7           3.1
```

A powerful insight in one line: **Engineering** has both the highest average salary
*and* the highest satisfaction, while **Sales** is lowest on both. This is the kind
of finding that drives real decisions.

```python
# --- Step 5: relationship between two numeric variables ---
print(round(df["age"].corr(df["salary"]), 3))
```

**Output:**
```text
-0.428
```

The correlation between age and salary is **−0.428** — a moderate *negative*
relationship in this small sample (older employees here happen to earn less, perhaps
because the young engineers are highly paid). *Remember Chapter 6: correlation is not
causation, and small samples can mislead.*

```python
# --- Step 6: a pivot table (count of employees per department) ---
print(df.pivot_table(index="department", values="salary", aggfunc="count"))
```

**Output:**
```text
            salary
department
Eng              3
HR               2
Sales            3
```

A **pivot table** summarises data by category — here counting employees per
department. Pivot tables can cross two categories and apply any aggregation
(`mean`, `sum`, `count`, etc.), making them the Swiss-army knife of data analysis.

::: keyidea
Notice the flow: we identified **types**, did **univariate** analysis (value counts,
salary stats), then **bivariate/grouped** analysis (department comparisons,
correlation), then **summarised** with a pivot table. This sequence — *understand
each variable, then their relationships, then summarise* — is the backbone of all
exploratory work (Chapter 15).
:::

::: tip
**Practical tips:** (1) `df["col"].value_counts(normalize=True)` gives proportions
(percentages) instead of counts. (2) `df.describe(include="all")` summarises both
numeric and categorical columns. (3) For correlations among many columns at once,
use `df.corr(numeric_only=True)` (Chapter 15 visualises this as a heatmap). (4)
Always sanity-check surprising results on a small sample before trusting them.
:::

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Treating categorical codes as numbers.** If "department" is stored as
1, 2, 3, never compute its mean — the number is just a label, not a quantity.
:::

- **Mistake 2 — Ignoring data types** and feeding text where numbers are expected
  (or vice versa).
- **Mistake 3 — Trusting averages on skewed data** — report the median too.
- **Mistake 4 — Drawing big conclusions from tiny samples** (like our 8-row
  example).
- **Mistake 5 — Confusing correlation with causation** when interpreting
  relationships.
- **Mistake 6 — Skipping the inspection step** and jumping straight to modelling.

## Best practices

- **Always identify each column's true type** before analysing.
- **Start univariate, then go bivariate, then multivariate** — build understanding
  layer by layer.
- **Report spread and skew, not just averages.**
- **Use `groupby` and pivot tables** to compare across categories.
- **Document your findings in plain English** as you go.
- **Be skeptical** of surprising results; verify them.

## Chapter Summary

- Data analysis is the essential first phase of ML — **80% of the work is
  understanding and preparing data**.
- **Data types:** structured vs unstructured; numerical (discrete, continuous);
  categorical (nominal, ordinal); plus datetime, text, boolean. The type dictates
  the right analysis and encoding.
- Data comes from **CSV, Excel, JSON, SQL, APIs, web, Parquet** — Pandas reads them
  all.
- The **analytics ladder**: descriptive (what), diagnostic (why), **predictive**
  (what next — where ML lives), prescriptive (what to do).
- Analysis proceeds **univariate → bivariate → multivariate**, using
  `value_counts`, `describe`, `groupby`, `corr`, and **pivot tables**.

---

::: {.qband}
Practice Zone — Chapter 9
:::

## Multiple-Choice Questions (MCQs)

**Q1.** Which is an example of *ordinal* data?
a) City names  b) Education level (school < bachelor < master)  c) Temperature
d) Phone numbers

**Q2.** Height in centimetres is:
a) Discrete numerical  b) Continuous numerical  c) Nominal  d) Ordinal

**Q3.** "What will sales be next quarter?" is which level of analytics?
a) Descriptive  b) Diagnostic  c) Predictive  d) Prescriptive

**Q4.** Which Pandas method counts each category in a column?
a) `describe()`  b) `value_counts()`  c) `corr()`  d) `merge()`

**Q5.** Analysing the relationship between *two* variables is called:
a) Univariate  b) Bivariate  c) Multivariate  d) Descriptive

**Q6.** Product reviews (free text) are an example of:
a) Structured data  b) Unstructured data  c) Ordinal data  d) Discrete data

**Q7.** Which is the WRONG operation for nominal data like "city"?
a) Counting categories  b) Computing the mean  c) Finding the mode  d) Grouping by it

**Q8.** A tool that summarises data by category with an aggregation is a:
a) Histogram  b) Pivot table  c) Correlation  d) Boolean mask

### MCQ Answers
**1:** b. **2:** b. **3:** c. **4:** b. **5:** b. **6:** b. **7:** b. **8:** b.

## Interview Questions (with answers)

**Q1. What is the difference between nominal and ordinal data?**
*Answer:* Both are categorical. Nominal categories have no inherent order (city,
colour). Ordinal categories have a meaningful order but not necessarily equal gaps
(low/medium/high, education levels). The distinction matters for encoding: ordinal
data should preserve its order, nominal data should not imply one.

**Q2. Explain the four levels of analytics.**
*Answer:* Descriptive (what happened — summaries), diagnostic (why it happened —
causes/relationships), predictive (what will happen — forecasting, where ML lives),
and prescriptive (what to do — recommended actions, often via optimisation/RL). Each
level adds value and difficulty.

**Q3. Why is identifying data types the first step of analysis?**
*Answer:* Because the type determines valid operations, charts, and encodings. You
can't average nominal data, continuous and categorical variables need different
plots, and ordinal order must be preserved. Wrong type handling leads to meaningless
statistics and broken models.

**Q4. What is the difference between univariate and bivariate analysis?**
*Answer:* Univariate analysis studies one variable alone (its distribution, centre,
spread). Bivariate analysis studies the relationship between two variables (e.g.
correlation, grouped comparisons). Multivariate extends this to three or more
together.

## Scenario-Based Questions (with answers)

**Q1.** *A dataset stores "satisfaction" as the text values "Low", "Medium",
"High". A colleague maps them to 1, 2, 3 and computes the mean. Is this valid?*
*Answer:* Cautiously. The data is **ordinal**, so the order 1<2<3 is meaningful and a
mean can be a rough summary — but the gaps between levels aren't guaranteed equal, so
the mean can mislead. Reporting the distribution (counts of each level) or the median
level is safer.

**Q2.** *Your boss sees "average customer spend = \$500" and wants to target premium
products, but you suspect a few whales skew it. What analysis confirms this and what
do you recommend?*
*Answer:* Compare mean vs median and look at the distribution (`describe`, a
histogram). If the median is much lower (e.g. \$60), the data is right-skewed and the
mean is misleading. Recommend reporting the median and segmenting the few high
spenders separately.

**Q3.** *You find a strong correlation between number of firefighters at a scene and
the amount of fire damage. Should the city send fewer firefighters?*
*Answer:* No — this is a correlation-vs-causation trap. A bigger fire causes *both*
more firefighters *and* more damage (the fire size is the hidden cause). Reducing
firefighters would worsen outcomes. Always look for confounders.

## Logic-Based Questions (with answers)

**Q1.** In the employee data, the mean salary (68,500) exceeds the median (61,000).
What does that imply about the salary distribution?
*Answer:* It implies a right-skew: a few high salaries pull the mean above the
median. Most employees earn below the mean.

**Q2.** `value_counts()` on a column returns 3 rows. What does that tell you about
the column?
*Answer:* The column has exactly three distinct categories. It is categorical with
three possible values.

**Q3.** If age and salary have correlation −0.428, and you add an older, very
high-paid CEO to the data, what will likely happen to the correlation?
*Answer:* The correlation would move toward positive (less negative), because the new
point pairs high age with high salary, weakening the existing negative trend — showing
how sensitive correlation is to individual points in small data.

## Practical Questions (with answers)

**Q1.** Write one line to get the *proportion* (not count) of each department.
*Answer:* `df["department"].value_counts(normalize=True)`.

**Q2.** Write one line to find the average satisfaction per department.
*Answer:* `df.groupby("department")["satisfaction"].mean()`.

**Q3.** Which single Pandas call summarises both numeric and categorical columns at
once?
*Answer:* `df.describe(include="all")`.

## Long Questions (with answers)

**Q1. Explain the different types of data (numerical and categorical, with their
sub-types), giving an example of each, and explain why correctly identifying the type
is critical before analysis or modelling.**

*Answer:* Data divides first into **numerical** and **categorical**. Numerical data is
**discrete** (countable whole numbers, e.g. number of children) or **continuous** (any
value in a range, e.g. height or temperature). Categorical data is **nominal**
(unordered categories, e.g. city or colour) or **ordinal** (ordered categories with
unequal/unknown gaps, e.g. education level or low/medium/high ratings). Other types
include datetime, free text, and boolean. Correct identification is critical because
the type governs which operations and tools are valid: you can average continuous
numbers but not nominal categories; ordinal data must preserve its order when encoded
while nominal must not imply one; numeric-looking category codes (department = 1,2,3)
must never be treated as quantities; and different types require different charts and
different model preprocessing. Mislabelling a type leads to meaningless statistics
(like the "average city"), misleading visualisations, and models that learn false
relationships — errors that are hard to detect later but easy to prevent by checking
types first.

**Q2. Describe the four levels of analytics and where Machine Learning fits, using a
single business example carried through all four levels.**

*Answer:* Take an online store. **Descriptive analytics** answers "what happened?" —
e.g. "revenue dropped 12% last month" — using summaries and charts of past data.
**Diagnostic analytics** answers "why?" — e.g. "revenue dropped because returning
customers fell after a checkout bug" — by digging into relationships and segments.
**Predictive analytics** answers "what will happen?" — e.g. "we forecast a further 8%
drop next month unless we act" — by learning patterns from history to forecast the
future; this is where **Machine Learning** primarily operates, building models that
classify or predict. **Prescriptive analytics** answers "what should we do?" — e.g.
"fix the bug and offer returning customers a 10% coupon to maximise recovered
revenue" — recommending optimal actions, often via optimisation or reinforcement
learning. The levels build on each other: you must understand and explain the past
before you can reliably predict the future or prescribe action, and each step up the
ladder adds both difficulty and business value.

## Exercises

1. Classify each as nominal, ordinal, discrete, or continuous: blood type, T-shirt
   size (S/M/L), number of pets, body temperature, postal code.
2. For each analytics level, write a question about a topic you care about (e.g.
   your studies, a sport, a business).
3. Load any CSV and report: its shape, each column's type, and one univariate
   summary per column.
4. Pick two numeric columns and compute their correlation. Interpret the sign and
   strength in words.
5. Explain why you cannot meaningfully compute the mean of a nominal variable.

## Mini-Project

**Project: A one-page data analysis report.**

1. Choose a dataset (e.g. Titanic, tips, or any CSV with at least 5 columns and 100+
   rows).
2. Identify and list each column's data type.
3. Do univariate analysis on every column (value counts for categorical, describe for
   numeric).
4. Do at least three grouped/bivariate analyses (e.g. average target per category,
   two correlations).
5. Write a one-page report of your top 5 findings in plain English, each backed by a
   number. Save it in `my-ml-journey/`.

## Assignments

1. **Coding:** Take the employee DataFrame from this chapter and add an "experience"
   column. Compute the correlation of experience with salary, and the average salary
   per department sorted from highest to lowest.
2. **Conceptual:** Write half a page explaining, with examples, why "80% of ML is
   data work." Reference at least three things that can go wrong if data analysis is
   skipped.
3. **Research:** Find a real dataset online (Kaggle, government open data). Document
   its source, format, number of rows/columns, and the type of each column.

::: tip
Next, in Chapter 10, we tackle the messy reality: real data is full of missing
values, errors, duplicates, and outliers. **Data cleaning** is where good analysis
becomes trustworthy analysis.
:::
