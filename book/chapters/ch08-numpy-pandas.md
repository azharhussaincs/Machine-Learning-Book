# NumPy, Pandas & Scientific Python

## Introduction

In Chapter 7 you learned core Python. Now you meet the two libraries you will use
in **every single remaining chapter** of this book: **NumPy** (fast numbers) and
**Pandas** (data tables). If Python is the language of ML, NumPy and Pandas are its
two most important words.

Remember the pure-Python data work from Chapter 7? It took loops and several lines.
With these libraries, the same work becomes one fast, readable line. More
importantly, **all ML libraries expect data as NumPy arrays or Pandas
DataFrames** — so this chapter is the bridge between raw data and real modelling.

By the end you will be able to:

- Create and manipulate **NumPy arrays**, and understand *why* they beat plain
  lists.
- Use **vectorisation** and **broadcasting** to do maths on whole datasets at once.
- Load, inspect, filter, transform, group, and summarise data with **Pandas
  DataFrames**.
- Handle **missing values** — your first taste of real-world data cleaning.

::: keyidea
NumPy is for **numbers in a grid** (fast maths). Pandas is for **labelled tables**
(real datasets with column names). Pandas is actually built *on top of* NumPy. Learn
both and you can wrangle almost any dataset.
:::

---

# Part A — NumPy: fast numerical arrays

## Why NumPy? Lists are slow; arrays are fast

A Python list can hold anything, which makes it flexible but **slow** for maths. A
NumPy **array** holds numbers of one type in a tight block of memory, which lets
NumPy do maths on the whole array at once, in fast C code. For large data, NumPy can
be **10–100× faster** than loops — and the code is shorter too.

```python
import numpy as np            # the standard nickname for numpy

a = np.array([1, 2, 3, 4, 5]) # make an array from a list
print(a, "| shape", a.shape, "| dtype", a.dtype)

print("zeros:", np.zeros(3))            # array of zeros
print("arange:", np.arange(0, 10, 2))   # like range(): start, stop, step
print("linspace:", np.linspace(0, 1, 5))# 5 evenly spaced numbers from 0 to 1
```

**Output:**
```text
[1 2 3 4 5] | shape (5,) | dtype int64
zeros: [0. 0. 0.]
arange: [0 2 4 6 8]
linspace: [0.   0.25 0.5  0.75 1.  ]
```

- **`shape`** tells you the dimensions (here `(5,)` = a 1-D array of 5 items).
- **`dtype`** is the data type of the elements (here 64-bit integers).
- `zeros`, `arange`, and `linspace` are quick ways to *generate* arrays — you'll use
  them constantly.

## Vectorised operations: maths without loops

This is NumPy's superpower. Apply an operation to an array and it happens to **every
element at once** — no loop needed. This is called **vectorisation**.

```python
a = np.array([1, 2, 3, 4, 5])
print("a*2 =", a * 2)        # multiply every element by 2
print("a+10 =", a + 10)      # add 10 to every element
print("a**2 =", a ** 2)      # square every element
print("mean,sum,max:", a.mean(), a.sum(), a.max())
```

**Output:**
```text
a*2 = [ 2  4  6  8 10]
a+10 = [11 12 13 14 15]
a**2 = [ 1  4  9 16 25]
mean,sum,max: 3.0 15 5
```

::: keyidea
`a * 2` did five multiplications in one short, fast operation. In Chapter 5's
gradient descent, `y_pred = w * X + b` worked on all data points at once for exactly
this reason. Vectorisation is *the* habit that makes ML code fast and clean.
:::

## 2-D arrays and the `axis` idea

Real data is 2-D (rows × columns). Many operations take an **`axis`** argument:
`axis=0` works **down the columns**, `axis=1` works **across the rows**.

```python
m = np.array([[1, 2, 3],
              [4, 5, 6]])
print("shape:", m.shape)               # (2, 3): 2 rows, 3 columns
print("col sums (axis=0):", m.sum(axis=0))  # sum down each column
print("row sums (axis=1):", m.sum(axis=1))  # sum across each row
```

**Output:**
```text
shape: (2, 3)
col sums (axis=0): [5 7 9]
row sums (axis=1): [ 6 15]
```

::: warning
**`axis` confuses everyone at first.** Trick: `axis=0` means "collapse the rows"
(operate down each column → one value per column). `axis=1` means "collapse the
columns" (operate across each row → one value per row). When unsure, test on a tiny
array and check the shape of the result.
:::

## Indexing, slicing, and boolean masks

```python
a = np.array([1, 2, 3, 4, 5])
print("a[a > 2] =", a[a > 2])    # keep only elements greater than 2
```

**Output:**
```text
a[a > 2] = [3 4 5]
```

This **boolean indexing** (or "masking") is one of the most useful tools in data
work: `a > 2` produces `[False False True True True]`, and `a[...]` keeps only the
`True` positions. You'll use it to filter data everywhere.

## Broadcasting: combining different shapes

**Broadcasting** lets NumPy automatically "stretch" a smaller array to match a
larger one, so you can combine them without manual loops.

```python
m = np.array([[1, 2, 3],
              [4, 5, 6]])
print(m + np.array([10, 20, 30]))   # the small array is added to EVERY row
```

**Output:**
```text
[[11 22 33]
 [14 25 36]]
```

The 1-D array `[10, 20, 30]` was automatically applied to each row of `m`.
Broadcasting powers feature scaling, bias addition in neural networks, and much
more.

## Reshaping and the dot product

```python
print(np.arange(6).reshape(2, 3))    # turn 6 numbers into a 2x3 grid
print("dot:", np.dot(np.array([1, 2, 3]),
                     np.array([4, 5, 6])))   # 1*4 + 2*5 + 3*6 = 32
```

**Output:**
```text
[[0 1 2]
 [3 4 5]]
dot: 32
```

`reshape` rearranges data into the shape a model needs (recall the `reshape(-1, 1)`
from Chapter 1). The **dot product** (`np.dot` or the `@` operator) is the core of
every prediction, as you learned in Chapter 5.

---

# Part B — Pandas: real data tables

## The DataFrame: a spreadsheet in Python

A Pandas **DataFrame** is a table with **named columns** and an **index** (row
labels). It is the single most important object for working with real datasets. A
single column is called a **Series**.

![Anatomy of a Pandas DataFrame: named columns across the top, a row index down the left, and the data values in the grid. Each column is a Series.](assets/images/ch08_dataframe.png)

```python
import pandas as pd            # standard nickname

df = pd.DataFrame({
    "name":  ["Ali", "Sara", "Omar", "Lina", "Zed"],
    "age":   [21, 22, 20, 23, 21],
    "city":  ["Lahore", "Karachi", "Lahore", "Karachi", "Lahore"],
    "score": [72, 88, 56, 91, 64],
})
print(df.head(3))              # show the first 3 rows
print("shape:", df.shape)      # (rows, columns)
```

**Output:**
```text
   name  age     city  score
0   Ali   21   Lahore     72
1  Sara   22  Karachi     88
2  Omar   20   Lahore     56
shape: (5, 4)
```

::: note
In real projects you load data from a file instead of typing it:
`df = pd.read_csv("data.csv")`. Pandas also reads Excel, JSON, SQL databases, and
more. Everything below works the same regardless of where the data came from.
:::

## Inspecting your data (always do this first)

The first thing to do with any dataset is *look at it*:

- `df.head()` / `df.tail()` — first / last rows.
- `df.shape` — (rows, columns).
- `df.info()` — column names, types, and missing-value counts.
- `df.describe()` — summary statistics for numeric columns.

```python
print(df["score"].describe())   # stats for one column
```

**Output:**
```text
count     5.00000
mean     74.20000
std      15.10629
min      56.00000
25%      64.00000
50%      72.00000
max      91.00000
Name: score, dtype: float64
```

Notice Pandas just gave you the **descriptive statistics** from Chapter 6 — count,
mean, std, min, quartiles, max — for free.

## Selecting and filtering data

```python
# Select a single column (a Series)
print(df["name"].tolist())

# Filter rows with a boolean mask, then pick two columns
print(df[df["score"] > 70][["name", "score"]])

# Select by position with iloc, by label with loc
print(df.iloc[0]["name"])       # first row's name
```

**Output:**
```text
['Ali', 'Sara', 'Omar', 'Lina', 'Zed']
   name  score
0   Ali     72
1  Sara     88
3  Lina     91
Ali
```

::: warning
**`loc` vs `iloc`** is a classic confusion. `iloc` uses **integer positions**
(`df.iloc[0]` = the first row). `loc` uses **labels** (`df.loc[0, "name"]` uses the
index label `0` and column name `"name"`). With the default numeric index they look
similar, but they behave very differently once you set a custom index.
:::

## Creating and modifying columns

```python
df["passed"] = df["score"] >= 60     # create a new boolean column
print(df[["name", "passed"]])
```

**Output:**
```text
   name  passed
0   Ali    True
1  Sara    True
2  Omar   False
3  Lina    True
4   Zed    True
```

Creating a column from others is **feature engineering** (Chapter 12) in miniature —
and it's just one line.

## GroupBy: split, apply, combine

**GroupBy** is one of Pandas' most powerful ideas: split the data into groups, apply
a calculation to each group, and combine the results.

![The split-apply-combine pattern of GroupBy: rows are split into groups by a key (here, city), an aggregation is applied to each group (mean score), and the results are combined into a summary.](assets/images/ch08_groupby.png)

```python
print(df.groupby("city")["score"].mean())   # average score per city
```

**Output:**
```text
city
Karachi    89.5
Lahore     64.0
Name: score, dtype: float64
```

In one line we learned that Karachi students in this tiny dataset averaged 89.5 vs
Lahore's 64.0. GroupBy answers "what is the average/total/count *per category*?" —
a question you'll ask constantly.

## Sorting

```python
print(df.sort_values("score", ascending=False)[["name", "score"]])
```

**Output:**
```text
   name  score
3  Lina     91
1  Sara     88
0   Ali     72
4   Zed     64
2  Omar     56
```

## Handling missing values (a first look)

Real data is messy and full of gaps (shown as `NaN` = "Not a Number"). Detecting and
handling these is essential — we cover it deeply in Chapter 10, but here's the core.

```python
import numpy as np
df2 = pd.DataFrame({"x": [1, np.nan, 3],
                    "y": [4, 5, np.nan]})
print(df2.isnull().sum().tolist())   # count missing values per column
print(df2.fillna(0).values.tolist()) # replace missing values with 0
```

**Output:**
```text
[1, 1]
[[1.0, 4.0], [0.0, 5.0], [3.0, 0.0]]
```

- **`isnull().sum()`** counts the gaps in each column (one missing in `x`, one in
  `y`).
- **`fillna(0)`** fills the gaps with 0. Other strategies: fill with the mean/median,
  or drop the rows with `dropna()`. Choosing wisely matters — Chapter 10.

::: keyidea
The Chapter 7 pure-Python "student summariser" took loops and many lines. Here, the
same kind of analysis — averages, filters, top values, per-group stats — took **one
line each** with Pandas, and runs far faster. This is why every ML practitioner
lives in Pandas.
:::

::: tip
**Practical workflow & debugging:** (1) After loading data, *always* run
`df.head()`, `df.info()`, and `df.describe()` before anything else. (2) A
`SettingWithCopyWarning` usually means you filtered then assigned — use `.loc` or
`.copy()` to be explicit. (3) `df.shape` after each step confirms you didn't
accidentally drop rows/columns. (4) Chaining operations is fine, but break long
chains across lines for readability.
:::

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Confusing `loc` and `iloc`.** `iloc` = integer position; `loc` =
label. Mixing them causes wrong rows or `KeyError`.
:::

- **Mistake 2 — Using Python loops over a DataFrame** when a vectorised Pandas/NumPy
  operation would be far faster. Avoid `iterrows()` for big data.
- **Mistake 3 — Forgetting `axis`** in NumPy/Pandas aggregations and getting the
  wrong direction.
- **Mistake 4 — Ignoring `NaN`s.** Missing values silently break calculations and
  models; check with `isnull().sum()` early.
- **Mistake 5 — Modifying a filtered slice** and expecting the original to change
  (or vice versa) — understand views vs copies; use `.copy()` when in doubt.
- **Mistake 6 — Mixed data types in a column** (numbers stored as text) — check
  `df.dtypes`.

## Best practices

- **Inspect first:** `head`, `info`, `describe`, `shape` on every new dataset.
- **Vectorise:** prefer NumPy/Pandas operations over Python loops.
- **Name things clearly:** meaningful column names make analysis self-documenting.
- **Check shapes constantly** when reshaping or joining.
- **Handle missing data deliberately**, not by accident.
- **Keep raw data untouched;** create new columns/DataFrames for transformations.

## Chapter Summary

- **NumPy arrays** store numbers compactly and enable **vectorised** maths (operate
  on whole arrays at once) — far faster than Python lists/loops.
- Key NumPy tools: `array`, `zeros`, `arange`, `linspace`, `shape`, `dtype`, the
  `axis` argument, **boolean masking**, **broadcasting**, `reshape`, and `dot`/`@`.
- **Pandas DataFrames** are labelled tables (columns = **Series**); they're the
  standard for real datasets.
- Core Pandas skills: `read_csv`, `head/info/describe/shape`, selecting columns,
  **boolean filtering**, `loc`/`iloc`, creating columns, **`groupby`** (split-apply-
  combine), `sort_values`, and handling missing values with `isnull`/`fillna`/
  `dropna`.
- These two libraries are the foundation for *all* the data work (Part III) and
  modelling (Parts IV+) that follows.

---

::: {.qband}
Practice Zone — Chapter 8
:::

## Multiple-Choice Questions (MCQs)

**Q1.** The main advantage of a NumPy array over a Python list is:
a) It can hold any type  b) Fast vectorised maths on whole arrays
c) It uses more memory  d) It cannot be indexed

**Q2.** `np.array([[1,2,3],[4,5,6]]).sum(axis=0)` gives:
a) `[6, 15]`  b) `[5, 7, 9]`  c) `21`  d) `[1, 2, 3]`

**Q3.** A single column of a Pandas DataFrame is a:
a) Array  b) Series  c) List  d) Index

**Q4.** `df[df["score"] > 70]` performs:
a) Sorting  b) Boolean filtering  c) Grouping  d) Reshaping

**Q5.** Which selects by *integer position*?
a) `df.loc`  b) `df.iloc`  c) `df.head`  d) `df.groupby`

**Q6.** `NaN` in a DataFrame represents:
a) A string  b) Zero  c) A missing value  d) Negative infinity

**Q7.** `np.arange(0, 10, 2)` produces:
a) `[0,1,2,...,9]`  b) `[0,2,4,6,8]`  c) `[2,4,6,8,10]`  d) `[0,10,2]`

**Q8.** GroupBy follows which pattern?
a) Sort-filter-merge  b) Split-apply-combine  c) Map-reduce-join  d) Read-write-close

### MCQ Answers
**1:** b. **2:** b. **3:** b. **4:** b. **5:** b. **6:** c. **7:** b. **8:** b.

## Interview Questions (with answers)

**Q1. Why use NumPy arrays instead of Python lists for ML?**
*Answer:* NumPy arrays store homogeneous numbers in contiguous memory and support
vectorised operations executed in fast C, making them far faster and more
memory-efficient than lists for numerical work. ML libraries also expect array
inputs.

**Q2. What is broadcasting in NumPy?**
*Answer:* Broadcasting is NumPy's automatic stretching of a smaller array to match
the shape of a larger one during arithmetic, so you can combine arrays of different
(but compatible) shapes without manual loops — e.g. adding a 1-D bias vector to
every row of a 2-D matrix.

**Q3. What is the difference between `loc` and `iloc` in Pandas?**
*Answer:* `loc` selects by labels (index names and column names), while `iloc`
selects by integer positions. With a default numeric index they can look the same,
but they differ once a custom index is set.

**Q4. Explain the split-apply-combine idea behind `groupby`.**
*Answer:* GroupBy splits rows into groups by one or more keys, applies an aggregation
or transformation to each group (e.g. mean, sum, count), and combines the per-group
results into a new structure — answering "what is X per category?".

**Q5. How do you handle missing values in Pandas, and what are the trade-offs?**
*Answer:* Detect with `isnull().sum()`. Options: drop rows/columns with `dropna()`
(simple but loses data), or fill with `fillna()` using a constant, mean, median, or
forward/backward fill (keeps data but introduces assumptions). The right choice
depends on how much is missing and why.

## Scenario-Based Questions (with answers)

**Q1.** *Your code that processes a 5-million-row dataset with a Python `for` loop
takes 20 minutes. How would you speed it up dramatically?*
*Answer:* Replace the loop with vectorised NumPy/Pandas operations (column
arithmetic, boolean masks, `groupby`, built-in aggregations). Vectorisation runs in
optimised C and can be orders of magnitude faster, often turning minutes into
seconds.

**Q2.** *After filtering a DataFrame and assigning to a new column, you get a
`SettingWithCopyWarning` and your change doesn't stick. What's happening?*
*Answer:* You modified a slice that may be a view of the original, so Pandas warns
the operation is ambiguous. Fix it by working on an explicit copy
(`subset = df[mask].copy()`) or by assigning with `.loc` on the original
(`df.loc[mask, "col"] = value`).

**Q3.** *A numeric column won't let you compute a mean and shows dtype `object`.
Why, and how do you fix it?*
*Answer:* The column likely contains numbers stored as text (or stray symbols), so
Pandas treats it as strings. Convert with `pd.to_numeric(df["col"],
errors="coerce")`, which turns valid entries into numbers and invalid ones into
`NaN` to handle separately.

## Logic-Based Questions (with answers)

**Q1.** For `m = [[1,2,3],[4,5,6]]`, why does `m.sum(axis=1)` give `[6, 15]`?
*Answer:* `axis=1` collapses the columns, summing *across* each row: row 0 is
1+2+3=6 and row 1 is 4+5+6=15, giving one value per row.

**Q2.** `a = np.array([1,2,3,4]); print(a[a % 2 == 0])`. What prints and why?
*Answer:* `[2 4]`. The mask `a % 2 == 0` is `[False, True, False, True]`, and boolean
indexing keeps only the elements at `True` positions (the even numbers).

**Q3.** If `df.groupby("city")["score"].mean()` returns two rows, what does that tell
you about the `city` column?
*Answer:* That the `city` column contains exactly two distinct values (groups) — here
Karachi and Lahore — since GroupBy produces one result row per unique group key.

## Practical Questions (with answers)

**Q1.** Write one line to get the average `age` per `city` from `df`.
*Answer:* `df.groupby("city")["age"].mean()`.

**Q2.** Write one line to keep only rows where `age` is 21 and show the `name` and
`score` columns.
*Answer:* `df[df["age"] == 21][["name", "score"]]`.

**Q3.** Using NumPy, create a 3×3 array of all ones and multiply it by 5.
*Answer:* `np.ones((3, 3)) * 5`.

## Long Questions (with answers)

**Q1. Explain vectorisation and broadcasting in NumPy, why they matter for Machine
Learning, and give concrete examples of each.**

*Answer:* **Vectorisation** means applying an operation to an entire array at once
instead of looping element by element. For example, `a * 2` doubles every element of
`a`, and `w * X + b` computes predictions for all data points simultaneously. Because
the loop runs in optimised C inside NumPy rather than in slow Python, vectorised code
is both shorter and dramatically faster — essential when datasets have millions of
rows. **Broadcasting** is the rule that lets NumPy combine arrays of different but
compatible shapes by automatically "stretching" the smaller one. For example, adding
a 1-D array `[10, 20, 30]` to a 2-D matrix adds it to every row, and subtracting a
per-column mean from a whole data matrix standardises features in one line. Together,
vectorisation and broadcasting let ML practitioners express operations on entire
datasets cleanly and run them at near-C speed, which is exactly why NumPy underpins
every major ML library and why GPUs (which excel at such bulk array maths) accelerate
deep learning.

**Q2. Describe the typical first steps of working with a new dataset in Pandas, and
why each step matters.**

*Answer:* The first goal is to *understand* the data before touching a model. Begin
by loading it (`pd.read_csv` or similar). Then run `df.head()` and `df.tail()` to see
real example rows and confirm columns loaded correctly. Use `df.shape` to know how
many rows and columns you have, which sets expectations for everything downstream.
Call `df.info()` to see each column's data type and how many non-null values it has —
this immediately reveals missing data and columns wrongly typed as text. Run
`df.describe()` for summary statistics (count, mean, std, min, quartiles, max) on
numeric columns, which surfaces ranges, scales, and possible outliers. Check
`df.isnull().sum()` to quantify missing values per column so you can plan cleaning.
These steps matter because most real-world ML failures come from data problems —
wrong types, missing values, outliers, or misunderstanding what the columns mean —
not from the algorithm. Inspecting first prevents hours of debugging later and
informs the cleaning, preprocessing, and feature-engineering decisions of Part III.

## Exercises

1. Create a NumPy array of the numbers 1–10 and print: their sum, mean, and only the
   values greater than 5 (use a boolean mask).
2. Make a 2×4 array with `reshape` and print its column sums and row sums.
3. Build a small DataFrame of 5 movies (title, year, rating). Print the average
   rating and the highest-rated movie.
4. From the movies DataFrame, filter movies with rating above 8 and show only their
   titles.
5. Explain `axis=0` vs `axis=1` in your own words with a tiny example.

## Mini-Project

**Project: Redo the Chapter 7 summariser in Pandas.**

1. Take the student/product data from the Chapter 7 mini-project and load it into a
   Pandas DataFrame.
2. Reproduce every result (average, filtering, top item, per-category stats) using
   Pandas — each in one line where possible.
3. Add at least one new computed column (e.g. a "grade" or "discounted price").
4. Compare your line count to the pure-Python version and write 3–4 sentences on what
   Pandas made easier. Save both versions in `my-ml-journey/`.

## Assignments

1. **Coding:** Download any small CSV (e.g. from Kaggle or create one). Load it, run
   `head`, `info`, `describe`, count missing values, and write 5 findings about the
   data in plain English.
2. **Coding:** Using NumPy only, implement standardisation: take an array, subtract
   its mean, divide by its standard deviation (use broadcasting). Confirm the result
   has mean ≈ 0 and std ≈ 1.
3. **Conceptual:** In one page, explain how NumPy and Pandas relate to each other and
   why both are needed, with examples of a task best suited to each.

::: tip
You now have the complete toolkit: maths (Ch 5), statistics (Ch 6), Python (Ch 7),
and data libraries (Ch 8). **Part III** puts them to work on the real, messy job that
consumes most of an ML practitioner's time — cleaning, preparing, and exploring
data.
:::
