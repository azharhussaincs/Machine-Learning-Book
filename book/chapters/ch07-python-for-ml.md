# Python for Machine Learning

## Introduction

Every idea in this book becomes *real* through code — and that code is **Python**.
If you have never programmed before, do not worry. Python was designed to read
almost like English, and this chapter teaches you everything you need from zero. If
you already know some Python, treat this as a focused refresher on the parts that
matter most for Machine Learning.

By the end of this chapter you will:

- Understand **why** Python dominates Machine Learning.
- Set up a proper Python working environment (and know what a "virtual
  environment" is and why it matters).
- Confidently use Python's core building blocks: variables, data structures,
  conditionals, loops, functions, and classes.
- Write clean, ML-friendly Python using **list comprehensions** and **lambda
  functions**.
- Know the **ML library ecosystem** and what each tool is for.
- Process a small dataset with pure Python — and see why libraries make it easier.

::: keyidea
You don't need to be a software engineer to do ML. You need *enough* Python to
load data, transform it, call libraries, and read errors calmly. That is exactly
what this chapter delivers.
:::

## Why Python for Machine Learning?

Other languages can do ML, so why does almost everyone use Python?

- **Simple, readable syntax** — you spend time on ideas, not on fighting the
  language.
- **The richest ML ecosystem** — NumPy, Pandas, scikit-learn, TensorFlow, PyTorch,
  and thousands more, all free.
- **A huge community** — almost any error you hit has already been answered online.
- **"Glue" language** — easily connects data, models, web apps, and the cloud.
- **Interactive tools** — Jupyter notebooks let you run code piece by piece and see
  results instantly, which is perfect for experimenting with data.

::: note
Python itself is actually "slow" compared to languages like C. The trick is that
ML libraries (NumPy, etc.) are written in fast C/C++ under the hood. You write
simple Python; the heavy maths runs at C speed. Best of both worlds.
:::

## Setting up your environment

You need three things: **Python**, a way to install **packages**, and a place to
**write code**.

1. **Install Python (3.10+).** Download from python.org, or use the **Anaconda**
   distribution which bundles Python plus most ML libraries in one installer
   (recommended for beginners).
2. **pip** — Python's package installer. Install any library with, e.g.:
   `pip install numpy pandas scikit-learn matplotlib`.
3. **An editor / notebook** — choose one:
   - **Jupyter Notebook / JupyterLab** — run code in cells; ideal for data work.
   - **VS Code** — a powerful free editor with great Python support.
   - **Google Colab** — free Jupyter notebooks in your browser, with free GPUs (no
     install needed — great if your computer is weak).

### Virtual environments (do this early!)

A **virtual environment** is an isolated, private box of packages for one project.
Without it, different projects fight over library versions and break each other.

```bash
# Create a virtual environment named ".venv"
python -m venv .venv

# Activate it
source .venv/bin/activate      # Mac/Linux
.venv\Scripts\activate         # Windows

# Now install packages safely inside this project only
pip install numpy pandas scikit-learn
```

::: tip
**Always use a virtual environment per project.** It is the single best habit for
avoiding the dreaded "it worked yesterday, now everything's broken" situation. The
`requirements.txt` file lists a project's packages so others can recreate your
environment with `pip install -r requirements.txt`.
:::

## Python basics: the building blocks

We now learn the essential Python you'll use constantly. Type every example
yourself.

### Variables and data types

A **variable** is a named box that stores a value. Python figures out the type
automatically.

```python
name = "Sara"          # str  (text, in quotes)
age = 25               # int  (whole number)
height = 5.6           # float (decimal number)
is_student = True      # bool (True or False)

print(name, "is", age, "years old.")
print("Type of height:", type(height))
```

**Output:**
```text
Sara is 25 years old.
Type of height: <class 'float'>
```

### Operators

```python
a, b = 10, 3
print(a + b)    # 13  addition
print(a - b)    # 7   subtraction
print(a * b)    # 30  multiplication
print(a / b)    # 3.333...  division (always float)
print(a // b)   # 3   floor division (drops the remainder)
print(a % b)    # 1   modulo (the remainder)
print(a ** b)   # 1000  power (10 to the 3rd)
```

Comparison operators (`==`, `!=`, `<`, `>`, `<=`, `>=`) return `True`/`False`, and
logical operators (`and`, `or`, `not`) combine conditions.

### Strings

```python
text = "Machine Learning"
print(len(text))             # 16  (number of characters)
print(text.upper())          # MACHINE LEARNING
print(text.lower())          # machine learning
print(text.replace("Machine", "Deep"))   # Deep Learning
print(text[0])               # M  (first character, index 0)
print(text[-1])              # g  (last character)
print(text[0:7])             # Machine  (slice: characters 0–6)

# f-strings: the modern way to build text (note the f before the quote)
score = 92.5
print(f"Your score is {score}% — well done!")
```

**Output:**
```text
16
MACHINE LEARNING
machine learning
Deep Learning
M
g
Machine
Your score is 92.5% — well done!
```

### Data structures: lists, tuples, dictionaries, sets

These four containers hold collections of values. Choosing the right one is a key
skill.

![Python's four core collections. Lists are ordered and changeable; tuples are ordered and fixed; sets hold unique unordered items; dictionaries store key→value pairs.](assets/images/ch07_data_structures.png)

**List** — an *ordered, changeable* collection (the workhorse):

```python
scores = [85, 90, 78, 92]
scores.append(100)        # add to the end -> [85, 90, 78, 92, 100]
scores[0] = 88            # change the first item
print(scores[1])          # 90  (access by index)
print(scores[1:3])        # [90, 78]  (slice)
print(len(scores))        # 5
print(sum(scores), max(scores), min(scores))  # 448 100 78
```

**Tuple** — an *ordered, UNchangeable* collection (use for fixed data):

```python
point = (3, 4)            # cannot be modified after creation
print(point[0])           # 3
```

**Dictionary** — *key → value* pairs (like a labelled lookup table):

```python
student = {"name": "Ali", "age": 21, "grade": "A"}
print(student["name"])    # Ali  (look up by key)
student["age"] = 22       # update a value
student["city"] = "Lahore"  # add a new key
print(student.keys())     # dict_keys(['name', 'age', 'grade', 'city'])
```

**Set** — an *unordered* collection of *unique* items (great for removing
duplicates):

```python
nums = [1, 2, 2, 3, 3, 3]
unique = set(nums)        # {1, 2, 3}
print(unique)
```

::: warning
**Common beginner trap:** Python indexing starts at **0**, not 1. The first item is
`list[0]`, and `list[-1]` is the last. Forgetting this causes endless
"off-by-one" and "index out of range" errors.
:::

### Conditionals: making decisions

```python
score = 75

if score >= 90:
    grade = "A"
elif score >= 70:        # "elif" = "else if"
    grade = "B"
else:
    grade = "C"

print("Grade:", grade)   # Grade: B
```

::: warning
**Indentation is not optional in Python — it *is* the syntax.** The spaces before
`grade = "A"` tell Python that line belongs inside the `if`. Use 4 spaces
consistently. Wrong indentation is the #1 beginner error.
:::

### Loops: repeating work

```python
# for loop: do something for each item
for s in [85, 90, 78]:
    print("Score:", s)

# range(): generate a sequence of numbers
for i in range(3):       # 0, 1, 2
    print("i =", i)

# while loop: repeat while a condition is true
count = 0
while count < 3:
    print("count =", count)
    count += 1           # same as count = count + 1
```

### Functions: reusable blocks of code

A **function** packages code so you can reuse it. This is the heart of clean
programming.

```python
def average(numbers):
    """Return the average of a list of numbers."""   # docstring
    return sum(numbers) / len(numbers)

# default arguments
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

print(average([10, 20, 30]))     # 20.0
print(greet("Sara"))             # Hello, Sara!
print(greet("Ali", "Welcome"))   # Welcome, Ali!
```

### List comprehensions: Python's superpower for data

A **list comprehension** builds a new list in one readable line. You will see these
*everywhere* in ML code.

```python
nums = [1, 2, 3, 4, 5]

squares = [x ** 2 for x in nums]              # [1, 4, 9, 16, 25]
evens = [x for x in nums if x % 2 == 0]       # [2, 4]  (with a filter)
labels = ["pass" if x >= 3 else "fail" for x in nums]  # transform each item

print(squares)
print(evens)
print(labels)
```

**Output:**
```text
[1, 4, 9, 16, 25]
[2, 4]
['fail', 'fail', 'pass', 'pass', 'pass']
```

::: keyidea
Read a comprehension as: *"give me `expression` for each `item` in `collection`
(optionally `if condition`)."* This single pattern replaces many clumsy loops and
is core to writing clean, Pythonic ML code.
:::

### Lambda functions: tiny one-line functions

A **lambda** is a small anonymous function, handy for quick operations (e.g.
sorting or with Pandas).

```python
double = lambda x: x * 2
print(double(5))         # 10

# common use: sorting by a custom key
people = [("Ali", 30), ("Sara", 25), ("Omar", 35)]
people.sort(key=lambda person: person[1])   # sort by age (index 1)
print(people)            # [('Sara', 25), ('Ali', 30), ('Omar', 35)]
```

### Error handling: failing gracefully

```python
def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        print("Cannot divide by zero!")
        return None

print(safe_divide(10, 2))   # 5.0
print(safe_divide(10, 0))   # prints the warning, returns None
```

### Classes: bundling data and behaviour (a gentle intro)

A **class** is a blueprint for creating objects that bundle data (attributes) and
actions (methods). ML libraries are built from classes — when you write
`model = LogisticRegression()`, you are creating an *object* from a class.

```python
class Student:
    def __init__(self, name, score):   # the constructor (runs on creation)
        self.name = name               # attribute
        self.score = score

    def has_passed(self):              # a method (an action)
        return self.score >= 50

s = Student("Sara", 72)
print(s.name)            # Sara
print(s.has_passed())    # True
```

::: note
You will mostly *use* classes from libraries rather than write your own at first.
But understanding `self`, `__init__`, attributes, and methods makes library
documentation (and your own future code) far less mysterious.
:::

## The Machine Learning library ecosystem

You will not build ML from scratch every time — you'll stand on the shoulders of
powerful libraries. Here is the map of the tools used throughout this book.

![The Python Machine Learning ecosystem, grouped by job: data handling, visualisation, classic ML, deep learning, and deployment.](assets/images/ch07_ecosystem.png)

| Library | What it does | Used in |
|---|---|---|
| **NumPy** | Fast arrays and maths | Everywhere (Ch 8) |
| **Pandas** | Tables/dataframes, data wrangling | Data work (Ch 8–15) |
| **Matplotlib / Seaborn** | Charts and plots | Visualisation (Ch 14) |
| **scikit-learn** | Classic ML algorithms + tools | Parts IV–V |
| **TensorFlow / Keras** | Deep learning (Google) | Part VI |
| **PyTorch** | Deep learning (Meta), research favourite | Part VI |
| **NLTK / spaCy / Hugging Face** | Natural language processing | Part VII |
| **OpenCV** | Computer vision / image processing | Ch 40 |
| **Flask / FastAPI / Streamlit** | Turning models into apps & APIs | Part VIII |

::: tip
Don't try to learn all of these now. We introduce each exactly when you need it.
For the next several chapters, **NumPy** and **Pandas** are your bread and butter.
:::

## Practical: process data with pure Python

Let's tie it together. Below we analyse some student data using *only* core Python —
no libraries — so you appreciate both the skill and (later) why libraries help.

```python
# A small "dataset": list of dictionaries (rows)
students = [
    {"name": "Ali",  "score": 72, "hours": 5},
    {"name": "Sara", "score": 88, "hours": 8},
    {"name": "Omar", "score": 56, "hours": 3},
    {"name": "Lina", "score": 91, "hours": 9},
    {"name": "Zed",  "score": 64, "hours": 4},
]

# 1) Average score (list comprehension + sum)
scores = [s["score"] for s in students]
avg = sum(scores) / len(scores)
print(f"Average score: {avg:.1f}")

# 2) Who passed (score >= 60)?
passed = [s["name"] for s in students if s["score"] >= 60]
print("Passed:", passed)

# 3) Top student (max by score, using a lambda key)
top = max(students, key=lambda s: s["score"])
print(f"Top student: {top['name']} ({top['score']})")

# 4) Add a "result" field to each student
for s in students:
    s["result"] = "Pass" if s["score"] >= 60 else "Fail"
print(students[2])   # show Omar's updated record
```

**Output:**
```text
Average score: 74.2
Passed: ['Ali', 'Sara', 'Lina', 'Zed']
Top student: Lina (91)
{'name': 'Omar', 'score': 56, 'hours': 3, 'result': 'Fail'}
```

### Explanation

- **(1)** We pull all scores with a list comprehension, then average them.
- **(2)** A comprehension *with a filter* keeps only names of students who passed.
- **(3)** `max(..., key=lambda s: s["score"])` finds the record with the highest
  score — the lambda tells `max` *what* to compare.
- **(4)** A loop adds a new computed field to every record.

::: keyidea
This is exactly the kind of work ML requires: load records, filter, transform,
summarise. In Chapter 8 you'll do all of this in *one or two lines* with Pandas —
but understanding the pure-Python version makes Pandas feel like magic instead of
mystery.
:::

::: tip
**Debugging tips for beginners:** (1) Read errors **bottom-up** — the last line
names the error type (e.g. `KeyError`, `IndexError`). (2) `print()` your variables
liberally to see what they actually contain. (3) Check types with `type(x)` when
something behaves oddly. (4) Most errors are typos, wrong indentation, or a 0-vs-1
index mistake.
:::

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Inconsistent indentation** (mixing tabs and spaces). Pick 4 spaces
and never mix. This causes `IndentationError`.
:::

- **Mistake 2 — Off-by-one / index errors** from forgetting indexing starts at 0.
- **Mistake 3 — Using `=` (assignment) when you mean `==` (comparison)** in an `if`.
- **Mistake 4 — Modifying a list while looping over it** — leads to skipped items;
  loop over a copy instead.
- **Mistake 5 — Installing packages globally** instead of in a virtual environment,
  causing version conflicts.
- **Mistake 6 — Naming a file `random.py` or `numpy.py`** — it shadows the real
  library and breaks imports.

## Best practices

- **Use virtual environments** for every project.
- **Write small functions** with clear names; avoid giant blocks of code.
- **Use f-strings** for readable text formatting.
- **Prefer list comprehensions** over manual loops for simple transformations.
- **Comment the *why*, not the obvious *what*.**
- **Use a notebook** (Jupyter/Colab) while exploring data; move to `.py` files for
  reusable code.
- **Keep a `requirements.txt`** so others can reproduce your environment.

## Chapter Summary

- **Python** dominates ML for its readability, vast free ecosystem, community, and
  interactive tools — while heavy maths runs at C speed inside libraries.
- Set up with **Python 3.10+**, **pip**, a **virtual environment**, and a notebook
  or editor.
- Core building blocks: **variables/types**, **operators**, **strings**, and four
  collections — **list** (ordered, changeable), **tuple** (ordered, fixed),
  **dict** (key→value), **set** (unique, unordered).
- Control flow uses **if/elif/else** and **for/while** loops; **indentation is
  syntax**.
- **Functions** make code reusable; **list comprehensions** and **lambdas** make
  data transformations concise; **try/except** handles errors gracefully.
- **Classes** bundle data + behaviour — the foundation of every ML library object.
- The **ML ecosystem**: NumPy, Pandas, Matplotlib/Seaborn, scikit-learn,
  TensorFlow/PyTorch, NLP and CV libraries, and deployment tools.

---

::: {.qband}
Practice Zone — Chapter 7
:::

## Multiple-Choice Questions (MCQs)

**Q1.** In Python, indexing of a list starts at:
a) 1  b) 0  c) −1  d) Depends on the list

**Q2.** Which data structure stores key→value pairs?
a) List  b) Tuple  c) Dictionary  d) Set

**Q3.** Which collection automatically removes duplicates?
a) List  b) Tuple  c) Dictionary  d) Set

**Q4.** What does `[x*2 for x in range(3)]` produce?
a) `[0, 2, 4]`  b) `[2, 4, 6]`  c) `[0, 1, 2]`  d) `[1, 2, 3]`

**Q5.** Which is an *immutable* (unchangeable) ordered collection?
a) List  b) Tuple  c) Dictionary  d) Set

**Q6.** What is the main purpose of a virtual environment?
a) Make code run faster  b) Isolate a project's package versions
c) Replace pip  d) Write documentation

**Q7.** `10 % 3` evaluates to:
a) 3  b) 3.33  c) 1  d) 0

**Q8.** Which library is the standard for fast numerical arrays in Python?
a) Pandas  b) NumPy  c) Flask  d) NLTK

### MCQ Answers
**1:** b. **2:** c. **3:** d. **4:** a (0×2, 1×2, 2×2). **5:** b. **6:** b.
**7:** c. **8:** b.

## Interview Questions (with answers)

**Q1. Why is Python so popular for Machine Learning?**
*Answer:* Readable syntax lets you focus on ideas; it has the richest free ML
ecosystem (NumPy, Pandas, scikit-learn, TensorFlow, PyTorch); a massive community;
it acts as glue between data, models, and apps; and interactive notebooks make
experimentation easy. Heavy computation runs in fast C under the hood.

**Q2. What is the difference between a list and a tuple?**
*Answer:* Both are ordered collections. A list is mutable (you can add, remove, or
change items) and uses `[]`. A tuple is immutable (fixed once created) and uses
`()`. Use tuples for fixed data (like coordinates) and as dictionary keys; use
lists when contents change.

**Q3. What is a list comprehension and why use it?**
*Answer:* It's a concise one-line way to build a list by transforming and/or
filtering an iterable, e.g. `[x*2 for x in nums if x > 0]`. It's more readable and
often faster than an equivalent for-loop, and is idiomatic in ML data code.

**Q4. What is a virtual environment and why is it important?**
*Answer:* An isolated directory containing a project's own Python packages. It
prevents version conflicts between projects and makes environments reproducible
(via `requirements.txt`), avoiding "works on my machine" problems.

**Q5. What does `self` mean in a Python class?**
*Answer:* `self` refers to the specific instance (object) the method is operating
on. It lets methods access and modify that object's own attributes. It's passed
automatically when you call a method on an object.

## Scenario-Based Questions (with answers)

**Q1.** *You install a new library for one project and suddenly another project
breaks with version errors. What practice would have prevented this?*
*Answer:* Using a separate **virtual environment** per project. Each project then
has its own isolated package versions, so installing or upgrading a library in one
cannot affect another.

**Q2.** *A teammate's script fails on your computer with "ModuleNotFoundError." What
is likely missing and how do you fix it cleanly?*
*Answer:* The required packages aren't installed in your environment. The clean fix
is a `requirements.txt` in the project; run `pip install -r requirements.txt` inside
a fresh virtual environment to install exactly the needed versions.

**Q3.** *You need to remove duplicate user IDs from a list of 1 million entries
quickly. Which data structure helps and why?*
*Answer:* Convert the list to a **set**, which stores only unique items, then back
to a list if needed: `list(set(ids))`. Sets handle uniqueness efficiently.

## Logic-Based Questions (with answers)

**Q1.** What will `print([x for x in range(6) if x % 2 == 1])` output, and why?
*Answer:* `[1, 3, 5]`. `range(6)` is 0–5; the filter keeps only odd numbers (those
with remainder 1 when divided by 2).

**Q2.** Why does modifying a list while iterating over it cause bugs?
*Answer:* Removing/adding items shifts the indices during iteration, so the loop can
skip elements or process them twice. The safe pattern is to iterate over a copy
(`for x in mylist[:]:`) or build a new list with a comprehension.

**Q3.** `a = [1, 2, 3]; b = a; b.append(4)`. What is `a` now, and what does this
teach about lists?
*Answer:* `a` is `[1, 2, 3, 4]`. `b = a` makes `b` point to the *same* list object,
not a copy, so changes through `b` affect `a`. To copy, use `b = a.copy()` or
`b = a[:]`.

## Practical Questions (with answers)

**Q1.** In the pure-Python practical, how does `max(students, key=lambda s:
s["score"])` work?
*Answer:* `max` compares the items in `students`; the `key` lambda tells it to
compare each student by their `"score"` value, so it returns the whole student
dictionary that has the highest score.

**Q2.** Rewrite "get the names of students who scored at least 80" as a list
comprehension.
*Answer:* `[s["name"] for s in students if s["score"] >= 80]`.

**Q3.** Write a function `is_even(n)` that returns `True` if `n` is even.
*Answer:*
```python
def is_even(n):
    return n % 2 == 0
```

## Long Questions (with answers)

**Q1. Explain Python's four core data structures (list, tuple, dictionary, set),
including when to use each, with examples.**

*Answer:* A **list** is an ordered, changeable (mutable) collection written with
`[]`, e.g. `scores = [85, 90, 78]`; use it for sequences whose contents may change,
like a growing list of predictions. A **tuple** is an ordered but unchangeable
(immutable) collection written with `()`, e.g. `point = (3, 4)`; use it for fixed
groupings such as coordinates or returning multiple values from a function, and
because immutability lets tuples be used as dictionary keys. A **dictionary** stores
key→value pairs written with `{key: value}`, e.g. `student = {"name": "Ali", "age":
21}`; use it for fast lookups by a meaningful key, like mapping a feature name to
its value. A **set** is an unordered collection of unique items written with `{}` or
`set()`, e.g. `set([1, 2, 2, 3]) == {1, 2, 3}`; use it to remove duplicates or test
membership quickly. Choosing correctly affects both clarity and performance: lists
for ordered changeable data, tuples for fixed data, dictionaries for labelled
lookups, and sets for uniqueness.

**Q2. Describe the Python Machine Learning ecosystem and how the libraries fit
together in a typical project workflow.**

*Answer:* A typical ML project flows through several libraries, each specialised for
a job. First, **Pandas** loads and wrangles tabular data (cleaning, filtering,
joining), built on top of **NumPy**, which provides the fast array maths underneath
nearly everything. During exploration, **Matplotlib** and **Seaborn** create charts
to understand the data. For modelling classic algorithms (regression, trees, SVMs,
clustering) and utilities like train/test splitting and metrics, **scikit-learn** is
the standard toolkit. When problems need deep learning — images, text, audio —
**TensorFlow/Keras** or **PyTorch** build and train neural networks, often using
domain libraries like **OpenCV** (vision), **NLTK/spaCy**, and **Hugging Face
Transformers** (language). Finally, to deliver the model to users, **Flask**,
**FastAPI**, or **Streamlit** wrap it in an API or web app. So the pieces connect as
a pipeline: NumPy/Pandas for data → Matplotlib/Seaborn to explore → scikit-learn or
TensorFlow/PyTorch to model → Flask/FastAPI/Streamlit to deploy. Mastering this
ecosystem means knowing not every detail of each library, but *which tool does which
job* and how to pass data between them.

## Exercises

1. Create a dictionary describing yourself (name, age, city, hobbies as a list) and
   print each value.
2. Write a list comprehension that produces the squares of even numbers from 1 to
   10.
3. Write a function `grade(score)` that returns "A", "B", "C", or "F" using
   if/elif/else.
4. Given `nums = [4, 1, 4, 2, 1, 3]`, print the unique values and the count of
   unique values.
5. Explain in your own words why indentation matters in Python.

## Mini-Project

**Project: A pure-Python data summariser.**

1. Make a list of at least 8 dictionaries representing products (name, price,
   category, in_stock).
2. Write functions to: (a) compute the average price, (b) list all product names in
   a given category, (c) find the most expensive in-stock product.
3. Use list comprehensions and a lambda where appropriate.
4. Print a neat summary report. *(In Chapter 8 you'll redo this with Pandas in a
   fraction of the code — keep this version to compare.)*

## Assignments

1. **Setup:** Install Python and create a virtual environment for this book's
   exercises. Install `numpy pandas scikit-learn matplotlib` inside it and save a
   `requirements.txt`. Write down the commands you used.
2. **Coding:** Write a program that reads a list of numbers, then prints the mean,
   the max, the min, and a new list containing only the numbers above the mean
   (use a list comprehension).
3. **Conceptual:** In half a page, explain the difference between a list and a
   dictionary, and give two real ML situations where each is the better choice.

::: tip
You now speak Python. Next, in Chapter 8, you'll meet **NumPy** and **Pandas** — the
two libraries you will use in literally every remaining chapter. The pure-Python
skills here are the foundation that makes those libraries click.
:::
