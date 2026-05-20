# Introduction to Artificial Intelligence

## Introduction

Imagine you wake up in the morning. Your phone has already silenced the spam
calls. Your email inbox has quietly moved junk mail into a "Spam" folder. When
you open a maps app to check traffic, it predicts you will reach your office in
23 minutes. A video app suggests exactly the kind of videos you enjoy. When you
type a message, your keyboard guesses the next word before you finish.

You did not program any of this. No human sat down and wrote a rule that said
"if this exact email arrives, mark it spam." Instead, **machines learned** to do
these things by looking at huge amounts of past data. This ability of machines
to *seem* intelligent — to make decisions, recognise patterns, understand
language, and improve over time — is called **Artificial Intelligence (AI)**.

This chapter is your gentle first step. We will not write heavy mathematics or
complicated code yet. Our only goals here are simple but important:

- Understand **what AI really is** (and what it is *not*).
- See how AI, Machine Learning, and Deep Learning are related.
- Understand the **big idea** that separates AI from normal computer programs.
- Learn the **types of AI** and the **branches of AI**.
- Write our very first tiny taste of "learning from data" in Python.

::: keyidea
By the end of this chapter you will be able to explain, in your own simple
words, what AI is to a friend who knows nothing about computers — and you will
have run your first line of "learning" code.
:::

## What does "Intelligence" mean?

Before we talk about *artificial* intelligence, let's think about plain
**intelligence**. When we say a person is intelligent, we usually mean they can:

- **Learn** from experience (a child touches a hot cup once and learns not to).
- **Recognise** things (your friend's face in a crowd).
- **Understand** language (you understand this sentence).
- **Solve problems** (finding your way to a new place).
- **Make decisions** (choosing what to eat).
- **Adapt** to new situations.

**Artificial Intelligence** is simply our attempt to give *machines* some of
these same abilities. The word "artificial" just means "made by humans," and
"intelligence" means the abilities listed above. So:

::: note
**Artificial Intelligence (AI)** is the field of building machines and software
that can perform tasks which normally require human intelligence — such as
seeing, understanding language, learning from experience, and making decisions.
:::

Notice the word *normally*. A calculator does maths faster than any human, but
we don't call it "intelligent," because following fixed arithmetic steps is not
something that requires *human-like* thinking. AI is about the harder, fuzzier
tasks — the ones where the "right answer" is not a simple formula.

## Real-world examples of AI around you

AI is not science fiction. It is already part of daily life. Here are everyday
examples, with *what* the AI does and *why* it is considered intelligent.

| Where you see it | What the AI does | Why it needs "intelligence" |
|---|---|---|
| Email spam filter | Decides if an email is junk | Spammers constantly change tricks; fixed rules fail |
| Maps / navigation | Predicts travel time and best route | Traffic changes every minute; it must adapt |
| Online shopping | Recommends products you may like | It must learn *your* taste from your behaviour |
| Video / music apps | Suggests next video or song | Millions of items; must personalise for each user |
| Phone face unlock | Recognises your face | Lighting, angle, beard, glasses all change |
| Voice assistants | Understands spoken commands | Human speech is messy and varied |
| Banks | Flags suspicious (fraud) transactions | Fraud patterns are hidden and change over time |
| Hospitals | Helps detect disease in X-rays | Subtle patterns even doctors can miss |
| Translation apps | Convert one language to another | Languages have grammar, context, and exceptions |

::: tip
A quick test to spot AI: ask *"Would this task be hard to solve with a fixed
list of if-else rules written by a human?"* If yes, it is probably a good
candidate for AI.
:::

## The big picture: AI vs Machine Learning vs Deep Learning

People often mix up three words: **AI**, **Machine Learning (ML)**, and **Deep
Learning (DL)**. They are *not* the same thing — they are nested, like boxes
inside boxes.

![How AI, Machine Learning, and Deep Learning are related. The outer field is the largest; each inner field is a more specific approach.](assets/images/ch01_ai_ml_dl_venn.png)

Let's go from the outside in:

- **Artificial Intelligence (the biggest circle)** — the whole goal of making
  machines act intelligently. This includes *any* method: hand-written rules,
  logic, search, or learning from data.
- **Machine Learning (inside AI)** — *one way* of achieving AI. Instead of a
  human writing the rules, the machine **learns the rules from data**. This is
  the main subject of this book.
- **Deep Learning (inside ML)** — *one kind* of machine learning that uses
  large "neural networks" with many layers. It powers modern image recognition,
  speech, and chatbots like the ones you may have used.

::: keyidea
**AI is the goal. Machine Learning is the most successful way to reach it. Deep
Learning is the most powerful kind of Machine Learning today.**
:::

You will also hear the term **Data Science**. Data Science is the broad practice
of getting useful insights from data; it *overlaps* with ML but also includes
things like reporting, dashboards, and statistics that are not "learning."

## The one idea that changes everything

Here is the single most important idea in this entire book. Understand this and
you understand the heart of Machine Learning.

In **traditional programming**, a human writes the **rules**, gives the computer
some **data**, and the computer produces **answers**:

> Data + Rules → (Computer) → Answers

In **Machine Learning**, we flip it around. We give the computer the **data**
*and* the **answers** (examples), and the computer figures out the **rules** by
itself:

> Data + Answers → (Machine Learning) → Rules (the "model")

![Traditional programming vs Machine Learning. In the classic approach humans write the rules; in machine learning the machine discovers the rules from examples.](assets/images/ch01_traditional_vs_ml.png)

### A simple story to make it stick

Suppose you want a program that decides if a fruit is an **apple** or an
**orange**.

- **Traditional way:** You sit and write rules — *"if colour is red and shape is
  round, it's an apple; if colour is orange and skin is rough, it's an
  orange."* But what about green apples? Or a reddish orange? You will keep
  adding rules forever and still miss cases.
- **Machine Learning way:** You show the computer 10,000 photos, each labelled
  "apple" or "orange." The computer **learns by itself** which features separate
  the two. When a new photo comes, it predicts the answer — and it handles cases
  you never thought of.

::: note
The "rules" that a Machine Learning algorithm discovers are together called the
**model**. The model is the final product of learning — it is the thing that
makes predictions on new data.
:::

## A short history (the one-minute version)

We will study the full, fascinating history in **Chapter 3**. For now, just a
quick timeline so you see AI is not new:

| Year | Milestone |
|---|---|
| 1950 | Alan Turing asks "Can machines think?" and proposes the *Turing Test*. |
| 1956 | The term *Artificial Intelligence* is coined at the Dartmouth workshop. |
| 1980s | "Expert systems" (giant rule-books) become popular, then hit limits. |
| 1997 | IBM's *Deep Blue* beats world chess champion Garry Kasparov. |
| 2012 | Deep learning wins the ImageNet image-recognition contest by a huge margin. |
| 2016 | *AlphaGo* beats the world champion at the game of Go. |
| 2020s | Large Language Models (like ChatGPT-style assistants) reach the public. |

The lesson: AI has had ups and downs (called "AI winters"), but the recent boom
is driven by three things together — **more data**, **more computing power**,
and **better algorithms**.

## Types of AI

There are two common ways to classify AI: by **how capable** it is, and by
**how it works**.

### By capability (the most important classification)

![The three types of AI by capability. Only Narrow AI exists today.](assets/images/ch01_types_of_ai.png)

- **Artificial Narrow Intelligence (ANI)** — also called "Weak AI." It is good
  at **one specific task**. A spam filter cannot drive a car; a chess engine
  cannot translate languages. **Every AI that exists today is Narrow AI** — yes,
  even the most advanced chatbots.
- **Artificial General Intelligence (AGI)** — "Strong AI." A machine that could
  do *any* intellectual task a human can, and switch between them. **This does
  not exist yet.**
- **Artificial Super Intelligence (ASI)** — A hypothetical machine far smarter
  than the best humans at *everything*. This is **purely theoretical** and a
  topic of much debate.

::: warning
A very common mistake (even in the news) is to call today's AI "AGI" or to fear
it as "super intelligence." In reality, all current systems are **Narrow AI** —
extremely good at specific tasks, but with no general understanding.
:::

### By functionality

A second classification describes *how* the system behaves:

- **Reactive machines** — react to the current input only, with no memory of the
  past. *Example:* a classic chess engine.
- **Limited memory** — use recent past data to make decisions. *Example:*
  self-driving cars remembering nearby cars' speeds. **Most modern ML is here.**
- **Theory of mind** — would understand emotions and beliefs of others. *Still
  research.*
- **Self-aware** — would have its own consciousness. *Science fiction for now.*

## The branches (sub-fields) of AI

AI is a big umbrella. Here are its main branches and what each one does. Many
get their own full chapters later in this book.

| Branch | What it does | Covered in |
|---|---|---|
| Machine Learning | Learning patterns from data | Most of this book |
| Deep Learning | ML using deep neural networks | Part VI |
| Natural Language Processing (NLP) | Understanding and generating human language | Chapter 38 |
| Computer Vision | Understanding images and videos | Chapter 40 |
| Robotics | Machines that sense and act in the physical world | (overview here) |
| Expert Systems | Rule-based decision systems (older approach) | Chapter 3 |
| Speech Recognition | Turning speech into text | Chapters 38–39 |
| Reinforcement Learning | Learning by trial, reward, and error | Chapter 31 |

## How does an AI system work? (the high-level pipeline)

Almost every modern AI/ML system follows the same basic flow. Memorise this
shape — you will see it again and again throughout the book.

```text
   ┌─────────┐    ┌──────────────┐    ┌─────────┐    ┌─────────────┐
   │  DATA   │ →  │  ALGORITHM   │ →  │  MODEL  │ →  │ PREDICTIONS │
   │(examples)│   │ (learns from │    │(learned │    │ (on new,    │
   │          │   │   the data)  │    │  rules) │    │ unseen data)│
   └─────────┘    └──────────────┘    └─────────┘    └─────────────┘
                                          ↑                 │
                                          └──── feedback ────┘
                                       (we measure mistakes and improve)
```

1. **Data** — we collect examples (emails labelled spam/not-spam, house prices,
   photos, etc.).
2. **Algorithm** — a learning method (you will learn many) studies the data.
3. **Model** — the learned "rules" produced by the algorithm.
4. **Predictions** — we feed new, unseen data to the model and it answers.
5. **Feedback** — we check how many answers were right and use that to improve.

## A little intuition about "learning" (no scary maths)

How can a machine "learn"? At the simplest level, learning means **adjusting
numbers to reduce mistakes**.

Imagine guessing house prices using only one fact: size in square metres. You
start with a rough guess rule:

```text
predicted_price = w × size + b
```

Here `w` and `b` are just two numbers ("knobs") the machine can turn. At first
they are random, so the guesses are bad. The machine compares its guesses to the
real prices, measures the total mistake (the "error"), and **nudges `w` and `b`
to make the error smaller.** It repeats this thousands of times until the guesses
are good.

That's it. That is the core of most machine learning: *make a guess → measure
the error → adjust to reduce the error → repeat.* We will make this precise (with
real maths) starting in Chapter 5, and you will see it in action in Chapter 17.

## Practical: your very first taste of Machine Learning

Let's *feel* the difference between writing rules and learning rules. We will
predict whether a student **passes** based on **hours studied**.

::: note
You do **not** need to understand every detail yet — Chapter 7 teaches Python
and Chapter 18 explains this exact model. The goal now is just to *see* a
machine learn a rule from data. Install the library first if needed:
`pip install scikit-learn`.
:::

### Step 1 — The "traditional programming" way (human writes the rule)

```python
# We, the humans, invent a rule: pass if the student studied 5 or more hours.
def will_pass_rule(hours):
    if hours >= 5:          # <- this threshold (5) is GUESSED by us
        return "Pass"
    else:
        return "Fail"

print(will_pass_rule(2))    # studied 2 hours
print(will_pass_rule(7))    # studied 7 hours
```

**Output:**
```text
Fail
Pass
```

**What happened?** *We* decided the magic number `5`. If it is wrong, the program
is wrong. We have no idea if 5 is actually the best threshold — we just guessed.

### Step 2 — The "machine learning" way (machine learns the rule from data)

```python
# 1) Import the tools we need.
import numpy as np                              # numpy handles number arrays
from sklearn.linear_model import LogisticRegression  # a simple ML classifier

# 2) Our DATA (examples). Each row is one past student.
#    X = hours studied. We must shape it as a column, hence reshape(-1, 1).
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)

# 3) The ANSWERS for those examples. 0 = Failed, 1 = Passed.
#    Notice we are GIVING the machine the answers, not the rule.
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# 4) Create the model and let it LEARN the rule from X and y.
model = LogisticRegression()
model.fit(X, y)            # <- this single line is "the learning"

# 5) Use the learned model to predict on NEW, unseen students.
new_students = np.array([[2], [7], [4.5]])     # studied 2, 7, and 4.5 hours
predictions = model.predict(new_students)

# 6) Show the results in plain language.
for hours, result in zip(new_students.ravel(), predictions):
    label = "Pass" if result == 1 else "Fail"
    print(f"Studied {hours} hours  ->  {label}")
```

**Output (yours may differ slightly):**
```text
Studied 2.0 hours  ->  Fail
Studied 7.0 hours  ->  Pass
Studied 4.5 hours  ->  Pass
```

Notice something interesting: the machine decided that **4.5 hours is already
enough to pass**. We never told it that — it worked out the best cut-off
(around 4.5) purely from the data. With our hand-written rule in Step 1, 4.5
hours would have been a "Fail" because we *guessed* the cut-off was 5. The
machine found a slightly better boundary on its own.

**Line-by-line explanation:**

- **Lines 2–3:** We import `numpy` (for number arrays) and a ready-made learning
  algorithm called `LogisticRegression`.
- **Line 6 (`X`):** Our input data — hours studied. `reshape(-1, 1)` turns the
  list into a column, which is the shape scikit-learn expects.
- **Line 10 (`y`):** The correct answers (labels). We *hand the machine the
  truth* so it can learn from it.
- **Line 14 (`model.fit`):** The heart of it. The machine looks at `X` and `y`
  and **figures out the rule itself** — including the best cut-off point. We
  never told it "5 hours."
- **Lines 17–18:** We give it brand-new students and ask for predictions.

::: keyidea
The difference is profound. In Step 1 *we* invented the rule "5 hours." In Step
2 the **machine discovered the rule from data**. If we feed it different or more
data, it updates the rule automatically — no human rewriting required. That is
the power of Machine Learning.
:::

::: tip
**Optimization / debugging tips for beginners:**
(1) If you get an error about input shape, remember scikit-learn wants `X` as a
2-D array (rows = examples, columns = features) — that is why we use
`reshape(-1, 1)`. (2) Always print `X.shape` and `y.shape` when confused; they
should have the same number of rows. (3) With so few data points the model is
just a demo — real models need much more data, which we cover in Part III.
:::

## Common mistakes & myths about AI

::: warning
**Myth 1: "AI thinks and understands like a human."** No. Today's AI finds
statistical patterns in data. It has no real understanding, feelings, or common
sense.
:::

- **Myth 2: "AI is always right."** AI can be confidently wrong, especially on
  data different from what it learned on.
- **Myth 3: "More data always means a better model."** Quality matters more than
  quantity. Bad or biased data produces a bad, biased model.
- **Myth 4: "AI will replace all jobs tomorrow."** AI is a tool. It changes
  jobs, automates parts of them, and creates new ones — but today's Narrow AI
  cannot replace general human judgement.
- **Myth 5: "You need to be a maths genius to start."** You need patience and
  curiosity. We build the maths gently, only as much as you need.
- **Mistake (technical):** Confusing AI, ML, and DL. Remember the nested
  circles: every DL is ML, every ML is AI, but not the reverse.

## Best practices & mindset for learning AI

- **Learn the *why*, not just the *what*.** Knowing *why* an algorithm is used
  matters more than memorising its name.
- **Code along.** Type every example yourself. Reading is not the same as doing.
- **Be comfortable being confused.** Confusion is the feeling of learning.
- **Connect everything to a real problem.** Always ask, "What would I use this
  for in the real world?"
- **Respect the data.** Most real ML work is understanding and cleaning data,
  not fancy algorithms.
- **Build a portfolio early.** Even tiny projects (like Section 1.11) count.

## Chapter Summary

- **AI** is the science of making machines do tasks that normally need human
  intelligence (seeing, understanding language, deciding, learning).
- AI is **nested**: Deep Learning ⊂ Machine Learning ⊂ Artificial Intelligence.
- The defining idea of ML: instead of *humans writing rules*, the **machine
  learns the rules (the "model") from data and answers**.
- The basic AI pipeline is **Data → Algorithm → Model → Predictions → Feedback**.
- By capability, AI is **Narrow (today), General (not yet), Super
  (hypothetical)**. Everything in use today is **Narrow AI**.
- "Learning" at its core means **adjusting numbers to reduce mistakes**,
  repeated many times.
- AI is a powerful **tool** — its value and risks depend on how humans use it.

---

::: {.qband}
Practice Zone — Chapter 1
:::

## Multiple-Choice Questions (MCQs)

**Q1.** Which statement best describes Artificial Intelligence?
a) Any program that uses a computer
b) Making machines perform tasks that normally need human intelligence
c) Only robots that look like humans
d) A type of calculator

**Q2.** The correct relationship is:
a) AI ⊂ ML ⊂ Deep Learning
b) Deep Learning ⊂ ML ⊂ AI
c) ML ⊂ AI ⊂ Deep Learning
d) They are all the same thing

**Q3.** In traditional programming, the human provides ___ and the computer
produces ___.
a) data and answers; rules
b) rules and data; answers
c) answers only; rules
d) nothing; everything

**Q4.** Which type of AI exists today?
a) Artificial General Intelligence
b) Artificial Super Intelligence
c) Artificial Narrow Intelligence
d) Self-aware AI

**Q5.** The "rules" learned by a machine learning algorithm are together called:
a) the dataset
b) the model
c) the feedback
d) the pipeline

**Q6.** Which of these is the *best* candidate for a Machine Learning solution?
a) Adding two numbers
b) Detecting spam emails that constantly change tricks
c) Printing "Hello World"
d) Converting kilometres to miles

**Q7.** A self-driving car remembering the speed of nearby cars is an example of:
a) Reactive machine
b) Limited memory
c) Self-aware AI
d) Expert system

### MCQ Answers
**1:** b — AI = doing tasks that normally need human intelligence.
**2:** b — Deep Learning is inside ML, which is inside AI.
**3:** b — humans give rules + data; the computer outputs answers.
**4:** c — only Narrow AI exists today.
**5:** b — the learned rules are the *model*.
**6:** b — spam keeps changing, so fixed rules fail; ML adapts.
**7:** b — using recent past data is *limited memory*.

## Interview Questions (with answers)

**Q1. What is the difference between AI, Machine Learning, and Deep Learning?**
*Answer:* AI is the broad goal of making machines act intelligently using *any*
technique. Machine Learning is a subset of AI where the machine learns patterns
from data instead of following hand-written rules. Deep Learning is a subset of
ML that uses multi-layered neural networks and works very well on images, speech,
and text. Relationship: DL ⊂ ML ⊂ AI.

**Q2. How is Machine Learning different from traditional programming?**
*Answer:* In traditional programming, humans write the rules and the computer
applies them to data to get answers. In ML, we give the computer data *and*
example answers, and it learns the rules (the model) itself. ML shines when the
rules are too complex or change too often to be written by hand.

**Q3. What are the types of AI by capability?**
*Answer:* Narrow AI (good at one task — the only kind today), General AI
(human-level across all tasks — not yet achieved), and Super AI (beyond human —
hypothetical).

**Q4. Give three real-world examples of AI and why they need it.**
*Answer:* Spam filters (spam tricks keep changing), navigation apps (traffic
changes constantly), and product recommendations (must personalise to each user
from behaviour). Each is hard to solve with fixed human-written rules.

**Q5. Is a calculator an example of AI? Why or why not?**
*Answer:* No. A calculator follows fixed arithmetic steps and does not learn,
adapt, or handle ambiguity. AI targets tasks that normally require human-like
judgement.

## Scenario-Based Questions (with answers)

**Q1.** *A company wants to automatically detect angry customer emails so they
can be answered first. A junior developer suggests writing if-else rules that
search for words like "angry" and "terrible." Why might Machine Learning be a
better choice?*
*Answer:* Anger is expressed in countless ways ("this is unacceptable," sarcasm,
ALL CAPS, no keywords at all). A fixed keyword list will miss many cases and
wrongly flag others. ML can learn the subtle patterns of tone from many labelled
examples and adapt as language changes.

**Q2.** *Your friend says, "I built an AI that can play chess perfectly, so AI
has reached human-level general intelligence." How do you respond?*
*Answer:* A chess engine is **Narrow AI** — superb at one task but unable to do
anything else (it cannot hold a conversation or recognise a cat). General
intelligence (AGI) means doing *any* intellectual task a human can; that does not
exist yet.

**Q3.** *A hospital has only 30 labelled X-ray images and wants an AI to detect a
rare disease. What is the main risk?*
*Answer:* Far too little data. The model will likely "memorise" those 30 images
and fail on new patients (poor generalisation), and any bias in the small set
will be amplified. More and more representative data is needed.

## Logic-Based Questions (with answers)

**Q1.** If *every* Deep Learning system is a Machine Learning system, and *every*
Machine Learning system is an AI system, is *every* AI system a Deep Learning
system?
*Answer:* No. The relationship is one-directional (nested). AI also includes
non-learning methods (like hand-written rule systems) that are neither ML nor DL.

**Q2.** A program plays tic-tac-toe using a fixed table of "best moves" written
by a human and never changes. Is it AI? Is it Machine Learning?
*Answer:* It can be called (rule-based) AI because it performs an
intelligence-like task, but it is **not** Machine Learning, because it does not
learn from data — the moves were written by a human.

**Q3.** You feed an ML model only photos of brown dogs and label them "dog."
Logically, what will likely happen when it sees a white dog?
*Answer:* It may fail to recognise the white dog, because it learned that
"brown" is part of being a dog. This shows how biased/incomplete data leads to a
biased model.

## Practical Questions (with answers)

**Q1.** In the Section 1.11 code, why do we write `X.reshape(-1, 1)`?
*Answer:* scikit-learn expects the input `X` to be 2-D (rows = examples, columns
= features). Our hours are a 1-D list, so we reshape it into a single column.
`-1` tells NumPy to figure out the number of rows automatically.

**Q2.** In the same code, what does `model.fit(X, y)` actually do?
*Answer:* It runs the learning process: the algorithm looks at the inputs `X`
and answers `y` and adjusts its internal numbers to best separate "pass" from
"fail." After this line, `model` holds the learned rule.

**Q3.** Modify the idea: how would you change the code so the model predicts a
student who studied **5.5 hours**?
*Answer:* Add `[5.5]` to the `new_students` array, e.g.
`new_students = np.array([[2], [7], [4.5], [5.5]])`, then run the same predict
loop. (Try it — this is your first experiment.)

## Long Questions (with answers)

**Q1. Explain, with a clear example, the fundamental difference between
traditional programming and Machine Learning. Why does this difference make ML
so powerful for real-world problems?**

*Answer:* In traditional programming the flow is **Data + Rules → Answers**: a
human studies the problem, writes explicit rules (code), and the computer applies
them. This works well when the rules are clear and stable — for example,
converting kilometres to miles. But many real problems have rules that are
either too complex or constantly changing. Consider spam detection: spammers
invent new tricks daily, so a human can never finish writing rules.

Machine Learning flips the flow to **Data + Answers → Rules**: we provide many
labelled examples (emails marked spam/not-spam) and the algorithm discovers the
rules itself, producing a *model*. When spammers change tactics, we simply
retrain on new data and the model updates — no human rewriting required.

This is powerful because (a) it handles problems too complex for hand-written
rules, (b) it adapts as the world changes, (c) it can find subtle patterns
humans miss, and (d) it scales to millions of cases. The trade-off is that ML
needs good data and can be wrong in surprising ways — themes we explore
throughout the book.

**Q2. Describe the types of AI by capability and explain why it is incorrect to
call today's most advanced systems "General AI."**

*Answer:* By capability there are three types. **Narrow AI (ANI)** is skilled at
a single task — spam filtering, face unlock, playing Go — and cannot transfer
that skill to unrelated tasks. **General AI (AGI)** would match a human across
*any* intellectual task and switch flexibly between them. **Super AI (ASI)**
would surpass the best humans at everything and is purely hypothetical.

Today's most advanced systems, including powerful chatbots, are still **Narrow
AI**: each is trained for a family of tasks (e.g., predicting text) and lacks
genuine general understanding, common sense, or the ability to autonomously
master truly unrelated domains. They can appear general because language touches
many topics, but they have no real comprehension or goals of their own. Calling
them AGI overstates their abilities and fuels both hype and unfounded fear. The
honest description is: extremely capable, *narrow*, statistical pattern matchers.

## Exercises

1. In your own words (3–4 sentences), explain AI to a 10-year-old child.
2. List five examples of AI you personally used in the last 24 hours. For each,
   write *why* fixed human-written rules would struggle.
3. Draw the nested circles of AI, ML, and DL from memory, and write one sentence
   defining each.
4. Re-draw the "Data + Rules → Answers" vs "Data + Answers → Rules" diagram and
   explain each arrow.
5. Find one news headline about "AI" and decide: is it really about Narrow AI?
   Justify your answer.

## Mini-Project

**Project: "Is it Spam?" — your first thinking-in-ML exercise (no heavy code).**

1. On paper, collect 10 example text messages and label each as "Spam" or "Not
   Spam."
2. As a *human*, write down 3 rules you would use to detect spam (e.g., "contains
   the word 'prize'").
3. Now find at least 2 spam messages from step 1 that your rules would **miss**,
   and 1 normal message your rules would **wrongly flag**.
4. Write a short paragraph: *why* would a Machine Learning approach handle these
   tricky cases better than your hand-written rules?

*Goal:* feel, in your own data, why ML beats fixed rules — before you write a
single line of ML code. (We build the real spam detector in Chapter 49.)

## Assignments

1. **Reading & writing:** Write a one-page essay titled *"Where I see AI in my
   daily life and how it might change my future career."*
2. **Coding warm-up:** Make sure Python is installed
   (`python --version`). Install scikit-learn (`pip install scikit-learn numpy`).
   Type and run the Section 1.11 code yourself. Then change the training data
   `y` so that students need 7 hours to pass, retrain, and report what changes in
   the predictions.
3. **Research:** In 5 bullet points, describe one real company that uses AI, what
   problem it solves, and which *type* of AI (by capability) it is. Cite your
   source.

::: tip
Keep all your answers, code, and projects in a folder called `my-ml-journey/`.
By the end of this book it will be your portfolio.
:::
