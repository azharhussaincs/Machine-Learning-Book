# Natural Language Processing (NLP)

## Introduction

Welcome to **Part VII**, where we apply everything from Parts I–VI to real-world domains.
We begin with **Natural Language Processing (NLP)** — teaching machines to understand and
generate **human language**. NLP powers search engines, spam filters, translation, voice
assistants, sentiment analysis, and the chatbots we'll explore in Chapter 39.

Language is *hard* for computers. It's full of **ambiguity** ("I saw her duck" — a bird or
a dodge?), **context** (sarcasm, idioms), and endless variation. The story of NLP is the
story of better and better ways to turn messy text into numbers a model can learn from.

::: keyidea
Models only understand numbers, so the central challenge of NLP is **turning text into
meaningful numbers** (vectors). The history of NLP is a progression of better
representations: **counts → TF-IDF → word embeddings → contextual embeddings
(Transformers)** — each capturing more meaning than the last.
:::

By the end of this chapter you will be able to:

- Apply **text preprocessing**: tokenization, stopwords, stemming, lemmatization.
- Represent text with **Bag of Words**, **TF-IDF**, and **word embeddings**.
- Understand the modern NLP pipeline and common tasks.
- Build a **text classifier** with scikit-learn.

## Text preprocessing

Raw text must be cleaned and standardised first (echoing Chapters 10–12 for text):

- **Tokenization** — split text into units (words or sub-words). "I love NLP" → ["I",
  "love", "NLP"].
- **Lowercasing** — "Movie" and "movie" become the same.
- **Stopword removal** — drop very common, low-information words ("the", "is", "and").
- **Stemming** — chop words to a crude root ("running" → "run", "studies" → "studi").
  Fast but rough.
- **Lemmatization** — reduce to the proper dictionary form ("better" → "good", "studies" →
  "study"). Smarter but slower.
- **Removing punctuation/numbers** as needed.

(Libraries: **NLTK** and **spaCy** — `pip install nltk spacy`.)

## Representing text as numbers

![The evolution of text representation: from one-hot/Bag-of-Words (sparse counts), to TF-IDF (weighted counts), to dense word embeddings (meaning captured in geometry), to contextual embeddings from Transformers (meaning depends on context).](assets/images/ch38_representations.png)

### Bag of Words (BoW)

The simplest representation: count how often each vocabulary word appears in a document,
ignoring order. "the cat sat" and "sat the cat" get the *same* vector. Simple but loses
word order and treats all words as equally important.

### TF-IDF

**TF-IDF (Term Frequency–Inverse Document Frequency)** improves BoW by **down-weighting
common words** and **up-weighting distinctive ones**. A word that appears in *this*
document but rarely in others is informative; a word in *every* document (like "the") is
not.

<div class="equation"><img class="eq" src="assets/images/eq_ch38_tfidf.png" alt="TF-IDF"></div>

where `tf` is how often the term appears in the document, `N` is the number of documents,
and `df` is how many documents contain the term. TF-IDF is a fast, strong baseline for text
classification (it powered classic spam filters and search).

### Word embeddings

BoW/TF-IDF don't capture **meaning** — "good" and "great" are as unrelated as "good" and
"car". **Word embeddings** (Word2Vec, GloVe) fix this by mapping each word to a **dense
vector** where *similar words sit close together* in space, learned from how words
co-occur. Famously, the geometry captures relationships: **king − man + woman ≈ queen**.

![Word embeddings place words in a space where meaning is geometry: similar words cluster, and relationships become directions (king − man + woman ≈ queen). Models learn these vectors from huge text corpora.](assets/images/ch38_embeddings.png)

### Contextual embeddings

Word2Vec gives each word *one* fixed vector — but "bank" means different things in "river
bank" vs "money bank". **Contextual embeddings** from Transformers (BERT, Chapter 37)
produce a *different* vector for a word depending on its **context**, capturing far more
meaning. These power modern NLP.

## The NLP pipeline and common tasks

![A typical NLP pipeline: raw text → preprocessing (clean, tokenize) → representation (TF-IDF or embeddings) → model → task output (e.g. sentiment, entities, translation).](assets/images/ch38_pipeline.png)

Common NLP **tasks**:

- **Text classification / sentiment analysis** — categorise text (spam, positive/negative).
- **Named-Entity Recognition (NER)** — find names, places, dates.
- **Machine translation** — language to language.
- **Summarisation, question answering, text generation** — increasingly done by LLMs
  (Chapter 39).

## Practical: a text classifier with scikit-learn

Let's build a sentiment classifier using **TF-IDF + logistic regression**.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import numpy as np

pos = ["i love this movie", "great film highly recommend", "wonderful acting and story",
       "an amazing and beautiful experience", "best movie ever fantastic",
       "truly enjoyed every minute", "brilliant and moving masterpiece",
       "so good i watched it twice"]
neg = ["i hate this movie", "terrible film waste of time", "boring and badly acted",
       "an awful and dull experience", "worst movie ever horrible",
       "i regret watching this", "poorly made and disappointing", "so bad i turned it off"]
texts = pos + neg; labels = [1] * len(pos) + [0] * len(neg)   # 1 = positive, 0 = negative

# TF-IDF turns text into weighted-count vectors
tfidf = TfidfVectorizer(); X = tfidf.fit_transform(texts)
print("vocabulary size:", len(tfidf.vocabulary_))
print("TF-IDF matrix shape:", X.shape)

# Pipeline: TF-IDF -> logistic regression
clf = make_pipeline(TfidfVectorizer(), LogisticRegression()).fit(texts, labels)
tests = ["this film is wonderful", "what a boring waste"]
print("predictions:", clf.predict(tests).tolist(), "(1=pos, 0=neg)")
print("prob positive:", np.round(clf.predict_proba(tests)[:, 1], 3).tolist())
```

**Output:**
```text
vocabulary size: 50
TF-IDF matrix shape: (16, 50)
predictions: [1, 0] (1=pos, 0=neg)
prob positive: [0.519, 0.43]
```

### Explanation

- **`TfidfVectorizer`** built a 50-word vocabulary and turned the 16 sentences into a
  16×50 TF-IDF matrix — text became numbers a model can learn from.
- The classifier correctly labelled **"this film is wonderful" → positive** and **"what a
  boring waste" → negative**. (Confidence is modest because we trained on only 16 tiny
  sentences; real datasets give far sharper probabilities.)
- This **TF-IDF + linear model** pipeline is a genuinely strong, fast baseline for text —
  often surprisingly competitive, and the classic approach before Transformers.

::: keyidea
Notice the pattern: **clean text → represent as numbers (TF-IDF) → standard ML model**.
Once text is vectorised, all of Part IV applies. The leap to modern NLP is simply using a
*much better representation* (contextual Transformer embeddings) — but the core idea of
"turn language into meaningful numbers" never changes.
:::

::: tip
**Practical & debugging tips:** (1) Start with **TF-IDF + Logistic Regression / Naive
Bayes** (Chapter 20) — a fast, strong baseline. (2) Tune `TfidfVectorizer` (`ngram_range`
for word pairs, `min_df`/`max_df`, `stop_words`). (3) For deeper meaning, use **pretrained
embeddings** or **fine-tune a Transformer** (Hugging Face, Chapter 37/39). (4) Always handle
text cleaning consistently between train and inference. (5) Watch for class imbalance and
use appropriate metrics (Chapter 25). (6) `pip install nltk spacy` for richer preprocessing.
:::

## The evolution of NLP (one paragraph)

NLP went through eras: **rule-based** systems (hand-written grammar rules — brittle),
**statistical** methods (n-grams, Naive Bayes, TF-IDF — the workhorses of the 2000s),
**word embeddings** (Word2Vec/GloVe, ~2013 — meaning as geometry), and finally
**Transformers** (2017+ — contextual understanding, LLMs). Each era captured more meaning;
today, pretrained Transformers dominate.

## Advantages, disadvantages, and use cases

| Approach | Strength | Weakness |
|---|---|---|
| BoW / TF-IDF | Fast, simple, strong baseline | Ignores order & meaning |
| Word embeddings | Capture word similarity | One vector per word (no context) |
| Transformers (BERT/GPT) | Deep contextual understanding | Heavy compute, data-hungry |

**Use cases:** spam/sentiment classification, search & information retrieval, chatbots,
translation, NER, summarisation, content moderation, and voice assistants.

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Inconsistent preprocessing** between training and inference (e.g. different
tokenization), which silently breaks the model. Apply the *same* pipeline to both.
:::

- **Mistake 2 — Using BoW/TF-IDF and expecting it to understand meaning or context** — it
  only counts words.
- **Mistake 3 — Forgetting to remove or down-weight stopwords** when using counts.
- **Mistake 4 — Over-stemming** (turning words into unreadable roots that lose meaning).
- **Mistake 5 — Training huge Transformers from scratch** for small tasks instead of
  fine-tuning pretrained ones.
- **Mistake 6 — Ignoring class imbalance** in text datasets.

## Best practices

- **Start with TF-IDF + a linear/Naive Bayes baseline.**
- **Keep preprocessing identical** across train and inference (use a pipeline).
- **Use pretrained embeddings / Transformers** for tasks needing real understanding.
- **Tune n-grams and vocabulary** settings.
- **Evaluate with the right metrics** (precision/recall/F1) for imbalanced text.

## Chapter Summary

- **NLP** teaches machines to understand and generate human language; the core challenge is
  **turning text into meaningful numbers**.
- **Preprocessing:** tokenization, lowercasing, stopword removal, stemming/lemmatization.
- **Representations** evolved: **Bag of Words** (counts) → **TF-IDF** (weighted counts,
  down-weighting common words) → **word embeddings** (dense vectors where similar words are
  close; king−man+woman≈queen) → **contextual embeddings** from Transformers (meaning
  depends on context).
- The pipeline is **text → preprocess → represent → model → task**; tasks include
  classification, sentiment, NER, translation, summarisation, and QA.
- We built a **TF-IDF + logistic regression** sentiment classifier that correctly labelled
  new sentences — a fast, strong baseline; modern NLP swaps in Transformer representations.

---

::: {.qband}
Practice Zone — Chapter 38
:::

## Multiple-Choice Questions (MCQs)

**Q1.** The core challenge of NLP is:
a) Faster CPUs  b) Turning text into meaningful numbers  c) Drawing charts  d) Storing files

**Q2.** Splitting text into words or sub-words is called:
a) Stemming  b) Tokenization  c) Lemmatization  d) Embedding

**Q3.** TF-IDF down-weights words that:
a) Are rare  b) Appear in many documents (common)  c) Are long  d) Are capitalised

**Q4.** Word embeddings place similar words:
a) Far apart  b) Close together in vector space  c) At the origin  d) Randomly

**Q5.** "king − man + woman ≈ queen" demonstrates:
a) TF-IDF  b) Embedding geometry capturing relationships  c) Stopword removal  d) Tokenization

**Q6.** Contextual embeddings (BERT) differ from Word2Vec because they:
a) Are faster  b) Give a word different vectors depending on context  c) Ignore context
d) Use no training

**Q7.** A fast, strong baseline for text classification is:
a) CNN from scratch  b) TF-IDF + Logistic Regression/Naive Bayes  c) KMeans  d) PCA

**Q8.** Bag of Words ignores:
a) Word counts  b) Word order  c) Vocabulary  d) Documents

### MCQ Answers
**1:** b. **2:** b. **3:** b. **4:** b. **5:** b. **6:** b. **7:** b. **8:** b.

## Interview Questions (with answers)

**Q1. How do you turn text into features a model can use?**
*Answer:* Preprocess (tokenize, lowercase, remove stopwords, stem/lemmatize), then represent
numerically: Bag of Words (counts), TF-IDF (weighted counts), word embeddings (dense vectors
capturing similarity), or contextual embeddings from Transformers. The representation choice
determines how much meaning the model can access.

**Q2. What is TF-IDF and why is it better than raw counts?**
*Answer:* TF-IDF multiplies term frequency by inverse document frequency, so it boosts words
that are frequent in a document but rare across the corpus (informative) and down-weights
ubiquitous words like "the" (uninformative). This highlights distinctive terms, making it a
stronger feature representation than raw counts.

**Q3. What's the difference between word embeddings and contextual embeddings?**
*Answer:* Word embeddings (Word2Vec/GloVe) assign each word a single fixed vector regardless
of context, capturing general similarity. Contextual embeddings (from Transformers like
BERT) produce a different vector for a word depending on the surrounding words, capturing
context-specific meaning (e.g. "bank" in "river bank" vs "money bank").

**Q4. What is the difference between stemming and lemmatization?**
*Answer:* Stemming crudely chops word endings to a root (often not a real word, e.g.
"studies"→"studi"), fast but imprecise. Lemmatization maps a word to its proper dictionary
base form using vocabulary and grammar (e.g. "better"→"good"), more accurate but slower.

**Q5. Why might a simple TF-IDF model still be used despite Transformers existing?**
*Answer:* It's fast, cheap, interpretable, needs little data, and is a strong baseline that
is often competitive on straightforward tasks (e.g. spam, simple sentiment). Transformers are
heavier and overkill when a lightweight, explainable model suffices.

## Scenario-Based Questions (with answers)

**Q1.** *You must build a spam filter quickly with limited compute. What approach do you
choose and why?*
*Answer:* TF-IDF features with Naive Bayes or Logistic Regression. It trains and predicts in
milliseconds, scales to high-dimensional text, needs little compute, and is a proven strong
baseline for spam — far more practical than a Transformer for this constraint.

**Q2.** *Your model treats "good" and "great" as completely unrelated, hurting sentiment
accuracy. What representation fixes this?*
*Answer:* Word embeddings (or contextual embeddings), which place semantically similar words
close together, so "good" and "great" share signal. TF-IDF/BoW treat them as unrelated
tokens; embeddings capture their similarity.

**Q3.** *A sentiment model works in testing but fails in production on real user text. You
find production text is lowercased differently and not stripped of punctuation. What's the
lesson?*
*Answer:* Preprocessing must be identical between training and inference. Inconsistent
cleaning changes the features the model sees, degrading performance. Encapsulate
preprocessing in a pipeline applied the same way everywhere.

## Logic-Based Questions (with answers)

**Q1.** Why does Bag of Words give the same vector to "the cat sat" and "sat the cat"?
*Answer:* Because BoW only counts word occurrences and ignores order; both sentences contain
the same words with the same counts, so their count vectors are identical.

**Q2.** Why does TF-IDF assign a low weight to the word "the"?
*Answer:* "the" appears in nearly every document, so its document frequency is very high,
making the inverse-document-frequency factor (log N/df) small — driving its TF-IDF weight
down as uninformative.

**Q3.** Why can the same word need different vectors in different sentences?
*Answer:* Because words are polysemous — their meaning depends on context (e.g. "bank").
Contextual embeddings produce different vectors for the same word in different contexts to
capture the intended sense, which fixed word embeddings cannot.

## Practical Questions (with answers)

**Q1.** Write code to convert a list of texts into a TF-IDF matrix.
*Answer:* `TfidfVectorizer().fit_transform(texts)`.

**Q2.** How would you add word-pair (bigram) features to TF-IDF?
*Answer:* `TfidfVectorizer(ngram_range=(1, 2))` includes unigrams and bigrams.

**Q3.** Why use a scikit-learn `Pipeline` for a text classifier?
*Answer:* So the same vectorizer is fit on training data and applied identically to new data,
preventing preprocessing mismatch and leakage, and making the model easy to deploy.

## Long Questions (with answers)

**Q1. Explain how text is represented numerically in NLP, tracing the evolution from Bag of
Words to contextual embeddings.**

*Answer:* Because models only handle numbers, NLP must convert text into vectors, and the
methods have grown steadily richer. **Bag of Words (BoW)** counts how often each vocabulary
word appears in a document, ignoring order — simple but it discards word order and treats all
words as equally important. **TF-IDF** improves on counts by weighting each term by its
frequency in the document times the log inverse of how many documents contain it, so
distinctive words are boosted and ubiquitous words like "the" are suppressed; it's a fast,
strong baseline. Both, however, capture no **meaning** — "good" and "great" are unrelated.
**Word embeddings** (Word2Vec, GloVe) solve this by learning, from large corpora, a dense
vector per word such that semantically similar words sit close together and relationships
become directions (king − man + woman ≈ queen). Their limitation is one fixed vector per
word, ignoring context. **Contextual embeddings** from Transformers (BERT) produce a
*different* vector for a word depending on its surrounding words, capturing sense and nuance
(distinguishing "river bank" from "money bank"). This progression — counts → weighted counts
→ static meaning vectors → contextual meaning vectors — is the central story of NLP, with
each step letting models access more of language's meaning.

**Q2. Describe the NLP pipeline and common tasks, and explain why TF-IDF baselines remain
useful in the age of Transformers.**

*Answer:* A typical **NLP pipeline** is: **raw text → preprocessing → representation → model →
task output**. Preprocessing cleans and standardises text (tokenization, lowercasing,
stopword removal, stemming/lemmatization). Representation converts tokens to vectors (TF-IDF
or embeddings). A model (a classic ML classifier or a neural network) then performs a
**task**: text classification and **sentiment analysis** (categorise text), **named-entity
recognition** (extract names/places/dates), **machine translation**, **summarisation**,
**question answering**, and **text generation** — the last increasingly handled by LLMs.
Despite Transformers' dominance, **TF-IDF baselines remain useful** because they are fast,
cheap, interpretable, require little data, and are often competitive on straightforward tasks
like spam detection and simple sentiment; they run on minimal hardware, are easy to deploy
and explain, and serve as an essential sanity-check baseline before investing in heavier
models. In practice, one starts with a TF-IDF + linear/Naive Bayes baseline and escalates to
pretrained or fine-tuned Transformers only when the task's complexity and the value of deeper
language understanding justify the extra cost.

## Exercises

1. List five text-preprocessing steps and what each does.
2. Explain why TF-IDF down-weights common words, using the formula.
3. What does "king − man + woman ≈ queen" tell you about embeddings?
4. Why do contextual embeddings beat Word2Vec for ambiguous words?
5. Give three common NLP tasks and an example of each.

## Mini-Project

**Project: Build a text classifier.**

1. Get a text dataset (e.g. SMS spam, movie reviews, or a labelled set you create).
2. Build a TF-IDF + Logistic Regression (and Naive Bayes) pipeline; report accuracy,
   precision, recall, F1.
3. Inspect the most informative words (largest coefficients) for each class.
4. Experiment with `ngram_range` and stopword removal; note the effect.
5. (Stretch) Compare against a pretrained Transformer via Hugging Face. Save in
   `my-ml-journey/`.

## Assignments

1. **Coding:** Build a sentiment classifier on a real dataset; tune the `TfidfVectorizer`
   and compare unigrams vs bigrams.
2. **Coding (stretch):** `pip install transformers` and run a sentiment `pipeline` on the
   same sentences; compare to your TF-IDF model.
3. **Conceptual:** Write one page tracing NLP's evolution (rules → statistical → embeddings →
   Transformers) and what each era added.

::: tip
TF-IDF and embeddings turn text into numbers. But the systems that *understand and generate*
language at a human-like level are **Large Language Models** — built on the Transformers of
Chapter 37. Chapter 39 explains how LLMs like ChatGPT actually work.
:::
