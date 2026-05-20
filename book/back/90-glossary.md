# Glossary {.unnumbered}

A comprehensive glossary of the key terms used throughout this book, in plain language.

**Activation function** — A non-linear function (ReLU, sigmoid, tanh, softmax) applied in a neuron so networks can learn complex patterns.

**Adam** — A popular adaptive optimiser combining momentum and per-parameter learning rates; the default for deep learning.

**Algorithm** — A step-by-step recipe a computer follows to solve a problem.

**Anomaly detection** — Finding rare, unusual data points (e.g. fraud), often unsupervised.

**Artificial Intelligence (AI)** — Making machines perform tasks that normally require human intelligence.

**ANI / AGI / ASI** — Artificial Narrow Intelligence (one task; today's AI), Artificial General Intelligence (human-level across tasks; not yet achieved), Artificial Super Intelligence (beyond human; hypothetical).

**Attention** — A mechanism where each element attends to all others (Query·Key→weights→Value); the core of Transformers.

**AUC** — Area Under the ROC Curve; threshold-independent classifier quality (1 perfect, 0.5 random).

**Autoencoder** — A network that compresses input to a bottleneck then reconstructs it; used for compression, denoising, anomaly detection.

**Backpropagation** — Computing gradients of the loss for every weight by applying the chain rule backward through the network.

**Bagging** — Bootstrap Aggregating: train models on bootstrap samples and average/vote; reduces variance (Random Forest).

**Batch size** — Number of samples processed before each weight update.

**Bayes' Theorem** — Updates a prior probability into a posterior using evidence: P(A|B) = P(B|A)·P(A)/P(B).

**Bias (statistical)** — Error from overly simple assumptions → underfitting.

**Bias–Variance Trade-off** — Balancing too-simple (high bias) vs too-complex (high variance) models to minimise total error.

**Boosting** — Sequentially train weak learners, each correcting the last's errors; reduces bias (AdaBoost, Gradient Boosting, XGBoost).

**Central Limit Theorem** — Sample means form a normal distribution even when the data isn't normal.

**Classification** — Supervised task predicting a category.

**Clustering** — Unsupervised grouping of similar items (K-Means, hierarchical, DBSCAN).

**CNN (Convolutional Neural Network)** — A network using convolution filters; ideal for images.

**Concept drift** — When the input→target relationship changes over time in production.

**Confusion matrix** — Table of TP/FP/TN/FN underlying classification metrics.

**Convolution** — Sliding a filter over an image to produce a feature map.

**Correlation** — Strength of linear association between two variables (−1 to +1); not causation.

**Cross-validation** — Splitting training data into k folds to get a robust performance estimate.

**Data augmentation** — Creating label-preserving variations (flips, rotations) to reduce overfitting.

**Data drift** — When the input data distribution changes over time in production.

**Dataset / Instance / Feature / Label** — The full data table / one row / an input column (X) / the target column (y).

**DBSCAN** — Density-based clustering that finds arbitrary shapes and detects outliers.

**Deep Learning** — ML using many-layered neural networks.

**Diffusion model** — Generative model that denoises random noise into images (DALL·E, Stable Diffusion).

**Dimensionality reduction** — Compressing many features into few (PCA, t-SNE, UMAP).

**Disparate impact ratio** — A fairness measure; favourable-outcome ratio across groups; <0.8 flags bias.

**Dropout** — Randomly disabling neurons during training to regularize.

**Edge AI** — Running models on local devices for low latency, privacy, and offline use.

**Embeddings (word)** — Dense vectors where similar words are close (king−man+woman≈queen); contextual embeddings depend on context.

**Ensemble** — Combining multiple models (bagging, boosting) for better performance.

**Epoch** — One full pass through the training data.

**Explainable AI (XAI)** — Methods (SHAP, LIME, interpretable models) to understand model decisions.

**F1-score** — Harmonic mean of precision and recall.

**Feature engineering** — Creating new informative features from raw data.

**Feature selection** — Keeping only the most useful features (filter, wrapper, embedded).

**Federated learning** — Training across devices without raw data leaving them.

**Foundation model** — A large model pretrained on vast data, adapted to many tasks.

**GAN** — Generative Adversarial Network: a generator vs a discriminator trained adversarially.

**Generalization** — A model's ability to perform well on new, unseen data — the goal of ML.

**Generative AI** — AI that creates new content (text, images, audio, video, code).

**Gradient** — Vector of partial derivatives; points uphill on the loss.

**Gradient descent** — Minimising loss by stepping opposite the gradient; learning rate sets step size.

**GRU** — Gated Recurrent Unit; a simpler, faster LSTM variant.

**Hallucination** — When a generative model produces confident but false content.

**Hierarchical clustering** — Builds a dendrogram of merges; no k needed.

**Hyperparameter** — A setting chosen before training (k, learning rate, depth); tuned, not learned.

**KNN (K-Nearest Neighbors)** — Lazy, instance-based: predict from the k closest stored examples.

**LLM (Large Language Model)** — A large Transformer trained to predict the next token; powers chatbots.

**Logistic Regression** — Linear model + sigmoid → probability; a classification baseline.

**Loss function** — Number measuring how wrong predictions are (MSE, cross-entropy); training minimises it.

**LSTM** — Long Short-Term Memory; an RNN with gates for long-term memory.

**Machine Learning (ML)** — Programs that learn patterns from data instead of explicit rules.

**Matrix factorization** — Decomposing the user–item matrix into latent factors (recommenders).

**Mean / Median / Mode** — Average / middle value / most frequent value.

**MLOps** — Practices and tools to deploy, monitor, and maintain ML in production.

**Model** — The learned rules that map inputs to predictions.

**MSE / RMSE / MAE** — Mean squared error / its root (units) / mean absolute error — regression metrics.

**Naive Bayes** — Probabilistic classifier using Bayes' theorem with a feature-independence assumption.

**Neural network** — Layers of artificial neurons, each computing φ(w·x+b).

**Normal (Gaussian) distribution** — The bell curve; 68/95/99.7% within 1/2/3 std devs.

**Normalization (Min-Max)** — Scaling features to [0, 1].

**One-hot encoding** — Turning a nominal category into separate 0/1 columns.

**Overfitting / Underfitting** — Memorising noise (great train, poor test) / too simple (poor on both).

**Parameter** — A value the model learns during training (e.g. a weight).

**PCA** — Principal Component Analysis: linear dimensionality reduction along directions of max variance.

**Pipeline** — A bundled sequence of preprocessing + model, applied consistently to prevent leakage.

**Pooling** — Downsampling feature maps (e.g. max pooling) in CNNs.

**Precision / Recall** — Of predicted positives, how many correct / of actual positives, how many caught.

**Pruning** — Removing unimportant tree branches or network weights to simplify.

**p-value** — Probability of data as extreme as observed assuming the null hypothesis; small → significant.

**Quantization** — Storing weights with fewer bits (e.g. 32→8) to shrink models for the edge.

**RAG (Retrieval-Augmented Generation)** — Grounding an LLM's answers in retrieved documents to reduce hallucination.

**Random Forest** — Bagging of decorrelated decision trees; accurate, robust tabular model.

**Recall** — See Precision / Recall.

**Regression** — Supervised task predicting a continuous number.

**Regularization** — Penalising complexity (L1/L2, dropout) to fight overfitting.

**Reinforcement Learning (RL)** — An agent learns from rewards by interacting with an environment.

**ReLU** — max(0, z); the default hidden-layer activation.

**RNN** — Recurrent Neural Network; processes sequences with a hidden-state memory.

**ROC curve** — Plots true-positive vs false-positive rate across thresholds.

**Self-supervised learning** — The data generates its own labels (e.g. predict a masked word); powers LLMs.

**Semi-supervised learning** — Learning from a few labels plus much unlabelled data.

**Sigmoid** — Squashes any value to (0, 1); used for binary probabilities.

**Softmax** — Turns scores into class probabilities summing to 1.

**Standardization (Z-score)** — Scaling features to mean 0, std 1.

**Standard deviation (σ)** — The most common measure of spread (√variance).

**Supervised learning** — Learning from labelled data (X → y).

**Support Vector Machine (SVM)** — Maximum-margin classifier; the kernel trick handles non-linear data.

**TF-IDF** — Term Frequency–Inverse Document Frequency; weights words by distinctiveness.

**Time series** — Time-ordered, dependent data; never shuffle; split chronologically.

**Tokenization / Token** — Splitting text into units / a word-piece an LLM processes.

**Training / Inference** — Learning parameters from data / using the model to predict.

**Transfer learning** — Adapting a pretrained model to a new task with little data.

**Transformer** — Attention-based architecture behind modern NLP and LLMs.

**Unsupervised learning** — Learning structure from unlabelled data (clustering, dimensionality reduction).

**Variance** — Error from over-sensitivity to training data → overfitting.

**Vector / Matrix / Tensor** — 1-D list / 2-D table / 3-D-plus array of numbers.

**XGBoost** — A fast, regularized gradient-boosting library; top performer on tabular data.

**z-score** — How many standard deviations a value is from the mean: (x − μ)/σ.
