# Appendix {.unnumbered}

Practical references to support your work throughout the book.

## A. Environment setup

```bash
# 1) Install Python 3.10+ (python.org or Anaconda)
# 2) Create a virtual environment (one per project)
python -m venv .venv
source .venv/bin/activate          # Mac/Linux
.venv\Scripts\activate             # Windows

# 3) Install the core ML stack
pip install numpy pandas matplotlib seaborn scikit-learn jupyter

# 4) Optional, as needed by later chapters
pip install scipy            # statistics (Ch 6)
pip install torch            # deep learning (Parts VI–VII)
pip install transformers     # LLMs/Transformers (Ch 37, 39)
pip install fastapi uvicorn streamlit   # deployment (Ch 44)
pip install xgboost          # gradient boosting (Ch 24)

# 5) Launch a notebook
jupyter notebook
```

**Tip:** save your environment with `pip freeze > requirements.txt`, and recreate it with
`pip install -r requirements.txt`.

## B. The ML library cheat-sheet

| Task | Library | Key calls |
|---|---|---|
| Arrays / maths | NumPy | `np.array`, `.shape`, `.mean`, `@`, `reshape` |
| Data tables | Pandas | `read_csv`, `head`, `info`, `describe`, `groupby`, `fillna` |
| Plots | Matplotlib / Seaborn | `plt.plot/hist/scatter`, `sns.heatmap` |
| Split / CV | scikit-learn | `train_test_split`, `cross_val_score`, `TimeSeriesSplit` |
| Preprocess | scikit-learn | `StandardScaler`, `OneHotEncoder`, `Pipeline`, `ColumnTransformer` |
| Models | scikit-learn | `.fit`, `.predict`, `.predict_proba` |
| Metrics | scikit-learn | `accuracy_score`, `classification_report`, `roc_auc_score`, `mean_squared_error` |
| Tuning | scikit-learn | `GridSearchCV`, `RandomizedSearchCV` |
| Save model | joblib | `joblib.dump`, `joblib.load` |
| Deep learning | PyTorch | `nn.Module`, `loss.backward()`, `optimizer.step()` |

## C. The universal scikit-learn workflow

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)        # split first

model = make_pipeline(StandardScaler(),                       # preprocess + model
                      RandomForestClassifier(random_state=42))
model.fit(X_train, y_train)                                   # train
print(classification_report(y_test, model.predict(X_test)))   # evaluate
```

## D. Mathematical notation used in this book

| Symbol | Meaning |
|---|---|
| `x`, `X` | A feature (input); the feature matrix |
| `y`, `ŷ` | The target (true); the prediction |
| `w`, `b` | Weights; bias |
| `η` (eta) | Learning rate |
| `λ` (lambda) | Regularization strength |
| `σ` | Sigmoid function / standard deviation |
| `∇` (nabla) | Gradient |
| `Σ` | Sum |
| `μ` | Mean |
| `θ` | Model parameters (general) |

## E. Choosing an algorithm (quick guide)

- **Predict a number?** → Linear Regression → Random Forest / Gradient Boosting.
- **Predict a category?** → Logistic Regression → Random Forest / XGBoost → (SVM, KNN, NB).
- **Tabular data, max accuracy?** → Gradient Boosting (XGBoost/LightGBM).
- **No labels, find groups?** → K-Means / DBSCAN.
- **Too many features?** → Feature selection / PCA.
- **Images?** → CNN / transfer learning.
- **Text/sequences?** → TF-IDF + linear (baseline) → Transformers/LLMs.
- **Time series?** → lag features + regressor → LSTM.

## F. Common errors and fixes

| Error/symptom | Likely cause | Fix |
|---|---|---|
| Shape mismatch | Wrong array dimensions | Check `.shape`; `reshape(-1, 1)` |
| Loss is `nan` | Learning rate too high | Lower the learning rate |
| 100% train, low test | Overfitting | Regularize, more data, simpler model |
| Great offline, bad live | Data leakage / drift | Split first, use pipelines, monitor |
| KNN/SVM poor | Unscaled features | `StandardScaler` |
| `ModuleNotFoundError` | Missing package | `pip install ...` (in the right venv) |
