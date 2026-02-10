# Traditional ML — Text Classification Pipeline

Modular implementation of a traditional ML pipeline for text classification
with TF-IDF features, SMOTE balancing, GridSearchCV tuning, and ensemble methods.

## Structure

```
traditional_ml/
├── __init__.py          # Package exports
├── preprocessing.py     # TweetPreprocessor class
├── features.py          # FeatureExtractor (word + char TF-IDF)
├── models.py            # Model definitions, SMOTE, GridSearch, evaluation
├── pipeline.py          # Binary & multi-class orchestration
├── main.py              # Example runner (adapt to your dataset)
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt

python -m traditional_ml.main \
    --train data/train.csv \
    --val data/val.csv \
    --test data/test.csv
```

## Adapting to Your Dataset

Edit the **CONFIGURATION** section in `main.py`:

1. **`TEXT_COLUMN`** — column name containing raw text
2. **`BINARY_TARGETS`** — list of binary (0/1) column names
3. **`MULTICLASS_NAMES`** — class labels for multi-class task
4. **`create_multiclass_label()`** — function mapping rows to a single int label

## Using as a Library

```python
from traditional_ml import (
    TweetPreprocessor,
    FeatureExtractor,
    run_binary_classification,
)

# Preprocess
preprocessor = TweetPreprocessor(remove_stopwords=False)
train['clean'] = preprocessor.transform_series(train['text'])

# Extract features
extractor = FeatureExtractor(word_ngram_range=(1, 2))
X_train, X_train_nb = extractor.fit_transform(train['clean'])
X_test, X_test_nb = extractor.transform(test['clean'])

# Run binary classification for a target
results = run_binary_classification(
    target_col='label',
    X_full={'train': X_train, 'val': X_val, 'test': X_test},
    X_nb={'train': X_train_nb, 'val': X_val_nb, 'test': X_test_nb},
    y_train=y_tr, y_val=y_va, y_test=y_te
)
```

## Pipeline Overview

1. **Preprocessing** — lowercasing, URL/mention removal, hashtag→word,
   elongation normalization, number→NUM, punctuation removal, stemming
2. **Feature Extraction** — FeatureUnion of word TF-IDF (unigrams+bigrams, 10K)
   and character TF-IDF (3-5 char_wb, 10K) → ~20K sparse features
3. **SMOTE** — synthetic oversampling of minority class per target
4. **GridSearchCV** — 5-fold CV with macro-F1 for each algorithm
5. **Models** — MultinomialNB, LinearSVC, SVM-RBF, LogisticRegression,
   RandomForest, VotingEnsemble (soft voting)
6. **Evaluation** — F1-macro, classification report, confusion matrix
