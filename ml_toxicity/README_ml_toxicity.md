
# 🧠 ML Toxicity Detection Pipeline

This module implements a structured, leakage-safe machine learning pipeline for toxic comment classification.

It replaces the experimental notebook-style workflow with a clean, modular architecture supporting:

- TF-IDF (word + char n-grams)
- Cross-validation comparisons
- Fine-tuning via GridSearchCV
- Ensemble methods (Voting + Stacking)
- Meta-learning
- Confusion matrices & error analysis exports
- Reproducible results with timestamped output folders

---

# 📂 Project Structure

Machine_Learning_Course_MScAI_2025-2026/
│
├── dataset_jigsaw/
│   └── jigsaw_miltilingual_valid_translated.csv
│
├── ml_toxicity/
│   ├── __init__.py
│   ├── features.py
│   ├── models.py
│   ├── pipeline.py
│   └── main.py
│
└── README.md

---

# 🏗 Architecture Overview

The pipeline follows four structured stages:

---

## 🔹 Stage A — Base Model Holdout Evaluation

- Train/test split (80/20 stratified)
- Evaluate base classifiers
- Save:
  - Macro F1 / precision / recall
  - Confusion matrices
  - False positives CSV
  - False negatives CSV

Models:

- Naive Bayes
- SVM (RBF)
- SVM (Linear Kernel)
- LinearSVC
- Random Forest
- AdaBoost
- XGBoost

---

## 🔹 Stage B — Cross-Validation Suite

- 5-fold Stratified CV
- Compare:
  - No sampler
  - RandomOverSampler (ROS)

Outputs:
- cv_results.csv
- Mean & std of macro-F1

---

## 🔹 Stage C — Fine-Tuning (GridSearchCV)

Tunes best candidates on training split only (no holdout leakage).

Targets:
- LinearSVC
- Random Forest
- XGBoost

Outputs:
- grid_<model>.csv
- tuning_summary.json

---

## 🔹 Stage D — Meta-Learner (Ensembles)

Builds:

### Soft Voting
Uses calibrated LinearSVC probabilities.

### Stacking
Meta-learner: Logistic Regression  
Base models:
- Calibrated LinearSVC
- Random Forest (ROS)
- XGBoost
- Naive Bayes (ROS)

Outputs:
- meta_holdout_macroavg.json
- Confusion matrices
- FP/FN CSVs

---

# ⚙️ Installation

Create environment (recommended):

conda create -n ml-env python=3.10  
conda activate ml-env

Install dependencies:

pip install -r requirements.txt

---

# 🚀 How To Run

Always run from repo root.

## Run Base Models Only

python ml_toxicity/main.py --run_base

## Run Cross-Validation

python ml_toxicity/main.py --run_cv

## Run Fine-Tuning

python ml_toxicity/main.py --run_tune

## Run Meta-Learner (Voting + Stacking)

python ml_toxicity/main.py --run_meta

## Run Everything

python ml_toxicity/main.py --run_base --run_cv --run_tune --run_meta

---

# 📁 Output Folder

Each run creates a timestamped directory:

results_ml_toxicity/
└── YYYYMMDD_HHMMSS/
    ├── base_holdout_macroavg.json
    ├── cv_results.csv
    ├── tuning_summary.json
    ├── meta_holdout_macroavg.json
    ├── cm_*.png
    ├── *_false_positives.csv
    └── *_false_negatives.csv

---

# 🧪 Important Design Decisions

## No Data Leakage

Vectorizers are inside sklearn Pipelines.

Cross-validation and tuning operate on raw text, ensuring:
- Vectorizer fit happens only inside folds
- Holdout split remains clean

## Sampling Strategy

Default:
- RandomOverSampler (safe for sparse TF-IDF)

Optional:
- SVD + SMOTE (dense representation)

## Why Calibrated LinearSVC?

LinearSVC does not output probabilities.

Calibration enables:
- Soft Voting
- Stacking
- Probability-based evaluation

---

# 📊 Notebook Usage

The notebook (ML-toxic_comment_analysis.ipynb) is used for:

- Analysis & reporting
- Visualization
- Interpretation

All training logic lives in ml_toxicity/.

---

# 🛠 Troubleshooting

### FileNotFoundError
Ensure dataset is inside:

dataset_jigsaw/

and you run from repo root.

### Windows filename errors
All model names are sanitized automatically when saving outputs.

---

# 📈 Suggested Improvements

- Add threshold tuning
- Add Precision-Recall curves
- Add SHAP for XGBoost interpretation
- Add experiment tracking (MLflow)

---

Designed for MSc AI Machine Learning coursework.
