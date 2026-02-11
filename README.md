
# 🧠 Social Media Abuse & Toxicity Detection  
## Traditional ML · Deep Learning · Structured ML Pipeline

This repository contains implementations and comparative studies for:

- 📌 Harassment Detection (ECML 2019 Dataset)
  - Traditional Machine Learning
  - Deep Learning (GRU + Attention architectures)

- 📌 Toxic Comment Classification (Jigsaw Dataset)
  - Structured, modular ML pipeline
  - Cross-validation, fine-tuning, meta-learning

The project explores the trade-offs between feature-engineered ML models and neural architectures under class imbalance and limited data.

---

# 📂 Project Structure

.
├── dataset_ecml2019/        # ECML 2019 Harassment dataset  
├── dataset_jigsaw/          # Jigsaw Toxicity dataset  
│  
├── traditional_ml/          # Traditional ML implementation for ECML harassment  
├── deep_learning/           # GRU-based Deep Learning models  
├── ml_toxicity/             # Modular toxicity detection pipeline  
│  
├── notebooks/               # Experimental notebooks & analysis  
│  
└── README.md  

---

# 🧩 Part 1 — Harassment Detection (ECML 2019)

Dataset: dataset_ecml2019/

Task:
- Binary classification per harassment target
- Multi-class classification (4 classes)

Classes:

| Label | Description |
|-------|------------|
| 0     | Non-harassment |
| 1     | IndirectH |
| 2     | SexualH |
| 3     | PhysicalH |

---

# 🔹 Traditional Machine Learning (traditional_ml/)

## Preprocessing Pipeline
- Lowercasing
- URL removal
- Mention removal
- Hashtag normalization
- Elongation normalization
- Stopword removal
- Snowball stemming

Example:
"@john I HAAATE you!!! #violence"
→ "haate violenc"

---

## Feature Extraction

Word TF-IDF
- ngram_range=(1,2)
- max_features=10,000
- sublinear_tf=True

Character TF-IDF
- analyzer='char_wb'
- ngram_range=(3,5)
- max_features=10,000

FeatureUnion → ~20K sparse features per tweet.

---

## Imbalance Handling
- SMOTE applied only on training data
- Separate binary classifier per target

---

## Models Used
- Multinomial Naive Bayes
- SVM (Linear & RBF)
- Logistic Regression
- Random Forest
- Voting Ensemble

---

## Results (Binary Classification)

| Target       | Best Model           | Test F1 |
|-------------|---------------------|--------|
| Harassment  | SVM-Linear          | 0.7344 |
| SexualH     | Random Forest       | 0.8202 |
| IndirectH   | Logistic Regression | 0.5272 |
| PhysicalH   | Naive Bayes         | 0.5243 |

Traditional ML outperformed deep learning across targets.

---

# 🔹 Deep Learning (deep_learning/)

## Representation
- GloVe Twitter 200d embeddings

## Data Augmentation
- Back-translation (German, Greek, French → English)

## Architectures Tested
- Vanilla RNN
- Projected RNN
- Attention RNN
- Multi-Attention RNN
- Projected Multi-Attention RNN

Training:
- BCEWithLogitsLoss
- Adam (lr=0.001)
- Early stopping
- 10 runs per model

---

## Best DL Model

MultiProjectedAttentionRNN  
Macro F1: 0.4708

Despite architectural complexity, traditional ML achieved higher F1 scores.

---

# 🧩 Part 2 — Toxic Comment Classification (Jigsaw)

Folder: ml_toxicity/  
Dataset: dataset_jigsaw/

---

# 🏗 ml_toxicity Pipeline Architecture

## Stage A — Base Holdout Evaluation
- 80/20 stratified split
- Confusion matrices
- False Positive / False Negative export

## Stage B — Cross-Validation
- Stratified 5-fold
- Compare:
  - No sampler
  - RandomOverSampler (ROS)

## Stage C — Fine-Tuning
GridSearchCV for:
- LinearSVC
- RandomForest
- XGBoost

Train split only (no leakage).

## Stage D — Meta-Learner
- Soft Voting
- Stacking (Logistic Regression meta-model)

---

# 🚀 Running Toxicity Pipeline

From repository root:

Base Models:
python ml_toxicity/main.py --run_base

Cross-Validation:
python ml_toxicity/main.py --run_cv

Fine-Tuning:
python ml_toxicity/main.py --run_tune

Meta-Learner:
python ml_toxicity/main.py --run_meta

Run Everything:
python ml_toxicity/main.py --run_base --run_cv --run_tune --run_meta

---

# 📁 Output Structure

Each run creates:

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

# 🧠 Engineering Improvements

- No data leakage (vectorizers inside pipelines)
- Clean stage separation
- Reproducible CV setup
- Windows-safe file saving
- Modular architecture
- Meta-learning integration

---

# 🎓 Academic Context

Developed as part of MSc AI coursework.

Demonstrates:
- Feature engineering vs representation learning
- Class imbalance handling
- Cross-validation best practices
- Ensemble learning
- Reproducible ML engineering workflow
