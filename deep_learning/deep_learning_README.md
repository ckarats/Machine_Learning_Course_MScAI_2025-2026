# Deep Learning — Tweet Harassment Detection

GRU-based deep learning pipeline with multiple attention variants for multi-label harassment detection in tweets.

## Structure

```
deep_learning/
├── requirements.txt
└── src/
    ├── globals.py                 # Paths, hyperparameters, model registry
    ├── encapsulations.py          # Tweet data class (tokenization, label logic)
    ├── preprocess.py              # Data loading, batching, embedding preparation
    ├── preprocess_utils.py        # Vocabulary, GloVe loading, padding, indexing
    ├── translation_script.py      # Back-translation data augmentation
    ├── utils.py                   # Training loop, evaluation, model save/load
    ├── main.py                    # Entry point — runs all 8 model variants
    └── modeling/
        ├── modules.py                       # Base layers (Embedding, GRU, MLP, Attention, Pooling)
        ├── vanilla_rnn.py                   # LastStateRNN / AvgRNN
        ├── projected_vanilla_rnn.py         # ProjectedLastStateRNN / ProjectedAvgRNN
        ├── attention_rnn.py                 # AttentionRNN
        ├── projected_attention.py           # ProjectedAttentionRNN
        ├── multi_attention_rnn.py           # MultiAttentionRNN
        └── projected_multi_attention_rnn.py # ProjectedMultiAttentionRNN (best model)
```

## Quick Start

```bash
pip install -r requirements.txt
cd src
python main.py
```

### Prerequisites

- **GloVe Embeddings**: Download `glove.twitter.27B.200d.txt` and place in `data/embeddings/glove.twitter.27B/`
- **Dataset CSVs**: Place train, validation, and test CSVs in `data/` (paths configured in `globals.py`)
- **Back-translations** (optional): Pre-generated translation CSVs in `data/` or run `python translation_script.py` to generate them

## Pipeline Overview

### 1. Preprocessing (`encapsulations.py`, `preprocess.py`, `preprocess_utils.py`)

Κάθε tweet tokenized μέσω της βιβλιοθήκης `tweet-preprocessor`, η οποία αναγνωρίζει και αντικαθιστά URLs, mentions, hashtags, emojis, κ.λπ. με ειδικά tokens. Στη συνέχεια εφαρμόζεται lowercasing και καθαρισμός αποστρόφων. Δεν γίνεται stemming ή αφαίρεση stopwords, αφού τα pre-trained embeddings εκμεταλλεύονται τις πλήρεις μορφές λέξεων.

Το vocabulary χτίζεται από τα tokens όλων των splits (train + val + test) με ελάχιστη συχνότητα ≥1. Κρατούνται μόνο τα tokens για τα οποία υπάρχει GloVe embedding, μαζί με ειδικά PAD και UNK tokens.

### 2. Word Embeddings

Pre-trained **GloVe Twitter 200d** embeddings (27B tokens). Κατά default τα embeddings είναι **frozen** (`trainable_embeddings=False`) για αποφυγή overfitting στο μικρό dataset. Ορισμένα μοντέλα (Attention, MultiAttention) δοκιμάζονται και με trainable embeddings μέσω της αντίστοιχης παραμέτρου στο `CONFIG`.

### 3. Data Augmentation — Back-Translation (`translation_script.py`)

Για την αντιμετώπιση του class imbalance, εφαρμόζεται **back-translation** αποκλειστικά στα minority samples (IndirectH=1 ή PhysicalH=1) του training set. Κάθε tweet μεταφράζεται EN→target language→EN μέσω Google Translate, σε τρεις γλώσσες:

- Γερμανικά (DE)
- Ελληνικά (EL)
- Γαλλικά (FR)

Οι μεταφράσεις προστίθενται στο training set, αυξάνοντας τα δείγματα των μειονοτικών κλάσεων χωρίς να εισάγουν ακριβή αντίγραφα.

### 4. Model Architectures (`modeling/`)

Όλα τα μοντέλα μοιράζονται κοινή βάση: **GloVe Embeddings → GRU (128 hidden) → Pooling/Attention → MLP → 4 outputs (sigmoid)**. Κάθε output αντιστοιχεί σε ένα binary target (harassment, SexualH, PhysicalH, IndirectH).

#### Base Layers (`modules.py`)

| Layer | Περιγραφή |
|---|---|
| `PretrainedEmbeddingLayer` | GloVe embeddings + SpatialDropout |
| `CellLayer` | GRU ή LSTM (configurable), unidirectional by default |
| `MLP` | Multi-layer perceptron με configurable depth, dropout, activations |
| `LastState` | Τελευταίο hidden state ως αναπαράσταση |
| `AvgPoolingState` | Μέσος όρος όλων των hidden states |
| `AttendedState` | MLP-based attention: score per timestep → softmax → weighted sum |

#### 8 Model Variants

| Μοντέλο | Pooling | Projection | Attention | Decision Heads |
|---|---|---|---|---|
| **LastStateRNN** | Last hidden | ✗ | ✗ | 1 shared MLP → 4 |
| **AvgRNN** | Avg pooling | ✗ | ✗ | 1 shared MLP → 4 |
| **ProjectedLastStateRNN** | Last hidden | ✓ (tanh, 128d) | ✗ | 1 shared MLP → 4 |
| **ProjectedAvgRNN** | Avg pooling | ✓ (tanh, 128d) | ✗ | 1 shared MLP → 4 |
| **AttentionRNN** | Attention | ✗ | 1 shared | 1 shared MLP → 4 |
| **ProjectedAttentionRNN** | Attention | ✓ (tanh, 128d) | 1 shared | 1 shared MLP → 4 |
| **MultiAttentionRNN** | Attention ×4 | ✗ | 4 separate | 4 separate MLP → 1 each |
| **ProjectedMultiAttentionRNN** | Attention ×4 | ✓ (tanh, 128d) | 4 separate | 4 separate MLP → 1 each |

- **Projection layer**: Linear(200→128) + tanh, μειώνει τη διάσταση των embeddings πριν το GRU
- **Multi-attention**: Ξεχωριστός attention mechanism ανά target — κάθε head εστιάζει σε διαφορετικά tokens ανάλογα με το target
- **Shared vs Separate decision heads**: Τα vanilla/attention μοντέλα χρησιμοποιούν ένα MLP (→4 outputs), ενώ τα multi-attention χρησιμοποιούν 4 ξεχωριστά MLPs (→1 output each)

### 5. Training (`utils.py`)

| Παράμετρος | Τιμή |
|---|---|
| Optimizer | Adam (lr=0.001) |
| Loss | BCEWithLogitsLoss (weighted: 0.5·harassment + 0.5·(0.2·sexual + 0.4·physical + 0.4·indirect)) |
| Epochs | 20 (max) |
| Early stopping | Patience=10 (βάσει mean AUC στο validation) |
| Batch size | 32 |
| Max sequence length | 100 tokens |
| Iterations | 10 runs ανά μοντέλο (για στατιστική αξιοπιστία) |

Η loss function εφαρμόζει **βαρυτημένη** BCEWithLogitsLoss: 50% βάρος στο harassment target και 50% κατανεμημένο στις 3 υποκατηγορίες, με μεγαλύτερο βάρος στα physical και indirect (0.4 each) σε σχέση με το sexual (0.2), αντανακλώντας τη δυσκολία ανίχνευσής τους.

Σε κάθε epoch γίνεται evaluation στο validation set μέσω mean AUC (macro average across 4 targets). Το best μοντέλο αποθηκεύεται στο disk (`models/`) και φορτώνεται μετά το training για τα τελικά predictions.

### 6. Inference & Post-processing (`utils.py`)

Κατά το inference, τα 4 sigmoid outputs μετατρέπονται σε binary predictions μέσω hierarchical thresholding:

1. Αν `harassment_score < 0.33` → **Non-harassment** (όλα τα targets = 0)
2. Αν `harassment_score ≥ 0.33` → επιλέγεται η υποκατηγορία με το υψηλότερο score (argmax μεταξύ IndirectH, SexualH, PhysicalH), εξασφαλίζοντας ότι ακριβώς μία υποκατηγορία ενεργοποιείται

### 7. Evaluation

Κάθε μοντέλο τρέχει **10 φορές** με διαφορετική τυχαιότητα. Τα F1, Precision και Recall ανά target υπολογίζονται σε κάθε run στο test set, και αναφέρονται ως μέσοι όροι. Τα αποτελέσματα αποθηκεύονται σωρευτικά στο `results/results.csv`.

## Configuration

Όλες οι βασικές παράμετροι ρυθμίζονται στο `globals.py`:

```python
CONFIG = {
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 20,
    'patience': 10,
    'batch_size': 32,
    'maxlen': 100,
    'trainable_embeddings': False,
    'iterations': 10
}
```

### Data Paths

Τα paths των αρχείων ρυθμίζονται επίσης στο `globals.py`. Για να χρησιμοποιήσεις τα δικά σου δεδομένα, άλλαξε τα αντίστοιχα paths:

```python
GLOVE_EMBEDDINGS_PATH = 'data/embeddings/glove.twitter.27B/glove.twitter.27B.200d.txt'
TRAIN_DATA_PATH = 'data/Train_data_competition.csv'
VALID_DATA_PATH = 'data/Validation_data_competition.csv'
TEST_DATA_PATH  = 'data/testset-competition.csv'
TEST_RESULTS    = 'data/testdata_gold_labels.csv'
```

### Expected CSV Format

Τα CSVs πρέπει να περιέχουν τουλάχιστον τις στήλες:

| Στήλη | Τύπος | Περιγραφή |
|---|---|---|
| `tweet_content` | str | Το κείμενο του tweet |
| `harassment` | int (0/1) | Binary label — γενική παρενόχληση |
| `SexualH` | int (0/1) | Binary label — σεξουαλική παρενόχληση |
| `IndirectH` | int (0/1) | Binary label — έμμεση παρενόχληση |
| `PhysicalH` | int (0/1) | Binary label — σωματική παρενόχληση |

## Results (Test Set — Mean of 10 Runs)

| Model | Sexual F1 | Indirect F1 | Physical F1 | Harassment F1 | Macro F1 |
|---|---|---|---|---|---|
| LastStateRNN | 0.6991 | 0.2584 | 0.1173 | 0.7101 | 0.4462 |
| AvgRNN | 0.6378 | 0.1752 | 0.1256 | 0.6881 | 0.4067 |
| ProjectedLastStateRNN | 0.6962 | 0.3347 | 0.0727 | 0.7080 | 0.4529 |
| ProjectedAvgRNN | 0.6557 | 0.2702 | 0.1559 | 0.6757 | 0.4394 |
| AttentionRNN | 0.6750 | 0.2963 | 0.0878 | 0.7095 | 0.4422 |
| ProjectedAttentionRNN | 0.6923 | 0.3153 | 0.0194 | 0.6941 | 0.4303 |
| MultiAttentionRNN | 0.6935 | 0.3253 | 0.1454 | 0.7004 | 0.4661 |
| **ProjectedMultiAttentionRNN** | **0.7141** | **0.3556** | 0.1268 | 0.6867 | **0.4708** |

Best model: **ProjectedMultiAttentionRNN** (Macro F1 = 0.4708)

## Dependencies

```
torch>=1.1.0
numpy
pandas
scikit-learn
tqdm
tweet-preprocessor==0.5.0
googletrans  (μόνο για back-translation)
```
