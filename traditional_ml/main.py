#!/usr/bin/env python3
"""
main.py
--------
Example usage of the traditional_ml package.

Adapt the DATA LOADING and LABEL CREATION sections to your own dataset.
The rest of the pipeline works independently of the data source.

Usage:
    python main.py --train traindata.csv --val val.csv --test test.csv
"""

import argparse
import warnings
import numpy as np
import pandas as pd

from traditional_ml import (
    TweetPreprocessor,
    FeatureExtractor,
    run_binary_classification,
    run_multiclass_classification,
)

warnings.filterwarnings('ignore')
np.random.seed(500)


# ══════════════════════════════════════════════════════════
# CONFIGURATION — adapt these to your dataset
# ══════════════════════════════════════════════════════════

# Column containing the raw text
TEXT_COLUMN = 'tweet_content'

# Binary target columns (each must be 0/1)
BINARY_TARGETS = ['harassment', 'SexualH', 'IndirectH', 'PhysicalH']

# Multi-class setup
MULTICLASS_NAMES = ['Non-harassment', 'IndirectH', 'SexualH', 'PhysicalH']


def create_multiclass_label(row):
    """Convert binary targets to a single multi-class label.

    Adapt this function to your own label schema.

    Returns: 0=Non-harassment, 1=IndirectH, 2=SexualH, 3=PhysicalH
    """
    if row['harassment'] == 0:
        return 0
    elif row['IndirectH'] == 1:
        return 1
    elif row['SexualH'] == 1:
        return 2
    else:
        return 3


# ══════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════

def main(train_path, val_path, test_path):

    # ── 1. Data Loading ──────────────────────────────────
    print("=" * 70)
    print("1. DATA LOADING")
    print("=" * 70)

    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)

    for df in [train, val, test]:
        if "Unnamed: 0" in df.columns:
            del df["Unnamed: 0"]

    print(f"  Train: {len(train)} samples")
    print(f"  Val:   {len(val)} samples")
    print(f"  Test:  {len(test)} samples")

    # ── 2. Preprocessing ─────────────────────────────────
    print("\n" + "=" * 70)
    print("2. PREPROCESSING")
    print("=" * 70)

    preprocessor = TweetPreprocessor(
        language='english',
        remove_stopwords=True,   # set False to preserve bigram phrases
        min_token_len=2
    )

    for name, df in [("Train", train), ("Val", val), ("Test", test)]:
        df['text_clean'] = preprocessor.transform_series(df[TEXT_COLUMN])
        print(f"  {name}: preprocessed {len(df)} texts")

    # ── 3. Feature Extraction ────────────────────────────
    print("\n" + "=" * 70)
    print("3. FEATURE EXTRACTION")
    print("=" * 70)

    extractor = FeatureExtractor(
        max_word_features=10000,
        max_char_features=10000,
        word_ngram_range=(1, 2),
        char_ngram_range=(3, 5),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    X_train_full, X_train_nb = extractor.fit_transform(train['text_clean'])
    X_val_full, X_val_nb = extractor.transform(val['text_clean'])
    X_test_full, X_test_nb = extractor.transform(test['text_clean'])

    # Pack into dicts for the pipeline
    X_full = {'train': X_train_full, 'val': X_val_full, 'test': X_test_full}
    X_nb = {'train': X_train_nb, 'val': X_val_nb, 'test': X_test_nb}

    # ── 4. Part A: Binary Classification ─────────────────
    print("\n" + "=" * 70)
    print("4. PART A — BINARY CLASSIFICATION")
    print("=" * 70)

    all_binary_results = {}

    for target_col in BINARY_TARGETS:
        y_tr = train[target_col].to_numpy()
        y_va = val[target_col].to_numpy()
        y_te = test[target_col].to_numpy()

        results = run_binary_classification(
            target_col=target_col,
            X_full=X_full, X_nb=X_nb,
            y_train=y_tr, y_val=y_va, y_test=y_te
        )
        all_binary_results[target_col] = results

    # ── 5. Part B: Multi-class Classification ────────────
    print("\n" + "=" * 70)
    print("5. PART B — MULTI-CLASS CLASSIFICATION")
    print("=" * 70)

    for df in [train, val, test]:
        df['multiclass'] = df.apply(create_multiclass_label, axis=1)

    y_tr_mc = train['multiclass'].to_numpy()
    y_va_mc = val['multiclass'].to_numpy()
    y_te_mc = test['multiclass'].to_numpy()

    mc_results = run_multiclass_classification(
        X_full=X_full, X_nb=X_nb,
        y_train=y_tr_mc, y_val=y_va_mc, y_test=y_te_mc,
        class_names=MULTICLASS_NAMES
    )

    # ── 6. Overall Summary ───────────────────────────────
    print("\n" + "=" * 70)
    print("BEST MODELS")
    print("=" * 70)

    summary_rows = []

    print("\n  Binary Classification:")
    for target_col, results in all_binary_results.items():
        best = max(results, key=lambda x: x['f1_test'])
        print(f"    {target_col:12s} → {best['model_name']:<40s} "
              f"Test F1: {best['f1_test']:.4f}")
        summary_rows.append({
            'problem': 'binary', 'target': target_col,
            'best_model': best['model_name'],
            'f1_val': best['f1_val'], 'f1_test': best['f1_test']
        })

    best_mc = max(mc_results, key=lambda x: x['f1_test'])
    print(f"\n  Multi-class Classification:")
    print(f"    4-class      → {best_mc['model_name']:<40s} "
          f"Test F1: {best_mc['f1_test']:.4f}")
    summary_rows.append({
        'problem': 'multiclass', 'target': '4-class',
        'best_model': best_mc['model_name'],
        'f1_val': best_mc['f1_val'], 'f1_test': best_mc['f1_test']
    })

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("results_summary.csv", index=False)
    print("\n  Results saved to results_summary.csv")

    return all_binary_results, mc_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Traditional ML pipeline for text classification"
    )
    parser.add_argument('--train', required=True, help='Path to train CSV')
    parser.add_argument('--val', required=True, help='Path to validation CSV')
    parser.add_argument('--test', required=True, help='Path to test CSV')

    args = parser.parse_args()
    main(args.train, args.val, args.test)
