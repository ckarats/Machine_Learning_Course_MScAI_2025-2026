"""
pipeline.py
------------
High-level pipeline that orchestrates:
  - Part A: Binary classification per target
  - Part B: Multi-class classification
"""

import numpy as np
import pandas as pd

from .models import (get_base_models, build_voting_ensemble,
                     apply_smote, run_gridsearch, evaluate_model,
                     print_summary)


def run_binary_classification(target_col, X_full, X_nb, y_train, y_val,
                              y_test, random_state=0):
    """Run all models for a single binary target.

    Parameters
    ----------
    target_col : str
        Name of the target column (for logging).
    X_full : dict with keys 'train', 'val', 'test'
        FeatureUnion sparse matrices.
    X_nb : dict with keys 'train', 'val', 'test'
        Word-only sparse matrices (for MultinomialNB).
    y_train, y_val, y_test : array-like
        Binary labels.
    random_state : int

    Returns
    -------
    list of result dicts
    """
    print("=" * 70)
    print(f"BINARY CLASSIFICATION — Target: {target_col}")
    print("=" * 70)

    pos_ratio = y_train.sum() / len(y_train)
    print(f"  Train balance: {y_train.sum()}/{len(y_train)} "
          f"({pos_ratio:.1%} positive)")

    # SMOTE on both feature sets
    X_tr_sm, y_tr_sm = apply_smote(X_full['train'], y_train, random_state)
    X_tr_nb_sm, y_tr_nb_sm = apply_smote(X_nb['train'], y_train, random_state)

    target_names = [f'Non-{target_col}', target_col]
    results = []
    best_models = {}

    # --- Individual models with GridSearch ---
    base_models = get_base_models(random_state)

    for name, (estimator, param_grid, uses_nb) in base_models.items():
        if uses_nb:
            X_tr, y_tr = X_tr_nb_sm, y_tr_nb_sm
            X_v, X_te = X_nb['val'], X_nb['test']
        else:
            X_tr, y_tr = X_tr_sm, y_tr_sm
            X_v, X_te = X_full['val'], X_full['test']

        best_est = run_gridsearch(
            estimator, param_grid, X_tr, y_tr,
            model_name=f"{name} ({target_col})"
        )
        best_models[name] = best_est

        # Build model name string with best params
        model_label = _format_model_name(name, best_est)

        res = evaluate_model(
            best_est, X_tr, y_tr, X_v, y_val, X_te, y_test,
            model_name=model_label, target_names=target_names
        )
        results.append(res)

    # --- Voting Ensemble ---
    if all(k in best_models for k in
           ['LogisticRegression', 'SVM-Linear', 'RandomForest']):
        voting = build_voting_ensemble(
            best_models['LogisticRegression'],
            best_models['SVM-Linear'],
            best_models['RandomForest'],
            random_state=random_state
        )
        res = evaluate_model(
            voting, X_tr_sm, y_tr_sm,
            X_full['val'], y_val, X_full['test'], y_test,
            model_name="VotingEnsemble (LR+SVM+RF)",
            target_names=target_names
        )
        results.append(res)

    print_summary(results, title=f"Binary [{target_col}]")
    return results


def run_multiclass_classification(X_full, X_nb, y_train, y_val, y_test,
                                  class_names=None, random_state=0):
    """Run all models for multi-class classification.

    Parameters
    ----------
    X_full : dict with keys 'train', 'val', 'test'
    X_nb : dict with keys 'train', 'val', 'test'
    y_train, y_val, y_test : array-like
        Multi-class labels (integers).
    class_names : list of str
        Human-readable class names.
    random_state : int

    Returns
    -------
    list of result dicts
    """
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(len(np.unique(y_train)))]

    print("=" * 70)
    print("MULTI-CLASS CLASSIFICATION")
    print("=" * 70)

    # SMOTE
    X_tr_sm, y_tr_sm = apply_smote(X_full['train'], y_train, random_state)
    X_tr_nb_sm, y_tr_nb_sm = apply_smote(X_nb['train'], y_train, random_state)

    results = []
    best_models = {}
    base_models = get_base_models(random_state)

    for name, (estimator, param_grid, uses_nb) in base_models.items():
        if uses_nb:
            X_tr, y_tr = X_tr_nb_sm, y_tr_nb_sm
            X_v, X_te = X_nb['val'], X_nb['test']
        else:
            X_tr, y_tr = X_tr_sm, y_tr_sm
            X_v, X_te = X_full['val'], X_full['test']

        best_est = run_gridsearch(
            estimator, param_grid, X_tr, y_tr,
            model_name=f"{name} (multiclass)"
        )
        best_models[name] = best_est

        model_label = _format_model_name(name, best_est)

        res = evaluate_model(
            best_est, X_tr, y_tr, X_v, y_val, X_te, y_test,
            model_name=model_label, target_names=class_names
        )
        results.append(res)

    # Voting Ensemble
    if all(k in best_models for k in
           ['LogisticRegression', 'SVM-Linear', 'RandomForest']):
        voting = build_voting_ensemble(
            best_models['LogisticRegression'],
            best_models['SVM-Linear'],
            best_models['RandomForest'],
            random_state=random_state
        )
        res = evaluate_model(
            voting, X_tr_sm, y_tr_sm,
            X_full['val'], y_val, X_full['test'], y_test,
            model_name="VotingEnsemble (LR+SVM+RF)",
            target_names=class_names
        )
        results.append(res)

    print_summary(results, title="Multi-class")
    return results


def _format_model_name(base_name, estimator):
    """Create descriptive model name from best estimator params."""
    if base_name == 'MultinomialNB':
        return f"MultinomialNB (α={estimator.alpha:.4f})"
    elif base_name == 'SVM-Linear':
        return f"SVM-Linear (C={estimator.C})"
    elif base_name == 'SVM-RBF':
        return f"SVM-RBF (C={estimator.C}, γ={estimator.gamma})"
    elif base_name == 'LogisticRegression':
        return f"LogReg (C={estimator.C}, pen={estimator.penalty})"
    elif base_name == 'RandomForest':
        return (f"RandomForest (n={estimator.n_estimators}, "
                f"d={estimator.max_depth})")
    return base_name
