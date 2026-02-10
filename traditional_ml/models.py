"""
models.py
----------
Model definitions, hyperparameter grids, SMOTE balancing,
GridSearchCV tuning, and evaluation utilities.
"""

import numpy as np
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (f1_score, classification_report,
                              confusion_matrix, make_scorer)


# ──────────────────────────────────────────────────────────
# Hyperparameter grids
# ──────────────────────────────────────────────────────────

PARAM_GRIDS = {
    'multinomial_nb': {
        'alpha': np.concatenate([
            np.linspace(0.001, 0.1, 10),
            np.linspace(0.1, 2.0, 10)
        ])
    },
    'svm_linear': {
        'C': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
    },
    'svm_rbf': {
        'C': [0.1, 1.0, 10.0, 50.0],
        'gamma': ['scale', 'auto', 0.01, 0.001]
    },
    'logistic_regression': {
        'C': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],
        'max_iter': [1000]
    },
    'random_forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 30, 50],
        'min_samples_split': [2, 5]
    }
}


# ──────────────────────────────────────────────────────────
# Model factory
# ──────────────────────────────────────────────────────────

def get_base_models(random_state=0):
    """Return dict of (estimator, param_grid, requires_nb_features) tuples."""
    return {
        'MultinomialNB': (
            MultinomialNB(),
            PARAM_GRIDS['multinomial_nb'],
            True   # needs non-negative features (word-only TF-IDF)
        ),
        'SVM-Linear': (
            LinearSVC(dual='auto'),
            PARAM_GRIDS['svm_linear'],
            False
        ),
        'SVM-RBF': (
            SVC(kernel='rbf'),
            PARAM_GRIDS['svm_rbf'],
            False
        ),
        'LogisticRegression': (
            LogisticRegression(random_state=random_state),
            PARAM_GRIDS['logistic_regression'],
            False
        ),
        'RandomForest': (
            RandomForestClassifier(random_state=random_state, n_jobs=-1),
            PARAM_GRIDS['random_forest'],
            False
        ),
    }


def build_voting_ensemble(best_lr, best_svm_lin, best_rf, random_state=0):
    """Build soft-voting ensemble from best tuned models.

    Parameters
    ----------
    best_lr : LogisticRegression (fitted or with best params)
    best_svm_lin : LinearSVC (fitted or with best params)
    best_rf : RandomForestClassifier (fitted or with best params)

    Returns
    -------
    VotingClassifier
    """
    calibrated_svm = CalibratedClassifierCV(
        LinearSVC(C=best_svm_lin.C, dual='auto'), cv=3
    )
    return VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(
                C=best_lr.C, penalty=best_lr.penalty,
                solver='liblinear', max_iter=1000,
                random_state=random_state)),
            ('svm', calibrated_svm),
            ('rf', RandomForestClassifier(
                n_estimators=best_rf.n_estimators,
                max_depth=best_rf.max_depth,
                random_state=random_state, n_jobs=-1)),
        ],
        voting='soft'
    )


# ──────────────────────────────────────────────────────────
# SMOTE
# ──────────────────────────────────────────────────────────

def apply_smote(X, y, random_state=42, verbose=True):
    """Apply SMOTE oversampling to balance classes.

    Parameters
    ----------
    X : sparse matrix or array
    y : array-like
    random_state : int
    verbose : bool

    Returns
    -------
    X_resampled, y_resampled
    """
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X, y)

    if verbose:
        unique, counts = np.unique(y_res, return_counts=True)
        dist = dict(zip(unique.astype(int), counts))
        print(f"  SMOTE: {X.shape[0]} → {X_res.shape[0]} samples  {dist}")

    return X_res, y_res


# ──────────────────────────────────────────────────────────
# Grid Search
# ──────────────────────────────────────────────────────────

def run_gridsearch(estimator, param_grid, X_train, y_train,
                   model_name="", cv=5, scoring='macro'):
    """Run GridSearchCV with F1-macro scoring.

    Parameters
    ----------
    estimator : sklearn estimator
    param_grid : dict
    X_train : sparse matrix or array
    y_train : array-like
    model_name : str (for logging)
    cv : int (number of folds)
    scoring : str ('macro', 'micro', 'weighted')

    Returns
    -------
    best_estimator : fitted sklearn estimator
    """
    scorer = make_scorer(f1_score, average=scoring)
    grid = GridSearchCV(
        estimator, param_grid,
        scoring=scorer, cv=cv, n_jobs=-1, refit=True
    )
    grid.fit(X_train, y_train)

    print(f"  GridSearch [{model_name}]")
    print(f"    Best params: {grid.best_params_}")
    print(f"    Best CV F1-{scoring}: {grid.best_score_:.4f}")

    return grid.best_estimator_


# ──────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────

def evaluate_model(model, X_train, y_train, X_val, y_val,
                   X_test, y_test, model_name="",
                   target_names=None, average='macro'):
    """Train on (X_train, y_train), evaluate on val and test.

    Parameters
    ----------
    model : sklearn estimator
    X_train, y_train : training data
    X_val, y_val : validation data
    X_test, y_test : test data
    model_name : str
    target_names : list of str (for classification report)
    average : str (F1 averaging method)

    Returns
    -------
    dict with keys: model_name, f1_val, f1_test, pred_test, model
    """
    model.fit(X_train, y_train)

    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    f1_val = f1_score(y_val, pred_val, average=average)
    f1_test = f1_score(y_test, pred_test, average=average)

    print(f"\n  {model_name}")
    print(f"  {'─' * 55}")
    print(f"  Val  F1-{average}: {f1_val:.4f}")
    print(f"  Test F1-{average}: {f1_test:.4f}")
    print(f"\n  Test Classification Report:")
    print(classification_report(y_test, pred_test,
                                target_names=target_names, digits=4))

    cm = confusion_matrix(y_test, pred_test)
    print(f"  Confusion Matrix (test):")
    for row in cm:
        print(f"    {row}")

    return {
        'model_name': model_name,
        'f1_val': f1_val,
        'f1_test': f1_test,
        'pred_test': pred_test,
        'model': model
    }


def print_summary(results, title=""):
    """Print a ranked summary table from a list of result dicts."""
    sorted_res = sorted(results, key=lambda x: x['f1_test'], reverse=True)
    best_f1 = sorted_res[0]['f1_test']

    print(f"\n  {'─' * 60}")
    print(f"  SUMMARY — {title}")
    print(f"  {'─' * 60}")
    print(f"  {'Model':<45s} {'Val F1':>8s} {'Test F1':>8s}")
    print(f"  {'─' * 60}")

    for r in sorted_res:
        marker = " ★" if r['f1_test'] == best_f1 else ""
        print(f"  {r['model_name']:<45s} "
              f"{r['f1_val']:>8.4f} {r['f1_test']:>8.4f}{marker}")
