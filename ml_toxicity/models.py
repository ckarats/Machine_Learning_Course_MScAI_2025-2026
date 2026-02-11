# ml_toxicity/models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb


@dataclass(frozen=True)
class BestCandidates:
    """
    Your 'best candidates' (can be used directly in ensembles and gridsearch starting points).
    """
    best_svc: Any
    best_rf: Any
    best_xgb: Any
    nb: Any


def get_base_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Base models exactly as you listed.
    Note: we keep SVC(probability=False) by default (fast). For soft voting/stacking we will use calibrated LinearSVC.
    """
    return {
        "Naive Bayes": MultinomialNB(),
        "SVM (Simple/RBF)": SVC(kernel="rbf", class_weight="balanced"),
        "SVM (Linear Kernel)": SVC(kernel="linear", class_weight="balanced"),
        "LinearSVC": LinearSVC(class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=1, random_state=random_state),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=random_state),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, scale_pos_weight=7, eval_metric="logloss", random_state=random_state),
    }


def get_best_candidates(random_state: int = 42) -> BestCandidates:
    best_svc = CalibratedClassifierCV(
        LinearSVC(C=0.1, loss="squared_hinge", class_weight="balanced", random_state=random_state),
        cv=3,
        method="sigmoid",
    )

    best_rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=1,
    )

    best_xgb = xgb.XGBClassifier(
        learning_rate=0.1,
        max_depth=6,
        n_estimators=200,
        scale_pos_weight=5,
        eval_metric="logloss",
        random_state=random_state,
    )

    nb = MultinomialNB()

    return BestCandidates(best_svc=best_svc, best_rf=best_rf, best_xgb=best_xgb, nb=nb)


def get_param_grids() -> Dict[str, Dict[str, Any]]:
    """
    Parameter grids aligned to your fine-tuning script.
    Keys match pipeline step names expected by GridSearchCV:
      - vec__* (optional, if you decide to tune vectorizer)
      - clf__*
    """
    return {
        "LinearSVC": {
            "clf__C": [0.1, 1, 10],
            "clf__loss": ["hinge", "squared_hinge"],
        },
        "Random Forest": {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [None, 20],
            "clf__min_samples_split": [2, 5],
        },
        "XGBoost": {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [4, 6, 8],
            "clf__learning_rate": [0.05, 0.1],
            "clf__scale_pos_weight": [3, 5, 7],
        },
    }


def ensure_probabilistic(estimator, *, cv: int = 3):
    """
    Ensure we have predict_proba for soft voting / stacking.
    - If estimator already has predict_proba, return as-is.
    - If it's LinearSVC, calibrate it.
    - Otherwise, try calibration as a generic fallback.
    """
    if hasattr(estimator, "predict_proba"):
        return estimator

    if isinstance(estimator, LinearSVC):
        return CalibratedClassifierCV(estimator, cv=cv, method="sigmoid")

    # Generic fallback (works for many margin-based classifiers)
    return CalibratedClassifierCV(estimator, cv=cv, method="sigmoid")


def build_voting_ensemble(
    named_estimators: List[Tuple[str, Any]],
    voting: str = "soft",
    *,
    calibrate_for_soft: bool = True,
) -> VotingClassifier:
    """
    named_estimators: list of (name, estimator)
    """
    if voting == "soft" and calibrate_for_soft:
        named_estimators = [(n, ensure_probabilistic(est)) for n, est in named_estimators]

    return VotingClassifier(estimators=named_estimators, voting=voting)


def build_stacking_ensemble(
    named_estimators: List[Tuple[str, Any]],
    meta_learner=None,
    *,
    passthrough: bool = False,
    calibrate_base: bool = True,
) -> StackingClassifier:
    if meta_learner is None:
        meta_learner = LogisticRegression(max_iter=2000)

    if calibrate_base:
        named_estimators = [(n, ensure_probabilistic(est)) for n, est in named_estimators]

    return StackingClassifier(
        estimators=named_estimators,
        final_estimator=meta_learner,
        passthrough=passthrough,
        stack_method="predict_proba",
        n_jobs=None,
    )
