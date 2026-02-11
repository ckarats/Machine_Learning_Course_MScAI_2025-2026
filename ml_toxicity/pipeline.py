# ml_toxicity/pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Literal

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold, cross_validate, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE

import matplotlib.pyplot as plt
import re
from pathlib import Path

def safe_filename(s: str, max_len: int = 120) -> str:
    s = s.strip()
    s = re.sub(r'[\\/:*?"<>|]+', "_", s)  # Windows-forbidden chars
    s = re.sub(r"\s+", "_", s)            # spaces -> _
    return s[:max_len]


SamplerMode = Literal["none", "ros", "smote_svd"]


@dataclass
class CVConfig:
    n_splits: int = 5
    shuffle: bool = True
    random_state: int = 42
    use_stratified: bool = True


@dataclass
class GridConfig:
    scoring: str = "f1_macro"
    n_jobs: int = -1
    verbose: int = 1


def get_cv(cfg: CVConfig):
    if cfg.use_stratified:
        return StratifiedKFold(n_splits=cfg.n_splits, shuffle=cfg.shuffle, random_state=cfg.random_state)
    return KFold(n_splits=cfg.n_splits, shuffle=cfg.shuffle, random_state=cfg.random_state)


def make_text_pipeline(vectorizer, clf, *, sampler_mode: SamplerMode = "none", random_state: int = 42):
    """
    Returns an imblearn Pipeline:
      vec -> (optional) reducer -> (optional) sampler -> clf

    Notes:
    - For sparse TF-IDF, ROS is safe.
    - SMOTE is safer after SVD (dense low-dim); we include Normalizer to stabilize.
    """
    steps: List[Tuple[str, Any]] = [("vec", vectorizer)]

    if sampler_mode == "ros":
        steps += [("sampler", RandomOverSampler(random_state=random_state))]
        return ImbPipeline(steps + [("clf", clf)])

    if sampler_mode == "smote_svd":
        steps += [
            ("svd", TruncatedSVD(n_components=300, random_state=random_state)),
            ("norm", Normalizer(copy=False)),
            ("sampler", SMOTE(random_state=random_state)),
        ]
        return ImbPipeline(steps + [("clf", clf)])

    return ImbPipeline(steps + [("clf", clf)])


def run_cv(
    X_raw,
    y,
    models: Dict[str, Any],
    vectorizer,
    *,
    sampler_modes: List[SamplerMode] = ("none", "ros"),
    cv_cfg: CVConfig = CVConfig(),
) -> pd.DataFrame:
    """
    Runs cross-validation for each (model, sampler_mode) with raw text input.
    Returns tidy DataFrame with mean/std for macro-F1/precision/recall.
    """
    scoring = {
        "f1_macro": "f1_macro",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
    }
    cv = get_cv(cv_cfg)

    rows = []
    for model_name, clf in models.items():
        for sm in sampler_modes:
            pipe = make_text_pipeline(vectorizer, clf, sampler_mode=sm, random_state=cv_cfg.random_state)
            scores = cross_validate(pipe, X=X_raw, y=y, cv=cv, scoring=scoring, n_jobs=None)
            row = {
                "model": model_name,
                "sampler": sm,
                "cv": "StratifiedKFold" if cv_cfg.use_stratified else "KFold",
                "n_splits": cv_cfg.n_splits,
            }
            for k, v in scores.items():
                if k.startswith("test_"):
                    metric = k.replace("test_", "")
                    row[f"{metric}_mean"] = float(np.mean(v))
                    row[f"{metric}_std"] = float(np.std(v))
            rows.append(row)

    return pd.DataFrame(rows).sort_values(["f1_macro_mean"], ascending=False).reset_index(drop=True)


def run_gridsearch(
    X_raw,
    y,
    base_pipeline,
    param_grid: Dict[str, Any],
    *,
    cv_cfg: CVConfig = CVConfig(n_splits=3),
    grid_cfg: GridConfig = GridConfig(),
):
    cv = get_cv(cv_cfg)
    gs = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring=grid_cfg.scoring,
        cv=cv,
        n_jobs=grid_cfg.n_jobs,
        verbose=grid_cfg.verbose,
    )
    gs.fit(X_raw, y)
    return gs


def evaluate_holdout(model, X_raw_train, y_train, X_raw_test, y_test, *, target_names=None) -> Dict[str, Any]:
    model.fit(X_raw_train, y_train)
    y_pred = model.predict(X_raw_test)
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    return {"report": report, "confusion": cm, "y_pred": y_pred}


def save_confusion_matrix(cm: np.ndarray, path: str, *, title: str = ""):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, str(val), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_error_csv(X_raw_test, y_test, y_pred, path_fp: str, path_fn: str):
    Path(path_fp).parent.mkdir(parents=True, exist_ok=True)
    Path(path_fn).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"text": list(X_raw_test), "y_true": y_test, "y_pred": y_pred})
    fp = df[(df.y_true == 0) & (df.y_pred == 1)]
    fn = df[(df.y_true == 1) & (df.y_pred == 0)]
    fp.to_csv(path_fp, index=False)
    fn.to_csv(path_fn, index=False)
