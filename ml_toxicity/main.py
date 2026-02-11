# ml_toxicity/main.py
from __future__ import annotations

import sys
from pathlib import Path

# Allow running this file directly: `python ml_toxicity/main.py`
if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

import argparse
import json
import os
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split

from ml_toxicity.features import clean_text_series, VectorizerConfig, build_feature_union, build_nb_vectorizer
from ml_toxicity.models import (
    get_base_models,
    get_best_candidates,
    build_voting_ensemble,
    build_stacking_ensemble,
)
from ml_toxicity.pipeline import (
    make_text_pipeline,
    run_cv,
    run_gridsearch,
    evaluate_holdout,
    safe_filename,
    save_confusion_matrix,
    save_error_csv,
    CVConfig,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--valid_csv", type=str, default="dataset_jigsaw/jigsaw_miltilingual_valid_translated.csv")
    p.add_argument("--test_csv", type=str, default=None, help="Optional labeled test CSV (same columns).")
    p.add_argument("--text_col", type=str, default="translated")
    p.add_argument("--label_col", type=str, default="toxic")
    p.add_argument("--results_dir", type=str, default="results_ml_toxicity")

    p.add_argument("--run_base", action="store_true")
    p.add_argument("--run_cv", action="store_true")
    p.add_argument("--run_tune", action="store_true")
    p.add_argument("--run_meta", action="store_true")
    return p.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def main():
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ensure_dir(os.path.join(args.results_dir, ts))

    # --- Load dataset ---
    print(f"[Load] Reading CSV: {args.valid_csv}")
    df = pd.read_csv(args.valid_csv)
    X_raw = clean_text_series(df[args.text_col])
    y = df[args.label_col].astype(int).values
    print(f"[Load] Rows: {len(df)} | Positive rate: {y.mean():.4f}")

    # Holdout split (used for base + meta evaluation)
    X_train_raw, X_eval_raw, y_train, y_eval = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[Split] Train: {len(y_train)} | Eval: {len(y_eval)}")

    vec_cfg = VectorizerConfig(max_features=10_000)
    vec_union = build_feature_union(vec_cfg)
    nb_vec = build_nb_vectorizer(vec_cfg)

    base_models = get_base_models()

    # -------------------------
    # Stage A: Base models
    # -------------------------
    if args.run_base:
        print("\n[Stage A] Base holdout evaluation...")
        base_results = {}

        for name, clf in base_models.items():
            print(f"  - Training/Evaluating: {name}")
            vec = nb_vec if name == "Naive Bayes" else vec_union
            pipe = make_text_pipeline(vec, clf, sampler_mode="none")

            res = evaluate_holdout(pipe, X_train_raw, y_train, X_eval_raw, y_eval)
            base_results[name] = res["report"]["macro avg"]

            sname = safe_filename(name)
            cm_path = os.path.join(out_dir, f"cm_{sname}.png")
            save_confusion_matrix(res["confusion"], cm_path, title=name)

            save_error_csv(
                X_eval_raw, y_eval, res["y_pred"],
                os.path.join(out_dir, f"{sname}_false_positives.csv"),
                os.path.join(out_dir, f"{sname}_false_negatives.csv"),
            )

            f1 = base_results[name].get("f1-score", None)
            if f1 is not None:
                print(f"    macro-F1: {f1:.4f}")

        with open(os.path.join(out_dir, "base_holdout_macroavg.json"), "w", encoding="utf-8") as f:
            json.dump(base_results, f, indent=2)

        print("[Stage A] Done.")

    # -------------------------
    # Stage B: Cross-validation
    # -------------------------
    if args.run_cv:
        print("\n[Stage B] Cross-validation (none vs ros)...")
        cv_cfg = CVConfig(n_splits=5, use_stratified=True, random_state=42)

        rows = []
        for name, clf in base_models.items():
            print(f"  - CV: {name}")
            vec = nb_vec if name == "Naive Bayes" else vec_union
            df_cv = run_cv(
                X_raw, y,
                {name: clf},
                vec,
                sampler_modes=["none", "ros"],
                cv_cfg=cv_cfg,
            )
            rows.append(df_cv)

        cv_df = pd.concat(rows, ignore_index=True).sort_values("f1_macro_mean", ascending=False)
        cv_path = os.path.join(out_dir, "cv_results.csv")
        cv_df.to_csv(cv_path, index=False)
        print(f"[Stage B] Done. Saved: {cv_path}")

    # -------------------------
    # Stage C: Fine-tuning (GridSearchCV)
    # -------------------------
    if args.run_tune:
        print("\n[Stage C] GridSearch tuning (train split only)...")
        from ml_toxicity.models import get_param_grids

        grids = get_param_grids()
        tune_targets = ["LinearSVC", "Random Forest", "XGBoost"]
        tuning_summary = {}

        for name in tune_targets:
            print(f"  - GridSearch: {name}")
            clf = base_models[name]
            pipe = make_text_pipeline(vec_union, clf, sampler_mode="none")

            # IMPORTANT: tune on TRAIN ONLY (keep eval split clean)
            gs = run_gridsearch(X_train_raw, y_train, pipe, grids[name])

            tuning_summary[name] = {
                "best_params": gs.best_params_,
                "best_score": float(gs.best_score_),
            }

            grid_path = os.path.join(out_dir, f"grid_{safe_filename(name)}.csv")
            pd.DataFrame(gs.cv_results_).to_csv(grid_path, index=False)
            print(f"    best_score: {gs.best_score_:.4f} | saved: {grid_path}")

        with open(os.path.join(out_dir, "tuning_summary.json"), "w", encoding="utf-8") as f:
            json.dump(tuning_summary, f, indent=2)

        print("[Stage C] Done.")

    # -------------------------
    # Stage D: Meta-learner (Voting + Stacking)
    # -------------------------
    if args.run_meta:
        print("\n[Stage D] Meta-learner ensembles (Voting + Stacking)...")
        best = get_best_candidates()

        svc_pipe = make_text_pipeline(vec_union, best.best_svc, sampler_mode="none")
        rf_pipe  = make_text_pipeline(vec_union, best.best_rf,  sampler_mode="ros")
        xgb_pipe = make_text_pipeline(vec_union, best.best_xgb, sampler_mode="none")
        nb_pipe  = make_text_pipeline(nb_vec,   best.nb,       sampler_mode="ros")  # safe for sparse

        voting = build_voting_ensemble(
            [("svc", svc_pipe), ("rf", rf_pipe), ("xgb", xgb_pipe), ("nb", nb_pipe)],
            voting="soft",
            calibrate_for_soft=False,  # svc is already calibrated in best candidates
        )

        stacking = build_stacking_ensemble(
            [("svc", svc_pipe), ("rf", rf_pipe), ("xgb", xgb_pipe), ("nb", nb_pipe)],
            passthrough=False,
            calibrate_base=False,
        )

        meta_results = {}
        for name, model in [("Voting", voting), ("Stacking", stacking)]:
            print(f"  - Training/Evaluating: {name}")
            res = evaluate_holdout(model, X_train_raw, y_train, X_eval_raw, y_eval)
            meta_results[name] = res["report"]["macro avg"]

            sname = safe_filename(name)
            cm_path = os.path.join(out_dir, f"cm_{sname}.png")
            save_confusion_matrix(res["confusion"], cm_path, title=name)

            save_error_csv(
                X_eval_raw, y_eval, res["y_pred"],
                os.path.join(out_dir, f"{sname}_false_positives.csv"),
                os.path.join(out_dir, f"{sname}_false_negatives.csv"),
            )

            f1 = meta_results[name].get("f1-score", None)
            if f1 is not None:
                print(f"    macro-F1: {f1:.4f}")

        with open(os.path.join(out_dir, "meta_holdout_macroavg.json"), "w", encoding="utf-8") as f:
            json.dump(meta_results, f, indent=2)

        print("[Stage D] Done.")

    print(f"\nDone. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()