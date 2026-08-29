#!/usr/bin/env python3
"""
Entrypoint for Week 5 optimization experiments.

Usage:
    python -m src.optimize --data data/telco_customer_churn.csv --out experiments --models_out models
"""
import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                             f1_score, accuracy_score)
from sklearn.model_selection import train_test_split

from optimization.grid_search import run_grid_search, run_logistic_grid
from optimization.random_search import run_random_search
from optimization.bayesian_optimization import run_optuna
from optimization.compare_models import summarize_results

from preprocessing import load_engineer, split_X_y, build_preprocessor

RND = 42


def evaluate_on_test(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = None
    try:
        probs = model.predict_proba(X_test)[:, 1]
    except Exception:
        try:
            probs = model.decision_function(X_test)
        except Exception:
            probs = np.zeros(len(y_test))
    return {
        "test_auc": float(roc_auc_score(y_test, probs)),
        "test_accuracy": float(accuracy_score(y_test, preds)),
        "test_recall": float(recall_score(y_test, preds)),
        "test_precision": float(precision_score(y_test, preds)),
        "test_f1": float(f1_score(y_test, preds)),
    }


def append_summary(out_dir, row):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "experiment_summary.csv"
    df = pd.DataFrame([row])
    if summary_path.exists():
        df_existing = pd.read_csv(summary_path)
        df = pd.concat([df_existing, df], ignore_index=True)
    df.to_csv(summary_path, index=False)
    print(f"Wrote summary to {summary_path}")


def main(args):
    data_path = Path(args.data)
    out_dir = Path(args.out)
    models_out = Path(args.models_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_out.mkdir(parents=True, exist_ok=True)

    # Load and engineer features (no encoding)
    df = load_engineer(str(data_path))
    X, y = split_X_y(df)

    # Train/test split (hold-out set must remain untouched during optimization)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RND, stratify=y
    )

    # Build preprocessor fitted only on training data schema
    preprocessor = build_preprocessor(X_train)

    # Logistic Regression baseline (Grid Search)
    print("Running Logistic Regression Grid Search (baseline)...")
    log_model, log_cv_score, log_record = run_logistic_grid(
        X_train, y_train, preprocessor=preprocessor, cv=5, out_dir=out_dir / "grid_search" / "logistic"
    )
    joblib.dump(log_model, models_out / "logistic_grid_best.joblib")
    log_eval = evaluate_on_test(log_model, X_test, y_test)
    log_row = {
        "method": "grid_search",
        "model": "LogisticRegression",
        "cv_auc": log_cv_score,
        "test_auc": log_eval["test_auc"],
        "accuracy": log_eval["test_accuracy"],
        "precision": log_eval["test_precision"],
        "recall": log_eval["test_recall"],
        "f1": log_eval["test_f1"],
        "best_params": json.dumps(log_record.get("best_params", {})),
        "generalization_gap": float(log_cv_score - log_eval["test_auc"]),
    }
    append_summary(out_dir, log_row)

    # Random Forest Grid Search
    print("Running Random Forest Grid Search...")
    gs_model, gs_cv_score, gs_record = run_grid_search(
        X_train, y_train, preprocessor=preprocessor, cv=5, out_dir=out_dir / "grid_search" / "random_forest"
    )
    joblib.dump(gs_model, models_out / "rf_grid_search_best.joblib")
    gs_eval = evaluate_on_test(gs_model, X_test, y_test)
    gs_row = {
        "method": "grid_search",
        "model": "RandomForest",
        "cv_auc": gs_cv_score,
        "test_auc": gs_eval["test_auc"],
        "accuracy": gs_eval["test_accuracy"],
        "precision": gs_eval["test_precision"],
        "recall": gs_eval["test_recall"],
        "f1": gs_eval["test_f1"],
        "best_params": json.dumps(gs_record.get("best_params", {})),
        "generalization_gap": float(gs_cv_score - gs_eval["test_auc"]),
    }
    append_summary(out_dir, gs_row)

    # Randomized Search
    print("Running Randomized Search...")
    rs_model, rs_cv_score, rs_record = run_random_search(
        X_train, y_train, preprocessor=preprocessor, cv=5, n_iter=30, out_dir=out_dir / "random_search", random_state=RND
    )
    joblib.dump(rs_model, models_out / "rf_random_search_best.joblib")
    rs_eval = evaluate_on_test(rs_model, X_test, y_test)
    rs_row = {
        "method": "random_search",
        "model": "RandomForest",
        "cv_auc": rs_cv_score,
        "test_auc": rs_eval["test_auc"],
        "accuracy": rs_eval["test_accuracy"],
        "precision": rs_eval["test_precision"],
        "recall": rs_eval["test_recall"],
        "f1": rs_eval["test_f1"],
        "best_params": json.dumps(rs_record.get("best_params", {})),
        "generalization_gap": float(rs_cv_score - rs_eval["test_auc"]),
    }
    append_summary(out_dir, rs_row)

    # Optuna Bayesian Optimization
    print("Running Optuna Bayesian Optimization...")
    opt_model, opt_cv_score, opt_record = run_optuna(
        X_train, y_train, preprocessor=preprocessor, cv=5, n_trials=args.optuna_trials, out_dir=out_dir / "optuna"
    )
    joblib.dump(opt_model, models_out / "rf_optuna_best.joblib")
    opt_eval = evaluate_on_test(opt_model, X_test, y_test)
    opt_row = {
        "method": "optuna",
        "model": "RandomForest",
        "cv_auc": opt_cv_score,
        "test_auc": opt_eval["test_auc"],
        "accuracy": opt_eval["test_accuracy"],
        "precision": opt_eval["test_precision"],
        "recall": opt_eval["test_recall"],
        "f1": opt_eval["test_f1"],
        "best_params": json.dumps(opt_record.get("best_params", {})),
        "generalization_gap": float(opt_cv_score - opt_eval["test_auc"]),
    }
    append_summary(out_dir, opt_row)

    # Summarize + plots
    print("Summarizing results and creating plots...")
    summarize_results(out_dir, out_dir / "optimization_plots.png")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to csv dataset")
    parser.add_argument("--out", default="experiments", help="Experiments output dir")
    parser.add_argument("--models_out", default="models", help="Where to save best models")
    parser.add_argument("--optuna_trials", type=int, default=50, help="Optuna trials")
    args = parser.parse_args()
    main(args)
