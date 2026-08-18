"""
run_evaluation.py

End-to-end entry point: trains both models and runs the full Week 4
evaluation suite (cross-validation, holdout metrics, overfitting check,
confusion matrix / ROC / PR plots, false-negative inspection).

Usage:
    python -m src.run_evaluation
"""

import argparse

from src.train import main as train_main
from src.evaluate import (
    cross_validate_model,
    evaluate_holdout,
    train_test_gap,
    plot_confusion_matrix,
    plot_roc_and_pr_curves,
    find_false_negatives,
)
from src.utils import get_logger

logger = get_logger(__name__)


def run(data_path: str, model_out_dir: str, show_plots: bool = False):
    artifacts = train_main(data_path, model_out_dir)
    rf = artifacts["rf"]
    X_train, X_test = artifacts["X_train"], artifacts["X_test"]
    y_train, y_test = artifacts["y_train"], artifacts["y_test"]

    logger.info("Running 5-fold cross-validation on Random Forest...")
    cv_results = cross_validate_model(rf, X_train, y_train, k=5)
    print("Cross-validation:", cv_results)

    logger.info("Evaluating on the held-out test set...")
    holdout_results = evaluate_holdout(rf, X_test, y_test, "Random Forest")
    print("Holdout:", holdout_results)

    logger.info("Checking train/test gap for overfitting...")
    gap_results = train_test_gap(rf, X_train, y_train, X_test, y_test)
    print("Train/test gap:", gap_results)

    y_pred = rf.predict(X_test)
    proba = rf.predict_proba(X_test)[:, 1]
    false_negatives = find_false_negatives(X_test, y_test, y_pred, proba)
    print(f"False negatives on test set: {len(false_negatives)}")

    if show_plots:
        plot_confusion_matrix(rf, X_test, y_test, title="Random Forest \u2014 Confusion Matrix")
        plot_roc_and_pr_curves(rf, X_test, y_test, name="Random Forest")

    return {
        "cv_results": cv_results,
        "holdout_results": holdout_results,
        "gap_results": gap_results,
        "false_negatives": false_negatives,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full training + evaluation pipeline.")
    parser.add_argument("--data", default="data/telco_customer_churn.csv")
    parser.add_argument("--out", default="models")
    parser.add_argument("--plots", action="store_true", help="Render diagnostic plots")
    args = parser.parse_args()

    run(args.data, args.out, show_plots=args.plots)
