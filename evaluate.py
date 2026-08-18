"""
evaluate.py

Evaluation and validation utilities for the churn prediction models:
cross-validation, holdout metrics, confusion matrix, ROC / PR curves,
and error analysis (false-negative inspection).

See the Week 4 report (Model Evaluation and Validation Techniques) for
the reasoning behind each metric and validation choice made here.
"""

import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, validation_curve

RANDOM_STATE = 42


def cross_validate_model(model, X_train, y_train, k: int = 5, scoring: str = "roc_auc") -> dict:
    """5-fold stratified cross-validation. Returns per-fold scores plus
    mean/std, since a low std is itself evidence the result is stable
    and not an artefact of one lucky split."""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring=scoring)
    return {
        "fold_scores": scores.round(3).tolist(),
        "mean": round(scores.mean(), 3),
        "std": round(scores.std(), 3),
    }


def evaluate_holdout(model, X_test, y_test, name: str, verbose: bool = True) -> dict:
    """Compute ROC-AUC, recall, and precision on the held-out test set.
    Accuracy is deliberately not the headline metric here -- see Week 4
    Section 3.2 on why accuracy is misleading on an imbalanced dataset."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results = {
        "model": name,
        "roc_auc": round(roc_auc_score(y_test, y_proba), 3),
        "recall": round(recall_score(y_test, y_pred), 3),
        "precision": round(precision_score(y_test, y_pred), 3),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    if verbose:
        print(f"--- {name} ---")
        print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    return results


def train_test_gap(model, X_train, y_train, X_test, y_test) -> dict:
    """Compare train vs. test ROC-AUC to spot overfitting. A gap in the
    0.05-0.08 range is typically mild and expected for a Random Forest;
    0.15+ warrants tuning (see Week 4 Section 9.2)."""
    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return {
        "train_auc": round(train_auc, 3),
        "test_auc": round(test_auc, 3),
        "gap": round(train_auc - test_auc, 3),
    }


def plot_confusion_matrix(model, X_test, y_test, title: str = "Confusion Matrix"):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    disp.plot(cmap="Blues")
    plt.title(title)
    return disp


def plot_roc_and_pr_curves(model, X_test, y_test, name: str = "Model"):
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"ROC Curve \u2014 {name}")

    PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"Precision-Recall Curve \u2014 {name}")


def find_false_negatives(X_test, y_test, y_pred, proba):
    """Isolate customers who churned but were missed by the model, for
    manual inspection. If false negatives cluster around a specific
    profile, that is actionable -- see Week 4 Section 10.3."""
    df = X_test.copy()
    df["actual"] = y_test.values
    df["predicted"] = y_pred
    df["churn_probability"] = proba
    return df[(df["actual"] == 1) & (df["predicted"] == 0)]


def depth_validation_curve(estimator_cls, X_train, y_train, depth_range=None, **estimator_kwargs):
    """Train/validation score across a range of max_depth, to find where
    the validation score peaks before the model starts overfitting."""
    if depth_range is None:
        depth_range = [3, 5, 7, 10, 15, 20, None]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    model = estimator_cls(random_state=RANDOM_STATE, **estimator_kwargs)

    train_scores, val_scores = validation_curve(
        model,
        X_train,
        y_train,
        param_name="max_depth",
        param_range=depth_range,
        cv=skf,
        scoring="roc_auc",
    )
    return {
        "depth_range": depth_range,
        "train_mean": train_scores.mean(axis=1).round(3).tolist(),
        "val_mean": val_scores.mean(axis=1).round(3).tolist(),
    }
