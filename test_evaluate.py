"""
tests/test_evaluate.py

Unit tests for src/evaluate.py. Run with: pytest tests/
"""

import pytest

from src.train import load_training_data, train_random_forest
from src.evaluate import (
    cross_validate_model,
    evaluate_holdout,
    train_test_gap,
    find_false_negatives,
)

DATA_PATH = "data/telco_customer_churn.csv"


@pytest.fixture(scope="module")
def trained_rf():
    X_train, X_test, y_train, y_test = load_training_data(DATA_PATH)
    model = train_random_forest(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


def test_cross_validation_returns_five_folds(trained_rf):
    model, X_train, X_test, y_train, y_test = trained_rf
    results = cross_validate_model(model, X_train, y_train, k=5)
    assert len(results["fold_scores"]) == 5
    assert 0 <= results["mean"] <= 1


def test_cross_validation_scores_are_reasonably_stable(trained_rf):
    model, X_train, X_test, y_train, y_test = trained_rf
    results = cross_validate_model(model, X_train, y_train, k=5)
    # A very high std would suggest the estimate is unreliable (Week 4, Section 7.2)
    assert results["std"] < 0.10


def test_holdout_metrics_are_within_valid_ranges(trained_rf):
    model, X_train, X_test, y_train, y_test = trained_rf
    results = evaluate_holdout(model, X_test, y_test, "Random Forest", verbose=False)
    for key in ("roc_auc", "recall", "precision"):
        assert 0 <= results[key] <= 1


def test_train_test_gap_is_non_negative_and_bounded(trained_rf):
    model, X_train, X_test, y_train, y_test = trained_rf
    gap_results = train_test_gap(model, X_train, y_train, X_test, y_test)
    assert gap_results["train_auc"] >= gap_results["test_auc"]
    # A gap beyond this would flag likely overfitting worth investigating.
    assert gap_results["gap"] < 0.25


def test_false_negative_finder_only_returns_missed_churners(trained_rf):
    model, X_train, X_test, y_train, y_test = trained_rf
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    fn = find_false_negatives(X_test, y_test, y_pred, proba)
    assert (fn["actual"] == 1).all()
    assert (fn["predicted"] == 0).all()
