"""
tests/test_train.py

Unit tests for src/train.py. Run with: pytest tests/
"""

import pytest

from src.train import load_training_data, train_logistic_regression, train_random_forest

DATA_PATH = "data/telco_customer_churn.csv"


@pytest.fixture(scope="module")
def split_data():
    return load_training_data(DATA_PATH)


def test_split_preserves_churn_rate(split_data):
    X_train, X_test, y_train, y_test = split_data
    full_rate = (y_train.sum() + y_test.sum()) / (len(y_train) + len(y_test))
    assert abs(y_train.mean() - full_rate) < 0.02
    assert abs(y_test.mean() - full_rate) < 0.02


def test_train_and_test_columns_match(split_data):
    X_train, X_test, y_train, y_test = split_data
    assert set(X_train.columns) == set(X_test.columns)


def test_logistic_regression_predicts_probabilities_in_range(split_data):
    X_train, X_test, y_train, y_test = split_data
    model, preprocessor = train_logistic_regression(X_train, y_train)
    X_test_scaled = preprocessor.transform(X_test)
    proba = model.predict_proba(X_test_scaled)[:, 1]
    assert (proba >= 0).all() and (proba <= 1).all()


def test_random_forest_predicts_probabilities_in_range(split_data):
    X_train, X_test, y_train, y_test = split_data
    model = train_random_forest(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    assert (proba >= 0).all() and (proba <= 1).all()


def test_random_forest_prediction_count_matches_input(split_data):
    X_train, X_test, y_train, y_test = split_data
    model = train_random_forest(X_train, y_train)
    predictions = model.predict(X_test)
    assert len(predictions) == len(X_test)
