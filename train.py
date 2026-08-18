"""
train.py

Trains the Logistic Regression baseline and Random Forest candidate
model for churn prediction, on top of the Week 2 preprocessing pipeline.

Usage:
    python -m src.train
"""

import argparse
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import full_clean_and_engineer, build_numeric_pipeline
from src.utils import log_run, get_logger

logger = get_logger(__name__)

RANDOM_STATE = 42


def load_training_data(data_path: str):
    """Load, clean, engineer, and split the dataset. Splitting happens
    before any imputer/scaler is fit, to avoid leaking test-set
    statistics into training (see Week 2 / Week 4 reports)."""
    df = full_clean_and_engineer(data_path)

    X = df.drop(columns=["Churn"])
    y = df["Churn"].map({"Yes": 1, "No": 0})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    logger.info("Train shape: %s, Test shape: %s", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train):
    """Fit the interpretable baseline. class_weight='balanced' accounts
    for the churn class being the minority (~27%) class."""
    preprocessor = build_numeric_pipeline()
    X_train_scaled = preprocessor.fit_transform(X_train)

    model = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
    )
    model.fit(X_train_scaled, y_train)
    return model, preprocessor


def train_random_forest(X_train, y_train):
    """Fit the primary candidate model. max_depth is capped conservatively
    as a starting point; Week 4's validation curve work refines this."""
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def main(data_path: str, model_out_dir: str):
    X_train, X_test, y_train, y_test = load_training_data(data_path)

    log_reg, preprocessor = train_logistic_regression(X_train, y_train)
    rf = train_random_forest(X_train, y_train)

    joblib.dump(log_reg, f"{model_out_dir}/logistic_regression.pkl")
    joblib.dump(preprocessor, f"{model_out_dir}/scaler.pkl")
    joblib.dump(rf, f"{model_out_dir}/random_forest.pkl")

    logger.info("Models saved to %s", model_out_dir)
    log_run("logistic_regression", {"max_iter": 1000}, {}, log_path=f"{model_out_dir}/run_log.csv")
    log_run(
        "random_forest",
        {"n_estimators": 300, "max_depth": 10},
        {},
        log_path=f"{model_out_dir}/run_log.csv",
    )

    return {
        "log_reg": log_reg,
        "rf": rf,
        "preprocessor": preprocessor,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train churn prediction models.")
    parser.add_argument("--data", default="data/telco_customer_churn.csv")
    parser.add_argument("--out", default="models")
    args = parser.parse_args()

    main(args.data, args.out)
