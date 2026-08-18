"""
preprocessing.py

Data cleaning, feature engineering, and preprocessing pipeline construction
for the customer churn prediction project.

Dataset: IBM Telco Customer Churn (public)
https://www.kaggle.com/datasets/blastchar/telco-customer-churn
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

SERVICE_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

BINARY_COLUMNS = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]

ONE_HOT_COLUMNS = ["InternetService", "PaymentMethod", "Contract"]


def load_and_clean(path: str) -> pd.DataFrame:
    """Load the raw CSV and fix the known TotalCharges parsing issue.

    TotalCharges is stored as a string in the source file because a small
    number of zero-tenure customers have a blank value instead of 0.00.
    Coercing to numeric and filling those blanks with 0 resolves this.
    """
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    df = df.drop_duplicates()
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived features defined in the Week 2 preprocessing strategy."""
    df = df.copy()

    # Average monthly spend: smooths billing anomalies vs. raw TotalCharges.
    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)

    # Tenure cohorts: churn risk is rarely linear across raw tenure months.
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49+"],
        include_lowest=True,
    )

    # Engagement score: count of active add-on services.
    df["num_services"] = (df[SERVICE_COLUMNS] == "Yes").sum(axis=1)

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Binary-map two-level fields, ordinally encode tenure_group, and
    one-hot encode nominal fields.

    customerID is dropped here: it is a unique identifier with no
    predictive value and a guaranteed source of overfitting if left in.
    """
    df = df.copy()

    for col in BINARY_COLUMNS:
        df[col] = df[col].map({"Yes": 1, "No": 0, "Male": 1, "Female": 0})

    # The six add-on service columns use Yes/No (and sometimes "No internet
    # service") -- map all of these to a simple 1/0 flag.
    service_map = {"Yes": 1, "No": 0, "No internet service": 0, "No phone service": 0}
    for col in SERVICE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(service_map)

    if "MultipleLines" in df.columns:
        df["MultipleLines"] = df["MultipleLines"].map(service_map)

    # tenure_group is ordinal (0-12 < 13-24 < 25-48 < 49+), so it is mapped
    # to integers rather than one-hot encoded, preserving that ordering.
    tenure_order = {"0-12": 0, "13-24": 1, "25-48": 2, "49+": 3}
    if "tenure_group" in df.columns:
        df["tenure_group"] = df["tenure_group"].astype(str).map(tenure_order)

    df = pd.get_dummies(df, columns=ONE_HOT_COLUMNS, drop_first=True)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    return df


def build_numeric_pipeline() -> Pipeline:
    """Median imputation + standard scaling, used for the Logistic
    Regression branch only. Tree-based models do not need this."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def full_clean_and_engineer(path: str) -> pd.DataFrame:
    """Convenience wrapper chaining load -> clean -> engineer -> encode."""
    df = load_and_clean(path)
    df = engineer_features(df)
    df = encode_categoricals(df)
    return df
