"""
tests/test_preprocessing.py

Unit tests for src/preprocessing.py. Run with: pytest tests/
"""

import pandas as pd
import pytest

from src.preprocessing import (
    load_and_clean,
    engineer_features,
    encode_categoricals,
)

DATA_PATH = "data/telco_customer_churn.csv"


@pytest.fixture(scope="module")
def cleaned_df():
    return load_and_clean(DATA_PATH)


def test_total_charges_is_numeric_after_cleaning(cleaned_df):
    assert pd.api.types.is_numeric_dtype(cleaned_df["TotalCharges"])
    assert cleaned_df["TotalCharges"].isnull().sum() == 0


def test_no_duplicate_rows_remain(cleaned_df):
    assert cleaned_df.duplicated().sum() == 0


def test_avg_monthly_spend_is_never_infinite(cleaned_df):
    df = engineer_features(cleaned_df)
    assert not df["avg_monthly_spend"].isin([float("inf"), float("-inf")]).any()


def test_tenure_group_has_expected_categories(cleaned_df):
    df = engineer_features(cleaned_df)
    expected = {"0-12", "13-24", "25-48", "49+"}
    assert set(df["tenure_group"].dropna().astype(str).unique()) <= expected


def test_num_services_within_valid_range(cleaned_df):
    df = engineer_features(cleaned_df)
    assert df["num_services"].between(0, 6).all()


def test_encoding_drops_customer_id(cleaned_df):
    df = engineer_features(cleaned_df)
    df = encode_categoricals(df)
    assert "customerID" not in df.columns


def test_encoding_produces_only_numeric_columns(cleaned_df):
    df = engineer_features(cleaned_df)
    df = encode_categoricals(df)
    non_numeric = df.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    # 'Churn' (the target) is the only expected non-numeric survivor here;
    # it gets mapped to 0/1 separately in train.py before modelling.
    assert set(non_numeric) <= {"Churn"}
