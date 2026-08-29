"""src/preprocessing.py

Safe, leakage-free preprocessing helpers for Week 5 optimization.
This module wraps the repository's root-level preprocessing functions and
provides utilities to build sklearn ColumnTransformer pipelines based on the
engineered dataframe. It is intentionally placed under src/ so optimization
modules can import it as `import preprocessing` when sys.path includes src/.
"""
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Reuse the project's top-level preprocessing implementation where possible
import preprocessing as base_preproc  # the root-level preprocessing.py


def load_engineer(path: str) -> pd.DataFrame:
    """Load CSV, run the repository's cleaning and feature engineering steps,
    but DO NOT perform categorical encoding (no get_dummies) so that we can
    safely build a ColumnTransformer and avoid leakage.

    Returns the engineered dataframe (with original categorical columns intact).
    """
    df = base_preproc.load_and_clean(path)
    df = base_preproc.engineer_features(df)
    # Do not call encode_categoricals here: encoding must happen inside a Pipeline
    return df


def split_X_y(df: pd.DataFrame, target_col: str = "Churn") -> Tuple[pd.DataFrame, pd.Series]:
    """Split engineered dataframe into X (features) and y (target series).

    If target_col exists with values 'Yes'/'No' it will be mapped to 1/0.
    Otherwise, the last column is used as target.
    """
    df = df.copy()
    if target_col in df.columns:
        y = df[target_col].map({"Yes": 1, "No": 0})
        X = df.drop(columns=[target_col])
    else:
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]
    return X, y.astype(int)


def _get_column_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    # numeric columns
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    # treat boolean as categorical? keep them in numeric group
    # categorical columns: object, category
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    # If any columns are numeric but actually categorical (tiny unique values), user can adjust.
    return numeric_cols, categorical_cols


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Construct a ColumnTransformer that imputes/scales numeric features and
    imputes/one-hot-encodes categorical features. This transformer must be
    fitted only on training data (it will be used inside an sklearn Pipeline
    that is itself passed to cross-validation/search).

    Returns the ColumnTransformer instance.
    """
    numeric_cols, categorical_cols = _get_column_types(X)

    # numeric pipeline: median imputation + standard scaling
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # categorical pipeline: constant imputation + one-hot (ignore unknowns)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse=False, drop="first"),
            ),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor
