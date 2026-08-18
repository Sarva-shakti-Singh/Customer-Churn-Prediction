"""
predict.py

Scores a single customer record against a trained model. Intended as
the building block for the batch-scoring deployment planned in Week 1.
"""

import pandas as pd


def predict_single(model, preprocessor, customer: dict) -> dict:
    """Score one customer record and return a probability + risk tier.

    Raises a clear ValueError if the input is missing an expected
    column, rather than letting scikit-learn fail with an opaque
    internal error -- see Week 3 Section 12.3.
    """
    try:
        row = pd.DataFrame([customer])
        row_processed = preprocessor.transform(row) if preprocessor is not None else row
    except (ValueError, KeyError) as exc:
        raise ValueError(f"Input record is missing an expected column: {exc}") from exc

    proba = model.predict_proba(row_processed)[0][1]
    tier = "High" if proba >= 0.6 else "Medium" if proba >= 0.3 else "Low"

    return {"churn_probability": round(float(proba), 3), "risk_tier": tier}


if __name__ == "__main__":
    import joblib

    model = joblib.load("models/random_forest.pkl")

    sample_customer = {
        "tenure": 3,
        "MonthlyCharges": 95.20,
        "TotalCharges": 285.60,
        "avg_monthly_spend": 71.4,
        "num_services": 1,
        # ... remaining engineered/encoded columns would go here in a real call
    }

    try:
        result = predict_single(model, None, sample_customer)
        print(result)
    except ValueError as e:
        print(f"Prediction failed: {e}")
