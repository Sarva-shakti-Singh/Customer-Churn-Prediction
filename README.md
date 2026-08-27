# Customer Churn Prediction Pipeline

An end-to-end machine learning pipeline that predicts whether a customer of a
subscription-based business will churn within the next 30 days, built as a
6-week Machine Learning Engineer internship project (NSDC).

This repository accompanies four written deliverables:

| Week | Report | Focus |
|------|--------|-------|
| 1 | Project Plan | Problem definition, timeline, risk management, roadmap |
| 2 | Data Preprocessing & Feature Engineering Strategy | Cleaning, encoding, feature design |
| 3 | Model Implementation & Code Documentation | Baseline model implementation |
| 4 | Model Evaluation & Validation Techniques | Cross-validation, metrics, error analysis |

## Dataset

This project uses the public **IBM Telco Customer Churn** dataset
(~7,043 rows, 21 columns). It is not included in this repository — download
it from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
and place it at:

```
data/telco_customer_churn.csv
```

## Project Structure

```
churn-prediction-pipeline/
├── data/                    # Dataset goes here (gitignored)
├── models/                  # Saved model artefacts + run log (gitignored)
├── notebooks/                # Optional exploratory notebooks
├── src/
│   ├── preprocessing.py     # Cleaning, feature engineering, pipeline builder
│   ├── train.py             # Trains Logistic Regression + Random Forest
│   ├── evaluate.py          # Cross-validation, metrics, error analysis
│   ├── run_evaluation.py    # End-to-end train + evaluate entry point
│   ├── predict.py           # Single-record scoring with error handling
│   └── utils.py             # Logging + experiment run logger
├── tests/
│   ├── test_preprocessing.py
│   ├── test_train.py
│   └── test_evaluate.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Train both models:

```bash
python -m src.train --data data/telco_customer_churn.csv --out models
```

Run the full evaluation suite (cross-validation, holdout metrics, overfitting
check, false-negative analysis):

```bash
python -m src.run_evaluation --data data/telco_customer_churn.csv --plots
```

Score a single customer:

```bash
python -m src.predict
```

Run the test suite:

```bash
pytest tests/ -v
```

## Models

- **Logistic Regression** — interpretable baseline, `class_weight='balanced'`
  to account for the ~27% churn class.
- **Random Forest** — primary candidate, expected to capture non-linear
  feature interactions better than the linear baseline.

Both are evaluated primarily on **ROC-AUC**, **recall**, and **precision**
on the churn class rather than accuracy, since the dataset is imbalanced
(see Week 4 report, Section 3.2, for the full reasoning).

## Target Metrics (from Week 1 plan)

| Metric | Target |
|--------|--------|
| ROC-AUC | ≥ 0.80 |
| Recall (churn class) | ≥ 0.70 |
| Precision (churn class) | ≥ 0.55 |


## Week 5 — Model Optimization & Experimentation

Week 5 extends the baseline machine-learning pipeline with systematic
hyperparameter optimization.

Three optimization strategies are compared:

1. Grid Search
2. Randomized Search
3. Bayesian Optimization using Optuna

### Experimental Design

All experiments use the same:

- 80/20 stratified train-test split
- Random seed: 42
- 5-fold Stratified Cross-Validation
- Primary optimization metric: ROC-AUC
- Held-out test set reserved for final evaluation

### Hyperparameters Optimized

For the Random Forest model:

- n_estimators
- max_depth
- min_samples_split
- min_samples_leaf
- max_features
- class_weight

### Run Week 5 Optimization

```bash
python -m src.optimize \
    --data data/telco_customer_churn.csv

## License

This is an educational/internship project.
