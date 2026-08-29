import json
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from .optimization_config import RF_PARAM_GRID, rf_base

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def run_grid_search(X, y, cv=5, out_dir="experiments/grid_search"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = rf_base()
    gs = GridSearchCV(
        estimator=model,
        param_grid=RF_PARAM_GRID,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
        refit=True
    )
    gs.fit(X, y)

    # Save cv results
    cvres = pd.DataFrame(gs.cv_results_)
    cvres.to_csv(out_dir / "grid_search_cv_results.csv", index=False)

    best = gs.best_estimator_
    best_score = float(gs.best_score_)
    best_params = gs.best_params_

    # record small metadata
    record = {"best_params": best_params, "best_score": best_score}
    with open(out_dir / "grid_search_record.json", "w") as f:
        json.dump(record, f, indent=2)

    return best, best_score, record


def run_logistic_grid(X, y, cv=5, out_dir="experiments/grid_search/logistic", random_state=42):
    """Run a small GridSearchCV for LogisticRegression as an interpretable baseline.

    This function accepts feature matrix X and target y (compatibility with tests)
    and performs GridSearch over C and penalty using a simple pipeline that
    scales numeric features.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight='balanced', random_state=random_state, max_iter=1000))
    ])

    param_grid = {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__penalty": ["l1", "l2"],
        # liblinear supports l1/l2 and is robust across sklearn versions
        "clf__solver": ["liblinear"],
    }

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
        refit=True
    )
    gs.fit(X, y)

    cvres = pd.DataFrame(gs.cv_results_)
    cvres.to_csv(Path(out_dir) / "logistic_grid_cv_results.csv", index=False)

    best = gs.best_estimator_
    best_score = float(gs.best_score_)
    best_params = gs.best_params_

    record = {"best_params": best_params, "best_score": best_score}
    with open(Path(out_dir) / "logistic_grid_record.json", "w") as f:
        json.dump(record, f, indent=2)

    return best, best_score, record
