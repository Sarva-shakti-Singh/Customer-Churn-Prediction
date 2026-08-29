import json
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from .optimization_config import RF_PARAM_GRID, rf_base


def _prefix_params(param_dict, prefix="clf__"):
    return {f"{prefix}{k}": v for k, v in param_dict.items()}


def run_grid_search(X, y, preprocessor=None, cv=5, out_dir="experiments/grid_search"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # If a preprocessor is provided, wrap the estimator in a Pipeline so that
    # transformations are fit only on training folds.
    if preprocessor is not None:
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("clf", rf_base())
        ])
        param_grid = _prefix_params(RF_PARAM_GRID, prefix="clf__")
        estimator = pipe
    else:
        estimator = rf_base()
        param_grid = RF_PARAM_GRID

    gs = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
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


def run_logistic_grid(X, y, preprocessor=None, cv=5, out_dir="experiments/grid_search/logistic", random_state=42):
    """Run a small GridSearchCV for LogisticRegression as an interpretable baseline.

    If a preprocessor (ColumnTransformer) is provided, it will be used inside
    a Pipeline so that encoding/scaling is fit only on train folds.
    """
    from sklearn.linear_model import LogisticRegression

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clf = LogisticRegression(class_weight='balanced', random_state=random_state, max_iter=1000)

    if preprocessor is not None:
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("clf", clf)
        ])
        estimator = pipe
        param_grid = {
            "clf__C": [0.01, 0.1, 1.0, 10.0],
            "clf__penalty": ["l1", "l2"],
            "clf__solver": ["liblinear"],
        }
    else:
        estimator = clf
        param_grid = {
            "C": [0.01, 0.1, 1.0, 10.0],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"],
        }

    gs = GridSearchCV(
        estimator=estimator,
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
