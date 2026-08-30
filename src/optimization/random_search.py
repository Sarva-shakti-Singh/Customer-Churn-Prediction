import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from .optimization_config import RF_PARAM_DIST, rf_base


def _prefix_params(param_dict, prefix="clf__"):
    return {f"{prefix}{k}": v for k, v in param_dict.items()}


def run_random_search(X, y, preprocessor=None, cv=5, n_iter=30, out_dir="experiments/random_search", random_state=42):
    """Run RandomizedSearchCV for Random Forest with optional preprocessing pipeline.
    
    If a preprocessor (ColumnTransformer) is provided, it will be used inside
    a Pipeline so that encoding/scaling is fit only on train folds.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # If a preprocessor is provided, wrap the estimator in a Pipeline so that
    # transformations are fit only on training folds.
    if preprocessor is not None:
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("clf", rf_base(random_state=random_state))
        ])
        param_dist = _prefix_params(RF_PARAM_DIST, prefix="clf__")
        estimator = pipe
    else:
        estimator = rf_base(random_state=random_state)
        param_dist = RF_PARAM_DIST

    # Use StratifiedKFold for consistent CV across all search methods
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    rs = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv_strategy,
        scoring="roc_auc",
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
        refit=True
    )
    rs.fit(X, y)

    results = pd.DataFrame(rs.cv_results_)
    results.to_csv(out_dir / "random_search_cv_results.csv", index=False)

    best = rs.best_estimator_
    best_score = float(rs.best_score_)
    best_params = rs.best_params_

    record = {"best_params": best_params, "best_score": best_score}
    with open(out_dir / "random_search_record.json", "w") as f:
        json.dump(record, f, indent=2)

    return best, best_score, record
