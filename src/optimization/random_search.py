import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

from .optimization_config import RF_PARAM_DIST, rf_base

def run_random_search(X, y, cv=5, n_iter=30, out_dir="experiments/random_search", random_state=42):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = rf_base(random_state=random_state)
    rs = RandomizedSearchCV(
        estimator=model,
        param_distributions=RF_PARAM_DIST,
        n_iter=n_iter,
        cv=cv,
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
