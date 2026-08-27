import json
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from .optimization_config import RF_PARAM_GRID, rf_base

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
