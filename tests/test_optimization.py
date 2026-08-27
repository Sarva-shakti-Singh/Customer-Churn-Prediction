import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# Ensure src package importable when running tests from repo root
sys.path.insert(0, "src")

from optimization.grid_search import run_grid_search
from optimization.random_search import run_random_search
from optimization.bayesian_optimization import run_optuna

def make_small_dataset(n_samples=200):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=5,
        n_redundant=0,
        weights=[0.7, 0.3],
        flip_y=0.01,
        random_state=42
    )
    import pandas as pd
    return pd.DataFrame(X), pd.Series(y)

def test_grid_search_runs():
    X, y = make_small_dataset()
    best, score, rec = run_grid_search(X, y, cv=3, out_dir="tests/_tmp_grid")
    assert isinstance(score, float)
    assert "best_params" in rec

def test_random_search_runs():
    X, y = make_small_dataset()
    best, score, rec = run_random_search(X, y, cv=3, n_iter=5, out_dir="tests/_tmp_rand")
    assert isinstance(score, float)
    assert "best_params" in rec

def test_optuna_runs():
    X, y = make_small_dataset()
    best, score, rec = run_optuna(X, y, cv=3, n_trials=5, out_dir="tests/_tmp_optuna")
    assert isinstance(score, float)
    assert "best_params" in rec
