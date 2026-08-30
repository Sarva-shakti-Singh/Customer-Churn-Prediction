import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# Ensure src package importable when running tests from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.optimization.grid_search import run_grid_search, run_logistic_grid
from src.optimization.random_search import run_random_search
from src.optimization.bayesian_optimization import run_optuna
from src.preprocessing import build_preprocessor


def make_small_dataset(n_samples=200):
    """Create a small synthetic dataset for testing."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=5,
        n_redundant=0,
        weights=[0.7, 0.3],
        flip_y=0.01,
        random_state=42
    )
    return pd.DataFrame(X), pd.Series(y)


def test_grid_search_runs():
    """Test that Grid Search completes without errors."""
    X, y = make_small_dataset()
    best, score, rec = run_grid_search(X, y, cv=3, out_dir="tests/_tmp_grid")
    assert isinstance(score, float)
    assert "best_params" in rec
    assert score > 0.0
    print("✓ Grid Search test passed")


def test_grid_search_with_preprocessor():
    """Test that Grid Search works with a preprocessor."""
    X, y = make_small_dataset()
    preprocessor = build_preprocessor(X)
    best, score, rec = run_grid_search(X, y, preprocessor=preprocessor, cv=3, out_dir="tests/_tmp_grid_preproc")
    assert isinstance(score, float)
    assert "best_params" in rec
    assert score > 0.0
    print("✓ Grid Search with preprocessor test passed")


def test_logistic_grid_runs():
    """Test that Logistic Regression Grid Search completes without errors."""
    X, y = make_small_dataset()
    best, score, rec = run_logistic_grid(X, y, cv=3, out_dir="tests/_tmp_log_grid")
    assert isinstance(score, float)
    assert "best_params" in rec
    assert score > 0.0
    print("✓ Logistic Grid Search test passed")


def test_random_search_runs():
    """Test that Random Search completes without errors."""
    X, y = make_small_dataset()
    best, score, rec = run_random_search(X, y, cv=3, n_iter=5, out_dir="tests/_tmp_rand")
    assert isinstance(score, float)
    assert "best_params" in rec
    assert score > 0.0
    print("✓ Random Search test passed")


def test_random_search_with_preprocessor():
    """Test that Random Search works with a preprocessor."""
    X, y = make_small_dataset()
    preprocessor = build_preprocessor(X)
    best, score, rec = run_random_search(X, y, preprocessor=preprocessor, cv=3, n_iter=5, out_dir="tests/_tmp_rand_preproc")
    assert isinstance(score, float)
    assert "best_params" in rec
    assert score > 0.0
    print("✓ Random Search with preprocessor test passed")


def test_optuna_runs():
    """Test that Optuna completes without errors."""
    X, y = make_small_dataset()
    best, score, rec = run_optuna(X, y, cv=3, n_trials=5, out_dir="tests/_tmp_optuna")
    assert isinstance(score, float)
    assert "best_params" in rec
    assert score > 0.0
    print("✓ Optuna test passed")


def test_optuna_with_preprocessor():
    """Test that Optuna works with a preprocessor."""
    X, y = make_small_dataset()
    preprocessor = build_preprocessor(X)
    best, score, rec = run_optuna(X, y, preprocessor=preprocessor, cv=3, n_trials=5, out_dir="tests/_tmp_optuna_preproc")
    assert isinstance(score, float)
    assert "best_params" in rec
    assert score > 0.0
    print("✓ Optuna with preprocessor test passed")


if __name__ == "__main__":
    # Run all tests when executed directly
    test_grid_search_runs()
    test_grid_search_with_preprocessor()
    test_logistic_grid_runs()
    test_random_search_runs()
    test_random_search_with_preprocessor()
    test_optuna_runs()
    test_optuna_with_preprocessor()
    print("\n✅ All tests passed!")
