from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Grid for GridSearchCV (small-ish grid so it completes reasonably)
RF_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}

# Broader distribution for RandomizedSearchCV
RF_PARAM_DIST = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [5, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", None]
}

# Base model factory
def rf_base(random_state=42):
    return RandomForestClassifier(class_weight="balanced", random_state=random_state)
