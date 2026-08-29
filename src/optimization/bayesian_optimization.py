import json
from pathlib import Path

import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from .optimization_config import rf_base

RND = 42


def run_optuna(X, y, preprocessor=None, cv=5, n_trials=50, out_dir="experiments/optuna"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 100, 500)
        max_depth = trial.suggest_int("max_depth", 3, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])

        # build model
        model = rf_base()
        # set params
        model.set_params(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features
        )

        # If a preprocessor is provided, wrap in a pipeline
        if preprocessor is not None:
            from sklearn.pipeline import Pipeline
            pipeline = Pipeline([("preprocessor", preprocessor), ("clf", model)])
            estimator = pipeline
            # For cross_val_score, use the pipeline directly
        else:
            estimator = model

        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RND)
        scores = cross_val_score(estimator, X, y, cv=cv_strategy, scoring="roc_auc", n_jobs=-1)
        return float(scores.mean())

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RND))
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_value = float(study.best_value)

    # train best model on full training set
    best_model = rf_base()
    best_model.set_params(**best_params)
    best_model.fit(X, y)

    # save trials dataframe
    df = study.trials_dataframe()
    df.to_csv(out_dir / "optuna_trials.csv", index=False)

    # Save optuna plots using matplotlib (if available)
    try:
        import matplotlib.pyplot as plt
        from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances

        fig1 = plot_optimization_history(study)
        fig1.savefig(out_dir / "optuna_history.png")
        fig2 = plot_param_importances(study)
        fig2.savefig(out_dir / "optuna_param_importance.png")
    except Exception as e:
        # Not critical; plots are best-effort
        print(f"Optuna plotting failed: {e}")

    record = {"best_params": best_params, "best_value": best_value}
    with open(out_dir / "optuna_record.json", "w") as f:
        json.dump(record, f, indent=2)

    return best_model, best_value, record
