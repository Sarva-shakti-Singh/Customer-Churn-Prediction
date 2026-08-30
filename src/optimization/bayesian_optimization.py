import json
from pathlib import Path

import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline

from .optimization_config import rf_base

RND = 42


def run_optuna(X, y, preprocessor=None, cv=5, n_trials=50, out_dir="experiments/optuna"):
    """Run Optuna Bayesian Optimization for Random Forest hyperparameters.
    
    If a preprocessor (ColumnTransformer) is provided, it will be used inside
    a Pipeline so that encoding/scaling is fit only on train folds during CV.
    
    After optimization, the best model is trained on the full training set
    (WITH preprocessing if provided).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 100, 500)
        max_depth = trial.suggest_int("max_depth", 3, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])

        # Build base model
        model = rf_base()
        # Set suggested parameters
        model.set_params(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features
        )

        # If a preprocessor is provided, wrap in a pipeline
        if preprocessor is not None:
            pipeline = Pipeline([("preprocessor", preprocessor), ("clf", model)])
            estimator = pipeline
        else:
            estimator = model

        # Use StratifiedKFold for consistent CV
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RND)
        scores = cross_val_score(estimator, X, y, cv=cv_strategy, scoring="roc_auc", n_jobs=-1)
        return float(scores.mean())

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RND))
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_value = float(study.best_value)

    # Train best model on full training set WITH preprocessing if provided
    best_model = rf_base()
    best_model.set_params(**best_params)
    
    if preprocessor is not None:
        best_pipeline = Pipeline([("preprocessor", preprocessor), ("clf", best_model)])
        best_pipeline.fit(X, y)
        # Return the pipeline, not just the model
        final_model = best_pipeline
    else:
        best_model.fit(X, y)
        final_model = best_model

    # Save trials dataframe
    df = study.trials_dataframe()
    df.to_csv(out_dir / "optuna_trials.csv", index=False)

    # Save optuna plots using matplotlib (if available)
    try:
        import matplotlib.pyplot as plt
        from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances

        fig1 = plot_optimization_history(study)
        fig1.savefig(out_dir / "optuna_history.png")
        plt.close(fig1)
        
        fig2 = plot_param_importances(study)
        fig2.savefig(out_dir / "optuna_param_importance.png")
        plt.close(fig2)
    except Exception as e:
        # Not critical; plots are best-effort
        print(f"Optuna plotting failed: {e}")

    record = {"best_params": best_params, "best_value": best_value}
    with open(out_dir / "optuna_record.json", "w") as f:
        json.dump(record, f, indent=2)

    return final_model, best_value, record
