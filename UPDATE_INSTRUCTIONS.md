UPDATE INSTRUCTIONS — Week 5 Optimization Additions

1. Copy files into your repo
   - Create directories:
     - src/optimization
     - tests
     - experiments (optional; scripts will create it)
   - Add the files provided:
     - src/optimize.py
     - src/optimization/optimization_config.py
     - src/optimization/grid_search.py
     - src/optimization/random_search.py
     - src/optimization/bayesian_optimization.py
     - src/optimization/compare_models.py
     - tests/test_optimization.py

2. Requirements
   - Update requirements.txt to include Optuna:
     - optuna>=3.0
   - Then create and activate virtualenv and install:
     - python -m venv .venv
     - source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows
     - pip install -r requirements.txt
     - pip install optuna

3. Run smoke tests
   - pytest tests/test_optimization.py -q

4. Run full optimization (example)
   - python -m src.optimize --data data/telco_customer_churn.csv --out experiments --models_out models --optuna_trials 50

5. After experiments finish
   - results: experiments/experiment_summary.csv
   - per-method outputs: experiments/grid_search/, experiments/random_search/, experiments/optuna/
   - models: models/rf_*.joblib

6. Commit & push
   - git checkout -b week5-optimization
   - git add .
   - git commit -m "Add Week 5 optimization: Grid/Random/Optuna + runner + tests"
   - git push --set-upstream origin week5-optimization

7. Notes
   - If you already have a preprocessing pipeline, change src/optimize.py to import and call it instead of the simple_preprocess() here.
   - For long runs, consider reducing GridSearchCV grid size or n_iter / n_trials.
