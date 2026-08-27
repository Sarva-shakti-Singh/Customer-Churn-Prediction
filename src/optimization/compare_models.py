import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def summarize_results(experiments_dir="experiments", out_plot="experiments/optimization_plots.png"):
    experiments_dir = Path(experiments_dir)
    summary_path = experiments_dir / "experiment_summary.csv"
    if not summary_path.exists():
        print("No experiment_summary.csv found. Skipping summary.")
        return
    df = pd.read_csv(summary_path)
    # ensure numeric columns
    for c in ["cv_roc_auc", "test_roc_auc"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # simple bar plot comparing cv_roc_auc and test_roc_auc
    df_melt = df.melt(id_vars=["method"], value_vars=[c for c in ["cv_roc_auc", "test_roc_auc"] if c in df.columns], var_name="stage", value_name="roc_auc")
    plt.figure(figsize=(8,4))
    sns.barplot(data=df_melt, x="method", y="roc_auc", hue="stage")
    plt.title("CV vs Test ROC-AUC by method")
    plt.tight_layout()
    out_plot = Path(out_plot)
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_plot)
    print(f"Wrote plot to {out_plot}")
