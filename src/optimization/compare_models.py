import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def summarize_results(experiments_dir="experiments", out_plot="experiments/optimization_plots.png"):
    """Create a summary plot comparing CV vs Test ROC-AUC across optimization methods.
    
    Reads the experiment_summary.csv file and generates a bar plot showing
    CV ROC-AUC and Test ROC-AUC for each optimization method.
    """
    experiments_dir = Path(experiments_dir)
    summary_path = experiments_dir / "experiment_summary.csv"
    if not summary_path.exists():
        print("No experiment_summary.csv found. Skipping summary.")
        return
    
    df = pd.read_csv(summary_path)
    
    # Ensure numeric columns (correct column names: cv_auc, test_auc)
    for c in ["cv_auc", "test_auc"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Melt for plotting: compare CV AUC vs Test AUC
    df_melt = df.melt(
        id_vars=["method", "model"],
        value_vars=[c for c in ["cv_auc", "test_auc"] if c in df.columns],
        var_name="stage",
        value_name="roc_auc"
    )
    
    # Create figure
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melt, x="method", y="roc_auc", hue="stage", palette="Set2")
    plt.title("CV vs Test ROC-AUC by Optimization Method", fontsize=14, fontweight="bold")
    plt.xlabel("Optimization Method", fontsize=12)
    plt.ylabel("ROC-AUC Score", fontsize=12)
    plt.ylim([0, 1.0])
    plt.legend(title="Stage", labels=["CV ROC-AUC", "Test ROC-AUC"])
    plt.tight_layout()
    
    out_plot = Path(out_plot)
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_plot, dpi=150)
    plt.close()
    print(f"Wrote plot to {out_plot}")
