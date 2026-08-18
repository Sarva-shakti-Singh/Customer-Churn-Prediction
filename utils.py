"""
utils.py

Shared helpers: logging setup and a minimal experiment run-logger.
"""

import csv
import datetime
import logging
import os


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def log_run(model_name: str, params: dict, metrics: dict, log_path: str = "models/run_log.csv"):
    """Append one row per training run to a CSV log, so every run is
    traceable back to the parameters and metrics that produced it."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "params": str(params),
        **metrics,
    }

    file_exists = os.path.exists(log_path) and os.path.getsize(log_path) > 0
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
