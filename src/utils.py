import os
import json
import random
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import platform
import sys


def print_versions() -> None:
    """Log key library versions for reproducibility."""
    try:
        import sklearn
        import pandas as pd
        import xgboost
        import lightgbm
    except Exception:
        sklearn = None
        pd = None
        xgboost = None
        lightgbm = None

    print("[Versions]")
    print(f"Python: {sys.version.split()[0]}  |  Platform: {platform.platform()}")
    if sklearn:
        print(f"scikit-learn: {sklearn.__version__}")
    if pd:
        print(f"pandas: {pd.__version__}")
    if xgboost:
        print(f"xgboost: {xgboost.__version__}")
    if lightgbm:
        print(f"lightgbm: {lightgbm.__version__}")
    print("-" * 40)


def set_global_seed(seed: int) -> None:
    """Set seeds for reproducibility across common libs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # For xgboost/lightgbm we also pass seeds via model params


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute primary regression metrics: MAE and R2."""
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"MAE": mae, "R2": r2}


def save_json(obj: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def plot_learning_curve(
    train_sizes: np.ndarray,
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    title: str = "Learning Curve",
    save_path: Optional[str] = None,
) -> None:
    """Plot simple learning curves (mean ± std over folds).

    Expects arrays with shape (n_sizes, n_folds) for scores.
    """
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(7, 5))
    plt.title(title)
    plt.plot(train_sizes, train_mean, "-o", label="Train score")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
    plt.plot(train_sizes, val_mean, "-o", label="CV score")
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120)
    plt.close()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residuals",
    save_path: Optional[str] = None,
) -> None:
    """Residual plot and histogram for quick error analysis."""
    residuals = y_true - y_pred

    plt.figure(figsize=(12, 5))

    # Scatter: y_true vs y_pred
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.7)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", linewidth=1)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("y_true vs y_pred")

    # Histogram of residuals
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=25, alpha=0.8)
    plt.axvline(0, color="r", linestyle="--", linewidth=1)
    plt.xlabel("Residual (y_true - y_pred)")
    plt.title(title)

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120)
    plt.close()
