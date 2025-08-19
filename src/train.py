import argparse
import json
import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from utils import (
    set_global_seed,
    evaluate_regression,
    save_json,
    plot_residuals,
    print_versions,
)


def try_load_boston(data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load Boston Housing from OpenML if possible; else from local CSV.
    Returns features (X) and target (y).

    Expected target column: 'MEDV'.
    """
    # Try OpenML first
    try:
        from sklearn.datasets import fetch_openml
        ds = fetch_openml(name="boston", version=1, as_frame=True)
        df = ds.frame.copy()
        if "MEDV" not in df.columns:
            # Sometimes the column is lowercase or named differently on mirrors
            for cand in ["medv", "target"]:
                if cand in df.columns:
                    df.rename(columns={cand: "MEDV"}, inplace=True)
                    break
        if "MEDV" not in df.columns:
            raise ValueError("Could not find 'MEDV' in fetched dataset.")
        X = df.drop(columns=["MEDV"]).copy()
        y = df["MEDV"].copy()
        return X, y
    except Exception:
        # Fallback to CSV
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"OpenML fetch failed and local CSV not found at {data_path}.\n"
                f"Place a Boston Housing CSV there with a 'MEDV' target column."
            )
        df = pd.read_csv(data_path)
        if "MEDV" not in df.columns:
            raise ValueError("Expected target column 'MEDV' in local CSV.")
        X = df.drop(columns=["MEDV"]).copy()
        y = df["MEDV"].copy()
        return X, y


def build_preprocessing(numeric_features: List[str]) -> ColumnTransformer:
    """
    ColumnTransformer that imputes and scales numeric features.

    Note: Tree models don't strictly need scaling, but we keep a consistent
    preprocessing pipeline for reproducibility and for linear models.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
        ],
        remainder="drop",
        n_jobs=None,
    )
    return preprocessor


def get_default_xgb_params(seed: int) -> Dict:
    """Reasonable tuned-ish defaults for Boston (MAE ~20–22k typical)."""
    return dict(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=seed,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
    )


def get_default_lgbm_params(seed: int) -> Dict:
    return dict(
        n_estimators=800,
        max_depth=-1,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=seed,
        objective="regression",
        n_jobs=-1,
        verbose=-1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Boston Housing models")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/boston.csv",
        help="Path to local CSV if OpenML fetch is unavailable.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global seed")
    parser.add_argument(
        "--model",
        type=str,
        choices=["xgb", "lgbm"],
        default="xgb",
        help="Which GBM to use for the final model",
    )
    parser.add_argument("--cv_folds", type=int, default=5, help="KFold CV folds for tuning")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="runs/exp1",
        help="Where to save artifacts (model, metrics, predictions)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_global_seed(args.seed)
    print_versions()

    # 1) Load data
    X, y = try_load_boston(args.data_path)
    feature_names = list(X.columns)
    print(f"[Data] X shape: {X.shape}, y shape: {y.shape}")

    # 2) Train/test split (no peeking at test during CV)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed
    )

    # 3) Preprocessing pipeline
    preprocessor = build_preprocessing(numeric_features=feature_names)

    # 4) Baselines (quick sanity checks; not saved)
    for name, model in [
        ("linreg", LinearRegression()),
        ("rf", RandomForestRegressor(n_estimators=300, random_state=args.seed, n_jobs=-1)),
    ]:
        baseline_pipe = Pipeline(steps=[("pre", preprocessor), ("model", model)])
        baseline_pipe.fit(X_train, y_train)
        y_pred = baseline_pipe.predict(X_test)
        metrics = evaluate_regression(y_test.to_numpy(), y_pred)
        print(f"[Baseline: {name}] MAE={metrics['MAE']:.3f}, R2={metrics['R2']:.3f}")

    # 5) Tuned GBM (target model)
    if args.model == "xgb":
        gbm = XGBRegressor(**get_default_xgb_params(args.seed))
        param_grid = {
            "model__n_estimators": [400, 600, 800, 1000],
            "model__max_depth": [3, 4, 6, 8],
            "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
            "model__min_child_weight": [1.0, 3.0, 5.0],
        }
    else:
        gbm = LGBMRegressor(**get_default_lgbm_params(args.seed))
        param_grid = {
            "model__n_estimators": [400, 600, 800, 1000],
            "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
            "model__subsample": [0.7, 0.8, 1.0],
            "model__colsample_bytree": [0.7, 0.8, 1.0],
            "model__min_child_samples": [10, 20, 30],
        }

    pipe = Pipeline(steps=[("pre", preprocessor), ("model", gbm)])

    cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    # simpler, avoids sign confusion:
    scorer = "neg_mean_absolute_error"

    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        scoring=scorer,
        n_jobs=-1,
        refit=True,
        verbose=0,
        return_train_score=False,
    )
    grid.fit(X_train, y_train)

    best_pipe = grid.best_estimator_
    best_params = grid.best_params_
    print("[GridSearch] Best params:")
    print(json.dumps(best_params, indent=2))

    # Save best params and CV table
    save_json(best_params, os.path.join(args.out_dir, "best_params.json"))
    cv_df = pd.DataFrame(grid.cv_results_)
    cv_df.to_csv(os.path.join(args.out_dir, "cv_results.csv"), index=False)

    # 6) Final evaluation on test
    y_pred_test = best_pipe.predict(X_test)
    metrics = evaluate_regression(y_test.to_numpy(), y_pred_test)
    print(f"[Final] Test MAE={metrics['MAE']:.4f}, R2={metrics['R2']:.4f}")

    # 7) Save artifacts
    model_path = os.path.join(args.out_dir, "best_model.pkl")
    joblib.dump(best_pipe, model_path)

    metrics_path = os.path.join(args.out_dir, "metrics.json")
    save_json(metrics, metrics_path)

    preds_path = os.path.join(args.out_dir, "predictions.csv")
    pd.DataFrame({"y_true": y_test, "y_pred": y_pred_test}).to_csv(preds_path, index=False)

    # Residual plot for quick error analysis
    try:
        plot_residuals(
            y_test.to_numpy(),
            y_pred_test,
            save_path=os.path.join(args.out_dir, "residuals.png"),
        )
    except Exception as e:
        print(f"[Warn] Residual plot failed: {e}")

    # Optional Kaggle-like submission (format may differ per competition)
    sub_path = os.path.join(args.out_dir, "submission.csv")
    pd.DataFrame({"Id": np.arange(len(y_pred_test)), "MEDV": y_pred_test}).to_csv(
        sub_path, index=False
    )


if __name__ == "__main__":
    main()
