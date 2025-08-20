
import argparse
import json
import os
import warnings
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer, FunctionTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import randint, uniform, loguniform
from lightgbm import LGBMRegressor



# small helper functions

def set_global_seed(seed: int) -> None:
    """make runs reproducible enough for a student project"""
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def save_json(obj: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def print_versions() -> None:
    """print library versions for my README/repro notes"""
    import platform, sys
    import sklearn, pandas as pd, lightgbm
    print("[Versions]")
    print(f"Python: {sys.version.split()[0]}  |  Platform: {platform.platform()}")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"pandas: {pd.__version__}")
    print(f"lightgbm: {lightgbm.__version__}")
    print("-" * 40)

def rmsle_like(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """Kaggle evaluates RMSE on log(SalePrice), so just compute RMSE in log space."""
    return float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))


# data loading

def load_ames(data_dir: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    expects:
      data/train.csv  (has 'SalePrice')
      data/test.csv   (no 'SalePrice')
    """
    train_path = os.path.join(data_dir, "train.csv")
    test_path  = os.path.join(data_dir, "test.csv")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Place Kaggle files at:\n  {train_path}\n  {test_path}\n"
            "Download from: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data"
        )
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    if "SalePrice" not in train_df.columns:
        raise ValueError("train.csv must contain the 'SalePrice' column.")
    y = train_df["SalePrice"].copy()
    X = train_df.drop(columns=["SalePrice"]).copy()
    return X, y, test_df


# simple feature engineering (FE)
# these are the extra numeric features i found helpful
ENGINEERED_NUM_COLS = [
    "TotalSF", "TotalBath", "Age", "RemodAge", "IsRemodeled",
    "TotalPorchSF", "OverallScore"
]

def add_ames_features(df: pd.DataFrame) -> pd.DataFrame:
    """create a few hand-crafted features. safe to call on both train and test."""
    d = df.copy()
    g = lambda c: d[c] if c in d.columns else 0

    # bigger houses cost more
    d["TotalSF"] = g("TotalBsmtSF") + g("1stFlrSF") + g("2ndFlrSF")

    # bathrooms weighted by halves 
    d["TotalBath"] = g("FullBath") + 0.5*g("HalfBath") + g("BsmtFullBath") + 0.5*g("BsmtHalfBath")

    # how old is the house when sold
    d["Age"] = (d["YrSold"] - d["YearBuilt"]) if ("YrSold" in d and "YearBuilt" in d) else 0
    if "YrSold" in d.columns and "YearRemodAdd" in d.columns:
        d["RemodAge"] = d["YrSold"] - d["YearRemodAdd"]
        d["IsRemodeled"] = (d["YearRemodAdd"] != d.get("YearBuilt", d["YearRemodAdd"])).astype(int)
    else:
        d["RemodAge"] = 0
        d["IsRemodeled"] = 0

    d["TotalPorchSF"] = g("OpenPorchSF") + g("EnclosedPorch") + g("3SsnPorch") + g("ScreenPorch") + g("WoodDeckSF")

    #  overall quality * condition
    d["OverallScore"] = d.get("OverallQual", 0) * d.get("OverallCond", 0)

    return d

# preprocessing
def split_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """figure out which columns are numeric and which are categorical (after FE preview)"""
    numeric_cols = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols

def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """
    Numeric: median impute -> Yeo-Johnson (handles skew incl. zeros/negatives) -> Standardize
    Categorical: most_frequent impute -> OneHotEncoder (ignore unknowns)
    """
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("power", PowerTransformer(method="yeo-johnson")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        n_jobs=None,
    )
    return pre


# model defaults
def get_lgbm_for_cv(seed: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression",
        random_state=seed,
        n_jobs=1,   
        verbose=-1,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--n_iter", type=int, default=60)
    parser.add_argument("--out_dir", type=str, default="runs/ames_pro")
    parser.add_argument("--submit_path", type=str, default="submission.csv")
    parser.add_argument("--no_search", action="store_true")
    parser.add_argument("--params_path", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_global_seed(args.seed)
    print_versions()

    # Load raw data
    X, y, test_df = load_ames(args.data_dir)

    X_preview = add_ames_features(X.head(5))
    num_cols, cat_cols = split_columns(X_preview)
    print(f"[Columns after FE] numeric={len(num_cols)}, categorical={len(cat_cols)} (+{len(ENGINEERED_NUM_COLS)} engineered)")

    pre = build_preprocessor(num_cols, cat_cols)

    if args.no_search:
        params_path = args.params_path or os.path.join(args.out_dir, "best_params.json")
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"--no_search set but params file not found: {params_path}")
        with open(params_path, "r", encoding="utf-8") as f:
            best_params = json.load(f)
        final_lgbm = LGBMRegressor(objective="regression", random_state=args.seed, n_jobs=-1, verbose=-1)
        for k, v in best_params.items():
            assert k.startswith("model__")
            setattr(final_lgbm, k.split("__", 1)[1], v)

        final_pipe = Pipeline(steps=[
            ("fe", FunctionTransformer(add_ames_features)),
            ("pre", pre),
            ("model", final_lgbm),
        ])
        final_pipe.fit(X, np.log1p(y.values))
        joblib.dump(final_pipe, os.path.join(args.out_dir, "final_model.pkl"))

        test_ids = test_df["Id"].values
        test_pred_log = final_pipe.predict(test_df)
        test_pred = np.expm1(test_pred_log)
        submission = pd.DataFrame({"Id": test_ids, "SalePrice": test_pred})
        submission.to_csv(args.submit_path, index=False)
        submission.to_csv(os.path.join(args.out_dir, "submission.csv"), index=False)
        print(f"[Submit] Wrote {args.submit_path} with {len(submission)} rows.")
        print("[Done --no_search]")
        return

    # train/valid split
    X_tr, X_va, y_tr_raw, y_va_raw = train_test_split(X, y, test_size=0.2, random_state=args.seed)
    y_tr = np.log1p(y_tr_raw.values)  # log targets for training metric
    y_va = np.log1p(y_va_raw.values)

    y_all_log = np.log1p(y.values)
    bins_all = pd.qcut(y_all_log, q=10, labels=False, duplicates="drop")
    bins_tr = bins_all[X_tr.index]  # only bins for the training subset

    cv_strat = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    cv_splits = list(cv_strat.split(X_tr, bins_tr))  # pass splits so scoring uses real y_tr

    lgbm_cv = get_lgbm_for_cv(args.seed)
    pipe = Pipeline(steps=[
        ("fe", FunctionTransformer(add_ames_features)),
        ("pre", pre),
        ("model", lgbm_cv),
    ])
    param_dist = {
        "model__n_estimators": randint(900, 2200),
        "model__learning_rate": loguniform(0.008, 0.06),
        "model__num_leaves": randint(32, 96),
        "model__min_child_samples": randint(8, 45),
        "model__subsample": uniform(0.7, 0.25),         # 0.70–0.95
        "model__colsample_bytree": uniform(0.7, 0.25),  # 0.70–0.95
        "model__reg_lambda": loguniform(1e-3, 3.0),
        "model__reg_alpha": loguniform(1e-4, 0.5),
    }

    scorer = "neg_root_mean_squared_error"  # aligns with RMSE on log target
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=args.n_iter,             # try 25 for laptops, 60+ on Kaggle/Colab
        scoring=scorer,
        cv=cv_splits,                   # our stratified splits on binned log target
        n_jobs=-1,                      # parallelize folds/iters
        refit=True,
        verbose=1,
        random_state=args.seed,
        return_train_score=False,
    )
    search.fit(X_tr, y_tr)

    #  persist best params + CV table 
    best_pipe = search.best_estimator_
    best_params = search.best_params_
    print("[Search] Best params:")
    print(json.dumps(best_params, indent=2))
    save_json(best_params, os.path.join(args.out_dir, "best_params.json"))
    pd.DataFrame(search.cv_results_).to_csv(os.path.join(args.out_dir, "cv_results.csv"), index=False)

 
    y_va_pred_log = best_pipe.predict(X_va)
    valid_rmse_log = rmsle_like(y_va, y_va_pred_log)
    y_va_pred = np.expm1(y_va_pred_log)
    valid_mae_usd = float(mean_absolute_error(y_va_raw.values, y_va_pred))
    valid_r2 = float(r2_score(y_va_raw.values, y_va_pred))
    metrics = {
        "valid_RMSE_log": valid_rmse_log,
        "valid_MAE_dollars": valid_mae_usd,
        "valid_R2": valid_r2,
    }
    print(f"[Final] Valid log-RMSE={valid_rmse_log:.5f} | MAE=${valid_mae_usd:,.0f} | R2={valid_r2:.4f}")
    save_json(metrics, os.path.join(args.out_dir, "metrics.json"))

    final_lgbm = LGBMRegressor(objective="regression", random_state=args.seed, n_jobs=-1, verbose=-1)
    for k, v in best_params.items():
        assert k.startswith("model__")
        setattr(final_lgbm, k.split("__", 1)[1], v)
    final_pipe = Pipeline(steps=[
        ("fe", FunctionTransformer(add_ames_features)),
        ("pre", pre),
        ("model", final_lgbm),
    ])
    final_pipe.fit(X, np.log1p(y.values))
    joblib.dump(final_pipe, os.path.join(args.out_dir, "final_model.pkl"))

    #predict Kaggle test and write submission
    test_ids = test_df["Id"].values
    test_pred_log = final_pipe.predict(test_df)
    test_pred = np.expm1(test_pred_log)
    submission = pd.DataFrame({"Id": test_ids, "SalePrice": test_pred})
    submission.to_csv(args.submit_path, index=False)
    submission.to_csv(os.path.join(args.out_dir, "submission.csv"), index=False)
    print(f"[Submit] Wrote {args.submit_path} with {len(submission)} rows.")
    print("[Done]")


if __name__ == "__main__":
    main()
