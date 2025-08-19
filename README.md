# Boston Housing Regression (Reproducible)
- Final test MAE: **1.77 k$**, R²: **0.918** (XGBoost, 5-fold CV grid search)
- Baselines: Linear 3.19, RandomForest 2.08
- Repro: `py src\train.py --data_path data\boston.csv --model xgb --seed 42 --out_dir runs\exp1`
- MEDV is in **$1,000s** → MAE 1.77 ≈ **$1,770**.