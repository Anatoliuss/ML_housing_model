# House Prices (Ames, LightGBM + FE + Stratified CV)

This is my **personal ML project** to practice a clean, reproducible workflow on the classic **Kaggle “House Prices: Advanced Regression Techniques”** dataset (Ames, Iowa).

## Current Kaggle Result

- **Public Leaderboard:** `RMSLE = 0.12868` (and **MAE ≈ $16,225** on my local hold-out)
- This was achieved with LightGBM, stratified CV on binned log(SalePrice), and a small set of features (TotalSF, baths, age/remodel flags, porch area, overall score).

---

## Repo Structure

```

.
├─ data/
│  ├─ train.csv            # (download from Kaggle)
│  └─ test.csv             # (download from Kaggle)
├─ runs/
│  └─ ames\_big/  -- a big Ames run
│  └─ ames\_pro/  -- a smaller run( the most effective for some reason)
│  └─ exp6/  -- a course run         
├─ src/
│  ├─ train\_ames.py        # Kaggle project 
│  ├─ train.py             # Boston Housing (I did a stanford x coursera course and reproduced the lab)
│  └─ utils.py             # helpers for the Boston project (metrics, plots, seeds)
└─ README.md

````

---

## Quickstart

**Set up environment (Windows PowerShell)**

```powershell
# create and activate a virtual env
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# install deps
py -m pip install -U pip
py -m pip install numpy pandas scikit-learn lightgbm joblib
````

**Get the data**

* Download `train.csv` and `test.csv` from the Kaggle competition page and place them in `./data`.

**Run (search mode – RandomizedSearchCV)**

```powershell
py src\train_ames.py --data_dir data --cv_folds 5 --n_iter 40 --out_dir runs\ames_pro --submit_path submission.csv
```

This will:

* Print library versions
* Do feature engineering + preprocessing
* Run a randomized hyperparameter search with **StratifiedKFold on binned log-target**
* Save artifacts to `runs\ames_pro\`
* Train a final model on full data and write `submission.csv` (1459 rows)

**Run (no-search mode – fast re-submit using saved params)**

```powershell
py src\train_ames.py --data_dir data --no_search --out_dir runs\ames_pro --submit_path submission.csv
```

---

## Feature Engineering

* `TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF`
* `TotalBath = FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath`
* `Age = YrSold - YearBuilt`, `RemodAge = YrSold - YearRemodAdd`, `IsRemodeled`
* `TotalPorchSF = OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch + WoodDeckSF`
* `OverallScore = OverallQual * OverallCond`



## Sranford x Coursera Project

I also included a simpler, “course-style” tabular project on the **Boston Housing** dataset:

* `src/train.py` — scikit-learn pipelines for Linear Regression, Random Forest, and tuned XGBoost/LightGBM
* `src/utils.py` — helper functions:

  * `set_global_seed`, `evaluate_regression` (MAE, R²)
  * `plot_residuals`, (and version logging in the newer code)

---

## 🔁 Reproducibility Notes

* Seeds fixed via `set_global_seed`.
* All versions printed at start.
* Best params + CV results saved for auditability and re-use.
* Final model saved to `final_model.pkl`.
