# House Prices (Ames) — Student ML Project (LightGBM + FE + Stratified CV)

- Build a baseline → improve with **feature engineering** → **tune** with CV → ship a **valid Kaggle submission**.
- Learn to keep things reproducible (seeds, versions, artifacts) and transparent (clear code comments and logs).

## Current Kaggle Result

- **Public Leaderboard:** `RMSLE = 0.12868` (and **MAE ≈ $16,225** on my local hold-out)
- This was achieved with **LightGBM**, **stratified CV on binned log(SalePrice)**, and a small set of **hand-crafted features** (TotalSF, baths, age/remodel flags, porch area, overall score).
