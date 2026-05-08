# F1-Pit-Stop-Prediction

## Problem
Predict whether the F1 car will pit in the next lap or not

## Dataset
439140 rows, 16 features, binary classification.

## Approach
- EDA: Found class imbalance, 
  LapNumber and RaceProgress as dominant features
- Feature Engineering: DegradationByStint, LapProgress, TyreLife_lag1,
- TyreLife_diff, driver_encoded, racer_encoded, compound_encoded
- Models: LightGBM, CatBoost, XGBoost
- Validation: StratifiedKFold (5 fold) due to class imbalance

## Results
Public leaderboard score: 0.94911
Best CV score: 0.9497

## Key Learnings
- optuna tuning
- ensemble learning by using 3 strong models
  
