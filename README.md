# F1-Pit-Stop-Prediction

## Problem
Predict whether the F1 car will pit in the next lap or not

## About
- This is competition of F1 Pit Stop which was featured on Kaggle. In this competition you have to predict
- if the F1 car will pit stop in the next lap or not.

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

## How to run app
- Copy the code and paste in python file then run the file with streamlit. Set the features to values and
- click predict button to get pit stop prediction

## Key Learnings
- optuna tuning
- ensemble learning by using 3 strong models
  
