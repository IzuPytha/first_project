
# Match Predictor Documentation
## Overview

This file provides an overview of the trained machine learning model used to predict football match results and number of goals scored.
The models were built using scikit-learn and XGBoost.

### Model Details

Type: XGBoost Classifier.

Training Dataset: Processed match data containing team stats, form,etc.

Target Variable: Match outcome (Home Win, Draw, Away Win) and Home and Away goals scored.

Hyperparameter Tuning: Used Hyperopt to optimize model performance.

**Features Used**

Team Information: Home team, Away team.

Recent Performance: Win/loss records, match stats.

**Model Performance**

*Evaluation Metrics*: Accuracy, Root mean squared error(RMSE).

Best Hyperparameters:

For goal_predictions.pkl

colsample_bytree: 0.8278581623261667

early_stopping_rounds : 80

gamma : 3.108377905034552

max_depth : 12

min_child_weight : 8

reg_alpha : 48

reg_lambda : 0.4848165801185895

For result_predictions.pkl:

colsample_bytree: 0.9705885718518942

early_stopping_rounds : 100

gamma : 6.551015896491389

max_depth : 8

min_child_weight : 7

reg_alpha : 53

reg_lambda : 0.5818292174394918

### Deployment
The model is deployed using Streamlit, allowing users to input match details and receive a predicted outcome.

**Running the Model in Streamlit**
Ensure all dependencies are installed:
```pip install -r requirements.txt```.
Run the Streamlit application:
```streamlit run app.py```
Upload match data or input team details manually.

The model will generate and display the predicted match result.

## Possible Issues & Fixes

FileNotFoundError: Ensure the 2 models are in the correct directory.
