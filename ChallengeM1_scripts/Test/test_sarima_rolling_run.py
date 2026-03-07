"""
SARIMAX Rolling Forecast Verification Script

This script verifies the SARIMAX rolling 1-step forecast logic developed in the notebook.
It performs model selection via AIC on a grid of SARIMA parameters, then generates
rolling forecasts using state updates (no refitting) for computational efficiency.

The script uses Load data with weekly seasonality (period=7) and tests the fast
rolling forecast approach with .append(refit=False) for state updates.

Inspiration
-----------
SARIMAX implementation: Statsmodels library
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools

# Load data
base_path = Path(__file__).resolve().parent / ".." / "Data"
train = pd.read_csv(base_path / "train.csv", parse_dates=["Date"]).set_index("Date").sort_index()
test = pd.read_csv(base_path / "test.csv", parse_dates=["Date"]).set_index("Date").sort_index()
y_train = train["Load"]

# Model selection: AIC-based grid search (small grid for speed)
best_aic = float("inf")
best_order = (1, 0, 2)
best_seasonal_order = (1, 0, 2, 7)

for p, q in itertools.product([0, 1, 2], [0, 1, 2]):
    for P, Q in itertools.product([0, 1, 2], [0, 1, 2]):
        try:
            res = SARIMAX(
                y_train,
                order=(p, 0, q),
                seasonal_order=(P, 0, Q, 7),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            if res.aic < best_aic:
                best_aic, best_order, best_seasonal_order = res.aic, (p, 0, q), (P, 0, Q, 7)
        except Exception:
            pass

print("Best order", best_order, best_seasonal_order)

# Fit best model on full training data
best_res = SARIMAX(
    y_train,
    order=best_order,
    seasonal_order=best_seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False,
).fit(disp=False)

# Rolling forecast with state updates (no refit)
res = best_res
preds = []
test_index = test.index
load_lag1 = test["Load.1"]

for i, current_date in enumerate(test_index):
    if i > 0:
        # Update state with previous day's true load
        prev_date = test_index[i - 1]
        true_prev_load = load_lag1.iloc[i]
        new_obs = pd.Series([true_prev_load], index=[prev_date], name="Load")
        res = res.append(new_obs, refit=False)

    # One-step forecast
    fc = res.forecast(steps=1)
    preds.append(fc.iloc[0])

# Results
y_pred_test = pd.Series(preds, index=test_index, name="Load_pred_sarimax")
test_with_pred = test.copy()
test_with_pred["Load_pred_sarimax"] = y_pred_test

print("Len preds", len(preds))
print(test_with_pred[["Load.1", "Load_pred_sarimax"]].head(10))
