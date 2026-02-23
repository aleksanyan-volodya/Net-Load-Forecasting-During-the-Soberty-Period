"""Run SARIMAX rolling 1-step forecast with expand (no refit). Verifies notebook logic."""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools

base_path = Path(__file__).resolve().parent / ".." / "Data"
train = pd.read_csv(base_path / "train.csv", parse_dates=["Date"]).set_index("Date").sort_index()
test = pd.read_csv(base_path / "test.csv", parse_dates=["Date"]).set_index("Date").sort_index()
y_train = train["Load"]

# AIC selection (small grid)
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

best_res = SARIMAX(
    y_train,
    order=best_order,
    seasonal_order=best_seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False,
).fit(disp=False)

res = best_res
preds = []
test_index = test.index
load_lag1 = test["Load.1"]
for i, current_date in enumerate(test_index):
    if i > 0:
        prev_date = test_index[i - 1]
        true_prev_load = load_lag1.iloc[i]
        new_obs = pd.Series([true_prev_load], index=[prev_date], name="Load")
        res = res.append(new_obs, refit=False)
    fc = res.forecast(steps=1)
    preds.append(fc.iloc[0])

y_pred_test = pd.Series(preds, index=test_index, name="Load_pred_sarimax")
test_with_pred = test.copy()
test_with_pred["Load_pred_sarimax"] = y_pred_test
print("Len preds", len(preds))
print(test_with_pred[["Load.1", "Load_pred_sarimax"]].head(10))
