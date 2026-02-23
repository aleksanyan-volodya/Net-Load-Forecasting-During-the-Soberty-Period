"""
XGBoost 1-day-ahead Wind_power forecast with rolling 1-step approach.
Objective: minimize pinball loss at quantile 0.6 on test.csv.
No future leakage; features built from past only.
"""

import numpy as np
import pandas as pd
from pathlib import Path

try:
    import xgboost as xgb
except ImportError:
    raise ImportError("Install xgboost: pip install xgboost")

# Project score module (pinball at quantile)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Python.score import pinball_loss

# Paths
DATA_DIR = Path(__file__).resolve().parents[1] / "Data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

QUANTILE_ALPHA = 0.8
VAL_FRAC = 0.2
RANDOM_STATE = 42
NUM_BOOST_ROUND = 1000  # increase for production; set lower (e.g. 300) for quick runs

# Feature names (must match train and test construction)
FEATURE_NAMES = [
    "lag1", "lag2", "lag7", "lag14",
    "roll_mean_7", "roll_mean_14", "roll_std_7",
    "dow", "month", "day_of_year",
    "doy_sin", "doy_cos",
]


def build_train_features(df: pd.DataFrame):
    """
    Build features and target from train DataFrame.
    All features use only past information (no leakage).
    - Lags 1, 2, 7, 14 from Wind_power
    - Rolling mean 7, 14 and rolling std 7 (computed on past values, then shifted)
    - Calendar: day of week, month, day of year, doy sin/cos
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    y = df["Wind_power"].copy()

    # Lags (already past: lag 1 = previous day, etc.)
    df["lag1"] = df["Wind_power"].shift(1)
    df["lag2"] = df["Wind_power"].shift(2)
    df["lag7"] = df["Wind_power"].shift(7)
    df["lag14"] = df["Wind_power"].shift(14)

    # Rolling stats: use only past. So at t we use values t-7..t-1 (shift(1) then rolling(7))
    past = df["Wind_power"].shift(1)
    df["roll_mean_7"] = past.rolling(7, min_periods=1).mean()
    df["roll_mean_14"] = past.rolling(14, min_periods=1).mean()
    df["roll_std_7"] = past.rolling(7, min_periods=1).std()

    # Calendar (from date at t, no future)
    df["dow"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["day_of_year"] = df["Date"].dt.dayofyear
    doy = df["day_of_year"] / 365.25
    df["doy_sin"] = np.sin(2 * np.pi * doy)
    df["doy_cos"] = np.cos(2 * np.pi * doy)

    X = df[FEATURE_NAMES]
    return X, y


def build_test_row_features(
    history: list,
    wind_power_1: float,
    date: pd.Timestamp,
) -> np.ndarray:
    """
    Build feature vector for one test row using:
    - history: list of past Wind_power (most recent last). history[-1] = yesterday.
    - wind_power_1: true previous day (same as history[-1] when updated).
    - date: current row date for calendar.
    """
    n = len(history)
    lag1 = wind_power_1
    lag2 = history[-2] if n >= 2 else np.nan
    lag7 = history[-7] if n >= 7 else np.nan
    lag14 = history[-14] if n >= 14 else np.nan

    arr = np.array(history, dtype=float)
    roll7 = arr[-7:] if n >= 7 else arr
    roll14 = arr[-14:] if n >= 14 else arr
    roll_mean_7 = np.nanmean(roll7) if len(roll7) else np.nan
    roll_mean_14 = np.nanmean(roll14) if len(roll14) else np.nan
    roll_std_7 = np.nanstd(roll7) if len(roll7) >= 2 else np.nan

    dow = date.dayofweek
    month = date.month
    day_of_year = date.dayofyear
    doy = day_of_year / 365.25
    doy_sin = np.sin(2 * np.pi * doy)
    doy_cos = np.cos(2 * np.pi * doy)

    return np.array([[
        lag1, lag2, lag7, lag14,
        roll_mean_7, roll_mean_14, roll_std_7,
        dow, month, day_of_year,
        doy_sin, doy_cos,
    ]], dtype=np.float64)


def main():
    print("Loading data...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    test["Date"] = pd.to_datetime(test["Date"])

    print("Building train features (no leakage)...")
    X_train_full, y_train_full = build_train_features(train)
    # Drop rows with NaN from lags/rolling (first 14 rows)
    valid = X_train_full.notna().all(axis=1) & y_train_full.notna()
    X_train_full = X_train_full.loc[valid].astype(np.float64)
    y_train_full = y_train_full.loc[valid].astype(np.float64)

    n = len(X_train_full)
    n_val = int(n * VAL_FRAC)
    n_tr = n - n_val
    X_tr = X_train_full.iloc[:n_tr]
    y_tr = y_train_full.iloc[:n_tr]
    X_val = X_train_full.iloc[n_tr:]
    y_val = y_train_full.iloc[n_tr:]

    print(f"Train size {n_tr}, Val size {n_val}")

    # Custom pinball metric for early stopping (quantile 0.6)
    def pinball_60(y_pred, dtrain):
        y = dtrain.get_label()
        res = np.mean((y - y_pred) * (QUANTILE_ALPHA - (y < y_pred)))
        return "pinball_60", res

    params = {
        "objective": "reg:quantileerror",
        "quantile_alpha": QUANTILE_ALPHA,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "tree_method": "hist",
        "verbosity": 0,
    }

    print("Training with early stopping on validation pinball (0.6)...")
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    evals = [(dtrain, "train"), (dval, "val")]
    evals_result = {}
    try:
        model_booster = xgb.train(
            params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            evals=evals,
            custom_metric=pinball_60,
            evals_result=evals_result,
            verbose_eval=False,
        )
    except TypeError:
        model_booster = xgb.train(
            params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            evals=evals,
            feval=pinball_60,
            evals_result=evals_result,
            verbose_eval=False,
        )
    val_pb = evals_result["val"]["pinball_60"]
    best_iter = int(np.argmin(val_pb)) + 1
    print(f"Best iteration (val pinball 0.6): {best_iter}, val pinball = {val_pb[best_iter-1]:.6f}")

    # Retrain on full train with best number of trees
    print("Retraining on full train set with best n_estimators...")
    dtrain_full = xgb.DMatrix(X_train_full, label=y_train_full)
    model_final_booster = xgb.train(
        params,
        dtrain_full,
        num_boost_round=best_iter,
        verbose_eval=False,
    )
    # Use booster for predict (DMatrix); pass feature names so booster accepts the array
    def predict_fn(X):
        return model_final_booster.predict(xgb.DMatrix(X, feature_names=FEATURE_NAMES))

    # Rolling 1-step prediction on test
    # history: Wind_power series from train (last 14+ values), then we append true values as we go
    wind_series = train["Wind_power"].astype(float).tolist()
    history = wind_series[-14:] if len(wind_series) >= 14 else wind_series.copy()
    if len(history) < 14:
        history = ([np.nan] * (14 - len(history))) + history

    predictions = []
    for i in range(len(test)):
        row = test.iloc[i]
        wind_power_1 = float(row["Wind_power.1"])
        date = row["Date"]
        # For first row, history ends with train's last day; Wind_power.1 is that day (previous to test date 0)
        # So we use wind_power_1 as lag1. History might already end with that value; ensure we have 14 past
        if i == 0:
            # history = last 14 of train. Row 0's Wind_power.1 is the day before first test date = last train day
            # So we don't change history; lag1 = wind_power_1
            pass
        else:
            # Append true value for previous test date ( = current row's Wind_power.1 )
            history.append(wind_power_1)

        feat = build_test_row_features(history, wind_power_1, date)
        pred = predict_fn(feat)[0]
        predictions.append(pred)

    predictions = np.array(predictions)

    # Actuals: for test row i, the true Wind_power is test row i+1's Wind_power.1
    # So we have actuals for rows 0..n-2 (n-1 points)
    n_eval = len(test) - 1
    y_test = test["Wind_power.1"].iloc[1 : n_eval + 1].astype(float).values
    pred_test = predictions[:n_eval]

    test_pinball = pinball_loss(y_test, pred_test, quant=np.array([QUANTILE_ALPHA]))
    print(f"\nTest pinball loss (quantile {QUANTILE_ALPHA}): {test_pinball:.6f}")
    print(f"Evaluated on {n_eval} test points (rows 0..{n_eval-1}; actuals from Wind_power.1 of next row).")

    # Optionally save forecasts
    out_dir = Path(__file__).resolve().parents[1] / "Results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "xgboost_wind_forecasts.csv"
    pd.DataFrame({
        "Date": test["Date"].iloc[:n_eval],
        "Wind_power_pred": pred_test,
        "Wind_power_actual": y_test,
    }).to_csv(out_path, index=False)
    print(f"Forecasts saved to {out_path}")

    pd.DataFrame({"Id": test["Id"], "Wind_forecast": predictions}).to_csv(out_dir / "xgboost_wind_forecast_id.csv", index=False)

    return test_pinball


if __name__ == "__main__":
    main()
