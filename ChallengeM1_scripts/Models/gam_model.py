"""
Simple GAM experiment script for expectile-based approximate quantile regression.

Design choices and notes:
- Uses pyGAM (https://pygam.readthedocs.io/). Picked because it's a
  standard, well-documented Python library for GAMs.
- pyGAM does not implement direct pinball (quantile) loss. It implements
  ExpectileGAM which fits "expectiles" (asymmetric squared error). Expectiles
  are not identical to quantiles (pinball loss), but are a close and standard
  alternative when a quantile GAM is unavailable. See comments below.
- Reuses the same features, one-hot encoding for WeekDays and the same
  normalization (continuous columns standardized) as in `test_linear.ipynb`.
- Minimal, readable code with clear output logs. No hyperparameter tuning.

Run from the repo root or from the `Models` directory. If `pygam` is not
installed, install it with: pip install pygam
"""

import os
import sys
import numpy as np
import pandas as pd
from pygam import s, ExpectileGAM

sys.path.append('../Python')
from score import pinball_loss
from Linear import LinearRegression


# Utility functions reused from the notebook to keep feature handling consistent
def find_continuous_columns(X):
    continuous_cols = []
    for col in X.columns:
        s = X[col]
        if s.dtype == bool:
            continue
        vals = pd.unique(s.dropna())
        if len(vals) <= 2 and set(vals).issubset({0, 1}):
            continue
        continuous_cols.append(col)
    return continuous_cols


def normalize(X, scale_cols=None):
    if isinstance(X, pd.DataFrame):
        X_norm = X.copy()
        if scale_cols is None:
            scale_cols = find_continuous_columns(X_norm)
        mean = X_norm[scale_cols].mean(axis=0)
        std = X_norm[scale_cols].std(axis=0)
        std = std.replace(0, 1.)
        if len(scale_cols) > 0:
            X_norm[scale_cols] = (X_norm[scale_cols] - mean) / std
        return X_norm, mean, std
    else:
        raise ValueError("normalize expects a pandas DataFrame")


def build_gam_terms(n_features):
    # Build terms s(0) + s(1) + ... + s(n_features)
    terms = s(0)
    for i in range(1, n_features):
        terms = terms + s(i)
    return terms


def main():
    # Reproducibility
    np.random.seed(0)

    print("Loading data...")
    Data_train = Data_train = pd.read_csv(
        "../Data/train.csv",
        parse_dates=["Date"])
    
    Data_test = Data_train = pd.read_csv(
        "../Data/tets.csv",
        parse_dates=["Date"])

    # Build features exactly like in the notebook
    X_train = Data_train.drop(columns=["Net_demand", "Date", "Solar_power", "Wind_power", "Load"])
    y_train = Data_train["Net_demand"]
    X_test = Data_test.drop(columns=["Date", "Usage", "Id"])

    X_train = pd.get_dummies(X_train, columns=['WeekDays'], prefix='WeekDays', drop_first=True, dtype=float)
    X_test = pd.get_dummies(X_test, columns=['WeekDays'], prefix='WeekDays', drop_first=True, dtype=float)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)

    # Use the same normalization as in notebook (important for comparability)
    scale_cols = find_continuous_columns(X_train)
    X_train, X_mean, X_std = normalize(X_train, scale_cols=scale_cols)
    # Apply same scaling to test set
    X_test = X_test.copy()
    if len(scale_cols) > 0:
        X_test[scale_cols] = (X_test[scale_cols] - X_mean) / X_std.replace(0, 1.0)

    # Numpy arrays for models
    X_train_np = X_train.values
    X_test_np = X_test.values
    y_train_np = y_train.values

    # target quantile/expectile
    tau = 0.8

    print("Building GAM terms and fitting ExpectileGAM (expectile=0.8).")
    terms = build_gam_terms(X_train_np.shape[1])

    # We keep default smoothing penalties and small model - no hyperparameter tuning here
    gam = ExpectileGAM(terms=terms, expectile=tau)

    print("Fitting GAM (this may take a short while)...")
    gam.fit(X_train_np, y_train_np)

    # Produce predictions on the official test set (no ground-truth available there)
    yhat_gam_test = gam.predict(X_test_np)

    # Create a time-based validation holdout from the training data for evaluation (last 20%)
    N = X_train_np.shape[0]
    split = int(N * 0.8)
    X_tr = X_train_np[:split]
    y_tr = y_train_np[:split]
    X_val = X_train_np[split:]
    y_val = y_train_np[split:]

    # Re-fit GAM on the training split to evaluate on the holdout (keeps train/val consistent)
    gam.fit(X_tr, y_tr)
    yhat_gam_val = gam.predict(X_val)

    # Evaluate on validation set
    pb_gam = pinball_loss(y_val, yhat_gam_val, quant=tau)
    coverage_gam = float(np.mean(y_val <= yhat_gam_val))

    # --- Baseline: Linear pinball model (explicit gradient-based solver) ---
    print("Fitting baseline Linear pinball model (tau=0.8) on the same train split...")
    lin = LinearRegression(learning_rate=0.02, maxIter=8000, tau=tau)
    lin.fit(X_tr, y_tr, loss="pinball", verbose=False)
    yhat_lin_val = lin.predict(X_val)
    pb_lin = pinball_loss(y_val, yhat_lin_val, quant=[tau])
    coverage_lin = float(np.mean(y_val <= yhat_lin_val))

    # Basic check for divergence / poor convergence of gradient-based linear solver
    if pb_lin > 1e5 or coverage_lin < 0.01:
        print("Warning: Linear baseline shows very large loss or near-zero coverage !!!")

    # Quick checks and printouts
    print("\n--- Summary (tau=0.8) ---")
    print("GAM (Expectile) loss -> : {:.6f}".format(pb_gam))
    print("GAM P(y <= y_hat): {:.3f}".format(coverage_gam))
    print("Linear pinball loss -> : {:.6f}".format(pb_lin))
    print("Linear P(y <= y_hat): {:.3f}".format(coverage_lin))

    # Confirm that GAM learned non-linear effects and outputs are on same scale
    print("\nDebug checks:")
    print(" mean(y_train_split)={:.3f}, mean(yhat_gam_val)={:.3f}, mean(yhat_lin_val)={:.3f}".format(
        float(y_tr.mean()), float(yhat_gam_val.mean()), float(yhat_lin_val.mean())
    ))

    # Check simple non-linearity presence by comparing linear vs GAM fit on training data
    yhat_gam_train = gam.predict(X_train_np)
    yhat_lin_train = lin.predict(X_train_np)

    # If GAM predictions differ non-trivially from linear predictions, we likely have non-linear effects
    diff = np.mean(np.abs(yhat_gam_train - yhat_lin_train))
    print(" mean absolute difference (GAM vs Linear) on train predictions: {:.3f}".format(diff))
    if diff > 1e-3:
        print(" Note: GAM shows non-linear behavior (predictions differ from linear model).")
    else:
        print(" Note: GAM predictions are very close to linear model -> may not have learned strong non-linearities.")

    print("\nDone.")

if __name__ == '__main__':
    main()
