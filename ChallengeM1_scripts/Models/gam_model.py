"""GAM experiment using pyGAM ExpectileGAM."""

import os
import sys
import numpy as np
import pandas as pd
from pygam import s, ExpectileGAM

sys.path.append('../Python')
from score import pinball_loss
from Linear import LinearRegression

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
    terms = s(0)
    for i in range(1, n_features):
        terms = terms + s(i)
    return terms


def main():
    np.random.seed(0)

    print("Loading data...")
    Data_train = Data_train = pd.read_csv(
        "../Data/train.csv",
        parse_dates=["Date"])
    
    Data_test = Data_train = pd.read_csv(
        "../Data/tets.csv",
        parse_dates=["Date"])

    X_train = Data_train.drop(columns=["Net_demand", "Date", "Solar_power", "Wind_power", "Load"])
    y_train = Data_train["Net_demand"]
    X_test = Data_test.drop(columns=["Date", "Usage", "Id"])

    X_train = pd.get_dummies(X_train, columns=['WeekDays'], prefix='WeekDays', drop_first=True, dtype=float)
    X_test = pd.get_dummies(X_test, columns=['WeekDays'], prefix='WeekDays', drop_first=True, dtype=float)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)

    scale_cols = find_continuous_columns(X_train)
    X_train, X_mean, X_std = normalize(X_train, scale_cols=scale_cols)
    X_test = X_test.copy()
    if len(scale_cols) > 0:
        X_test[scale_cols] = (X_test[scale_cols] - X_mean) / X_std.replace(0, 1.0)

    X_train_np = X_train.values
    X_test_np = X_test.values
    y_train_np = y_train.values

    tau = 0.8

    print("Building GAM terms and fitting ExpectileGAM (expectile=0.8).")
    terms = build_gam_terms(X_train_np.shape[1])

    gam = ExpectileGAM(terms=terms, expectile=tau)

    print("Fitting GAM (this may take a short while)...")
    gam.fit(X_train_np, y_train_np)

    yhat_gam_test = gam.predict(X_test_np)

    N = X_train_np.shape[0]
    split = int(N * 0.8)
    X_tr = X_train_np[:split]
    y_tr = y_train_np[:split]
    X_val = X_train_np[split:]
    y_val = y_train_np[split:]

    gam.fit(X_tr, y_tr)
    yhat_gam_val = gam.predict(X_val)

    pb_gam = pinball_loss(y_val, yhat_gam_val, quant=tau)
    coverage_gam = float(np.mean(y_val <= yhat_gam_val))

    print("Fitting baseline Linear pinball model (tau=0.8) on the same train split...")
    lin = LinearRegression(learning_rate=0.02, maxIter=8000, tau=tau)
    lin.fit(X_tr, y_tr, loss="pinball", verbose=False)
    yhat_lin_val = lin.predict(X_val)
    pb_lin = pinball_loss(y_val, yhat_lin_val, quant=[tau])
    coverage_lin = float(np.mean(y_val <= yhat_lin_val))

    if pb_lin > 1e5 or coverage_lin < 0.01:
        print("Warning: Linear baseline shows very large loss or near-zero coverage !!!")

    print("\n--- Summary (tau=0.8) ---")
    print("GAM (Expectile) loss -> : {:.6f}".format(pb_gam))
    print("GAM P(y <= y_hat): {:.3f}".format(coverage_gam))
    print("Linear pinball loss -> : {:.6f}".format(pb_lin))
    print("Linear P(y <= y_hat): {:.3f}".format(coverage_lin))

    print("\nDebug checks:")
    print(" mean(y_train_split)={:.3f}, mean(yhat_gam_val)={:.3f}, mean(yhat_lin_val)={:.3f}".format(
        float(y_tr.mean()), float(yhat_gam_val.mean()), float(yhat_lin_val.mean())
    ))

    yhat_gam_train = gam.predict(X_train_np)
    yhat_lin_train = lin.predict(X_train_np)

    diff = np.mean(np.abs(yhat_gam_train - yhat_lin_train))
    print(" mean absolute difference (GAM vs Linear) on train predictions: {:.3f}".format(diff))
    if diff > 1e-3:
        print(" Note: GAM shows non-linear behavior (predictions differ from linear model).")
    else:
        print(" Note: GAM predictions are very close to linear model -> may not have learned strong non-linearities.")

    print("\nDone.")

if __name__ == '__main__':
    main()
