"""
Simple GAM experiment script for expectile-based approximate quantile regression.

Design choices and notes:
- Uses pyGAM (https://pygam.readthedocs.io/). Picked because it's a
  standard, well-documented Python library for Generalized Additive Models.
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

# Make sure we can import supporting scripts regardless of current working dir
HERE = os.path.dirname(__file__)
sys.path.append(os.path.join(HERE, '..', 'Python'))  # for score.py

# Import pinball_loss implementation used elsewhere in the project
try:
    from score import pinball_loss
except Exception:
    # Fallback: simple local implementation (keeps script runnable)
    def pinball_loss(y, yhat_quant, quant, output_vect=False):
        yhat_quant = np.asarray(yhat_quant)
        quant = np.asarray(quant)
        if yhat_quant.ndim == 1:
            yhat_quant = yhat_quant[:, None]
        nq = yhat_quant.shape[1]
        loss_q = np.zeros(nq)
        for q in range(nq):
            loss_q[q] = np.nanmean(
                (y - yhat_quant[:, q]) * (quant[q] - (y < yhat_quant[:, q]))
            )
        if output_vect:
            return loss_q
        else:
            return np.mean(loss_q)


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
    # Lazily build terms s(0) + s(1) + ... for pygam
    from pygam import s
    terms = s(0)
    for i in range(1, n_features):
        terms = terms + s(i)
    return terms


def main():
    # Reproducibility
    np.random.seed(0)

    # Load data using same paths as the notebook
    data_dir = os.path.join(HERE, '..', 'Data')
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')

    print("Loading data...")
    Data_train = pd.read_csv(train_path, parse_dates=["Date"]) 
    Data_test = pd.read_csv(test_path, parse_dates=["Date"]) 

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

    # --- Fit Expectile GAM (pyGAM) ---
    try:
        from pygam import ExpectileGAM
    except ImportError:
        print("pygam is not installed. Install it with: pip install pygam")
        return

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
    pb_gam = pinball_loss(y_val, yhat_gam_val, quant=[tau])
    coverage_gam = float(np.mean(y_val <= yhat_gam_val))

    # --- Baseline: Linear pinball model from project (explicit gradient-based solver) ---
    # Import local LinearRegression implementation
    sys.path.append(HERE)  # ensure current Models dir is importable
    try:
        from Linear import LinearRegression
    except Exception as e:
        print("Could not import LinearRegression from Linear.py; skipping baseline comparison.", e)
        LinearRegression = None

    if LinearRegression is not None:
        print("Fitting baseline Linear pinball model (tau=0.8) on the same train split...")
        lin = LinearRegression(learning_rate=0.02, maxIter=17000, tau=tau)
        lin.fit(X_tr, y_tr, loss="pinball", verbose=False)
        yhat_lin_val = lin.predict(X_val)
        pb_lin = pinball_loss(y_val, yhat_lin_val, quant=[tau])
        coverage_lin = float(np.mean(y_val <= yhat_lin_val))

        # Basic check for divergence / poor convergence of gradient-based linear solver
        if pb_lin > 1e5 or coverage_lin < 0.01:
            print("Warning: Linear baseline shows very large loss or near-zero coverage. Consider tuning learning_rate/maxIter or checking feature scaling.")
    else:
        pb_lin = None
        coverage_lin = None

    # Quick checks and printouts
    print("\n--- Summary (tau=0.8) ---")
    print("GAM (Expectile) -> pinball_loss (on same tau): {:.6f}".format(pb_gam))
    print("GAM empirical coverage P(y <= y_hat): {:.3f}".format(coverage_gam))
    if pb_lin is not None:
        print("Linear pinball -> pinball_loss: {:.6f}".format(pb_lin))
        print("Linear empirical coverage P(y <= y_hat): {:.3f}".format(coverage_lin))

    # Confirm that GAM learned non-linear effects and outputs are on same scale
    print("\nDebug checks:")
    print(" mean(y_train_split)={:.3f}, mean(yhat_gam_val)={:.3f}, mean(yhat_lin_val)={:.3f}".format(
        float(y_tr.mean()), float(yhat_gam_val.mean()), float(yhat_lin_val.mean() if LinearRegression else np.nan)
    ))

    # Check simple non-linearity presence by comparing linear vs GAM fit on training data
    yhat_gam_train = gam.predict(X_train_np)
    if LinearRegression is not None:
        yhat_lin_train = lin.predict(X_train_np)
        # If GAM predictions differ non-trivially from linear predictions, we likely have non-linear effects
        diff = np.mean(np.abs(yhat_gam_train - yhat_lin_train))
        print(" mean absolute difference (GAM vs Linear) on train predictions: {:.3f}".format(diff))
        if diff > 1e-3:
            print(" Note: GAM shows non-linear behavior (predictions differ from linear model). ✅")
        else:
            print(" Note: GAM predictions are very close to linear model -> may not have learned strong non-linearities. ⚠️")
    else:
        print(" Note: No linear baseline to compare non-linearity against.")

    print("\nDone.")


if __name__ == '__main__':
    main()
