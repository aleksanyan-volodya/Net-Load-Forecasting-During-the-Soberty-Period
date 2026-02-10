"""
Compare Linear pinball (baseline) vs Linear pinball + spline-expanded features (GAM-style)

- Uses `spline_features.add_spline_features` to expand ['Temp','toy','Load.1'] only.
- Normalizes continuous (non-binary) columns after expansion (same behavior as notebook).
- Trains the project's `LinearRegression` in pinball mode (same hyperparameters as prior experiments).
- Prints clear, labeled outputs and simple correctness checks.
"""

import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
# Import project LinearRegression
sys.path.append(HERE)
# Add project Python utilities path (for score.py)
sys.path.append(os.path.join(HERE, '..', 'Python'))
from Linear import LinearRegression
from spline_features import add_spline_features
# pinball loss from project utilities
sys.path.append(os.path.join(HERE, '..', 'Python'))
from score import pinball_loss


# Small helpers mirroring notebook behavior
def find_continuous_columns(X: pd.DataFrame):
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


def normalize(X: pd.DataFrame, scale_cols=None):
    X_norm = X.copy()
    if scale_cols is None:
        scale_cols = find_continuous_columns(X_norm)
    mean = X_norm[scale_cols].mean(axis=0)
    std = X_norm[scale_cols].std(axis=0)
    std = std.replace(0, 1.)
    if len(scale_cols) > 0:
        X_norm[scale_cols] = (X_norm[scale_cols] - mean) / std
    return X_norm, mean, std


def main():
    np.random.seed(0)
    data_dir = os.path.join(HERE, '..', 'Data')
    train_path = os.path.join(data_dir, 'train.csv')

    Data_train = pd.read_csv(train_path, parse_dates=["Date"]) 

    # Build base feature set exactly like in the notebook
    X = Data_train.drop(columns=["Net_demand", "Date", "Solar_power", "Wind_power", "Load"])
    y = Data_train["Net_demand"]

    X = pd.get_dummies(X, columns=['WeekDays'], prefix='WeekDays', drop_first=True, dtype=float)

    # Train/validation time-based split (last 20% as validation)
    N = len(X)
    split = int(N * 0.8)
    X_tr_full = X.iloc[:split].copy()
    y_tr_full = y.iloc[:split].copy()
    X_val_full = X.iloc[split:].copy()
    y_val = y.iloc[split:].copy()

    # Model hyperparameters consistent with existing experiments
    tau = 0.8
    learning_rate = 0.02
    maxIter = 17000

    print(f'features before expansion: {X_tr_full.shape[1]}')

    # Note: Baseline normalization and training are performed AFTER spline expansion
    # so the global order becomes: spline expansion -> normalization -> training
    # (this keeps the pipeline consistent and ensures normalization uses expanded feature sets where applicable)

    # We'll compute y arrays now for later use
    y_tr_np = y_tr_full.values
    y_val_np = y_val.values

    # --- Spline expansion on selected columns ---
    spline_cols = ['Temp', 'toy', 'Load.1']

    print('\n--- Applying spline expansion to columns:', spline_cols, '---')
    X_tr_spl, transformers = add_spline_features(X_tr_full, spline_cols, n_knots=5, degree=3, include_bias=False)
    X_val_spl, _ = add_spline_features(X_val_full, spline_cols, transformers=transformers)

    print(f'features after expansion: {X_tr_spl.shape[1]}')

    # Normalize continuous columns AFTER expansion (for spline pipeline)
    scale_cols_spl = find_continuous_columns(X_tr_spl)
    X_tr_spl_norm, X_mean_spl, X_std_spl = normalize(X_tr_spl, scale_cols=scale_cols_spl)
    X_val_spl_norm = X_val_spl.copy()
    if len(scale_cols_spl) > 0:
        X_val_spl_norm[scale_cols_spl] = (X_val_spl_norm[scale_cols_spl] - X_mean_spl) / X_std_spl.replace(0, 1.0)

    X_tr_spl_np = X_tr_spl_norm.values
    X_val_spl_np = X_val_spl_norm.values

    # --- NOW normalize the baseline features (after expansion step) so the sequence is consistent ---
    scale_cols = find_continuous_columns(X_tr_full)
    X_tr_norm, X_mean, X_std = normalize(X_tr_full, scale_cols=scale_cols)
    X_val_norm = X_val_full.copy()
    if len(scale_cols) > 0:
        X_val_norm[scale_cols] = (X_val_norm[scale_cols] - X_mean) / X_std.replace(0, 1.0)

    X_tr_np = X_tr_norm.values
    X_val_np = X_val_norm.values

    # Train baseline AFTER normalization (so the global order is: expansion -> normalization -> training)
    print('\n--- Running baseline Linear pinball (no splines) ---')
    lin = LinearRegression(learning_rate=learning_rate, maxIter=maxIter, tau=tau)
    lin.fit(X_tr_np, y_tr_np, loss='pinball', verbose=False)
    yhat_lin_val = lin.predict(X_val_np)

    pb_lin = pinball_loss(y_val_np, yhat_lin_val, quant=[tau])
    coverage_lin = float(np.mean(y_val_np <= yhat_lin_val))

    # Basic checks for baseline
    mean_y = float(y_tr_np.mean())
    mean_lin = float(yhat_lin_val.mean())
    var_lin = float(np.var(yhat_lin_val))

    # Train pinball linear model on spline-expanded features
    print('\n--- Training Linear pinball on spline-expanded features (GAM-style) ---')
    lin_spl = LinearRegression(learning_rate=learning_rate, maxIter=maxIter, tau=tau)
    lin_spl.fit(X_tr_spl_np, y_tr_np, loss='pinball', verbose=False)
    yhat_spl_val = lin_spl.predict(X_val_spl_np)

    pb_spl = pinball_loss(y_val_np, yhat_spl_val, quant=[tau])
    coverage_spl = float(np.mean(y_val_np <= yhat_spl_val))

    mean_spl = float(yhat_spl_val.mean())
    var_spl = float(np.var(yhat_spl_val))

    # Non-linearity check: compare train predictions
    yhat_lin_train = lin.predict(X_tr_np)
    yhat_spl_train = lin_spl.predict(X_tr_spl_np)
    mad_train = float(np.mean(np.abs(yhat_lin_train - yhat_spl_train)))

    # Weight comparison for extra debug
    w_lin = lin.weights
    b_lin = lin.bias
    w_spl = lin_spl.weights
    b_spl = lin_spl.bias

    # Compare shapes and partial overlaps safely
    if w_lin is None or w_spl is None:
        weight_diff_norm = np.nan
        weight_shape_info = (None, None)
    else:
        weight_shape_info = (w_lin.shape[0], w_spl.shape[0])
        # Compare common prefix
        m = min(w_lin.shape[0], w_spl.shape[0])
        weight_diff_norm = float(np.linalg.norm(w_lin[:m] - w_spl[:m]))

    # Collapse checks
    collapse_lin = var_lin < 1e-6
    collapse_spl = var_spl < 1e-6

    print('\n--- Results ---')
    print('Weight vector shapes (baseline, spline):', weight_shape_info)
    print('Weight vector norm difference on common coordinates (lin vs spline): {:.6f}'.format(weight_diff_norm))
    print('First 10 baseline weights:', (w_lin[:10] if w_lin is not None else 'None'))
    print('First 10 spline-expanded weights:', (w_spl[:10] if w_spl is not None else 'None'))
    print('Baseline bias:', b_lin)
    print('Spline bias:', b_spl)
    print('Linear pinball (no splines):')
    print(f'  pinball_loss={pb_lin:.6f}, coverage={coverage_lin:.3f}, mean_yhat={mean_lin:.3f}, var_yhat={var_lin:.3f}')

    print('\nLinear pinball + spline features (GAM-style):')
    print(f'  pinball_loss={pb_spl:.6f}, coverage={coverage_spl:.3f}, mean_yhat={mean_spl:.3f}, var_yhat={var_spl:.3f}')

    print('\nAdditional checks:')
    print(f'  mean absolute difference (train preds) GAM-style vs linear: {mad_train:.3f}')
    print(f'  predictions collapsed to constant? baseline={collapse_lin}, splines={collapse_spl}')

    # Quick verification of goals
    print('\nVerification checklist:')
    print(f' - Model optimized with pinball loss: Yes (used LinearRegression with loss="pinball")')
    print(f' - Spline-expanded model non-linear behavior (mad_train>{1e-3}): {mad_train > 1e-3}')
    print(f' - Empirical coverage improved or stayed stable: {coverage_spl >= coverage_lin - 1e-6}')
    print(' - Change isolated and documented in this script and in spline_features.py')

    # Suggest next steps if checks fail
    if collapse_spl or collapse_lin:
        print('\nWarning: predictions collapsed to near-constant. Check scaling or learning rate / maxIter for solver convergence.')

    print('\nDone.')

    # --- Optional synthetic sanity check to demonstrate non-linear capacity ---
    print('\n--- Synthetic sanity check: target = non-linear function of Temp ---')
    run_synthetic = True
    if run_synthetic:
        # create synthetic target from Temp (non-linear) on the full original train set
        Temp = X['Temp'].values
        # normalize Temp to 0-1 for stable sin frequency
        Temp_norm = (Temp - Temp.min()) / (Temp.max() - Temp.min())
        y_synth = 10000.0 * np.sin(2 * np.pi * Temp_norm) + 200.0 * np.random.randn(len(Temp_norm))

        # split same way
        y_s_tr = y_synth[:split]
        y_s_val = y_synth[split:]

        # Baseline on original features
        lin_synth = LinearRegression(learning_rate=learning_rate, maxIter=maxIter, tau=tau)
        lin_synth.fit(X_tr_np, y_s_tr, loss='pinball', verbose=False)
        yhat_lin_s_val = lin_synth.predict(X_val_np)
        pb_lin_s = pinball_loss(y_s_val, yhat_lin_s_val, quant=[tau])

        # Spline pipeline for synthetic target
        X_tr_spl_np = X_tr_spl_norm.values
        X_val_spl_np = X_val_spl_norm.values
        lin_spl_synth = LinearRegression(learning_rate=learning_rate, maxIter=maxIter, tau=tau)
        lin_spl_synth.fit(X_tr_spl_np, y_s_tr, loss='pinball', verbose=False)
        yhat_spl_s_val = lin_spl_synth.predict(X_val_spl_np)
        pb_spl_s = pinball_loss(y_s_val, yhat_spl_s_val, quant=[tau])

        print('Synthetic baseline pinball_loss:', pb_lin_s)
        print('Synthetic spline pinball_loss:', pb_spl_s)
        print('Synthetic mean absolute diff (val preds):', float(np.mean(np.abs(yhat_lin_s_val - yhat_spl_s_val))))
        if pb_spl_s < pb_lin_s:
            print('Result: spline-expanded model fits the non-linear synthetic target better (as expected). GOOD')
        else:
            print('Result: spline did NOT improve synthetic non-linear fit — unexpected. PROBLEM')


if __name__ == '__main__':
    main()
