"""
Run controlled experiments to measure effect of L2 regularization on quantile (pinball) regression.

- Uses `LinearRegression` from `Models.Linear` with `loss='pinball'` and `tau=0.8`.
- Trains on the first 80% of `Data/train.csv` (time-based split) and validates on the last 20%.
- Reports: training pinball loss, validation pinball loss, empirical coverage P(y <= y_hat),
  and weight L2 norm (||w||).

This script keeps the optimizer unchanged (plain gradient descent).
"""

import numpy as np
import pandas as pd
import os, sys
from Linear import LinearRegression
sys.path.append('../Python')
from score import pinball_loss

def coverage(y, y_hat):
    return float(np.mean(y <= y_hat))

def load_data(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    return df

if __name__ == "__main__":
    df = load_data("../Data/train.csv")

    # Use the observed Net_demand as the target
    target = "Net_demand"

    # Select numeric features (exclude the target)
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if target not in numeric:
        raise RuntimeError(f"target {target} not found among numeric columns")

    features = [c for c in numeric if c != target]

    # time-based split: first 80% train, last 20% validation
    n = len(df)
    split = int(n * 0.8)
    train = df.iloc[:split]
    val = df.iloc[split:]

    X_train = train[features].values
    y_train = train[target].values
    X_val = val[features].values
    y_val = val[target].values

    # NOTE: we avoid centering/standardizing here because, for pinball using the
    # subgradient of the SUM loss, centering features can make the initial
    # gradients (X.T @ grad_factor) almost zero (columns have zero mean), which
    # prevents the model from moving away from zero weights. To keep results
    # comparable to the original (unregularized) implementation we use raw
    # features and tune a small learning rate for numerical stability.

    # Experiment grid for lambda_reg (from no regularization to strong)
    lambdas = [0.0, 1e-8, 1e-6, 5e-5, 1e-5, 1e-4, 5e-4, 5e-3, 1e-3, 1e-2, 1e-1, 1.0, 5.0, 8.0, 9.0, 10.0]
    results = []

    for lam in lambdas:
        print(f"\nTraining with lambda_reg={lam}")
        # Use a small learning rate for the pinball gradient (features are large-scale).
        # Still plain gradient descent; reducing the step size keeps training stable.
        # Increase maxIter to give the optimizer more time to converge with small steps.
        model = LinearRegression(learning_rate=1e-4, maxIter=10000, tau=0.8, lambda_reg=lam)
        model.fit(X_train, y_train, loss="pinball", verbose=False)

        y_hat_train = model.predict(X_train)
        y_hat_val = model.predict(X_val)

        train_loss = pinball_loss(y_train, y_hat_train, tau=0.8)
        val_loss = pinball_loss(y_val, y_hat_val, tau=0.8)
        cov_train = coverage(y_train, y_hat_train)
        cov_val = coverage(y_val, y_hat_val)
        w_norm = float(np.linalg.norm(model.weights))

        results.append({
            "lambda": lam,
            "train_pinball": train_loss,
            "val_pinball": val_loss,
            "cov_train": cov_train,
            "cov_val": cov_val,
            "w_norm": w_norm,
            "weights_head": model.weights[:8].tolist(),
        })

        print(f"lambda={lam} train_pinball={train_loss:.6f} val_pinball={val_loss:.6f} ")
        print(f"coverage(train)={cov_train:.3f} coverage(val)={cov_val:.3f} ||w||={w_norm:.6f}")

    # Sanity checks
    w0 = next(r for r in results if r["lambda"] == 0.0)
    wmax = results[-1]

    if abs(w0["train_pinball"] - results[0]["train_pinball"]) > 1e-9:
        print("Warning: unexpected change in zero-regularization objective")

    print("\nSummary table:")
    print("lambda\ttrain_pinball\tval_pinball\tcov_train\tcov_val\t||w||")
    for r in results:
        print(f"{r['lambda']}\t{r['train_pinball']:.6f}\t{r['val_pinball']:.6f}\t{r['cov_train']:.3f}\t{r['cov_val']:.3f}\t{r['w_norm']:.6f}")

    # Check that stronger regularization shrinks the weight norm (at least compared to lambda=0)
    if results[-1]["w_norm"] >= results[0]["w_norm"] - 1e-8:
        print("\nWarning: the largest lambda did not reduce the weight norm compared to lambda=0")
    else:
        print("\nObserved: increasing lambda reduced the weight norm (coefs were shrunk).")

    # Save results to csv for easy inspection
    pd.DataFrame(results).to_csv("regularization_experiments_summary.csv", index=False)
    print('\nSaved results to regularization_experiments_summary.csv')