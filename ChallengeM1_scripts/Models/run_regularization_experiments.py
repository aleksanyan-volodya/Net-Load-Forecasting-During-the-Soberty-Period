"""Run L2 regularization sweep experiments on pinball regression.

Train models with different regularization strengths and measure loss,
coverage, and weight norm on a time-based train/validation split.

Notes
-----
- Validation split is time-based (last 20% of rows).
- Results saved to regularization_experiments_summary.csv.

Inspiration
-----------
- Existing project linear model and score utility.
"""

import numpy as np
import pandas as pd
import os, sys
from Linear import LinearRegression
sys.path.append('../Python')
from score import pinball_loss

def coverage(y, y_hat):
    """Return fraction of predictions >= actual values.

    Parameters
    ----------
    y : array_like
        Actual target values.
    y_hat : array_like
        Predicted values.

    Returns
    -------
    float
        P(y <= y_hat)
    """
    return float(np.mean(y <= y_hat))

def load_data(path):
    """Load CSV file with date parsing.

    Parameters
    ----------
    path : str
        Path to CSV file.

    Returns
    -------
    pandas.DataFrame
        Loaded data.
    """
    df = pd.read_csv(path, parse_dates=["Date"])
    return df

if __name__ == "__main__":
    df = load_data("../Data/train.csv")

    target = "Net_demand"

    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if target not in numeric:
        raise RuntimeError(f"target {target} not found among numeric columns")

    features = [c for c in numeric if c != target]

    # Time split: first 80% train, last 20% validation.
    n = len(df)
    split = int(n * 0.8)
    train = df.iloc[:split]
    val = df.iloc[split:]

    X_train = train[features].values
    y_train = train[target].values
    X_val = val[features].values
    y_val = val[target].values

    # Features are not centered to avoid zero gradients at start.
    lambdas = [0.0, 1e-8, 1e-6, 5e-5, 1e-5, 1e-4, 5e-4, 5e-3, 1e-3, 1e-2, 1e-1, 1.0, 5.0, 8.0, 9.0, 10.0]
    results = []

    for lam in lambdas:
        print(f"\nTraining with lambda_reg={lam}")
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

    w0 = next(r for r in results if r["lambda"] == 0.0)
    wmax = results[-1]

    if abs(w0["train_pinball"] - results[0]["train_pinball"]) > 1e-9:
        print("Warning: unexpected change in zero-regularization objective")

    print("\nSummary table:")
    print("lambda\ttrain_pinball\tval_pinball\tcov_train\tcov_val\t||w||")
    for r in results:
        print(f"{r['lambda']}\t{r['train_pinball']:.6f}\t{r['val_pinball']:.6f}\t{r['cov_train']:.3f}\t{r['cov_val']:.3f}\t{r['w_norm']:.6f}")

    if results[-1]["w_norm"] >= results[0]["w_norm"] - 1e-8:
        print("\nWarning: the largest lambda did not reduce the weight norm compared to lambda=0")
    else:
        print("\nObserved: increasing lambda reduced the weight norm (coefs were shrunk).")

    pd.DataFrame(results).to_csv("regularization_experiments_summary.csv", index=False)
    print('\nSaved results to regularization_experiments_summary.csv')