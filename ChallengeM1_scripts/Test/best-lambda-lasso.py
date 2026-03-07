"""
Lasso Regularization Parameter Selection via Time Series Cross-Validation

This script performs hyperparameter tuning for Lasso (L1) regularization in a linear
pinball regression model. It uses time series cross-validation to find the optimal
lambda_reg value that minimizes pinball loss at quantile 0.8.

Key steps:
1. Grid search over lambda values (logarithmic scale)
2. Time series cross-validation with temporal splits
3. Feature standardization (fitted only on training folds)
4. Pinball loss evaluation at quantile 0.8
5. Final model training with optimal lambda

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from Linear import LinearModel

# Grid search setup
lambda_values = [0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]  # Logarithmic scale
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)
mean_scores = []

print(f"Starting time series cross-validation with {n_splits} splits...")
print("-" * 60)

# Cross-validation loop
for lam in lambda_values:
    fold_scores = []

    for train_index, val_index in tscv.split(X):
        # Temporal split
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]

        # Standardization (fit only on training fold to avoid leakage)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)

        # Train Lasso model
        model = LinearModel(
            learning_rate=0.01,
            maxIter=2000,
            tau=0.8,           # Target quantile 0.8
            lambda_reg=lam,    # Lambda being tested
            penalty='l1'       # Lasso
        )

        model.fit(X_train_scaled, y_train_fold, loss="pinball", verbose=False)

        # Validation prediction and pinball loss calculation
        y_pred_val = model.predict(X_val_scaled)
        diff = y_val_fold - y_pred_val
        score_fold = np.mean(np.maximum(0.8 * diff, (0.8 - 1.0) * diff))

        fold_scores.append(score_fold)

    # Average score for this lambda
    avg_score = np.mean(fold_scores)
    mean_scores.append(avg_score)
    print(f"Lambda = {lam:7.4f} | Mean Pinball Loss = {avg_score:.4f}")

# Select best lambda
best_idx = np.argmin(mean_scores)
best_lambda = lambda_values[best_idx]
best_score = mean_scores[best_idx]

print("-" * 60)
print(f"BEST RESULT: Lambda = {best_lambda} (Loss = {best_score:.4f})")

# Plot lambda vs validation loss
plt.figure(figsize=(10, 6))
plt.errorbar(np.log10(lambdas), mean_scores, yerr=std_scores, fmt='-o', capsize=5)
plt.title('Cross-Validation: Lasso Regularization Impact')
plt.xlabel('log10(lambda_reg)')
plt.ylabel('Pinball Loss (Validation)')
plt.axvline(np.log10(best_lambda), color='r', linestyle='--', label=f'Best lambda: {best_lambda:.4f}')
plt.legend()
plt.grid(True)
plt.show()

# Final model training with optimal lambda
print("\nTraining final model on full dataset with best lambda...")

scaler_final = StandardScaler()
X_final_scaled = scaler_final.fit_transform(X)

final_model = LinearModel(
    learning_rate=0.01,
    maxIter=5000,
    tau=0.8,
    lambda_reg=best_lambda,
    penalty='l1'
)

final_model.fit(X_final_scaled, y, loss="pinball", verbose=True)

print("Model ready for predictions on submission file!")