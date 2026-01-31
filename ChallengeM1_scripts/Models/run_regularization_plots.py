"""
Sweep learning rates and lambda_reg values, collect metrics, and produce plots
showing the effect of L2 regularization and learning rate on pinball regression.

Saves:
 - regularization_sweep_results.csv
 - plots: losses_vs_lambda_{lr}.png, norm_vs_lambda.png, coverage_vs_lambda.png,
   coeffs_vs_lambda.png
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append(os.path.dirname(__file__))
from Linear import LinearRegression

sns.set(style="whitegrid")


def pinball_loss(y, y_hat, tau=0.8):
    r = y - y_hat
    return np.mean(np.maximum(tau * r, (tau - 1) * r))


def coverage(y, y_hat):
    return float(np.mean(y <= y_hat))


def run_sweep(train_csv='../Data/train.csv'):
    df = pd.read_csv(train_csv, parse_dates=['Date'])
    target = 'Net_demand'
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric if c != target]
    n = len(df)
    split = int(n * 0.8)
    train = df.iloc[:split]
    val = df.iloc[split:]
    X_train = train[features].values
    y_train = train[target].values
    X_val = val[features].values
    y_val = val[target].values

    # Use small learning rates that are stable for the (mean-scaled) pinball gradient
    learning_rates = [1e-8, 1e-9, 1e-10]
    lambdas = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

    rows = []
    weight_matrix = {}

    for lr in learning_rates:
        for lam in lambdas:
            print(f"Training lr={lr} lambda={lam}")
            model = LinearRegression(learning_rate=lr, maxIter=3000, tau=0.8, lambda_reg=lam)
            model.fit(X_train, y_train, loss='pinball', verbose=False)
            y_hat_train = model.predict(X_train)
            y_hat_val = model.predict(X_val)
            train_loss = pinball_loss(y_train, y_hat_train, tau=0.8)
            val_loss = pinball_loss(y_val, y_hat_val, tau=0.8)
            cov_tr = coverage(y_train, y_hat_train)
            cov_val = coverage(y_val, y_hat_val)
            w_norm = float(np.linalg.norm(model.weights))

            # detect divergence / numerical explosion and mark status
            status = 'ok'
            if (not np.isfinite(train_loss)) or (not np.isfinite(val_loss)) or train_loss > 1e8 or val_loss > 1e8:
                status = 'diverged'
                print(f" -> Diverged for lr={lr}, lambda={lam}: train={train_loss:.3e}, val={val_loss:.3e}")

            rows.append({
                'lr': lr,
                'lambda': lam,
                'train_pinball': train_loss,
                'val_pinball': val_loss,
                'cov_train': cov_tr,
                'cov_val': cov_val,
                'w_norm': w_norm,
                'status': status
            })
            # store weights for later coefficient plot
            weight_matrix[(lr, lam)] = model.weights.copy()

    results = pd.DataFrame(rows)
    results.to_csv('regularization_sweep_results.csv', index=False)
    print('Saved regularization_sweep_results.csv')

    # Plotting
    os.makedirs('figures', exist_ok=True)

    # 1) For each lr, plot train & val pinball vs lambda (skip diverged runs)
    for lr in learning_rates:
        sub = results[(results['lr'] == lr) & (results['status'] == 'ok')].sort_values('lambda')
        if sub.empty:
            print(f"No successful runs to plot for lr={lr} (all diverged)")
            continue
        plt.figure(figsize=(8,5))
        plt.semilogx(sub['lambda'].replace(0, 1e-12), sub['train_pinball'], marker='o', label='train')
        plt.semilogx(sub['lambda'].replace(0, 1e-12), sub['val_pinball'], marker='o', label='val')
        plt.xlabel('lambda (log scale)')
        plt.ylabel('Pinball loss')
        plt.title(f'Pinball loss vs lambda (lr={lr})')
        plt.legend()
        plt.tight_layout()
        fn = f'figures/losses_vs_lambda_lr_{lr}.png'
        plt.savefig(fn)
        plt.close()
        print('Saved', fn)

    # 2) Weight norm vs lambda (one line per lr)
    plt.figure(figsize=(8,5))
    for lr in learning_rates:
        sub = results[results['lr'] == lr].sort_values('lambda')
        plt.semilogx(sub['lambda'].replace(0, 1e-12), sub['w_norm'], marker='o', label=f'lr={lr}')
    plt.xlabel('lambda (log scale)')
    plt.ylabel('||w||')
    plt.title('Weight norm vs lambda')
    plt.legend()
    plt.tight_layout()
    fn = 'figures/w_norm_vs_lambda.png'
    plt.savefig(fn)
    plt.close()
    print('Saved', fn)

    # 3) Coverage vs lambda (validation)
    plt.figure(figsize=(8,5))
    for lr in learning_rates:
        sub = results[results['lr'] == lr].sort_values('lambda')
        plt.semilogx(sub['lambda'].replace(0, 1e-12), sub['cov_val'], marker='o', label=f'lr={lr}')
    plt.xlabel('lambda (log scale)')
    plt.ylabel('Empirical coverage P(y <= y_hat)')
    plt.title('Validation coverage vs lambda')
    plt.legend()
    plt.tight_layout()
    fn = 'figures/coverage_vs_lambda.png'
    plt.savefig(fn)
    plt.close()
    print('Saved', fn)

    # 4) Coefficient paths vs lambda for top-k coefficients (by abs at lambda=0, lr=middle)
    lr_mid = learning_rates[1]
    w0 = weight_matrix[(lr_mid, 0.0)]
    topk = np.argsort(np.abs(w0))[-6:][::-1]
    plt.figure(figsize=(10,6))
    for idx in topk:
        vals = [weight_matrix[(lr_mid, lam)][idx] for lam in lambdas]
        plt.semilogx([l if l>0 else 1e-12 for l in lambdas], vals, marker='o', label=f'feat_{features[idx]} (idx={idx})')
    plt.xlabel('lambda (log scale)')
    plt.ylabel('coefficient value')
    plt.title(f'Coefficient paths vs lambda (lr={lr_mid})')
    plt.legend(fontsize='small')
    plt.tight_layout()
    fn = 'figures/coeffs_vs_lambda.png'
    plt.savefig(fn)
    plt.close()
    print('Saved', fn)

    print('\nDone. Figures saved in ./figures and results in regularization_sweep_results.csv')


if __name__ == '__main__':
    run_sweep()
