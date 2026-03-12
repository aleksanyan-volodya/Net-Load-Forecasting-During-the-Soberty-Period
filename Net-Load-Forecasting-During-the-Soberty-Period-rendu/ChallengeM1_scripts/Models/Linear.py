"""Linear regression with RMSE/pinball loss and L1/L2 regularization.

Simple implementation trained with gradient descent.

Inspiration
-----------
- Project baseline model for regression tasks.
- Support for quantile (pinball) loss for expectile-like learning.
"""

import numpy as np

class LinearModel:
    """Linear model with Quantile Regression and L1/L2 Regularization."""
    
    def __init__(self, learning_rate=0.01, maxIter=1000, tau=0.8, lambda_reg=0.0, penalty='l2'):
        """Initialize linear model.

        Parameters
        ----------
        learning_rate : float, default=0.01
            Step size for gradient descent.
        maxIter : int, default=1000
            Maximum number of iterations.
        tau : float, default=0.8
            Quantile level for pinball loss. Ignored if loss='rmse'.
        lambda_reg : float, default=0.0
            Regularization strength. Higher = more shrinkage.
        penalty : {'l2', 'l1'}, default='l2'
            Penalty type. 'l2' for Ridge, 'l1' for Lasso.
        """
        self.learning_rate = learning_rate
        self.maxIter = maxIter
        self.weights = None
        self.bias = None
        self.errors = []
        self.objective_history = []
        self.reg_history = []
        self.tau = tau
        self.lambda_reg = float(lambda_reg)
        self.penalty = penalty.lower()

        if self.penalty not in ['l1', 'l2']:
            raise ValueError("Penalty must be 'l1' or 'l2'")
    
    def fit(self, X, y, loss="rmse", verbose=False, log_every=500):
        """Fit the model using gradient descent.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Training feature matrix.
        y : array_like, shape (n_samples,)
            Training target vector.
        loss : {'rmse', 'pinball'}, default='rmse'
            Loss function to minimize.
        verbose : bool, default=False
            Print training progress.
        log_every : int, default=500
            Print interval during training.

        Returns
        -------
        self
        """
        N, d = X.shape

        self.bias = 0.0
        self.weights = np.zeros(d)
        
        if loss == "pinball" and not hasattr(self, 'tau'):
             self.tau = 0.5 

        for i in range(self.maxIter):
            y_pred = np.dot(X, self.weights) + self.bias 
            error = y_pred - y

            # Compute gradient based on loss type.
            if loss == "rmse":
                grad_w_data = (2 / N) * (X.T @ error)
                grad_b = (2 / N) * np.sum(error)
                current_loss_val = np.sqrt(np.mean(error**2))

            elif loss == "pinball":
                # Quantile loss: gradient depends on sign of residual.
                r = y - y_pred
                grad_factor = (r < 0).astype(float) - self.tau
                grad_w_data = (1.0 / N) * (X.T @ grad_factor)
                grad_b = (1.0 / N) * np.sum(grad_factor)
                current_loss_val = np.mean(np.maximum((1 - self.tau) * error, self.tau * (-error)))
                

            grad_reg = np.zeros_like(self.weights)
            reg_penalty_val = 0.0

            # Add regularization penalty and gradient.
            if self.lambda_reg > 0:
                if self.penalty == 'l2':
                    grad_reg = 2.0 * self.lambda_reg * self.weights
                    reg_penalty_val = self.lambda_reg * np.sum(self.weights ** 2)
                elif self.penalty == 'l1':
                    grad_reg = self.lambda_reg * np.sign(self.weights)
                    reg_penalty_val = self.lambda_reg * np.sum(np.abs(self.weights))

            # Update weights and bias (bias is never regularized).
            self.weights -= self.learning_rate * (grad_w_data + grad_reg)
            self.bias -= self.learning_rate * grad_b

            # Track history.
            objective = current_loss_val + reg_penalty_val
            
            self.errors.append(current_loss_val)
            self.reg_history.append(reg_penalty_val)
            self.objective_history.append(objective)

            if verbose and (i % log_every == 0 or i == self.maxIter - 1):
                msg = f"[{loss}] iter={i} loss={current_loss_val:.4f} reg({self.penalty})={reg_penalty_val:.4f} obj={objective:.4f}"
                if loss == "pinball":
                    frac_ge = float(np.mean(y_pred >= y))
                    msg += f" frac(y_hat>=y)={frac_ge:.3f}"
                print(msg)

            if objective < 1e-6:
                break
        
        if verbose and loss == "pinball":
            y_pred_final = np.dot(X, self.weights) + self.bias
            coverage = float(np.mean(y <= y_pred_final))
            print(f"[pinball] Final coverage P(y <= y_hat)={coverage:.3f} (target tau={self.tau})")

        return self

    def predict(self, X):
        """Make predictions on new data.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        array_like
            Predicted values.
        """
        return np.dot(X, self.weights) + self.bias


# Alias for backward compatibility.
LinearRegression = LinearModel