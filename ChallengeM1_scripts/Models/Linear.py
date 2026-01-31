# Base model : Linear Regression from scratch
import numpy as np

# import the score.py file from ../Python for loss functions
# from ChallengeM1_scripts.Python import score
# TODO: use these loss functions, espacially rmse, mape and piball_loss

class LinearRegression:
    """Linear model"""
    def __init__(self, learning_rate=0.01, maxIter=1000, tau=0.8, lambda_reg=0.0):
        """
        Linear regression model with optional L2 regularization for pinball (quantile) loss.

        Objective (for pinball mode):
            objective = pinball_loss + lambda_reg * ||w||^2
        Note: the bias/intercept is NOT regularized.

        lambda_reg : L2 regularization strength (>= 0.0). This penalty shrinks coefficients
                     (reduces variance), improves numerical stability when features are
                     correlated, and helps generalization by discouraging large weights.
                     Set to 0.0 to recover the original unregularized pinball model.
        """
        self.learning_rate = learning_rate
        self.maxIter = maxIter
        self.weights = None
        self.bias = None
        self.errors = []              # history of (unregularized) pinball loss
        self.objective_history = []   # history of (pinball + lambda * ||w||^2) objective
        self.reg_history = []         # history of regularization penalty (lambda * ||w||^2)
        self.tau = tau
        self.lambda_reg = float(lambda_reg)  # regularization strength (>= 0.0)
    
    # def __ones_trick(self, X):
    #     """
    #     Adds a colomn of 1 in X for bias
    #     """
    #     N = X.shape[0]
    #     return np.hstack((np.ones((N, 1)), X))

    # def __gradient(self, X, y, y_pred, N):
    #     """
    #     grad for RMSE
    #     """
    #     return -2/N * np.dot(X.T, (y-y_pred)), -2 * np.mean()

    def fit(self, X, y, loss="rmse", verbose=False, log_every=500):
        """
        Fitting the model returning the errors, the weights, and f
        """
        N, d = X.shape

        self.bias = 0.0
        self.weights = np.zeros(d)
        
        # Sécurité : Si on fait du pinball mais que tau n'est pas défini, on met la médiane par défaut
        if loss == "pinball" and not hasattr(self, 'tau'):
             self.tau = 0.5 

        for i in range(self.maxIter):
            y_pred = np.dot(X, self.weights) + self.bias 
            error = y_pred - y

            if (loss == "rmse"):
                grad_w = (2 / N) * (X.T @ error)
                grad_b = (2 / N) * np.sum(error)

                self.weights -= self.learning_rate * grad_w
                self.bias -= self.learning_rate * grad_b

                # On stocke la RMSE
                l = np.sqrt(np.mean(error**2))
                self.errors.append(l)

            elif (loss == "pinball"):
                # Pinball loss at quantile tau:
                # loss(y, y_hat) = max(tau*(y - y_hat), (tau-1)*(y - y_hat))
                # With residual r = y - y_hat, this is max(tau*r, (tau-1)*r)
                # Subgradient wrt y_hat:
                #   if r > 0  (y_hat < y):  dL/dy_hat = -tau
                #   if r < 0  (y_hat > y):  dL/dy_hat = 1 - tau
                # (at equality, any value in [-tau, 1-tau] is a valid subgradient)
                r = y - y_pred
                # Subgradient of per-sample pinball loss w.r.t. prediction y_hat:
                #   if r > 0 : dL/dy_hat = -tau
                #   if r < 0 : dL/dy_hat = 1 - tau
                # Implemented compactly as (r < 0).astype(float) - tau
                grad_factor = (r < 0).astype(float) - self.tau

                # We use the subgradient of the SUM pinball loss (no division by N)
                # so step magnitudes are comparable to the existing RMSE implementation.
                # For the L2 penalty term (lambda * ||w||^2), the gradient w.r.t. w is 2 * lambda * w.
                # We add this gradient to the pinball-gradient for the full objective:
                #   grad_w = grad_pinball + 2 * lambda_reg * w
                # The bias/intercept is NOT regularized.
                grad_w = (X.T @ grad_factor) + 2.0 * self.lambda_reg * self.weights
                grad_b = np.sum(grad_factor)

                self.weights -= self.learning_rate * grad_w
                self.bias -= self.learning_rate * grad_b

                # 2. Compute the (unregularized) average Pinball Loss for logging
                # Formula : max((1-tau)*error, -tau*error)
                # Recall: error = y_pred - y
                pinball_loss = np.mean(np.maximum((1 - self.tau) * error, self.tau * (-error)))

                # regularization penalty (added to objective; bias is not regularized)
                reg_penalty = self.lambda_reg * np.sum(self.weights ** 2)
                objective = pinball_loss + reg_penalty

                # keep histories: raw pinball loss (for interpretability), regularization, and full objective
                self.errors.append(pinball_loss)
                self.reg_history.append(reg_penalty)
                self.objective_history.append(objective)

                # Use the full objective as stopping criterion
                l = objective

                if verbose and (i % log_every == 0 or i == self.maxIter - 1):
                    frac_ge = float(np.mean(y_pred >= y))
                    print(
                        f"[pinball] iter={i} pinball={pinball_loss:.6f} reg={reg_penalty:.6f} "
                        f"obj={objective:.6f} mean(y_hat)={float(np.mean(y_pred)):.3f} "
                        f"mean(y)={float(np.mean(y)):.3f} frac(y_hat>=y)={frac_ge:.3f}"
                    )

            if l < 1e-6:
                break
        
        # Final diagnostic: empirical coverage on training data
        if verbose and loss == "pinball":
            y_pred_final = np.dot(X, self.weights) + self.bias
            coverage = float(np.mean(y <= y_pred_final))
            print(f"[pinball] final empirical coverage P(y <= y_hat)={coverage:.3f} (target tau={self.tau})")

        return self
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias