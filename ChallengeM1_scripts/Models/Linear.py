# Base model : Linear Regression from scratch
import numpy as np

# import the score.py file from ../Python for loss functions
# from ChallengeM1_scripts.Python import score
# TODO: use these loss functions, espacially rmse, mape and piball_loss

class LinearRegression:
    """Linear model"""
    def __init__(self, learning_rate=0.01, maxIter=1000, tau = 0.8):
        self.learning_rate = learning_rate
        self.maxIter = maxIter
        self.weights = None
        self.bias = None 
        self.errors = []
        self.tau = tau
    
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
                # With error = y_hat - y, this is max((1-tau)*error, tau*(-error))
                # Subgradient wrt y_hat:
                #   if y_hat > y: (1 - tau)
                #   else:         (-tau)
                # (at equality, any value in [-tau, 1-tau] is a valid subgradient)
                indicator = (y_pred > y).astype(float) 
                
                # Le terme de gradient est (Indicator - tau)
                # Si surestimation : (1 - tau)
                # Si sous-estimation : (0 - tau) = -tau
                grad_factor = indicator - self.tau
                
                grad_w = (1 / N) * (X.T @ grad_factor)
                grad_b = (1 / N) * np.sum(grad_factor)

                self.weights -= self.learning_rate * grad_w
                self.bias -= self.learning_rate * grad_b

                # 2. Calcul de la Pinball Loss moyenne pour l'historique
                # Formule : max((1-tau)*error, -tau*error)
                # Rappel: error = y_pred - y
                pinball_loss = np.mean(np.maximum((1 - self.tau) * error, self.tau * (-error)))
                self.errors.append(pinball_loss)
                
                l = pinball_loss
                
                if verbose and (i % log_every == 0 or i == self.maxIter - 1):
                    frac_ge = float(np.mean(y_pred >= y))
                    print(
                        f"[pinball] iter={i} loss={pinball_loss:.6f} "
                        f"mean(y_hat)={float(np.mean(y_pred)):.3f} "
                        f"mean(y)={float(np.mean(y)):.3f} "
                        f"frac(y_hat>=y)={frac_ge:.3f}"
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