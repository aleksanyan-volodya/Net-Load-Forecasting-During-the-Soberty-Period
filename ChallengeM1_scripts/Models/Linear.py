# Base model : Linear Regression from scratch
import numpy as np

# import the score.py file from ../Python for loss functions
# from ChallengeM1_scripts.Python import score
# TODO: use these loss functions, espacially rmse, mape and piball_loss

class LinearModel:
    """Linear model with Quantile Regression and configurable Regularization"""
    
    def __init__(self, learning_rate=0.01, maxIter=1000, tau=0.8, lambda_reg=0.0, penalty='l2'):
        """
        penalty : 'l2' (Ridge) or 'l1' (Lasso).
                  - 'l2': Adds lambda * ||w||^2. Good for correlated features.
                  - 'l1': Adds lambda * ||w||_1. Good for feature selection (sparsity).
        """
        self.learning_rate = learning_rate
        self.maxIter = maxIter
        self.weights = None
        self.bias = None
        self.errors = []              # history of (unregularized) loss
        self.objective_history = []   # history of objective (loss + reg)
        self.reg_history = []         # history of regularization penalty
        self.tau = tau
        self.lambda_reg = float(lambda_reg)
        self.penalty = penalty.lower() # 'l1' or 'l2'

        # Validation simple
        if self.penalty not in ['l1', 'l2']:
            raise ValueError("Penalty must be 'l1' or 'l2'")
    
    def fit(self, X, y, loss="rmse", verbose=False, log_every=500):
        N, d = X.shape

        self.bias = 0.0
        self.weights = np.zeros(d)
        
        # Sécurité pour pinball
        if loss == "pinball" and not hasattr(self, 'tau'):
             self.tau = 0.5 

        for i in range(self.maxIter):
            y_pred = np.dot(X, self.weights) + self.bias 
            error = y - y_pred

            # --- 1. Calcul du Gradient de base (Data term) ---
            if loss == "rmse":
                # Gradient MSE
                grad_w_data = (2 / N) * (X.T @ error)
                grad_b = (2 / N) * np.sum(error)
                
                # Metric pour l'historique
                current_loss_val = np.sqrt(np.mean(error**2))

            elif loss == "pinball":
                # Gradient Pinball
                # r = y - y_pred. Si r < 0 (donc y < y_pred, surestimation), grad = 1-tau
                # Sinon (sous-estimation), grad = -tau.
                # Note: Votre code original utilisait `r = y - y_pred` pour le calcul
                r = y - y_pred
                grad_factor = (r < 0).astype(float) - self.tau
                
                grad_w_data = (1.0 / N) * (X.T @ grad_factor)
                grad_b = (1.0 / N) * np.sum(grad_factor)

                # Metric pour l'historique
                # max((1-tau)*error_pos, -tau*error_neg) where error = y_pred - y
                current_loss_val = np.mean(np.maximum((1 - self.tau) * error, self.tau * (-error)))

            # --- 2. Ajout de la Régularisation (Gradient & Loss) ---
            grad_reg = np.zeros_like(self.weights)
            reg_penalty_val = 0.0

            if self.lambda_reg > 0:
                if self.penalty == 'l2': # RIDGE
                    # Grad: 2 * lambda * w
                    grad_reg = 2.0 * self.lambda_reg * self.weights
                    # Loss: lambda * sum(w^2)
                    reg_penalty_val = self.lambda_reg * np.sum(self.weights ** 2)
                
                elif self.penalty == 'l1': # LASSO
                    # Grad: lambda * sign(w)
                    grad_reg = self.lambda_reg * np.sign(self.weights)
                    # Loss: lambda * sum(|w|)
                    reg_penalty_val = self.lambda_reg * np.sum(np.abs(self.weights))

            # --- 3. Mise à jour des poids ---
            # Le bias n'est jamais régularisé
            self.weights -= self.learning_rate * (grad_w_data + grad_reg)
            self.bias -= self.learning_rate * grad_b

            # --- 4. Historique ---
            objective = current_loss_val + reg_penalty_val
            
            self.errors.append(current_loss_val)
            self.reg_history.append(reg_penalty_val)
            self.objective_history.append(objective)

            # Logs
            if verbose and (i % log_every == 0 or i == self.maxIter - 1):
                msg = f"[{loss}] iter={i} loss={current_loss_val:.4f} reg({self.penalty})={reg_penalty_val:.4f} obj={objective:.4f}"
                if loss == "pinball":
                    frac_ge = float(np.mean(y_pred >= y))
                    msg += f" frac(y_hat>=y)={frac_ge:.3f}"
                print(msg)

            # Critère d'arrêt
            if objective < 1e-6:
                break
        
        # Diagnostic final pour pinball
        if verbose and loss == "pinball":
            y_pred_final = np.dot(X, self.weights) + self.bias
            coverage = float(np.mean(y <= y_pred_final))
            print(f"[pinball] Final coverage P(y <= y_hat)={coverage:.3f} (target tau={self.tau})")

        return self

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias