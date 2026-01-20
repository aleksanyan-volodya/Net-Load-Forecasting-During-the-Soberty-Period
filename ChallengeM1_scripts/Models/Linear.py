# Base model : Linear Regression from scratch

import numpy as np

# import the score.py file from ../Python for loss functions
# from ChallengeM1_scripts.Python import score
# TODO: use these loss functions, espacially rmse, mape and piball_loss

class LinearRegression:
    """Linear model"""
    def __init__(self, learning_rate=0.01, maxIter=1000):
        self.learning_rate = learning_rate
        self.maxIter = maxIter
        self.weights = None
        self.bias = None 
        self.errors = []
    
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

    def fit(self, X, y):
        """
        Fitting the model returning the errorn, the weights, and f
        """
        # X = self.__ones_trick(X)
        N, d = X.shape

        self.bias = 0.0
        self.weights = np.zeros(d)

        for i in range(self.maxIter):
            y_pred = np.dot(X, self.weights) + self.bias 
            error = y_pred - y

            grad_w = (2 / N) * (X.T @ error)
            grad_b = (2 / N) * np.sum(error)

            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

            rmse = np.sqrt(np.mean(error**2))
            self.errors.append(rmse)
            
            if rmse < 1e-6:
                break

        return self
    
    def predict(self, X):
        """
        Predicting
        """
        return np.dot(X, self.weights) + self.bias
    