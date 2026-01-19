# Base model : Linear Regression from scratch

import numpy as np

# import the score.py file from ../Python for loss functions
# from ChallengeM1_scripts.Python import score
# TODO: use these loss functions, espacially rmse, mape and piball_loss

class LinearRegression:
    """Linear model"""
    def __init__(self, loss_function = None, learning_rate=0.01, maxIter=1000):
        self.loss_function = loss_function
        # use score.py for loss functions
        self.learning_rate = learning_rate
        self.maxIter = maxIter
        self.weights = None
        self.bias = None 
    
    def __ones_trick(self, X):
        """
        Adds a colomn of 1 in X for bias
        """
        N = X.shape[0]
        return np.hstack((np.ones((N, 1)), X))

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
        N, nb_param = X.shape
        Err = []
        F = []
        # wieghts init
        self.bias = 0 # No need with ones trick
        self.weights = np.zeros(nb_param)

        for i in range(self.maxIter):
            Y_pred = np.dot(X, self.weights) + self.bias 
            error = y - Y_pred
            # rmse
            mse = np.mean(error**2)
            rmse = np.sqrt(mse)
            
            # Adding a small epsilon (1e-8) to avoid division by zero
            grad = -(1 / (N * (rmse + 1e-8))) * (X.T @ error)

            self.weights -= self.learning_rate * grad
            self.bias -= self.learning_rate * (-2 * np.mean(error) / (rmse + 1e-8))

            Err.append(rmse)
            F.append(np.dot(X, self.weights) + self.bias)

        return Err, F, self.weights
    
    def predict(self, X):
        """
        Predicting
        """
        return np.dot(X, self.weights) + self.bias
    




