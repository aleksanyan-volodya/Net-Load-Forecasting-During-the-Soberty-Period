# Base model : Linear Regression from scratch

import numpy as np

# import the score.py file from ../Python for loss functions
# from ChallengeM1_scripts.Python import score
# TODO: use these loss functions, espacially rmse, mape and piball_loss

class LinearRegression:
    def __init__(self, loss_function = None, learning_rate=0.02, maxIter=1000):
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
        Fitting the model
        """
        X = self.__ones_trick(X)
        N, nb_param = X.shape

        # wieghts init
        # self.bias = 0 # No need with ones trick
        self.weights = np.random.rand(nb_param)

        for i in range(self.maxIter):
            y_pred = np.dot(X, self.weights) # + self.bias 

            grad = -2/N * np.dot(X.T, (y-y_pred)) #MSE
            # TODO: add general function

            self.weights -= self.learning_rate * grad 
            # self.bias -= self.learning_rate * np.mean(-2*(y-y_pred))

        return self
    
    def predict(self, X):
        """
        Predicting
        """
        X = self.__ones_trick(X)
        return np.dot(X, self.weights) # + self.bias
    




