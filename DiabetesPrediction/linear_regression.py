import numpy as np


class LinearRegression:

    def __init__(self, lr=0.01, num_iter=1000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose

    def __add_intercept(self, X):
        X = np.array(X)
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __cost(self, X, y, theta):
        m = y.size
        h = X.dot(theta)
        J = (1 / (2 * m)) * np.sum(np.square(h - y))
        return J

    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        self.theta = np.zeros(X.shape[1])
        for i in range(self.num_iter):
            h = X.dot(self.theta)
            gradient = X.T.dot(h - y) / y.size
            self.theta -= self.lr * gradient
            if self.verbose and i % 100 == 0:
                print(f"Cost: {self.__cost(X, y, self.theta)}")

    def predict(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        return X.dot(self.theta)
