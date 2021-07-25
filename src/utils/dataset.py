import numpy as np
import random

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles

###


class DatasetUtils:
    @staticmethod
    def shuffle(X, Y):
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)

        return X[indices], Y[indices]

    @staticmethod
    def split(X, Y, train_size=0.8):
        return train_test_split(X, Y, train_size=train_size)


###


class DatasetGenerator:
    @staticmethod
    def linear():
        N_POINTS_TOT = 200
        N_POINTS_FOR_CLASS = int(N_POINTS_TOT / 2)

        # TODO: improve this value
        mean1 = np.array([0, 2])
        # TODO: improve this value
        mean2 = np.array([2, 0])
        # TODO: improve this value
        cov = np.array([[0.8, 0.6], [0.6, 0.8]])

        X_class1 = np.random.multivariate_normal(mean1, cov, N_POINTS_FOR_CLASS)
        X_class2 = np.random.multivariate_normal(mean2, cov, N_POINTS_FOR_CLASS)

        Y_class1 = np.full(N_POINTS_FOR_CLASS, 1)
        Y_class2 = np.full(N_POINTS_FOR_CLASS, -1)

        X = np.concatenate((X_class1, X_class2))
        Y = np.concatenate((Y_class1, Y_class2)).astype(np.double)

        return DatasetUtils.shuffle(X, Y)

    @staticmethod
    def non_linear1():
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[1.5, 1.0], [1.0, 1.5]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        X = np.concatenate((X1, X2))
        Y = np.concatenate((y1, y2)).astype(np.double)
        return X, Y

    @staticmethod
    def non_linear2():
        X, y = make_moons(n_samples=100, noise=0)
        y = np.where(y, 1, -1).astype(np.double)
        return X, y.astype(np.double)

    @staticmethod
    def non_linear3():
        X, y = make_circles(n_samples=100, noise=0)
        y = np.where(y, 1, -1).astype(np.double)
        return X, y.astype(np.double)

    @staticmethod
    def non_linear4():
        mean1 = [-1, 2]
        mean2 = [1, -1]
        mean3 = [4, -4]
        mean4 = [-4, 4]
        cov = [[1.0, 0.8], [0.8, 1.0]]
        X1 = np.random.multivariate_normal(mean1, cov, 50)
        X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 50)
        X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
        y2 = np.ones(len(X2)) * -1
        X = np.concatenate((X1, X2))
        Y = np.concatenate((y1, y2)).astype(np.double)
        return X, Y

    @staticmethod
    def random():
        N_POINTS_TOT = 200

        X = np.random.randn(N_POINTS_TOT, 2)
        Y = np.array([random.choice([-1, 1]) for i in range(N_POINTS_TOT)]).astype(np.double)

        return DatasetUtils.shuffle(X, Y)
