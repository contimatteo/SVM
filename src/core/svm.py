import cvxopt
import numpy as np

from core.optimizer import Optimizer

###


class SVM():
    def __init__(self):
        self.kernel = None

    #

    def __extract_multipliers(self, optimizer_solution):
        return np.array(optimizer_solution['x']).flatten()

    def __compute_bias(self):
        return []

    #

    def fit(self, X, Y):
        optimizer = Optimizer()

        # QP problem solution
        solution = optimizer.cvxopt_solve(X, Y)

        # lagrangian multipliers
        multipliers = self.__extract_multipliers(solution)

        print()
        print(f"[INFO] multipliers.shape = {multipliers.shape}")
        print()

        # bias
        bias = self.__compute_bias()

    def predict(self, X_test):
        return []


###


class Kernel:
    @staticmethod
    def linear(x, y):
        return np.dot(x, y)
