import math
import numpy as np

from numpy import linalg

###


class Kernel:
    @staticmethod
    def linear(x, y):
        return np.dot(x, y)

    @staticmethod
    def polynomial(x, y, degree):
        return (Kernel.linear(x, y) + 1)**degree

    # TODO: remove this ...
    # @staticmethod
    # def rbf(x, y, gamma):
    #     return np.exp(-gamma * linalg.norm((x-y), 2))

    # TODO: remove this ...
    # @staticmethod
    # def sigmoid(x, y, b):
    #     return np.tanh(Kernel.linear(x, y) + b)

    # TODO: remove this ...
    # def gaussian(x, y, sigma=5.0):
    #     return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))
