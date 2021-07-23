import math
import numpy as np

from numpy import linalg

###


class Kernel:
    @staticmethod
    def linear(M1, M2):
        return np.dot(M1, M2)

    @staticmethod
    def polynomial(M1, M2, exponent, alpha):
        return (Kernel.linear(M1, M2) + alpha)**exponent

    @staticmethod
    def sigmoid(M1, M2, b):
        return np.tanh(Kernel.linear(M1, M2) - b)
