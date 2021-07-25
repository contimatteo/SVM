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
