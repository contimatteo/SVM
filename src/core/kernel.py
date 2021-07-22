import numpy as np

###


class Kernel:
    @staticmethod
    def linear(x, y):
        return np.dot(x, y)
