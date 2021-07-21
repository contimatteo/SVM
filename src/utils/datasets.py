import numpy as np

###


class Dataset:
    @staticmethod
    def linear():
        N_POINTS_TOT = 200
        N_POINTS_FOR_CLASS = int(N_POINTS_TOT / 2)

        # TODO: change this value
        mean1 = np.array([0, 2])
        # TODO: change this value
        mean2 = np.array([2, 0])
        # TODO: change this value
        cov = np.array([[0.8, 0.6], [0.6, 0.8]])

        X_class1 = np.random.multivariate_normal(mean1, cov, N_POINTS_FOR_CLASS)
        X_class2 = np.random.multivariate_normal(mean2, cov, N_POINTS_FOR_CLASS)

        Y_class1 = np.full(N_POINTS_FOR_CLASS, 1)
        Y_class2 = np.full(N_POINTS_FOR_CLASS, -1)

        # X = np.concatenate((X_class1, X_class2))
        # Y = np.concatenate((Y_class1, Y_class2))

        return X_class1, Y_class1, X_class2, Y_class2
