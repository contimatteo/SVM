import numpy as np

from core.kernel import Kernel
from core.optimizer import Optimizer

###


class SVM():
    def __init__(self, kernel='linear', C=None, deg=None):
        self._kernel = None
        self._kernel_type = kernel
        self._kernel_function = None

        self._multipliers = None
        self._lambdas = None
        self._bias = None

        self._sv = None
        self._sv_Y = None
        self._sv_idxs = None

        self.C = C
        self.deg = deg if deg is not None else 6

        if self._kernel_type == 'linear':
            self._kernel_function = lambda x1, x2: Kernel.linear(x1, x2)
        elif self._kernel_type == 'poly':
            self._kernel_function = lambda x1, x2: Kernel.polynomial(x1, x2, degree=self.deg)
        else:
            raise Exception(f"SVM: invalid 'kernel={self._kernel_type}' parameter value.")

    #

    @property
    def support_vectors(self):
        return self._sv

    @property
    def support_vectors_Y(self):
        return self._sv_Y

    @property
    def bias(self):
        return self._bias

    #

    def fit(self, X, Y):
        ###  compute the kernel
        self._kernel = SVMCore.apply_kernel(self._kernel_function, X)

        optimizer = Optimizer()
        optimizer.initialize()

        ###  QP problem solution
        solution = None
        if self.C:
            solution = optimizer.cvxopt_soft_margin_solve(Y, self._kernel, self.C)
        else:
            solution = optimizer.cvxopt_hard_margin_solve(Y, self._kernel)

        ###  lagrangian multipliers
        self._multipliers = SVMCore.multipliers(solution)

        self._sv_idxs = SVMCore.support_vectors_indexes(self._multipliers)

        ###  lambda params (filtered multipliers)
        self._lambdas = self._multipliers[self._sv_idxs]

        ###  support vectors
        self._sv = X[self._sv_idxs]
        self._sv_Y = Y[self._sv_idxs]

        ###  bias
        self._bias = SVMCore.bias(self._lambdas, self._kernel, self._sv_Y, self._sv_idxs)

    def project(self, points):
        return SVMCore.hyperplane_projection(
            self._kernel_type, self._kernel_function, self._lambdas, self._sv, self._sv_Y,
            self._bias
        )(points)

    def predict(self, X_test):
        projections = self.project(X_test)

        return np.sign(projections)


###


class SVMCore():
    @staticmethod
    def apply_kernel(kernel_function, points):
        return kernel_function(points, points.T)

    @staticmethod
    def multipliers(optimizer_solution):
        """
        The solver returns the list of optimum variables values. \\
        In our case, variables are the lagrangian multipliers.
        """
        return np.array(optimizer_solution['x']).flatten()

    @staticmethod
    def support_vectors_indexes(multipliers):
        """
        In the solution, all points `xi` having the corresponding multiplier 
        `λi` strictly positive are named support vectors. All other points 
        `xi` have the corresponding `λi = 0` have no effect on the classifier.
        """
        zero_threshold = 1e-5
        bool_idxs = multipliers > zero_threshold
        return np.arange(multipliers.shape[0])[bool_idxs]

    @staticmethod
    def bias(lambdas, kernel, Y, sv_idxs):
        """
        TODO: missing explaination of the following computations
        """
        bias = 0
        for n in range(lambdas.shape[0]):
            bias += Y[n] - np.sum(lambdas * Y * kernel[sv_idxs[n], sv_idxs])
        bias /= lambdas.shape[0]

        return bias

    @staticmethod
    def __hyperplane_linear_coefficients(lambdas, sv, sv_Y):
        """
        TODO: missing formulation ...

        given the hyperplane equation \\
        `f(x) = (w * x) + b`

        and given the original Lagrangian formulation of our problem

        we obtain the following partial derivate of `L(w,b,λ)` (respect to `w`) \\
        `𝟃L/𝟃w = w - (λ * Y * X)`

        and then by applying the KKT (1) condition (used to have
        guarantees on the optimality of the result) we get \\
        `𝟃L/𝟃w = 0` \\
        `w - (λ * Y * X) = 0` \\
        `w = λ * Y * X`
        """
        X = sv
        Y = sv_Y
        coefficients_to_sum = np.array(lambdas * Y * X.T)
        return np.sum(coefficients_to_sum, axis=1)

    @staticmethod
    def hyperplane_projection(kernel_type, kernel_function, lambdas, sv, sv_Y, bias):
        """
        TODO: missing formulation ...
        """
        def linear_projection(points):
            coefficients = SVMCore.__hyperplane_linear_coefficients(lambdas, sv, sv_Y)
            return np.dot(points, coefficients) + bias

        def non_linear_projection(points):
            projections = np.zeros(points.shape[0])
            for (idx1, point) in enumerate(points):
                sign = 0
                for idx2 in range(sv.shape[0]):
                    sign += lambdas[idx2] * sv_Y[idx2] * kernel_function(point, sv[idx2])
                projections[idx1] = sign + bias
            return projections

        if kernel_type == 'linear':
            return lambda points: linear_projection(points)
        else:
            return lambda points: non_linear_projection(points)

    @staticmethod
    def hyperplane_equation(coefficients, bias):
        """
        given the hyperplane equation (where the default value of `c` is 0) \\
        `w1 * x1 + w2 * x2 + b = c`

        we obtain 
        `x2 = (-w1 * x1 - b + c) / w2`
        """
        def equation(w, b, x1, c):
            return (-w[0] * x1 - b + c) / w[1]

        return lambda x1, c: equation(coefficients, bias, x1, c)
