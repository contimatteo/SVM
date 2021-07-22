import cvxopt
import numpy as np

from core.kernel import Kernel
from core.optimizer import Optimizer

###


class SVM():
    def __init__(self):
        self._kernel = None
        self._kernel_function = Kernel.linear

        self._lambdas = None
        self._bias = None
        self._w = None

        self._sv = None
        self._sv_kernel = None
        self._sv_Y = None

    #

    @property
    def support_vectors(self):
        return self._sv

    @property
    def weights(self):
        return self._w

    @property
    def bias(self):
        return self._bias

    #

    def project_to_hyperplane(self, points):
        return SVMCore.hyperplane_projection_function(self._w, self._bias)(points)

    def hyperplane_equation(self, x1, c=0):
        return SVMCore.hyperplane_equation(self._w, self._bias)(x1, c)

    #

    def fit(self, X, Y):
        ###  compute the kernel
        self._kernel = SVMCore.compute_kernel(self._kernel_function, X)

        optimizer = Optimizer()
        optimizer.initialize()

        ###  QP problem solution
        solution = optimizer.cvxopt_solve(X, Y, self._kernel)

        ###  lagrangian multipliers
        multipliers = SVMCore.extract_multipliers(solution)

        sv_idxs = SVMCore.extract_support_vectors_indexes(multipliers)

        ###  lambda params (filtered multipliers)
        self._lambdas = multipliers[sv_idxs]

        ###  support vectors
        self._sv = X[sv_idxs]
        self._sv_Y = Y[sv_idxs]
        self._sv_kernel = SVMCore.compute_kernel(self._kernel_function, self._sv)

        ###  bias
        self._bias = SVMCore.compute_bias(self._lambdas, self._sv_kernel, self._sv_Y)

        ### w (hyperplane equation coefficients)
        self._w = SVMCore.compute_hyperplane_coefficients(self._lambdas, self._sv, self._sv_Y)

    def predict(self, X_test):
        projections = self.project_to_hyperplane(X_test)
        return np.sign(projections)


###


class SVMCore():
    @staticmethod
    def compute_kernel(kernel_function, points):
        return kernel_function(points, points.T)

    @staticmethod
    def extract_multipliers(optimizer_solution):
        """
        The solver returns the list of optimum variables values. \\
        In our case, variables are the lagrangian multipliers.
        """
        return np.array(optimizer_solution['x']).flatten()

    @staticmethod
    def extract_support_vectors_indexes(multipliers):
        """
        In the solution, all points `xi` having the corresponding multiplier 
        `λi` strictly positive are named support vectors. All other points 
        `xi` have the corresponding `λi = 0` have no effect on the classifier.
        """
        zero_threshold = 1e-5
        return multipliers > zero_threshold

    @staticmethod
    def compute_bias(lambdas, kernel, Y):
        """
        given the equation of the hyperplane \\
        `f(x) = (W * X) + b`

        we know from the partial derivative of the primal Lagrangian 
        formulation and by the application of the KKT (1) condition \\
        `w = λ * Y * X`

        as a conseguence, the equation of the hyperplane becomes  \\
        `f(x) = ((λ * Y * X) * X) + b`

        
        given that `X * X` is replaced by the Kernel 
        function and given that `f(x) = 1/|λ|`
        
        we obtain the following \\
        `f(x) = (λ * Y * Kernel) + b`

        and now we can extract the bias \\
        `b = f(x) - ((λ * Y * X) * X)`

        // TODO: missing explaination of the following computations \\
        // ...
        """

        ### TODO: why {b} parameter is formulated as follow?
        # ...

        ### TODO: remote the following simplification
        ### from
        ### ->  b = 1/|λ| * ∑ (Y - ∑ (λ * Y * Kernel))
        ### to
        ### -> c1 = 1 / |λ|
        ###    c2 = ∑ (λ * Y * Kernel)
        ###    c3 = ∑ (Y - c2)
        ###     b = c1 * c3

        c1 = 1 / lambdas.shape[0]
        c2 = np.array(lambdas * Y * kernel).sum()
        c3 = np.array(Y - c2).sum()

        return c1 * c3

    @staticmethod
    def compute_hyperplane_coefficients(lambdas, X, Y):
        """
        given the hyperplane equation \\
        `f(x) = (w * x) + b`

        and given the original Lagrangian formulation of our problem \\
        `TODO: missing formulation ...` \\
        `......`

        we obtain the following partial derivate of `L(w,b,λ)` (respect to `w`) \\
        `𝟃L/𝟃w = w - (λ * Y * X)`

        and then by applying the KKT (1) condition (used to have
        guarantees on the optimality of the result) we get \\
        `𝟃L/𝟃w = 0` \\
        `w - (λ * Y * X) = 0` \\
        `w = λ * Y * X`
        """
        coefficients_to_sum = np.array(lambdas * Y * X.T)
        return np.sum(coefficients_to_sum, axis=1)

    @staticmethod
    def hyperplane_projection_function(coefficients, bias):
        """
        given the hyperplane equation \\
        `f(x) = (w * X) + b`
        """
        def project_to_hyperplane(coefficients, bias, points):
            return np.dot(points, coefficients) + bias

        return lambda X: project_to_hyperplane(coefficients, bias, X)

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
