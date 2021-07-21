import cvxopt
import numpy as np

###


class SVM():
    def __init__(self):
        self.kernel = None

    #

    def __cvxopt_formulation(self, X, Y):
        """
        Lagrangian Formulation (MIN) \\
        `L(λ, w, b) = 1/2 * ||w||^2 + ∑ λi * (yi * (w * xi + b) - 1)`

        Dual of Lagrangian Formulation (MAX) \\
        `F(λ) = ∑ λi - 1/2 * (∑ ∑ λi * λi * yi * yj * xi * xj)`
        
        CVXopt Formulation (MIN) \\
        `1/2 * (x.T * P * x) + (q.T * x)` \\
        `Gx ≤ h` \\
        `Ax = b`

        in order to obtain a MIN problem, we start from the dual
        and we multiply by -1 the entire objective function.

        CVXopt Formulation Applied (MIN) \\
        `1/2 * (λ.T * (X * Y) * λ) - (I.T * λ)` \\
        `(-1 * I * λi) ≤ 0` \\
        `∑ yi * λi = 0`

        CVXopt Variables:
         - P = (X * Y) = Hessian
         - q = -I.T
         - G = -I
         - h = 0 (vector)
         - A = Y
         - b = 0 (number)
        """

        # {cvxopt} array types compatibility.
        points = X.copy().astype(np.double)
        classes = Y.copy().astype(np.double)

        N = classes.shape[0]

        # {cvxopt} Parameters Formulation
        P_raw = np.outer(classes, classes.T) * np.dot(points, points.T)
        q_raw = -1 * np.full(N, 1)
        G_raw = -1 * np.identity(N)
        h_raw = np.full(N, 0)
        A_raw = classes
        b_raw = .0

        # {cvxopt} matrix shape compatibility.
        A_raw = A_raw.reshape((1, N))
        # {cvxopt} array types compatibility.
        q_raw = q_raw.astype(np.double)
        G_raw = G_raw.astype(np.double)
        h_raw = h_raw.astype(np.double)

        # {cvxopt} casting required.
        P = cvxopt.matrix(P_raw)
        q = cvxopt.matrix(q_raw)
        G = cvxopt.matrix(G_raw)
        h = cvxopt.matrix(h_raw)
        A = cvxopt.matrix(A_raw)
        b = cvxopt.matrix(b_raw)

        return (P, q, G, h, A, b)

    def fit(self, X, Y):
        # solve QP problem
        P, q, G, h, A, b = self.__cvxopt_formulation(X, Y)
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

    def predict(self, X_test):
        return []