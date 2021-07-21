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

        N = X.shape[0]

        P_raw = np.dot(X, X.T) * np.outer(Y, Y.T)
        q_raw = np.ones(N) * -1
        G_raw = -1 * np.identity(N)
        h_raw = np.zeros(N)
        A_raw = Y
        b_raw = np.zeros(N)

        print()
        print()
        print(f"[INFO] P_raw.shape = {P_raw.shape}")
        print(f"[INFO] q_raw.shape = {q_raw.shape}")
        print(f"[INFO] G_raw.shape = {G_raw.shape}")
        print(f"[INFO] h_raw.shape = {h_raw.shape}")
        print(f"[INFO] A_raw.shape = {A_raw.shape}")
        print(f"[INFO] b_raw.shape = {b_raw.shape}")
        print()
        print()

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

        pass

    def predict(self, X_test):
        return []
