import cvxopt
import numpy as np

###


class Optimizer():
    def __cvxopt_hard_formulation(self, kernel, classes):
        """
        TODO: re-write the formulation below ...

        Lagrangian Formulation \\
        `min L(w,b)` \\
        `max L(λ)` \\
        `L(w, b, λ) = 1/2 * ||w||^2 + ∑ λi * (yi * (w * xi + b) - 1)`

        Dual of Lagrangian Formulation \\
        `max F(λ)` \\
        `F(λ) = ∑ λi - 1/2 * (∑ ∑ λi * λj * yi * yj * xi * xj)`
        
        CVXOPT Formulation \\
        `min F(x)` \\
        `F(x) = 1/2 * (x.T * P * x) + (q.T * x)` \\
        `Gx ≤ h` \\
        `Ax = b`

        Our problem is to `maximize` the `F(λ)` (lagrangian dual-problem), but the
        library CVXOPT accepts a problem formulated as a `minimization` problem.
        In order to obtain a MIN problem, we start from the dual and we multiply
        by -1 the entire objective function `F(λ)`.

        Lagrangian Dual Problem as Minimization Problem \\
        `min -F(λ)` \\
        `-F(λ) = 1/2 * (∑ ∑ λi * λj * yi * yj * xi * xj) - ∑ λi`

        CVXOPT Formulation Applied \\
        `min -F(λ)` \\
        `-F(λ) = 1/2 * (λ.T * (X * Y) * λ) - (I.T * λ)` \\
        `(-1 * I * λi) ≤ 0` \\
        `∑ λi * yi = 0`

        CVXOPT Variables:
         - P = (X * Y)
         - q = -I.T
         - G = -I
         - h = 0 (vector)
         - A = Y
         - b = 0 (number)
        """

        ### number of Lagrangian multipliers parameters.
        N = classes.shape[0]

        # {cvxopt} Parameters Formulation
        P = np.outer(classes, classes.T) * kernel
        q = np.full(N, 1) * -1
        G = np.identity(N) * -1
        h = np.full(N, 0)
        A = classes
        b = .0

        ### {cvxopt} matrix shape compatibility (for moltiplication).
        A = A.reshape((1, N))
        ###  {cvxopt} array types compatibility.
        q = q.astype(np.double)
        G = G.astype(np.double)
        h = h.astype(np.double)

        return (P, q, G, h, A, b)

    #

    def __cvxopt_soft_formulation(self, kernel, classes, C):
        """
        Objective Function \\
        `min (1/2 ||W||^2) + (C • ∑ ξ)`

        Constraints \\
        `∀i . yi (xi • W + b) -1 + ξ ≥ 0`

        Primal Lagrangian Formulation \\
        `min Lp(w,b)` \\
        `max Ld(λ)` \\
        `L(w, b, λ) = (1/2 ||W||^2) + (C ∑ ξi) - (∑ λi yi (xi • W + b) - 1 + ξi) - (∑ μi ξi)`

        Bordered Hessian \\
        `H = (Y • Y.T) (X.T • X)`

        Dual Lagrangian Formulation \\
        `max Ld(λ)` \\
        `F(λ) = ∑ λi - 1/2 (λ • H • λ.T)`
        
        CVXOPT Formulation \\
        `min F(x)` \\
        `F(x) = 1/2 * (x.T * P * x) + (q.T * x)` \\
        `Gx ≤ h` \\
        `Ax = b`

        Our problem is to `maximize` the `Ld(λ)` (lagrangian dual-problem), but the
        library CVXOPT accepts a problem formulated as a `minimization` problem.
        In order to obtain a MIN problem, we start from the dual and we multiply
        by -1 the entire objective function `Ld(λ)`.

        Lagrangian Dual Problem as Minimization Problem \\
        `min -Ld(λ)` \\
        `-F(λ) = 1/2 (λ • H • λ.T) - (∑ λi)`

        CVXOPT Formulation Applied \\
        `min -F(λ)` \\
        `-F(λ) = 1/2 * (λ.T * (X * Y) * λ) - (I.T * λ)` \\
        `∀i . -I λi ≤ 0` \\
        `∀i .  I λi ≤ C` \\
        `∑  . yi λi = 0`

        CVXOPT Variables:
         - P = (X * Y)
         - q = -I.T
         - G = -I :: I
         - h = 0s :: C
         - A = Y
         - b = 0 (number)
        """

        ### number of Lagrangian multipliers parameters.
        N = classes.shape[0]

        # {cvxopt} Parameters Formulation
        P = np.outer(classes, classes.T) * kernel
        q = np.full(N, 1) * -1
        G = np.vstack((np.identity(N) * -1, np.identity(N)))
        h = np.hstack((np.full(N, 0), np.full(N, C)))
        A = classes
        b = .0

        ### {cvxopt} matrix shape compatibility (for moltiplication).
        A = A.reshape((1, N))
        ###  {cvxopt} array types compatibility.
        q = q.astype(np.double)
        G = G.astype(np.double)
        h = h.astype(np.double)

        return (P, q, G, h, A, b)

    def __cvxopt_matrix_conversion(self, P_form, q_form, G_form, h_form, A_form, b_form):
        P = cvxopt.matrix(P_form.copy())
        q = cvxopt.matrix(q_form.copy())
        G = cvxopt.matrix(G_form.copy())
        h = cvxopt.matrix(h_form.copy())
        A = cvxopt.matrix(A_form.copy())
        b = cvxopt.matrix(b_form)

        return (P, q, G, h, A, b)

    def __cvxopt_qp_solve(self, P, q, G, h, A, b):
        return cvxopt.solvers.qp(P, q, G, h, A, b)

    #

    def initialize(self):
        cvxopt.solvers.options['show_progress'] = False

    def cvxopt_hard_margin_solve(self, classes, kernel):
        P_form, q_form, G_form, h_form, A_form, b_form = self.__cvxopt_hard_formulation(
            kernel, classes
        )

        P, q, G, h, A, b = self.__cvxopt_matrix_conversion(
            P_form, q_form, G_form, h_form, A_form, b_form
        )

        solution = self.__cvxopt_qp_solve(P, q, G, h, A, b)

        return solution

    def cvxopt_soft_margin_solve(self, classes, kernel, C):
        P_form, q_form, G_form, h_form, A_form, b_form = self.__cvxopt_soft_formulation(
            kernel, classes, C
        )

        P, q, G, h, A, b = self.__cvxopt_matrix_conversion(
            P_form, q_form, G_form, h_form, A_form, b_form
        )

        return cvxopt.solvers.qp(P, q, G, h, A, b)

    def scipy_solve(self, points, classes, kernel=None):
        pass
