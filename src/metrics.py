import numpy as np


def relative_error(x_computed, x_exact):
    """||x~ - x||_2 / ||x||_2 -- относительная ошибка решения."""
    return np.linalg.norm(x_computed - x_exact) / np.linalg.norm(x_exact)


def relative_residual(A, x_computed, b):
    """||b - A x~||_2 / ||b||_2 -- относительная невязка."""
    return np.linalg.norm(b - A @ x_computed) / np.linalg.norm(b)


def backward_error_1(A, x_computed, b):
    """eta_back,1 = ||b - A x~||_1 / (||A||_1 ||x~||_1 + ||b||_1), 0/0 = 0."""
    residual = np.linalg.norm(b - A @ x_computed, 1)
    denominator = np.linalg.norm(A, 1) * np.linalg.norm(x_computed, 1) \
        + np.linalg.norm(b, 1)
    if denominator == 0:
        return 0.0
    return residual / denominator


def condition_1_estimate(A, solve):
    """kappa~_1(A) = ||A||_1 * max_j ||z_j||_1, где A z_j = e_j.

    :param solve: функция b -> x, решающая Ax = b по готовым факторам,
                  например lambda b: lup_solve(L, U, p, b)
    """
    n = A.shape[0]
    largest_column = 0.0
    for j in range(n):
        e_j = np.zeros(n, dtype=A.dtype)
        e_j[j] = 1
        z_j = solve(e_j)                      # j-й столбец A^{-1}
        largest_column = max(largest_column, np.linalg.norm(z_j, 1))
    return np.linalg.norm(A, 1) * largest_column


def growth_factor(A, U):
    """rho_U = max |u_ij| / max |a_ij| -- рост элементов при исключении."""
    return np.max(np.abs(U)) / np.max(np.abs(A))
