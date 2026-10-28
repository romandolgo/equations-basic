import numpy as np

# =============================== upper_solve ================================
def upper_solve(U, b):
    """
    :param U: np.array((n, n)), float32 or float64, nonsingular upper triangular
    :param b: np.array((n,)), same dtype as U

    :return: np.array((n,)), same dtype as U -- x
    """
    n = U.shape[0]
    x = np.zeros(U.shape[0], dtype=U.dtype)
    for i in range(n - 1, -1, -1):
        sum_upper = U[i, i + 1:] @ x[i + 1:]
        if (U[i, i] == 0):
            raise ZeroDivisionError(f"zero diagonal element at step {i}")
        x[i] = (b[i] - sum_upper) / U[i, i]
    return x
