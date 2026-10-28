# =============================== lower_solve ================================
import numpy as np


def lower_solve(L, b):
    """
    :param L: np.array((n, n)), float32 or float64, nonsingular lower
              triangular, diagonal not necessarily unit
    :param b: np.array((n,)), same dtype as L

    :return: np.array((n,)), same dtype as L -- x
    """

    n = L.shape[0]
    x = np.zeros(L.shape[0], dtype=L.dtype)
    for i in range(n):
        sum_lower = L[i, :i] @ x[:i]
        if (L[i, i] == 0):
            raise ZeroDivisionError(f"zero diagonal element at step {i}")
        x[i] = (b[i] - sum_lower) / L[i, i]
    return x
