import numpy as np

# =============================== ldlt_factor ================================
def ldlt_factor(A):
    """LDL^T factorization without pivoting.

    :param A: np.array((n, n)), float32 or float64, symmetric

    :raises ZeroDivisionError: if some d_k is exactly zero

    :return: (L, d), computed in the dtype of A
        L - np.array((n, n)), unit lower triangular
        d - np.array((n,)), A == L @ np.diag(d) @ L.T
    """
    n = A.shape[0]
    L = np.eye(n, dtype=A.dtype)
    d = np.zeros(n, dtype=A.dtype)

    for k in range(n):

        w = (L[k, :k] * d[:k])
        d[k] = A[k, k] - L[k, :k] @ w

        if d[k] == 0:
            raise ZeroDivisionError(f"d at step {k} is exactly zero")
        if not np.isfinite(d[k]):
            raise ArithmeticError(f"d at step {k} is infinite")

        if k < n - 1:
            L[k + 1:, k] = (A[k + 1:, k] - L[k + 1:, :k] @ w) / d[k]

    return L, d
