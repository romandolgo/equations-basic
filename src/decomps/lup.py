import numpy as np

# ================================ lup_factor ================================
def lup_factor(A):
    """LUP factorization with partial pivoting.

    :param A: np.array((n, n)), float32 or float64, nonsingular

    :raises ZeroDivisionError: if the pivot at some step is exactly zero

    :return: (L, U, p), computed in the dtype of A
        L - np.array((n, n)), unit lower triangular
        U - np.array((n, n)), upper triangular
        p - np.array((n,)), int, row permutation from 0: A[p] == L @ U
    """
    n = A.shape[0]
    L = np.eye(n, dtype = A.dtype)
    U = A.copy()
    p = np.arange(n)

    for k in range(n):
        q = k + int(np.argmax(np.abs(U[k:, k])))
        pivot = U[q, k]

        if pivot == 0:
            raise ZeroDivisionError(f"the pivot at {k} step is exactly zero")
        if not np.isfinite(pivot):
            raise ArithmeticError(f"the pivot at {k} step is infinite")

        if q != k:
            U[[k, q]] = U[[q, k]]
            p[[k, q]] = p[[q, k]]
            L[[k, q], :k] = L[[q, k], :k]

        if k < n - 1:
            L[k + 1:, k] = U[k + 1:, k] / U[k, k]
            U[k + 1:, k + 1:] -= np.outer(L[k + 1:, k], U[k, k + 1:])
            U[k + 1:, k] = 0

    return L, U, p



