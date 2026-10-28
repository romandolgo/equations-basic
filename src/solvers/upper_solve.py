import numpy as np
from numpy.typing import NDArray


def upper_solve(U: NDArray, b: NDArray) -> NDArray:
    """
    :param U: np.array((n, n)), float32 or float64, nonsingular upper triangular
    :param b: np.array((n,)), same dtype as U

    :return: np.array((n,)), same dtype as U -- x
    """
    if not np.all(np.isfinite(U)) or not np.all(np.isfinite(b)):
        raise ArithmeticError("Матрица U или вектор b содержат неконечные значения")

    n: int = U.shape[0]
    x: NDArray = np.zeros(n, dtype=U.dtype)
    for i in range(n - 1, -1, -1):
        sum_upper: np.floating = U.dtype.type(U[i, i + 1:] @ x[i + 1:])
        if U[i, i] == 0:
            raise ZeroDivisionError(
                f"Нулевой диагональный элемент на шаге {i=}: {U[i, i]=}"
            )
        x[i] = (b[i] - sum_upper) / U[i, i]
        if not np.isfinite(x[i]):
            raise ArithmeticError(
                f"Неконечный результат обратной подстановки на шаге {i=}: {x[i]=}"
            )
    return x
