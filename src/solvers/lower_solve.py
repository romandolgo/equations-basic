import numpy as np
from numpy.typing import NDArray


def lower_solve(L: NDArray, b: NDArray) -> NDArray:
    """
    :param L: np.array((n, n)), float32 or float64, nonsingular lower
              triangular, diagonal not necessarily unit
    :param b: np.array((n,)), same dtype as L

    :return: np.array((n,)), same dtype as L -- x
    """
    if not np.all(np.isfinite(L)) or not np.all(np.isfinite(b)):
        raise ArithmeticError("Матрица L или вектор b содержат неконечные значения")

    n: int = L.shape[0]
    x: NDArray = np.zeros(n, dtype=L.dtype)
    for i in range(n):
        sum_lower: np.floating = L.dtype.type(L[i, :i] @ x[:i])
        if L[i, i] == 0:
            raise ZeroDivisionError(
                f"Нулевой диагональный элемент на шаге {i=}: {L[i, i]=}"
            )
        x[i] = (b[i] - sum_lower) / L[i, i]
        if not np.isfinite(x[i]):
            raise ArithmeticError(
                f"Неконечный результат прямой подстановки на шаге {i=}: {x[i]=}"
            )
    return x
