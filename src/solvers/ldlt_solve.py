import numpy as np
from numpy.typing import NDArray

from solvers.lower_solve import lower_solve
from solvers.upper_solve import upper_solve


def ldlt_solve(L: NDArray, d: NDArray, b: NDArray) -> NDArray:
    """
    :param L, d: returned by ldlt_factor, must not be modified
    :param b: np.array((n,)), same dtype as L

    :return: np.array((n,)), same dtype as L -- x
    """
    if not np.all(np.isfinite(d)):
        raise ArithmeticError("Диагональ d содержит неконечные значения")
    if np.any(d == 0):
        zero_idx: int = int(np.flatnonzero(d == 0)[0])
        raise ZeroDivisionError(f"Нулевой диагональный элемент: {zero_idx=}")

    y: NDArray = lower_solve(L, b)
    x: NDArray = upper_solve(L.T, y / d)
    return x
