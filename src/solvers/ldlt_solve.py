from solvers.lower_solve import lower_solve
from solvers.upper_solve import upper_solve

# ================================ ldlt_solve ================================
def ldlt_solve(L, d, b):
    """
    :param L, d: returned by ldlt_factor, must not be modified
    :param b: np.array((n,)), same dtype as L

    :return: np.array((n,)), same dtype as L -- x
    """
    y = lower_solve(L, b)
    x = upper_solve(L.T, y / d)
    return x
