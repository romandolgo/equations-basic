from solvers.lower_solve import lower_solve
from solvers.upper_solve import upper_solve


# ================================ lup_solve =================================
def lup_solve(L, U, p, b):
    """
    :param L, U, p: returned by lup_factor, must not be modified
    :param b: np.array((n,)), same dtype as L

    :return: np.array((n,)), same dtype as L -- x
    """
    y = lower_solve(L, b[p])
    x = upper_solve(U, y)
    return x