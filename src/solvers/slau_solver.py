import sys
from pathlib import Path

import numpy as np

# make `from decomps.lup import lup_factor` work when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decomps.lup import lup_factor
from decomps.ldlt import ldlt_factor 
from decomps.qr import qr_factor
from solvers.lup_solve import lup_solve
from solvers.ldlt_solve import ldlt_solve
from solvers.qr_solve import qr_solve

METHODS = ("lup", "ldlt", "qr")
PROMPT_A = "Введите матрицу A:"
PROMPT_B = "Введите вектор b:"


def parse_args(args):
    """Возвращает (path, method); path = None означает ввод с консоли."""
    path, method = None, "lup"
    if len(args) == 1:
        if args[0] in METHODS:
            method = args[0]
        else:
            path = args[0]
    elif len(args) >= 2:
        path, method = args[0], args[1]
    return path, method


def parse_row(line):
    return [float(token) for token in line.split()]


def read_from_file(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()

    rows = []
    i = 0
    while i < len(lines) and lines[i].strip():
        rows.append(parse_row(lines[i]))
        i += 1

    b = [float(line) for line in lines[i:] if line.strip()]
    return np.array(rows, dtype=np.float64), np.array(b, dtype=np.float64)


def read_from_console():
    print(PROMPT_A)
    rows = []
    while True:
        line = input()
        if not line.strip():
            break
        rows.append(parse_row(line))

    n = len(rows)
    print(PROMPT_B)
    b = [float(input()) for _ in range(n)]
    return np.array(rows, dtype=np.float64), np.array(b, dtype=np.float64)


def solve(A, b, method):
    if method == "lup":
        L, U, p = lup_factor(A)
        return lup_solve(L, U, p, b)
    if method == "ldlt":
        L, d = ldlt_factor(A)
        return ldlt_solve(L, d, b)
    F, tau = qr_factor(A)
    return qr_solve(F, tau, b)


# =================================== main ===================================
def main(args):
    """
    :param args: list[str], command-line arguments without the program name
    """
    path, method = parse_args(args)
    if path is None:
        A, b = read_from_console()
    else:
        A, b = read_from_file(path)

    x = solve(A, b, method)
    for value in x:
        print(format(value, ".17g"))


if __name__ == "__main__":
    main(sys.argv[1:])
