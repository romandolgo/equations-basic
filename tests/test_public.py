import ast
import importlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.linalg
from numpy.testing import assert_allclose, assert_array_equal

SRC = Path(__file__).resolve().parents[1] / "src"
UROUND = 2.0**-53
C = 16  # moderate constant C(n) in backward error bounds
REF = 1e-10  # agreement with the library reference on well-conditioned data
LAYOUTS = ["fortran", "strided", "negative-strides"]
METHODS = ["lup", "ldlt", "qr"]
LS_METHODS = ["qr", "normal_ldlt", "normal_lup"]


def _raw(module, name):
    return getattr(importlib.import_module(module), name)


def _student(module, name, copy=()):
    """Import lazily; pass the arguments listed in `copy` as copies (factors are passed as is)."""

    def call(*args, **kwargs):
        args = [
            (
                np.array(value, copy=True)
                if index in copy and isinstance(value, np.ndarray)
                else value
            )
            for index, value in enumerate(args)
        ]
        return _raw(module, name)(*args, **kwargs)

    call.__name__ = name
    return call


lup_factor = _student("decomps.lup", "lup_factor", copy={0})
ldlt_factor = _student("decomps.ldlt", "ldlt_factor", copy={0})
qr_factor = _student("decomps.qr", "qr_factor", copy={0})
apply_q = _student("decomps.qr", "apply_q", copy={2})
form_q = _student("decomps.qr", "form_q")
lower_solve = _student("solvers.lower_solve", "lower_solve", copy={1})
upper_solve = _student("solvers.upper_solve", "upper_solve", copy={1})
lup_solve = _student("solvers.lup_solve", "lup_solve", copy={3})
ldlt_solve = _student("solvers.ldlt_solve", "ldlt_solve", copy={2})
qr_solve = _student("solvers.qr_solve", "qr_solve", copy={2})
lstsq = _student("mnk.lstsq", "lstsq", copy={0, 1})
tikhonov = _student("mnk.tikhonov", "tikhonov", copy={0, 1, 2, 3})


def _random(shape, seed):
    return np.random.default_rng(seed).normal(size=shape)


def _spd(n, seed):
    M = _random((n, n), seed)
    A = M.T @ M + n * np.eye(n)
    return (A + A.T) / 2


def _with_condition(n, kappa, seed):
    """Singular values from 1 to 1/kappa."""
    rng = np.random.default_rng(seed)
    left, _ = np.linalg.qr(rng.normal(size=(n, n)))
    right, _ = np.linalg.qr(rng.normal(size=(n, n)))
    return (left * np.geomspace(1.0, 1.0 / kappa, n)) @ right.T


def _wilkinson(n):
    W = np.eye(n) - np.tril(np.ones((n, n)), -1)
    W[:, -1] = 1.0
    return W


def _planted_ldlt(n, seed):
    """Indefinite A = L diag(d) L^T whose LDLT is exact in any operation order."""
    rng = np.random.default_rng(seed)
    L = np.tril(rng.integers(-4, 5, size=(n, n)) / 4.0, -1) + np.eye(n)
    d = rng.choice([-1.0, 1.0], n) * 2.0 ** rng.integers(-2, 3, n)
    if n > 1:
        d[:2] = [-1.0, 2.0]
    return (L * d) @ L.T, L, d


def _chain_network(nodes=20):
    """Reduced Kirchhoff matrix of the chain from Task IV."""
    g = 10.0 ** (-3 + 4 * np.arange(nodes) / (nodes - 1))
    B = np.eye(nodes) - np.eye(nodes, k=-1)
    return B.T @ (g[:, None] * B), g


def _layout(A, layout):
    if layout == "fortran":
        return np.asfortranarray(A)
    if layout == "strided":
        storage = np.zeros(
            (2 * A.shape[0], 3 * A.shape[-1])
            if A.ndim == 2
            else 3 * A.shape[0]
        )
        if A.ndim == 2:
            storage[::2, ::3] = A
            return storage[::2, ::3]
        storage[::3] = A
        return storage[::3]
    if layout == "negative-strides":
        storage = A[(slice(None, None, -1),) * A.ndim].copy()
        return storage[(slice(None, None, -1),) * A.ndim]
    raise ValueError(layout)


def _check_array(value, shape, name):
    assert isinstance(
        value, np.ndarray
    ), f"{name}: expected numpy.ndarray, got {type(value).__name__}"
    assert (
        value.shape == shape
    ), f"{name}: shape {value.shape}, expected {shape}"
    assert (
        value.dtype == np.float64
    ), f"{name}: dtype {value.dtype}, expected float64"
    assert np.all(np.isfinite(value)), f"{name}: contains NaN or inf"
    return value


def _check_unit_lower(L, name="L"):
    assert np.all(
        np.triu(L, 1) == 0
    ), f"{name}: must be exactly zero above the diagonal"
    assert np.all(np.diag(L) == 1), f"{name}: must have a unit diagonal"


def _backward_error(A, x, b):
    """eta_back,1 = ||b - Ax||_1 / (||A||_1 ||x||_1 + ||b||_1), 0/0 = 0."""
    denominator = np.linalg.norm(A, 1) * np.linalg.norm(x, 1) + np.linalg.norm(
        b, 1
    )
    return (
        0.0 if denominator == 0 else np.linalg.norm(b - A @ x, 1) / denominator
    )


def _omega(A, x, b):
    """omega from Task II."""
    norm = np.linalg.norm(A)
    denominator = norm * (norm * np.linalg.norm(x) + np.linalg.norm(b))
    return (
        0.0
        if denominator == 0
        else np.linalg.norm(A.T @ (b - A @ x)) / denominator
    )


def _assert_solution(A, b, x, x_true=None):
    n = len(b)
    _check_array(x, (n,), "x")
    eta = _backward_error(A, x, b)
    assert (
        eta <= C * n * UROUND
    ), f"backward error {eta:.3e} > {C * n * UROUND:.3e}"
    if x_true is not None:
        kappa = np.linalg.cond(A, 1)
        error = np.linalg.norm(x - x_true, 1) / np.linalg.norm(x_true, 1)
        bound = 2 * C * n * UROUND * kappa
        assert (
            error <= bound
        ), f"relative error {error:.3e} > {bound:.3e} (kappa_1 = {kappa:.3e})"


def _lup_reference(A):
    lu, piv = scipy.linalg.lu_factor(A)
    p = np.arange(len(A))
    for k, q in enumerate(piv):
        p[[k, q]] = p[[q, k]]
    return np.tril(lu, -1) + np.eye(len(A)), np.triu(lu), p


def _check_lup(A, result, compare=True):
    n = len(A)
    assert len(result) == 3, "lup_factor must return (L, U, p)"
    L, U, p = result
    _check_array(L, (n, n), "L")
    _check_array(U, (n, n), "U")
    assert isinstance(p, np.ndarray) and p.shape == (
        n,
    ), f"p: expected numpy.ndarray of shape ({n},)"
    assert np.issubdtype(
        p.dtype, np.integer
    ), f"p: expected an integer array, got {p.dtype}"
    assert_array_equal(
        np.sort(p), np.arange(n), err_msg="p must be a permutation of 0..n-1"
    )
    _check_unit_lower(L)
    assert np.all(
        np.tril(U, -1) == 0
    ), "U: must be exactly zero below the diagonal"
    largest = np.max(np.abs(L))
    assert (
        largest <= 1 + 4 * UROUND
    ), f"partial pivoting gives |l_ik| <= 1, got {largest!r}"
    residual = np.linalg.norm(A[p] - L @ U)
    bound = C * n * UROUND * np.linalg.norm(np.abs(L) @ np.abs(U))
    assert residual <= bound, f"||A[p] - LU||_F = {residual:.3e} > {bound:.3e}"
    if compare:
        L_ref, U_ref, p_ref = _lup_reference(A)
        gap = 1 - np.max(np.abs(np.tril(L_ref, -1)), initial=0.0)
        assert gap > 1e-8, "test data: near tie between pivot candidates"
        assert_array_equal(p, p_ref, err_msg="wrong pivot order")
        assert_allclose(
            L, L_ref, rtol=0, atol=REF, err_msg="L differs from reference"
        )
        assert_allclose(
            U,
            U_ref,
            rtol=0,
            atol=REF * np.max(np.abs(U_ref)),
            err_msg="U differs from reference",
        )
    return L, U, p


def _assert_lup(A, compare=True):
    return _check_lup(A, lup_factor(A), compare)


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 16, 31, 64])
def test_lup_random_matrices(n, seed):
    _assert_lup(_random((n, n), seed))


def test_lup_later_swap_moves_computed_multipliers():
    A = np.array([[4.0, 1.0, 2.0], [2.0, 1.0, 3.0], [1.0, 3.0, 1.0]])
    L, U, p = _assert_lup(A)
    assert_array_equal(
        p, [0, 2, 1], err_msg="row 2 must be the pivot at step 2"
    )
    assert_allclose(
        L,
        [[1, 0, 0], [0.25, 1, 0], [0.5, 2 / 11, 1]],
        rtol=0,
        atol=1e-15,
        err_msg="swapped rows must carry their multipliers",
    )
    assert_allclose(
        U, [[4, 1, 2], [0, 2.75, 0.5], [0, 0, 21 / 11]], rtol=0, atol=1e-14
    )


def test_lup_tie_in_first_column_takes_smallest_index():
    A = np.array([[1.0, 2.0, 0.0], [-4.0, 1.0, 1.0], [4.0, 0.0, 2.0]])
    L, U, p = _assert_lup(A, compare=False)
    assert_array_equal(
        p, [1, 0, 2], err_msg="ties go to the smallest row index"
    )
    assert_allclose(
        L, [[1, 0, 0], [-0.25, 1, 0], [-1, 4 / 9, 1]], rtol=0, atol=1e-15
    )
    assert_allclose(
        U, [[-4, 1, 1], [0, 2.25, 0.25], [0, 0, 26 / 9]], rtol=0, atol=1e-14
    )


def test_lup_tie_uses_current_row_index():
    # after step 1 original row 0 sits at position 2; the tie at step 2 keeps position 1
    A = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [2.0, 0.0, 1.0]])
    L, U, p = _assert_lup(A, compare=False)
    assert_array_equal(
        p, [2, 1, 0], err_msg="ties go to the smallest current row index"
    )
    assert_array_equal(L, [[1, 0, 0], [0, 1, 0], [0.5, 1, 1]])
    assert_array_equal(U, [[2, 0, 1], [0, 1, 0], [0, 0, -0.5]])


@pytest.mark.parametrize("eps", [1e-3, 2.0**-30, 1e-20])
def test_lup_small_leading_entry_is_not_a_pivot(eps):
    A = np.array([[eps, 1.0], [1.0, 1.0]])
    L, U, p = _assert_lup(A)
    assert_array_equal(p, [1, 0], err_msg="a small a_11 must not be the pivot")
    assert_allclose(L, [[1, 0], [eps, 1]], rtol=1e-15, atol=0)
    assert_allclose(U, [[1, 1], [0, 1 - eps]], rtol=1e-15, atol=0)


@pytest.mark.parametrize("n", [5, 12])
def test_lup_upper_triangular_input_needs_no_swaps(n):
    A = np.triu(_random((n, n), n))
    L, U, p = _assert_lup(A)
    assert_array_equal(
        p, np.arange(n), err_msg="no swaps for a triangular matrix"
    )
    assert_array_equal(L, np.eye(n))
    assert_array_equal(U, A)


@pytest.mark.parametrize("seed", [3, 4])
def test_lup_permutation_matrix(seed):
    order = np.random.default_rng(seed).permutation(9)
    A = np.eye(9)[order]
    L, U, p = _assert_lup(A)
    assert_array_equal(A[p], np.eye(9), err_msg="A[p] must be I")
    assert_array_equal(L, np.eye(9))
    assert_array_equal(U, np.eye(9))


@pytest.mark.parametrize("n", [2, 8, 16, 32])
def test_lup_wilkinson_growth(n):
    L, U, p = _assert_lup(_wilkinson(n), compare=False)
    expected_U = np.eye(n)
    expected_U[:, -1] = 2.0 ** np.arange(n)
    assert_array_equal(p, np.arange(n), err_msg="ties must not cause swaps")
    assert_array_equal(L, np.eye(n) - np.tril(np.ones((n, n)), -1))
    assert_array_equal(
        U, expected_U, err_msg="last column of U must be 1, 2, ..., 2^(n-1)"
    )


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_lup_row_order_of_input_does_not_change_factors(seed):
    A = _random((12, 12), 5)
    L, U, p = _assert_lup(A)
    order = np.random.default_rng(seed).permutation(12)
    L2, U2, p2 = _assert_lup(A[order])
    assert_array_equal(
        order[p2], p, err_msg="pivoting must not depend on the input row order"
    )
    assert_allclose(L2, L, rtol=0, atol=1e-14)
    assert_allclose(U2, U, rtol=0, atol=1e-14 * np.max(np.abs(U)))


@pytest.mark.parametrize("factor", [2.0**-600, 0.5, 2.0**40, -(2.0**300)])
def test_lup_power_of_two_scaling_is_exact(factor):
    A = _random((10, 10), 21)
    L, U, p = _assert_lup(A)
    L2, U2, p2 = _assert_lup(A * factor)
    assert_array_equal(p2, p, err_msg="scaling must not change pivots")
    assert_array_equal(L2, L, err_msg="scaling by 2^k must not change L")
    assert_array_equal(
        U2, U * factor, err_msg="U must scale exactly (no absolute thresholds)"
    )


@pytest.mark.parametrize("layout", LAYOUTS)
def test_lup_memory_layouts(layout):
    A = _random((13, 13), 50)
    _check_lup(A, _raw("decomps.lup", "lup_factor")(_layout(A, layout)))


@pytest.mark.parametrize("kappa", [1e8, 1e14])
def test_lup_ill_conditioned_is_backward_stable(kappa):
    _assert_lup(_with_condition(20, kappa, seed=17), compare=False)


def _ldlt_from_cholesky(A):
    H = np.linalg.cholesky(A)
    s = np.diag(H)
    return H / s, s**2


def _check_ldlt(A, result, L_expected=None, d_expected=None, atol=REF):
    n = len(A)
    assert len(result) == 2, "ldlt_factor must return (L, d)"
    L, d = result
    _check_array(L, (n, n), "L")
    _check_array(d, (n,), "d")
    _check_unit_lower(L)
    assert np.all(d != 0), "zero d_k for nonzero leading minors"
    residual = np.linalg.norm(A - (L * d) @ L.T)
    bound = (
        C * n * UROUND * np.linalg.norm((np.abs(L) * np.abs(d)) @ np.abs(L).T)
    )
    assert (
        residual <= bound
    ), f"||A - L diag(d) L^T||_F = {residual:.3e} > {bound:.3e}"
    if L_expected is not None:
        assert_allclose(
            L, L_expected, rtol=0, atol=atol, err_msg="L differs from expected"
        )
        assert_allclose(
            d,
            d_expected,
            rtol=0,
            atol=atol * np.max(np.abs(d_expected)),
            err_msg="d differs from expected",
        )
    return L, d


def _assert_ldlt(A, L_expected=None, d_expected=None, atol=REF):
    return _check_ldlt(A, ldlt_factor(A), L_expected, d_expected, atol)


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 16, 40])
def test_ldlt_spd_matrices(n, seed):
    A = _spd(n, seed)
    _, d = _assert_ldlt(A, *_ldlt_from_cholesky(A))
    assert np.all(d > 0), "d_k must be positive for an SPD matrix"


@pytest.mark.parametrize(
    "A,L_expected,d_expected",
    [
        pytest.param([[4, 2], [2, 3]], [[1, 0], [0.5, 1]], [4, 2], id="spd"),
        pytest.param(
            [[1, 2], [2, 1]], [[1, 0], [2, 1]], [1, -3], id="indefinite"
        ),
        pytest.param(
            [[1, 2], [2, 5]], [[1, 0], [2, 1]], [1, 1], id="no-pivoting"
        ),
        pytest.param(
            [[2, -1, 0], [-1, 2, -1], [0, -1, 2]],
            [[1, 0, 0], [-0.5, 1, 0], [0, -2 / 3, 1]],
            [2, 1.5, 4 / 3],
            id="tridiagonal",
        ),
    ],
)
def test_ldlt_known_factors(A, L_expected, d_expected):
    _assert_ldlt(
        np.array(A, dtype=float),
        np.array(L_expected),
        np.array(d_expected),
        atol=1e-15,
    )


@pytest.mark.parametrize("n,seed", [(3, 0), (6, 1), (12, 2)])
def test_ldlt_indefinite_planted_factors(n, seed):
    A, L, d = _planted_ldlt(n, seed)
    _assert_ldlt(A, L, d, atol=1e-15)


def test_ldlt_diagonal_matrix():
    d = np.array([3.0, -0.5, 2.0**-40, -7.0, 1e10])
    L, result = _assert_ldlt(np.diag(d))
    assert_array_equal(L, np.eye(5))
    assert_array_equal(result, d)


def test_ldlt_saddle_point_system():
    G, _ = _chain_network()
    c = np.zeros(20)
    c[0], c[-1] = 1.0, -1.0
    H = np.block([[G, c[:, None]], [c[None, :], np.zeros((1, 1))]])
    _, d = _assert_ldlt(H)
    assert np.all(d[:-1] > 0), "leading SPD block must give positive d_k"
    expected = -c @ np.linalg.solve(G, c)
    assert d[-1] == pytest.approx(
        expected, rel=1e-8
    ), "last d_k must equal -c^T G^{-1} c"


@pytest.mark.parametrize("seed", [5, 6])
def test_ldlt_symmetric_power_of_two_scaling_is_exact(seed):
    A = _spd(9, seed)
    L, d = _assert_ldlt(A)
    s = 2.0 ** np.random.default_rng(seed).integers(-200, 200, 9)
    L2, d2 = _assert_ldlt(A * s[:, None] * s)
    assert_array_equal(
        L2, L * s[:, None] / s, err_msg="L must scale exactly for S A S"
    )
    assert_array_equal(d2, d * s**2, err_msg="d must scale exactly for S A S")


@pytest.mark.parametrize("n", [6, 10])
def test_ldlt_hilbert_is_backward_stable(n):
    _assert_ldlt(scipy.linalg.hilbert(n))


@pytest.mark.parametrize("layout", LAYOUTS)
def test_ldlt_memory_layouts(layout):
    A = _spd(11, 52)
    _check_ldlt(
        A,
        _raw("decomps.ldlt", "ldlt_factor")(_layout(A, layout)),
        *_ldlt_from_cholesky(A),
    )


def _qr_reference(A):
    (F, tau), _ = scipy.linalg.qr(A, mode="raw")
    n, m = A.shape
    return np.array(F), tau[: min(m, n - 1)]


def _check_qr(A, result, compare=True, full=True):
    n, m = A.shape
    r = min(m, n - 1)
    assert len(result) == 2, "qr_factor must return (F, tau)"
    F, tau = result
    _check_array(F, (n, m), "F")
    _check_array(tau, (r,), "tau")
    R = np.triu(F[:m])
    scale = np.linalg.norm(A)
    for k in range(r):
        tail = F[k + 1 :, k]
        if tau[k] == 0:
            assert not np.any(tail), f"tau[{k}] = 0 requires a zero tail"
            continue
        expected = 2 / (1 + tail @ tail)
        assert (
            abs(tau[k] - expected) <= C * n * UROUND * expected
        ), f"tau[{k}] = {tau[k]!r}, expected 2 / (v^T v) = {expected!r}"
        assert (
            1 - C * n * UROUND <= tau[k] <= 2 + C * n * UROUND
        ), f"tau[{k}] = {tau[k]!r} outside [1, 2]: wrong sign of beta"
    padded = np.vstack((R, np.zeros((n - m, m))))
    columns = np.unique(np.linspace(0, m - 1, min(m, 64)).astype(int))
    QtA = np.column_stack(
        [apply_q(F, tau, A[:, j], transpose=True) for j in columns]
    )
    error = np.linalg.norm(QtA - padded[:, columns])
    assert error <= C * n * UROUND * scale, f"||Q^T A - [R; 0]|| = {error:.3e}"
    QR = np.column_stack([apply_q(F, tau, padded[:, j]) for j in columns])
    error = np.linalg.norm(QR - A[:, columns])
    assert error <= C * n * UROUND * scale, f"||Q [R; 0] - A|| = {error:.3e}"
    Q1 = _check_array(form_q(F, tau), (n, m), "form_q(thin=True)")
    error = np.linalg.norm(Q1.T @ Q1 - np.eye(m))
    assert error <= C * n * UROUND, f"||Q1^T Q1 - I||_F = {error:.3e}"
    error = np.linalg.norm(Q1 @ R - A)
    assert error <= C * n * UROUND * scale, f"||Q1 R - A|| = {error:.3e}"
    if full:
        Q = _check_array(
            form_q(F, tau, thin=False), (n, n), "form_q(thin=False)"
        )
        error = np.linalg.norm(Q.T @ Q - np.eye(n))
        assert error <= C * n * UROUND, f"||Q^T Q - I||_F = {error:.3e}"
        assert_allclose(
            Q[:, :m],
            Q1,
            rtol=0,
            atol=C * n * UROUND,
            err_msg="full Q must extend thin Q",
        )
    if compare:
        F_ref, tau_ref = _qr_reference(A)
        assert_allclose(
            tau,
            tau_ref,
            rtol=0,
            atol=REF,
            err_msg="tau differs from reference",
        )
        assert_allclose(
            R,
            np.triu(F_ref[:m]),
            rtol=0,
            atol=REF * scale,
            err_msg="R differs from reference",
        )
        assert_allclose(
            np.tril(F, -1),
            np.tril(F_ref, -1),
            rtol=0,
            atol=REF,
            err_msg="reflector tails differ from reference",
        )
    return F, tau


def _assert_qr(A, compare=True, full=True):
    return _check_qr(A, qr_factor(A), compare, full)


QR_SHAPES = [
    (1, 1),
    (2, 1),
    (5, 1),
    (2, 2),
    (3, 3),
    (6, 3),
    (7, 7),
    (20, 5),
    (40, 40),
    (100, 10),
]


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("n,m", QR_SHAPES)
def test_qr_random_matrices(n, m, seed):
    _assert_qr(_random((n, m), seed))


def test_qr_known_factors():
    A = np.array([[3.0, 1.0], [4.0, 2.0]])
    F, tau = _assert_qr(A)
    assert_allclose(F, [[-5, -2.2], [0.5, 0.4]], rtol=0, atol=1e-15)
    assert_allclose(tau, [1.6], rtol=0, atol=1e-15)
    assert_allclose(
        form_q(F, tau), [[-0.6, -0.8], [-0.8, 0.6]], rtol=0, atol=1e-15
    )


def test_qr_zero_leading_entry_gives_negative_beta():
    F, tau = _assert_qr(np.array([[0.0], [3.0], [4.0]]))
    assert_allclose(
        F, [[-5.0], [0.6], [0.8]], rtol=0, atol=1e-15, err_msg="sgn+(0) = 1"
    )
    assert_allclose(tau, [1.0], rtol=0, atol=1e-15)


@pytest.mark.parametrize("n,m", [(3, 3), (4, 2), (5, 5)])
def test_qr_zero_tail_still_reflects(n, m):
    # x = (x_1, 0, ..., 0) is still reflected (tau = 2); only x = 0 gives tau = 0, unlike LAPACK
    A = np.eye(n, m)
    F, tau = _assert_qr(A, compare=False)
    r = min(m, n - 1)
    assert_array_equal(tau, np.full(r, 2.0))
    expected = np.zeros((n, m))
    expected[np.arange(m), np.arange(m)] = [-1.0] * r + [1.0] * (m - r)
    assert_array_equal(F, expected)
    Q = np.diag([-1.0] * r + [1.0] * (n - r))
    assert_array_equal(form_q(F, tau, thin=False), Q)


def test_qr_zero_tail_then_zero_leading_entry():
    A = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 5.0]])
    F, tau = _assert_qr(A, compare=False)
    assert_array_equal(tau, [2.0, 1.0])
    assert_array_equal(F, [[-1.0, 0.0], [0.0, -5.0], [0.0, 1.0]])
    assert_array_equal(
        form_q(F, tau, thin=False), [[-1, 0, 0], [0, 0, -1], [0, -1, 0]]
    )


def test_qr_zero_column_keeps_tau_zero():
    A = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 2.0]])
    F, tau = _assert_qr(A, compare=False)
    assert tau[0] == 0.0, "tau_k must be 0 for x = 0"
    assert_array_equal(F[:, 0], [0.0, 0.0, 0.0])
    root = np.sqrt(2.0)
    assert_allclose(F[:, 1], [1.0, -2 * root, root - 1], rtol=0, atol=1e-15)
    assert_allclose(tau, [0.0, 1 + root / 2], rtol=0, atol=1e-15)


@pytest.mark.parametrize("n,m", [(1, 1), (4, 3), (4, 4)])
def test_qr_zero_matrix(n, m):
    A = np.zeros((n, m))
    F, tau = _assert_qr(A, compare=False)
    assert_array_equal(F, A)
    assert_array_equal(tau, np.zeros(min(m, n - 1)))
    assert_array_equal(form_q(F, tau), np.eye(n, m))
    b = _random(n, 1)
    assert_array_equal(apply_q(F, tau, b), b)


@pytest.mark.parametrize("value", [3.0, -2.0])
def test_qr_single_entry(value):
    F, tau = _assert_qr(np.array([[value]]))
    assert tau.shape == (0,), "tau must be empty for n = m = 1"
    assert_array_equal(F, [[value]])
    assert_array_equal(form_q(F, tau, thin=False), [[1.0]])


@pytest.mark.parametrize("n,m", [(4, 4), (9, 3), (30, 30), (50, 7)])
def test_apply_q_matches_library_q(n, m):
    A, b = _random((n, m), n), _random(n, m)
    F, tau = qr_factor(A)
    Q = scipy.linalg.qr(A)[0]
    y = _check_array(apply_q(F, tau, b), (n,), "apply_q")
    assert_allclose(
        y,
        Q @ b,
        rtol=0,
        atol=C * n * UROUND * np.linalg.norm(b),
        err_msg="apply_q must compute Q b by default",
    )
    z = _check_array(
        apply_q(F, tau, b, transpose=True), (n,), "apply_q(transpose=True)"
    )
    assert_allclose(
        z, Q.T @ b, rtol=0, atol=C * n * UROUND * np.linalg.norm(b)
    )
    assert_allclose(
        apply_q(F, tau, z), b, rtol=0, atol=C * n * UROUND * np.linalg.norm(b)
    )
    assert np.linalg.norm(y) == pytest.approx(
        np.linalg.norm(b), rel=C * n * UROUND
    )


def test_form_q_default_is_thin():
    F, tau = qr_factor(_random((7, 3), 5))
    Q = form_q(F, tau)
    assert Q.shape == (7, 3), "form_q must return thin Q_1 by default"
    assert_allclose(Q, form_q(F, tau, thin=True), rtol=0, atol=0)


@pytest.mark.parametrize("seed", [8, 9])
def test_qr_power_of_two_column_scaling_is_exact(seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(9, 5))
    s = rng.choice([-1.0, 1.0], 5) * 2.0 ** rng.integers(-300, 300, 5)
    F, tau = _assert_qr(A)
    F2, tau2 = _assert_qr(A * s)
    assert_array_equal(tau2, tau)
    assert_array_equal(np.tril(F2, -1), np.tril(F, -1))
    assert_array_equal(np.triu(F2[:5]), np.triu(F[:5]) * s)


def test_qr_dependent_columns():
    B = _random((8, 3), 12)
    A = np.column_stack((B, B[:, 0] + B[:, 1], -2 * B[:, 2]))
    _assert_qr(A, compare=False)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_qr_memory_layouts(layout):
    A = _random((15, 6), 53)
    _check_qr(A, _raw("decomps.qr", "qr_factor")(_layout(A, layout)))


TRIANGULAR = {"lower": (lower_solve, np.tril), "upper": (upper_solve, np.triu)}


def _triangular(n, seed, kind):
    rng = np.random.default_rng(seed)
    T = rng.normal(size=(n, n)) / np.sqrt(n)
    T[np.diag_indices(n)] = rng.choice([-1.0, 1.0], n) * rng.uniform(
        1.0, 2.0, n
    )
    return TRIANGULAR[kind][1](T)


def _assert_triangular_solution(T, b, x, kind):
    n = len(b)
    _check_array(x, (n,), "x")
    residual = np.linalg.norm(b - T @ x, np.inf)
    bound = C * n * UROUND * np.linalg.norm(np.abs(T) @ np.abs(x), np.inf)
    assert (
        residual <= bound
    ), f"{kind}_solve: ||b - Tx||_inf = {residual:.3e} > {bound:.3e}"


@pytest.mark.parametrize("kind", ["lower", "upper"])
@pytest.mark.parametrize("n", [1, 2, 5, 17, 64, 200])
def test_triangular_random(kind, n):
    T, b = _triangular(n, n, kind), _random(n, n + 1)
    x = TRIANGULAR[kind][0](T, b)
    _assert_triangular_solution(T, b, x, kind)
    expected = scipy.linalg.solve_triangular(T, b, lower=kind == "lower")
    kappa = np.linalg.cond(T, np.inf)
    assert_allclose(
        x,
        expected,
        rtol=0,
        atol=2 * C * n * UROUND * kappa * np.max(np.abs(expected)),
    )


@pytest.mark.parametrize("kind", ["lower", "upper"])
@pytest.mark.parametrize("n", [4, 30])
def test_triangular_exact_integer_system(kind, n):
    rng = np.random.default_rng(40 + n)
    T = rng.integers(-3, 4, size=(n, n)).astype(float)
    T = TRIANGULAR[kind][1](T, -1 if kind == "lower" else 1)
    T += np.diag(rng.choice([-2.0, -1.0, 1.0, 2.0, 4.0], n))
    x = rng.integers(-5, 6, n).astype(float)
    assert_array_equal(TRIANGULAR[kind][0](T, T @ x), x)


@pytest.mark.parametrize("kind", ["lower", "upper"])
def test_triangular_non_unit_diagonal(kind):
    L = np.array([[2.0, 0.0, 0.0], [1.0, 4.0, 0.0], [-1.0, 2.0, 0.5]])
    T = L if kind == "lower" else L.T
    x = np.array([1.5, -0.25, 3.0])
    assert_array_equal(TRIANGULAR[kind][0](T, T @ x), x)


@pytest.mark.parametrize("kind", ["lower", "upper"])
def test_triangular_ill_conditioned_is_backward_stable(kind):
    n = 60
    L = np.eye(n) - np.tril(np.ones((n, n)), -1)  # kappa ~ 2^n
    T = L if kind == "lower" else L.T
    b = _random(n, 61)
    _assert_triangular_solution(T, b, TRIANGULAR[kind][0](T, b), kind)


@pytest.mark.parametrize("kind", ["lower", "upper"])
def test_triangular_zero_rhs(kind):
    T = _triangular(7, 3, kind)
    assert_array_equal(TRIANGULAR[kind][0](T, np.zeros(7)), np.zeros(7))


@pytest.mark.parametrize("kind", ["lower", "upper"])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_triangular_memory_layouts(kind, layout):
    T, b = _triangular(12, 54, kind), _random(12, 55)
    x = _raw(f"solvers.{kind}_solve", f"{kind}_solve")(
        _layout(T, layout), _layout(b, layout)
    )
    _assert_triangular_solution(T, b, x, kind)


def _factor_and_solve(method, A, rhs):
    if method == "lup":
        L, U, p = lup_factor(A)
        return [lup_solve(L, U, p, b) for b in rhs]
    if method == "ldlt":
        L, d = ldlt_factor(A)
        return [ldlt_solve(L, d, b) for b in rhs]
    F, tau = qr_factor(A)
    return [qr_solve(F, tau, b) for b in rhs]


def _system(method, n, seed):
    return _spd(n, seed) if method == "ldlt" else _random((n, n), seed)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n", [1, 2, 5, 20, 64])
def test_solve_random_systems(method, n):
    A, x_true = _system(method, n, 30 + n), _random(n, 60 + n)
    b = A @ x_true
    [x] = _factor_and_solve(method, A, [b])
    _assert_solution(A, b, x, x_true)


@pytest.mark.parametrize("n", [3, 30])
def test_lup_solve_with_library_factors(n):
    A, b = _random((n, n), 70 + n), _random(n, 71)
    L, U, p = _lup_reference(A)
    x = lup_solve(L, U, p, b)
    _assert_solution(A, b, x, np.linalg.solve(A, b))


@pytest.mark.parametrize("n,seed", [(3, 3), (9, 4)])
def test_ldlt_solve_with_planted_factors(n, seed):
    A, L, d = _planted_ldlt(n, seed)
    x_true = _random(n, seed)
    b = A @ x_true
    _assert_solution(A, b, ldlt_solve(L, d, b), x_true)


@pytest.mark.parametrize("n", [3, 30])
def test_qr_solve_with_library_factors(n):
    A, b = _random((n, n), 80 + n), _random(n, 81)
    F, tau = _qr_reference(A)
    _assert_solution(A, b, qr_solve(F, tau, b), np.linalg.solve(A, b))


@pytest.mark.parametrize("method", METHODS)
def test_factors_are_reused_for_many_right_hand_sides(method):
    A = _system(method, 12, 90)
    X = _random((8, 12), 91)
    rhs = [A @ x for x in X]
    for x_true, b, x in zip(X, rhs, _factor_and_solve(method, A, rhs)):
        _assert_solution(A, b, x, x_true)


@pytest.mark.parametrize("method", METHODS)
def test_identity_system_is_solved_exactly(method):
    b = np.array([0.1, -1 / 3, np.pi, 1e-300, -2.5e300, 7.0])
    [x] = _factor_and_solve(method, np.eye(6), [b])
    assert_array_equal(x, b)


@pytest.mark.parametrize("method", METHODS)
def test_zero_rhs(method):
    [x] = _factor_and_solve(method, _system(method, 6, 92), [np.zeros(6)])
    assert_array_equal(x, np.zeros(6))


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("k", [0, 20, 40])
@pytest.mark.parametrize("direction", ["none", "u", "v"])
def test_perturbation_direction_family(method, k, direction):
    delta, eta = 2.0**-k, 2.0**-30
    A = 0.5 * np.array([[1 + delta, 1 - delta], [1 - delta, 1 + delta]])
    u, v = np.array([1.0, 1.0]), np.array([1.0, -1.0])
    b, x_true = {
        "none": (u, u),
        "u": (u + eta * u, (1 + eta) * u),
        "v": (u + eta * v, u + eta / delta * v),
    }[direction]
    [x] = _factor_and_solve(method, A, [b])
    _assert_solution(A, b, x, x_true)


SOLVER = SRC / "solvers" / "slau_solver.py"
PROMPT_A = "Введите матрицу A:"
PROMPT_B = "Введите вектор b:"
EXAMPLE = "2 1\n1 3\n\n4\n7\n"


def _system_text(A, b, final_newline=True):
    rows = "\n".join(
        " ".join(repr(float(value)) for value in row) for row in A
    )
    text = rows + "\n\n" + "\n".join(repr(float(value)) for value in b)
    return text + "\n" if final_newline else text


def _solver_env():
    path = os.pathsep.join(
        filter(None, [str(SRC), os.environ.get("PYTHONPATH")])
    )
    return dict(
        os.environ,
        PYTHONPATH=path,
        PYTHONIOENCODING="utf-8",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )


def _run_solver(args, cwd, stdin=""):
    result = subprocess.run(
        [sys.executable, str(SOLVER), *args],
        cwd=cwd,
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        env=_solver_env(),
        check=False,
    )
    assert (
        result.returncode == 0
    ), f"slau_solver.py {' '.join(args)} exited with {result.returncode}\n{result.stderr}"
    return result.stdout


def _numbers(text, n, what):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    assert (
        len(lines) == n
    ), f"{what}: expected {n} lines, got {len(lines)}:\n{text}"
    try:
        return np.array([float(line) for line in lines])
    except ValueError:
        pytest.fail(f"{what}: expected one number per line:\n{text}")


def _parse_console(stdout, n):
    first, second = stdout.find(PROMPT_A), stdout.find(PROMPT_B)
    assert first >= 0, f"missing prompt {PROMPT_A!r}:\n{stdout}"
    assert (
        second > first
    ), f"missing prompt {PROMPT_B!r} after {PROMPT_A!r}:\n{stdout}"
    assert not stdout[
        :first
    ].strip(), f"unexpected output before {PROMPT_A!r}:\n{stdout}"
    rest = stdout[first:].replace(PROMPT_A, "\n").replace(PROMPT_B, "\n")
    return _numbers(rest, n, "output")


def _assert_file_mode_output(stdout, A, b):
    assert (
        PROMPT_A not in stdout and PROMPT_B not in stdout
    ), "no prompts in file mode"
    x = _numbers(stdout, len(b), "output")
    _assert_solution(A, b, x, np.linalg.solve(A, b))
    return x


@pytest.mark.cli
@pytest.mark.parametrize("method", [None, "lup", "ldlt", "qr"])
def test_cli_example_file(tmp_path, method):
    (tmp_path / "input.txt").write_text(EXAMPLE, encoding="utf-8")
    stdout = _run_solver(
        ["input.txt"] + ([method] if method else []), tmp_path
    )
    x = _assert_file_mode_output(
        stdout, np.array([[2.0, 1.0], [1.0, 3.0]]), np.array([4.0, 7.0])
    )
    assert_allclose(x, [1.0, 2.0], rtol=0, atol=1e-15)


@pytest.mark.cli
@pytest.mark.parametrize("args", [[], ["lup"], ["ldlt"], ["qr"]])
def test_cli_console_input(tmp_path, args):
    x = _parse_console(_run_solver(args, tmp_path, stdin=EXAMPLE), 2)
    assert_allclose(x, [1.0, 2.0], rtol=0, atol=1e-15)


@pytest.mark.cli
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n", [3, 12])
def test_cli_random_system_from_file(tmp_path, method, n):
    A, b = _system(method, n, 100 + n), _random(n, 101)
    (tmp_path / "system.txt").write_text(_system_text(A, b), encoding="utf-8")
    _assert_file_mode_output(
        _run_solver(["system.txt", method], tmp_path), A, b
    )


@pytest.mark.cli
@pytest.mark.parametrize("method", METHODS)
def test_cli_random_system_from_console(tmp_path, method):
    A, b = _system(method, 5, 102), _random(5, 103)
    x = _parse_console(
        _run_solver([method], tmp_path, stdin=_system_text(A, b)), 5
    )
    _assert_solution(A, b, x, np.linalg.solve(A, b))


@pytest.mark.cli
def test_cli_number_formats(tmp_path):
    text = "4e0 -1.5E+0 0\n-1 2.5e-1 1E1\n0.5 0 -3\n\n1e-3\n-2E2\n7\n"
    (tmp_path / "formats.txt").write_text(text, encoding="utf-8")
    A = np.array([[4.0, -1.5, 0.0], [-1.0, 0.25, 10.0], [0.5, 0.0, -3.0]])
    _assert_file_mode_output(
        _run_solver(["formats.txt"], tmp_path),
        A,
        np.array([1e-3, -200.0, 7.0]),
    )


@pytest.mark.cli
def test_cli_file_without_final_newline(tmp_path):
    A, b = _random((4, 4), 104), _random(4, 105)
    (tmp_path / "system.txt").write_text(
        _system_text(A, b, final_newline=False), encoding="utf-8"
    )
    _assert_file_mode_output(_run_solver(["system.txt", "qr"], tmp_path), A, b)


@pytest.mark.cli
def test_cli_relative_path_with_spaces(tmp_path):
    folder = tmp_path / "data dir"
    folder.mkdir()
    A, b = _spd(4, 106), _random(4, 107)
    (folder / "my system.txt").write_text(_system_text(A, b), encoding="utf-8")
    _assert_file_mode_output(
        _run_solver(["data dir/my system.txt", "ldlt"], tmp_path), A, b
    )


@pytest.mark.cli
@pytest.mark.parametrize("method", METHODS)
def test_cli_prints_full_precision(tmp_path, method):
    b = np.array([0.1, -1 / 3, np.pi, 1e-300, -2.5e300, 123456789.12345679])
    (tmp_path / "system.txt").write_text(
        _system_text(np.eye(6), b), encoding="utf-8"
    )
    x = _numbers(_run_solver(["system.txt", method], tmp_path), 6, "output")
    assert_array_equal(
        x, b, err_msg="x must be printed with full float64 precision"
    )


SOLVER_FUNCTIONS = {
    "lup": [("decomps.lup", "lup_factor"), ("solvers.lup_solve", "lup_solve")],
    "ldlt": [
        ("decomps.ldlt", "ldlt_factor"),
        ("solvers.ldlt_solve", "ldlt_solve"),
    ],
    "qr": [("decomps.qr", "qr_factor"), ("solvers.qr_solve", "qr_solve")],
}


def _run_solver_in_process(args, stdin, monkeypatch, capfd):
    """Run slau_solver.py in-process and record factorization and solver calls."""
    import io
    import runpy

    calls = []
    for functions in SOLVER_FUNCTIONS.values():
        for module_name, name in functions:
            module = importlib.import_module(module_name)
            original = getattr(module, name)

            def recorder(*a, _original=original, _name=name, **k):
                calls.append(_name)
                return _original(*a, **k)

            for loaded in list(sys.modules.values()):
                filename = getattr(loaded, "__file__", None)
                if filename and Path(filename).resolve().is_relative_to(
                    SRC.resolve()
                ):
                    for attribute, value in list(vars(loaded).items()):
                        if value is original:
                            monkeypatch.setattr(loaded, attribute, recorder)
    monkeypatch.setattr(sys, "argv", [str(SOLVER), *args])
    monkeypatch.setattr(
        sys,
        "stdin",
        io.TextIOWrapper(io.BytesIO(stdin.encode("utf-8")), encoding="utf-8"),
    )
    capfd.readouterr()
    try:
        runpy.run_path(str(SOLVER), run_name="__main__")
    except SystemExit as stop:
        assert stop.code in (
            None,
            0,
        ), f"slau_solver.py exited with {stop.code}"
    return capfd.readouterr().out, calls


@pytest.mark.cli
@pytest.mark.parametrize(
    "args,source,method",
    [
        pytest.param(["system.txt"], "file", "lup", id="file-default"),
        pytest.param(["system.txt", "lup"], "file", "lup", id="file-lup"),
        pytest.param(["system.txt", "ldlt"], "file", "ldlt", id="file-ldlt"),
        pytest.param(["system.txt", "qr"], "file", "qr", id="file-qr"),
        pytest.param([], "console", "lup", id="console-default"),
        pytest.param(["ldlt"], "console", "ldlt", id="console-ldlt"),
        pytest.param(["qr"], "console", "qr", id="console-qr"),
    ],
)
def test_cli_uses_factor_and_solver_of_chosen_method(
    tmp_path, monkeypatch, capfd, args, source, method
):
    A, b = _spd(4, 108), _random(4, 109)
    (tmp_path / "system.txt").write_text(_system_text(A, b), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    stdout, calls = _run_solver_in_process(
        args,
        _system_text(A, b) if source == "console" else "",
        monkeypatch,
        capfd,
    )
    solver = SOLVER_FUNCTIONS[method][1][1]
    others = {
        name
        for key, functions in SOLVER_FUNCTIONS.items()
        if key != method
        for _, name in functions
    }
    assert (
        solver in calls
    ), f"method {method} must call {solver}; called: {calls}"
    assert not others & set(
        calls
    ), f"method {method} called other methods: {calls}"
    x = (
        _parse_console(stdout, 4)
        if source == "console"
        else _numbers(stdout, 4, "output")
    )
    _assert_solution(A, b, x, np.linalg.solve(A, b))


@pytest.mark.cli
def test_cli_console_does_not_wait_for_end_of_input(tmp_path):
    process = subprocess.Popen(
        [sys.executable, str(SOLVER)],
        cwd=tmp_path,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=_solver_env(),
    )
    try:
        process.stdin.write(EXAMPLE)
        process.stdin.flush()
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        pytest.fail("solver waits for EOF after the last component of b")
    stdout, stderr = process.communicate()
    assert (
        process.returncode == 0
    ), f"slau_solver.py exited with {process.returncode}\n{stderr}"
    assert_allclose(_parse_console(stdout, 2), [1.0, 2.0], rtol=0, atol=1e-15)


def _assert_least_squares(A, b, x, method, x_expected=None):
    n, m = A.shape
    _check_array(x, (m,), "x")
    if x_expected is None:
        x_expected = np.linalg.lstsq(A, b, rcond=None)[0]
    kappa, norm_A, norm_x = (
        np.linalg.cond(A),
        np.linalg.norm(A, 2),
        np.linalg.norm(x_expected),
    )
    data = 1 + np.linalg.norm(b) / (norm_A * norm_x)
    if method == "qr":
        residual = np.linalg.norm(b - A @ x_expected) / (norm_A * norm_x)
        bound = C * (n + m) * UROUND * (kappa * data + kappa**2 * residual)
    else:
        bound = C * (n + m) * UROUND * kappa**2 * data
    error = np.linalg.norm(x - x_expected) / norm_x
    assert (
        error <= bound
    ), f"{method}: relative error {error:.3e} > {bound:.3e}"
    omega = _omega(A, x, b)
    assert (
        omega <= C * (n + m * m) * UROUND
    ), f"{method}: A^T r != 0, omega = {omega:.3e}"


def _ls_problem(n, m, seed, ratio):
    """b = A x + r with r orthogonal to range(A), ||r|| = ratio ||Ax||."""
    rng = np.random.default_rng(seed)
    A, x = rng.normal(size=(n, m)), rng.normal(size=m)
    Q = np.linalg.qr(A, mode="complete")[0]
    r = Q[:, m:] @ rng.normal(size=n - m)
    r *= ratio * np.linalg.norm(A @ x) / np.linalg.norm(r)
    return A, A @ x + r, x, r


@pytest.mark.parametrize("method", LS_METHODS)
@pytest.mark.parametrize(
    "n,m", [(1, 1), (3, 1), (4, 4), (10, 3), (50, 8), (200, 20)]
)
def test_lstsq_random_problems(method, n, m):
    A, b = _random((n, m), n + m), _random(n, n * m)
    _assert_least_squares(A, b, lstsq(A, b, method=method), method)


def test_lstsq_default_method_is_qr():
    A, b = _random((20, 4), 1), _random(20, 2)
    assert_allclose(lstsq(A, b), lstsq(A, b, method="qr"), rtol=1e-15, atol=0)


@pytest.mark.parametrize("method", LS_METHODS)
def test_lstsq_exact_line_fit(method):
    t = np.arange(5.0)
    A = np.column_stack((np.ones(5), t))
    assert_allclose(
        lstsq(A, 2 + 3 * t, method=method), [2.0, 3.0], rtol=0, atol=1e-13
    )


@pytest.mark.parametrize("method", LS_METHODS)
@pytest.mark.parametrize("ratio", [0.0, 1.0, 100.0])
def test_lstsq_known_solution_and_residual(method, ratio):
    A, b, x_true, r = _ls_problem(40, 6, 5, ratio)
    x = lstsq(A, b, method=method)
    _assert_least_squares(A, b, x, method, x_true)
    assert_allclose(b - A @ x, r, rtol=0, atol=1e-10 * np.linalg.norm(b))


@pytest.mark.parametrize("method", LS_METHODS)
def test_lstsq_rhs_orthogonal_to_columns_gives_zero(method):
    A = _random((12, 4), 3)
    b = np.linalg.qr(A, mode="complete")[0][:, 4:] @ _random(8, 4)
    x = _check_array(lstsq(A, b, method=method), (4,), "x")
    bound = (
        C
        * 16
        * UROUND
        * np.linalg.cond(A) ** 2
        * np.linalg.norm(b)
        / np.linalg.norm(A, 2)
    )
    assert (
        np.linalg.norm(x) <= bound
    ), "b orthogonal to range(A) must give x = 0"


@pytest.mark.parametrize("k", [20, 27])
def test_lstsq_qr_accuracy_on_lauchli_matrix(k):
    # Task II: at k = 27 the computed A^T A is singular
    m, eps = 8, 2.0**-k
    A = np.vstack((np.ones((1, m)), eps * np.eye(m)))
    x_true = 1 + np.arange(1, m + 1) / m
    x = _check_array(lstsq(A, A @ x_true), (m,), "x")
    kappa = np.sqrt(m + eps**2) / eps
    error = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
    bound = C * (m + 1) * UROUND * kappa
    assert error <= bound, f"qr: relative error {error:.3e} > {bound:.3e}"


@pytest.mark.parametrize("method", LS_METHODS)
@pytest.mark.parametrize(
    "change", ["permute-rows", "duplicate-rows", "zero-rows"]
)
def test_lstsq_row_operations_keep_solution(method, change):
    A, b = _random((15, 4), 7), _random(15, 8)
    if change == "permute-rows":
        order = np.random.default_rng(9).permutation(15)
        changed = A[order], b[order]
    elif change == "duplicate-rows":
        changed = np.vstack((A, A)), np.concatenate((b, b))
    else:
        changed = np.vstack((A, np.zeros((5, 4)))), np.concatenate(
            (b, np.zeros(5))
        )
    expected = np.linalg.lstsq(A, b, rcond=None)[0]
    _assert_least_squares(
        *changed, lstsq(*changed, method=method), method, expected
    )


@pytest.mark.parametrize("method", LS_METHODS)
def test_lstsq_single_column(method):
    a, b = _random(9, 3), _random(9, 4)
    x = _check_array(lstsq(a[:, None], b, method=method), (1,), "x")
    assert x[0] == pytest.approx((a @ b) / (a @ a), rel=1e-13)


@pytest.mark.parametrize("method", LS_METHODS)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_lstsq_memory_layouts(method, layout):
    A, b = _random((17, 5), 56), _random(17, 57)
    x = _raw("mnk.lstsq", "lstsq")(
        _layout(A, layout), _layout(b, layout), method=method
    )
    _assert_least_squares(A, b, x, method)


def _tikhonov_data(n, m, q, seed):
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(n, m)),
        rng.normal(size=n),
        rng.normal(size=(q, m)),
        rng.normal(size=m),
    )


def _regularized_reference(A, b, Gamma, x0, lam):
    """Solve (A^T A + lam^2 G^T G) x = A^T b + lam^2 G^T G x0."""
    GtG = Gamma.T @ Gamma
    M = A.T @ A + lam**2 * GtG
    return np.linalg.solve(M, A.T @ b + lam**2 * GtG @ x0), np.linalg.cond(M)


@pytest.mark.parametrize(
    "n,m,q", [(5, 3, 3), (4, 6, 6), (8, 5, 4), (3, 3, 1), (10, 4, 7)]
)
@pytest.mark.parametrize("lam", [0.5, 3.0])
def test_tikhonov_builds_stacked_problem(n, m, q, lam):
    A, b, Gamma, x0 = _tikhonov_data(n, m, q, 100 * n + 10 * m + q)
    result = tikhonov(A, b, Gamma, x0, lam)
    assert len(result) == 2, "tikhonov must return (A_reg, b_reg)"
    A_reg = _check_array(result[0], (n + q, m), "A_reg")
    b_reg = _check_array(result[1], (n + q,), "b_reg")
    assert_array_equal(A_reg[:n], A)
    assert_array_equal(b_reg[:n], b)
    assert_allclose(A_reg[n:], lam * Gamma, rtol=4 * UROUND, atol=0)
    scale = lam * np.max(np.abs(Gamma) @ np.abs(x0))
    assert_allclose(
        b_reg[n:], lam * (Gamma @ x0), rtol=0, atol=C * m * UROUND * scale
    )


@pytest.mark.parametrize("seed", [1, 2])
def test_tikhonov_objective_identity(seed):
    A, b, Gamma, x0 = _tikhonov_data(9, 4, 3, seed)
    lam = 1.7
    A_reg, b_reg = tikhonov(A, b, Gamma, x0, lam)
    for x in _random((5, 4), seed + 10):
        left = np.linalg.norm(A_reg @ x - b_reg) ** 2
        right = (
            np.linalg.norm(A @ x - b) ** 2
            + lam**2 * np.linalg.norm(Gamma @ (x - x0)) ** 2
        )
        assert left == pytest.approx(
            right, rel=1e-12
        ), "||A_reg x - b_reg||^2 = ||Ax - b||^2 + lam^2 ||G(x - x0)||^2"


@pytest.mark.parametrize("method", LS_METHODS)
def test_tikhonov_solution_matches_normal_equations(method):
    A, b, Gamma, x0 = _tikhonov_data(20, 6, 5, 3)
    lam = 0.7
    expected, kappa = _regularized_reference(A, b, Gamma, x0, lam)
    x = _check_array(
        lstsq(*tikhonov(A, b, Gamma, x0, lam), method=method), (6,), "x"
    )
    assert_allclose(
        x,
        expected,
        rtol=0,
        atol=C * 30 * UROUND * kappa * np.linalg.norm(expected),
    )


@pytest.mark.parametrize("method", LS_METHODS)
def test_tikhonov_identity_penalty_with_dependent_columns(method):
    B = _random((10, 3), 1)
    A = np.column_stack((B, B[:, 0] - 2 * B[:, 2]))
    b, x0, lam = _random(10, 2), _random(4, 3), 0.3
    expected, kappa = _regularized_reference(A, b, np.eye(4), x0, lam)
    x = _check_array(
        lstsq(*tikhonov(A, b, np.eye(4), x0, lam), method=method), (4,), "x"
    )
    assert_allclose(
        x,
        expected,
        rtol=0,
        atol=C * 30 * UROUND * kappa * np.linalg.norm(expected),
    )


@pytest.mark.parametrize("method", LS_METHODS)
def test_tikhonov_first_differences_on_blur(method):
    m = 12
    t = np.linspace(0.0, 1.0, m)
    K = np.exp(-((t[:, None] - t[None, :]) ** 2) / (2 * 0.1**2))
    A = K / K.sum(axis=1, keepdims=True)
    D = np.diff(np.eye(m), axis=0)
    b, lam = np.sin(3 * t) + 0.5, 0.1
    expected, kappa = _regularized_reference(A, b, D, np.zeros(m), lam)
    A_reg, b_reg = tikhonov(A, b, D, np.zeros(m), lam)
    assert A_reg.shape == (2 * m - 1, m) and b_reg.shape == (2 * m - 1,)
    x = _check_array(lstsq(A_reg, b_reg, method=method), (m,), "x")
    assert_allclose(
        x,
        expected,
        rtol=0,
        atol=C * 3 * m * UROUND * kappa * np.linalg.norm(expected),
    )


def test_tikhonov_large_lambda_returns_reference_vector():
    A, b, _, x0 = _tikhonov_data(15, 5, 5, 4)
    lam = 1e6
    x = lstsq(*tikhonov(A, b, np.eye(5), x0, lam))
    bound = 2 * np.linalg.norm(A.T @ (A @ x0 - b)) / lam**2
    assert np.linalg.norm(x - x0) <= bound, "x must tend to x0 for large lam"


IMPLEMENTATION_FILES = [
    "decomps/lup.py",
    "decomps/ldlt.py",
    "decomps/qr.py",
    "solvers/lower_solve.py",
    "solvers/upper_solve.py",
    "solvers/lup_solve.py",
    "solvers/ldlt_solve.py",
    "solvers/qr_solve.py",
    "solvers/slau_solver.py",
    "mnk/lstsq.py",
    "mnk/tikhonov.py",
]
NUMPY_LINALG_ALLOWED = {
    "norm",
    "vector_norm",
    "matrix_norm",
    "matmul",
    "vecdot",
    "outer",
    "matrix_transpose",
    "diagonal",
    "trace",
    "tensordot",
    "multi_dot",
    "cross",
    "vecmat",
    "matvec",
    "LinAlgError",
}
FORBIDDEN_MODULES = ("scipy.linalg", "scipy.sparse.linalg")
FORBIDDEN_NAMES = {"numpy.matrix", "numpy.asmatrix"}


def _is_forbidden(name):
    parts = name.split(".")
    return (
        name in FORBIDDEN_NAMES
        or any(name.startswith(module + ".") for module in FORBIDDEN_MODULES)
        or (
            parts[:2] == ["numpy", "linalg"]
            and len(parts) > 2
            and parts[2] not in NUMPY_LINALG_ALLOWED
        )
    )


def _star_import_names(module):
    """Names that `from module import *` brings into scope."""
    if module == "numpy":
        return {
            "linalg": "numpy.linalg",
            "matrix": "numpy.matrix",
            "asmatrix": "numpy.asmatrix",
        }
    if module == "scipy":
        return {"linalg": "scipy.linalg", "sparse": "scipy.sparse"}
    if module in ("numpy.linalg", *FORBIDDEN_MODULES):
        names = getattr(importlib.import_module(module), "__all__", [])
        return {name: f"{module}.{name}" for name in names}
    return {}


def _forbidden_references(source):
    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.aliases = {}
            self.bad = set()

        def resolve(self, node):
            if isinstance(node, ast.Name):
                return self.aliases.get(node.id, "")
            if isinstance(node, ast.Attribute):
                parent = self.resolve(node.value)
                return f"{parent}.{node.attr}" if parent else ""
            return ""

        def check(self, name):
            if name and _is_forbidden(name):
                self.bad.add(name)

        def visit_Import(self, node):
            for item in node.names:
                if item.asname:
                    self.aliases[item.asname] = item.name
                else:
                    top = item.name.split(".")[0]
                    self.aliases[top] = top
                self.check(item.name)

        def visit_ImportFrom(self, node):
            module = node.module or ""
            if node.level:
                return
            for item in node.names:
                if item.name == "*":
                    self.aliases.update(_star_import_names(module))
                else:
                    full = f"{module}.{item.name}"
                    self.aliases[item.asname or item.name] = full
                    self.check(full)

        def visit_Attribute(self, node):
            self.check(self.resolve(node))
            self.generic_visit(node)

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load):
                self.check(self.resolve(node))

        def visit_Assign(self, node):
            self.visit(node.value)
            full = self.resolve(node.value)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.aliases[target.id] = full

        def visit_FunctionDef(self, node):
            outer = self.aliases.copy()
            args = (
                node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            )
            args += [arg for arg in (node.args.vararg, node.args.kwarg) if arg]
            for arg in args:
                self.aliases[arg.arg] = ""
            self.generic_visit(node)
            self.aliases = outer

        visit_AsyncFunctionDef = visit_FunctionDef

    visitor = Visitor()
    visitor.visit(ast.parse(source))
    return sorted(visitor.bad)


@pytest.mark.algorithm
@pytest.mark.parametrize("filename", IMPLEMENTATION_FILES)
def test_sources_use_permitted_tools(filename):
    path = SRC / filename
    assert path.exists(), f"src/{filename} not found"
    bad = _forbidden_references(path.read_text(encoding="utf-8"))
    assert (
        not bad
    ), f"src/{filename}: library factorizations and solvers are not allowed: {bad}"


@pytest.mark.algorithm
@pytest.mark.parametrize(
    "source",
    [
        "import numpy as np; np.linalg.solve(A, b)",
        "import numpy as np; np.linalg.inv(A)",
        "import numpy.linalg as la; la.cholesky(A)",
        "from numpy.linalg import qr as factor; factor(A)",
        "from scipy.linalg import lu_factor",
        "from scipy import linalg; linalg.solve_triangular(L, b)",
        "import scipy.linalg as sla; sla.ldl(A)",
        "import numpy as np; la = np.linalg; la.lstsq(A, b)",
        "from numpy.linalg import *; solve(A, b)",
        "from numpy import *; linalg.pinv(A)",
        "import numpy as np; np.linalg.matrix_power(A, -1)",
        "import scipy.sparse.linalg as spla; spla.spsolve(A, b)",
        "import numpy as np; np.matrix(A).I",
        "from scipy.linalg.lapack import dgetrf",
        "import numpy as np; np.linalg.det(A)",
    ],
)
def test_checker_detects_disallowed_operations(source):
    assert _forbidden_references(source)


@pytest.mark.algorithm
@pytest.mark.parametrize(
    "source",
    [
        "import numpy as np; np.linalg.norm(x); x @ y; np.outer(v, w)",
        "import numpy as np; np.dot(a, b); np.tril(A, -1); np.triu(A); np.eye(3)",
        "from numpy.linalg import norm; norm(x)",
        "import numpy as np; raise np.linalg.LinAlgError('singular')",
        "from solvers.lower_solve import lower_solve; lower_solve(L, b)",
        "def solve(A, b): return b\nsolve(A, b)",
        "# np.linalg.solve(A, b)\ntext = 'np.linalg.inv'",
        "import numpy as np\ndef helper(np): return np.linalg.solve()",
        "import sys\nimport numpy as np\nnp.loadtxt(sys.argv[1])",
        "from numpy import *; tril(A)",
    ],
)
def test_checker_accepts_permitted_code(source):
    assert _forbidden_references(source) == []


@pytest.mark.stress
def test_stress_lup():
    _assert_lup(_random((256, 256), 2026))


@pytest.mark.stress
def test_stress_ldlt():
    A = _spd(256, 2026)
    _assert_ldlt(A, *_ldlt_from_cholesky(A))


@pytest.mark.stress
def test_stress_qr_tall():
    _assert_qr(_random((2000, 40), 2026))


@pytest.mark.stress
@pytest.mark.parametrize("kind", ["lower", "upper"])
def test_stress_triangular(kind):
    T, b = _triangular(1500, 2026, kind), _random(1500, 2027)
    _assert_triangular_solution(T, b, TRIANGULAR[kind][0](T, b), kind)


@pytest.mark.stress
@pytest.mark.parametrize("method", METHODS)
def test_stress_solve(method):
    A, x_true = _system(method, 300, 2026), _random(300, 2027)
    b = A @ x_true
    [x] = _factor_and_solve(method, A, [b])
    _assert_solution(A, b, x, x_true)


@pytest.mark.stress
@pytest.mark.parametrize("method", LS_METHODS)
def test_stress_lstsq_tall(method):
    A, b = _random((5000, 30), 2026), _random(5000, 2027)
    _assert_least_squares(A, b, lstsq(A, b, method=method), method)
