"""Ridge regression for tuning-curve-to-weight inference.

Solves the regularized least-squares problem ``X ≈ J @ Phi`` for the weight
matrix ``J``, with three options for handling the ``N_theta`` vs ``N``
asymmetry: ``primal``, ``dual``, and ``leave_one_out``.
"""
import numpy as np


def compute_J(X, Phi, lamda, method='opt'):
    """Solve ridge regression ``X ≈ J @ Phi`` with diagonal subtraction.

    Removes the autapse (self-coupling) by subtracting a diagonal correction
    from the unconstrained solution ``J_uc``; the returned ``J`` has zero
    diagonal in the relevant inner product.

    Parameters
    ----------
    X : ndarray, shape (N_theta, N)
        Target activity (e.g. preactivations or trajectories).
    Phi : ndarray, shape (N_theta, N)
        Predictor activity (e.g. tuning curves).
    lamda : float
        Ridge regularization strength, in the ``N``-normalized convention
        described below.
    method : {'opt', 'primal', 'dual', 'leave_one_out'}, default 'opt'
        Solver variant:

        - ``'opt'``: auto-select ``'dual'`` if ``N_theta < N``, else ``'primal'``.
        - ``'primal'``: invert ``Phi.T @ Phi / N_theta + (N * lamda) * I`` (``N x N``).
          Cheaper when ``N <= N_theta``.
        - ``'dual'``: invert ``Phi @ Phi.T / N + (lamda * N_theta) * I``
          (``N_theta x N_theta``). Cheaper when ``N_theta < N``.
        - ``'leave_one_out'``: for each neuron ``i``, solve the regression
          excluding column ``i`` of ``Phi`` (leave-one-out cross-validation
          style). Slow (O(N) regressions); ``J_uc`` is not returned. Its ridge
          coefficient uses ``N - 1``, not ``N``, because that regression has
          ``N - 1`` predictors.

    Notes
    -----
    The data term of the loss carries a ``1 / N`` alongside the ``1 / N_theta``,

    .. math::

        L(J) = \\frac{1}{N} \\frac{1}{2\\pi} \\int d\\theta \\sum_i
               \\Big( \\sum_j J_{ij} \\phi_j(\\theta) - x_i(\\theta) \\Big)^2
               + \\lambda \\sum_{ij} J_{ij}^2 ,

    so that ``lamda`` has a well-defined ``N -> infinity`` limit. Without the
    ``1 / N`` the data term grows with ``N`` while the penalty does not, the
    effective regularization is ``lamda / N``, and it vanishes at fixed
    ``lamda``. The dual kernel is then ``Phi @ Phi.T / N + N_theta * lamda * I``,
    with eigenvalues ``N * N_theta * (Ghat_phi_n + lamda)``, free of both ``N``
    and ``N_theta``.

    This convention was adopted on 2026-07-31 and is a change of normalization,
    not a correction. To convert a value from the earlier convention, in which
    the data term had no ``1 / N``, use ``lamda_new = lamda_old / N``.

    Returns
    -------
    J : ndarray, shape (N, N)
        Weight matrix with the diagonal correction applied.
    J_uc : ndarray or None, shape (N, N)
        Unconstrained ridge solution before diagonal subtraction. ``None``
        when ``method='leave_one_out'``.
    """
    N_theta, N = Phi.shape
    if method == 'opt':
        method = 'dual' if N_theta < N else 'primal'

    if method == 'dual':
        K = (Phi @ Phi.T) / N + (lamda * N_theta) * np.eye(N_theta)
        K_inv = np.linalg.inv(K)
        A = np.eye(N) - (1 / N) * (Phi.T @ K_inv @ Phi)
        J_uc = (1 / N) * (X.T @ K_inv @ Phi)
        J = J_uc - A * (np.diag(J_uc) / np.diag(A))[:, None]
        return J, J_uc

    elif method == 'primal':
        M = (Phi.T @ Phi) / N_theta + np.eye(N) * (N * lamda)
        M_inv = np.linalg.inv(M)
        J_uc = ((X.T @ Phi) / N_theta) @ M_inv
        J = J_uc - M_inv * (np.diag(J_uc) / np.diag(M_inv))[:, None]
        return J, J_uc

    elif method == 'leave_one_out':
        J = np.zeros((N, N))
        for i in range(N):
            mask = np.ones(N, dtype=bool)
            mask[i] = False
            Phi_excl = Phi[:, mask]
            X_i = X[:, i]
            M_excl = (Phi_excl.T @ Phi_excl) / N_theta + np.eye(N - 1) * ((N - 1) * lamda)
            J_i = np.linalg.inv(M_excl) @ ((1 / N_theta) * (Phi_excl.T @ X_i))
            J[mask, i] = J_i
        return J, None

    else:
        raise ValueError(f"method must be one of 'opt', 'primal', 'dual', 'leave_one_out'; got {method!r}")
