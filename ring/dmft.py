"""Dynamical mean-field theory (DMFT) for the ring-attractor model.

Single class ``DMFTModel`` constructs the wraparound Gaussian covariance,
derives the analytical recurrent weights J and the deviation operator U,
and simulates the DMFT equations (Euler integration of the order parameter
``m`` and its conjugate ``m_hat``). Used by ``notebooks/dmft.ipynb`` and
``notebooks/weight_matrix_analyses.ipynb``.

Public API on a ``DMFTModel`` instance:
    ``F``, ``J``                                — analytical functions / weights
    ``simulate``                                — Euler integration
    ``compute_bump_jacobian``                   — Jacobian at the bump fixed point
    ``compute_homogeneous_jacobian``            — Jacobian at the homogeneous solution

Other attributes (``Gamma_*``, ``U``, ``dF_*``, ``compute_*``,
``rfft_sum``, ``J_m_prod`` etc.) are internal implementation detail.
"""
import numpy as np
from tqdm import tqdm
from scipy.linalg import circulant
from . import fn, tc


class DMFTModel:
    """Analytical + simulation toolkit for the ring-attractor DMFT.

    On construction, derives the wraparound Gaussian covariance ``Gamma_x``
    (length scale ``scale``, ``N_theta`` angular points), the nonlinear
    covariance map ``F``, the analytical recurrent weights ``J`` (from
    ``Gamma_x / Gamma_phi`` in Fourier space), and the deviation operator
    ``U``. After construction the instance exposes ``F``, ``J``, and the
    ``simulate`` / ``compute_*_jacobian`` methods listed in the module
    docstring.

    Parameters
    ----------
    N_theta : int, default 1000
        Number of angular grid points on the ring.
    scale : float, default 1.5
        Wrapped-Gaussian length scale (in units of Fourier-frequency index).
    g : float, default 2.0
        Gain parameter scaling the nonlinearity.
    lamda : float, default 1e-12
        Regularization entering the mean-field kernels as
        ``Jhat_n = Ghat_x_n / (Ghat_phi_n + lamda)`` and
        ``Uhat_n = Ghat_x_n / (Ghat_phi_n + lamda)^2``.

        It has the same PLACE in the formula as the ``lamda`` of
        ``regression.compute_J``, but not the same JOB, which is why the
        default differs. Here the inversion is exactly singular, because this
        is the ``N -> infinity`` limit and ``Ghat_phi_n -> 0`` as ``n -> inf``,
        so the constant is a numerical guard against dividing by zero and
        should be as small as the arithmetic allows. In the finite-``N``
        inference ``lamda`` is the actual ridge of the loss and does real work
        against the sampling structure of the empirical ``(1/N) Phi Phi^T``,
        which is what ``1e-8`` there is sized for.

        Keeping this at ``1e-12`` matters for the stability sweep, since at
        low gain ``Ghat_phi_n`` becomes comparable to ``1e-8`` and the guard
        would start suppressing ``J`` for real, moving the phase boundary and
        leaving cells where the homogeneous solution has no instability
        timescale at all. Pass the finite-``N`` value explicitly if you are
        comparing this theory against a specific fit rather than mapping the
        model.
    """

    def __init__(self, N_theta=1000, scale=1.5, g=2.0, lamda=1e-12):
        self.N_theta = N_theta
        self.scale = scale
        self.g = g
        self.lamda = lamda
        self.theta = np.arange(N_theta) * 2 * np.pi / N_theta
        self.dtheta = self.theta[1] - self.theta[0]

        self.nonlin = lambda x: 1 + fn.unity_erf(self.g * x)
        self.compute_F_function()
        self.compute_Gamma()
        self.compute_J_and_U()

    def compute_F_function(self):
        """(internal) Derive the nonlinear covariance map ``F`` and its derivatives.

        Sets three instance attributes:

        - ``self.F(C_11, C_22, C_12)`` — the 2-point Gaussian average of the
          erf-based nonlinearity for joint variances/covariance ``C_*``.
        - ``self.dF_dC12``, ``self.dF_dC11`` — partial derivatives.
        """
        g = self.g

        def F(C_11, C_22, C_12):
            in_sqrt = ((self.g**2) * C_11 + 2/np.pi) * ((self.g**2) * C_22 + 2/np.pi) - ((self.g**2) * C_12)**2
            in_sqrt = np.clip(in_sqrt, a_min=1e-20, a_max=np.inf)
            t_12 = (2/np.pi) * np.arctan((self.g**2) * C_12 / np.sqrt(in_sqrt))
            return 1 + t_12

        def dF_dC12(C_11, C_22, C_12):
            A = ((g**2)*C_11 + (2/np.pi)) * ((g**2)*C_22 + (2/np.pi))
            B = A - ((g**2)*C_12)**2
            return (g**2) * (2/np.pi) * 1./np.sqrt(B)

        def dF_dC11(C_11, C_22, C_12):
            A = ((g**2)*C_11 + (2/np.pi)) * ((g**2)*C_22 + (2/np.pi))
            B = A - ((g**2)*C_12)**2
            dF_dA = -(2/np.pi) * ((g**2)*C_12 / (2*A)) * 1./np.sqrt(B)
            dA_dC11 = (g**2) * ((g**2)*C_22 + (2/np.pi))
            return dF_dA * dA_dC11

        self.F = F
        self.dF_dC12 = dF_dC12
        self.dF_dC11 = dF_dC11

    def compute_Gamma(self):
        """(internal) Build ``Gamma_x`` (input covariance) and ``Gamma_phi`` (output)."""
        self.Gamma_x, self.Gamma_x_ft = tc.wrapped_gaussian_cov(
            eq_var=1., scale=self.scale, N_theta=self.N_theta)
        self.Gamma_phi = self.F(self.Gamma_x[0], self.Gamma_x[0], self.Gamma_x)
        self.Gamma_phi_ft = np.fft.rfft(self.Gamma_phi, norm='forward').real

    def compute_J_and_U(self):
        """(internal) Derive ``J = Gamma_x / (Gamma_phi + lamda)`` and ``U = Gamma_x / (Gamma_phi + lamda)^2`` in Fourier space.

        The ridge ``self.lamda`` is added to the Fourier-space ``Gamma_phi``,
        which is what regularizes the inversion at high ``n`` where
        ``Gamma_phi_ft`` decays toward zero.
        """
        self.J_ft = self.Gamma_x_ft / (self.Gamma_phi_ft + self.lamda)
        self.J = np.fft.irfft(self.J_ft, norm='forward')
        self.U_ft = self.Gamma_x_ft / (self.Gamma_phi_ft + self.lamda)**2
        self.U = np.fft.irfft(self.U_ft, norm='forward')

    def rfft_sum(self, x):
        """(internal) Reduce real Fourier coefficients ``x`` to the implied real-space sum."""
        return x[0].real + 2 * x[1:].real.sum()

    def J_m_prod(self, m):
        """(internal) Convolve ``m`` with ``J`` via FFT."""
        m_ft = np.fft.rfft(m, norm='forward')
        return np.fft.irfft(self.J_ft * m_ft, norm='forward')

    def J_deriv_m_hat_prod(self, m_hat):
        """(internal) Convolve ``m_hat`` with the derivative of ``J`` via FFT."""
        N_freq = len(self.J_ft)
        n = np.arange(N_freq)
        J_deriv_ft = 1j * n * self.J_ft
        m_hat_ft = np.fft.rfft(m_hat, norm='forward')
        return np.fft.irfft(J_deriv_ft * m_hat_ft, norm='forward')

    def compute_Q(self, m, m_hat):
        """(internal) Compute the DMFT Q functional from ``m``, ``m_hat``.

        Notes
        -----
        Both the mean field and its variance are driven by the same single
        source ``s = m - d(m_hat)/dtheta``, whose Fourier coefficients are
        ``s_n = m_n - i n mhat_n``. The mean field is ``J`` convolved with
        ``s``, since ``J_m_prod(m) - J_deriv_m_hat_prod(m_hat)`` equals
        ``irfft(J_ft * s_ft)``, and ``Q`` is the same source measured with
        ``U`` instead of ``J``,

        ``Q = sum_n U_n |m_n - i n mhat_n|^2``,

        which is what ``t1 + t2 + t3`` below expands to. That expansion fixes
        the conjugation in ``t3``: ``m_hat_ft.conj() * m_ft`` gives the cross
        term of ``|m_n - i n mhat_n|^2``, whereas conjugating the other factor
        would flip its sign and give ``|m_n + i n mhat_n|^2``, a source
        inconsistent with the one the mean field uses. Both choices are
        positive, so positivity of the variance does not settle it and
        consistency with ``H`` does. Verified numerically against the closed
        form, 2026-07-31. The two agree only when ``m_hat = 0``, that is at
        zero drift, so nothing computed at ``omega = 0`` was ever sensitive
        to this.
        """
        m_ft = np.fft.rfft(m, norm='forward')
        m_hat_ft = np.fft.rfft(m_hat, norm='forward')
        N_freq = len(self.U_ft)
        n = np.arange(N_freq)
        U_deriv_ft = 1j * n * self.U_ft
        U_double_deriv_ft = -n**2 * self.U_ft
        t1 = self.rfft_sum(self.U_ft * np.abs(m_ft)**2)
        t2 = -self.rfft_sum(U_double_deriv_ft * np.abs(m_hat_ft)**2)
        t3 = 2 * self.rfft_sum(U_deriv_ft * m_hat_ft.conj() * m_ft)
        return t1 + t2 + t3

    def simulate(self, T, dt, T_eval, omega_func=None, m_init=None,
                 use_tqdm=True, tau=1., exclude_Q=False):
        """Euler-integrate the DMFT equations for ``m`` and ``m_hat``.

        Parameters
        ----------
        T : float
            Total integration time.
        dt : float
            Integration step.
        T_eval : float
            Save state every ``T_eval`` time units.
        omega_func : callable or None
            Time-varying drift ``t -> omega(t)``. ``None`` -> zero drift.
        m_init : ndarray, shape (N_theta,), optional
            Initial order parameter. Defaults to ``self.Gamma_phi``.
        use_tqdm : bool, default True
            Show progress bar.
        tau : float, default 1.
            Membrane time constant.
        exclude_Q : bool, default False
            If True, replace ``Q`` with ``Gamma_x[0]`` (no fluctuations).

        Returns
        -------
        t : ndarray, shape (N_eval,)
            Sampled timesteps.
        m : ndarray, shape (N_eval, N_theta)
            Order-parameter snapshots.
        m_hat : ndarray, shape (N_eval, N_theta)
            Conjugate order-parameter snapshots.
        g_dyn : ndarray, shape (N_eval, N_theta)
            Snapshots of the input ``G`` driving the dynamics.
        """
        N_t = int(T / dt)
        N_eval = int(T / T_eval)
        eval_step = int(T_eval / dt)
        t = np.arange(N_eval) * T_eval

        omega = np.zeros(N_t) if omega_func is None else omega_func(np.arange(N_t) * dt)

        m = np.zeros((N_eval, self.N_theta))
        m_hat = np.zeros((N_eval, self.N_theta))
        g_dyn = np.zeros((N_eval, self.N_theta))

        m_current = m_init if m_init is not None else self.Gamma_phi
        m_hat_current = tau * omega[0] * m_current

        m[0] = m_current
        m_hat[0] = m_hat_current
        g_dyn[0] = m_current

        iterator = tqdm(range(1, N_t)) if use_tqdm else range(1, N_t)

        eval_index = 1
        for i in iterator:
            H = self.J_m_prod(m_current) - self.J_deriv_m_hat_prod(m_hat_current)
            Q = self.compute_Q(m_current, m_hat_current) if not exclude_Q else self.Gamma_x[0]
            G = self.F(self.Gamma_x[0], Q, H)

            m_deriv = (-m_current + G) / tau
            m_hat_deriv = (-m_hat_current + tau * omega[i] * G) / tau

            m_current += dt * m_deriv
            m_hat_current += dt * m_hat_deriv

            if i % eval_step == 0 and eval_index < N_eval:
                m[eval_index] = m_current
                m_hat[eval_index] = m_hat_current
                g_dyn[eval_index] = G
                eval_index += 1

        return t, m, m_hat, g_dyn

    def compute_bump_jacobian(self, exclude_Q=False):
        """Compute the Jacobian and eigenvalues at the bump solution.

        Parameters
        ----------
        exclude_Q : bool, default False
            If True, omit the Q-fluctuation contribution.

        Returns
        -------
        jac : ndarray, shape (N_theta, N_theta)
            Jacobian matrix at the bump.
        eigenvals : ndarray, shape (N_theta,)
            Eigenvalues sorted by descending real part.
        """
        G1 = self.dF_dC12(self.Gamma_x[0], self.Gamma_x[0], self.Gamma_x)
        G2 = self.dF_dC11(self.Gamma_x[0], self.Gamma_x[0], self.Gamma_x)

        T1 = G1[:, None] * circulant(self.J)
        T2 = G2[:, None] * 2 * self.J[None, :] if not exclude_Q else 0.

        jac = -np.eye(self.N_theta) + (T1 + T2) / self.N_theta
        eigenvals = np.linalg.eigvals(jac)
        eigenvals = eigenvals[np.argsort(eigenvals.real)[::-1]]
        return jac, eigenvals

    def find_homogeneous_solution(self, tol=1e-14, max_iter=2000, exclude_Q=False):
        """Locate the homogeneous (theta-independent) fixed point ``m0`` via bisection.

        Parameters
        ----------
        tol : float, default 1e-14
            Convergence tolerance on the residual.
        max_iter : int, default 2000
            Maximum bisection iterations.
        exclude_Q : bool, default False
            If True, replace Q with Gamma_x[0] in the fixed-point equation.

        Returns
        -------
        m0 : float
            Homogeneous fixed-point value.

        Raises
        ------
        ValueError
            If bisection fails to converge within ``max_iter`` iterations.
        """
        def compute_err(m0):
            Q_homo = self.U.mean() * m0**2 if not exclude_Q else self.Gamma_x[0]
            h_homo = self.J.mean() * m0
            return -m0 + self.F(Q_homo, self.Gamma_x[0], h_homo)

        # Bisection bounds — solution is typically close to 1.
        left, right = 1., 10.
        for _ in range(max_iter):
            m0 = (left + right) / 2
            err = compute_err(m0)
            if abs(err) < tol:
                return m0
            if err > 0:
                left = m0
            else:
                right = m0
        raise ValueError(
            f"Failed to converge within {max_iter} iterations, g = {self.g}, scale = {self.scale}")

    def compute_homogeneous_jacobian(self, m0=None, exclude_Q=False, exclude_Q_for_m0=False):
        """Compute the Jacobian and eigenvalues at the homogeneous fixed point.

        Parameters
        ----------
        m0 : float, optional
            Homogeneous solution value. Computed via ``find_homogeneous_solution``
            if not provided.
        exclude_Q : bool, default False
            If True, omit the Q-fluctuation contribution.
        exclude_Q_for_m0 : bool, default False
            If True (and ``m0`` is computed here), pass ``exclude_Q=True`` to
            ``find_homogeneous_solution``.

        Returns
        -------
        jac : ndarray, shape (N_theta, N_theta)
            Jacobian matrix at the homogeneous solution.
        eigenvals : ndarray, shape (N_theta,)
            Eigenvalues sorted by descending real part.
        """
        if m0 is None:
            m0 = self.find_homogeneous_solution(exclude_Q=exclude_Q_for_m0)

        Q_homo = self.U.mean() * m0**2 if not exclude_Q else self.Gamma_x[0]
        h_homo = self.J.mean() * m0

        t1 = self.dF_dC12(Q_homo, self.Gamma_x[0], h_homo)
        t3 = self.dF_dC11(Q_homo, self.Gamma_x[0], h_homo)
        t4 = 2 * self.U.mean() * m0
        Q_term = t3 * t4 if not exclude_Q else 0.

        jac = -np.eye(self.N_theta) + (t1 * circulant(self.J) + Q_term) / self.N_theta
        eigenvals = np.linalg.eigvals(jac)
        eigenvals = eigenvals[np.argsort(eigenvals.real)[::-1]]
        return jac, eigenvals
