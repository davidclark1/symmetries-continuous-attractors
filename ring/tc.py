"""Tuning curve sampling and wraparound covariance helpers.

Used by other ring/* modules and notebooks for sampling random tuning curves
(``sample_tcs``) and constructing the wrapped Gaussian covariance kernel
(``wrapped_gaussian_cov``) used in DMFT and grid-search optimization.
"""
import numpy as np
import torch

from . import fn


def wrapped_gaussian_cov(eq_var, scale, N_theta, backend='numpy'):
    """Wraparound (periodic) Gaussian covariance and its real Fourier transform.

    The kernel is the wraparound generalization of ``N(0, scale^2)`` sampled
    at ``N_theta`` evenly-spaced points on the circle, normalized so the
    Fourier amplitude at DC is ``eq_var``.

    Parameters
    ----------
    eq_var : float
        Equilibrium variance.
    scale : float
        Gaussian length scale (in units of Fourier-frequency index).
    N_theta : int
        Number of angular grid points.
    backend : str, default 'numpy'
        ``'numpy'`` or ``'torch'``. Torch ops use the default torch device.

    Returns
    -------
    Gamma : ndarray or torch.Tensor, shape (N_theta,)
        Real-space wraparound covariance.
    Gamma_ft : ndarray or torch.Tensor, shape (N_theta // 2 + 1,)
        Fourier-space (real) covariance.
    """
    N_freq = N_theta // 2 + 1
    if backend == 'numpy':
        freq_range = np.arange(N_freq)
        Gamma_ft = (eq_var / np.sqrt(2 * np.pi) / scale) * np.exp(-freq_range**2 / (2 * scale**2))
        Gamma = np.fft.irfft(Gamma_ft, norm='forward')
    else:  # torch
        freq_range = torch.arange(N_freq)
        Gamma_ft = (eq_var / np.sqrt(2 * np.pi) / scale) * torch.exp(-freq_range**2 / (2 * scale**2))
        Gamma = torch.fft.irfft(Gamma_ft, norm='forward')
    return Gamma, Gamma_ft


def sample_tcs(N, Gamma_ft, cutoff=None):
    """Sample ``N`` tuning curves from a wraparound Gaussian process.

    Each sampled curve is a real periodic function of theta whose Fourier-
    space covariance matches ``Gamma_ft``. The procedure draws iid complex
    normal Fourier coefficients, scales by the sqrt of the spectrum, and
    inverse-FFTs to real space. Uses the global numpy RNG, so the caller is
    responsible for seeding it when the draw needs to be reproducible.

    Parameters
    ----------
    N : int
        Number of curves to sample.
    Gamma_ft : ndarray, shape (N_freq,)
        Fourier-space covariance spectrum (real). ``N_theta`` is inferred
        as ``2 * (N_freq - 1)``.
    cutoff : int, optional
        If given, zero out spectrum beyond this index (high-frequency cutoff).
        Default uses all frequencies.

    Returns
    -------
    X : ndarray, shape (N_theta, N)
        Sampled tuning curves, one per column.

    Notes
    -----
    Draws the complex Fourier coefficients in ``(N_freq, N)`` layout;
    ``gp_opt.sample_z`` uses the transposed ``(N_samples, N_freq)`` layout, so
    the two yield different realizations under the same seed and are
    intentionally NOT interchangeable.
    """
    N_freq = len(Gamma_ft)
    N_theta = (N_freq - 1) * 2
    if cutoff is None:
        cutoff = N_freq
    X_ft = np.zeros((N_freq, N), dtype=complex)
    X_ft[:cutoff, :] = (np.random.randn(cutoff, N) + 1j * np.random.randn(cutoff, N)) / np.sqrt(2.)
    X_ft[0, :] = X_ft[0, :].real * np.sqrt(2.)
    X_ft *= np.sqrt(np.clip(Gamma_ft[:, None], a_min=0, a_max=np.inf))
    X = np.fft.irfft(X_ft, axis=0, norm='forward')
    return X


def gen_tuning_curves_idealized(N_samples, N_theta, scale, beta):
    """Sample idealized (erf) tuning curves for the large-N generative process.

    Draws input currents from the wraparound Gaussian process
    (``wrapped_gaussian_cov`` + ``sample_tcs``) and applies the idealized
    error-function nonlinearity (``fn.nonlin_idealized``). Uses the global numpy
    RNG via ``sample_tcs``; the caller is responsible for seeding.

    Parameters
    ----------
    N_samples : int
        Number of tuning curves to sample.
    N_theta : int
        Number of angular grid points.
    scale : float
        Wraparound-Gaussian Fourier decay scale (``sigma``).
    beta : float
        Response sharpness of the nonlinearity.

    Returns
    -------
    X : ndarray, shape (N_theta, N_samples)
        Input-current tuning curves.
    Phi : ndarray, shape (N_theta, N_samples)
        Firing-rate tuning curves.
    """
    _, Gamma_x_ft = wrapped_gaussian_cov(eq_var=1., scale=scale, N_theta=N_theta)
    X = sample_tcs(N=N_samples, Gamma_ft=Gamma_x_ft)
    Phi = fn.nonlin_idealized(X, beta=beta)
    return X, Phi
