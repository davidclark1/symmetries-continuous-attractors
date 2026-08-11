"""Low-level math helpers: nonlinearities and Fourier-space covariance.

Used by other ring/* modules and notebooks for activation functions
(``unity_erf``, ``softplus``) and circulant covariance estimation
(``calc_C``).
"""
import numpy as np
import torch
from scipy.special import erf


def unity_erf(x):
    """Error-function nonlinearity normalized so ``phi'(0) = 1``.

    ``phi(x) = erf(sqrt(pi) * x / 2)``.

    Parameters
    ----------
    x : float or ndarray or torch.Tensor
        Input.

    Returns
    -------
    Same type and shape as `x`.
    """
    if isinstance(x, torch.Tensor):
        return torch.erf(np.sqrt(np.pi) * x / 2.)
    return erf(np.sqrt(np.pi) * x / 2.)


def unity_erf_deriv(x):
    """Derivative of ``unity_erf``: ``exp(-pi * x^2 / 4)``.

    Parameters
    ----------
    x : float or ndarray or torch.Tensor
        Input.

    Returns
    -------
    Same type and shape as `x`.
    """
    if isinstance(x, torch.Tensor):
        return torch.exp(-(np.pi * x**2) / 4)
    return np.exp(-(np.pi * x**2) / 4)


def nonlin_idealized(x, beta):
    """Idealized (saturating) activation: ``1 + unity_erf(beta * x)``.

    The error-function nonlinearity used for the large-N "idealized" generative
    process (saturates to 0 and 2). Shared by the spectra / weight-matrix /
    spurious-fixed-point notebooks.

    Parameters
    ----------
    x : float or ndarray or torch.Tensor
        Input.
    beta : float
        Response sharpness.

    Returns
    -------
    Same type and shape as `x`.
    """
    return 1 + unity_erf(beta * x)


def softplus(x, beta=1.):
    """Numerically stable softplus: ``log(1 + exp(beta * x)) / beta``.

    Uses ``x + log(1 + exp(-beta*x))/beta`` for ``x >= 0`` and
    ``log(1 + exp(beta*x))/beta`` for ``x < 0`` to avoid overflow.

    Parameters
    ----------
    x : float or ndarray or torch.Tensor
        Input. Python floats are promoted to a 1-element numpy array.
    beta : float
        Sharpness parameter; approaches ReLU as ``beta -> inf``.

    Returns
    -------
    Same type and shape as the (possibly promoted) input.
    """
    if isinstance(x, (int, float)):
        x = np.array([x], dtype=float)
    if isinstance(x, torch.Tensor):
        result = torch.zeros_like(x)
        mask_pos = x >= 0
        result[mask_pos] = x[mask_pos] + torch.log(1 + torch.exp(-beta * x[mask_pos])) / beta
        result[~mask_pos] = torch.log(1 + torch.exp(beta * x[~mask_pos])) / beta
    else:
        result = np.zeros_like(x)
        mask_pos = x >= 0
        result[mask_pos] = x[mask_pos] + np.log(1 + np.exp(-beta * x[mask_pos])) / beta
        result[~mask_pos] = np.log(1 + np.exp(beta * x[~mask_pos])) / beta
    return result


def calc_C(X, return_ft=True):
    """Circulant (wraparound) covariance of rows of ``X`` via FFT.

    Uses the identity ``E[|FFT(x)|^2] = FT(C)``: averaging the squared
    Fourier magnitudes across samples yields the power spectrum, whose
    inverse transform is the circulant covariance.

    Parameters
    ----------
    X : ndarray or torch.Tensor, shape (N_samples, N_theta)
        Sample matrix; rows are samples.
    return_ft : bool, default True
        If True, return both ``C`` (real-space) and ``C_ft`` (Fourier).
        Otherwise return only ``C``.

    Returns
    -------
    C : same type as `X`, shape (N_theta,)
        Real-space circulant covariance.
    C_ft : same type as `X`, shape (N_theta // 2 + 1,)
        Fourier-space covariance (the power spectrum). Only returned
        if ``return_ft`` is True.
    """
    if isinstance(X, np.ndarray):
        C_ft = np.mean(np.abs(np.fft.rfft(X, axis=1, norm='forward'))**2, axis=0)
        C = np.fft.irfft(C_ft, norm='forward')
    else:  # torch
        C_ft = torch.mean(torch.abs(torch.fft.rfft(X, dim=1, norm='forward'))**2, dim=0)
        C = torch.fft.irfft(C_ft, norm='forward')
    if return_ft:
        return C, C_ft
    return C
