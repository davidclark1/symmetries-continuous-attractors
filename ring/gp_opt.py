"""Grid-search optimizer for the (scale, bias, beta) tuning-curve generative parameters.

Walks a (scale, bias, beta) grid, generates ``Phi_m`` samples from each parameter
triple, computes the resulting Fourier-space covariance, and compares to a
target ``Gamma_phi_data_ft`` (typically derived from real mouse data in
``notebooks/data_and_generative_model.ipynb``).

Run as a script to (re)produce ``grid_search_results.npz``::

    python -m ring.gp_opt

Public surface used by notebooks: ``gen_Phi_m`` and ``compute_cov_from_params``
(see e.g. ``data_and_generative_model.ipynb``); the other functions support the
script flow.
"""
from pathlib import Path

import numpy as np
import torch

from . import tc, fn
from .seed import set_global_seed


# --- Configuration ----------------------------------------------------------

N_THETA = 100          # Number of angular grid points.
N_SAMPLES = 500000     # Monte Carlo samples per parameter evaluation.
DEVICE = 0             # int -> CUDA device index; None -> CPU.

# Paths resolve relative to the ring/ package (not the caller's cwd), so
# ``python -m ring.gp_opt`` works from anywhere.
_DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_GAMMA_FILE = str(_DATA_DIR / 'gamma_phi_info.npz')      # Target Fourier coeffs (written by data_and_generative_model.ipynb).
OUTPUT_RESULTS_FILE = str(_DATA_DIR / 'grid_search_results.npz')

# Search bounds.
SCALE_BOUNDS = (1.0, 2.0)
BIAS_BOUNDS = (0.01, 5.0)
BETA_BOUNDS = (0.01, 10.0)

# Grid resolution along each axis.
N_SCALE_POINTS = 20
N_BIAS_POINTS = 30
N_BETA_POINTS = 30


# --- GP sampling primitives -------------------------------------------------

def sample_z(N_samples, N_freq, device=None):
    """Draw iid complex normal Fourier coefficients ``z`` for GP sampling.

    The DC coefficient (column 0) is constrained to be real (and rescaled
    by ``sqrt(2)``) so the inverse Fourier transform produces a real series.

    Parameters
    ----------
    N_samples : int
        Number of independent draws.
    N_freq : int
        Number of Fourier frequencies (typically ``N_theta // 2 + 1``).
    device : int or None
        ``None`` -> numpy on CPU; otherwise the torch CUDA device index.

    Returns
    -------
    z : ndarray or torch.Tensor, shape (N_samples, N_freq), complex

    Notes
    -----
    Draws in ``(N_samples, N_freq)`` layout. ``tc.sample_tcs`` draws the same
    kind of coefficients in the transposed ``(N_freq, N)`` layout, so the two
    fill the RNG stream in a different order and yield *different* realizations
    under the same seed. They are intentionally NOT interchangeable, so do not
    reroute one through the other.
    """
    if device is None:  # numpy
        z = (np.random.randn(N_samples, N_freq) + 1j * np.random.randn(N_samples, N_freq)) / np.sqrt(2)
        z[:, 0] = z[:, 0].real * np.sqrt(2)
    else:  # torch
        z_real = torch.randn(N_samples, N_freq, device=device)
        z_imag = torch.randn(N_samples, N_freq, device=device)
        z = (z_real + 1j * z_imag) / np.sqrt(2)
        z[:, 0] = z[:, 0].real * np.sqrt(2)
    return z


def gen_Phi_m(scale, bias, beta, N_samples, N_theta, device=None,
              return_Gamma_x_ft=False, normalize=False):
    """Generate ``Phi_m`` samples from a wraparound Gaussian process + softplus.

    ``X_m`` is a draw from the wraparound-Gaussian prior with length
    scale ``scale``; ``Phi_m = softplus(X_m - bias, beta=beta)`` applies
    the nonlinearity.

    Parameters
    ----------
    scale : float
        Wraparound-Gaussian length scale.
    bias : float
        Offset subtracted before the nonlinearity.
    beta : float
        Sharpness of the softplus nonlinearity.
    N_samples : int
        Number of independent samples.
    N_theta : int
        Number of angular grid points.
    device : int or None
        CUDA device index, or ``None`` for CPU/numpy.
    return_Gamma_x_ft : bool, default False
        Also return the Fourier-space covariance.
    normalize : bool, default False
        Normalize each sample by its mean before returning (avoids the
        all-zero collapse for some parameter combinations).

    Returns
    -------
    Phi_m : ndarray or torch.Tensor, shape (N_samples, N_theta)
    X_m : same type/shape, optional
        Returned only if ``return_Gamma_x_ft`` is True.
    Gamma_x_ft : same type, shape (N_theta // 2 + 1,), optional
        Returned only if ``return_Gamma_x_ft`` is True.
    """
    N_freq = N_theta // 2 + 1
    eq_var = 1.
    backend = 'numpy' if device is None else 'torch'

    _, Gamma_x_ft = tc.wrapped_gaussian_cov(eq_var, scale, N_theta, backend=backend)
    if device is not None:
        Gamma_x_ft = Gamma_x_ft.to(device)

    z = sample_z(N_samples, N_freq, device)

    if device is None:
        X_m = np.fft.irfft(z * np.sqrt(Gamma_x_ft)[None, :], axis=1, norm='forward')
    else:
        X_m = torch.fft.irfft(z * torch.sqrt(Gamma_x_ft)[None, :], dim=1, norm='forward')

    Phi_m = fn.softplus(X_m - bias, beta=beta)

    if normalize:
        if device is None:
            Phi_m = Phi_m / (Phi_m.mean(1, keepdims=True) + 1e-30)
        else:
            Phi_m = Phi_m / (Phi_m.mean(1, keepdim=True) + 1e-30)

    if return_Gamma_x_ft:
        return Phi_m, X_m, Gamma_x_ft
    return Phi_m


def compute_cov_from_params(params, N_samples, N_theta, device=None):
    """Generate Phi_m samples for ``params = (scale, bias, beta)`` and return their covariance.

    Parameters
    ----------
    params : sequence of 3 floats
        (scale, bias, beta).
    N_samples : int
    N_theta : int
    device : int or None

    Returns
    -------
    Gamma_phi : ndarray or torch.Tensor, shape (N_theta,)
    Gamma_phi_ft : ndarray or torch.Tensor, shape (N_theta // 2 + 1,)
    """
    Phi_m = gen_Phi_m(*params, N_samples, N_theta, device=device, normalize=True)
    Gamma_phi, Gamma_phi_ft = fn.calc_C(Phi_m)
    return Gamma_phi, Gamma_phi_ft


def eval_loss(params, Gamma_phi_data_ft, N_samples, N_theta, device=None):
    """L2 distance between the model and data covariance in the first 20 Fourier components.

    Parameters
    ----------
    params : sequence of 3 floats
        (scale, bias, beta).
    Gamma_phi_data_ft : ndarray, shape (>= 20,)
        Target Fourier-space covariance.
    N_samples, N_theta, device :
        Forwarded to ``compute_cov_from_params``.

    Returns
    -------
    err : float
    """
    _, Gamma_phi_ft = compute_cov_from_params(params, N_samples, N_theta, device)
    if device is None:
        err = np.sqrt(np.sum((Gamma_phi_ft[:20] - Gamma_phi_data_ft[:20])**2))
    else:
        err = torch.sqrt(torch.sum(
            (Gamma_phi_ft[:20] - torch.from_numpy(Gamma_phi_data_ft[:20]).to(device))**2))
        err = err.item()
    return err


# --- Grid search ------------------------------------------------------------

def run_grid_search(Gamma_phi_data_ft):
    """Run the (scale, bias, beta) grid search and write intermediate + final results.

    Seeds the global RNG to 42 before any sampling, so results are bit-reproducible.

    Parameters
    ----------
    Gamma_phi_data_ft : ndarray, shape (>= 20,)
        Target Fourier coefficients.

    Returns
    -------
    best_params : dict
        ``{'scale', 'bias', 'beta', 'loss'}`` at the minimum.
    loss_vals : ndarray, shape (N_SCALE_POINTS, N_BIAS_POINTS, N_BETA_POINTS)
        Full grid of loss values.
    """
    # Pin the RNG so grid_search_results.npz is bit-reproducible.
    set_global_seed(42)

    scale_vals = np.linspace(SCALE_BOUNDS[0], SCALE_BOUNDS[1], N_SCALE_POINTS)
    bias_vals = np.linspace(BIAS_BOUNDS[0], BIAS_BOUNDS[1], N_BIAS_POINTS)
    beta_vals = np.linspace(BETA_BOUNDS[0], BETA_BOUNDS[1], N_BETA_POINTS)
    loss_vals = np.zeros((len(scale_vals), len(bias_vals), len(beta_vals)))

    device = torch.device(f'cuda:{DEVICE}') if isinstance(DEVICE, int) else None

    for i, scale in enumerate(scale_vals):
        print(f"Processing scale {i+1}/{len(scale_vals)}")
        for j, bias in enumerate(bias_vals):
            for k, beta in enumerate(beta_vals):
                loss_vals[i, j, k] = eval_loss(
                    [scale, bias, beta], Gamma_phi_data_ft,
                    N_SAMPLES, N_THETA, device=device)

        # Track + checkpoint best-so-far each scale-slab.
        min_idx = np.unravel_index(np.argmin(loss_vals), loss_vals.shape)
        current_best = {
            'scale': scale_vals[min_idx[0]],
            'bias': bias_vals[min_idx[1]],
            'beta': beta_vals[min_idx[2]],
            'loss': loss_vals[min_idx],
        }
        np.savez(OUTPUT_RESULTS_FILE,
                 scale_vals=scale_vals, bias_vals=bias_vals, beta_vals=beta_vals,
                 loss_vals=loss_vals,
                 bounds=[SCALE_BOUNDS, BIAS_BOUNDS, BETA_BOUNDS],
                 best_params=current_best)

    min_idx = np.unravel_index(np.argmin(loss_vals), loss_vals.shape)
    best_params = {
        'scale': scale_vals[min_idx[0]],
        'bias': bias_vals[min_idx[1]],
        'beta': beta_vals[min_idx[2]],
        'loss': loss_vals[min_idx],
    }
    return best_params, loss_vals


def main():
    """Script entry point: load ``INPUT_GAMMA_FILE`` and run the grid search."""
    print("Starting grid search with parameters:")
    print(f"  N_THETA: {N_THETA}")
    print(f"  N_SAMPLES: {N_SAMPLES}")
    print(f"  Device: {DEVICE}")
    print(f"  Grid resolution: {N_SCALE_POINTS}x{N_BIAS_POINTS}x{N_BETA_POINTS}")
    print(f"  Input file: {INPUT_GAMMA_FILE}")
    print(f"  Output file: {OUTPUT_RESULTS_FILE}")

    with np.load(INPUT_GAMMA_FILE) as data:
        Gamma_phi_data_ft = data['Gamma_phi_data_ft']
    print("\nInput data loaded successfully.")

    best_params, loss_vals = run_grid_search(Gamma_phi_data_ft)
    print("\nGrid search completed successfully!")
    print("\nBest parameters found:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")


if __name__ == "__main__":
    main()
