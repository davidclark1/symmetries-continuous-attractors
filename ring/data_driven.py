"""Data-driven ring-attractor toolkit.

Builds the empirical weight matrix ``J`` from mouse tuning curves, runs
the resulting recurrent network with closed-loop head-direction drive,
projects trajectories into PC space, and plots them. Consumed primarily
by ``notebooks/data_driven_attractor.ipynb`` (Fig 3 producer) and the
``data_driven_analyses`` / ``grid_sims`` chains.
"""
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from . import sim
from . import fn
from . import regression
from . import analysis
from . import style  # noqa: F401  (imported for the matplotlib rcParams side effect)


# Paths resolve relative to the ring/ package (not the notebook cwd) so they
# work regardless of where the consumer runs from.
TC_DATA_PATH = str(Path(__file__).parent.parent / "data" / "mouse_data" / "tc_data.npz")
HD_TIMESERIES_PATH = str(Path(__file__).parent.parent / "data" / "actual_hd_timeseries_2.npz")


# --- Tuning-curve loading / preparation -------------------------------------

def load_tuning_curve_data():
    """(internal) Load tuning curves from ``TC_DATA_PATH``, dropping all-zero rows.

    Returns
    -------
    tcs : ndarray, shape (N_units, N_theta)
        Cross-validation-fold-averaged tuning curves (zero-row units dropped).
    plain_tcs : ndarray, shape (N_units, 2, N_theta)
        Plain (un-smoothed) per-fold tuning curves.
    hd_mask, fs_mask, exc_mask : ndarray of bool, shape (N_units,)
        Per-unit head-direction / fast-spiking / excitatory classifiers.
    num_units : int
        Number of surviving units.
    """
    data = np.load(TC_DATA_PATH, "r")
    tcs = data['opt_tcs'].mean(1)
    nz_mask = ~np.all(tcs == 0, axis=1)
    tcs = tcs[nz_mask]
    plain_tcs = data['plain_tcs'][nz_mask]
    hd_mask = data['hd_mask'][nz_mask]
    fs_mask = data['fs_mask'][nz_mask]
    exc_mask = data['exc_mask'][nz_mask]
    num_units = np.sum(nz_mask)
    return tcs, plain_tcs, hd_mask, fs_mask, exc_mask, num_units


def prepare_Phi(tcs, hd_mask):
    """(internal) Select HD-classified units, mean-normalize, and transpose to (N_theta, N)."""
    Phi = tcs[hd_mask].copy()
    scalings = 1. / Phi.mean(axis=1)
    Phi = Phi * scalings[:, None]
    return Phi.T  # shape = (N_theta, N)


def load_data_Phi():
    """Load ``Phi`` for the HD-classified subset of mouse units.

    Returns
    -------
    Phi : ndarray, shape (N_theta, N)
        Mean-normalized tuning curves transposed to (N_theta, N).
    """
    tcs, _, hd_mask, _, _, _ = load_tuning_curve_data()
    return prepare_Phi(tcs, hd_mask)


# --- Nonlinearities ---------------------------------------------------------

def nonlin(x, beta=2):
    """Softplus nonlinearity. Thin wrapper around ``fn.softplus(x, beta)``."""
    return fn.softplus(x, beta=beta)


def inv_nonlin(phi, beta=2):
    """(internal) Inverse softplus: ``log(exp(beta * phi) - 1) / beta``.

    Supports numpy or torch input; raises TypeError otherwise.
    """
    if isinstance(phi, np.ndarray):
        return np.log(np.exp(beta * phi) - 1) / beta
    elif isinstance(phi, torch.Tensor):
        return torch.log(torch.exp(beta * phi) - 1) / beta
    else:
        raise TypeError("Input must be either a NumPy array or a PyTorch tensor")


# --- Resampling / smoothing -------------------------------------------------

def resample_curves_fft(X, N_new):
    """(internal) FFT-based resampling of curves from ``N_original`` to ``N_new`` points.

    Parameters
    ----------
    X : ndarray, shape (N_original, N_curves)
    N_new : int

    Returns
    -------
    X_resampled : ndarray, shape (N_new, N_curves)
    """
    N_original, N_curves = X.shape
    X_fft = np.fft.rfft(X, axis=0)
    new_fft = np.zeros((N_new // 2 + 1, N_curves), dtype=complex)
    new_fft[:min(X_fft.shape[0], new_fft.shape[0])] = X_fft[:min(X_fft.shape[0], new_fft.shape[0])]
    new_fft *= N_new / N_original
    return np.fft.irfft(new_fft, n=N_new, axis=0)


def _sinkhorn_normalize(Phi, n_iter=1000):
    """(internal) Sinkhorn-Knopp double normalization, in place.

    Alternately rescales rows (axis 1) then columns (axis 0) to unit mean for
    ``n_iter`` sweeps, driving both marginal means to 1. Mutates ``Phi`` in
    place and returns it.

    Parameters
    ----------
    Phi : ndarray, shape (N_theta, N)
    n_iter : int, default 1000

    Returns
    -------
    Phi : ndarray
        The same array, normalized in place.
    """
    for _ in range(n_iter):
        Phi /= Phi.mean(1, keepdims=True)
        Phi /= Phi.mean(0, keepdims=True)
    return Phi


def prepare_smoothed_normalized_resampled_tuning_curves_OLD(
        Phi, N_theta=500, sigma=2, normalize=True, normalize_across_neurons=True):
    """Legacy gaussian-smoothing + Sinkhorn-iterate normalization pipeline.

    Retained for backward compatibility with ``data_driven_attractor.ipynb``
    and ``data_driven_reconstruction.ipynb`` which call it with the literal ``_OLD``
    name. The newer ``prepare_smoothed_normalized_resampled_tuning_curves``
    uses FFT cutoff smoothing instead of Gaussian.

    Parameters
    ----------
    Phi : ndarray, shape (N_theta_orig, N)
    N_theta : int, default 500
        Output resolution after FFT resampling.
    sigma : float, default 2
        Gaussian smoothing width in bin units.
    normalize : bool, default True
        If True, apply Sinkhorn-iterate normalization.
    normalize_across_neurons : bool, default True
        If True, normalize across both axes; otherwise only across theta.

    Returns
    -------
    X_smooth_norm_resampled : ndarray, shape (N_theta, N)
    Phi_smooth_norm_resampled : ndarray, shape (N_theta, N)
    """
    X = inv_nonlin(Phi, beta=2.)
    X_smooth = gaussian_filter1d(X, axis=0, sigma=sigma, mode='wrap')
    Phi_smooth = nonlin(X_smooth, beta=2.)
    Phi_smooth_norm = Phi_smooth.copy()
    if normalize:
        if normalize_across_neurons:
            _sinkhorn_normalize(Phi_smooth_norm)
        else:
            Phi_smooth_norm /= Phi_smooth_norm.mean(0, keepdims=True)
        X_smooth_norm = inv_nonlin(Phi_smooth_norm, beta=2.)
    else:
        X_smooth_norm = X_smooth
    X_smooth_norm_resampled = resample_curves_fft(X_smooth_norm, N_theta)
    Phi_smooth_norm_resampled = nonlin(X_smooth_norm_resampled, beta=2.)
    return X_smooth_norm_resampled, Phi_smooth_norm_resampled


def prepare_smoothed_normalized_resampled_tuning_curves(Phi_m, N_theta=500, cutoff=13):
    """FFT-cutoff smoothing + Sinkhorn normalization + FFT resampling.

    Smooths in Fourier space by zeroing coefficients beyond ``cutoff``,
    Sinkhorn-iterates the normalization (1000 sweeps), then resamples to
    ``N_theta``.

    Parameters
    ----------
    Phi_m : ndarray, shape (N_theta_orig, N)
    N_theta : int, default 500
        Output resolution after FFT resampling.
    cutoff : int, default 13
        Fourier-coefficient cutoff (rounded up to odd).

    Returns
    -------
    X_m_sm_resamp : ndarray, shape (N_theta, N)
    Phi_m_sm_resamp : ndarray, shape (N_theta, N)
    """
    if cutoff % 2 == 0:
        cutoff += 1

    X_m = inv_nonlin(Phi_m)

    X_m_ft = np.fft.rfft(X_m, axis=0)
    X_m_ft[cutoff:] = 0.
    X_m_sm = np.fft.irfft(X_m_ft, axis=0)

    Phi_m_sm = nonlin(X_m_sm)
    _sinkhorn_normalize(Phi_m_sm)
    X_m_sm = inv_nonlin(Phi_m_sm)
    X_m_sm_resamp = resample_curves_fft(X_m_sm, N_theta)
    Phi_m_sm_resamp = nonlin(X_m_sm_resamp)
    return X_m_sm_resamp, Phi_m_sm_resamp


# --- Weights + simulation ---------------------------------------------------

def compute_weights(X, Phi, lamda=1e-8):
    """Compute the recurrent weight matrix ``J`` and its skew derivative ``J_prime``.

    ``J`` is the ridge regression solution to ``X ≈ J @ Phi`` (via
    ``regression.compute_J``); ``J_prime`` is the analogous solution for
    the angular derivative of ``X``, used as the coupling to the omega
    drift term in the dynamics.

    Parameters
    ----------
    X : ndarray, shape (N_theta, N)
        Preactivation tuning curves.
    Phi : ndarray, shape (N_theta, N)
        Output (post-nonlinearity) tuning curves.
    lamda : float, default 1e-8
        Ridge regularization strength, in the ``N``-normalized convention of
        ``regression.compute_J`` (see its Notes).

    Returns
    -------
    J : ndarray, shape (N, N)
        Recurrent weight matrix.
    J_prime : ndarray, shape (N, N)
        Skew coupling matrix (for omega drift).
    """
    J, _ = regression.compute_J(X, Phi, lamda=lamda)
    N_theta = X.shape[0]
    dtheta = 2 * np.pi / N_theta
    dX = np.diff(X, axis=0) / dtheta
    dX = np.concatenate(((X[0:1] - X[-1:]) / dtheta, dX), axis=0)
    J_prime, _ = regression.compute_J(X=dX, Phi=Phi, lamda=lamda)
    return J, J_prime


def _auto_device(device=None):
    """(internal) Resolve ``device``: ``None`` -> CUDA if available, else CPU."""
    if device is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def run_simulation(J, x_init, T, tau, dt, T_eval,
                   J_prime=None, omega_func=None, c_const=1., device=None):
    """Run the data-driven recurrent dynamics (CUDA if available, else CPU).

    Wraps ``sim.run_sim`` with the softplus nonlinearity, the mean-
    stabilizer (target_mean=1.), and torch conversion of ``J``/``J_prime``
    /``x_init``. Returns only the saved trajectory snapshots.

    Parameters
    ----------
    J : ndarray, shape (N, N)
        Recurrent weights.
    x_init : ndarray, shape (N, N_batch)
        Initial states.
    T, tau, dt, T_eval : float
        Integration time, time constant, step, save period.
    J_prime : ndarray, shape (N, N), optional
        Required if ``omega_func`` is not None.
    omega_func : callable or None
        Time-varying drift.
    c_const : float, default 1.
        Mean-stabilizer strength.
    device : str or int or None, default None
        Torch device. ``None`` auto-selects CUDA if available, else CPU.

    Returns
    -------
    x_save : ndarray, shape (N_eval, N, N_batch)
        Saved state snapshots.
    """
    device = _auto_device(device)
    _, x_save, _ = sim.run_sim(
        T=T, tau=tau, dt=dt, T_eval=T_eval,
        J=torch.Tensor(J).to(device),
        J_prime=torch.Tensor(J_prime).to(device) if J_prime is not None else None,
        x_init=torch.Tensor(x_init).to(device),
        nonlin=nonlin,
        disable_tqdm=False, target_mean=1., const=c_const,
        omega_func=omega_func, break_if_stopped=False)
    return x_save


def compute_m(x_save, X):
    """Project trajectories onto tuning-curve basis via tensor contraction.

    Parameters
    ----------
    x_save : ndarray, shape (T_eval, N, N_batch)
        Trajectories.
    X : ndarray, shape (N_theta, N)
        Tuning-curve basis.

    Returns
    -------
    m : ndarray, shape (T_eval, N_theta, N_batch)
        ``m[t, a, b] = sum_i x_save[t, i, b] * X[a, i]``.
    """
    return np.einsum('tib,ai->tab', x_save, X, optimize='optimal')


# --- Head-direction utilities + plot helpers --------------------------------

def remove_discontinuities_mod_2pi(hd):
    """Mod-2pi-wrap an angle series, marking wrap discontinuities as NaN.

    Used for plotting head-direction traces without spurious vertical lines
    at each wrap-around.

    Parameters
    ----------
    hd : ndarray, shape (T,)
        Continuous (possibly unwrapped) angles.

    Returns
    -------
    hd_plot : ndarray, shape (T,)
        Mod-2pi-wrapped; entries flanking a > pi jump are NaN.
    """
    hd_plot = np.mod(hd, 2 * np.pi)
    discontinuities = np.where(np.abs(np.diff(hd_plot)) > np.pi)[0]
    for idx in discontinuities:
        hd_plot[idx:idx + 2] = np.nan
    return hd_plot


def plot_head_direction(ax, m, hd, m_dt, hd_dt, vmin=0):
    """Plot ``m`` as a heatmap with the head-direction track overlaid.

    Parameters
    ----------
    ax : matplotlib axis
    m : ndarray, shape (T_eval, N_theta)
    hd : ndarray, shape (T_hd,)
    m_dt : float
        Time step between ``m`` rows.
    hd_dt : float
        Time step between ``hd`` samples.
    vmin : float, default 0
        Lower bound for the imshow colormap (vmax is auto-set to the 97.5
        percentile of ``m``).
    """
    vmax = np.percentile(m.flatten(), 97.5)
    T = m_dt * len(m)
    ax.imshow(m.T[::-1], extent=(0, T, 0, 2 * np.pi), aspect='auto',
              cmap=style.field_cmap('ice'), vmin=vmin, vmax=vmax)

    hd_time = np.arange(int(T / hd_dt)) * hd_dt
    ax.plot(hd_time, hd[:len(hd_time)], lw=1, c=style.HD_COLOR, label='Head Direction')

    ax.set_yticks([0, np.pi, 2 * np.pi], ['0', 'π', '2π'])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("$\\theta$", rotation=0, va='center')


# --- PCA + 3D projection ---------------------------------------------------

def do_pca_on_tcs_and_trajectories(X, x_save, N_pcs=3, comp_axis=None,
                                   return_V=False, return_eig=False, N_retain=None):
    """PCA on tuning curves and projection of trajectories into PC space.

    Decomposes the centered tuning-curve matrix ``X - X.mean(0)`` and
    projects both the tuning curves and the trajectories ``x_save`` onto
    the top ``N_pcs`` principal components. Uses the kernel trick when
    ``N_theta < N`` to avoid building the (N x N) covariance.

    Parameters
    ----------
    X : ndarray, shape (N_theta, N)
        Tuning-curve matrix (samples x features).
    x_save : ndarray, shape (T_eval, N, N_batch)
        Trajectories to project.
    N_pcs : int, default 3
        Number of principal components.
    comp_axis : int or None, default None
        Force the kernel method (``0``) or the standard method (``1``).
        ``None`` auto-picks the smaller-dim option.
    return_V : bool, default False
        Also return the eigenvector matrix ``V``.
    return_eig : bool, default False
        Also return eigenvalues ``w``.
    N_retain : int, optional
        Number of PCs to use for the projection of ``X``; default keeps all.

    Returns
    -------
    pcs : ndarray, shape (N_theta, N_retain)
    pcs_traj : ndarray, shape (T_eval, N_pcs, N_batch)
    V : ndarray, shape (N, N_pcs), optional
    w : ndarray, optional
        Returned in (pcs, pcs_traj, V, w) order when both flags are True.
    """
    N_theta, N = X.shape
    mu = X.mean(0)
    X_ctd = X - mu
    x_save_ctd = x_save - mu[None, :, None]

    use_kernel = (comp_axis == 0) if comp_axis is not None else (N_theta < N)

    if use_kernel:
        K = np.dot(X_ctd, X_ctd.T) / N_theta  # (N_theta, N_theta)
        w_K, U = np.linalg.eigh(K)
        w_K, U = w_K[::-1], U[:, ::-1]
        V = np.zeros((N, N_pcs))
        for i in range(N_pcs):
            V[:, i] = np.dot(X_ctd.T, U[:, i]) / np.sqrt(w_K[i] * N_theta)
        w = w_K
    else:
        C = np.dot(X_ctd.T, X_ctd) / N_theta
        w, V = np.linalg.eigh(C)
        w, V = w[::-1], V[:, ::-1]
        V = V[:, :N_pcs]

    if N_retain is None:
        N_retain = N
    pcs = X_ctd @ V[:, :N_retain]
    pcs_traj = np.einsum('tib,ik->tkb', x_save_ctd, V, optimize='optimal')

    if return_V:
        return (pcs, pcs_traj, V, w) if return_eig else (pcs, pcs_traj, V)
    return (pcs, pcs_traj, w) if return_eig else (pcs, pcs_traj)


def plot_3d_pca(ax, pcs, pcs_traj, torus=False, cloud_color='black',
                cloud_alpha=0.1, cloud_s=10, traj_lw=None, end_s=12, start_s=5):
    """3D plot of trajectories in PC space with cyclic color coding.

    Parameters
    ----------
    ax : matplotlib 3D axis
    pcs : ndarray, shape (N_theta, >= 3)
    pcs_traj : ndarray, shape (T_eval, 3, N_batch)
    torus : bool, default False
        If True, render ``pcs`` as a scatter cloud instead of a connected line.
    cloud_color, cloud_alpha, cloud_s : str, float, float
        Appearance of the ``torus=True`` manifold cloud. The defaults reproduce
        the original dense black cloud; the torus figure passes a lighter, smaller
        cloud so the coloured trajectories on top of it stay readable.
    traj_lw : float or None
        Trajectory line width. None keeps the rcParams default.
    end_s, start_s : float
        Marker sizes for the trajectory end dot and start cross. Smaller values
        stop the black markers from swamping the trajectory colours.
    """
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)

    if torus:
        ax.scatter(pcs[:, 0], pcs[:, 1], pcs[:, 2], lw=0, s=cloud_s,
                   alpha=cloud_alpha, c=cloud_color)
    else:
        ax.plot(pcs[:, 0], pcs[:, 1], pcs[:, 2], c='0.6', ls='-', zorder=0)

    # Trajectories are indexed by their initial angle on the ring, which is
    # cyclic, so they take ANGLE_CMAP. This used to read raw `cm.hsv`, which
    # callers had to monkey-patch to get the house map; grid_sims did not patch
    # it and so drew the neon rainbow instead.
    n_trajectories = pcs_traj.shape[2]
    colors = style.ANGLE_CMAP(np.linspace(0, 1, n_trajectories))

    for ic_idx in range(n_trajectories):
        color = colors[ic_idx]
        ax.scatter(pcs_traj[-1, 0, ic_idx], pcs_traj[-1, 1, ic_idx], pcs_traj[-1, 2, ic_idx],
                   lw=0, s=end_s, c='black', marker='.')
        ax.scatter(pcs_traj[0, 0, ic_idx], pcs_traj[0, 1, ic_idx], pcs_traj[0, 2, ic_idx],
                   lw=0.4, s=start_s, c='black', marker='x')
        ax.plot(pcs_traj[:, 0, ic_idx], pcs_traj[:, 1, ic_idx], pcs_traj[:, 2, ic_idx],
                c=color, lw=traj_lw)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.view_init(elev=20, azim=120)


def torus_display_frame(pcs, phase_a, phase_b):
    """Build a 3-axis display frame that makes a 2-torus manifold read as a torus.

    Taking the top three principal components is not a stable way to view this
    manifold. On the hexagonal grid the leading PCA block is SIX-fold
    near-degenerate (three lattice directions times cos/sin), with the six
    eigenvalues spanning under 5% and a 92% gap to the seventh. Which three of
    the six come out on top is therefore decided by noise, and a perturbation as
    small as a few percent in ``J`` reshuffles them, so the projection can swing
    from showing the hole to hiding it while the manifold itself is unchanged.

    Since the two toroidal phases are known, the frame can be pinned to them
    instead:

    - ``x``, ``y`` span the plane in which ``phase_a`` traces a closed circle,
      obtained by regressing its cosine and sine onto the block;
    - ``z`` is the residual direction that best encodes ``phase_b``, which lifts
      the tube out of that plane.

    A projection shows a hole exactly when one toroidal angle sweeps a full
    circle in the plotted plane while the other displaces along the third axis,
    which is what this construction guarantees. It is deterministic given the
    lattice, so it does not drift with the PC ordering.

    Parameters
    ----------
    pcs : ndarray, shape (N_points, N_pcs)
        Manifold projected onto the leading PCs. Use the whole degenerate block
        (six components for the hexagonal grid), not three.
    phase_a : ndarray, shape (N_points,)
        Toroidal phase to place in the plotted plane, in radians.
    phase_b : ndarray, shape (N_points,)
        The other toroidal phase, in radians, which sets the third axis.

    Returns
    -------
    frame : ndarray, shape (3, N_pcs)
        Orthonormal rows. Apply to the manifold as ``pcs @ frame.T`` and to
        trajectories as ``np.einsum('tkb,mk->tmb', pcs_traj, frame)``.
    """
    B = np.asarray(pcs, dtype=float)
    B = B - B.mean(0)

    def _fit(target):
        c, *_ = np.linalg.lstsq(B, target - target.mean(), rcond=None)
        norm = np.linalg.norm(c)
        if norm < 1e-12:
            raise ValueError("phase is not represented in the supplied PC block")
        return c / norm

    axes = []
    for v in (_fit(np.cos(phase_a)), _fit(np.sin(phase_a)), _fit(np.cos(phase_b))):
        for u in axes:
            v = v - (v @ u) * u
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            raise ValueError("degenerate display frame; the phases are collinear "
                             "in this PC block")
        axes.append(v / norm)
    return np.array(axes)


# --- Distance-to-ring helpers ----------------------------------------------

def compute_distances_to_ring(X, x_save, device='cpu'):
    """(internal) Closest-tuning-curve distance per timestep, normalized by sqrt(N)."""
    X_torch = torch.Tensor(X).to(device)
    x_save_torch = torch.Tensor(x_save).to(device)
    distances = []
    for t_idx in tqdm(range(len(x_save))):
        X_dyn = x_save_torch[t_idx, :]
        A = X_dyn[None, :, :] - X_torch[:, :, None]
        dist_matrix = torch.linalg.norm(A, dim=1)
        nearest_dist = torch.min(dist_matrix, dim=0).values
        distances.append(nearest_dist.cpu().numpy())
    distances = np.array(distances)
    N = X.shape[1]
    return distances / np.sqrt(N)  # not squared


def compute_distances_to_ring_batched(X, x_save, device='cpu', chunk_size=10):
    """Chunked wrapper around ``compute_distances_to_ring`` to keep GPU memory in check.

    Parameters
    ----------
    X : ndarray, shape (N_theta, N)
    x_save : ndarray, shape (T_eval, N, N_batch)
    device : str, default 'cpu'
    chunk_size : int, default 10
        Number of batch elements per GPU chunk.

    Returns
    -------
    all_dists : ndarray, shape (T_eval, N_batch)
        Normalized closest-tuning-curve distance per timestep per batch.
    """
    N_chunks = int(np.ceil(x_save.shape[-1] / chunk_size))
    all_dists = np.zeros((x_save.shape[0], x_save.shape[2]))
    for i in range(N_chunks):
        print('chunk =', i + 1, 'of', N_chunks)
        i1, i2 = i * chunk_size, (i + 1) * chunk_size
        all_dists[:, i1:i2] = compute_distances_to_ring(X, x_save[:, :, i1:i2], device=device)
    return all_dists


# --- Misc circular geometry + plotting helpers -----------------------------

def circular_com(data, return_idx=False):
    """Circular center of mass of curves indexed by theta.

    The float-angle computation is delegated to
    ``analysis.circular_center_of_mass`` (bit-identical single source); this
    function adds the optional integer-bin (floor) return.

    Parameters
    ----------
    data : ndarray, shape (N_curves, N_theta)
    return_idx : bool, default False
        If True, return integer bin indices (floor) instead of float angles.

    Returns
    -------
    com : ndarray, shape (N_curves,)
        Either angles in [0, 2*pi) (default) or integer bin indices.
    """
    result = analysis.circular_center_of_mass(data)
    if return_idx:
        return np.floor(result * data.shape[1] / (2 * np.pi)).astype(int)
    return result


def sort_weight_matrix(J, Phi):
    """Sort ``J`` rows and columns by the circular COM of ``Phi``'s tuning curves.

    Parameters
    ----------
    J : ndarray, shape (N, N)
    Phi : ndarray, shape (N_theta, N)

    Returns
    -------
    J_srt : ndarray, shape (N, N)
        ``J`` reindexed so units appear in circular order.
    order : ndarray, shape (N,)
        Sorting permutation (apply via ``arr[order]``).
    """
    order = np.argsort(circular_com(Phi.T, True))
    J_srt = J[order, :][:, order]
    return J_srt, order


# --- HD timeseries pipeline ------------------------------------------------

def load_hd_timeseries_data(file_path=HD_TIMESERIES_PATH):
    """(internal) Load (hd, dt) from a Peyrache-style npz."""
    print("loading", file_path)
    with np.load(file_path) as data:
        hd = data['hd']
        data_dt = data['dt'].item()
    return hd, data_dt


def interpolate_angle_nans(hd):
    """(internal) Linearly interpolate NaN gaps in a wrapped-angle series.

    Interpolation is done on the complex-exponential representation
    ``z = exp(1j * hd)`` to handle wrap-around correctly.
    """
    non_nan_indices = np.where(~np.isnan(hd))[0]
    if len(non_nan_indices) == 0 or len(non_nan_indices) == len(hd):
        return hd
    all_indices = np.arange(len(hd))
    z = np.exp(1j * hd[non_nan_indices])
    real_interp = np.interp(all_indices, non_nan_indices, np.real(z))
    imag_interp = np.interp(all_indices, non_nan_indices, np.imag(z))
    return np.mod(np.angle(real_interp + 1j * imag_interp), 2 * np.pi)


def preprocess_hd_data(hd, data_dt, smoothing_sigma=10):
    """(internal) NaN-interpolate + unwrap + Gaussian-smooth + compute omega(t).

    Prepends 2000 stationary samples to the unwrapped series so the
    dynamics has a quiet start.

    Returns
    -------
    hd_unwrap_smooth : ndarray
    omega : ndarray
        Angular velocity = d(hd)/dt of the smoothed series.
    """
    hd = interpolate_angle_nans(hd)
    hd_unwrap = np.unwrap(hd)
    hd_unwrap = np.concatenate((np.ones(2000) * hd_unwrap[0], hd_unwrap))
    hd_unwrap_smooth = gaussian_filter1d(hd_unwrap, sigma=smoothing_sigma)
    omega = np.diff(np.unwrap(hd_unwrap_smooth)) / data_dt
    return hd_unwrap_smooth, omega


def load_hd_and_omega():
    """Load mouse HD timeseries and compute its smoothed angular velocity.

    Returns
    -------
    hd : ndarray
    omega : ndarray
    data_dt : float
    """
    hd, data_dt = load_hd_timeseries_data()
    hd, omega = preprocess_hd_data(hd, data_dt)
    return hd, omega, data_dt
