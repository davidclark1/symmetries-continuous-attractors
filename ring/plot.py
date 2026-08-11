"""Plotting helpers for the paper figures.

Each public ``plot_*`` function takes one or more matplotlib axes and renders
a specific panel of the paper figures (mostly Figs 4, 6 from
``notebooks/data_and_generative_model.ipynb``). The helpers ``setup_tc_axes`` and
``setup_profile_axes`` are internal axis-formatting utilities shared by
several plotters.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as _mcolors
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path
from scipy.stats import pearsonr

from . import gp_opt
from . import tc
from . import fn
from . import analysis
from . import style
from .seed import set_global_seed


# Color palette for data-vs-model comparisons (centralized in ring.style).
gen_process_color = style.MODEL
data_color = style.DATA


# --- Axis-formatting helpers (internal) ------------------------------------

def setup_tc_axes(axes):
    """(internal) Standardize tuning-curve panel formatting.

    Sets x-tick positions to [0, pi, 2*pi] with symbolic labels, x-limits
    to [0, 2*pi], y-limits to [0, 10]; suppresses x-labels on non-bottom
    rows and y-tick labels on non-left columns.

    Parameters
    ----------
    axes : ndarray of matplotlib axes
        1-D or 2-D grid of axes.
    """
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)
    for ax in axes.flatten():
        ax.set_xticks([0, np.pi, 2 * np.pi])
        ax.set_xticklabels(["0", "$\\pi$", "$2\\pi$"])
        ax.set_xlabel("$\\theta$")
        ax.set_ylim(0, 10)
        ax.set_xlim(0, 2 * np.pi)
    for ax in axes[:-1].flatten():
        ax.set_xlabel("")
    for ax in axes[:, 1:].flatten():
        ax.set_yticklabels([])


def setup_profile_axes(ax, ylabel):
    """(internal) Standardize profile-plot panel formatting.

    Parameters
    ----------
    ax : matplotlib axis
    ylabel : str
    """
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_ylim(0, 6)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticklabels(['$-\\pi$', '$0$', '$\\pi$'])
    ax.set_ylabel(ylabel)


# --- Optimization landscape plots ------------------------------------------

def plot_log_sigma_scatter(ax, opt_sigmas):
    """Cross-validation sigma-smoothing comparison scatter.

    Plots ``log10(sigma_fold1)`` vs ``log10(sigma_fold2)`` and overlays the
    Pearson correlation.

    Parameters
    ----------
    ax : matplotlib axis
    opt_sigmas : ndarray, shape (N_units, 2)
        Optimal sigmas per CV fold.
    """
    log_sigmas = np.log10(opt_sigmas)
    ax.scatter(log_sigmas[:, 0], log_sigmas[:, 1],
               lw=0, s=2.5, alpha=0.25, clip_on=False)
    ax.set_xlim(-1, 2)
    ax.set_ylim(-1, 2)
    ax.set_aspect(1.0)

    rho, _ = pearsonr(log_sigmas[:, 0], log_sigmas[:, 1])
    ax.text(0.05, 0.95, f'ρ = {rho:.2f}', transform=ax.transAxes, verticalalignment='top')
    return ax


def plot_loss_vs_scale(ax, scale_vals, min_losses, opt_scale):
    """Optimization loss vs the wrap-Gaussian scale parameter, with optimum marked.

    Parameters
    ----------
    ax : matplotlib axis or None
        If None, a fresh ``plt.subplots(figsize=(4, 3))`` is created.
    scale_vals : ndarray, shape (N_scale,)
    min_losses : ndarray, shape (N_scale,)
        Minimum loss at each scale (after marginalizing over bias, beta).
    opt_scale : float
        Optimal scale value (where to draw the vertical line).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3))
    # A plain solid black line: this is a smooth measured error curve, so markers
    # are unjustified per the plotting convention. clip_on=False so the peak of the
    # curve (which touches the y-limit) is not sliced off by the axes box.
    ax.plot(scale_vals, min_losses, color='black', clip_on=False, label='Loss')
    ax.axvline(x=opt_scale, color='black', linestyle='--', alpha=1,
               label=f'Min Scale={opt_scale:.2f}')
    ax.set_xlabel("$\\sigma$ [Fourier decay scale]")
    ax.set_ylabel('min. error at $\\sigma$')
    ax.set_ylim(0, 0.11)
    return ax


def plot_loss_landscape(ax, loss_slice, bias_vals, beta_vals, opt_bias, opt_beta):
    """2D loss-landscape heatmap in (bias, beta) at the optimal scale, with optimum marked.

    Parameters
    ----------
    ax : matplotlib axis or None
        If None, a fresh ``plt.subplots(figsize=(4, 3))`` is created.
    loss_slice : ndarray, shape (N_bias, N_beta)
        Loss landscape at the optimal scale.
    bias_vals : ndarray, shape (N_bias,)
    beta_vals : ndarray, shape (N_beta,)
    opt_bias, opt_beta : float
        Optimal (bias, beta).

    Returns
    -------
    ax : matplotlib axis
    cbar : matplotlib colorbar
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(loss_slice.T, origin='lower', aspect='auto',
                   extent=[bias_vals[0], bias_vals[-1], beta_vals[0], beta_vals[-1]],
                   cmap=style.field_cmap('aurora'),
                   vmin=0, vmax=loss_slice.max(), interpolation="None")
    ax.plot(opt_bias, opt_beta, 'w.', markersize=5,
            label=f'Min (bias={opt_bias:.2f}, beta={opt_beta:.2f})')
    cbar = plt.colorbar(im, ax=ax, label='error', location='top',
                        fraction=style.CBAR_FRACTION, pad=style.CBAR_PAD,
                        aspect=style.CBAR_ASPECT)
    cbar.outline.set_linewidth(0.5)
    ax.set_xlabel('$b$ [soft threshold]')
    ax.set_ylabel("$\\beta$ [sharpness]")
    ax.set_ylim(0, 10)
    ax.set_xlim(0, 5)
    return ax, cbar


# --- Tuning-curve panels ---------------------------------------------------

def plot_data_tuning_curves(axes, mouse_nums_to_plot, mouse_nums, mouse_names,
                            plain_tcs, scalings, Phi_m_data, fold_colors=None):
    """Per-mouse tuning curves: plain CV-fold pairs + smoothed canonical.

    For each mouse in ``mouse_nums_to_plot``, randomly samples
    ``axes.shape[1]`` units and plots both CV-fold tuning curves plus the
    smoothed canonical curve.

    Parameters
    ----------
    axes : ndarray of matplotlib axes, shape (N_mice, N_cols)
    mouse_nums_to_plot : sequence of int
        Mouse ids to render (one per row).
    mouse_nums : ndarray, shape (N_curves,)
        Per-cell mouse-id labels.
    mouse_names : ndarray, shape (N_curves,)
        Per-cell mouse names (strings).
    plain_tcs : ndarray, shape (N_curves, 2, N_theta)
        Plain (per-CV-fold) tuning curves.
    scalings : ndarray, shape (N_curves,)
        Per-cell scaling factors (mean-normalize).
    Phi_m_data : ndarray, shape (N_curves, N_theta)
        Smoothed canonical tuning curves.
        fold_colors : tuple of 2 str or None
        Colours for the two cross-validation folds. None keeps the original
        bright pair; Fig. 2 passes the muted published pair.
"""
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)
    N_theta = Phi_m_data.shape[1]
    theta = np.arange(N_theta) * 2 * np.pi / N_theta
    # CV-fold colours. Defaults are the original bright pair; the published
    # figure passes the muted pair.
    c1, c2 = fold_colors if fold_colors is not None else (style.FOLD_A, style.FOLD_B)

    for row_idx, mouse_num in enumerate(mouse_nums_to_plot):
        mouse_mask = mouse_nums == mouse_num
        mouse_name = mouse_names[mouse_mask][0]

        np.random.seed(row_idx)
        cell_idx_to_show = np.random.choice(np.arange(mouse_mask.sum()),
                                            replace=False, size=axes.shape[1])

        for i, ax in zip(cell_idx_to_show, axes[row_idx]):
            plain_tcs_to_show = plain_tcs[mouse_mask][i] * scalings[mouse_mask][i]
            ax.plot(theta, plain_tcs_to_show[0], clip_on=True, color=c1, lw=1, label="partition 1")
            ax.plot(theta, plain_tcs_to_show[1], clip_on=True, color=c2, lw=1,
                    label="partition 2")
            ax.plot(theta, Phi_m_data[mouse_mask][i], color=style.DATA,
                    clip_on=True, label="smoothed")
            if ax == axes[row_idx][0]:
                ax.set_ylabel(f'mouse\n{mouse_name}')

    setup_tc_axes(axes)


def plot_model_tuning_curves(axes, opt_params, seed=42):
    """Random samples from the generative process at the optimum parameters.

    One independent sample per sub-panel, so every panel shows a DISTINCT curve.
    Until 2026-08-02 this drew a fixed 16 samples and then picked one per panel
    with ``np.random.choice``, i.e. WITH replacement, which on the published 4x6
    grid gave 24 panels holding only 11 distinct curves with pixel-identical
    duplicates. That defeats the panel's purpose, which is to show the diversity
    of the process, so the sample count now follows the number of panels and each
    sample is used exactly once. The samples are i.i.d., so consuming them in
    order is already a random assignment and no extra permutation is needed.

    Parameters
    ----------
    axes : ndarray of matplotlib axes (1-D or 2-D)
    opt_params : sequence of 3 floats
        (scale, bias, beta).
    seed : int
        Seed for the global RNG that ``gp_opt.gen_Phi_m`` draws from, set here so
        the panel is reproducible whatever the caller did beforehand.
    """
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)
    set_global_seed(seed)
    theta = np.arange(100) * 2 * np.pi / 100
    Phi_m = gp_opt.gen_Phi_m(*opt_params, N_samples=axes.size, N_theta=100)
    Phi_m = Phi_m / Phi_m.mean(1, keepdims=True)

    for i, ax in enumerate(axes.flatten()):
        ax.plot(theta, Phi_m[i], color='black', clip_on=False)

    setup_tc_axes(axes)


# --- Mean / std profile panels --------------------------------------------

def plot_profile(ax, theta, profiles, mean_profile=None, profile_label='',
                 mean_label='mean', color=style.INDIVIDUAL_DATA, model_profile=None):
    """(internal) Plot individual + mean profiles with optional model overlay.

    Parameters
    ----------
    ax : matplotlib axis
    theta : ndarray, shape (N_theta,)
    profiles : ndarray, shape (N_curves, N_theta)
    mean_profile : ndarray or None, shape (N_theta,)
    profile_label : str
        Y-axis label.
    mean_label : str
        (Unused; kept for backward compat with signature.)
    color : str
        Color for the individual-profile lines.
    model_profile : ndarray or None, shape (N_theta,)
    """
    ax.plot(theta, profiles.T, alpha=.5, c=color, zorder=0, clip_on=False)
    ax.plot(theta, profiles[0], alpha=.5, c=color, zorder=0, clip_on=False, label='individual mice')
    if mean_profile is not None:
        # Only lighten and widen the data curve when a nearly coincident model
        # curve is present. Data-only panels retain the bold charcoal aggregate.
        mean_color = style.DATA_UNDERLAY if model_profile is not None else style.DATA
        mean_lw = style.OVERLAP_DATA_LW if model_profile is not None else 1.3
        ax.plot(theta, mean_profile, c=mean_color, zorder=2, lw=mean_lw,
                label='mouse average')
    if model_profile is not None:
        # A solid persimmon center stroke over the wider stone data stroke makes
        # both nearly coincident curves continuously visible at print size.
        ax.plot(theta, model_profile, color=style.MODEL, zorder=3,
                lw=style.OVERLAP_MODEL_LW, label='generative process')
    setup_profile_axes(ax, profile_label)
    ax.set_xlabel('Δθ')


def plot_mean_std_profiles(ax_mu, ax_std, all_mus, all_stds, opt_params=None):
    """Mean + std profile panels, with optional generative-process overlay.

    Parameters
    ----------
    ax_mu, ax_std : matplotlib axes
    all_mus, all_stds : ndarray, shape (n_mice, N_theta)
        Per-mouse mean and std profiles (from
        ``analysis.compute_mean_and_std_profiles_for_all_mice``).
    opt_params : sequence of 3 floats or None
        If given, also draws a model curve from
        ``gp_opt.gen_Phi_m(*opt_params, ...)``.
    """
    n_angles = all_mus.shape[1]
    mid_idx = n_angles // 2
    theta = np.arange(-mid_idx, n_angles - mid_idx) * 2 * np.pi / n_angles

    model_profiles = None
    if opt_params is not None:
        model_data = gp_opt.gen_Phi_m(*opt_params, N_samples=100000, N_theta=n_angles,
                                      device=None, normalize=True)
        model_mu, model_std = analysis.compute_mean_and_std_profiles(model_data)
        model_profiles = (np.roll(model_mu, -mid_idx), np.roll(model_std, -mid_idx))

    plot_profile(ax_mu, theta, np.roll(all_mus, -mid_idx, axis=1),
                 mean_profile=np.roll(all_mus.mean(0), -mid_idx),
                 profile_label='mean',
                 model_profile=model_profiles[0] if model_profiles else None)

    plot_profile(ax_std, theta, np.roll(all_stds, -mid_idx, axis=1),
                 mean_profile=np.roll(all_stds.mean(0), -mid_idx),
                 profile_label='std. dev.',
                 model_profile=model_profiles[1] if model_profiles else None)


# --- Covariance + comparison panels ---------------------------------------

def plot_covariance_comparison(ax, Gamma_phi_data, Gamma_phi_opt):
    """Overlay data + generative-process covariance functions.

    Parameters
    ----------
    ax : matplotlib axis
    Gamma_phi_data : ndarray, shape (N_theta,)
    Gamma_phi_opt : ndarray, shape (N_theta,)
    """
    # These curves lie almost exactly on top of each other. A solid persimmon center
    # stroke over a wider stone data stroke leaves both curves continuously visible
    # without offsetting either scientific quantity.
    theta = np.linspace(-np.pi, np.pi, 100)
    ax.plot(theta, np.roll(Gamma_phi_data, -50), lw=style.OVERLAP_DATA_LW,
            label="data",
            color=style.DATA_UNDERLAY, zorder=1)
    ax.plot(theta, np.roll(Gamma_phi_opt, -50), lw=style.OVERLAP_MODEL_LW,
            label="gen.\nprocess", color=style.MODEL, zorder=2)
    ax.legend(loc='upper right', bbox_to_anchor=[1.1, 0.9])
    ax.set_ylim(0, 4)
    ax.set_ylabel("$\\Gamma_{\\phi}(\\Delta\\theta)$")
    ax.set_xlabel("$\\Delta\\theta$")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels(["$-\\pi$", "$0$", "$\\pi$"])


def plot_peak_count_comparison(ax, data_counts, model_counts):
    """Side-by-side bar chart of peak-count distributions: data vs model.

    Parameters
    ----------
    ax : matplotlib axis
    data_counts, model_counts : ndarray, shape (N_bins,)
    """
    bar_width = 0.35
    positions = np.arange(1, len(data_counts) + 1)
    ax.bar(positions - bar_width / 2 - 0.01, data_counts, bar_width,
           color=data_color, label="Data", edgecolor='white', zorder=2)
    ax.bar(positions + bar_width / 2 + 0.01, model_counts, bar_width,
           color=gen_process_color, label="Generative\nprocess", edgecolor='white', zorder=2)
    ax.set_xlim(0.5, len(data_counts) + 0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks(positions)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_ylabel("frac. neurons")
    ax.set_xlabel("peak count")


def plot_peak_value_histograms(ax, data_peaks, model_peaks):
    """Peak-value distribution: data vs model.

    Parameters
    ----------
    ax : matplotlib axis
    data_peaks, model_peaks : ndarray
        Flat arrays of peak heights.
    """
    ax.hist(data_peaks, bins=80, range=(0, 15), density=True,
            color=data_color, label="data", clip_on=False)
    ax.hist(model_peaks, bins=80, range=(0, 15), density=True,
            color=gen_process_color, histtype='step', label="model", clip_on=False)
    ax.set_ylim(0, 1.)
    ax.set_xlim(0, 15.)
    ax.set_xticks([0, 5, 10, 15])
    ax.set_xlabel("peak height")
    ax.set_ylabel("density")


def plot_symmetry_histogram(ax, data_scores, model_scores, ylim=None):
    """Flip-symmetry score distribution: data vs model.

    Parameters
    ----------
    ax : matplotlib axis
    data_scores, model_scores : ndarray
        Per-curve Pearson r in [-1, 1].
    ylim : tuple or None
        Optional y-axis limits.
    """
    ax.hist(data_scores, density=True, range=(-1., 1.), bins=80,
            color=data_color, label="data")
    ax.hist(model_scores, density=True, range=(-1., 1.), bins=80,
            histtype='step', color=gen_process_color, label="model")
    if ylim:
        ax.set_ylim(ylim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('density')
    ax.set_xlabel("$\\rho^{flip}$")


def plot_information_histogram(ax, data_info, model_info):
    """Bits-per-spike distribution: data vs model.

    Parameters
    ----------
    ax : matplotlib axis
    data_info, model_info : ndarray
    """
    ax.hist(data_info, density=True, range=(0, 4), bins=80,
            color=data_color, label="data")
    ax.hist(model_info, density=True, range=(0, 4), bins=80,
            histtype='step', color=gen_process_color, label="model")
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xlim(0, 3.5)
    ax.set_ylabel("density")
    ax.set_xlabel("$I_{HD}$")


# --- COM / covariance grid panels -----------------------------------------

def plot_mouse_coms(ax, mouse_coms, null=False, color=style.COM_GREEN, seed=42):
    """Per-mouse circular center-of-mass scatter (one row per mouse).

    Parameters
    ----------
    ax : matplotlib axis
    mouse_coms : list of ndarray
        Length-n_mice; from ``analysis.compute_coms_for_all_mice``.
    null : bool, default False
        If True, replace each mouse's COMs with a uniform-random null draw
        of the same size (for a visual significance baseline).
    color : str
        Scatter color.
    seed : int
        Seed for the global RNG, set only when ``null`` is True, which is the
        only branch that draws random numbers. Before 2026-08-02 the null draw
        used whatever global state it inherited, so the panel changed on every
        run, contradicting the paper's Reproducibility statement.
    """
    n_mice = len(mouse_coms)
    if null:
        set_global_seed(seed)
    for mouse_idx in range(n_mice):
        if null:
            coms = 2 * np.pi * np.random.rand(len(mouse_coms[mouse_idx]))
        else:
            coms = mouse_coms[mouse_idx]
        ax.scatter(coms, np.ones(len(coms)) * (mouse_idx + 1),
                   lw=0, s=2., color=color, alpha=1, clip_on=False)

    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(0.5, n_mice + 0.5)
    ax.set_xticks([0, np.pi, 2 * np.pi])
    ax.set_xticklabels(['$0$', '$\\pi$', '$2\\pi$'])
    ax.set_ylabel('mouse index')
    ax.set_xlabel('center of mass')
    ax.set_title('random uniform' if null else 'data')


def plot_covariance_grid(axes, mouse_nums, mouse_nums_to_plot, mouse_names,
                         Phi_m_data, subsample_sizes):
    """Grid of per-mouse subsampled covariance matrices, demonstrating convergence.

    Rows = mice in ``mouse_nums_to_plot``; columns = subsample sizes in
    ``subsample_sizes``. Each cell shows the empirical covariance of a
    random N_sub subset of that mouse's tuning curves.

    Parameters
    ----------
    axes : ndarray of matplotlib axes, shape (N_mice_to_plot, N_subsample_sizes)
    mouse_nums : ndarray
    mouse_nums_to_plot : sequence of int
    mouse_names : ndarray
    Phi_m_data : ndarray, shape (N_curves, N_theta)
    subsample_sizes : sequence of int

    Returns
    -------
    im : matplotlib image handle (last drawn)
    """
    np.random.seed(42)
    vmax = 5

    for i, mouse_num in enumerate(mouse_nums_to_plot):
        mouse_tcs = Phi_m_data[mouse_nums == mouse_num]
        num_tcs = mouse_tcs.shape[0]
        mouse_name = mouse_names[mouse_nums == mouse_num][0]
        if i == 0:
            for j, N_sub in enumerate(subsample_sizes):
                axes[i, j].set_title("$N_{sub} =" + str(N_sub) + "$")
        axes[i, 0].set_ylabel(f'mouse\n{mouse_name}')

        for j, N_sub in enumerate(subsample_sizes):
            sub_tcs = mouse_tcs[np.random.choice(np.arange(num_tcs), replace=False, size=N_sub)]
            cov_sub = (sub_tcs.T @ sub_tcs) / N_sub
            # origin='lower' so row 0 sits at theta' = 0, matching the extent's
            # y axis. Without it imshow's default puts row 0 at the top, the image
            # renders mirrored against its own tick labels, and the true diagonal
            # draws as the anti-diagonal.
            im = axes[i, j].imshow(cov_sub, extent=[0, 2 * np.pi, 0, 2 * np.pi],
                                   origin='lower',
                                   vmin=0, vmax=vmax, cmap=style.COVARIANCE_CMAP,
                                   interpolation="nearest")

            ax = axes[i, j]
            if i == len(mouse_nums_to_plot) - 1:
                ax.set_xticks([0, np.pi, 2 * np.pi])
                ax.set_xticklabels(["$0$", "$\\pi$", "$2\\pi$"])
            else:
                ax.set_xticks([])
            if j == 0:
                ax.set_yticks([0, np.pi, 2 * np.pi])
                ax.set_yticklabels(["$0$", "$\\pi$", "$2\\pi$"])
            else:
                ax.set_yticks([])
            ax.set_aspect(1.)
            ax.text(0.5, 0, "$\\theta$", ha='center', va='bottom',
                    c='white', transform=ax.transAxes, size=6)
            ax.text(0, 0.5, "$\\theta'$", ha='left', va='center',
                    c='white', transform=ax.transAxes, size=6)
    return im


# --- Generative-process example panels ------------------------------------

def plot_cov_examples(ax):
    """Wrap-Gaussian covariance functions at a few scales.

    Parameters
    ----------
    ax : matplotlib axis
    """
    scales = [0.8, 1.05, 1.25, 1.6, 2.25]
    theta = np.linspace(-np.pi, np.pi, 100)
    # Fig 4's cool palette: the sigma family takes 'ocean'.
    colors = style.sweep_colors(len(scales), 'ocean')
    for scale, color in zip(scales, colors):
        Gamma_x, _ = tc.wrapped_gaussian_cov(1, scale=scale, N_theta=len(theta))
        ax.plot(theta, np.roll(Gamma_x, len(theta) // 2), color=color, alpha=0.8)
    ax.set_xlabel(r"$\Delta \theta$")
    ax.set_ylabel(r"$\Gamma_x(\Delta \theta)$")
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels([r"$-\pi$", "$0$", r"$\pi$"])
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(0, 1.02)
    ax.set_title("covariance")


def plot_pre_examples(ax):
    """Sample pre-nonlinearity tuning curves at a few scales, vertically offset.

    Parameters
    ----------
    ax : matplotlib axis
    """
    scales = [0.8, 1.05, 1.25, 1.6, 2.25]
    theta = np.linspace(-np.pi, np.pi, 100)
    # Same 'ocean' ramp as panel A. The per-curve sigma labels below pick up the
    # same color, so text and curve stay in register; the ramp's capped lightness
    # is what keeps the largest-sigma label legible.
    colors = style.sweep_colors(len(scales), 'ocean')
    for i, (scale, color) in enumerate(zip(scales, colors)):
        _, Gamma_x_ft = tc.wrapped_gaussian_cov(1, scale=scale, N_theta=len(theta))
        X = tc.sample_tcs(N=10, Gamma_ft=Gamma_x_ft).T
        ax.plot(theta + np.pi, 5 * i + X[:5].T, color=color, alpha=0.8)
        # sigma labels sit just LEFT of the left spine, tilted so they fit in the
        # narrow gap without stealing the width the panel now extends into on the
        # right (see Fig. 4 layout).
        ax.text(-0.15, 5 * i + np.mean(X[:5]),
                f'$\\sigma$ = {scale:.2f}', va='center', ha='right',
                rotation=35, rotation_mode='anchor', color=color, size=6)
    ax.set_xlabel(r"$\theta$")
    ax.set_xticks([0, np.pi, 2 * np.pi])
    ax.set_xticklabels(["$0$", r"$\pi$", r"$2\pi$"])
    ax.set_xlim(0, 2 * np.pi)
    ax.set_yticks([])
    ax.set_title("samples")


def plot_nonlin_examples(ax):
    """Softplus nonlinearity examples at a few (bias, beta) pairs, plus Gaussian prior overlay.

    Parameters
    ----------
    ax : matplotlib axis
    """
    x_vals = np.linspace(-4, 4, 1000)
    # Beta takes the same warm 'ember' ramp used for beta in the DMFT figure,
    # while panels A and B use the cool 'ocean' ramp for sigma.
    # Each of the three thresholds b redraws the same beta-colored fan, shifted
    # horizontally; the b arrow in the figure marks that shift.
    betas = [1.3, 2, 3.5, 10]
    beta_colors = style.sweep_colors(len(betas), 'ember')
    for bias in [0, 1.5, 2.5]:
        for beta, color in zip(betas, beta_colors):
            y_vals = fn.softplus(x_vals - bias, beta=beta) * 0.07
            ax.plot(x_vals, y_vals, color=color, alpha=0.85)

    pdf = (1. / np.sqrt(2 * np.pi)) * np.exp(-x_vals**2 / 2.)
    ax.plot(x_vals, pdf, zorder=0, color='0.25', alpha=1)
    ax.fill_between(x_vals, 0, pdf, zorder=0, color="0.5", alpha=0.1)

    ax.set_xlim(-4, 4)
    ax.set_xlabel(r"$x(\theta)$")
    ax.set_ylabel(r"$\phi(\theta)$")
    ax.set_title("nonlinearity")


# --------------------------------------------------------------------------- #
# Classical ring-attractor schematic panels (Fig. 2A-C)
#
# These three are drawn fully in code, so the whole figure is reproducible
# from the notebook.
# --------------------------------------------------------------------------- #

def _shaded_sphere_rgba(size=320, base=(0.70, 0.70, 0.72),
                        light=(-0.40, 0.52)):
    """(internal) Lambertian-shaded sphere as an RGBA array.

    Rendered once and stamped at each soma position, which is far cheaper than
    building a gradient per neuron. Diffuse-dominant with only a broad, low
    specular term, so the somata read matte rather than glossy.

    Parameters
    ----------
    size : int
        Pixel resolution of the (square) sprite.
    base : tuple of 3 floats
        Base RGB colour of the sphere.
    light : tuple of 2 floats
        (x, y) of the unit light direction; z is inferred.

    Returns
    -------
    ndarray, shape (size, size, 4)
        RGBA image, fully transparent outside the disc.
    """
    y, x = np.mgrid[-1:1:size * 1j, -1:1:size * 1j]
    rr = x ** 2 + y ** 2
    inside = rr <= 1.0
    z = np.sqrt(np.clip(1 - rr, 0, 1))
    lx, ly = light
    lz = np.sqrt(max(0.0, 1 - lx ** 2 - ly ** 2))
    lam = np.clip(x * lx + y * ly + z * lz, 0, 1)
    shade = 0.50 + 0.58 * lam
    spec = lam ** 12
    rgba = np.zeros((size, size, 4))
    for c in range(3):
        rgba[..., c] = np.clip(base[c] * shade + 0.18 * spec, 0, 1)
    rim = np.clip((rr - 0.74) / 0.26, 0, 1) * inside
    rgba[..., :3] *= (1 - 0.28 * rim[..., None])
    rgba[..., 3] = inside.astype(float)
    return rgba


def plot_ring_schematic(ax, n_neurons=19, soma_r=0.155, e_targets=(1, 2),
                        i_targets=(3, 4, 5), e_color=None,
                        i_color=None, label_fontsize=20,
                        theta_fontsize=15):
    """Classical ring-attractor connectivity schematic (Fig. 2A).

    Somata are evenly spaced on a unit ring, with excitatory (short-range) and
    inhibitory (longer-range) projections drawn from the top neuron. Every
    connection begins and ends on the *inner tangent circle* of radius
    ``1 - soma_r``, the circle inscribed against all somata, so terminals meet
    their target squarely: excitatory discs are centred on the soma edge and
    inhibitory bars lie along the tangent there. Terminal placement is therefore
    parametric, and re-anchors itself if ``soma_r`` or ``n_neurons`` changes.

    Parameters
    ----------
    ax : matplotlib axes
    n_neurons : int
        Number of somata on the ring.
    soma_r : float
        Soma radius in ring-radius units. Somata stay visibly separated while
        ``2 * soma_r < 2 * sin(pi / n_neurons)``.
    e_targets, i_targets : sequence of int
        Neuron offsets (both signs are drawn) receiving excitatory and
        inhibitory projections from the source neuron.
    e_color, i_color : str or None
        Optional override colours for the E/I labels. When None (default), the
        labels take ``style.E_COLOR`` / ``style.I_COLOR``, the persimmon and
        petrol anchors that also define the signed-matrix map.
    """
    r_inner = 1.0 - soma_r
    ang = np.pi / 2 - 2 * np.pi * np.arange(n_neurons) / n_neurons
    pos = np.column_stack([np.cos(ang), np.sin(ang)])

    def inner_point(p):
        u = p / np.linalg.norm(p)
        return u * r_inner, u

    src_in, _ = inner_point(pos[0])

    # Edge colours are tints of the two semantic anchors style.E_COLOR /
    # style.I_COLOR, so the whole set reads as one persimmon-to-petrol sweep.
    # Nearest-target excitation starts at full-strength persimmon and fades toward
    # the center; inhibition continues from pale petrol to full-strength petrol.
    # The sweep therefore passes through light in the middle instead of becoming
    # two disconnected fans. The larger inhibitory fan makes the Mexican-hat
    # profile intuitive and echoes the signed-matrix map in panel C.
    def _tint(hex_color, w):
        """Blend a colour toward white by fraction ``w``."""
        return tuple(np.asarray(_mcolors.to_rgb(hex_color)) * (1 - w) + w)

    e_shades = [_tint(style.E_COLOR, w) for w in np.linspace(0.0, 0.45, len(e_targets))]
    i_shades = [_tint(style.I_COLOR, w) for w in np.linspace(0.45, 0.0, len(i_targets))]

    # projections, drawn beneath the somata (thinner lines than before, so the
    # terminal discs/bars read cleanly against them)
    for offsets, shades, kind, pull in ((e_targets, e_shades, 'E', 0.22),
                                        (i_targets, i_shades, 'I', 0.52)):
        for k, col in zip(offsets, shades):
            for sgn in (+1, -1):
                p_term, u_rad = inner_point(pos[(sgn * k) % n_neurons])
                # quadratic Bezier whose control point is pulled toward the ring
                # centre, making the inward bow exact rather than sign-inferred
                ctrl = ((src_in + p_term) / 2) * (1.0 - pull)
                ax.add_patch(PathPatch(
                    Path([src_in, ctrl, p_term],
                         [Path.MOVETO, Path.CURVE3, Path.CURVE3]),
                    fill=False, edgecolor=col, lw=1.52, zorder=3))
                if kind == 'E':
                    ax.add_patch(Circle(p_term, 0.050, facecolor=col,
                                        edgecolor='none', zorder=6))
                else:
                    t = np.array([-u_rad[1], u_rad[0]])
                    ax.plot(*np.column_stack([p_term - t * 0.066,
                                              p_term + t * 0.066]),
                            color=col, lw=1.6, solid_capstyle='butt',
                            zorder=6)

    # labels take the full-strength anchors, which are also the most saturated
    # drawn edges: E the nearest excitatory shade, I the farthest inhibitory one
    e_color = e_color if e_color is not None else style.E_COLOR
    i_color = i_color if i_color is not None else style.I_COLOR

    sprite = _shaded_sphere_rgba()
    for p in pos:
        ax.imshow(sprite, extent=[p[0] - soma_r, p[0] + soma_r,
                                  p[1] - soma_r, p[1] + soma_r],
                  zorder=5, interpolation='bilinear')

    ax.text(-0.20, -0.06, 'E', color=e_color, fontsize=label_fontsize,
            fontweight='bold', ha='center', va='center', zorder=7)
    ax.text(0.20, -0.06, 'I', color=i_color, fontsize=label_fontsize,
            fontweight='bold', ha='center', va='center', zorder=7)

    # theta direction arrow, sweeping counter-clockwise up the right-hand side
    # from about 5 o'clock to about 2:45. Clock face -> degrees: 3 o'clock is 0
    # and each hour is 30 degrees, so 5:00 is -60 and 2:45 is +7.5.
    a_tail, a_head, rr = np.deg2rad(-60.0), np.deg2rad(7.5), 1.30
    arrow = FancyArrowPatch((rr * np.cos(a_tail), rr * np.sin(a_tail)),
                            (rr * np.cos(a_head), rr * np.sin(a_head)),
                            connectionstyle='arc3,rad=0.26', arrowstyle='-|>',
                            mutation_scale=15, color='black', lw=1.6,
                            shrinkA=0, shrinkB=0)
    arrow.set_joinstyle('miter')   # sharp head, not matplotlib's rounded default
    arrow.set_capstyle('butt')
    ax.add_patch(arrow)
    a_mid, r_lab = (a_tail + a_head) / 2, 1.56
    ax.text(r_lab * np.cos(a_mid), r_lab * np.sin(a_mid), r"$\theta$",
            fontsize=theta_fontsize, ha='center', va='center')

    # tight bounds: the ring spans +/-1.14, so only the theta arrow and its label
    # need extra room, and only on the right
    # y-window is exactly the ring's outer diameter, so the ring fills the panel
    # height; that lets the caller match it to the heatmap's height in Fig. 2.
    lim = 1.0 + soma_r
    ax.set_xlim(-lim - 0.02, lim + 0.46)   # extra room on the right for theta
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.axis('off')


def plot_circulant_weights(ax, n_theta=1000, width=1.0, offset=0.5, title_fontsize=12):
    """Circulant Mexican-hat weight matrix J(theta - theta') (Fig. 1, panel b).

    Schematic only, with no colorbar and no meaningful weight units.

    The profile is a Gaussian minus a constant, ``exp(-(dt/width)**2) - offset``,
    so it is positive near the diagonal (local excitation) and negative beyond
    ``dt = width * sqrt(-ln offset)`` (longer-range inhibition), which is what
    the caption and the Introduction claim.

    Before 2026-08-02 the offset was absent, so the function was strictly
    positive with no inhibitory surround at all, and it read as a Mexican hat
    only because an autoscaled diverging colormap painted its near-zero tail
    with the negative-end colour. Subtracting ``offset = 0.5`` and taking colour
    limits symmetric about zero makes the sign change real and puts it exactly
    where an autoscaled midpoint would fall, so the picture looks the same
    while the colour scale is honest.

    Parameters
    ----------
    ax : matplotlib axes
    n_theta : int
        Grid resolution along each angular axis.
    width : float
        Gaussian width of the excitatory lobe, in radians.
    offset : float
        Constant subtracted from the Gaussian. Sets the sign change, and must be
        0.5 to reproduce the pre-2026-08-02 rendering exactly.
    """
    t = np.linspace(0, 2 * np.pi, n_theta)
    dt = np.abs(t[:, None] - t[None, :])
    dt = np.minimum(dt, 2 * np.pi - dt)
    J = np.exp(-(dt / width) ** 2) - offset
    lim = np.abs(J).max()
    # origin='lower' so row 0 sits at theta' = 0, matching the extent's y axis and
    # putting the excitatory band on the true diagonal rather than the anti-diagonal.
    ax.imshow(J, cmap=style.DIVERGING_CMAP, vmin=-lim, vmax=lim,
              extent=[0, 2 * np.pi, 0, 2 * np.pi], aspect=1.0, origin='lower')
    ax.set_xticks([0, np.pi, 2 * np.pi])
    ax.set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
    ax.set_yticks([0, np.pi, 2 * np.pi])
    ax.set_yticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
    ax.set_title(r"$J(\theta-\theta')$", fontsize=title_fontsize)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\theta'$", rotation=0, va='center')


def plot_circulant_tuning_curves(ax, n_neurons=50, step=5, width2=40.0):
    """Translation-invariant tuning curves predicted by classical models (Fig. 2C).

    Identical bumps tiling the theta axis, shaded light-to-dark so the
    neuron-specific shift is legible.

    Parameters
    ----------
    ax : matplotlib axes
    n_neurons : int
        Number of neurons spanning the ring.
    step : int
        Plot every ``step``-th neuron's tuning curve.
    width2 : float
        Squared Gaussian width, in neuron-index units.
    """
    idx = np.arange(n_neurons)
    for i in range(0, n_neurons, step):
        ax.plot(idx, np.exp(-(idx - i) ** 2 / width2),
                color=str(0.75 * (1 - i / n_neurons)), clip_on=False, lw=1.5)
    ax.set_xticks([0, n_neurons // 2 - 1, n_neurons - 1])
    ax.set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
    ax.set_yticks([])
    ax.set_ylabel("firing rate")
    ax.set_xlabel(r"head direction $\theta$")
    ax.set_xlim(0, n_neurons - 1)
    ax.set_ylim(0, 1)


def plot_flow_schematic(ax, box_color=style.SCHEMATIC_BOX, edge_color='black',
                        fontsize=6.4, arrow_label_fontsize=7.4,
                        box_w=0.246, box_h=0.345, lw=1.1):
    """Flow of the paper: data to data-driven RNN, generative process, DMFT.

    Laid out three columns by two rows. Two of its neighbours in the top band
    are aspect-locked, so shortening that band widens this panel; the wide
    layout is what keeps the boxes undistorted at the resulting aspect (~2.3).

    Parameters
    ----------
    ax : matplotlib axes
    box_color, edge_color : str
        Fill and outline of the boxes.
    fontsize : float
        Box label size.
    arrow_label_fontsize : float
        Size of the N -> infinity label on the synth -> dmft arrow.
    box_w, box_h : float
        Box size in axes fractions.
    lw : float
        Outline and arrow width.
    """
    # columns are unevenly spaced so that synth -> dmft, the one arrow carrying
    # a label, gets the longer run and the label is not crowded by either box
    x1, x2, x3 = 0.133, 0.473, 0.867           # column centres
    y1, y2 = 0.790, 0.210                      # row centres

    nodes = {
        'data':      (x1, y1, 'Neural data'),
        'ddrnn':     (x2, y1, 'Data-derived\nRNN'),
        'gen':       (x1, y2, 'Generative\nprocess'),
        'synth':     (x2, y2, 'Synthetic\ndata-derived\nRNN'),
        'dmft':      (x3, y2, 'Dynamical\nmean-field\ntheory'),
    }
    for x, y, label in nodes.values():
        ax.add_patch(FancyBboxPatch((x - box_w / 2, y - box_h / 2), box_w, box_h,
                                    boxstyle='round,pad=0,rounding_size=0.028',
                                    facecolor=box_color, edgecolor=edge_color,
                                    linewidth=lw, zorder=2))
        ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
                zorder=3, linespacing=1.25)

    def arrow(a, b, label=None):
        (xa, ya, _), (xb, yb, _) = nodes[a], nodes[b]
        if abs(ya - yb) < 1e-9:                       # horizontal
            start, end = (xa + box_w / 2, ya), (xb - box_w / 2, yb)
        else:                                          # vertical
            sgn = -1 if yb < ya else 1
            start, end = (xa, ya + sgn * box_h / 2), (xb, yb - sgn * box_h / 2)
        p = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=8,
                            color=edge_color, lw=lw, shrinkA=0, shrinkB=0,
                            zorder=1)
        p.set_joinstyle('miter')
        ax.add_patch(p)
        if label:
            if abs(ya - yb) < 1e-9:            # horizontal arrow: label above it
                # nudge the label a hair left of the raw gap midpoint: the mathtext
                # advance width centres the N->infinity glyph run slightly right of
                # its ink, so this lands it visually centred between the two boxes.
                ax.text((start[0] + end[0]) / 2 - 0.0015, ya + 0.050, label,
                        ha='center', va='bottom', fontsize=arrow_label_fontsize)
            else:                              # vertical arrow: label beside it
                ax.text(xa + 0.035, (start[1] + end[1]) / 2, label,
                        ha='left', va='center', fontsize=arrow_label_fontsize)

    arrow('data', 'ddrnn')
    arrow('data', 'gen')
    arrow('gen', 'synth')
    arrow('synth', 'dmft', r"$N\!\to\!\infty$")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def plot_embedding_schematic(ax, aspect=0.77, ordered=style.EMBED_ORDERED,
                             disordered=style.EMBED_DISORDERED,
                             grey="#E8E8E8", seed=0):
    """Ring-embedding schematic: geometric ring -> ordered vs disordered tuning.

    A warped closed ring (circular symmetry) is mapped, through a Fourier embedding,
    to ordered tuning curves that are identical up to shifts, and, through a Gaussian
    embedding, to disordered heterogeneous tuning curves. Portrait "Y" layout (ring
    on top, two embedding branches fanning down) so it fits as a full-height left
    panel of the spectral-structure figure. ``aspect`` is the panel's width/height; pass the real value so the
    equal-aspect drawing fills the panel.

    Parameters
    ----------
    ax : matplotlib axes
    aspect : float
        Panel width/height, so the drawing fills the panel without distortion.
    ordered, disordered, grey : str
        Colours for the ordered (Fourier) branch, the disordered (Gaussian) branch,
        and the tuning-box fill. The two branch colours are the SAME pair the
        spectral panels use for J and its circulant surrogate, so one colour
        means one thing across the whole figure.
    seed : int
        Seeds the disordered example curves (determinism).
    """
    # Orthographic view angles (azimuth A, elevation E) shared by the ring curve
    # and the axis gizmo, so the gizmo honestly labels the ring's orientation.
    A_view, E_view = np.deg2rad(50.0), np.deg2rad(28.0)

    def project(X, Y, Z, A=A_view, E=E_view):
        # Orthographic projection of a 3D point onto screen coords (u, v).
        u = -np.sin(A) * X + np.cos(A) * Y
        v = (-np.cos(A) * np.sin(E) * X - np.sin(A) * np.sin(E) * Y
             + np.cos(E) * Z)
        return u, v

    def ring_xy(n=900):
        # True orthographic projection of the embedded circle
        # (X, Y, Z) = (sin theta, cos theta, cos 2theta): a circle carrying a
        # second harmonic, so the projected loop bobs up and down twice and
        # crosses itself, reading as a warped ring rather than a flat circle.
        t = np.linspace(0, 2 * np.pi, n)
        return project(np.sin(t), np.cos(t), np.cos(2 * t))

    def ordered_curve(shift, x):
        return np.exp(-((((x - shift + 0.5) % 1.0) - 0.5) / 0.14) ** 2)

    def disordered_curve(rng, x):
        y = np.zeros_like(x)
        for k in range(1, 6):
            y += rng.normal() / k * np.sin(2 * np.pi * k * x + rng.uniform(0, 2 * np.pi))
        y = y - y.min()
        return y / (y.max() + 1e-9)

    ax.set_xlim(0, aspect); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
    cxmid = aspect / 2

    # ---- geometric ring, top (label fully above it) ----
    # The ring is a true orthographic projection of the embedded circle
    # (sin, cos, cos 2theta); it carries a soft offset drop shadow so the warped
    # loop reads as a 3D curve. Centered in the panel; the axis gizmo that names
    # the projection sits off to the lower-left.
    ring_h, cy = 0.235, 0.760
    rx, ry = ring_xy()
    rx = rx - rx.mean(); ry = ry - ry.mean()
    s = ring_h / (ry.max() - ry.min())
    cx_ring = cxmid
    for k, (dx, dy, a) in enumerate([(0.006, -0.010, 0.18), (0.012, -0.020, 0.12)]):
        ax.plot(cx_ring + rx * s + dx, cy + ry * s + dy, color="0.35", lw=2.4,
                solid_capstyle="round", solid_joinstyle="round", zorder=2, alpha=a)
    ax.plot(cx_ring + rx * s, cy + ry * s, color="black", lw=2.0,
            solid_capstyle="round", solid_joinstyle="round", zorder=3)
    ax.text(cxmid, 0.998, "geometric ring\n(circular symmetry)", fontsize=6.6,
            ha="center", va="top", linespacing=1.2)

    # ---- axis gizmo: the same projection's three coordinate axes, drawn small
    #      off to the lower-left so the reader can see how the circle is embedded
    #      (a circle in x-y carrying a second harmonic along z = cos 2theta). ----
    gx, gy = cx_ring + rx.min() * s - 0.082, cy - 0.080
    glen = 0.058
    axis_specs = [
        ((-np.sin(A_view), -np.cos(A_view) * np.sin(E_view)), r"$\sin\theta$"),
        (( np.cos(A_view), -np.sin(A_view) * np.sin(E_view)), r"$\cos\theta$"),
        (( 0.0,             np.cos(E_view)),                  r"$\cos 2\theta$"),
    ]
    y_lab = gy - 0.042          # sin/cos labels share this height, just below the arrows
    for (dxu, dyv), lab in axis_specs:
        tipx, tipy = gx + dxu * glen, gy + dyv * glen
        ax.add_patch(FancyArrowPatch((gx, gy), (tipx, tipy), arrowstyle="-|>",
                     mutation_scale=6, color="0.35", lw=0.9,
                     joinstyle="miter", capstyle="butt",
                     shrinkA=0, shrinkB=0, zorder=4))
        if lab == r"$\cos 2\theta$":
            lx, ly = tipx + dxu * 0.024, tipy + dyv * 0.024   # above the up-arrow
        elif lab == r"$\sin\theta$":
            lx, ly = tipx - 0.011, y_lab                       # left, shared height
        else:                                                  # cos theta
            lx, ly = tipx + 0.009, y_lab                       # right, shared height
        ax.text(lx, ly, lab, fontsize=5.0, ha="center", va="center",
                color="0.25", zorder=4)

    # ---- two tuning columns (fill the panel down to the bottom edge) ----
    box_w, box_h = 0.30, 0.10
    xL, xR = 0.006, aspect - box_w - 0.006
    row_tops = [0.452, 0.300, 0.120]        # top edge of each neuron box
    theta_y = 0.005
    vdots_y = (row_tops[1] - box_h + row_tops[2]) / 2   # between neuron 2 and N

    def column(x0, curves, title, tcolor, inner_x):
        labels = ["neuron 1", "neuron 2", "neuron $N$"]
        lab_x = x0 + 0.035 * box_w          # left-aligned label inside box (both cols)
        inners = []
        for cv, lab, ytop_i in zip(curves, labels, row_tops):
            y0 = ytop_i - box_h
            ax.add_patch(Rectangle((x0, y0), box_w, box_h, facecolor=grey,
                                   edgecolor="none", zorder=1))
            # curve confined to the lower part so it never touches the top-left label
            xx = np.linspace(0, 1, 200)
            ax.plot(x0 + xx * box_w, y0 + 0.10 * box_h + cv(xx) * 0.48 * box_h,
                    color="black", lw=1.05, solid_capstyle="round", zorder=3)
            ax.text(lab_x, y0 + box_h - 0.06 * box_h, lab, fontsize=6.0,
                    ha="left", va="top", zorder=4)
            inners.append((inner_x, y0 + box_h / 2))
        ax.text(x0 + box_w / 2, vdots_y, r"$\vdots$", fontsize=8,
                ha="center", va="center")
        ax.text(x0 + box_w / 2, theta_y, r"$\theta$", fontsize=8,
                ha="center", va="top")
        # Regular weight, matching the "geometric ring" caption. The BOLD labels
        # are reserved for the two embeddings, which name the operation; these
        # name its output and should not compete with it.
        ax.text(x0 + box_w / 2, row_tops[0] + 0.022, title, fontsize=6.6,
                ha="center", va="bottom", color=tcolor)
        return inners

    ordered_fns = [lambda x, sh=sh: ordered_curve(sh, x) for sh in (0.30, 0.46, 0.74)]
    disordered_fns = [lambda x, r=np.random.default_rng(seed + sd): disordered_curve(r, x)
                  for sd in (1, 4, 7)]
    iL = column(xL, ordered_fns, "ordered tuning", ordered, xL + box_w + 0.008)
    iR = column(xR, disordered_fns, "disordered tuning", disordered, xR - 0.008)

    # ---- arrow fans: shafts leave a short base at the ring bottom and bow inward
    #      toward the panel centre, the lower/longer arrows bowing the most ----
    def fan(bx, targets, color, sign, rads):
        oy = cy - ring_h * 0.52
        for j, (tgt, rad) in enumerate(zip(targets, rads)):
            # every shaft in a fan starts at the exact same base point
            # miter join + butt cap: a SHARP arrowhead. matplotlib's default
            # rounded join blunts the tip into a little dome.
            ax.add_patch(FancyArrowPatch((bx, oy), tgt, arrowstyle="-|>",
                         mutation_scale=9, color=color, lw=1.3,
                         joinstyle="miter", capstyle="butt",
                         shrinkA=2, shrinkB=2,
                         connectionstyle="arc3,rad={}".format(rad), zorder=2))
    bows = [0.03, 0.08, 0.15]               # top least, bottom most
    fan(cxmid - 0.034, iL, ordered, -1, [-b for b in bows])
    fan(cxmid + 0.034, iR, disordered, +1, bows)

    ax.text(xL + box_w / 2, 0.560, "Fourier\nembedding", fontsize=6.2, color=ordered,
            ha="center", va="center", fontweight="bold", linespacing=1.05)
    ax.text(xR + box_w / 2, 0.560, "disordered\nembedding", fontsize=6.2, color=disordered,
            ha="center", va="center", fontweight="bold", linespacing=1.05)
