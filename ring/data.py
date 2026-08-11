"""Mouse head-direction tuning-curve producer.

Walks the Peyrache lab NWB sessions on disk, computes per-unit tuning
curves (with Poisson-likelihood-optimized Gaussian smoothing per
cross-validation fold), and writes
``tc_data.npz``, the tuning-curve file consumed by ``notebooks/data_and_generative_model.ipynb``
and the ``data_driven`` chain.

External entry point: run as a script,
::

    python -m ring.data

(or ``python ring/data.py``). Consumers reference the produced cache as
``data/mouse_data/tc_data.npz``.

Running this is optional. The cache it writes is distributed with the rest
of the caches, so it is only needed to rebuild the tuning curves from the
raw recordings.
"""
import os
from pathlib import Path

import numpy as np
from pynwb import NWBHDF5IO
from scipy.ndimage import gaussian_filter1d
from scipy.stats import poisson
from tqdm import tqdm


# Raw Peyrache sessions from DANDI Dandiset 000939 (~52 GB). Not in the repo,
# and too large to want a copy inside it, so the default is a sibling of the
# repo rather than a path under it. Set ``RING_DATA_DIR`` to your own download:
#
#     RING_DATA_DIR=/path/to/000939 python -m ring.data
#
DATA_DIR = os.environ.get(
    "RING_DATA_DIR",
    str(Path(__file__).parent.parent.parent / "PeyracheData" / "000939"),
)

# Producer output. Defaults into the repo's data/ tree so consumers
# (``data_driven.TC_DATA_PATH``) pick it up with no manual copy step.
# NOTE: re-running the producer overwrites the released cache. Set
# ``RING_OUTPUT_DIR`` to write elsewhere and keep both.
OUTPUT_DIR = os.environ.get(
    "RING_OUTPUT_DIR",
    str(Path(__file__).parent.parent / "data" / "mouse_data"),
)

# Behavioral / spike-train binning resolution (seconds).
DATA_DT = 0.01


def get_session_info():
    """Enumerate NWB sessions in ``DATA_DIR``.

    Returns
    -------
    sessions : list of (fname, subject_id, has_optogenetics) tuples
        Sorted by filename. ``fname`` is the absolute path to the NWB
        file; ``subject_id`` is the trailing subject identifier from the
        directory name; ``has_optogenetics`` is True if the session is
        the ``+ogen.nwb`` variant.
    """
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(
            f"Raw Peyrache sessions not found at {DATA_DIR}.\n"
            "These are the ~52 GB NWB recordings from DANDI Dandiset 000939 "
            "(https://dandiarchive.org/dandiset/000939), which are not in the repository.\n"
            "Download them and set RING_DATA_DIR to the directory holding the sub-* "
            "session folders.\n"
            "Reproducing the paper's figures does not require them; the cached tuning "
            "curves in data/mouse_data/tc_data.npz are enough."
        )
    session_info = []
    for dirname in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, dirname)
        if os.path.isfile(path):
            continue
        subj = dirname.split("-")[-1]
        fname = f"sub-{subj}_behavior+ecephys"
        full_path_ogen = os.path.join(DATA_DIR, dirname, fname + "+ogen.nwb")
        full_path_standard = os.path.join(DATA_DIR, dirname, fname + ".nwb")
        if os.path.exists(full_path_ogen):
            session_info.append((full_path_ogen, subj, True))
        elif os.path.exists(full_path_standard):
            session_info.append((full_path_standard, subj, False))
    return sorted(session_info)


def compute_angular_occupancy_and_spike_counts(N_theta, hd_idx, spike_time_idx, mask=None):
    """Bin behavioral occupancy and spike counts by head-direction angle.

    Parameters
    ----------
    N_theta : int
        Number of angular bins on the ring.
    hd_idx : ndarray, shape (N_bins,)
        Per-time-bin angular bin index (in [0, N_theta)); negative entries
        flag invalid bins (e.g. NaN HD samples).
    spike_time_idx : ndarray, shape (N_spikes,)
        Time-bin indices at which spikes occurred.
    mask : ndarray of bool or None, shape (N_bins,)
        Optional per-time-bin mask (e.g. cross-validation fold). Negative
        ``hd_idx`` entries are unconditionally masked out.

    Returns
    -------
    angular_occupancy_counts : ndarray, shape (N_theta,)
    angular_spike_counts : ndarray, shape (N_theta,)
    """
    N_bins = len(hd_idx)
    mask = np.ones(N_bins, dtype=bool) if mask is None else mask.copy()
    mask[hd_idx < 0] = False

    masked_spike_time_idx = spike_time_idx[np.in1d(spike_time_idx, np.arange(N_bins)[mask])]
    angular_occupancy_counts = np.zeros(N_theta)
    angular_spike_counts = np.zeros(N_theta)

    unique_hd_idx, hd_idx_counts = np.unique(hd_idx[mask], return_counts=True)
    angular_occupancy_counts[unique_hd_idx] = hd_idx_counts

    hd_idxs_with_spikes = hd_idx[masked_spike_time_idx]
    unique_hd_idx_with_spikes, hd_idx_with_spikes_counts = np.unique(hd_idxs_with_spikes, return_counts=True)
    angular_spike_counts[unique_hd_idx_with_spikes] = hd_idx_with_spikes_counts

    return angular_occupancy_counts, angular_spike_counts


def compute_tuning_curve(N_theta, hd_idx, spike_time_idx, mask=None):
    """Tuning curve = (spike count) / (occupancy * DATA_DT) per angular bin.

    Zero-rate bins are floored at ``mean * 1e-3`` so the downstream
    Poisson likelihood never sees a log(0).

    Parameters
    ----------
    N_theta : int
        Number of angular bins on the ring.
    hd_idx : ndarray, shape (N_bins,)
        Per-time-bin angular bin index; negative entries flag invalid bins.
    spike_time_idx : ndarray, shape (N_spikes,)
        Time-bin indices at which spikes occurred.
    mask : ndarray of bool or None, shape (N_bins,)
        Optional per-time-bin mask (e.g. cross-validation fold).

    Returns
    -------
    tc : ndarray, shape (N_theta,)
        Firing-rate tuning curve (Hz).
    """
    aoc, asc = compute_angular_occupancy_and_spike_counts(N_theta, hd_idx, spike_time_idx, mask)
    tc = asc / (aoc * DATA_DT)
    tc[tc == 0.] = tc.mean() * 1e-3
    return tc


def compute_poisson_ll(tuning_curve, hd_idx, temporal_spike_counts, mask=None):
    """Poisson log-likelihood of observed spike counts given a tuning curve.

    Parameters
    ----------
    tuning_curve : ndarray, shape (N_theta,)
        Firing-rate tuning curve (Hz).
    hd_idx : ndarray, shape (N_bins,)
        Per-time-bin angular bin index; negatives are masked.
    temporal_spike_counts : ndarray, shape (N_bins,)
        Spike count per time bin.
    mask : ndarray of bool or None, shape (N_bins,)
        Optional cross-validation mask.

    Returns
    -------
    ll : float
        Summed log-pmf across the masked bins.
    """
    N_bins = len(hd_idx)
    mask = np.ones(N_bins, dtype=bool) if mask is None else mask.copy()
    mask[hd_idx < 0] = False
    poisson_rates = tuning_curve[hd_idx[mask]] * DATA_DT
    poisson_observations = temporal_spike_counts[mask]
    return poisson.logpmf(k=poisson_observations, mu=poisson_rates).sum()


def generate_alternating_mask(N_bins, K):
    """Alternating-chunk True/False mask for ``2*K`` blocks of cross-validation.

    Used as a cross-validation fold split: ``2*K`` contiguous chunks of
    near-equal size, alternating between True and False.

    Parameters
    ----------
    N_bins : int
        Total number of time bins.
    K : int
        Number of "True" chunks (and number of "False" chunks); total
        ``2*K`` alternating chunks.

    Returns
    -------
    mask : ndarray of bool, shape (N_bins,)
    """
    if K <= 0 or N_bins <= 0:
        return np.array([])
    chunk_size = N_bins // (2 * K)
    remainder = N_bins % (2 * K)
    mask = np.zeros(N_bins, dtype=bool)
    current_value = True
    start_index = 0
    for _ in range(2 * K):
        this_chunk_size = chunk_size + (1 if remainder > 0 else 0)
        remainder -= 1
        mask[start_index:start_index + this_chunk_size] = current_value
        start_index += this_chunk_size
        current_value = not current_value
    return mask


def process_file(nwbfile, N_theta=100, sigma_vals=np.logspace(-1, 2, 100), K=4):
    """Compute per-unit tuning curves for one session.

    For each unit, computes a plain (occupancy-normalized) tuning curve
    per cross-validation fold, then sweeps Gaussian smoothing widths
    ``sigma_vals`` and picks the one that maximizes the held-out Poisson
    log-likelihood.

    Parameters
    ----------
    nwbfile : pynwb NWBFile (already-opened)
        Session to process. Expects ``behavior/CompassDirection/head-direction``
        and ``units/spike_times`` to be present, plus an ``epochs`` table
        with a ``wake_square`` tag.
    N_theta : int, default 100
        Number of angular bins.
    sigma_vals : ndarray, default ``np.logspace(-1, 2, 100)``
        Gaussian smoothing widths (in bin units) to sweep.
    K : int, default 4
        Number of alternating-chunk cross-validation folds.

    Returns
    -------
    plain_tcs : ndarray, shape (num_units, 2, N_theta)
        Plain (un-smoothed) tuning curves per fold.
    opt_tcs : ndarray, shape (num_units, 2, N_theta)
        Smoothed tuning curve at the held-out-LL-optimal sigma per fold.
    opt_sigmas : ndarray, shape (num_units, 2)
        Optimal sigma per fold.
    """
    # Behavioral epoch for the "wake_square" task.
    epoch_df = nwbfile.epochs.to_dataframe()
    ti_nominal, tf_nominal = epoch_df[epoch_df["tags"] == "wake_square"][["start_time", "stop_time"]].values[0]

    # Head-direction track.
    hd_spatial_series = nwbfile.processing["behavior"].data_interfaces["CompassDirection"].spatial_series["head-direction"]
    hd_full = hd_spatial_series.data[:]
    t_full = hd_spatial_series.timestamps[:]

    ti_idx = np.argmin((ti_nominal - t_full)**2)
    tf_idx = np.argmin((tf_nominal - t_full)**2)
    N_bins = tf_idx - ti_idx
    ti, tf = t_full[[ti_idx, tf_idx]]

    hd = hd_full[ti_idx:tf_idx].copy()
    d_theta = 2 * np.pi / N_theta
    hd[np.isnan(hd)] = -1
    hd_idx = np.floor(hd / d_theta).astype(int)

    mask = generate_alternating_mask(N_bins, K=K)
    masks = (mask, ~mask)

    num_units = len(nwbfile.units)
    opt_tcs = np.zeros((num_units, 2, N_theta))
    opt_sigmas = np.zeros((num_units, 2))
    plain_tcs = np.zeros((num_units, 2, N_theta))

    for unit_idx in tqdm(range(num_units), desc="Computing tuning curves"):
        spike_times = nwbfile.units["spike_times"][unit_idx][:]
        within_epoch_spike_times = spike_times[(spike_times >= ti) & (spike_times < tf)]
        spike_time_idx = np.floor((within_epoch_spike_times - ti) / DATA_DT).astype(int)

        temporal_spike_counts = np.zeros(N_bins)
        unique_spike_time_idx, spike_time_idx_counts = np.unique(spike_time_idx, return_counts=True)
        temporal_spike_counts[unique_spike_time_idx] = spike_time_idx_counts

        for mask_idx, mask in enumerate(masks):
            ll_vals = np.zeros(len(sigma_vals))
            smoothed_tcs = np.zeros((len(sigma_vals), N_theta))
            tuning_curve = compute_tuning_curve(N_theta, hd_idx, spike_time_idx, mask=mask)
            plain_tcs[unit_idx, mask_idx] = tuning_curve

            for i in range(len(sigma_vals)):
                smoothed_tcs[i] = gaussian_filter1d(tuning_curve, sigma=sigma_vals[i], mode='wrap')
                ll_vals[i] = compute_poisson_ll(smoothed_tcs[i], hd_idx, temporal_spike_counts, mask=~mask)

            opt_idx = ll_vals.argmax()
            opt_tcs[unit_idx, mask_idx] = smoothed_tcs[opt_idx]
            opt_sigmas[unit_idx, mask_idx] = sigma_vals[opt_idx]

    return plain_tcs, opt_tcs, opt_sigmas


def process_all_tcs():
    """Process all sessions in ``DATA_DIR`` and write ``tc_data.npz``.

    Output keys (concatenated across sessions on axis 0 where applicable):

    - ``hd_mask, exc_mask, fs_mask`` : per-unit boolean classifiers from
      the NWB units table.
    - ``plain_tcs``: shape ``(N_total_units, 2, N_theta)``, per-fold
      occupancy-normalized tuning curves.
    - ``opt_tcs, opt_sigmas`` : optimal smoothed tuning curves and
      smoothing widths.
    - ``fnames`` : per-session NWB filenames.
    - ``unit_counts`` : per-session unit count.
    - ``mouse_nums, mouse_names`` : per-unit session and subject id.
    - ``has_ogen`` : per-unit optogenetics flag.
    """
    session_info = get_session_info()
    all_hd_masks, all_exc_masks, all_fs_masks = [], [], []
    all_plain_tcs, all_opt_tcs, all_opt_sigmas = [], [], []
    all_num_units = []
    all_mouse_nums, all_mouse_names, all_has_ogen = [], [], []

    for session_idx, (fname, subject_id, has_ogen) in enumerate(session_info):
        print(f"Processing session {session_idx + 1}/{len(session_info)}: {subject_id}")
        io = NWBHDF5IO(fname, mode="r")
        nwbfile = io.read()

        num_units = len(nwbfile.units)
        hd_mask = nwbfile.units["is_head_direction"][:].astype(bool)
        exc_mask = nwbfile.units["is_excitatory"][:].astype(bool)
        fs_mask = nwbfile.units["is_fast_spiking"][:].astype(bool)

        plain_tcs, opt_tcs, opt_sigmas = process_file(nwbfile)

        all_hd_masks.append(hd_mask)
        all_exc_masks.append(exc_mask)
        all_fs_masks.append(fs_mask)
        all_plain_tcs.append(plain_tcs)
        all_opt_tcs.append(opt_tcs)
        all_opt_sigmas.append(opt_sigmas)
        all_num_units.append(num_units)
        all_mouse_nums.extend([session_idx] * num_units)
        all_mouse_names.extend([subject_id] * num_units)
        all_has_ogen.extend([has_ogen] * num_units)

        io.close()

    np.savez(
        os.path.join(OUTPUT_DIR, "tc_data.npz"),
        hd_mask=np.concatenate(all_hd_masks),
        exc_mask=np.concatenate(all_exc_masks),
        fs_mask=np.concatenate(all_fs_masks),
        plain_tcs=np.concatenate(all_plain_tcs, axis=0),
        opt_tcs=np.concatenate(all_opt_tcs, axis=0),
        opt_sigmas=np.concatenate(all_opt_sigmas, axis=0),
        fnames=[info[0] for info in session_info],
        unit_counts=np.array(all_num_units),
        mouse_nums=np.array(all_mouse_nums),
        mouse_names=np.array(all_mouse_names),
        has_ogen=np.array(all_has_ogen),
    )


if __name__ == "__main__":
    process_all_tcs()
