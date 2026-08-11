"""Tuning-curve statistics + model-vs-data comparison for the paper analysis.

Provides circular-statistics primitives (COM, centering, peak-finding),
per-mouse aggregate computations, scalar comparison metrics (peak counts,
peak values, flip symmetry, bits-per-spike), and a Kuiper-test-with-
FDR-correction pipeline. The top-level ``analyze_tuning_curves`` runs all
four comparison metrics on data vs model and returns a dict; the npz
serialization round-trip uses ``save_analysis_results`` /
``load_analysis_results``. Consumed by
``notebooks/data_and_generative_model.ipynb`` and ``ring/plot.py``.
"""
import numpy as np
from astropy.stats import kuiper
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests


# --- Core tuning-curve operations ------------------------------------------

def circular_center_of_mass(tuning_curves):
    """Angular center of mass of one or more tuning curves.

    Parameters
    ----------
    tuning_curves : ndarray, shape (N_curves, N_theta)

    Returns
    -------
    coms : ndarray, shape (N_curves,)
        Angular positions in [0, 2*pi).
    """
    theta = np.linspace(0, 2 * np.pi, tuning_curves.shape[1], endpoint=False)
    x_comp = np.sum(tuning_curves * np.cos(theta), axis=1)
    y_comp = np.sum(tuning_curves * np.sin(theta), axis=1)
    return np.arctan2(y_comp, x_comp) % (2 * np.pi)


def center_tuning_curves(tuning_curves):
    """Roll each tuning curve so its circular center of mass is at index 0.

    Parameters
    ----------
    tuning_curves : ndarray, shape (N_curves, N_theta)

    Returns
    -------
    centered : ndarray, shape (N_curves, N_theta)
    """
    n_angles = tuning_curves.shape[1]
    coms = circular_center_of_mass(tuning_curves)
    shifts = (-(n_angles * coms / (2 * np.pi)).round()).astype(int)
    centered = np.zeros_like(tuning_curves)
    for i, shift in enumerate(shifts):
        centered[i] = np.roll(tuning_curves[i], shift)
    return centered


def find_local_maxima(signal, z_thresh):
    """Indices of peaks in ``signal`` exceeding ``z_thresh`` SDs above mean.

    Treats the signal as circular (wrap-around at the boundary).

    Parameters
    ----------
    signal : ndarray, shape (N,)
    z_thresh : float

    Returns
    -------
    maxima : ndarray of int
    """
    # diff[i] is the rise into bin i, with diff[0] wrapping from the last bin.
    diff = np.concatenate(([signal[0] - signal[-1]], np.diff(signal)))
    # Bin i is a local maximum when the rise into it is positive and the rise out of it is
    # negative. Rolling keeps that test defined at the last bin, which np.diff cannot reach.
    sign_change = np.sign(np.roll(diff, -1)) - np.sign(diff)
    maxima = np.where(sign_change == -2)[0]
    return maxima[(signal[maxima] - signal.mean()) / signal.std() > z_thresh]


# --- Per-mouse aggregates --------------------------------------------------

def select_mice_by_cell_count(mouse_nums, n_mice):
    """Select the ``n_mice`` mice with the most recorded cells.

    Parameters
    ----------
    mouse_nums : ndarray, shape (N_curves,)
        Per-cell mouse-id labels.
    n_mice : int
        Number of mice to keep.

    Returns
    -------
    selected : ndarray, shape (n_mice,)
        Mouse-id values, ordered by descending cell count.
    """
    total_mice = int(np.max(mouse_nums) + 1)
    counts = np.array([np.sum(mouse_nums == i) for i in range(total_mice)])
    return np.argsort(counts)[::-1][:n_mice]


def compute_mean_and_std_profiles(tuning_curves):
    """COM-centered mean and std profiles across one set of tuning curves.

    Parameters
    ----------
    tuning_curves : ndarray, shape (N_curves, N_theta)

    Returns
    -------
    mean_profile : ndarray, shape (N_theta,)
    std_profile : ndarray, shape (N_theta,)
    """
    tuning_curves_centered = center_tuning_curves(tuning_curves)
    return tuning_curves_centered.mean(0), tuning_curves_centered.std(0)


def compute_mean_and_std_profiles_for_all_mice(Phi_m_data, mouse_nums):
    """Per-mouse mean + std profiles.

    Parameters
    ----------
    Phi_m_data : ndarray, shape (N_curves, N_theta)
    mouse_nums : ndarray, shape (N_curves,)

    Returns
    -------
    means : ndarray, shape (n_mice, N_theta)
    stds : ndarray, shape (n_mice, N_theta)
    """
    all_mus, all_stds = [], []
    n_mice = int(np.max(mouse_nums) + 1)
    for mouse_num in range(n_mice):
        mu, std = compute_mean_and_std_profiles(Phi_m_data[mouse_nums == mouse_num])
        all_mus.append(mu)
        all_stds.append(std)
    return np.array(all_mus), np.array(all_stds)


def compute_coms_for_all_mice(Phi_m_data, mouse_nums):
    """Per-mouse list of tuning-curve COMs.

    Parameters
    ----------
    Phi_m_data : ndarray, shape (N_curves, N_theta)
    mouse_nums : ndarray, shape (N_curves,)

    Returns
    -------
    mouse_coms : list of ndarray
        Length-``n_mice``; element ``m`` is the array of COMs for mouse ``m``.
    """
    n_mice = int(np.max(mouse_nums) + 1)
    return [circular_center_of_mass(Phi_m_data[mouse_nums == m]) for m in range(n_mice)]


# --- Per-curve scalar metrics ----------------------------------------------

def compute_peak_counts(tuning_curves, z_thresh, max_peaks=6):
    """Normalized histogram of number-of-peaks across tuning curves.

    Parameters
    ----------
    tuning_curves : ndarray, shape (N_curves, N_theta)
    z_thresh : float
        Z-score threshold for peak detection.
    max_peaks : int, default 6
        Histogram extends to ``max_peaks - 1`` peaks per curve.

    Returns
    -------
    hist : ndarray, shape (max_peaks - 1,)
        Probabilities (sums to 1).
    """
    peak_counts = np.array([len(find_local_maxima(tc, z_thresh)) for tc in tuning_curves])
    counts = np.array([(peak_counts == i).sum() for i in range(1, max_peaks)]).astype(float)
    return counts / counts.sum()


def get_peak_values(tuning_curves, num_peaks):
    """Top-``num_peaks`` peak heights per curve (NaN-padded for short rows).

    Parameters
    ----------
    tuning_curves : ndarray, shape (N_curves, N_theta)
    num_peaks : int

    Returns
    -------
    results : ndarray, shape (N_curves, num_peaks)
        Sorted descending; missing values are NaN.
    """
    results = np.zeros((len(tuning_curves), num_peaks)) * np.nan
    for i, tc in enumerate(tuning_curves):
        maxima = find_local_maxima(tc, z_thresh=1)
        max_vals = np.sort(tc[maxima])[::-1]
        n = min(num_peaks, len(max_vals))
        results[i, :n] = max_vals[:n]
    return results


def compute_symmetry_scores(tuning_curves):
    """Flip-symmetry correlation for each tuning curve.

    For each curve, reflect across its circular COM and compute the
    Pearson correlation between the original and the reflection.

    Parameters
    ----------
    tuning_curves : ndarray, shape (N_curves, N_theta)

    Returns
    -------
    scores : ndarray, shape (N_curves,)
        Pearson r in [-1, 1].
    """
    def _com_bin(x):
        # Inline integer-bin COM. Uses the average-with-weights form rather
        # than the module-level circular_center_of_mass, because the released
        # results were computed with this form and the two differ in the last
        # float bits.
        theta = np.arange(len(x)) * 2 * np.pi / len(x)
        cos_avg = np.average(np.cos(theta), weights=x)
        sin_avg = np.average(np.sin(theta), weights=x)
        angle = np.mod(np.arctan2(sin_avg, cos_avg), 2 * np.pi)
        return int(np.round(angle * len(x) / (2 * np.pi)))

    scores = []
    for tc in tuning_curves:
        com = _com_bin(tc)
        reversed_tc = np.roll(tc, -2 * com - 1)[::-1]
        scores.append(pearsonr(tc, reversed_tc)[0])
    return np.array(scores)


def compute_bits_per_spike(tuning_curves):
    """Per-curve bits-per-spike (mean of ``tc * log2(tc)``).

    Parameters
    ----------
    tuning_curves : ndarray, shape (N_curves, N_theta)

    Returns
    -------
    bits : ndarray, shape (N_curves,)
    """
    return (tuning_curves * np.log2(tuning_curves)).mean(1)


# --- Statistical tests -----------------------------------------------------

def analyze_circular_datasets(datasets, alpha=0.05):
    """Kuiper uniformity test per dataset + Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    datasets : list of ndarray
        Each entry is an array of angular values to test against uniformity.
    alpha : float, default 0.05
        FDR target significance level.

    Returns
    -------
    results : list of dict
        Per-dataset ``{dataset, statistic, p_value, corrected_p_value, rejected}``.
    """
    results = []
    p_values = []
    for i, data in enumerate(datasets):
        statistic, p_value = kuiper(data, lambda x: x / (2 * np.pi))
        results.append({'dataset': i + 1, 'statistic': statistic, 'p_value': p_value})
        p_values.append(p_value)

    rejected, corrected_p_values, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    for i, result in enumerate(results):
        result['corrected_p_value'] = corrected_p_values[i]
        result['rejected'] = rejected[i]
    return results


def analyze_and_print_stats(mouse_coms):
    """Run uniformity tests on per-mouse COM distributions; print a summary.

    Parameters
    ----------
    mouse_coms : list of ndarray
        From ``compute_coms_for_all_mice``.
    """
    results = analyze_circular_datasets(mouse_coms)
    p_vals = np.array([r['p_value'] for r in results])
    p_vals_corrected = np.array([r['corrected_p_value'] for r in results])

    print("\nUniformity Test Results:")
    print(f"Number of mice: {len(p_vals)}")
    print(f"Mice with p < 0.05 (uncorrected): {np.sum(p_vals < 0.05)}")
    print(f"Mice with p < 0.05 (FDR-corrected): {np.sum(p_vals_corrected < 0.05)}")
    print("\nP-value ranges:")
    print(f"Uncorrected: {p_vals.min():.1e} - {p_vals.max():.1e}")
    print(f"FDR-corrected: {p_vals_corrected.min():.1e} - {p_vals_corrected.max():.1e}")


# --- Top-level data-vs-model comparison + I/O ------------------------------

def analyze_tuning_curves(data_curves, model_curves, z_thresholds=[1.0, 0.5, 0.0]):
    """Run all four data-vs-model comparison metrics.

    Parameters
    ----------
    data_curves : ndarray, shape (N_curves, N_theta)
    model_curves : ndarray, shape (N_curves, N_theta)
    z_thresholds : list of float, default [1.0, 0.5, 0.0]
        Thresholds for peak detection in the peak-count histogram.

    Returns
    -------
    results : dict
        Keys: ``peak_counts`` (list of (data, model) tuples per threshold),
        ``peak_values`` (data, model), ``symmetry`` (data, model),
        ``information`` (data, model).
    """
    return {
        'peak_counts': [(compute_peak_counts(data_curves, z),
                         compute_peak_counts(model_curves, z))
                        for z in z_thresholds],
        'peak_values': (get_peak_values(data_curves, 3),
                        get_peak_values(model_curves, 3)),
        'symmetry': (compute_symmetry_scores(data_curves),
                     compute_symmetry_scores(model_curves)),
        'information': (compute_bits_per_spike(data_curves),
                        compute_bits_per_spike(model_curves)),
    }


def save_analysis_results(results, filename='tuning_curve_analysis.npz'):
    """Serialize the dict from ``analyze_tuning_curves`` to compressed npz.

    Parameters
    ----------
    results : dict
    filename : str
    """
    np.savez_compressed(
        filename,
        peak_counts_data=[x[0] for x in results['peak_counts']],
        peak_counts_model=[x[1] for x in results['peak_counts']],
        peak_values_data=results['peak_values'][0],
        peak_values_model=results['peak_values'][1],
        symmetry_data=results['symmetry'][0],
        symmetry_model=results['symmetry'][1],
        information_data=results['information'][0],
        information_model=results['information'][1],
    )


def load_analysis_results(filename='tuning_curve_analysis.npz'):
    """Inverse of ``save_analysis_results``.

    Parameters
    ----------
    filename : str

    Returns
    -------
    results : dict
        Same structure produced by ``analyze_tuning_curves``.
    """
    data = np.load(filename)
    return {
        'peak_counts': list(zip(data['peak_counts_data'], data['peak_counts_model'])),
        'peak_values': (data['peak_values_data'], data['peak_values_model']),
        'symmetry': (data['symmetry_data'], data['symmetry_model']),
        'information': (data['information_data'], data['information_model']),
    }
