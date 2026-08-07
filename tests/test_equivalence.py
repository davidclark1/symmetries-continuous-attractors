"""Bit-level equivalence + invariant guards for the ring package.

Pins the invariants established during the deduplication pass so a future edit
that silently changes numerics is caught. Bit-identical claims use
``np.array_equal`` (not ``allclose``). Run with pytest, or directly:

    python tests/test_equivalence.py
"""
import numpy as np
from scipy.special import erf

from ring import fn, tc, analysis, data_driven as dd


# --- consolidated: these MUST stay bit-identical -----------------------------

def test_circular_com_matches_center_of_mass():
    """circular_com (float path) == analysis.circular_center_of_mass, bit-for-bit."""
    rng = np.random.RandomState(0)
    for shape in [(200, 100), (1, 360), (500, 64)]:
        A = np.abs(rng.randn(*shape)) + 0.3
        assert np.array_equal(dd.circular_com(A), analysis.circular_center_of_mass(A))


def test_nonlin_idealized_matches_expression():
    """fn.nonlin_idealized(x, beta) == 1 + fn.unity_erf(beta*x), bit-for-bit."""
    x = np.linspace(-5, 5, 40001)
    for beta in (1.0, 2.0, 2.7659):
        assert np.array_equal(fn.nonlin_idealized(x, beta), 1 + fn.unity_erf(beta * x))


def test_gen_tuning_curves_idealized_matches_recipe():
    """tc.gen_tuning_curves_idealized == (wrapped_gaussian_cov + sample_tcs + nonlin)."""
    def recipe(Ns, Nt, sc, bt):
        _, g = tc.wrapped_gaussian_cov(eq_var=1., scale=sc, N_theta=Nt)
        X = tc.sample_tcs(N=Ns, Gamma_ft=g)
        return X, 1 + fn.unity_erf(bt * X)
    for (Ns, Nt, sc, bt) in [(2500, 500, 1.4211, 2.7659), (50, 64, 1.0, 2.0)]:
        np.random.seed(45)
        Xa, Pa = tc.gen_tuning_curves_idealized(Ns, Nt, sc, bt)
        np.random.seed(45)
        Xb, Pb = recipe(Ns, Nt, sc, bt)
        assert np.array_equal(Xa, Xb) and np.array_equal(Pa, Pb)


def test_sinkhorn_helper_matches_inline_loop():
    """_sinkhorn_normalize == the inline 1000-sweep double-normalization loop."""
    rng = np.random.RandomState(5)
    P0 = np.abs(rng.randn(80, 120)) + 0.2
    ref = P0.copy()
    for _ in range(1000):
        ref /= ref.mean(1, keepdims=True)
        ref /= ref.mean(0, keepdims=True)
    assert np.array_equal(dd._sinkhorn_normalize(P0.copy()), ref)


def test_ring_imports_clean():
    import ring  # noqa: F401
    import ring.tc, ring.fn, ring.gp_opt, ring.dmft, ring.sim  # noqa: F401
    import ring.regression, ring.data_driven, ring.analysis, ring.plot  # noqa: F401
    import ring.seed, ring.style, ring.data  # noqa: F401


# --- intentionally DISTINCT: guards so nobody "helpfully" merges them ---------

def test_sample_z_layout_is_not_sample_tcs():
    """gp_opt.sample_z uses a transposed layout -> DIFFERENT draws under one seed.

    Regression guard: rerouting one through the other would silently change
    every figure that samples tuning curves. If this ever starts matching,
    someone changed the RNG draw order.
    """
    from ring import gp_opt
    N_theta, N = 64, 50
    _, Gamma_ft = tc.wrapped_gaussian_cov(1., 1.4, N_theta)
    np.random.seed(7)
    X_a = tc.sample_tcs(N=N, Gamma_ft=Gamma_ft)
    np.random.seed(7)
    z = gp_opt.sample_z(N, N_theta // 2 + 1)
    X_b = np.fft.irfft(z * np.sqrt(Gamma_ft)[None, :], axis=1, norm='forward').T
    assert not np.allclose(X_a, X_b)


def test_mexican_hat_phi_is_mathematically_equal_but_bit_fragile():
    """The notebook's phi = 1 + erf(beta*sqrt(pi)*x/2) equals nonlin_idealized
    mathematically (allclose), but the multiply order differs so it is NOT
    guaranteed bit-identical. Documented reason to keep them separate.
    """
    x = np.linspace(-4, 4, 20001)
    beta = 2.7659
    phi_notebook = 1. + erf(beta * np.sqrt(np.pi) * x / 2.)
    assert np.allclose(phi_notebook, fn.nonlin_idealized(x, beta))


# --- colormap system invariants ----------------------------------------------

def _contrast_on_white(rgb):
    """WCAG contrast ratio of each color against a white page."""
    c = np.asarray(rgb)[..., :3]
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return 1.05 / (lin @ np.array([0.2126, 0.7152, 0.0722]) + 0.05)


def _simulate_cvd(rgb, matrix):
    """Approximate complete color-vision deficiency in linear sRGB."""
    c = np.asarray(rgb)[..., :3]
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    transformed = np.clip(lin @ np.asarray(matrix).T, 0, 1)
    return np.where(
        transformed <= 0.0031308,
        transformed * 12.92,
        1.055 * transformed ** (1 / 2.4) - 0.055,
    )


def test_sweep_ramps_are_never_too_pale_to_read():
    """Every sampled sweep color clears SWEEP_MIN_CONTRAST against the page.

    This is the invariant the ordered-parameter ramps exist to provide: curves
    and the per-curve text labels drawn in the same color must stay legible at
    print size. Viridis's yellow end reached only 1.5:1, which is what these
    ramps replaced.
    """
    from ring import style
    for kind in style.SWEEP_CMAPS:
        # Ramps whose colours are also drawn as TEXT (per-curve labels) are held
        # to the stricter floor; ramps read through a legend may run bolder.
        floor = (style.SWEEP_LABEL_CONTRAST if kind in style.SWEEP_LABEL_KINDS
                 else style.SWEEP_MIN_CONTRAST)
        for n in (2, 4, 5, 9, 16):
            worst = _contrast_on_white(style.sweep_colors(n, kind)).min()
            assert worst >= floor, (kind, n, worst, floor)


def test_sweep_ramps_are_monotone_in_lightness():
    """Each ramp reads as ORDERED: perceived lightness increases along it."""
    from ring import style
    for kind in style.SWEEP_CMAPS:
        c = _contrast_on_white(style.sweep_colors(12, kind))
        assert np.all(np.diff(c) < 0), kind  # contrast falls == lightness rises


def test_colormap_kinds_are_the_documented_set():
    """The kind vocabulary is a contract: notebooks pass these strings by name."""
    from ring import style
    assert set(style.SWEEP_CMAPS) == {'ocean', 'ember', 'slate'}
    assert set(style.FIELD_CMAPS) == {'ice', 'ember', 'aurora', 'stability'}
    # Ordered line sweeps remain generated from the paper hue arms.
    for cm in style.SWEEP_CMAPS.values():
        assert cm.name.startswith('ring_'), cm.name
    # Heatmaps and signed matrices use the paper's perceptually ordered maps.
    assert {cm.name for cm in style.FIELD_CMAPS.values()} == {
        'ring_field_activity', 'ring_field_sunstone',
        'ring_field_stability'}
    assert style.DIVERGING_CMAP.name == 'ring_diverging_petrol_persimmon'
    assert style.COVARIANCE_CMAP.name == 'ring_field_lagoon'
    assert style.ANGLE_CMAP.name == 'ring_angle'
    assert style.GRID_CELL_CMAP.name == 'ring_field_lagoon'


def test_main_figure_semantic_colors_are_shared():
    """Repeated scientific roles use one source-of-truth color."""
    from matplotlib.colors import to_rgb
    from ring import style
    assert style.J_COLOR == style.MODEL
    assert style.EMBED_DISORDERED == style.MODEL
    assert style.J2_COLOR == style.CLASSICAL
    assert style.EMBED_ORDERED == style.CLASSICAL
    assert style.HD_COLOR != style.MODEL
    assert _contrast_on_white(
        np.array([to_rgb(style.HD_COLOR)])).min() >= 4.5
    assert style.DATA != style.MODEL
    assert style.E_COLOR == style.MODEL
    assert style.I_COLOR == style.CLASSICAL


def test_data_model_profile_uses_a_continuous_layered_stroke():
    """The model stays visible without weakening data-only aggregate curves."""
    import matplotlib.pyplot as plt
    from ring import plot, style

    theta = np.linspace(-np.pi, np.pi, 9)
    profiles = np.vstack((np.cos(theta) + 2, np.cos(theta) + 2.2))
    mean = profiles.mean(axis=0)

    fig, (ax_data, ax_overlay) = plt.subplots(1, 2)
    plot.plot_profile(ax_data, theta, profiles, mean_profile=mean)
    plot.plot_profile(ax_overlay, theta, profiles, mean_profile=mean,
                      model_profile=mean)

    data_only = next(line for line in ax_data.lines
                     if line.get_label() == 'mouse average')
    underlay = next(line for line in ax_overlay.lines
                    if line.get_label() == 'mouse average')
    model = next(line for line in ax_overlay.lines
                 if line.get_label() == 'generative process')

    assert data_only.get_color() == style.DATA
    assert underlay.get_color() == style.DATA_UNDERLAY
    assert model.get_color() == style.MODEL
    assert underlay.get_linewidth() > model.get_linewidth()
    assert model.get_linestyle() == '-'
    plt.close(fig)


def test_main_semantic_pairs_remain_separable_under_cvd():
    """Critical pairs retain a visible cue under complete CVD simulation.

    Principal scientific comparisons are also redundantly encoded by dashes,
    markers, line weight, labels, or spatial separation. Figure 1's two data
    partitions are the deliberate colour-only exception, so their stronger
    simulated-CVD distance is guarded separately below.
    """
    from matplotlib.colors import to_rgb
    from ring import style

    # Full-severity linear-RGB approximations for protanopia, deuteranopia and
    # tritanopia. These guard the categorical anchors; the continuous maps are
    # also reviewed as rendered figures in each simulated viewing condition.
    matrices = (
        ((0.152286, 1.052583, -0.204868),
         (0.114503, 0.786281, 0.099216),
         (-0.003882, -0.048116, 1.051998)),
        ((0.367322, 0.860646, -0.227968),
         (0.280085, 0.672501, 0.047413),
         (-0.011820, 0.042940, 0.968881)),
        ((1.255528, -0.076749, -0.178779),
         (-0.078411, 0.930809, 0.147602),
         (0.004733, 0.691367, 0.303900)),
    )
    core = np.array([to_rgb(style.MODEL), to_rgb(style.CLASSICAL)])
    partitions = np.array([to_rgb(style.PARTITION_A), to_rgb(style.PARTITION_B)])
    for matrix in matrices:
        assert np.linalg.norm(np.diff(_simulate_cvd(core, matrix), axis=0)) > 0.40
        assert np.linalg.norm(np.diff(_simulate_cvd(partitions, matrix), axis=0)) > 0.40

    # The principal line colors also stay comfortably readable on a white page.
    assert _contrast_on_white(core).min() >= 4.5
    assert _contrast_on_white(partitions).min() >= 4.5


def test_sweep_kinds_are_visually_distinct():
    """Different KINDS of swept parameter must not read as one series.

    Compares ramps pairwise in RGB; 'slate' is excluded because it is the
    deliberately neutral grey fallback.
    """
    from ring import style
    kinds = [k for k in style.SWEEP_CMAPS if k != 'slate']
    for i, a in enumerate(kinds):
        for b in kinds[i + 1:]:
            ca = style.sweep_colors(16, a)[:, :3]
            cb = style.sweep_colors(16, b)[:, :3]
            assert np.abs(ca - cb).max() > 0.25, (a, b)


def test_scalar_heatmaps_use_perceptually_ordered_paper_maps():
    """Every scalar heatmap uses a monotone-lightness paper map."""
    from ring import style
    assert style.field_cmap('ice').name == 'ring_field_activity'
    assert style.field_cmap('ember').name == 'ring_field_sunstone'
    assert style.field_cmap('aurora').name == 'ring_field_sunstone'
    assert style.field_cmap('stability').name == 'ring_field_stability'

    # Relative luminance must remain strictly ordered under every complete-CVD
    # simulation, not merely for a standard trichromatic viewer.
    matrices = (
        ((0.152286, 1.052583, -0.204868),
         (0.114503, 0.786281, 0.099216),
         (-0.003882, -0.048116, 1.051998)),
        ((0.367322, 0.860646, -0.227968),
         (0.280085, 0.672501, 0.047413),
         (-0.011820, 0.042940, 0.968881)),
        ((1.255528, -0.076749, -0.178779),
         (-0.078411, 0.930809, 0.147602),
         (0.004733, 0.691367, 0.303900)),
    )
    samples = np.linspace(0, 1, 256)
    for cmap in (style.LAGOON_CMAP, style.ACTIVITY_CMAP,
                 style.SUNSTONE_CMAP, style.STABILITY_CMAP):
        rgb = cmap(samples)[:, :3]
        for matrix in matrices:
            transformed = _simulate_cvd(rgb, matrix)
            luminance = 1 / _contrast_on_white(transformed)
            assert np.all(np.diff(luminance) > 0), cmap.name


def test_angle_map_is_cyclic_and_isoluminant():
    """ANGLE_CMAP must close on itself and hold one lightness.

    Angle is periodic, so 0 and 2*pi have to land on the same colour; and every
    trajectory must read with the same weight against white, which is what raw
    HSV fails at (its yellow is far lighter than its blue).
    """
    from ring import style
    cols = style.ANGLE_CMAP(np.linspace(0, 1, 361))[:, :3]
    assert np.abs(cols[0] - cols[-1]).max() < 0.02          # closes the circle
    c = _contrast_on_white(style.ANGLE_CMAP(np.linspace(0, 1, 64, endpoint=False)))
    assert c.max() / c.min() < 1.25                          # near-isoluminant


def test_diverging_map_pivots_through_neutral():
    """The signed-map endpoints carry equal weight around a near-white zero."""
    from ring import style
    lo, mid, hi = style.DIVERGING_CMAP([0.0, 0.5, 1.0])[:, :3]
    assert mid.min() > 0.9                                   # pivot is near-white
    assert lo[2] > lo[0]                                  # petrol negative end
    assert hi[0] > hi[2]                                  # persimmon positive end
    c = _contrast_on_white(np.array([lo, hi]))
    assert abs(c[0] - c[1]) < 1.2                            # ends carry equal weight


def test_find_local_maxima_is_circular():
    """The peak detector must be rotation-equivariant, including at the last bin.

    It was not: the final entry of the sign-change array was a raw amplitude difference
    standing in for a comparison that np.diff cannot reach, so a peak landing in the last
    angular bin was never counted.
    """
    N = 100
    theta = np.arange(N) * 2 * np.pi / N

    # a single peak swept over every bin must be found at every bin
    for k in range(N):
        found = analysis.find_local_maxima(np.cos(theta - theta[k]), 0.5)
        assert list(found) == [k], f"peak at bin {k} reported as {list(found)}"

    # and the peak set must rotate with the signal
    rng = np.random.RandomState(0)
    for _ in range(50):
        coef = rng.randn(6, 2)
        s = sum(a * np.cos(j * theta) + b * np.sin(j * theta)
                for j, (a, b) in enumerate(coef, start=1))
        base = analysis.find_local_maxima(s, 1.0)
        for shift in (1, 7, 50, 99):
            rolled = analysis.find_local_maxima(np.roll(s, shift), 1.0)
            assert np.array_equal(np.sort((base + shift) % N), np.sort(rolled))


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    fails = 0
    for f in fns:
        try:
            f()
            print(f"PASS  {f.__name__}")
        except Exception:
            fails += 1
            print(f"FAIL  {f.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - fails}/{len(fns)} passed")
    raise SystemExit(1 if fails else 0)
