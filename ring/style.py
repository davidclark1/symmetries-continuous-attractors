"""Matplotlib style configuration and small text-formatting helpers.

Imported by every notebook to set consistent fonts, sizes, and PDF/SVG export
settings for paper figures. Also patches ``matplotlib.ticker.ScalarFormatter``
so tick labels that round to zero never render with a leading minus sign
(e.g. "-0.00" -> "0.00"); see ``fmt0`` for the manual-text-annotation
equivalent.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as _mcolors
from matplotlib.ticker import ScalarFormatter as _ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable as _make_axes_locatable


# Paper-figure rcParams, applied once at import time.
plt.rcParams.update({
    # Journal figure text: 5-7 pt. Body/labels/titles 7 pt, ticks and
    # legends 6 pt; panel letters are 8 pt bold (see panel_letter). Figures are now
    # authored at true on-page size (FULL_W/COL_W), so these pt values are literal.
    'font.size': 7,
    'font.family': 'Arial',
    'lines.linewidth': 0.7,
    'legend.fontsize': 6,
    'legend.frameon': False,
    'axes.titlesize': 7,
    'axes.labelsize': 7,
    'axes.labelpad': 1,
    'axes.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.formatter.limits': (-3, 3),
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'xtick.major.size': 2,
    'ytick.major.size': 2,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.pad': 1,
    'ytick.major.pad': 1,
    'xtick.minor.size': 1.5,
    'ytick.minor.size': 1.5,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
    'figure.dpi': 225,
    'figure.figsize': [4 / 2.54, 3 / 2.54],  # default panel size; override per-panel
    'path.simplify': True,
    'pdf.fonttype': 42,
    'svg.fonttype': 'none',
    'image.interpolation': 'none',
    'image.aspect': 'auto',
    'patch.edgecolor': 'none',
})


# ---------------------------------------------------------------------------
# THE PALETTE  (single source of truth -- edit here once)
# ---------------------------------------------------------------------------
# This palette was designed independently for the final main-text figures.
# Petrol and persimmon carry the two recurring scientific roles, warm graphite
# carries experimental data, and steel/ochre provide a second categorical pair.
# The hues stay legible in dense scientific panels and remain distinct at print
# size and under protanopia, deuteranopia and tritanopia simulations.
#
#   COOL ARM      petrol -> jade -> lichen
#   WARM ARM      persimmon -> copper -> ochre
#   NEUTRALS      warm graphite -> stone -> parchment
#
# Continuous fields are custom monotone-lightness maps. Signed fields use the
# same petrol/persimmon opposition as the categorical system. This avoids jet's
# false boundaries and makes the whole paper read as one designed object.
#
# SATURATION MODEL. Every chromatic value here is placed ON the sRGB gamut
# boundary rather than at an absolute CIELAB chroma. A requested chroma that
# sRGB cannot render has to be shrunk, and the amount shrunk varies with hue and
# lightness, so an absolute request delivers whatever survives, which was as
# little as a third of the ask in the dark half of the cool arm. Instead each
# colour asks for a FRACTION of the chroma that actually exists at its (L*, hue)
# (``_max_chroma``), so the request is always deliverable and the delivered
# saturation is uniform across hues. Anchors sit as light as the 4.5:1 page
# contrast guard allows (L* = 49.4) because in that band the available chroma
# still rises with lightness.

# Nominal hue of each arm, used by the signed and cyclic maps, which pivot and
# start on the arms rather than on the categorical anchors.
_H_PETROL, _H_PERSIMMON = 225, 38

# Hue of each categorical anchor. Placed by eye at the vivid level, then pushed
# to the gamut edge at _ANCHOR_L (see hexes below).
_ANCHOR_L = 49.4


# --- Neutrals ---------------------------------------------------------------
BLACK = '#242321'          # warm graphite, softer than process black
SLATE = '#68645E'          # secondary stone neutral
DATA = BLACK               # aggregate experimental reference
DATA_UNDERLAY = '#7B746A'  # shoulders around nearly coincident model/data curves
OVERLAP_DATA_LW = 2.8
OVERLAP_MODEL_LW = 1.4
INDIVIDUAL_DATA = '#A69D90'  # quiet warm gray for individual-mouse traces

# --- Chromatic anchors ------------------------------------------------------
# Each is L* = _ANCHOR_L at the hue in the trailing comment, at the full chroma
# sRGB offers there (AUBERGINE at 62% of it, since magenta's gamut is so wide
# that the full value reads as neon next to the others). PETROL and STEEL gain
# the least from the gamut-edge placement because the whole teal-to-blue sector
# of sRGB is chroma-poor at every lightness, which is a fact about the gamut.
PETROL     = '#008386'  # h 200; classical, ordered and reference structures
PERSIMMON  = '#D34300'  # h  48; generative, disordered and model structures
STEEL      = '#007CBA'  # h 262; first data partition
OCHRE      = '#AF6200'  # h  66; second data partition
PINE       = '#00865D'  # h 162; cool categorical support
AUBERGINE  = '#AB4FBE'  # h 322; quiet tertiary support
SIGNAL_ORANGE = '#E52300'  # h  42; high-salience overlay on activity fields

# Compatibility names retained for notebook and SI call sites. They map into
# this palette rather than preserving the hues implied by their old names.
INDIGO = AUBERGINE
BLUE = PETROL
CYAN = STEEL
VIOLET = AUBERGINE
PURPLE = AUBERGINE
MAGENTA = PERSIMMON
PINK = PERSIMMON
ORANGE = PERSIMMON
GREEN = PINE
GOLD = OCHRE
CRIMSON = PERSIMMON
CORAL = PERSIMMON
WINE = AUBERGINE
TEAL = PETROL

# --- Semantic roles ---------------------------------------------------------
MODEL = PERSIMMON
CLASSICAL = PETROL
J_COLOR = MODEL
J2_COLOR = CLASSICAL
LINE_A = CLASSICAL
LINE_B = MODEL
E_COLOR = PERSIMMON
I_COLOR = PETROL
HD_COLOR = SIGNAL_ORANGE

# These folds use identical geometry and line style, so this stronger steel /
# ochre pair is reserved for them. It clears the CVD-distance guard in all three
# complete-deficiency simulations.
PARTITION_A, PARTITION_B = STEEL, OCHRE
FOLD_A, FOLD_B = PARTITION_A, PARTITION_B
COM_GREEN = PERSIMMON
EMBED_ORDERED = J2_COLOR
EMBED_DISORDERED = J_COLOR
SCHEMATIC_BOX = '#F4EFE6'

# Unordered support series. Critical comparisons use the semantic pairs above
# and redundant marks; this cycle is for trajectories and secondary series.
CATEGORICAL = [PETROL, PERSIMMON, OCHRE, AUBERGINE, PINE, SLATE]


# ---------------------------------------------------------------------------
# COLORMAPS, generated from the same cool and warm arms
# ---------------------------------------------------------------------------
# The public accessor names remain stable because notebooks use them as semantic
# slots. Their actual maps are new:
#
#   ocean  : ordered cool parameter family, indigo -> jade -> lichen
#   ember  : ordered warm parameter family, persimmon -> copper -> ochre
#   ice    : indigo -> blue -> ice for activity fields
#   aurora : sunstone field map for warm loss landscapes
#   stability: indigo -> magenta -> coral -> gold for analytical phase diagrams
#   diverging: petrol -> ivory -> persimmon for signed matrices
#
# Chroma is everywhere a FRACTION of the locally available chroma, not an
# absolute CIELAB value; see the saturation-model note in the palette section.

SWEEP_L = (26, 61)          # CIELAB L* at the low and high ends of a line ramp
SWEEP_FRAC = 1.0            # share of the available chroma at each point on it
SWEEP_MIN_CONTRAST = 2.4    # floor for ANY sampled sweep colour
SWEEP_LABEL_CONTRAST = 3.0  # stricter floor where the colour is also drawn as TEXT
SWEEP_LABEL_KINDS = ('ocean',)
# 'slate' is achromatic, so it keeps the wider lightness span it needs to
# separate its steps; the chromatic ramps separate by hue as well as lightness.
_SWEEP_ENV = {
    'slate': ((20, 61), 1.0),
}

# 'ocean' STARTS in indigo rather than petrol. That is the only way to buy real
# chroma in the dark half of a cool ramp, since at L* = 30 the petrol hue tops
# out near C = 21 while indigo reaches 42. Still one cool arm, still monotone.
_SWEEP_HUES = {
    'ocean': [276, 180, 124],
    'ember': [26, 48, 82],
    'slate': None,
}

# 2-D scalar FIELD maps: one arm each, run over the full lightness range. Fields
# are area fill with a colorbar, so unlike the line ramps they may go pale.
FIELD_L = (8, 96)
FIELD_FRAC = 1.0
# Per-kind overrides. 'ember' retains a full near-black to near-white span for
# any future warm scalar field that needs the full dynamic range.
_FIELD_ENV = {
    'lagoon': ((22, 97), 1.0),
    'activity': ((7, 99), 1.0),
    'ember': ((16, 97), 1.0),
    'aurora': ((16, 97), 1.0),
    'stability': ((10, 97), 1.0),
}
_FIELD_HUES = {
    'lagoon': [232, 165, 110],
    # Activity deliberately stays on the blue side of the wheel: this path ends
    # in pale blue rather than the yellow-green a full cool arm would reach, so
    # it never competes with the orange integrated-velocity trace drawn over it.
    'activity': [285, 262, 234, 203],
    'ember': [26, 54, 90],
    'aurora': [26, 54, 90],
    # A long, monotone-lightness hue path gives the compact DMFT phase diagrams
    # visibly distinct levels without jet-like reversals or false boundaries.
    'stability': [298, 332, 14, 46, 80],
}

# --- generator machinery ----------------------------------------------------
_XYZ2RGB = np.linalg.inv(np.array([[0.4124564, 0.3575761, 0.1804375],
                                   [0.2126729, 0.7151522, 0.0721750],
                                   [0.0193339, 0.1191920, 0.9503041]]))
_D65 = np.array([0.95047, 1.0, 1.08883])


def _lab_to_rgb(L, a, b):
    """(internal) CIELAB -> sRGB, unclipped so callers can detect out-of-gamut."""
    fy = (L + 16) / 116
    f = np.stack([fy + a / 500, fy, fy - b / 200], axis=-1)
    d = 6 / 29
    xyz = np.where(f > d, f ** 3, 3 * d ** 2 * (f - 4 / 29)) * _D65
    lin = xyz @ _XYZ2RGB.T
    return np.where(lin <= 0.0031308, lin * 12.92,
                    1.055 * np.abs(lin) ** (1 / 2.4) - 0.055)


def _lch_to_rgb(L, C, h_deg):
    """(internal) One LCh point -> in-gamut sRGB, clipped by reducing CHROMA only.

    Lightness is never touched, so whatever contrast guarantee the caller built
    into its lightness profile survives the clip.
    """
    h = np.radians(h_deg)
    c = C
    for _ in range(90):
        rgb = _lab_to_rgb(L, c * np.cos(h), c * np.sin(h))
        if rgb.min() >= -1e-6 and rgb.max() <= 1 + 1e-6:
            return np.clip(rgb, 0, 1)
        c *= 0.96
    return np.clip(rgb, 0, 1)


def _hue_path(hues, t):
    """(internal) Interpolate a hue path, joining anchors the short way round."""
    hs = np.asarray(hues, dtype=float)
    for i in range(1, len(hs)):
        hs[i] = hs[i - 1] + (hs[i] - hs[i - 1] + 180) % 360 - 180
    return np.interp(t, np.linspace(0, 1, len(hs)), hs)


def _max_chroma(L, h_deg, hi=170.0, iters=28):
    """(internal) The largest CIELAB chroma sRGB can render at this L* and hue.

    Every saturated value in this module is expressed as a fraction of this,
    which is what keeps the delivered saturation even across hues; see the
    saturation-model note in the palette section.
    """
    lo, h = 0.0, np.radians(h_deg)
    for _ in range(iters):
        mid = (lo + hi) / 2
        rgb = _lab_to_rgb(L, mid * np.cos(h), mid * np.sin(h))
        if rgb.min() >= -1e-6 and rgb.max() <= 1 + 1e-6:
            lo = mid
        else:
            hi = mid
    return lo


def _smooth(x, sigma):
    """(internal) Gaussian-smooth a 1-D profile with edge padding."""
    if sigma <= 0:
        return x
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(x, r, mode='edge'), k, mode='valid')


def _lch_ramp(hues, name, L, frac, n=256, gamma=1.0, sigma=26.0):
    """Sequential ramp along a CIELAB hue path with linearly rising lightness.

    Lightness rising monotonically is what makes the ramp read as ORDERED; the
    hue rotation is what makes neighbouring samples DISCERNIBLE. A ramp needs
    both: lightness alone over a short hue arc cannot separate four or five
    curves, which is exactly why viridis rotates hue too.

    Chroma is ``frac`` of the chroma available at each point. The gamut ceiling
    is bumpy along a long hue path and following it exactly puts kinks in the
    relative luminance that break the CVD monotonicity guard, so the ceiling is
    smoothed first. That costs a little peak chroma and buys a clean ordering.
    """
    t = np.linspace(0, 1, n)
    u = t ** gamma
    Ls = L[0] + u * (L[1] - L[0])
    if hues is None:
        rgb = np.array([_lch_to_rgb(Ls[i], 0.0, 0.0) for i in range(n)])
    else:
        h = _hue_path(hues, u)
        ceil = np.array([_max_chroma(Ls[i], h[i]) for i in range(n)])
        ceil = np.minimum(_smooth(ceil, sigma), ceil)
        rgb = np.array([_lch_to_rgb(Ls[i], frac * ceil[i], h[i]) for i in range(n)])
    return _mcolors.LinearSegmentedColormap.from_list(name, rgb)


def _diverging_ramp(h_neg, h_pos, name, L_end=38, L_mid=97, frac=1.0, n=257):
    """Diverging map: one arm per sign, pivoting through near-white at zero.

    Built from the same two hue arms as everything else, so a signed matrix sits
    inside the paper palette. Chroma falls to zero at the pivot, so the zero
    level is unambiguous. The 0.40 exponent holds each arm near its available
    chroma over most of its span and releases it only close to the pivot, so a
    signed matrix reads as saturated rather than as two washed-out halves.
    """
    half = n // 2
    rgb = np.empty((n, 3))
    for i in range(half + 1):
        s = i / half                        # 0 at the saturated negative end
        L = L_end + s * (L_mid - L_end)
        rgb[i] = _lch_to_rgb(L, frac * (1 - s) ** 0.40 * _max_chroma(L, h_neg), h_neg)
    for i in range(half, n):
        s = (i - half) / (n - 1 - half)     # 0 at the pivot
        L = L_mid + s * (L_end - L_mid)
        rgb[i] = _lch_to_rgb(L, frac * s ** 0.40 * _max_chroma(L, h_pos), h_pos)
    return _mcolors.LinearSegmentedColormap.from_list(name, rgb)


def _cyclic_ramp(name, L=55, C=52, start_h=225, n=256):
    """Cyclic map for ANGLE: a full hue circle at CONSTANT lightness.

    Angle is periodic, so this is the one quantity that genuinely needs every hue.
    Holding lightness fixed is what makes it usable: every trajectory reads with
    the same weight against white, and 0 and 2*pi land on the identical colour.
    An honest hue circle in CIELAB rather than raw HSV, whose lightness swings
    wildly between yellow and blue and whose saturation reads as neon.

    ``C`` is requested at a level the sRGB gamut cannot hold at every hue, and
    the excess is clipped in RGB rather than by dropping chroma, which buys a
    markedly more saturated wheel at the cost of a small lightness ripple. The
    ripple is bounded by the isoluminance guard in tests/test_equivalence.py.
    """
    h = np.linspace(start_h, start_h + 360, n, endpoint=True)
    rgb = np.array([np.clip(_lab_to_rgb(L, C * np.cos(np.radians(hi)),
                                        C * np.sin(np.radians(hi))), 0, 1)
                    for hi in h])
    return _mcolors.ListedColormap(rgb, name=name)


SWEEP_CMAPS = {k: _lch_ramp(h, f'ring_sweep_{k}', *_SWEEP_ENV.get(k, (SWEEP_L, SWEEP_FRAC)))
               for k, h in _SWEEP_HUES.items()}
LAGOON_CMAP = _lch_ramp(
    _FIELD_HUES['lagoon'], 'ring_field_lagoon', *_FIELD_ENV['lagoon'], gamma=0.65)
ACTIVITY_CMAP = _lch_ramp(
    _FIELD_HUES['activity'], 'ring_field_activity', *_FIELD_ENV['activity'])
SUNSTONE_CMAP = _lch_ramp(
    _FIELD_HUES['aurora'], 'ring_field_sunstone', *_FIELD_ENV['aurora'], gamma=0.68)
STABILITY_CMAP = _lch_ramp(
    _FIELD_HUES['stability'], 'ring_field_stability',
    *_FIELD_ENV['stability'], gamma=0.92)
COVARIANCE_CMAP = LAGOON_CMAP
FIELD_CMAPS = {
    'ice': ACTIVITY_CMAP,
    'ember': SUNSTONE_CMAP,
    'aurora': SUNSTONE_CMAP,
    'stability': STABILITY_CMAP,
}

DIVERGING_CMAP = _diverging_ramp(
    _H_PETROL, _H_PERSIMMON, 'ring_diverging_petrol_persimmon',
    L_end=38, L_mid=97, frac=1.0)
ANGLE_CMAP = _cyclic_ramp('ring_angle', L=57, C=70, start_h=_H_PETROL)
GRID_CELL_CMAP = LAGOON_CMAP

for _cm in list(SWEEP_CMAPS.values()) + list(FIELD_CMAPS.values()) + [DIVERGING_CMAP, ANGLE_CMAP]:
    if _cm.name not in plt.colormaps():
        plt.matplotlib.colormaps.register(_cm)


def sweep_cmap(kind='slate'):
    """The colormap for an ordered-parameter family of curves. See ``_SWEEP_HUES``."""
    return SWEEP_CMAPS[kind]


def sweep_colors(n, kind='slate', reverse=False):
    """``n`` colors, low value -> high value, for an ordered-parameter family.

    ``kind`` names a RAMP ('ocean', 'ember', 'slate'); see the per-figure table
    in this module for which figure uses which. The full [0, 1] span is sampled:
    these ramps have no unreadable end, so call sites do not clip.
    """
    t = np.linspace(1, 0, n) if reverse else np.linspace(0, 1, n)
    return SWEEP_CMAPS[kind](t)


def field_cmap(kind='ice'):
    """The colormap for a 2-D scalar field, keyed by quantity. See ``_FIELD_HUES``."""
    return FIELD_CMAPS[kind]


# Make the palette the default cycle for all line plots, and the activity field
# map the default image cmap so a bare imshow() is controlled here (every other
# field kind, and the diverging map, are passed explicitly via cmap=).
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=CATEGORICAL)
plt.rcParams['image.cmap'] = FIELD_CMAPS['ice'].name


# ---------------------------------------------------------------------------
# STANDARD FIGURE WIDTHS  (journal specification)
# ---------------------------------------------------------------------------
FULL_W = 183 / 25.4   # double column = 183 mm ~ 7.20 in
COL_W  = 89 / 25.4    # single column =  89 mm ~ 3.50 in
MAX_H  = 170 / 25.4   # max figure height = 170 mm ~ 6.69 in

# Two standard heights for double-column main figures, so they share width AND
# height for a harmonious set. SHORT is the reference (compact 3-4 panel row);
# TALL is for figures that genuinely need vertical room. Both < MAX_H. Save each
# figure at its true on-page size (FULL_W/COL_W wide) and \includegraphics it at
# the SAME physical width (183mm / 89mm), so the pt sizes in code == pt on page.
# Two standard heights for full-width main figures. TALL is the default (== the
# beloved Fig-7 height); SHORT is the compact variant (Fig 3). Both deliberately
# small so figures place inline near their text rather than piling up as floats.
TALL_H  = 3.5         # ~89 mm  (default; Figs 1,2,6,7,8)
SHORT_H = 2.5         # ~64 mm  (compact; Fig 3)

# Standard colorbar geometry (one thickness everywhere). CBAR_THICK is the bar's
# short-axis size as a fraction of its parent axis, for inset_axes-style bars;
# FRACTION/PAD/ASPECT are for plt.colorbar(...) attached bars.
CBAR_THICK    = 0.052   # bumped 1.15x per review (Fig 2A reference thickness)
CBAR_FRACTION = 0.046
CBAR_PAD      = 0.03
CBAR_ASPECT   = 30
CBAR_SIZE     = '4.6%' # THE standard colorbar thickness (% of parent panel's short side; 1.15x per review)
CBAR_GAP      = 0.06   # gap between a panel and its colorbar


def attach_cbar(fig, mappable, ax, side='right', ticks=None, ticklabels=None, label=None):
    """THE standard colorbar: a thin bar attached to ONE panel (never spanning
    several panels). ``side`` = 'right'/'left' (vertical) or 'top'/'bottom'
    (horizontal). Identical thickness (CBAR_SIZE) and gap everywhere -- use this
    instead of a hand-rolled inset so every colorbar in the paper matches.
    """
    orient = 'vertical' if side in ('right', 'left') else 'horizontal'
    cax = _make_axes_locatable(ax).append_axes(side, size=CBAR_SIZE, pad=CBAR_GAP)
    cbar = fig.colorbar(mappable, cax=cax, orientation=orient)
    if ticks is not None:
        cbar.set_ticks(ticks)
    if ticklabels is not None:
        cbar.set_ticklabels(ticklabels)
    if label is not None:
        cbar.set_label(label)
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(width=0.5)
    return cbar


def add_colorbar(fig, mappable, ax, orientation='vertical', ticks=None,
                 ticklabels=None, label=None, **kw):
    """Attach a colorbar with the house thickness/pad to one or more axes.

    Standardizes bar thickness across figures. For bars placed in an explicit
    inset/added axis, size that axis with ``CBAR_THICK`` instead and pass it as
    ``cax`` to ``fig.colorbar`` directly.
    """
    cbar = fig.colorbar(mappable, ax=ax, orientation=orientation,
                        fraction=CBAR_FRACTION, pad=CBAR_PAD,
                        aspect=CBAR_ASPECT, **kw)
    if ticks is not None:
        cbar.set_ticks(ticks)
    if ticklabels is not None:
        cbar.set_ticklabels(ticklabels)
    if label is not None:
        cbar.set_label(label)
    cbar.outline.set_linewidth(0.5)
    return cbar


def panel_letter(ax, letter, x=-0.02, y=1.02, ha='right', va='bottom', **kw):
    """Draw a panel letter in the house style: 8 pt bold, upright.

    Standardizes size/weight/anchor across all figures (position is still
    tunable via x, y in axes-fraction coordinates). For figure-level placement
    use ``fig.text(x, y, letter, fontsize=8, fontweight='bold', ...)`` directly.
    Letters are LOWERCASE, matching the captions and in-text callouts.
    """
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8,
            fontweight='bold', ha=ha, va=va, **kw)


# Patch ScalarFormatter so tick labels that round to zero ("-0.00") render
# without the leading minus. The patch is applied at module import; see
# `fmt0` for the manual-annotation equivalent.
_original_scalar_call = _ScalarFormatter.__call__


def _scalar_call_no_neg_zero(self, x, pos=None):
    s = _original_scalar_call(self, x, pos)
    if s and s[0] in ('-', '−'):
        rest = s[1:]
        try:
            if float(rest.replace('−', '-')) == 0:
                return rest
        except ValueError:
            pass
    return s


_ScalarFormatter.__call__ = _scalar_call_no_neg_zero


def fmt0(value, fmt='{:.2f}'):
    """Format a number, mapping near-zero to positive zero (avoids "-0.00").

    Tick labels are handled automatically by the ScalarFormatter patch in this
    module; use ``fmt0`` for manually-formatted text annotations inside
    ``ax.text`` / ``ax.annotate`` calls, e.g.
    ``ax.text(x, y, f"$\\\\kappa = {fmt0(k)}$")``.

    Parameters
    ----------
    value : float
        Number to format.
    fmt : str
        Python ``str.format`` spec. Default ``"{:.2f}"``.

    Returns
    -------
    s : str
        Formatted string with the leading minus stripped if it represents a
        rounded zero; otherwise the unchanged format output.
    """
    s = fmt.format(value)
    if s.startswith('-'):
        try:
            if float(s) == 0:
                return s[1:]
        except ValueError:
            pass
    return s
