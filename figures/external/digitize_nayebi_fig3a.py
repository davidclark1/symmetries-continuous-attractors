"""
digitize_nayebi_fig3a.py

Digitize the GREY experimental data histogram from Nayebi et al. 2021, Fig 3A
(top-left sub-panel, "Grid Cell Model"). The grey histogram is the distribution
of grid scores of medial entorhinal cortex (MEC) units (experimental data of
Mallory et al. 2021); the thin dark-blue bars overlaid on it are the model's
grid-score distribution. We recover ONLY the grey experimental distribution so it
can be redrawn as the reference distribution in our own figure panel B.

Source: a clean zoomed screenshot crop of the published figure
    'Screenshot 2026-07-23 at 11.30.30 AM.png'  (886 x 514 px)
cross-checked against the full four-panel crop '...11.30.20 AM.png'.

Method
------
1. Axis calibration (the axis is linear even though tick VALUES are unevenly
   labelled). Pixel positions are read from the published figure:
     x-axis  "Grid Score":  tick label centres (px) map to values
        0.00 -> 251.5, 0.25 -> 344, 0.50 -> 442.5, 1.0 -> 621, 1.5 -> 806
        A linear fit gives  px = 368.97*value + 253.17  (residuals < 5 px).
     y-axis  normalized "Unit Count" (density): tick label centres (row px)
        0.25 -> 96.5, 0.15 -> 247.5, 0.00 -> 474.5  =>  1512 px per unit.
     The bars sit on the x-axis line at row 468 (measured), which is the y=0
     baseline used for bar heights.
2. Grey-pixel classification. A pixel is a grey data-bar pixel if R,G,B are
   roughly equal (channel spread < 18), it is not white background (max < 232),
   not black axis/text (min > 60), and not the blue model overlay (B > R+25).
3. Per-column bar height = length of the grey run CONNECTED UPWARD FROM THE
   BASELINE (small gaps tolerated). Measuring from the baseline (rather than the
   topmost grey pixel anywhere in the column) ignores the anti-aliased grey edges
   of the dark-blue "Grid Cell Model / KS=..." caption text that floats in the
   upper-right of the panel and is not connected to any bar.
   The blue overlay is opaque, so at columns where a blue BAR (also measured as a
   run connected from the baseline) is TALLER than the grey behind it the grey is
   fully occluded and reads ~0. Blue bars are thin (~8 px) relative to the 37 px
   bins, so within each bin most columns are un-occluded; per bin we take the
   MEDIAN grey height over the NON-occluded interior columns, which recovers the
   true grey height even for bins carrying a tall thin blue bar.
4. Bin grid. Clean bar-to-bar transitions in the left half of the panel occur
   every ~36.8 px (bin width ~0.10 in grid-score units), with the first left
   edge at col 123. We fit a uniform grid from these transitions and use it for
   all bins, including those partly hidden by the blue overlay.

Outputs (written next to this script, in derived/)
    gridness_hist_digitized.npz : bin_edges, density  (+ metadata arrays)
    gridness_scores.npy         : a representative sample whose
        np.histogram(scores, bins=bin_edges, density=True) reproduces `density`
    panelB_compare.png          : digitized histogram vs the source crop

Reproducible: reads the screenshot, no hidden state; run under `ring-local`.

Citation for the recovered data:
    Nayebi et al. 2021 (NeurIPS) Fig 3A grey data histogram (grid scores of MEC
    units; experimental data of Mallory et al. 2021); digitized from the
    published figure.
"""

import glob
import os

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_GLOB = os.path.join(
    os.path.dirname(HERE), "Screenshot*11.30.30*.png"
)

# --- calibration constants (see module docstring) ---
X_SLOPE, X_INTCP = 368.966, 253.172          # px = X_SLOPE*value + X_INTCP
PX_PER_YUNIT = 1512.0                         # y pixels per density unit
BASELINE_ROW = 468                            # x-axis line (y = 0)
BIN_W_PX = (307 - 123) / 5.0                  # 36.8 px  (clean transitions)
BIN_LEFT_PX = 123.0                           # left edge of first grey bin
N_BINS = 21                                   # covers grey extent to ~col 895


def col_to_value(px):
    return (np.asarray(px) - X_INTCP) / X_SLOPE


GAP_TOL = 3  # px gaps tolerated inside a connected run (anti-aliasing)


def classify(im):
    R, G, B = (im[:, :, i].astype(int) for i in range(3))
    mx = np.maximum(np.maximum(R, G), B)
    mn = np.minimum(np.minimum(R, G), B)
    spread = (np.abs(R - G) < 18) & (np.abs(G - B) < 18) & (np.abs(R - B) < 18)
    is_blue = B > R + 25
    grey = spread & (mx < 232) & (mn > 60) & ~is_blue
    blue = is_blue & (B > 90)
    return grey, blue


def connected_height(mask, col):
    """Length (px) of the run of True in `mask` connected upward from BASELINE_ROW,
    tolerating gaps up to GAP_TOL."""
    h = 0
    gap = 0
    r = BASELINE_ROW - 1
    while r > 0:
        if mask[r, col]:
            h = BASELINE_ROW - r
            gap = 0
        else:
            gap += 1
            if gap > GAP_TOL:
                break
        r -= 1
    return h


def main():
    src = sorted(glob.glob(SRC_GLOB))[0]
    im = np.array(Image.open(src).convert("RGB"))
    grey, blue = classify(im)

    W = im.shape[1]
    grey_h = np.array([connected_height(grey, c) for c in range(W)])
    blue_h = np.array([connected_height(blue, c) for c in range(W)])
    occluded = blue_h > grey_h  # blue bar fully hides the grey behind it

    # uniform bin grid in pixels -> grid-score values
    edge_px = BIN_LEFT_PX + BIN_W_PX * np.arange(N_BINS + 1)
    bin_edges = col_to_value(edge_px)

    # per-bin height = median grey over NON-occluded interior columns
    density = np.zeros(N_BINS)
    for i in range(N_BINS):
        lo, hi = edge_px[i], edge_px[i + 1]
        cols = np.arange(int(np.ceil(lo + 3)), int(np.floor(hi - 3)) + 1)
        cols = cols[(cols >= 0) & (cols < W)]
        usable = cols[~occluded[cols]]
        if len(usable) == 0:            # entirely occluded -> interpolate later
            density[i] = np.nan
            continue
        density[i] = np.median(grey_h[usable]) / PX_PER_YUNIT

    # recover any fully-occluded bin by interpolation from neighbours
    nan = np.isnan(density)
    if nan.any():
        idx = np.arange(N_BINS)
        density[nan] = np.interp(idx[nan], idx[~nan], density[~nan])

    # clip tiny noise-floor bins (single stray grey rows) to zero
    density[density < 0.004] = 0.0

    # ---- build a representative sample reproducing the SHAPE ----
    # The Nayebi y-axis is a NORMALIZED unit count (relative frequency): the bar
    # heights sum to ~1 (see printed sum below), they do NOT integrate to 1,
    # so `density` is a per-bin fraction, not a probability density. A sample that
    # reproduces the shape therefore has per-bin count proportional to the height:
    #     count_i = round(density_i * N_total).
    # A consumer can then redraw the grey reference either as
    #   ax.hist(scores, bins=bin_edges, density=True)   (same shape, y rescaled), or
    #   ax.hist(scores, bins=bin_edges, weights=ones/N)  (heights as shown).
    binwidth = np.diff(bin_edges)
    N_total = 620  # number of MEC units in the source data (Nayebi 2021 states
    #                "the 620 units in the data"; experimental data of Mallory et al.)
    counts = np.round(density * N_total).astype(int)
    rng = np.random.default_rng(0)
    samples = []
    for i, n in enumerate(counts):
        if n > 0:
            samples.append(rng.uniform(bin_edges[i], bin_edges[i + 1], size=n))
    scores = np.concatenate(samples) if samples else np.array([])

    # ---- save ----
    np.savez(
        os.path.join(HERE, "gridness_hist_digitized.npz"),
        bin_edges=bin_edges,
        density=density,
        bin_centers=0.5 * (bin_edges[:-1] + bin_edges[1:]),
        counts=counts,
        source=os.path.basename(src),
    )
    np.save(os.path.join(HERE, "gridness_scores.npy"), scores)

    # ---- verification figure: digitized vs source crop ----
    fig, axes = plt.subplots(2, 1, figsize=(7, 7))
    axes[0].imshow(im)
    axes[0].set_title("source crop (Nayebi 2021 Fig 3A, top-left)")
    axes[0].axis("off")

    ax = axes[1]
    ax.bar(bin_edges[:-1], density, width=binwidth, align="edge",
           color="0.7", edgecolor="0.4", linewidth=0.4, label="digitized grey data")
    cnt_re, _ = np.histogram(scores, bins=bin_edges)
    d_re = cnt_re / scores.size  # relative frequency, matches the shown heights
    ax.step(bin_edges, np.append(d_re, d_re[-1]), where="post",
            color="crimson", lw=1.2, label="resampled (from gridness_scores.npy)")
    ax.set_xlabel("Grid Score")
    ax.set_ylabel("Unit Count (normalized)")
    ax.set_xlim(bin_edges[0], bin_edges[-1])
    for v in [-0.25, 0.0, 0.25, 0.50, 1.0, 1.5]:
        ax.axvline(v, color="0.85", lw=0.6, zorder=0)
    ax.set_xticks([-0.25, 0.0, 0.25, 0.50, 1.0, 1.5])
    ax.set_yticks([0.0, 0.15, 0.25])
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "panelB_compare.png"), dpi=150)

    # ---- console table ----
    print(f"source: {os.path.basename(src)}")
    print(f"bin width: {BIN_W_PX:.2f} px = {np.mean(binwidth):.4f} grid-score units")
    print(f"{'bin':>3} {'edge_lo':>8} {'edge_hi':>8} {'density':>8} {'count':>6}")
    for i in range(N_BINS):
        print(f"{i:>3} {bin_edges[i]:>8.3f} {bin_edges[i+1]:>8.3f} "
              f"{density[i]:>8.4f} {counts[i]:>6d}")
    print(f"N samples: {scores.size}")
    print(f"sum(density) = {np.sum(density):.4f}   (relative-frequency, ~1)")
    print(f"integral sum(density*binwidth) = {np.sum(density*binwidth):.4f}")


if __name__ == "__main__":
    main()
