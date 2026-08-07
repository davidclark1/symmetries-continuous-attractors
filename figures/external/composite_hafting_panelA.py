"""
composite_hafting_panelA.py

Build a candidate panel-A image for our grid-cell figure from Hafting et al. 2005
(Nature 03721) Fig 2C: three example medial-entorhinal grid cells (t7c1, t7c2,
t7c3), each shown with three circular maps:
  (1) spike-on-trajectory  (red spikes on the grey run path),
  (2) firing-rate map       (blue->red rate heatmap; peak rate 15/14/12 Hz),
  (3) spatial autocorrelogram (the hexagonal grid pattern).

In the source screenshot the three cells are stacked TOP-TO-BOTTOM. Our figure
top row is wide and short, so this composite lays the three cells out
LEFT-TO-RIGHT as three groups, each group its three maps in a small row, with the
cell label above and the peak-rate (Hz) below the rate map.

The composite is authored at the EXACT on-page size of panel A's axes box in
grid_cell_simulations.png (3.88 x 0.957 in, aspect 4.05, measured from the
figure's gridspec), so it maps into that box with matplotlib's aspect='equal'
imshow leaving essentially no left/right whitespace, and so the label-to-circle
gap is a literal 6 pt (matching the rcParams axes.titlepad used by the pure
matplotlib panels of the figure). The nine circular maps are packed edge-to-edge
across the full width (they are width-limited by nine-across-in-a-4:1-box), which
is as large as round maps can be in this band. The nine individual circular crops
are also written to derived/crops/ so the layout can be reassembled differently.

Source screenshot: 'Screenshot 2026-07-23 at 11.33.19 AM.png' (1476 x 1594).
Circle bounding boxes were located by detecting saturated coloured pixels
(max-min channel spread > 45) and splitting the merged spike+rate band at the
content valley near col 553; boxes below are the resulting per-map windows,
padded to squares.
"""

import glob
import os

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CROPDIR = os.path.join(HERE, "crops")
os.makedirs(CROPDIR, exist_ok=True)

# Source screenshot lives outside the repo, because it is a capture of a
# copyrighted published figure and is not ours to redistribute. Point
# ``HAFTING_SRC_DIR`` at a directory holding a capture of Hafting et al. (2005)
# Fig. 2c to rerun this. The committed crops and composite were built from such
# a capture, so a reader who only wants the figure needs none of this.
SRC_DIR = os.environ.get("HAFTING_SRC_DIR", os.path.join(HERE, "src"))
_matches = sorted(glob.glob(os.path.join(SRC_DIR, "Screenshot*11.33.19*.png")))
if not _matches:
    raise SystemExit(
        f"No source capture found in {SRC_DIR}.\n"
        "Set HAFTING_SRC_DIR to a directory containing the Hafting et al. (2005) "
        "Fig. 2c capture. See README.md in this directory."
    )
SRC = _matches[0]

# column windows for the three map types (spike-traj, rate map, autocorrelogram)
COL_WIN = [(130, 553), (553, 978), (1028, 1458)]
# row windows for the three cells (t7c1, t7c2, t7c3)
ROW_WIN = [(26, 497), (561, 1049), (1090, 1563)]
CELLS = ["t7c1", "t7c2", "t7c3"]
TYPES = ["spike", "rate", "autocorr"]
HZ = ["15 Hz", "14 Hz", "12 Hz"]
PAD = 3  # white px around the tight coloured bbox (small -> circle fills its crop)

# On-page geometry of panel A's axes box in grid_cell_simulations.png (inches),
# measured from that figure's gridspec at height style.TALL_H. Authoring at this
# exact size makes aspect='equal' imshow fill the box and makes the 6 pt label pad
# literal.
BOX_W, BOX_H = 3.88, 0.957
DPI = 700                 # ~2700 px wide, matching the previous export's sharpness
WS = 0.05                 # inter-map gap (fraction of a map's width)
BAND_TOP, BAND_BOT = 0.72, 0.29   # circle row occupies this vertical fraction
LABEL_PAD_PT = 6.0        # t7cN label sits this many pt above the circles (== titlepad)
LABEL_FS = 7              # cell label size (matches the 7 pt panel titles)
HZ_FS = 6                 # peak-rate size (matches 6 pt tick text)


def tight_square(im, r0, r1, c0, c1):
    """Square crop centred on the coloured map blob, with everything OUTSIDE the
    circular map set to white. Masking the corners removes the neighbouring cell
    label ('t7c1' ...) and the baked-in 'NN Hz' text that would otherwise bleed
    into the crop."""
    sub = im[r0:r1, c0:c1]
    R, G, B = (sub[:, :, i].astype(int) for i in range(3))
    mx = np.maximum(np.maximum(R, G), B)
    mn = np.minimum(np.minimum(R, G), B)
    colored = (mx - mn > 45) & (mx > 90)
    ys, xs = np.where(colored)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    ay0, ay1 = r0 + y0, r0 + y1
    ax0, ax1 = c0 + x0, c0 + x1
    cy, cx = (ay0 + ay1) / 2, (ax0 + ax1) / 2
    # circle radius from the coloured extent; square half-size adds PAD margin
    radius = max(ay1 - ay0, ax1 - ax0) / 2
    half = radius + PAD
    ry0 = int(max(0, cy - half)); ry1 = int(min(im.shape[0], cy + half))
    rx0 = int(max(0, cx - half)); rx1 = int(min(im.shape[1], cx + half))
    crop = im[ry0:ry1, rx0:rx1].copy()

    # white out everything beyond the circle (removes stray label / Hz text)
    hh, ww = crop.shape[:2]
    yy, xx = np.mgrid[0:hh, 0:ww]
    ccx, ccy = cx - rx0, cy - ry0
    outside = (xx - ccx) ** 2 + (yy - ccy) ** 2 > (radius + 4) ** 2
    crop[outside] = 255
    return crop


def main():
    im = np.array(Image.open(SRC).convert("RGB"))

    crops = {}
    for ci, (r0, r1) in enumerate(ROW_WIN):
        for ti, (c0, c1) in enumerate(COL_WIN):
            crop = tight_square(im, r0, r1, c0, c1)
            crops[(ci, ti)] = crop
            fn = os.path.join(CROPDIR, f"{CELLS[ci]}_{TYPES[ti]}.png")
            Image.fromarray(crop).save(fn)

    # ---- assemble: nine maps edge-to-edge in one row, authored at panel A's
    #      on-page box size so it drops into the figure with no wasted margin ----
    fig = plt.figure(figsize=(BOX_W, BOX_H), dpi=DPI)
    # square-ish cells (cell width ~= band height) so the round maps fill each cell
    gs = fig.add_gridspec(1, 9, left=0.004, right=0.996,
                          top=BAND_TOP, bottom=BAND_BOT, wspace=WS)
    axes = []
    for ci in range(3):
        for ti in range(3):
            ax = fig.add_subplot(gs[0, 3 * ci + ti])
            ax.imshow(crops[(ci, ti)])  # aspect defaults to 'equal' -> round
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            axes.append(ax)

    fig.canvas.draw()  # positions are final; place labels relative to the maps
    pad_frac = LABEL_PAD_PT / (72.0 * BOX_H)
    for ci in range(3):
        grp = [axes[3 * ci + t].get_position() for t in range(3)]
        xc = (grp[0].x0 + grp[2].x1) / 2
        top = max(p.y1 for p in grp)
        fig.text(xc, top + pad_frac, CELLS[ci], ha="center", va="bottom",
                 fontsize=LABEL_FS)
        # peak-rate under the rate map (middle map of the group)
        rate = grp[1]
        fig.text((rate.x0 + rate.x1) / 2, rate.y0 - pad_frac, HZ[ci],
                 ha="center", va="top", fontsize=HZ_FS)

    out = os.path.join(HERE, "hafting2005_grid_ratemaps.png")
    fig.savefig(out, dpi=DPI)
    print("wrote", out)
    print("wrote 9 crops to", CROPDIR)
    for k, v in crops.items():
        print(f"  {CELLS[k[0]]}_{TYPES[k[1]]}: {v.shape[1]}x{v.shape[0]}")


if __name__ == "__main__":
    main()
