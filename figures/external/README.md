# Borrowed figure assets

**The files here are not covered by this repository's MIT licence.** They are other groups'
published experimental results, reproduced under permission granted for this paper only. To
reuse them you need your own permission from the rights holder. Everything else in
`figures/` is ours and is covered by the licence.

| file | used in | source | terms |
|---|---|---|---|
| `hafting2005_grid_ratemaps.png`, `crops/` | grid-cell figure, panel a | Hafting et al. (2005), *Nature* **436**, 801–806, Fig. 2c | Springer Nature licence 6323150710897 |
| `experimental_gridness_scores.npy` | grid-cell figure, panel b | grid scores digitized from Nayebi et al. (2021) Fig. 3A, which analysed data from Mallory et al. (2021) | these are data rather than artwork, and the underlying recordings are CC BY |

Only these two panels come from elsewhere. Every other panel in that figure, and every other
figure in the paper, is generated from the code in this repository.

## How each was made

`hafting2005_grid_ratemaps.png` is assembled by `composite_hafting_panelA.py`, which crops
the nine circular maps (spike-on-trajectory, rate map and autocorrelogram, for cells t7c1,
t7c2 and t7c3) out of a capture of the published figure and lays them edge to edge at the
size of the panel's axes box. The capture itself is deliberately not in this repository,
since it is an image of a copyrighted figure. To rerun the script, point `HAFTING_SRC_DIR`
at a directory holding your own capture. The committed composite and the crops under
`crops/` are its output, so reproducing the figure does not need any of this.

`experimental_gridness_scores.npy` holds values digitized from the published histogram by
`digitize_nayebi_fig3a.py` and replotted as our own panel. It ships with the cached data on
Zenodo rather than with the code, since it is a data array.
