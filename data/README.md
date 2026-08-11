# `data/`: cached intermediate results

**This directory is empty in the repository. Download its contents from Zenodo:**

    https://doi.org/10.5281/zenodo.21827767

Unpack the archive here, so that the `.npz` files sit directly in `data/` and `tc_data.npz`
sits in `data/mouse_data/`. There are 21 files totalling 2.13 GB. Verify the download with:

```bash
md5sum -c CHECKSUMS.md5
```

The notebooks load these files instead of re-running the expensive simulations and fits, so
the figure-plotting cells finish quickly. The notebooks will not run without these files.

## Where each file comes from

Every cache can be rebuilt with the code in this repository; only the three files marked
below also need the raw recordings. The
cell that writes a cache only runs when `REGENERATE_CACHES = True` is set at the top of its
notebook, so a normal run cannot overwrite what you downloaded. `FIGURES.md` says which caches each figure reads.

- **`mouse_data/tc_data.npz`**: tuning curves built from the raw NWB recordings by
  `python -m ring.data`. The raw data are from the Peyrache lab,
  [DANDI Dandiset 000939](https://dandiarchive.org/dandiset/000939), 52 GB and not in the
  repository. Point `RING_DATA_DIR` at your own download; output lands in
  `data/mouse_data/` unless you override `RING_OUTPUT_DIR`. Rerunning the command replaces
  the released file, so keep a copy if you want to compare the two.
- **`grid_search_results.npz`**: the (σ, β, b) generative-parameter fit, produced by
  `python -m ring.gp_opt` in about 13 minutes. The command reads `gamma_phi_info.npz`, which
  `data_and_generative_model.ipynb` writes. Both file paths resolve relative to the `ring`
  package, so the command works from any directory.
- **`spurious_fixed_points.npz`**: PC-projected initial and final states of the finite-*N*
  spurious-fixed-point search, written by `spurious_fixed_points.ipynb` in about 2 minutes.
- **`closing_the_loop.npz`** and **`hd_timeseries.npz`**: written by
  `closing_the_loop.ipynb`, the one notebook that reads the raw NWB recordings directly
  rather than working from caches. Regenerating these two files needs
  `RING_DATA_DIR`. `data_driven_attractor.ipynb`, which reads them, needs only the caches,
  not the recordings.
- **All other caches**: each is written by the notebook that reads it.
