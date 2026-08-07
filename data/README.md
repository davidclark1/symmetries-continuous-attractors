# `data/` — cached intermediate results

**This directory is empty in the repository. Download its contents from Zenodo:**

    https://doi.org/10.5281/zenodo.21827768

Unpack the archive here, so that the `.npz` files sit directly in `data/` and `tc_data.npz`
sits in `data/mouse_data/`. There are 21 files totalling 2.13 GB. Verify the download with:

```bash
md5sum -c CHECKSUMS.md5
```

The notebooks load these files instead of re-running the expensive simulations and fits, so
the figure-plotting cells finish quickly. The notebooks will not run without them.

## Where each file comes from

Every cache is reproducible from the raw recordings and the code in this repository. Each
producer is guarded by a `REGENERATE_CACHES = False` flag at the top of its notebook, so a
normal run cannot overwrite one. `FIGURES.md` lists the producer of each cache, and
`EQUIVALENCE_LEDGER.md` gives the per-cache regeneration verdicts.

- **`mouse_data/tc_data.npz`** — tuning curves built from the raw NWB recordings by
  `python -m ring.data`. The raw data are from the Peyrache lab,
  [DANDI Dandiset 000939](https://dandiarchive.org/dandiset/000939), 52 GB and not in the
  repository. Point `RING_DATA_DIR` at your own download; output lands in
  `data/mouse_data/` unless you override `RING_OUTPUT_DIR`. Re-running overwrites the
  released cache, whose md5 is `6e1f3a17ab67e2e0eb8fac35d89e2d57`.
- **`grid_search_results.npz`** — the (σ, β, b) generative-parameter fit, produced by
  `python -m ring.gp_opt` in about 13 minutes. It reads `gamma_phi_info.npz`, which
  `data_and_generative_model.ipynb` writes. Both paths resolve relative to the `ring`
  package, so the command works from any directory.
- **`spurious_fixed_points.npz`** — PC-projected initial and final states of the finite-*N*
  spurious-fixed-point search, written by `spurious_fixed_points.ipynb` in about 2 minutes.
- **`closing_the_loop.npz`** and **`actual_hd_timeseries_2.npz`** — written by
  `closing_the_loop.ipynb`, the one notebook that reads the raw NWB recordings directly
  rather than working from caches. Regenerating them needs `RING_DATA_DIR`. Their consumer,
  `data_driven_attractor.ipynb`, needs only the caches.
- **All other caches** — written by the notebook whose cell produces them.
