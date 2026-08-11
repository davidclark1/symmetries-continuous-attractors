# Symmetries and Continuous Attractors in Disordered Neural Circuits

Documentation for the code accompanying the paper of that name, by David G. Clark,
L. F. Abbott and Haim Sompolinsky. The code is on
[GitHub](https://github.com/davidclark1/symmetries-continuous-attractors) and archived at
[10.5281/zenodo.21829584](https://doi.org/10.5281/zenodo.21829584).

Each figure in the paper is drawn by one of the eleven notebooks. The notebooks read
precomputed results from `data/`: tuning curves extracted from the recordings, and the
outputs of the slow simulations and fits. The recordings themselves are from the Peyrache
lab, published separately as
[DANDI Dandiset 000939](https://dandiarchive.org/dandiset/000939); you do not need them
to reproduce the figures.

## Start here

Install the environment and the package.

```bash
git clone https://github.com/davidclark1/symmetries-continuous-attractors.git
cd symmetries-continuous-attractors
conda env create -f ring-local.yml
conda activate ring-local
pip install -e .
```

Download the cached results, 2.1 GB, from
[10.5281/zenodo.21827767](https://doi.org/10.5281/zenodo.21827767) and unpack the archive into
`data/`, so that the `.npz` files sit directly in `data/` and `tc_data.npz` sits in
`data/mouse_data/`. Then check that the download is intact and that the package imports and
computes what it should.

```bash
cd data && md5sum -c CHECKSUMS.md5 && cd ..
python tests/test_equivalence.py
```

Now open any notebook and run it from top to bottom.

```bash
jupyter lab notebooks/dmft.ipynb
```

`dmft.ipynb` is a good notebook to start with, because it reads two small caches, involves no random
numbers and finishes quickly. Figure 7 of the paper appears inline as the last cell runs.
Nothing is written to disk unless you change one of the two settings described below.

## What is in each notebook

| Notebook | Purpose | Figures |
|----------|--------------|---------|
| `data_and_generative_model.ipynb` | Characterizes the tuning curves measured in each mouse, tests the circular symmetry of their distribution, and fits the generative process to them | 1, 3, 4, 5, S13 |
| `data_driven_attractor.ipynb` | Builds the ring attractor directly from the recordings and tests its closed-loop velocity integration | 2 |
| `spectra.ipynb` | Compares the spectra and eigenvector geometry of the disordered and circulant weight matrices | 6 |
| `dmft.ipynb` | Solves the dynamical mean-field theory of the disordered ring attractor and maps the stability of its solutions | 7 |
| `grid_sims.ipynb` | Simulates grid cells from the disordered attractor and computes the spectrum of their connectivity | 8, S15 |
| `weight_matrix_analyses.ipynb` | Examines the weight matrix's low-rank structure and singular vectors, compares it with spectrum-matched surrogates, and measures drift under weight noise | S1, S2, S3, S4, S10 |
| `data_driven_reconstruction.ipynb` | Tests the reconstruction's dependence on normalization, regularization and population size, and its convergence in PC space | S7, S8, S9, S14 |
| `mexican_hat_and_asymmetry.ipynb` | Derives the origin of the Mexican-hat connectivity profile and simulates asymmetric connectivity models | S5, S6 |
| `spurious_fixed_points.ipynb` | Searches for spurious fixed points and simulates the attractor's breakup at small N | S11, S12 |
| `reservoir_rnn.ipynb` | Trains a reservoir RNN to embed a ring attractor and compares its behaviour with the model's | S16 |
| `closing_the_loop.ipynb` | Builds the closed-loop integration results that Figure 2 uses, reading the raw recordings | none directly |

The [figure page](figures.md) goes the other way, from each figure to the notebook and cell
that draws it.

## Flags

Every notebook starts with two True/False settings, both `False`, so a normal run writes
nothing to disk.

`SAVE_FIGURES = True` writes the figure to `figures/`, replacing the committed image. Turn it
on when you want the figure as a file rather than inline.

`REGENERATE_CACHES = True` recomputes the cached inputs instead of loading them, then
overwrites the files you downloaded. Leave it off unless you specifically want to rebuild a
cache, since some of the caches take hours to rebuild.

## Rebuilding cached results

Every file in `data/` can be rebuilt with the code in this repository. Three of the files
are built directly from the raw recordings; rebuilding any of the others does not involve
the recordings at all.

**Files built from the raw recordings.** `python -m ring.data` reads the recordings and
builds the tuning curves, `mouse_data/tc_data.npz`. `closing_the_loop.ipynb` reads the
recordings and builds `closing_the_loop.npz` and `hd_timeseries.npz`. To rebuild any of
these three, download the recordings, 52 GB, from
[Dandiset 000939](https://dandiarchive.org/dandiset/000939) and point `RING_DATA_DIR` at
the directory holding the `sub-*` folders.

**All other files.** Each of these files is written by one of the notebooks, using
simulations and fits rather than the recordings. To rebuild one of these files, open the
notebook that writes that file
([`data/README.md`](https://github.com/davidclark1/symmetries-continuous-attractors/blob/main/data/README.md)
says which), set `REGENERATE_CACHES = True` at the top, and run the notebook.

**One special case.** `grid_search_results.npz` is written by a separate command rather
than by a notebook: `python -m ring.gp_opt` fits the generative parameters and writes the
file, taking about 13 minutes. The command reads `gamma_phi_info.npz`, which the early
cells of `data_and_generative_model.ipynb` write, and the later cells of that same notebook
read the finished fit back in. Rebuilding this chain from scratch therefore takes three
steps: run the notebook, which writes `gamma_phi_info.npz` and then stops with an error
where the fit is missing; run `python -m ring.gp_opt`; run the notebook again.

## Hardware

The code runs on a CPU and uses a GPU when it finds one. Two figures are sensitive to
the hardware that generates them. `reservoir_rnn_dynamics.png` was made on a GPU and cannot
be reproduced pixel for pixel on a CPU, because torch draws different random numbers on a
CPU than on a GPU, even from the same seed. `grid_cell_eigenvalues.pdf` shows a six-fold
degenerate eigenvalue spectrum, and the order in which the solver returns eigenvalues
within a degenerate cluster depends on the number of BLAS threads. Neither difference
changes what the figure shows. The [figure page](figures.md) explains both cases on the
figures they affect.

## Other pages

- The [figure page](figures.md) maps every figure to the notebook, cell and caches behind it.
- The [API reference](api.md) documents the `ring` package function by function.
- `tools/check_figure_equivalence.py` and `tools/check_cache_equivalence.py` compare a figure
  or cache you regenerated against the released version.
