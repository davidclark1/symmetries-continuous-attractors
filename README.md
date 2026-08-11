# Symmetries and Continuous Attractors in Disordered Neural Circuits

Code and analysis for the paper of that name, by David G. Clark, L. F. Abbott and
Haim Sompolinsky.

**Documentation: [davidclark1.github.io/symmetries-continuous-attractors](https://davidclark1.github.io/symmetries-continuous-attractors/)**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21829584.svg)](https://doi.org/10.5281/zenodo.21829584)

## Abstract

A core challenge in neuroscience is reconciling idealized models with experimental data.
Classical continuous-attractor models describe neural representations of continuous
variables such as head direction. Symmetry in the weights generates a manifold of stable
states, with tuning curves identical up to shifts. Rodent head-direction cells, by
contrast, are markedly heterogeneous. Such heterogeneity, we show, is nevertheless
compatible with these models. We build, directly from head-direction cell data, recurrent
networks that behave as quasi-continuous attractors. A generative process then reproduces
the heterogeneous tuning of these cells while making connectivity and dynamics tractable at
large *N*. Although the weights appear disordered, they carry a circular symmetry whose
eigenvalue degeneracies could signify continuous-attractor organization in connectome data.
Dynamical mean-field theory recovers the ingredients of classical ring attractors, namely
Mexican-hat interactions and a spontaneously broken symmetry producing bump states.
Continuous-attractor mechanisms may thus operate in complex mammalian circuits.

## Project structure

Every figure in the paper is drawn by one of eleven notebooks. The notebooks read
precomputed results from `data/`: tuning curves extracted from the recordings, and the
outputs of the slow simulations and fits, about 2 GB in total, released on Zenodo. The
recordings themselves are head-direction data from the Peyrache lab, published separately
as [DANDI Dandiset 000939](https://dandiarchive.org/dandiset/000939); you do not need them
to reproduce the figures. Most notebooks run in under a minute.

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

## Installation

```bash
conda env create -f ring-local.yml
conda activate ring-local
pip install -e .
```

These commands install the `ring` package that the notebooks import. The package uses a
GPU when one is available and the CPU otherwise. Check the installation with `python tests/test_equivalence.py`.

## Data

Download the cached results, 2.1 GB, and unpack them into `data/`:

    https://doi.org/10.5281/zenodo.21827767

The `.npz` files should sit directly in `data/`, with `tc_data.npz` in `data/mouse_data/`.
Verify the download with `md5sum -c CHECKSUMS.md5` from inside `data/`. The notebooks will
not run without these files. [`data/README.md`](data/README.md) says where each one comes
from.

You only need the raw recordings from DANDI if you want to rebuild the caches from scratch
rather than download them. Only two pieces of code read the raw recordings: `python -m ring.data`, which builds
the tuning curves in `mouse_data/tc_data.npz`, and `notebooks/closing_the_loop.ipynb`, which
builds `closing_the_loop.npz` and `hd_timeseries.npz`. All three output files are in the
Zenodo download. Point `RING_DATA_DIR` at your own copy of the recordings to rerun either.

## Reproducing a figure

Each figure comes from one notebook. Run it from top to bottom and the figure is recomputed
and displayed inline. [`FIGURES.md`](FIGURES.md) lists every figure with its notebook, the
cell that draws it, and the caches it reads.

Two variables at the top of each notebook control whether anything is written to disk.
Both variables are `False` by default, so a normal run changes no files.

- `SAVE_FIGURES = True` writes the figure to `figures/`, replacing the committed image.
- `REGENERATE_CACHES = True` recomputes the cached inputs instead of loading them. This takes
  hours and overwrites the files you downloaded.

Most notebooks are quick. `weight_matrix_analyses.ipynb` takes about five minutes because it
works through a 145 MB cache, and `reservoir_rnn.ipynb` trains five reservoirs from scratch.

Nearly every figure regenerates identically. A few cannot, for reasons that are properties of
the computation rather than faults, such as a spectrum whose degenerate eigenvalues come back
in a different order. [`FIGURES.md`](FIGURES.md) names each figure this affects and
explains the difference. If you want to check a figure or a cache you regenerated against
ours, `tools/check_figure_equivalence.py` and `tools/check_cache_equivalence.py` do the
comparison.

## Repository layout

```
notebooks/    the eleven analysis notebooks
ring/         the analysis code, installed as an importable package
tests/        checks on the numerical behaviour of the ring package
data/         cached results, downloaded from Zenodo (see data/README.md)
figures/      every figure image the paper uses
docs/         the figure manifest and the documentation site source
tools/        comparison utilities and the documentation generator
```

`FIGURES.md` is generated from `docs/figure_manifest.yaml` by `python tools/gen_docs.py`, so
edit the manifest rather than the generated page.

## Citation

If you use this code, please cite the
[preprint](https://www.biorxiv.org/content/10.1101/2025.01.26.634933):

```bibtex
@article{clark2025symmetries,
  title   = {Symmetries and continuous attractors in disordered neural circuits},
  author  = {Clark, David G. and Abbott, L. F. and Sompolinsky, Haim},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.01.26.634933}
}
```

The code itself is archived at [10.5281/zenodo.21829584](https://doi.org/10.5281/zenodo.21829584)
and the cached results at [10.5281/zenodo.21827767](https://doi.org/10.5281/zenodo.21827767).

## Licence

This repository is released under the MIT licence, in [`LICENSE`](LICENSE).

Panel a of the grid-cell figure reproduces published images from Hafting et al. (2005), used
with permission from Springer Nature. The Hafting et al. images are excluded from the licence, as
described in [`figures/external/README.md`](figures/external/README.md).
