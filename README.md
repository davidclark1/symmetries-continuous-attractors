# Symmetries and Continuous Attractors in Disordered Neural Circuits

Code and analysis for the paper of that name, by David G. Clark, L. F. Abbott and
Haim Sompolinsky.

**Documentation: [davidclark1.github.io/symmetries-continuous-attractors](https://davidclark1.github.io/symmetries-continuous-attractors/)**

[![DOI](https://zenodo.org/badge/1325787724.svg)](https://doi.org/10.5281/zenodo.21829584)

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

This repository contains everything needed to reproduce the figures.

## Installation

```bash
conda env create -f ring-local.yml
conda activate ring-local
pip install -e .
```

This installs the `ring` package, which the notebooks import. The code uses a GPU when one
is available and falls back to the CPU otherwise. You can check the installation by running
`python tests/test_equivalence.py`.

## Data

Download the cached results, 2.1 GB, and unpack them into `data/`:

    https://doi.org/10.5281/zenodo.21827768

Ten of the eleven notebooks need nothing else. The exceptions are
`notebooks/closing_the_loop.ipynb` and `python -m ring.data`, which read the raw recordings
from [DANDI Dandiset 000939](https://dandiarchive.org/dandiset/000939), 52 GB. Point
`RING_DATA_DIR` at that download if you want to run them. Neither is needed to reproduce a
figure.

## Reproducing a figure

Each figure comes from one notebook in `notebooks/`. Run the notebook top to bottom and the
figure is recomputed and displayed inline. [`FIGURES.md`](FIGURES.md) gives the notebook and
cell for every figure, and [`PROVENANCE.md`](PROVENANCE.md) is the same mapping in one table.

Two flags at the top of each notebook default to off, so a normal run changes nothing on
disk. Setting `SAVE_FIGURES = True` writes the figure to `figures/`, overwriting the
committed image. Setting `REGENERATE_CACHES = True` recomputes the cached inputs, which
takes hours and overwrites the downloaded caches.

Most notebooks finish in under a minute on a GPU. `weight_matrix_analyses.ipynb` takes about
five minutes because it works through a 145 MB cache.

## Repository layout

```
notebooks/    11 analysis notebooks (10 produce figures, closing_the_loop builds a cache)
ring/         the analysis code, installed as an importable package
tests/        checks on the numerical behaviour of the ring package
data/         cached inputs, downloaded from Zenodo (see data/README.md)
figures/      every figure image the paper uses
docs/         the figure manifest and the documentation site source
tools/        utilities for checking results and generating documentation
```

`FIGURES.md` and `PROVENANCE.md` are generated from `docs/figure_manifest.yaml` by
`python tools/gen_docs.py`, so edit the manifest rather than the generated files.
`EQUIVALENCE_LEDGER.md` records whether regenerating each figure and cache reproduces the
original.

## Known differences on regeneration

Most figures regenerate byte-for-byte, or differ only in PDF timestamp metadata. Two do not,
and neither indicates a problem: `grid_cell_eigenvalues.pdf` varies with the BLAS thread
count, and `reservoir_rnn_dynamics.png` differs between CPU and GPU. `EQUIVALENCE_LEDGER.md`
explains both.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{clark2026symmetries,
  title  = {Symmetries and continuous attractors in disordered neural circuits},
  author = {Clark, David G. and Abbott, L. F. and Sompolinsky, Haim},
  year   = {2026}
}
```

The code itself is archived at [10.5281/zenodo.21829584](https://doi.org/10.5281/zenodo.21829584)
and the cached results at [10.5281/zenodo.21827768](https://doi.org/10.5281/zenodo.21827768).

## Licence

This repository is released under the MIT licence, in [`LICENSE`](LICENSE).

Panel a of the grid-cell figure reproduces published images from Hafting et al. (2005),
used with permission from Springer Nature. Those images are excluded from the licence, as
described in `figures/external/README.md`.
