# Symmetries and Continuous Attractors in Disordered Neural Circuits

Documentation for the code accompanying the paper of that name, by David G. Clark,
L. F. Abbott and Haim Sompolinsky.

- **[Figure provenance](provenance.md)** maps every figure to the notebook that produces it.
- **[Figure recipes](figures.md)** gives per-figure reproduction instructions.
- **[API reference](api.md)** documents the `ring` package, generated from its docstrings.

## Quick start

```bash
conda env create -f ring-local.yml
conda activate ring-local
pip install -e .
```

Then download the cached results, 2.1 GB, from
[Zenodo](https://doi.org/10.5281/zenodo.21827768) and unpack them into `data/`. The
notebooks will not run without them.

With that in place, run any notebook in `notebooks/` top to bottom to regenerate its figure.
The [figure recipes](figures.md) say which notebook produces which figure.

The head-direction recordings analysed in the paper are from the Peyrache lab and are
available separately from
[DANDI Dandiset 000939](https://dandiarchive.org/dandiset/000939). They are not needed to
reproduce a figure.

The repository `README.md` covers installation and reproduction in more detail.
