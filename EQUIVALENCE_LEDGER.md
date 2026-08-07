# Equivalence ledger

Before release, every figure and cache in this repository was regenerated from the released
code and compared against the committed version. This file records the method and the
results. Per-figure verdicts are tabulated in [`PROVENANCE.md`](PROVENANCE.md), with
reproduction instructions in [`FIGURES.md`](FIGURES.md).

## Verdict vocabulary

- **BIT-EXACT** — regenerating produces a byte-identical file.
- **PIXEL-EXACT** — regenerating produces a pixel-identical image; only invisible file
  metadata differs.
- **PDF-METADATA-ONLY** — identical rendered content; only PDF timestamps and ids differ.
  Compare PDFs after stripping `CreationDate`, `ModDate`, and the document ID.
- **TIER-3-STATISTICAL** — a different random draw with the same statistics. Applies to a
  few cache arrays whose released values were produced before seeds were added; the released
  caches remain the reference, and the seeds now in the notebooks make any future
  regeneration deterministic.
- **ARTWORK-ONLY** — a change confined to text or layout, verified by digesting every
  plotted artist (line data, image arrays, axes limits) and comparing with `np.array_equal`.
- **NON-DETERMINISTIC** — residual run-to-run variation beyond seeding control; the
  per-figure note says why.

## How figures were compared

Bitmaps were compared byte-for-byte or pixel-for-pixel. Vector PDFs were compared after
stripping timestamp metadata. Where a legitimate difference was expected (see below), the
plotted artists were extracted from both runs and compared numerically with
`np.array_equal`, so "equivalent" always means the drawn content is identical, not that the
files merely look similar.

## Seed policy

Every notebook that draws random numbers seeds its RNG explicitly (`np.random.seed`,
`torch.manual_seed`, `torch.cuda.manual_seed_all`, or `ring.seed.set_global_seed`), so runs
are deterministic given the released code. The `Fixes` identifiers in
[`PROVENANCE.md`](PROVENANCE.md) (F2 through F14) mark the figures whose seeds were added
during this verification pass; the seeds are in the released notebooks, so no action is
needed to benefit from them. A handful of cache arrays predate their seeds and are
TIER-3-STATISTICAL as described above; each is named in its figure's note in
[`FIGURES.md`](FIGURES.md).

## Known differences on regeneration

Two figures do not regenerate identically, and neither indicates a problem.

- **`grid_cell_eigenvalues.pdf`** (Supplementary Fig. S15). The spectrum is exactly 6-fold
  degenerate, and the ordering the eigenvalue solver returns within a degenerate cluster
  depends on the BLAS thread count and on CPU vs GPU execution. A regenerated plot shows the
  same 6-fold degeneracy with individual points swapped within clusters.
- **`reservoir_rnn_dynamics.png`** (Supplementary Fig. S16). torch's CPU and CUDA
  random-number streams differ even from the same seed, so a CPU run draws a different
  reservoir than the committed GPU run (about 10.6% of pixels differ, same qualitative
  content). Regenerate on a GPU to match the committed file.

Additionally, singular vectors and eigenvectors are defined only up to sign, so panels that
plot them can show sign flips across BLAS builds (noted per figure in
[`FIGURES.md`](FIGURES.md)), and the categorical color cycle can shift across
matplotlib/Python patch versions in one figure (Supplementary Fig. S8).

## Caches

Each cache in `data/` was regenerated from the raw recordings and the released code and
compared against the released file key by key with `np.array_equal(..., equal_nan=True)`.
`data/README.md` lists each cache's producer. Most keys are bit-exact; the exceptions are
the TIER-3-STATISTICAL arrays noted per figure in [`FIGURES.md`](FIGURES.md), for which the
released caches are the reference.
