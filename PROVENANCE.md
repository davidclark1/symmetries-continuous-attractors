# Figure provenance

Auto-generated from [`docs/figure_manifest.yaml`](docs/figure_manifest.yaml) by `tools/gen_docs.py` — **do not edit by hand.**

Every paper figure mapped to its producer notebook and its verification verdict from the equivalence-checking pass. See [figure recipes](FIGURES.md) for per-figure reproduction recipes.

## Summary

- **24** paper figures (17 direct, 7 composite).
- Verdicts: BIT-EXACT (13), NON-DETERMINISTIC (1), PDF-METADATA-ONLY (10).

## Terms used here

**Figure kind**
- *direct* — a single notebook cell saves the whole figure; reproducible from the notebook alone.
- *composite* — several panels are drawn by one notebook cell and assembled into the final image in code.
- *response* — an additional figure produced by the notebooks; not a numbered figure in the paper.

**Verification verdict** — how faithfully re-running the code reproduces the committed figure:
- *BIT-EXACT* — regenerates a byte-identical file.
- *PDF-METADATA-ONLY* — identical image; only PDF timestamps and ids differ.
- *NON-DETERMINISTIC* — small run-to-run variation remains even with a fixed seed; the per-figure note says why.

**Other terms**
- *Producer* — the notebook, and specific cell, that generates a figure or panel.
- *Input caches* — precomputed results in `data/` that a notebook loads instead of recomputing.

## Figures

| Fig | Manuscript file | Producer | Verdict |
|----:|-----------------|----------|---------|
| 1 | `figures/classical_attractor_vs_data.pdf` | `data_and_generative_model.ipynb` | BIT-EXACT |
| 2 | `figures/data_driven_attractor.pdf` | `data_driven_attractor.ipynb` | BIT-EXACT |
| 3 | `figures/circular_symmetry_in_data.pdf` | `data_and_generative_model.ipynb` | BIT-EXACT |
| 4 | `figures/generative_model_tuning.pdf` | `data_and_generative_model.ipynb` | PDF-METADATA-ONLY |
| 5 | `figures/model_vs_data.pdf` | `data_and_generative_model.ipynb` | PDF-METADATA-ONLY |
| 6 | `figures/disordered_vs_circulant_spectra.pdf` | `spectra.ipynb` | BIT-EXACT |
| 7 | `figures/dmft_theory.pdf` | `dmft.ipynb` | PDF-METADATA-ONLY |
| 8 | `figures/grid_cell_simulations.pdf` | `grid_sims.ipynb` | BIT-EXACT |
| S1 | `figures/low_rank_connectivity.pdf` | `weight_matrix_analyses.ipynb` | PDF-METADATA-ONLY |
| S2 | `figures/low_rank_mixture_torus.pdf` | `weight_matrix_analyses.ipynb` | PDF-METADATA-ONLY |
| S3 | `figures/left_right_singular_vectors.png` | `weight_matrix_analyses.ipynb` | BIT-EXACT |
| S4 | `figures/real_vs_surrogate_matrices.png` | `weight_matrix_analyses.ipynb` | BIT-EXACT |
| S5 | `figures/mexican_hat_origin.pdf` | `mexican_hat_and_asymmetry.ipynb` | PDF-METADATA-ONLY |
| S6 | `figures/asymmetric_models.png` | `mexican_hat_and_asymmetry.ipynb` | BIT-EXACT |
| S7 | `figures/double_normalization.pdf` | `data_driven_reconstruction.ipynb` | PDF-METADATA-ONLY |
| S8 | `figures/pca_convergence.png` | `data_driven_reconstruction.ipynb` | BIT-EXACT |
| S9 | `figures/error_vs_regularization.pdf` | `data_driven_reconstruction.ipynb` | PDF-METADATA-ONLY |
| S10 | `figures/noise_driven_dynamics.png` | `weight_matrix_analyses.ipynb` | BIT-EXACT |
| S11 | `figures/spurious_fixed_points.pdf` | `spurious_fixed_points.ipynb` | PDF-METADATA-ONLY |
| S12 | `figures/finite_size_dynamics.pdf` | `spurious_fixed_points.ipynb` | PDF-METADATA-ONLY |
| S13 | `figures/normalized_covariance_eigenvectors.png` | `data_and_generative_model.ipynb` | BIT-EXACT |
| S14 | `figures/data_and_subsample_spectra.png` | `data_driven_reconstruction.ipynb` | BIT-EXACT |
| S15 | `figures/grid_cell_eigenvalues.pdf` | `grid_sims.ipynb` | NON-DETERMINISTIC |
| S16 | `figures/reservoir_rnn_dynamics.png` | `reservoir_rnn.ipynb` | BIT-EXACT |

## Additional figures

Figures produced by the analysis notebooks that are not numbered figures in the paper.

| Ref | Manuscript file | Producer | Verdict |
|-----|-----------------|----------|---------|
| R1 | `figures/weight_spectrum_vs_theory.png` | `weight_matrix_analyses.ipynb` | BIT-EXACT |

