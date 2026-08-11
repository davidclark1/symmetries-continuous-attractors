# Figures

Generated from [`figure_manifest.yaml`](figure_manifest.yaml) by `tools/gen_docs.py`. Edit the manifest, not this page.

Every figure in the paper, the notebook that draws it, and what that notebook reads. To reproduce one, install the package, unpack the cached results into `data/`, and run the notebook from top to bottom; the figure appears inline. Set `SAVE_FIGURES = True` at the top of the notebook to write it to `figures/` as well.

| Figure | What it shows | Notebook |
|--------|---------------|----------|
| [1](#figure-1) | Classical ring-attractor model and mouse head-direction tuning curves | `data_and_generative_model.ipynb` |
| [2](#figure-2) | Ring attractor built from the data: PCA convergence, connectivity, closed-loop integration | `data_driven_attractor.ipynb` |
| [3](#figure-3) | Circular symmetry in the distribution of head-direction responses | `data_and_generative_model.ipynb` |
| [4](#figure-4) | Tuning curves and connectivity from the generative model | `data_and_generative_model.ipynb` |
| [5](#figure-5) | Model vs. data: profiles, peak statistics, symmetry and information | `data_and_generative_model.ipynb` |
| [6](#figure-6) | Circular geometry, disordered embedding, and the spectrum of the optimal weights | `spectra.ipynb` |
| [7](#figure-7) | Dynamical mean-field theory of the disordered ring attractor | `dmft.ipynb` |
| [8](#figure-8) | Grid-cell simulations from the disordered attractor | `grid_sims.ipynb` |
| [S1](#figure-s1) | Low-rank connectivity: kappa trajectories | `weight_matrix_analyses.ipynb` |
| [S2](#figure-s2) | Mixture loadings: the attractor is the whole torus | `weight_matrix_analyses.ipynb` |
| [S3](#figure-s3) | Left and right singular vectors of the connectivity | `weight_matrix_analyses.ipynb` |
| [S4](#figure-s4) | Real vs. surrogate connectivity matrices and their spectra | `weight_matrix_analyses.ipynb` |
| [S5](#figure-s5) | Origin of the Mexican-hat connectivity profile | `mexican_hat_and_asymmetry.ipynb` |
| [S6](#figure-s6) | Asymmetric connectivity models | `mexican_hat_and_asymmetry.ipynb` |
| [S7](#figure-s7) | Double normalization of tuning curves | `data_driven_reconstruction.ipynb` |
| [S8](#figure-s8) | PCA convergence of the data-driven ring attractor | `data_driven_reconstruction.ipynb` |
| [S9](#figure-s9) | Reconstruction error vs. regularization strength | `data_driven_reconstruction.ipynb` |
| [S10](#figure-s10) | Noise-driven dynamics: circulant vs. disordered | `weight_matrix_analyses.ipynb` |
| [S11](#figure-s11) | Searching for spurious fixed points in finite-N attractors | `spurious_fixed_points.ipynb` |
| [S12](#figure-s12) | Attractor dynamics at finite N | `spurious_fixed_points.ipynb` |
| [S13](#figure-s13) | Normalized covariance matrices and their eigenvectors | `data_and_generative_model.ipynb` |
| [S14](#figure-s14) | Eigenvalue and singular-value spectra of the data and of subsamples | `data_driven_reconstruction.ipynb` |
| [S15](#figure-s15) | Eigenvalue spectrum of the grid-cell connectivity | `grid_sims.ipynb` |
| [S16](#figure-s16) | Attractor dynamics in a reservoir RNN | `reservoir_rnn.ipynb` |

## Main text

### Figure 1

Classical ring-attractor model and mouse head-direction tuning curves.

- **Image:** `figures/classical_attractor_vs_data.pdf`
- **Notebook:** `notebooks/data_and_generative_model.ipynb`
- **Cell:** the one containing `plt.savefig("../figures/classical_attractor_vs_data.pdf", dpi=400)`
- **Reads:** `data/mouse_data/tc_data.npz`
- **Panels:**
    - a: ring-attractor schematic
    - b: circulant weight matrix
    - c: translation-invariant tuning curves
    - d: flow of the paper
    - e: tuning curves measured in each mouse
    - f, g: mean and fluctuation profiles

One cell draws all seven panels. Panel b shows the classical connectivity kernel J(theta - theta') = J0 + J1 cos(theta - theta') with J0 = 0 and J1 = 1. Its colour limits are symmetric about zero so that white marks the sign change, and it carries no colorbar because the kernel is a schematic with no units.

### Figure 2

Ring attractor built from the data: PCA convergence, connectivity, closed-loop integration.

- **Image:** `figures/data_driven_attractor.pdf`
- **Notebook:** `notebooks/data_driven_attractor.ipynb`
- **Cell:** the one containing `plt.savefig("../figures/data_driven_attractor.pdf", dpi=400)`
- **Reads:** `data/mouse_data/tc_data.npz`, `data/closing_the_loop.npz`, `data/convergence_pcs.npz`, `data/convergence_distances.npz`, `data/surrogate_bump_sims.npz`, `data/surrogate_spectra.npz`, `data/velocity_integration_sims.npz`

One cell draws the whole figure. The closed-loop integration panels come from closing_the_loop.npz, which closing_the_loop.ipynb builds from the raw recordings; the other panels come from caches written by this notebook. Both random steps, the scrambled-noise shuffle and the initial conditions of the convergence analysis, are seeded.

### Figure 3

Circular symmetry in the distribution of head-direction responses.

- **Image:** `figures/circular_symmetry_in_data.pdf`
- **Notebook:** `notebooks/data_and_generative_model.ipynb`
- **Cell:** the one containing `plt.savefig("../figures/circular_symmetry_in_data.pdf", dpi=400)`
- **Reads:** `data/mouse_data/tc_data.npz`, `data/grid_search_results.npz`

Panel b's matched-uniform null is the only random draw in the figure, and it is seeded. Panels a and c are deterministic.

*On rerunning:* Rebuilding grid_search_results.npz with python -m ring.gp_opt can land on a different, equally good optimum, because the loss surface is a Monte Carlo estimate that is nearly flat near its minimum. The released cache holds the fit the paper reports, so panels drawn from a rebuilt fit shift slightly while showing the same thing.

### Figure 4

Tuning curves and connectivity from the generative model.

- **Image:** `figures/generative_model_tuning.pdf`
- **Notebook:** `notebooks/data_and_generative_model.ipynb`
- **Cell:** the one containing `plt.savefig("../figures/generative_model_tuning.pdf")  # SAVE_FIGURES-gated`
- **Reads:** `data/mouse_data/tc_data.npz`, `data/grid_search_results.npz`
- **Panels:**
    - a-f: covariance, Gaussian-process samples, softplus nonlinearity, error vs. sigma, error landscape, and the data/model correlation
    - g: 24 independent samples of tuning curves from the fitted process

One cell assembles all seven panels. Panel e's axes are labelled '$b$ [soft threshold]' and '$\beta$ [sharpness]', matching the notation in the paper.

*On rerunning:* Rebuilding grid_search_results.npz with python -m ring.gp_opt can land on a different, equally good optimum, because the loss surface is a Monte Carlo estimate that is nearly flat near its minimum. The released cache holds the fit the paper reports, so panels drawn from a rebuilt fit shift slightly while showing the same thing.

### Figure 5

Model vs. data: profiles, peak statistics, symmetry and information.

- **Image:** `figures/model_vs_data.pdf`
- **Notebook:** `notebooks/data_and_generative_model.ipynb`
- **Cell:** the one containing `plt.savefig("../figures/model_vs_data.pdf")  # SAVE_FIGURES-gated`
- **Reads:** `data/analysis_results.npz`, `data/mouse_data/tc_data.npz`, `data/grid_search_results.npz`
- **Panels:**
    - a, b: mean and fluctuation profiles, model vs. data
    - c: peak counts at three z-thresholds
    - d: peak-height histograms for the first three peaks
    - e: flip symmetry, full distribution and zoomed tail
    - f: head-direction information

Peak detection treats the tuning curve as circular and checks the final angular bin, so every curve contributes at least one peak, as a circular signal must.

*On rerunning:* Rebuilding analysis_results.npz redraws the four generative-model peak-count arrays, since they are a fresh sample from the model rather than a function of the data, so panel c shifts slightly. The four arrays measured from the recordings are exact. Rebuilding grid_search_results.npz can likewise land on a different, equally good optimum (its loss surface is a nearly flat Monte Carlo estimate), shifting the model-side panels. Reading the released caches reproduces the committed figure.

### Figure 6

Circular geometry, disordered embedding, and the spectrum of the optimal weights.

- **Image:** `figures/disordered_vs_circulant_spectra.pdf`
- **Notebook:** `notebooks/spectra.ipynb`
- **Cell:** the one containing `plt.savefig("../figures/disordered_vs_circulant_spectra.pdf", dpi=400)`
- **Reads:** `data/spectra_cache.npz`
- **Panels:**
    - a: ring-embedding schematic
    - b-h: weight matrices, spectrum, eigenvalue planes, and eigenvector scatters

The tuning curves are generated in the notebook rather than measured; spectra_cache.npz only stores the results of the expensive computations, and the notebook recomputes it from scratch if it is absent. The spectra use ridge regularization lambda = 1e-8, applied as Gamma_x / (Gamma_phi + lambda); the leading eigenvalues and the pattern of doublets and singlets are stable under that choice.

*On rerunning:* Singular vectors are defined only up to sign and the seed does not fix which sign the solver returns, so the rightmost subplot of panel g can come out sign-flipped on different hardware or a different BLAS build. Deleting data/spectra_cache.npz forces a recompute of the mean-field weight profile; the released cache predates the seed on that computation, so a recompute shifts the circulant-side panels slightly while recomputes agree with each other exactly.

### Figure 7

Dynamical mean-field theory of the disordered ring attractor.

- **Image:** `figures/dmft_theory.pdf`
- **Notebook:** `notebooks/dmft.ipynb`
- **Cell:** the one containing `plt.savefig("../figures/dmft_theory.pdf", dpi=400)`
- **Reads:** `data/jacobian_results.npz`, `data/dmft_simulation_results.npz`

No random numbers enter anywhere in this pipeline. Panels b, c and f use (sigma, beta) = (1.4211, 2.7659), the values quoted in the Methods. DMFTModel carries its own regularization constant, 1e-12 by default, which guards the continuum inversion against becoming singular as Ghat^phi_n approaches zero. That constant is a numerical guard, unrelated to the ridge used when fitting finite-N networks elsewhere in the code.

### Figure 8

Grid-cell simulations from the disordered attractor.

- **Image:** `figures/grid_cell_simulations.pdf`
- **Notebook:** `notebooks/grid_sims.ipynb`
- **Cell:** the one containing `plt.savefig("../figures/grid_cell_simulations.pdf", bbox_inches="tight", dpi=450)  # SAVE_FIGURES-gated`
- **Reads:** `data/grid_analysis_results.npz`, `data/eta_analysis.npz`, `data/gridness_hist_digitized.npz`, `data/experimental_gridness_scores.npy`
- **Panels:**
    - a: rat grid-cell rate maps published by Hafting et al. (2005), loaded from figures/external/hafting2005_grid_ratemaps.png and reproduced with permission from Springer Nature
    - b: grid scores digitized from Nayebi et al. (2021), replotted as our own histogram
    - c-f: simulation output, drawn from the caches

Panels a and b are experimental results from other groups rather than model output; `figures/external/README.md` gives their terms of use. Panels c to f read `grid_analysis_results.npz`, which despite the similar name is a different file from the `analysis_results.npz` that Figure 5 uses. The example cells shown in panels d and e are whichever simulated cells fall nearest a target grid score, so a different seed changes which cells appear without changing what the panel shows. Panel d's input-current maps are displayed sign-flipped to positive skewness where needed, because the generative process is sign-symmetric and only the positively skewed realization looks like a grid cell once rectified; the flip leaves panel c's grid scores and panel e's autocorrelations exactly as they were. Panel f is not a raw PCA projection. The leading components are near-degenerate in eigenvalue and defined only up to sign, so which three of them a standard projection would pick is arbitrary; the panel instead uses a fixed display frame pinned to the two known toroidal phases, with axes labelled torus 1, 2 and 3.

## Supplementary figures

### Figure S1

Low-rank connectivity: kappa trajectories.

- **Image:** `figures/low_rank_connectivity.pdf`
- **Notebook:** `notebooks/weight_matrix_analyses.ipynb`
- **Cell:** the one containing `plt.savefig("../figures/low_rank_connectivity.pdf"`
- **Reads:** nothing; everything it needs is computed in the notebook

An illustrative synthetic figure. The low-rank connectivity is built from a seeded random draw in the cell above the plotting cell.

### Figure S2

Mixture loadings: the attractor is the whole torus.

- **Image:** `figures/low_rank_mixture_torus.pdf`
- **Notebook:** `notebooks/weight_matrix_analyses.ipynb`
- **Cell:** the one containing `plt.savefig("../figures/low_rank_mixture_torus.pdf"`
- **Reads:** nothing; everything it needs is computed in the notebook

Built on the same rank-four geometry and the same seed as Figure S1, with one change: the neuron loadings are drawn from a two-component Gaussian mixture, so each neuron loads onto one doublet only. Every ring is then stable at once and the attractor is the whole torus, rather than the union of rings in Figure S1. Layout, kappa normalization, colormap and axis limits are copied from Figure S1 so that the two read as a pair.

### Figure S3

Left and right singular vectors of the connectivity.

- **Image:** `figures/left_right_singular_vectors.png`
- **Notebook:** `notebooks/weight_matrix_analyses.ipynb`
- **Cell:** the one containing `plt.savefig("../figures/left_right_singular_vectors.png"`
- **Reads:** nothing; everything it needs is computed in the notebook

The singular vectors come from the decomposition computed in the cell above, whose tuning curves are drawn from a seeded generator.

### Figure S4

Real vs. surrogate connectivity matrices and their spectra.

- **Image:** `figures/real_vs_surrogate_matrices.png`
- **Notebook:** `notebooks/weight_matrix_analyses.ipynb`
- **Cell:** the one containing `plt.savefig("../figures/real_vs_surrogate_matrices.png"`
- **Reads:** nothing; everything it needs is computed in the notebook

The surrogate-matrix panels are identical on every run, because their random orthogonal matrices are drawn from a separately seeded generator; only the real-J heatmap and its singular-value curve depend on the seeded tuning-curve draw.

### Figure S5

Origin of the Mexican-hat connectivity profile.

- **Image:** `figures/mexican_hat_origin.pdf`
- **Notebook:** `notebooks/mexican_hat_and_asymmetry.ipynb`
- **Cell:** the one containing `plt.savefig("../figures/mexican_hat_origin.pdf"`
- **Reads:** nothing; everything it needs is computed in the notebook

The plotting cell deconvolves normalized probability densities, stabilized by a hardcoded 1e-6 constant. That constant is specific to this deconvolution and is not the ridge used elsewhere in the code. The tuning curves it works from are generated earlier in the notebook from a seeded generator.

### Figure S6

Asymmetric connectivity models.

- **Image:** `figures/asymmetric_models.png`
- **Notebook:** `notebooks/mexican_hat_and_asymmetry.ipynb`
- **Cell:** the one containing `plt.savefig("../figures/asymmetric_models.png"`
- **Reads:** nothing; everything it needs is computed in the notebook

This cell uses a ridge of 1e-6 rather than the 1e-8 used elsewhere, and the difference matters. The connectivity here comes from a rank-2 covariance, points on a two-dimensional ring, so the ridge acts on only two directions and enters twice through the inverse. At 1e-8 several near-null modes are admitted, and because the heatmaps autoscale, their large magnitudes wash out the structure the figure is about.

### Figure S7

Double normalization of tuning curves.

- **Image:** `figures/double_normalization.pdf`
- **Notebook:** `notebooks/data_driven_reconstruction.ipynb`
- **Cell:** the one containing `plt.savefig("../figures/double_normalization.pdf"`
- **Reads:** `data/mouse_data/tc_data.npz`

The orange curve carries two transformations, not one: the sigma = 2 Gaussian smoothing of the input currents as well as the across-neuron normalization used to build the network. The neuron subsample it plots is seeded.

### Figure S8

PCA convergence of the data-driven ring attractor.

- **Image:** `figures/pca_convergence.png`
- **Notebook:** `notebooks/data_driven_reconstruction.ipynb`
- **Cell:** the one containing `plt.savefig("../figures/pca_convergence.png"`
- **Reads:** `data/mouse_data/tc_data.npz`, `data/pc_convergence_grid.npz`

A sweep over the ridge strength lambda, with the operating value lambda = 1e-8.

*On rerunning:* Two things can move. The pcs_traj3_grid array in the cache is a stochastic realization, so rebuilding the cache redraws those trajectories, though the released cache reproduces the committed figure. And a different matplotlib release can reassign the categorical colour cycle, which changes the hue and drawing order of the trajectories without moving any of them.

### Figure S9

Reconstruction error vs. regularization strength.

- **Image:** `figures/error_vs_regularization.pdf`
- **Notebook:** `notebooks/data_driven_reconstruction.ipynb`
- **Cell:** the one containing `plt.savefig("../figures/error_vs_regularization.pdf")`
- **Reads:** `data/mouse_data/tc_data.npz`, `data/pc_convergence_grid.npz`

A deterministic log-log plot of the mean squared flow error against lambda, swept from 1e-12 to 1e-2 in decades. The operating value lambda = 1e-8 sits one grid step above the curve's true minimum near 3e-11, so it should not be read as minimizing the curve: the apparent upturn below 3e-11 is an artifact of finite-precision conditioning rather than a property of the flow. The figure is saved at full canvas width without a tight bounding box, so the surrounding whitespace is intentional.

### Figure S10

Noise-driven dynamics: circulant vs. disordered.

- **Image:** `figures/noise_driven_dynamics.png`
- **Notebook:** `notebooks/weight_matrix_analyses.ipynb`
- **Cell:** the one containing `plt.savefig("../figures/noise_driven_dynamics.png", bbox_inches="tight", dpi=400)  # SAVE_FIGURES-gated`
- **Reads:** `data/ring_attractor_noise_results.npz`
- **Panels:**
    - a: the two connectivity matrices
    - b: angular-drift trajectories, disordered network on the left and circulant on the right
    - c: angular stability across noise levels

One cell draws all three panels from the cache. The cache itself is written by a cell that only runs when REGENERATE_CACHES is True; it is seeded for both numpy and torch and is deterministic on a GPU.

### Figure S11

Searching for spurious fixed points in finite-N attractors.

- **Image:** `figures/spurious_fixed_points.pdf`
- **Notebook:** `notebooks/spurious_fixed_points.ipynb`
- **Cell:** the one containing `plt.savefig("../figures/spurious_fixed_points.pdf"`
- **Reads:** `data/mouse_data/tc_data.npz`, `data/spurious_fixed_points.npz`

Two rows of panels at five matched noise levels, row a for the generative network and row b for the network derived from data. The data-derived network converges from every initial condition tested, including a per-neuron standard deviation of 1000, where states begin about 22 ring radii away and still end within 2e-4 of a ring radius of the manifold. Under unstructured initial conditions it also shows a preference for particular bump angles, which weakens as the initialization noise grows. Rebuilding the cache takes about two minutes.

### Figure S12

Attractor dynamics at finite N.

- **Image:** `figures/finite_size_dynamics.pdf`
- **Notebook:** `notebooks/spurious_fixed_points.ipynb`
- **Cell:** the one containing `plt.savefig("../figures/finite_size_dynamics.pdf"`
- **Reads:** `data/mouse_data/tc_data.npz`

Seeded for both numpy and torch before any random draw, so it is deterministic on CPU and GPU alike.

### Figure S13

Normalized covariance matrices and their eigenvectors.

- **Image:** `figures/normalized_covariance_eigenvectors.png`
- **Notebook:** `notebooks/data_and_generative_model.ipynb`
- **Cell:** the one containing `plt.savefig("../figures/normalized_covariance_eigenvectors.png"`
- **Reads:** `data/mouse_data/tc_data.npz`

Which neurons appear is a seeded random choice, so the panels are the same on every run.

### Figure S14

Eigenvalue and singular-value spectra of the data and of subsamples.

- **Image:** `figures/data_and_subsample_spectra.png`
- **Notebook:** `notebooks/data_driven_reconstruction.ipynb`
- **Cell:** the one containing `plt.savefig("../figures/data_and_subsample_spectra.png", bbox_inches="tight", dpi=400)  # SAVE_FIGURES-gated`
- **Reads:** `data/mouse_data/tc_data.npz`
- **Panels:**
    - a: spectrum of the mouse-derived weights at N = 1533
    - b: how the spectrum converges with sample size, at N = 100, 500, 1000 and 1533

Panel a reuses the weights and tuning curves computed earlier in the same notebook and involves no random draw. Panel b builds idealized tuning curves and weights at each N with a ridge of 1e-8; it is a seeded random realization rather than an exact function of the data, so the same seed reproduces it but a different one would not.

### Figure S15

Eigenvalue spectrum of the grid-cell connectivity.

- **Image:** `figures/grid_cell_eigenvalues.pdf`
- **Notebook:** `notebooks/grid_sims.ipynb`
- **Cell:** the one containing `plt.savefig("../figures/grid_cell_eigenvalues.pdf"`
- **Reads:** nothing; everything it needs is computed in the notebook

The leading eigenvalues of the connectivity, computed in the notebook from sampled data.

*On rerunning:* The spectrum is exactly six-fold degenerate, and eigenvalue solvers do not promise any particular ordering inside a degenerate cluster. Changing the number of BLAS threads, or running on a GPU instead of a CPU, can therefore swap individual points within a cluster. The six-fold degeneracy the figure is about is unaffected.

### Figure S16

Attractor dynamics in a reservoir RNN.

- **Image:** `figures/reservoir_rnn_dynamics.png`
- **Notebook:** `notebooks/reservoir_rnn.ipynb`
- **Cell:** the one containing `plt.savefig("../figures/reservoir_rnn_dynamics.png", bbox_inches='tight', dpi=300)`
- **Reads:** nothing; everything it needs is computed in the notebook

One cell trains five reservoirs from scratch, which makes this the slowest figure here to regenerate. The reservoir weights, the input projection and the initial state are all drawn fresh from a seeded generator on each run.

*On rerunning:* Torch's CPU and GPU random-number streams differ even from the same seed, so a CPU run draws different reservoirs than a GPU run and about 10.6 percent of pixels change. The gain sweep, the convergent and non-convergent regimes, and the boundary between them all stay as they are. The committed image was made on a GPU, so match it on a GPU.

