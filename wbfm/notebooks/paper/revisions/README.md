# Paper figures (revised version)

Code for the figures of the revised manuscript
"An intrinsic neuronal manifold underlies brain-wide hierarchical organization of behavior in *C. elegans*".

Each analysis is a Jupyter notebook; a Python export (`.py`) of every notebook is committed next to it
(the `.ipynb` files themselves are not tracked). The exports are not meant to run top to bottom as scripts: they
mirror the notebook cells, and loading the data requires the raw recordings (see the main README and
[docs/paper_datasets.md](../../../../docs/paper_datasets.md)).

Outputs are written to a folder per figure (e.g. `fig5/`), and each final figure is assembled from these panels
(main figures as `figN/FIGN.svg`). A few panels were not changed during the revision and are still produced by the
scripts of the original submission in the parent folder (`wbfm/notebooks/paper/`); these are marked "(original
submission)" below.

## Main figures

| Figure | Content | Script | Output folder |
|---|---|---|---|
| Fig. 1 | Example recording: heatmap, PCA modes, behavior, trajectory | `fig_1_big_kymograph.py` | `fig1/` |
| Fig. 2 | PCA and CCA manifolds, behavior variance explained, eigenworms | `fig_2_CCA_with_eigenworms.py` | `fig2/` |
| Fig. 3 | PC1 weights in freely moving vs. immobilized animals, intrinsic categories, dimensionality | `fig_3_wbfm_immob_comparison.py` | `fig3/` |
| Fig. 4 | Example traces and undulation-triggered averages of residual activity (VB02, DB01, DD01, AVB) | `../fig_3_examples_residual_decomposition.py` (original submission, updated); `fig_4_triggered_average_examples.py` (checks of the undulation triggers) | `../multiplexing/wbfm_trigger/` |
| Fig. 5 | Bayesian hierarchical model: model comparison, gating, phase encoding, eigenworms 3-4 | `fig_bayesian_ALT_CONDA.py` | `fig5/` |
| Fig. 6 | O2/CO2 reafferent neurons, manifold participation, fraction of active neurons | `fig_6_pie_chart_and_triple_plot.py`, `fig_supp_O2_table.py` | `fig6/`, `fig_supp_o2/` |

## Supplementary figures

| Figure | Content | Script | Output folder |
|---|---|---|---|
| Hardware and pipeline (c-d) | Tracking accuracy of the pipeline steps | `fig_supp_tracking_accuracy.py` | `pipeline_accuracy/` |
| Neuron identification | Identified neurons per dataset and per neuron class | `../supplement/fig_S2_ided_neurons_table.py` (original submission) | `../supplement/ids/` |
| Behavior quantifications | (a) GCaMP vs. GFP behavior statistics; (b-c) with vs. without imaging light | (a) `fig_1_big_kymograph.py`; (b-c) `fig_supp_no_laser_behavior.py` | `fig_supp_beh/` |
| GCaMP vs. GFP | GFP controls, variance, number of objects, dimensionality | `../supplement/fig_S4_gfp.py`, `../fig_2_intrinsic_dimensionality.py` (original submission) | `../supplement/gfp/`, `../supplement/intrinsic_dimension/` |
| Higher modes of CCA | CCA correlations and weights | `fig_2_CCA_with_eigenworms.py` | `fig2/` |
| Example PCA manifolds | PCA manifolds of all freely moving and immobilized datasets | `fig_supp_example_pca_manifolds.py` | `fig_supp_manifolds/` |
| PC1 weights with different preprocessing | Fig. 3 analysis with dR/R20 normalization | `fig_3_wbfm_immob_comparison.py` (`_r20` outputs) | `fig3/` |
| Bayesian modeling (a-l) | Example traces, Fourier analysis, triggered averages of the global component | `../supplement/fig_S7_neuron_fft.py`, `../fig_3_examples_residual_decomposition.py` (original submission) | `../supplement/fft_examples/`, `../multiplexing/` |
| Bayesian modeling (m-q) | PC1/PC2 coefficient tests, gating vs. manifold variance, phase-shift tests | `fig_bayesian_ALT_CONDA.py` | `fig5/` |
| Bayesian models with alternative manifolds | Hierarchical model with CCA or discrete-label manifolds | `bayesian_alt_manifold/fig_bayesian-CCA_ALT_CONDA.py`, `bayesian_alt_manifold/fig_bayesian-discrete_ALT_CONDA.py`, `bayesian_alt_manifold/comparison_plot.py` | `bayesian_alt_manifold/` |
| O2/CO2 responding neurons | Triggered averages and t-tests, wild type and gcy-31;-35;-9 | `fig_supp_O2_table.py` (freely moving); `../fig_4_O2_neurons.py` (immobilized columns, original submission) | `fig_supp_o2/`, `../multiplexing/` |
| Wild type vs. gcy-31;-35;-9, all neurons | Genotype comparison across all identified neurons, behavior of both genotypes | `fig_supp_O2_table.py` | `fig_supp_o2/` |

## Analyses for the response to reviewers (not in the paper figures)

| Analysis | Script |
|---|---|
| Bayesian model fit to AVB HisCl recordings and controls | `bayesian_avb_hiscl/` |
| Manifolds with more than two PCA modes | `fig_2_assemble_n_pca_modes.py`, `fig_2_supp_CCA_n_pca_modes.py` |
| Decoding head swings from neuronal activity | `fig_3_supp_decode_headswings.py` |

## Shared code

Most of the analysis code lives in the package itself, in particular:
- `wbfm/utils/general/utils_paper.py`: figure settings, PCA weight tests, packaging of Bayesian results
- `wbfm/utils/general/utils_paper_revisions_bayes.py`: statistics on the Bayesian model fits
- `wbfm/utils/external/utils_pymc.py`: the Bayesian models (PyMC)
- `wbfm/utils/visualization/paper_multidataset_triggered_average.py`: triggered averages and their statistics
- `wbfm/utils/general/utils_hardcoded.py`: paper dataset lists and neuron annotations
