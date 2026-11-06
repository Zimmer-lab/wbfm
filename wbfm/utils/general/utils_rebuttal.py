
# ======================================
# Configuration parameters (edit as needed)
# ======================================
"""
SUPPLEMENTAL FIGURES S2A, S2B, S2C: DUAL-MODE VISUALIZATION (March 2026 REFACTOR)
=================================================================================

ARCHITECTURE OVERVIEW:
This module generates publication-quality supplement figures with two visualization modes:

1. CLASSIC MODE (use_mean_and_shading=False):
   - S2A: Stacked subplots (original layout)
   - S2B: Grouped boxplots (percent neurons above threshold)
   - S2C: Bar plots (quiet proportions)

2. MEAN+SHADING MODE (use_mean_and_shading=True):
   - S2A: 2 rows × 4 cols grid (PSD/CDF rows, conditions 0-3 columns)
   - S2B: Line plots with mean ± shading per wavelength
   - S2C: Line plots with mean ± shading per wavelength

KEY DESIGN DECISIONS:
--------------------

DATA FORMAT (Critical for Understanding):
- Data preparation functions (prepare_s2a_psd_data_for_mean_and_shading, etc.)
  intentionally produce ONE ROW PER FREQUENCY PER RECORDING PER GROUP.
- NO pre-averaging in data prep. This separation of concerns prevents bugs.
- Aggregation happens downstream in plotly_plot_mean_and_shading:
  * Groups by (condition, wavelength)
  * Computes mean/std ACROSS RECORDINGS for each group
  
Example data structure after prepare_s2a_psd_data_for_mean_and_shading:
  frequency  psd         group
  0.008      0.5         0 | Green       <- freq bin from recording 1, cond 0, wavelength Green
  0.008      0.6         0 | Green       <- freq bin from recording 2, cond 0, wavelength Green
  0.008      0.3         1 | Green       <- freq bin from recording 1, cond 1, wavelength Green
  (continues for all freq bins, all recordings, all groups...)

S2A SPECIAL ARCHITECTURE (2×4 GRID):
When use_mean_and_shading=True:
  - Creates 2 rows (PSD, CDF) × 4 cols (Condition 0, 1, 2, 3)
  - CONDITION MUST be in [0,1,2,3] for grid layout to work
  - Each subplot shows all wavelengths for that condition (color-coded)
  - plotly_plot_mean_and_shading computes separate mean/std per wavelength within condition
  - Band-edge lines (F_LOW, F_HIGH) shown on all subplots
  - X-axis uniform across all subplots: [0, 2*F_HIGH]

COLOR MAPPING STRATEGY:
- make_wavelength_color_map(results) returns {wavelength -> color} palette
- Type consistency critical: use original wavelength objects from results dict
- Do NOT convert wavelengths to strings before building color map
- Build group_cmap directly from results.items() to maintain type alignment

SPECTRUM CONFIGURATION:
- Frequency band: [F_LOW=0.007, F_HIGH=0.033] Hz
- X-axis limit: [0, 2*F_HIGH] Hz
- Post-processing: percentile normalization per recording
- CDF computed via complementary empirical CDF (1 - standard CDF)

METRICS COMPUTED (compute_all_metrics output):
- PSD: Power Spectral Density (L/period, normalized per recording)
- CDF: Cumulative Distribution Function of activity levels
- S2B: Percent neurons with frequency band activity > THRESHOLD_FRACTION
- S2C: Percent recordings with >QUIET_THRESHOLD neurons above threshold

WORKFLOW:
1. reproduce_figures_plotly: Main entry point
2. compute_all_metrics: Extract PSD/CDF/thresholds per recording
3. Conditional logic based on use_mean_and_shading flag:
   IF False: Use original plot_s2a_plotly_simple, etc. (stacked layout)
   IF True: 
     a. prepare_s2a_psd_data_for_mean_and_shading -> DataFrame
     b. prepare_s2a_cdf_data_for_mean_and_shading -> DataFrame
     c. plot_s2a_mean_and_shading -> 2×4 grid figure
     (Similar pattern for S2B, S2C)

ERROR PREVENTION NOTES:
- Ensure results dict has condition keys in [0,1,2,3] (not strings like "0")
- Don't aggregate data before prepare_s2a_* (causes group mixing)
- Check wavelength type consistency in color map building
- Use direct results dict keys for cmap, not derived values
"""
from collections import defaultdict
import logging
import os
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, norm
from tqdm.auto import tqdm

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from wbfm.utils.external.utils_pandas import fill_missing_indices_with_nan
from wbfm.utils.external.utils_plotly import combine_plotly_figures, extract_shapes_as_figure, plotly_plot_mean_and_shading
from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes, options_for_ethogram
from wbfm.utils.general.utils_paper import apply_figure_settings, split_time_series_with_laser_switches
from wbfm.utils.projects.finished_project_data import split_project_data_in_time
from wbfm.utils.visualization.plot_summary_statistics import calc_speed_dataframe
from wbfm.utils.visualization.plot_traces import make_summary_heatmap_and_subplots


F_LOW = 0.007    # Hz, lower bound of band [^1]
F_HIGH = 0.033   # Hz, upper bound of band [^1]
THRESHOLD_FRACTION = 0.2  # per-neuron band fraction threshold (Figure S2B metric) [^2]
QUIET_THRESHOLD = 20.0    # % of neurons above THRESHOLD_FRACTION to classify "quiet" [^1]
ALPHA_BH = 0.05           # Benjamini–Hochberg FDR level [^2]

# ==================================
# Helper functions: spectral metrics
# ==================================


def make_heatmap_stack(these_heatmaps: dict, these_ethograms: dict, output_folder=None, prefix='', DEBUG=False):
        
    n = len(these_heatmaps)
    base_row_heights = np.array([0.85, 0.1, 0.05])
    base_row_heights = list(base_row_heights / base_row_heights.sum())
    
    all_row_heights = list(np.array(n*base_row_heights) / n)
    
    subplot_opt = dict(rows=len(all_row_heights), row_heights=all_row_heights, 
                    vertical_spacing=0.0
                    )
    
    all_figs = []
    for k in these_heatmaps.keys():
        all_figs.append(these_heatmaps[k])
        all_figs.append(extract_shapes_as_figure(these_ethograms[k], only_include_shapes_with_yref='y'))
        all_figs.append(go.Figure())  # Dummy empty figure
    
    fig = combine_plotly_figures(all_figs, horizontal=False, custom_subplot_opt=subplot_opt, hide_interior_xlabels=True,
                                force_yref_paper=False)
    
    fig.update_yaxes(title="", showticklabels=False, overwrite=True)
    fig.update_xaxes(title="", showticklabels=False, overwrite=True)
    fig.update_xaxes(title="Seconds", showticklabels=True, row=len(all_row_heights), overwrite=True)
    
    apply_figure_settings(fig=fig, width_factor=0.15, height_factor=1.0)
    
    if output_folder is not None:
        fname = os.path.join(output_folder, f'stacked_heatmaps_with_ethograms-{prefix}.png')
        fig.write_image(fname, scale=3)

    return fig


def add_vline_based_on_splits(_fig, vps, splits):
    for s in splits[:-1]:
        opt = dict(x=s[1]/vps, line_width=2, line_color='black')#, line_dash='dash')
        _fig.add_vline(**opt, y0=0, y1=1)


def make_heatmap(dat, splits=None, vps=None):
    x_for_plots_volumes = dat.columns
    heatmap = go.Heatmap(y=dat.index, z=dat, x=x_for_plots_volumes,
                         zmin=-0.25, zmax=1.25, colorscale='jet', xaxis="x", yaxis="y",
                         coloraxis='coloraxis1')
    
    fig = go.Figure()
    fig.add_trace(heatmap)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(showlegend=False, autosize=False, #**plotly_opt,
                       coloraxis=dict(colorscale="jet"))
    
    fig.update_coloraxes(cmin=-0.25, cmax=0.75, colorbar=dict(
        # thickness=10,
        # title=dict(text=r'ΔR / R₅₀', **font_dict)
        # title=dict(text=r'$\frac{\Delta R}{R_{50}}$', **font_dict)
    ))

    if splits is not None and vps is not None:
        add_vline_based_on_splits(fig, vps, splits)

    return fig


def make_ethogram(df_beh, splits=None, vps=None, use_alternate_cmap=False):
    ethogram_cmap_opt = dict()
    
    ethogram_opt = options_for_ethogram(df_beh, **ethogram_cmap_opt, include_turns=False,
                                        to_extend_short_states=False, use_alternate_cmap=use_alternate_cmap)
    fig_beh = go.Figure()
    fig_beh.update_layout(shapes=[opt for opt in ethogram_opt])

    if splits is not None and vps is not None:
        add_vline_based_on_splits(fig_beh, vps, splits)

    return fig_beh
