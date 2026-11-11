#!/usr/bin/env python
# coding: utf-8
"""
Figure 2 CCA panels, redone with more neuronal PCA modes kept before CCA, plus supplementary comparison figures.

In fig_2_CCA_with_eigenworms, the neuronal traces are truncated to 3 PCA modes before CCA. With 3 PCA modes and 3 CCA
components, CCA can only rotate the top-3 PCA subspace. Here we keep N_PCA_MODES (default 5) instead.

Part 1: The same figures as fig_2_CCA_with_eigenworms (identical settings except the number of PCA modes)
        The pure-PCA panel (pca-2d) and the state-transition graphs do not depend on this, so they are not redone
Part 2: Comparisons between BASELINE_PCA_MODES (3) and N_PCA_MODES (supplement)
        A: Sweep over the number of PCA modes: neuronal variance explained, behavioral variance explained, and
           canonical correlations
        B: Which PCA modes each CCA mode uses (squared cosine heatmap)
        C: Similarity of the per-dataset neuronal CCA weights (3 vs N PCA modes)
        D: Median neuronal CCA weights per neuron (3 vs N PCA modes)

All files go to fig2_{N_PCA_MODES}pcs/ next to this script. Part 1 filenames match those in fig2/, and part 2 files are
prefixed with 'supp_npcs_'

Run from the repo root (~5 min to load data, then ~10 min):
    /home/charles/anaconda3/envs/wbfm/bin/python -I wbfm/notebooks/paper/revisions/fig_2_supp_CCA_n_pca_modes.py
Or run interactively cell-by-cell (# %% cells)
"""

# %% Setup
import os
from pathlib import Path

import matplotlib

try:
    get_ipython()  # noqa: F821
    INTERACTIVE = True
except NameError:
    INTERACTIVE = False
if not INTERACTIVE:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm.auto import tqdm

if not INTERACTIVE:
    # Figures are only saved; do not open a browser tab for every plotly figure
    go.Figure.show = lambda self, *args, **kwargs: None

from wbfm.utils.external.utils_plotly import plotly_plot_mean_and_shading
from wbfm.utils.general.utils_hardcoded import load_paper_datasets, neurons_with_confident_ids
from wbfm.utils.general.utils_paper import (apply_figure_settings, behavior_name_mapping, plot_foldchange_boxes,
                                            plotly_paper_color_discrete_map)
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.visualization.utils_cca import (CCAPlotter, calc_cca_weights_for_all_projects,
                                                calc_mode_correlation_for_all_projects,
                                                calc_r_squared_for_all_projects)

# %% Parameters
N_PCA_MODES = 5  # Number of neuronal PCA modes kept before CCA (the original figure uses 3)
BASELINE_PCA_MODES = 3  # For the comparison figures
N_PCA_MODES_SWEEP = [3, 4, 5, 6, 7, 8, 10]
N_CCA_COMPONENTS = 3  # Number of CCA components shown in all figures (same as the original figure)

TRACE_KWARGS = dict(use_paper_options=True)
BEH_KWARGS = dict(additional_behaviors=[f"eigenworm{i}" for i in range(4)])
EXAMPLE_PROJECT = "/lisc/data/scratch/neurobiology/zimmer/fieseler/wbfm_projects/manually_annotated/paper_data/ZIM2165_Gcamp7b_worm1-2022_11_28_updated_format/project_config.yaml"

try:
    BASE_FOLDER = Path(__file__).resolve().parent
except NameError:
    BASE_FOLDER = Path.cwd()
output_folder = str(BASE_FOLDER / f'fig2_{N_PCA_MODES}pcs')
os.makedirs(output_folder, exist_ok=True)
print(f"Saving figures to {output_folder}")

# Options shared by all CCA calculations of the original figure; only the number of PCA modes is changed
cca_opt = dict(preprocess_traces_using_pca=True, truncate_traces_to_n_components=N_PCA_MODES,
               trace_kwargs=TRACE_KWARGS, beh_kwargs=BEH_KWARGS)


def save_plotly(fig, fname, scale=3):
    fname = os.path.join(output_folder, fname)
    fig.write_image(fname, scale=scale)
    fig.write_image(str(Path(fname).with_suffix('.svg')))


def save_matplotlib(fname, dpi=100):
    fname = os.path.join(output_folder, fname)
    plt.savefig(fname, dpi=dpi)
    plt.savefig(str(Path(fname).with_suffix('.svg')))


def rename_eigenworms(name):
    return {f"eigenworm{i}": f"Eigenworm {i + 1}" for i in range(4)}.get(name, name)


# %% Load data
project_data_gcamp = ProjectData.load_final_project_data_from_config(EXAMPLE_PROJECT)
all_projects_gcamp = load_paper_datasets(['gcamp', 'hannah_O2_fm'])

# %% [markdown]
# # Part 1: Same figures as fig_2_CCA_with_eigenworms, with N_PCA_MODES

# %% Example dataset: 2d CCA latent spaces (continuous and discrete behaviors)
project_data_gcamp.use_physical_x_axis = True
cca_plotter = CCAPlotter(project_data_gcamp, preprocess_behavior_using_pca=True, **cca_opt)

fig = cca_plotter.plot(plot_3d=False, output_folder=output_folder, show_legend=False)
fig = cca_plotter.plot(plot_3d=False, binary_behaviors=True, show_legend=False, output_folder=output_folder)

# %% Neuronal variance explained (PCA vs CCA vs discrete CCA)
all_cca_classes, df_r_squared_melt, r_squared_per_row = calc_r_squared_for_all_projects(
    all_projects_gcamp, r_squared_kwargs=dict(n_components=[1, 2, 3]), melt=True, **cca_opt)

fig = px.box(df_r_squared_melt, x='n_components', color='Method', y='Variance Explained',
             color_discrete_map=plotly_paper_color_discrete_map())
apply_figure_settings(fig, width_factor=0.4, height_factor=0.2)
fig.update_yaxes(title='Neuronal Variance<br>Explained (cumulative)', range=[0, 1.1])
fig.update_xaxes(title='Number of components')
fig.update_layout(showlegend=True)
save_plotly(fig, 'top_mode_reconstruction_boxplot.png')
fig.show()

# %% Latent space quality: canonical correlations
all_cca_classes3, df_mode_correlations, df_mode_correlations_binary = calc_mode_correlation_for_all_projects(
    all_projects_gcamp, correlation_kwargs=dict(n_components=N_CCA_COMPONENTS), **cca_opt)

df_mode_correlations.index = np.arange(1, N_CCA_COMPONENTS + 1)
df_mode_correlations_binary.index = np.arange(1, N_CCA_COMPONENTS + 1)
df0 = df_mode_correlations.T.copy()
df1 = df_mode_correlations_binary.T.copy()
df0['Behavior type'] = 'Continuous'
df1['Behavior type'] = 'Discrete'
df_mode_combined = pd.concat([df0, df1])

fig = px.box(df_mode_combined, color='Behavior type', color_discrete_map=plotly_paper_color_discrete_map())
fig.update_xaxes(title="Component")
fig.update_yaxes(title="Correlation of <br> CCA latent spaces")
fig.update_layout(showlegend=False)
apply_figure_settings(fig, width_factor=0.2, height_factor=0.2)
save_plotly(fig, 'paired_boxplot_latent_space_combined12345.png', scale=7)
fig.show()

# %% PCA-CCA similarity (cosine between each CCA mode and the PCA mode with the same index)
all_dots = {i + 1: {name: c.calc_mode_dot_product(i) for name, c in all_cca_classes3.items()}
            for i in range(N_CCA_COMPONENTS)}
df_all_dots = pd.DataFrame(all_dots).melt(var_name='Component', value_name='PCA-CCA similarity')
df_all_dots['Comparison Method'] = 'CCA'

all_dots_discrete = {i + 1: {name: c.calc_mode_dot_product(i, binary_behaviors=True) for name, c in all_cca_classes3.items()}
                     for i in range(N_CCA_COMPONENTS)}
df_all_dots_discrete = pd.DataFrame(all_dots_discrete).melt(var_name='Component', value_name='PCA-CCA similarity')
df_all_dots_discrete['Comparison Method'] = 'CCA Discrete'

df_all_dots = pd.concat([df_all_dots, df_all_dots_discrete])
df_all_dots['PCA-CCA similarity'] = df_all_dots['PCA-CCA similarity'].abs()

fig = px.box(df_all_dots, x='Component', y='PCA-CCA similarity', color='Comparison Method',
             color_discrete_map=plotly_paper_color_discrete_map())
fig.update_yaxes(title='PCA-CCA similarity', range=[0, 1.1])
fig.update_layout(showlegend=False)
apply_figure_settings(fig, width_factor=0.2, height_factor=0.2)
save_plotly(fig, 'mode_dot_product.png')
fig.show()

# %% Behavioral variance explained by the neuronal latent space
all_cca_classes_beh, df_r_squared_melt_beh, all_r_squared_per_row_beh = calc_r_squared_for_all_projects(
    all_projects_gcamp, r_squared_kwargs=dict(n_components=[1, 2, 3], use_behavior=True, DEBUG=False),
    melt=True, **cca_opt)

df = all_r_squared_per_row_beh.copy()
df['Behavior Variable'] = df['Behavior Variable'].map(rename_eigenworms)

cmap = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
cmap.pop(2)  # Remove the horrible neon green

for method in df['Method'].unique():
    _df = df[df['Method'] == method].copy()
    cols = ['Behavior Variable', 'Components']
    vals = 'Cumulative Variance explained'
    df_sorted = _df.groupby(cols)[vals].mean().reset_index().sort_values(by=['Components', vals], ascending=False)
    category_orders = list(df_sorted[df_sorted['Components'] == 3]['Behavior Variable'].dropna())
    _df['Behavior Variable'] = pd.Categorical(_df['Behavior Variable'], categories=category_orders, ordered=True)
    _df = _df.sort_values('Behavior Variable')

    fig = px.box(_df, y='Cumulative Variance explained', color='Components', x='Behavior Variable',
                 facet_col='Method', color_discrete_sequence=cmap)
    show_legend = (method == 'CCA')
    fig.update_yaxes(range=[-0.01, 0.8])
    fig.update_layout(showlegend=show_legend)
    if show_legend:
        fig.update_layout(legend=dict(yanchor="middle", y=0.25, xanchor="left", x=1.01))
    else:
        fig.update_yaxes(title='')
    fig.update_xaxes(title=f"Method: {method}")
    apply_figure_settings(fig, width_factor=0.55 if show_legend else 0.4, height_factor=0.3)
    fig.update_layout(xaxis=dict(showline=True, linecolor="black"), overwrite=True)
    save_plotly(fig, f'behavior_variance_explained_{method}.png', scale=5)
    fig.show()

# %% Behavioral variance explained: colored boxes (PCA and CCA)
df = all_r_squared_per_row_beh.copy()
df = df[df['Method'].isin(['PCA', 'CCA'])]
df = df[df['Components'].isin([1, 2, 3])]
df['Behavior Variable'] = df['Behavior Variable'].map(rename_eigenworms)

groups = {'Discrete': ['Fwd-Bwd State', 'Dorsal turn', 'Ventral turn', 'Self-collision', 'Pause']}
_low_level = ['Ventral head', 'Dorsal head']
_low_level.extend([f"Eigenworm {i + 1}" for i in range(4)])
groups['Low level'] = _low_level
groups['High level'] = ['Velocity', 'Dorsal body', 'Ventral body']

cols = ['Method', 'Behavior Variable', 'Components']
vals = 'Cumulative Variance explained'
df_sorted = df.groupby(cols)[vals].mean().reset_index().sort_values(by=['Method', 'Components', vals], ascending=True)
neuron_order = list(df_sorted[(df_sorted['Method'] == 'PCA') & (df_sorted['Components'] == 3)]['Behavior Variable'])

ax, fig, ordered_neurons = plot_foldchange_boxes(
    df=df, behavior_col='Method', groups=groups, subtitle_behavior_col='Components', rows_col='Behavior Variable',
    value_col='Cumulative Variance explained', figsize=(8, 6), row_vspace=0.5, behavior_hspace=3.0, cmap='Greens',
    add_text=False, vmax=0.6, neuron_order=neuron_order, DEBUG=False)
plt.tight_layout()
apply_figure_settings(fig, width_factor=0.4, height_factor=0.25, plotly_not_matplotlib=False)
save_matplotlib('colored_box_beh_variance_explained.png')
plt.show()

# %% Behavioral variance explained: colored boxes (discrete CCA)
df = all_r_squared_per_row_beh.copy()
df = df[df['Method'].isin(['CCA Discrete'])]
df = df[df['Components'].isin([1, 2, 3])]

groups = {'Discrete': ['Fwd-Bwd State', 'Dorsal turn', 'Ventral turn', 'Self-collision', 'Pause']}
groups['Low level'] = ['eigenworm3', 'eigenworm2', 'eigenworm1', 'eigenworm0', 'Ventral head', 'Dorsal head']
groups['High level'] = ['Velocity', 'Dorsal body', 'Ventral body']

df_sorted = df.groupby(cols)[vals].mean().reset_index().sort_values(by=['Method', 'Components', vals], ascending=True)
neuron_order = list(df_sorted[(df_sorted['Method'] == 'CCA Discrete') & (df_sorted['Components'] == 3)]['Behavior Variable'])

ax, fig, ordered_neurons = plot_foldchange_boxes(
    df=df, behavior_col='Method', groups=groups, subtitle_behavior_col='Components', rows_col='Behavior Variable',
    value_col='Cumulative Variance explained', figsize=(8, 6), row_vspace=0.25, behavior_hspace=2.0, cmap='Greens',
    add_text=False, vmax=0.6, neuron_order=neuron_order, DEBUG=False)
plt.tight_layout()
apply_figure_settings(fig, width_factor=0.25, height_factor=0.4, plotly_not_matplotlib=False)
save_matplotlib('colored_box_beh_variance_explained_binary.png')
plt.show()

# %% SUPP: Variance explained per neuron (cumulative histogram)
_df = r_squared_per_row.rename(columns={'Behavior Variable': 'Neuron Name'})[r_squared_per_row['Components'] == 2]
df_var_exp_hist = _df.copy()

bins = np.linspace(0, 1, 51)
func = lambda Z: np.cumsum(np.histogram(Z, bins=bins)[0])
df_var_exp_hist = df_var_exp_hist.groupby(['Dataset Name', 'Method'])['Cumulative Variance explained'].apply(func)
long_vars = (df_var_exp_hist / df_var_exp_hist.apply(max)).reset_index().explode('Cumulative Variance explained').reset_index(drop=True)
fraction_count = df_var_exp_hist.apply(lambda x: bins[1:]).reset_index().explode('Cumulative Variance explained').reset_index(drop=True).rename(columns={'Cumulative Variance explained': 'bins'})
long_vars['bins'] = fraction_count['bins']
long_vars['Cumulative Variance explained'] = long_vars['Cumulative Variance explained'].astype(float)
long_vars['bins'] = long_vars['bins'].astype(float)

df_subset = long_vars[long_vars['Method'].isin(['CCA', 'CCA Discrete'])]
fig = plotly_plot_mean_and_shading(df_subset, x='bins', y='Cumulative Variance explained', color='Method',
                                   cmap=plotly_paper_color_discrete_map(), x_intersection_annotation=0.5)
fig.update_xaxes(title='Var. explained (fraction)', range=[0, 1.05])
fig.update_yaxes(title='Fraction of neurons <br> (cumulative)', range=[0, 1.05])
fig.update_layout(showlegend=False)
apply_figure_settings(fig, width_factor=0.3, height_factor=0.2)
save_plotly(fig, 'variance_explained_by_cca_cumulative.png')
fig.show()

# %% Neuronal CCA weights: mode 1
weights_opt = dict(combine_left_and_right=True, preprocess_behavior_using_pca=True, **cca_opt)
all_cca_classes1, df_weights1, df_weights_binary1 = calc_cca_weights_for_all_projects(
    all_projects_gcamp, which_mode=0, min_datasets_present=6, weights_kwargs=dict(n_components=2), **weights_opt)


def melt_both_weights(df_weights, df_weights_binary):
    df_both = df_weights.reset_index().melt(id_vars='index')
    df_both['Behavior Type'] = 'Continuous'
    df_both_binary = df_weights_binary.reset_index().melt(id_vars='index')
    df_both_binary['Behavior Type'] = 'Discrete'
    df_both = pd.concat([df_both, df_both_binary])
    df_both.columns = ['Dataset Name', 'Neuron', 'Weight', 'Behavior Type']
    return df_both


neurons_to_plot = neurons_with_confident_ids(combine_left_right=True)
df_both1 = melt_both_weights(df_weights1, df_weights_binary1)
fig = px.box(df_both1[df_both1['Neuron'].isin(neurons_to_plot)], x='Neuron', y='Weight', color='Behavior Type',
             hover_data=['Dataset Name'], color_discrete_map=plotly_paper_color_discrete_map())
apply_figure_settings(fig, width_factor=1.0, height_factor=0.2, plotly_not_matplotlib=True)
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="CCA Weight<br>(Component 1)")
fig.update_layout(legend=dict(yanchor="top", y=1.05, xanchor="left", x=0.8))
save_plotly(fig, 'paired_boxplot_neural_weights_BOTH.png')
fig.show()

# %% SUPP: Neuronal CCA weights: mode 2
all_cca_classes2, df_weights2, df_weights_binary2 = calc_cca_weights_for_all_projects(
    all_projects_gcamp, which_mode=1, min_datasets_present=6, weights_kwargs=dict(n_components=3),
    correct_sign_using_top_weight=True, **weights_opt)
df_weights2 = df_weights2[[c for c in df_weights2.columns if c in neurons_to_plot]]
df_weights_binary2 = df_weights_binary2[[c for c in df_weights_binary2.columns if c in neurons_to_plot]]

df_both2 = melt_both_weights(df_weights2, df_weights_binary2)
fig = px.box(df_both2[df_both2['Neuron'].isin(neurons_to_plot)], x='Neuron', y='Weight', color='Behavior Type',
             hover_data=['Dataset Name'], color_discrete_map=plotly_paper_color_discrete_map())
apply_figure_settings(fig, width_factor=1.0, height_factor=0.3, plotly_not_matplotlib=True)
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="CCA Weight (mode 2)")
fig.update_xaxes(title="", tickfont_size=12)
fig.update_layout(legend=dict(yanchor="top", y=1.0, xanchor="left", x=0.5))
save_plotly(fig, 'paired_boxplot_neural_weights2_BOTH.png')
fig.show()

# %% Behavioral CCA weights: mode 1 (continuous and discrete)
beh_weights_opt = dict(neural_not_behavioral=False, beh_kwargs=BEH_KWARGS, trace_kwargs=TRACE_KWARGS,
                       preprocess_traces_using_pca=True, truncate_traces_to_n_components=N_PCA_MODES)
all_cca_classes_beh1, df_weights_beh1, df_weights_binary_beh1 = calc_cca_weights_for_all_projects(
    all_projects_gcamp, which_mode=0, min_datasets_present=5, weights_kwargs=dict(n_components=2),
    preprocess_behavior_using_pca=False, **beh_weights_opt)

fig = px.box(df_weights_beh1.rename(columns=behavior_name_mapping(shorten=True)),
             color_discrete_sequence=[plotly_paper_color_discrete_map()['CCA']])
apply_figure_settings(fig, width_factor=0.25, height_factor=0.25, plotly_not_matplotlib=True)
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="CCA Weight<br>(Component 1)")
fig.update_xaxes(title="", tickfont_size=12)
save_plotly(fig, 'paired_boxplot_beh_weights.png')
fig.show()

fig = px.box(df_weights_binary_beh1.rename(columns=behavior_name_mapping(shorten=True)),
             color_discrete_sequence=[plotly_paper_color_discrete_map()['Discrete']])
fig.update_layout(showlegend=False)
apply_figure_settings(fig, width_factor=0.25, height_factor=0.25, plotly_not_matplotlib=True)
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="Discrete CCA<br>Weight<br>(Component 1)",
                 range=[-0.2, 1.1])
fig.update_xaxes(title="", tickfont_size=12)
save_plotly(fig, 'paired_boxplot_beh_weights_binary.png')
fig.show()

# %% SUPP: Behavioral CCA weights: mode 2 (continuous and discrete)
all_cca_classes_beh2, df_weights_beh2, df_weights_binary_beh2 = calc_cca_weights_for_all_projects(
    all_projects_gcamp, which_mode=1, min_datasets_present=5, weights_kwargs=dict(n_components=3),
    preprocess_behavior_using_pca=False, **beh_weights_opt)

fig = px.box(df_weights_beh2.rename(columns=behavior_name_mapping(shorten=True)),
             color_discrete_sequence=[plotly_paper_color_discrete_map()['CCA']])
apply_figure_settings(fig, width_factor=0.25, height_factor=0.3, plotly_not_matplotlib=True)
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="CCA Weight <br> (mode 2)")
fig.update_xaxes(title="", tickfont_size=12)
save_plotly(fig, 'paired_boxplot_beh_weights2.png')
fig.show()

fig = px.box(df_weights_binary_beh2.rename(columns=behavior_name_mapping(shorten=True)),
             color_discrete_sequence=[plotly_paper_color_discrete_map()['Discrete']])
apply_figure_settings(fig, width_factor=0.25, height_factor=0.3, plotly_not_matplotlib=True)
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="CCA Weight <br> (discrete mode 2)")
fig.update_xaxes(title="")
save_plotly(fig, 'paired_boxplot_beh_weights_binary2.png')
fig.show()

# %% SUPP: Behavioral CCA weights: mode 3 (continuous)
# Note: as in the original notebook, this one uses behavior PCA preprocessing (which skips the sign anchoring)
all_cca_classes_beh3, df_weights_beh3, df_weights_binary_beh3 = calc_cca_weights_for_all_projects(
    all_projects_gcamp, which_mode=2, min_datasets_present=5, weights_kwargs=dict(n_components=3),
    preprocess_behavior_using_pca=True, **beh_weights_opt)

fig = px.box(df_weights_beh3.rename(columns=behavior_name_mapping(shorten=True)),
             color_discrete_sequence=[plotly_paper_color_discrete_map()['CCA']])
apply_figure_settings(fig, width_factor=0.25, height_factor=0.2, plotly_not_matplotlib=True)
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="Weight <br> (mode 3)")
fig.update_xaxes(title="")
save_plotly(fig, 'paired_boxplot_beh_weights3.png')
fig.show()

# %% [markdown]
# # Part 2: Comparisons across the number of PCA modes (supplement)

# %% Sweep: build CCA classes for each number of PCA modes and calculate summary statistics
# Same CCA options as the neuronal variance explained calculation above (no behavior PCA preprocessing)
sweep_plotters = {k: {name: CCAPlotter(p, preprocess_traces_using_pca=True, truncate_traces_to_n_components=k,
                                       trace_kwargs=TRACE_KWARGS, beh_kwargs=BEH_KWARGS)
                      for name, p in all_projects_gcamp.items()}
                  for k in tqdm(N_PCA_MODES_SWEEP, desc='Building CCA classes for each number of PCA modes')}

method_opts = {'CCA': dict(binary_behaviors=False), 'CCA Discrete': dict(binary_behaviors=True)}
sweep_rows = []
for k, plotters in tqdm(sweep_plotters.items(), desc='Sweep statistics'):
    for name, c in plotters.items():
        for method, opt in method_opts.items():
            r2_neur, _ = c.calc_r_squared(use_pca=False, n_components=N_CCA_COMPONENTS, **opt)
            r2_beh, _ = c.calc_r_squared(use_pca=False, n_components=N_CCA_COMPONENTS, use_behavior=True, **opt)
            corrs = c.calc_mode_correlation(n_components=N_CCA_COMPONENTS, **opt)
            row = {'n_pca_modes': k, 'Dataset Name': name, 'Method': method,
                   'Neuronal variance explained': r2_neur, 'Behavioral variance explained': r2_beh}
            row.update({f'Canonical correlation {i + 1}': corr for i, corr in enumerate(corrs)})
            sweep_rows.append(row)
        if k == N_PCA_MODES_SWEEP[0]:
            # PCA reference (independent of the number of PCA modes kept for CCA); 3 PCA modes
            r2_neur, _ = c.calc_r_squared(use_pca=True, n_components=N_CCA_COMPONENTS)
            r2_beh, _ = c.calc_r_squared(use_pca=True, n_components=N_CCA_COMPONENTS, use_behavior=True)
            for _k in N_PCA_MODES_SWEEP:
                sweep_rows.append({'n_pca_modes': _k, 'Dataset Name': name, 'Method': 'PCA',
                                   'Neuronal variance explained': r2_neur, 'Behavioral variance explained': r2_beh})
df_sweep = pd.DataFrame(sweep_rows)
df_sweep.to_csv(os.path.join(output_folder, 'supp_npcs_sweep_statistics.csv'), index=False)
print(df_sweep.groupby(['Method', 'n_pca_modes']).mean(numeric_only=True).round(3))

# %% A: Neuronal and behavioral variance explained vs. number of PCA modes
x_title = 'Neuronal PCA modes<br>kept before CCA'
for y_col, y_title, fname in [
        ('Neuronal variance explained', f'Neuronal variance<br>explained ({N_CCA_COMPONENTS} comp.)', 'supp_npcs_neuronal_variance_explained.png'),
        ('Behavioral variance explained', f'Behavioral variance<br>explained ({N_CCA_COMPONENTS} comp.)', 'supp_npcs_behavioral_variance_explained.png')]:
    fig = plotly_plot_mean_and_shading(df_sweep, x='n_pca_modes', y=y_col, color='Method',
                                       cmap=plotly_paper_color_discrete_map())
    fig.add_vline(x=N_PCA_MODES, line=dict(color='black', width=1, dash='dash'))
    fig.update_xaxes(title=x_title, tickvals=N_PCA_MODES_SWEEP)
    fig.update_yaxes(title=y_title, range=[0, 1.0])
    fig.update_layout(legend=dict(yanchor="bottom", y=0.0, xanchor="right", x=1.0))
    apply_figure_settings(fig, width_factor=0.3, height_factor=0.2)
    save_plotly(fig, fname)
    fig.show()

# %% A: Canonical correlations vs. number of PCA modes
mode_cmap = {f'Component {i + 1}': c for i, c in enumerate(px.colors.qualitative.Safe[:N_CCA_COMPONENTS])}
for method in method_opts:
    _df = df_sweep[df_sweep['Method'] == method].melt(
        id_vars=['n_pca_modes', 'Dataset Name'], value_vars=[f'Canonical correlation {i + 1}' for i in range(N_CCA_COMPONENTS)],
        var_name='Component', value_name='Canonical correlation')
    _df['Component'] = _df['Component'].str.replace('Canonical correlation', 'Component')
    fig = plotly_plot_mean_and_shading(_df, x='n_pca_modes', y='Canonical correlation', color='Component', cmap=mode_cmap)
    fig.add_vline(x=N_PCA_MODES, line=dict(color='black', width=1, dash='dash'))
    fig.update_xaxes(title=x_title, tickvals=N_PCA_MODES_SWEEP)
    fig.update_yaxes(title=f'Correlation of CCA<br>latent spaces ({method})', range=[0, 1.05])
    fig.update_layout(legend=dict(yanchor="bottom", y=0.0, xanchor="right", x=1.0))
    apply_figure_settings(fig, width_factor=0.3, height_factor=0.2)
    save_plotly(fig, f'supp_npcs_canonical_correlation_{method.replace(" ", "_")}.png')
    fig.show()


# %% B: Which PCA modes does each CCA mode use? (squared cosine, averaged over datasets)
def calc_cca_pca_overlap(c: CCAPlotter, binary_behaviors: bool) -> np.ndarray:
    """Squared cosine between each neuronal CCA mode and each kept PCA mode; shape (n_cca_modes, n_pca_modes)

    The CCA weights live in the space of the kept PCA modes, so each row sums to 1
    """
    _, _, cca = c.calc_cca(n_components=N_CCA_COMPONENTS, binary_behaviors=binary_behaviors)
    w = cca.x_weights_.T  # (n_cca_modes, n_pca_modes), in PCA coordinates (orthonormal basis)
    return w ** 2 / (w ** 2).sum(axis=1, keepdims=True)


ks_to_compare = [BASELINE_PCA_MODES, N_PCA_MODES]
for method, opt in method_opts.items():
    fig = make_subplots(rows=1, cols=2, column_widths=[BASELINE_PCA_MODES, N_PCA_MODES], horizontal_spacing=0.08,
                        subplot_titles=[f'{k} PCA modes kept' for k in ks_to_compare])
    rows_csv = []
    for i_col, k in enumerate(ks_to_compare):
        overlap = np.mean([calc_cca_pca_overlap(c, **opt) for c in sweep_plotters[k].values()], axis=0)
        x_labels = [f'PC{j + 1}' for j in range(k)]
        y_labels = [f'CCA {i + 1}' for i in range(N_CCA_COMPONENTS)]
        fig.add_trace(go.Heatmap(z=overlap, x=x_labels, y=y_labels, zmin=0, zmax=1, colorscale='Greens',
                                 text=np.round(overlap, 2), texttemplate='%{text:.2f}', showscale=(i_col == 1),
                                 colorbar=dict(title='cos²', len=0.9)),
                      row=1, col=i_col + 1)
        rows_csv.append(pd.DataFrame(overlap, index=y_labels, columns=x_labels).assign(n_pca_modes=k))
    fig.update_yaxes(autorange='reversed')
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    apply_figure_settings(fig, width_factor=0.45, height_factor=0.15)
    save_plotly(fig, f'supp_npcs_cca_pca_overlap_{method.replace(" ", "_")}.png')
    pd.concat(rows_csv).to_csv(os.path.join(output_folder, f'supp_npcs_cca_pca_overlap_{method.replace(" ", "_")}.csv'))
    fig.show()

# %% C: Per-dataset similarity of the neuronal CCA weights (baseline vs N_PCA_MODES)
sim_rows = []
for method, opt in method_opts.items():
    for name in all_projects_gcamp:
        all_w = []
        for k in ks_to_compare:
            c = sweep_plotters[k][name]
            _, _, cca = c.calc_cca(n_components=N_CCA_COMPONENTS, **opt)
            all_w.append(c.get_weights_from_cca(cca, opt['binary_behaviors'])[0].values)
        for i in range(N_CCA_COMPONENTS):
            w0, w1 = all_w[0][i], all_w[1][i]
            sim_rows.append({'Dataset Name': name, 'Behavior type': 'Continuous' if method == 'CCA' else 'Discrete',
                             'Component': i + 1, 'Weight similarity': abs(w0 @ w1) / (np.linalg.norm(w0) * np.linalg.norm(w1))})
df_weight_similarity = pd.DataFrame(sim_rows)
df_weight_similarity.to_csv(os.path.join(output_folder, 'supp_npcs_weight_similarity.csv'), index=False)
print(df_weight_similarity.groupby(['Behavior type', 'Component'])['Weight similarity'].median().round(3))

fig = px.box(df_weight_similarity, x='Component', y='Weight similarity', color='Behavior type',
             color_discrete_map=plotly_paper_color_discrete_map())
fig.update_yaxes(title=f'Neuronal weight similarity<br>({BASELINE_PCA_MODES} vs. {N_PCA_MODES} PCA modes)', range=[0, 1.05])
fig.update_layout(showlegend=False)
apply_figure_settings(fig, width_factor=0.2, height_factor=0.2)
save_plotly(fig, 'supp_npcs_weight_similarity.png')
fig.show()

# %% D: Median neuronal CCA weight per neuron (baseline vs N_PCA_MODES), same settings as the weight figures above
baseline_weights_opt = weights_opt.copy()
baseline_weights_opt['truncate_traces_to_n_components'] = BASELINE_PCA_MODES
_, df_weights1_base, df_weights_binary1_base = calc_cca_weights_for_all_projects(
    all_projects_gcamp, which_mode=0, min_datasets_present=6, weights_kwargs=dict(n_components=2), **baseline_weights_opt)
_, df_weights2_base, df_weights_binary2_base = calc_cca_weights_for_all_projects(
    all_projects_gcamp, which_mode=1, min_datasets_present=6, weights_kwargs=dict(n_components=3), **baseline_weights_opt)

all_medians = []
for mode, beh_type, df_base, df_new in [(1, 'Continuous', df_weights1_base, df_weights1),
                                        (1, 'Discrete', df_weights_binary1_base, df_weights_binary1),
                                        (2, 'Continuous', df_weights2_base, df_weights2),
                                        (2, 'Discrete', df_weights_binary2_base, df_weights_binary2)]:
    neurons = [n for n in neurons_to_plot if n in df_base.columns and n in df_new.columns]
    _df = pd.DataFrame({'baseline': df_base[neurons].median(), 'new': df_new[neurons].median()})
    # The overall sign is set separately for each number of PCA modes (by the largest weight), so align it here
    if _df['baseline'].corr(_df['new']) < 0:
        print(f"Flipping the overall sign of the {N_PCA_MODES}-mode weights for component {mode} ({beh_type})")
        _df['new'] *= -1
    _df['Component'] = mode
    _df['Behavior Type'] = beh_type
    all_medians.append(_df.rename_axis('Neuron').reset_index())
df_medians = pd.concat(all_medians)
df_medians.to_csv(os.path.join(output_folder, 'supp_npcs_median_weights.csv'), index=False)

fig = make_subplots(rows=1, cols=2, subplot_titles=['Component 1', 'Component 2'], horizontal_spacing=0.15)
cmap = plotly_paper_color_discrete_map()
n_labels = 6  # Label the neurons with the largest weights (per component and behavior type)
for i_col, mode in enumerate([1, 2]):
    for beh_type in ['Continuous', 'Discrete']:
        _df = df_medians[(df_medians['Component'] == mode) & (df_medians['Behavior Type'] == beh_type)]
        top = set(_df.assign(m=_df[['baseline', 'new']].abs().max(axis=1)).nlargest(n_labels, 'm')['Neuron'])
        r = _df['baseline'].corr(_df['new'])
        fig.add_trace(go.Scatter(x=_df['baseline'], y=_df['new'], mode='markers+text', name=f'{beh_type} (r={r:.2f})',
                                 text=[n if n in top else '' for n in _df['Neuron']], textposition='top center',
                                 textfont=dict(size=8), marker=dict(color=cmap[beh_type], size=6),
                                 hovertext=_df['Neuron'], showlegend=True),
                      row=1, col=i_col + 1)
    lim = df_medians.loc[df_medians['Component'] == mode, ['baseline', 'new']].abs().max().max() * 1.1
    fig.add_shape(type='line', x0=-lim, y0=-lim, x1=lim, y1=lim, line=dict(color='gray', dash='dash', width=1),
                  row=1, col=i_col + 1)
    fig.update_xaxes(title=f'Median weight<br>({BASELINE_PCA_MODES} PCA modes)', range=[-lim, lim],
                     zeroline=True, zerolinecolor='black', zerolinewidth=1, row=1, col=i_col + 1)
    fig.update_yaxes(range=[-lim, lim], zeroline=True, zerolinecolor='black', zerolinewidth=1, row=1, col=i_col + 1)
fig.update_yaxes(title=f'Median weight<br>({N_PCA_MODES} PCA modes)', row=1, col=1)
fig.update_layout(legend=dict(yanchor="top", y=-0.35, xanchor="left", x=0.0, orientation='h'))
apply_figure_settings(fig, width_factor=0.5, height_factor=0.25)
save_plotly(fig, 'supp_npcs_median_weights_scatter.png')
fig.show()

# %%
print(f"Done; figures saved in {output_folder}")
