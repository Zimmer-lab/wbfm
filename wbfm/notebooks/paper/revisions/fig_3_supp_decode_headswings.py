#!/usr/bin/env python
# coding: utf-8
"""
Reviewer: "How does the GFP data perform for the analyses in Fig. 3? E.g. since you can not obtain a true manifold,
can you decode headswings from GFP data? And does the same pattern appear when annotating fwd/backward from behavior,
rather than PC1?"

Opposite direction of the Fig. 3 Bayesian model: decode head swings (eigenworm 1+2) from the whole population of each
dataset, for GCaMP and GFP, from raw activity and from residuals (PCA manifold / CCA continuous / discrete behaviors).
Methods text: see DECODING_METHODS in wbfm/utils/general/utils_paper_revisions_decoding.py (filled in and saved below).

Run from the repo root:
    ~/anaconda3/envs/wbfm/bin/python -I wbfm/notebooks/paper/revisions/fig_3_supp_decode_headswings.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from wbfm.utils.general.utils_hardcoded import get_hierarchical_modeling_dir
from wbfm.utils.general.utils_paper import apply_figure_settings, plotly_paper_color_discrete_map, \
    data_type_name_mapping
import wbfm.utils.general.utils_paper_revisions_decoding as dec

output_folder = Path(__file__).parent / 'fig5_decoding'
output_folder.mkdir(exist_ok=True)
recalculate = '--recalculate' in sys.argv
N_SHUFFLES, N_FOLDS, GAP, MAX_NAN = 100, 5, 10, 0.05

# Load the same data as the Bayesian model
Xy = pd.read_hdf(os.path.join(get_hierarchical_modeling_dir(), 'data.h5'))
Xy_gfp = pd.read_hdf(os.path.join(get_hierarchical_modeling_dir(gfp=True), 'data.h5'))
name_gcamp, name_gfp = data_type_name_mapping()['gcamp'], data_type_name_mapping()['gfp']

# Decode (or load cached)
fname = output_folder / 'headswing_decoding_results.csv'
if fname.exists() and not recalculate:
    df = pd.read_csv(fname)
else:
    df = dec.run_population_decoding({name_gcamp: Xy, name_gfp: Xy_gfp}, n_shuffles=N_SHUFFLES, n_folds=N_FOLDS,
                                     gap=GAP, max_nan_fraction=MAX_NAN)
    df.to_csv(fname, index=False)

# Summary table + group statistics
mode_order = list(dec.RESIDUAL_MODES.keys())
df['significant'] = df['r2'] > df['r2_null_95']
summary = df.groupby(['residual_mode', 'datatype']).agg(
    n_datasets=('r2', 'size'), median_r2=('r2', 'median'), iqr_low=('r2', lambda x: x.quantile(0.25)),
    iqr_high=('r2', lambda x: x.quantile(0.75)), median_null_95=('r2_null_95', 'median'),
    frac_significant=('significant', 'mean'), median_n_neurons=('n_neurons', 'median')).reindex(mode_order, level=0)
stats = {}
for mode in mode_order:
    a = df.query('residual_mode == @mode and datatype == @name_gcamp')['r2']
    b = df.query('residual_mode == @mode and datatype == @name_gfp')['r2']
    stats[mode] = mannwhitneyu(a, b, alternative='two-sided').pvalue
summary['mannwhitney_p_gcamp_vs_gfp'] = summary.index.get_level_values(0).map(stats)
print(summary.round(3).to_string())
summary.to_csv(output_folder / 'headswing_decoding_summary.csv')

# Methods text, with numbers filled in
methods = dec.DECODING_METHODS.format(
    n_gcamp=Xy.dataset_name.nunique(), n_gfp=Xy_gfp.dataset_name.nunique(), max_nan_pct=int(100 * MAX_NAN),
    n_neurons_gcamp=int(df.query('datatype == @name_gcamp')['n_neurons'].median()),
    n_neurons_gfp=int(df.query('datatype == @name_gfp')['n_neurons'].median()),
    n_folds=N_FOLDS, gap=GAP, gap_s=GAP * 0.288, n_alphas=len(dec.ALPHAS),
    a_min=np.log10(dec.ALPHAS.min()), a_max=np.log10(dec.ALPHAS.max()), n_shuffles=N_SHUFFLES)
(output_folder / 'methods_headswing_decoding.txt').write_text(methods)

# Plot: per-dataset cross-validated R2
cmap = plotly_paper_color_discrete_map()
fig = px.box(df, x='residual_mode', y='r2', color='datatype', points='all',
             category_orders={'residual_mode': mode_order, 'datatype': [name_gcamp, name_gfp]},
             color_discrete_map={k: cmap.get(k, None) for k in [name_gcamp, name_gfp]},
             labels={'r2': 'Head swing decoding<br>(cross-validated R<sup>2</sup>)', 'residual_mode': '',
                     'datatype': ''})
fig.add_hline(y=0, line=dict(color='black', dash='dash', width=1))
for i, mode in enumerate(mode_order):
    p = stats[mode]
    label = 'n.s.' if p > 0.05 else ('***' if p < 0.001 else '**' if p < 0.01 else '*')
    fig.add_annotation(x=mode, y=df['r2'].max() * 1.08, text=label, showarrow=False)
apply_figure_settings(fig, width_factor=0.6, height_factor=0.3)
fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='left', x=0))
fig.update_xaxes(tickangle=20)
for ext in ['png', 'svg']:
    fig.write_image(output_folder / f'headswing_decoding_r2.{ext}', **({'scale': 3} if ext == 'png' else {}))
fig.write_html(output_folder / 'headswing_decoding_r2.html')
