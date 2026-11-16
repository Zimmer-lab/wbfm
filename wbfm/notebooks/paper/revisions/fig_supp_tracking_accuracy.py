#!/usr/bin/env python
# coding: utf-8

# # Supplementary figure: tracking accuracy of pipeline steps
# 
# Compares three sets of tracks to the manually corrected ground truth (GT):
# 1. **Single reference frame**: global tracker using only one reference frame
# 2. **Multiple reference frames**: global tracker (intermediate output)
# 3. **Full pipeline**: global tracker + tracklets (initial pipeline output, before manual correction)
# 
# Accuracy for a neuron = fraction of time points (where the GT is not NaN) at which the tracker assigned the same segmented object as the GT.
# 
# Main dataset: `ZIM2165_Gcamp7b_worm1-2022_11_28` (the only GT dataset that has all three track sets).

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.performance.comparing_ground_truth import calculate_accuracy_from_dataframes, calc_accuracy_of_pipeline_steps
from wbfm.utils.general.utils_paper import apply_figure_settings


# In[2]:


gt_parent_folder = Path("/lisc/data/scratch/neurobiology/zimmer/fieseler/wbfm_projects/manually_annotated/paper_data")
# Note: the folders without '_updated_format' have old config files that no longer load
fname = gt_parent_folder / "ZIM2165_Gcamp7b_worm1-2022_11_28_updated_format" / "project_config.yaml"
project_data_gcamp = ProjectData.load_final_project_data_from_config(fname)

output_folder = "pipeline_accuracy"
os.makedirs(output_folder, exist_ok=True)
to_save = True

def save_plotly(fig, name):
    if not to_save:
        return
    fname = os.path.join(output_folder, name + ".png")
    fig.write_image(fname, scale=5)
    fig.write_image(str(Path(fname).with_suffix('.svg')))


# In[3]:


# Fixed order and colors for the three methods, used in all plots
method_order = ['Single reference frame', 'Multiple reference frames', 'Full pipeline']
method_colors = {'Single reference frame': '#B0B0B0',
                 'Multiple reference frames': '#6A8FBF',
                 'Full pipeline': '#1F3A93'}


# ## Calculate per-neuron accuracy

# In[4]:


plt.rcParams["font.family"] = "DejaVu Sans"  # Default

# Also produces a paired (matplotlib) boxplot
df_acc = calc_accuracy_of_pipeline_steps(project_data_gcamp, remove_gt_nan=True)
df_acc = df_acc[method_order]
df_acc.index.name = 'Neuron'

apply_figure_settings(fig=None, width_factor=0.5, height_factor=0.3, plotly_not_matplotlib=False)
if to_save:
    fname = os.path.join(output_folder, "pipeline_steps_boxplots.png")
    plt.savefig(fname, transparent=True)
    plt.savefig(str(Path(fname).with_suffix('.svg')))


# ### Summary table (useful for the text / caption)

# In[5]:


def summarize_accuracy(df_acc):
    summary = pd.DataFrame({
        'Median accuracy': df_acc.median(),
        'Mean accuracy': df_acc.mean(),
        'Neurons >= 0.90': (df_acc >= 0.90).sum(),
        'Neurons >= 0.95': (df_acc >= 0.95).sum(),
        'Fraction neurons >= 0.95': (df_acc >= 0.95).mean(),
        'N neurons': df_acc.count(),
    })
    return summary

df_summary = summarize_accuracy(df_acc)
if to_save:
    df_summary.to_csv(os.path.join(output_folder, "pipeline_steps_summary.csv"))
df_summary.round(3)


# In[6]:


# Paired tests (same neurons tracked by each method), all pairwise comparisons, Bonferroni corrected
from itertools import combinations
from scipy.stats import wilcoxon
from wbfm.utils.visualization.utils_plot_traces import p_value_to_stars
pairs = list(combinations(method_order, 2))
n_tests = len(pairs)
p_values = {}
for m0, m1 in pairs:
    diff = df_acc[m1] - df_acc[m0]
    stat, p = wilcoxon(df_acc[m1], df_acc[m0])
    p_values[(m0, m1)] = min(p * n_tests, 1.0)
    print(f"{m1} vs {m0}: better for {(diff > 0).sum()}/{len(diff)} neurons, "
          f"worse for {(diff < 0).sum()}; median change {diff.median():.3f}; "
          f"Wilcoxon W={stat:.0f}, p={p:.2e} (Bonferroni p={p_values[(m0, m1)]:.2e}, {p_value_to_stars(p_values[(m0, m1)])})")


# ## Boxplot

# In[7]:


df_acc_melt = df_acc.melt(var_name='Algorithm option', value_name='Accuracy')

fig = px.box(df_acc_melt, x='Algorithm option', color='Algorithm option', y='Accuracy',
             category_orders={'Algorithm option': method_order}, color_discrete_map=method_colors)
fig.update_layout(boxmode='overlay')
apply_figure_settings(fig, width_factor=0.15, height_factor=0.3, plotly_not_matplotlib=True)

# Significance brackets (paired Wilcoxon, Bonferroni corrected; see above)
# Short brackets (adjacent boxes) on the lower level, the long one on top
bracket_heights = {pair: (1.17 if abs(method_order.index(pair[0]) - method_order.index(pair[1])) > 1 else 1.04)
                   for pair in p_values}
tick = 0.015
for (m0, m1), p in p_values.items():
    i0, i1 = method_order.index(m0), method_order.index(m1)
    y = bracket_heights[(m0, m1)]
    # Small gap so adjacent brackets at the same height do not touch
    x0, x1 = i0 + 0.06, i1 - 0.06
    fig.add_shape(type='path', path=f"M {x0},{y - tick} L {x0},{y} L {x1},{y} L {x1},{y - tick}",
                  line=dict(color='black', width=1), xref='x', yref='y')
    fig.add_annotation(x=(i0 + i1) / 2, y=y, text=p_value_to_stars(p), showarrow=False,
                       yanchor='bottom', yshift=-4, font=dict(size=12))

fig.update_xaxes(dict(showticklabels=False, title=""), row=1, col=1)
fig.update_yaxes(title="Fraction correctly tracked points", range=[0, 1.26], tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
fig.update_layout(showlegend=False)
fig.show()

save_plotly(fig, "pipeline_steps_scatterplot")


# ## Cumulative accuracy plot
# 
# For each accuracy threshold (x), the fraction of neurons tracked at least that accurately (y).
# A curve that lies higher/further right is better; reading off a vertical line gives "X% of neurons are tracked with >= Y accuracy".
# This replaces the per-neuron "improvement" scatterplot, which was confusing to reviewers.

# In[8]:


def plot_cumulative_accuracy(df_acc, x_min=0.5, colors=None, order=None, line_dash=None, fig=None, name_suffix=''):
    """Survival-style curve: fraction of neurons with accuracy >= threshold."""
    if fig is None:
        fig = go.Figure()
    order = order or list(df_acc.columns)
    thresholds = np.linspace(x_min, 1, 501)
    for m in order:
        vals = df_acc[m].dropna().values
        frac = [(vals >= t).mean() for t in thresholds]
        fig.add_trace(go.Scatter(x=thresholds, y=frac, mode='lines', name=m + name_suffix,
                                 line=dict(color=(colors or {}).get(m), width=2, dash=line_dash, shape='hv')))
    fig.update_xaxes(title="Accuracy threshold<br>(fraction of points correct)", range=[x_min, 1.0])
    fig.update_yaxes(title="Fraction of neurons<br>with accuracy >= threshold", range=[0, 1.02])
    return fig

fig = plot_cumulative_accuracy(df_acc, colors=method_colors, order=method_order)
apply_figure_settings(fig, width_factor=0.35, height_factor=0.3, plotly_not_matplotlib=True)
fig.update_layout(margin=dict(r=15), showlegend=True,
                  legend=dict(title='', yanchor="bottom", y=0.02, xanchor="left", x=0.02))
fig.show()

save_plotly(fig, "pipeline_steps_cumulative_accuracy")


# ## Note: other GT datasets cannot be used here
# 
# `2022-11-23_worm11` and `ZIM2165_Gcamp7b_worm1-2022-12-10` also have manual GT, but (checked Nov 2026):
# - they do not have the multi-reference-frame global tracks saved, and
# - their `initial_pipeline_tracks` are identical to the GT (full-pipeline accuracy = 1.000 for every neuron), i.e. the
#   pre-correction pipeline output was overwritten. They would need the tracking step to be rerun to be included.

# # Scratch

# ## Look at accuracy as an image

# In[9]:


neuron_names = project_data_gcamp.finished_neuron_names()
df_gt = project_data_gcamp.final_tracks[neuron_names]
df_global = project_data_gcamp.intermediate_global_tracks[neuron_names]
df_single_reference = project_data_gcamp.single_reference_frame_tracks(0)[neuron_names]
df_pipeline = project_data_gcamp.initial_pipeline_tracks[neuron_names]

def _ind(df):
    return df.loc[:, (neuron_names, 'raw_neuron_ind_in_list')].droplevel(1, axis=1)


# In[10]:


px.imshow(_ind(df_gt) == _ind(df_global), title="Correct global (multiple reference frames)")


# In[11]:


px.imshow(_ind(df_gt) == _ind(df_single_reference), title="Correct global (single reference frame)")


# In[12]:


px.imshow(_ind(df_gt) == _ind(df_pipeline), title="Correct full pipeline")


# In[13]:


px.imshow(~np.isnan(_ind(df_gt)), title="Non-nan points in the ground truth")


# ## Plot fraction tracked (gaps) vs. fraction correct

# In[14]:


opt = dict(column_names=['raw_neuron_ind_in_list'])
df_acc_pipeline = calculate_accuracy_from_dataframes(df_gt, df_pipeline, **opt)

df_pipeline_nonnan = _ind(df_pipeline).count() / df_pipeline.shape[0]
df_acc_pipeline['fraction_tracked'] = df_pipeline_nonnan
df_acc_pipeline['enough_tracked'] = df_acc_pipeline['fraction_tracked'] > 0.9
df_acc_pipeline.head()


# In[15]:


opt = dict(trendline='ols', trendline_scope="overall", trendline_color_override="black")
px.scatter(df_acc_pipeline, x='matches', y='fraction_tracked', **opt,
           color='enough_tracked', title="Better tracking implies more correct matches")


# In[16]:


px.scatter(df_acc_pipeline, x='mismatches', y='fraction_tracked', **opt,
           color='enough_tracked', title="Better tracking implies fewer mismatches")


# ## Estimate tracking and segmentation statistics across all paper datasets (slow)

# In[17]:


run_all_datasets = False
if run_all_datasets:
    from tqdm.auto import tqdm
    from wbfm.utils.general.utils_hardcoded import load_paper_datasets
    projects_to_compare = dict(gcamp=load_paper_datasets('gcamp'), gfp=load_paper_datasets('gfp'))

    rows = []
    thresholds = [0.9, 0.5, 0.1]
    for key, project_list in tqdm(projects_to_compare.items()):
        for dataset_name, p in project_list.items():
            for t in thresholds:
                try:
                    rows.append([len(p.well_tracked_neuron_names(t)), t, key])
                except AttributeError:
                    pass
    cols = ["Neurons > threshold", "Threshold for successfully tracked frames (fraction)", "genotype"]
    df_tracked = pd.DataFrame(rows, columns=cols)

    fig = px.box(df_tracked, y=cols[0], color='genotype', x=cols[1],
                 title="A majority of detected objects are well tracked")
    fig.update_layout(font=dict(size=18))
    fig.show()
    dpi, width, height = 200, 6.4, 4.8
    save_opt = dict(width=width*dpi, height=height*dpi, scale=1)
    fname = os.path.join(output_folder, 'threshold_for_tracked_objects.png')
    fig.write_image(fname, **save_opt)
    fig.write_image(str(Path(fname).with_suffix('.svg')), **save_opt)

