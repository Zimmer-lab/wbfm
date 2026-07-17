#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import matplotlib.pyplot as plt
from wbfm.utils.projects.finished_project_data import ProjectData
import napari
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from collections import defaultdict
import zarr
from pathlib import Path
import os
import seaborn as sns


# In[2]:


from sklearn.decomposition import PCA
from wbfm.utils.visualization.plot_traces import make_grid_plot_from_dataframe
import seaborn as sns
from wbfm.utils.visualization.filtering_traces import fill_nan_in_dataframe
import plotly.express as px
from wbfm.utils.general.utils_filenames import add_name_suffix


# In[4]:


# Load multiple datasets
from wbfm.utils.general.utils_hardcoded import load_paper_datasets
all_projects_gcamp = load_paper_datasets(['gcamp', 'hannah_O2_fm'])


# In[5]:


all_projects_gfp = load_paper_datasets('gfp')


# In[6]:


all_projects_immob = load_paper_datasets('immob')


# In[7]:


# Get specific example datasets
project_data_gcamp = all_projects_gcamp['ZIM2165_Gcamp7b_worm1-2022_11_28']
# project_data_immob = all_projects_immob['2022-12-13_15-16_ZIM2165_immob_worm9-2022-12-13']

# Comparing 2 datasets
project_data_gcamp2 = all_projects_gcamp['ZIM2165_Gcamp7b_worm1-2022-11-30']
project_data_immob2 = all_projects_immob['ZIM2165_immob_adj_set_2_worm2-2022-11-30']


# In[7]:


# # Same individual: fm and immob
# fname = '/lisc/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-12-06_wbfm_to_immob/2022-12-06_17-23_ZIM2165_worm5-2022-12-06/project_config.yaml'
# project_data_fm2immob_fm = ProjectData.load_final_project_data_from_config(fname, verbose=0)

# fname = '/lisc/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-12-06_wbfm_to_immob/2022-12-06_17-41_ZIM2165_immob_worm5-2022-12-06'
# project_data_fm2immob_immob = ProjectData.load_final_project_data_from_config(fname, verbose=0)

# # Same individual: fm and immob
# # fname = '/lisc/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-12-06_wbfm_to_immob/2022-12-06_17-23_ZIM2165_worm5-2022-12-06/project_config.yaml'
# # project_data_fm2immob_fm2 = ProjectData.load_final_project_data_from_config(fname, verbose=0)

# fname = '/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-12-06_wbfm_to_immob/2022-12-06_11-07_ZIM2165_immob_worm1-2022-12-06/project_config.yaml'
# project_data_fm2immob_immob2 = ProjectData.load_final_project_data_from_config(fname, verbose=0)



# In[8]:


# [str(p.project_config.self_path) for p in all_projects_gcamp.values()]


# In[8]:


# path_to_saved_data = "../step1_analysis/figure_1"
# path_to_shared_saved_data = "/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/step1_analysis/shared"


# In[10]:


# # Optional: clear just one cache
# for project_dict in tqdm([all_projects_immob]):
#     for name, project_data in tqdm(project_dict.items()):
#         project_data.data_cacher.clear_disk_cache(delete_invalid_indices=False, delete_traces=True)


# In[11]:


# # Optional: clear the trace cache
# for project_dict in tqdm([all_projects_gcamp, all_projects_gfp, 
#                           all_projects_immob]):
#     for name, project_data in tqdm(project_dict.items()):
#         project_data.data_cacher.clear_disk_cache(delete_invalid_indices=False, delete_traces=True)


# In[12]:


for project_dict in tqdm([all_projects_gcamp, all_projects_gfp, all_projects_immob]):
    for name, project_data in tqdm(project_dict.items()):
        df_traces = project_data.calc_paper_traces()
        df_res = project_data.calc_paper_traces(residual_mode='pca')
        df_global = project_data.calc_paper_traces(residual_mode='pca_global')
        if df_res is None or df_global is None or df_traces is None:
            raise ValueError


# In[13]:


# project_data_fm2immob_immob.data_cacher.clear_disk_cache(delete_invalid_indices=False, delete_traces=True)


# In[14]:


# # Also for FM to IMMOB datasets

# for project_data in [project_data_fm2immob_fm, project_data_fm2immob_immob]:
#     df_traces = project_data.calc_paper_traces()
#     df_res = project_data.calc_paper_traces(residual_mode='pca')
#     df_global = project_data.calc_paper_traces(residual_mode='pca_global')
#     if df_res is None or df_global is None or df_traces is None:
#         raise ValueError


# # Plots

# In[15]:


from wbfm.utils.visualization.plot_traces import make_summary_interactive_heatmap_with_pca, make_summary_heatmap_and_subplots


# In[16]:


# project_data_gcamp.use_physical_x_axis = True
# project_data_immob.use_physical_x_axis = True


# In[17]:


# NOT USED (combined plot)
# fig = make_summary_interactive_heatmap_with_pca(project_data_gcamp, to_save=True, to_show=True, output_folder="intro/example_summary_plots_wbfm")


# In[18]:


# Print number of neurons
project_data_gcamp.calc_paper_traces().shape


# In[19]:


# fig = make_summary_interactive_heatmap_with_pca(project_data_immob, to_save=True, to_show=True, output_folder="example_summary_plots_immob")
##del project_data_gcamp.worm_posture_class


# In[20]:


# USED: different figures for each
fig1, fig2 = make_summary_heatmap_and_subplots(project_data_gcamp, trace_opt=dict(use_paper_options=True, interpolate_nan=True), 
                                               to_save=True, to_show=True, 
                                               base_height=[0.25, 0.2], base_width=0.6, output_folder="intro/example_summary_plots_wbfm")


# In[21]:


# %debug


# In[22]:


# # Comparison: interpolated values
# fig1, fig2 = make_summary_heatmap_and_subplots(project_data_gcamp, trace_opt=dict(use_paper_options=True, interpolate_nan=True), to_save=True, to_show=True, 
#                                                output_folder="intro/example_summary_plots_wbfm")


# In[23]:


# fig1, fig2 = make_summary_heatmap_and_subplots(project_data_immob, trace_opt=dict(use_paper_options=True), include_speed_subplot=False,
#                                                to_save=True, to_show=True, output_folder="intro/example_summary_plots_immob")


# ### Just plot the legend for reversal shading

# In[35]:


from wbfm.utils.general.utils_paper import export_legend_for_paper

fname = 'intro/reversal_legend.png'
export_legend_for_paper(reversal_shading=True, fname=fname)


# In[36]:


fname = 'intro/reversal_and_collision_legend.png'
export_legend_for_paper(reversal_shading=True, fname=fname, include_self_collision=True)


# In[37]:


from wbfm.utils.general.utils_paper import export_legend_for_paper

fname = 'intro/ethogram_legend.png'
export_legend_for_paper(ethogram=True, fname=fname)


# ## PCA variance explained plot of all datasets

# In[9]:


from wbfm.utils.visualization.multiproject_wrappers import get_all_variance_explained
from wbfm.utils.visualization.utils_plot_traces import plot_with_shading
from wbfm.utils.general.utils_paper import apply_figure_settings, plotly_paper_color_discrete_map


# In[10]:


gcamp_var, gfp_var, immob_var, gcamp_var_sum, gfp_var_sum, immob_var_sum = get_all_variance_explained(all_projects_gcamp, all_projects_gfp, all_projects_immob)


# In[12]:


fig, ax = plt.subplots(dpi=200, figsize=(5,5))

var_sum_dict = {'Freely Moving (GCaMP)': gcamp_var_sum, 'Immobilized (GCaMP)': immob_var_sum, 'Freely Moving (GFP)': gfp_var_sum}
cmap = plotly_paper_color_discrete_map()

for name, mat in var_sum_dict.items():
    means = np.mean(mat, axis=1)
    color = cmap[name]
    plot_with_shading(means, np.std(mat, axis=1), label=name, ax=ax, lw=2,
                      x=np.arange(1, len(means) + 1), color=color)
# plt.legend()
# plt.title("Dimensionality")
plt.ylabel("Cumulative explained variance")
plt.ylim(0.2, 1.0)
plt.xlabel("Mode")

from matplotlib.ticker import MaxNLocator
ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))

apply_figure_settings(fig, width_factor=0.2, height_factor=0.25, plotly_not_matplotlib=False)
plt.tight_layout()

output_foldername = 'fig3'
fname = f"pca_cumulative_variance.png"
fname = os.path.join(output_foldername, fname)
plt.savefig(fname, transparent=True)
fig.savefig(fname.replace(".png", ".svg"), transparent=True)


# In[ ]:





# In[ ]:





# # PCA weights across wbfm and immob

# In[200]:


from wbfm.utils.visualization.utils_cca import calc_pca_weights_for_all_projects
from wbfm.utils.external.utils_plotly import plotly_boxplot_colored_boxes
from wbfm.utils.general.utils_paper import apply_figure_settings
from wbfm.utils.general.utils_hardcoded import neurons_with_confident_ids
from wbfm.utils.visualization.utils_plot_traces import add_p_value_annotation
from wbfm.utils.general.utils_paper import data_type_name_mapping, plotly_paper_color_discrete_map, calc_p_values_for_pca_weights
import plotly.graph_objects as go
from wbfm.utils.general.utils_paper import intrinsic_categories_color_discrete_map
from wbfm.utils.external.utils_plotly import colored_text


# In[14]:


wbfm_weights = calc_pca_weights_for_all_projects(all_projects_gcamp, use_paper_options=True, combine_left_right=True,
                                                 include_only_confident_ids=True)


# In[15]:


immob_weights = calc_pca_weights_for_all_projects(all_projects_immob, use_paper_options=True, combine_left_right=True,
                                                  include_only_confident_ids=True)


# In[222]:


df_both, df_significant_diff, df_4states_counts = calc_p_values_for_pca_weights(wbfm_weights, immob_weights,
                                                                                intrinsic_categories_fname='fig3/intrinsic_categories.xlsx')


# ## Main plot: dr/r50 traces

# In[231]:


# Plot
fig = px.box(df_both, y='PC1 weight', x='neuron_name_html', 
             color='Dataset Type', 
            color_discrete_map=plotly_paper_color_discrete_map(),
            category_orders={'Dataset Type': ['Immobilized (GCaMP)', 'Freely Moving (GCaMP)']})

add_p_value_annotation(fig, x_label='all', show_ns=False, show_only_stars=True, precalculated_p_values=df_significant_diff['p_value_corrected_diff'],
                      height_mode='top_of_data')
apply_figure_settings(fig, width_factor=0.75, height_factor=0.3, plotly_not_matplotlib=True)

fig.update_layout(legend=dict(
    yanchor="top",
    y=0.95,
    xanchor="left",
    x=0.6
))

fig.update_yaxes(dict(title="PC1 weight"), zeroline=True, zerolinewidth=1, zerolinecolor="black", )#range=[-0.2, 0.55])
fig.update_xaxes(dict(title="Neuron Name"), tickmode='linear')
fig.show()

fname = os.path.join("fig3", 'fm_and_immob_pca_weights.png')
fig.write_image(fname, scale=3)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)


# In[227]:


from wbfm.utils.general.utils_paper import intrinsic_categories_color_discrete_map

df_4states_counts = df_4states['Result'].value_counts().reset_index()
df_4states_counts['Result_simple'] = df_4states_counts['Result'].map(intrinsic_definition)
df_4states['Result_simple'] = df_4states['Result'].map(intrinsic_definition)

fig = px.pie(df_4states_counts, names='Result_simple', values='count', color='Result_simple', 
             color_discrete_map=intrinsic_categories_color_discrete_map())
apply_figure_settings(fig, width_factor=0.2, height_factor=0.3)
fig.update_layout(legend=dict(
    yanchor="top",
    y=-0.1,
    xanchor="left",
    x=0.2
))
fig.update_traces(texttemplate='%{percent:.2p}')
fig.show()

output_foldername = 'fig3'
fname = os.path.join(output_foldername, 'manifold_participation_pie_chart.png')
fig.write_image(fname, scale=3)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)


# In[163]:


df_4states['Result_simple'].value_counts()


# ## Supp figure: dr/r20 traces

# In[210]:


wbfm_weights_r20 = calc_pca_weights_for_all_projects(all_projects_gcamp, use_paper_options=True, combine_left_right=True,
                                                 include_only_confident_ids=True, channel_mode='dr_over_r_20')


# In[211]:


immob_weights_r20 = calc_pca_weights_for_all_projects(all_projects_immob, use_paper_options=True, combine_left_right=True,
                                                  include_only_confident_ids=True, channel_mode='dr_over_r_20')


# In[219]:


df_both_r20, df_significant_diff_r20, df_4states_counts_r20 = calc_p_values_for_pca_weights(wbfm_weights_r20, immob_weights_r20,
                                                                                intrinsic_categories_fname='fig3/intrinsic_categories_r20.xlsx')


# In[230]:


# Plot
fig = px.box(df_both_r20, y='PC1 weight', x='neuron_name_html', 
             color='Dataset Type', 
            color_discrete_map=plotly_paper_color_discrete_map(),
            category_orders={'Dataset Type': ['Immobilized (GCaMP)', 'Freely Moving (GCaMP)']})

add_p_value_annotation(fig, x_label='all', show_ns=False, show_only_stars=True, precalculated_p_values=df_significant_diff_r20['p_value_corrected_diff'],
                      height_mode='top_of_data')
apply_figure_settings(fig, width_factor=0.75, height_factor=0.3, plotly_not_matplotlib=True)

fig.update_layout(legend=dict(
    yanchor="top",
    y=0.95,
    xanchor="left",
    x=0.6
))

fig.update_yaxes(dict(title="PC1 weight"), zeroline=True, zerolinewidth=1, zerolinecolor="black", )#range=[-0.2, 0.55])
fig.update_xaxes(dict(title="Neuron Name"), tickmode='linear')
fig.show()

fname = os.path.join("fig3", 'fm_and_immob_pca_weights_r20.png')
fig.write_image(fname, scale=3)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)


# In[228]:


from wbfm.utils.general.utils_paper import intrinsic_categories_color_discrete_map

fig = px.pie(df_4states_counts_r20, names='Result_simple', values='count', color='Result_simple', 
             color_discrete_map=intrinsic_categories_color_discrete_map())
apply_figure_settings(fig, width_factor=0.2, height_factor=0.3)
fig.update_layout(legend=dict(
    yanchor="top",
    y=-0.1,
    xanchor="left",
    x=0.2
))
fig.update_traces(texttemplate='%{percent:.2p}')
fig.show()

output_foldername = 'fig3'
fname = os.path.join(output_foldername, 'manifold_participation_pie_chart_r20.png')
fig.write_image(fname, scale=3)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Variance explained by mode 1 across neurons
# 
# i.e. the cumulative histogram, with error bar per dataset

# In[139]:


from wbfm.utils.visualization.multiproject_wrappers import build_dataframe_of_variance_explained
from wbfm.utils.general.utils_paper import plotly_paper_color_discrete_map
from wbfm.utils.general.utils_paper import apply_figure_settings


# In[ ]:


trace_opt = dict(use_paper_options=True, interpolate_nan=True)

all_dfs = []
for n in tqdm([1, 2]):
    opt = dict(n_components=n, melt=True)

    df_var_exp_gcamp = build_dataframe_of_variance_explained(all_projects_gcamp, **opt, **trace_opt)
    df_var_exp_gcamp['Type of data'] = 'gcamp'
    df_var_exp_gcamp['n_components'] = n

    df_var_exp_immob = build_dataframe_of_variance_explained(all_projects_immob, **opt, **trace_opt)
    df_var_exp_immob['Type of data'] = 'immob'
    df_var_exp_immob['n_components'] = n

    df_var_exp_gfp = build_dataframe_of_variance_explained(all_projects_gfp, **opt, **trace_opt)
    df_var_exp_gfp['Type of data'] = 'gfp'
    df_var_exp_gfp['n_components'] = n
    
    all_dfs.extend([df_var_exp_gcamp, df_var_exp_immob, df_var_exp_gfp])

df_var_exp = pd.concat(all_dfs, axis=0)
df_var_exp.head()


# In[ ]:


df_var_exp[(df_var_exp['dataset_name'] == '2022-11-23_worm10') & (df_var_exp['neuron_name'] == 'ALA')]


# In[ ]:


# px.histogram(df_var_exp, color='dataset_name', x='fraction_variance_explained', cumulative=True, 
#              facet_row='Type of data',
#              barmode='overlay', histnorm='percent')


# In[ ]:


df_var_exp_hist = df_var_exp.copy()

# Get counts of neurons in each bin
bins = np.linspace(0, 1, 50)
func = lambda Z: np.cumsum(np.histogram(Z, bins=bins)[0])
df_var_exp_hist = df_var_exp_hist.groupby(['dataset_name', 'n_components'])['fraction_variance_explained'].apply(func)
# df_var_exp_hist.head()

# Explode to long form
long_vars = df_var_exp_hist.reset_index().explode('fraction_variance_explained')
long_vars.rename(columns={'fraction_variance_explained': 'cumulative_fraction_variance_explained'}, inplace=True)
long_vars.sort_values(by=['dataset_name', 'cumulative_fraction_variance_explained'], inplace=True)
# Just remake the bins
long_vars['cumcount'] = long_vars.groupby(['dataset_name', 'n_components']).cumcount()
long_vars['fraction_count'] = long_vars['cumcount'] / long_vars['cumcount'].max()

# Add back datatype column
long_vars = long_vars.merge(df_var_exp[['dataset_name', 'n_components', 'Type of data']], on=['dataset_name', 'n_components'])

# Normalize by number of total neurons (Only take one component to avoid duplication)
total_num_neurons = df_var_exp[df_var_exp['n_components']==1].dropna()['dataset_name'].value_counts()
long_vars.index = long_vars['dataset_name']  # So the division matches
long_vars['cumulative_fraction_variance_explained'] = long_vars['cumulative_fraction_variance_explained'] / total_num_neurons
long_vars.reset_index(drop=True, inplace=True)

long_vars.head()


# In[ ]:


# px.line(long_vars, x='fraction_count', 
#         y='cumulative_fraction_variance_explained', color='dataset_name',
#        facet_row='Type of data')


# In[ ]:


from wbfm.utils.external.utils_plotly import plotly_plot_mean_and_shading

opt = dict(x='fraction_count', y='cumulative_fraction_variance_explained', color='n_components', 
           cmap=plotly_paper_color_discrete_map()
          )

fig = None
g = 'gcamp'
df_subset = long_vars[(long_vars['Type of data']==g)]
fig = plotly_plot_mean_and_shading(df_subset, line_name=g, fig=fig, **opt,
                                   x_intersection_annotation=0.5)
# for n in [1, 2]:
#     df_subset = long_vars[(long_vars['Type of data']==g)]# & (long_vars['n_components']==n)]
#     fig = plotly_plot_mean_and_shading(df_subset, line_name=g, fig=fig, **opt,
#                                        x_intersection_annotation=0.5)

fig.update_xaxes(title='Var. explained (fraction)', range=[0, 1.05])
fig.update_yaxes(title='Fraction of neurons <br> (cumulative)', range=[0, 1.05])
fig.update_layout(
        showlegend=True,
        legend=dict(
            title='Mode',
          yanchor="middle",
          y=0.25,
          xanchor="left",
          x=0.6
        )
    )# fig.update_traces(line=dict(color=plotly_paper_color_discrete_map()['PCA']))
fig.update_traces(name='1 + 2', selector=dict(name='2'))

apply_figure_settings(fig, width_factor=0.3, height_factor=0.2)

fig.show()

to_save = True
if to_save:
    output_foldername = 'intro/dimensionality'
    fname = os.path.join(output_foldername, 'variance_explained_by_pc1_and_pc2_cumulative.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)


# In[ ]:


# %debug


# In[ ]:


long_vars['Type of data'].unique()


# In[ ]:


from wbfm.utils.external.utils_plotly import plotly_plot_mean_and_shading
# Same as above, but with FM and immob, not only FM

opt = dict(x='fraction_count', y='cumulative_fraction_variance_explained', color='Type of data', 
           cmap=plotly_paper_color_discrete_map()
          )
# fig = None
# g = 'gcamp'
# df_subset = long_vars[(long_vars['Type of data']==g)]
# fig = plotly_plot_mean_and_shading(df_subset, line_name=g, fig=fig, **opt,
#                                    x_intersection_annotation=0.5)


# Add immob
n_components = 2
fig = None
fig = plotly_plot_mean_and_shading(long_vars[long_vars['n_components']==n_components], line_name='immob', fig=fig, **opt,
                                  x_intersection_annotation=0.5, annotation_position='right')

# Add gfp
n_components = 2
fig = None
fig = plotly_plot_mean_and_shading(long_vars[long_vars['n_components']==n_components], line_name='gfp', fig=fig, **opt,
                                  x_intersection_annotation=0.5, annotation_position='right')



fig.update_xaxes(title='Var. explained by modes 1 and 2 (fraction)', range=[0, 1.05])
fig.update_yaxes(title='Fraction of neurons (cumulative)', range=[0, 1.05])
fig.update_traces(name='Immobilized', selector=dict(name='immob'))
fig.update_traces(name='Freely Moving (GFP)', selector=dict(name='gfp'))
fig.update_traces(name='Freely Moving (GCaMP)', selector=dict(name='gcamp'))
fig.update_layout(
        showlegend=True,
        legend=dict(
          title='Datatype',
          yanchor="middle",
          y=0.15,
          xanchor="left",
          x=0.5
        )
    )
# In the supp, so it's larger
apply_figure_settings(fig, width_factor=0.5, height_factor=0.4)


fig.show()

output_foldername = 'intro/dimensionality'
fname = os.path.join(output_foldername, 'variance_explained_by_pc1_cumulative_with_immob.png')
fig.write_image(fname, scale=3)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)


# In[74]:


# %debug


# # Dimensionality of single neurons

# In[232]:


from wbfm.utils.visualization.utils_dimensionality import calculate_dimensionality_of_single_neurons
from wbfm.utils.general.utils_hardcoded import get_hierarchical_modeling_dir


# In[181]:


parent_folder = get_hierarchical_modeling_dir(immobilized=True)
fname = 'data.h5'
df_immob = pd.read_hdf(os.path.join(parent_folder, fname))


# In[182]:


parent_folder


# In[184]:


# [print(c) for c in df_immob.columns]


# In[185]:


df_immob['AVAL']


# In[186]:


df_immob['AVAL_manifold']


# In[236]:


calculate_dimensionality_of_single_neurons(combine_left_right=False)


# In[235]:


calculate_dimensionality_of_single_neurons(combine_left_right=True)


# # Scratch

# ## Where is DD01?

# In[75]:


'DD01' in wbfm_weights, 'DD01' in immob_weights


# In[ ]:





# In[ ]:




