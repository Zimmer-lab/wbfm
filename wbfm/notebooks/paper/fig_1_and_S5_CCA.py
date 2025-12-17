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
import plotly.express as px
from wbfm.utils.visualization.utils_plot_traces import add_p_value_annotation


# In[3]:


# fname = "/scratch/neurobiology/zimmer/Charles/dlc_stacks/2022-11-27_spacer_7b_2per_agar/ZIM2165_Gcamp7b_worm1-2022_11_28/project_config.yaml"
# Manually corrected version
fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/manually_annotated/paper_data/ZIM2165_Gcamp7b_worm1-2022_11_28/project_config.yaml"
project_data_gcamp = ProjectData.load_final_project_data_from_config(fname)


# In[4]:


from wbfm.utils.general.hardcoded_paths import load_paper_datasets
all_projects_gcamp = load_paper_datasets(['gcamp', 'hannah_O2_fm'])


# # Do PCA, CCA on real behaviors, and CCA on binarized behaviors

# In[5]:


output_folder = 'cca'


# In[6]:


from wbfm.utils.visualization.utils_cca import CCAPlotter


# In[7]:


project_data_gcamp.use_physical_x_axis = True

cca_plotter = CCAPlotter(project_data_gcamp, truncate_traces_to_n_components=5, preprocess_behavior_using_pca=True, trace_kwargs=dict(use_paper_options=True))


# In[8]:


# fig = cca_plotter.plot(output_folder=output_folder, show_legend=False)


# In[9]:


# For fixing cutoff labels
# camera = dict(
#     eye=dict(x=2, y=2, z=2))
# fig.update_layout(scene_camera=camera)
# fig.show()


# In[10]:


# fig = cca_plotter.plot(binary_behaviors=True, output_folder=output_folder, show_legend=False)#, beh_annotation_kwargs=dict(include_collision=True))


# In[11]:


# fig = cca_plotter.plot(binary_behaviors=False, use_pca=True, output_folder=output_folder, show_legend=False)


# ## Also plot correlation matrix for behaviors

# In[12]:


# df_beh = cca_plotter._df_beh
# fig = px.imshow(df_beh.corr(), width=1000, height=1000)
# fig.show()


# In[ ]:





# ## Example dataset modes in 2d

# In[13]:


fig = cca_plotter.plot(plot_3d=False, output_folder=output_folder, show_legend=False)


# In[14]:


fig = cca_plotter.plot(plot_3d=False, binary_behaviors=True, show_legend=False, output_folder=output_folder)#, beh_annotation_kwargs=dict(include_collision=True))


# In[15]:


fig = cca_plotter.plot(plot_3d=False, binary_behaviors=False, show_legend=False, use_pca=True, output_folder=output_folder)


# In[ ]:





# In[ ]:





# In[ ]:





# ## Example traces: Top mode

# In[16]:


# fig = cca_plotter.plot_single_mode(output_folder=output_folder, show_legend=False)


# In[17]:


# fig = cca_plotter.plot_single_mode(binary_behaviors=True, output_folder=output_folder, show_legend=False)#, beh_annotation_kwargs=dict(include_collision=True))


# In[18]:


# fig = cca_plotter.plot_single_mode(binary_behaviors=False, use_pca=True, output_folder=output_folder, show_legend=False)


# In[ ]:





# In[ ]:





# # Calculate variance explained per dataset

# In[7]:


from wbfm.utils.visualization.utils_cca import calc_r_squared_for_all_projects
from wbfm.utils.general.utils_paper import plotly_paper_color_discrete_map
from wbfm.utils.general.utils_paper import apply_figure_settings


# In[8]:


all_cca_classes, df_r_squared_melt, r_squared_per_row = calc_r_squared_for_all_projects(all_projects_gcamp, r_squared_kwargs=dict(n_components=[1, 2, 3]), 
                                                                preprocess_traces_using_pca=True, truncate_traces_to_n_components=5, trace_kwargs=dict(use_paper_options=True),
                                                               melt=True)


# In[9]:


# %debug


# In[10]:


df_r_squared_melt.head()


# In[11]:


# df_r_squared_melt = df_r_squared.melt(var_name='Model Type', value_name='Variance explained')
# df_r_squared_melt


# In[13]:


from wbfm.utils.external.utils_plotly import plotly_plot_mean_and_shading
# fig = px.box(df_r_squared_melt, x='Model Type', y='$R^2$', 
#              color='Model Type', color_discrete_map=plotly_paper_color_discrete_map())#, title="Reconstruction quality for single modes")
fig = px.box(df_r_squared_melt, x='n_components', color='Method', y='Variance Explained', 
             color_discrete_map=plotly_paper_color_discrete_map())#, title="Reconstruction quality for single modes")

apply_figure_settings(fig, width_factor=0.4, height_factor=0.2)
fig.update_yaxes(title='Neuronal Variance<br>Explained (cumulative)', 
                 range=[0, 1.1])#, showgrid=True, overwrite=True)
fig.update_xaxes(title='Number of components')
fig.update_layout(showlegend=True)

to_save = False
if to_save:
    fname = os.path.join(output_folder, 'top_mode_reconstruction_boxplot.png')
    fig.write_image(fname, scale=3)
    fname = fname.replace('.png', '.svg')
    fig.write_image(fname)

fig.show()


# In[25]:


# fig = paired_boxplot_from_dataframes(df_r_squared.T, num_rows=3, add_median_line=False)
# plt.ylabel("R2")
# plt.title("Reconstruction (mode 1)")

# apply_figure_settings(fig, width_factor=0.3, height_factor=0.2, plotly_not_matplotlib=False)

# fname = os.path.join(output_folder, 'paired_boxplot.png')
# plt.savefig(fname, transparent=True)
# fname = Path(fname).with_suffix('.svg')
# plt.savefig(fname)


# ## Test: use direct var explained from pca code

# In[23]:


# opt = dict(filter_mode='rolling_mean', interpolate_nan=True, nan_tracking_failure_points=True,
#                    use_physical_time=True, rename_neurons_using_manual_ids=True, use_paper_options=True)


# all_var_explained = {}
# for name, project in tqdm(all_projects_gcamp.items()):
#     _, exp_var_ratio = project.calc_pca_modes(2, **opt)
#     all_var_explained[name] = sum(exp_var_ratio)
    


# In[22]:


# px.box(pd.DataFrame(all_var_explained, index=[0]).T)


# ## Variance explained per neuron (cumulative plot)

# In[26]:


from wbfm.utils.visualization.utils_cca import calc_r_squared_for_all_projects
from wbfm.utils.general.utils_paper import plotly_paper_color_discrete_map
from wbfm.utils.general.utils_paper import apply_figure_settings


# In[27]:


# _, df_r_squared, r_squared_per_row = calc_r_squared_for_all_projects(all_projects_gcamp, r_squared_kwargs=dict(n_components=[1, 2, 3]), 
#                                                                 preprocess_traces_using_pca=True, truncate_traces_to_n_components=5, trace_kwargs=dict(use_paper_options=True),
#                                                                melt=True)


# In[28]:


_df = r_squared_per_row.rename(columns={'Behavior Variable': 'Neuron Name'})[r_squared_per_row['Components'] == 2]
_df


# In[29]:


df_var_exp = _df.copy()
df_var_exp_hist = _df.copy()

# Get counts of neurons in each bin
bins = np.linspace(0, 1, 51)
func = lambda Z: np.cumsum(np.histogram(Z, bins=bins)[0])
df_var_exp_hist = df_var_exp_hist.groupby(['Dataset Name', 'Method'])['Cumulative Variance explained'].apply(func)

# Explode to long form
long_vars = (df_var_exp_hist / df_var_exp_hist.apply(max)).reset_index().explode('Cumulative Variance explained').reset_index(drop=True)

# Just remake the bins
fraction_count = df_var_exp_hist.apply(lambda x: bins[1:]).reset_index().explode('Cumulative Variance explained').reset_index(drop=True).rename(columns={'Cumulative Variance explained': 'bins'})
long_vars['bins'] = fraction_count['bins']

long_vars


# In[30]:


from wbfm.utils.external.utils_plotly import plotly_plot_mean_and_shading

opt = dict(x='bins', y='Cumulative Variance explained', color='Method', 
           cmap=plotly_paper_color_discrete_map()
          )

fig = None
g = ['CCA', 'CCA Discrete']
df_subset = long_vars[(long_vars['Method'].isin(g))]
fig = plotly_plot_mean_and_shading(df_subset, fig=fig, **opt,
                                   x_intersection_annotation=0.5)

fig.update_xaxes(title='Var. explained (fraction)', range=[0, 1.05])
fig.update_yaxes(title='Fraction of neurons <br> (cumulative)', range=[0, 1.05])
fig.update_layout(
        showlegend=False,
        legend=dict(
            title='Mode',
          yanchor="middle",
          y=0.25,
          xanchor="left",
          x=0.6
        )
    )# fig.update_traces(line=dict(color=plotly_paper_color_discrete_map()['PCA']))
# fig.update_traces(name='1 + 2', selector=dict(name='2'))

apply_figure_settings(fig, width_factor=0.3, height_factor=0.2)

fig.show()

to_save = True
if to_save:
    output_foldername = 'intro/dimensionality'
    fname = os.path.join(output_foldername, 'variance_explained_by_cca_cumulative.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)


# In[31]:


plotly_paper_color_discrete_map()['PCA']


# In[32]:


long_vars.groupby(['Dataset Name', 'Method']).apply(np.cumsum)


# In[33]:


df_var_exp_hist


# In[34]:


df_var_exp_hist.reset_index().explode('Cumulative Variance explained').groupby(['Dataset Name', 'Method']).cumcount()


# In[35]:


df_var_exp_hist.reset_index().explode('Cumulative Variance explained')


# In[ ]:





# In[ ]:





# In[ ]:





# ## Also calculate variance explained of behavior time series

# In[36]:


all_cca_classes_beh, df_r_squared_melt_beh, all_r_squared_per_row_beh = calc_r_squared_for_all_projects(all_projects_gcamp, 
                                                                                                        r_squared_kwargs=dict(n_components=[1, 2, 3], use_behavior=True, DEBUG=False), 
                                                                preprocess_traces_using_pca=True, truncate_traces_to_n_components=5, trace_kwargs=dict(use_paper_options=True),
                                                               melt=True)


# In[37]:


df_r_squared_melt_beh.head()


# In[38]:


# all_r_squared_per_row_beh


# In[39]:


# from wbfm.utils.external.utils_plotly import plotly_plot_mean_and_shading
# # fig = px.box(df_r_squared_melt, x='Model Type', y='$R^2$', 
# #              color='Model Type', color_discrete_map=plotly_paper_color_discrete_map())#, title="Reconstruction quality for single modes")
# fig = px.box(df_r_squared_melt, x='n_components', color='Method', y='Variance Explained', 
#              color_discrete_map=plotly_paper_color_discrete_map())#, title="Reconstruction quality for single modes")

# apply_figure_settings(fig, width_factor=0.4, height_factor=0.2)
# fig.update_yaxes(title='Variance Explained<br>(cumulative)', 
#                  range=[0, 1.1])#, showgrid=True, overwrite=True)
# fig.update_xaxes(title='Number of components')
# fig.update_layout(showlegend=True)

# to_save = False
# if to_save:
#     fname = os.path.join(output_folder, 'top_mode_reconstruction_boxplot_behavior.png')
#     fig.write_image(fname, scale=3)
#     fname = fname.replace('.png', '.svg')
#     fig.write_image(fname)

# fig.show()


# In[40]:


# from wbfm.utils.general.utils_paper import behavior_name_mapping
# # Also plot the variance explained per row
# data = all_r_squared_per_row_beh

# # Step 1: Flatten the nested dictionary
# flat_data = []
# for outer_key, level1_dict in data.items():
#     for level1_key, level2_dict in level1_dict.items():
#         for level2_key, level3_dict in level2_dict.items():
#             for level3_key, value in level3_dict.items():
#                 flat_data.append({
#                     'Components': outer_key,
#                     'Dataset Name': level1_key,
#                     'Method': level2_key,
#                     'Behavior Variable': behavior_name_mapping(shorten=True)[level3_key],
#                     'Cumulative Variance explained': value
#                 })

# # Step 2: Convert to a DataFrame
# df = pd.DataFrame(flat_data)
# df


# In[41]:


# fig = px.box(df[df['Method'] != 'PCA'], y='Cumulative Variance explained', color='Components', x='Behavior Variable', facet_col='Method')
# apply_figure_settings(fig, width_factor=1, height_factor=0.4)

# fig.show()


# In[42]:


df = all_r_squared_per_row_beh

cmap = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
cmap.pop(2)  # Remove the horrible neon green

for method in df['Method'].unique():
    _df = df[df['Method']==method]
    fig = px.box(_df, y='Cumulative Variance explained', color='Components', x='Behavior Variable', facet_col='Method', color_discrete_sequence=cmap)
    show_legend = (method=='CCA')
    fig.update_layout(showlegend=show_legend)
    if show_legend:  
        fig.update_layout(
            legend=dict(
              yanchor="middle",
              y=0.25,
              xanchor="left",
              x=1.01
            )
        )
    else:
        fig.update_yaxes(title='')
    # fig.update_traces(boxpoints=False)
    fig.update_xaxes(title=f"Method: {method}")
    apply_figure_settings(fig, width_factor=0.55 if show_legend else 0.45, height_factor=0.3)
    
    to_save = True
    if to_save:
        fname = os.path.join(output_folder, f'behavior_variance_explained_{method}.png')
        fig.write_image(fname, scale=3)
        fname = fname.replace('.png', '.svg')
        fig.write_image(fname)

    fig.show()


# ## Also do ttests

# In[43]:


# df_r_squared.head()


# In[44]:


# from scipy.stats import ttest_ind
# from itertools import combinations


# In[45]:


# cols = list(df_r_squared.columns)
# col_pairs = combinations(cols, 2)

# for col_pair in col_pairs:
#     a = df_r_squared.loc[:, col_pair[0]]
#     b = df_r_squared.loc[:, col_pair[1]]

#     result = ttest_ind(a, b, equal_var=False)
#     print(f"Ttest for correlations of columns {col_pair}: {result.pvalue}")


# In[ ]:





# # Calculate dot product between the pca and cca modes

# In[46]:


all_dots = {i+1: {name: c.calc_mode_dot_product(i) for name, c in tqdm(all_cca_classes.items())} for i in range(3)}
df_all_dots = pd.DataFrame(all_dots).melt(var_name='Component', value_name='PCA-CCA similarity')
df_all_dots['Comparison Method'] = 'CCA'

all_dots_discrete = {i+1: {name: c.calc_mode_dot_product(i, binary_behaviors=True) for name, c in tqdm(all_cca_classes.items())} for i in range(3)}
df_all_dots_discrete = pd.DataFrame(all_dots_discrete).melt(var_name='Component', value_name='PCA-CCA similarity')
df_all_dots_discrete['Comparison Method'] = 'CCA Discrete'

df_all_dots = pd.concat([df_all_dots, df_all_dots_discrete])


# In[47]:


df_all_dots['PCA-CCA similarity'] = df_all_dots['PCA-CCA similarity'].abs()


# In[48]:


fig = px.box(df_all_dots, x='Component', y='PCA-CCA similarity', color='Comparison Method',
            color_discrete_map=plotly_paper_color_discrete_map())
# fig.update_traces(marker=dict(color=plotly_paper_color_discrete_map()['PCA']))
# fig.update_xaxes(title='Component')
fig.update_yaxes(title='PCA-CCA similarity', range=[0, 1.1])
fig.update_layout(showlegend=False)

apply_figure_settings(fig, width_factor=0.2, height_factor=0.2)

fig.show()

to_save = True
if to_save:
    fname = os.path.join(output_folder, 'mode_dot_product.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)


# # Supp: Look at the correlation of the modes across datasets

# In[49]:


from wbfm.utils.visualization.utils_cca import calc_mode_correlation_for_all_projects
from wbfm.utils.external.utils_matplotlib import paired_boxplot_from_dataframes
from wbfm.utils.general.utils_paper import apply_figure_settings


# In[50]:


output_folder = 'cca'


# In[51]:


all_cca_classes3, df_mode_correlations, df_mode_correlations_binary = calc_mode_correlation_for_all_projects(all_projects_gcamp, correlation_kwargs=dict(n_components=5),
                                                                                                             preprocess_traces_using_pca=True, truncate_traces_to_n_components=6, 
                                                                                                             trace_kwargs=dict(use_paper_options=True))


# In[52]:


df_mode_correlations.index = np.arange(1, 6)
df_mode_correlations_binary.index = np.arange(1, 6)


# In[53]:


# fig = paired_boxplot_from_dataframes(df_mode_correlations, num_rows=5, add_median_line=False)
# plt.ylabel("Correlation")
# plt.xlabel("CCA mode index")
# plt.title("CCA")

# apply_figure_settings(fig, width_factor=0.3, height_factor=0.2, plotly_not_matplotlib=False)

# fname = os.path.join(output_folder, 'paired_boxplot_latent_space.png')
# plt.savefig(fname, transparent=True)
# fname = Path(fname).with_suffix('.svg')
# plt.savefig(fname)


# In[54]:


# fig = paired_boxplot_from_dataframes(df_mode_correlations_binary, num_rows=5, add_median_line=False)
# plt.ylabel("Correlation")
# plt.xlabel("CCA mode index")
# plt.title("CCA (binary)")

# apply_figure_settings(fig, width_factor=0.3, height_factor=0.2, plotly_not_matplotlib=False)

# fname = os.path.join(output_folder, 'paired_boxplot_latent_space_binary.png')
# plt.savefig(fname, transparent=True)
# fname = Path(fname).with_suffix('.svg')
# plt.savefig(fname)


# In[55]:


df0 = df_mode_correlations.T.copy()
df1 = df_mode_correlations_binary.T.copy()

df0['Behavior type'] = 'Continuous'
df1['Behavior type'] = 'Discrete'
df_mode_combined = pd.concat([df0, df1])


# In[56]:


fig = px.box(df_mode_combined.drop(columns=[4, 5]), color='Behavior type',
             color_discrete_map=plotly_paper_color_discrete_map())

fig.update_xaxes(title="Component")
fig.update_yaxes(title="Correlation of <br> CCA latent spaces")
fig.update_layout(showlegend=False)

apply_figure_settings(fig, width_factor=0.2, height_factor=0.2)

# fig.update_layout(
#     showlegend=True,
#     legend=dict(
#       yanchor="top",
#       y=1,
#       xanchor="right",
#       x=1.0
#     )
# )

fig.show()


fname = os.path.join(output_folder, 'paired_boxplot_latent_space_combined.png')
fig.write_image(fname, scale=7)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)


# In[90]:


fig = px.box(df_mode_combined, color='Behavior type',
             color_discrete_map=plotly_paper_color_discrete_map())

fig.update_xaxes(title="Component")
fig.update_yaxes(title="Correlation of <br> CCA latent spaces")
fig.update_layout(showlegend=False)

apply_figure_settings(fig, width_factor=0.5, height_factor=0.3)

# fig.update_layout(
#     showlegend=True,
#     legend=dict(
#       yanchor="top",
#       y=1,
#       xanchor="right",
#       x=1.0
#     )
# )

fig.show()


fname = os.path.join(output_folder, 'paired_boxplot_latent_space_combined12345.png')
fig.write_image(fname, scale=7)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)


# ## Also do t-tests

# In[57]:


from scipy.stats import ttest_ind


# In[58]:


df_mode_correlations


# In[59]:


# a = df_mode_correlations.loc[1, :]
# b = df_mode_correlations_binary.loc[1, :]

# result = ttest_ind(a, b, equal_var=False)
# print(f"Ttest for correlations of mode 0: {result.pvalue}")


# In[60]:


# a = df_mode_correlations.loc[2, :]
# b = df_mode_correlations_binary.loc[2, :]

# result = ttest_ind(a, b, equal_var=False)
# print(f"Ttest for correlations of mode 1: {result.pvalue}")


# # Get the neural and behavioral weights across datasets

# In[61]:


from wbfm.utils.visualization.utils_cca import calc_cca_weights_for_all_projects
import plotly.express as px
from wbfm.utils.general.utils_paper import apply_figure_settings, plotly_paper_color_discrete_map
from wbfm.utils.general.hardcoded_paths import neurons_with_confident_ids
from wbfm.utils.visualization.utils_plot_traces import add_p_value_annotation
output_folder = 'cca'


# In[62]:


all_cca_classes1, df_weights1, df_weights_binary1 = calc_cca_weights_for_all_projects(all_projects_gcamp, which_mode=0, min_datasets_present=6,
                                                                                       weights_kwargs=dict(n_components=2),
                                                                                                             preprocess_traces_using_pca=True, truncate_traces_to_n_components=5,
                                                                                                            preprocess_behavior_using_pca=True,
                                                                                    combine_left_and_right=True,
                                                                                   trace_kwargs=dict(use_paper_options=True))


# In[63]:


# df_weights1 = df_weights1[[c for c in df_weights1.columns if c in neurons_with_confident_ids(combine_left_right=True)]]

# fig = px.box(df_weights1)#, title="CCA weights of mode 1 across recordings")

# apply_figure_settings(fig, width_factor=0.75, height_factor=0.2, plotly_not_matplotlib=True)
# # fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="Weight <br> (mode 1)")
# fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="Weight")
# fig.update_xaxes(title="")
# fig.show()

# to_save = True
# if to_save:
#     fname = os.path.join(output_folder, 'paired_boxplot_neural_weights.png')
#     fig.write_image(fname, scale=3)
#     fname = Path(fname).with_suffix('.svg')
#     fig.write_image(fname)


# In[64]:


# df_weights_binary1 = df_weights_binary1[[c for c in df_weights_binary1.columns if c in neurons_with_confident_ids(combine_left_right=True)]]

# fig = px.box(df_weights_binary1)#, title="CCA weights of mode 1 across recordings (binary)")

# apply_figure_settings(fig, width_factor=0.75, height_factor=0.2, plotly_not_matplotlib=True)
# fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="Weight <br> (discrete <br> mode 1)")
# fig.update_xaxes(title="")
# fig.show()

# fname = os.path.join(output_folder, 'paired_boxplot_neural_weights_binary.png')
# fig.write_image(fname, scale=3)
# fname = Path(fname).with_suffix('.svg')
# fig.write_image(fname)


# In[65]:


# Both modes together
df_both1 = df_weights1.reset_index().melt(id_vars='index')
df_both1['Behavior Type'] = 'Continuous'
df_both1_binary = df_weights_binary1.reset_index().melt(id_vars='index')
df_both1_binary['Behavior Type'] = 'Discrete'
df_both1 = pd.concat([df_both1, df_both1_binary])
df_both1.columns = ['Dataset Name', 'Neuron', 'Weight', 'Behavior Type']
# df_both1


# In[66]:


from wbfm.utils.general.hardcoded_paths import neurons_with_confident_ids
neurons_to_plot = neurons_with_confident_ids(combine_left_right=True)
fig = px.box(df_both1[df_both1['Neuron'].isin(neurons_to_plot)], x='Neuron', y='Weight', color='Behavior Type',
             hover_data=['Dataset Name'],
             color_discrete_map=plotly_paper_color_discrete_map())

apply_figure_settings(fig, width_factor=1.0, height_factor=0.2, plotly_not_matplotlib=True)
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="CCA Weight<br>(Component 1)")
# fig.update_xaxes(title="", tickfont_size=12)

fig.update_layout(legend=dict(
    yanchor="top",
    y=1.05,
    xanchor="left",
    x=0.8
))
fig.show()

fname = os.path.join(output_folder, 'paired_boxplot_neural_weights_BOTH.png')
fig.write_image(fname, scale=3)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)


# In[67]:


add_p_value_annotation(fig, x_label='all', show_only_stars=True)
fig.show()


# ## SUPP: Same but for mode 2

# In[68]:


all_cca_classes2, df_weights2, df_weights_binary2 = calc_cca_weights_for_all_projects(all_projects_gcamp, which_mode=1, min_datasets_present=6,
                                                                                       weights_kwargs=dict(n_components=3),
                                                                                    correct_sign_using_top_weight=True,
                                                                                                             preprocess_traces_using_pca=True, truncate_traces_to_n_components=5,
                                                                                                            preprocess_behavior_using_pca=True,
                                                                                    combine_left_and_right=True,
                                                                                   trace_kwargs=dict(use_paper_options=True))


# In[69]:


df_weights2 = df_weights2[[c for c in df_weights2.columns if c in neurons_with_confident_ids(combine_left_right=True)]]

# fig = px.box(df_weights2, color_discrete_sequence=[plotly_paper_color_discrete_map()['CCA']])

# apply_figure_settings(fig, width_factor=1.0, height_factor=0.2, plotly_not_matplotlib=True)
# fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="Weight <br> (mode 2)")
# fig.update_xaxes(title="")
# fig.show()

# to_save = True
# if to_save:
#     fname = os.path.join(output_folder, 'paired_boxplot_neural_weights2.png')
#     fig.write_image(fname, scale=3)
#     fname = Path(fname).with_suffix('.svg')
#     fig.write_image(fname)


# In[70]:


df_weights_binary2 = df_weights_binary2[[c for c in df_weights_binary2.columns if c in neurons_with_confident_ids(combine_left_right=True)]]

# fig = px.box(df_weights_binary2, color_discrete_sequence=[plotly_paper_color_discrete_map()['Discrete']])#, title="CCA weights of mode 2 across recordings (binary)")

# apply_figure_settings(fig, width_factor=1.0, height_factor=0.2, plotly_not_matplotlib=True)
# fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="Weight <br> (discrete <br> mode 2)")
# fig.update_xaxes(title="")
# fig.show()

# fname = os.path.join(output_folder, 'paired_boxplot_neural_weights_binary2.png')
# fig.write_image(fname, scale=3)
# fname = Path(fname).with_suffix('.svg')
# fig.write_image(fname)


# In[71]:


from wbfm.utils.general.hardcoded_paths import neurons_with_confident_ids
neurons_to_plot = neurons_with_confident_ids(combine_left_right=True)

# Both modes together
df_both2 = df_weights2.reset_index().melt(id_vars='index')
df_both2['Behavior Type'] = 'Continuous'
df_both2_binary = df_weights_binary2.reset_index().melt(id_vars='index')
df_both2_binary['Behavior Type'] = 'Discrete'
df_both2 = pd.concat([df_both2, df_both2_binary])
df_both2.columns = ['Dataset Name', 'Neuron', 'Weight', 'Behavior Type']
# df_both1

fig = px.box(df_both2[df_both2['Neuron'].isin(neurons_to_plot)], x='Neuron', y='Weight', color='Behavior Type',
             hover_data=['Dataset Name'],
             color_discrete_map=plotly_paper_color_discrete_map())

apply_figure_settings(fig, width_factor=1.0, height_factor=0.3, plotly_not_matplotlib=True)
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="CCA Weight (mode 2)")
fig.update_xaxes(title="", tickfont_size=12)

fig.update_layout(legend=dict(
    yanchor="top",
    y=1.0,
    xanchor="left",
    x=0.5
))
fig.show()

fname = os.path.join(output_folder, 'paired_boxplot_neural_weights2_BOTH.png')
fig.write_image(fname, scale=3)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)


# ## SUPP: Same but for mode 3

# In[72]:


all_cca_classes3, df_weights3, df_weights_binary3 = calc_cca_weights_for_all_projects(all_projects_gcamp, which_mode=2, min_datasets_present=5,
                                                                                       weights_kwargs=dict(n_components=3),
                                                                                      combine_left_and_right=True,
                                                                                    correct_sign_using_top_weight=True,
                                                                                                             preprocess_traces_using_pca=True, truncate_traces_to_n_components=5,
                                                                                                            preprocess_behavior_using_pca=True,
                                                                                   trace_kwargs=dict(use_paper_options=True))


# In[73]:


# df_weights3 = df_weights3[[c for c in df_weights3.columns if c in neurons_with_confident_ids(combine_left_right=True)]]

# fig = px.box(df_weights3, color_discrete_sequence=[plotly_paper_color_discrete_map()['CCA']])#, title="CCA weights of mode 3 across recordings")
# apply_figure_settings(fig, width_factor=1.0, height_factor=0.2, plotly_not_matplotlib=True)
# fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="Weight <br> (mode 3)")
# fig.update_xaxes(title="")
# fig.show()

# fname = os.path.join(output_folder, 'paired_boxplot_neural_weights3.png')
# fig.write_image(fname, scale=3)
# fname = Path(fname).with_suffix('.svg')
# fig.write_image(fname)


# In[74]:


# fig = px.box(df_weights_binary, title="CCA weights of mode 3 across recordings (binary)")
# fig.show()

# fname = os.path.join(output_folder, 'paired_boxplot_neural_weights_binary3.png')
# fig.write_image(fname)


# ## Same but for behavior weights

# In[75]:


from wbfm.utils.general.utils_paper import behavior_name_mapping


# In[76]:


all_cca_classes_beh1, df_weights_beh1, df_weights_binary_beh1 = calc_cca_weights_for_all_projects(all_projects_gcamp, which_mode=0, min_datasets_present=5,
                                                                                       weights_kwargs=dict(n_components=2), neural_not_behavioral=False,
                                                                                                             preprocess_traces_using_pca=True, truncate_traces_to_n_components=5,
                                                                                                            preprocess_behavior_using_pca=False,
                                                                                   trace_kwargs=dict(use_paper_options=True))


# In[77]:


df_weights_beh1.rename(columns=behavior_name_mapping(shorten=True)).head()


# In[78]:


fig = px.box(df_weights_beh1.rename(columns=behavior_name_mapping(shorten=True)), color_discrete_sequence=[plotly_paper_color_discrete_map()['CCA']])
apply_figure_settings(fig, width_factor=0.25, height_factor=0.2, plotly_not_matplotlib=True)
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="CCA Weight<br>(Component 1)",
                #range=[-1.1, 1.1]
                )
# fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="Weight  <br> (mode 1)")
fig.update_xaxes(title="", tickfont_size=12)
fig.show()

fname = os.path.join(output_folder, 'paired_boxplot_beh_weights.png')
fig.write_image(fname, scale=3)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)


# In[79]:


cmap = px.colors.qualitative.Plotly
# df_weights_binary_beh1['color'] = cmap[1]

fig = px.box(df_weights_binary_beh1.rename(columns=behavior_name_mapping(shorten=True)), color_discrete_sequence=[plotly_paper_color_discrete_map()['Discrete']])
fig.update_layout(showlegend=False)
apply_figure_settings(fig, width_factor=0.3, height_factor=0.22, plotly_not_matplotlib=True)
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="Discrete CCA<br>Weight<br>(Component 1)",
                range=[-0.2, 1.1])
fig.update_xaxes(title="", tickfont_size=12)
fig.show()

fname = os.path.join(output_folder, 'paired_boxplot_beh_weights_binary.png')
fig.write_image(fname, scale=3)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)


# ### Debug

# In[80]:


# px.line(all_cca_classes_beh1['2022-11-23_worm10']._df_beh)


# In[81]:


# px.line(all_cca_classes_beh1['2022-11-23_worm10']._df_traces_truncated)


# In[82]:


c = all_cca_classes_beh1['2022-11-23_worm10']


# In[83]:


opt = dict(binary_behaviors=False)
_, _, cca = cca_plotter.calc_cca(**opt)
trace_weights, behavior_weights = c.get_weights_from_cca(cca, **opt)
behavior_weights


# In[ ]:





# In[ ]:





# ## Supp: behavior

# In[84]:


all_cca_classes_beh2, df_weights_beh2, df_weights_binary_beh2 = calc_cca_weights_for_all_projects(all_projects_gcamp, which_mode=1, min_datasets_present=5,
                                                                                       weights_kwargs=dict(n_components=3), neural_not_behavioral=False,
                                                                                                             preprocess_traces_using_pca=True, truncate_traces_to_n_components=5,
                                                                                                            preprocess_behavior_using_pca=False,
                                                                                   trace_kwargs=dict(use_paper_options=True))


# In[85]:


fig = px.box(df_weights_beh2.rename(columns=behavior_name_mapping(shorten=True)), color_discrete_sequence=[plotly_paper_color_discrete_map()['CCA']])
apply_figure_settings(fig, width_factor=0.25, height_factor=0.3, plotly_not_matplotlib=True)
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="CCA Weight <br> (mode 2)")
fig.update_xaxes(title="")
fig.show()

fname = os.path.join(output_folder, 'paired_boxplot_beh_weights2.png')
fig.write_image(fname, scale=3)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)


# In[86]:


fig = px.box(df_weights_binary_beh2.rename(columns=behavior_name_mapping(shorten=True)), color_discrete_sequence=[plotly_paper_color_discrete_map()['Discrete']])

apply_figure_settings(fig, width_factor=0.25, height_factor=0.3, plotly_not_matplotlib=True)
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="CCA Weight <br> (discrete mode 2)")
fig.update_xaxes(title="")
fig.show()

fname = os.path.join(output_folder, 'paired_boxplot_beh_weights_binary2.png')
fig.write_image(fname, scale=3)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)


# In[87]:


all_cca_classes_beh3, df_weights_beh3, df_weights_binary_beh3 = calc_cca_weights_for_all_projects(all_projects_gcamp, which_mode=2, min_datasets_present=5,
                                                                                       weights_kwargs=dict(n_components=3), neural_not_behavioral=False,
                                                                                                             preprocess_traces_using_pca=True, truncate_traces_to_n_components=5,
                                                                                                            preprocess_behavior_using_pca=True,
                                                                                   trace_kwargs=dict(use_paper_options=True))


# In[88]:


fig = px.box(df_weights_beh3.rename(columns=behavior_name_mapping(shorten=True)), color_discrete_sequence=[plotly_paper_color_discrete_map()['CCA']])
apply_figure_settings(fig, width_factor=0.25, height_factor=0.2, plotly_not_matplotlib=True)
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="Weight <br> (mode 3)")
fig.update_xaxes(title="")
fig.show()

fname = os.path.join(output_folder, 'paired_boxplot_beh_weights3.png')
fig.write_image(fname, scale=3)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)


# In[ ]:





# In[ ]:




