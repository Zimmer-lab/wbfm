#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[18]:


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


# In[19]:


from sklearn.decomposition import PCA
from wbfm.utils.visualization.plot_traces import make_grid_plot_from_dataframe
import seaborn as sns
from wbfm.utils.visualization.filtering_traces import fill_nan_in_dataframe
import plotly.express as px
from wbfm.utils.general.utils_filenames import add_name_suffix


# In[20]:


# Load multiple datasets
from wbfm.utils.general.hardcoded_paths import load_paper_datasets
all_projects_gcamp = load_paper_datasets('gcamp')
all_projects_gfp = load_paper_datasets('gfp')


# In[21]:


all_projects_immob = load_paper_datasets('immob')


# In[22]:


# Get specific example datasets
project_data_gcamp = all_projects_gcamp['ZIM2165_Gcamp7b_worm1-2022_11_28']
project_data_immob = all_projects_immob['2022-12-13_15-16_ZIM2165_immob_worm9-2022-12-13']


# # Plot example heatmap and pca modes

# In[24]:


from wbfm.utils.visualization.plot_traces import make_summary_interactive_heatmap_with_pca, make_summary_heatmap_and_subplots


# In[25]:


project_data_gfp = all_projects_gfp['ZIM2319_GFP_worm3-2022-12-10']

# project_data_gfp = all_projects_gfp['ZIM2319_GFP_worm2-2022-12-10']
project_data_gfp.use_physical_x_axis = True


# In[26]:


fig1, fig2 = make_summary_heatmap_and_subplots(project_data_gfp, trace_opt=dict(use_paper_options=True), to_save=True, to_show=True, 
                                               output_folder="gfp", base_width=0.5)


# In[23]:


# fig2.update_yaxes(row=5, overwrite=True, tickangle=0, griddash='dash', 
#                   range=[-0.2, 0.2], tickmode='array', tickvals=[-0.2, 0, 0.2], zeroline=False, showline=True , showgrid=True, gridcolor='black')


# In[6]:


# fig1.update_layout(
#     title = 'My figure',
#     xaxis_title = r'$\Delta t\textrm{(s)}$',
#     yaxis_title = r'This text is not displayed/$\sqrt{2}\textrm{(ps)}$',
# )


# In[46]:


# fig2['layout']


# # Get traces for each datatype

# In[27]:


from wbfm.utils.visualization.multiproject_wrappers import build_trace_time_series_from_multiple_projects
from wbfm.utils.external.utils_pandas import apply_to_dict_of_dfs_and_concat
from wbfm.utils.general.utils_paper import apply_figure_settings, plotly_paper_color_discrete_map


# In[28]:


df_all_gcamp = build_trace_time_series_from_multiple_projects(all_projects_gcamp, use_paper_options=True)
df_all_gfp = build_trace_time_series_from_multiple_projects(all_projects_gfp, use_paper_options=True)
df_all_immob = build_trace_time_series_from_multiple_projects(all_projects_immob, use_paper_options=True)


# In[29]:


dict_of_dfs = {'wbfm': df_all_gcamp, 'immob': df_all_immob, 'gfp': df_all_gfp}


# In[30]:


# df_all_gcamp


# In[60]:


func = lambda df: df.drop(columns=['local_time']).groupby('dataset_name').var().melt()

df_all_var = apply_to_dict_of_dfs_and_concat(dict_of_dfs, func)
df_all_var.columns = ['Neuron name', 'Variance', 'Data Type']
df_all_var['Data Type'] = df_all_var['Data Type'].map({'wbfm': 'Freely Moving (GCaMP)',
                              'immob': 'Immobilized (GCaMP)',
                              'gfp': 'Freely Moving (GFP)',})

fig = px.box(df_all_var, y='Variance', color='Data Type', log_y=True, points='all',
            color_discrete_map=plotly_paper_color_discrete_map(),
            category_orders={'Data Type': ['Freely Moving (GFP)', 'Immobilized (GCaMP)', 'Freely Moving (GCaMP)']})
apply_figure_settings(fig, width_factor=0.33, height_factor=0.3)
fig.update_layout(showlegend=False, boxgroupgap=0.5)
fig.update_xaxes(range=[-0.4,0.4])

fig.show()

fname = os.path.join('gfp', 'variance_all_types.png')
fig.write_image(fname, scale=7)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)


# In[32]:


# fig = px.scatter(df_all_var, x='Data Type', y='Variance', color='Data Type', log_y=True, marginal_y='box')
# fig.show()


# In[16]:


# # fig = px.histogram(df_all_var, x='value', color='name', #log_x=True, 
# #                    barmode='group')
# fig = px.histogram(df_all_var, x='value', color='name', #log_x=True, 
#                    cumulative=True, histnorm='probability')#, barmode='group')
# fig.update_xaxes(title='Variance')
# fig.show()


# In[59]:


# import plotly.graph_objects as go

# # from: https://stackoverflow.com/questions/55220935/plotly-how-to-plot-a-cumulative-steps-histogram
# # use matplotlib to get "n" and "bins"
# # n_bins will affect the resolution of the cumilative histogram but not dictate the bin widths.
# n_bins = 100
# n, bins, patches = plt.hist(df_all_var['value'], n_bins, density=True, histtype='step', cumulative=1)

# # use plotly (v3) to plot
# data = []
# trace = go.Scatter(
#     x=bins,
#     y=n,
#     mode='lines',
#     name= "test",
#     line=dict(
#         shape='hvh'
#     )
# )

# data.append(trace)
# fig = go.Figure(data=data)
# fig.show()


# # Same but for green and red

# In[49]:


from wbfm.utils.visualization.multiproject_wrappers import build_trace_time_series_from_multiple_projects
from wbfm.utils.external.utils_pandas import apply_to_dict_of_dfs_and_concat
from wbfm.utils.general.utils_paper import apply_figure_settings, plotly_paper_color_discrete_map, data_type_name_mapping


# In[50]:


df_all_gcamp_red = build_trace_time_series_from_multiple_projects(all_projects_gcamp, use_paper_options=True, channel_mode='red', min_nonnan=0.7)
df_all_gfp_red = build_trace_time_series_from_multiple_projects(all_projects_gfp, use_paper_options=True, channel_mode='red', min_nonnan=0.7)
df_all_immob_red = build_trace_time_series_from_multiple_projects(all_projects_immob, use_paper_options=True, channel_mode='red', min_nonnan=0.7)


# In[51]:


df_all_gcamp_green = build_trace_time_series_from_multiple_projects(all_projects_gcamp, use_paper_options=True, channel_mode='green', min_nonnan=0.7)
df_all_gfp_green = build_trace_time_series_from_multiple_projects(all_projects_gfp, use_paper_options=True, channel_mode='green', min_nonnan=0.7)
df_all_immob_green = build_trace_time_series_from_multiple_projects(all_projects_immob, use_paper_options=True, channel_mode='green', min_nonnan=0.7)


# In[52]:


# px.line(df_all_gcamp_red['AVAL'])


# In[53]:


dict_of_dfs_red = {'wbfm': df_all_gcamp_red, 'immob': df_all_immob_red, 'gfp': df_all_gfp_red}
dict_of_dfs_green = {'wbfm': df_all_gcamp_green, 'immob': df_all_immob_green, 'gfp': df_all_gfp_green}


# In[54]:


# func = lambda df: df.drop(columns=['local_time']).groupby('dataset_name').var().melt()

# df_all_var_green = apply_to_dict_of_dfs_and_concat(dict_of_dfs_green, func)
# df_all_var_green.columns = ['Neuron name', 'Variance', 'Data Type']
# df_all_var_red['Data Type'].replace(data_type_name_mapping(), inplace=True)

# fig = px.box(df_all_var_green, y='Variance', color='Data Type', log_y=True, points='all')
# apply_figure_settings(fig, width_factor=1.0, height_factor=0.3)

# fig.show()

# fname = os.path.join('gfp', 'variance_all_types_green.png')
# fig.write_image(fname, scale=7)
# fname = Path(fname).with_suffix('.svg')
# fig.write_image(fname)


# In[55]:


# func = lambda df: df.drop(columns=['local_time']).groupby('dataset_name').var().melt()

# df_all_var_red = apply_to_dict_of_dfs_and_concat(dict_of_dfs_red, func)
# df_all_var_red.columns = ['Neuron name', 'Variance', 'Data Type']
# df_all_var_red['Data Type'].replace(data_type_name_mapping(), inplace=True)

# fig = px.box(df_all_var_red, y='Variance', color='Data Type', log_y=True, points='all')
# apply_figure_settings(fig, width_factor=1.0, height_factor=0.3)

# fig.show()

# fname = os.path.join('gfp', 'variance_all_types_red.png')
# fig.write_image(fname, scale=7)
# fname = Path(fname).with_suffix('.svg')
# fig.write_image(fname)


# In[64]:


func = lambda df: df.drop(columns=['local_time']).groupby('dataset_name').mean().melt()

df_all_mean_red = apply_to_dict_of_dfs_and_concat(dict_of_dfs_red, func)
df_all_mean_red.columns = ['Neuron name', 'Mean Red Brightness', 'Data Type']

df_all_mean_green = apply_to_dict_of_dfs_and_concat(dict_of_dfs_green, func)
df_all_mean_green.columns = ['Neuron name', 'Mean Green Brightness', 'Data Type']

df_all_mean_both = df_all_mean_green.copy()
df_all_mean_both['Mean Red Brightness'] = df_all_mean_red['Mean Red Brightness']
df_all_mean_both['Data Type'].replace({'wbfm': 'Freely Moving (GCaMP)',
                              'immob': 'Immobilized (GCaMP)',
                              'gfp': 'Freely Moving (GFP)',}, inplace=True)

fig = px.scatter(df_all_mean_both, x='Mean Red Brightness', y='Mean Green Brightness', color='Data Type', log_x=True, log_y=True,
                color_discrete_map=plotly_paper_color_discrete_map(), 
                marginal_y='box', marginal_x='box',
                category_orders={'Data Type': ['Freely Moving (GFP)', 'Immobilized (GCaMP)', 'Freely Moving (GCaMP)']})
apply_figure_settings(fig, width_factor=0.5, height_factor=0.3)

fig.update_layout(showlegend=False)
fig.show()

fname = os.path.join('gfp', 'mean_all_types_both.png')
fig.write_image(fname, scale=7)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)


# # Plot correlations between behavior and pca/ava

# In[2]:


from wbfm.utils.general.hardcoded_paths import get_hierarchical_modeling_dir
fname = os.path.join(get_hierarchical_modeling_dir(), 'data.h5')
Xy = pd.read_hdf(fname)

fname = os.path.join(get_hierarchical_modeling_dir(gfp=True), 'data.h5')
Xy_gfp = pd.read_hdf(fname)


# In[3]:


# column_names = ['AVAL', 'pca_0', 'fwd', 'speed']
# fig = px.scatter_matrix(Xy, dimensions=column_names)
# fig.update_traces(diagonal_visible=False, showupperhalf=False)


# In[11]:


import pandas as pd
import numpy as np
from wbfm.utils.external.utils_pandas import combine_columns_with_suffix

column_names = ['AVAL', 'AVAR', 'pca_0', 'fwd', 'speed']

def get_unique_correlations(group):
    # Exclude the grouping column
    group = group.drop(columns='dataset_name')
    
    # Calculate the correlation matrix of a subset of columns
    corr_matrix = combine_columns_with_suffix(group[column_names]).corr()#.abs()
    
    # Extract the upper triangle of the correlation matrix, excluding the diagonal
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Convert the upper triangle matrix to a long-format DataFrame
    pairs = (
        upper_triangle
        .stack()
        .reset_index()
        .rename(columns={'level_0': 'Variable1', 'level_1': 'Variable2', 0: 'Correlation'})
    )
    return pairs

# Group by and apply the correlation calculation
df_corr_gcamp = Xy.groupby('dataset_name').apply(get_unique_correlations).reset_index(level=1, drop=True).reset_index()
df_corr_gcamp['both_variables'] = df_corr_gcamp['Variable1'] + '-' + df_corr_gcamp['Variable2']
df_corr_gcamp['Data Type'] = 'gcamp'

# Also for gfp
df_corr_gfp = Xy_gfp.groupby('dataset_name').apply(get_unique_correlations).reset_index(level=1, drop=True).reset_index()
df_corr_gfp['both_variables'] = df_corr_gfp['Variable1'] + '-' + df_corr_gfp['Variable2']
df_corr_gfp['Data Type'] = 'gfp'

df_corr = pd.concat([df_corr_gcamp, df_corr_gfp], axis=0)

# Display the result
# print(df_corr)


# In[32]:


from wbfm.utils.general.utils_paper import plotly_paper_color_discrete_map, apply_figure_settings

fig = px.box(df_corr, x='both_variables', y='Correlation', color='Data Type', points='all', 
            color_discrete_map=plotly_paper_color_discrete_map())
apply_figure_settings(fig, width_factor=0.5, height_factor=0.3)
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black")
fig.update_xaxes(title="Pair of correlated variables")

fig.show()

fname = os.path.join('gfp', 'comparison_to_speed.png')
fig.write_image(fname, scale=7)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)


# ## Autocovariance

# In[57]:


# df_all = pd.concat(dict_of_dfs, ignore_index=True).drop(columns=['local_time'])
# 0lag=1

# df_all_autocorr = df_all.groupby('dataset_name').apply(lambda group: group.select_dtypes(include='number').apply(autocorr_func)).melt()
# df_all_autocorr.head()


# In[20]:


# lag=1
# autocorr_func = lambda col: col.autocorr(lag=lag) * col.var()

# func = lambda df: df.drop(columns=['local_time']).groupby('dataset_name').apply(
#     lambda subdf: subdf.apply(autocorr_func)).melt()

# df_all_autocorr = apply_to_dict_of_dfs_and_concat(dict_of_dfs, func)
# df_all_autocorr.columns = ['Neuron name', 'Autocovariance', 'Data Type']
# df_all_autocorr['Data Type'].replace(data_type_name_mapping(), inplace=True)

# fig = px.box(df_all_autocorr, y='Autocovariance', color='Data Type', log_y=True, points='all',
#             color_discrete_map=plotly_paper_color_discrete_map())
# apply_figure_settings(fig, width_factor=1.0, height_factor=0.3)

# fig.show()

# fname = os.path.join('gfp', 'autocorr_all_types.png')
# fig.write_image(fname, scale=7)
# fname = Path(fname).with_suffix('.svg')
# fig.write_image(fname)


# In[58]:


# df_combined = df_all_autocorr.copy()
# df_combined['Variance'] = df_all_var['Variance']

# px.scatter(df_combined, x='Variance', y='Autocovariance')


# # Scratch

# In[19]:


df_red = project_data_gcamp.calc_default_traces(use_paper_options=True, channel_mode='red', min_nonnan=0.9)


# In[34]:


# df_red['AVAL'].plot()


# In[ ]:




