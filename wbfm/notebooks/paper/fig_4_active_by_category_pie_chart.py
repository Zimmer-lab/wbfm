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
import plotly.express as px


# In[2]:


from sklearn.decomposition import PCA
from wbfm.utils.visualization.plot_traces import make_grid_plot_from_dataframe
import seaborn as sns
from wbfm.utils.visualization.behavior_comparison_plots import NeuronToMultivariateEncoding
from wbfm.utils.traces.gui_kymograph_correlations import build_all_gui_dfs_multineuron_correlations


# In[3]:


fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-11-27_spacer_7b_2per_agar/ZIM2165_Gcamp7b_worm1-2022_11_28/project_config.yaml"
# Manually corrected version
# fname = "/scratch/neurobiology/zimmer/Charles/dlc_stacks/manually_annotated/paper_data/ZIM2165_Gcamp7b_worm1-2022_11_28/project_config.yaml"
# project_data_gcamp = ProjectData.load_final_project_data_from_config(fname)


# In[4]:


# Load multiple datasets
from wbfm.utils.general.hardcoded_paths import load_paper_datasets
# all_projects_gcamp = load_paper_datasets('gcamp')
# INSTEAD: use Hannah's projects, because they are ID'ed with the O2 neurons
all_projects_gcamp = load_paper_datasets(['hannah_O2_fm', 'gcamp'])

all_projects_gfp = load_paper_datasets('gfp')


# In[5]:


all_projects_immob = load_paper_datasets('immob')


# # Get all traces

# In[6]:


from wbfm.utils.visualization.multiproject_wrappers import build_trace_time_series_from_multiple_projects


# In[7]:


df_traces_gcamp = build_trace_time_series_from_multiple_projects(all_projects_gcamp, use_paper_options=True)
df_traces_gfp = build_trace_time_series_from_multiple_projects(all_projects_gfp, use_paper_options=True)
df_traces_immob = build_trace_time_series_from_multiple_projects(all_projects_immob, use_paper_options=True)


# ## Define "active" based on a gfp cutoff

# In[8]:


# Get the threshold
gfp_var = df_traces_gfp.drop(columns=['local_time']).groupby('dataset_name').var().melt().dropna()
active_threshold = gfp_var['value'].quantile(0.95)
active_threshold


# # Get number of "active" neurons in each dataset

# In[9]:


from wbfm.utils.general.utils_paper import apply_figure_settings, plotly_paper_color_discrete_map


# In[10]:


def get_active_neurons_per_dataset(df_traces):
    df_group = df_traces.drop(columns=['local_time']).groupby('dataset_name')

    # Get the fraction and number of active neurons per dataset
    total_neurons_per_dataset = df_group.var().T.count()

    # Matrix indices; note: nan counts as false
    active_ind = df_group.var() > active_threshold
    active_neurons_per_dataset = active_ind.T.sum()

    fraction_active_neurons_per_dataset = active_neurons_per_dataset / total_neurons_per_dataset
    
    return active_neurons_per_dataset, total_neurons_per_dataset, fraction_active_neurons_per_dataset


# In[11]:


active_neurons_gcamp, total_neurons_gcamp, fraction_active_neurons_gcamp = get_active_neurons_per_dataset(df_traces_gcamp)
active_neurons_gfp, total_neurons_gfp, fraction_active_neurons_gfp = get_active_neurons_per_dataset(df_traces_gfp)
active_neurons_immob, total_neurons_immob, fraction_active_neurons_immob = get_active_neurons_per_dataset(df_traces_immob)


# In[12]:


column_names = ['Active Neurons', 'Total Neurons']
df_all_active = pd.concat([active_neurons_gcamp, total_neurons_gcamp], axis=1)
df_all_active.columns = column_names
df_all_active['Datatype'] = 'Freely Moving (GCaMP)'

_df = pd.concat([active_neurons_gfp, total_neurons_gfp], axis=1)
_df.columns = column_names
_df['Datatype'] = 'Freely Moving (GFP)'

_df2 = pd.concat([active_neurons_immob, total_neurons_immob], axis=1)
_df2.columns = column_names
_df2['Datatype'] = 'Immobilized (GCaMP)'

df_all_active = pd.concat([df_all_active, _df, _df2])
df_all_active.head()


# In[13]:


fig = px.scatter(df_all_active, x='Total Neurons', y='Active Neurons', color='Datatype',
          marginal_x='box', marginal_y='box', color_discrete_map=plotly_paper_color_discrete_map(),
                category_orders={'Datatype': ['Freely Moving (GFP)', 'Immobilized (GCaMP)', 'Freely Moving (GCaMP)']})
apply_figure_settings(fig, width_factor=0.67, height_factor=0.3)
# fig.update_layout(showlegend=False)

fig.show()

to_save = True
if to_save:
    fname = os.path.join("active_by_category", "number_active_scatter_and_box.png")
    fig.write_image(fname, scale=3)
    fname = str(Path(fname).with_suffix('.svg'))
    fig.write_image(fname)


# In[14]:


# df = pd.DataFrame({'Freely Moving (GCaMP)': active_neurons_gcamp, 'Immobilized (GCaMP)': active_neurons_immob,
#                   'GFP': active_neurons_gfp})
# fig = px.box(df)



# In[15]:


df = pd.DataFrame({'Freely Moving (GCaMP)': fraction_active_neurons_gcamp, 'Immobilized (GCaMP)': fraction_active_neurons_immob,
                  'Freely Moving (GFP)': fraction_active_neurons_gfp})
df = df.melt(var_name='Datatype', value_name='Fraction active neurons')

fig = px.box(df, x='Datatype', color='Datatype', y='Fraction active neurons', color_discrete_map=plotly_paper_color_discrete_map(),
            category_orders={'Datatype': ['Freely Moving (GFP)', 'Immobilized (GCaMP)', 'Freely Moving (GCaMP)']})
fig.update_layout(showlegend=False)
fig.update_xaxes(title="", showticklabels=False)
apply_figure_settings(fig, width_factor=0.25, height_factor=0.25)
fig.show()


to_save = True
if to_save:
    fname = os.path.join("active_by_category", "fraction_active_box.png")
    fig.write_image(fname, scale=3)
    fname = str(Path(fname).with_suffix('.svg'))
    fig.write_image(fname)


# # Define categories based on if they are active in immob

# In[16]:


from wbfm.utils.general.hardcoded_paths import list_of_gas_sensing_neurons


# In[17]:


# Get the active ID'ed neurons
# Have number of all active neurons above
df_group = df_traces_immob.drop(columns=['local_time']).groupby('dataset_name')
df_active = (df_group.std() > active_threshold).sum()
num_active_per_ID = df_active[['neuron' not in name for name in df_active.index]]

num_immob_datasets = len(df_traces_immob['dataset_name'].unique())

neurons_active_and_ided_in_immob = set(num_active_per_ID[num_active_per_ID > 0.4*num_immob_datasets].index)
len(neurons_active_and_ided_in_immob)


# In[18]:


df_group = df_traces_gcamp.drop(columns=['local_time']).groupby('dataset_name')
df_active = (df_group.std() > active_threshold).sum()
print(df_active.head())

real_name_idx = [('neuron' not in name and 'VG_' not in name) for name in df_active.index]
num_active_per_ID = df_active[real_name_idx]

num_gcamp_datasets = len(df_traces_gcamp['dataset_name'].unique())

neurons_active_and_ided_in_fm = set(num_active_per_ID[num_active_per_ID > 0.4*num_gcamp_datasets].index)
len(neurons_active_and_ided_in_fm)


# In[19]:


neurons_active_in_both = neurons_active_and_ided_in_fm.intersection(neurons_active_and_ided_in_immob)
neurons_active_in_only_fm = neurons_active_and_ided_in_fm - neurons_active_and_ided_in_immob
neurons_active_in_only_immob = neurons_active_and_ided_in_immob - neurons_active_and_ided_in_fm
gas_sensing_neurons = list_of_gas_sensing_neurons()

len(neurons_active_in_both), len(neurons_active_in_only_fm), len(neurons_active_in_only_immob), len(gas_sensing_neurons)


# In[20]:


# 3-step boxplot: All active, ided + non-ided, ided shared + ided unique

# Get active and ID'ed (often) per dataset
df_group = df_traces_gcamp.drop(columns=['local_time']).groupby('dataset_name')

# Get the fraction and number of active neurons per dataset
total_neurons_per_dataset = df_group.std().T.count()

# Matrix indices; note: nan counts as false
active_ind = df_group.std() > active_threshold
neurons_active_and_ided_in_fm_per_dataset = active_ind.loc[:, list(neurons_active_and_ided_in_fm)].T.sum()
neurons_active_in_both_and_ided_in_fm_per_dataset = active_ind.loc[:, list(neurons_active_in_both)].T.sum()
neurons_active_in_only_fm_and_ided_in_fm_per_dataset = active_ind.loc[:, list(neurons_active_in_only_fm)].T.sum()
neurons_active_gas_sensing_and_ided_in_fm_per_dataset = active_ind.loc[:, [name in gas_sensing_neurons for name in active_ind.columns]].T.sum()


# In[21]:


df = pd.DataFrame({#'Total active neurons': active_neurons_gcamp, 
                   'IDed neurons': neurons_active_and_ided_in_fm_per_dataset,
                   'Shared immob active neurons': neurons_active_in_both_and_ided_in_fm_per_dataset,
                   'Unique fm active neurons': neurons_active_in_only_fm_and_ided_in_fm_per_dataset,
                   'Gas sensing fm active neurons': neurons_active_gas_sensing_and_ided_in_fm_per_dataset,
})

fig = px.box(df, points='all')
apply_figure_settings(fig, width_factor=0.5, height_factor=0.25)
fig.update_yaxes(title='Number')
fig.update_xaxes(title='')

fig.show()

to_save = True
if to_save:
    fname = os.path.join("active_by_category", "fraction_ided_box.png")
    fig.write_image(fname, scale=3)
    fname = str(Path(fname).with_suffix('.svg'))
    fig.write_image(fname)


# # Define categories based on participation in the global manifold

# In[22]:


from wbfm.utils.visualization.utils_cca import calc_pca_weights_for_all_projects
from wbfm.utils.visualization.multiproject_wrappers import build_dataframe_of_variance_explained
from wbfm.utils.general.hardcoded_paths import neurons_with_confident_ids


# ## Immob

# In[24]:


df_var_exp_immob = build_dataframe_of_variance_explained(all_projects_immob, use_paper_options=True, interpolate_nan=True)


# In[25]:


manifold_threshold = 0.5
num_immob_datasets = df_var_exp_immob.shape[0]

# df_group = df_var_exp_immob.drop(columns=['local_time']).groupby('dataset_name')

# Take the mean variance explained across datasets
_df_active = (df_var_exp_immob.dropna(thresh = 0.5*num_immob_datasets, axis=1).groupby('neuron_name')['fraction_variance_explained'].mean(numeric_only=True) > manifold_threshold)

real_name_idx = [('neuron' not in name and 'VG_' not in name) for name in _df_active.index]
num_active_per_ID = _df_active[real_name_idx]
# print(df_active)

neurons_active_and_manifold_in_immob = set(num_active_per_ID[num_active_per_ID].index)
len(neurons_active_and_ided_in_immob), len(neurons_active_and_manifold_in_immob)


# ## FM

# In[26]:


df_weights_pc0_gcamp = calc_pca_weights_for_all_projects(all_projects_gcamp, which_mode=0, 
                                                         drop_unlabeled_neurons=False, min_datasets_present=0,
                                                        use_paper_options=True, interpolate_nan=True)


# In[28]:


df_var_exp_gcamp = build_dataframe_of_variance_explained(all_projects_gcamp, use_paper_options=True, interpolate_nan=True)


# In[29]:


# Get long dataframe, to be used for final pie chart
df_var_exp_gcamp_melt = df_var_exp_gcamp.reset_index().melt(id_vars=['dataset_name', 'neuron_name'],
                                                           value_vars='fraction_variance_explained')
df_var_exp_gcamp_melt.rename(columns={'value': 'fraction_variance_explained'}, inplace=True)

# Add a column for the simple variance threshold
df_group = df_traces_gcamp.drop(columns=['local_time']).groupby('dataset_name')
_df_active = df_group.var()
df_active_melt = _df_active.reset_index().melt(id_vars='dataset_name')
df_active_melt.rename(columns={'index': 'dataset_name', 'variable': 'neuron_name', 'value': 'variance'}, inplace=True)

# Make sure the datatypes match
melt_on_vars = ['dataset_name', 'neuron_name']
# for m in melt_on_vars:
#     df_var_exp_gcamp_melt[m] = df_var_exp_gcamp_melt[m].astype(pd.StringDtype())
#     df_active_melt[m] = df_active_melt[m].astype(pd.StringDtype())
# print(df_active_melt['dataset_name'].unique(), df_var_exp_gcamp_melt['dataset_name'].unique())
# print(df_active_melt['variable'].unique(), df_var_exp_gcamp_melt['variable'].unique())


# In[30]:


df_var_exp_gcamp_melt = df_var_exp_gcamp_melt.merge(df_active_melt, on=melt_on_vars)
df_var_exp_gcamp_melt.head()


# In[31]:


# Add columns for the relevant categories
id_list = neurons_with_confident_ids()

df_var_exp_gcamp_melt['has_id'] = df_var_exp_gcamp_melt['neuron_name'].apply(lambda x: x in id_list)
df_var_exp_gcamp_melt['is_o2'] = df_var_exp_gcamp_melt['neuron_name'].apply(lambda x: x in gas_sensing_neurons)
# df_var_exp_gcamp_melt['active_in_immob'] = df_var_exp_gcamp_melt['variable'].apply(lambda x: x in neurons_active_and_ided_in_immob)
df_var_exp_gcamp_melt['active_in_immob'] = df_var_exp_gcamp_melt['neuron_name'].apply(lambda x: x in neurons_active_and_manifold_in_immob)


# In[32]:


df_var_exp_gcamp_melt['category'] = np.nan
df_var_exp_gcamp_melt['category'] = df_var_exp_gcamp_melt['has_id'].replace({True: np.nan, False: 'Not Identified'})
tmp = df_var_exp_gcamp_melt['is_o2'].replace({True: 'O2 or CO2 sensing', False: np.nan})
df_var_exp_gcamp_melt['category'].fillna(tmp, inplace=True)
tmp = df_var_exp_gcamp_melt['active_in_immob'].replace({True: 'Intrinsic (shared with immobilized)', False: np.nan})
df_var_exp_gcamp_melt['category'].fillna(tmp, inplace=True)
df_var_exp_gcamp_melt['category'].fillna('Manifold in Freely Moving only', inplace=True)

df_var_exp_gcamp_melt['marker_size'] = [0.3 if c=='Not Identified' else 1 for c in df_var_exp_gcamp_melt['category']]
print(df_var_exp_gcamp_melt['category'].value_counts())


# In[33]:


df_var_exp_gcamp_melt[df_var_exp_gcamp_melt['category']=='Manifold in Freely Moving only']['neuron_name'].unique()


# In[34]:


from wbfm.utils.general.utils_paper import plotly_paper_color_discrete_map, apply_figure_settings

active_idx = df_var_exp_gcamp_melt['variance'] > active_threshold

fig = px.scatter(df_var_exp_gcamp_melt[active_idx], y='variance', x='fraction_variance_explained', 
           color='category', size='marker_size', size_max=10,
          color_discrete_map=plotly_paper_color_discrete_map(),
                title='Active neurons by category')

apply_figure_settings(fig, height_factor=0.2)

fig.show()

output_folder = "/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/supplement/active_by_category"
fname = os.path.join(output_folder, 'variance_and_manifold_explained_all_neuron_types.png')
fig.write_image(fname, scale=4)
fname = fname.replace('.png', '.svg')
fig.write_image(fname)


# In[42]:


interesting_idx = (df_var_exp_gcamp_melt['variance'] > active_threshold) & \
    (df_var_exp_gcamp_melt['fraction_variance_explained'] > 0.5)
 
fig = px.pie(df_var_exp_gcamp_melt[interesting_idx], names='category', color='category',
          color_discrete_map=plotly_paper_color_discrete_map(),
            )#title='High manifold-active neurons by category')
# Round percentages: https://community.plotly.com/t/decimal-precision-in-pie-charts/31731/5
# fig.update_traces(texttemplate='%{percent:.0%f}')

apply_figure_settings(fig, height_factor=0.2, width_factor=0.5)

fig.show()

output_folder = "/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/supplement/active_by_category"
fname = os.path.join(output_folder, 'manifold_explained_pie_chart.png')
fig.write_image(fname, scale=4)
fname = fname.replace('.png', '.svg')
fig.write_image(fname)


# In[36]:


# valid_idx = ['neuron' not in name and 'VG_' not in name for name in df_var_exp_gcamp.columns]
# df_var_exp_gcamp_subset = df_var_exp_gcamp.loc[:, valid_idx]
# df_var_exp_gcamp_subset = df_var_exp_gcamp_subset.loc[:, df_var_exp_gcamp_subset.count() > 3]
# df_var_exp_gcamp_subset = df_var_exp_gcamp_subset.reindex(df_var_exp_gcamp_subset.median().sort_values().index, axis=1)


# In[ ]:


# px.histogram(df_var_exp_gcamp_subset)


# In[ ]:


# px.histogram(df_var_exp_gcamp.T)


# In[ ]:


# df_var_exp_gcamp_subset.head()


# In[ ]:


# Category with 3 outputs: not IDed, is_o2, or active_in_immob
# px.histogram(df_var_exp_gcamp_melt, x='value', color='active_in_immob', barmode='group')


# In[ ]:


# px.histogram(df_var_exp_gcamp_melt, x='value', color='category', barmode='group')


# In[ ]:


df_var_exp_gcamp_melt['category'].value_counts()


# # Scratch: Alternate pie chart: residual neurons

# In[ ]:


from wbfm.utils.visualization.multiproject_wrappers import calc_all_autocovariance


# In[ ]:


output_folder = "/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/supplement/active_by_category"
fig, df_summary, significance_line, cmap = calc_all_autocovariance(all_projects_gcamp, all_projects_gfp, lag=1, output_folder=output_folder,
                                                                   loop_not_facet_row=True,
                                                                   use_paper_options=True, include_gfp=True, include_legend=True)


# In[ ]:


px.histogram(df_summary['Autocorrelation'])


# In[ ]:


df_var_exp_gcamp_melt.shape, df_summary.shape


# In[ ]:


# Combine both dataframes
df_both = df_var_exp_gcamp_melt.rename(columns = {'variable': 'neuron_name'}).merge(
    df_summary, on=['dataset_name', 'neuron_name'])


# In[ ]:


# px.scatter(df_both, x='acv', y='variance', color='Type of data')


# In[ ]:


# Add column: has a high residual activity
df = df_both.copy()

# Assuming your DataFrame is named df

# Step 1: Apply threshold for 'residual gcamp' data
df['threshold_passed'] = (df['Type of data'] == 'residual gcamp') & (df['acv'] > significance_line)

# Step 2: Duplicate the boolean column for other 'Type of data'
df['high_residual_activity'] = df.groupby(['dataset_name', 'neuron_name'])['threshold_passed'].transform('max')

# Optionally, drop the intermediate column 'threshold_passed'
df.drop(columns=['threshold_passed'], inplace=True)


# In[ ]:


# Add final category column: both high residual activity and high manifold variance explained

df['high_residual_activity_str'] = df['high_residual_activity'].map({True: 'High residual', False: 'Low residual'})
df['fraction_variance_explained_str'] = (df['fraction_variance_explained']> 0.5).map({True: '; High manifold participation', False: '; Low manifold participation'})

df['Residual Activity Category'] = df['high_residual_activity_str'] + df['fraction_variance_explained_str'] 
df['Residual Activity Category'].unique()


# In[ ]:


# fig = px.box(df, x='Type of data', y='acv', color='high_residual_activity',
#           color_discrete_map=plotly_paper_color_discrete_map())
# fig.show()


# In[ ]:


# interesting_idx = (df['variance'] > active_threshold) & \
#     (df['Type of data'] == 'gcamp')
 
# fig = px.pie(df[interesting_idx], names='Residual Activity Category', color='Residual Activity Category')
#           #color_discrete_map=plotly_paper_color_discrete_map())

# apply_figure_settings(fig, height_factor=0.2, width_factor=0.5)
# fig.show()

# output_folder = "/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/supplement/active_by_category"
# fname = os.path.join(output_folder, 'manifold_explained_residual_pie_chart.png')
# fig.write_image(fname, scale=4)
# fname = fname.replace('.png', '.svg')
# fig.write_image(fname)

# # Also save version with values, not percent
# fig.update_traces(textinfo='value')

# fname = os.path.join(output_folder, 'manifold_explained_values_residual_pie_chart.png')
# fig.write_image(fname, scale=4)
# fname = fname.replace('.png', '.svg')
# fig.write_image(fname)


# In[ ]:


significance_line, df_summary[df_summary['Type of data'] == 'gfp']['acv'].quantile(0.95)


# In[ ]:


len(df['dataset_name'].unique())


# In[ ]:


df[df['Type of data'] == 'gcamp'].shape, interesting_idx.value_counts()


# # Scratch: adding these columns to previous summary matrix

# In[ ]:


from wbfm.utils.visualization.multiproject_wrappers import calc_summary_dataframe_all_datasets, plot_variance_all_neurons


# In[ ]:


df_summary, _, _ = calc_summary_dataframe_all_datasets(all_projects_gcamp, all_projects_gfp)


# In[ ]:


df_summary['Genotype and datatype'].value_counts()


# In[ ]:


# all_figs, df_summary, significance_line, cmap = plot_variance_all_neurons(all_projects_gcamp, all_projects_gfp, output_folder=None)


# In[ ]:


# x='fraction_variance_explained'
# y='acv'
# scatter_opt = dict(y=y, x=x, symbol='Simple Neuron ID', marginal_y='box', size='multiplex_size', log_y=True,
#                    size_max=5)
# fig = px.scatter(df_summary, facet_row='Type of data',
#                  color_discrete_sequence=cmap, range_y=[0.00005, 0.3],
#                  color='Genotype and datatype', **scatter_opt)

# fig.update_traces(marker=dict(line=dict(width=0,
#                                         color='White')))
# fig.show()


# In[ ]:




