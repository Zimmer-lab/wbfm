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
from wbfm.utils.general.hardcoded_paths import get_hierarchical_modeling_dir

from wbfm.utils.general.utils_filenames import add_name_suffix


# In[3]:


# # Load multiple datasets
# from wbfm.utils.general.hardcoded_paths import load_paper_datasets
# all_projects_gcamp = load_paper_datasets(['gcamp', 'hannah_O2_fm'])
# # all_projects_gfp = load_paper_datasets('gfp')


# In[4]:


# all_projects_immob = load_paper_datasets('immob')


# In[5]:


# all_projects_immob_O2 = load_paper_datasets('hannah_O2_immob')


# In[6]:


from wbfm.utils.general.hardcoded_paths import load_all_data_as_dataframe


# In[7]:


get_hierarchical_modeling_dir()


# In[8]:


Xy = load_all_data_as_dataframe()


# In[9]:


Xy['fwd'].unique()


# In[10]:


'IL1LL' in Xy


# In[11]:


# Xy['dataset_name'].unique()


# In[12]:


# _Xy = Xy.loc[Xy['dataset_name'].apply(lambda x: 'hiscl' in x), :].dropna().copy()
# cols = _Xy.columns
# for c in cols:
#     if 'manifold' not in c and 'IL' in c.upper():
#         print(c)
#         print(Xy[[c, 'dataset_name']].dropna()['dataset_name'].unique())


# In[13]:


Xy['dataset_name'].apply(lambda x: 'hiscl' in x)


# # Plot ided neurons, counted by dataset and colored by datatype

# In[17]:


from wbfm.utils.general.hardcoded_paths import neurons_with_confident_ids
from wbfm.utils.general.utils_paper import apply_figure_settings, plotly_paper_color_discrete_map, data_type_name_mapping


# In[18]:


def func(col):
    if 'neuron' in col.name or '_' in col.name:
        return np.nan
    # print(col.name)
    non_null_this_col = col.dropna().index
    return len(Xy.loc[non_null_this_col, 'dataset_name'].unique())

Xy_count = Xy.apply(func)
# Xy_count = Xy_count.div(Xy_count['local_time'].values, axis=0)
Xy_count


# In[19]:


non_nan_groups = Xy.groupby(['dataset_name', 'dataset_type'])[neurons_with_confident_ids()].agg(lambda x: x.count() > 0).reset_index()
# non_nan_groups = Xy.groupby(['dataset_name', 'dataset_type']).agg(lambda x: x.count() > 0).reset_index()

non_nan_totals = non_nan_groups.groupby('dataset_type').sum(numeric_only=True).sort_values(by='gcamp', axis=1)
# non_nan_totals /= non_nan_totals.max(axis=1, numeric_only=True)
non_nan_melt = non_nan_totals.reset_index().melt(id_vars='dataset_type', var_name='neuron_name', value_name='count')

non_nan_fraction = (non_nan_totals.T / non_nan_totals.max(axis=1)).T
non_nan_fraction = non_nan_fraction.reset_index().melt(id_vars='dataset_type', var_name='neuron_name', value_name='count')


# In[20]:


print(non_nan_totals)


# In[21]:


non_nan_melt.columns = ['Dataset Type', 'Neuron Name', 'Count']
non_nan_melt['Dataset Type'] = non_nan_melt['Dataset Type'].map(data_type_name_mapping())

non_nan_fraction.columns = ['Dataset Type', 'Neuron Name', 'Count']
non_nan_fraction['Dataset Type'] = non_nan_fraction['Dataset Type'].map(data_type_name_mapping())


# In[22]:


non_nan_melt[non_nan_melt['Neuron Name'] == 'AIBR']


# In[23]:


fig = px.bar(non_nan_melt, color='Dataset Type', x='Neuron Name', y='Count',
       barmode='group', color_discrete_map=plotly_paper_color_discrete_map(),
            category_orders={'Dataset Type': ['Freely Moving (GCaMP)', 'Immobilized (GCaMP)', 'Freely Moving (GFP)']})

fig.update_layout(
    showlegend=False,
    legend=dict(
      yanchor="top",
      y=1.5,
      xanchor="left",
      x=0.01
    ),
)
fig.update_xaxes(tickfont_size=10)
apply_figure_settings(fig, width_factor=1.0, height_factor=0.2)
fig.update_yaxes(showgrid=True, gridwidth=1, griddash='dash', gridcolor='black', overwrite=True)

fig.show()


fname = 'ids/neuron_ids_per_neuron.png'
fig.write_image(fname, scale=3)
fname = fname.replace('.png', '.svg')
fig.write_image(fname)


# In[24]:


# Same, but no grid and with horizontal lines where the total number of datasets are

fig = px.bar(non_nan_melt, color='Dataset Type', x='Neuron Name', y='Count',
       barmode='group', color_discrete_map=plotly_paper_color_discrete_map(),
            category_orders={'Dataset Type': ['Freely Moving (GCaMP)', 'Immobilized (GCaMP)', 'Freely Moving (GFP)']})

fig.update_layout(
    showlegend=False,
    legend=dict(
      yanchor="top",
      y=1.5,
      xanchor="left",
      x=0.01
    ),
)
fig.update_xaxes(tickfont_size=10)
apply_figure_settings(fig, width_factor=1.0, height_factor=0.2)
# fig.update_yaxes(showgrid=True, gridwidth=1, griddash='dash', gridcolor='black', overwrite=True)

# New: lines
all_maxes = non_nan_melt.groupby('Dataset Type').max().to_dict()['Count']
for datatype, y_val in all_maxes.items():
    color = plotly_paper_color_discrete_map()[datatype]
    fig.add_shape(type="line",
                  x0=0, y0=y_val,  # start of the line (bottom of the plot)
                  x1=1, y1=y_val,  # end of the line (top of the plot)
                  line=dict(color=color, width=2),
                  xref='paper',
                  yref='y')
fig.show()


fname = 'ids/neuron_ids_per_neuron_with_lines.png'
fig.write_image(fname, scale=3)
fname = fname.replace('.png', '.svg')
fig.write_image(fname)


# In[25]:


fig = px.bar(non_nan_fraction, color='Dataset Type', x='Neuron Name', y='Count',
       barmode='group', color_discrete_map=plotly_paper_color_discrete_map(),
            category_orders={'Dataset Type': ['Freely Moving (GCaMP)', 'Immobilized (GCaMP)', 'Freely Moving (GFP)']})

fig.update_layout(
    showlegend=False
)
fig.update_xaxes(tickfont_size=10)
apply_figure_settings(fig, width_factor=1.0, height_factor=0.2)
fig.update_yaxes(showgrid=True, gridwidth=1, griddash='dash', gridcolor='black', overwrite=True)

fig.show()


fname = 'ids/fraction_neuron_ids_per_neuron.png'
fig.write_image(fname, scale=3)
fname = fname.replace('.png', '.svg')
fig.write_image(fname)


# In[26]:


non_nan_fraction['Neuron Name'].value_counts()


# ## Export to an excel sheet, with total numbers

# In[26]:


id_export = non_nan_totals.copy()
id_export['Number of Datasets'] = Xy.groupby('dataset_type')['dataset_name'].agg(lambda x: len(x.unique()))
id_export.index = id_export.index.map(data_type_name_mapping(include_mutant=True))
# id_export.head()


# In[27]:


fname = 'ids/ids_per_dataset_type.xlsx'
id_export.to_excel(fname)


# # Boxplot of IDed neurons per dataset

# In[28]:


non_nan_groups_melt = non_nan_groups.melt(id_vars=['dataset_type', 'dataset_name'])
non_nan_groups_melt = non_nan_groups_melt[non_nan_groups_melt['value']]
non_nan_id_per_dataset = non_nan_groups_melt.groupby('dataset_type')['dataset_name'].value_counts().reset_index()

non_nan_id_per_dataset.columns = ['Dataset Type', 'Dataset Name', 'Number of IDed neurons']
non_nan_id_per_dataset['Dataset Type'] = non_nan_id_per_dataset['Dataset Type'].map(data_type_name_mapping())
non_nan_id_per_dataset


# In[29]:


fig = px.box(non_nan_id_per_dataset, x='Dataset Type', y='Number of IDed neurons', color='Dataset Type', 
      color_discrete_map=plotly_paper_color_discrete_map(), points='all',
            category_orders={'Dataset Type': ['Freely Moving (GFP)', 'Freely Moving (GCaMP)', 'Immobilized (GCaMP)', ]})

fig.update_layout(
    showlegend=True,
    legend=dict(
      yanchor="top",
      y=0.3,
      xanchor="left",
      x=0.3
    ),
)
fig.update_xaxes(title="", showticklabels=False)

apply_figure_settings(fig, width_factor=0.4, height_factor=0.3)
fig.show()

fname = 'ids/neuron_ids_per_dataset.png'
fig.write_image(fname, scale=3)
fname = fname.replace('.png', '.svg')
fig.write_image(fname)


# # Fraction of IDs per dataset

# In[28]:


Xy.head()


# In[48]:


ided_columns = [col for col in Xy.columns if col in neurons_with_confident_ids() and 'manifold' not in col]
nonided_columns = [col for col in Xy.columns if col.startswith('neuron_') and 'manifold' not in col]

# Count non-null neuron IDs per dataset
ided_result = Xy.groupby(['dataset_name', 'dataset_type'])[ided_columns].apply(lambda x: (x.notna().any()).sum()).reset_index()
nonided_result = Xy.groupby(['dataset_name', 'dataset_type'])[nonided_columns].apply(lambda x: (x.notna().any()).sum()).reset_index()

ided_result.columns = ['dataset_name', 'Dataset Type', 'neuron_id_count']
nonided_result.columns = ['dataset_name', 'Dataset Type', 'neuron_id_count']


# In[49]:


ided_result['Fraction of IDed neurons'] = ided_result['neuron_id_count'] / (ided_result['neuron_id_count'] + nonided_result['neuron_id_count'])


# In[51]:


fig = px.box(ided_result, x='Dataset Type', y='Fraction of IDed neurons', color='Dataset Type', 
      color_discrete_map=plotly_paper_color_discrete_map(), points='all',
            category_orders={'Dataset Type': ['Freely Moving (GFP)', 'Freely Moving (GCaMP)', 'Immobilized (GCaMP)', ]})
fig.show()


# In[ ]:





# # Alt: go back to projects

# # Freely moving: ID'ed neurons

# In[30]:


# all_ids = defaultdict(int)

# for name, proj in tqdm(all_projects_gcamp.items()):
#     df_traces = proj.calc_default_traces(use_paper_options=True)
#     neuron_columns = [c for c in df_traces.columns if 'neuron' not in c]
#     for c in neuron_columns:
#         all_ids[c] += 1


# In[31]:


# df_ids = pd.DataFrame(all_ids, index=[0]).T.sort_values(0, ascending=False)
# px.scatter(df_ids)


# In[32]:


# fname = os.path.join('ids', 'id_counts.csv')
# df_ids.to_csv(fname)


# # Immobilized: ID'ed neurons

# In[33]:


# all_ids = defaultdict(int)

# for name, proj in tqdm(all_projects_immob.items()):
#     df_traces = proj.calc_default_traces(use_paper_options=True)
#     neuron_columns = [c for c in df_traces.columns if 'neuron' not in c]
#     for c in neuron_columns:
#         all_ids[c] += 1


# In[34]:


# df_ids = pd.DataFrame(all_ids, index=[0]).T.sort_values(0, ascending=False)
# px.scatter(df_ids)


# In[35]:


# fname = os.path.join('ids', 'id_counts_immob.csv')
# df_ids.to_csv(fname)


# # Immobilized with O2 stimulus: ID'ed neurons

# In[36]:


# all_ids = defaultdict(int)

# for name, proj in tqdm(all_projects_immob_O2.items()):
#     df_traces = proj.calc_default_traces(use_paper_options=True)
#     neuron_columns = [c for c in df_traces.columns if 'neuron' not in c]
#     for c in neuron_columns:
#         all_ids[c] += 1


# In[37]:


# df_ids = pd.DataFrame(all_ids, index=[0]).T.sort_values(0, ascending=False)
# px.scatter(df_ids)


# In[38]:


# fname = os.path.join('ids', 'id_counts_immob_O2.csv')
# df_ids.to_csv(fname)


# # Same but with simple dataframes, not full projects

# In[39]:


# from wbfm.utils.general.hardcoded_paths import get_hierarchical_modeling_dir
# # fname = 'data.h5'

# fname = os.path.join(get_hierarchical_modeling_dir(), 'data.h5')
# Xy = pd.read_hdf(fname)

# fname = os.path.join(get_hierarchical_modeling_dir(gfp=True), 'data.h5')
# Xy_gfp = pd.read_hdf(fname)


# In[40]:


# def get_counts(Xy):
#     all_ids = np.ceil(Xy.count() / Xy.shape[0] * len(Xy['dataset_name'].unique()))
#     strs_to_drop = ['neuron', 'eigenworm', 'pca', 'name', 'time', 'curvature', 'manifold',
#                    'fwd', 'speed', 'collision']
#     for s in strs_to_drop:
#         all_ids = all_ids[~all_ids.index.str.contains(s)]
#     all_ids = all_ids.sort_values()
#     return all_ids

# all_ids = get_counts(Xy)
# all_ids_gfp = get_counts(Xy_gfp)


# In[41]:


# px.scatter(all_ids_gfp)


# In[ ]:





# In[ ]:





# # Scratch: Do any projects have no IDs?

# In[42]:


# all_num_ids = defaultdict(int)

# for name, proj in tqdm(all_projects_gcamp.items()):
#     df_traces = proj.calc_default_traces(use_paper_options=True)
#     neuron_columns = [c for c in df_traces.columns if 'neuron' not in c]
#     if 'RMED' not in neuron_columns:
#         print(name)
#     if 'AQ' in neuron_columns:
#         print(name)
#     all_num_ids[name] = len(neuron_columns)


# In[43]:


# p = all_projects_gcamp['ZIM2165_Gcamp7b_worm2-2022-12-10']
# df = p.calc_default_traces(use_paper_options=True)


# In[44]:


# all_num_ids = defaultdict(int)

# for name, proj in tqdm(all_projects_immob.items()):
#     df_traces = proj.calc_default_traces(use_paper_options=True)
#     neuron_columns = [c for c in df_traces.columns if 'neuron' not in c]
#     if 'RMED' not in neuron_columns:
#         print(name)
#     if 'AQ' in neuron_columns:
#         print(name)
#     all_num_ids[name] = len(neuron_columns)
# min(list(all_num_ids.values()))


# In[ ]:




