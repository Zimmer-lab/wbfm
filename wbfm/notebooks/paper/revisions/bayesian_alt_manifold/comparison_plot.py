#!/usr/bin/env python
# coding: utf-8

# In[116]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm
import os 
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
import plotly.express as px

from wbfm.utils.general.utils_paper import apply_figure_settings

from wbfm.utils.general.utils_hardcoded import neurons_with_less_confident_ids


# In[40]:


fname = 'df_params.h5'
df_main = pd.read_hdf(fname)
df_main['Model Type'] = 'PCA'

fname = os.path.join('fig5_discrete', 'df_params.h5')
df_discrete = pd.read_hdf(fname)
df_discrete['Model Type'] = 'Discrete'

fname = os.path.join('fig5_cca_continuous', 'df_params.h5')
df_cca = pd.read_hdf(fname)
df_cca['Model Type'] = 'CCA'


# In[41]:


df = pd.concat([df_main, df_discrete, df_cca])


# In[43]:


df['Category']


# In[165]:


_df = df[df['Dataset Type'] == "Freely Moving (GCaMP, residual)"].copy()

_df = _df.melt(id_vars=['neuron_name', 'Model Type'], value_vars=['hyper_pca0_p_value_is_significant', 'Behavior is above GFP', 'Hierarchy is above GFP'])

# Remove neurons with no results
bad_neurons = _df.groupby('neuron_name')['value'].sum() == 0
good_neurons = bad_neurons.index[~bad_neurons].values
# _df = _df[_df['neuron_name'].isin(good_neurons)]

_df['Neuron Name'] = _df['neuron_name'].map(neurons_with_less_confident_ids(return_mapping=True, combine_left_right=False))

_df['Category'] = _df['variable'].map({'hyper_pca0_p_value_is_significant': 'Mode 1 significance',
                                      'Hierarchy is above GFP': 'Hierarchy > GFP',
                                      'Behavior is above GFP': 'Behavior > GFP'})

# Add summary of all models values
_df2 = _df.groupby(['neuron_name', 'variable'])['value'].sum().reset_index()
_df2['Model Type'] = 'All models together'

_df


# In[108]:


_df2


# In[169]:


import plotly.express as px

# 1. Ensure your boolean values are integers for the color scale
_df['value_int'] = _df['value'].astype(int)

# 2. Create the Faceted Grid
fig = px.scatter(
    _df,
    x='Category',
    y='Neuron Name',
    color='value_int',
    facet_col='Model Type',
    # Using 'square' symbol makes it look like a checkbox grid
    symbol_sequence=['square'], 
    color_continuous_scale=[[0, 'white'], [1, '#1f77b4']], # White for False, Blue for True
    height=1100, # Adjust based on number of neurons
    width=330
)

# 3. Scientific Styling
fig.update_traces(
    marker=dict(size=12, line=dict(width=1, color='black')), # Add borders to squares
)

fig.update_layout(
    plot_bgcolor='white', # Clean white background for papers
    coloraxis_showscale=False, # We don't need a legend for binary data
    margin=dict(t=50, b=50, l=100, r=50)
)

# Sorting
neuron_order = _df.groupby('Neuron Name')['value_int'].sum().sort_values(ascending=True).index
fig.update_yaxes(
    categoryorder='array', 
    categoryarray=neuron_order,
    # ticks='array'
)

# 4. Clean up the Facet Labels (remove "Model Type=")
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.for_each_annotation(lambda a: a.update(y=1.12, font=dict(size=16)))#, xanchor='left'))

# 5. Ensure X-axis is at the top for readability

# apply_figure_settings(fig, width_factor=0.3, height_factor=1)
fig.update_xaxes(side="top", showgrid=False, gridcolor='lightgrey', tickangle=45, title="",
                title_standoff=10, ticks='outside')
fig.update_yaxes(showgrid=False, gridcolor='lightgrey',
                range=[-0.5, _df['neuron_name'].nunique() - 0.5]
                )
fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

fig.show()

to_save = True
if to_save:
    fname = 'individual_model_comparison.png'
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)
    fname = Path(fname).with_suffix('.html')
    fig.write_html(fname)


# In[106]:


import plotly.express as px

# 1. Ensure your boolean values are integers for the color scale
_df2['value_int'] = _df2['value'].astype(int)

discrete_colors = [
    [0.0, "white"],   # 0
    [0.25, "white"],
    [0.25, "#d1e5f0"], # 1
    [0.5, "#d1e5f0"],
    [0.5, "#67a9cf"],  # 2
    [0.75, "#67a9cf"],
    [0.75, "#02818a"], # 3
    [1.0, "#02818a"]
]

# 2. Create the Faceted Grid
fig = px.scatter(
    _df2,
    x='variable',
    y='neuron_name',
    color='value_int',
    # facet_col='Model Type',
    # Using 'square' symbol makes it look like a checkbox grid
    symbol_sequence=['square'], 
    color_continuous_scale=discrete_colors,
    # color_continuous_scale=[[0, 'white'], [1, '#1f77b4']], # White for False, Blue for True
    height=1000, # Adjust based on number of neurons
    width=300
)

# 3. Scientific Styling
fig.update_traces(
    marker=dict(size=12, line=dict(width=1, color='black')), # Add borders to squares
)

fig.update_layout(
    plot_bgcolor='white', # Clean white background for papers
    # coloraxis_showscale=False, # We don't need a legend for binary data
    margin=dict(t=50, b=50, l=100, r=50)
)

# 4. Clean up the Facet Labels (remove "Model Type=")
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

# 5. Ensure X-axis is at the top for readability
fig.update_xaxes(side="top", showgrid=False, gridcolor='lightgrey', tickangle=-45)
fig.update_yaxes(showgrid=False, gridcolor='lightgrey')
fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

fig.show()

to_save = True
if to_save:
    fname = 'total_model_comparison.png'
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)
    fname = Path(fname).with_suffix('.html')
    fig.write_html(fname)


# In[ ]:




