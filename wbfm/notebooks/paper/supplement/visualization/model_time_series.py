#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm
import os 
import numpy as np
from pathlib import Path
import plotly.express as px


# In[2]:


from wbfm.utils.general.hardcoded_paths import get_hierarchical_modeling_dir

fname = os.path.join(get_hierarchical_modeling_dir(), 'data.h5')
print(fname)
Xy = pd.read_hdf(fname)

fname = os.path.join(get_hierarchical_modeling_dir(gfp=True), 'data.h5')
print(fname)
Xy_gfp = pd.read_hdf(fname)


# In[3]:


import arviz as az
from wbfm.utils.general.hardcoded_paths import neurons_with_confident_ids

def load_all_traces(foldername):
    fnames = neurons_with_confident_ids()
    all_traces = {}
    for neuron in tqdm(fnames):
        trace_fname = os.path.join(foldername, f'{neuron}_hierarchical_pca_trace.nc')
        if os.path.exists(trace_fname):
            try:
                trace = az.from_netcdf(trace_fname)
                all_traces[neuron] = trace
            except ValueError:
                pass
    return all_traces

parent_folder = '/lisc/data/scratch/neurobiology/zimmer/fieseler/paper/hierarchical_modeling'
# suffix = '_only_eigenworms'
# suffix = '_eigenworms34_speed'
# suffix = ''
suffix = '_new_ids_only_eigenworms'
            
foldername = os.path.join(parent_folder, f'output{suffix}')
all_traces_gcamp = load_all_traces(foldername)

# foldername = os.path.join(f'{parent_folder}_gfp', f'output{suffix}')
# all_traces_gfp = load_all_traces(foldername)


# # Look at individual reconstructions

# In[6]:


from wbfm.utils.external.utils_arviz import plot_ts, plot_model_elements


# In[7]:


# Surprisingly large curvature encoding
fig = plot_model_elements(all_traces_gcamp['URXL'])


# In[8]:


# Should be the largest oscillating neuron
fig = plot_model_elements(all_traces_gcamp['VB02'])


# In[9]:


# Possibly not really oscillation, just modeled by a sigmoid
fig = plot_model_elements(all_traces_gcamp['RID'])


# In[10]:


# Not modeled well by behavior alone, only hierarchy
fig = plot_model_elements(all_traces_gcamp['URYVR'])


# In[11]:


# Not modeled well by behavior alone, only hierarchy
fig = plot_model_elements(all_traces_gcamp['URADL'])

