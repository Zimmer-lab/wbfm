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


# In[3]:


fname = "/lisc/data/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-11-27_spacer_7b_2per_agar/ZIM2165_Gcamp7b_worm1-2022_11_28/project_config.yaml"
# Manually corrected version
# fname = "/scratch/neurobiology/zimmer/Charles/dlc_stacks/manually_annotated/paper_data/ZIM2165_Gcamp7b_worm1-2022_11_28/project_config.yaml"
project_data_gcamp = ProjectData.load_final_project_data_from_config(fname)


# # Visualize the body postures at which the hilbert code triggers

# In[4]:


from wbfm.utils.traces.triggered_averages import FullDatasetTriggeredAverages


# In[5]:


trace_opt = dict(use_paper_options=True, residual_mode='pca', interpolate_nan=True)
trigger_opt = dict(use_hilbert_phase=True, state=None)
trigger_class = FullDatasetTriggeredAverages.load_from_project(project_data_gcamp, trace_opt=trace_opt, trigger_opt=trigger_opt)


# In[6]:


# trigger_class.plot_events_over_trace('VB02', fig_opt=dict(figsize=(50, 10)))


# In[7]:


fpv = project_data_gcamp.physical_unit_conversion.frames_per_volume
print(trigger_class.ind_class.idx_onsets)
idx_beh_time_series = (trigger_class.ind_class.idx_onsets.copy() * fpv).astype(int)
idx_beh_time_series


# In[8]:


beh_video = project_data_gcamp.worm_posture_class.raw_behavior_video
beh_video.shape


# In[11]:


# kymo.mean().abs().mean()


# In[14]:


# obj.onset_vector()


# In[13]:


kymo = project_data_gcamp.worm_posture_class.curvature(fluorescence_fps=True).copy()
_df = (kymo-kymo.mean()).reset_index(drop=True).T
# _df = kymo.reset_index(drop=True).T
# _df = (kymo-kymo.mean()).T
# px.imshow(_df, zmin=-0.01, zmax=0.01, color_continuous_scale='RdBu')

trace = trigger_class.df_traces['VB02'].copy().reset_index(drop=True)


obj = trigger_class.ind_class
fig = px.line({'raw':_df.loc[15, :]/_df.loc[15, :].max(), 
               'hilbert':obj.behavioral_annotation/obj.behavioral_annotation.max(), 
               'binary':obj.cleaned_binary_state.copy().astype(float),
               'vb02': trace/trace.max(),
               'final_onsets': obj.onset_vector()})
project_data_gcamp.shade_axis_using_behavior(plotly_fig=fig)
fig.show()


# In[ ]:


obj.onset_vector()


# In[ ]:


obj.idx_onsets


# In[17]:


from ipywidgets import interact

def f(i=0):
    t = idx_beh_time_series[i]
    fig = px.imshow(beh_video[t], width=1000, height=1000, title=f"Index: {t}")
    fig.show()

interact(f, i=(0, len(idx_beh_time_series)))


# ## Why are not all events captured?

# In[ ]:


trigger_class.ind_class.triggered_average_indices(dict_of_events_to_keep=None)


# In[ ]:


get_ipython().run_line_magic('debug', '')


# In[ ]:




