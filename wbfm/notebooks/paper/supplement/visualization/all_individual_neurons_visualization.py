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
from wbfm.utils.general.hardcoded_paths import load_paper_datasets


# In[2]:


from sklearn.decomposition import PCA
from wbfm.utils.visualization.plot_traces import make_grid_plot_from_dataframe
import seaborn as sns
import plotly.express as px


# In[3]:


# Load multiple datasets
all_projects_good = load_paper_datasets('gcamp_good', initialization_kwargs=dict(use_physical_time=True))


# In[3]:


all_projects_gcamp = load_paper_datasets(['gcamp', 'hannah_O2_fm'], initialization_kwargs=dict(use_physical_time=True))


# In[5]:


all_projects_immob = load_paper_datasets('immob', initialization_kwargs=dict(use_physical_time=True))


# In[6]:


all_projects_O2_immob = load_paper_datasets('hannah_O2_immob')


# In[7]:


all_projects_immob['2022-12-12_15-59_ZIM2165_immob_worm1-2022-12-12'].use_physical_time


# # Use a function to plot traces across multiple projects (with proper behavior shading)

# In[4]:


from wbfm.utils.general.utils_behavior_annotation import plot_stacked_figure_with_behavior_shading_using_plotly


# ## Plot

# In[6]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_gcamp, 
                                                             ['AQR', 'BAG', 'URX'], full_path_title=True,
                                                             to_save=True, trace_kwargs=dict(use_paper_options=True), combine_neuron_pairs=True)
fig.show()


# In[9]:


# fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_gcamp, 
#                                                              ['SMDDR', 'SMDDL', 'SMDVL', 'SMDVR', 'RIVL', 'RIVR', 'AVAL', 'VG_post_turning_L', 'VG_post_turning_R', 'VB02'], 
#                                                              to_save=True, trace_kwargs=dict(use_paper_options=True), combine_neuron_pairs=False)
# fig.show()


# ## Plot: immobilized

# In[15]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_immob, ['AVAL', 'AVAR', 'DB01', 'VB02'], full_path_title=True,
                                                             to_save=True, trace_kwargs=dict(use_paper_options=True), fname_suffix='immob', combine_neuron_pairs=False)
fig.show()


# In[ ]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_immob, ['AVAL', 'AVBL', 'AVBR', 'RIS'], to_save=False, trace_kwargs=dict(use_paper_options=True))
fig.show()


# In[ ]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_immob, ['AVAL', 'AVBL', 'RIVL', 'RID'], to_save=False, trace_kwargs=dict(use_paper_options=True))
fig.show()


# In[ ]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_immob, ['AVAL', 'AVBL', 'RIVL', 'RMED', 'RMEV'], to_save=False, trace_kwargs=dict(use_paper_options=True))
fig.show()


# # Plot: immobilized with stimulation

# In[11]:


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_O2_immob, ['AVAL', 'AVAR', 'BAGL', 'BAGR', 'AQR'], to_save=False, trace_kwargs=dict(use_paper_options=True))
fig.show()


# In[ ]:





# # DEBUG: why is the immobilized frame rate slightly wrong?

# In[24]:


all_projects_immob['ZIM2165_immob_adj_set_2_worm3-2022-11-30']


# In[ ]:




