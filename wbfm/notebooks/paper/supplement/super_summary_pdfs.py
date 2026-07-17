#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[9]:


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
import surpyval
from wbfm.utils.general.utils_paper import apply_figure_settings
from wbfm.utils.general.utils_paper import plotly_paper_color_discrete_map
import plotly.express as px


# In[3]:


# fname = "/scratch/neurobiology/zimmer/Charles/dlc_stacks/2022-11-27_spacer_7b_2per_agar/ZIM2165_Gcamp7b_worm1-2022_11_28/project_config.yaml"
# Manually corrected version
fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/manually_annotated/paper_data/ZIM2165_Gcamp7b_worm1-2022_11_28/project_config.yaml"
project_data_gcamp = ProjectData.load_final_project_data_from_config(fname)


# # Plot everything

# In[4]:


from wbfm.utils.visualization.plot_traces import make_full_summary_interactive_plot


# In[166]:


fig = make_full_summary_interactive_plot(project_data_gcamp, to_save=False, to_show=True,
                                                      apply_figure_size_settings=True, showlegend=True, crop_x_axis=False,
                                                       # row_heights=[0.25, 0.05, 0.2, 0.2, 0.2]
                                        )

to_save = True
if to_save:
    fname = os.path.join("behavior", "all_information summary.png")
    fig.write_image(fname, scale=5)
    fname = str(Path(fname).with_suffix('.svg'))
    fig.write_image(fname)
    fname = str(Path(fname).with_suffix('.pdf'))
    fig.write_image(fname)


# In[163]:


# [f['xaxis'] for f in fig.to_dict()['data']]


# # Make for all projects

# In[1]:


from wbfm.utils.general.hardcoded_paths import load_all_paper_datasets
all_projects = load_all_paper_datasets()


# In[2]:


from wbfm.utils.visualization.plot_traces import make_full_summary_interactive_plot


# In[31]:


for name, project in tqdm(all_projects.items()):
    fname = os.path.join("all_dataset_summaries", name)
    fname = f'{fname}.png'
    if os.path.exists(fname):
        continue
    
    fig = make_full_summary_interactive_plot(project, to_save=True, to_show=False,
                                                          apply_figure_size_settings=True, showlegend=True, crop_x_axis=False)
    
    fig.write_image(fname, scale=5)
    fname = str(Path(fname).with_suffix('.svg'))
    fig.write_image(fname)
    fname = str(Path(fname).with_suffix('.pdf'))
    fig.write_image(fname)


# In[33]:


fig = make_full_summary_interactive_plot(project, to_save=False, to_show=True,
                                                      apply_figure_size_settings=True, showlegend=True, crop_x_axis=False)


# # Debug high dimensional (messed up) times in the curvature

# In[44]:


import plotly.express as px
from sklearn.decomposition import PCA


# In[34]:


df = project.worm_posture_class.curvature()


# In[61]:


pca = PCA(n_components=10, whiten=True)
df_subset = df.iloc[:, 3:-3]
df_subset_proj = pca.inverse_transform(pca.fit_transform(df_subset))


# In[62]:


df_residual = df_subset - df_subset_proj
px.line(np.linalg.norm(df_residual, axis=1))


# In[63]:


xy = project.worm_posture_class.centerline_absolute_coordinates(nan_high_dimensional=True)


# In[73]:


px.scatter(x=xy.loc[:, (50, 'X')], y=xy.loc[:, (50, 'Y')])


# In[75]:


# xy_raw = project.worm_posture_class.centerline_absolute_coordinates(nan_high_dimensional=False)


# In[80]:


px.scatter(xy.loc[:, (50, 'X')].diff().abs().values)


# In[81]:


xy.loc[:, (50, 'X')].diff().abs().median()

