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


# In[3]:


# Load multiple datasets
from wbfm.utils.general.hardcoded_paths import load_paper_datasets
all_projects_gcamp = load_paper_datasets(['gcamp', 'hannah_O2_fm'])


# In[4]:


all_projects_gfp = load_paper_datasets('gfp')


# In[5]:


all_projects_immob = load_paper_datasets('immob')


# In[ ]:





# # Just print the paths

# In[6]:


print("="*30, "GCAMP", "="*30)
for p in all_projects_gcamp.values():
    print(p.project_dir)

print("="*30, "GFP", "="*30)
for p in all_projects_gfp.values():
    print(p.project_dir)

print("="*30, "IMMOB", "="*30)
for p in all_projects_immob.values():
    print(p.project_dir)
    


# In[ ]:




