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
from wbfm.utils.projects.finished_project_data import ProjectData

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


# In[6]:


fname = "/lisc/scratch/neurobiology/zimmer/schaar/wbfm/results/20250301/worm1-2025-03-07/project_config.yaml"
# Manually corrected version
# fname = "/lisc/scratch/neurobiology/zimmer/ItamarLev/feedback_story/WBFM/test_behavior/2023-09-06_13-56_GCamP7b_2per_worm3-2023-09-06"
project_data_gcamp = ProjectData.load_final_project_data_from_config(fname)


# # Read behavior

# In[7]:


project_data_gcamp.worm_posture_class.stage_position()


# In[9]:


project_data_gcamp.project_config.get_behavior_config()


# In[ ]:





# In[ ]:





# # Turn annotation

# In[69]:


# del project_data_gcamp.worm_posture_class


# In[77]:


project_data_gcamp.worm_posture_class.beh_annotation_already_converted_to_fluorescence_fps


# In[80]:


project_data_gcamp.worm_posture_class.beh_annotation(fluorescence_fps=True)


# In[79]:


get_ipython().run_line_magic('debug', '')


# In[74]:


project_data_gcamp.worm_posture_class.filename_beh_annotation


# In[ ]:





# # Which eigenworm phase correlates to which body segment?

# In[5]:


Xy.head()


# In[10]:


import sklearn, math

x = ['eigenworm1', 'eigenworm2']
all_theta = []

for i in range(1, 100):
    y = f'curvature_{i}'
    # print(y)

    Xy_for_model = Xy[x].join(Xy[y]).dropna()

    model = sklearn.linear_model.LinearRegression(fit_intercept=False)
    model.fit(Xy_for_model[x], Xy_for_model[y])

    # Convert to polar coordinates
    cx, cy = model.coef_
    r = np.sqrt(cx**2 + cy**2)
    theta = np.arctan2(cy, cx)
    all_theta.append(theta)
    # print(r, 360*theta/(2*math.pi))
fig = px.line(all_theta)
fig.add_hline(y=math.pi/2)
fig.add_hline(y=math.pi)
fig.add_hline(y=-math.pi/2)


# In[12]:





# In[8]:


all_y = []
for i in [5, 10, 15, 20]:
    y = f'curvature_{i}'
    all_y.append(y)
px.line(Xy[all_y])


# In[ ]:




