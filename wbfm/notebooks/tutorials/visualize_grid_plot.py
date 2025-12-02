#!/usr/bin/env python
# coding: utf-8

# In[ ]:



get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.general.utils_paper import apply_figure_settings
import plotly.express as px


# # Step 1: using my project class
# 
# Note: you will need to update the path if you are not on the cluster. If you have /scratch mounted, this might work:
# 
# fname = "Z:/neurobiology/zimmer/fieseler/wbfm_projects/2022-11-27_spacer_7b_2per_agar/ZIM2165_Gcamp7b_worm1-2022_11_28/project_config.yaml"

# In[ ]:


# Use a project with external stimulus as an additional behavior annotation
fname = "/lisc/data/scratch/neurobiology/zimmer/brenner/wbfm_projects/analyze/immobilized_wt/2023-09-19_11-42_worm1-2023-09-19/project_config.yaml"
project_data_gcamp = ProjectData.load_final_project_data_from_config(fname)


# # Step 2: Use a function to make a grid plot

# In[ ]:


from wbfm.utils.visualization.plot_traces import make_grid_plot_from_project

# trace_options = dict(use_paper_options=True)  # If you like
trace_options = dict()
fig = make_grid_plot_from_project(project_data_gcamp, trace_kwargs=trace_options,
                                  min_nonnan=0.95,
                                  to_save=False)


# In[ ]:


# There are many options for this function, use help to check them out:
help(make_grid_plot_from_project)


# # Advanced: adding stimulus
# 
# Requires a .csv file with the starts and ends of the stimulus period in Volumes. Example:
# 
# /lisc/data/scratch/neurobiology/zimmer/brenner/wbfm_projects/analyze/immobilized_wt/2023-09-19_11-42_worm1-2023-09-19/3-tracking/manual_annotation/stimulus.csv
# 

# ## Sort by correlation to a stimulus
# 

# In[ ]:


from wbfm.utils.visualization.plot_traces import make_grid_plot_from_project
from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes

# trace_options = dict(use_paper_options=True)  # If you like
trace_options = dict()
sorting_options = dict(behavioral_correlation_shading='stimulus', sort_using_shade_value=True)
base_options = dict(min_nonnan=0.99, to_save=False, trace_kwargs=trace_options)
fig = make_grid_plot_from_project(project_data_gcamp, **sorting_options, **base_options)


# ## Plot the behavior as a shaded overlay (not sorting)

# In[ ]:


from wbfm.utils.visualization.plot_traces import make_grid_plot_from_project
from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes

trace_options = dict()
base_options = dict(min_nonnan=0.99, to_save=False, trace_kwargs=trace_options)
shade_plot_options = dict(shade_plot_kwargs=dict(additional_shaded_states=[BehaviorCodes.STIMULUS], behaviors_to_ignore=[BehaviorCodes.REV]))

fig = make_grid_plot_from_project(project_data_gcamp, **base_options, **shade_plot_options, **sorting_options)


# # Advanced: plot another time series on top of this dataframe
# 
# In principle, any dataframe can be plotted on top of this grid. We will make a dummy one here
# 
# Note: takes a long time! Maybe a couple of minutes to plot

# ## Option 1: Behavior on top of traces

# In[ ]:


# Make dataframe to add
import numpy as np

# First get the traces
df_traces = project_data_gcamp.calc_default_traces()

# Use a behavior time series as the second trace
y_stimulus = project_data_gcamp.worm_posture_class.calc_behavior_from_alias('stimulus')
# The columns should be the same as the original trace dataframe
df_behavior = df_traces.copy()
df_behavior[:] = np.array([y_stimulus] * df_behavior.shape[1]).T

df_behavior.head()


# In[ ]:


# Use a different function to easily plot two dataframes on top of each other
from wbfm.utils.visualization.plot_traces import make_grid_plot_from_two_dataframes

fig = make_grid_plot_from_two_dataframes(df_traces, df_behavior)


# ## Option 2: Single neuron trace on top of all others

# In[ ]:


# Make dataframe to add
import numpy as np

# First get the traces
df_traces = project_data_gcamp.calc_default_traces()

# Use a single neuron as the second trace
y = df_traces['neuron_001']
# The columns should be the same as the original trace dataframe
df_baseline_neuron = df_traces.copy()
df_baseline_neuron[:] = np.array([y] * df_baseline_neuron.shape[1]).T

df_baseline_neuron.head()


# In[ ]:


# Use a different function to easily plot two dataframes on top of each other
from wbfm.utils.visualization.plot_traces import make_grid_plot_from_two_dataframes

fig = make_grid_plot_from_two_dataframes(df_traces, df_baseline_neuron)


# In[ ]:




