#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.general.utils_paper import apply_figure_settings
import plotly.express as px


# # Step 1: using my project class
# 
# Note: you will need to update the path if you are not on the cluster. If you have /scratch mounted, this might work:
# 
# fname = "Z:/neurobiology/zimmer/fieseler/wbfm_projects/2022-11-27_spacer_7b_2per_agar/ZIM2165_Gcamp7b_worm1-2022_11_28/project_config.yaml"

# In[2]:


# Use a project with external stimulus as an additional behavior annotation
fname = "/lisc/data/scratch/neurobiology/zimmer/brenner/wbfm_projects/analyze/immobilized_wt/2023-09-19_11-42_worm1-2023-09-19/project_config.yaml"
project_data_gcamp = ProjectData.load_final_project_data_from_config(fname)


# # Step 2: get the traces as a pandas dataframe

# In[3]:


# For convinience, use pre-calculated traces that are used in the paper
df_traces = project_data_gcamp.calc_default_traces(use_paper_options=True)


# In[4]:


df_traces.head()


# # Step 3: plot your favorite (no behavior at first)
# 
# I like the plotly library, because it is interactive.

# In[5]:


neuron_to_plot = 'AVAL'
fig = px.line(df_traces, y=neuron_to_plot)
fig.show()


# ## Additional options for making it prettier

# In[6]:


neuron_to_plot = 'AVAL'
fig = px.line(df_traces, y=neuron_to_plot, color_discrete_sequence=['black'])

project_data_gcamp.use_physical_time = True
xlabel = project_data_gcamp.x_label_for_plots

fig.update_xaxes(title_text=xlabel)

# This line adds the behavior shading
project_data_gcamp.shade_axis_using_behavior(plotly_fig=fig)


apply_figure_settings(fig, height_factor=0.2)
fig.show()


# In[10]:


from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes

neuron_to_plot = 'URXL'
fig = px.line(df_traces, y=neuron_to_plot, color_discrete_sequence=px.colors.qualitative.D3)

project_data_gcamp.use_physical_time = True
xlabel = project_data_gcamp.x_label_for_plots

fig.update_xaxes(title_text=xlabel)

# This line adds the behavior shading, now with additional ones, and with reversal removed
# See BehaviorCodes for all behavior options
project_data_gcamp.shade_axis_using_behavior(plotly_fig=fig, 
                                             additional_shaded_states=[BehaviorCodes.STIMULUS],
                                             default_reversal_shading=False)


apply_figure_settings(fig, height_factor=0.2)
fig.show()


# # Step 4 (optional): save

# In[8]:


fname = f"{neuron_to_plot}_trace.png"
fig.write_image(fname)

