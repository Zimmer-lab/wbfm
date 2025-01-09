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


fname = "/lisc/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-11-27_spacer_7b_2per_agar/ZIM2165_Gcamp7b_worm1-2022_11_28/project_config.yaml"
project_data_gcamp = ProjectData.load_final_project_data_from_config(fname)


# # Step 2: get the traces as a pandas dataframe

# In[3]:


# For convinience, use pre-calculated traces that are used in the paper
df_traces = project_data_gcamp.calc_default_traces(use_paper_options=True)


# In[4]:


df_traces.head()


# # Step 3: plot your favorite!
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
project_data_gcamp.shade_axis_using_behavior(plotly_fig=fig)
apply_figure_settings(fig, height_factor=0.2)
fig.show()


# # Step 3 (alternate): plot multiple neurons

# In[7]:


neuron_to_plot = ['AVAL', 'AVAR', 'RID']
fig = px.line(df_traces, y=neuron_to_plot)
fig.show()


# # Step 4 (optional): save

# In[8]:


fname = f"{neuron_to_plot}_trace.png"
fig.write_image(fname)

