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
import plotly.express as px
from wbfm.utils.visualization.utils_plot_traces import add_p_value_annotation


# In[3]:


from wbfm.utils.general.utils_hardcoded import load_paper_datasets


# In[23]:


all_projects_no_light = load_paper_datasets('no_light_control_fm')


# In[6]:


all_projects_wt = load_paper_datasets(['gcamp', 'hannah_O2_fm'])


# # Look at the basic histograms: speed, rev/fwd duration, and rev/fwd frequency

# ## Angular speed

# In[44]:


from wbfm.utils.visualization.plot_summary_statistics import calc_speed_dataframe
from wbfm.utils.general.utils_paper import apply_figure_settings, plotly_paper_color_discrete_map
from wbfm.utils.general.utils_paper import data_type_name_mapping


# In[47]:


df_speed_wt = calc_speed_dataframe(all_projects_wt)
df_speed_no_light = calc_speed_dataframe(all_projects_no_light)


# In[60]:


df_speed_wt['Genotype'] = 'Wild Type'
df_speed_no_light['Genotype'] = 'No Light'
df_speed = pd.concat([df_speed_wt, df_speed_no_light])

# df_speed['Genotype'] = df_speed['Genotype'].map(data_type_name_mapping())
speed_types = [#'abs_stage_speed', 'middle_body_speed', 
               'signed_speed_angular']
for x in speed_types:
    fig = px.histogram(df_speed, x=x, facet_row='Genotype', color='Genotype', color_discrete_map=plotly_paper_color_discrete_map(), #title="Speed",#x, 
                       histnorm='probability')
    fig.update_layout(title=dict(x=0.4, y=0.99))
    # Remove facet_row annotations
    for anno in fig['layout']['annotations']:
        anno['text']=''
    fig.update_layout(showlegend=True)
    # fig.update_yaxes(dict(title="Probability", range=[0, 0.019]))
    fig.update_yaxes(dict(title="Probability", range=[0, 0.07])
                    )
    fig.update_xaxes(dict(title="Angular Velocity (degrees?/s)", 
                          range=[-0.04, 0.04]))
    fig.update_xaxes(dict(title=""), row=2, col=1)
    
    fig.update_traces(xbins=dict( # bins used for histogram
        size=0.002
    ))
    apply_figure_settings(fig, width_factor=0.4, height_factor=0.2, plotly_not_matplotlib=True)
    
    fig.show()
    
    fname = f"{x}_histogram.png"
    fname = os.path.join("fig_supp_beh", fname)
    fig.write_image(fname, scale=3)
    fig.write_image(fname.replace(".png", ".svg"))


# In[87]:


df_speed_wt


# ## Durations

# In[56]:


from wbfm.utils.visualization.plot_summary_statistics import calc_durations_dataframe


# In[57]:


df_duration_wt = calc_durations_dataframe(all_projects_wt)
df_duration_no_light = calc_durations_dataframe(all_projects_no_light)


# In[33]:


# %debug


# In[58]:


df_duration_wt['genotype'] = 'Wild Type'
df_duration_no_light['genotype'] = 'No Light'

df_duration = pd.concat([df_duration_wt, df_duration_no_light])

fps = 3.5
df_duration['BehaviorCodes.FWD'] /= fps
df_duration['BehaviorCodes.REV'] /= fps


# In[63]:


states = ['BehaviorCodes.FWD', 'BehaviorCodes.REV']
titles = ["Fwd", "Bwd"]

for x, t in zip(states, titles):
    fig = px.histogram(df_duration, x=x, facet_row='genotype', color='genotype', color_discrete_map=plotly_paper_color_discrete_map(), 
                       title=f"<br>                {t} duration", 
                       # title=f"                              {t} duration", 
                       histnorm='probability')

    fig.update_layout(title=dict(x=0.5, y=0.99))
    # Remove facet_row annotations
    for anno in fig['layout']['annotations']:
        anno['text']=''
    fig.update_layout(xaxis_title="Time (s)", showlegend=False)
    fig.update_traces(xbins=dict( # bins used for histogram
        # start=0.0,
        # end=60.0,
        size=1
    ))
    fig.update_xaxes(dict(range=[0, 90]))
    fig.update_yaxes(dict(range=[0, 0.4]), title="")
    width_factor = 0.2
    if t == 'Reversal':
        # fig.update_xaxes(dict(range=[0, 20]))
        fig.update_yaxes(showticklabels=False, overwrite=True)
        width_factor -= 0.01
    # else:
    #     fig.update_xaxes(dict(range=[0, 90]))
    #     fig.update_yaxes(dict(range=[0, 0.19]))
                        
    apply_figure_settings(fig, width_factor=width_factor, height_factor=0.2, plotly_not_matplotlib=True)
    fig.show()
    
    fname = f"duration_histogram_{x.split('.')[1]}.png"
    fname = os.path.join("fig_supp_beh", fname)
    fig.write_image(fname, scale=3)
    fig.write_image(fname.replace(".png", ".svg"))


# ## Turns

# In[70]:


from wbfm.utils.visualization.plot_summary_statistics import calc_turn_amplitude_dataframe
from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes


# In[67]:


df_turns_wt = calc_turn_amplitude_dataframe(all_projects_wt)
df_turns_no_light = calc_turn_amplitude_dataframe(all_projects_no_light)


# In[71]:


df_turns_wt['genotype'] = 'Wild Type'
df_turns_no_light['genotype'] = 'No Light'

df_turns = pd.concat([df_turns_wt, df_turns_no_light])


# In[72]:


beh_list = [BehaviorCodes.VENTRAL_TURN, BehaviorCodes.DORSAL_TURN]
cmap = [BehaviorCodes.ethogram_cmap()[beh] for beh in beh_list]


# In[80]:


df_turns['Amplitude'] = df_turns['Amplitude'].abs()

fig = px.histogram(df_turns, color="Turn Direction", histnorm='probability', color_discrete_sequence=cmap,
                  barmode='overlay', facet_row='genotype')
fig.update_layout(xaxis=dict(title="Peak Head Curvature (1/mm)"), showlegend=False)
fig.update_layout(yaxis=dict(title="Probability"), showlegend=False)
fig.update_traces(xbins=dict( # bins used for histogram
    # start=0.0,
    # end=60.0,
    size=1
))
fig.update_yaxes(range=[0, 0.25])

fig.update_layout(
    showlegend=True,
    # legend=dict(
    #   yanchor="middle",
    #   y=1.75,
    #   xanchor="left",
    #   x=0.5
    # )
)
apply_figure_settings(fig, width_factor=0.5, height_factor=0.3, plotly_not_matplotlib=True)

fig.show()

fname = f"first_head_bend_absolute_curvature_histogram.png"
fname = os.path.join("fig_supp_beh", fname)
fig.write_image(fname, scale=3)
fig.write_image(fname.replace(".png", ".svg"))


# In[81]:


fig = px.pie(df_turns, names="Turn Direction", color_discrete_sequence=cmap)

apply_figure_settings(fig, width_factor=0.15, height_factor=0.1, plotly_not_matplotlib=True)
fig.update_layout(showlegend=False)
fig.show()

fname = f"first_head_bend_absolute_curvature_pie_chart.png"
fname = os.path.join("fig_supp_beh", fname)
fig.write_image(fname, scale=3)
fig.write_image(fname.replace(".png", ".svg"))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Trajectory example

# In[7]:


from wbfm.utils.general.utils_paper import plot_trajectory


# In[84]:


p = all_projects_no_light['2025-12-03_17-34_no_light_control_worm10_auto-2025-12-03']

plot_trajectory(p, beh_annotation_kwargs=dict(include_pause=False), to_save=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Debug not finding behavior

# In[24]:





# In[30]:


p.worm_posture_class


# In[35]:


p.worm_posture_class._raw_curvature

