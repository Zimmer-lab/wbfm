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
all_projects_good = load_paper_datasets('gcamp_good')


# In[4]:


all_projects_immob = load_paper_datasets('immob')


# In[ ]:





# # PCA plots for immobilized

# In[5]:


from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
from wbfm.utils.visualization.utils_plot_traces import modify_dataframe_to_allow_gaps_for_plotly
import plotly.graph_objects as go
from wbfm.utils.general.utils_paper import apply_figure_settings
from wbfm.utils.general.utils_paper import behavior_name_mapping


# In[13]:


plot_3d = False

output_foldername = 'manifolds'

# Only actually plot a couple
# good_names = ['ZIM2165_immob_adj_set_2_worm2-2022-11-30', 
#               '2022-12-13_15-16_ZIM2165_immob_worm9-2022-12-13']

for name, p in tqdm(all_projects_immob.items()):
    # if name not in good_names:
    #     continue
        
    modes_to_plot = [0, 1, 2]
    beh_annotation_kwargs = {}
    ethogram_cmap_kwargs = {}
    
    X_r, var_explained = p.calc_pca_modes(n_components=3, multiply_by_variance=False)
    df_latents = pd.DataFrame(X_r)
    
    beh_annotation = dict(fluorescence_fps=True, reset_index=True, include_collision=False, include_turns=True,
                          include_head_cast=False, include_pause=False, include_slowing=False)
    beh_annotation.update(beh_annotation_kwargs)
    state_vec = p.worm_posture_class.beh_annotation(**beh_annotation)
    state_vec[1041] = BehaviorCodes.FWD
    state_vec = state_vec.apply(BehaviorCodes.convert_to_simple_states)
    df_latents['state'] = state_vec.values  # Ignore the index here, since it may not be physical time
    # ethogram_cmap_kwargs.setdefault('include_turns', beh_annotation['include_turns'])
    # ethogram_cmap_kwargs.setdefault('include_quiescence', beh_annotation['include_pause'])
    # ethogram_cmap_kwargs.setdefault('include_collision', beh_annotation['include_collision'])
    ethogram_cmap = BehaviorCodes.ethogram_cmap(**ethogram_cmap_kwargs, include_reversal_turns=False)
    df_out, col_names = modify_dataframe_to_allow_gaps_for_plotly(df_latents, modes_to_plot, 'state')
    state_codes = df_latents['state'].unique()
    
    # Actually plot
    phase_plot_list = []
    # Loop over behaviorally-colored short segments and plot
    for i, state_code in enumerate(state_codes):
        legend_name = behavior_name_mapping()[state_code.individual_names[0]]
        scatter_opt = dict(mode='lines', name=legend_name,
                           line=dict(color=ethogram_cmap.get(state_code, None), width=4))
        if plot_3d:
            phase_plot_list.append(go.Scatter3d(x=df_out[col_names[0][i]], y=df_out[col_names[1][i]], z=df_out[col_names[2][i]],
                             **scatter_opt))
        else:
            phase_plot_list.append(go.Scatter(x=df_out[col_names[0][i]], y=df_out[col_names[1][i]],
                           **scatter_opt))

    # Need to manually add the list of traces instead of plotly express
    print(name)
    fig = go.Figure()#layout=dict(height=800, width=1000))
    fig.add_traces(phase_plot_list)
    fig.update_xaxes(title=f'Neuronal component 1<br>(immobilized; PCA; {100*var_explained[0]:.01f}%)')
    fig.update_yaxes(title=f'Neuronal component 2<br>(immobilized; PCA; {100*var_explained[1]:.01f}%)')

    if name == good_names[0]:
        width_factor = 0.6 #if len(good_names) > 1 else 1.0
        # width_factor = 0.6
        fig.update_layout(showlegend=True)
    else:
        width_factor = 0.4
        fig.update_layout(showlegend=False)
    
    apply_figure_settings(fig, plotly_not_matplotlib=True, width_factor=width_factor, height_factor=0.25)
    
    fig.show()
    
    # Also just time series
    # fig2 = px.line(X_r)
    # fig2.show()
    
    # Save
    fname = os.path.join(output_foldername, f'immob-{name}-pca{3 if plot_3d else 2}d.png')
    fig.write_image(fname)
    # fname = fname.replace('.png', '.html')
    # fig.write_html(fname)
    
    # # Save
    # fname = os.path.join(output_foldername, f'{name}-pca-lines.png')
    # fig2.write_image(fname)
    # fname = fname.replace('.png', '.html')
    # fig2.write_html(fname)
    # break


# # Also freely moving

# In[14]:


plot_3d = False

output_foldername = 'manifolds'

# Only actually plot a couple
# good_names = ['ZIM2165_immob_adj_set_2_worm2-2022-11-30', 
#               '2022-12-13_15-16_ZIM2165_immob_worm9-2022-12-13']

for name, p in tqdm(all_projects_good.items()):
    # if name not in good_names:
    #     continue
        
    modes_to_plot = [0, 1, 2]
    beh_annotation_kwargs = {}
    ethogram_cmap_kwargs = {}
    
    X_r, var_explained = p.calc_pca_modes(n_components=3, multiply_by_variance=False)
    df_latents = pd.DataFrame(X_r)
    
    beh_annotation = dict(fluorescence_fps=True, reset_index=True, include_collision=False, include_turns=True,
                          include_head_cast=False, include_pause=False, include_slowing=False)
    beh_annotation.update(beh_annotation_kwargs)
    state_vec = p.worm_posture_class.beh_annotation(**beh_annotation)
    # state_vec[1041] = BehaviorCodes.FWD
    state_vec = state_vec.apply(BehaviorCodes.convert_to_simple_states)
    df_latents['state'] = state_vec.values  # Ignore the index here, since it may not be physical time
    # ethogram_cmap_kwargs.setdefault('include_turns', beh_annotation['include_turns'])
    # ethogram_cmap_kwargs.setdefault('include_quiescence', beh_annotation['include_pause'])
    # ethogram_cmap_kwargs.setdefault('include_collision', beh_annotation['include_collision'])
    ethogram_cmap = BehaviorCodes.ethogram_cmap(**ethogram_cmap_kwargs, include_reversal_turns=False)
    df_out, col_names = modify_dataframe_to_allow_gaps_for_plotly(df_latents, modes_to_plot, 'state')
    state_codes = df_latents['state'].unique()
    
    # Actually plot
    phase_plot_list = []
    # Loop over behaviorally-colored short segments and plot
    for i, state_code in enumerate(state_codes):
        legend_name = behavior_name_mapping()[state_code.individual_names[0]]
        scatter_opt = dict(mode='lines', name=legend_name,
                           line=dict(color=ethogram_cmap.get(state_code, None), width=4))
        if plot_3d:
            phase_plot_list.append(go.Scatter3d(x=df_out[col_names[0][i]], y=df_out[col_names[1][i]], z=df_out[col_names[2][i]],
                             **scatter_opt))
        else:
            phase_plot_list.append(go.Scatter(x=df_out[col_names[0][i]], y=df_out[col_names[1][i]],
                           **scatter_opt))

    # Need to manually add the list of traces instead of plotly express
    print(name)
    fig = go.Figure()#layout=dict(height=800, width=1000))
    fig.add_traces(phase_plot_list)
    fig.update_xaxes(title=f'Neuronal component 1<br>(freely moving; PCA; {100*var_explained[0]:.01f}%)')
    fig.update_yaxes(title=f'Neuronal component 2<br>(freely moving; PCA; {100*var_explained[1]:.01f}%)')

    if name == good_names[0]:
        width_factor = 0.6 #if len(good_names) > 1 else 1.0
        # width_factor = 0.6
        fig.update_layout(showlegend=True)
    else:
        width_factor = 0.4
        fig.update_layout(showlegend=False)
    
    apply_figure_settings(fig, plotly_not_matplotlib=True, width_factor=width_factor, height_factor=0.25)
    
    fig.show()
    
    # Also just time series
    # fig2 = px.line(X_r)
    # fig2.show()
    
    # Save
    fname = os.path.join(output_foldername, f'fm-{name}-pca{3 if plot_3d else 2}d.png')
    fig.write_image(fname)
    # fname = fname.replace('.png', '.html')
    # fig.write_html(fname)
    
    # # Save
    # fname = os.path.join(output_foldername, f'{name}-pca-lines.png')
    # fig2.write_image(fname)
    # fname = fname.replace('.png', '.html')
    # fig2.write_html(fname)
    # break


# In[ ]:




