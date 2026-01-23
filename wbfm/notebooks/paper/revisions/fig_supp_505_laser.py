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


# In[4]:


all_projects_505_488_505_immob = load_paper_datasets('505_488_505_immob')


# In[5]:


all_projects_505_488_505_immob_inactive = load_paper_datasets('505_488_505_immob_inactive')


# In[6]:


all_projects_488_505_488_immob = load_paper_datasets('488_505_488_immob')


# In[7]:


all_projects_488_505_488_immob_inactive = load_paper_datasets('488_505_488_immob_inactive')


# In[8]:


all_projects_505_488_505_fm = load_paper_datasets('505_488_505_fm')


# In[9]:


all_projects_488_505_488_fm = load_paper_datasets('488_505_488_fm')


# # Freely moving laser switch experiments

# In[10]:


from wbfm.utils.projects.finished_project_data import split_project_data_in_time
from wbfm.utils.general.utils_paper import split_time_series_with_laser_switches, plotly_paper_color_discrete_map, apply_figure_settings
from wbfm.utils.visualization.plot_summary_statistics import calc_speed_dataframe


# In[11]:


df_reversals = []
df_speed = []

laser_wavelengths = [488, 505, 488]
for name, p in all_projects_488_505_488_fm.items():
    # For each project, split it into 3 and then append to the appropriate list
    starts_stops = split_time_series_with_laser_switches(p.green_traces)
    all_segments = split_project_data_in_time(p, starts_stops, verbose=0)

    this_dict = {}
    for i, (laser_wavelength, seg) in enumerate(zip(laser_wavelengths, all_segments)):
        raw_num_rev = len(seg.worm_posture_class.get_starts_and_ends_of_reversals()[0])
        raw_pause_length = seg.worm_posture_class.calc_behavior_from_alias('pause').sum() / p.physical_unit_conversion.volumes_per_second
        num_minutes = p.num_frames / p.physical_unit_conversion.volumes_per_second / 60
        df_reversals.append({'dataset_name': name, 
                             'position': i, 
                             'Laser wavelength': laser_wavelength, 
                             'raw_num_rev': raw_num_rev,
                             'raw_pause_length': raw_pause_length,
                             'num_minutes': num_minutes,
                             'rev_per_minute': raw_num_rev / num_minutes,
                             'pause_per_minute': raw_pause_length / num_minutes})

        _df_speed = calc_speed_dataframe({seg.shortened_name: seg})
        _df_speed['position'] = i
        _df_speed['Laser wavelength'] = laser_wavelength
        df_speed.append(_df_speed)
    # df_reversals.append({name: pd.Series(this_dict)})
    # df_reversals[name] = this_dict
df_reversals = pd.DataFrame.from_dict(df_reversals)
df_speed = pd.concat(df_speed)


# In[12]:


df_reversals2 = []
df_speed2 = []

laser_wavelengths = [505, 488, 505]
for name, p in all_projects_505_488_505_fm.items():
    # For each project, split it into 3 and then append to the appropriate list
    starts_stops = split_time_series_with_laser_switches(p.green_traces)
    all_segments = split_project_data_in_time(p, starts_stops, verbose=0)

    this_dict = {}
    for i, (laser_wavelength, seg) in enumerate(zip(laser_wavelengths, all_segments)):
        raw_num_rev = len(seg.worm_posture_class.get_starts_and_ends_of_reversals()[0])
        raw_pause_length = seg.worm_posture_class.calc_behavior_from_alias('pause').sum() / p.physical_unit_conversion.volumes_per_second
        num_minutes = p.num_frames / p.physical_unit_conversion.volumes_per_second / 60
        df_reversals2.append({'dataset_name': name, 
                             'position': i, 
                             'Laser wavelength': laser_wavelength, 
                             'raw_num_rev': raw_num_rev,
                             'raw_pause_length': raw_pause_length,
                             'num_minutes': num_minutes,
                             'rev_per_minute': raw_num_rev / num_minutes,
                            'pause_per_minute': raw_pause_length / num_minutes})

        _df_speed = calc_speed_dataframe({seg.shortened_name: seg})
        _df_speed['position'] = i
        _df_speed['Laser wavelength'] = laser_wavelength
        df_speed2.append(_df_speed)
    # df_reversals.append({name: pd.Series(this_dict)})
    # df_reversals[name] = this_dict
df_reversals2 = pd.DataFrame.from_dict(df_reversals2)
df_speed2 = pd.concat(df_speed2)


# In[13]:


df_reversals_both = pd.concat([df_reversals, df_reversals2])
df_reversals_both['Index name'] = df_reversals_both['position'].map({0: 'min 0-4', 1:'min 4-8', 2:'min 8-12'})

df_speed_both = pd.concat([df_speed, df_speed2])
df_speed_both['Index name'] = df_speed_both['position'].map({0: 'min 0-4', 1:'min 4-8', 2:'min 8-12'})


# In[ ]:





# In[14]:


# px.box(df_reversals, color='Laser wavelength', y='rev_per_minute', points='all', x='position', boxmode='overlay', 
#        color_discrete_map=plotly_paper_color_discrete_map())


# In[15]:


# px.box(df_reversals2, color='Laser wavelength', y='rev_per_minute', points='all', x='position', boxmode='overlay',
#       color_discrete_map=plotly_paper_color_discrete_map())


# In[ ]:





# In[16]:


fig = px.box(df_reversals_both, color='Laser wavelength', y='rev_per_minute', points='all', x='Index name', #boxmode='overlay',
      color_discrete_map=plotly_paper_color_discrete_map(),
            category_orders={'Index name': ['min 0-4', 'min 4-8', 'min 8-12']})

fig.update_yaxes(title='Reversals per minute')
fig.update_xaxes(title='')

apply_figure_settings(fig, width_factor=0.5, height_factor=0.2, plotly_not_matplotlib=True)
fig.update_layout(showlegend=False)

fig.show()

fname = f"laser_switch_rev_per_minute.png"
fname = os.path.join("fig_reviewers_505", fname)
fig.write_image(fname, scale=3)
fig.write_image(fname.replace(".png", ".svg"))


# In[17]:


fig = px.box(df_reversals_both, color='Laser wavelength', y='pause_per_minute', points='all', x='Index name', #boxmode='overlay',
      color_discrete_map=plotly_paper_color_discrete_map(),
            category_orders={'Index name': ['min 0-4', 'min 4-8', 'min 8-12']})

fig.update_yaxes(title='Fraction time pausing')
fig.update_xaxes(title='')

apply_figure_settings(fig, width_factor=0.5, height_factor=0.3, plotly_not_matplotlib=True)

fig.show()

fname = f"laser_switch_pause_per_minute.png"
fname = os.path.join("fig_reviewers_505", fname)
fig.write_image(fname, scale=3)
fig.write_image(fname.replace(".png", ".svg"))


# ## Speed histogram

# In[18]:


df_speed_both['sign'] = np.sign(df_speed_both['signed_middle_body_speed'])
df_speed_both['Behavior State'] = df_speed_both['sign'].map({1: 'Forward', -1: 'Backward'})

fig = px.box(df_speed_both.dropna(), color='Laser wavelength', y='signed_middle_body_speed', #points='all', 
             x='Index name', facet_col='Behavior State', #boxmode='overlay',
      color_discrete_map=plotly_paper_color_discrete_map(),
            category_orders={'Index name': ['min 0-4', 'min 4-8', 'min 8-12']})

fig.update_yaxes(title='Velocity per state<br>(mm/s)', row=1, col=1)
# fig.update_yaxes(title='Backward Velocity<br>(mm/s)', row=2)
fig.update_xaxes(title='Forward state', col=1)
fig.update_xaxes(title='Backward state', col=2)
fig.update_layout(showlegend=False)

apply_figure_settings(fig, width_factor=0.5, height_factor=0.25, plotly_not_matplotlib=True)

fig.show()

fname = f"laser_switch_speed_histogram.png"
fname = os.path.join("fig_reviewers_505", fname)
fig.write_image(fname, scale=3)
fig.write_image(fname.replace(".png", ".svg"))


# In[ ]:





# In[ ]:





# ## Example heatmap of different segments

# In[19]:


from wbfm.utils.visualization.plot_traces import make_summary_interactive_heatmap_with_pca


# In[20]:


laser_wavelengths = [505, 488, 505]
for name, p in all_projects_505_488_505_fm.items():
    # For each project, split it into 3 and then append to the appropriate list
    starts_stops = split_time_series_with_laser_switches(p.green_traces)
    all_segments = split_project_data_in_time(p, starts_stops, verbose=0)

    for i, (laser_wavelength, seg) in enumerate(zip(laser_wavelengths, all_segments)):
        print(i, laser_wavelength, seg.shortened_name)
        fig = make_summary_interactive_heatmap_with_pca(seg, to_save=False, to_show=False, 
                                                        trace_opt={'use_paper_options': False, 'interpolate_nan': True})

        fname = f"laser_switch_summary_project{seg.shortened_name}_segment{i}_laser{laser_wavelength}.png"
        fname = os.path.join("fig_reviewers_505", fname)
        fig.write_image(fname, scale=3)
        fig.write_image(fname.replace(".png", ".svg"))


# In[21]:


laser_wavelengths = [488, 505, 488]
for name, p in all_projects_488_505_488_fm.items():
    # For each project, split it into 3 and then append to the appropriate list
    starts_stops = split_time_series_with_laser_switches(p.green_traces)
    all_segments = split_project_data_in_time(p, starts_stops, verbose=0)

    this_dict = {}
    
    for i, (laser_wavelength, seg) in enumerate(zip(laser_wavelengths, all_segments)):
        print(i, laser_wavelength, seg.shortened_name)
        fig = make_summary_interactive_heatmap_with_pca(seg, to_save=False, to_show=False, 
                                                        trace_opt={'use_paper_options': False, 'interpolate_nan': True})
        
        fname = f"laser_switch_summary_project{seg.shortened_name}_segment{i}_laser{laser_wavelength}.png"
        fname = os.path.join("fig_reviewers_505", fname)
        fig.write_image(fname, scale=3)
        fig.write_image(fname.replace(".png", ".svg"))


# ## Unified pca plot for one experiment

# In[22]:


p = all_projects_488_505_488_fm['2025-11-20_11-52_shifts_488_505_488_worm2-2025-11-20']

starts_stops = split_time_series_with_laser_switches(p.green_traces)
all_segments = split_project_data_in_time(p, starts_stops, verbose=0)


# In[23]:


# Build unified trace dataframe, colored by minute
df_traces = []
pca_list = []
trace_opt = {'use_paper_options': False, 'interpolate_nan': True}

for i, (laser_wavelength, seg) in enumerate(zip(laser_wavelengths, all_segments)):
    _df = seg.calc_default_traces(**trace_opt)
    df_traces.append(_df)

    _df = seg.calc_pca_modes(n_components=3, **trace_opt)[0]
    _df['position'] = i
    _df['Laser Wavelength'] = laser_wavelength
    pca_list.append(_df)


# In[24]:


df_pca = pd.concat(pca_list)
df_pca['Index name'] = df_pca['position'].map({0: 'min 0-4', 1:'min 4-8', 2:'min 8-12'})

fig = px.scatter(df_pca, x=0, y=1, color='Index name')

apply_figure_settings(fig, width_factor=0.5, height_factor=0.3, plotly_not_matplotlib=True)

fig.update_xaxes(title='PCA Mode 1')
fig.update_yaxes(title='PCA Mode 2')
fig.show()

fname = f"laser_switch_summary_project{seg.shortened_name}_all_segments_pca.png"
fname = os.path.join("fig_reviewers_505", fname)
fig.write_image(fname, scale=3)
fig.write_image(fname.replace(".png", ".svg"))


# In[25]:


df_pca = pd.concat(pca_list)
df_pca['Index name'] = df_pca['position'].map({0: 'min 0-4', 1:'min 4-8', 2:'min 8-12'})
df_pca['Laser Wavelength'] = df_pca['Laser Wavelength'].astype(str)

fig = px.line(df_pca, x=0, y=1, color='Laser Wavelength', color_discrete_map=plotly_paper_color_discrete_map(),)

apply_figure_settings(fig, width_factor=0.5, height_factor=0.3, plotly_not_matplotlib=True)

fig.update_xaxes(title='PCA Mode 1')
fig.update_yaxes(title='PCA Mode 2')
fig.show()

fname = f"laser_switch_summary_project{seg.shortened_name}_all_segments_pca.png"
fname = os.path.join("fig_reviewers_505", fname)
fig.write_image(fname, scale=3)
fig.write_image(fname.replace(".png", ".svg"))


# # Immobilized laser switch experiments

# In[26]:


from wbfm.utils.projects.finished_project_data import split_project_data_in_time
from wbfm.utils.general.utils_paper import split_time_series_with_laser_switches, plotly_paper_color_discrete_map, apply_figure_settings
from wbfm.utils.visualization.plot_summary_statistics import calc_speed_dataframe


manual_split_annotation = {'2025-09-15_17-12_505_6min_488_6min_561_6min_worm2-2025-09-15':
                              [[0, 774], [776, 1527], [1529, 2269]],
                           '2025-09-15_16-47_505_6min_488_6min_561_6min_worm1-2025-09-15':
                              [[0, 777], [778, 1525], [1527, 2269]],
                           '2025-09-15_17-38_505_6min_488_6min_561_6min_worm3-2025-09-15':
                              [[0, 774], [776, 1526], [1528, 2269]],
                           '2025-09-15_15-55_488_6min_505_6min_488_6min_worm5-2025-09-15':
                              [[0, 776], [778, 1402], [1404, 2269]],
                           '2025-09-15_15-30_488_6min_505_6min_488_6min_worm4-2025-09-15':
                              [[0, 774], [781, 1540], [1542, 2269]]}


# In[27]:


df_reversals_immob = []

laser_wavelengths = [488, 505, 488]
for name, p in all_projects_488_505_488_immob.items():
    # For each project, split it into 3 and then append to the appropriate list
    # print(p.shortened_name)
    # split_time_series_with_laser_switches(p.green_traces, brightness_threshold=605e3, DEBUG=True)
    # continue

    starts_stops = manual_split_annotation.get(p.shortened_name, None)
    if starts_stops is None:
        starts_stops = split_time_series_with_laser_switches(p.green_traces, brightness_threshold=605e3)
    all_segments = split_project_data_in_time(p, starts_stops, verbose=0)
    if not len(all_segments) == 3:
        split_time_series_with_laser_switches(p.green_traces, brightness_threshold=2e4, DEBUG=True)
        raise ValueError

    this_dict = {}
    for i, (laser_wavelength, seg) in enumerate(zip(laser_wavelengths, all_segments)):
        raw_num_rev = len(seg.worm_posture_class.get_starts_and_ends_of_reversals()[0])
        num_minutes = p.num_frames / p.physical_unit_conversion.volumes_per_second / 60
        df_reversals_immob.append({'dataset_name': name, 
                             'position': i, 
                             'Laser wavelength': laser_wavelength, 
                             'raw_num_rev': raw_num_rev,
                             'num_minutes': num_minutes,
                             'rev_per_minute': raw_num_rev / num_minutes})
        print(i, (laser_wavelength, seg.shortened_name))

df_reversals_immob = pd.DataFrame.from_dict(df_reversals_immob)


# In[28]:


df_reversals_immob2 = []
df_speed2 = []

laser_wavelengths = [505, 488, 505]
for name, p in all_projects_505_488_505_immob.items():
    # For each project, split it into 3 and then append to the appropriate list
    # print(p.shortened_name)
    # split_time_series_with_laser_switches(p.green_traces, brightness_threshold=605e3, DEBUG=True)
    # continue

    starts_stops = manual_split_annotation.get(p.shortened_name, None)
    if starts_stops is None:
        starts_stops = split_time_series_with_laser_switches(p.green_traces, brightness_threshold=605e3)
    print(starts_stops)
    all_segments = split_project_data_in_time(p, starts_stops, verbose=0)
    assert len(all_segments) == 3

    this_dict = {}
    for i, (laser_wavelength, seg) in enumerate(zip(laser_wavelengths, all_segments)):
        raw_num_rev = len(seg.worm_posture_class.get_starts_and_ends_of_reversals()[0])
        num_minutes = p.num_frames / p.physical_unit_conversion.volumes_per_second / 60
        df_reversals_immob2.append({'dataset_name': name, 
                             'position': i, 
                             'Laser wavelength': laser_wavelength, 
                             'raw_num_rev': raw_num_rev,
                             'num_minutes': num_minutes,
                             'rev_per_minute': raw_num_rev / num_minutes})
        print(i, (laser_wavelength, seg.shortened_name))

df_reversals_immob2 = pd.DataFrame.from_dict(df_reversals_immob2)


# In[29]:


df_reversals_immob_both = pd.concat([df_reversals_immob, df_reversals_immob2])
df_reversals_immob_both['Index name'] = df_reversals_immob_both['position'].map({0: 'min 0-6', 1:'min 6-12', 2:'min 12-18'})


# In[ ]:





# In[30]:


fig = px.box(df_reversals_immob_both, color='Laser wavelength', y='rev_per_minute', points='all', x='Index name', #boxmode='overlay',
      color_discrete_map=plotly_paper_color_discrete_map(),
            category_orders={'Index name': ['min 0-6', 'min 6-12', 'min 12-18']})

fig.update_yaxes(title='Reversal command<br>states per minute')
fig.update_xaxes(title='')

apply_figure_settings(fig, width_factor=0.25, height_factor=0.25, plotly_not_matplotlib=True)
fig.update_layout(showlegend=False)
# add_p_value_annotation(fig, x_label='all', show_ns=True, show_only_stars=True) #height_mode='top_of_data')

fig.show()

fname = f"laser_switch_rev_per_minute.png"
fname = os.path.join("fig_reviewers_505_immob", fname)
fig.write_image(fname, scale=3)
fig.write_image(fname.replace(".png", ".svg"))


# In[31]:


# df_reversals_immob_both


# ## Number of completely inactive datasets

# In[32]:


df = pd.DataFrame()
df['Number of Datasets'] = [len(all_projects_505_488_505_immob_inactive),
                   len(all_projects_488_505_488_immob_inactive),
                   len(all_projects_505_488_505_immob),
                   len(all_projects_488_505_488_immob)]
df['Activity Type'] = ['Inactive', 'Inactive', 'Active', 'Active']
df['Condition Type'] = ['505 First', '488 First', '505 First', '488 First']


# In[33]:


fig = px.bar(df, y='Number of Datasets', x='Condition Type', color='Activity Type', text='Number of Datasets', color_discrete_map=plotly_paper_color_discrete_map(),)


apply_figure_settings(fig, width_factor=0.25, height_factor=0.25, plotly_not_matplotlib=True)

fig.show()

fname = f"inactive_fraction.png"
fname = os.path.join("fig_reviewers_505_immob", fname)
fig.write_image(fname, scale=3)
fig.write_image(fname.replace(".png", ".svg"))


# ## Example heatmap of different segments

# In[34]:


from wbfm.utils.visualization.plot_traces import make_summary_interactive_heatmap_with_pca


# In[35]:


manual_split_annotation = {'2025-09-15_17-12_505_6min_488_6min_561_6min_worm2-2025-09-15':
                              [[0, 774], [776, 1527], [1529, 2269]],
                           '2025-09-15_16-47_505_6min_488_6min_561_6min_worm1-2025-09-15':
                              [[0, 777], [778, 1525], [1527, 2269]],
                           '2025-09-15_17-38_505_6min_488_6min_561_6min_worm3-2025-09-15':
                              [[0, 774], [776, 1526], [1528, 2269]],
                           '2025-09-15_15-55_488_6min_505_6min_488_6min_worm5-2025-09-15':
                              [[0, 776], [778, 1402], [1404, 2269]],
                           '2025-09-15_15-30_488_6min_505_6min_488_6min_worm4-2025-09-15':
                              [[0, 774], [781, 1540], [1542, 2269]]}


# In[36]:


laser_wavelengths = [505, 488, 505]
for name, p in all_projects_505_488_505_immob.items():
    # For each project, split it into 3 and then append to the appropriate list
    
    starts_stops = manual_split_annotation.get(p.shortened_name, None)
    if starts_stops is None:
        starts_stops = split_time_series_with_laser_switches(p.green_traces, brightness_threshold=605e3)
    all_segments = split_project_data_in_time(p, starts_stops, verbose=0)

    for i, (laser_wavelength, seg) in enumerate(zip(laser_wavelengths, all_segments)):
        print(i, laser_wavelength, seg.shortened_name)
        fig = make_summary_interactive_heatmap_with_pca(seg, to_save=False, to_show=False, 
                                                        trace_opt={'use_paper_options': False, 'interpolate_nan': True})

        fname = f"laser_switch_summary_project{seg.shortened_name}_segment{i}_laser{laser_wavelength}.png"
        fname = os.path.join("fig_reviewers_505_immob", fname)
        fig.write_image(fname, scale=3)
        fig.write_image(fname.replace(".png", ".svg"))


# In[37]:


laser_wavelengths = [488, 505, 488]
for name, p in all_projects_488_505_488_immob.items():
    # For each project, split it into 3 and then append to the appropriate list

    starts_stops = manual_split_annotation.get(p.shortened_name, None)
    if starts_stops is None:
        starts_stops = split_time_series_with_laser_switches(p.green_traces, brightness_threshold=605e3)
    all_segments = split_project_data_in_time(p, starts_stops, verbose=0)
    
    for i, (laser_wavelength, seg) in enumerate(zip(laser_wavelengths, all_segments)):
        print(i, laser_wavelength, seg.shortened_name)
        fig = make_summary_interactive_heatmap_with_pca(seg, to_save=False, to_show=False, 
                                                        trace_opt={'use_paper_options': False, 'interpolate_nan': True})
        
        fname = f"laser_switch_summary_project{seg.shortened_name}_segment{i}_laser{laser_wavelength}.png"
        fname = os.path.join("fig_reviewers_505_immob", fname)
        fig.write_image(fname, scale=3)
        fig.write_image(fname.replace(".png", ".svg"))


# In[ ]:





# ## Fourier spectrum analysis

# In[38]:


manual_split_annotation = {'2025-09-15_17-12_505_6min_488_6min_561_6min_worm2-2025-09-15':
                              [[0, 774], [776, 1527], [1529, 2269]],
                           '2025-09-15_16-47_505_6min_488_6min_561_6min_worm1-2025-09-15':
                              [[0, 777], [778, 1525], [1527, 2269]],
                           '2025-09-15_17-38_505_6min_488_6min_561_6min_worm3-2025-09-15':
                              [[0, 774], [776, 1526], [1528, 2269]],
                           '2025-09-15_15-55_488_6min_505_6min_488_6min_worm5-2025-09-15':
                              [[0, 776], [778, 1402], [1404, 2269]],
                           '2025-09-15_15-30_488_6min_505_6min_488_6min_worm4-2025-09-15':
                              [[0, 774], [781, 1540], [1542, 2269]]}

all_dfs = defaultdict(list)

trace_opt = {'use_paper_options': False, 'interpolate_nan': True}


# In[39]:


trace_opt


# In[40]:


laser_wavelengths = [505, 488, 505]
for name, p in all_projects_505_488_505_immob.items():
    # For each project, split it into 3 and then append to the appropriate list
    
    starts_stops = manual_split_annotation.get(p.shortened_name, None)
    if starts_stops is None:
        starts_stops = split_time_series_with_laser_switches(p.green_traces, brightness_threshold=605e3)
    all_segments = split_project_data_in_time(p, starts_stops, verbose=0)
    print(starts_stops)

    for i, (laser_wavelength, seg) in enumerate(zip(laser_wavelengths, all_segments)):
        print(i, laser_wavelength, seg.shortened_name)
        all_dfs[(i, laser_wavelength)].append(seg.calc_default_traces(**trace_opt))


# In[41]:


laser_wavelengths = [488, 505, 488]
for name, p in all_projects_488_505_488_immob.items():
    # For each project, split it into 3 and then append to the appropriate list

    starts_stops = manual_split_annotation.get(p.shortened_name, None)
    if starts_stops is None:
        starts_stops = split_time_series_with_laser_switches(p.green_traces, brightness_threshold=605e3)
    all_segments = split_project_data_in_time(p, starts_stops, verbose=0)
    print(starts_stops)
    
    for i, (laser_wavelength, seg) in enumerate(zip(laser_wavelengths, all_segments)):
        print(i, laser_wavelength, seg.shortened_name)
        all_dfs[(i, laser_wavelength)].append(seg.calc_default_traces(**trace_opt))


# ### Main figure

# In[42]:


from wbfm.utils.general.utils_paper_revisions import compute_fig1_metrics, plot_fig1E_F_plotly_simple


# In[43]:


sampling_intervals_all = {k: len(v)*[1/3.5] for k, v in all_dfs.items()}

fig1_results = compute_fig1_metrics(all_dfs, sampling_intervals_all)


# In[44]:


fig1_figs = plot_fig1E_F_plotly_simple(fig1_results['per_condition'])


# In[45]:


fig = fig1_figs['Fig1E']

apply_figure_settings(fig, width_factor=0.25, height_factor=0.25, plotly_not_matplotlib=True)
fig.update_layout(showlegend=False)
add_p_value_annotation(fig, x_label='all', show_ns=True, show_only_stars=True)
fig.show()

fname = f"fraction_neurons_in_fft_band.png"
fname = os.path.join("fig_reviewers_505_immob", fname)
fig.write_image(fname, scale=3)
fig.write_image(fname.replace(".png", ".svg"))


# In[46]:


fig = fig1_figs['Fig1F']

apply_figure_settings(fig, width_factor=0.25, height_factor=0.25, plotly_not_matplotlib=True)
fig.update_layout(showlegend=False)
add_p_value_annotation(fig, x_label='all', show_ns=True, show_only_stars=True)
fig.show()

fname = f"spectral_edge.png"
fname = os.path.join("fig_reviewers_505_immob", fname)
fig.write_image(fname, scale=3)
fig.write_image(fname.replace(".png", ".svg"))


# ### Supp figures

# In[47]:


from wbfm.utils.general.utils_paper_revisions import reproduce_figures_plotly


# In[48]:


sampling_intervals_all = {k: len(v)*[1/3.5] for k, v in all_dfs.items()}

results = reproduce_figures_plotly(all_dfs, sampling_intervals_all)


# In[49]:


# %debug


# In[50]:


fig = results['figures']['S2A']

apply_figure_settings(fig, width_factor=0.25, height_factor=0.5, plotly_not_matplotlib=True)
fig.update_layout(showlegend=False)
fig.show()

fname = f"cdf_and_psd.png"
fname = os.path.join("fig_reviewers_505_immob", fname)
fig.write_image(fname, scale=3)
fig.write_image(fname.replace(".png", ".svg"))


# In[51]:


fig = results['figures']['S2B']
apply_figure_settings(fig, width_factor=0.25, height_factor=0.25, plotly_not_matplotlib=True)
fig.update_layout(showlegend=False)
add_p_value_annotation(fig, x_label='all', show_ns=True, show_only_stars=True)
fig.show()

fname = f"percent_neurons_over_threshold.png"
fname = os.path.join("fig_reviewers_505_immob", fname)
fig.write_image(fname, scale=3)
fig.write_image(fname.replace(".png", ".svg"))


# In[52]:


results['figures']['S2C']


# ## Also do the PC1 comparison like the main figure

# In[53]:


from wbfm.utils.visualization.utils_cca import calc_pca_weights_for_all_projects
from wbfm.utils.external.utils_plotly import plotly_boxplot_colored_boxes
from wbfm.utils.general.utils_paper import apply_figure_settings
from wbfm.utils.general.utils_hardcoded import neurons_with_confident_ids


# In[54]:


all_sub_projects = defaultdict(dict)


# In[55]:


trace_opt = {'use_paper_options': False, 'interpolate_nan': True, 'rename_neurons_using_manual_ids': True, 'manual_id_confidence_threshold': 0}


# In[56]:


laser_wavelengths = [505, 488, 505]
for name, p in all_projects_505_488_505_immob.items():
    # For each project, split it into 3 and then append to the appropriate list
    
    starts_stops = manual_split_annotation.get(p.shortened_name, None)
    if starts_stops is None:
        starts_stops = split_time_series_with_laser_switches(p.green_traces, brightness_threshold=605e3)
    all_segments = split_project_data_in_time(p, starts_stops, verbose=0)
    print(starts_stops)

    for i, (laser_wavelength, seg) in enumerate(zip(laser_wavelengths, all_segments)):
        print(i, laser_wavelength, seg.shortened_name)
        all_sub_projects[(i, laser_wavelength)][seg.shortened_name] = seg


# In[57]:


laser_wavelengths = [488, 505, 488]
for name, p in all_projects_488_505_488_immob.items():
    # For each project, split it into 3 and then append to the appropriate list

    starts_stops = manual_split_annotation.get(p.shortened_name, None)
    if starts_stops is None:
        starts_stops = split_time_series_with_laser_switches(p.green_traces, brightness_threshold=605e3)
    all_segments = split_project_data_in_time(p, starts_stops, verbose=0)
    print(starts_stops)
    
    for i, (laser_wavelength, seg) in enumerate(zip(laser_wavelengths, all_segments)):
        print(i, laser_wavelength, seg.shortened_name)
        all_sub_projects[(i, laser_wavelength)][seg.shortened_name] = seg


# In[58]:


all_weights = dict()

for key, projects in tqdm(all_sub_projects.items()):
    print(key)
    all_weights[key] = calc_pca_weights_for_all_projects(projects, **trace_opt, combine_left_right=True, include_only_confident_ids=True)


# In[59]:


from wbfm.utils.general.utils_paper_revisions import calc_statistics_for_pc1_comparison_plots, plot_pc1_comparison


# In[60]:


# Make three plots with each pair of laser light, i.e. split by 1st/2nd/3rd part
df_both, df_pvalue, df_medians_gcamp, df_medians_immob, df_significant_diff = calc_statistics_for_pc1_comparison_plots(all_weights,
                                                                                                             [(0, 488), (0, 505)])

# In the first segment, calculate the order for all panels
x_order = list(df_medians_gcamp.sort_values(ascending=False).index)

fig = plot_pc1_comparison(df_both, df_significant_diff, x_order=x_order)

apply_figure_settings(fig, width_factor=0.8, height_factor=0.15, plotly_not_matplotlib=True)
fig.update_yaxes(dict(title="PC1 weight"), zeroline=True, zerolinewidth=1, zerolinecolor="black", overwrite=True)
fig.update_layout(showlegend=False)
fig.update_yaxes(title="")
fig.show()

fname = f"pc1_weights_segment0.png"
fname = os.path.join("fig_reviewers_505_immob", fname)
fig.write_image(fname, scale=3)
fig.write_image(fname.replace(".png", ".svg"))


# In[61]:


# Make three plots with each pair of laser light, i.e. split by 1st/2nd/3rd part
df_both, df_pvalue, df_medians_gcamp, df_medians_immob, df_significant_diff = calc_statistics_for_pc1_comparison_plots(all_weights,
                                                                                                             [(1, 488), (1, 505)])
fig = plot_pc1_comparison(df_both, df_significant_diff, x_order=x_order)

apply_figure_settings(fig, width_factor=0.8, height_factor=0.15, plotly_not_matplotlib=True)
fig.update_yaxes(dict(title="PC1 weight"), zeroline=True, zerolinewidth=1, zerolinecolor="black", overwrite=True)
fig.update_layout(showlegend=False)
fig.update_yaxes(title="")
fig.show()

fname = f"pc1_weights_segment1.png"
fname = os.path.join("fig_reviewers_505_immob", fname)
fig.write_image(fname, scale=3)
fig.write_image(fname.replace(".png", ".svg"))


# In[62]:


# Make three plots with each pair of laser light, i.e. split by 1st/2nd/3rd part
df_both, df_pvalue, df_medians_gcamp, df_medians_immob, df_significant_diff = calc_statistics_for_pc1_comparison_plots(all_weights,
                                                                                                             [(2, 488), (2, 505)])
fig = plot_pc1_comparison(df_both, df_significant_diff, x_order=x_order)

apply_figure_settings(fig, width_factor=0.8, height_factor=0.15, plotly_not_matplotlib=True)
fig.update_yaxes(dict(title="PC1 weight"), zeroline=True, zerolinewidth=1, zerolinecolor="black", overwrite=True)
fig.update_layout(showlegend=False)
fig.update_yaxes(title="")
fig.show()

fname = f"pc1_weights_segment2.png"
fname = os.path.join("fig_reviewers_505_immob", fname)
fig.write_image(fname, scale=3)
fig.write_image(fname.replace(".png", ".svg"))


# In[63]:


df_both.dropna().groupby(['neuron_name', 'Dataset Type']).size().unstack(fill_value=0)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Debug

# In[64]:


fname = "/lisc/data/scratch/neurobiology/zimmer/ItamarLev/WBFM/WBFM_projects/freely_moving_505/488_505_488/2025-11-20_11-08_shifts_488_505_488_worm1-2025-11-20"

p = ProjectData.load_final_project_data(fname)


# In[65]:


p.worm_posture_class


# In[66]:


# p = all_projects_505_488_505_fm['2025-11-20_11-35_shifts_505_488_505_worm1-2025-11-20']


# In[67]:


from wbfm.utils.external.custom_errors import NoManualBehaviorAnnotationsError, NoBehaviorAnnotationsError, \
    MissingAnalysisError, DataSynchronizationError


# In[68]:


from wbfm.utils.projects.finished_project_data import split_project_data_in_time
from wbfm.utils.general.utils_paper import split_time_series_with_laser_switches


# In[ ]:





# In[69]:


starts_stops = split_time_series_with_laser_switches(p.green_traces)
all_segments = split_project_data_in_time(p, starts_stops, verbose=0)


# In[70]:


p.red_traces.shape, all_segments[0].red_traces.shape, all_segments[1].red_traces.shape


# In[71]:


# p.worm_posture_class.()


# In[72]:


p.worm_posture_class.beh_annotation(fluorescence_fps=True).shape, all_segments[0].worm_posture_class.beh_annotation(fluorescence_fps=True).shape, all_segments[1].worm_posture_class.beh_annotation(fluorescence_fps=True).shape, all_segments[2].worm_posture_class.beh_annotation(fluorescence_fps=True).shape


# In[73]:


# px.imshow(p.green_traces.loc[:, (slice(None), 'intensity_image')].T.diff(axis=1), height=1000)


# In[74]:


# px.imshow(p.red_traces.loc[:, (slice(None), 'intensity_image')].T.diff(axis=1), height=1000)


# In[75]:


p.background_per_pixel


# In[76]:


px.line({'green': p.green_traces.loc[:, (slice(None), 'intensity_image')].T.sum() - 100*p.green_traces.loc[:, (slice(None), 'area')].T.sum(),
        # 'red': p.red_traces.loc[:, (slice(None), 'intensity_image')].T.diff(axis=1).median()
        })


# In[77]:


px.line(p.green_traces.loc[:, (slice(None), 'area')].T.sum())

