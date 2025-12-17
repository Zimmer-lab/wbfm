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
import plotly.express as px


# In[2]:


from sklearn.decomposition import PCA
from wbfm.utils.visualization.plot_traces import make_grid_plot_from_dataframe
import seaborn as sns
from wbfm.utils.visualization.behavior_comparison_plots import NeuronToMultivariateEncoding
from wbfm.utils.traces.gui_kymograph_correlations import build_all_gui_dfs_multineuron_correlations


# In[3]:


fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-11-27_spacer_7b_2per_agar/ZIM2165_Gcamp7b_worm1-2022_11_28/project_config.yaml"
# Manually corrected version
# fname = "/scratch/neurobiology/zimmer/Charles/dlc_stacks/manually_annotated/paper_data/ZIM2165_Gcamp7b_worm1-2022_11_28/project_config.yaml"
project_data_gcamp = ProjectData.load_final_project_data_from_config(fname)


# In[4]:


fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-12-10_spacer_7b_2per_agar_GFP/ZIM2319_GFP_worm1-2022-12-10/project_config.yaml"
project_data_gfp = ProjectData.load_final_project_data_from_config(fname)


# In[5]:


# fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-12-12_immob/2022-12-13_15-16_ZIM2165_immob_worm9-2022-12-13/project_config.yaml"
fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-12-12_immob/2022-12-13_10-38_ZIM2165_immob_worm8-2022-12-13/project_config.yaml"
project_data_immob = ProjectData.load_final_project_data_from_config(fname)


# In[6]:


# Load multiple datasets
from wbfm.utils.general.hardcoded_paths import load_paper_datasets
all_projects_gcamp = load_paper_datasets(['gcamp', 'hannah_O2_fm'])
all_projects_gfp = load_paper_datasets('gfp')


# In[7]:


all_projects_immob = load_paper_datasets('immob')


# # Triple plots

# ## Example: WBFM

# In[8]:


from wbfm.utils.visualization.paper_multidataset_triggered_average import PaperExampleTracePlotter


# In[9]:


# wbfm_plotter = PaperExampleTracePlotter(project_data_gcamp, xlim=[0, 120], ylim=[-0.33, 0.24])
# wbfm_plotter = PaperExampleTracePlotter(all_projects_gcamp['ZIM2165_Gcamp7b_worm3-2022_11_28'], xlim=[65, 145], ylim=[-0.24, 0.3])
wbfm_plotter = PaperExampleTracePlotter(all_projects_gcamp['ZIM2165_Gcamp7b_worm3-2022_11_28'], xlim=[65, 145], ylim=[-0.19, 0.3])


# In[30]:


output_foldername = '/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm'
triple_opt = dict(output_foldername=output_foldername, width_factor=0.3, height_factor=0.23,
                               round_y_ticks=True, round_yticks_kwargs=dict(max_ticks=3), title=True, )

for n in ['VB02', 'DB01', 'DD01', 'AVB']:

    wbfm_plotter.plot_triple_traces(n, legend=False,#n=='VB02', 
                                    combine_lr=True, **triple_opt);
    


# In[34]:


# For the supp
output_foldername = '/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm'
wbfm_plotter.plot_triple_traces('BAG', ylim=None, title=True, legend=True, combine_lr=True, 
                                output_foldername=output_foldername,width_factor=0.45, height_factor=0.25)
# wbfm_plotter.plot_single_trace('BAG', title=False, legend=False, output_foldername=output_foldername,
#                                    trace_options=dict(trace_type='raw'), width_factor=0.4, height_factor=0.2)


# In[12]:


# wbfm_plotter.project.neuron_name_to_manual_id_mapping(remove_unnamed_neurons=True, confidence_threshold=0)


# In[ ]:





# In[ ]:





# # Triggered averages

# ## Initial calculations

# In[13]:


from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
from wbfm.utils.visualization.paper_multidataset_triggered_average import PaperMultiDatasetTriggeredAverage
from wbfm.utils.general.utils_behavior_annotation import approximate_behavioral_annotation_using_pc1


# In[14]:


triggered_average_gcamp_plotter = PaperMultiDatasetTriggeredAverage(all_projects_gcamp)


# In[15]:


# %debug


# In[16]:


# from wbfm.utils.general.utils_behavior_annotation import approximate_behavioral_annotation_using_ava
# # Run if the behavior for the immobilized isn't there or needs to be updated
# for p in all_projects_immob.values():
#     try:
#         beh_vec = approximate_behavioral_annotation_using_ava(p)
#     except:
#         print("Dataset failed")
#         print(p.shortened_name)


# In[17]:


# triggered_average_immob_plotter = PaperMultiDatasetTriggeredAverage(all_projects_immob)


# ## Motor (ACTUALLY PLOTTED BELOW)

# In[18]:


# trigger_types = [('global_rev', ''), 
#                 ('residual_rectified_rev', 'Reversal'),
#                 ('residual_rectified_fwd', 'Forward'),
#                 ('residual', '')]

# for trigger_type, title in trigger_types:
#     triggered_average_gcamp_plotter.plot_triggered_average_single_neuron('VB02', trigger_type, 
#                                                                          title=title, show_title=True, ylim=[-0.09, 0.055],
#                                                                          output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm_trigger',
#                                                                         i_figure=4)


# In[19]:


# trigger_types = [('global_rev', ''), 
#                 ('residual_rectified_rev', 'Reversal'),
#                 ('residual_rectified_fwd', 'Forward'),
#                 ('residual', '')]

# for trigger_type, title in trigger_types:
#     triggered_average_gcamp_plotter.plot_triggered_average_single_neuron('DB01', trigger_type, 
#                                                                          title=title, show_title=True, ylim=[-0.09, 0.055],
#                                                                          output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm_trigger',
#                                                                         i_figure=4)


# In[20]:


from wbfm.utils.general.utils_paper import apply_figure_settings
# Plot neurons as a rainbow on a single plot

# trigger_types = [('global_rev', ''), 
#                 ('residual_rectified_rev', 'Reversal'),
#                 ('residual_rectified_fwd', 'Forward'),
#                 ('residual', '')]

trigger_type = 'residual_rectified_fwd'
title = ''

# neurons = ['VB03', 'VB02', 'RMED', 'RMDVL', 'SMDDL', 'DB01', 'RMEV']
neurons = ['VB02', 'RMED', 'RMDV', 'SMDD', 'DB01', 'RMEV', 'VB03', 'DD01', 
          ]

fig, ax = None, None
cmap = px.colors.qualitative.Plotly
for i, neuron in tqdm(enumerate(neurons)):
    color = cmap[i]
    fig, ax = triggered_average_gcamp_plotter.plot_triggered_average_single_neuron(neuron, trigger_type, fig=fig, ax=ax, use_plotly=True,
                                                                         title=title, show_title=True, ylim=[-0.09, 0.055], color=color,
                                                                         output_folder=None,#'/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm_trigger',
                                                                        i_figure=4)


# In[29]:


# apply_figure_settings(fig, plotly_not_matplotlib=True, width_factor=1.0, height_factor=0.3)
# fig.update_xaxes(range=[-3, 3])
# fig.update_yaxes(range=[-0.1, 0.1])
# fig.update_layout(showlegend=True)
# fig.show()


# ## BAG

# In[22]:


# Actually plotted in the O2 multiplexing notebook

# trigger_types = [('global_rev', 'Reversal Triggered'), 
#                 ('residual_collision', 'Collision Triggered')]

# for trigger_type, title in trigger_types:
#     triggered_average_gcamp_plotter.plot_triggered_average_single_neuron('BAGL', trigger_type, 
#                                                                          title=title, show_title=True,
#                                                                          output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm_trigger',
#                                                                         fig_opt=dict(width_factor=0.5,height_factor=0.2))


# ## All O2 neurons

# In[24]:


# from wbfm.utils.general.hardcoded_paths import list_of_gas_sensing_neurons

# trigger_types = [('global_rev', 'Reversal Triggered'), 
#                 ('residual_collision', 'Collision Triggered')]

# for n in list_of_gas_sensing_neurons():
#     print(n)
#     for trigger_type, title in trigger_types:
#         triggered_average_gcamp_plotter.plot_triggered_average_single_neuron(n, trigger_type, 
#                                                                              title="",#title,
#                                                                              output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm_trigger',
#                                                                             i_figure=5)


# In[ ]:





# # Additional triple plots and triggered averages (examples)

# ## Additional triple plots (example dataset)

# In[ ]:


# wbfm_plotter.plot_triple_traces('RID', output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm')


# # Triggered averages (multiple datasets, ACTUALLY USED)

# In[25]:


from wbfm.utils.visualization.utils_plot_traces import convert_channel_mode_to_axis_label
convert_channel_mode_to_axis_label(triggered_average_gcamp_plotter.trace_opt)


# In[33]:


trigger_types = [('global_rev', ''), 
                ('global_fwd', ''), 
                # ('residual_rectified_rev', 'Reversal'),
                # ('residual_rectified_fwd', 'Forward'),
                # ('residual', '')
                ]

# Global triggered averages
for neuron in [#'VB01', 'VB03', 'DB02', 'VA02', 'VA01', 'DA01', 
               'VB02', 'DB01',
               # 'SIAD', 'SIAV', 'RIB',
               'DD01', 
               'AVB', #'RMEV', 'RMED', 'RME',
               #'SAAV', 'RIA', 'RID', 'URAD', 'AVF', 'AVB', 'AVA' # Also try some that shouldn't be rectified
              ]:
    for trigger_type, title in trigger_types:
        
        if 'residual' in trigger_type:
            opt = dict(ylim=[-0.09, 0.1], xlim=[-3.5, 3.5], height_factor=0.15, width_factor=0.3)
        elif 'global' in trigger_type:
            opt = dict(ylim=[-0.2, 0.12], 
                       xlim=[-4, 10], 
                       height_factor=0.2, width_factor=0.3)
        else:
            opt = dict()
        
        triggered_average_gcamp_plotter.plot_triggered_average_single_neuron(neuron, trigger_type, title=neuron, show_title=True, 
                                                                             output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm_trigger',
                                                                             i_figure=4, width_factor_addition=-0.05, **opt)
        plt.show()


# In[28]:


# STACKED residual triggered averages
trigger_types = ['residual_rectified_fwd',
                'residual_rectified_rev',
                ]
output_folder = '/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm_trigger'
# Global triggered averages
for neuron in [#'VB01', 'VB03', 'DB02', 'VA02', 'VA01', 'DA01', 
               'VB02', 'DB01',
               # 'SIAD', 'SIAV', 'RIB',
               'DD01', 
               'AVB', #'RMEV', 'RMED', 'RME',
               #'SAAV', 'RIA', 'RID', 'URAD', 'AVF', 'AVB', 'AVA' # Also try some that shouldn't be rectified
              ]:
    opt = dict(ylim=[-0.09, 0.12], xlim=[-3.5, 3.5], height_factor=0.23, width_factor=0.3)
    
    triggered_average_gcamp_plotter.plot_triggered_average_single_neuron(neuron, trigger_types, title=neuron, show_title=False,
                                                                         output_folder=output_folder,
                                                                         **opt)
    # fig, axes = plt.subplots(nrows=2, **triggered_average_gcamp_plotter.get_fig_opt())
    # output_folder = None
#     for trigger_type, ax in zip(trigger_types, axes):

#         triggered_average_gcamp_plotter.plot_triggered_average_single_neuron(neuron, trigger_type, title=neuron, show_title=False, ax=ax, fig=fig,
#                                                                              output_folder=output_folder,
#                                                                              **opt)
#     output_folder = '/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm_trigger'
#     apply_figure_settings(width_factor=opt['width_factor'], height_factor=opt['height_factor'], plotly_not_matplotlib=False)
#     plt.subplots_adjust(hspace=0)
    
#     fname = os.path.join(output_folder, f'{neuron}-stacked_triggered_averages.png')
#     # plt.tight_layout()
#     plt.savefig(fname, transparent=True)
#     plt.savefig(fname.replace(".png", ".svg"))
    
    plt.show(fig)


# In[ ]:


# %debug


# In[ ]:





# In[ ]:





# # DEBUG

# In[ ]:


# y = immob_plotter.df_traces['BAGL']
# y2 = immob_plotter.df_traces_global['BAGL']
# y3 = immob_plotter.df_traces_residual['BAGL']

# df = pd.DataFrame({'y': y, 'resid': y3, 'global': y2})
# px.line(df)


# ## Plot DB01 (all lines)

# In[ ]:


trigger_types = [('global_rev', ''), 
                ('global_fwd', ''), 
                ('residual_rectified_rev', 'Reversal'),
                ('residual_rectified_fwd', 'Forward'),
                # ('residual', '')
                ]

for neuron in [#'VB01', 'VB03', 'DB02', 'VA02', 'VA01', 'DA01', 
               # 'VB02', 
    'DB01',
               # 'SIAD', 'SIAV', 'RIB',
               # 'DD01', 
               # 'AVB', #'RMEV', 'RMED', 'RME',
               #'SAAV', 'RIA', 'RID', 'URAD', 'AVF', 'AVB', 'AVA' # Also try some that shouldn't be rectified
              ]:
    for trigger_type, title in trigger_types:
        if 'residual' in trigger_type:
            opt = dict(ylim=[-0.09, 0.1], xlim=[-3.5, 3.5], height_factor=0.15, width_factor=0.3)
        elif 'global' in trigger_type:
            opt = dict(#ylim=[-0.1, 0.5], 
                       xlim=[-5, 10], 
                       height_factor=0.15, width_factor=0.3)
        else:
            opt = dict()
        
        triggered_average_gcamp_plotter.plot_triggered_average_single_neuron(neuron, trigger_type, title=neuron, show_title=True, 
                                                                             output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/wbfm_trigger',
                                                                             show_individual_lines=True, **opt)
        plt.show()


# In[ ]:


from wbfm.utils.general.utils_paper import apply_figure_settings
# Plot stacked triggered averages
trigger_types = ['residual_rectified_rev',
                'residual_rectified_fwd',
                # ('residual', '')
                ]
neuron = 'AVB'

fig, axes = plt.subplots(nrows=2, **triggered_average_gcamp_plotter.get_fig_opt())
for trigger_type, ax in zip(trigger_types, axes):
    opt = dict(ylim=[-0.09, 0.1], xlim=[-3.5, 3.5], height_factor=0.2, width_factor=0.3)
    triggered_average_gcamp_plotter.plot_triggered_average_single_neuron(neuron, trigger_type, show_y_label=True, #title=neuron, show_title=True, 
                                                                         output_folder=None,
                                                                         show_individual_lines=False, fig=fig, ax=ax, **opt)
    
apply_figure_settings(width_factor=opt['width_factor'], height_factor=opt['height_factor'], plotly_not_matplotlib=False)
plt.subplots_adjust(hspace=0)

plt.show()


# In[ ]:


fig


# In[ ]:





# In[ ]:


df_subset = triggered_average_gcamp_plotter.get_traces_single_neuron('DB01', 'global_fwd', return_individual_traces=False)
# df_subset


# In[ ]:


# px.line(df_subset)


# In[ ]:


all_projects_gcamp['ZIM2165_Gcamp7b_worm10-2022-12-05'].project_dir


# In[ ]:


all_projects_gcamp['ZIM2165_Gcamp7b_worm3-2022-12-05'].project_dir


# In[ ]:


all_projects_gcamp['ZIM2165_Gcamp7b_worm5-2022-12-10'].project_dir


# In[ ]:


all_projects_gcamp['2022-11-23_worm11'].project_dir


# In[ ]:


beh_vec = immob_plotter.project.worm_posture_class.beh_annotation(fluorescence_fps=True)


# In[ ]:





# # Plot of all neurons with signal (autocovariance)
# 

# In[ ]:


from wbfm.utils.visualization.multiproject_wrappers import plot_variance_all_neurons


# In[ ]:


output_folder = 'multiplexing'
fig, df_summary, significance_line, cmap = plot_variance_all_neurons(all_projects_gcamp, all_projects_gfp, lag=1, output_folder=output_folder,
                                                                   loop_not_facet_row=True,
                                                                     names_to_keep_in_simple_id=('VB02', 'DB01'),
                                                                   use_paper_options=True, include_gfp=True, include_legend=True)


# In[ ]:


# output_folder = None
# fig, df_summary, significance_line, cmap = plot_variance_all_neurons(all_projects_gcamp, all_projects_gfp, lag=1, output_folder=output_folder,
#                                                                    loop_not_facet_row=True,
#                                                                      names_to_keep_in_simple_id=('AVA', 'AVE', 'RIM', 'ALA'),
#                                                                    use_paper_options=True, include_gfp=True, include_legend=True)


# ## Total count of significant neurons in each data category

# In[ ]:


df_significant_numbers = df_summary.groupby('Type of data')['Significant'].value_counts().reset_index()
df_significant_numbers


# In[ ]:


df_total_numbers = df_significant_numbers.groupby('Type of data').sum().reset_index().drop(columns=['Significant'])
df_total_numbers


# In[ ]:


df_signficant_summary = df_significant_numbers.merge(df_total_numbers, on='Type of data')
df_signficant_summary


# In[ ]:


df_signficant_summary['percent_significant'] = df_signficant_summary['count_x'] / df_signficant_summary['count_y']
df_signficant_summary


# ## Count of significant neurons in residual AND global, and high (or low) PC correlation

# In[ ]:


df_subset = df_summary[(df_summary['Type of data']=='global gcamp') | (df_summary['Type of data']=='residual gcamp')]
df_subset = df_subset[df_subset['pc0_high'] | df_subset['pc0_low']]
df_subset = df_subset[df_subset['Significant']].reset_index()
df_subset.head()


# In[ ]:


# Count which indices are doubled, meaning both residual and global were significant
counts = df_subset.groupby('index')['Significant'].count()
result = counts[counts == 2].index
count_of_true_for_both_versions = len(result)
print(count_of_true_for_both_versions, count_of_true_for_both_versions/2682)


# In[ ]:


# df_subset.merge(df_total_numbers, on='Type of data')


# In[ ]:




