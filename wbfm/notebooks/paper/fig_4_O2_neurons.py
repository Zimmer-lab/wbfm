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


# Load multiple datasets
from wbfm.utils.general.hardcoded_paths import load_paper_datasets
all_projects_O2_fm = load_paper_datasets(['hannah_O2_fm', 'gcamp'])
all_projects_O2_fm_mutant = load_paper_datasets('hannah_O2_fm_mutant')
all_projects_O2_immob_mutant = load_paper_datasets('hannah_O2_immob_mutant')


# In[4]:


all_projects_O2_immob = load_paper_datasets('immob_o2')


# In[5]:


all_projects_immob = load_paper_datasets('immob')


# In[6]:


all_projects_O2_hiscl = load_paper_datasets('O2_hiscl')


# In[7]:


all_projects_gfp = load_paper_datasets('gfp')


# In[8]:


list_of_all_dicts = [all_projects_O2_fm, all_projects_O2_fm_mutant, all_projects_O2_immob, all_projects_O2_immob_mutant,
                     all_projects_immob, all_projects_O2_hiscl, all_projects_gfp]


# # Check ID'ed neurons

# In[9]:


all_projects_ids = {}
for name, p in all_projects_O2_fm.items():
    ids = p.neuron_name_to_manual_id_mapping(remove_unnamed_neurons=True, confidence_threshold=0)
    if len(ids) > 4:
        all_projects_ids[name] = p
        print(name, len(ids))
    # else:
    #     print(name, ids)


# # Check that the behavior annotations make sense

# In[10]:


from wbfm.utils.visualization.plot_traces import make_summary_interactive_kymograph_with_behavior
from wbfm.utils.general.utils_behavior_annotation import approximate_behavioral_annotation_using_pc1


# In[11]:


for p in all_projects_O2_immob.values():
    try:
        approximate_behavioral_annotation_using_pc1(p)
    except (PermissionError, AttributeError) as e:
        print(e)


# In[12]:


# Reload just the worm_posture_class
for p in all_projects_O2_immob.values():
    del p.worm_posture_class


# In[13]:


# del all_projects_O2_immob['2024-09-30_14-54_ZIM2165_immob_o2_worm5-2024-09-30']


# In[14]:


# p = all_projects_ids['2023-11-21_14-46_wt_worm4-2023-11-21']
# p.plot_neuron_with_kymograph('neuron_010')

# speed = p.worm_posture_class.worm_speed(fluorescence_fps=True, strong_smoothing_before_derivative=True, signed=False)
# ang_speed = p.worm_posture_class.worm_angular_velocity(fluorescence_fps=True)

# # px.line(speed)
# # px.line(speed).show()

# plt.plot(10*speed)
# plt.plot(10*ang_speed)


# In[15]:


# speed = p.worm_posture_class.worm_speed(fluorescence_fps=True, strong_smoothing_before_derivative=True, signed=False)
# df = pd.DataFrame({'speed': speed, 'speed+1': speed.shift(1), 'speed-1': speed.shift(-1)})
# df['nan'] = (df.abs() < 0.001).all(axis=1).astype(int)
# fig = px.line(df)
# fig.show()


# In[16]:


# beh_vec = p.worm_posture_class.beh_annotation(fluorescence_fps=True, reset_index=True, use_pause_to_exclude_other_states=True)
# # beh_vec = pd.Series(p.worm_posture_class.tracking_failure_idx)
# beh_vec.iloc[1390:1400]


# In[17]:


# p = all_projects_O2_immob['2024-09-30_12-01_ZIM2165_immob_o2_worm2-2024-09-30']
# p.worm_posture_class


# In[18]:


# p.worm_posture_class.calc_triggered_average_indices(min_lines=3, ind_preceding=20, state=BehaviorCodes.STIMULUS)


# In[19]:


# for name, p in all_projects_O2_fm.items():
#     # print(p.worm_posture_class)
#     name_dict = p.neuron_name_to_manual_id_mapping(remove_unnamed_neurons=True, confidence_threshold=0, flip_names_and_ids=True)
#     neuron = name_dict.get('AVAL', None)
#     print(p.shortened_name, neuron)
#     plt.show()
#     if neuron is not None:
#         p.plot_neuron_with_kymograph(neuron)
#     # break


# In[20]:


# for name, p in all_projects_O2_fm_mutant.items():
#     # print(p.worm_posture_class)
#     name_dict = p.neuron_name_to_manual_id_mapping(remove_unnamed_neurons=True, confidence_threshold=0, flip_names_and_ids=True)
#     neuron = name_dict.get('AVAL', None)
#     print(p.shortened_name, neuron)
#     plt.show()
#     if neuron is not None:
#         p.plot_neuron_with_kymograph(neuron)
#     # break


# # Same list of neurons with example time series

# In[21]:


from wbfm.utils.visualization.paper_multidataset_triggered_average import PaperExampleTracePlotter
from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes


# In[22]:


project_data_gcamp = all_projects_O2_fm['ZIM2165_Gcamp7b_worm1-2022_11_28']
# For ANTIcorL and RMDV
project_data_gcamp2 = all_projects_O2_fm['2023-11-30_14-31_wt_worm5_FM-2023-11-30']


# In[23]:


wbfm_plotter = PaperExampleTracePlotter(project_data_gcamp, xlim=[0, 120])#, ylim=[-0.35, 0.35])
wbfm_plotter2 = PaperExampleTracePlotter(project_data_gcamp2, xlim=[50, 120])#, ylim=[-0.35, 0.35])


# In[24]:


# neuron_list = ['AQR', #'AUAL', 
#                'IL1LR', 'IL2LL', #'RMDVL', 
#                'BAGL',
#               'URXL', ]#'ANTIcorR']

# output_folder = os.path.join('multiplexing', 'o2_example_traces')

# for i, neuron in enumerate(neuron_list):
#     if 'URX' in neuron:
#         xlabels=True
#     else:
#         xlabels=False
#     try:
#         print(neuron)
#         fig, ax = wbfm_plotter.plot_single_trace(neuron, title=False, round_y_ticks=True, xlabels=xlabels,
#                                        output_foldername=output_folder, use_plotly=True,
#                                        shading_kwargs=dict(additional_shaded_states=[BehaviorCodes.REV, BehaviorCodes.SELF_COLLISION],
#                                                           DEBUG=False))
#         # plt.show()
#         fig.show()
#         # break
#     except ValueError as e:
#         print(e)
#         pass


# In[25]:


# # Second set of neurons
# neuron_list = ['AUAL', 'AUAR', 'RMDVL', 'RMDVR',
#                'ANTIcorR', 'ANTIcorL']

# output_folder = os.path.join('multiplexing', 'o2_example_traces')

# for i, neuron in enumerate(neuron_list):
#     if 'URX' in neuron:
#         xlabels=True
#     else:
#         xlabels=False
#     try:
#         print(neuron)
#         fig, ax = wbfm_plotter2.plot_single_trace(neuron, title=False, round_y_ticks=True, xlabels=xlabels,
#                                        output_foldername=output_folder, use_plotly=True,
#                                        shading_kwargs=dict(additional_shaded_states=[BehaviorCodes.REV, BehaviorCodes.SELF_COLLISION],
#                                                           DEBUG=False))
#         # plt.show()
#         fig.show()

#     except ValueError as e:
#         # print(e)
#         pass


# # FM

# In[26]:


from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
from wbfm.utils.visualization.paper_multidataset_triggered_average import PaperMultiDatasetTriggeredAverage
from wbfm.utils.general.utils_behavior_annotation import approximate_behavioral_annotation_using_pc1


# In[27]:


# triggered_average_gcamp_plotter = PaperMultiDatasetTriggeredAverage(all_projects_O2_fm, calculate_residual=False, calculate_global=False, calculate_turns=False, calculate_self_collision=True,
#                                                                    trace_opt=dict(use_paper_options=False, manual_id_confidence_threshold=0), 
#                                                                     trigger_opt=dict(fixed_num_points_after_event=40)
#                                                                    )


# In[28]:


# trigger_types = [('raw_rev', 'Reversal'),
#                 ('raw_fwd', 'Forward')]

# for raw_neuron in ['AVA', 'AQR', 'IL1L', 'IL2L', 'BAG', 'RMDV', 'ANTIcor', 'AUA', 'URX']:
#     for suffix in ['L', 'R']:
#         if raw_neuron != 'AQR':
#             neuron = f"{raw_neuron}{suffix}"
#         else:
#             neuron = raw_neuron
#             if suffix == 'R':
#                 continue
#         for trigger_type, title in trigger_types:
#             try:
#                 triggered_average_gcamp_plotter.plot_triggered_average_single_neuron(neuron, trigger_type, 
#                                                                                      title=f"{neuron} {title}", show_title=True, #ylim=[-0.09, 0.055],
#                                                                                      output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_fm',
#                                                                                      min_lines=1, xlim=[-5, 12],
#                                                                                     i_figure=4)
#             except Exception as e:
#                 print(e)
#                 raise e


# In[29]:



# # neuron_list = ['AQR', 'PQR', 'URXL', 'URXR', 'AUAL', 'AUAR', 'RMDVL', 'RMDVR', 'ANTIcorR', 'ANTIcorL']
# neuron_list = ['AQR', 'PQR', 'URX', 'AUA', 'RMDV', 'ANTIcor', 'BAG']

# color_list = px.colors.qualitative.D3
# trigger_types = [('raw_rev', 'Reversal'),
#                 ('raw_fwd', 'Forward'),
#                 ('self_collision', 'Self-collision')]

# for trigger_type, title in trigger_types:
#     triggered_average_gcamp_plotter.plot_triggered_average_multiple_neurons(neuron_list, trigger_type, 
#                                                                          title=f"multineuron {title}", show_title=True, #ylim=[-0.09, 0.055],
#                                                                          output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_fm',
#                                                                          min_lines=1, xlim=[-5, 15], color_list=color_list,
#                                                                         i_figure=0, legend=True)


# # Immob

# In[30]:


# all_projects_O2_immob = load_paper_datasets('hannah_O2_immob')


# In[31]:


from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
from wbfm.utils.visualization.paper_multidataset_triggered_average import PaperMultiDatasetTriggeredAverage
from wbfm.utils.general.utils_behavior_annotation import approximate_behavioral_annotation_using_pc1


# In[32]:


# triggered_average_gcamp_plotter_immob = PaperMultiDatasetTriggeredAverage(all_projects_O2_immob, calculate_residual=False, calculate_global=False, calculate_turns=False,
#                                                                    trace_opt=dict(use_paper_options=False, manual_id_confidence_threshold=0),
#                                                                    calculate_stimulus=True, 
#                                                                           trigger_opt=dict(fixed_num_points_after_event=40, ind_delay=6))


# In[33]:


# trigger_types = [('raw_rev', 'Reversal'),
#                 ('raw_fwd', 'Forward'),
#                 ('stimulus', 'Stimulus')]

# for raw_neuron in ['AVA', 'AQR', 'IL1L', 'IL2L', 'BAG', 'RMDV', 'ANTIcor', 'AUA', 'URX']:
#     for suffix in ['L', 'R']:
#         if raw_neuron != 'AQR':
#             neuron = f"{raw_neuron}{suffix}"
#         else:
#             neuron = raw_neuron
#             if suffix == 'R':
#                 continue
#         for trigger_type, title in trigger_types:
#             try:
#                 triggered_average_gcamp_plotter_immob.plot_triggered_average_single_neuron(neuron, trigger_type, 
#                                                                                      title=f"{neuron} {title}", show_title=True, #ylim=[-0.09, 0.055],
#                                                                                      output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_immob',
#                                                                                      min_lines=1, xlim=[-5, 12],
#                                                                                     i_figure=4)
#             except Exception as e:
#                 print(e)
#                 raise e


# In[34]:



# # neuron_list = ['AQR', 'PQR', 'URXL', 'URXR', 'AUAL', 'AUAR', 'RMDVL', 'RMDVR', 'ANTIcorR', 'ANTIcorL']
# neuron_list = ['AQR', 'PQR', 'URX', 'AUA', 'RMDV', 'ANTIcor', 'BAG']
# color_list = px.colors.qualitative.D3
# trigger_types = [('raw_rev', 'Reversal'),
#                 ('raw_fwd', 'Forward'),
#                 ('stimulus', 'Stimulus')]

# for trigger_type, title in trigger_types:
#     triggered_average_gcamp_plotter_immob.plot_triggered_average_multiple_neurons(neuron_list, trigger_type, 
#                                                                          title=f"multineuron {title}", show_title=True, #ylim=[-0.09, 0.055],
#                                                                          output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_immob',
#                                                                          min_lines=2, xlim=[-5, 15], color_list=color_list,
#                                                                         i_figure=0, legend=True)


# ## Same but downshift

# In[35]:


triggered_average_gcamp_plotter_immob_downshift = PaperMultiDatasetTriggeredAverage(all_projects_O2_immob, calculate_residual=False, calculate_global=False, calculate_turns=False,
                                                                   trace_opt=dict(use_paper_options=False, manual_id_confidence_threshold=0),
                                                                   calculate_stimulus=True, 
                                                                                    trigger_opt=dict(fixed_num_points_after_event=40, trigger_on_downshift=True, ind_delay=6))


# In[ ]:


# trigger_types = [#('raw_rev', 'Reversal'),
#                 #('raw_fwd', 'Forward'),
#                 ('stimulus', 'Stimulus')]

# for raw_neuron in ['AVA', 'AQR', 'IL1L', 'IL2L', 'BAG', 'RMDV', 'ANTIcor', 'AUA', 'URX']:
#     for suffix in ['L', 'R']:
#         if raw_neuron != 'AQR':
#             neuron = f"{raw_neuron}{suffix}"
#         else:
#             neuron = raw_neuron
#             if suffix == 'R':
#                 continue
#         for trigger_type, title in trigger_types:
#             try:
#                 triggered_average_gcamp_plotter_immob_downshift.plot_triggered_average_single_neuron(neuron, trigger_type, 
#                                                                                      title=f"{neuron} {title}", show_title=True, #ylim=[-0.09, 0.055],
#                                                                                      output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_immob_downshift',
#                                                                                      min_lines=1, xlim=[-5, 15],
#                                                                                     i_figure=4)
#             except Exception as e:
#                 print(e)
#                 raise e


# In[ ]:


# # Plot all on one plot
# # neuron_list = ['AQR', 'PQR', 'URXL', 'URXR', 'AUAL', 'AUAR', 'RMDVL', 'RMDVR', 'ANTIcorR', 'ANTIcorL']
# neuron_list = ['AQR', 'PQR', 'URX', 'AUA', 'RMDV', 'ANTIcor', 'BAG']

# color_list = px.colors.qualitative.D3
# trigger_types = [#('raw_rev', 'Reversal'),
#                 #('raw_fwd', 'Forward'),
#                 ('stimulus', 'Stimulus')]

# for trigger_type, title in trigger_types:
#     triggered_average_gcamp_plotter_immob_downshift.plot_triggered_average_multiple_neurons(neuron_list, trigger_type, 
#                                                                          title=f"multineuron {title}", show_title=True, #ylim=[-0.09, 0.055],
#                                                                          output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_immob_downshift',
#                                                                          min_lines=2, xlim=[-5, 15], color_list=color_list,
#                                                                         i_figure=0, legend=True)


# # Immob muscle hiscl

# In[ ]:


# triggered_average_gcamp_plotter_immob_hiscl = PaperMultiDatasetTriggeredAverage(all_projects_O2_hiscl, calculate_residual=False, calculate_global=False, calculate_turns=False,
#                                                                    trace_opt=dict(use_paper_options=False, manual_id_confidence_threshold=0),
#                                                                    calculate_stimulus=True, 
#                                                                           trigger_opt=dict(fixed_num_points_after_event=40, ind_delay=6))


# In[ ]:





# # FM_mutant

# In[ ]:


from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
from wbfm.utils.visualization.paper_multidataset_triggered_average import PaperMultiDatasetTriggeredAverage
from wbfm.utils.general.utils_behavior_annotation import approximate_behavioral_annotation_using_pc1


# In[ ]:


# triggered_average_gcamp_plotter_fm_mutant = PaperMultiDatasetTriggeredAverage(all_projects_O2_fm_mutant, calculate_residual=False, calculate_global=False, calculate_turns=False, calculate_self_collision=True,
#                                                                    trace_opt=dict(use_paper_options=False, manual_id_confidence_threshold=0), 
#                                                                     trigger_opt=dict(fixed_num_points_after_event=40)
#                                                                    )


# In[ ]:


# trigger_types = [('raw_rev', 'Reversal'),
#                 ('raw_fwd', 'Forward')]

# for raw_neuron in ['AVA', 'AQR', 'IL1L', 'IL2L', 'BAG', 'RMDV', 'ANTIcor', 'AUA', 'URX']:
#     for suffix in ['L', 'R']:
#         if raw_neuron != 'AQR':
#             neuron = f"{raw_neuron}{suffix}"
#         else:
#             neuron = raw_neuron
#             if suffix == 'R':
#                 continue
#         for trigger_type, title in trigger_types:
#             try:
#                 triggered_average_gcamp_plotter_fm_mutant.plot_triggered_average_single_neuron(neuron, trigger_type, 
#                                                                                      title=f"{neuron} {title}", show_title=True, #ylim=[-0.09, 0.055],
#                                                                                      output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_fm_mutant',
#                                                                                      min_lines=1, xlim=[-5, 12],
#                                                                                     i_figure=4, is_mutant=True)
#             except Exception as e:
#                 print(e)
#                 raise e


# In[ ]:


# # Plot all on one plot
# trigger_types = [('raw_rev', 'Reversal'),
#                 ('raw_fwd', 'Forward'),
#                 ('self_collision', 'Self-collision')]
# # neuron_list = ['AQR', 'PQR', 'URXL', 'URXR', 'AUAL', 'AUAR', 'RMDVL', 'RMDVR', 'ANTIcorR', 'ANTIcorL']
# neuron_list = ['AQR', 'PQR', 'URX', 'AUA', 'RMDV', 'ANTIcor', 'BAG']

# color_list = px.colors.qualitative.D3

# for trigger_type, title in trigger_types:
#     triggered_average_gcamp_plotter_fm_mutant.plot_triggered_average_multiple_neurons(neuron_list, trigger_type, 
#                                                                          title=f"multineuron {title}", show_title=True, #ylim=[-0.09, 0.055],
#                                                                          output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_fm_mutant',
#                                                                          min_lines=2, xlim=[-5, 15], color_list=color_list,
#                                                                         i_figure=0, legend=True)


# # immob_mutant

# In[ ]:


from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
from wbfm.utils.visualization.paper_multidataset_triggered_average import PaperMultiDatasetTriggeredAverage
from wbfm.utils.general.utils_behavior_annotation import approximate_behavioral_annotation_using_pc1


# In[ ]:


# triggered_average_gcamp_plotter_immob_mutant = PaperMultiDatasetTriggeredAverage(all_projects_O2_immob_mutant, calculate_residual=False, calculate_global=False, calculate_turns=False,
#                                                                    trace_opt=dict(use_paper_options=False, manual_id_confidence_threshold=0), calculate_stimulus=True, 
#                                                                     trigger_opt=dict(fixed_num_points_after_event=40, ind_delay=6)
#                                                                    )


# In[ ]:


# trigger_types = [('raw_rev', 'Reversal'),
#                 ('raw_fwd', 'Forward'),
#                 ('stimulus', 'Stimulus')]

# for raw_neuron in ['AVA', 'AQR', 'IL1L', 'IL2L', 'BAG', 'RMDV', 'ANTIcor', 'AUA', 'URX']:
#     for suffix in ['L', 'R']:
#         if raw_neuron != 'AQR':
#             neuron = f"{raw_neuron}{suffix}"
#         else:
#             neuron = raw_neuron
#             if suffix == 'R':
#                 continue
#         for trigger_type, title in trigger_types:
#             try:
#                 triggered_average_gcamp_plotter_immob_mutant.plot_triggered_average_single_neuron(neuron, trigger_type, 
#                                                                                      title=f"{neuron} {title}", show_title=True, #ylim=[-0.09, 0.055],
#                                                                                      output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_immob_mutant',
#                                                                                      min_lines=1, xlim=[-5, 12],
#                                                                                     i_figure=4, is_mutant=True)
#             except Exception as e:
#                 print(e)
#                 # raise e


# In[ ]:


# # Plot all on one plot
# # neuron_list = ['AQR', 'PQR', 'URXL', 'URXR', 'AUAL', 'AUAR', 'RMDVL', 'RMDVR', 'ANTIcorR', 'ANTIcorL']
# neuron_list = ['AQR', 'PQR', 'URX', 'AUA', 'RMDV', 'ANTIcor', 'BAG']
# color_list = px.colors.qualitative.D3
# trigger_types = [('raw_rev', 'Reversal'),
#                 ('raw_fwd', 'Forward'),
#                 ('stimulus', 'Stimulus')]

# for trigger_type, title in trigger_types:
#     triggered_average_gcamp_plotter_immob_mutant.plot_triggered_average_multiple_neurons(neuron_list, trigger_type, 
#                                                                          title=f"multineuron {title}", show_title=True, #ylim=[-0.09, 0.055],
#                                                                          output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_immob_mutant',
#                                                                          min_lines=2, xlim=[-5, 15], color_list=color_list,
#                                                                         i_figure=0, legend=True)


# ## Same but downshift

# In[ ]:


triggered_average_gcamp_plotter_immob_mutant_downshift = PaperMultiDatasetTriggeredAverage(all_projects_O2_immob_mutant, calculate_residual=False, calculate_global=False, calculate_turns=False,
                                                                   trace_opt=dict(use_paper_options=False, manual_id_confidence_threshold=0),
                                                                   calculate_stimulus=True, trigger_opt=dict(fixed_num_points_after_event=30, trigger_on_downshift=True, ind_delay=6))


# In[ ]:


# trigger_types = [#('raw_rev', 'Reversal'),
#                 #('raw_fwd', 'Forward'),
#                 ('stimulus', 'Stimulus')]

# for raw_neuron in ['AVA', 'AQR', 'IL1L', 'IL2L', 'BAG', 'RMDV', 'ANTIcor', 'AUA', 'URX']:
#     for suffix in ['L', 'R']:
#         if raw_neuron != 'AQR':
#             neuron = f"{raw_neuron}{suffix}"
#         else:
#             neuron = raw_neuron
#             if suffix == 'R':
#                 continue
#         for trigger_type, title in trigger_types:
#             try:
#                 triggered_average_gcamp_plotter_immob_mutant_downshift.plot_triggered_average_single_neuron(neuron, trigger_type, 
#                                                                                      title=f"{neuron} {title}", show_title=True, #ylim=[-0.09, 0.055],
#                                                                                      output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_immob_mutant_downshift',
#                                                                                      min_lines=1, xlim=[-5, 15],
#                                                                                     i_figure=4)
#             except Exception as e:
#                 print(e)
#                 # raise e


# In[ ]:


# # Plot all on one plot
# # neuron_list = ['AQR', 'PQR', 'URXL', 'URXR', 'AUAL', 'AUAR', 'RMDVL', 'RMDVR', 'ANTIcorR', 'ANTIcorL']
# neuron_list = ['AQR', 'PQR', 'URX', 'AUA', 'RMDV', 'ANTIcor', 'BAG']

# color_list = px.colors.qualitative.D3
# trigger_types = [#('raw_rev', 'Reversal'),
#                 #('raw_fwd', 'Forward'),
#                 ('stimulus', 'Stimulus')]

# for trigger_type, title in trigger_types:
#     triggered_average_gcamp_plotter_immob_mutant_downshift.plot_triggered_average_multiple_neurons(neuron_list, trigger_type, 
#                                                                          title=f"multineuron {title}", show_title=True, #ylim=[-0.09, 0.055],
#                                                                          output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_immob_mutant_downshift',
#                                                                          min_lines=2, xlim=[-5, 15], color_list=color_list,
#                                                                         i_figure=0, legend=True)


# # Both WT and mutant on the same plot

# In[ ]:


# # URX type neuron, i.e. responds to O2 upshift and forward behavior
# neuron_list = ['AQR', 'URX','AUA']
# is_mutant_vec = [False, True]

# trigger_type = 'stimulus'
# plotter_classes = [triggered_average_gcamp_plotter_immob, triggered_average_gcamp_plotter_immob_mutant]
# for neuron in neuron_list:
#     ax = None
#     for obj, is_mutant in zip(plotter_classes, is_mutant_vec):
#         fig, ax = obj.plot_triggered_average_single_neuron(neuron, trigger_type, 
#                                                          title=f"O2 Upshift", show_title=True, #ylim=[-0.05, 0.5],
#                                                          output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_wt_and_mutant',
#                                                          min_lines=1, xlim=[-5, 12], ax=ax,
#                                                            width_factor_addition=-0.03,
#                                                            round_y_ticks=False, show_y_ticks=True, show_y_label=False,
#                                                          i_figure=4, is_mutant=is_mutant, show_x_ticks=('URX' in neuron), use_plotly=False)


# trigger_type = 'raw_fwd'
# plotter_classes = [triggered_average_gcamp_plotter, triggered_average_gcamp_plotter_fm_mutant]
# for neuron in neuron_list:
#     ax = None
#     for obj, is_mutant in zip(plotter_classes, is_mutant_vec):
#         fig, ax = obj.plot_triggered_average_single_neuron(neuron, trigger_type, 
#                                                          title=f"FWD Triggered", show_title=True, #ylim=[-0.05, 0.5],
#                                                          output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_wt_and_mutant',
#                                                            width_factor_addition=-0.03,
#                                                          min_lines=1, xlim=[-5, 12], ax=ax, round_y_ticks=False, show_y_label=False,
#                                                          i_figure=4, is_mutant=is_mutant, show_x_ticks=('URX' in neuron), use_plotly=False)
        


# In[ ]:


# # Type 2: BAG-type, which responds to O2 downshift and reversals
# neuron_list = ['BAG', 'ANTIcor', 'RMDV']
# is_mutant_vec = [False, True]

# trigger_type = 'stimulus'
# plotter_classes = [triggered_average_gcamp_plotter_immob_downshift, triggered_average_gcamp_plotter_immob_mutant_downshift]

# for neuron in neuron_list:
#     ax = None
#     for obj, is_mutant in zip(plotter_classes, is_mutant_vec):
#         fig, ax = obj.plot_triggered_average_single_neuron(neuron, trigger_type, 
#                                                          title=f"O2 Downshift", show_title=True, #ylim=[-0.05, 0.5],
#                                                          output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_wt_and_mutant',
#                                                          min_lines=1, xlim=[-5, 12], ax=ax, 
#                                                            round_y_ticks=False, show_y_ticks=True, show_y_label=False,
#                                                          i_figure=4, is_mutant=is_mutant, show_x_ticks=False, use_plotly=False,
#                                                           width_factor_addition=-0.03)

# trigger_type = 'raw_rev'
# plotter_classes = [triggered_average_gcamp_plotter, triggered_average_gcamp_plotter_fm_mutant]

# for neuron in neuron_list:
#     ax = None
#     for obj, is_mutant in zip(plotter_classes, is_mutant_vec):
#         fig, ax = obj.plot_triggered_average_single_neuron(neuron, trigger_type, 
#                                                          title=f"REV Triggered", show_title=True, #ylim=[-0.05, 0.5],
#                                                          output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_wt_and_mutant',
#                                                          min_lines=1, xlim=[-5, 12], ax=ax, round_y_ticks=False, show_y_label=False,
#                                                          i_figure=4, is_mutant=is_mutant, show_x_ticks=False, use_plotly=False,
#                                                           width_factor_addition=-0.03)
        


# In[ ]:


# # IL neurons, i.e. responds to O2 upshift and forward behavior, BUT only in hiscl
# # Do not show mutants, which are not hiscl
# neuron_list = ['IL1L', 'IL2L']
# is_mutant_vec = [False, True]

# trigger_type = 'stimulus'
# plotter_classes = [triggered_average_gcamp_plotter_immob_hiscl, 
#                    triggered_average_gcamp_plotter_immob_mutant
#                   ]
# for neuron in neuron_list:
#     ax = None
#     for obj, is_mutant in zip(plotter_classes, is_mutant_vec):
#         if not is_mutant:
#             color = plt.get_cmap('tab20b')(4)
#         else:
#             color = None
#         fig, ax = obj.plot_triggered_average_single_neuron(neuron, trigger_type, 
#                                                          title=f"O2 Upshift", show_title=True, #ylim=[-0.05, 0.5],
#                                                          output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_wt_and_mutant',
#                                                          min_lines=1, xlim=[-5, 12], 
#                                                            color=color,
#                                                            ax=ax, show_individual_lines=False,
#                                                            width_factor_addition=-0.03,
#                                                            round_y_ticks=False, show_y_ticks=True, show_y_label=False,
#                                                          i_figure=4, is_mutant=is_mutant, show_x_ticks=('URX' in neuron), use_plotly=False)

# # trigger_type = 'raw_fwd'
# # plotter_classes = [triggered_average_gcamp_plotter, triggered_average_gcamp_plotter_fm_mutant]
# # for neuron in neuron_list:
# #     ax = None
# #     for obj, is_mutant in zip(plotter_classes, is_mutant_vec):
# #         fig, ax = obj.plot_triggered_average_single_neuron(neuron, trigger_type, 
# #                                                          title=f"Forward Triggered", show_title=True, #ylim=[-0.05, 0.5],
# #                                                          output_folder='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_wt_and_mutant',
# #                                                          min_lines=1, xlim=[-5, 12], ax=ax, round_y_ticks=True, show_y_label=False,
# #                                                          i_figure=4, is_mutant=is_mutant, show_x_ticks=('URX' in neuron), use_plotly=True)
        


# ## Just export legend

# In[ ]:


# from wbfm.utils.general.utils_paper import export_legend_for_paper

# export_legend_for_paper('/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/legend.png')


# ## Do ttests to compare the triggered averages

# In[ ]:


# from wbfm.utils.general.utils_paper import plotly_paper_color_discrete_map, plot_box_multi_axis
# from wbfm.utils.visualization.paper_multidataset_triggered_average import plot_ttests_from_triggered_average_classes


# In[ ]:


# # Everything in one list in order to properly do multiple comparison correction
# neuron_list = ['AQR', 'URX', 'AUA', 'BAG', 'ANTIcor', 'RMDV', 'IL1', 'IL2']
# is_mutant_vec = [False, True]
# output_dir = '/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_wt_and_mutant_ttests'

# # FM
# plotter_classes = [triggered_average_gcamp_plotter, triggered_average_gcamp_plotter_fm_mutant]

# trigger_type = 'raw_fwd'
# all_figs, df_boxplot, df_p_values, _ = plot_ttests_from_triggered_average_classes(neuron_list, plotter_classes, is_mutant_vec, 
#                                                                                trigger_type, output_dir=output_dir)

# trigger_type = 'raw_rev'
# all_figs, df_boxplot, df_p_values, _ = plot_ttests_from_triggered_average_classes(neuron_list, plotter_classes, is_mutant_vec, 
#                                                                                trigger_type, output_dir=output_dir)

# # Also immob to stimulus
# trigger_type = 'stimulus'
# plotter_classes = [triggered_average_gcamp_plotter_immob, triggered_average_gcamp_plotter_immob_mutant]
# all_figs, df_boxplot, df_p_values, _ = plot_ttests_from_triggered_average_classes(neuron_list, plotter_classes, is_mutant_vec, 
#                                                                                trigger_type, output_dir=output_dir)

# # Special, targetted at ILs only (but should still be corrected)
# trigger_type = 'stimulus'
# plotter_classes = [triggered_average_gcamp_plotter_immob_hiscl, triggered_average_gcamp_plotter_immob_mutant]
# all_figs, df_boxplot, df_p_values, _ = plot_ttests_from_triggered_average_classes(neuron_list, plotter_classes, is_mutant_vec, 
#                                                                                trigger_type, output_dir=output_dir)


# # Final version: combine triggered averages, ttests, and example traces together

# In[ ]:


from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
from wbfm.utils.visualization.paper_multidataset_triggered_average import PaperMultiDatasetTriggeredAverage
from wbfm.utils.general.utils_behavior_annotation import approximate_behavioral_annotation_using_pc1
from wbfm.utils.general.utils_paper import plotly_paper_color_discrete_map, plot_box_multi_axis
from wbfm.utils.visualization.paper_multidataset_triggered_average import plot_ttests_from_triggered_average_classes, plot_triggered_averages_from_triggered_average_classes
from wbfm.utils.external.utils_plotly import combine_plotly_figures
from wbfm.utils.visualization.paper_multidataset_triggered_average import PaperExampleTracePlotter
from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes


# ## Load all classes

# In[ ]:


# Make sure physical indices are set
for this_dict in list_of_all_dicts:
    for p in this_dict.values():
        p.use_physical_time = True


# In[ ]:


opt = dict(calculate_global=False, calculate_turns=False,
          trace_opt=dict(use_paper_options=True, channel_mode='dr_over_r_20'))

# FM Wild type
triggered_average_gcamp_plotter = PaperMultiDatasetTriggeredAverage(all_projects_O2_fm, calculate_self_collision=True, **opt,
                                                                    trigger_opt=dict(fixed_num_points_after_event=40), calculate_residual=True)

# Fm mutant
triggered_average_gcamp_plotter_fm_mutant = PaperMultiDatasetTriggeredAverage(all_projects_O2_fm_mutant, calculate_self_collision=True,
                                                                              trigger_opt=dict(fixed_num_points_after_event=40), calculate_residual=True, **opt)


# In[ ]:



opt = dict(calculate_residual=False, calculate_global=False, calculate_turns=False,
          trace_opt=dict(use_paper_options=True, channel_mode='dr_over_r_20'))

# Immob (no O2; only used for final export)
triggered_average_gcamp_plotter_immob_no_O2 = PaperMultiDatasetTriggeredAverage(all_projects_immob, **opt,
                                                                         trigger_opt=dict(fixed_num_points_after_event=40, ind_delay=6, use_manual_annotation=True),
                                                                         calculate_stimulus=False)


# In[ ]:


opt = dict(calculate_residual=False, calculate_global=False, calculate_turns=False,
          trace_opt=dict(use_paper_options=True, channel_mode='dr_over_r_20'))

# Immob (with O2)
triggered_average_gcamp_plotter_immob = PaperMultiDatasetTriggeredAverage(all_projects_O2_immob, **opt,
                                                                         trigger_opt=dict(fixed_num_points_after_event=40, ind_delay=6, use_manual_annotation=True),
                                                                         calculate_stimulus=True)

triggered_average_gcamp_plotter_immob_downshift = PaperMultiDatasetTriggeredAverage(all_projects_O2_immob, **opt,
                                                                                    calculate_stimulus=True,
                                                                                    trigger_opt=dict(fixed_num_points_after_event=40, trigger_on_downshift=True, ind_delay=6, use_manual_annotation=True))


# In[ ]:


# %debug


# In[ ]:


# Immob mutant
triggered_average_gcamp_plotter_immob_mutant = PaperMultiDatasetTriggeredAverage(all_projects_O2_immob_mutant, **opt, 
                                                                                 calculate_stimulus=True, 
                                                                                 trigger_opt=dict(fixed_num_points_after_event=40, ind_delay=6, use_manual_annotation=True))

triggered_average_gcamp_plotter_immob_mutant_downshift = PaperMultiDatasetTriggeredAverage(all_projects_O2_immob_mutant, **opt,
                                                                   calculate_stimulus=True, trigger_opt=dict(fixed_num_points_after_event=40, trigger_on_downshift=True, ind_delay=6, use_manual_annotation=True))

# Immob Hiscl
triggered_average_gcamp_plotter_immob_hiscl = PaperMultiDatasetTriggeredAverage(all_projects_O2_hiscl, **opt, calculate_stimulus=True, 
                                                                                trigger_opt=dict(fixed_num_points_after_event=40, ind_delay=6, use_manual_annotation=True))


# In[ ]:





# In[ ]:





# ## Actually plot

# In[ ]:


project_data_gcamp = all_projects_O2_fm['ZIM2165_Gcamp7b_worm1-2022_11_28']
# For ANTIcorL and RMDV
project_data_gcamp2 = all_projects_O2_fm['2023-11-30_14-31_wt_worm5_FM-2023-11-30']


# In[ ]:


trace_options = dict(channel_mode='dr_over_r_20')

wbfm_plotter = PaperExampleTracePlotter(project_data_gcamp, xlim=[0, 120], trace_options=trace_options)#, ylim=[-0.35, 0.35])
wbfm_plotter2 = PaperExampleTracePlotter(project_data_gcamp2, xlim=[0, 120], trace_options=trace_options)#, ylim=[-0.35, 0.35])


# ### Example traces (left column); shared regardless of ttests

# In[ ]:



neuron_list = ['AQR', #'AUAL', 
               'IL1L', 'IL2L', #'RMDVL', 
               'BAG',
              'URX', ]#'ANTIcorR']
all_figs_examples = {}

opt = dict(title=False, round_y_ticks=True, output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_example_traces', use_plotly=True,
           shading_kwargs=dict(additional_shaded_states=[BehaviorCodes.REV, BehaviorCodes.SELF_COLLISION],
                                                          DEBUG=False))
for i, neuron in enumerate(neuron_list):
    xlabels = 'URX' in neuron
    try:
        fig, ax = wbfm_plotter.plot_single_trace(neuron, xlabels=xlabels, **opt)
        all_figs_examples[neuron] = fig
    except ValueError as e:
        pass


# In[ ]:


# Second set of neurons
neuron_list = ['AUA', #'AUAR', 
               'RMDV', #'RMDVR',
               #'ANTIcor', #'ANTIcorL'
              ]
for i, neuron in enumerate(neuron_list):
    xlabels='URX' in neuron
    try:
        fig, ax = wbfm_plotter2.plot_single_trace(neuron, xlabels=xlabels, **opt)
        all_figs_examples[neuron] = fig
    except ValueError as e:
        pass


# ### Helper function

# In[ ]:


from wbfm.utils.general.utils_paper import apply_figure_settings

# Everything in one list in order to properly do multiple comparison correction
neuron_list = ['AQR', 'URX', 'AUA', 'BAG', #'ANTIcor', 
               'RMDV', 'IL1L', 'IL2L']
is_mutant_vec = [False, True]
output_dir = '/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_wt_and_mutant_with_ttests'
# output_dir = None
opt = dict(show_title=True, #output_dir=output_dir, 
           min_lines=1, xlim=[-5, 12], show_individual_lines=False,
           round_y_ticks=False, show_y_ticks=True, show_y_label=False, show_y_label_only_export=True, i_figure=4,  
           use_plotly=True, to_show=False)


def _combine_and_save(all_figs_box, all_figs_trig, all_figs_examples, width_factor=None, is_supp=False, to_show=True, suffix='', is_immobilized=True, **kwargs):
    for key in all_figs_box.keys():
        print(key)
        if all_figs_examples is not None:
            fig = combine_plotly_figures([all_figs_examples[key], all_figs_box[key], all_figs_trig[key]], 
                                         show_legends=[False, False, False], 
                                         column_widths=[0.3, 0.3, 0.4], **kwargs)
            apply_figure_settings(fig, width_factor=0.6 if is_supp else width_factor, height_factor=0.15)
        else:
            is_supp = True
            # Add y label, but only for the supp version, but not immobilized
            fig_trig = all_figs_trig[key]
            # Adding the y label doesn't work at this stage, because the label and ticks overlap
            # if not is_immobilized:
            #     fig_trig.update_yaxes(title=r"$\Delta R/R_{20}$")
            fig = combine_plotly_figures([all_figs_box[key], all_figs_trig[key]], 
                                         show_legends=[False, False], 
                                         column_widths=[0.3, 0.7], **kwargs)
            # All of these are supp by default
            apply_figure_settings(fig, width_factor=0.45 if width_factor is None else width_factor, height_factor=0.12)
        if output_dir is not None:
            fname = os.path.join(output_dir, f'{key}-{trigger_type}-supp_{is_supp}-combined_triggered_average_ttests-{suffix}.png')
            fig.write_image(fname, scale=3)
            fname = fname.replace('.png', '.svg')
            fig.write_image(fname)
        if to_show:
            fig.show()


# ### triggered averages (middle column) and ttests (right column) for all conditions

# In[ ]:


# Shared variables
ttest_kwargs=dict(dynamic_window_center=True, dynamic_window_length=6)

ttest_dir = '/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_wt_and_mutant_ttests'
trigger_dir = '/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_wt_and_mutant'
trigger_immob_dir = '/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_immob'
trigger_immob_downshift_dir = '/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_immob_downshift'

is_mutant_vec = [True, False]
neuron_list_fwd = ['AQR', 'URX', 'AUA', 'IL1', 'IL2']#, 'RMDV']
neuron_list_upshift = ['AQR', 'URX', 'AUA']#,'RMDV']
neuron_list_hiscl = ['IL1', 'IL2']
neuron_list_rev = ['BAG']


# ### Calculate all p values and subplots
# Note: these p values will not be used directly, because they need to be corrected across conditions

# In[ ]:


## FWD (most neurons)
plotter_classes = [triggered_average_gcamp_plotter_fm_mutant, triggered_average_gcamp_plotter]
trigger_type = 'raw_fwd'
all_figs_trig_fwd, df_boxplot_fwd, df_p_values_fwd, df_idx_range_fwd = plot_ttests_from_triggered_average_classes(neuron_list_fwd, plotter_classes, is_mutant_vec, 
                                                                                    trigger_type, output_dir=None, to_show=False,
                                                                                    ttest_kwargs=ttest_kwargs)
all_figs_box_fwd = plot_triggered_averages_from_triggered_average_classes(neuron_list_fwd, plotter_classes, is_mutant_vec, trigger_type, df_idx_range_fwd, 
                                                                      return_individual_traces=False, output_dir=None, **opt)
## REV (only BAG)
plotter_classes = [triggered_average_gcamp_plotter_fm_mutant, triggered_average_gcamp_plotter]
trigger_type = 'raw_rev'
all_figs_trig_rev, df_boxplot_rev, df_p_values_rev, df_idx_range_rev = plot_ttests_from_triggered_average_classes(neuron_list_rev, plotter_classes, is_mutant_vec, 
                                                                               trigger_type, output_dir=None, to_show=False, ttest_kwargs=ttest_kwargs)
all_figs_box_rev = plot_triggered_averages_from_triggered_average_classes(neuron_list_rev, plotter_classes, is_mutant_vec, trigger_type, df_idx_range_rev, 
                                                                      output_dir=None, **opt)
## IMMOB (upshift is same as fwd without ILs, downshift is same as REV, i.e. just BAG)
trigger_type = 'stimulus'
plotter_classes = [triggered_average_gcamp_plotter_immob_mutant, triggered_average_gcamp_plotter_immob]
all_figs_trig_immob_up, df_boxplot_immob_up, df_p_values_immob_up, df_idx_range_immob_up = plot_ttests_from_triggered_average_classes(neuron_list_upshift, plotter_classes, is_mutant_vec, 
                                                                               trigger_type, output_dir=None, to_show=False, ttest_kwargs=ttest_kwargs)
title = 'Stimulus'
all_figs_box_immob_up = plot_triggered_averages_from_triggered_average_classes(neuron_list_upshift, plotter_classes, is_mutant_vec, trigger_type, df_idx_range_immob_up, 
                                                                      **opt, output_dir=None)
# Downshift
trigger_type = 'stimulus'
plotter_classes = [triggered_average_gcamp_plotter_immob_mutant_downshift, triggered_average_gcamp_plotter_immob_downshift]
all_figs_trig_immob_down, df_boxplot_immob_down, df_p_values_immob_down, df_idx_range_immob_down = plot_ttests_from_triggered_average_classes(neuron_list_rev, plotter_classes, is_mutant_vec, 
                                                                               trigger_type, output_dir=None, to_show=False, ttest_kwargs=ttest_kwargs)
title = 'Stimulus'
all_figs_box_immob_down = plot_triggered_averages_from_triggered_average_classes(neuron_list_rev, plotter_classes, is_mutant_vec, trigger_type, df_idx_range_immob_down, 
                                                                      **opt, output_dir=None)
## Special: hiscl of ILs only
trigger_type = 'stimulus'
plotter_classes = [triggered_average_gcamp_plotter_immob_hiscl]
all_figs_trig_hiscl, df_boxplot_hiscl, df_p_values_hiscl, df_idx_range_hiscl = plot_ttests_from_triggered_average_classes(neuron_list_hiscl, plotter_classes, [False], 
                                                                               trigger_type, output_dir=None, to_show=False, ttest_kwargs=ttest_kwargs)
all_figs_box_hiscl = plot_triggered_averages_from_triggered_average_classes(neuron_list_hiscl, plotter_classes, [False], trigger_type, df_idx_range_hiscl, 
                                                                      **opt, output_dir=None)


# In[ ]:


# %debug


# In[ ]:


# NOT USED
# # Immob rev
# trigger_type = 'raw_rev'
# plotter_classes = [triggered_average_gcamp_plotter_immob, triggered_average_gcamp_plotter_immob_mutant]
# all_figs_trig, df_boxplot, df_p_values = plot_ttests_from_triggered_average_classes(neuron_list, plotter_classes, is_mutant_vec, 
#                                                                                trigger_type, output_dir=ttest_dir, to_show=False, ttest_kwargs=ttest_kwargs)
# all_figs_box = plot_triggered_averages_from_triggered_average_classes(neuron_list, plotter_classes, is_mutant_vec, trigger_type, **opt)

# # Combine and actually plot
# _combine_and_save(all_figs_box, all_figs_trig, all_figs_examples)

# # Immob rev
# trigger_type = 'raw_rev'
# plotter_classes = [triggered_average_gcamp_plotter_immob_hiscl, triggered_average_gcamp_plotter_immob_mutant]
# all_figs_trig, df_boxplot, df_p_values = plot_ttests_from_triggered_average_classes(neuron_list, plotter_classes, is_mutant_vec, 
#                                   1                                             trigger_type, output_dir=ttest_dir, to_show=False, ttest_kwargs=ttest_kwargs)
# all_figs_box = plot_triggered_averages_from_triggered_average_classes(neuron_list, plotter_classes, is_mutant_vec, trigger_type, **opt)

# Combine and actually plot
# _combine_and_save(all_figs_box, all_figs_trig, all_figs_examples)
#
# trigger_type = 'stimulus'
# plotter_classes = [triggered_average_gcamp_plotter_immob_mutant, triggered_average_gcamp_plotter_immob_hiscl]
# all_figs_trig, df_boxplot, df_p_values, df_idx_range = plot_ttests_from_triggered_average_classes(neuron_list, plotter_classes, is_mutant_vec, 
#                                                                                trigger_type, output_dir=ttest_dir, to_show=False, ttest_kwargs=ttest_kwargs)
# all_figs_box = plot_triggered_averages_from_triggered_average_classes(neuron_list, plotter_classes, is_mutant_vec, trigger_type, df_idx_range, 
#                                                                       **opt, output_dir=trigger_immob_downshift_dir, is_immobilized=True)

# # Combine and actually plot
# _combine_and_save(all_figs_box, all_figs_trig, all_figs_examples=None, suffix='-hiscl', is_immobilized=True)


# ### Properly correct the p values, replot the subplots, then plot everything

# In[ ]:


from statsmodels.stats.multitest import multipletests
df_p_values_all = pd.concat([df_p_values_hiscl, df_p_values_immob_down, df_p_values_immob_up, df_p_values_rev, df_p_values_fwd])
df_p_values_all['p_value_corrected'] = multipletests(df_p_values_all['p_value'].values.squeeze(), method='fdr_bh', alpha=0.05)[1]

df_p_values_fm = df_p_values_all[~df_p_values_all.is_immobilized]
df_p_values_immob = df_p_values_all[df_p_values_all.is_immobilized]


# In[ ]:


# Replot panels; mostly copied from above code, but now using these p values

## FWD (most neurons)
plotter_classes = [triggered_average_gcamp_plotter_fm_mutant, triggered_average_gcamp_plotter]
trigger_type = 'raw_fwd'
all_figs_trig_fwd, df_boxplot_fwd, df_p_values_fwd, df_idx_range_fwd = plot_ttests_from_triggered_average_classes(neuron_list_fwd, plotter_classes, is_mutant_vec, 
                                                                                    trigger_type, output_dir=ttest_dir, to_show=False,
                                                                                    df_p_values_all=df_p_values_fm, ttest_kwargs=ttest_kwargs)
all_figs_box_fwd = plot_triggered_averages_from_triggered_average_classes(neuron_list_fwd, plotter_classes, is_mutant_vec, trigger_type, df_idx_range_fwd, 
                                                                      return_individual_traces=False, output_dir=trigger_dir, **opt)
## REV (only BAG)
plotter_classes = [triggered_average_gcamp_plotter_fm_mutant, triggered_average_gcamp_plotter]
trigger_type = 'raw_rev'
all_figs_trig_rev, df_boxplot_rev, df_p_values_rev, df_idx_range_rev = plot_ttests_from_triggered_average_classes(neuron_list_rev, plotter_classes, is_mutant_vec, 
                                                                               trigger_type, output_dir=ttest_dir, to_show=False, 
                                                                                                                  df_p_values_all=df_p_values_fm, ttest_kwargs=ttest_kwargs)
all_figs_box_rev = plot_triggered_averages_from_triggered_average_classes(neuron_list_rev, plotter_classes, is_mutant_vec, trigger_type, df_idx_range_rev, 
                                                                      output_dir=trigger_dir, **opt)
## IMMOB (upshift is same as fwd without ILs, downshift is same as REV, i.e. just BAG)
trigger_type = 'stimulus'
plotter_classes = [triggered_average_gcamp_plotter_immob_mutant, triggered_average_gcamp_plotter_immob]
all_figs_trig_immob_up, df_boxplot_immob_up, df_p_values_immob_up, df_idx_range_immob_up = plot_ttests_from_triggered_average_classes(neuron_list_upshift, plotter_classes, is_mutant_vec, 
                                                                               trigger_type, output_dir=ttest_dir, to_show=False, 
                                                                                                                                      df_p_values=df_p_values_immob, ttest_kwargs=ttest_kwargs)
title = 'Stimulus'
all_figs_box_immob_up = plot_triggered_averages_from_triggered_average_classes(neuron_list_upshift, plotter_classes, is_mutant_vec, trigger_type, df_idx_range_immob_up, 
                                                                      **opt, output_dir=trigger_immob_dir, annotation_kwargs=dict(is_immobilized=True))
# Downshift
trigger_type = 'stimulus'
plotter_classes = [triggered_average_gcamp_plotter_immob_mutant_downshift, triggered_average_gcamp_plotter_immob_downshift]
all_figs_trig_immob_down, df_boxplot_immob_down, df_p_values_immob_down, df_idx_range_immob_down = plot_ttests_from_triggered_average_classes(neuron_list_rev, plotter_classes, is_mutant_vec, 
                                                                               trigger_type, output_dir=ttest_dir, to_show=False, df_p_values=df_p_values_immob, ttest_kwargs=ttest_kwargs)
title = 'Stimulus'
all_figs_box_immob_down = plot_triggered_averages_from_triggered_average_classes(neuron_list_rev, plotter_classes, is_mutant_vec, trigger_type, df_idx_range_immob_down, 
                                                                      **opt, output_dir=trigger_immob_downshift_dir, annotation_kwargs=dict(is_immobilized=True))
## Special: hiscl of ILs only
trigger_type = 'stimulus'
plotter_classes = [triggered_average_gcamp_plotter_immob_hiscl]
all_figs_trig_hiscl, df_boxplot_hiscl, df_p_values_hiscl, df_idx_range_hiscl = plot_ttests_from_triggered_average_classes(neuron_list_hiscl, plotter_classes, [False], 
                                                                               trigger_type, output_dir=ttest_dir, to_show=False, df_p_values=df_p_values_immob, ttest_kwargs=ttest_kwargs)
all_figs_box_hiscl = plot_triggered_averages_from_triggered_average_classes(neuron_list_hiscl, plotter_classes, [False], trigger_type, df_idx_range_hiscl, 
                                                                      **opt, output_dir=trigger_immob_dir, annotation_kwargs=dict(is_immobilized=True))


# In[ ]:


# Combine and actually plot,
# _combine_and_save(all_figs_box_fwd, all_figs_trig_fwd, all_figs_examples, to_show=True)
_combine_and_save(all_figs_box_fwd, all_figs_trig_fwd, all_figs_examples=None, to_show=True)
# Combine and actually plot
# _combine_and_save(all_figs_box_rev, all_figs_trig_rev, all_figs_examples, to_show=True)
_combine_and_save(all_figs_box_rev, all_figs_trig_rev, all_figs_examples=None, to_show=True)
# Combine and actually plot
_combine_and_save(all_figs_box_hiscl, all_figs_trig_hiscl, all_figs_examples=None, suffix='-hiscl', is_immobilized=True, to_show=True)
# Combine and actually plot
_combine_and_save(all_figs_box_immob_up, all_figs_trig_immob_up, all_figs_examples=None, suffix='-upshift', is_immobilized=True, to_show=True)
# Combine and actually plot
_combine_and_save(all_figs_box_immob_down, all_figs_trig_immob_down, all_figs_examples=None, suffix='-downshift', is_immobilized=True, to_show=True)


# ## Export for alternative modeling

# In[ ]:


# Add gfp class, just for exporting
opt = dict(calculate_global=False, calculate_turns=False,
          trace_opt=dict(use_paper_options=True, channel_mode='dr_over_r_20'))

# FM Wild type
triggered_average_gcamp_plotter_GFP = PaperMultiDatasetTriggeredAverage(all_projects_gfp, calculate_self_collision=False, **opt,
                                                                        trigger_opt=dict(fixed_num_points_after_event=40), calculate_residual=True)


# In[ ]:


from wbfm.utils.external.utils_pandas import combine_columns_with_suffix

all_classes = {'triggered_average_gcamp_plotter': triggered_average_gcamp_plotter,
               'triggered_average_gcamp_plotter_mutant': triggered_average_gcamp_plotter_fm_mutant,
               'triggered_average_gcamp_plotter_immob': triggered_average_gcamp_plotter_immob,
               'triggered_average_gcamp_plotter_immob_no_O2': triggered_average_gcamp_plotter_immob_no_O2,
               'triggered_average_gcamp_plotter_immob_downshift': triggered_average_gcamp_plotter_immob_downshift,
               'triggered_average_gcamp_plotter_immob_mutant': triggered_average_gcamp_plotter_immob_mutant,
               'triggered_average_gcamp_plotter_immob_mutant_downshift': triggered_average_gcamp_plotter_immob_mutant_downshift,
               'triggered_average_gcamp_plotter_immob_hiscl': triggered_average_gcamp_plotter_immob_hiscl,
               'triggered_average_gcamp_plotter_GFP': triggered_average_gcamp_plotter_GFP}

folder_name = '/lisc/data/scratch/neurobiology/zimmer/fieseler/paper/triggered_average_dataframes'

for name, trigger_class in tqdm(all_classes.items()):
    
    all_trigger_types = list(trigger_class.intermediates_dict.keys())
    print(name, all_trigger_types)
    for trigger_type in all_trigger_types:
        
        df = trigger_class.get_df_triggered_from_trigger_type_all_traces_as_df(trigger_type)

        fname = f'{name}-{trigger_type}.h5'
        fname = os.path.join(folder_name, fname)
        df.to_hdf(fname, key='df_with_missing')
        
        # ALSO: generate and save the L/R pooled versions
        df_combined = combine_columns_with_suffix(df, DEBUG=False)
        
        fname = f'{name}-{trigger_type}-LR_pooled.h5'
        fname = os.path.join(folder_name, fname)
        df_combined.to_hdf(fname, key='df_with_missing')


# In[ ]:


# df_combined['AVA'].head()


# In[ ]:


# df = triggered_average_gcamp_plotter.get_df_triggered_from_trigger_type_all_traces_as_df('raw_rev', melt_neuron='AVAL')

# # df_grouped = df.groupby(['dataset_name', 'trial_idx', 'before']).median().reset_index()

# px.box(df, x='dataset_name', y='value', color='before', #points='all',
#        category_orders={'before': ['True', 'False', True, False]})


# # Also export an excel with number of IDs and events

# In[ ]:


from wbfm.utils.general.hardcoded_paths import get_triggered_average_dataframe_fname, get_all_trigger_suffixes, get_triggered_average_modeling_dir
from wbfm.utils.general.hardcoded_paths import neurons_with_confident_ids


# In[ ]:


all_trigger_suffixes = get_all_trigger_suffixes()
all_dfs = {}
data_dir = get_triggered_average_modeling_dir()
print(data_dir)
for suffix in tqdm(all_trigger_suffixes):
    fname = f'triggered_average_gcamp_plotter'
    fname += suffix + '.h5'
    fname = os.path.join(data_dir, fname)
    # fname, _ = get_triggered_average_dataframe_fname(suffix=suffix)
    try:
        all_dfs[suffix] = pd.read_hdf(fname)
    except FileNotFoundError:
        print(f"Not found: {fname}")
        pass


# In[ ]:


# all_dfs.keys()


# In[ ]:


from wbfm.utils.general.hardcoded_paths import excel_event_full_description
from wbfm.utils.general.utils_paper import add_figure_panel_references_to_df
all_events_dict = {}

trigger_type_to_dataset_name = {'raw_rev': 'num_datasets_freely_moving_gcamp',
                                   'immob-raw_rev': 'num_datasets_immob_gcamp',
                                   'mutant-raw_rev': 'num_datasets_mutant_gcamp',
                                   'immob_no_O2-raw_rev': 'num_datasets_immob_no_O2',
                                   'immob_mutant-raw_rev': 'num_datasets_mutant_immob',
                                   'immob_hiscl-raw_rev': 'num_datasets_immob_hiscl',
                                   'GFP-raw_rev': 'num_datasets_gfp'}

neuron_list = neurons_with_confident_ids()
neuron_list.extend(neurons_with_confident_ids(combine_left_right=True))
neuron_list = list(set(neuron_list))

for trigger_type, df in all_dfs.items():
    for neuron in neuron_list:
        # Will append if doing multiple trigger types
        neuron_dict = all_events_dict.get(neuron, defaultdict(lambda: np.nan))
        # Remove -LR_pooled; even if the neuron already exists, the value should be the same
        trigger_type = trigger_type.strip('-_').replace('-LR_pooled', '')
        
        # Otherwise, leave it as the default value (nan)
        if neuron in df:
            df_neuron = df[neuron]
            # Add rows for the total number of datasets
            if trigger_type in trigger_type_to_dataset_name:
                neuron_dict[trigger_type_to_dataset_name[trigger_type]] = len(df_neuron.columns.get_level_values('dataset_name').unique())
            neuron_dict[trigger_type] = df_neuron.shape[1]

        all_events_dict[neuron] = neuron_dict
df = pd.DataFrame(all_events_dict)

# Add a column referring to specific figure panels
df['Figure panel references'] = ''
add_figure_panel_references_to_df(df)

# Add a column with full descriptions
df['Description'] = pd.Series(df.index).astype(str).map(excel_event_full_description()).values
df = df[df.Description != 'DROP']


# In[ ]:


# df['DB01']


# In[ ]:



fname = 'supplement/ids/events_per_id.xlsx'
df.to_excel(fname)


# # Alternative: collision-triggered averages (only showing BAG)

# In[ ]:


ttest_kwargs=dict(dynamic_window_center=True, dynamic_window_length=6)
opt = dict(show_title=True, #output_dir=output_dir, 
           min_lines=1, xlim=[-5, 12], show_individual_lines=False,
           round_y_ticks=False, show_y_ticks=True, show_y_label=True, show_y_label_only_export=True, i_figure=4,  
           use_plotly=True, to_show=False)

## Only BAG
neuron_list_col = ['BAG']
plotter_classes = [triggered_average_gcamp_plotter_fm_mutant, triggered_average_gcamp_plotter]
is_mutant_vec = [True, False]
trigger_type = 'residual_collision'
all_figs_trig_col, df_boxplot_col, df_p_values_col, df_idx_range_col = plot_ttests_from_triggered_average_classes(neuron_list_col, plotter_classes, is_mutant_vec, 
                                                                               trigger_type, output_dir=None, to_show=False, ttest_kwargs=ttest_kwargs)
all_figs_box_col = plot_triggered_averages_from_triggered_average_classes(neuron_list_col, plotter_classes, is_mutant_vec, trigger_type, df_idx_range_col, 
                                                                      output_dir=None, annotation_kwargs=dict(is_residual=True), **opt)


# In[ ]:


_combine_and_save(all_figs_box_col, all_figs_trig_col, all_figs_examples=None, to_show=True, suffix='collision', width_factor=0.5)


# ## Legends for everything

# In[ ]:


from wbfm.utils.general.utils_paper import export_legend_for_paper


# In[ ]:


output_foldername = '/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing'
fname = os.path.join(output_foldername, 'o2_supp_legend.png')
export_legend_for_paper(fname, o2_supp=True)


# In[ ]:


output_foldername = '/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing'
fname = os.path.join(output_foldername, 'triple_plots_legend.png')
export_legend_for_paper(fname, triple_plots=True)


# In[ ]:


output_foldername = '/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing'
fname = os.path.join(output_foldername, 'o2_legend.png')
export_legend_for_paper(fname, o2=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




