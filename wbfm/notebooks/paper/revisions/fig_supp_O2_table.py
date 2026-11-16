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
from wbfm.utils.general.utils_hardcoded import neurons_with_confident_ids


# In[2]:


from sklearn.decomposition import PCA
from wbfm.utils.visualization.plot_traces import make_grid_plot_from_dataframe
import seaborn as sns
from wbfm.utils.visualization.behavior_comparison_plots import NeuronToMultivariateEncoding


# In[3]:


# Load multiple datasets
from wbfm.utils.general.utils_hardcoded import load_paper_datasets
all_projects_O2_fm = load_paper_datasets(['hannah_O2_fm', 'gcamp'])
all_projects_O2_immob_mutant = load_paper_datasets('hannah_O2_immob_mutant')


# In[4]:


all_projects_O2_fm_mutant = load_paper_datasets('hannah_O2_fm_mutant')


# In[5]:


all_projects_O2_immob = load_paper_datasets('immob_o2')


# In[6]:


all_projects_immob = load_paper_datasets('immob')


# In[7]:


all_projects_O2_hiscl = load_paper_datasets('O2_hiscl')


# In[8]:


all_projects_gfp = load_paper_datasets('gfp')


# In[9]:


list_of_all_dicts = [all_projects_O2_fm, all_projects_O2_fm_mutant, all_projects_O2_immob, all_projects_O2_immob_mutant,
                     all_projects_immob, all_projects_O2_hiscl, all_projects_gfp]


# # Optional: Remake paper traces dataframes with updated IDs

# In[10]:


# list_of_all_dicts = [#all_projects_O2_fm, 
#                      all_projects_O2_fm_mutant, 
#     # all_projects_O2_immob, all_projects_O2_immob_mutant,
#     #                  all_projects_O2_hiscl
# ]

# # Optional: clear the trace cache
# for project_dict in tqdm(list_of_all_dicts):
#     for name, project_data in tqdm(project_dict.items()):
#         # Delete
#         project_data.data_cacher.clear_disk_cache(delete_invalid_indices=False, delete_traces=True)

#         # Remake
#         df_traces = project_data.calc_paper_traces()
#         df_res = project_data.calc_paper_traces(residual_mode='pca')
#         df_global = project_data.calc_paper_traces(residual_mode='pca_global')


# # Final version: combine triggered averages, ttests, and example traces together

# In[11]:


from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
from wbfm.utils.visualization.paper_multidataset_triggered_average import PaperMultiDatasetTriggeredAverage
from wbfm.utils.general.utils_behavior_annotation import approximate_behavioral_annotation_using_pc1
from wbfm.utils.general.utils_paper import plotly_paper_color_discrete_map, plot_box_multi_axis
from wbfm.utils.visualization.paper_multidataset_triggered_average import plot_ttests_from_triggered_average_classes, plot_triggered_averages_from_triggered_average_classes
from wbfm.utils.external.utils_plotly import combine_plotly_figures
from wbfm.utils.visualization.paper_multidataset_triggered_average import PaperExampleTracePlotter
from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes


# ## Load all classes

# In[12]:


# Make sure physical indices are set
for this_dict in list_of_all_dicts:
    for p in this_dict.values():
        p.use_physical_time = True


# In[75]:


opt = dict(calculate_global=False, calculate_turns=False,
          trace_opt=dict(use_paper_options=True, channel_mode='dr_over_r_20'))


# In[ ]:


# FM Wild type
triggered_average_gcamp_plotter = PaperMultiDatasetTriggeredAverage(all_projects_O2_fm, calculate_self_collision=True, **opt,
                                                                    trigger_opt=dict(fixed_num_points_after_event=40), calculate_residual=True)


# In[76]:


# Fm mutant
triggered_average_gcamp_plotter_fm_mutant = PaperMultiDatasetTriggeredAverage(all_projects_O2_fm_mutant, calculate_self_collision=False,
                                                                              trigger_opt=dict(fixed_num_points_after_event=40, use_manual_annotation=True), calculate_residual=True, **opt)


# In[77]:


# Fm mutant, but don't use manual annotation for collisions
triggered_average_gcamp_plotter_fm_mutant_collision = PaperMultiDatasetTriggeredAverage(all_projects_O2_fm_mutant, calculate_self_collision=True,
                                                                              trigger_opt=dict(fixed_num_points_after_event=40, use_manual_annotation=False), calculate_residual=True, **opt)


# In[16]:


# triggered_average_gcamp_plotter.print_statistics()


# In[17]:


# del triggered_average_gcamp_plotter.intermediates_dict[0]


# In[18]:


# triggered_average_gcamp_plotter


# In[19]:



opt = dict(calculate_residual=False, calculate_global=False, calculate_turns=False,
          trace_opt=dict(use_paper_options=True, channel_mode='dr_over_r_20'))

# Immob (no O2; only used for final export)
triggered_average_gcamp_plotter_immob_no_O2 = PaperMultiDatasetTriggeredAverage(all_projects_immob, **opt,
                                                                         trigger_opt=dict(fixed_num_points_after_event=40, ind_delay=6, use_manual_annotation=True),
                                                                         calculate_stimulus=False)


# In[20]:


opt = dict(calculate_residual=False, calculate_global=False, calculate_turns=False,
          trace_opt=dict(use_paper_options=True, channel_mode='dr_over_r_20'))

# Immob (with O2)
triggered_average_gcamp_plotter_immob = PaperMultiDatasetTriggeredAverage(all_projects_O2_immob, **opt,
                                                                         trigger_opt=dict(fixed_num_points_after_event=40, ind_delay=6, use_manual_annotation=True),
                                                                         calculate_stimulus=True)

triggered_average_gcamp_plotter_immob_downshift = PaperMultiDatasetTriggeredAverage(all_projects_O2_immob, **opt,
                                                                                    calculate_stimulus=True,
                                                                                    trigger_opt=dict(fixed_num_points_after_event=40, trigger_on_downshift=True, ind_delay=6, use_manual_annotation=True))


# In[21]:


# %debug


# In[22]:


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

# In[23]:


project_data_gcamp = all_projects_O2_fm['ZIM2165_Gcamp7b_worm1-2022_11_28']
# For ANTIcorL and RMDV
project_data_gcamp2 = all_projects_O2_fm['2023-11-30_14-31_wt_worm5_FM-2023-11-30']


# In[24]:


trace_options = dict(channel_mode='dr_over_r_20')

wbfm_plotter = PaperExampleTracePlotter(project_data_gcamp, xlim=[0, 120], trace_options=trace_options)#, ylim=[-0.35, 0.35])
wbfm_plotter2 = PaperExampleTracePlotter(project_data_gcamp2, xlim=[0, 120], trace_options=trace_options)#, ylim=[-0.35, 0.35])


# ### Example traces (left column); shared regardless of ttests

# In[25]:



# neuron_list = ['AQR', #'AUAL', 
#                'IL1L', 'IL2L', #'RMDVL', 
#                'BAG',
#               'URX', ]#'ANTIcorR']
# all_figs_examples = {}

# opt = dict(title=False, round_y_ticks=True, output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/revisions/fig_supp_o2', use_plotly=True,
#            shading_kwargs=dict(additional_shaded_states=[BehaviorCodes.REV, BehaviorCodes.SELF_COLLISION],
#                                                           DEBUG=False))
# for i, neuron in enumerate(neuron_list):
#     xlabels = 'URX' in neuron
#     try:
#         fig, ax = wbfm_plotter.plot_single_trace(neuron, xlabels=xlabels, **opt)
#         all_figs_examples[neuron] = fig
#     except ValueError as e:
#         pass


# In[26]:


# # Second set of neurons
# neuron_list = ['AUA', #'AUAR', 
#                'RMDV', #'RMDVR',
#                #'ANTIcor', #'ANTIcorL'
#               ]
# for i, neuron in enumerate(neuron_list):
#     xlabels='URX' in neuron
#     try:
#         fig, ax = wbfm_plotter2.plot_single_trace(neuron, xlabels=xlabels, **opt)
#         all_figs_examples[neuron] = fig
#     except ValueError as e:
#         pass


# ### Helper function

# In[27]:


from wbfm.utils.general.utils_paper import apply_figure_settings

# Everything in one list in order to properly do multiple comparison correction
# neuron_list = ['AQR', 'URX', 'AUA', 'BAG', #'ANTIcor', 
#                'RMDV', 'IL1L', 'IL2L']
is_mutant_vec = [False, True]
output_dir = '/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/revisions/fig_supp_o2/wt_and_mutant_ttests'
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

# In[28]:


# Shared variables
ttest_kwargs=dict(dynamic_window_center=True, dynamic_window_length=6)

ttest_dir = '/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/revisions/fig_supp_o2/wt_and_mutant_ttests'
trigger_dir = '/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/revisions/fig_supp_o2/wt_and_mutant'
trigger_immob_dir = '/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_immob'
trigger_immob_downshift_dir = '/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing/o2_trigger_immob_downshift'

is_mutant_vec = [True, False]

neuron_list_fwd = neurons_with_confident_ids(combine_left_right=True)
neuron_list_upshift = neurons_with_confident_ids(combine_left_right=True)
neuron_list_hiscl = neurons_with_confident_ids(combine_left_right=True)
neuron_list_rev = neurons_with_confident_ids(combine_left_right=True)

# neuron_list_fwd = ['AQR', 'URX', 'AUA', 'IL1', 'IL2']#, 'RMDV']
# neuron_list_upshift = ['AQR', 'URX', 'AUA']#,'RMDV']
# neuron_list_hiscl = ['IL1', 'IL2']
# neuron_list_rev = ['BAG']


# ### Calculate all p values and subplots
# Note: these p values will not be used directly, because they need to be corrected across conditions

# In[29]:


## FWD (most neurons)
plotter_classes = [triggered_average_gcamp_plotter_fm_mutant, triggered_average_gcamp_plotter]
trigger_type = 'raw_fwd'
all_figs_trig_fwd_raw, df_boxplot_fwd_raw, df_p_values_fwd_raw, df_idx_range_fwd = plot_ttests_from_triggered_average_classes(neuron_list_fwd, plotter_classes, is_mutant_vec, 
                                                                                    trigger_type, output_dir=None, to_show=False,
                                                                                    ttest_kwargs=ttest_kwargs, fdr_method=None,
                                                                                                                             combine_left_right=True)
# all_figs_box_fwd = plot_triggered_averages_from_triggered_average_classes(neuron_list_fwd, plotter_classes, is_mutant_vec, trigger_type, df_idx_range_fwd, 
#                                                                       return_individual_traces=False, output_dir=None, **opt)
## REV (only BAG)
plotter_classes = [triggered_average_gcamp_plotter_fm_mutant, triggered_average_gcamp_plotter]
trigger_type = 'raw_rev'
all_figs_trig_rev_raw, df_boxplot_rev_raw, df_p_values_rev_raw, df_idx_range_rev = plot_ttests_from_triggered_average_classes(neuron_list_rev, plotter_classes, is_mutant_vec, 
                                                                               trigger_type, output_dir=None, to_show=False, ttest_kwargs=ttest_kwargs,
                                                                                                                 fdr_method=None,
                                                                                                                             combine_left_right=True)
# all_figs_box_rev = plot_triggered_averages_from_triggered_average_classes(neuron_list_rev, plotter_classes, is_mutant_vec, trigger_type, df_idx_range_rev, 
#                                                                       output_dir=None, **opt)
## IMMOB (upshift is same as fwd without ILs, downshift is same as REV, i.e. just BAG)
# trigger_type = 'stimulus'
# plotter_classes = [triggered_average_gcamp_plotter_immob_mutant, triggered_average_gcamp_plotter_immob]
# all_figs_trig_immob_up, df_boxplot_immob_up, df_p_values_immob_up, df_idx_range_immob_up = plot_ttests_from_triggered_average_classes(neuron_list_upshift, plotter_classes, is_mutant_vec, 
#                                                                                trigger_type, output_dir=None, to_show=False, ttest_kwargs=ttest_kwargs)
# title = 'Stimulus'
# all_figs_box_immob_up = plot_triggered_averages_from_triggered_average_classes(neuron_list_upshift, plotter_classes, is_mutant_vec, trigger_type, df_idx_range_immob_up, 
#                                                                       **opt, output_dir=None)
# # Downshift
# trigger_type = 'stimulus'
# plotter_classes = [triggered_average_gcamp_plotter_immob_mutant_downshift, triggered_average_gcamp_plotter_immob_downshift]
# all_figs_trig_immob_down, df_boxplot_immob_down, df_p_values_immob_down, df_idx_range_immob_down = plot_ttests_from_triggered_average_classes(neuron_list_rev, plotter_classes, is_mutant_vec, 
#                                                                                trigger_type, output_dir=None, to_show=False, ttest_kwargs=ttest_kwargs)
# title = 'Stimulus'
# all_figs_box_immob_down = plot_triggered_averages_from_triggered_average_classes(neuron_list_rev, plotter_classes, is_mutant_vec, trigger_type, df_idx_range_immob_down, 
#                                                                       **opt, output_dir=None)
# ## Special: hiscl of ILs only
# trigger_type = 'stimulus'
# plotter_classes = [triggered_average_gcamp_plotter_immob_hiscl]
# all_figs_trig_hiscl, df_boxplot_hiscl, df_p_values_hiscl, df_idx_range_hiscl = plot_ttests_from_triggered_average_classes(neuron_list_hiscl, plotter_classes, [False], 
#                                                                                trigger_type, output_dir=None, to_show=False, ttest_kwargs=ttest_kwargs)
# all_figs_box_hiscl = plot_triggered_averages_from_triggered_average_classes(neuron_list_hiscl, plotter_classes, [False], trigger_type, df_idx_range_hiscl, 
#                                                                       **opt, output_dir=None)


# In[30]:


df_p_values_rev_raw.head()


# ### Properly correct the p values, replot the subplots, then plot everything

# In[31]:


from statsmodels.stats.multitest import multipletests
df_p_values_all = pd.concat([#df_p_values_hiscl, df_p_values_immob_down, df_p_values_immob_up, 
                             df_p_values_rev_raw, df_p_values_fwd_raw])
# Index is (neuron, is_mutant_str, trigger_type) after library fix — unique, no collisions
raw = df_p_values_all['p_value'].dropna()
df_p_values_all.loc[raw.index, 'p_value_corrected'] = multipletests(raw.values.squeeze(), method='fdr_bh', alpha=0.05)[1]

df_p_values_fm = df_p_values_all[~df_p_values_all.is_immobilized]
# df_p_values_immob = df_p_values_all[df_p_values_all.is_immobilized]
df_p_values_fm.index.names


# In[32]:


# Replot panels; mostly copied from above code, but now using these p values

## FWD (most neurons)
plotter_classes = [triggered_average_gcamp_plotter_fm_mutant, triggered_average_gcamp_plotter]
trigger_type = 'raw_fwd'
all_figs_trig_fwd, df_boxplot_fwd, df_p_values_fwd, df_idx_range_fwd = plot_ttests_from_triggered_average_classes(neuron_list_fwd, plotter_classes, is_mutant_vec, 
                                                                                    trigger_type, output_dir=ttest_dir, to_show=False,
                                                                                    df_p_values=df_p_values_fm, ttest_kwargs=ttest_kwargs,
                                                                                                                             combine_left_right=True)
# all_figs_box_fwd = plot_triggered_averages_from_triggered_average_classes(neuron_list_fwd, plotter_classes, is_mutant_vec, trigger_type, df_idx_range_fwd, 
#                                                                       return_individual_traces=False, output_dir=trigger_dir, **opt)
## REV (only BAG)
plotter_classes = [triggered_average_gcamp_plotter_fm_mutant, triggered_average_gcamp_plotter]
trigger_type = 'raw_rev'
all_figs_trig_rev, df_boxplot_rev, df_p_values_rev, df_idx_range_rev = plot_ttests_from_triggered_average_classes(neuron_list_rev, plotter_classes, is_mutant_vec, 
                                                                               trigger_type, output_dir=ttest_dir, to_show=False, 
                                                                                                                  df_p_values=df_p_values_fm, ttest_kwargs=ttest_kwargs,
                                                                                                                             combine_left_right=True)
# all_figs_box_rev = plot_triggered_averages_from_triggered_average_classes(neuron_list_rev, plotter_classes, is_mutant_vec, trigger_type, df_idx_range_rev, 
#                                                                       output_dir=trigger_dir, **opt)
## IMMOB (upshift is same as fwd without ILs, downshift is same as REV, i.e. just BAG)
# trigger_type = 'stimulus'
# plotter_classes = [triggered_average_gcamp_plotter_immob_mutant, triggered_average_gcamp_plotter_immob]
# all_figs_trig_immob_up, df_boxplot_immob_up, df_p_values_immob_up, df_idx_range_immob_up = plot_ttests_from_triggered_average_classes(neuron_list_upshift, plotter_classes, is_mutant_vec, 
#                                                                                trigger_type, output_dir=ttest_dir, to_show=False, 
#                                                                                                                                       df_p_values=df_p_values_immob, ttest_kwargs=ttest_kwargs)
# title = 'Stimulus'
# all_figs_box_immob_up = plot_triggered_averages_from_triggered_average_classes(neuron_list_upshift, plotter_classes, is_mutant_vec, trigger_type, df_idx_range_immob_up, 
#                                                                       **opt, output_dir=trigger_immob_dir, annotation_kwargs=dict(is_immobilized=True))
# # Downshift
# trigger_type = 'stimulus'
# plotter_classes = [triggered_average_gcamp_plotter_immob_mutant_downshift, triggered_average_gcamp_plotter_immob_downshift]
# all_figs_trig_immob_down, df_boxplot_immob_down, df_p_values_immob_down, df_idx_range_immob_down = plot_ttests_from_triggered_average_classes(neuron_list_rev, plotter_classes, is_mutant_vec, 
#                                                                                trigger_type, output_dir=ttest_dir, to_show=False, df_p_values=df_p_values_immob, ttest_kwargs=ttest_kwargs)
# title = 'Stimulus'
# all_figs_box_immob_down = plot_triggered_averages_from_triggered_average_classes(neuron_list_rev, plotter_classes, is_mutant_vec, trigger_type, df_idx_range_immob_down, 
#                                                                       **opt, output_dir=trigger_immob_downshift_dir, annotation_kwargs=dict(is_immobilized=True))
# ## Special: hiscl of ILs only
# trigger_type = 'stimulus'
# plotter_classes = [triggered_average_gcamp_plotter_immob_hiscl]
# all_figs_trig_hiscl, df_boxplot_hiscl, df_p_values_hiscl, df_idx_range_hiscl = plot_ttests_from_triggered_average_classes(neuron_list_hiscl, plotter_classes, [False], 
#                                                                                trigger_type, output_dir=ttest_dir, to_show=False, df_p_values=df_p_values_immob, ttest_kwargs=ttest_kwargs)
# all_figs_box_hiscl = plot_triggered_averages_from_triggered_average_classes(neuron_list_hiscl, plotter_classes, [False], trigger_type, df_idx_range_hiscl, 
#                                                                       **opt, output_dir=trigger_immob_dir, annotation_kwargs=dict(is_immobilized=True))


# In[33]:


# %debug


# In[34]:


# # Combine and actually plot,
# # _combine_and_save(all_figs_box_fwd, all_figs_trig_fwd, all_figs_examples, to_show=True)
# _combine_and_save(all_figs_box_fwd, all_figs_trig_fwd, all_figs_examples=None, to_show=True)
# # Combine and actually plot
# # _combine_and_save(all_figs_box_rev, all_figs_trig_rev, all_figs_examples, to_show=True)
# _combine_and_save(all_figs_box_rev, all_figs_trig_rev, all_figs_examples=None, to_show=True)
# # Combine and actually plot
# _combine_and_save(all_figs_box_hiscl, all_figs_trig_hiscl, all_figs_examples=None, suffix='-hiscl', is_immobilized=True, to_show=True)
# # Combine and actually plot
# _combine_and_save(all_figs_box_immob_up, all_figs_trig_immob_up, all_figs_examples=None, suffix='-upshift', is_immobilized=True, to_show=True)
# # Combine and actually plot
# _combine_and_save(all_figs_box_immob_down, all_figs_trig_immob_down, all_figs_examples=None, suffix='-downshift', is_immobilized=True, to_show=True)


# ## Export for alternative modeling

# In[35]:


# Add gfp class, just for exporting
opt = dict(calculate_global=False, calculate_turns=False,
          trace_opt=dict(use_paper_options=True, channel_mode='dr_over_r_20'))

# FM Wild type
triggered_average_gcamp_plotter_GFP = PaperMultiDatasetTriggeredAverage(all_projects_gfp, calculate_self_collision=False, **opt,
                                                                        trigger_opt=dict(fixed_num_points_after_event=40), calculate_residual=True)


# In[36]:


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


# In[37]:


# df_combined['RIBL'].head()


# In[38]:


# df = triggered_average_gcamp_plotter.get_df_triggered_from_trigger_type_all_traces_as_df('raw_rev', melt_neuron='AVAL')

# # df_grouped = df.groupby(['dataset_name', 'trial_idx', 'before']).median().reset_index()

# px.box(df, x='dataset_name', y='value', color='before', #points='all',
#        category_orders={'before': ['True', 'False', True, False]})


# # Also export an excel with number of IDs and events

# In[39]:


from wbfm.utils.general.utils_hardcoded import get_triggered_average_dataframe_fname, get_all_trigger_suffixes, get_triggered_average_modeling_dir
from wbfm.utils.general.utils_hardcoded import neurons_with_confident_ids


# In[40]:


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


# In[41]:


# all_dfs.keys()


# In[42]:


from wbfm.utils.general.utils_hardcoded import excel_event_full_description
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


# In[43]:



fname = 'ids/events_per_id.xlsx'
df.to_excel(fname)


# # Alternative: collision-triggered averages (only showing BAG)

# In[ ]:


ttest_kwargs=dict(dynamic_window_center=True, dynamic_window_length=6)
opt = dict(show_title=True, #output_dir=output_dir, 
           min_lines=1, xlim=[-5, 12], show_individual_lines=False,
           round_y_ticks=False, show_y_ticks=True, show_y_label=True, show_y_label_only_export=True, i_figure=4,  
           use_plotly=True, to_show=False)

## Only BAG
neuron_list_col = neurons_with_confident_ids(combine_left_right=True)
plotter_classes = [triggered_average_gcamp_plotter_fm_mutant_collision, triggered_average_gcamp_plotter]
is_mutant_vec = [True, False]
trigger_type = 'residual_collision'
all_figs_trig_col_raw, df_boxplot_col_raw, df_p_values_col_raw, df_idx_range_col = plot_ttests_from_triggered_average_classes(neuron_list_col, plotter_classes, is_mutant_vec, 
                                                                               trigger_type, output_dir=None, to_show=False, ttest_kwargs=ttest_kwargs,
                                                                                                                             combine_left_right=True)
# all_figs_box_col = plot_triggered_averages_from_triggered_average_classes(neuron_list_col, plotter_classes, is_mutant_vec, trigger_type, df_idx_range_col, 
#                                                                       output_dir=None, annotation_kwargs=dict(is_residual=True), **opt)


# In[ ]:


df_boxplot_col_raw


# In[ ]:


# df_boxplot_col_raw.melt(id_vars=['neuron', 'is_mutant_str'])


# In[ ]:


df_p_values_col_raw


# In[ ]:


# _combine_and_save(all_figs_box_col, all_figs_trig_col, all_figs_examples=None, to_show=True, suffix='collision', width_factor=0.5)


# ## Legends for everything

# In[ ]:


# from wbfm.utils.general.utils_paper import export_legend_for_paper


# In[ ]:


# output_foldername = '/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing'
# fname = os.path.join(output_foldername, 'o2_supp_legend.png')
# export_legend_for_paper(fname, o2_supp=True)


# In[ ]:


# output_foldername = '/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing'
# fname = os.path.join(output_foldername, 'triple_plots_legend.png')
# export_legend_for_paper(fname, triple_plots=True)


# In[ ]:


# output_foldername = '/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/multiplexing'
# fname = os.path.join(output_foldername, 'o2_legend.png')
# export_legend_for_paper(fname, o2=True)


# # Alternate summary plot: all p values together

# In[344]:


from wbfm.utils.external.utils_plotly import get_nonoverlapping_text_positions


# In[78]:


ttest_kwargs=dict(dynamic_window_center=True, dynamic_window_length=6)
opt = dict(show_title=True, #output_dir=output_dir, 
           min_lines=1, xlim=[-5, 12], show_individual_lines=False,
           round_y_ticks=False, show_y_ticks=True, show_y_label=True, show_y_label_only_export=True, i_figure=4,  
           use_plotly=True, to_show=False)

## Only BAG
neuron_list_col = neurons_with_confident_ids(combine_left_right=True)
plotter_classes = [triggered_average_gcamp_plotter_fm_mutant, triggered_average_gcamp_plotter]
is_mutant_vec = [True, False]
trigger_type = 'raw_rev'

all_figs_trig_col_raw, df_boxplot_col_raw, df_p_values_col_raw, df_idx_range_col = plot_ttests_from_triggered_average_classes(neuron_list_col, plotter_classes, is_mutant_vec, 
                                                                                   trigger_type, output_dir=None, to_show=False, ttest_kwargs=ttest_kwargs,
                                                                                                                             combine_left_right=True)


# In[79]:


from wbfm.utils.general.utils_paper import plot_foldchange_boxes


# In[80]:


df_boxplot_fwd_raw.tail()


# In[81]:



idx = df_boxplot_fwd_raw['neuron'] == 'RIB'
# idx2 = df_boxplot_fwd_raw['Data Type'] == 'Fwd-Mutant'
idx2 = df_boxplot_fwd_raw['is_mutant'] == True
df_boxplot_fwd_raw.loc[idx*idx2, :]


# In[133]:


# group_sizes = df_boxplot_fwd_raw.groupby(['neuron', 'is_mutant_str', 'before']).size()
# group_sizes.loc['RIB', :]


# In[134]:


# df_boxplot_rev_raw.groupby(['neuron', 'is_mutant_str', 'before_str'])['mean'].mean()['RIB']


# In[135]:


def _get_effect_size(_df):
    return _df.groupby(['neuron', 'is_mutant_str']).apply(
        lambda group: pd.Series({
            'fc': group[group['before'] == False]['mean'].median() - 
                  group[group['before'] == True]['mean'].median()
        })
    )


# In[293]:


# Combine
all_dfs = []

def _select_trigger(df_p, trigger):
    """Filter a p-value table to one trigger; tolerate stale pre-fix tables."""
    if isinstance(df_p.index, pd.MultiIndex) and 'trigger_type' in df_p.index.names:
        return df_p.xs(trigger, level='trigger_type')
    # Stale table from before library fix (no trigger_type in index):
    # assume every row belongs to this single-trigger calculation
    print(f'WARNING: {trigger} p-value table has no trigger_type level; using all rows')
    return df_p

# df_p_values_fwd/rev are the same object (df_p_values_fm) after cell 42;
# filter each to its own trigger so Forward/Backward rows are not mixed.
_df = _select_trigger(df_p_values_fwd, 'raw_fwd').join(
    _get_effect_size(df_boxplot_fwd_raw)
).reset_index()
_df['Trigger Type'] = 'Forward'
all_dfs.append(_df.copy())

_df = _select_trigger(df_p_values_rev, 'raw_rev').join(
    _get_effect_size(df_boxplot_rev_raw)
).reset_index()
_df['Trigger Type'] = 'Backward'
all_dfs.append(_df.copy())

# _df = _select_trigger(df_p_values_col_raw, 'residual_collision').join(
#     _get_effect_size(df_boxplot_col_raw)
# ).reset_index()
# _df['Trigger Type'] = 'Self-Collision'
# all_dfs.append(_df.copy())

df = pd.concat(all_dfs).reset_index(drop=True)

# Correct (across all triggers in this summary table)
_p = df['p_value'].dropna()
df.loc[_p.index, 'p_value_corrected'] = multipletests(_p.values.squeeze(), method='fdr_bh', alpha=0.05)[1]

# Finalize
df['p_value_adj'] = df['p_value_corrected']
# Only non-zero fc can be logged; do NOT replace(0) on p-values
df['log2_fc'] = np.where(df['fc'] != 0,
                         np.log2(np.abs(df['fc'])) * -np.sign(df['fc']),
                         np.nan)

df = df.dropna(subset=['p_value_adj', 'fc', 'count', 'neuron', 'is_mutant_str', 'Trigger Type'])

df.head()


# In[294]:


# idx = df['neuron'] == 'AVA'
# df[idx]


# In[315]:


neuron_order = list(df.groupby(['is_mutant_str', 'Trigger Type', 'neuron'])['fc'].mean().sort_values().loc['Wild Type', :].loc['Forward', :].index)

# _df = df

# neuron_order = list(df_p_values_fwd.loc[(slice(None), 'Wild Type'), 'p_value_corrected'].sort_values().reset_index()['neuron'].unique())
to_remove = []
# to_remove = ['IL1', 'BAG', 'AQR', 'AUA', 'IL2', 'URX']
neuron_order = [n for n in neuron_order if n not in to_remove]
idx = df['neuron'].isin(neuron_order)
_df = df.loc[idx, :]

# Remove neurons with too few trials
min_num_trials = 4
_df = _df[df['count'] >= min_num_trials]

groups = {'all_neurons': neuron_order}
# groups = ['all_fwd', 'all_rev']


# In[316]:


# idx = df_mvw['neuron'].isin(neuron_order)


# In[317]:


# Add explicit comparison columns between WT and mutant

from wbfm.utils.visualization.paper_multidataset_triggered_average import calc_mutant_vs_wt_test

df_mvw = pd.concat([
    calc_mutant_vs_wt_test(neuron_list_fwd, triggered_average_gcamp_plotter,
                           triggered_average_gcamp_plotter_fm_mutant, 'raw_fwd', min_num_datasets=4),
    calc_mutant_vs_wt_test(neuron_list_rev, triggered_average_gcamp_plotter,
                           triggered_average_gcamp_plotter_fm_mutant, 'raw_rev', min_num_datasets=4),
]).reset_index()
df_mvw['Trigger Type'] = df_mvw['trigger_type'].map({'raw_fwd': 'Forward', 'raw_rev': 'Backward'})
df_mvw['p_value_adj'] = df_mvw['p_value_corrected']   # already corrected within this family
df_mvw['log2_fc'] = df_mvw['fc']                      # only the sign is used by plot_foldchange_boxes

idx = df_mvw['neuron'].isin(neuron_order)

_df = pd.concat([_df, df_mvw.loc[idx, :].reset_index(drop=True)]).reset_index(drop=True)   # after the existing correction, don't re-correct


# In[318]:


# df_mvw.loc[idx, :]['neuron'].unique()


# In[319]:


# df_mvw


# In[320]:


# df_mvw.loc[idx, :]


# In[321]:


ax, fig, ordered_neurons = plot_foldchange_boxes(
    df=_df,
    behavior_col='Trigger Type',
    groups=groups,
    subtitle_behavior_col='is_mutant_str',
    rows_col='neuron',
    value_col='p_value_corrected',
    figsize=(8, 6),
    row_vspace=0.5,
    behavior_hspace=3.0,
    cmap='PiYG',
    add_text=False,
    use_pval_log10=True,
    # vmax=10,
    # vmin=0,
    neuron_order=neuron_order,
    DEBUG=False
)


apply_figure_settings(fig=fig, width_factor=1.0, height_factor=0.8, plotly_not_matplotlib=False)


# In[322]:


# Same but with categories instead of log p value, which reflects dataset number
from matplotlib.colors import ListedColormap

def _sig_category(p):
    if pd.isna(p) or p >= 0.05:
        return np.nan  # n.s. -> grey (NaN is grey in fold-change mode)
    return 3 if p < 0.001 else (2 if p < 0.01 else 1)

_df = _df.copy()
_df['sig_category'] = _df['p_value_corrected'].map(_sig_category) * np.sign(_df['fc'])
cmap_cat = ListedColormap(plt.get_cmap('PiYG')(np.linspace(0, 1, 7)))  # one colour per level -3..3

ax, fig, ordered_neurons = plot_foldchange_boxes(
    df=_df,
    behavior_col='Trigger Type',
    groups=groups,
    subtitle_behavior_col='is_mutant_str',
    rows_col='neuron',
    value_col='sig_category',
    use_pval_log10=False,          # fold-change mode: colours value_col directly
    cmap=cmap_cat, vmin=-3.5, vmax=3.5,
    figsize=(8, 6), row_vspace=0.5, behavior_hspace=3.0,
    neuron_order=neuron_order,
)

cbar_ax = fig.axes[-1]
cbar_ax.set_ylim(-3.5, 3.5)
cbar_ax.set_yticks([-3, -2, -1, 1, 2, 3])
cbar_ax.set_yticklabels(['*** down', '** down', '* down', '* up', '** up', '*** up'])
cbar_ax.set_ylabel('Adj. p-value (grey: n.s.)')


# In[323]:


_df


# In[324]:


idx = _df['neuron'] == 'URX'
_df.loc[idx, :]


# In[325]:


df_compare_p_value = _df.pivot_table(index=['neuron', 'Trigger Type'], columns='is_mutant_str', values='p_value_corrected').reset_index()
df_compare_p_value.head()


# In[326]:


df_compare = _df.pivot_table(index=['neuron', 'Trigger Type'], columns='is_mutant_str', values='fc').reset_index()
df_compare['size'] = 0.1

df_compare['name_sig'] = df_compare['neuron']

p_value = np.log10(_df['p_value_corrected'])

# idx = _df['is_mutant_str'] == 'Mutant vs WT'
# p_value = _df['p_value_corrected']
# idx = np.logical_and(idx, p_value > 0.05)
# non_sig_neurons = _df.loc[idx, 'neuron']
# idx = df_compare['neuron'].isin(non_sig_neurons)
idx = df_compare_p_value['Mutant vs WT'] > 0.05
df_compare.loc[idx, 'name_sig'] = ''
df_compare['Mutant vs WT - p_value'] = p_value


# idx = df_compare['Mutant vs WT'] > 0.5
# df_compare.loc[idx, 'name_sig'] = ''

df_compare.head()


# In[463]:


# Categorical significance of Mutant vs WT, colored by sign (same scheme as the fold-change box plot above)
from matplotlib.colors import to_hex

sig_labels = {-3: '*** down', -2: '** down', -1: '* down', 1: '* up', 2: '** up', 3: '*** up'}
sig_order = ['n.s.'] + list(sig_labels.values())  # n.s. first, so its trace is drawn in the background
_piyg = plt.get_cmap('PiYG')(np.linspace(0, 1, 7))  # one colour per level -3..3
sig_color_map = {label: to_hex(_piyg[level + 3]) for level, label in sig_labels.items()}
sig_color_map['n.s.'] = 'lightgray'

# Merge on keys rather than relying on row order between the two pivot tables
_p_mvw = df_compare_p_value[['neuron', 'Trigger Type', 'Mutant vs WT']].rename(columns={'Mutant vs WT': 'Mutant vs WT - p_value_adj'})
df_compare = df_compare.drop(columns=['Mutant vs WT - p_value_adj'], errors='ignore').merge(
    _p_mvw, on=['neuron', 'Trigger Type'], how='left')

_level = df_compare['Mutant vs WT - p_value_adj'].map(_sig_category) * np.sign(df_compare['Mutant vs WT'])
df_compare['Significance'] = _level.map(sig_labels).fillna('n.s.')  # NaN p-value or zero fold change -> n.s.

def add_full_sig_legend(fig):
    """Show every significance category in the legend, even ones absent from this subplot's data."""
    import plotly.graph_objects as go
    present = {t.name for t in fig.data}
    for label in sig_order:
        if label not in present:
            # Invisible placeholder: no data, so it only appears in the legend
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name=label, legendgroup=label,
                                     marker=dict(color=sig_color_map[label]), hoverinfo='skip', showlegend=True))
    # Legend in sig_order (independent of drawing order); equal-size legend markers
    fig.for_each_trace(lambda t: t.update(legendrank=sig_order.index(t.name)) if t.name in sig_order else None)
    fig.update_layout(legend_itemsizing='constant', legend_title_text='Significance')
    return fig


def add_unity_line(fig):
    """Dotted y = x line spanning the current axis ranges, drawn below the data."""
    xr, yr = fig.layout.xaxis.range, fig.layout.yaxis.range
    lo, hi = min(xr[0], yr[0]), max(xr[1], yr[1])
    fig.add_shape(type='line', x0=lo, y0=lo, x1=hi, y1=hi, xref='x', yref='y',
                  line=dict(color='black', width=1, dash='dot'), layer='below')
    return fig


wt_vs_mutant_scatter_dir = '/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/revisions/fig_supp_o2/wt_vs_mutant_scatter'
os.makedirs(wt_vs_mutant_scatter_dir, exist_ok=True)

def save_wt_vs_mutant_scatter(fig, trigger_type):
    fname = os.path.join(wt_vs_mutant_scatter_dir, f'{trigger_type}-wt_vs_mutant_fold_change_scatter.png')
    fig.write_image(fname, scale=3)
    fig.write_image(fname.replace('.png', '.svg'))
    print(f"Saved to {fname}")

df_compare['Significance'].value_counts()


# In[464]:


print(df_compare.loc[:, 'name_sig'].unique())
# print(df_compare.loc[:, 'name_sig'], df_compare.loc[:, 'Wild Type'], df_compare.loc[:, 'gcy-31;-35;-9']) 


# In[465]:


# df_compare


# In[475]:


_df2 = df_compare[df_compare['Trigger Type'] == 'Forward']

fig = px.scatter(_df2, x='Wild Type', y='gcy-31;-35;-9', color='Significance', color_discrete_map=sig_color_map,
                 category_orders={'Significance': sig_order},
                 title='Forward Crawling Triggered',
                # text='name_sig', 
                 size='size', size_max=10, #facet_col='Trigger Type'
                )
add_full_sig_legend(fig)
apply_figure_settings(fig=fig, width_factor=0.5, height_factor=0.25, plotly_not_matplotlib=True)
# apply_figure_settings sets the top margin to 0, which clips the title; same margin in both panels keeps plot heights equal
fig.update_layout(margin_t=25, title_x=0.5, title_xanchor='center')
fig.update_yaxes(range=[-0.25, 1], title='gcy-31;-35;-9 (fold change)')
fig.update_xaxes(range=[0, 0.25], title='Wild Type (fold change)')
fig.update_traces(
    cliponaxis=False,
    marker=dict(
        line=dict(
            width=1,      # Thickness of the border in pixels
            color='black'   # Color of the border (can be hex, rgb, or color name)
        )
    )
)

get_nonoverlapping_text_positions(_df2['Wild Type'], _df2['gcy-31;-35;-9'], _df2['name_sig'], fig, #weight=0.1, k=0.25,
                                weight=1,
                                k=0.1,
                                  add_nodes_with_no_text=False,
                                  DEBUG=False,
                                 x_range=[0.02, 0.2], 
                                  # y_range=[-300, 1000], 
                                  # size=12
                                 )
add_unity_line(fig)
save_wt_vs_mutant_scatter(fig, 'Forward')
fig.show()


# In[476]:


_df2 = df_compare[df_compare['Trigger Type'] == 'Backward']

fig = px.scatter(_df2, x='Wild Type', y='gcy-31;-35;-9', color='Significance', color_discrete_map=sig_color_map,
                 category_orders={'Significance': sig_order},
                 title='Backward Crawling Triggered',
                # text='name_sig', 
                 size='size', size_max=10, #facet_col='Trigger Type'
                )
apply_figure_settings(fig=fig, width_factor=0.35, height_factor=0.25, plotly_not_matplotlib=True)
# apply_figure_settings sets the top margin to 0, which clips the title; same margin in both panels keeps plot heights equal
fig.update_layout(margin_t=25, title_x=0.5, title_xanchor='center')
fig.update_yaxes(range=[-0.25, 1], title='gcy-31;-35;-9 (fold change)')
fig.update_xaxes(range=[0, 0.75], title='Wild Type (fold change)')
fig.update_traces(
    showlegend=False,
    cliponaxis=False,
    marker=dict(
        line=dict(
            width=1,      # Thickness of the border in pixels
            color='black'   # Color of the border (can be hex, rgb, or color name)
        )
    )
)

get_nonoverlapping_text_positions(_df2['Wild Type'], _df2['gcy-31;-35;-9'], _df2['name_sig'], fig, #weight=0.1, k=0.25,
                                weight=1,
                                k=0.1,
                                  add_nodes_with_no_text=False,
                                  DEBUG=False,
                                 x_range=[0.04, 0.75], 
                                  # y_range=[-300, 1000], 
                                  # size=12
                                 )
add_unity_line(fig)
save_wt_vs_mutant_scatter(fig, 'Backward')
fig.show()


# In[ ]:





# In[331]:



# _df['sig_category'] = _df['p_value_corrected'].map(_sig_category) * np.sign(_df['fc'])
# cmap_cat = ListedColormap(plt.get_cmap('PiYG')(np.linspace(0, 1, 7)))  # one colour per level -3..3

# cbar_ax = fig.axes[-1]
# cbar_ax.set_ylim(-3.5, 3.5)
# cbar_ax.set_yticks([-3, -2, -1, 1, 2, 3])
# cbar_ax.set_yticklabels(['*** down', '** down', '* down', '* up', '** up', '*** up'])
# cbar_ax.set_ylabel('Adj. p-value (grey: n.s.)')


# In[336]:


fig = px.scatter(df_compare, x='Wild Type', y='gcy-31;-35;-9', color='Mutant vs WT - p_value', 
                 # color_continuous_scale='PiYG',
                text='name_sig', size='size', size_max=10, facet_col='Trigger Type')
apply_figure_settings(fig=fig, width_factor=1.0, height_factor=0.25, plotly_not_matplotlib=True)
# fig.update_traces(textposition='top center', textfont=dict(color='black'), cliponaxis=False)
fig.update_traces(
    mode='markers+text',         # <-- Forces text to display permanently on the plot
    textposition='top center', 
    texttemplate='%{text}',      # <-- Forces evaluation of the text column across facets
    textfont=dict(color='black'), 
    cliponaxis=False
)
fig.show()


# In[ ]:





# # Behavior differences

# In[ ]:


# Per-worm behavior statistics: reversal frequency, speed, etc. (automatic behavior annotation)
# WT = all_projects_O2_fm (gcamp + hannah_O2_fm); mutant = all_projects_O2_fm_mutant
# Calculating takes ~20s per project, so the result is cached; delete the csv to recalculate
from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes

beh_output_folder = os.path.join('fig_supp_o2', 'behavior')
os.makedirs(beh_output_folder, exist_ok=True)
fname_beh_cache = os.path.join(beh_output_folder, 'per_worm_behavior_metrics.csv')


def calc_per_worm_behavior_metrics(projects: dict, group: str) -> list:
    rows = []
    for name, p in tqdm(projects.items(), desc=group):
        try:
            w = p.worm_posture_class
            vps = p.physical_unit_conversion.volumes_per_second
            beh = w.beh_annotation(fluorescence_fps=True, reset_index=True)
            n = len(beh)
            dur_min = n / vps / 60
            is_rev = beh.apply(lambda b: BehaviorCodes.REV in b).values.astype(int)
            starts = np.where(np.diff(np.r_[0, is_rev]) == 1)[0]
            ends = np.where(np.diff(np.r_[is_rev, 0]) == -1)[0]
            rev_durs = (ends - starts + 1) / vps
            is_turn = beh.apply(lambda b: (BehaviorCodes.VENTRAL_TURN in b) or (BehaviorCodes.DORSAL_TURN in b)).values
            is_pause = beh.apply(lambda b: BehaviorCodes.PAUSE in b).values
            aspd = np.abs(w.worm_speed(fluorescence_fps=True, signed=True).values[:n])
            fwd = ~is_rev.astype(bool)
            rows.append(dict(group=group, name=name, vps=vps, dur_min=dur_min, n_rev=len(starts),
                             rev_per_min=len(starts) / dur_min,
                             rev_dur_s=np.nanmedian(rev_durs) if len(rev_durs) else np.nan,
                             frac_rev=is_rev.mean(), frac_turn=is_turn.mean(), frac_pause=is_pause.mean(),
                             speed_all=np.nanmean(aspd), speed_fwd=np.nanmean(aspd[fwd]),
                             speed_rev=np.nanmean(aspd[~fwd]), speed_fwd_p90=np.nanpercentile(aspd[fwd], 90)))
        except Exception as e:
            print(f'FAILED {group} {name}: {type(e).__name__}: {e}')
    return rows


if os.path.exists(fname_beh_cache):
    df_beh = pd.read_csv(fname_beh_cache)
else:
    df_beh = pd.DataFrame(calc_per_worm_behavior_metrics(all_projects_O2_fm, 'wt') +
                          calc_per_worm_behavior_metrics(all_projects_O2_fm_mutant, 'mutant'))
    df_beh.to_csv(fname_beh_cache, index=False)

wt_name, mutant_name = 'Freely Moving (GCaMP)', 'gcy-31;-35;-9'
df_beh['Genotype'] = np.where(df_beh['group'] == 'mutant', mutant_name, wt_name)
beh_category_orders = {'Genotype': [wt_name, mutant_name]}
print(df_beh['Genotype'].value_counts())
df_beh.groupby('Genotype')[['rev_per_min', 'rev_dur_s', 'frac_rev', 'frac_turn', 'frac_pause',
                            'speed_all', 'speed_fwd', 'speed_rev']].median().T


# In[ ]:


# Boxplots, with Welch t-test (FDR corrected within each figure) via add_p_value_annotation
from wbfm.utils.general.utils_paper import apply_figure_settings, plotly_paper_color_discrete_map
from wbfm.utils.visualization.utils_plot_traces import add_p_value_annotation


def plot_genotype_boxplot(df_long, y_title, fname_base, width_factor=0.3):
    fig = px.box(df_long, x='Metric', y='value', color='Genotype', points='all',
                 color_discrete_map=plotly_paper_color_discrete_map(), category_orders=beh_category_orders)
    add_p_value_annotation(fig, x_label='all', show_only_stars=True, height_mode='top_of_data')
    apply_figure_settings(fig, width_factor=width_factor, height_factor=0.2, plotly_not_matplotlib=True)
    fig.update_layout(legend=dict(title='', yanchor='bottom', y=1.05, xanchor='left', x=0, orientation='h'))
    fig.update_xaxes(title='')
    fig.update_yaxes(title=y_title, rangemode='tozero')
    fig.show()

    fname = os.path.join(beh_output_folder, f'{fname_base}.png')
    fig.write_image(fname, scale=3)
    fig.write_image(fname.replace('.png', '.svg'))
    return fig


# Reversal frequency
df_rev = df_beh.melt(id_vars=['Genotype', 'name'], value_vars=['rev_per_min'], var_name='Metric')
df_rev['Metric'] = 'Reversals'
fig_rev = plot_genotype_boxplot(df_rev, 'Reversals / min', 'boxplot_reversal_frequency')

# Speed, split by forward and reversal states
df_speed = df_beh.melt(id_vars=['Genotype', 'name'], value_vars=['speed_fwd', 'speed_rev'], var_name='Metric')
df_speed['Metric'] = df_speed['Metric'].map({'speed_fwd': 'Forward', 'speed_rev': 'Reversal'})
fig_speed = plot_genotype_boxplot(df_speed, 'Speed (mm/s)', 'boxplot_speed_fwd_rev')


# # Sanity check: do the behaviors exist?

# In[ ]:


for p in all_projects_O2_fm.values():
    print(p)
    print(p.worm_posture_class)
    break
    


# In[ ]:


p = all_projects_O2_fm['2023-11-21_14-46_wt_worm4-2023-11-21']


# In[ ]:


del p.worm_posture_class


# In[ ]:


p


# In[ ]:


# p.worm_posture_class.hilbert_phase(fluorescence_fps=True)


# # Debug why AVA triggered average is so bad

# In[ ]:


out = triggered_average_gcamp_plotter.plot_triggered_average_single_neuron('AVAL', 'raw_rev', show_individual_lines=False)


# In[ ]:


out = triggered_average_gcamp_plotter_fm_mutant.plot_triggered_average_single_neuron('AVAL', 'raw_rev', show_individual_lines=False)


# In[ ]:


triggered_average_gcamp_plotter_fm_mutant.print_statistics()


# In[ ]:


# out = triggered_average_gcamp_plotter.plot_events_over_trace('AVAL', 'raw_rev', ignore_errors=False)


# In[ ]:


out = triggered_average_gcamp_plotter_fm_mutant.plot_events_over_trace('AVAL', 'raw_rev', ignore_errors=False)


# # Debug why other neuron mutant triggered average is so bad

# In[337]:


neuron = 'RIV'
trigger_type = 'raw_rev'

out = triggered_average_gcamp_plotter.plot_triggered_average_single_neuron(neuron, trigger_type, show_individual_lines=False)
out = triggered_average_gcamp_plotter_fm_mutant.plot_triggered_average_single_neuron(neuron, trigger_type, show_individual_lines=False)


# # Debug duplicate rows

# In[ ]:


traces = triggered_average_gcamp_plotter_fm_mutant.get_traces_single_neuron('RIB', 'raw_fwd')
traces


# In[ ]:


# triggered_average_gcamp_plotter_fm_mutant


# In[ ]:


pd.Series(neuron_list_fwd).value_counts()


# In[ ]:


pd.Series(list(set(neuron_list))).value_counts().max()


# # Debug why the behaviors aren't found

# In[ ]:


# name = '2023-11-21_14-46_wt_worm4-2023-11-21'
p = all_projects_O2_fm_mutant['2023-11-21_15-23_gcy9_gcy31_gcy35_worm1-2023-11-21']


# In[ ]:


# del p.worm_posture_class


# In[ ]:


p.worm_posture_class.beh_annotation(fluorescence_fps=True, include_collision=True).shape


# In[ ]:


df_traces = p.calc_default_traces(use_paper_options=True)


# In[ ]:


fig = px.line(df_traces['AVAL'])
p.shade_axis_using_behavior(plotly_fig=fig)
fig.show()


# In[ ]:


p.physical_unit_conversion


# In[ ]:


p2 = all_projects_O2_fm['2023-11-21_14-46_wt_worm4-2023-11-21']


# In[ ]:


df_traces2 = p2.calc_default_traces(use_paper_options=True)


# In[ ]:


fig = px.line(df_traces2['AVAL'])
p2.shade_axis_using_behavior(plotly_fig=fig)
fig.show()

