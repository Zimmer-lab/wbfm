#!/usr/bin/env python
# coding: utf-8




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
from wbfm.utils.general.utils_hardcoded import load_paper_datasets


from sklearn.decomposition import PCA
from wbfm.utils.visualization.plot_traces import make_grid_plot_from_dataframe
import seaborn as sns
import plotly.express as px


# Load multiple datasets
all_projects_good = load_paper_datasets('gcamp_good', initialization_kwargs=dict(use_physical_time=True))


all_projects_gcamp = load_paper_datasets(['gcamp', 'hannah_O2_fm'], initialization_kwargs=dict(use_physical_time=True))


all_projects_immob = load_paper_datasets('immob', initialization_kwargs=dict(use_physical_time=True))


all_projects_immob['2022-12-12_15-59_ZIM2165_immob_worm1-2022-12-12'].use_physical_time


# # Use a function to plot traces across multiple projects (with proper behavior shading)

from wbfm.utils.general.utils_behavior_annotation import plot_stacked_figure_with_behavior_shading_using_plotly


# ## Plot

fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_gcamp, 
                                                             ['AVAL', 'AVAR', 'AVBL', 'AVBR', 'RIML', 'RIMR'], 
                                                             to_save=True, trace_kwargs=dict(use_paper_options=True), combine_neuron_pairs=False)
fig.show()


# fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_gcamp, 
#                                                              ['SMDDR', 'SMDDL', 'SMDVL', 'SMDVR', 'RIVL', 'RIVR', 'AVAL', 'VG_post_turning_L', 'VG_post_turning_R', 'VB02'], 
#                                                              to_save=True, trace_kwargs=dict(use_paper_options=True), combine_neuron_pairs=False)
# fig.show()


# ## Same but for all projects

# fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_gcamp, ['AVAL', 'RIVL'], to_save=False, trace_kwargs=dict(use_paper_options=True))
# fig.show()


# ## Plot: immobilized

fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_immob, ['AVAL', 'AVAR'], 
                                                             to_save=True, trace_kwargs=dict(use_paper_options=True), fname_suffix='immob', combine_neuron_pairs=False)
fig.show()


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_immob, ['AVAL', 'AVBL', 'AVBR', 'RIS'], to_save=False, trace_kwargs=dict(use_paper_options=True))
fig.show()


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_immob, ['AVAL', 'AVBL', 'RIVL', 'RID'], to_save=False, trace_kwargs=dict(use_paper_options=True))
fig.show()


fig = plot_stacked_figure_with_behavior_shading_using_plotly(all_projects_immob, ['AVAL', 'AVBL', 'RIVL', 'RMED', 'RMEV'], to_save=False, trace_kwargs=dict(use_paper_options=True))
fig.show()


# # DEBUG: why is the immobilized frame rate slightly wrong?

all_projects_immob['ZIM2165_immob_adj_set_2_worm3-2022-11-30']




