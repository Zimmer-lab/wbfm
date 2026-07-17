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


from wbfm.utils.general.hardcoded_paths import load_all_data_as_dataframe


# In[4]:


Xy = load_all_data_as_dataframe()


# In[5]:


Xy.head()


# In[22]:


Xy['dataset_type'].unique()


# In[35]:


idx = Xy['VB02'].dropna().index
Xy.loc[idx, 'dataset_type'].unique()


# In[124]:


# fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-12-12_immob/2022-12-13_15-16_ZIM2165_immob_worm9-2022-12-13/project_config.yaml"
fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-12-12_immob/2022-12-13_15-16_ZIM2165_immob_worm9-2022-12-13/project_config.yaml"
project_data_immob = ProjectData.load_final_project_data_from_config(fname)


# In[78]:


fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-11-27_spacer_7b_2per_agar/ZIM2165_Gcamp7b_worm3-2022_11_28/project_config.yaml"
# Manually corrected version
# fname = "/scratch/neurobiology/zimmer/Charles/dlc_stacks/manually_annotated/paper_data/ZIM2165_Gcamp7b_worm1-2022_11_28/project_config.yaml"
project_data_gcamp = ProjectData.load_final_project_data_from_config(fname)


# # FFT of VB02 in immob and freely moving

# In[134]:


from wbfm.utils.projects.finished_project_data import plot_frequencies_for_fm_and_immob_projects
from wbfm.utils.general.utils_paper import plotly_paper_color_discrete_map, apply_figure_settings, data_type_name_mapping
from wbfm.utils.external.utils_plotly import plotly_plot_mean_and_shading
from wbfm.utils.external.utils_pandas import combine_columns_with_suffix
from wbfm.utils.visualization.filtering_traces import fill_nan_in_dataframe


# In[135]:


output_folder = 'multiplexing'


# In[136]:


from scipy import signal
fs = 3.5


# In[137]:


def package_df_for_fft(neuron_name):

    def func(y): 
        y = fill_nan_in_dataframe(y, do_filtering=False)
        output = signal.welch(y, fs)
        pxx = np.array(output[1])
        # pxx = (pxx - np.mean(pxx)) / np.std(pxx)
        pxx = (pxx) / np.std(pxx)
        return pd.Series({'freq': output[0], 'pxx': pxx})
    
    if neuron_name not in Xy:
        _Xy = combine_columns_with_suffix(Xy)
    else:
        _Xy = Xy
    pxx_vb02 = _Xy.groupby(['dataset_name', 'dataset_type'])[neuron_name].apply(func).to_frame()
    
    pxx_vb02_explode = pxx_vb02.unstack(level=2).apply(pd.Series.explode).droplevel(level=0, axis=1).reset_index(level=1)

    _df = pxx_vb02_explode.copy()
    _df = _df[_df['dataset_type'] != 'gfp']
    _df['dataset_type'] = _df['dataset_type'].map(data_type_name_mapping())
    
    _df = _df.dropna()
    print(f"Found the following dataset types: {_df['dataset_type'].unique()}")
    
    return _df.dropna()


# In[59]:


# neuron_name = 'VB02'

# def func(y): 
#     output = signal.welch(y, fs)
#     pxx = np.array(output[1])
#     # pxx = (pxx - np.mean(pxx)) / np.std(pxx)
#     pxx = (pxx) / np.std(pxx)
#     return pd.Series({'freq': output[0], 'pxx': pxx})

# pxx_vb02 = Xy.groupby(['dataset_name', 'dataset_type'])[neuron_name].apply(func).to_frame()

# pxx_vb02_explode = pxx_vb02.unstack(level=2).apply(pd.Series.explode).droplevel(level=0, axis=1).reset_index(level=1)

# _df = pxx_vb02_explode.copy()
# _df = _df[_df['dataset_type'] != 'gfp']
# _df['dataset_type'] = _df['dataset_type'].map(data_type_name_mapping())
# print(f"Found the following dataset types: {_df['dataset_type'].unique()}")

# _df = _df.dropna()
# print(f"Found the following dataset types: {_df['dataset_type'].unique()}")


# In[60]:


# y = Xy[np.logical_and(Xy['dataset_name']=='ZIM2165_Gcamp7b_worm1-2022-11-30', Xy['dataset_type']=='gcamp')]
# signal.welch(y['VB02'], 3.5)
# y['VB02'].count()


# In[61]:


# Xy['dataset_name'].unique()


# In[63]:


# px.line(Xy, y=neuron_name, facet_row='dataset_type', color='dataset_name', height=1000)


# In[41]:


# px.line(df[df['dataset_type'] == 'Freely Moving (GCaMP)'].reset_index(), x='freq', y='pxx', color='dataset_name',)


# In[183]:


# Do not plot gfp

neurons_to_plot = ['VB02', 'DB01', 'DD01', 'AVB', 
                   # 'VB03', 'VB01', 'DB02', 'VA01', 'VA02', 'DA01',
                   # 'SIAVL', 'SIAVR', 'DD01', 'RMDDR', 'RMDDL', 'RMDVL', 'RMDVR', 'SMDDL', 'SMDDR', 
                   # 'RMEV', 'RMED', 'RMER', 'RMEL', 'AVFL', 'AVFR', 'AVAL', 'AVAR'
                  ]

for n in neurons_to_plot:

    df = package_df_for_fft(n)
    print(n)

    fig = plotly_plot_mean_and_shading(df, x='freq', y='pxx', color='dataset_type', line_name='immob',
                                      cmap = plotly_paper_color_discrete_map(), title=n)
    fig.update_xaxes(range=[0, 0.4], title='Frequency (Hz)')
    fig.update_yaxes(title='Normalized<br>power')
    fig.update_layout(
        showlegend=False,#(n=='VB02'),
        legend=dict(
          yanchor="top",
          y=1.0,
          xanchor="left",
          x=0.1
        ),
    )

    apply_figure_settings(fig, width_factor=0.25, height_factor=0.14)

    fig.show()

    output_foldername = 'fft_examples'
    fname = os.path.join(output_foldername, f'fft_neuron_{n}.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)
    
    # break


# In[21]:


df['dataset_type'].unique()


# In[ ]:


get_ipython().run_line_magic('debug', '')


# In[ ]:


# # Do not plot gfp

# df = package_df_for_fft('DB01')

# fig = plotly_plot_mean_and_shading(df, x='freq', y='pxx', color='dataset_type',
#                                   cmap = plotly_paper_color_discrete_map())
# fig.update_xaxes(range=[0, 0.5], title='Frequency (Hz)')
# fig.update_yaxes(title='Normalized power')
# fig.update_layout(showlegend=False)

# apply_figure_settings(fig, width_factor=0.4, height_factor=0.2)

# fig.show()

# # output_foldername = 'intro/dimensionality'
# # fname = os.path.join(output_foldername, 'variance_explained_by_pc1_cumulative.png')
# # fig.write_image(fname, scale=3)
# # fname = Path(fname).with_suffix('.svg')
# # fig.write_image(fname)


# In[ ]:





# # Example traces: immob

# In[126]:


from wbfm.utils.visualization.paper_multidataset_triggered_average import PaperExampleTracePlotter


# In[170]:


immob_plotter = PaperExampleTracePlotter(project_data_immob, xlim=[200, 350], ylim=[-0.3, 0.39])


# In[171]:


# project_data_immob.worm_posture_class.beh_annotation(fluorescence_fps=True, use_manual_annotation=True)


# In[172]:


# immob_plotter.project.physical_unit_conversion.volumes_per_second


# In[173]:


# project_data_immob.data_cacher.clear_disk_cache(delete_invalid_indices=False, delete_traces=True)
# project_data_immob.calc_paper_traces()
# project_data_immob.calc_paper_traces_global()
# project_data_immob.calc_paper_traces_residual()


# In[179]:


neurons = ['VB02', 'DB01', 'DD01', 'AVB']

output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/supplement/fft_examples/immob'

for n in neurons:

    # immob_plotter.plot_triple_traces(n, output_foldername=output_foldername)
    immob_plotter.plot_single_trace(n, color_type='immob', output_foldername=output_foldername,
                                   trace_options=dict(trace_type='raw'), height_factor=0.12, width_factor=0.3, 
                                    shading_kwargs=dict(use_manual_annotation=True), round_y_ticks=True, round_yticks_kwargs=dict(max_ticks=2))


# ## Example traces: WBFM

# In[87]:


from wbfm.utils.visualization.paper_multidataset_triggered_average import PaperExampleTracePlotter


# In[ ]:





# In[168]:


# wbfm_plotter = PaperExampleTracePlotter(project_data_gcamp, xlim=[0, 120], ylim=[-0.33, 0.24])
wbfm_plotter = PaperExampleTracePlotter(project_data_gcamp, xlim=[50, 200], ylim=[-0.19, 0.3])


# In[169]:


neurons = ['VB02', 'DB01', 'DD01', 'AVB']

output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/supplement/fft_examples/wbfm'

for n in neurons:
    # wbfm_plotter.plot_triple_traces('VB02', title=True, legend=True, output_foldername=output_foldername, width_factor=0.3)
    wbfm_plotter.plot_single_trace(n, title=True, legend=False, output_foldername=output_foldername,
                                   trace_options=dict(trace_type='raw'), height_factor=0.15, width_factor=0.3, round_y_ticks=True, round_yticks_kwargs=dict(max_ticks=2))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Legend for everything

# In[184]:


from wbfm.utils.general.utils_paper import export_legend_for_paper


# In[188]:


output_foldername='/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/supplement/fft_examples'
fname = os.path.join(output_foldername, 'fft_legend.png')
export_legend_for_paper(fname, bayesian_supp=True)


# # Alternative

# ## Sanity check: fft of small vb02 time series

# In[ ]:


Xy['dataset_name'].unique()


# In[ ]:


y = Xy[Xy['dataset_name']=='ZIM2165_Gcamp7b_worm1-2022_11_28']['VB02'].reset_index(drop=True)
px.line(y)


# In[ ]:


y_subset = y.iloc[900:1100].values
output = signal.welch(y_subset, fs)
pxx = np.array(output[1])
# pxx = (pxx - np.mean(pxx)) / np.std(pxx)
# pxx = (pxx) / np.std(pxx)

px.line(y=pxx, x=output[0])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Use projects

# In[ ]:


# # Load multiple datasets
# from wbfm.utils.general.hardcoded_paths import load_paper_datasets
# all_projects_gcamp = load_paper_datasets(['gcamp', 'hannah_O2_fm'])


# In[ ]:


# all_projects_gfp = load_paper_datasets('gfp')


# In[ ]:


# all_projects_immob = load_paper_datasets('immob')


# In[ ]:


# opt = dict(rename_neurons_using_manual_ids=True, interpolate_nan=True)
# df_pxx_wbfm, df_pxx_immob, all_pxx_wbfm, all_pxx_immob = plot_frequencies_for_fm_and_immob_projects(all_projects_gcamp, all_projects_immob, 'VB02', 
#                                                                                                     output_folder=output_folder,**opt)


# In[ ]:


# opt = dict(rename_neurons_using_manual_ids=True, interpolate_nan=True)
# df_pxx_wbfm, df_pxx_immob, all_pxx_wbfm, all_pxx_immob = plot_frequencies_for_fm_and_immob_projects(all_projects_gcamp, all_projects_immob, 'AVAL', 
#                                                                                                     output_folder=output_folder, **opt)


# In[ ]:





# In[ ]:




