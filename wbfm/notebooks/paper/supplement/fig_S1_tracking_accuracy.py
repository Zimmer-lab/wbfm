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


# In[3]:


# fname = "/scratch/neurobiology/zimmer/Charles/dlc_stacks/2022-11-27_spacer_7b_2per_agar/ZIM2165_Gcamp7b_worm1-2022_11_28/project_config.yaml"
# Manually corrected version
fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/manually_annotated/paper_data/ZIM2165_Gcamp7b_worm1-2022_11_28/project_config.yaml"
project_data_gcamp = ProjectData.load_final_project_data_from_config(fname)


# In[4]:


# fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/manually_annotated/paper_data/2022-11-23_worm11/project_config.yaml"
fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/manually_annotated/paper_data/ZIM2165_Gcamp7b_worm1-2022-12-10/project_config.yaml"
project_data_gcamp2 = ProjectData.load_final_project_data_from_config(fname)
len(project_data_gcamp2.finished_neuron_names())


# In[5]:


# # Load multiple datasets
# from wbfm.utils.visualization.hardcoded_paths import load_paper_datasets
# all_projects_gcamp = load_paper_datasets('gcamp')
# all_projects_gfp = load_paper_datasets('gfp')


# In[ ]:





# In[6]:


output_folder = "pipeline_accuracy"


# # Compare three sets of tracks:
# 1. Global tracker only
# 2. Initial pipeline tracks
# 3. Manually corrected tracks

# In[7]:


from wbfm.utils.performance.comparing_ground_truth import calculate_accuracy_from_dataframes, calc_accuracy_of_pipeline_steps
import plotly.express as px
from wbfm.utils.general.utils_paper import apply_figure_settings


# In[8]:


plt.rcParams["font.family"] = "DejaVu Sans"  # Default

df_acc = calc_accuracy_of_pipeline_steps(project_data_gcamp, remove_gt_nan=True)

apply_figure_settings(fig=None, width_factor=0.5, height_factor=0.3, plotly_not_matplotlib=False)

to_save = True
if to_save:
    fname = os.path.join("pipeline_accuracy", "pipeline_steps_boxplots.png")
    plt.savefig(fname, transparent=True)
    fname = str(Path(fname).with_suffix('.svg'))
    plt.savefig(fname)


# In[9]:


df_acc.index.name = 'Neuron'
df_acc_melt = df_acc.melt(var_name='Algorithm option', value_name='Accuracy')


# In[35]:


df_acc


# In[10]:


df_acc_melt


# In[33]:


# fig = px.scatter(df_acc.sort_values(by='Full pipeline'), marginal_y='box')
fig = px.box(df_acc_melt, color='Algorithm option')
apply_figure_settings(fig, width_factor=0.15, height_factor=0.3, plotly_not_matplotlib=True)

fig.update_xaxes(dict(showticklabels=False, title=""),row=1, col=1)
fig.update_yaxes(title="Fraction correctly tracked points", range=[0.5, 1.03])
fig.update_layout(showlegend=False)
fig.show()


to_save = True
if to_save:
    fname = os.path.join("pipeline_accuracy", "pipeline_steps_scatterplot.png")
    fig.write_image(fname, scale=5)
    fname = str(Path(fname).with_suffix('.svg'))
    fig.write_image(fname)


# In[34]:


df_delta = (df_acc.T - df_acc['Single reference frame'].values).T
fig = px.scatter(df_delta.sort_values(by='Full pipeline'))#, marginal_y='box')
apply_figure_settings(fig, width_factor=0.35, height_factor=0.3, plotly_not_matplotlib=True)

fig.update_xaxes(tickfont=dict(color="rgba(0,0,0,0)", size=1), row=1, col=1, title='Neurons with<br>ground truth tracking')
fig.update_yaxes(title="Improvement in tracks<br>(fraction of frames)", zeroline=True, zerolinecolor='black')#, range=[0.5, 1.03])
fig.update_layout(
    showlegend=True,
    legend=dict(
      title='',
      yanchor="top",
      y=0.95,
      xanchor="left",
      x=0.05
    ),
)


fig.show()

to_save = True
if to_save:
    fname = os.path.join("pipeline_accuracy", "pipeline_steps_improvement_scatterplot.png")
    fig.write_image(fname, scale=5)
    fname = str(Path(fname).with_suffix('.svg'))
    fig.write_image(fname)


# # Scratch

# ## Look at accuracy as an image

# In[13]:


neuron_names = project_data_gcamp.finished_neuron_names()


# In[14]:


df_acc_image = df_gt.loc[:, (neuron_names, 'raw_neuron_ind_in_list')] == df_global.loc[:, (neuron_names, 'raw_neuron_ind_in_list')]
px.imshow(df_acc_image.droplevel(1, axis=1), title="Correct global (multiple reference frames)")


# In[ ]:


df_acc_image = df_gt.loc[:, (neuron_names, 'raw_neuron_ind_in_list')] == df_single_reference.loc[:, (neuron_names, 'raw_neuron_ind_in_list')]
px.imshow(df_acc_image.droplevel(1, axis=1), title="Correct global (single reference frames)")


# In[ ]:


df_acc_image = df_gt.loc[:, (neuron_names, 'raw_neuron_ind_in_list')] == df_pipeline.loc[:, (neuron_names, 'raw_neuron_ind_in_list')]
px.imshow(df_acc_image.droplevel(1, axis=1), title="Correct full pipeline")


# In[ ]:


df_acc_image = ~np.isnan(df_gt.loc[:, (neuron_names, 'raw_neuron_ind_in_list')])
px.imshow(df_acc_image.droplevel(1, axis=1), title="Nan points in the ground truth")


# In[ ]:


# (df_gt.loc[:, (neuron_names, 'raw_neuron_ind_in_list')].count() / 1666).hist()


# ## Another project (todo: recalculate tracks using ground truth)

# In[ ]:


from wbfm.utils.performance.comparing_ground_truth import  calc_accuracy_of_pipeline_steps


# In[ ]:


# Original project
# calc_accuracy_of_pipeline_steps(project_data_gcamp, remove_gt_nan=True)


# In[ ]:


# calc_accuracy_of_pipeline_steps(project_data_gcamp2)


# In[ ]:





# In[ ]:





# In[ ]:





# ## Plot fraction tracked (gaps) vs. fraction correct

# In[ ]:


neuron_names = project_data_gcamp.finished_neuron_names()

df_pipeline = project_data_gcamp.initial_pipeline_tracks[neuron_names]
df_gt = project_data_gcamp.final_tracks[neuron_names]


# In[ ]:


opt = dict(column_names=['raw_neuron_ind_in_list'])

df_acc_pipeline = calculate_accuracy_from_dataframes(df_gt, df_pipeline, **opt)


# In[ ]:


df_pipeline_nonnan = df_pipeline.loc[:, (slice(None), 'raw_neuron_ind_in_list')].droplevel(1, axis=1).count()
df_pipeline_nonnan /= df_pipeline.shape[0]


# In[ ]:


df_pipeline_nonnan


# In[ ]:


df_acc_pipeline['fraction_tracked'] = df_pipeline_nonnan
df_acc_pipeline['enough_tracked'] = df_acc_pipeline['fraction_tracked'] > 0.9


# In[ ]:


opt = dict(trendline='ols', trendline_scope="overall", trendline_color_override="black")

px.scatter(df_acc_pipeline, x='matches', y='fraction_tracked', **opt,
          color='enough_tracked', title="Better tracking implies more correct matches")


# In[ ]:


px.scatter(df_acc_pipeline, x='mismatches', y='fraction_tracked', **opt,
          color='enough_tracked', title="Better tracking implies fewer mismatches")


# In[ ]:


df_acc_pipeline.head()


# ## Estimate tracking and segmentation statistics

# In[ ]:


projects_to_compare = dict(
    gcamp = all_projects_gcamp,
    gfp = all_projects_gfp
)

num_tracked_dict = {}
i = 0
thresholds = [0.9, 0.5, 0.1]
for key, project_list in tqdm(projects_to_compare.items()):
    for dataset_name, p in project_list.items():
        for t in thresholds:
            new_key = f"{key}_{i}"
            try:
                num_tracked = len(p.well_tracked_neuron_names(t))
                num_tracked_dict[new_key] = [num_tracked, t, key]
                i += 1
            except AttributeError:
                pass
        
df_tracked = pd.DataFrame(num_tracked_dict).T
cols = ["Neurons > threshold", "Threshold for successfully tracked frames (fraction)", "genotype"]
df_tracked.columns = cols


# In[ ]:


df_tracked.head()


# In[ ]:


# Export options
dpi = 200
width = 6.4 # Defaults for matplotlib
height = 4.8
save_opt = dict(width=width*dpi, height=height*dpi, scale=1)
fig_opt = dict(
    font=dict(
        size=18,
    )
)


# In[ ]:


to_plot = cols[0]
df = df_tracked

fig = px.box(df, y=to_plot, color='genotype', x=cols[1],
            title="A majority of detected objects are well tracked")

fig.update_layout(**fig_opt)
fig.show()

fname = os.path.join(output_folder, 'threshold_for_tracked_objects.png')
fig.write_image(fname, **save_opt)
fname = str(Path(fname).with_suffix('.svg'))
fig.write_image(fname, **save_opt)


# In[ ]:





# In[ ]:





# ## Fix reading of manual annotation

# In[ ]:


fname = "/scratch/neurobiology/zimmer/Charles/dlc_stacks/manually_annotated/paper_data/ZIM2165_Gcamp7b_worm1-2022_11_28/3-tracking/manual_annotation/ZIM2165_Gcamp7b_worm1-2022_11_28.xlsx"


# In[ ]:


df = pd.read_excel(fname)


# In[ ]:


df['Finished?'].count()


# In[ ]:




