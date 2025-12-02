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
all_projects_gcamp = load_paper_datasets(['gcamp', 'hannah_O2_fm'])


# In[4]:


all_projects_gfp = load_paper_datasets('gfp')


# In[5]:


all_projects_immob = load_paper_datasets('immob')


# In[6]:


# Get specific example datasets
project_data_gcamp = all_projects_gcamp['ZIM2165_Gcamp7b_worm1-2022_11_28']
# project_data_immob = all_projects_immob['2022-12-13_15-16_ZIM2165_immob_worm9-2022-12-13']

# Comparing 2 datasets
project_data_gcamp2 = all_projects_gcamp['ZIM2165_Gcamp7b_worm1-2022-11-30']
project_data_immob2 = all_projects_immob['ZIM2165_immob_adj_set_2_worm2-2022-11-30']


# In[7]:


# # Same individual: fm and immob
# fname = '/lisc/data/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-12-06_wbfm_to_immob/2022-12-06_17-23_ZIM2165_worm5-2022-12-06/project_config.yaml'
# project_data_fm2immob_fm = ProjectData.load_final_project_data_from_config(fname, verbose=0)

# fname = '/lisc/data/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-12-06_wbfm_to_immob/2022-12-06_17-41_ZIM2165_immob_worm5-2022-12-06'
# project_data_fm2immob_immob = ProjectData.load_final_project_data_from_config(fname, verbose=0)

# # Same individual: fm and immob
# # fname = '/lisc/data/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-12-06_wbfm_to_immob/2022-12-06_17-23_ZIM2165_worm5-2022-12-06/project_config.yaml'
# # project_data_fm2immob_fm2 = ProjectData.load_final_project_data_from_config(fname, verbose=0)

# fname = '/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-12-06_wbfm_to_immob/2022-12-06_11-07_ZIM2165_immob_worm1-2022-12-06/project_config.yaml'
# project_data_fm2immob_immob2 = ProjectData.load_final_project_data_from_config(fname, verbose=0)


# In[8]:


# [str(p.project_config.self_path) for p in all_projects_gcamp.values()]


# In[9]:


path_to_saved_data = "../step1_analysis/figure_1"
path_to_shared_saved_data = "/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/step1_analysis/shared"


# # Precalculate the trace dataframes (cached to disk)

# In[10]:


# # Optional: clear just one cache
# for project_dict in tqdm([all_projects_immob]):
#     for name, project_data in tqdm(project_dict.items()):
#         project_data.data_cacher.clear_disk_cache(delete_invalid_indices=False, delete_traces=True)


# In[11]:


# # Optional: clear the trace cache
# for project_dict in tqdm([all_projects_gcamp, all_projects_gfp, 
#                           all_projects_immob]):
#     for name, project_data in tqdm(project_dict.items()):
#         project_data.data_cacher.clear_disk_cache(delete_invalid_indices=False, delete_traces=True)


# In[12]:


for project_dict in tqdm([all_projects_gcamp, all_projects_gfp, all_projects_immob]):
    for name, project_data in tqdm(project_dict.items()):
        df_traces = project_data.calc_paper_traces()
        df_res = project_data.calc_paper_traces(residual_mode='pca')
        df_global = project_data.calc_paper_traces(residual_mode='pca_global')
        if df_res is None or df_global is None or df_traces is None:
            raise ValueError


# In[13]:


# project_data_fm2immob_immob.data_cacher.clear_disk_cache(delete_invalid_indices=False, delete_traces=True)


# In[14]:


# # Also for FM to IMMOB datasets

# for project_data in [project_data_fm2immob_fm, project_data_fm2immob_immob]:
#     df_traces = project_data.calc_paper_traces()
#     df_res = project_data.calc_paper_traces(residual_mode='pca')
#     df_global = project_data.calc_paper_traces(residual_mode='pca_global')
#     if df_res is None or df_global is None or df_traces is None:
#         raise ValueError


# # Plots

# ## Heatmaps: immobilized and WBFM

# In[15]:


from wbfm.utils.visualization.plot_traces import make_summary_interactive_heatmap_with_pca, make_summary_heatmap_and_subplots


# In[16]:


# project_data_gcamp.use_physical_x_axis = True
# project_data_immob.use_physical_x_axis = True


# In[17]:


# NOT USED (combined plot)
# fig = make_summary_interactive_heatmap_with_pca(project_data_gcamp, to_save=True, to_show=True, output_folder="intro/example_summary_plots_wbfm")


# In[18]:


# Print number of neurons
project_data_gcamp.calc_paper_traces().shape


# In[19]:


# fig = make_summary_interactive_heatmap_with_pca(project_data_immob, to_save=True, to_show=True, output_folder="example_summary_plots_immob")
##del project_data_gcamp.worm_posture_class


# In[20]:


# USED: different figures for each
fig1, fig2 = make_summary_heatmap_and_subplots(project_data_gcamp, trace_opt=dict(use_paper_options=True, interpolate_nan=True), 
                                               to_save=True, to_show=True, 
                                               base_height=[0.25, 0.2], base_width=0.6, output_folder="intro/example_summary_plots_wbfm")


# In[21]:


# %debug


# In[22]:


# # Comparison: interpolated values
# fig1, fig2 = make_summary_heatmap_and_subplots(project_data_gcamp, trace_opt=dict(use_paper_options=True, interpolate_nan=True), to_save=True, to_show=True, 
#                                                output_folder="intro/example_summary_plots_wbfm")


# In[23]:


# fig1, fig2 = make_summary_heatmap_and_subplots(project_data_immob, trace_opt=dict(use_paper_options=True), include_speed_subplot=False,
#                                                to_save=True, to_show=True, output_folder="intro/example_summary_plots_immob")


# ## Heatmaps (actually figure 2): comparing fm to immob

# In[24]:


from wbfm.utils.visualization.plot_traces import make_summary_interactive_heatmap_with_pca, make_summary_heatmap_and_subplots


# In[25]:


project_data_gcamp2.use_physical_x_axis = True
project_data_immob2.use_physical_x_axis = True


# In[26]:


# from wbfm.utils.general.utils_behavior_annotation import approximate_behavioral_annotation_using_pc1
# approximate_behavioral_annotation_using_pc1(project_data_fm2immob_immob)
# approximate_behavioral_annotation_using_pc1(project_data_fm2immob_fm)  # This is an old dataset and the behavior was deleted


# In[27]:


##del project_data_immob2.worm_posture_class


# In[28]:


# fig1, fig2 = make_summary_heatmap_and_subplots(project_data_immob2, trace_opt=dict(use_paper_options=True, interpolate_nan=True, verbose=True), include_speed_subplot=False,
#                                                to_save=True, to_show=True, base_width=0.55, output_folder="intro/fm_to_immob/immob",
#                                               ethogram_on_top=True)


# In[29]:


project_data_gcamp2.calc_paper_traces().shape


# In[30]:


fig1, fig2 = make_summary_heatmap_and_subplots(project_data_gcamp2, trace_opt=dict(use_paper_options=True, interpolate_nan=True, verbose=True), include_speed_subplot=False,
                                               to_save=True, to_show=True, base_width=0.55, ethogram_on_top=True, output_folder="intro/fm_to_immob/fm")


# In[31]:


# trace_opt=dict(use_paper_options=True, interpolate_nan=False, verbose=True)
# df = project_data_fm2immob_fm.calc_default_traces(**trace_opt)


# In[32]:


# project_data_immob2.calc_paper_traces()


# In[33]:


# project_data_immob2.tail_neuron_names()


# In[34]:


# # Test: include tail neurons
# fig1, fig2 = make_summary_heatmap_and_subplots(project_data_fm2immob_immob2, 
#                                                trace_opt=dict(use_paper_options=True, interpolate_nan=False), 
#                                                include_speed_subplot=False,
#                                                to_save=False, to_show=True, output_folder="intro/fm_to_immob/immob")


# ### Just plot the legend for reversal shading

# In[35]:


from wbfm.utils.general.utils_paper import export_legend_for_paper

fname = 'intro/reversal_legend.png'
export_legend_for_paper(reversal_shading=True, fname=fname)


# In[36]:



fname = 'intro/reversal_and_collision_legend.png'
export_legend_for_paper(reversal_shading=True, fname=fname, include_self_collision=True)


# In[37]:


from wbfm.utils.general.utils_paper import export_legend_for_paper

fname = 'intro/ethogram_legend.png'
export_legend_for_paper(ethogram=True, fname=fname)


# ## Triggered average examples

# In[38]:


# from wbfm.utils.visualization.plot_traces import make_grid_plot_using_project
from wbfm.utils.traces.triggered_averages import FullDatasetTriggeredAverages
from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes, shade_using_behavior
from wbfm.utils.visualization.utils_plot_traces import plot_triggered_averages


# In[ ]:





# In[39]:


# plot_triggered_averages([project_data_gcamp, project_data_immob], output_foldername="intro/basic_triggered_average")


# ## PCA variance explained plot of all datasets

# In[40]:


from wbfm.utils.visualization.multiproject_wrappers import get_all_variance_explained
from wbfm.utils.visualization.utils_plot_traces import plot_with_shading
from wbfm.utils.general.utils_paper import apply_figure_settings, plotly_paper_color_discrete_map


# In[41]:


gcamp_var, gfp_var, immob_var, gcamp_var_sum, gfp_var_sum, immob_var_sum = get_all_variance_explained(all_projects_gcamp, all_projects_gfp, all_projects_immob)


# In[42]:


fig, ax = plt.subplots(dpi=200, figsize=(5,5))

var_sum_dict = {'Freely Moving (GCaMP)': gcamp_var_sum, 'Immobilized (GCaMP)': immob_var_sum, 'Freely Moving (GFP)': gfp_var_sum}
cmap = plotly_paper_color_discrete_map()

for name, mat in var_sum_dict.items():
    means = np.mean(mat, axis=1)
    color = cmap[name]
    plot_with_shading(means, np.std(mat, axis=1), label=name, ax=ax, lw=2,
                      x=np.arange(1, len(means) + 1), color=color)
# plt.legend()
# plt.title("Dimensionality")
plt.ylabel("Cumulative explained variance")
plt.ylim(0.2, 1.0)
plt.xlabel("Mode")

from matplotlib.ticker import MaxNLocator
ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))

apply_figure_settings(fig, width_factor=0.2, height_factor=0.25, plotly_not_matplotlib=False)
plt.tight_layout()

output_foldername = 'intro'
fname = f"pca_cumulative_variance.png"
fname = os.path.join(output_foldername, fname)
plt.savefig(fname, transparent=True)
fig.savefig(fname.replace(".png", ".svg"), transparent=True)


# In[ ]:





# In[ ]:





# # PCA weights across wbfm and immob

# In[43]:


from wbfm.utils.visualization.utils_cca import calc_pca_weights_for_all_projects
from wbfm.utils.external.utils_plotly import plotly_boxplot_colored_boxes
from wbfm.utils.general.utils_paper import apply_figure_settings
from wbfm.utils.general.hardcoded_paths import neurons_with_confident_ids


# In[191]:



wbfm_weights = calc_pca_weights_for_all_projects(all_projects_gcamp, use_paper_options=True, combine_left_right=True,
                                                 include_only_confident_ids=True)


# In[192]:


immob_weights = calc_pca_weights_for_all_projects(all_projects_immob, use_paper_options=True, combine_left_right=True,
                                                  include_only_confident_ids=True)


# In[193]:


len(neurons_with_confident_ids())


# In[194]:


len(neurons_with_confident_ids(combine_left_right=True))


# In[211]:


len(wbfm_weights.count()), len(immob_weights.count())


# In[196]:


# # Create a list of colors to highlight BAG
# base_color = '#1F77B4'  # Gray
# names = list(wbfm_weights.columns)
# colors = ['#000000' if 'BAG' in n else base_color for n in names]

# # fig = px.box(wbfm_weights)
# fig = plotly_boxplot_colored_boxes(wbfm_weights, colors)
# # Transparent background
# apply_figure_settings(fig, width_factor=0.6, height_factor=0.2, plotly_not_matplotlib=True)
# fig.update_yaxes(dict(title="PC1 weight"), zeroline=True, zerolinewidth=1, zerolinecolor="black")
# fig.update_xaxes(dict(title=""))
# # Add the 0 line back

# fig.show()

# fname = os.path.join("intro", 'wbfm_pca_weights.png')
# fig.write_image(fname, scale=3)
# fname = Path(fname).with_suffix('.svg')
# fig.write_image(fname)


# In[197]:


# # Create a list of colors to highlight BAG
# base_color = '#FF7F0E'  # Orange, overall immob color
# names = list(immob_weights.columns)
# # colors = ['#EF553B' if 'BAG' in n else base_color for n in names]
# colors = ['#000000' if 'BAG' in n else base_color for n in names]

# fig = plotly_boxplot_colored_boxes(immob_weights, colors)
# apply_figure_settings(fig, width_factor=0.6, height_factor=0.2, plotly_not_matplotlib=True)
# fig.update_yaxes(dict(title="PC1 weight"), zeroline=True, zerolinewidth=1, zerolinecolor="black")
# fig.update_yaxes(dict(title="PC1 weight"))
# fig.update_xaxes(dict(title="Neuron Name"))
# fig.show()

# fname = os.path.join("intro", 'immob_pca_weights.png')
# fig.write_image(fname, scale=3)
# fname = Path(fname).with_suffix('.svg')
# fig.write_image(fname)


# ## FM and immob on same plot

# In[212]:


from wbfm.utils.visualization.utils_plot_traces import add_p_value_annotation
from wbfm.utils.general.utils_paper import data_type_name_mapping, plotly_paper_color_discrete_map
import plotly.graph_objects as go
from wbfm.utils.general.utils_paper import intrinsic_categories_color_discrete_map
from wbfm.utils.external.utils_plotly import colored_text


# ### Precalculate all p values that will be needed

# In[213]:


'URX' in df_both['neuron_name'], 'URX' in names_to_keep


# In[254]:


from scipy import stats
from statsmodels.stats.multitest import multipletests

opts_multipletests = dict(method='fdr_bh', alpha=0.05)
# opts_multipletests = dict(method='sidak')

names_to_keep = set(wbfm_weights.columns).intersection(immob_weights.columns)
wbfm_melt = wbfm_weights.melt(var_name='neuron_name', value_name='PC1 weight').assign(dataset_type='gcamp')
immob_melt = immob_weights.melt(var_name='neuron_name', value_name='PC1 weight').assign(dataset_type='immob')
df_both = pd.concat([wbfm_melt, immob_melt], axis=0)
df_both = df_both[df_both['neuron_name'].isin(names_to_keep)]
df_both['Dataset Type'] = df_both['dataset_type'].map(data_type_name_mapping())
print(len(df_both['neuron_name'].unique()))

# Significantly different from 0... need a permutation version, so use an extra function
# From: https://stackoverflow.com/questions/73569894/permutation-based-alternative-to-scipy-stats-ttest-1samp
def _t_statistic(x, axis=-1):
    # return stats.ttest_1samp(x, popmean=0, axis=axis).statistic
    return stats.ttest_1samp(x, popmean=0).statistic

def t_statistic_permutation(x):
    return stats.permutation_test((x.values, ), _t_statistic, permutation_type='samples', ).pvalue

# func = lambda x: stats.ttest_1samp(x, 0)[1]
df_groupby = df_both.dropna().groupby(['neuron_name', 'dataset_type'])
df_pvalue = df_groupby['PC1 weight'].apply(t_statistic_permutation).to_frame()
df_pvalue.columns = ['p_value']

# Multiple comparison correction in the same way for all tests
output = multipletests(df_pvalue.values.squeeze(), **opts_multipletests)
df_pvalue['p_value_corrected'] = output[1]
df_pvalue['significance_corrected'] = output[0]

# Sign of medians
df_medians_gcamp = df_groupby['PC1 weight'].median()[(slice(None), 'gcamp')]
df_medians_immob = df_groupby['PC1 weight'].median()[(slice(None), 'immob')]

# Significantly different from each other (should be exact same as the boxplot)
df_groupby = df_both.dropna().groupby(['neuron_name'])
func = lambda x: stats.ttest_ind(x[x['dataset_type']=='gcamp']['PC1 weight'], x[x['dataset_type']=='immob']['PC1 weight'], 
                                 equal_var=False, permutations=1000)[1]
df_significant_diff = df_groupby.apply(func).to_frame()
df_significant_diff.columns = ['p_value_diff']
# Multiple comparison correction in the same way for all tests
output = multipletests(df_significant_diff.values.squeeze(), **opts_multipletests)
df_significant_diff['p_value_corrected_diff'] = output[1]
df_significant_diff['significance_corrected_diff'] = output[0]
# df_significant_diff.head()


# In[255]:


# Color xticks by later pie chart colors
# Load from disk from previous run
# NOTE: IF UPDATING NEURONS: this will remove neurons, which then will not get into the pie chart later
df_categories = pd.read_excel('intro/intrinsic_categories.xlsx')
if len(df_both['neuron_name'].unique()) > len(df_categories['neuron_name'].unique()):
    print("Not all neurons have colors; skipping coloring")
    df_both['neuron_name_html'] = df_both['neuron_name']
else:
    df_categories['Result_simple_color'] = df_categories['Result_simple'].map(intrinsic_categories_color_discrete_map(return_hex=False))
    df_both = pd.merge(df_both, df_categories, on='neuron_name', validate='many_to_one')
    df_both['neuron_name_html'] = df_both.apply(lambda x: colored_text(x['neuron_name'], x['Result_simple_color'], bold=True), axis=1)
    print(len(df_both['neuron_name'].unique()))


# ### Box plot with significance

# In[266]:


# Plot
fig = px.box(df_both, y='PC1 weight', x='neuron_name_html', 
             color='Dataset Type', 
            color_discrete_map=plotly_paper_color_discrete_map(),
            category_orders={'Dataset Type': ['Immobilized (GCaMP)', 'Freely Moving (GCaMP)']})

# add_p_value_annotation(fig, x_label='all', show_ns=False, show_only_stars=True, permutations=1000,
#                       height_mode='top_of_data')#, _format=dict(text_height=0.075))
add_p_value_annotation(fig, x_label='all', show_ns=False, show_only_stars=True, precalculated_p_values=df_significant_diff['p_value_corrected_diff'],
                      height_mode='top_of_data')
apply_figure_settings(fig, width_factor=0.83, height_factor=0.3, plotly_not_matplotlib=True)

fig.update_layout(legend=dict(
    yanchor="top",
    y=0.85,
    xanchor="left",
    x=0.6
))
# Add a fake trace to get a legend entry for GFP
# dummy_data = go.Box(
#     x=[None],
#     y=[None],
#     # mode="markers",
#     name="Freely Moving (GFP)",
#     #fillcolor='gray', 
#     line=dict(color='gray')
#     # marker=dict(size=7, color="gray"),
# )
# fig.add_trace(dummy_data)

fig.update_yaxes(dict(title="PC1 weight"), zeroline=True, zerolinewidth=1, zerolinecolor="black", )#range=[-0.2, 0.55])
fig.update_xaxes(dict(title="Neuron Name"))
fig.show()

fname = os.path.join("intro", 'fm_and_immob_pca_weights.png')
fig.write_image(fname, scale=3)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)


# ## Pie chart summarizing above
# 
# Get the p values between immob and fm and build 4 categories:
# 1. No difference, both positive
# 2. No difference, both negative
# 3. Positive difference
# 4. Negative difference
# 
# ... problem: AVA will then show a very strong difference, as will many that stay "on the same side"... what we really want is a switch in correlations for category 3/4
# New categories (including above):
# 1. immob NS diff from 0, fm positive
# 2. immob NS diff from 0, fm negative
# 3. Both NS diff from 0, but not from each other (will this exist?)
# 4. fm NS diff from 0, immob positive
# 5. fm NS diff from 0, immob negative
# 6. Both NS diff from 0
# 
# ... this is a lot of categories! I can maybe just ignore categories 4-6, as they are not interesting in FM
# 
# Restart:
# 1. Both significant, both same sign
# 2. Both significant, switch sign
# 3. Only significant in FM (ignore sign)
# 4. Not significant in FM (ignore significance in immob)
# 
# Fundamentally, we don't care much about the ones that are significant in immob... intuitively we think they are just less noisy, but it's hard to quantify
# 
# ... Unfortunately, many neurons are slightly on one edge of significance in immob or fm, creating a lot more "switching" or "fm only" guys than really make sense...
# 
# Restart, removing the focus on the (not shown) comparison to 0:
# 1. Same sign, both sig from 0, NS or sig difference: Intrinsic
# 2. Different sign, but sig from 
# 3. 
# 

# In[257]:


from wbfm.utils.general.hardcoded_paths import intrinsic_definition


# In[258]:


# from scipy import stats
# from statsmodels.stats.multitest import multipletests

# # Significantly different from 0
# func = lambda x: stats.ttest_1samp(x, 0)[1]
# df_groupby = df_both.dropna().groupby(['neuron_name', 'dataset_type'])
# df_pvalue = df_groupby['PC1 weight'].apply(func).to_frame()
# df_pvalue.columns = ['p_value']

# # Multiple comparison correction in the same way for all tests
# output = multipletests(df_pvalue.values.squeeze(), method='fdr_bh', alpha=0.05)
# df_pvalue['p_value_corrected'] = output[1]
# df_pvalue['significance_corrected'] = output[0]

# # Sign of medians
# df_medians_gcamp = df_groupby['PC1 weight'].median()[(slice(None), 'gcamp')]
# df_medians_immob = df_groupby['PC1 weight'].median()[(slice(None), 'immob')]

# # Significantly different from each other (should be exact same as the boxplot)
# df_groupby = df_both.dropna().groupby(['neuron_name'])
# func = lambda x: stats.ttest_ind(x[x['dataset_type']=='gcamp']['PC1 weight'], x[x['dataset_type']=='immob']['PC1 weight'], 
#                                  equal_var=False)[1]
# df_significant_diff = df_groupby.apply(func).to_frame()
# df_significant_diff.columns = ['p_value_diff']
# # Multiple comparison correction in the same way for all tests
# output = multipletests(df_significant_diff.values.squeeze(), method='fdr_bh', alpha=0.05)
# df_significant_diff['p_value_corrected_diff'] = output[1]
# df_significant_diff['significance_corrected_diff'] = output[0]
# df_significant_diff.head()


# In[259]:


# Process p value comparisons to 0
df_pvalue_thresh = df_pvalue['significance_corrected'].reset_index()

# Collect signficance calculations per datatype
df_pivot = df_pvalue_thresh.pivot_table(index='neuron_name', columns='dataset_type', values='significance_corrected', aggfunc='first')
df_4states_complex = df_pivot.astype(str).radd(df_pivot.columns + '_')
df_4states_complex = (df_4states_complex['gcamp'] + '_' + df_4states_complex['immob'])#.reset_index()

# Add suffix to the state: are both medians on the same side?
df_medians_gcamp.name = 'same_sign'
df_medians_immob.name = 'same_sign'
df_medians_same_sign = ((df_medians_gcamp>0) == (df_medians_immob>0)).astype(str).radd(df_medians_gcamp.name + '_')
df_4states_complex = df_4states_complex.to_frame().join(df_medians_same_sign)#.reset_index()

# Add suffix to the state: is the difference between them significant?
df_4states_complex = df_4states_complex.join(df_significant_diff['significance_corrected_diff'].astype(str).radd('diff_'))

# Combine into final categories
df_4states_complex.columns = ['pvalue_result', 'diff_sign', 'diff_sig']
df_4states = (df_4states_complex['pvalue_result'] + '_' + df_4states_complex['diff_sign'] + '_' + df_4states_complex['diff_sig']).to_frame()
df_4states.columns = ['Result']
df_4states.head()


# In[265]:


from wbfm.utils.general.utils_paper import intrinsic_categories_color_discrete_map

df_4states_counts = df_4states['Result'].value_counts().reset_index()
df_4states_counts['Result_simple'] = df_4states_counts['Result'].map(intrinsic_definition)
df_4states['Result_simple'] = df_4states['Result'].map(intrinsic_definition)

# Drop uninteresting rows
# df_4states_counts = df_4states_counts.drop(df_4states_counts[df_4states_counts['Result'].str.contains('No')].index)
# df_4states_counts = df_4states_counts.drop(df_4states_counts[df_4states_counts['Result'].str.contains('immob only')].index)

fig = px.pie(df_4states_counts, names='Result_simple', values='count', color='Result_simple', 
             color_discrete_map=intrinsic_categories_color_discrete_map())
apply_figure_settings(fig, width_factor=0.2, height_factor=0.25)
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.0,
    xanchor="left",
    x=0.2
))
fig.update_traces(texttemplate='%{percent:.2p}')

# fig.update_traces(
#         textposition="outside",
#         texttemplate='%{percent:01f}')
fig.show()

output_foldername = 'intro'
fname = os.path.join(output_foldername, 'manifold_participation_pie_chart.png')
fig.write_image(fname, scale=3)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)


# In[261]:


df_4states['Result_simple'].value_counts()


# ### Add english language column and export

# In[262]:


from wbfm.utils.general.hardcoded_paths import intrinsic_categories_short_description
df_4states['Result_description'] = df_4states['Result'].map(intrinsic_categories_short_description())

# Also add the original booleans that lead to these categories
df_4states_export = df_4states.copy().join(df_4states_complex.loc[:, ['pvalue_result', 'diff_sign', 'diff_sig']]).drop(columns='Result')
# df_4states_export

fname = 'intro/intrinsic_categories.xlsx'
df_4states.sort_values(by='Result_description').to_excel(fname)


# # Variance explained by mode 1 across neurons
# 
# i.e. the cumulative histogram, with error bar per dataset

# In[64]:


from wbfm.utils.visualization.multiproject_wrappers import build_dataframe_of_variance_explained
from wbfm.utils.general.utils_paper import plotly_paper_color_discrete_map
from wbfm.utils.general.utils_paper import apply_figure_settings


# In[65]:


trace_opt = dict(use_paper_options=True, interpolate_nan=True)

all_dfs = []
for n in tqdm([1, 2]):
    opt = dict(n_components=n, melt=True)

    df_var_exp_gcamp = build_dataframe_of_variance_explained(all_projects_gcamp, **opt, **trace_opt)
    df_var_exp_gcamp['Type of data'] = 'gcamp'
    df_var_exp_gcamp['n_components'] = n

    df_var_exp_immob = build_dataframe_of_variance_explained(all_projects_immob, **opt, **trace_opt)
    df_var_exp_immob['Type of data'] = 'immob'
    df_var_exp_immob['n_components'] = n

    df_var_exp_gfp = build_dataframe_of_variance_explained(all_projects_gfp, **opt, **trace_opt)
    df_var_exp_gfp['Type of data'] = 'gfp'
    df_var_exp_gfp['n_components'] = n
    
    all_dfs.extend([df_var_exp_gcamp, df_var_exp_immob, df_var_exp_gfp])

df_var_exp = pd.concat(all_dfs, axis=0)
df_var_exp.head()


# In[66]:


df_var_exp[(df_var_exp['dataset_name'] == '2022-11-23_worm10') & (df_var_exp['neuron_name'] == 'ALA')]


# In[67]:


# px.histogram(df_var_exp, color='dataset_name', x='fraction_variance_explained', cumulative=True, 
#              facet_row='Type of data',
#              barmode='overlay', histnorm='percent')


# In[68]:


df_var_exp_hist = df_var_exp.copy()

# Get counts of neurons in each bin
bins = np.linspace(0, 1, 50)
func = lambda Z: np.cumsum(np.histogram(Z, bins=bins)[0])
df_var_exp_hist = df_var_exp_hist.groupby(['dataset_name', 'n_components'])['fraction_variance_explained'].apply(func)
# df_var_exp_hist.head()

# Explode to long form
long_vars = df_var_exp_hist.reset_index().explode('fraction_variance_explained')
long_vars.rename(columns={'fraction_variance_explained': 'cumulative_fraction_variance_explained'}, inplace=True)
long_vars.sort_values(by=['dataset_name', 'cumulative_fraction_variance_explained'], inplace=True)
# Just remake the bins
long_vars['cumcount'] = long_vars.groupby(['dataset_name', 'n_components']).cumcount()
long_vars['fraction_count'] = long_vars['cumcount'] / long_vars['cumcount'].max()

# Add back datatype column
long_vars = long_vars.merge(df_var_exp[['dataset_name', 'n_components', 'Type of data']], on=['dataset_name', 'n_components'])

# Normalize by number of total neurons (Only take one component to avoid duplication)
total_num_neurons = df_var_exp[df_var_exp['n_components']==1].dropna()['dataset_name'].value_counts()
long_vars.index = long_vars['dataset_name']  # So the division matches
long_vars['cumulative_fraction_variance_explained'] = long_vars['cumulative_fraction_variance_explained'] / total_num_neurons
long_vars.reset_index(drop=True, inplace=True)

long_vars.head()


# In[69]:


# px.line(long_vars, x='fraction_count', 
#         y='cumulative_fraction_variance_explained', color='dataset_name',
#        facet_row='Type of data')


# In[70]:


from wbfm.utils.external.utils_plotly import plotly_plot_mean_and_shading

opt = dict(x='fraction_count', y='cumulative_fraction_variance_explained', color='n_components', 
           cmap=plotly_paper_color_discrete_map()
          )

fig = None
g = 'gcamp'
df_subset = long_vars[(long_vars['Type of data']==g)]
fig = plotly_plot_mean_and_shading(df_subset, line_name=g, fig=fig, **opt,
                                   x_intersection_annotation=0.5)
# for n in [1, 2]:
#     df_subset = long_vars[(long_vars['Type of data']==g)]# & (long_vars['n_components']==n)]
#     fig = plotly_plot_mean_and_shading(df_subset, line_name=g, fig=fig, **opt,
#                                        x_intersection_annotation=0.5)

fig.update_xaxes(title='Var. explained (fraction)', range=[0, 1.05])
fig.update_yaxes(title='Fraction of neurons <br> (cumulative)', range=[0, 1.05])
fig.update_layout(
        showlegend=True,
        legend=dict(
            title='Mode',
          yanchor="middle",
          y=0.25,
          xanchor="left",
          x=0.6
        )
    )# fig.update_traces(line=dict(color=plotly_paper_color_discrete_map()['PCA']))
fig.update_traces(name='1 + 2', selector=dict(name='2'))

apply_figure_settings(fig, width_factor=0.3, height_factor=0.2)

fig.show()

to_save = True
if to_save:
    output_foldername = 'intro/dimensionality'
    fname = os.path.join(output_foldername, 'variance_explained_by_pc1_and_pc2_cumulative.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)


# In[71]:


# %debug


# In[72]:


long_vars['Type of data'].unique()


# In[73]:


from wbfm.utils.external.utils_plotly import plotly_plot_mean_and_shading
# Same as above, but with FM and immob, not only FM

opt = dict(x='fraction_count', y='cumulative_fraction_variance_explained', color='Type of data', 
           cmap=plotly_paper_color_discrete_map()
          )
# fig = None
# g = 'gcamp'
# df_subset = long_vars[(long_vars['Type of data']==g)]
# fig = plotly_plot_mean_and_shading(df_subset, line_name=g, fig=fig, **opt,
#                                    x_intersection_annotation=0.5)


# Add immob
n_components = 2
fig = None
fig = plotly_plot_mean_and_shading(long_vars[long_vars['n_components']==n_components], line_name='immob', fig=fig, **opt,
                                  x_intersection_annotation=0.5, annotation_position='right')

# Add gfp
n_components = 2
fig = None
fig = plotly_plot_mean_and_shading(long_vars[long_vars['n_components']==n_components], line_name='gfp', fig=fig, **opt,
                                  x_intersection_annotation=0.5, annotation_position='right')



fig.update_xaxes(title='Var. explained by modes 1 and 2 (fraction)', range=[0, 1.05])
fig.update_yaxes(title='Fraction of neurons (cumulative)', range=[0, 1.05])
fig.update_traces(name='Immobilized', selector=dict(name='immob'))
fig.update_traces(name='Freely Moving (GFP)', selector=dict(name='gfp'))
fig.update_traces(name='Freely Moving (GCaMP)', selector=dict(name='gcamp'))
fig.update_layout(
        showlegend=True,
        legend=dict(
          title='Datatype',
          yanchor="middle",
          y=0.15,
          xanchor="left",
          x=0.5
        )
    )
# In the supp, so it's larger
apply_figure_settings(fig, width_factor=0.5, height_factor=0.4)


fig.show()

output_foldername = 'intro/dimensionality'
fname = os.path.join(output_foldername, 'variance_explained_by_pc1_cumulative_with_immob.png')
fig.write_image(fname, scale=3)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)


# In[74]:


# %debug


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Scratch

# ## Where is DD01?

# In[75]:


'DD01' in wbfm_weights, 'DD01' in immob_weights


# In[ ]:





# In[ ]:




