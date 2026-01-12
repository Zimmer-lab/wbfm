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


# fname = "/scratch/neurobiology/zimmer/Charles/dlc_stacks/2022-11-27_spacer_7b_2per_agar/ZIM2165_Gcamp7b_worm1-2022_11_28/project_config.yaml"
# Manually corrected version
fname = "/lisc/data/scratch/neurobiology/zimmer/fieseler/wbfm_projects/manually_annotated/paper_data/ZIM2165_Gcamp7b_worm1-2022_11_28_updated_format/project_config.yaml"
project_data_gcamp = ProjectData.load_final_project_data_from_config(fname)


# In[4]:


from wbfm.utils.general.utils_hardcoded import load_paper_datasets
all_projects_gcamp = load_paper_datasets(['gcamp', 'hannah_O2_fm'])


# # Diagram with correct behavior transitions

# In[ ]:


from wbfm.utils.external.utils_pandas import get_dataframe_of_transitions
from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
from functools import reduce

# For each project, get the transition probability dataframe
all_transitions = []
for name, p in tqdm(all_projects_gcamp.items()):
    beh_vec = p.worm_posture_class.beh_annotation(fluorescence_fps=True)
    beh_vec = BehaviorCodes.convert_to_simple_states_vector(beh_vec)
    beh_vec = [b.value for b in beh_vec]
    df_transition = get_dataframe_of_transitions(beh_vec, convert_to_probabilities=False, ignore_diagonal=True)
    
    all_transitions.append(df_transition)
    


# In[182]:


df_all_transitions = reduce(lambda a, b: a.add(b, fill_value=0), all_transitions).astype(int)
df_all_transitions


# In[312]:


mapper = lambda val: BehaviorCodes(val).name

df = df_all_transitions.rename(columns=mapper).rename(index=mapper)


# In[313]:


df = df.T.drop(columns=['UNKNOWN', 'PAUSE']).T.drop(columns=['UNKNOWN', 'PAUSE'])


# In[ ]:


df.sum(axis=1)


# In[ ]:


df.div(df.sum(axis=1), axis=0)


# In[ ]:


px.imshow(df)


# In[319]:


import networkx as nx
plt.figure()

df = df_all_transitions.rename(columns=mapper).rename(index=mapper)
df = df.T.drop(columns=['UNKNOWN', 'PAUSE']).T.drop(columns=['UNKNOWN', 'PAUSE'])
# Build graph: a node per row
df = df.div(df.sum(axis=1), axis=0)
print(df)
G = nx.from_pandas_adjacency(df, create_using=nx.DiGraph)

threshold = 0.03
edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] < threshold]
print(edges_to_remove)
G.remove_edges_from(edges_to_remove)

# Draw with edge weights
pos = nx.circular_layout(G)

edges = G.edges(data=True)
f = lambda x : 4*x #np.log(x+1e-6)
weights = [f(d["weight"]) for _, _, d in edges]
print(weights)

nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=10000,
    width=weights,  # makes stronger edges thicker
    connectionstyle="arc3,rad=0.2",
    nodelist=list(G)
)

ax = plt.gca()
x_values, y_values = zip(*pos.values())
ax.set_xlim(min(x_values) - 0.5, max(x_values) + 0.5)
ax.set_ylim(min(y_values) - 0.5, max(y_values) + 0.5)

plt.show()


# In[ ]:


list(G)


# # Do PCA, CCA on real behaviors, and CCA on binarized behaviors

# In[8]:


# project_data_gcamp.worm_posture_class


# In[64]:


output_folder = 'fig2'


# In[65]:


from wbfm.utils.visualization.utils_cca import CCAPlotter


# In[66]:


project_data_gcamp.use_physical_x_axis = True
beh_kwargs = dict(additional_behaviors=[f"eigenworm{i}" for i in range(4)])

cca_plotter = CCAPlotter(project_data_gcamp, truncate_traces_to_n_components=3, preprocess_behavior_using_pca=True, trace_kwargs=dict(use_paper_options=True),
                        beh_kwargs=beh_kwargs)



# ## Alt: more manifold modes

# In[67]:


# project_data_gcamp.use_physical_x_axis = True
# beh_kwargs = dict(additional_behaviors=[f"eigenworm{i}" for i in range(4)])

# cca_plotter6 = CCAPlotter(project_data_gcamp, truncate_traces_to_n_components=100, preprocess_behavior_using_pca=True, trace_kwargs=dict(use_paper_options=True),
#                          beh_kwargs=beh_kwargs)



# In[68]:


# fig = cca_plotter6.plot(plot_3d=False, output_folder=output_folder, show_legend=False)


# In[69]:


# fig = cca_plotter6.plot(plot_3d=True, output_folder=output_folder, show_legend=False)


# ## Example dataset modes in 2d

# In[70]:


fig = cca_plotter.plot(plot_3d=False, output_folder=output_folder, show_legend=False)


# In[71]:


# fig = cca_plotter.plot(plot_3d=True, output_folder=output_folder, show_legend=False)


# In[72]:


fig = cca_plotter.plot(plot_3d=False, binary_behaviors=True, show_legend=False, output_folder=output_folder)#, beh_annotation_kwargs=dict(include_collision=True))


# In[73]:


fig = cca_plotter.plot(plot_3d=False, binary_behaviors=False, show_legend=False, use_pca=True, output_folder=output_folder)


# # Calculate variance explained per dataset

# In[74]:


from wbfm.utils.visualization.utils_cca import calc_r_squared_for_all_projects
from wbfm.utils.general.utils_paper import plotly_paper_color_discrete_map
from wbfm.utils.general.utils_paper import apply_figure_settings


# In[75]:


beh_kwargs = dict(additional_behaviors=[f"eigenworm{i}" for i in range(4)])


# In[76]:


all_cca_classes, df_r_squared_melt, r_squared_per_row = calc_r_squared_for_all_projects(all_projects_gcamp, r_squared_kwargs=dict(n_components=[1, 2, 3]#, 4, 5]
                                                                                                                                 ), 
                                                                preprocess_traces_using_pca=True, truncate_traces_to_n_components=3, 
                                                                                        trace_kwargs=dict(use_paper_options=True),
                                                               melt=True, beh_kwargs=beh_kwargs)


# In[77]:


from wbfm.utils.external.utils_plotly import plotly_plot_mean_and_shading
# fig = px.box(df_r_squared_melt, x='Model Type', y='$R^2$', 
#              color='Model Type', color_discrete_map=plotly_paper_color_discrete_map())#, title="Reconstruction quality for single modes")
fig = px.box(df_r_squared_melt, x='n_components', color='Method', y='Variance Explained', 
             color_discrete_map=plotly_paper_color_discrete_map())#, title="Reconstruction quality for single modes")

apply_figure_settings(fig, width_factor=0.4, height_factor=0.2)
fig.update_yaxes(title='Neuronal Variance<br>Explained (cumulative)', 
                 range=[0, 1.1])#, showgrid=True, overwrite=True)
fig.update_xaxes(title='Number of components')
fig.update_layout(showlegend=True)

to_save = True
if to_save:
    fname = os.path.join(output_folder, 'top_mode_reconstruction_boxplot.png')
    fig.write_image(fname, scale=3)
    fname = fname.replace('.png', '.svg')
    fig.write_image(fname)

fig.show()


# In[ ]:





# # Latent space quality

# In[78]:


from wbfm.utils.visualization.utils_cca import calc_mode_correlation_for_all_projects
from wbfm.utils.external.utils_matplotlib import paired_boxplot_from_dataframes
from wbfm.utils.general.utils_paper import apply_figure_settings


# In[79]:


n_components = 3


# In[80]:


all_cca_classes3, df_mode_correlations, df_mode_correlations_binary = calc_mode_correlation_for_all_projects(all_projects_gcamp, correlation_kwargs=dict(n_components=n_components),
                                                                                                             preprocess_traces_using_pca=True, truncate_traces_to_n_components=3, 
                                                                                                             trace_kwargs=dict(use_paper_options=True),
                                                                                                            beh_kwargs=beh_kwargs)


# In[81]:


df_mode_correlations.index = np.arange(1, n_components+1)
df_mode_correlations_binary.index = np.arange(1, n_components+1)

df0 = df_mode_correlations.T.copy()
df1 = df_mode_correlations_binary.T.copy()

df0['Behavior type'] = 'Continuous'
df1['Behavior type'] = 'Discrete'
df_mode_combined = pd.concat([df0, df1])


# In[82]:


fig = px.box(df_mode_combined, color='Behavior type',
             color_discrete_map=plotly_paper_color_discrete_map())

fig.update_xaxes(title="Component")
fig.update_yaxes(title="Correlation of <br> CCA latent spaces")
fig.update_layout(showlegend=False)

apply_figure_settings(fig, width_factor=0.2, height_factor=0.2)

# fig.update_layout(
#     showlegend=True,
#     legend=dict(
#       yanchor="top",
#       y=1,
#       xanchor="right",
#       x=1.0
#     )
# )

fig.show()


fname = os.path.join(output_folder, 'paired_boxplot_latent_space_combined12345.png')
fig.write_image(fname, scale=7)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)


# In[ ]:





# In[83]:


all_dots = {i+1: {name: c.calc_mode_dot_product(i) for name, c in tqdm(all_cca_classes3.items())} for i in range(n_components)}
df_all_dots = pd.DataFrame(all_dots).melt(var_name='Component', value_name='PCA-CCA similarity')
df_all_dots['Comparison Method'] = 'CCA'


# In[84]:


all_dots_discrete = {i+1: {name: c.calc_mode_dot_product(i, binary_behaviors=True) for name, c in tqdm(all_cca_classes3.items())} for i in range(n_components)}
df_all_dots_discrete = pd.DataFrame(all_dots_discrete).melt(var_name='Component', value_name='PCA-CCA similarity')
df_all_dots_discrete['Comparison Method'] = 'CCA Discrete'


# In[85]:


df_all_dots = pd.concat([df_all_dots, df_all_dots_discrete])


# In[86]:


df_all_dots['PCA-CCA similarity'] = df_all_dots['PCA-CCA similarity'].abs()


# In[87]:


fig = px.box(df_all_dots, x='Component', y='PCA-CCA similarity', color='Comparison Method',
            color_discrete_map=plotly_paper_color_discrete_map())
# fig.update_traces(marker=dict(color=plotly_paper_color_discrete_map()['PCA']))
# fig.update_xaxes(title='Component')
fig.update_yaxes(title='PCA-CCA similarity', range=[0, 1.1])
fig.update_layout(showlegend=False)

apply_figure_settings(fig, width_factor=0.2, height_factor=0.2)

fig.show()

to_save = True
if to_save:
    fname = os.path.join(output_folder, 'mode_dot_product.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Also calculate variance explained of behavior time series

# In[88]:


from wbfm.utils.visualization.utils_cca import calc_r_squared_for_all_projects
from wbfm.utils.general.utils_paper import plotly_paper_color_discrete_map
from wbfm.utils.general.utils_paper import apply_figure_settings


# In[89]:


beh_kwargs = dict(additional_behaviors=[f"eigenworm{i}" for i in range(4)])


# In[90]:


all_cca_classes_beh, df_r_squared_melt_beh, all_r_squared_per_row_beh = calc_r_squared_for_all_projects(all_projects_gcamp, 
                                                                                                        r_squared_kwargs=dict(n_components=[1, 2, 3], use_behavior=True, DEBUG=False), 
                                                                                                        preprocess_traces_using_pca=True, truncate_traces_to_n_components=3, 
                                                                                                        trace_kwargs=dict(use_paper_options=True),
                                                                                                        beh_kwargs=beh_kwargs,
                                                                                                        melt=True)


# In[131]:


df = all_r_squared_per_row_beh.copy()
df['Behavior Variable'] = df['Behavior Variable'].map(lambda k: {f"eigenworm{i}": f"Eigenworm {i+1}" for i in range(4)}.get(k, k))

cmap = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
cmap.pop(2)  # Remove the horrible neon green

for method in df['Method'].unique():
    _df = df[df['Method']==method]
    cols = ['Behavior Variable', 'Components']
    vals = 'Cumulative Variance explained'
    df_sorted = _df.groupby(cols)[vals].mean().reset_index().sort_values(by=['Components', vals], ascending=False)
    category_orders = list(df_sorted[df_sorted['Components']==3]['Behavior Variable'].dropna())
    # Actually sort the dataframe by this order
    _df['Behavior Variable'] = pd.Categorical(
        _df['Behavior Variable'], 
        categories=category_orders, 
        ordered=True
    )
    
    _df = _df.sort_values('Behavior Variable')
    
    fig = px.box(_df, y='Cumulative Variance explained', color='Components', x='Behavior Variable', facet_col='Method', color_discrete_sequence=cmap,
                #category_orders={'Behavior Variable': category_orders}
                )
    show_legend = (method=='CCA')
    fig.update_yaxes(range=[-0.01, 0.8])
    fig.update_layout(showlegend=show_legend)
    if show_legend:  
        fig.update_layout(
            legend=dict(
              yanchor="middle",
              y=0.25,
              xanchor="left",
              x=1.01
            )
        )
    else:
        fig.update_yaxes(title='')
    # fig.update_traces(boxpoints=False)
    fig.update_xaxes(title=f"Method: {method}")#, zeroline=True, zerolinewidth=2, zerolinecolor='black')
    apply_figure_settings(fig, width_factor=0.55 if show_legend else 0.4, height_factor=0.3)
    
    fig.update_layout(
    xaxis=dict(
            showline=True, linecolor="black"
        ), overwrite=True
    )
    to_save = True
    if to_save:
        fname = os.path.join(output_folder, f'behavior_variance_explained_{method}.png')
        fig.write_image(fname, scale=5)
        fname = fname.replace('.png', '.svg')
        fig.write_image(fname)

    fig.show()


# In[127]:





# ## Alternative visualization: Flavell-style boxes in columns

# In[92]:


all_r_squared_per_row_beh.head()


# In[93]:


df.groupby(['Method', 'Behavior Variable', 'Components'])['Cumulative Variance explained'].mean()


# In[94]:


# df_multi = df.set_index(['Method', 'Behavior Variable', 'Components'])
# df_multi.to_dict()['Cumulative Variance explained']


# In[95]:


from wbfm.utils.general.utils_paper import plot_foldchange_boxes


# In[154]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = all_r_squared_per_row_beh.copy()
df = df[df['Method'].isin(['PCA', 'CCA'])]
df = df[df['Components'].isin([1,2,3])]

df['Behavior Variable'] = df['Behavior Variable'].map(lambda k: {f"eigenworm{i}": f"Eigenworm {i+1}" for i in range(4)}.get(k, k))

# Define behaviors and groups
behavior_col = 'Method'
behaviors = df[behavior_col].unique()

groups = {}
groups = {'Discrete': ['Fwd-Bwd State', 'Dorsal turn', 'Ventral turn', 'Self-collision', 'Pause']}
_low_level = ['Ventral head', 'Dorsal head',]
_low_level.extend([f"Eigenworm {i+1}" for i in range(4)])
groups['Low level'] = _low_level
groups['High level'] = ['Velocity', 'Dorsal body', 'Ventral body']

cols = ['Method', 'Behavior Variable', 'Components']
vals = 'Cumulative Variance explained'
df_sorted = df.groupby(cols)[vals].mean().reset_index().sort_values(by=['Method', 'Components', vals], ascending=True)
neuron_order = list(df_sorted[(df_sorted['Method']=='PCA')*(df_sorted['Components']==3)]['Behavior Variable'])
# print(neuron_order)

# Call the function with default fold-change mode
ax, fig, ordered_neurons = plot_foldchange_boxes(
    df=df,
    behavior_col=behavior_col,
    groups=groups,
    subtitle_behavior_col='Components',
    rows_col='Behavior Variable',
    value_col='Cumulative Variance explained',
    figsize=(8, 6),
    row_vspace=0.5,
    behavior_hspace=3.0,
    cmap='Greens',
    add_text=False,
    vmax=0.6,
    neuron_order=neuron_order,
    DEBUG=False
)

plt.tight_layout()
apply_figure_settings(fig, width_factor=0.4, height_factor=0.25, plotly_not_matplotlib=False)

fname = os.path.join(output_folder, 'colored_box_beh_variance_explained.png')
plt.savefig(fname, dpi=100)
plt.show()
# fig.write_image(fname, scale=7)
# fname = Path(fname).with_suffix('.svg')
# fig.write_image(fname)


# In[155]:


# SAME but discrete only
df = all_r_squared_per_row_beh.copy()
df = df[df['Method'].isin(['CCA Discrete'])]
df = df[df['Components'].isin([1,2,3])]

# Define behaviors and groups
behavior_col = 'Method'
behaviors = df[behavior_col].unique()

groups = {}
groups = {'Discrete': ['Fwd-Bwd State', 'Dorsal turn', 'Ventral turn', 'Self-collision', 'Pause']}
groups['Low level'] = ['eigenworm3',
                       'eigenworm2',
                       'eigenworm1',
                       'eigenworm0',
                       'Ventral head',
                       'Dorsal head',]
groups['High level'] = ['Velocity', 'Dorsal body', 'Ventral body']

cols = ['Method', 'Behavior Variable', 'Components']
vals = 'Cumulative Variance explained'
df_sorted = df.groupby(cols)[vals].mean().reset_index().sort_values(by=['Method', 'Components', vals], ascending=True)
neuron_order = list(df_sorted[(df_sorted['Method']=='CCA Discrete')*(df_sorted['Components']==3)]['Behavior Variable'])
print(neuron_order)


# Call the function with default fold-change mode
ax, fig, ordered_neurons = plot_foldchange_boxes(
    df=df,
    behavior_col=behavior_col,
    groups=groups,
    subtitle_behavior_col='Components',
    rows_col='Behavior Variable',
    value_col='Cumulative Variance explained',
    figsize=(8, 6),
    row_vspace=0.25,
    behavior_hspace=2.0,
    cmap='Greens',
    add_text=False,
    vmax=0.6,
    neuron_order=neuron_order,
    DEBUG=False
)

plt.tight_layout()
apply_figure_settings(fig, width_factor=0.25, height_factor=0.4, plotly_not_matplotlib=False)

fname = os.path.join(output_folder, 'colored_box_beh_variance_explained_binary.png')
plt.savefig(fname, dpi=100)
plt.show()
# fig.write_image(fname, scale=7)
# fname = Path(fname).with_suffix('.svg')
# fig.write_image(fname)


# ## Gut check: reconstruction of other curvatures from eigenworms

# In[98]:


from sklearn.linear_model import LinearRegression


# In[99]:


def _calc_r_squared(X, X_r_recon):
    residual_variance = (X - X_r_recon).var().sum()
    total_variance = X.var().sum()
    r_squared = 1 - residual_variance / total_variance
    return r_squared


# In[100]:


all_beh_var_explained = []
beh_to_explain = ['signed_middle_body_speed', 'ventral_only_body_curvature', 'ventral_only_head_curvature',
                  'dorsal_only_body_curvature', 'dorsal_only_head_curvature']
beh_to_use = [f"eigenworm{i}" for i in range(4)]
binary_behaviors = False

for name, c in all_cca_classes_beh.items():
    one_dataset_var_explained = {}
    _df_beh = c._get_beh_df(binary_behaviors=binary_behaviors, raw_not_truncated=True).copy()
    X_r = _df_beh[beh_to_use]
    for beh in beh_to_explain:
        X = _df_beh[beh]
        reg = LinearRegression().fit(X_r, X)
        X_r_recon = reg.predict(X_r)
        one_dataset_var_explained[beh] = _calc_r_squared(X, X_r_recon)
    all_beh_var_explained.append(pd.DataFrame.from_dict(one_dataset_var_explained, orient='index', columns=['r_squared']).reset_index().assign(dataset=name))
    


# In[101]:


df_beh_r_squared = pd.concat(all_beh_var_explained)
df_beh_r_squared


# In[102]:


px.box(df_beh_r_squared, x='index', y='r_squared')


# ## Same but shaded lines

# In[103]:


from wbfm.utils.external.utils_plotly import plotly_plot_mean_and_shading
import plotly


# In[104]:


# method = 'CCA'

# _df = df[df['Method']==method]

# cmap = dict(zip(_df['Behavior Variable'].unique(), px.colors.qualitative.D3))

# fig = plotly_plot_mean_and_shading(_df, x='Components', y='Cumulative Variance explained', color='Behavior Variable',
#                                   shade_style='quantile', cmap=cmap)
# apply_figure_settings(fig, width_factor=0.8, height_factor=0.5)
# fig.show()
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # SUPP

# In[ ]:





# ## Variance explained per neuron (cumulative plot)

# In[105]:


from wbfm.utils.visualization.utils_cca import calc_r_squared_for_all_projects
from wbfm.utils.general.utils_paper import plotly_paper_color_discrete_map
from wbfm.utils.general.utils_paper import apply_figure_settings


# In[106]:


# _, df_r_squared, r_squared_per_row = calc_r_squared_for_all_projects(all_projects_gcamp, r_squared_kwargs=dict(n_components=[1, 2, 3]), 
#                                                                 preprocess_traces_using_pca=True, truncate_traces_to_n_components=3, trace_kwargs=dict(use_paper_options=True),
#                                                                melt=True)


# In[107]:


_df = r_squared_per_row.rename(columns={'Behavior Variable': 'Neuron Name'})[r_squared_per_row['Components'] == 2]
_df


# In[108]:


df_var_exp = _df.copy()
df_var_exp_hist = _df.copy()

# Get counts of neurons in each bin
bins = np.linspace(0, 1, 51)
func = lambda Z: np.cumsum(np.histogram(Z, bins=bins)[0])
df_var_exp_hist = df_var_exp_hist.groupby(['Dataset Name', 'Method'])['Cumulative Variance explained'].apply(func)

# Explode to long form
long_vars = (df_var_exp_hist / df_var_exp_hist.apply(max)).reset_index().explode('Cumulative Variance explained').reset_index(drop=True)

# Just remake the bins
fraction_count = df_var_exp_hist.apply(lambda x: bins[1:]).reset_index().explode('Cumulative Variance explained').reset_index(drop=True).rename(columns={'Cumulative Variance explained': 'bins'})
long_vars['bins'] = fraction_count['bins']

long_vars


# In[109]:


from wbfm.utils.external.utils_plotly import plotly_plot_mean_and_shading

opt = dict(x='bins', y='Cumulative Variance explained', color='Method', 
           cmap=plotly_paper_color_discrete_map()
          )

fig = None
g = ['CCA', 'CCA Discrete']
df_subset = long_vars[(long_vars['Method'].isin(g))]
fig = plotly_plot_mean_and_shading(df_subset, fig=fig, **opt,
                                   x_intersection_annotation=0.5)

fig.update_xaxes(title='Var. explained (fraction)', range=[0, 1.05])
fig.update_yaxes(title='Fraction of neurons <br> (cumulative)', range=[0, 1.05])
fig.update_layout(
        showlegend=False,
        legend=dict(
            title='Mode',
          yanchor="middle",
          y=0.25,
          xanchor="left",
          x=0.6
        )
    )# fig.update_traces(line=dict(color=plotly_paper_color_discrete_map()['PCA']))
# fig.update_traces(name='1 + 2', selector=dict(name='2'))

apply_figure_settings(fig, width_factor=0.3, height_factor=0.2)

fig.show()

to_save = True
if to_save:
    output_foldername = 'fig2'
    fname = os.path.join(output_foldername, 'variance_explained_by_cca_cumulative.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)


# In[110]:


plotly_paper_color_discrete_map()['PCA']


# In[111]:


long_vars.groupby(['Dataset Name', 'Method']).apply(np.cumsum)


# In[112]:


df_var_exp_hist


# In[113]:


df_var_exp_hist.reset_index().explode('Cumulative Variance explained').groupby(['Dataset Name', 'Method']).cumcount()


# In[114]:


df_var_exp_hist.reset_index().explode('Cumulative Variance explained')


# # Get the neural and behavioral weights across datasets

# In[116]:


from wbfm.utils.visualization.utils_cca import calc_cca_weights_for_all_projects
import plotly.express as px
from wbfm.utils.general.utils_paper import apply_figure_settings, plotly_paper_color_discrete_map
from wbfm.utils.general.utils_hardcoded import neurons_with_confident_ids
from wbfm.utils.visualization.utils_plot_traces import add_p_value_annotation
output_folder = 'fig2'


# In[117]:


all_cca_classes1, df_weights1, df_weights_binary1 = calc_cca_weights_for_all_projects(all_projects_gcamp, which_mode=0, min_datasets_present=6,
                                                                                       weights_kwargs=dict(n_components=2),
                                                                                                             preprocess_traces_using_pca=True, truncate_traces_to_n_components=3,
                                                                                                            preprocess_behavior_using_pca=True,
                                                                                    combine_left_and_right=True, beh_kwargs=beh_kwargs,
                                                                                   trace_kwargs=dict(use_paper_options=True))


# In[118]:


# Both modes together
df_both1 = df_weights1.reset_index().melt(id_vars='index')
df_both1['Behavior Type'] = 'Continuous'
df_both1_binary = df_weights_binary1.reset_index().melt(id_vars='index')
df_both1_binary['Behavior Type'] = 'Discrete'
df_both1 = pd.concat([df_both1, df_both1_binary])
df_both1.columns = ['Dataset Name', 'Neuron', 'Weight', 'Behavior Type']
# df_both1


# In[134]:


from wbfm.utils.general.utils_hardcoded import neurons_with_confident_ids
neurons_to_plot = neurons_with_confident_ids(combine_left_right=True)
fig = px.box(df_both1[df_both1['Neuron'].isin(neurons_to_plot)], x='Neuron', y='Weight', color='Behavior Type',
             hover_data=['Dataset Name'],
             color_discrete_map=plotly_paper_color_discrete_map())

apply_figure_settings(fig, width_factor=1.0, height_factor=0.2, plotly_not_matplotlib=True)
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="CCA Weight<br>(Component 1)")
# fig.update_xaxes(title="", tickfont_size=12)

fig.update_layout(legend=dict(
    yanchor="top",
    y=1.05,
    xanchor="left",
    x=0.8
))
fig.show()

fname = os.path.join(output_folder, 'paired_boxplot_neural_weights_BOTH.png')
fig.write_image(fname, scale=3)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)


# In[135]:


add_p_value_annotation(fig, x_label='all', show_only_stars=True)
fig.show()


# ## SUPP: Same but for mode 2

# In[136]:


all_cca_classes2, df_weights2, df_weights_binary2 = calc_cca_weights_for_all_projects(all_projects_gcamp, which_mode=1, min_datasets_present=6,
                                                                                       weights_kwargs=dict(n_components=3),
                                                                                    correct_sign_using_top_weight=True,
                                                                                                             preprocess_traces_using_pca=True, truncate_traces_to_n_components=3,
                                                                                                            preprocess_behavior_using_pca=True,
                                                                                    combine_left_and_right=True, beh_kwargs=beh_kwargs,
                                                                                   trace_kwargs=dict(use_paper_options=True))


# In[137]:


df_weights2 = df_weights2[[c for c in df_weights2.columns if c in neurons_with_confident_ids(combine_left_right=True)]]

# fig = px.box(df_weights2, color_discrete_sequence=[plotly_paper_color_discrete_map()['CCA']])

# apply_figure_settings(fig, width_factor=1.0, height_factor=0.2, plotly_not_matplotlib=True)
# fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="Weight <br> (mode 2)")
# fig.update_xaxes(title="")
# fig.show()

# to_save = True
# if to_save:
#     fname = os.path.join(output_folder, 'paired_boxplot_neural_weights2.png')
#     fig.write_image(fname, scale=3)
#     fname = Path(fname).with_suffix('.svg')
#     fig.write_image(fname)


# In[138]:


df_weights_binary2 = df_weights_binary2[[c for c in df_weights_binary2.columns if c in neurons_with_confident_ids(combine_left_right=True)]]

# fig = px.box(df_weights_binary2, color_discrete_sequence=[plotly_paper_color_discrete_map()['Discrete']])#, title="CCA weights of mode 2 across recordings (binary)")

# apply_figure_settings(fig, width_factor=1.0, height_factor=0.2, plotly_not_matplotlib=True)
# fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="Weight <br> (discrete <br> mode 2)")
# fig.update_xaxes(title="")
# fig.show()

# fname = os.path.join(output_folder, 'paired_boxplot_neural_weights_binary2.png')
# fig.write_image(fname, scale=3)
# fname = Path(fname).with_suffix('.svg')
# fig.write_image(fname)


# In[140]:


from wbfm.utils.general.utils_hardcoded import neurons_with_confident_ids
neurons_to_plot = neurons_with_confident_ids(combine_left_right=True)

# Both modes together
df_both2 = df_weights2.reset_index().melt(id_vars='index')
df_both2['Behavior Type'] = 'Continuous'
df_both2_binary = df_weights_binary2.reset_index().melt(id_vars='index')
df_both2_binary['Behavior Type'] = 'Discrete'
df_both2 = pd.concat([df_both2, df_both2_binary])
df_both2.columns = ['Dataset Name', 'Neuron', 'Weight', 'Behavior Type']
# df_both1

fig = px.box(df_both2[df_both2['Neuron'].isin(neurons_to_plot)], x='Neuron', y='Weight', color='Behavior Type',
             hover_data=['Dataset Name'],
             color_discrete_map=plotly_paper_color_discrete_map())

apply_figure_settings(fig, width_factor=1.0, height_factor=0.3, plotly_not_matplotlib=True)
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="CCA Weight (mode 2)")
fig.update_xaxes(title="", tickfont_size=12)

fig.update_layout(legend=dict(
    yanchor="top",
    y=1.0,
    xanchor="left",
    x=0.5
))
fig.show()

fname = os.path.join(output_folder, 'paired_boxplot_neural_weights2_BOTH.png')
fig.write_image(fname, scale=3)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)


# ## SUPP: Same but for mode 3

# In[141]:


all_cca_classes3, df_weights3, df_weights_binary3 = calc_cca_weights_for_all_projects(all_projects_gcamp, which_mode=2, min_datasets_present=5,
                                                                                       weights_kwargs=dict(n_components=3),
                                                                                      combine_left_and_right=True, beh_kwargs=beh_kwargs,
                                                                                    correct_sign_using_top_weight=True,
                                                                                                             preprocess_traces_using_pca=True, truncate_traces_to_n_components=3,
                                                                                                            preprocess_behavior_using_pca=True,
                                                                                   trace_kwargs=dict(use_paper_options=True))


# In[142]:


# df_weights3 = df_weights3[[c for c in df_weights3.columns if c in neurons_with_confident_ids(combine_left_right=True)]]

# fig = px.box(df_weights3, color_discrete_sequence=[plotly_paper_color_discrete_map()['CCA']])#, title="CCA weights of mode 3 across recordings")
# apply_figure_settings(fig, width_factor=1.0, height_factor=0.2, plotly_not_matplotlib=True)
# fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="Weight <br> (mode 3)")
# fig.update_xaxes(title="")
# fig.show()

# fname = os.path.join(output_folder, 'paired_boxplot_neural_weights3.png')
# fig.write_image(fname, scale=3)
# fname = Path(fname).with_suffix('.svg')
# fig.write_image(fname)


# In[143]:


# fig = px.box(df_weights_binary, title="CCA weights of mode 3 across recordings (binary)")
# fig.show()

# fname = os.path.join(output_folder, 'paired_boxplot_neural_weights_binary3.png')
# fig.write_image(fname)


# ## Same but for behavior weights

# In[144]:


from wbfm.utils.general.utils_paper import behavior_name_mapping


# In[145]:


all_cca_classes_beh1, df_weights_beh1, df_weights_binary_beh1 = calc_cca_weights_for_all_projects(all_projects_gcamp, which_mode=0, min_datasets_present=5,
                                                                                       weights_kwargs=dict(n_components=2), neural_not_behavioral=False,
                                                                                                             preprocess_traces_using_pca=True, truncate_traces_to_n_components=3,
                                                                                                            preprocess_behavior_using_pca=False,
                                                                                   trace_kwargs=dict(use_paper_options=True), beh_kwargs=beh_kwargs)


# In[146]:


df_weights_beh1.rename(columns=behavior_name_mapping(shorten=True)).head()


# In[147]:


fig = px.box(df_weights_beh1.rename(columns=behavior_name_mapping(shorten=True)), color_discrete_sequence=[plotly_paper_color_discrete_map()['CCA']])
apply_figure_settings(fig, width_factor=0.25, height_factor=0.25, plotly_not_matplotlib=True)
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="CCA Weight<br>(Component 1)",
                #range=[-1.1, 1.1]
                )
# fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="Weight  <br> (mode 1)")
fig.update_xaxes(title="", tickfont_size=12)
fig.show()

fname = os.path.join(output_folder, 'paired_boxplot_beh_weights.png')
fig.write_image(fname, scale=3)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)


# In[148]:


cmap = px.colors.qualitative.Plotly
# df_weights_binary_beh1['color'] = cmap[1]

fig = px.box(df_weights_binary_beh1.rename(columns=behavior_name_mapping(shorten=True)), color_discrete_sequence=[plotly_paper_color_discrete_map()['Discrete']])
fig.update_layout(showlegend=False)
apply_figure_settings(fig, width_factor=0.25, height_factor=0.25, plotly_not_matplotlib=True)
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="Discrete CCA<br>Weight<br>(Component 1)",
                range=[-0.2, 1.1])
fig.update_xaxes(title="", tickfont_size=12)
fig.show()

fname = os.path.join(output_folder, 'paired_boxplot_beh_weights_binary.png')
fig.write_image(fname, scale=3)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)


# ## Supp: behavior

# In[149]:


all_cca_classes_beh2, df_weights_beh2, df_weights_binary_beh2 = calc_cca_weights_for_all_projects(all_projects_gcamp, which_mode=1, min_datasets_present=5,
                                                                                       weights_kwargs=dict(n_components=3), neural_not_behavioral=False,
                                                                                                             preprocess_traces_using_pca=True, truncate_traces_to_n_components=3,
                                                                                                            preprocess_behavior_using_pca=False,
                                                                                   trace_kwargs=dict(use_paper_options=True), beh_kwargs=beh_kwargs)


# In[150]:


fig = px.box(df_weights_beh2.rename(columns=behavior_name_mapping(shorten=True)), color_discrete_sequence=[plotly_paper_color_discrete_map()['CCA']])
apply_figure_settings(fig, width_factor=0.25, height_factor=0.3, plotly_not_matplotlib=True)
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="CCA Weight <br> (mode 2)")
fig.update_xaxes(title="")
fig.update_xaxes(title="", tickfont_size=12)

fig.show()

fname = os.path.join(output_folder, 'paired_boxplot_beh_weights2.png')
fig.write_image(fname, scale=3)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)


# In[151]:


fig = px.box(df_weights_binary_beh2.rename(columns=behavior_name_mapping(shorten=True)), color_discrete_sequence=[plotly_paper_color_discrete_map()['Discrete']])

apply_figure_settings(fig, width_factor=0.25, height_factor=0.3, plotly_not_matplotlib=True)
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="CCA Weight <br> (discrete mode 2)")
fig.update_xaxes(title="")
fig.show()

fname = os.path.join(output_folder, 'paired_boxplot_beh_weights_binary2.png')
fig.write_image(fname, scale=3)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)


# In[152]:


all_cca_classes_beh3, df_weights_beh3, df_weights_binary_beh3 = calc_cca_weights_for_all_projects(all_projects_gcamp, which_mode=2, min_datasets_present=5,
                                                                                       weights_kwargs=dict(n_components=3), neural_not_behavioral=False,
                                                                                                             preprocess_traces_using_pca=True, truncate_traces_to_n_components=3,
                                                                                                            preprocess_behavior_using_pca=True,
                                                                                   trace_kwargs=dict(use_paper_options=True), beh_kwargs=beh_kwargs)


# In[153]:


fig = px.box(df_weights_beh3.rename(columns=behavior_name_mapping(shorten=True)), color_discrete_sequence=[plotly_paper_color_discrete_map()['CCA']])
apply_figure_settings(fig, width_factor=0.25, height_factor=0.2, plotly_not_matplotlib=True)
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", title="Weight <br> (mode 3)")
fig.update_xaxes(title="")
fig.show()

fname = os.path.join(output_folder, 'paired_boxplot_beh_weights3.png')
fig.write_image(fname, scale=3)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # SCRATCH

# # Correlation matrix for behaviors

# In[ ]:


# df_beh = all_cca_classes_beh['2022-11-23_worm10']._df_beh
# fig = px.imshow(df_beh.corr(), width=1000, height=1000)
# fig.show()


# # Interactive GUI

# In[ ]:


# # Same GUI but for all datasets
# from ipywidgets import interact

# def f(dataset_name):
#     cca_plotter = all_cca_classes_beh[dataset_name]
#     fig = cca_plotter.visualize_modes_and_weights(binary_behaviors=False, n_components=3)
    
# interact(f, dataset_name=list(all_cca_classes_beh.keys()))


# In[ ]:


# cca_plotter.visualize_modes_and_weights(binary_behaviors=False, n_components=3)


# # Debugging

# In[ ]:


from wbfm.utils.general.utils_filenames import resolve_mounted_path_in_current_os
from wbfm.utils.general.postures.centerline_classes import WormFullVideoPosture


# In[ ]:


# fname = "/lisc/data/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-11-23_spacer_7b_2per_agar/2022-11-23_worm10"
fname = "/lisc/data/scratch/neurobiology/zimmer/brenner/wbfm_projects/analyze/freely_moving_wt/2024-01-30_17-00_wt_worm2-2024-01-30"
project_data_test = ProjectData.load_final_project_data_from_config(fname)


# In[ ]:


project_data_test.worm_posture_class


# In[ ]:


worm = WormFullVideoPosture.load_from_project(project_data_test, DEBUG=True)


# In[ ]:


project_data_test.project_config.get_behavior_raw_parent_folder_from_red_fname()


# In[ ]:


path, flag = project_data_test.project_config.get_raw_data_fname(True)


# In[ ]:


resolve_mounted_path_in_current_os(str(path), allow_only_parent_to_exist=True, verbose=2)


# In[ ]:




