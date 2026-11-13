#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm
import os 
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go


# In[2]:


from wbfm.utils.general.utils_hardcoded import get_hierarchical_modeling_dir

fname = os.path.join(get_hierarchical_modeling_dir(), 'data.h5')
print(fname)
Xy = pd.read_hdf(fname)

fname = os.path.join(get_hierarchical_modeling_dir(gfp=True), 'data.h5')
print(fname)
Xy_gfp = pd.read_hdf(fname)


# In[3]:


# from wbfm.utils.external.utils_pandas import get_dataframe_for_single_neuron
# get_dataframe_for_single_neuron(Xy_gfp, 'VB02', residual_mode='discrete')


# # All together one big function

# In[4]:


from wbfm.utils.general.utils_paper import apply_figure_settings
from wbfm.utils.external.utils_plotly import get_nonoverlapping_text_positions
from wbfm.utils.general.utils_paper import apply_figure_settings, plotly_paper_color_discrete_map, data_type_name_mapping, package_bayesian_df_for_plot


# In[5]:


from wbfm.utils.general.utils_paper_revisions_bayes import calculate_all_bayesian_model_data
parent_folder_kwargs = dict(suffix='avb_hiscl_control')
df_params, df_params_full, y_max_gfp, x_max_gfp = calculate_all_bayesian_model_data(suffix='_pca_global', recalculate_sigmoid=False, parent_folder_kwargs=parent_folder_kwargs)


# In[6]:


df_params


# In[7]:



idx = df_params['Dataset Type'] == 'Freely Moving (GCaMP, residual)'
_df = df_params[idx]
mapping = dict(zip(_df['neuron_name'], _df['Neuron Type']))
df_params_full['Neuron Type'] = df_params_full['neuron_name'].map(mapping)


# In[8]:


df_params[df_params['neuron_name'] == 'URXL']


# In[9]:


from wbfm.utils.general.utils_paper import plot_bayesian_model_comparison, plotly_paper_color_discrete_map
from wbfm.utils.external.utils_plotly import get_nonoverlapping_text_positions


# In[16]:



cmap = px.colors.qualitative.D3.copy()

y, x = 'Relative Hierarchy Score (raw)', 'Behavior Score (raw)'
_df = df_params[df_params['Dataset Type'] == 'Freely Moving (GFP, residual)']
x_max_gfp = _df[x].max()
y_max_gfp = _df[y].max()

# df_params['Category_new'] = 'PC1 not significant'
df_params['Category_new'] = 'Not above artifacts'
idx_beh = df_params['Behavior Score'] > x_max_gfp
df_params.loc[idx_beh, 'Category_new'] = 'Behavior is above GFP'
idx = df_params['Relative Hierarchy Score'] > y_max_gfp
# df_params.loc[idx, 'Category_new'] = 'Hierarchy is above GFP'
idx_hierarchy = np.logical_and(df_params['hyper_pca0_p_value_is_significant'] == True, df_params['Relative Hierarchy Score'] > y_max_gfp)
df_params.loc[idx_hierarchy, 'Category_new'] = 'Hierarchy > GFP and PC1 is significant'

color_discrete_map = {True: cmap[0], 'PC1 is significant': cmap[0], 'Hierarchy is above GFP': cmap[0], 'Hierarchy > GFP and PC1 is significant':cmap[0],
                     False: cmap[7], 'PC1 not significant': cmap[7],'Not above artifacts': cmap[7],
                     'PC1 not significant, but behavior is good': cmap[2], 'Behavior is above GFP': cmap[2]}

smap = {'Hierarchy > GFP and PC1 is significant': 'diamond', True:'diamond', 'True':'diamond',
        'Not above artifacts': 'circle', False: 'circle', 'False': 'circle',
        'Behavior is above GFP': 'circle'}

# Neurons that will be plotted in the end
idx_final = np.logical_or(idx_hierarchy, idx_beh)

smap = {'Hierarchy > GFP and PC1 is significant': 'diamond', True:'diamond', 'True':'diamond', "PC1 coefficient significant": 'diamond',
        'Not above artifacts': 'circle', False: 'circle', 'False': 'circle', "PC1 coefficient not significant": 'circle',
        'Behavior is above GFP': 'circle',
       "GFP": 'square'}

# Neurons that will be plotted in the end
idx_final = np.logical_or(idx_hierarchy, idx_beh)

# Give a new value for hyper_pca0_p_value_is_significant, that combines with GFP
df_params['Model parameter significance'] = "GFP"
idx_gcamp = df_params['Dataset Type'] == "Freely Moving (GCaMP, residual)"
idx = np.logical_and(idx_gcamp, df_params['hyper_pca0_p_value_is_significant'])
df_params.loc[idx, 'Model parameter significance'] = "PC1 coefficient significant"
idx = np.logical_and(idx_gcamp, ~df_params['hyper_pca0_p_value_is_significant'])
df_params.loc[idx, 'Model parameter significance'] = "PC1 coefficient not significant"


# In[17]:


# y, x = 'Relative Hierarchy Score', 'Behavior Score'
y, x = 'Relative Hierarchy Score (raw)', 'Behavior Score (raw)'


text = df_params['text_raw'].copy()
text[:] = ''
text[idx_final] = df_params['text_raw'].copy()[idx_final]

fig = px.scatter(df_params, 
                 y=df_params[y] + 1, x=df_params[x] + 1,
                 log_y=True, log_x=True,
                 # color='hyper_pca0_p_value_is_significant',
                 # color='Dataset Type',
                 color='Category_new',
                 color_discrete_map=color_discrete_map,
                 # category_orders={'Category':['PC1 not significant', 'PC1 not significant, but behavior is good', 'PC1 is significant']},
                 symbol='Model parameter significance',
                 symbol_map=smap,
                 size_max=50,
                 hover_data=['neuron_name', 'Dataset Type'],
                 # text=text,
                )
opt = dict(line=dict(dash='dot'), col=1)
_df = df_params[df_params['Dataset Type'] == 'Freely Moving (GFP, residual)']
fig.add_hline(y=_df[y].max(), **opt)
fig.add_vline(x=_df[x].max(), **opt)


# get_nonoverlapping_text_positions(df_params[x], df_params[y], text, fig, weight=10, k=10, add_nodes_with_no_text=False,
#                                  x_range=[-5, 40], y_range=[-5, 40], 
#                                   size=12
#                                  )

# Modify legend to only show colors, not symbols
fig.for_each_trace(lambda t: t.update(showlegend=False))
colors = df_params['Category_new'].unique()
symbols = df_params['Model parameter significance'].unique()
cmap = color_discrete_map

for i, c in enumerate(colors):
    # Color legend entries
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(color=cmap[c], symbol="circle"),
        name=f"{c}",
        legendgroup="color",
        legendgrouptitle_text="GFP comparison threshold" if i == 0 else None,
        showlegend=True
    ))
for i, s in enumerate(symbols):
    # if np.isnan(s):
    #     continue
    # Symbol legend entries
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(color="black", symbol=smap[s]),
        name=f"{s}",
        legendgroup="symbol",
        legendgrouptitle_text="PC1 coefficient significance" if i == 0 else None,
        showlegend=True
    ))

    

fig.update_layout(
                  showlegend=True, 
                  legend=dict(
        yanchor="top",
        y=1.9,
        xanchor="left",
        x=-0.0,
                      title="",
                      bordercolor="Black", borderwidth=2, #bgcolor = 'white'
    )
                 )

fig.update_traces(
    marker=dict(
        size=10,
        line=dict(
            width=2,
            color="black"
        )
    )
)
# z-order
# Put a specific trace on top
# fig.data = tuple(
#     sorted(fig.data, key=lambda t: t.name != "PC1 not significant")
# )

apply_figure_settings(fig, width_factor=0.6, height_factor=0.3)
fig.show()


to_save = True
if to_save:
    fname = os.path.join("fig5", 'model_improvement-avb_hiscl_control.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)
    fname = Path(fname).with_suffix('.html')
    fig.write_html(fname)
    


# In[ ]:


# Still sort by median, conditional on whether the beta parameter is significant
_df = df_params[idx_final].copy()

only_beh_neurons = _df[_df['Category_new']== 'Behavior is above GFP']['neuron_name'].values
idx_beta_sorted = _df['log_hyper_beta'].sort_values().index
neurons_beta_sorted = _df.loc[idx_beta_sorted, 'neuron_name'].reset_index(drop=True)
map_beta_sorted = dict(zip(neurons_beta_sorted.values, neurons_beta_sorted.index))

idx_gamma_sorted = _df['log_amplitude_mu'].sort_values().index
neurons_gamma_sorted = _df.loc[idx_gamma_sorted, 'neuron_name'].reset_index(drop=True)
map_gamma_sorted = dict(zip(neurons_gamma_sorted.values, neurons_gamma_sorted.index))

all_neurons = neurons_gamma_sorted.values
rank = {neuron: (map_gamma_sorted[neuron]+100 if neuron in only_beh_neurons else map_beta_sorted[neuron]) for neuron in all_neurons }

neurons_final = df_params.loc[idx_final, 'neuron_name']
idx_final_neurons = df_params_full['neuron_name'].isin(neurons_final)
idx_final_vals = df_params_full['parameter'].isin(['log_amplitude_mu', 'log_hyper_beta'])
idx = np.logical_and(idx_final_vals, idx_final_neurons)
_df = df_params_full[idx].copy().sort_values(by="neuron_name", key=lambda x: x.map(rank))

# Build columns myself
x = np.zeros(len(_df), dtype=np.dtypes.StringDType)
y = np.zeros_like(x, dtype=float)
color = np.zeros_like(y, dtype=np.dtypes.StringDType)

idx_mu = _df['parameter'] == 'log_amplitude_mu'
n = np.sum(idx_mu)
y[idx_mu] = np.exp(_df.loc[idx_mu, 'value'])
x[idx_mu] = _df.loc[idx_mu, 'neuron_name'].values
color[idx_mu] = [px.colors.qualitative.D3[2]] * n

idx_beta = _df['parameter'] == 'log_hyper_beta'
n = np.sum(idx_beta)
_x = _df.loc[idx_beta, 'neuron_name'].values
x[idx_beta] = _x
# Remove values corresponding to neurons that are not hierarchical
idx_only_beh = pd.Series(_x).isin(only_beh_neurons).values
_y = np.exp(_df.loc[idx_beta, 'value'])
_y_to_add = _y.copy()
_y[idx_only_beh] = np.nan
y[idx_beta] = _y
color[idx_beta] = [px.colors.qualitative.D3[0]] * n
# Also: add 0.5 * beta to the behavior-only model
# subset = y[idx_beta]
# subset[~idx_only_beh] = subset[~idx_only_beh] + 0.5 * _y[~idx_only_beh]
# y[idx_beta] = subset


fig = px.box(x=x,
                 y=y,
                 # category_orders=order,
                 color=color,
                 facet_row=color,
                 color_discrete_sequence=[px.colors.qualitative.D3[2], px.colors.qualitative.D3[0]],
             boxmode='overlay',
             range_y=[0, 1.2]
                )

fig.update_layout(showlegend=False, legend=dict(
        yanchor="top",
        y=1,
        xanchor="left",
        x=0.05,
        title="Sigmoid region",
        bordercolor="Black", borderwidth=2,# bgcolor = 'white'
))
fig.update_traces(name="Beta", selector=dict(name="#1F77B4"))
fig.update_traces(name="Gamma", selector=dict(name="#2CA02C"))

fig.update_traces(
    marker=dict(
        size=2,
        # line=dict(
        #     width=2,
        #     color="black"
        # )
    )
)
# fig.add_hline(y=0)
fig.update_xaxes(title="Neuron", row=2)
fig.update_yaxes(title="Beta", row=1)
fig.update_yaxes(title="Gamma", row=2)
fig.update_xaxes(
    type='category',
    tickmode='linear'
)

apply_figure_settings(fig, width_factor=0.4, height_factor=0.5)

# fig.show()

to_save = True
if to_save:
    fname = os.path.join("fig5", 'behavior_change_sigmoid_box-avb_hiscl.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)


# In[32]:


import plotly.graph_objects as go

idx_gcamp = df_params['Dataset Type'] == 'Freely Moving (GCaMP, residual)'
idx_name = df_params['neuron_name'].isin(['VB02', 'RMED', 'RMEV', 'SMDDL', 'SMDDR'])
idx = np.logical_and(idx_gcamp, idx_name)

_df = df_params[idx]#[idx_final]

r = np.exp(_df['log_hyper_beta']) + np.exp(_df['log_amplitude_mu'])
idx = df_params['Category_new'] == 'Behavior is above GFP'
r[idx] = np.exp(_df['log_amplitude_mu']) + 0.5*np.exp(_df['log_amplitude_mu'])

text = _df['text_complete'].copy()
# text[r < 0.05] = ''

fig = px.scatter_polar(_df, r=r,#'r', 
                       log_r=True,
                       theta='phase_shift', direction='counterclockwise', start_angle=0,
                       text=text,#'text_complete',
                       # color='Category_new', 
                       # color_discrete_map=color_discrete_map,
                       symbol='Category_new',
                       symbol_map=smap,
                       color='Neuron Type',
                       color_discrete_map=plotly_paper_color_discrete_map(),
                       size='size', size_max=15, #log_r=True,
    )

fig.for_each_trace(lambda t: t.update(showlegend=False))
fig.update_traces(thetaunit='radians', textposition='top center', textfont_size=12, )

colors = _df['Neuron Type'].unique()
symbols = _df['Category_new'].unique()
cmap = plotly_paper_color_discrete_map()

# for i, c in enumerate(colors):
#     # Color legend entries
#     fig.add_trace(go.Scatterpolar(
#         r=[None], theta=[None],
#         mode="markers",
#         marker=dict(color=cmap[c], symbol="circle"),
#         name=f"{c}",
#         legendgroup="color",
#         legendgrouptitle_text="Color" if i == 0 else None,
#         showlegend=True
#     ))

# for i, s in enumerate(symbols):
#     # Symbol legend entries
#     fig.add_trace(go.Scatterpolar(
#         r=[None], theta=[None],
#         mode="markers",
#         marker=dict(color="black", symbol=smap[s]),
#         name=f"{s}",
#         legendgroup="symbol",
#         legendgrouptitle_text="Symbol" if i == 0 else None,
#         showlegend=True
#     ))

    
fig.update_traces(
    marker=dict(
        size=15,
        line=dict(
            width=2,
            color="black"
        )
    )
)


# Add reference lines with manual coloring to match the example worm
cmap = ['#FF0F0F', '#CCCCCC', '#0000FF']
for i, (theta, seg_idx) in enumerate(zip([0, 90, 180], [6, 25, 37])):
    
    fig.add_trace(go.Scatterpolar(
            r = [0.35,0.6],
            theta = [theta, theta],
            mode = 'lines',
            line=dict(width=5, color=cmap[i]),
            name=seg_idx, opacity=1,
        showlegend=False
        ))

# apply_figure_settings(fig, width_factor=0.4, height_factor=0.4)
apply_figure_settings(fig, width_factor=0.6, height_factor=0.6)

fig.update_layout(polar=dict(
    angularaxis = dict(thetaunit = "radians"),
    radialaxis = dict(#range=[0, 0.6],
                      nticks=2, tickformat="~g")
), 
                  showlegend=True, 
                  legend=dict(
        yanchor="top",
        y=0.45,
        xanchor="left",
        x=0.5,
                      title="Neuron role and model results",
                      bordercolor="Black", borderwidth=2, bgcolor = 'white'
    )
                 )

fig.show()

to_save = True
if to_save:
    fname = os.path.join("fig5", 'phase_shift_and_oscillation_amplitude-avb_hiscl_control.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)
    fname = Path(fname).with_suffix('.html')
    fig.write_html(fname)
    


# In[ ]:


_df = df_params_full[df_params_full['parameter'] == 'eigenworm3_coefficient'].copy()

neurons_to_plot = df_params.loc[idx_final, 'neuron_name']
_df = _df[_df['neuron_name'].isin(neurons_to_plot)]

order = [val for i, val in enumerate(_df.groupby('neuron_name')['value'].median().sort_values().index)]

fig = px.box(_df,
      x='neuron_name',
      y='value',
      color='Neuron Type',
      color_discrete_map=plotly_paper_color_discrete_map(),
      boxmode='overlay'
)

fig.update_xaxes(categoryorder="array",
                 categoryarray=order)
fig.update_layout(showlegend=False)
apply_figure_settings(fig, width_factor=0.5, height_factor=0.2)
fig.update_xaxes(
    type='category',
    tickmode='linear'
)
fig.add_hline(y=0)

fig.update_xaxes(title="Neuron Name")
fig.update_yaxes(title="EW 3 Coefficient")
# fig.show()

to_save = True
if to_save:
    fname = os.path.join("fig5", 'ew3_boxplot-avb_hiscl_control.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)


# In[ ]:


_df = df_params_full[df_params_full['parameter'] == 'eigenworm4_coefficient'].copy()

neurons_to_plot = df_params.loc[idx_final, 'neuron_name']
_df = _df[_df['neuron_name'].isin(neurons_to_plot)]

order = [val for i, val in enumerate(_df.groupby('neuron_name')['value'].median().sort_values().index)]

fig = px.box(_df,
      x='neuron_name',
      y='value',
      color='Neuron Type',
      color_discrete_map=plotly_paper_color_discrete_map(),
      boxmode='overlay'
)

fig.update_xaxes(categoryorder="array",
                 categoryarray=order)
apply_figure_settings(fig, width_factor=0.5, height_factor=0.2)
fig.update_xaxes(
    type='category',
    tickmode='linear'
)
fig.add_hline(y=0)
fig.update_layout(showlegend=False)

fig.update_xaxes(title="Neuron Name")
fig.update_yaxes(title="EW 4 Coefficient")
# fig.show()

to_save = True
if to_save:
    fname = os.path.join("fig5", 'ew4_boxplot-avb_hiscl_control.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)


# ## Supp

# In[ ]:


# Phase shift ttests
import itertools, math
from statsmodels.stats.multitest import multipletests

def posterior_twosided_adjacent(df, neurons, n_mc=20000):
    results = []
    
    # Pre-store values for speed
    values = {
        n: df.loc[df["neuron_name"] == n, "value"].to_numpy()
        for n in neurons
    }
    
    for n1, n2 in zip(neurons[:-1], neurons[1:]):
        
        v1 = values[n1]
        v2 = values[n2]
        
        if len(v1) == 0:
            print(f"Found no data for {n1}")
            continue
        if len(v2) == 0:
            print(f"Found no data for {n2}")
            continue
        
        assert len(v1) == len(v2), "Posterior draws must match length"
        
        
        # Monte Carlo sampling (independent draws); phase is circular, so wrap the difference to [-pi, pi)
        from wbfm.utils.general.utils_paper_revisions_bayes import circular_posterior_difference_pvalue
        p_two_sided, mean_diff = circular_posterior_difference_pvalue(v1, v2, n_mc=n_mc)

        results.append({
            "neuron_1": n1,
            "neuron_2": n2,
            "p_two_sided": p_two_sided,
            "circ_mean_diff": mean_diff
        })

    return pd.DataFrame(results)

# Different
neurons_to_test = ['VB02', 'RMED', 'RMDVL', 'SMDDR', 'DB02', 'DD01', #'RMDDR', 'DB01', 
                   'RMEV', 'URXL', 'VB03', 'VB02']

# Sets of pairs which should be the same
neurons_to_test2 = [['DD01', 'DB01'],
                    ['URXL', 'URXR'],
                    ['SMDDL', 'SMDDR'],
                    ['RMDDL', 'RMDDR'],
                    ['RIAL', 'RIAR']]

_df = df_params_full[df_params_full['parameter'] == 'phase_shift'].copy()
all_neurons_to_test2 = set()
# idx = _df['neuron_name'].isin(neurons_to_test)
_df['phase_shift_set'] = "Very different"
for i, n in enumerate(neurons_to_test2):
    idx = _df['neuron_name'].isin(n)
    _df.loc[idx, 'phase_shift_set'] = f"Set {i}"
    all_neurons_to_test2.update(n)
    
results = posterior_twosided_adjacent(_df, neurons_to_test)
all_results = [results]

for n in neurons_to_test2:
    neuron_pairs = list(zip(n[:-1], n[1:]))
    results = posterior_twosided_adjacent(_df, n)
    all_results.append(results)
    # print(results)

all_results = pd.concat(all_results)

all_results['p_value_corrected'] = multipletests(all_results['p_two_sided'].values.squeeze(), method='fdr_bh', alpha=0.05)[1]


# In[ ]:


all_results


# In[ ]:


# _df = df_params_full[df_params_full['parameter'] == 'phase_shift']

fig = px.box(_df[_df['neuron_name'].isin(neurons_to_test)], x='neuron_name', y='value',
            category_orders={'neuron_name': neurons_to_test},
            range_y=[-math.pi, math.pi],
             color='Neuron Type',
             color_discrete_map=plotly_paper_color_discrete_map(),
             boxmode='overlay'
            )
fig.update_layout(showlegend=False)
fig.update_yaxes(title='Phase shift', col=1)
fig.update_xaxes(title='Neuron Name', col=1)
apply_figure_settings(fig, width_factor=0.4, height_factor=0.25)
# fig.show()

to_save = True
if to_save:
    fname = os.path.join("fig5", 'phase_shift_ttests-avb_hiscl_control.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)
    fname = Path(fname).with_suffix('.html')
    fig.write_html(fname)


# In[ ]:


# Build a dictionary mapping each facet to its x categories
category_orders = {
    facet: _df.loc[_df['phase_shift_set'] == facet, 'neuron_name'].unique().tolist()
    for facet in _df['phase_shift_set'].unique()
}

fig = px.box(_df[_df['neuron_name'].isin(all_neurons_to_test2)], x='neuron_name', y='value',
        range_y=[-math.pi, math.pi],
         color='Neuron Type',
             color_discrete_map=plotly_paper_color_discrete_map(),
         boxmode='overlay',
         facet_col='phase_shift_set',
             category_orders=category_orders
        )
# Update axes per facet
for facet_val, categories in category_orders.items():
    fig.update_xaxes(
        matches=None,  # allow independent x axes per facet
        categoryorder='array',
        categoryarray=categories,
        title_text="",  # remove individual x-axis titles
        col=_df['phase_shift_set'].unique().tolist().index(facet_val)+1  # map facet to column
    )

fig.update_yaxes(title='Phase shift', col=1)

# Add one big shared x-axis label
fig.add_annotation(
    x=0.5,
    y=-0.35,  # below the x-axes
    xref='paper',
    yref='paper',
    showarrow=False,
    text="Neuron Pairs",
    font=dict(size=16),
    align="center"
)

fig.update_xaxes(tickangle=45, tickfont=dict(size=10))
fig.update_layout(showlegend=False)
apply_figure_settings(fig, width_factor=0.4, height_factor=0.25)
# fig.show()

to_save = True
if to_save:
    fname = os.path.join("fig5", 'phase_shift_ttests_pairs-avb_hiscl_control.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)
    fname = Path(fname).with_suffix('.html')
    fig.write_html(fname)


# In[18]:


eps = 1e-3
# _df = df_params[df_params['Category'] == 'Hierarchy']
_df = df_params
y = -np.log(_df['hyper_pca0_p_value_corrected'] + eps)

text = _df['text_raw'].copy()
idx = ~_df['hyper_pca0_p_value_is_significant'].astype(bool)
text[idx] = ''

x = _df['hyper_pca0']
fig = px.scatter(_df,
                 x=x,
                 y=y,
                 # color='Category_new',
                 color_discrete_map=color_discrete_map,
                 #   color_discrete_sequence=[cmap[7], cmap[3]],
                 color='Category_new',
                 symbol='hyper_pca0_p_value_is_significant',
                 symbol_map=smap,
                 range_y=[0, 9],
                 hover_data='text_complete',
                 # text=text
                )

# get_nonoverlapping_text_positions(x, y, text, fig, weight=100, k=10, add_nodes_with_no_text=False,
#                                  # x_range=[-0.4, 0.3], 
#                                   # y_range=[0, 8], 
#                                   size=12
#                                  )
# fig.update_layout(showlegend=False)
fig.update_xaxes(title="PC1 coefficient (A_1)")
fig.update_yaxes(title="-log(p_value)")

# Only show symbols in legend
fig.for_each_trace(lambda t: t.update(showlegend=False))
# seen = set()
# for trace in fig.data:
#     symbol = trace.marker.symbol
#     if symbol not in seen:
#         parts = trace.name.split(",")
#         color_part, symbol_part = parts               # "color_group=A"
#         trace.update(showlegend=True, name=symbol_part)
#         seen.add(symbol)

colors = df_params['Category_new'].unique()
symbols = df_params['hyper_pca0_p_value_is_significant'].unique()
cmap = color_discrete_map

for i, c in enumerate(colors):
    # Color legend entries
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(color=cmap[c], symbol="circle"),
        name=f"{c}",
        legendgroup="color",
        legendgrouptitle_text="GFP comparison threshold" if i == 0 else None,
        showlegend=True
    ))
for i, s in enumerate(symbols):
    if np.isnan(s):
        continue
    # Symbol legend entries
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(color="black", symbol=smap[s]),
        name=f"{s}",
        legendgroup="symbol",
        legendgrouptitle_text="PC1 coefficient significance" if i == 0 else None,
        showlegend=True
    ))
    
fig.add_hline(y=-np.log(0.05), line=dict(dash='dot'))
fig.update_layout(showlegend=True,
legend=dict(
        yanchor="top",
        y=-0.4,
        xanchor="left",
        x=0.0,
                      title='PC1 coefficient significance',
                      bordercolor="Black", borderwidth=2, bgcolor = 'white'
    )
)
fig.update_traces(
    marker=dict(
        size=10,
        line=dict(
            width=2,
            color="black"
        )
    )
)
apply_figure_settings(fig, width_factor=0.4, height_factor=0.34)
fig.show()


to_save = True
if to_save:
    fname = os.path.join("fig5", 'pc0_coefficient-avb_hiscl_control.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)
    fname = Path(fname).with_suffix('.html')
    fig.write_html(fname)
    


# In[19]:


eps = 1e-6
# _df = df_params[df_params['Category'] == 'Hierarchy']
_df = df_params
y = -np.log(_df['hyper_pca1_p_value_corrected'] + eps)

cmap = px.colors.qualitative.D3

fig = px.scatter(_df,
                 x=x,
                 y=y,
                   color_discrete_sequence=[cmap[7], cmap[3]],
                 color='hyper_pca1_p_value_is_significant',
                 range_y=[0, 9]
                )
fig.add_hline(y=-np.log(0.05), line=dict(dash='dot'))
fig.update_layout(showlegend=False)
fig.update_xaxes(title="PC2 (A_2)")
fig.update_yaxes(title="-log(p_value)")
apply_figure_settings(fig, width_factor=0.15, height_factor=0.15)
fig.update_traces(
    marker=dict(
        size=10,
        # line=dict(
        #     width=2,
        #     color="black"
        # )
    )
)
fig.show()

to_save = True
if to_save:
    fname = os.path.join("fig5", 'pc1_coefficient-avb_hiscl_control.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)
    fname = Path(fname).with_suffix('.html')
    fig.write_html(fname)


# In[ ]:


from wbfm.utils.external.utils_plotly import add_trendline_annotation

# Manifold explained x axis
# x, y = 'manifold_variance', 'Relative Hierarchy Score'
x, y = 'manifold_variance', 'log_hyper_beta'

# idx = np.logical_and(df_params['Dataset Type'] == 'Freely Moving (GCaMP, residual)', ~np.isnan(df_params[y]))
idx = np.logical_and(df_params['Category_new'] == 'Hierarchy > GFP and PC1 is significant', ~np.isnan(df_params[y]))
_df = df_params[idx]

text = df_params['text'].copy()
text[:] = ''
text[idx_final] = df_params['text_raw']
text = text[idx]

y_vals = np.exp(_df[y])

# Actual plot
fig = px.scatter(_df, 
                 y=y_vals, x=x,
                 # text='neuron_name', 
                 # text=text, 
                 color='Category_new',
                 hover_data='text_raw',
                 color_discrete_map=color_discrete_map, #size='Size', size_max=10,
                 trendline='ols'
                 # marginal_y='rug'
                )

get_nonoverlapping_text_positions(_df[x], y_vals, text, fig, weight=100, k=1, add_nodes_with_no_text=False,
                                 x_range=[0, 1], y_range=[0, 1], 
                                  size=12
                                 )

fig.update_xaxes(title='Variance Explained by Manifold')
fig.update_yaxes(title='Gamma')

fig.update_layout(showlegend=False, 
                  legend=dict(
    yanchor="top",
    y=1.6,
    xanchor="left",
    x=0.0, title="Model results",
    bordercolor="Black", borderwidth=2, bgcolor = 'white'
    )
)
fig.update_traces(
    marker=dict(
        size=10,
        line=dict(
            width=2,
            color="black"
        )
    )
)
add_trendline_annotation(fig, x_offset=0.6, y_offset=0.25)
apply_figure_settings(fig, width_factor=0.5, height_factor=0.15)

fig.show()

to_save = True
if to_save:
    fname = os.path.join("fig5", 'hierarchy_behavior_score_and_manifold_variance-avb_hiscl_control.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)


# In[ ]:



# fig = px.scatter_polar(df_params[df_params['Neuron Type'].str.contains('Motor')], r='r', theta='phase_shift', direction='counterclockwise', start_angle=0,
#                        color='Neuron Type', 
#                        color_discrete_map=plotly_paper_color_discrete_map(),
#                        size='Relative Hierarchy Score', size_max=15, #log_r=True,
#                        #color='Neuron Role',
#                        #color='muscle_position'
#                       )
# fig.update_traces(thetaunit='radians', textposition='top center', textfont_size=12, )

# apply_figure_settings(fig, width_factor=0.4, height_factor=0.4)
# # apply_figure_settings(fig, width_factor=1.0, height_factor=1.0)

# fig.update_layout(polar=dict(
#     angularaxis = dict(thetaunit = "radians"),
#     radialaxis = dict(range=[0, 0.5],
#                       nticks=3)
# ), 
#                   showlegend=True, 
#                   legend=dict(
#         yanchor="top",
#         y=0.4,
#         xanchor="left",
#         x=0.6,
#                       bordercolor="Black", borderwidth=2, bgcolor = 'white'
#     )
#                  )

# fig.show()

# to_save = True
# if to_save:
#     fname = os.path.join("fig5", 'phase_shift_and_oscillation_amplitude_only_motor-no_text.png')
#     fig.write_image(fname, scale=3)
#     fname = Path(fname).with_suffix('.svg')
#     fig.write_image(fname)
#     fname = Path(fname).with_suffix('.html')
#     fig.write_html(fname)
    


# In[ ]:


import xarray as xr

neuron_name = 'VB02'
idata = all_traces[neuron_name]

# 1. Get the log-likelihood
# Dimensions: (chain, draw, obs_id)
log_lik = idata.log_likelihood['y']

# 2. Define your trial mapping
# This should be a 1D array/Series of length obs_id, 
# where each entry is the trial ID for that observation.
# Example: trial_ids = [0, 0, 0, 1, 1, 2, 2, 2, 2...]
idx = Xy[neuron_name].dropna().index
dataset_vector = Xy.loc[idx, 'dataset_name'].reset_index(drop=True)
trial_mapping = xr.DataArray(dataset_vector, dims=['y_dim_0'])

# 3. Perform Grouped Summation
# This handles the unequal trial lengths automatically.
# It sums the log-probs of all points within a trial.
# Resulting dims: (chain, draw, trial_id)
print("Summing...")
log_lik_grouped = log_lik.groupby(trial_mapping).sum(dim='y_dim_0')


# In[ ]:


# 4. Compute LOO on the grouped data

log_lik_grouped.name = "y" 
idata_loto = az.InferenceData(
    posterior=idata.posterior, 
    log_likelihood=log_lik_grouped.to_dataset()
)

loto_result = az.loo(idata_loto, pointwise=True)

print(loto_result)


# In[ ]:


loto_result.keys()


# In[ ]:


loto_result.keys()


# In[ ]:


# from wbfm.utils.external.utils_arviz import leave_one_trial_out_cv_all_neurons, leave_one_trial_out_cv_all_neurons_parallel
# from joblib import Parallel, delayed
# loto_results = leave_one_trial_out_cv_all_neurons(all_traces, Xy)

# # results = leave_one_trial_out_cv_all_neurons_parallel(all_traces, Xy)


# # Look at eigenworms

# In[27]:


from wbfm.utils.general.utils_hardcoded import load_paper_datasets


# In[30]:


all_projects = load_paper_datasets('avb_hiscl_control')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Try to diagnose why LOTO is so bad

# In[ ]:


ax = az.plot_forest(
    idata,
    var_names=["hyper_pca0_"],
    filter_vars="like",
    # figsize=(11.5, 5),
    # colors="C1",
    ess=True,
    combined=True,
    # r_hat=True,
)


# In[ ]:


ax = az.plot_forest(
    idata,
    var_names=["pca0_amplitude"],
    # filter_vars="like",
    # figsize=(11.5, 5),
    # colors="C1",
    ess=True,
    combined=True,
    # r_hat=True,
)


# In[ ]:


az.plot_forest(
    idata, 
    var_names=["pca0_amplitude"], 
    kind='ridgeplot', 
    combined=True, 
    ridgeplot_overlap=3, 
    figsize=(8, 12)
)


# In[ ]:


# import numpy as np
# import arviz as az
# import matplotlib.pyplot as plt

# # 1. Extract the population hyperparameters from your trace
# mu_pop = idata.posterior['hyper_pca0_amplitude'].values.flatten()
# sigma_pop = idata.posterior['hyper_pca0_sigma'].values.flatten()

# # 2. Generate "Potential" trials from those hyperparameters
# # This represents the 'Marginal' distribution (the model's expectation for any trial)
# marginal_samples = np.random.normal(mu_pop, sigma_pop)

# # 3. Plotting
# plt.figure(figsize=(10, 6))

# # Plot the Marginal Distribution (The "Rule")
# az.plot_dist(marginal_samples, color="black", label="Marginal (Population Rule)", 
#              plot_kwargs={"linewidth": 3, "linestyle": "--"})

# # Plot a few individual trials to see how they sit within the rule

# for i in range(36):
#     trial_0 = idata.posterior['pca0_amplitude'].sel(pca0_amplitude_dim_0=i).values.flatten()
#     # trial_5 = idata.posterior['pca0_amplitude'].sel(pca0_amplitude_dim_0=1).values.flatten()
    
#     az.plot_dist(trial_0, color="blue", label=f"Trial {i} Posterior")
# # az.plot_dist(trial_5, color="red", label="Trial 5 Posterior")

# plt.xlabel("pca0_amplitude")
# plt.legend()
# plt.title("Local Posteriors vs. Marginal Population Distribution")


# In[ ]:


import numpy as np
import arviz as az
import matplotlib.pyplot as plt

# 1. Get Population Samples (The "Rule")
mu_pop = idata.posterior['hyper_pca0_amplitude'].values.flatten()
sigma_pop = idata.posterior['hyper_pca0_sigma'].values.flatten()
# Sample from the implied population distribution
marginal_samples = np.random.normal(mu_pop, sigma_pop)

# 2. Get Trial-specific samples
# (chains, draws, 36 trials) -> flatten to (samples, 36 trials)
trial_samples = idata.posterior['pca0_amplitude'].stack(sample=("chain", "draw")).values.T

# 3. Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the individual trials as thin, transparent lines
for i in range(trial_samples.shape[1]):
    az.plot_dist(trial_samples[:, i], ax=ax, #color="gray", 
                 plot_kwargs={"alpha": 0.5, "linewidth": 1}, 
                 label="Individual Trials" if i == 0 else "")

# Plot the Population Marginal as a thick, high-contrast line
az.plot_dist(marginal_samples, ax=ax, color="black", 
             plot_kwargs={"linewidth": 3, "linestyle": "--"}, 
             label="Population Marginal (Prior for new trials)")

ax.set_title("Hierarchical Summary: Local Posteriors vs. Population Rule")
ax.legend()
plt.show()


# In[ ]:


idata


# In[ ]:


# import matplotlib.pyplot as plt
# import numpy as np

# # 1. Get Pareto k values from your existing LOO object
# k_values = loto_result.pareto_k.values

# # 2. Calculate the deviation of local parameters
# # Replace 'theta_trial' and 'mu_pop' with your actual variable names
# theta_means = idata.posterior['pca0_amplitude'].mean(dim=['chain', 'draw'])
# pop_mean = idata.posterior['hyper_pca0_amplitude'].mean(dim=['chain', 'draw'])

# # Simple distance (absolute difference)
# deviation = np.abs(theta_means - pop_mean)

# # 3. Plot
# plt.figure(figsize=(8, 6))
# plt.scatter(deviation, k_values, alpha=0.6)
# plt.axhline(0.7, color='r', linestyle='--', label='Danger Threshold (0.7)')
# plt.xlabel("Trial Parameter Deviation (Distance from Pop Mean)")
# plt.ylabel("Pareto k statistic")
# plt.title("Diagnostic: Do unique trials break LOO?")
# plt.legend()
# plt.show()


# In[ ]:


az.plot_khat(loto_result, show_hlines=True, show_bins=True)


# In[ ]:





# ### Recalculate marginal likelihoods using original model

# In[ ]:


from wbfm.utils.external.utils_pymc import build_baseline_priors, build_curvature_term, build_final_likelihood, build_sigmoid_term_pca
from wbfm.utils.external.utils_pandas import get_dataframe_for_single_neuron


# In[ ]:


neuron_name = 'VB02'
curvature_terms_to_use = ['eigenworm0', 'eigenworm1']
curvature_terms_to_use.extend(['eigenworm2', 'eigenworm3'])
residual_mode='pca_global'
dataset_name = "all"

# First pack into a single dataframe to drop nan, then unpack
df_model = get_dataframe_for_single_neuron(Xy, neuron_name, dataset_name=dataset_name,
                                           curvature_terms=curvature_terms_to_use, residual_mode=residual_mode)

pca_modes = df_model[['x_pca0', 'x_pca1']].values
y = df_model['y'].values
curvature = df_model[curvature_terms_to_use].values


# In[ ]:


import pymc as pm
import arviz as az
import xarray as xr
import numpy as np


dataset_name_idx, dataset_name_values = df_model.dataset_name.factorize()
coords = {'dataset_name': dataset_name_values}
dims = 'dataset_name'
dim_opt = dict(dims=dims, dataset_name_idx=dataset_name_idx)

# 1. Re-instantiate the exact model structure
with pm.Model(coords=coords) as marginal_model:
    # Use pm.Data to pass in the original inputs
    # 'y' must be the original 50k length vector
    # 'pca_modes' and 'curvature' must be the original inputs
    
    intercept, sigma = build_baseline_priors()
    sigmoid_term = build_sigmoid_term_pca(pca_modes, **dim_opt)
    curvature_term = build_curvature_term(curvature, 
                                          curvature_terms_to_use=curvature_terms_to_use, 
                                          **dim_opt)

    mu = pm.Deterministic('mu', intercept + sigmoid_term * curvature_term)
    
    # This creates the 'y' log_likelihood group ArviZ needs
    likelihood = build_final_likelihood(mu, sigma, y)

# 2. Prepare the "Prior-Swapped" InferenceData

    
trial_params = ["zscore_pca0_amplitude", "zscore_pca1_amplitude", "zscore_intercept"]
idata_marginal = idata.copy()

# Remove the existing likelihood so compute_log_likelihood has a clean slate
if "log_likelihood" in idata_marginal:
    del idata_marginal.log_likelihood
    
for param in trial_params:
    if param in idata_marginal.posterior:
        # Get shape: (chain, draw, 36)
        post_shape = idata_marginal.posterior[param].shape
        # Replace the 'spiky' posterior with standard normal noise (the prior)
        idata_marginal.posterior[param] = (("chain", "draw", "trial_dim"), 
                                           np.random.normal(0, 1, size=post_shape))


# In[ ]:


# 3. Compute the Pointwise Log-Likelihood
rng = 424242
opt = dict(draws=1000, tune=1000, random_seed=rng, target_accept=0.96)
with marginal_model:
    # This runs the 'mu' math using posterior fixed effects + random prior z-scores
    pm.compute_log_likelihood(idata_marginal)#, **opt)


# In[ ]:


import xarray as xr

# 1. Get the pointwise log-likelihood from the marginalized idata
# Shape is (chain, draw, time_dim)
pointwise_ll = idata_marginal.log_likelihood["y"]

# 2. Add the trial mapping as a coordinate to the xarray object
# This 'labels' every one of the 50k points with its trial ID
# Example: trial_ids = [0, 0, 0, 1, 1, 2, 2, 2, 2...]
idx = Xy[neuron_name].dropna().index
dataset_vector = Xy.loc[idx, 'dataset_name'].reset_index(drop=True)
trial_mapping = xr.DataArray(dataset_vector, dims=['y_dim_0'])

pointwise_ll = pointwise_ll.assign_coords(trial_id=("y_dim_0", np.asarray(trial_mapping)))

# 3. Use xarray's vectorized groupby to sum everything into 36 buckets
# This is the 'Leave-One-Trial-Out' summation step
trial_ll = pointwise_ll.groupby("trial_id").sum(dim="y_dim_0")

# 4. Rebuild the InferenceData for ArviZ
# The trial_ll now has shape (chain, draw, 36)
# Create the final scorable object including the posterior
loto_idata = az.InferenceData(
    posterior=idata_marginal.posterior,  # Re-attach the posterior samples
    log_likelihood=xr.Dataset({"y": trial_ll.rename({"trial_id": "trial"})}),
)


# In[ ]:



loto_result = az.loo(loto_idata)

print(loto_result)


# In[ ]:


az.plot_khat(loto_result, show_hlines=True, show_bins=True)


# ### Refit using grouped K folds

# In[ ]:


from wbfm.utils.external.utils_pymc import grouped_cv_refitting, compute_cv_loo


# In[ ]:


neuron_name = "VB02"
cv_results = grouped_cv_refitting(Xy, neuron_name, dataset_name='all', DEBUG=False)


# In[ ]:


idata, loo = compute_cv_loo(cv_results)


# In[ ]:


loo


# In[ ]:





# In[ ]:





# In[ ]:





# ### Reanalyze the LOO directly... I think this isn't enough information

# In[ ]:





# In[ ]:





# In[ ]:


# def load_all_traces_nonhierarchical(foldername):
def load_all_loo(foldername):
    fnames = neurons_with_confident_ids()
    all_traces = {}
    for neuron in tqdm(fnames):
        trace_fname = os.path.join(foldername, f'{neuron}_loo.h5')
        if os.path.exists(trace_fname):
            try:
                trace = pd.read_hdf(trace_fname)
                # trace = az.from_netcdf(trace_fname)
                all_traces[neuron] = trace
            except (ValueError, OSError) as e:
                print(f"Error for neuron {neuron}; this is not surprising if some are still being written: {e}")
    return all_traces

parent_folder = '/lisc/data/scratch/neurobiology/zimmer/fieseler/paper/hierarchical_modeling'
# suffix = '_only_eigenworms'
# suffix = '_eigenworms34_speed'
suffix = '_original'
# suffix = '_new_ids_only_eigenworms'
            
foldername = os.path.join(parent_folder, f'output{suffix}')
# all_traces_gcamp_nonhierarchical = load_all_traces_nonhierarchical(foldername)
all_loo_gcamp = load_all_loo(foldername)


# In[ ]:





# In[ ]:


neuron_name = 'ALA'

# idata_full = all_traces_gcamp[neuron_name]
# idata_null = all_traces_gcamp_nonhierarchical[neuron_name]

# cmp = az.compare(
#     {"null": idata_null, "full": idata_full},
#     ic="loo"
# )
cmp = all_loo_gcamp[neuron_name]
az.plot_compare(cmp)


# In[ ]:


all_loo_gcamp['VB02']


# In[ ]:


delta = cmp.loc["nonhierarchical", "elpd_diff"]# - cmp.loc["hierarchical_pca", "elpd_loo"]
bf = np.exp(delta / 2)
bf


# In[ ]:


df_to_plot_gcamp


# In[ ]:


cmp


# ## DEBUG: Specific segment extra annotations

# In[ ]:


# import sklearn, math

# x = ['eigenworm1', 'eigenworm2']
# all_theta = []

# for i in range(1, 100):
#     y = f'curvature_{i}'
#     # print(y)

#     Xy_for_model = Xy[x].join(Xy[y]).dropna()

#     model = sklearn.linear_model.LinearRegression(fit_intercept=False)
#     model.fit(Xy_for_model[x], Xy_for_model[y])

#     # Convert to polar coordinates
#     cx, cy = model.coef_
#     r = np.sqrt(cx**2 + cy**2)
#     # theta = np.arctan2(cy, cx)
#     theta = np.arctan(cy/cx)
#     all_theta.append(theta)
#     # print(r, 360*theta/(2*math.pi))
# fig = px.line(all_theta)
# fig.add_hline(y=math.pi/2)
# # fig.add_hline(y=math.pi)
# fig.add_hline(y=-math.pi/2)


# In[ ]:





# In[ ]:





# ## Sigmoid slope

# In[ ]:


df_params.head()


# In[ ]:


# Just plot slope median and variance
fig = px.scatter(df_params.sort_values(by='sigmoid_term'), y='sigmoid_term', x='sigmoid_term_variance',#x=df_params.index,
            text=df_params.index)

apply_figure_settings(fig, width_factor=1.0, height_factor=0.3)
fig.show()


# In[ ]:


# Get the full posterior and plot
# var_names2 = ["hyper_pca0_amplitude"]
var_names2 = ["hyper_pca0_amplitude", "hyper_pca1_amplitude", 
              'eigenworm3_coefficient', 'eigenworm4_coefficient']

all_dfs_hierarchy = {}
for n in tqdm(neurons_to_plot):
    dat = az.extract(all_traces[n], group='posterior', var_names=var_names2, filter_vars='like')
    dat_hierarchy = dat.to_dataframe().drop(columns=['chain', 'draw'])
    
    all_dfs_hierarchy[n] = dat_hierarchy
    # all_dfs_hierarchy[n] = np.squeeze(dat_hierarchy.values)
# df_hierarchy = pd.DataFrame(all_dfs_hierarchy)
df_hierarchy = pd.concat(all_dfs_hierarchy, axis=1).swaplevel(0, 1, axis=1)


# In[ ]:


df_hierarchy_melt = df_hierarchy.melt(var_name=['Variable', 'Neuron Name'], value_name='PC1 Coefficient', value_vars=df_hierarchy.columns.tolist())
# df_hierarchy_melt['Neuron Type'] = df_hierarchy_melt['Neuron Name'].map(df_params['Neuron Type'].to_dict())
# df_hierarchy_melt['Neuron Type'] = df_hierarchy_melt['Neuron Name'].map(role_of_neuron_dict(include_fwd_rev=True, include_ventral_dorsal=True, include_basic=False)).replace('', 'Other')
df_hierarchy_melt['Neuron Type'] = df_hierarchy_melt['Neuron Name'].map(role_of_neuron_dict(only_fwd_rev=True)).replace('', 'Other')
df_hierarchy_melt['Neuron Type Complex'] = df_hierarchy_melt['Neuron Name'].map(role_of_neuron_dict(include_fwd_rev=True, include_ventral_dorsal=True, include_basic=False)).replace('', 'Other')
df_hierarchy_melt['Neuron Type VD'] = df_hierarchy_melt['Neuron Name'].map(role_of_neuron_dict(include_ventral_dorsal=True)).replace('', 'Other')


# In[ ]:


df_hierarchy_melt['Neuron Type'].unique()


# In[ ]:


df_hierarchy_melt['Variable'].unique()


# In[ ]:


ordering = df_hierarchy_melt[df_hierarchy_melt['Variable'] == 'hyper_pca0_amplitude'].groupby('Neuron Name')['PC1 Coefficient'].median().sort_values()

_df = df_hierarchy_melt[df_hierarchy_melt['Variable']=='hyper_pca0_amplitude']

fig = px.box(_df, y='PC1 Coefficient', x='Neuron Name', #color='Neuron Type', #facet_row='Variable',
            category_orders={'Neuron Name': ordering.index}, color_discrete_map=plotly_paper_color_discrete_map())
apply_figure_settings(fig, width_factor=0.5, height_factor=0.25)
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", )#range=[-7, 10])
fig.update_layout(boxmode = "overlay") # Remove offset caused by invisible multiple types per neuron name
fig.update_yaxes(title_text='Gating Parameter')
fig.update_xaxes(title_text='', tickfont_size=10)

fig.update_layout(
    legend=dict(
        # itemsizing='constant',  # Display legend items as colored boxes and text
        x=0.1,  # Adjust the x position of the legend
        y=1.2, #0.54,  # Adjust the y position of the legend
        # bgcolor='rgba(0, 0, 0, 0.00)',  # Set the background color of the legend
        # bordercolor='Black',  # Set the border color of the legend
        # borderwidth=1,  # Set the border width of the legend
        # font=dict(size=base_font_size)  # Set the font size of the legend text
    )
)

# Smaller dots to make outlier transition more visible
fig.update_traces(marker=dict(size=2))

# Annotations to make the y axis more interpretable
# fig.add_annotation(x=-0.04, y=0.75,
#             text="Forward", textangle=-90,
#             showarrow=True,
#             arrowhead=1, xref="paper", yref="paper", #axref='paper', 
#                    ax=10, ay=10)
# fig.add_annotation(x=-0.03, y=0,
#             text="Reverse", textangle=-90,
#             showarrow=True,
#             arrowhead=1, xref="paper", yref="paper")

fig.show()

to_save = True
if to_save:
    fname = os.path.join("/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/bayesian_modeling/plots", 'sigmoid_coefficient_basic_colors.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)
    fname = Path(fname).with_suffix('.html')
    fig.write_html(fname)


# In[ ]:


ordering = df_hierarchy_melt[df_hierarchy_melt['Variable'] == 'hyper_pca1_amplitude'].groupby('Neuron Name')['PC1 Coefficient'].median().sort_values()

_df = df_hierarchy_melt[df_hierarchy_melt['Variable']=='hyper_pca1_amplitude']

fig = px.box(_df, y='PC1 Coefficient', x='Neuron Name', #color='Neuron Type', #facet_row='Variable',
            category_orders={'Neuron Name': ordering.index}, color_discrete_map=plotly_paper_color_discrete_map())
apply_figure_settings(fig, width_factor=0.5, height_factor=0.25)
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", )#range=[-7, 10])
fig.update_layout(boxmode = "overlay") # Remove offset caused by invisible multiple types per neuron name
fig.update_yaxes(title_text='Gating Parameter (2)<br>(PCA mode 2 parameter)')
fig.update_xaxes(title_text='', tickfont_size=9)

fig.update_layout(
    legend=dict(
        # itemsizing='constant',  # Display legend items as colored boxes and text
        x=0.1,  # Adjust the x position of the legend
        y=1.2, #0.54,  # Adjust the y position of the legend
        # bgcolor='rgba(0, 0, 0, 0.00)',  # Set the background color of the legend
        # bordercolor='Black',  # Set the border color of the legend
        # borderwidth=1,  # Set the border width of the legend
        # font=dict(size=base_font_size)  # Set the font size of the legend text
    )
)

# Smaller dots to make outlier transition more visible
fig.update_traces(marker=dict(size=2))

# Annotations to make the y axis more interpretable
# fig.add_annotation(x=-0.04, y=0.75,
#             text="Forward", textangle=-90,
#             showarrow=True,
#             arrowhead=1, xref="paper", yref="paper", #axref='paper', 
#                    ax=10, ay=10)
# fig.add_annotation(x=-0.03, y=0,
#             text="Reverse", textangle=-90,
#             showarrow=True,
#             arrowhead=1, xref="paper", yref="paper")

fig.show()

to_save = True
if to_save:
    fname = os.path.join("/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/bayesian_modeling/plots", 'sigmoid_coefficient_basic_colors2.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)
    fname = Path(fname).with_suffix('.html')
    fig.write_html(fname)


# In[ ]:


state_to_plot = 'eigenworm3_coefficient'

ordering = df_hierarchy_melt[df_hierarchy_melt['Variable'] == state_to_plot].groupby('Neuron Name')['PC1 Coefficient'].median().sort_values()

_df = df_hierarchy_melt[df_hierarchy_melt['Variable']==state_to_plot]

fig = px.box(_df, y='PC1 Coefficient', x='Neuron Name', color='Neuron Type VD', #facet_row='Variable',
            category_orders={'Neuron Name': ordering.index}, color_discrete_map=plotly_paper_color_discrete_map())
apply_figure_settings(fig, width_factor=0.5, height_factor=0.3)
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black", )#range=[-6, 2])
fig.update_layout(boxmode = "overlay") # Remove offset caused by invisible multiple types per neuron name
fig.update_yaxes(title_text='Turning Mode')#<br>Model Parameter')
fig.update_xaxes(title_text='', tickfont_size=10)

fig.update_layout(
    legend=dict(
        # itemsizing='constant',  # Display legend items as colored boxes and text
        yanchor="bottom",
        y=0.8,
        xanchor="left",
        x=0,
        title='Neuron Type',
        orientation='h',  # See: https://stackoverflow.com/questions/29189097/how-to-make-a-plotly-legend-span-two-columns
        entrywidth=0.4,
        entrywidthmode='fraction',
        # bgcolor='rgba(0, 0, 0, 0.00)',  # Set the background color of the legend
        # bordercolor='Black',  # Set the border color of the legend
        # borderwidth=0,  # Set the border width of the legend
        # font=dict(size=base_font_size)  # Set the font size of the legend text
    )
)
fig.show()

to_save = True
if to_save:
    fname = os.path.join("/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/bayesian_modeling/plots", f'{state_to_plot}.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)
    fname = Path(fname).with_suffix('.html')
    fig.write_html(fname)


# In[ ]:


state_to_plot = 'eigenworm4_coefficient'

ordering = df_hierarchy_melt[df_hierarchy_melt['Variable'] == state_to_plot].groupby('Neuron Name')['PC1 Coefficient'].median().sort_values()

_df = df_hierarchy_melt[df_hierarchy_melt['Variable']==state_to_plot]

fig = px.box(_df, y='PC1 Coefficient', x='Neuron Name', color='Neuron Type VD', #facet_row='Variable',
            category_orders={'Neuron Name': ordering.index}, color_discrete_map=plotly_paper_color_discrete_map())
apply_figure_settings(fig, width_factor=0.5, height_factor=0.25)
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black")
fig.update_layout(boxmode = "overlay", showlegend=True) # Remove offset caused by invisible multiple types per neuron name
fig.update_yaxes(title_text='Turning Mode<br>(Eigenworm 4 coefficient)')
fig.update_xaxes(title_text='', tickfont_size=10)

fig.update_layout(
    legend=dict(
        # itemsizing='constant',  # Display legend items as colored boxes and text
        yanchor="bottom",
        y=0.8,
        xanchor="left",
        x=-0,
        title='Neuron Type',
        orientation='h',  # See: https://stackoverflow.com/questions/29189097/how-to-make-a-plotly-legend-span-two-columns
        entrywidth=0.4,
        entrywidthmode='fraction',
        # bgcolor='rgba(0, 0, 0, 0.00)',  # Set the background color of the legend
        # bordercolor='Black',  # Set the border color of the legend
        # borderwidth=0,  # Set the border width of the legend
        # font=dict(size=base_font_size)  # Set the font size of the legend text
    )
)

fig.show()

to_save = True
if to_save:
    fname = os.path.join("/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/bayesian_modeling/plots", f'{state_to_plot}.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)
    fname = Path(fname).with_suffix('.html')
    fig.write_html(fname)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




