#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm
import os 
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
import plotly.express as px


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
df_params, df_params_full, y_max_gfp, x_max_gfp = calculate_all_bayesian_model_data(suffix='_pca_global', recalculate_sigmoid=False)


# In[6]:



idx = df_params['Dataset Type'] == 'Freely Moving (GCaMP, residual)'
_df = df_params[idx]
mapping = dict(zip(_df['neuron_name'], _df['Neuron Type']))
df_params_full['Neuron Type'] = df_params_full['neuron_name'].map(mapping)


# In[7]:


from wbfm.utils.general.utils_paper import plot_bayesian_model_comparison, plotly_paper_color_discrete_map
from wbfm.utils.external.utils_plotly import get_nonoverlapping_text_positions
from wbfm.utils.general.utils_hardcoded import neurons_with_less_confident_ids


# In[8]:



cmap = px.colors.qualitative.D3.copy()

# Recalculate the gfp thresholds
y, x = 'Relative Hierarchy Score (raw)', 'Behavior Score (raw)'
_df = df_params[df_params['Dataset Type'] == 'Freely Moving (GFP, residual)']
x_max_gfp = _df[x].max()
y_max_gfp = _df[y].max()

# df_params['Category_new'] = 'PC1 not significant'
df_params['Category_new'] = 'Not above artifacts'
df_params['Behavior is above GFP'] = False # Helper category with information subset
df_params['Hierarchy is above GFP'] = False # Helper category with information subset

idx_beh = df_params[x] > x_max_gfp
df_params.loc[idx_beh, 'Category_new'] = 'Behavior is above GFP'
df_params.loc[idx_beh, 'Behavior is above GFP'] = True
idx = df_params[y] > y_max_gfp
df_params.loc[idx_beh, 'Hierarchy is above GFP'] = True
# df_params.loc[idx, 'Category_new'] = 'Hierarchy is above GFP'
idx_hierarchy = np.logical_and(df_params['hyper_pca0_p_value_is_significant'] == True, df_params[y] > y_max_gfp)
df_params.loc[idx_hierarchy, 'Category_new'] = 'Hierarchy > GFP and PC1 is significant'

color_discrete_map = {True: cmap[0], 'PC1 is significant': cmap[0], 'Hierarchy is above GFP': cmap[0], 'Hierarchy > GFP and PC1 is significant':cmap[0],
                     False: cmap[7], 'PC1 not significant': cmap[7],'Not above artifacts': cmap[7],
                     'PC1 not significant, but behavior is good': cmap[2], 'Behavior is above GFP': cmap[2]}

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

# Add parentheses to less certain neurons
df_params['Neuron Name'] = df_params['neuron_name'].map(neurons_with_less_confident_ids(return_mapping=True, combine_left_right=False))
df_params_full['Neuron Name'] = df_params_full['neuron_name'].map(neurons_with_less_confident_ids(return_mapping=True, combine_left_right=False))


# In[9]:


# Save to disk for comparison to other manifold models

fname = os.path.join("bayesian_alt_manifold", 'df_params.h5')
df_params.to_hdf(fname, key="df_with_missing")


# In[10]:


# y, x = 'Relative Hierarchy Score', 'Behavior Score'
y, x = 'Relative Hierarchy Score (raw)', 'Behavior Score (raw)'

symbol_name = "Model parameter significance" 

text = df_params['Neuron Name'].copy()
text[:] = ''
text[idx_final] = df_params['Neuron Name'].copy()[idx_final]

fig = px.scatter(df_params, 
                 y=y, x=x,
                 # y=df_params[y] + 1, x=df_params[x] + 1,
                 # log_y=True, log_x=True,
                 # color='hyper_pca0_p_value_is_significant',
                 color='Category_new',
                 color_discrete_map=color_discrete_map,
                 # category_orders={'Category':['PC1 not significant', 'PC1 not significant, but behavior is good', 'PC1 is significant']},
                 symbol=symbol_name,
                 symbol_map=smap,
                 size_max=50,
                 hover_data=['text_raw', 'Dataset Type'],
                 # text=text,
                )
opt = dict(line=dict(dash='dot'), col=1)

_df = df_params[df_params['Dataset Type'] == 'Freely Moving (GFP, residual)']
fig.add_hline(y=_df[y].max(), **opt)
fig.add_vline(x=_df[x].max(), **opt)

fig.update_yaxes(dict(tickformat="~g"), title='Improvement of Hierarchy Model')#title=y)
fig.update_xaxes(dict(tickformat="~g"), title='Performance of Behavior-only Model')#title=x)

get_nonoverlapping_text_positions(df_params[x], df_params[y], text, fig, weight=10, k=1500, add_nodes_with_no_text=False,
                                 # x_range=[-5, 40], 
                                  y_range=[-300, 1000], 
                                  size=12
                                 )
fig.update_layout(showlegend=False)

# Modify legend to only show colors, not symbols
fig.for_each_trace(lambda t: t.update(showlegend=False))

colors = df_params['Category_new'].unique()
symbols = df_params[symbol_name].unique()
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
        legendgrouptitle_text=symbol_name if i == 0 else None,
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
fig.data = tuple(
    sorted(fig.data, key=lambda t: t.name != "Not above artifacts")
)

apply_figure_settings(fig, width_factor=0.6, height_factor=0.3)
fig.show()


to_save = True
if to_save:
    fname = os.path.join("fig5", 'model_improvement.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)
    fname = Path(fname).with_suffix('.html')
    fig.write_html(fname)
    


# In[11]:


# Still sort by median, conditional on whether the beta parameter is significant
_df = df_params[idx_final].copy()

only_beh_neurons = _df[_df['Category_new']== 'Behavior is above GFP']['Neuron Name'].values
idx_beta_sorted = _df['log_hyper_beta'].sort_values().index
neurons_beta_sorted = _df.loc[idx_beta_sorted, 'Neuron Name'].reset_index(drop=True)
map_beta_sorted = dict(zip(neurons_beta_sorted.values, neurons_beta_sorted.index))

idx_gamma_sorted = _df['log_amplitude_mu'].sort_values().index
neurons_gamma_sorted = _df.loc[idx_gamma_sorted, 'Neuron Name'].reset_index(drop=True)
map_gamma_sorted = dict(zip(neurons_gamma_sorted.values, neurons_gamma_sorted.index))

all_neurons = neurons_gamma_sorted.values
rank = {neuron: (map_gamma_sorted[neuron]+100 if neuron in only_beh_neurons else map_beta_sorted[neuron]) for neuron in all_neurons }

neurons_final = df_params.loc[idx_final, 'Neuron Name']
idx_final_neurons = df_params_full['Neuron Name'].isin(neurons_final)
idx_final_vals = df_params_full['parameter'].isin(['log_amplitude_mu', 'log_hyper_beta'])
idx = np.logical_and(idx_final_vals, idx_final_neurons)
_df = df_params_full[idx].copy().sort_values(by='Neuron Name', key=lambda x: x.map(rank))

# Build columns myself
x = np.zeros(len(_df), dtype=np.dtypes.StringDType)
y = np.zeros_like(x, dtype=float)
color = np.zeros_like(y, dtype=np.dtypes.StringDType)

idx_mu = _df['parameter'] == 'log_amplitude_mu'
n = np.sum(idx_mu)
y[idx_mu] = np.exp(_df.loc[idx_mu, 'value'])
x[idx_mu] = _df.loc[idx_mu, 'Neuron Name'].values
color[idx_mu] = [px.colors.qualitative.D3[2]] * n

idx_beta = _df['parameter'] == 'log_hyper_beta'
n = np.sum(idx_beta)
_x = _df.loc[idx_beta, 'Neuron Name'].values
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
        size=3,
        # line=dict(
        #     width=2,
        #     color="black"
        # )
    )
)
# fig.add_hline(y=0)
fig.update_xaxes(title="")
fig.update_yaxes(title="")
fig.update_annotations(text="")
# fig.update_xaxes(title="Neuron", row=2)
# fig.update_yaxes(title="Beta", row=1)
# fig.update_yaxes(title="Gamma", row=2)
fig.update_xaxes(
    type='category',
    tickmode='linear'
)

apply_figure_settings(fig, width_factor=0.4, height_factor=0.5)

# fig.show()

to_save = True
if to_save:
    fname = os.path.join("fig5", 'behavior_change_sigmoid_box.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)


# In[12]:


import plotly.graph_objects as go

_df = df_params[idx_final]

r = np.exp(_df['log_hyper_beta']) + np.exp(_df['log_amplitude_mu'])
idx = df_params['Category_new'] == 'Behavior is above GFP'
r[idx] = np.exp(_df['log_amplitude_mu']) + 0.5*np.exp(_df['log_amplitude_mu'])

text = _df['Neuron Name'].copy()
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

for i, c in enumerate(colors):
    # Color legend entries
    fig.add_trace(go.Scatterpolar(
        r=[None], theta=[None],
        mode="markers",
        marker=dict(color=cmap[c], symbol="circle"),
        name=f"{c}",
        legendgroup="color",
        legendgrouptitle_text="Color" if i == 0 else None,
        showlegend=True
    ))

for i, s in enumerate(symbols):
    # Symbol legend entries
    fig.add_trace(go.Scatterpolar(
        r=[None], theta=[None],
        mode="markers",
        marker=dict(color="black", symbol=smap[s]),
        name=f"{s}",
        legendgroup="symbol",
        legendgrouptitle_text="Symbol" if i == 0 else None,
        showlegend=True
    ))

    
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
    fname = os.path.join("fig5", 'phase_shift_and_oscillation_amplitude.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)
    fname = Path(fname).with_suffix('.html')
    fig.write_html(fname)
    


# In[13]:


_df = df_params_full[df_params_full['parameter'] == 'eigenworm3_coefficient'].copy()

neurons_to_plot = df_params.loc[idx_final, 'neuron_name']
_df = _df[_df['neuron_name'].isin(neurons_to_plot)]

order = [val for i, val in enumerate(_df.groupby('Neuron Name')['value'].median().sort_values().index)]

fig = px.box(_df,
      x='Neuron Name',
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

fig.update_traces(
    marker=dict(
        size=3,
    )
)
# fig.update_xaxes(title="Neuron Name")
fig.update_yaxes(title="EW 3 Coefficient")
# fig.show()

to_save = True
if to_save:
    fname = os.path.join("fig5", 'ew3_boxplot.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)


# In[14]:


_df = df_params_full[df_params_full['parameter'] == 'eigenworm4_coefficient'].copy()

neurons_to_plot = df_params.loc[idx_final, 'neuron_name']
_df = _df[_df['neuron_name'].isin(neurons_to_plot)]

order = [val for i, val in enumerate(_df.groupby('Neuron Name')['value'].median().sort_values().index)]

fig = px.box(_df,
      x='Neuron Name',
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

fig.update_traces(
    marker=dict(
        size=3,
    )
)
fig.update_xaxes(title="Neuron Name")
fig.update_yaxes(title="EW 4 Coefficient")
# fig.show()

to_save = True
if to_save:
    fname = os.path.join("fig5", 'ew4_boxplot.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)


# In[ ]:





# ## Supp

# In[15]:


df_params_full.head()


# In[16]:


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


# In[17]:


all_results


# In[18]:


# _df = df_params_full[df_params_full['parameter'] == 'phase_shift']

fig = px.box(_df[_df['neuron_name'].isin(neurons_to_test)], 
            x='Neuron Name', y='value',
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
    fname = os.path.join("fig5", 'phase_shift_ttests.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)
    fname = Path(fname).with_suffix('.html')
    fig.write_html(fname)


# In[19]:


# Build a dictionary mapping each facet to its x categories
category_orders = {
    facet: _df.loc[_df['phase_shift_set'] == facet, 'neuron_name'].unique().tolist()
    for facet in _df['phase_shift_set'].unique()
}

fig = px.box(_df[_df['neuron_name'].isin(all_neurons_to_test2)], 
             x='Neuron Name', y='value',
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
    fname = os.path.join("fig5", 'phase_shift_ttests_pairs.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)
    fname = Path(fname).with_suffix('.html')
    fig.write_html(fname)


# In[20]:


eps = 1e-3
# _df = df_params[df_params['Category'] == 'Hierarchy']
_df = df_params
y = -np.log(_df['hyper_pca0_p_value_corrected'] + eps)

text = _df['Neuron Name'].copy()
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

get_nonoverlapping_text_positions(x, y, text, fig, weight=10, k=1, add_nodes_with_no_text=False,
                                 # x_range=[-0.4, 0.3], 
                                  # y_range=[0, 8], 
                                  size=12
                                 )
# fig.update_layout(showlegend=False)
fig.update_xaxes(title=r"$\text{PC1 coefficient} (A_1)$")
fig.update_yaxes(title="-log(p value)")

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
    fname = os.path.join("fig5", 'pc0_coefficient.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)
    fname = Path(fname).with_suffix('.html')
    fig.write_html(fname)
    


# In[21]:


eps = 1e-6
# _df = df_params[df_params['Category'] == 'Hierarchy']
_df = df_params
y = -np.log(_df['hyper_pca1_p_value_corrected'] + eps)
x = _df['hyper_pca1']  # Was previously reusing x (PC1) from the cell above

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
fig.update_xaxes(title=r"$\text{PC2} (A_2)$")
fig.update_yaxes(title="-log(p value)")
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
    fname = os.path.join("fig5", 'pc1_coefficient.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)
    fname = Path(fname).with_suffix('.html')
    fig.write_html(fname)


# In[22]:


from wbfm.utils.external.utils_plotly import add_trendline_annotation, extend_trendline

# Manifold explained x axis
# x, y = 'manifold_variance', 'Relative Hierarchy Score'
x, y = 'manifold_variance', 'log_hyper_beta'

# idx = np.logical_and(df_params['Dataset Type'] == 'Freely Moving (GCaMP, residual)', ~np.isnan(df_params[y]))
idx = np.logical_and(df_params['Category_new'] == 'Hierarchy > GFP and PC1 is significant', ~np.isnan(df_params[y]))
_df = df_params[idx]

text = df_params['Neuron Name'].copy()
text[:] = ''
text[idx_final] = df_params['Neuron Name']
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
                 trendline='ols', 
                 # marginal_y='rug'
                )

fig.data[1].update(line=dict(color="black", dash='dot'))
get_nonoverlapping_text_positions(_df[x], y_vals, text, fig, weight=100, k=1, add_nodes_with_no_text=False,
                                 x_range=[0, 1], y_range=[0, 1], 
                                  size=12
                                 )

fig.update_xaxes(title='Variance Explained by Manifold')
fig.update_yaxes(title=r'$\beta$')  # Plots exp(log_hyper_beta), i.e. beta (was mislabeled gamma)

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
extend_trendline(fig, x_min=0, x_max=1)
apply_figure_settings(fig, width_factor=0.5, height_factor=0.15)

fig.show()

to_save = True
if to_save:
    fname = os.path.join("fig5", 'hierarchy_behavior_score_and_manifold_variance.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)


# In[ ]:





# In[23]:


# eps = 1e-6

# _df = df_params_full[df_params_full['parameter'] == 'log_hyper_beta'].copy()
# # _df_color = df_params_full[df_params_full['parameter'] == 'log_amplitude_mu'].copy()

# idx = df_params['Category_new'] == 'PC1 is significant'

# neurons_to_plot = df_params.loc[idx, 'neuron_name']
# _df = _df[_df['neuron_name'].isin(neurons_to_plot)]
# _df_color = _df_color[_df_color['neuron_name'].isin(neurons_to_plot)]

# # idx = _df['value'].reset_index(drop=True).sort_values().index
# idx = _df['neuron_name'].map(
#     _df.groupby('neuron_name')['value']
#        .median()
#        .sort_values()
#        .reset_index()
#         .reset_index()
#         .set_index('neuron_name')['index']
# ).sort_values().index
# # idx = idx[:100]

# x = _df.loc[idx, 'neuron_name']
# y = list(np.exp(_df.loc[idx, 'value']))

# # Plotly boxplots can only do discrete colors, so bin these floats
# # color = list(np.exp(_df_color.reset_index(drop=True).loc[idx, 'value']))
# _df_color = df_params[df_params['Category_new'] == 'PC1 is significant'].copy()
# _df_color['gamma'] = np.exp(_df_color['log_amplitude_mu'])

# _df_color['color_binned'] = pd.cut(
#     _df_color['gamma'],
#     bins=5,
#     # labels=False
# )
# color_mapping = dict(zip(_df_color['neuron_name'], _df_color['color_binned']))
# color_plotly = x.map(color_mapping)

# fig = px.box(x=x,
#              y=y,
#              # color=color_plotly,
#             boxmode='overlay',
#              color_discrete_sequence=[px.colors.qualitative.D3[0]]
#              # color_discrete_sequence=px.colors.sequential.Greens
# )

# fig.update_traces(marker=dict(opacity=1))

# # fig.update_layout(legend=dict(
# #         yanchor="top",
# #         y=1,
# #         xanchor="left",
# #         x=0.05,
# #                       title="Sigmoid region",
# #                       bordercolor="Black", borderwidth=2,# bgcolor = 'white'
# #     ))
# # fig.update_traces(name="High Sigmoid", selector=dict(name="#1F77B4"))
# # fig.update_traces(name="Low Sigmoid", selector=dict(name="#2CA02C"))

# fig.update_xaxes(title="Neuron")#($\gamma$/$\beta$)")
# fig.update_yaxes(title="Modulated behavior amplitude")
# fig.update_xaxes(
#     type='category',
#     tickmode='linear'
# )

# # apply_figure_settings(fig, width_factor=0.3, height_factor=0.3)
# apply_figure_settings(fig, width_factor=0.3, height_factor=0.15)

# fig.show()

# to_save = True
# if to_save:
#     fname = os.path.join("fig5", 'behavior_change_boxplot.png')
#     fig.write_image(fname, scale=3)
#     fname = Path(fname).with_suffix('.svg')
#     fig.write_image(fname)


# In[24]:


# eps = 1e-6
# _df = df_params[df_params['Category_new'] == 'PC1 is significant'].copy()

# idx = _df['log_hyper_beta'].sort_values().index

# y = list(np.exp(_df.loc[idx, 'log_hyper_beta']))
# # y.extend(list(np.exp(_df.loc[idx, 'log_hyper_beta']) + np.exp(_df.loc[idx, 'log_amplitude_mu'])))
# # x = list(_df.loc[idx, 'neuron_name']) * 2

# color = [px.colors.qualitative.D3[2]] * len(_df) + [px.colors.qualitative.D3[0]] * len(_df)
# # _df['color'] = color

# fig = px.scatter(x=_df.loc[idx, 'neuron_name'],
#                  y=y,
#                  # category_orders=order,
#                  # color=color,
#                  # color_discrete_sequence=[px.colors.qualitative.D3[2], px.colors.qualitative.D3[0]],
#                  # text='text_complete',#df_params.index,
#                  # hover_data='Category',
#                  # color_discrete_map=color_discrete_map, 
#                  # color='Category_new'
#                 )

# fig.update_layout(legend=dict(
#         yanchor="top",
#         y=1,
#         xanchor="left",
#         x=0.05,
#                       title="Sigmoid region",
#                       bordercolor="Black", borderwidth=2,# bgcolor = 'white'
#     ))
# # fig.update_traces(name="High Sigmoid", selector=dict(name="#1F77B4"))
# # fig.update_traces(name="Low Sigmoid", selector=dict(name="#2CA02C"))

# fig.update_xaxes(title="Neuron")#($\gamma$/$\beta$)")
# fig.update_yaxes(title="Modulated behavior amplitude")
# fig.update_xaxes(
#     type='category',
#     tickmode='linear'
# )

# apply_figure_settings(fig, width_factor=0.3, height_factor=0.3)

# fig.show()


# to_save = False
# if to_save:
#     fname = os.path.join("fig5", 'behavior_change_sigmoid_simple2.png')
#     fig.write_image(fname, scale=3)
#     fname = Path(fname).with_suffix('.svg')
#     fig.write_image(fname)
#     fname = Path(fname).with_suffix('.html')
#     fig.write_html(fname)


# In[25]:


# eps = 1e-6
# _df = df_params[df_params['Category'] == 'Hierarchy']
# y = -np.log(_df['log_amplitude_mu_p_value_corrected'] + eps)
# x = np.exp(_df['log_amplitude_mu'])
# fig = px.scatter(_df,
#                  x=x,
#                  y=y,
#                  text='text_complete',#df_params.index,
#                  # hover_data='Category',
#                  color_discrete_map=plotly_paper_color_discrete_map(), 
#                  color='log_amplitude_sigma'
#                 )
# fig.add_hline(y=-np.log(0.05))
# fig.update_xaxes(title="Suppression/gating index")#($\gamma$/$\beta$)")

# fig.show()


# ## Alt

# In[26]:


# # idx = np.logical_or(df_params['Category_raw'].isin(['Hierarchical Behavior', 'Behavior only']),
#                    # df_params['hyper_pca0_p_value_is_significant'])
# _df = df_params[idx_final]
# # _df = df_params[df_params['Category'] == 'Hierarchy']
# # _df = df_params[df_params['hyper_pca0_p_value_is_significant']]

# r = np.exp(_df['log_hyper_beta']) + np.exp(_df['log_amplitude_mu'])

# idx = df_params['Category_new'] == 'PC1 not significant, but behavior is good'
# r[idx] = np.exp(_df['log_amplitude_mu']) + 0.5*np.exp(_df['log_amplitude_mu'])

# cmap = px.colors.qualitative.D3.copy()

# fig = px.scatter_polar(_df, r=r,#'r', 
#                        log_r=True,
#                        theta='phase_shift', direction='counterclockwise', start_angle=0,
#                        text='text_complete',
#                        # color='Category_new', 
#                        # color_discrete_map=color_discrete_map,
#                        symbol='Category_new',
#                        color='Neuron Type',
#                        color_discrete_map=plotly_paper_color_discrete_map(),
#                        size='size', size_max=15, #log_r=True,
#     )
        
# fig.update_traces(
#     marker=dict(
#         size=15,
#         line=dict(
#             width=2,
#             color="black"
#         )
#     )
# )
# fig.update_traces(thetaunit='radians', textposition='top center', textfont_size=12, )

# # apply_figure_settings(fig, width_factor=0.4, height_factor=0.4)
# apply_figure_settings(fig, width_factor=0.7, height_factor=0.7)

# fig.update_layout(polar=dict(
#     angularaxis = dict(thetaunit = "radians"),
#     radialaxis = dict(#range=[0, 0.6],
#                       nticks=3)
# ), 
#                   showlegend=True, 
#                   legend=dict(
#         yanchor="top",
#         y=0.4,
#         xanchor="left",
#         x=0.45,
#                       title="Hierarchical encoding",
#                       bordercolor="Black", borderwidth=2, bgcolor = 'white'
#     )
#                  )

# fig.show()

# # to_save = True
# # if to_save:
# #     fname = os.path.join("fig5", 'phase_shift_and_oscillation_amplitude.png')
# #     fig.write_image(fname, scale=3)
# #     fname = Path(fname).with_suffix('.svg')
# #     fig.write_image(fname)
# #     fname = Path(fname).with_suffix('.html')
# #     fig.write_html(fname)
    


# In[27]:


# y, x = 'Relative Hierarchy Score (raw)', 'Behavior Score (raw)'

# _y_max_gfp = df_params[df_params['Dataset Type'] == 'Freely Moving (GFP, residual)'][y].max()
# _x_max_gfp = df_params[df_params['Dataset Type'] == 'Freely Moving (GFP, residual)'][x].max()


# fig = px.scatter(df_params, 
#                  y=df_params[y]+1e-1, x=df_params[x]+1e-1,
#                  log_y=True, log_x=True,
#                  # color='hyper_pca0_p_value_is_significant',
#                  color='Category_new',
#                  color_discrete_map=color_discrete_map,
#                  category_orders={'Category':['PC1 not significant', 'PC1 not significant, but behavior is good', 'PC1 is significant']},
#                  # symbol='Category_new',
#                  # size_max=50,
#                  hover_data=['text_raw', 'Dataset Type'],
#                 )
# opt = dict(line=dict(dash='dot'), col=1)
# fig.add_hline(y=_y_max_gfp, **opt)
# fig.add_vline(x=_x_max_gfp, **opt)


# text = df_params['text_complete'].copy()
# idx = df_params['Category_new'] == 'PC1 not significant'
# text[idx] = ''

# # get_nonoverlapping_text_positions(df_params[x], df_params[y], text, fig, weight=10, k=10, add_nodes_with_no_text=False,
# #                                  # x_range=[-5, 40], y_range=[-5, 40], 
# #                                   size=12
# #                                  )
# # fig.update_layout(showlegend=False)
# fig.update_layout(
#                   showlegend=True, 
#                   legend=dict(
#         yanchor="top",
#         y=1.4,
#         xanchor="left",
#         x=-0.0,
#                       title="Category",
#                       bordercolor="Black", borderwidth=2, #bgcolor = 'white'
#     )
#                  )

# fig.update_traces(
#     marker=dict(
#         size=10,
#         line=dict(
#             width=2,
#             color="black"
#         )
#     )
# )
# # z-order
# # Put a specific trace on top
# fig.data = tuple(
#     sorted(fig.data, key=lambda t: t.name != "PC1 not significant")
# )

# apply_figure_settings(fig, width_factor=0.4, height_factor=0.4)
# fig.show()


# to_save = True
# if to_save:
#     fname = os.path.join("fig5", 'model_improvement_raw.png')
#     fig.write_image(fname, scale=3)
#     fname = Path(fname).with_suffix('.svg')
#     fig.write_image(fname)
#     fname = Path(fname).with_suffix('.html')
#     fig.write_html(fname)
    


# In[28]:


# eps = 1e-6
# _df = df_params[idx_final].copy()

# idx = _df['log_hyper_beta'].sort_values().index

# y = list(np.exp(_df.loc[idx, 'log_amplitude_mu']))
# y.extend(list(np.exp(_df.loc[idx, 'log_hyper_beta'])))
# # y.extend(list(np.exp(_df.loc[idx, 'log_hyper_beta']) + np.exp(_df.loc[idx, 'log_amplitude_mu'])))
# x = list(_df.loc[idx, 'neuron_name']) * 2

# color = [px.colors.qualitative.D3[2]] * len(_df) + [px.colors.qualitative.D3[0]] * len(_df)
# # _df['color'] = color

# fig = px.scatter(x=x,
#                  y=y,
#                  # category_orders=order,
#                  color=color,
#                  facet_row=color,
#                  color_discrete_sequence=[px.colors.qualitative.D3[2], px.colors.qualitative.D3[0]],
#                 )

# fig.update_layout(showlegend=False, legend=dict(
#         yanchor="top",
#         y=1,
#         xanchor="left",
#         x=0.05,
#         title="Sigmoid region",
#         bordercolor="Black", borderwidth=2,# bgcolor = 'white'
# ))
# fig.update_traces(name="Beta", selector=dict(name="#1F77B4"))
# fig.update_traces(name="Gamma", selector=dict(name="#2CA02C"))

# fig.update_traces(
#     marker=dict(
#         size=10,
#         line=dict(
#             width=2,
#             color="black"
#         )
#     )
# )
# # fig.add_hline(y=0)
# fig.update_xaxes(title="Neuron", row=2)
# fig.update_yaxes(title="Beta", row=1)
# fig.update_yaxes(title="Gamma", row=2)
# fig.update_xaxes(
#     type='category',
#     tickmode='linear'
# )

# apply_figure_settings(fig, width_factor=0.4, height_factor=0.5)

# fig.show()

# to_save = True
# if to_save:
#     fname = os.path.join("fig5", 'behavior_change_sigmoid_centroid.png')
#     fig.write_image(fname, scale=3)
#     fname = Path(fname).with_suffix('.svg')
#     fig.write_image(fname)
#     fname = Path(fname).with_suffix('.html')
#     fig.write_html(fname)


# In[29]:



_df = df_params[idx_final].copy()
only_beh_neurons = _df[_df['Category_new']== 'Behavior is above GFP']['neuron_name'].values
idx_beta_sorted = _df['log_hyper_beta'].sort_values().index
neurons_beta_sorted = _df.loc[idx_beta_sorted, 'neuron_name']
map_beta_sorted = dict(zip(neurons_gamma_sorted.values, neurons_gamma_sorted.index))
idx_gamma_sorted = _df['log_amplitude_mu'].sort_values().index
neurons_gamma_sorted = _df.loc[idx_gamma_sorted, 'neuron_name']
map_gamma_sorted = dict(zip(neurons_gamma_sorted.values, neurons_gamma_sorted.index))

rank = {v: i for i, v in enumerate(neurons_sorted.values)}


# In[ ]:


map_beta_sorted


# In[ ]:


x, y, z = 'null_normalized', 'hierarchical_pca_normalized', 'nonhierarchical_normalized'

x, y, z = df_params[x], df_params[y], df_params[z]

fig = px.scatter(df_params, 
                 y=y-z, x=z, #range_x=[-2, 60],
                 text=df_params['text'],# if display_text else None, 
                 color='Category',
                 color_discrete_map=plotly_paper_color_discrete_map(), 
                 symbol='Dataset Type',
                 size_max=20,
                 hover_data=['text_raw'],
                )
# xy_max = np.min([df_params[x].max(), df_params[y].max()])
# fig.add_shape(type="line",
#             x0=-1.5, y0=-1.5,  # start of the line (bottom of the plot)
#             x1=-1, y1=-1,  # end of the line (top of the plot)
#             line=dict(color="black", width=1, dash="dash"),
#             xref='x',
#             yref='y')
# fig.add_hline(y=y_max_gfp)
fig.show()


# In[ ]:


# px.scatter(df_params,
#           y='Relative Hierarchy Score',
#           x='Behavior Score',
#           color='null_normalized',
#           text='text_complete')


# In[ ]:


# px.scatter(df_params,
#           y='Relative Hierarchy Score',
#           x='Behavior Score',
#           color='hierarchical_pca_normalized',
#           text='text_complete')


# In[ ]:


# eps = 1e-6
# _df = df_params[df_params['Category'] == 'Hierarchy']
# y = -np.log(_df['hyper_pca0_p_value_corrected'] + eps)
# x = np.exp(_df['log_amplitude_mu']) / np.exp(_df['log_hyper_beta'])
# fig = px.scatter(_df,
#                  x=x,
#                  y=y,
#                  text='text_complete',#df_params.index,
#                  # hover_data='Category',
#                  color_discrete_map=plotly_paper_color_discrete_map(), 
#                  # color='log_hyper_beta_sigma'
#                 )
# fig.add_hline(y=-np.log(0.05))
# fig.update_xaxes(title="Suppression/gating index")#($\gamma$/$\beta$)")

# fig.show()


# In[ ]:


# idx = np.logical_or(df_params['Category_raw'].isin(['Hierarchical Behavior', 'Behavior only']),
#                    df_params['hyper_pca0_p_value_is_significant'])
# _df = df_params[idx]
# # _df = df_params

# eps = 1e-3
# # _df = df_params[df_params['Category'].isin(['Hierarchy', 'Behavior only'])]
# y = -np.log(_df['eigenworm3_coefficient_p_value_corrected'] + eps)
# x = _df['eigenworm3_coefficient']
# fig = px.scatter(_df,
#                  x=x,
#                  y=y,
#                  # text='text_complete',#df_params.index,
#                  # hover_data='Category',
#                  color='Category_new', 
#                  # color_discrete_sequence=px.colors.qualitative.G10,
#                  color_discrete_map=color_discrete_map, 
#                  # color='Category'
#                 )
# fig.add_hline(y=-np.log(0.05))
# fig.update_layout(showlegend=False)
# fig.update_xaxes(title="eigenworm3_coefficient")#($\gamma$/$\beta$)")

# get_nonoverlapping_text_positions(x, y, _df['text'], fig, weight=1, k=None, add_nodes_with_no_text=False,
#                                  # x_range=[-0.4, 0.3], 
#                                   # y_range=[0, 8], 
#                                   size=12
#                                  )

# apply_figure_settings(fig, width_factor=0.5, height_factor=0.3)
# fig.show()

# to_save = True
# if to_save:
#     fname = os.path.join("fig5", 'eigenworm3.png')
#     fig.write_image(fname, scale=3)
#     fname = Path(fname).with_suffix('.svg')
#     fig.write_image(fname)
#     fname = Path(fname).with_suffix('.html')
#     fig.write_html(fname)


# In[ ]:


# idx = np.logical_or(df_params['Category_raw'].isin(['Hierarchical Behavior', 'Behavior only']),
#                    df_params['hyper_pca0_p_value_is_significant'])
# _df = df_params[idx]
# # _df = df_params
# # _df = df_params[df_params['Category'].isin(['Hierarchy', 'Behavior only'])]
# eps = 1e-3
# # _df = df_params[df_params['Category'] == 'Hierarchy']
# y = -np.log(_df['eigenworm4_coefficient_p_value_corrected'] + eps)
# x = _df['eigenworm4_coefficient']
# fig = px.scatter(_df,
#                  x=x,
#                  y=y,
#                  # text='text_complete',#df_params.index,
#                  # hover_data='Category',
#                  color_discrete_map=plotly_paper_color_discrete_map(), 
#                  color='Category'
#                 )
# fig.add_hline(y=-np.log(0.05))
# fig.update_xaxes(title="eigenworm4_coefficient")#($\gamma$/$\beta$)")

# get_nonoverlapping_text_positions(x, y, _df['text'], fig, weight=1, k=1, add_nodes_with_no_text=False,
#                                  # x_range=[-0.4, 0.3], 
#                                   # y_range=[0, 8], 
#                                   size=12
#                                  )

# apply_figure_settings(fig, width_factor=0.5, height_factor=0.3)
# fig.show()

# to_save = True
# if to_save:
#     fname = os.path.join("fig5", 'eigenworm4.png')
#     fig.write_image(fname, scale=3)
#     fname = Path(fname).with_suffix('.svg')
#     fig.write_image(fname)
#     fname = Path(fname).with_suffix('.html')
#     fig.write_html(fname)


# In[ ]:


df_params['Category_new'].unique()


# In[ ]:


idx = df_params['Category_new'] != 'PC1 not significant'
df_params.loc[idx, ['phase_shift_var', 'neuron_name', 'Category_new']]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Load model results

# In[ ]:


from wbfm.utils.external.utils_plotly import get_nonoverlapping_text_positions


# In[ ]:


# suffix = '_original'
suffix = '_pca_global'
# suffix = '_new_ids_only_eigenworms'
# suffix = '_eigenworms34_speed'
# suffix = '_only_eigenworms'

# Load data from many dataframes
output_dir = '/lisc/data/scratch/neurobiology/zimmer/fieseler/paper/hierarchical_modeling/output' + suffix
# output_dir = os.path.join(get_hierarchical_modeling_dir(), 'output')
_all_dfs = {}
for filename in tqdm(Path(output_dir).iterdir()):
    if filename.name.endswith('.h5') and 'single' not in filename.name:
        neuron_name = '_'.join(filename.name.split('_')[:-1])
        _all_dfs[neuron_name] = pd.read_hdf(filename)
df = pd.concat(_all_dfs).reset_index(names=['neuron_name', 'model_type'])


# In[ ]:


# df


# In[ ]:


# Also load gfp
output_dir = '/lisc/data/scratch/neurobiology/zimmer/fieseler/paper/hierarchical_modeling_gfp/output' + suffix

# output_dir = os.path.join(get_hierarchical_modeling_dir(gfp=True), 'output')
all_dfs_gfp = {}
for filename in tqdm(Path(output_dir).iterdir()):
    if filename.name.endswith('.h5') and 'single' not in filename.name:
        neuron_name = '_'.join(filename.name.split('_')[:-1])
        all_dfs_gfp[neuron_name] = pd.read_hdf(filename)
df_gfp = pd.concat(all_dfs_gfp).reset_index(names=['neuron_name', 'model_type'])


# In[ ]:


df_gfp[df_gfp['neuron_name']=='RID']


# In[ ]:


# px.scatter(df_gfp, color='neuron_name', #x='neuron_name', 
#            x='dse',
#            y='elpd_diff', facet_row='model_type', 
#            height=1000)


# ### Load CV results

# In[ ]:


# suffix = '_cv'
# # suffix = '_cv_time_series'

# # Load data from many dataframes
# output_dir = '/lisc/data/scratch/neurobiology/zimmer/fieseler/paper/hierarchical_modeling/output' + suffix
# # output_dir = os.path.join(get_hierarchical_modeling_dir(), 'output')
# all_dfs = {}
# for filename in tqdm(Path(output_dir).iterdir()):
#     if filename.name.endswith('.h5') and 'single' not in filename.name and 'compare' in filename.name:
#         neuron_name = '_'.join(filename.name.split('_')[:-1])
#         all_dfs[neuron_name] = pd.read_hdf(filename)
# df_cv = pd.concat(all_dfs).reset_index(names=['neuron_name', 'model_type'])

# # Also load gfp
# output_dir = '/lisc/data/scratch/neurobiology/zimmer/fieseler/paper/hierarchical_modeling_gfp/output' + suffix

# # output_dir = os.path.join(get_hierarchical_modeling_dir(gfp=True), 'output')
# all_dfs_gfp = {}
# for filename in tqdm(Path(output_dir).iterdir()):
#     if filename.name.endswith('.h5') and 'single' not in filename.name and 'compare' in filename.name:
#         neuron_name = '_'.join(filename.name.split('_')[:-1])
#         all_dfs_gfp[neuron_name] = pd.read_hdf(filename)
# df_cv_gfp = pd.concat(all_dfs_gfp).reset_index(names=['neuron_name', 'model_type'])


# # Plot model comparison statistics

# In[ ]:


# What I want to plot:
# x = scaled difference between the null and non-hierarchical model
# y = same but for hierarchical
from wbfm.utils.general.utils_paper import apply_figure_settings, plotly_paper_color_discrete_map, data_type_name_mapping, package_bayesian_df_for_plot
from wbfm.utils.general.utils_hardcoded import neurons_with_confident_ids

# df_to_plot_gcamp = package_bayesian_df_for_plot(df, relative_improvement=False).assign(datatype='Freely Moving (GCaMP, residual)')
# df_to_plot_gfp = package_bayesian_df_for_plot(df_gfp, relative_improvement=False).assign(datatype='Freely Moving (GFP, residual)')
df_to_plot_gcamp = package_bayesian_df_for_plot(df, df_normalization=Xy, 
                                                # min_num_datapoints=10000
                                               ).assign(datatype='Freely Moving (GCaMP, residual)')
df_to_plot_gfp = package_bayesian_df_for_plot(df_gfp, df_normalization=Xy_gfp, 
                                              # min_num_datapoints=5000
                                             ).assign(datatype='Freely Moving (GFP, residual)')

df_to_plot = pd.concat([df_to_plot_gcamp, df_to_plot_gfp])
df_to_plot['Dataset Type'] = df_to_plot['datatype']
df_to_plot['Size'] = 1


# In[ ]:


import plotly.graph_objects as go
from wbfm.utils.general.utils_paper import plot_bayesian_model_comparison

output_folder = 'fig5'


# In[ ]:


# y, x = 'Hierarchy Score', 'Relative Hierarchy Score'
# fig, df_to_plot_with_var, text = plot_bayesian_model_comparison(x, y, df_to_plot_gfp, df_to_plot_gcamp, 
#                                                                 display_text=True, remove_names_of_ns=False,
#                                                output_folder=output_folder)


# In[ ]:


# y, x = 'Relative Hierarchy Score', 'Behavior Score'
# fig, df_to_plot_with_var, text = plot_bayesian_model_comparison(x, y, df_to_plot_gfp, df_to_plot_gcamp, 
#                                                                 display_text=True, remove_names_of_ns=False,
#                                                output_folder=output_folder)


# In[ ]:


# y, x = 'Relative Hierarchy Score', 'Behavior Score'
# fig, df_to_plot_with_var, text = plot_bayesian_model_comparison(x, y, df_to_plot_gfp, df_to_plot_gcamp, 
#                                                                 display_text=True, remove_names_of_ns=True,
#                                                output_folder=output_folder)


# In[ ]:


# y, x = 'Hierarchy Score', 'Behavior Score'
# fig, _df, text = plot_bayesian_model_comparison(x, y, df_to_plot_gfp, df_to_plot_gcamp, display_text=False,
#                                                output_folder=None)

# get_nonoverlapping_text_positions(_df[x], _df[y], text, fig, weight=1, k=4, add_nodes_with_no_text=True,
#                                  x_range=[0, 29], size=12)

# fig.update_xaxes(title='Behavior-only Model Performance')
# fig.update_yaxes(title='Hierarchy Model Performance')

# fname = os.path.join(output_folder, f'x-{x}_y-{y}-annotations.png')
# fig.write_image(fname, scale=3)
# fname = Path(fname).with_suffix('.svg')
# fig.write_image(fname)
# fname = Path(fname).with_suffix('.html')
# fig.write_html(fname)

# fig.show()


# In[ ]:


# # Same but with ALL names
# y, x = 'hierarchical_pca_normalized', 'null_normalized'
# fig, _df, text = plot_bayesian_model_comparison(x, y, df_to_plot_gfp, df_to_plot_gcamp ,
#                             display_text=True, to_show=True, remove_names_of_ns=False)


# In[ ]:


# # Same but with ALL names
# y, x = 'Hierarchy Score', 'Behavior Score'
# fig, _df, text = plot_bayesian_model_comparison(x, y, df_to_plot_gfp, df_to_plot_gcamp ,
#                             display_text=True, to_show=True, remove_names_of_ns=False)


# ## Supplement with alternate axes: manifold variance

# In[ ]:




# def calc_var_ratio(Xy):
#     Xy_var = Xy.groupby('dataset_name').var()
#     Xy_var_ratio = {}
#     for col in Xy_var.columns:
#         if 'neuron' in col:
#             continue
#         col_manifold = f'{col}_manifold'
#         if col_manifold in Xy_var.columns:
#             Xy_var_ratio[col] = Xy_var[col_manifold] / Xy_var[col]
#     Xy_var_ratio = pd.DataFrame(Xy_var_ratio)
#     return Xy_var_ratio
    
# df_var_exp_gcamp = calc_var_ratio(Xy)
# df_var_exp_gcamp['Dataset Type'] = 'Freely Moving (GCaMP, residual)'
# df_var_exp_gfp = calc_var_ratio(Xy_gfp)
# df_var_exp_gfp['Dataset Type'] = 'Freely Moving (GFP, residual)'
# df_var_exp = pd.concat([df_var_exp_gcamp, df_var_exp_gfp], axis=0)
    
# # px.box(df_var_exp.dropna(thresh=3, axis=1), color='Dataset Type')


# In[ ]:



# df_var_exp_median = df_var_exp.groupby('Dataset Type').median().reset_index().melt(
#     id_vars='Dataset Type', var_name='neuron_name', value_name='manifold_variance')
# df_to_plot_with_var = df_to_plot.merge(df_var_exp_median, on=['neuron_name', 'Dataset Type'])

# # df_to_plot_with_var


# In[ ]:


# x, y = 'manifold_variance', 'Relative Hierarchy Score'

# # Define threshold(s)
# # x_max_gfp = df_to_plot_gfp[x].max()
# y_max_gfp = df_to_plot_gfp[y].max()
# print('GFP thresholds: ', y_max_gfp)
# df_to_plot_with_var['above_gfp'] = df_to_plot_with_var[y] > y_max_gfp
# df_to_plot_with_var['text_simple'] = df_to_plot_with_var['text']
# df_to_plot_with_var.loc[~df_to_plot_with_var['above_gfp'], 'text_simple'] = ''

# # Remove or include temporary names
# df_to_plot_with_var = df_to_plot_with_var.loc[df_to_plot_with_var['neuron_name'].isin(neurons_with_confident_ids())].copy()
# df_to_plot_with_var = df_to_plot_with_var.loc[df_to_plot_with_var['neuron_name'] != '']

# # Only include neurons that make it above the prior threshold, but not GFP ones
# cols_to_plot = ['Hierarchy only', 'Hierarchical Behavior']
# _neurons_to_plot = df_to_plot_with_var['neuron_name'][df_to_plot_with_var['Category'].isin(cols_to_plot)]
# _df_to_plot_with_var = df_to_plot_with_var.loc[df_to_plot_with_var['neuron_name'].isin(_neurons_to_plot)]
# _df_to_plot_with_var = _df_to_plot_with_var.loc[df_to_plot_with_var['Dataset Type'].isin(['Freely Moving (GCaMP, residual)'])]


# # Actual plot
# fig = px.scatter(_df_to_plot_with_var, 
#                  y=y, x=x,
#                  # text='neuron_name', 
#                  text='text', 
#                  color='Dataset Type',
#                 color_discrete_map=plotly_paper_color_discrete_map(), size='Size', size_max=10,
#                  # marginal_y='rug'
#                 )
# # fig.add_shape(type="line",
# #               x0=0, y0=y_max_gfp,  # start of the line (bottom of the plot)
# #               x1=1, y1=y_max_gfp,  # end of the line (top of the plot)
# #               line=dict(color="black", width=2, dash="dash"),
# #               xref='paper',
# #               yref='y')
# fig.update_xaxes(title='Variance Explained by Manifold')
# fig.update_yaxes(title='Hierarchical Model Improvement<br>(compared to behavior-only model)')

# fig.update_layout(showlegend=False, 
#                   legend=dict(
#     yanchor="top",
#     y=0.99,
#     xanchor="left",
#     x=0.4
# ))

# to_save = True
# if to_save:
#     apply_figure_settings(fig, width_factor=0.6, height_factor=0.3)
#     fname = os.path.join("/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/bayesian_modeling/plots", 'hierarchy_behavior_score_and_manifold_variance-no_text.png')
#     fig.write_image(fname, scale=3)
#     fname = Path(fname).with_suffix('.svg')
#     fig.write_image(fname)

# # text = _df_to_plot_with_var['text']
# # get_nonoverlapping_text_positions(_df_to_plot_with_var[x], _df_to_plot_with_var[y], text, fig, #weight=1, 
# #                                   k=0.2, 
# #                                   #add_nodes_with_no_text=True, 
# #                                   size=12, #x_range=[-0.2, 1.2]
# #                                  )

# apply_figure_settings(fig, width_factor=0.45, height_factor=0.45)


# fig.show()


# to_save = True
# if to_save:
#     fname = os.path.join("/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/bayesian_modeling/plots", 'hierarchy_behavior_score_and_manifold_variance.png')
#     # fname = os.path.join("/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/presentations_and_grants/CSH", 'hierarchy_behavior_score_with_gfp.png')
#     fig.write_image(fname, scale=3)
#     fname = Path(fname).with_suffix('.svg')
#     fig.write_image(fname)


# ## Same, but no text

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
    


# # Additional subplots: model parameters

# In[ ]:


import arviz as az
from wbfm.utils.general.utils_hardcoded import neurons_with_confident_ids, role_of_neuron_dict


# In[ ]:


def load_all_traces(foldername):
    fnames = neurons_with_confident_ids()
    all_traces = {}
    for neuron in tqdm(fnames):
        trace_fname = os.path.join(foldername, f'{neuron}_hierarchical_pca_trace.nc')
        if os.path.exists(trace_fname):
            try:
                trace = az.from_netcdf(trace_fname)
                all_traces[neuron] = trace
            except (ValueError, OSError, KeyError) as e:
                print(f"Error for neuron {neuron}; this is not surprising if some are still being written: {e}")
    return all_traces

parent_folder = '/lisc/data/scratch/neurobiology/zimmer/fieseler/paper/hierarchical_modeling'
# suffix = '_only_eigenworms'
# suffix = '_eigenworms34_speed'
# suffix = '_original'
suffix = '_pca_global'
# suffix = '_new_ids_only_eigenworms'
            
foldername = os.path.join(parent_folder, f'output{suffix}')
all_traces_gcamp = load_all_traces(foldername)

# foldername = os.path.join(f'{parent_folder}_gfp', f'output{suffix}')
# all_traces_gfp = load_all_traces(foldername)


# In[ ]:


# Also load the NMJ connectivity
# fname = '/home/charles/Current_work/repos/dlc_for_wbfm/paper/NeuronFixedPoints.xls'
# df_connect = pd.read_excel(fname)
# _df = df_connect[df_connect['Landmark'].str.contains('M')]
# muscle_position = _df.groupby('Neuron')['Landmark Position'].mean()


# In[ ]:


# Plot all that make it above the gfp line
cols_to_plot = ['Hierarchy only', 'Hierarchical Behavior']

_neurons_to_plot = df_to_plot_with_var['neuron_name'][df_to_plot_with_var['Category'].isin(cols_to_plot)]
# neurons_to_plot = neurons_with_confident_ids()
# neurons_to_plot = ['URXR', 'URXL']

# neurons_to_plot = ['SMDDL', 'SMDDR', 'VB02', 'DB01', 'DB02']
# neurons_to_plot = ['BAGL', 'BAGR', 'AVAL', 'AVAR']
# neurons_to_plot = fnames
# neurons_to_plot = ['SMDDL', 'SMDDR', 'VG_post_turning_R', 'VG_post_turning_L']
neurons_to_plot = list(set(_neurons_to_plot).intersection(set(all_traces_gcamp.keys())))

# Just plot all
neurons_to_plot = set(all_traces_gcamp.keys())

removed_neurons = set(_neurons_to_plot) - set(neurons_to_plot)
if len(removed_neurons) > 0:
    print(f"Warning: some neurons should be plotted, but do not have a stored trace: {removed_neurons}")
# df_to_plot_with_var[df_to_plot_with_var['neuron_name'] == 'AVAL']


# In[ ]:


# var_names = ["self_collision", 'speed', 'eigenworm3', 'eigenworm4', 'amplitude_mu']
var_names = ["self_collision", 'dorsal', 'ventral', 'amplitude_mu']

all_traces = all_traces_gcamp
# all_traces = all_traces_gfp

# az.plot_forest([all_traces[n] for n in neurons_to_plot], model_names=neurons_to_plot,
#                var_names=var_names, combined=True, 
#               filter_vars='like', kind='ridgeplot', figsize=(9, 7), ridgeplot_overlap=3)


# In[ ]:


# Scatter plot of median model parameters
from collections import defaultdict
import xarray as xr
from wbfm.utils.general.utils_paper_revisions_bayes import calculate_bayesian_ttests

df_params = calculate_bayesian_ttests(suffix = '_pca_global', #single_neuron=None, do_gfp=False,
                               recalculate_sigmoid=False, 
                               all_traces=all_traces, Xy=Xy, verbose=0)


# In[ ]:


df_params


# In[ ]:


fig = px.scatter_polar(df_params[df_params['Neuron Type'].str.contains('Motor')], r='r', theta='phase_shift', direction='counterclockwise', start_angle=0,
                       text='text_polar',
                       color='Neuron Type', 
                       color_discrete_map=plotly_paper_color_discrete_map(),
                       size='size', size_max=15, #log_r=True,
                       #color='Neuron Role',
                       #color='muscle_position'
                      )
fig.update_traces(thetaunit='radians', textposition='top center', textfont_size=12, )

apply_figure_settings(fig, width_factor=0.4, height_factor=0.4)
# apply_figure_settings(fig, width_factor=1.0, height_factor=1.0)

fig.update_layout(polar=dict(
    angularaxis = dict(thetaunit = "radians"),
    radialaxis = dict(range=[0, 0.6],
                      nticks=3)
), 
                  showlegend=True, 
                  legend=dict(
        yanchor="top",
        y=0.4,
        xanchor="left",
        x=0.6,
                      bordercolor="Black", borderwidth=2, bgcolor = 'white'
    )
                 )

fig.show()

to_save = False
if to_save:
    fname = os.path.join("fig5", 'phase_shift_and_oscillation_amplitude_only_motor.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)
    fname = Path(fname).with_suffix('.html')
    fig.write_html(fname)
    


# In[ ]:


# fig = px.scatter_polar(df_params, r='r', theta='phase_shift', direction='counterclockwise', start_angle=0,
#                        text='text', #'text_complete',
#                        color='Neuron Type', 
#                        color_discrete_map=plotly_paper_color_discrete_map(),
#                        size='Relative Hierarchy Score', size_max=15, #log_r=True,
#                        #color='Neuron Role',
#                        #color='muscle_position'
#                       )
# fig.update_traces(thetaunit='radians', textposition='top center', textfont_size=12, showlegend=False)

# apply_figure_settings(fig, width_factor=0.4, height_factor=0.3)
# # apply_figure_settings(fig, width_factor=1.0, height_factor=1.0)

# # Show legend, but only for reference lines
# fig.update_layout(polar=dict(
#     angularaxis = dict(thetaunit = "radians"),
#     radialaxis = dict(#range=[0, 0.5],
#                       nticks=3)
#     ), 
#     showlegend=False, 
#     legend=dict(
#         yanchor="bottom",
#         y=0.05,
#         xanchor="left",
#         x=0.5,
#         bordercolor="Black", borderwidth=1, bgcolor = 'white',
#         font=dict(size=10),
#         title='Body segment',
#         # orientation='h',
#     ))

# # Add reference lines with manual coloring to match the example worm
# cmap = ['#FF0F0F', '#CCCCCC', '#0000FF']
# for i, (theta, seg_idx) in enumerate(zip([0, 90, 180], [6, 25, 37])):
    
#     fig.add_trace(go.Scatterpolar(
#             r = [0.35,0.6],
#             theta = [theta, theta],
#             mode = 'lines',
#             line=dict(width=5, color=cmap[i]),
#             # marker=dict(symbol='arrow'),
#             name=seg_idx, opacity=1
#             # text=f'<b>{seg_idx}</b>',
#             # textposition='bottom center',
#             # textfont=dict(
#             #     size=18,
#             #     family='Arial',
#             #     color=cmap[i],
#             # ),
#         ))
#     # Add Background rectangles directly and separately to get the background color option
#     # fig.add_trace(go.Scatterpolar(
#     #     r=[r0], 
#     #     theta=[theta], 
#     #     mode='markers',
#     #     marker=dict(
#     #         size=20,  # Size of the background
#     #         color='white',  # Background color
#     #         # opacity=0.5,
#     #         symbol='square'
#     #     ),
#     #     showlegend=False
#     # ))
    

# fig.show()

# to_save = True
# if to_save:
#     fname = os.path.join("fig5", 'phase_shift_and_oscillation_amplitude.png')
#     fig.write_image(fname, scale=3)
#     fname = Path(fname).with_suffix('.svg')
#     fig.write_image(fname)
#     fname = Path(fname).with_suffix('.html')
#     fig.write_html(fname)
    


# ## Alt: volcano plots

# In[ ]:



df_params.columns


# In[ ]:


eps = 1e-6
y = -np.log(df_params['log_hyper_pca0_p_value_corrected'] + eps)
fig = px.scatter(df_params,
                 x='log_hyper_pca0',
                 y=y,
                 text='text_complete',#df_params.index,
                 # hover_data='Category',
                 # color='Category'
                )
fig.add_hline(y=-np.log(0.05))
fig.show()


# In[ ]:


eps = 1e-6
y = -np.log(df_params['pca1_amplitude_p_value_corrected'] + eps)
fig = px.scatter(df_params,
                 x='pca1_amplitude',
                 y=y,
                 text='text_complete',#df_params.index,
                 # hover_data='Category',
                 # color='Category'
                )
fig.add_hline(y=-np.log(0.05))
fig.show()


# In[ ]:


eps = 1e-6
y = -np.log(df_params['hyper_beta_p_value'] + eps)
fig = px.scatter(df_params,
                 x='hyper_beta',
                 y=y,
                 text='text_complete',#df_params.index,
                 # hover_data='Category',
                 # color='Category'
                )
fig.add_hline(y=-np.log(0.05))
fig.show()


# In[ ]:


# px.scatter(df_params, x=df_params.index, y='hyper_beta', color='hyper_beta_p_value_corrected')


# In[ ]:


# eps = 1e-6
# y = -np.log(df_params['beta_p_value_corrected'] + eps)
# fig = px.scatter(df_params,
#           x='beta',
#           y=y,
#                  text=df_params.index
#                 )
# fig.add_hline(y=-np.log(0.05))
# fig.show()


# In[ ]:


# fig = px.scatter(df_params['hyper_pca0_amplitude_p_value_corrected'])
# fig.add_hline(y=0.05)


# In[ ]:


basename = 'eigenworm3_coefficient'
eps = 1e-6

y = -np.log(df_params[f'{basename}_p_value_corrected'] + eps)
fig = px.scatter(df_params,
          x=basename,
          y=y,
                 text=df_params.index
                )
fig.add_hline(y=-np.log(0.05))
fig.show()


# In[ ]:


basename = 'eigenworm4_coefficient'
eps = 1e-6

y = -np.log(df_params[f'{basename}_p_value_corrected'] + eps)
fig = px.scatter(df_params,
          x=basename,
          y=y,
                 text=df_params.index
                )
fig.add_hline(y=-np.log(0.05))
fig.show()


# In[ ]:


# name = 'VB02'

# posterior = az.extract(all_traces[name], group='posterior', var_names=var_names, filter_vars='like')
# # dat = dat.to_dataframe().drop(columns=['chain', 'draw'])
# median_ds = posterior[var_names].median()
# median_df = _convert_0d_xarray(median_ds)
# all_dfs[n] = [median_df.T]

# # Also calculate the bayesian p-value vs. 0 for all columns
# prob_gt_zero = (posterior[var_names] > 0).mean()
# prob_gt_zero


# # Look at single neurons (should be significant)

# In[ ]:


import arviz as az
from wbfm.utils.general.utils_hardcoded import neurons_with_confident_ids, role_of_neuron_dict


# In[ ]:


def load_all_traces(foldername):
    fnames = neurons_with_confident_ids()
    all_traces = {}
    for neuron in tqdm(fnames):
        trace_fname = os.path.join(foldername, f'{neuron}_hierarchical_pca_trace.nc')
        if os.path.exists(trace_fname):
            try:
                trace = az.from_netcdf(trace_fname)
                all_traces[neuron] = trace
            except (ValueError, OSError, KeyError) as e:
                print(f"Error for neuron {neuron}; this is not surprising if some are still being written: {e}")
    return all_traces

parent_folder = '/lisc/data/scratch/neurobiology/zimmer/fieseler/paper/hierarchical_modeling'
# suffix = '_only_eigenworms'
# suffix = '_eigenworms34_speed'
# suffix = '_original'
suffix = '_pca_global'
# suffix = '_new_ids_only_eigenworms'
            
foldername = os.path.join(parent_folder, f'output{suffix}')
all_traces_gcamp = load_all_traces(foldername)

# foldername = os.path.join(f'{parent_folder}_gfp', f'output{suffix}')
# all_traces_gfp = load_all_traces(foldername)


# In[ ]:


# all_traces[n]


# In[ ]:


# var_names = [
#              'hyper_pca0_amplitude', "pca1_amplitude",  
#     # 'pca0_amplitude', 'pca1_amplitude', 
#     'hyper_beta', 'gamma'
# ]
# this_neuron = ['DB01', 'DB02', 'VB01', 'VB02',
#                'DD01', 'AVER', 
#                'VB03', 'DA01']


# _all_dfs = {}
# _all_dfs_sign = {}
# for n in tqdm(all_traces.keys()):
# # for n in tqdm(this_neuron):
#     # Original set of variables
#     posterior = az.extract(all_traces[n], group='posterior', var_names=var_names, filter_vars='like')
#     # dat = dat.to_dataframe().drop(columns=['chain', 'draw'])
# #     median_ds = posterior[var_names].median()
# #     median_df = _convert_0d_xarray(median_ds)
#     # _all_dfs[n] = posterior['pca0_amplitude'].values
#     _all_dfs_sign[n] = posterior['hyper_beta'].values.flatten()

# #     # Also calculate the bayesian p-value vs. 0 for all columns
# #     prob_gt_zero = (posterior[var_names] > 0).mean()
# #     # Get the smaller one (or vector) and multiply by 2
# #     p_two_sided = 2 * xr.where(
# #         prob_gt_zero <= 0.5,
# #         prob_gt_zero,
# #         1 - prob_gt_zero
# #     )
# #     _df = _convert_0d_xarray(p_two_sided, suffix="_p_value")
# #     _all_dfs[n].append(_df.T)
# df_params_test = pd.DataFrame.from_dict(_all_dfs, orient='index')
# df_params_test_sign = pd.DataFrame.from_dict(_all_dfs_sign, orient='index')


# In[ ]:


# # _all_dfs_sign['AVAL'].shape
# _all_dfs_sign.keys()


# In[ ]:


# _all_dfs


# In[ ]:


# fig = px.box(df_params_test_sign.T, x=df_params_test_sign.index, height = 1000)
# fig.add_vline(0.0)
# fig.show()


# In[ ]:


# idata = all_traces['VB02']

# az.plot_posterior(idata, var_names=['gamma', 'beta', 
#                                     # 'zscore_pca0_amplitude'
# ],
#                  ref_val=0.0)


# In[ ]:


# idata = all_traces['VB02']

# az.plot_ess(idata, var_names=['beta'])


# In[ ]:


idata


# In[ ]:


# idata = all_traces_gcamp['AVER']

# az.plot_trace(idata, var_names=['phase_shift', 'inflection_mu', 'log_hyper_beta', 'log_hyper_beta_sigma'])


# In[ ]:


# idata = all_traces_gcamp['RMDDL']
# var_names=['phase_shift', 'log_amplitude_mu', 'inflection_mu', 'log_hyper_beta', 'log_hyper_beta_sigma']
# az.plot_trace(idata, var_names=var_names)
# # az.plot_pair(idata, var_names=var_names, marginals=True)


# In[ ]:


idata = all_traces_gcamp['ALA']
# var_names=['phase_shift', 'log_amplitude_mu', 'inflection_mu', 'log_hyper_beta', 'hyper_pca0', 'hyper_pca1']
var_names=['hyper_pca0', 'log_hyper_beta', 'log_amplitude_mu']
az.plot_trace(idata, var_names=var_names)
az.plot_posterior(idata, var_names=var_names, ref_val=0)
az.plot_pair(idata, var_names=var_names, marginals=True)


# In[ ]:


idata = all_traces_gcamp['SMDDR']
# var_names=['phase_shift', 'log_amplitude_mu', 'inflection_mu', 'log_hyper_beta', 'hyper_pca0', 'hyper_pca1']
var_names=['hyper_pca0', 'log_hyper_beta', 'log_amplitude_mu']
az.plot_trace(idata, var_names=var_names, )
az.plot_posterior(idata, var_names=var_names, ref_val=0)
az.plot_pair(idata, var_names=var_names, marginals=True)


# In[ ]:


idata = all_traces_gcamp['VB02']
# var_names=['phase_shift', 'log_amplitude_mu', 'inflection_mu', 'log_hyper_beta', 'hyper_pca0', 'hyper_pca1']
var_names=['hyper_pca0', 'log_hyper_beta', 'log_amplitude_mu']
az.plot_trace(idata, var_names=var_names, )
az.plot_posterior(idata, var_names=var_names, ref_val=0)
az.plot_pair(idata, var_names=var_names, marginals=True)


# In[ ]:


idata = all_traces_gcamp['ALA']

az.plot_posterior(idata, var_names=['sigma_pca0', 
                                    # 'zscore_pca0_amplitude'
],
                 ref_val=0.0)


# In[ ]:


idata.posterior.beta


# In[ ]:


idata = all_traces_gcamp['VB01']
# var_names=['phase_shift', 'log_amplitude_mu', 'inflection_mu', 'log_hyper_beta', 'hyper_pca0', 'hyper_pca1']
var_names=['beta']
az.plot_posterior(idata, var_names=var_names, ref_val=0.01, combine_dims=set(['chain', 'draw', 'beta_dim_0']))


# In[ ]:


idata = all_traces_gcamp['ALA']
# var_names=['phase_shift', 'log_amplitude_mu', 'inflection_mu', 'log_hyper_beta', 'hyper_pca0', 'hyper_pca1']
var_names=['beta']
az.plot_posterior(idata, var_names=var_names, ref_val=0.01, combine_dims=set(['chain', 'draw', 'beta_dim_0']))


# In[ ]:


# idata = all_traces['VB02']

# var_names = ['beta']

# rhat = az.rhat(idata, var_names=var_names)                    # per variable
# # ess_bulk = az.ess(idata, method="bulk")  # bulk ESS
# # ess_tail = az.ess(idata, method="tail")  # tail ESS

# # bad_rhat = rhat.to_series()[lambda s: s > 1.01]
# # low_ess = ess_bulk.to_series()[lambda s: s < 400]
# # low_ess_tail = ess_tail.to_series()[lambda s: s < 200]

# px.scatter(rhat.to_array())

# # print("High R-hat:", rhat)#.sort_values(ascending=False))
# # print("Low bulk ESS:", low_ess.sort_values())
# # print("Low tail ESS:", low_ess_tail.sort_values())


# In[ ]:


idata = all_traces_gcamp['ALA']

az.plot_posterior(idata, var_names=['log_hyper_beta', 'inflection_mu'
                                    # 'log_amplitude_mu'
],
                 ref_val=0.0)

az.plot_trace(idata, var_names=['inflection_mu', 'log_hyper_beta'])


# In[ ]:


# idata = all_traces_gcamp['SMDDL']

# az.plot_posterior(idata, var_names=['phase_shift', 'eigenworm3_coefficient', 'eigenworm4_coefficient'
#                                     # 'log_amplitude_mu'
# ],
#                  ref_val=0.0)

# # az.plot_trace(idata, var_names=['phase_shift'])


# In[ ]:


# idata = all_traces_gcamp['SMDDR']

# az.plot_posterior(idata, var_names=['phase_shift', 'eigenworm3_coefficient', 'eigenworm4_coefficient'
#                                     # 'log_amplitude_mu'
# ],
#                  ref_val=0.0)

# # az.plot_trace(idata, var_names=['phase_shift'])


# In[ ]:


# idata = all_traces_gcamp['SMDDR']

# az.plot_trace(idata, var_names=['inflection_mu', 'hyper_beta'])


# In[ ]:


# az.plot_energy(idata)


# In[ ]:


idata = all_traces_gcamp['VB02']
az.plot_trace(idata, var_names=['inflection_mu'], compact=False)#filter_vars='like')


# In[ ]:


idata = all_traces_gcamp['AVER']
az.plot_pair(idata, var_names=['log_hyper_beta', 'inflection_mu', 'log_amplitude_mu'], combine_dims=set('draw'), marginals=True, #reference_values={'beta': [0,0], 'gamma': [0]}
            )


# In[ ]:


# # posterior = az.extract(idata, group="posterior")

# fig, ax = plt.subplots()

# az.plot_dist(
#     idata.posterior["hyper_beta"],
#     ax=ax,
#     label="hyper_beta"
# )

# az.plot_dist(
#     idata.posterior["log_amplitude_mu"],
#     ax=ax,
#     color='orange',
#     label="log_amplitude_mu"
# )

# ax.legend()
# plt.show()


# In[ ]:


import math

def plot_hierarchical_overlay(
    idata,
    var1,
    var2,
    dim=None,
    ncols=None,
    coords=None,
    kind="kde",
    ref_val=None,
    figsize=None,
):
    """
    Plot two hierarchical posterior variables overlaid, one subplot per coordinate.

    Parameters
    ----------
    idata : az.InferenceData
        InferenceData containing posterior samples.
    var1, var2 : str
        Variable names in the posterior group.
    dim : str
        Name of the hierarchical dimension (e.g. "beta_dim_0").
    coords : dict, optional
        Optional coordinate selection before plotting.
        Example: {"year": 2022}
    kind : {"kde", "hist"}
        Type of distribution plot.
    figsize : tuple, optional
        Size of the full figure.
    """

    da1 = idata.posterior[var1]
    da2 = idata.posterior[var2]

    if dim is None:
        dim1 = f'{var1}_dim_0'
        dim2 = f'{var2}_dim_0'
    else:
        dim1 = dim
        dim2 = dim

    dim_vals = da1.coords[dim1].values
    n = len(dim_vals)

    # Grid shape
    if ncols is None:
        ncols = int(math.ceil(math.sqrt(n)))
    nrows = int(math.ceil(n / ncols))

    if figsize is None:
        figsize = (4 * ncols, 3 * nrows)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        sharey=True,
    )

    axes = axes.flatten()

    # fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True)

    if n == 1:
        axes = [axes]

    for ax, d in zip(axes, dim_vals):
        az.plot_dist(
            da1.sel({dim1: d}),
            ax=ax,
            kind=kind,
            label=f"{var1}[{d}]",
        )
        az.plot_dist(
            da2.sel({dim2: d}),
            ax=ax,
            kind=kind,
            color='orange',
            label=f"{var2}[{d}]",
        )
        ax.set_title(f"{dim1} = {d}")
        if ref_val is not None:
            ax.axvline(x=ref_val, color='k')
        ax.legend()

    fig.tight_layout()
    return fig, axes



# In[ ]:


# from arviz import hdi
# idata = all_traces_gcamp['VB01']

# x = idata.posterior["beta"].median(dim=["draw", 'chain'])
# y = idata.posterior["gamma"].median(dim=["draw", 'chain'])

# plt.scatter(x, y)
# plt.xlabel("median(beta)")
# plt.ylabel("median(gamma)")
# plt.show()


# In[ ]:


# idata = all_traces['VB02']
# plot_hierarchical_overlay(idata, 'beta', 'gamma', ref_val=0)


# In[ ]:


# idata = all_traces['VB02']
# fig = px.scatter(az.rhat(idata, var_names=["pca0_amplitude"])['pca0_amplitude'].values)
# fig.show()

# fig = px.scatter(az.ess(idata, var_names=["pca0_amplitude"])['pca0_amplitude'].values)
# fig.show()


# In[ ]:


# import pymc as pm
# import arviz as az
# import matplotlib.pyplot as plt

# with pm.Model() as model:
#     hyper_pca0_amplitude = pm.Normal('hyper_pca0_amplitude', mu=0, sigma=5)
#     hyper_pca0_sigma = pm.Exponential('hyper_pca0_sigma', lam=10)
#     hyper_pca0_sigma2 = pm.HalfNormal('hyper_pca0_sigma2', sigma=0.5)
#     zscore_pca0_amplitude = pm.Normal('zscore_pca0_amplitude', mu=0, sigma=1)

#     prior = pm.sample_prior_predictive(samples=10000)

# # ArviZ does the plotting nicely
# az.plot_dist(prior.prior["hyper_pca0_amplitude"], label="amplitude")
# plt.show()

# az.plot_dist(prior.prior["hyper_pca0_sigma"], label="sigma")
# plt.show()

# az.plot_dist(prior.prior["hyper_pca0_sigma2"], label="sigma")
# plt.show()

# az.plot_dist(prior.prior["zscore_pca0_amplitude"], label="sigma")
# plt.show()


# In[ ]:


# idata = all_traces['VB02']
# az.plot_rank(idata, var_names=["pca0_amplitude"])


# In[ ]:


idata = all_traces['VB01']
px.scatter(az.rhat(idata, var_names=["pca0_amplitude"])['pca0_amplitude'].values)


# In[ ]:


idata = all_traces['VB02']
az.plot_trace(idata, var_names=['pca0_amplitude'], compact=False)


# In[ ]:


idata = all_traces['VB02']
az.plot_trace(idata, var_names=['zscore_pca0_amplitude'], compact=True);


# In[ ]:


# idata = all_traces['VB02']
# az.plot_posterior(idata, var_names=['hyper_pca0_sigma'])

# idata = all_traces['DB01']
# az.plot_posterior(idata, var_names=['hyper_pca0_sigma'])


# In[ ]:


idata = all_traces['VB02']
az.plot_posterior(idata, var_names=['hyper_pca0_amplitude'], ref_val=0)

idata = all_traces['VB01']
az.plot_posterior(idata, var_names=['hyper_pca0_amplitude'], ref_val=0)


# In[ ]:


# vals = {}
# for k, idata in all_traces_gcamp.items():
#     vals[k] = np.array(idata.posterior.eigenworm3_coefficient).flatten()#.median(dim='chain')


# In[ ]:


# px.box(vals)


# ### Reconstruct model outputs

# In[ ]:


from wbfm.utils.external.utils_pymc import sample_prior_predictive_for_neuron, reconstruct_model_term_from_trace
n = 'AVER'

idata = all_traces_gcamp[n]

posterior, model_data = reconstruct_model_term_from_trace(idata.posterior, n, Xy)


# In[ ]:


sigmoid_term_median = posterior['sigmoid_term'].median(dim=['chain', 'draw'])


# In[ ]:


posterior


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# idata


# In[ ]:


from wbfm.utils.external.utils_pandas import get_dataframe_for_single_neuron
neuron_name = 'VB02'
dataset_name = 'all'

curvature_terms_to_use = ['eigenworm0', 'eigenworm1']
curvature_terms_to_use.extend(['eigenworm2', 'eigenworm3'])
residual_mode='pca_global'
verbose = 1


# In[ ]:


df_model = get_dataframe_for_single_neuron(Xy, neuron_name, dataset_name=dataset_name,
                                               curvature_terms=curvature_terms_to_use, 
                                               residual_mode=residual_mode, verbose=verbose)


# In[ ]:


idata = all_traces_gcamp[neuron_name]

posterior, model_data = reconstruct_model_term_from_trace(idata.posterior, neuron_name, Xy, var_names=['sigmoid_term', 'mu'])


# In[ ]:


sigmoid_term_median = posterior['sigmoid_term'].median(dim=['chain', 'draw'])
mu_median = posterior['mu'].median(dim=['chain', 'draw'])


# In[ ]:


posterior


# In[ ]:


import pandas as pd
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display

df = df_model
# Unique dataset names
unique_datasets = df['dataset_name'].unique()

# Slider for selecting dataset
dataset_index_widget = widgets.IntSlider(
    value=0,
    min=0,
    max=len(unique_datasets)-1,
    step=1,
    description='Dataset:',
    continuous_update=False
)

# Dropdown for selecting which column to plot
column_widget = widgets.Dropdown(
    options=df.columns.tolist(),
    value='y',
    description='Column:'
)


def plot_dataset_plotly(dataset_index, column):
    dataset_name = unique_datasets[dataset_index]
    dataset_idx = df['dataset_name'] == dataset_name
    subset = df[dataset_idx]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=subset.index,
        y=subset[column],
        # fillcolor='rev',
        # mode='lines+markers',
        name=f'{neuron_name} ({dataset_name})'
    ))
    fig.add_trace(go.Scatter(
        x=subset.index,
        y=subset['rev'],
        name=f'rev ({dataset_name})'
    ))
    fig.add_trace(go.Scatter(
        x=subset.index,
        y=subset['x_pca0'],
        name=f'pca0 ({dataset_name})'
    ))
    fig.add_trace(go.Scatter(
        x=subset.index,
        y=sigmoid_term_median.values[dataset_idx],
        name=f'sigmoid ({dataset_name})'
    ))
    fig.add_trace(go.Scatter(
        x=subset.index,
        y=mu_median.values[dataset_index],
        name='mu'
    ))
    
    
    fig.update_layout(
        title=f"{column} for dataset '{dataset_name}'",
        xaxis_title="Time point index",
        yaxis_title=column,
        template="plotly_white",
        height=400
    )
    fig.show()


# Interactive widget
widgets.interact(plot_dataset_plotly, dataset_index=dataset_index_widget, column=column_widget)


# In[ ]:


df


# In[ ]:





# In[ ]:





# ## Alt: summarize parameters using flavell-style table

# In[ ]:


from wbfm.utils.general.utils_paper import plot_foldchange_boxes


# In[ ]:


df_params.head()


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_hierarchy = df_params.copy().reset_index(names='Neuron Name')#.swaplevel(0, 1, axis=1)

# Reshape to tall-style dataframe; can't just reuse var_names because they are substrings not the full strings
# var_names_to_plot = var_names
# var_names_to_plot.append('sigmoid_term_quantile')
var_names = ["r", 'sigmoid_term_quantile', 
             'eigenworm2_coefficient', 'eigenworm3_coefficient', 
             #'speed', 'phase', 'dorsal', 'ventral', 
             'hyper_pca0_amplitude', 'hyper_pca1_amplitude',
             "sigmoid_term"]

val_name = 'values'
# print(df_hierarchy.head())
df_hierarchy_melt = df_hierarchy.melt(#var_name='Neuron Name', 
                                      id_vars='Neuron Name', 
                                      value_name=val_name,  
                                      value_vars=var_names#df_hierarchy.columns.tolist()
                                     )
print(df_hierarchy_melt)
df_hierarchy_melt['Neuron Type'] = df_hierarchy_melt['Neuron Name'].map(role_of_neuron_dict(only_fwd_rev=True)).replace('', 'Other')
df_hierarchy_melt['Neuron Type Complex'] = df_hierarchy_melt['Neuron Name'].map(role_of_neuron_dict(include_fwd_rev=True, include_ventral_dorsal=True, include_basic=False)).replace('', 'Other')
df_hierarchy_melt['Neuron Type VD'] = df_hierarchy_melt['Neuron Name'].map(role_of_neuron_dict(include_ventral_dorsal=True)).replace('', 'Other')

neuron_order = df_hierarchy_melt[df_hierarchy_melt['variable'] == 'hyper_pca0_amplitude'].groupby('Neuron Name')[val_name].median().sort_values().index
df_hierarchy_melt['variable'] = df_hierarchy_melt['variable'].map(lambda x: {'r': 'r', 'sigmoid_term_quantile': 'sig_q', 'eigenworm2_coefficient': 'e2',
       'eigenworm3_coefficient': 'e3', 'hyper_pca0_amplitude': 'pc0', 'hyper_pca1_amplitude': 'pc1', 'sigmoid_term': 'sig'}[x])
df_hierarchy_melt.head()


# In[ ]:



_df = df_hierarchy_melt[df_hierarchy_melt['variable'].isin(['pc0', 'pc1'])]
groups = {'neurons': _df['Neuron Name'].unique()}
opt = dict(
    behavior_col='variable',
    groups=groups,
    # subtitle_behavior_col='Components',
    rows_col='Neuron Name',
    value_col=val_name,
    # figsize=(8, 6),
    # row_vspace=0.5,
    # behavior_hspace=3.0,
    add_text=False,
    # vmax=0.6,
    neuron_order=neuron_order,
    center_at_zero=True,
    DEBUG=False
)

# Call the function with default fold-change mode
ax, fig, ordered_neurons = plot_foldchange_boxes(
    df=_df, **opt,
    cmap='RdBu',
)

plt.tight_layout()
apply_figure_settings(fig, width_factor=0.5, height_factor=0.5, plotly_not_matplotlib=False)


_df = df_hierarchy_melt[df_hierarchy_melt['variable'].isin(['e2', 'e3', 'r'])]
# Call the function with default fold-change mode
ax, fig, ordered_neurons = plot_foldchange_boxes(
    df=_df, **opt,
    cmap='PiYG',
)

plt.tight_layout()
apply_figure_settings(fig, width_factor=0.5, height_factor=0.5, plotly_not_matplotlib=False)


# fname = os.path.join(output_folder, 'bayesian_variables_foldchange.png')
# plt.savefig(fname, dpi=100)

# fname = Path(fname).with_suffix('.svg')
# plt.savefig(fname)

# plt.show()


# In[ ]:


get_ipython().run_line_magic('debug', '')


# In[ ]:


# _df


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Alt: direct model comparison (needs non-hierarchical traces also)

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


# ### Try to diagnose why LOTO is so bad

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




