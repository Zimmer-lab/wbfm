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

# In[5]:


from wbfm.utils.general.utils_paper import apply_figure_settings
from wbfm.utils.external.utils_plotly import get_nonoverlapping_text_positions
from wbfm.utils.general.utils_paper import apply_figure_settings, plotly_paper_color_discrete_map, data_type_name_mapping, package_bayesian_df_for_plot


# In[6]:


from wbfm.utils.general.utils_paper_revisions_bayes import calculate_all_bayesian_model_data
df_params, df_params_full, y_max_gfp, x_max_gfp = calculate_all_bayesian_model_data(suffix='_cca_continuous', recalculate_sigmoid=False)


# In[7]:



idx = df_params['Dataset Type'] == 'Freely Moving (GCaMP, residual)'
_df = df_params[idx]
mapping = dict(zip(_df['neuron_name'], _df['Neuron Type']))
df_params_full['Neuron Type'] = df_params_full['neuron_name'].map(mapping)


# In[8]:


from wbfm.utils.general.utils_paper import plot_bayesian_model_comparison, plotly_paper_color_discrete_map
from wbfm.utils.external.utils_plotly import get_nonoverlapping_text_positions


# In[30]:



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


# In[31]:


# Save to disk for comparison to other manifold models

fname = os.path.join("fig5_cca_continuous", 'df_params.h5')
df_params.to_hdf(fname, key="df_with_missing")


# In[43]:


# y, x = 'Relative Hierarchy Score', 'Behavior Score'
y, x = 'Relative Hierarchy Score (raw)', 'Behavior Score (raw)'

symbol_name = "Model parameter significance" 

_idx = df_params['Category_new'] == 'Hierarchy > GFP and PC1 is significant'
text = df_params['text_raw'].copy()
text[:] = ''
text[_idx] = df_params['text_raw'].copy()[idx_final]

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

get_nonoverlapping_text_positions(df_params[x], df_params[y], text, fig, weight=200, k=2000, add_nodes_with_no_text=False,
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
                  showlegend=False, 
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

apply_figure_settings(fig, width_factor=0.75, height_factor=0.3)
fig.show()


to_save = True
if to_save:
    fname = os.path.join("fig5_cca_continuous", 'model_improvement.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)
    fname = Path(fname).with_suffix('.html')
    fig.write_html(fname)
    


# In[12]:


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
    fname = os.path.join("fig5_cca_continuous", 'behavior_change_sigmoid_box.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)


# In[15]:


import plotly.graph_objects as go

_df = df_params[idx_final]

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
    fname = os.path.join("fig5_cca_continuous", 'phase_shift_and_oscillation_amplitude.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)
    fname = Path(fname).with_suffix('.html')
    fig.write_html(fname)
    


# In[16]:


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
    fname = os.path.join("fig5_cca_continuous", 'ew3_boxplot.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)


# In[17]:


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
    fname = os.path.join("fig5_cca_continuous", 'ew4_boxplot.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)


# In[ ]:





# ## Supp

# In[18]:


df_params_full.head()


# In[19]:


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


# In[20]:


all_results


# In[21]:


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
    fname = os.path.join("fig5_cca_continuous", 'phase_shift_ttests.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)
    fname = Path(fname).with_suffix('.html')
    fig.write_html(fname)


# In[22]:


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
    fname = os.path.join("fig5_cca_continuous", 'phase_shift_ttests_pairs.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)
    fname = Path(fname).with_suffix('.html')
    fig.write_html(fname)


# In[27]:


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

# get_nonoverlapping_text_positions(x, y, text, fig, weight=50, k=1, add_nodes_with_no_text=False,
#                                  # x_range=[-0.4, 0.3], 
#                                   # y_range=[0, 8], 
#                                   size=12
#                                  )
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
    fname = os.path.join("fig5_cca_continuous", 'pc0_coefficient.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)
    fname = Path(fname).with_suffix('.html')
    fig.write_html(fname)
    


# In[24]:


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
    fname = os.path.join("fig5_cca_continuous", 'pc1_coefficient.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)
    fname = Path(fname).with_suffix('.html')
    fig.write_html(fname)


# In[25]:


from wbfm.utils.external.utils_plotly import add_trendline_annotation, extend_trendline

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
                 trendline='ols', 
                 # marginal_y='rug'
                )

fig.data[1].update(line=dict(color="black", dash='dot'))
get_nonoverlapping_text_positions(_df[x], y_vals, text, fig, weight=100, k=1, add_nodes_with_no_text=False,
                                 x_range=[0, 1], y_range=[0, 1], 
                                  size=12
                                 )

fig.update_xaxes(title='Variance Explained by Manifold')
fig.update_yaxes(title=r'$\gamma$')

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
    fname = os.path.join("fig5_cca_continuous", 'hierarchy_behavior_score_and_manifold_variance.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




