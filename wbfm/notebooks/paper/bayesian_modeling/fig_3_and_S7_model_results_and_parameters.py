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


# In[2]:


from wbfm.utils.general.hardcoded_paths import get_hierarchical_modeling_dir

fname = os.path.join(get_hierarchical_modeling_dir(), 'data.h5')
print(fname)
Xy = pd.read_hdf(fname)

fname = os.path.join(get_hierarchical_modeling_dir(gfp=True), 'data.h5')
print(fname)
Xy_gfp = pd.read_hdf(fname)


# # Load model results

# In[3]:


from wbfm.utils.external.utils_plotly import get_nonoverlapping_text_positions


# In[4]:


suffix = ''
# suffix = '_new_ids_only_eigenworms'
# suffix = '_eigenworms34_speed'
# suffix = '_only_eigenworms'

# Load data from many dataframes
output_dir = '/lisc/scratch/neurobiology/zimmer/fieseler/paper/hierarchical_modeling/output' + suffix
# output_dir = os.path.join(get_hierarchical_modeling_dir(), 'output')
all_dfs = {}
for filename in tqdm(Path(output_dir).iterdir()):
    if filename.name.endswith('.h5') and 'single' not in filename.name:
        neuron_name = '_'.join(filename.name.split('_')[:-1])
        all_dfs[neuron_name] = pd.read_hdf(filename)
df = pd.concat(all_dfs).reset_index(names=['neuron_name', 'model_type'])


# In[5]:


# Also load gfp
output_dir = '/lisc/scratch/neurobiology/zimmer/fieseler/paper/hierarchical_modeling_gfp/output' + suffix

# output_dir = os.path.join(get_hierarchical_modeling_dir(gfp=True), 'output')
all_dfs_gfp = {}
for filename in tqdm(Path(output_dir).iterdir()):
    if filename.name.endswith('.h5') and 'single' not in filename.name:
        neuron_name = '_'.join(filename.name.split('_')[:-1])
        all_dfs_gfp[neuron_name] = pd.read_hdf(filename)
df_gfp = pd.concat(all_dfs_gfp).reset_index(names=['neuron_name', 'model_type'])


# # Plot model comparison statistics

# In[6]:


# What I want to plot:
# x = scaled difference between the null and non-hierarchical model
# y = same but for hierarchical
from wbfm.utils.general.utils_paper import apply_figure_settings, plotly_paper_color_discrete_map, data_type_name_mapping, package_bayesian_df_for_plot
from wbfm.utils.general.hardcoded_paths import neurons_with_confident_ids

# df_to_plot_gcamp = package_bayesian_df_for_plot(df, relative_improvement=False).assign(datatype='Freely Moving (GCaMP, residual)')
# df_to_plot_gfp = package_bayesian_df_for_plot(df_gfp, relative_improvement=False).assign(datatype='Freely Moving (GFP, residual)')
df_to_plot_gcamp = package_bayesian_df_for_plot(df, df_normalization=Xy, 
                                                min_num_datapoints=10000).assign(datatype='Freely Moving (GCaMP, residual)')
df_to_plot_gfp = package_bayesian_df_for_plot(df_gfp, df_normalization=Xy_gfp, 
                                              min_num_datapoints=5000).assign(datatype='Freely Moving (GFP, residual)')

# Add a couple names back in
# df_to_plot_gfp.loc['VB02', 'text'] = 'VB02 (gfp)'
# df_to_plot_gfp.loc['RMED', 'text'] = 'RMED (gfp)'
# df_to_plot_gfp.loc['RMEV', 'text'] = 'RMEV (gfp)'
rename_func = lambda x: f'{x} (gfp)' if x != '' else ''
df_to_plot_gfp.loc[:, 'text'] = df_to_plot_gfp.loc[:, 'text'].apply(rename_func)
# df_to_plot_gcamp.loc['RMDVL', 'text'] = 'RMDVL'
# df_to_plot_gcamp.loc['SMDVR', 'text'] = 'SMDVR'
# df_to_plot_gcamp.loc['VB03', 'text'] = 'VB03'
# Remove a couple names
# df_to_plot_gcamp.loc['BAGL', 'text'] = ''
# df_to_plot_gcamp.loc['URADL', 'text'] = ''

df_to_plot = pd.concat([df_to_plot_gcamp, df_to_plot_gfp])
df_to_plot['Dataset Type'] = df_to_plot['datatype']
df_to_plot['Size'] = 1


# In[7]:


import plotly.graph_objects as go
def paper_plot(x, y, to_save=True, remove_names_of_ns=True, display_text=True, to_show=True, **kwargs):

    x_max_gfp = df_to_plot_gfp[x].max()
    y_max_gfp = df_to_plot_gfp[y].max()
    print('GFP thresholds: ', y_max_gfp, x_max_gfp)

    def categorize_row(row):
        if row[y] > y_max_gfp and row[x] > x_max_gfp:
            return 'Hierarchical Behavior'
        elif row[y] <= y_max_gfp and row[x] > x_max_gfp:
            return 'Behavior only'
        elif row[y] > y_max_gfp and row[x] <= x_max_gfp:
            return 'Hierarchy only'
        else:
            return 'No Behavior or Hierarchy'

    # Apply function to create new column
    df_to_plot_gcamp['Category'] = df_to_plot_gcamp.apply(categorize_row, axis=1)
    df_to_plot['Category'] = df_to_plot.apply(categorize_row, axis=1)
    _df = df_to_plot[df_to_plot.index.isin(neurons_with_confident_ids())]
    text = _df['text'].copy()
    if remove_names_of_ns:
        text[_df[y] <= y_max_gfp] = ''
    
    fig = px.scatter(_df, 
                     # x='Hierarchy Score', y='Behavior Score', range_y=[-2, 60],
                     y=y, x=x, #range_x=[-2, 60],
                     text=text if display_text else None, 
                     # color='Category', #
                     color='Dataset Type',
                     color_discrete_map=plotly_paper_color_discrete_map(), 
                     #size='Size', 
                     size_max=10,
                     hover_data=['Category'],
                     **kwargs
                    )
    fig.update_traces(textposition='middle left')

    apply_figure_settings(fig, width_factor=1.0, height_factor=0.3)

    # gfp lines
    # fig.add_shape(type="line",
    #               x0=x_max_gfp, y0=0,  # start of the line (bottom of the plot)
    #               x1=x_max_gfp, y1=1,  # end of the line (top of the plot)
    #               line=dict(color="black", width=1, dash="dash"),
    #               xref='x',
    #               yref='paper')
    fig.add_shape(type="line",
                  x0=0, y0=y_max_gfp,  # start of the line (bottom of the plot)
                  x1=1, y1=y_max_gfp,  # end of the line (top of the plot)
                  line=dict(color="black", width=1, dash="dash"),
                  xref='paper',
                  yref='y')
    # Diagonal line
    xy_max = np.min([df_to_plot[x].max(), df_to_plot[y].max()])
    fig.add_shape(type="line",
                  x0=0, y0=0,  # start of the line (bottom of the plot)
                  x1=xy_max, y1=xy_max,  # end of the line (top of the plot)
                  line=dict(color="black", width=1, dash="dash"),
                  xref='x',
                  yref='y')
    fig.update_layout(legend=dict(
        yanchor="top",
        y=1.02,
        xanchor="left",
        x=0.02
    ))
    fig.update_xaxes(title=f'{x}')# over Behavior model')
    fig.update_yaxes(title=f'{y}')# <br>over Trivial model')
    
    # Add contour plot for the gfp points
    # _df2 = _df[_df['Dataset Type'] == 'Freely Moving (GFP, residual)']
    # hist, x_edges, y_edges = np.histogram2d(_df2[x], _df2[y], bins=4)
    # x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    # y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    # # contour = go.Figure(data=
    # contour = go.Contour(
    #         x=x_centers,
    #         y=y_centers,
    #         z=hist.T * 2,
    #         contours_coloring='lines',
    #         line_width=2,
    #     )
    # )
    # Combine the scatter and contour plots
    # fig.add_trace(contour)

    if to_save:
        ##
        # Make a figure for presentations with fewer names
        ##
        apply_figure_settings(fig, height_factor=0.4, width_factor=0.5)
        # fig.show()  # Showing here messes it up for the next save
        fname = os.path.join("/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/bayesian_modeling/plots", 'hierarchy_behavior_score_with_gfp_presentation.png')
        fig.write_image(fname, scale=7)

    ##
    # Final settings
    ##

    # # Add some additional annotations with arrows and offsets (gfp)
    # annotations_to_add = ['VB02', 'RMED', 'RMEV']
    # offset_list = [[10, -150], [250, -50], [130, -50]]
    # for offset, neuron in zip(offset_list, annotations_to_add):
    #     ind = df_to_plot['datatype'] == 'Freely Moving (GFP, residual)'
    #     xy = list(df_to_plot[ind].loc[neuron, [x, y]])
    #     text = f'{neuron} (GFP)'
    #     fig.add_annotation(x=xy[0], y=xy[1], ax=offset[0], ay=offset[1],
    #                        text=text, showarrow=True)

    # # Add some additional annotations with arrows and offsets (gcamp)
    # annotations_to_add = ['RIS']
    # offset_list = [[100, -50]]
    # for offset, neuron in zip(offset_list, annotations_to_add):
    #     ind = df_to_plot['datatype'] == 'Freely Moving (GCaMP, residual)'
    #     xy = list(df_to_plot[ind].loc[neuron, [x, y]])
    #     text = f'{neuron}'
    #     fig.add_annotation(x=xy[0], y=xy[1], ax=offset[0], ay=offset[1],
    #                        text=text, showarrow=True)

    apply_figure_settings(fig, height_factor=0.25, width_factor=0.5)

    if to_save:
        fname = os.path.join("/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/bayesian_modeling/plots", f'x-{x}_y-{y}.png')
        fig.write_image(fname, scale=3)
        fname = Path(fname).with_suffix('.svg')
        fig.write_image(fname)
        fname = Path(fname).with_suffix('.html')
        fig.write_html(fname)
        
    if to_show:
        fig.show()
    
    return fig, _df, text


# In[8]:


y, x = 'Hierarchy Score', 'Relative Hierarchy Score'
fig, _df, text = paper_plot(x, y, display_text=False)


# In[9]:


y, x = 'Hierarchy Score', 'Behavior Score'
fig, _df, text = paper_plot(x, y, #size='Relative Hierarchy Score', 
                            display_text=False, to_show=False)

get_nonoverlapping_text_positions(_df[x], _df[y], text, fig, weight=1, k=4, add_nodes_with_no_text=True,
                                 x_range=[0, 29], size=12)

fig.update_xaxes(title='Behavior-only Model Performance')
fig.update_yaxes(title='Hierarchy Model Performance')

fname = os.path.join("/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/bayesian_modeling/plots", f'x-{x}_y-{y}-annotations.png')
fig.write_image(fname, scale=3)
fname = Path(fname).with_suffix('.svg')
fig.write_image(fname)
fname = Path(fname).with_suffix('.html')
fig.write_html(fname)

fig.show()


# In[10]:


# Same but with ALL names
y, x = 'Hierarchy Score', 'Behavior Score'
fig, _df, text = paper_plot(x, y, #size='Relative Hierarchy Score', 
                            display_text=True, to_show=True, remove_names_of_ns=False)
# fig.show()
# get_nonoverlapping_text_positions(_df[x], _df[y], text, fig, weight=1, k=4, add_nodes_with_no_text=True,
#                                  x_range=[0, 29], size=12)


# ## Supplement with alternate axes: manifold variance

# In[11]:


def calc_var_ratio(Xy):
    Xy_var = Xy.groupby('dataset_name').var()
    Xy_var_ratio = {}
    for col in Xy_var.columns:
        if 'neuron' in col:
            continue
        col_manifold = f'{col}_manifold'
        if col_manifold in Xy_var.columns:
            Xy_var_ratio[col] = Xy_var[col_manifold] / Xy_var[col]
    Xy_var_ratio = pd.DataFrame(Xy_var_ratio)
    return Xy_var_ratio
    
df_var_exp_gcamp = calc_var_ratio(Xy)
df_var_exp_gcamp['Dataset Type'] = 'Freely Moving (GCaMP, residual)'
df_var_exp_gfp = calc_var_ratio(Xy_gfp)
df_var_exp_gfp['Dataset Type'] = 'Freely Moving (GFP, residual)'
df_var_exp = pd.concat([df_var_exp_gcamp, df_var_exp_gfp], axis=0)
    
# px.box(df_var_exp.dropna(thresh=3, axis=1), color='Dataset Type')


# In[12]:


df_var_exp_median = df_var_exp.groupby('Dataset Type').median().reset_index().melt(
    id_vars='Dataset Type', var_name='neuron_name', value_name='manifold_variance')
df_to_plot_with_var = df_to_plot.merge(df_var_exp_median, on=['neuron_name', 'Dataset Type'])

# df_to_plot_with_var


# In[13]:


x, y = 'manifold_variance', 'Relative Hierarchy Score'

# Define threshold(s)
# x_max_gfp = df_to_plot_gfp[x].max()
y_max_gfp = df_to_plot_gfp[y].max()
print('GFP thresholds: ', y_max_gfp)
df_to_plot_with_var['above_gfp'] = df_to_plot_with_var[y] > y_max_gfp
df_to_plot_with_var['text_simple'] = df_to_plot_with_var['text']
df_to_plot_with_var.loc[~df_to_plot_with_var['above_gfp'], 'text_simple'] = ''

# Remove or include temporary names
df_to_plot_with_var = df_to_plot_with_var.loc[df_to_plot_with_var['neuron_name'].isin(neurons_with_confident_ids())].copy()
df_to_plot_with_var = df_to_plot_with_var.loc[df_to_plot_with_var['neuron_name'] != '']

# Only include neurons that make it above the prior threshold, but not GFP ones
cols_to_plot = ['Hierarchy only', 'Hierarchical Behavior']
_neurons_to_plot = df_to_plot_with_var['neuron_name'][df_to_plot_with_var['Category'].isin(cols_to_plot)]
_df_to_plot_with_var = df_to_plot_with_var.loc[df_to_plot_with_var['neuron_name'].isin(_neurons_to_plot)]
_df_to_plot_with_var = _df_to_plot_with_var.loc[df_to_plot_with_var['Dataset Type'].isin(['Freely Moving (GCaMP, residual)'])]


# Actual plot
fig = px.scatter(_df_to_plot_with_var, 
                 y=y, x=x,
                 # text='neuron_name', 
                 text='text', 
                 color='Dataset Type',
                color_discrete_map=plotly_paper_color_discrete_map(), size='Size', size_max=10,
                 # marginal_y='rug'
                )
# fig.add_shape(type="line",
#               x0=0, y0=y_max_gfp,  # start of the line (bottom of the plot)
#               x1=1, y1=y_max_gfp,  # end of the line (top of the plot)
#               line=dict(color="black", width=2, dash="dash"),
#               xref='paper',
#               yref='y')
fig.update_xaxes(title='Variance Explained by Manifold')
fig.update_yaxes(title='Hierarchical Model Improvement<br>(compared to behavior-only model)')

fig.update_layout(showlegend=False, 
                  legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.4
))

to_save = True
if to_save:
    apply_figure_settings(fig, width_factor=0.6, height_factor=0.3)
    fname = os.path.join("/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/bayesian_modeling/plots", 'hierarchy_behavior_score_and_manifold_variance-no_text.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)

# text = _df_to_plot_with_var['text']
# get_nonoverlapping_text_positions(_df_to_plot_with_var[x], _df_to_plot_with_var[y], text, fig, #weight=1, 
#                                   k=0.2, 
#                                   #add_nodes_with_no_text=True, 
#                                   size=12, #x_range=[-0.2, 1.2]
#                                  )

apply_figure_settings(fig, width_factor=0.45, height_factor=0.45)


fig.show()


to_save = True
if to_save:
    fname = os.path.join("/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/bayesian_modeling/plots", 'hierarchy_behavior_score_and_manifold_variance.png')
    # fname = os.path.join("/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/presentations_and_grants/CSH", 'hierarchy_behavior_score_with_gfp.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)


# # Additional subplots: model parameters

# In[14]:


import arviz as az
from wbfm.utils.general.hardcoded_paths import neurons_with_confident_ids
from wbfm.utils.general.hardcoded_paths import role_of_neuron_dict


# In[15]:


def load_all_traces(foldername):
    fnames = neurons_with_confident_ids()
    all_traces = {}
    for neuron in tqdm(fnames):
        trace_fname = os.path.join(foldername, f'{neuron}_hierarchical_pca_trace.nc')
        if os.path.exists(trace_fname):
            try:
                trace = az.from_netcdf(trace_fname)
                all_traces[neuron] = trace
            except (ValueError, OSError) as e:
                print(f"Error for neuron {neuron}; this is not surprising if some are still being written: {e}")
    return all_traces

parent_folder = '/lisc/scratch/neurobiology/zimmer/fieseler/paper/hierarchical_modeling'
# suffix = '_only_eigenworms'
# suffix = '_eigenworms34_speed'
suffix = ''
# suffix = '_new_ids_only_eigenworms'
            
foldername = os.path.join(parent_folder, f'output{suffix}')
all_traces_gcamp = load_all_traces(foldername)

# foldername = os.path.join(f'{parent_folder}_gfp', f'output{suffix}')
# all_traces_gfp = load_all_traces(foldername)


# In[17]:


# Also load the NMJ connectivity
fname = '/home/charles/Current_work/repos/dlc_for_wbfm/paper/NeuronFixedPoints.xls'
df_connect = pd.read_excel(fname)
_df = df_connect[df_connect['Landmark'].str.contains('M')]
muscle_position = _df.groupby('Neuron')['Landmark Position'].mean()


# In[18]:


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
removed_neurons = set(_neurons_to_plot) - set(neurons_to_plot)
if len(removed_neurons) > 0:
    print(f"Warning: some neurons should be plotted, but do not have a stored trace: {removed_neurons}")
# df_to_plot_with_var[df_to_plot_with_var['neuron_name'] == 'AVAL']


# In[19]:


# var_names = ["self_collision", 'speed', 'eigenworm3', 'eigenworm4', 'amplitude_mu']
var_names = ["self_collision", 'dorsal', 'ventral', 'amplitude_mu']

all_traces = all_traces_gcamp
# all_traces = all_traces_gfp

# az.plot_forest([all_traces[n] for n in neurons_to_plot], model_names=neurons_to_plot,
#                var_names=var_names, combined=True, 
#               filter_vars='like', kind='ridgeplot', figsize=(9, 7), ridgeplot_overlap=3)


# In[20]:


# Scatter plot of median model parameters
from collections import defaultdict
var_names = ["self_collision", 'amplitude_mu', 'eigenworm', 'speed', 'phase', 'dorsal', 'ventral', 'hyper_pca0_amplitude']
var_names2 = ["sigmoid_term"]

all_dfs = {}
for n in tqdm(neurons_to_plot):
    # Original set of variables
    dat = az.extract(all_traces[n], group='posterior', var_names=var_names, filter_vars='like')
    all_dfs[n] = [dat.to_dataframe().drop(columns=['chain', 'draw']).median()]
    
    # Variables with specific postprocessing
    dat = az.extract(all_traces[n], group='posterior', var_names=var_names2, filter_vars='like')
    dat_sigmoid = dat.to_dataframe().drop(columns=['chain', 'draw'])
    dat_sigmoid_quantile = dat_sigmoid.quantile(0.8)#.rename('sigmoid_term_quantile')
    dat_sigmoid_quantile.index = ['sigmoid_term_quantile']
    dat_sigmoid_variance = dat_sigmoid.var()
    dat_sigmoid_variance.index = ['sigmoid_term_variance']
    all_dfs[n].extend([dat_sigmoid.median(), dat_sigmoid_quantile, dat_sigmoid_variance])
    
    all_dfs[n] = pd.concat(all_dfs[n])


# In[21]:


'DD01' in list(df_to_plot_with_var.neuron_name.astype(str).values)


# In[22]:


# Add final columns
df_params = pd.concat(all_dfs, axis=1).T
df_params['dataset_type'] = 'residual'

df_params['muscle_position'] = muscle_position
df_params.loc['RID', 'muscle_position'] = np.nan  # RID is strange

_df = df_to_plot_with_var[df_to_plot_with_var['datatype']=='Freely Moving (GCaMP, residual)'].copy()
_df.index = _df['neuron_name']
df_params['Hierarchy Score'] = _df['Hierarchy Score']
df_params['Relative Hierarchy Score'] = _df['Relative Hierarchy Score']

df_params['Neuron Type'] = list(pd.Series(df_params.index).map(role_of_neuron_dict()))

df_params.head()


# In[41]:


from wbfm.utils.general.hardcoded_paths import role_of_neuron_dict
# Get radial term: combination of raw curvature amplitude and median of the sigmoid term
df_params['r'] = np.exp(df_params['log_amplitude_mu']) * df_params['sigmoid_term_quantile'] 
df_params['size'] = df_params['Relative Hierarchy Score'] + 1  # Add a minimum size

df_params['Neuron Type'] = pd.Series(df_params.index).map(role_of_neuron_dict(include_ventral_dorsal=True)).values

# r = df_params['log_amplitude_mu']
df_params['text'] = np.array(df_params.index)
df_params['text_complete'] = np.array(df_params.index)
df_params.loc[df_params['r'] < 0.1, 'text'] = ''
# size = 3*np.ones(len(df_params.index))
# size[r < 0.2] = 1
df_params.head()


# In[42]:


fig = px.scatter_polar(df_params[df_params['Neuron Type'].str.contains('Motor')], r='r', theta='phase_shift', direction='counterclockwise', start_angle=0,
                       text='text',
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

to_save = True
if to_save:
    fname = os.path.join("/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/bayesian_modeling/plots", 'phase_shift_and_oscillation_amplitude_only_motor.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)
    fname = Path(fname).with_suffix('.html')
    fig.write_html(fname)
    


# In[25]:


fig = px.scatter_polar(df_params, r='r', theta='phase_shift', direction='counterclockwise', start_angle=0,
                       text='text', #'text_complete',
                       color='Neuron Type', 
                       color_discrete_map=plotly_paper_color_discrete_map(),
                       size='Relative Hierarchy Score', size_max=15, #log_r=True,
                       #color='Neuron Role',
                       #color='muscle_position'
                      )
fig.update_traces(thetaunit='radians', textposition='top center', textfont_size=12, showlegend=False)

apply_figure_settings(fig, width_factor=0.4, height_factor=0.3)
# apply_figure_settings(fig, width_factor=1.0, height_factor=1.0)

# Show legend, but only for reference lines
fig.update_layout(polar=dict(
    angularaxis = dict(thetaunit = "radians"),
    radialaxis = dict(range=[0, 0.5],
                      nticks=3)
    ), 
    showlegend=False, 
    legend=dict(
        yanchor="bottom",
        y=0.05,
        xanchor="left",
        x=0.5,
        bordercolor="Black", borderwidth=1, bgcolor = 'white',
        font=dict(size=10),
        title='Body segment',
        # orientation='h',
    ))

# Add reference lines with manual coloring to match the example worm
cmap = ['#FF0F0F', '#CCCCCC', '#0000FF']
for i, (theta, seg_idx) in enumerate(zip([0, 90, 180], [6, 25, 37])):
    
    fig.add_trace(go.Scatterpolar(
            r = [0.35,0.6],
            theta = [theta, theta],
            mode = 'lines',
            line=dict(width=5, color=cmap[i]),
            # marker=dict(symbol='arrow'),
            name=seg_idx, opacity=1
            # text=f'<b>{seg_idx}</b>',
            # textposition='bottom center',
            # textfont=dict(
            #     size=18,
            #     family='Arial',
            #     color=cmap[i],
            # ),
        ))
    # Add Background rectangles directly and separately to get the background color option
    # fig.add_trace(go.Scatterpolar(
    #     r=[r0], 
    #     theta=[theta], 
    #     mode='markers',
    #     marker=dict(
    #         size=20,  # Size of the background
    #         color='white',  # Background color
    #         # opacity=0.5,
    #         symbol='square'
    #     ),
    #     showlegend=False
    # ))
    

fig.show()

to_save = True
if to_save:
    fname = os.path.join("/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/bayesian_modeling/plots", 'phase_shift_and_oscillation_amplitude.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)
    fname = Path(fname).with_suffix('.html')
    fig.write_html(fname)
    


# ## Same, but no text

# In[26]:


fig = px.scatter_polar(df_params[df_params['Neuron Type'].str.contains('Motor')], r='r', theta='phase_shift', direction='counterclockwise', start_angle=0,
                       color='Neuron Type', 
                       color_discrete_map=plotly_paper_color_discrete_map(),
                       size='Relative Hierarchy Score', size_max=15, #log_r=True,
                       #color='Neuron Role',
                       #color='muscle_position'
                      )
fig.update_traces(thetaunit='radians', textposition='top center', textfont_size=12, )

apply_figure_settings(fig, width_factor=0.4, height_factor=0.4)
# apply_figure_settings(fig, width_factor=1.0, height_factor=1.0)

fig.update_layout(polar=dict(
    angularaxis = dict(thetaunit = "radians"),
    radialaxis = dict(range=[0, 0.5],
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

to_save = True
if to_save:
    fname = os.path.join("/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/bayesian_modeling/plots", 'phase_shift_and_oscillation_amplitude_only_motor-no_text.png')
    fig.write_image(fname, scale=3)
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(fname)
    fname = Path(fname).with_suffix('.html')
    fig.write_html(fname)
    


# ### Specific segment extra annotations

# In[27]:


import sklearn, math

x = ['eigenworm1', 'eigenworm2']
all_theta = []

for i in range(1, 100):
    y = f'curvature_{i}'
    # print(y)

    Xy_for_model = Xy[x].join(Xy[y]).dropna()

    model = sklearn.linear_model.LinearRegression(fit_intercept=False)
    model.fit(Xy_for_model[x], Xy_for_model[y])

    # Convert to polar coordinates
    cx, cy = model.coef_
    r = np.sqrt(cx**2 + cy**2)
    # theta = np.arctan2(cy, cx)
    theta = np.arctan(cy/cx)
    all_theta.append(theta)
    # print(r, 360*theta/(2*math.pi))
fig = px.line(all_theta)
fig.add_hline(y=math.pi/2)
# fig.add_hline(y=math.pi)
fig.add_hline(y=-math.pi/2)


# In[ ]:





# In[ ]:





# ## Sigmoid slope

# In[28]:


df_params.head()


# In[29]:


# Just plot slope median and variance
fig = px.scatter(df_params.sort_values(by='sigmoid_term'), y='sigmoid_term', x='sigmoid_term_variance',#x=df_params.index,
            text=df_params.index)

apply_figure_settings(fig, width_factor=1.0, height_factor=0.3)
fig.show()


# In[30]:


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


# In[31]:


df_hierarchy_melt = df_hierarchy.melt(var_name=['Variable', 'Neuron Name'], value_name='PC1 Coefficient', value_vars=df_hierarchy.columns.tolist())
# df_hierarchy_melt['Neuron Type'] = df_hierarchy_melt['Neuron Name'].map(df_params['Neuron Type'].to_dict())
# df_hierarchy_melt['Neuron Type'] = df_hierarchy_melt['Neuron Name'].map(role_of_neuron_dict(include_fwd_rev=True, include_ventral_dorsal=True, include_basic=False)).replace('', 'Other')
df_hierarchy_melt['Neuron Type'] = df_hierarchy_melt['Neuron Name'].map(role_of_neuron_dict(only_fwd_rev=True)).replace('', 'Other')
df_hierarchy_melt['Neuron Type Complex'] = df_hierarchy_melt['Neuron Name'].map(role_of_neuron_dict(include_fwd_rev=True, include_ventral_dorsal=True, include_basic=False)).replace('', 'Other')
df_hierarchy_melt['Neuron Type VD'] = df_hierarchy_melt['Neuron Name'].map(role_of_neuron_dict(include_ventral_dorsal=True)).replace('', 'Other')


# In[32]:


df_hierarchy_melt['Neuron Type'].unique()


# In[33]:


df_hierarchy_melt['Variable'].unique()


# In[34]:


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


# In[35]:


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


# In[36]:


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


# In[37]:


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




