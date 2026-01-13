import os
import plotly.express as px
import pandas as pd
from scipy import stats
import numpy as np
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm

from wbfm.utils.general.utils_paper import apply_figure_settings, plotly_paper_color_discrete_map
from wbfm.utils.general.utils_hardcoded import (get_hierarchical_modeling_dir, role_of_neuron_dict,
                                                neurons_with_confident_ids)


def main(combine_left_right=True, verbose=0):
    output_folder = '/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/intro/dimensionality'

    # Import traces (gcamp and immob)
    parent_folder = get_hierarchical_modeling_dir()
    fname = 'data.h5'
    df_gcamp = pd.read_hdf(os.path.join(parent_folder, fname))
    parent_folder = get_hierarchical_modeling_dir(immobilized=True)
    fname = 'data.h5'
    df_immob = pd.read_hdf(os.path.join(parent_folder, fname))

    # Get the columns we want: variance explained by the manifold in %
    # First get the shared neuron names
    shared_neurons = df_gcamp.columns.intersection(df_immob.columns)
    nontarget_strings = ['neuron', 'manifold', 'pca', 'local_time', 'dataset']
    for string in nontarget_strings:
        shared_neurons = shared_neurons[~shared_neurons.str.contains(string)]

    # Get the target columns by calculating the variance explained by the raw trace and the manifold, and the ratio
    # But do this per dataset
    all_dfs = []
    df_gcamp_grouped = df_gcamp.groupby('dataset_name')
    df_immob_grouped = df_immob.groupby('dataset_name')

    for neuron in shared_neurons:
        # Gcamp
        var_raw = df_gcamp_grouped[neuron].var()
        var_manifold = df_gcamp_grouped[neuron + '_manifold'].var()
        ratio = var_manifold / var_raw
        # Make a small df from this series
        _df = pd.DataFrame(ratio, columns=['variance_explained'])
        _df['neuron'] = neuron
        _df['dataset_type'] = 'Freely Moving'
        all_dfs.append(_df)
        # Immob
        var_raw = df_immob_grouped[neuron].var()
        var_manifold = df_immob_grouped[neuron + '_manifold'].var()
        ratio = var_manifold / var_raw
        # Make a small df from this series
        _df = pd.DataFrame(ratio, columns=['variance_explained'])
        _df['neuron'] = neuron
        _df['dataset_type'] = 'Immobilized'
        all_dfs.append(_df)

    # Create a dataframe and plot via plotly
    df = pd.concat(all_dfs)

    # New column for left/right combined neurons (remove 'L' or 'R' from the name, if it is the last character)
    def _neuron_pair_name(neuron_name):
        return neuron_name[:-1] if (neuron_name[-1] in ['L', 'R'] and len(neuron_name) > 3) else neuron_name
    df['neuron_combined'] = df['neuron'].apply(_neuron_pair_name)
    neuron_row_name = 'neuron_combined' if combine_left_right else 'neuron'

    df.dropna(inplace=True)
    # Only include neurons with more than the mininum neurons in both immob and gcamp
    min_neurons = 3
    neuron_counts_gcamp = df[df['dataset_type'] == 'Freely Moving'][neuron_row_name].value_counts()
    neuron_counts_immob = df[df['dataset_type'] == 'Immobilized'][neuron_row_name].value_counts()
    enough_gcamp = neuron_counts_gcamp[neuron_counts_gcamp >= min_neurons].index
    enough_immob = neuron_counts_immob[neuron_counts_immob >= min_neurons].index
    enough_neurons = enough_gcamp.intersection(enough_immob).intersection(
        neurons_with_confident_ids(combine_left_right))
    if verbose >= 1:
        print(f'Number of neurons with at least {min_neurons} datasets in both conditions: {len(enough_neurons)}')
        if verbose >= 2:
            print(f'Neurons: {enough_neurons.tolist()}')
            print(f'Immob neurons: {enough_immob.tolist()}')
            print(f'Gcamp neurons: {enough_gcamp.tolist()}')
    df = df[df[neuron_row_name].isin(enough_neurons)]

    # Sort by mean variance explained in immob (not gcamp) per neuron
    # df_grouped = df[df['dataset_type'] == 'Immobilized'].groupby('neuron_combined')
    # mean_var_explained = df_grouped['variance_explained'].mean()
    # # copy mean_var_explained to all rows of the same neuron, including gcamp
    # df = df.set_index('neuron_combined')
    # df['mean_var_explained'] = mean_var_explained
    # df = df.sort_values(by='mean_var_explained', ascending=True).reset_index()

    # Sort by difference in variance explained between immob and gcamp
    df_grouped = df[df['dataset_type'] == 'Freely Moving'].groupby(neuron_row_name)
    mean_var_explained_gcamp = df_grouped['variance_explained'].median()
    df_grouped = df[df['dataset_type'] == 'Immobilized'].groupby(neuron_row_name)
    mean_var_explained_immob = df_grouped['variance_explained'].median()
    diff_var_explained = mean_var_explained_immob - mean_var_explained_gcamp
    df = df.set_index(neuron_row_name)
    df['diff_var_explained'] = diff_var_explained
    df = df.sort_values(by='diff_var_explained', ascending=True).reset_index()

    fig = px.box(df, x=neuron_row_name, y='variance_explained', color='dataset_type', #points='all',
                 color_discrete_map=plotly_paper_color_discrete_map())
                 #title='Variance explained by the manifold')
    apply_figure_settings(fig, width_factor=1.0, height_factor=0.25)
    fig.update_yaxes(title='Var. explained by PC1+2')
    fig.update_xaxes(title='Neuron', tickfont_size=10)

    fname = os.path.join(output_folder, f'variance_explained_boxplot-LRcombined{combine_left_right}.svg')
    fig.write_image(fname)
    fname = fname.replace('.svg', '.png')
    fig.write_image(fname, scale=3)

    fig.show(renderer='browser')

    # Calculate the p values and effect sizes of the difference in variance explained between immob and gcamp
    # for each neuron
    p_values = {}
    effect_sizes = {}
    permutations = 100000
    for neuron in tqdm(enough_neurons):
        df_neuron = df[df[neuron_row_name] == neuron]
        immob = df_neuron[df_neuron['dataset_type'] == 'Immobilized']['variance_explained']
        gcamp = df_neuron[df_neuron['dataset_type'] == 'Freely Moving']['variance_explained']
        p_value = stats.ttest_ind(immob, gcamp, equal_var=False, random_state=4242, permutations=permutations)[1]
        # p_value = stats.ttest_ind(immob, gcamp, equal_var=False, random_state=4242)[1]# permutations=permutations)[1]
        effect_size = immob.median() - gcamp.median()
        p_values[neuron] = p_value
        effect_sizes[neuron] = effect_size

    # Second plot: volcano plot of p values and effect sizes
    df_p_values = pd.DataFrame.from_dict(p_values, orient='index', columns=['p_value_raw'])
    df_effect_sizes = pd.DataFrame.from_dict(effect_sizes, orient='index', columns=['effect_size'])
    df_combined = df_p_values.join(df_effect_sizes)

    # Correct for multiple comparisons, same as other figures
    alpha = 0.05
    output = multipletests(df_combined['p_value_raw'].values.squeeze(), method='fdr_bh', alpha=alpha)
    df_combined['p_value_corrected'] = output[1]
    df_combined['significant'] = output[0]

    df_combined['median_immob'] = df_combined.index.map(
        lambda x: df[(df['neuron_combined'] == x) & (df['dataset_type'] == 'Immobilized')]['variance_explained'].median())
    df_combined['median_gcamp'] = df_combined.index.map(
        lambda x: df[(df['neuron_combined'] == x) & (df['dataset_type'] == 'Freely Moving')]['variance_explained'].median())
    df_combined['minus_log_p'] = -np.log10(df_combined['p_value_corrected'] + 1e-6)
    df_combined['neuron_name'] = df_combined.index
    df_combined['role'] = df_combined['neuron_name'].map(role_of_neuron_dict())
    df_combined['fwd_rev'] = df_combined['neuron_name'].map(lambda x: role_of_neuron_dict(only_fwd_rev=True).get(x, ''))
    # Concatenate columns
    new_col = df_combined['role'] + ', ' + df_combined['fwd_rev']
    # Combine all motor roles
    new_col = new_col.map(lambda x: x if 'Motor' not in x else 'Motor')
    new_col = new_col.str.replace('Interneuron', 'Inter')
    new_col = new_col.map(lambda x: x if x != 'Inter, ' else 'Interneuron')
    new_col = new_col.str.replace('Sensory, ', 'Sensory')
    new_col = new_col.str.replace('SensorySensory', 'Sensory')
    new_col = new_col.str.replace('SensoryReverse', 'Sensory')
    # new_col = new_col.str.replace('Forward', 'Forward')
    # new_col = new_col.str.replace('Reverse', 'Reverse')
    df_combined['combined_role'] = new_col
    # Add a black line around the points
    fig = px.scatter(df_combined, x='effect_size', y='minus_log_p',
                     # symbol='role', color='fwd_rev',
                     color='combined_role',
                     color_discrete_map=plotly_paper_color_discrete_map(),
                     category_orders={'combined_role': ['Sensory', 'Interneuron', 'Motor', 'Inter, Forward', 'Inter, Reverse']},
                     hover_name=df_combined.index)
                     #title='Effect sizes and p values of the difference in variance explained between immob and gcamp')

    # Add significance line (horizontal at -log10(0.05))
    # Don't change the x axis range though
    # x_min, x_max = fig.full_figure_for_development().layout.xaxis.range
    x_min, x_max = -0.51, 0.51
    fig.add_shape(type='line', x0=x_min, y0=-np.log10(alpha), x1=x_max, y1=-np.log10(alpha),
                  line=dict(color='black', width=1, dash='dash'))

    fig.update_xaxes(title='Change in Dimensionality <br> in Freely Moving')
    fig.update_yaxes(title='-log10(p value)')
    fig.update_layout(legend_title='Neuron Type')
    # apply_figure_settings(fig, width_factor=1.0, height_factor=1.0)
    apply_figure_settings(fig, width_factor=0.45, height_factor=0.25)
    fig.update_traces(marker=dict(size=12, line=dict(width=0.5, color='Black')))#, opacity=0.7)

    # Save
    output_folder = '/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper/intro/dimensionality'
    fname = os.path.join(output_folder, f'variance_explained_difference-LRcombined{combine_left_right}.svg')
    fig.write_image(fname)
    fname = fname.replace('.svg', '.png')
    fig.write_image(fname, scale=3)
    fname = fname.replace('.png', '.html')
    fig.write_html(fname)

    fig.show(renderer='browser')

    # Additional plot: scatter plot of immob vs gcamp variance explained
    # fig = px.scatter(df_combined, x='median_immob', y='median_gcamp', color='combined_role',
    #                  #text='neuron_name',
    #                  size='minus_log_p',
    #                  color_discrete_map=mapping,
    #                  title='Variance explained by the manifold')
    # # fig.update_traces(textposition='top center')
    # apply_figure_settings(fig, width_factor=0.5, height_factor=0.3)
    # # Add y=x line
    # fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(color='black', width=1))
    #
    # fig.show(renderer='browser')


if __name__ == '__main__':
    main(combine_left_right=False)
    main(combine_left_right=True)
