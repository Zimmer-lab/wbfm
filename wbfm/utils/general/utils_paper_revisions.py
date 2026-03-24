import numpy as np
import pandas as pd
import plotly.express as px

from scipy import stats
from statsmodels.stats.multitest import multipletests

from wbfm.utils.general.utils_paper import apply_figure_settings, plotly_paper_color_discrete_map, data_type_name_mapping
from wbfm.utils.visualization.utils_plot_traces import add_p_value_annotation



def calc_statistics_for_pc1_comparison_plots(pc1_weights_all_conditions, keys_to_compare):
    opts_multipletests = dict(method='fdr_bh', alpha=0.05)
    # opts_multipletests = dict(method='sidak')

    pc1_weights1 = pc1_weights_all_conditions[keys_to_compare[0]]
    pc1_weights2 = pc1_weights_all_conditions[keys_to_compare[1]]

    def _key2type(k):
        if isinstance(k, str):
            return k
        elif isinstance(k, tuple):
            return str(k[1])
            # return f"condition{k[0]}_wavelength{k[1]}"

    names_to_keep = set(pc1_weights1.columns).intersection(pc1_weights2.columns)
    wbfm_melt = pc1_weights1.melt(var_name='neuron_name', value_name='PC1 weight').assign(dataset_type=_key2type(keys_to_compare[0]))
    immob_melt = pc1_weights2.melt(var_name='neuron_name', value_name='PC1 weight').assign(dataset_type=_key2type(keys_to_compare[1]))
    df_both = pd.concat([wbfm_melt, immob_melt], axis=0)
    df_both = df_both[df_both['neuron_name'].isin(names_to_keep)]
    df_both['Dataset Type'] = df_both['dataset_type'].map(lambda k: data_type_name_mapping().get(k, k))
    print(len(df_both['neuron_name'].unique()))

    # Significantly different from 0... need a permutation version, so use an extra function
    # From: https://stackoverflow.com/questions/73569894/permutation-based-alternative-to-scipy-stats-ttest-1samp
    def _t_statistic(x, axis=-1):
        # return stats.ttest_1samp(x, popmean=0, axis=axis).statistic
        return stats.ttest_1samp(x, popmean=0).statistic

    def t_statistic_permutation(x):
        try:
            return stats.permutation_test((x.values, ), _t_statistic, permutation_type='samples', ).pvalue
        except ValueError:
            return 0

    # func = lambda x: stats.ttest_1samp(x, 0)[1]
    df_groupby = df_both.dropna().groupby(['neuron_name', 'dataset_type'])
    df_pvalue = df_groupby['PC1 weight'].apply(t_statistic_permutation).to_frame()
    df_pvalue.columns = ['p_value']

    # Multiple comparison correction in the same way for all tests
    output = multipletests(df_pvalue.values.squeeze(), **opts_multipletests)
    df_pvalue['p_value_corrected'] = output[1]
    df_pvalue['significance_corrected'] = output[0]

    # Sign of medians
    df_medians_gcamp = df_groupby['PC1 weight'].median()[(slice(None), _key2type(keys_to_compare[0]))]
    df_medians_immob = df_groupby['PC1 weight'].median()[(slice(None), _key2type(keys_to_compare[1]))]

    # Significantly different from each other (should be exact same as the boxplot)
    df_groupby = df_both.dropna().groupby(['neuron_name'])
    func = lambda x: stats.ttest_ind(x[x['dataset_type']==_key2type(keys_to_compare[0])]['PC1 weight'], x[x['dataset_type']==_key2type(keys_to_compare[1])]['PC1 weight'], 
                                    equal_var=False, permutations=1000)[1]
    df_significant_diff = df_groupby.apply(func).to_frame()
    df_significant_diff.columns = ['p_value_diff']
    # Multiple comparison correction in the same way for all tests
    output = multipletests(df_significant_diff.values.squeeze(), **opts_multipletests)
    df_significant_diff['p_value_corrected_diff'] = output[1]
    df_significant_diff['significance_corrected_diff'] = output[0]
    # df_significant_diff.head()

    return df_both, df_pvalue, df_medians_gcamp, df_medians_immob, df_significant_diff


def plot_pc1_comparison(df_both, df_significant_diff, minimum_number_neurons=2, x_order=None):
    df_both = df_both.copy()

    if minimum_number_neurons > 1:
        neuron_counts = df_both.dropna().groupby(['neuron_name', 'Dataset Type']).size().unstack(fill_value=0)
        neurons_to_keep = neuron_counts[(neuron_counts >= minimum_number_neurons).all(axis=1)].index
        df_both = df_both[df_both['neuron_name'].isin(neurons_to_keep)]
        print(f"After filtering for minimum {minimum_number_neurons} neurons per dataset type, {len(neurons_to_keep)} neurons remain.")

    # Plot
    x_name = 'neuron_name_html' if 'neuron_name_html' in df_both.columns else 'neuron_name'
    if x_order is not None:
        df_both[x_name] = pd.Categorical(df_both[x_name], categories=x_order, ordered=True)
        df_both = df_both.sort_values(x_name)

    fig = px.box(df_both, y='PC1 weight', x=x_name, 
                 color='Dataset Type', 
                 color_discrete_map=plotly_paper_color_discrete_map(),
                 category_orders={'Dataset Type': ['Immobilized (GCaMP)', 'Freely Moving (GCaMP)', 488, 505, '488', '505']})

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

    fig.update_yaxes(dict(title="PC1 weight"), zeroline=True, zerolinewidth=1, zerolinecolor="black", overwrite=True)#range=[-0.2, 0.55])
    fig.update_xaxes(dict(title="Neuron Name"))
    fig.show()

    return fig
