#!/usr/bin/env python3
"""PCA per dataset and boxplot of PC1 loadings for labeled neurons."""

import argparse
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from load_neurons import load_neurons
from wbfm.utils.external.utils_pandas import combine_columns_with_suffix
from wbfm.utils.external.utils_plotly import plotly_plot_mean_and_shading
from wbfm.utils.general.utils_hardcoded import neurons_with_confident_ids
from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
from wbfm.utils.general.utils_paper import apply_figure_settings, plotly_paper_color_discrete_map, behavior_name_mapping
import plotly.graph_objects as go
from wbfm.utils.visualization.utils_plot_traces import add_p_value_annotation, modify_dataframe_to_allow_gaps_for_plotly

BEHAVIOR_COLUMNS = ['velocity', 'angular_velocity', 'head_curvature', 'body_curvature', 'pumping', 'reversal', 'speed']
METADATA_COLUMNS = ['dataset_name', 'local_time', 'source']
COLUMN_EXCLUDE_PATTERNS = ['_manifold', 'eigenworm', 'curvature', '_frequency', '_turn', '_collision', 
                           'CCA_neural_mode', 'local_time_physical', 'pca_', 'rev', 'pause']

# Page layout (fractions of an A4 page, see apply_figure_settings):
#   row 1: PC1 weight boxplots, full width
#   row 2: variance explained (no legend) + two example manifolds (legend on the first)
WEIGHTS_WIDTH_FACTOR, WEIGHTS_HEIGHT_FACTOR = 1.0, 0.3
PANEL_WIDTH_FACTOR, PANEL_HEIGHT_FACTOR = 0.33, 0.25
PHASE_PLOT_TOP_MARGIN = 80 # pixels; room for the legend above the example manifold
EXAMPLE_PHASE_DATASETS = ['atanas_kim_2023-2023-01-23-01', 'atanas_kim_2023-2022-07-20-01']
# Source whose medians set the neuron order (falls back to the next if absent), to match the main paper
SORT_SOURCE_PRIORITY = ['zimmer', 'flavell', 'immob']


def get_neuron_columns(df):
    """Get columns that are neurons (not metadata, behavior, or derived columns)."""
    exclude_cols = set(METADATA_COLUMNS + BEHAVIOR_COLUMNS)
    for c in df.columns:
        if any(p in c for p in COLUMN_EXCLUDE_PATTERNS):
            exclude_cols.add(c)
    return [c for c in df.columns if c not in exclude_cols]


def get_labeled_neurons(neuron_cols):
    """Get neurons with real labels (not matching neuron_XXX pattern)."""
    return [c for c in neuron_cols if not c.startswith('neuron_')]


def count_neuron_datasets(df, neuron_cols):
    """Count how many datasets each neuron appears in."""
    counts = {n: 0 for n in neuron_cols}
    for dataset_name, group in df.groupby('dataset_name'):
        for neuron in neuron_cols:
            if neuron in group.columns:
                if not group[neuron].isna().all():
                    counts[neuron] += 1
    return counts


def count_neuron_datasets_by_source(df, neuron_cols):
    """Count how many datasets each neuron appears in, broken down by source."""
    counts = {}
    for neuron in neuron_cols:
        counts[neuron] = {}
    for (dataset_name, source), group in df.groupby(['dataset_name', 'source']):
        for neuron in neuron_cols:
            if neuron in group.columns:
                if not group[neuron].isna().all():
                    if source not in counts[neuron]:
                        counts[neuron][source] = 0
                    counts[neuron][source] += 1
    return counts


def get_neurons_in_group(group, neuron_cols):
    """Get neurons that exist in this dataset (not all NaN)."""
    available = []
    for neuron in neuron_cols:
        if neuron in group.columns:
            if not group[neuron].isna().all():
                available.append(neuron)
    return available


def pca_per_dataset(df, all_neurons, labeled_neurons_to_return, anchor_neuron='AVA', n_components=1, zscore_neurons=False, nan_log=None):
    """Run PCA per dataset using ALL neurons, but return loadings only for labeled neurons.
    
    Anchors PCA by flipping sign so anchor_neuron has positive loading.
    
    Args:
        df: DataFrame with neuron columns
        all_neurons: List of all neuron column names to use for PCA
        labeled_neurons_to_return: List of labeled neurons to return loadings for
        anchor_neuron: Neuron to use for anchoring PCA sign
        n_components: Number of PCA components
        zscore_neurons: Whether to z-score neurons before PCA
        nan_log: Optional list to collect partial NaN info (dataset, neurons with partial NaN)
    """
    if nan_log is None:
        nan_log = []
    results = {}
    pc_scores = {}
    variance_explained = {}

    for dataset_name, group in df.groupby('dataset_name'):
        available_all = get_neurons_in_group(group, all_neurons)
        available_labeled = get_neurons_in_group(group, labeled_neurons_to_return)
        
        X = group[available_all].values.astype(np.float64)
        
        nan_by_neuron = np.isnan(X).all(axis=0)
        partial_nan = np.isnan(X).any(axis=0) & ~nan_by_neuron
        
        if partial_nan.any():
            neurons_with_partial_nan = [n for n, has_partial in zip(available_all, partial_nan) if has_partial]
            nan_log.append({
                'dataset': dataset_name,
                'neurons_with_partial_nan': neurons_with_partial_nan
            })
        
        X = X.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0)
        
        if zscore_neurons:
            X_std = StandardScaler().fit_transform(X)
        else:
            X_std = X
        
        pca = PCA(n_components=n_components)
        pca.fit(X_std)
        
        loadings_all = pca.components_[0]
        neuron_to_loading = dict(zip(available_all, loadings_all))
        neuron_to_idx = {n: i for i, n in enumerate(available_all)}
        
        flip_sign = anchor_neuron in available_labeled and neuron_to_loading[anchor_neuron] < 0
        if flip_sign:
            neuron_to_loading = {k: -v for k, v in neuron_to_loading.items()}

        results[dataset_name] = {n: neuron_to_loading[n] for n in labeled_neurons_to_return if n in neuron_to_loading}

        scores = pca.transform(X_std)
        if flip_sign:
            # Keep the PC1 scores consistent with the flipped loadings (high AVA = positive PC1)
            scores[:, 0] = -scores[:, 0]
        pc_scores[dataset_name] = scores
        
        variance_explained[dataset_name] = pca.explained_variance_ratio_.copy()
    
    return results, nan_log, pc_scores, variance_explained


def compute_variance_explained(df, neuron_cols, n_components=20, zscore_neurons=False):
    """Run PCA per dataset and return explained variance ratios.
    
    Args:
        df: DataFrame with neuron columns
        neuron_cols: List of neuron column names to use for PCA
        n_components: Number of PCA components
        zscore_neurons: Whether to z-score neurons before PCA
        
    Returns:
        dict: dataset_name -> array of explained variance ratios
    """
    variance_explained = {}
    
    for dataset_name, group in df.groupby('dataset_name'):
        available = get_neurons_in_group(group, neuron_cols)
        if len(available) < 2:
            continue
            
        X = group[available].values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0)
        
        if zscore_neurons:
            X_std = StandardScaler().fit_transform(X)
        else:
            X_std = X
        
        n_comp = min(n_components, X_std.shape[0], X_std.shape[1])
        if n_comp < 2:
            continue
            
        pca = PCA(n_components=n_comp)
        pca.fit(X_std)
        variance_explained[dataset_name] = pca.explained_variance_ratio_.copy()
    
    return variance_explained


def sort_neurons_by_median(plot_df):
    """Sort neurons by median PC1 loading (decreasing)."""
    medians = plot_df.groupby('neuron')['PC1_loading'].median()
    sorted_neurons = medians.sort_values(ascending=False).index.tolist()
    plot_df['neuron'] = pd.Categorical(plot_df['neuron'], categories=sorted_neurons, ordered=True)
    return plot_df


def combine_suffixes_per_dataset(df):
    """Apply combine_columns_with_suffix per dataset (after dropping all-NaN cols)."""
    dfs = []
    for dataset_name, group in df.groupby('dataset_name'):
        group = group.dropna(axis=1, how='all')
        group = combine_columns_with_suffix(group)
        dfs.append(group)
    return pd.concat(dfs, ignore_index=True)


def compute_neuron_ava_correlation(df, neurons):
    """Compute correlation between each neuron and AVA per dataset."""
    results = []
    for dataset_name, group in df.groupby('dataset_name'):
        if 'AVA' not in group.columns:
            continue
        ava_vals = group['AVA'].values
        if np.isnan(ava_vals).all():
            continue
        
        for neuron in neurons:
            if neuron == 'AVA' or neuron not in group.columns:
                continue
            neuron_vals = group[neuron].values
            if np.isnan(neuron_vals).all():
                continue
            
            mask = ~np.isnan(ava_vals) & ~np.isnan(neuron_vals)
            if mask.sum() > 10:
                corr = np.corrcoef(ava_vals[mask], neuron_vals[mask])[0, 1]
                if not np.isnan(corr):
                    results.append({
                        'dataset': dataset_name,
                        'neuron': neuron,
                        'correlation': corr
                    })
    return pd.DataFrame(results)


def print_diagnostic_correlations(df, neurons, anchor_neuron='AVA'):
    """Print correlation with AVA for neurons."""
    print(f"\n=== Diagnostic: Correlation with {anchor_neuron} ===")
    
    corr_df = compute_neuron_ava_correlation(df, neurons)
    
    # Summary statistics per neuron
    corr_summary = corr_df.groupby('neuron')['correlation'].agg(['mean', 'std', 'min', 'max', 'count'])
    corr_summary = corr_summary.sort_values('mean', ascending=False)
    
    print(f"\nCorrelation with {anchor_neuron} (mean ± std [min, max] across datasets):")
    print("-" * 65)
    for neuron in corr_summary.index:
        row = corr_summary.loc[neuron]
        marker = " <-- ANCHOR" if neuron == anchor_neuron else ""
        print(f"  {neuron:10s}: {row['mean']:+.3f} ± {row['std']:.3f} [{row['min']:+.3f}, {row['max']:+.3f}] (n={int(row['count'])}){marker}")
    
    return corr_summary


def prepare_dataframe(include_hierarchical=False, include_immob=False):
    """Load and prepare the neuron dataframe.
    
    Args:
        include_hierarchical: Include hierarchical data (zimmer)
        include_immob: Include immob data
    
    Returns:
        DataFrame with loaded and prepared neuron data
    """
    df = load_neurons(include_hierarchical=include_hierarchical, include_immob=include_immob)
    
    print("\nCombining L/R neuron suffixes (per dataset)...")
    df = combine_suffixes_per_dataset(df)
    
    # Filter to datasets that have AVA (required anchor)
    datasets_with_ava = []
    for dataset_name, group in df.groupby('dataset_name'):
        if 'AVA' in group.columns and group['AVA'].notna().sum() > 0:
            datasets_with_ava.append(dataset_name)
    
    original_count = df['dataset_name'].nunique()
    df = df[df['dataset_name'].isin(datasets_with_ava)]
    print(f"Filtered to {len(datasets_with_ava)}/{original_count} datasets with AVA")
    
    return df


def filter_neurons_by_coverage(df, neuron_cols, labeled_neurons, min_datasets=5, pairwise_min_datasets=3):
    """Filter neurons by coverage across datasets.
    
    Args:
        df: DataFrame with neuron data
        neuron_cols: List of all neuron column names
        labeled_neurons: List of labeled (non-neuron_XXX) neurons
        min_datasets: Minimum datasets for "all" neurons
        pairwise_min_datasets: Minimum datasets for pairwise comparisons
    
    Returns:
        dict with keys: 'all', 'flavell_zimmer', 'flavell_immob', 'zimmer_immob'
    """
    confident_list = neurons_with_confident_ids(combine_left_right=True)
    confident_set = set(confident_list)
    print(f"Confident neuron base names: {len(confident_set)}")
    
    dataset_counts_by_source = count_neuron_datasets_by_source(df, neuron_cols)
    sources = df['source'].unique().tolist()
    
    def get_neurons_for_pair(sources_pair, counts_dict, confident_set, labeled_neurons, min_datasets):
        """Get neurons that have min_datasets in both sources of the pair."""
        neurons = []
        for neuron in labeled_neurons:
            if neuron not in confident_set:
                continue
            counts = counts_dict.get(neuron, {})
            
            passes = True
            for src in sources_pair:
                if counts.get(src, 0) < min_datasets:
                    passes = False
                    break
            if passes:
                neurons.append(neuron)
        
        return neurons
    
    # Get neurons for each comparison
    plot_neurons_all = get_neurons_for_pair(sources, dataset_counts_by_source, confident_set, labeled_neurons, min_datasets)
    
    neurons_fz = []
    neurons_fi = []
    neurons_zi = []
    
    if 'flavell' in sources and 'zimmer' in sources:
        neurons_fz = get_neurons_for_pair(['flavell', 'zimmer'], dataset_counts_by_source, confident_set, labeled_neurons, pairwise_min_datasets)
    if 'flavell' in sources and 'immob' in sources:
        neurons_fi = get_neurons_for_pair(['flavell', 'immob'], dataset_counts_by_source, confident_set, labeled_neurons, pairwise_min_datasets)
    if 'zimmer' in sources and 'immob' in sources:
        neurons_zi = get_neurons_for_pair(['zimmer', 'immob'], dataset_counts_by_source, confident_set, labeled_neurons, pairwise_min_datasets)
    
    print(f"Plot neurons (all {len(sources)} sources): {len(plot_neurons_all)}")
    print(f"Plot neurons (flavell vs zimmer): {len(neurons_fz)}")
    print(f"Plot neurons (flavell vs immob): {len(neurons_fi)}")
    print(f"Plot neurons (zimmer vs immob): {len(neurons_zi)}")
    print(f"AVA in all neurons: {'AVA' in plot_neurons_all}")
    
    return {
        'all': plot_neurons_all,
        'flavell_zimmer': neurons_fz,
        'flavell_immob': neurons_fi,
        'zimmer_immob': neurons_zi
    }


def build_pca_plot_dataframe(pca_results, df):
    """Build DataFrame for plotting from PCA results.
    
    Args:
        pca_results: Dict of dataset_name -> dict of neuron -> PC1 loading
        df: Original DataFrame (used for source information)
    
    Returns:
        DataFrame with columns: dataset, neuron, PC1_loading, source
    """
    data_for_plot = []
    for dataset_name, loadings in pca_results.items():
        for neuron, loading in loadings.items():
            row = {
                'dataset': dataset_name,
                'neuron': neuron,
                'PC1_loading': loading
            }
            if 'source' in df.columns:
                group = df[df['dataset_name'] == dataset_name]
                if 'source' in group.columns:
                    row['source'] = group['source'].iloc[0]
            data_for_plot.append(row)
    
    return pd.DataFrame(data_for_plot)


def compute_pvalues_for_comparison(plot_df, src1, src2):
    """Compute p-values for a pairwise comparison.
    
    Args:
        plot_df: DataFrame with PC1_loading, neuron, source columns
        src1: First source name
        src2: Second source name
    
    Returns:
        dict of neuron -> corrected p-value
    """
    from scipy import stats
    from statsmodels.stats.multitest import multipletests
    
    p_value_dict = {}
    
    try:
        df_src1 = plot_df[plot_df['source'] == src1].pivot(index='neuron', columns='dataset', values='PC1_loading')
        df_src2 = plot_df[plot_df['source'] == src2].pivot(index='neuron', columns='dataset', values='PC1_loading')
        
        if df_src1.shape[0] > 0 and df_src2.shape[0] > 0:
            common_neurons = set(df_src1.index).intersection(set(df_src2.index))
            p_values = {}
            for neuron in common_neurons:
                vals1 = df_src1.loc[neuron].dropna().values
                vals2 = df_src2.loc[neuron].dropna().values
                if len(vals1) >= 3 and len(vals2) >= 3:
                    _, p = stats.ttest_ind(vals1, vals2, equal_var=False)
                    p_values[neuron] = p
            
            if len(p_values) > 0:
                neurons = list(p_values.keys())
                p_arr = np.array([p_values[n] for n in neurons])
                _, p_corrected, _, _ = multipletests(p_arr, method='fdr_bh')
                p_value_dict = {n: p_corrected[i] for i, n in enumerate(neurons)}
                print(f"Calculated p-values for {src1} vs {src2}: {len(p_value_dict)} neurons")
    except Exception as e:
        print(f"Warning: Could not calculate p-values for {src1} vs {src2}: {e}")
    
    return p_value_dict


def make_phase_plot_data(pc_scores, df):
    """Build DataFrame for phase plots.
    
    Args:
        pc_scores: Dict of dataset_name -> PCA scores array (n_timepoints, n_components)
        df: Original DataFrame (used for source and behavioral data)
    
    Returns:
        DataFrame with columns: PC0, PC1, dataset, source, timepoint, reversal
    """
    phase_data = []
    for dataset_name, scores in pc_scores.items():
        group = df[df['dataset_name'] == dataset_name]
        source = group['source'].iloc[0] if 'source' in group.columns else 'unknown'
        n_timepoints = scores.shape[0]
        phase_data.append(pd.DataFrame({
            'PC0': scores[:, 0],
            'PC1': scores[:, 1],
            'dataset': dataset_name,
            'source': source,
            'timepoint': np.arange(n_timepoints),
            'reversal': group['reversal'].values if 'reversal' in group.columns else np.nan,
        }))
    
    return pd.concat(phase_data, ignore_index=True)


def make_boxplot(df_subset, filename, title, sort_neurons=None):
    """Create a boxplot of PC1 loadings using plotly.
    
    Args:
        df_subset: DataFrame with 'neuron', 'PC1_loading', and optionally 'source' columns
        filename: Base filename for saving (without extension)
        title: Plot title
        sort_neurons: Optional list of neuron names to order the x-axis (by median PC1 loading)
    """
    if len(df_subset) == 0:
        return
    if sort_neurons is None:
        medians = df_subset.groupby('neuron')['PC1_loading'].median().sort_values(ascending=False)
        sort_neurons = medians.index.tolist()
    
    plot_kwargs = dict(
        data_frame=df_subset, 
        x='neuron', 
        y='PC1_loading',
        title=title,
        labels={'neuron': 'Neuron', 'PC1_loading': 'PC1 Loading'},
        category_orders={'neuron': sort_neurons}
    )
    
    if 'source' in df_subset.columns:
        plot_kwargs['color'] = 'source'
    
    fig = px.box(**plot_kwargs)
    fig.update_traces(marker=dict(size=4, opacity=0.5), boxpoints='all')
    fig.update_layout(
        xaxis=dict(tickangle=45),
        height=600,
        width=1600
    )
    
    fig.write_html(f'{filename}.html')
    fig.write_image(f'{filename}.png', scale=2)


def get_comparisons_to_run(plot_df, filtered_neurons):
    """Determine which source comparisons to run based on available data.
    
    Args:
        plot_df: DataFrame with 'source' column containing PCA results
        filtered_neurons: Dict with keys 'all', 'flavell_zimmer', 'flavell_immob', 'zimmer_immob'
                        mapping to lists of neurons for each comparison
    
    Returns:
        List of comparison tuples. Each tuple is either:
        - (src1, src2, neurons, filename) for 2-way comparisons
        - (src1, src2, src3, neurons, filename) for 3-way comparisons
    """
    sources = plot_df['source'].unique().tolist()
    comparisons = []
    
    if 'flavell' in sources and 'zimmer' in sources:
        comparisons.append(('flavell', 'zimmer', filtered_neurons['flavell_zimmer'], 'pc1_loadings_flavell_zimmer'))
    if 'flavell' in sources and 'immob' in sources:
        comparisons.append(('flavell', 'immob', filtered_neurons['flavell_immob'], 'pc1_loadings_flavell_immob'))
    if 'zimmer' in sources and 'immob' in sources:
        comparisons.append(('zimmer', 'immob', filtered_neurons['zimmer_immob'], 'pc1_loadings_zimmer_immob'))
    if 'flavell' in sources and 'zimmer' in sources and 'immob' in sources:
        comparisons.append(('flavell', 'zimmer', 'immob', filtered_neurons['all'], 'pc1_loadings_all_three'))
    
    return comparisons


def save_nan_log(nan_log, nan_log_file, DEBUG=False):
    """Save NaN log to a CSV file.
    
    Args:
        nan_log: List of dicts with 'dataset' and 'neurons_with_partial_nan' keys
        nan_log_file: Path to output file
        DEBUG: If True, print confirmation message
    """
    if nan_log and nan_log_file:
        with open(nan_log_file, 'w') as f:
            f.write("Dataset,Neurons with Partial NaN\n")
            for entry in nan_log:
                f.write(f"{entry['dataset']},{','.join(entry['neurons_with_partial_nan'])}\n")
        if DEBUG:
            print(f"Saved NaN log to {nan_log_file}")


def make_variance_explained_plot(variance_explained, df, zscore_neurons, DEBUG, gfp_h5_path=None):
    """Create variance explained plot from PCA results.
    
    Args:
        variance_explained: Dict of dataset_name -> array of explained variance ratios
                           (from pca_per_dataset)
        df: Original DataFrame (used for source information)
        zscore_neurons: Whether neurons were z-scored before PCA
        DEBUG: If True, print debug messages
        gfp_h5_path: Optional path to GFP control data H5 file. If None, uses default path.
                    Set to None to skip GFP control data entirely.
    
    Returns:
        DataFrame with variance explained data (for potential further analysis)
    """
    DEFAULT_GFP_PATH = '/lisc/data/scratch/neurobiology/zimmer/fieseler/paper/hierarchical_modeling_gfp/data.h5'
    
    var_exp_data = []
    for dataset_name, var_ratios in variance_explained.items():
        group = df[df['dataset_name'] == dataset_name]
        source = group['source'].iloc[0] if 'source' in group.columns else 'unknown'
        cumulative_var = np.cumsum(var_ratios)
        for mode_idx, var_exp in enumerate(cumulative_var):
            var_exp_data.append({
                'source': source,
                'dataset': dataset_name,
                'pca_mode': mode_idx + 1,
                'variance_explained': var_exp
            })
    
    gfp_path = gfp_h5_path if gfp_h5_path is not None else DEFAULT_GFP_PATH
    
    if gfp_path is not None:
        try:
            df_gfp = pd.read_hdf(gfp_path, key='df_with_missing')
            df_gfp = df_gfp.dropna(axis=1, how='all')
            
            gfp_neurons = get_neuron_columns(df_gfp)
            gfp_var_exp = compute_variance_explained(df_gfp, gfp_neurons, n_components=20, zscore_neurons=zscore_neurons)
            
            for dataset_name, var_ratios in gfp_var_exp.items():
                cumulative_var = np.cumsum(var_ratios)
                for mode_idx, var_exp in enumerate(cumulative_var):
                    var_exp_data.append({
                        'source': 'gfp_control',
                        'dataset': dataset_name,
                        'pca_mode': mode_idx + 1,
                        'variance_explained': var_exp
                    })
            if DEBUG:
                print(f"Added GFP control data to variance explained plot")
        except Exception as e:
            print(f"Could not load GFP control data: {e}")
    
    var_exp_df = pd.DataFrame(var_exp_data)
    
    if len(var_exp_df) > 0:
        zscore_text = ', z-scored' if zscore_neurons else ''
        
        fig_var = plotly_plot_mean_and_shading(
            var_exp_df,
            x='pca_mode',
            y='variance_explained',
            color='source',
            line_name='Mean',
            shade_style='std',
            add_individual_lines=False,
            cmap=plotly_paper_color_discrete_map()
        )
        fig_var.update_layout(
            xaxis_title='PCA Mode',
            yaxis_title='Cumulative fraction<br>variance explained',
        )
        apply_figure_settings(fig_var, width_factor=PANEL_WIDTH_FACTOR, height_factor=PANEL_HEIGHT_FACTOR)
        # Same top margin as the manifold panels, so the axes line up in the same row
        # Legend inside, bottom right (empty, since the curves rise); needed because gfp_control is not in any other legend
        fig_var.update_layout(margin=dict(t=PHASE_PLOT_TOP_MARGIN), showlegend=True,
                              legend=dict(title=None, yanchor='bottom', y=0.0, xanchor='right', x=1.0,
                                          bgcolor='rgba(0,0,0,0)'))
        fig_var.write_html('variance_explained_pca.html')
        fig_var.write_image('variance_explained_pca.png', scale=2)
        if DEBUG:
            print("Saved variance explained plot to variance_explained_pca.html and .png")
    
    return var_exp_df


def make_all_phase_plots(pc_scores, df, zscore_text, DEBUG=False, legend_dataset=EXAMPLE_PHASE_DATASETS[0]):
    """Create phase plots (PC0 vs PC1) for flavell datasets colored by reversal.

    Plots are sized as panels for the paper figure; only legend_dataset gets a legend.
    The example datasets for the figure are in EXAMPLE_PHASE_DATASETS.

    Args:
        pc_scores: Dict of dataset_name -> PCA scores array (n_timepoints, n_components)
        df: Original DataFrame (used for source and behavioral data)
        zscore_text: String to include in plot titles (e.g. ', z-scored' or '')
        DEBUG: If True, print debug messages
    
    Returns:
        int: Number of phase plots created
    """
    phase_data = []
    for dataset_name, scores in pc_scores.items():
        group = df[df['dataset_name'] == dataset_name]
        source = group['source'].iloc[0] if 'source' in group.columns else 'unknown'
        n_timepoints = scores.shape[0]
        phase_data.append(pd.DataFrame({
            'PC0': scores[:, 0],
            'PC1': scores[:, 1],
            'dataset': dataset_name,
            'source': source,
            'timepoint': np.arange(n_timepoints),
            'reversal': group['reversal'].values if 'reversal' in group.columns else np.nan,
            'post_reversal_turn': group['post_reversal_turn'].values if 'post_reversal_turn' in group.columns else False,
        }))

    phase_df = pd.concat(phase_data, ignore_index=True)

    flavell_datasets = phase_df[phase_df['source'] == 'flavell']['dataset'].unique()

    ethogram_cmap = BehaviorCodes.ethogram_cmap()
    beh_names = behavior_name_mapping()
    reversal_to_behavior_name = {'no reversal': beh_names['FWD'], 'reversal': beh_names['REV']}
    # Turn direction can't be recovered reliably from the Flavell data, so all post-reversal turns get the
    # ventral turn color (most of our post-reversal turns, ~70%, are ventral) but a direction-free name
    turn_name = 'Post-reversal turn'
    behavior_cmap = {beh_names['FWD']: ethogram_cmap[BehaviorCodes.FWD],
                     beh_names['REV']: ethogram_cmap[BehaviorCodes.REV],
                     turn_name: ethogram_cmap[BehaviorCodes.VENTRAL_TURN]}

    plot_count = 0
    for dataset_name in flavell_datasets:
        dataset_phase = phase_df[phase_df['dataset'] == dataset_name]
        
        safe_name = dataset_name.replace('/', '_').replace(' ', '_')
        
        if 'reversal' not in dataset_phase.columns:
            continue
        valid_mask = ~dataset_phase['reversal'].isna()
        if valid_mask.sum() == 0:
            continue
            
        # Same style as the local (zimmer) PCA manifold plots (see plot_traces.py):
        # continuous lines, split by behavior, with ethogram colors and paper behavior names
        plot_data = dataset_phase[valid_mask].reset_index(drop=True)
        plot_data['behavior'] = plot_data['reversal'].map(reversal_to_behavior_name)
        is_turn = plot_data['post_reversal_turn'].fillna(False).astype(bool)  # NaN for rows from other sources
        plot_data.loc[is_turn, 'behavior'] = turn_name
        df_gaps, col_names = modify_dataframe_to_allow_gaps_for_plotly(plot_data, ['PC0', 'PC1'], 'behavior')
        states = list(plot_data['behavior'].unique())
        show_legend = dataset_name == legend_dataset

        fig_phase = go.Figure()
        for name in behavior_cmap.keys():
            if name not in states:
                continue
            i = states.index(name)
            fig_phase.add_trace(go.Scatter(x=df_gaps[col_names[0][i]], y=df_gaps[col_names[1][i]], mode='lines',
                                           name=name, line=dict(color=behavior_cmap[name], width=2),
                                           showlegend=show_legend))
        fig_phase.update_xaxes(title='PC1')
        fig_phase.update_yaxes(title='PC2')

        apply_figure_settings(fig_phase, width_factor=PANEL_WIDTH_FACTOR, height_factor=PANEL_HEIGHT_FACTOR)
        # Legend above the axes; same top margin on all panels so the example axes line up
        fig_phase.update_layout(margin=dict(t=PHASE_PLOT_TOP_MARGIN),
                                legend=dict(orientation='h', yanchor='bottom', y=1.0, xanchor='left', x=0.0))

        fig_phase.write_html(f'phase_plot_{safe_name}_reversal.html')
        fig_phase.write_image(f'phase_plot_{safe_name}_reversal.png', scale=2)
        plot_count += 1
    
    if DEBUG:
        print(f"Saved {plot_count} phase plots with reversal coloring (HTML and PNG)")
    
    return plot_count


def make_full_page_figure(weights_png='pc1_loadings_all_three.png', variance_png='variance_explained_pca.png',
                          example_datasets=EXAMPLE_PHASE_DATASETS, output_png='pca_comparison_full_page.png'):
    """Tile the already-saved panel PNGs into one page.

    Row 1: PC1 weights; row 2: variance explained + example manifolds.
    Panel sizes come from apply_figure_settings, so the tiled page keeps real page proportions.
    """
    from PIL import Image

    phase_pngs = [f"phase_plot_{d.replace('/', '_').replace(' ', '_')}_reversal.png" for d in example_datasets]
    weights = Image.open(weights_png).convert('RGBA')
    panels = [Image.open(f).convert('RGBA') for f in [variance_png] + phase_pngs]

    width = max(weights.width, sum(p.width for p in panels))
    height = weights.height + max(p.height for p in panels)
    page = Image.new('RGBA', (width, height), 'white')
    page.alpha_composite(weights, (0, 0))
    x = 0
    for p in panels:
        page.alpha_composite(p, (x, weights.height))
        x += p.width
    page.convert('RGB').save(output_png)
    return output_png


def make_pairwise_comparisons(df, neuron_cols, comparisons, zscore_neurons=False, min_datasets=5, zscore_text='', DEBUG=False):
    """Run pairwise PCA and generate boxplots for source comparisons.
    
    Args:
        df: DataFrame with neuron data
        neuron_cols: List of all neuron column names
        comparisons: List of comparison tuples from get_comparisons_to_run()
        zscore_neurons: Whether to z-score neurons before PCA
        min_datasets: Minimum datasets for neuron inclusion
        zscore_text: String to include in plot titles
        DEBUG: If True, print debug messages
    """
    for comp in comparisons:
        if len(comp) == 4:
            src1, src2, neurons, filename = comp
            title = f'PC1 Loadings: {src1.capitalize()} vs {src2.capitalize()} (confident + >= {min_datasets} datasets in each, anchored to AVA{zscore_text})'
            df_subset = df[df['source'].isin([src1, src2])]
        else:
            src1, src2, src3, neurons, filename = comp
            title = f'PC1 Loadings: {src1.capitalize()} vs {src2.capitalize()} vs {src3.capitalize()} (confident + >= {min_datasets} datasets in each, anchored to AVA{zscore_text})'
            df_subset = df[df['source'].isin([src1, src2, src3])]
        
        pca_results_pair, _, pc_scores_pair, _ = pca_per_dataset(
            df_subset, neuron_cols, neurons, anchor_neuron='AVA', 
            n_components=20, zscore_neurons=zscore_neurons, nan_log=[]
        )
        
        data_pair = []
        for dataset_name, loadings in pca_results_pair.items():
            for neuron, loading in loadings.items():
                row = {
                    'dataset': dataset_name,
                    'neuron': neuron,
                    'PC1_loading': loading
                }
                group = df_subset[df_subset['dataset_name'] == dataset_name]
                if 'source' in group.columns:
                    row['source'] = group['source'].iloc[0]
                data_pair.append(row)
        
        df_pair = pd.DataFrame(data_pair)
        
        p_value_dict = {}
        if len(comp) == 4:
            try:
                df_src1 = df_pair[df_pair['source'] == src1].pivot(index='neuron', columns='dataset', values='PC1_loading')
                df_src2 = df_pair[df_pair['source'] == src2].pivot(index='neuron', columns='dataset', values='PC1_loading')
                if df_src1.shape[0] > 0 and df_src2.shape[0] > 0:
                    from scipy import stats
                    from statsmodels.stats.multitest import multipletests
                    
                    common_neurons = set(df_src1.index).intersection(set(df_src2.index))
                    p_values = {}
                    for neuron in common_neurons:
                        vals1 = df_src1.loc[neuron].dropna().values
                        vals2 = df_src2.loc[neuron].dropna().values
                        if len(vals1) >= 3 and len(vals2) >= 3:
                            _, p = stats.ttest_ind(vals1, vals2, equal_var=False)
                            p_values[neuron] = p
                    
                    if len(p_values) > 0:
                        neurons_list = list(p_values.keys())
                        p_arr = np.array([p_values[n] for n in neurons_list])
                        _, p_corrected, _, _ = multipletests(p_arr, method='fdr_bh')
                        p_value_dict = {n: p_corrected[i] for i, n in enumerate(neurons_list)}
                        print(f"Calculated p-values for {src1} vs {src2}: {len(p_value_dict)} neurons")
                        if DEBUG:
                            for n, p in p_value_dict.items():
                                print(f"  {n}: corrected p = {p:.4e}")
            except Exception as e:
                print(f"Warning: Could not calculate p-values for {src1} vs {src2}: {e}")
        
        if len(df_pair) == 0:
            continue
        sort_source = next((s for s in SORT_SOURCE_PRIORITY if s in df_pair['source'].values), None)
        df_sort = df_pair[df_pair['source'] == sort_source] if sort_source is not None else df_pair
        sort_neurons = df_sort.groupby('neuron')['PC1_loading'].median().sort_values(ascending=False).index.tolist()
        # Neurons missing from the sorting source go at the end
        sort_neurons += [n for n in df_pair['neuron'].unique() if n not in sort_neurons]
        
        plot_kwargs = dict(
            data_frame=df_pair, 
            x='neuron', 
            y='PC1_loading',
            # title=title,
            labels={'neuron': 'Neuron', 'PC1_loading': 'PC1 Weight'},
            category_orders={'neuron': sort_neurons}
        )
        
        if 'source' in df_pair.columns:
            plot_kwargs['color'] = 'source'
            plot_kwargs['color_discrete_map'] = plotly_paper_color_discrete_map()
        
        fig = px.box(**plot_kwargs)
        fig.update_traces(marker=dict(size=4, opacity=0.5), boxpoints='all')
        fig.update_layout(
            xaxis=dict(tickangle=45)
        )

        if p_value_dict:
            try:
                fig = add_p_value_annotation(fig, x_label='all', precalculated_p_values=p_value_dict, show_only_stars=True, show_ns=DEBUG, 
                                             category_orders={'neuron': sort_neurons}, height_mode='top_of_data', DEBUG=DEBUG)
            except Exception as e:
                raise e

        apply_figure_settings(fig, width_factor=WEIGHTS_WIDTH_FACTOR, height_factor=WEIGHTS_HEIGHT_FACTOR)
        fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='black')
        # Legend inside the plot (as in the main paper); the right side has mostly negative weights
        fig.update_layout(legend=dict(title=None, yanchor='top', y=1.0, xanchor='right', x=1.0, bgcolor='rgba(0,0,0,0)'))

        fig.write_html(f'{filename}.html')
        fig.write_image(f'{filename}.png', scale=2)
        if DEBUG:
            print(f"Saved {filename} plot")


def main(include_hierarchical=False, zscore_neurons=False, nan_log_file=None, min_datasets=5, include_immob=False, DEBUG=False, gfp_h5_path=None):
    nan_log = []
    
    df = prepare_dataframe(include_hierarchical=include_hierarchical, include_immob=include_immob)
    neuron_cols = get_neuron_columns(df)
    labeled_neurons = get_labeled_neurons(neuron_cols)
    
    print(f"Total neuron columns after combining: {len(neuron_cols)}")
    print(f"Labeled neurons after combining: {len(labeled_neurons)}")
    
    sources = df['source'].unique().tolist()
    
    pairwise_min_datasets = max(3, min_datasets // 2)
    
    filtered = filter_neurons_by_coverage(
        df, neuron_cols, labeled_neurons, 
        min_datasets=min_datasets, 
        pairwise_min_datasets=pairwise_min_datasets
    )
    plot_neurons_all = filtered['all']
    neurons_fz = filtered['flavell_zimmer']
    neurons_fi = filtered['flavell_immob']
    neurons_zi = filtered['zimmer_immob']
    
    print(f"Plot neurons (all {len(sources)} sources): {len(plot_neurons_all)}")
    print(f"Plot neurons (flavell vs zimmer): {len(neurons_fz)}")
    print(f"Plot neurons (flavell vs immob): {len(neurons_fi)}")
    print(f"Plot neurons (zimmer vs immob): {len(neurons_zi)}")
    print(f"AVA in all neurons: {'AVA' in plot_neurons_all}")
    
    # Print diagnostic: correlation with AVA
    if DEBUG:
        print_diagnostic_correlations(df, plot_neurons_all, anchor_neuron='AVA')
    
    # Run PCA for all sources combined (20 components for variance explained plot)
    pca_results, nan_log, pc_scores, variance_explained = pca_per_dataset(
        df, neuron_cols, plot_neurons_all, anchor_neuron='AVA', n_components=20, 
        zscore_neurons=zscore_neurons, nan_log=nan_log
    )
    
    # Build dataframe for all PCA results
    plot_df = build_pca_plot_dataframe(pca_results, df)
    sources = plot_df['source'].unique().tolist()
    zscore_text = ', z-scored' if zscore_neurons else ''
    
    # Combined plot (all sources)
    title_combined = f'PC1 Loadings for Filtered Neurons (confident + >= {min_datasets} datasets in each source, anchored to AVA{zscore_text}, sorted by median)'
    make_boxplot(plot_df, 'pc1_loadings_boxplot', title_combined)
    if DEBUG:
        print("Saved combined plot to pc1_loadings_boxplot.html and .png")
    
    # Cumulative variance explained per PCA mode
    make_variance_explained_plot(variance_explained, df, zscore_neurons, DEBUG, gfp_h5_path)
    
    # Pairwise and all-three plots: run separate PCAs for each comparison
    comparisons = get_comparisons_to_run(plot_df, filtered)
    make_pairwise_comparisons(
        df, neuron_cols, comparisons, 
        zscore_neurons=zscore_neurons, 
        min_datasets=min_datasets,
        zscore_text=zscore_text,
        DEBUG=DEBUG
    )
    
    # Phase plot: PC0 vs PC1 for flavell datasets only, colored by reversal
    make_all_phase_plots(pc_scores, df, zscore_text, DEBUG)

    # Full page: PC1 weights (all three sources) on top, variance explained + example manifolds below
    if any(len(c) == 5 for c in comparisons):
        make_full_page_figure()
    
    save_nan_log(nan_log, nan_log_file, DEBUG)
    
    return plot_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PCA per dataset and boxplot of PC1 loadings')
    parser.add_argument('--flavell-only', action='store_true',
                        help='Use only flavell data (exclude zimmer and immob)')
    parser.add_argument('--include-immob', action='store_true',
                        help='Include immob data (along with zimmer)')
    parser.add_argument('--zscore-neurons', action='store_true',
                        help='Z-score neurons before PCA')
    parser.add_argument('--nan-log-file', type=str, default=None,
                        help='File to save NaN log')
    parser.add_argument('--min-datasets', type=int, default=5,
                        help='Minimum number of datasets (per source) a neuron must appear in (default: 5)')
    parser.add_argument('--debug', action='store_true', 
                        help='Enable debug mode with extra prints')
    parser.add_argument('--gfp-h5-path', type=str, default=None,
                        help='Path to GFP control data H5 file (default: use built-in path)')
    args = parser.parse_args()
    
    include_hierarchical = not args.flavell_only
    main(include_hierarchical=include_hierarchical, zscore_neurons=args.zscore_neurons, 
         nan_log_file=args.nan_log_file, min_datasets=args.min_datasets, 
         include_immob=args.include_immob, DEBUG=args.debug, gfp_h5_path=args.gfp_h5_path)
