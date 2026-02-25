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
from wbfm.utils.visualization.utils_plot_traces import add_p_value_annotation

BEHAVIOR_COLUMNS = ['velocity', 'angular_velocity', 'head_curvature', 'body_curvature', 'pumping', 'reversal', 'speed']
METADATA_COLUMNS = ['dataset_name', 'local_time', 'source']
COLUMN_EXCLUDE_PATTERNS = ['_manifold', 'eigenworm', 'curvature', '_frequency', '_turn', '_collision', 
                           'CCA_neural_mode', 'local_time_physical', 'pca_', 'rev', 'pause']


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
        
        if anchor_neuron in available_labeled:
            anchor_loading = neuron_to_loading[anchor_neuron]
            if anchor_loading < 0:
                neuron_to_loading = {k: -v for k, v in neuron_to_loading.items()}
        
        results[dataset_name] = {n: neuron_to_loading[n] for n in labeled_neurons_to_return if n in neuron_to_loading}
        
        scores = pca.transform(X_std)
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


def main(include_hierarchical=False, zscore_neurons=False, nan_log_file=None, min_datasets=5, include_immob=False, DEBUG=False):
    nan_log = []
    df = load_neurons(include_hierarchical=include_hierarchical, include_immob=include_immob)
    
    print("\nCombining L/R neuron suffixes (per dataset)...")
    df = combine_suffixes_per_dataset(df)
    
    neuron_cols = get_neuron_columns(df)
    labeled_neurons = get_labeled_neurons(neuron_cols)
    
    print(f"Total neuron columns after combining: {len(neuron_cols)}")
    print(f"Labeled neurons after combining: {len(labeled_neurons)}")
    
    # Filter to datasets that have AVA (required anchor)
    datasets_with_ava = []
    for dataset_name, group in df.groupby('dataset_name'):
        if 'AVA' in group.columns and group['AVA'].notna().sum() > 0:
            datasets_with_ava.append(dataset_name)
    
    original_count = df['dataset_name'].nunique()
    df = df[df['dataset_name'].isin(datasets_with_ava)]
    print(f"Filtered to {len(datasets_with_ava)}/{original_count} datasets with AVA")
    
    confident_list = neurons_with_confident_ids(combine_left_right=True)
    confident_set = set(confident_list)
    print(f"Confident neuron base names: {len(confident_set)}")
    
    dataset_counts_by_source = count_neuron_datasets_by_source(df, neuron_cols)
    sources = df['source'].unique().tolist()
    
    def get_neurons_for_pair(sources_pair, counts_dict, confident_set, labeled_neurons, min_datasets):
        """Get neurons that have min_datasets in both sources of the pair."""
        neurons = []
        debug_info = {src: [] for src in sources_pair}
        all_neurons_debug = []
        for neuron in labeled_neurons:
            if neuron not in confident_set:
                continue
            counts = counts_dict.get(neuron, {})
            src_counts = {src: counts.get(src, 0) for src in sources_pair}
            all_neurons_debug.append((neuron, src_counts))
            
            passes = True
            for src in sources_pair:
                if counts.get(src, 0) < min_datasets:
                    passes = False
                    break
            if passes:
                neurons.append(neuron)
        
        filtered = [(n, c) for n, c in all_neurons_debug if n not in neurons]
        return neurons
    
    # Get neurons for each comparison
    all_sources = sources
    plot_neurons_all = get_neurons_for_pair(sources, dataset_counts_by_source, confident_set, labeled_neurons, min_datasets)
    
    # Use lower threshold for pairwise comparisons since datasets are limited per source
    pairwise_min_datasets = max(3, min_datasets // 2)
    
    # Only compute neurons for pairs that actually exist in the data
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
    
    # Print diagnostic: correlation with AVA
    if DEBUG:
        print_diagnostic_correlations(df, plot_neurons_all, anchor_neuron='AVA')
    
    # Run PCA for all sources combined (20 components for variance explained plot)
    pca_results, nan_log, pc_scores, variance_explained = pca_per_dataset(
        df, neuron_cols, plot_neurons_all, anchor_neuron='AVA', n_components=20, 
        zscore_neurons=zscore_neurons, nan_log=nan_log
    )
    
    # Build dataframe for all PCA results
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
    
    plot_df = pd.DataFrame(data_for_plot)
    sources = plot_df['source'].unique().tolist()
    zscore_text = ', z-scored' if zscore_neurons else ''
    
    def make_boxplot(df_subset, filename, title, sort_neurons=None):
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
    
    # Combined plot (all sources)
    title_combined = f'PC1 Loadings for Filtered Neurons (confident + >= {min_datasets} datasets in each source, anchored to AVA{zscore_text}, sorted by median)'
    make_boxplot(plot_df, 'pc1_loadings_boxplot', title_combined)
    if DEBUG:
        print("Saved combined plot to pc1_loadings_boxplot.html and .png")
    
    # Cumulative variance explained per PCA mode (use existing PCA results from pca_per_dataset)
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
    
    # Add GFP control data using the dedicated function (use same zscore flag as main analysis)
    gfp_h5_path = '/lisc/data/scratch/neurobiology/zimmer/fieseler/paper/hierarchical_modeling_gfp/data.h5'
    try:
        df_gfp = pd.read_hdf(gfp_h5_path, key='df_with_missing')
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
        fig_var = plotly_plot_mean_and_shading(
            var_exp_df,
            x='pca_mode',
            y='variance_explained',
            color='source',
            line_name='Mean',
            shade_style='std',
            add_individual_lines=False
        )
        fig_var.update_layout(
            title=f'Cumulative Variance Explained per PCA Mode by Source (mean ± std{zscore_text})',
            xaxis_title='PCA Mode',
            yaxis_title='Cumulative Variance Explained',
            height=600,
            width=800
        )
        fig_var.write_html('variance_explained_pca.html')
        fig_var.write_image('variance_explained_pca.png', scale=2)
        if DEBUG:
            print("Saved variance explained plot to variance_explained_pca.html and .png")
    
    # Pairwise and all-three plots: run separate PCAs for each comparison
    comparisons = []
    if 'flavell' in sources and 'zimmer' in sources:
        comparisons.append(('flavell', 'zimmer', neurons_fz, 'pc1_loadings_flavell_zimmer'))
    if 'flavell' in sources and 'immob' in sources:
        comparisons.append(('flavell', 'immob', neurons_fi, 'pc1_loadings_flavell_immob'))
    if 'zimmer' in sources and 'immob' in sources:
        comparisons.append(('zimmer', 'immob', neurons_zi, 'pc1_loadings_zimmer_immob'))
    if 'flavell' in sources and 'zimmer' in sources and 'immob' in sources:
        comparisons.append(('flavell', 'zimmer', 'immob', plot_neurons_all, 'pc1_loadings_all_three'))
    
    for comp in comparisons:
        if len(comp) == 4:
            src1, src2, neurons, filename = comp
            title = f'PC1 Loadings: {src1.capitalize()} vs {src2.capitalize()} (confident + >= {min_datasets} datasets in each, anchored to AVA{zscore_text})'
            df_subset = df[df['source'].isin([src1, src2])]
        else:
            src1, src2, src3, neurons, filename = comp
            title = f'PC1 Loadings: {src1.capitalize()} vs {src2.capitalize()} vs {src3.capitalize()} (confident + >= {min_datasets} datasets in each, anchored to AVA{zscore_text})'
            df_subset = df[df['source'].isin([src1, src2, src3])]
        
        # Run PCA for this comparison
        pca_results_pair, _, pc_scores_pair, _ = pca_per_dataset(df_subset, neuron_cols, neurons, anchor_neuron='AVA', n_components=20, zscore_neurons=zscore_neurons, nan_log=[])
        
        # Build dataframe
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
        
        # Calculate p-values for 2-way comparisons only
        p_value_dict = {}
        if len(comp) == 4:
            try:
                df_src1 = df_pair[df_pair['source'] == src1].pivot(index='neuron', columns='dataset', values='PC1_loading')
                df_src2 = df_pair[df_pair['source'] == src2].pivot(index='neuron', columns='dataset', values='PC1_loading')
                if df_src1.shape[0] > 0 and df_src2.shape[0] > 0:
                    # Compute p-values directly: t-test between the two sources for each neuron
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
                        # FDR correction
                        neurons = list(p_values.keys())
                        p_arr = np.array([p_values[n] for n in neurons])
                        _, p_corrected, _, _ = multipletests(p_arr, method='fdr_bh')
                        p_value_dict = {n: p_corrected[i] for i, n in enumerate(neurons)}
                        print(f"Calculated p-values for {src1} vs {src2}: {len(p_value_dict)} neurons")
                        if DEBUG:
                            for n, p in p_value_dict.items():
                                print(f"  {n}: corrected p = {p:.4e}")
            except Exception as e:
                print(f"Warning: Could not calculate p-values for {src1} vs {src2}: {e}")
        
        # Sort neurons by flavell median if available
        if 'flavell' in df_pair['source'].values:
            flavell_medians = df_pair[df_pair['source'] == 'flavell'].groupby('neuron')['PC1_loading'].median().sort_values(ascending=False)
            sort_neurons = flavell_medians.index.tolist()
        else:
            sort_neurons = None
        
        # Make boxplot
        if len(df_pair) == 0:
            return
        if sort_neurons is None:
            medians = df_pair.groupby('neuron')['PC1_loading'].median().sort_values(ascending=False)
            sort_neurons = medians.index.tolist()
        
        plot_kwargs = dict(
            data_frame=df_pair, 
            x='neuron', 
            y='PC1_loading',
            title=title,
            labels={'neuron': 'Neuron', 'PC1_loading': 'PC1 Loading'},
            category_orders={'neuron': sort_neurons}
        )
        
        if 'source' in df_pair.columns:
            plot_kwargs['color'] = 'source'
        
        fig = px.box(**plot_kwargs)
        fig.update_traces(marker=dict(size=4, opacity=0.5), boxpoints='all')
        fig.update_layout(
            xaxis=dict(tickangle=45),
            height=600,
            width=1600
        )
        
        if p_value_dict:
            try:
                fig = add_p_value_annotation(fig, x_label='all', precalculated_p_values=p_value_dict, show_only_stars=True, show_ns=False, 
                                             category_orders={'neuron': sort_neurons}, DEBUG=DEBUG)
            except Exception as e:
                # print(f"Warning: Could not add p-value annotations: {e}")
                raise e
        
        fig.write_html(f'{filename}.html')
        fig.write_image(f'{filename}.png', scale=2)
        if DEBUG:
            print(f"Saved {filename} plot")
    
    # Phase plot: PC0 vs PC1 for flavell datasets only, colored by reversal
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
    
    phase_df = pd.concat(phase_data, ignore_index=True)
    
    zscore_text = ', z-scored' if zscore_neurons else ''
    
    flavell_datasets = phase_df[phase_df['source'] == 'flavell']['dataset'].unique()
    
    for dataset_name in flavell_datasets:
        dataset_phase = phase_df[phase_df['dataset'] == dataset_name]
        
        safe_name = dataset_name.replace('/', '_').replace(' ', '_')
        
        if 'reversal' not in dataset_phase.columns:
            continue
        valid_mask = ~dataset_phase['reversal'].isna()
        if valid_mask.sum() == 0:
            continue
            
        fig_phase = px.scatter(
            dataset_phase[valid_mask],
            x='PC0',
            y='PC1',
            color='reversal',
            title=f'Phase Plot: {dataset_name} (reversal, anchored to AVA{zscore_text})',
            color_discrete_map={'no reversal': 'blue', 'reversal': 'red'}
        )
        fig_phase.update_traces(marker=dict(size=4, opacity=0.7))
        fig_phase.update_layout(
            height=600,
            width=800
        )
        fig_phase.write_html(f'phase_plot_{safe_name}_reversal.html')
        fig_phase.write_image(f'phase_plot_{safe_name}_reversal.png', scale=2)
    
    if DEBUG:
        print(f"Saved {len(flavell_datasets)} phase plots with reversal coloring (HTML and PNG)")
    
    if nan_log and nan_log_file:
        with open(nan_log_file, 'w') as f:
            f.write("Dataset,Neurons with Partial NaN\n")
            for entry in nan_log:
                f.write(f"{entry['dataset']},{','.join(entry['neurons_with_partial_nan'])}\n")
        print(f"Saved NaN log to {nan_log_file}")
    
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
    args = parser.parse_args()
    
    include_hierarchical = not args.flavell_only
    main(include_hierarchical=include_hierarchical, zscore_neurons=args.zscore_neurons, 
         nan_log_file=args.nan_log_file, min_datasets=args.min_datasets, 
         include_immob=args.include_immob, DEBUG=args.debug)
