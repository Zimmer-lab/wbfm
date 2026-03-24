#!/usr/bin/env python3
"""Tests for pca_boxplot.py refactoring.

This test suite uses cached fixtures to verify that extracted functions
produce the same results as the original implementation.
"""

import os
import pickle
import pytest
import numpy as np
import pandas as pd

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')


def save_fixture(data, name):
    """Save fixture to pickle file."""
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    path = os.path.join(FIXTURES_DIR, f'{name}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved fixture: {path}")


def load_fixture(name):
    """Load fixture from pickle file."""
    path = os.path.join(FIXTURES_DIR, f'{name}.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)


def fixture_exists(name):
    """Check if a specific fixture exists."""
    return os.path.exists(os.path.join(FIXTURES_DIR, f'{name}.pkl'))


def compare_dataframes(actual, expected, name='DataFrame', rtol=1e-4, atol=1e-6):
    """Compare two DataFrames and raise assertion error if different."""
    if not isinstance(actual, pd.DataFrame):
        raise AssertionError(f"{name}: expected DataFrame, got {type(actual)}")
    if not isinstance(expected, pd.DataFrame):
        raise AssertionError(f"{name}: expected DataFrame, got {type(expected)}")
    
    if set(actual.columns) != set(expected.columns):
        missing_in_actual = set(expected.columns) - set(actual.columns)
        missing_in_expected = set(actual.columns) - set(expected.columns)
        raise AssertionError(f"{name}: column mismatch. Missing in actual: {missing_in_actual}, Missing in expected: {missing_in_expected}")
    
    for col in actual.columns:
        actual_vals = actual[col].values
        expected_vals = expected[col].values
        
        if actual[col].dtype == object or actual[col].dtype == 'string':
            if not np.array_equal(actual_vals, expected_vals):
                raise AssertionError(f"{name}: Different values in column: {col}")
        elif not np.allclose(actual_vals, expected_vals, rtol=rtol, atol=atol, equal_nan=True):
            raise AssertionError(f"{name}: Different values in column: {col}")


def compare_dicts(actual, expected, name='dict', rtol=1e-5, atol=1e-8):
    """Compare two dicts with nested numpy arrays/DataFrames."""
    if set(actual.keys()) != set(expected.keys()):
        raise AssertionError(f"{name}: key mismatch. Actual: {set(actual.keys())}, Expected: {set(expected.keys())}")
    
    for key in actual.keys():
        a, e = actual[key], expected[key]
        
        if isinstance(a, np.ndarray) and isinstance(e, np.ndarray):
            if not np.allclose(a, e, rtol=rtol, atol=atol):
                raise AssertionError(f"{name}[{key}]: arrays not close")
        elif isinstance(a, pd.DataFrame) and isinstance(e, pd.DataFrame):
            if not a.equals(e):
                raise AssertionError(f"{name}[{key}]: DataFrames not equal")
        elif isinstance(a, dict) and isinstance(e, dict):
            compare_dicts(a, e, f"{name}[{key}]", rtol, atol)
        elif isinstance(a, list) and isinstance(e, list):
            if len(a) != len(e):
                raise AssertionError(f"{name}[{key}]: list length mismatch")
            for i, (ai, ei) in enumerate(zip(a, e)):
                if isinstance(ai, np.ndarray) and isinstance(ei, np.ndarray):
                    if not np.allclose(ai, ei, rtol=rtol, atol=atol):
                        raise AssertionError(f"{name}[{key}][{i}]: arrays not close")
                elif ai != ei:
                    raise AssertionError(f"{name}[{key}][{i}]: {ai} != {ei}")
        elif isinstance(a, (int, float)) and isinstance(e, (int, float)):
            if not np.isclose(a, e, rtol=rtol, atol=atol):
                raise AssertionError(f"{name}[{key}]: {a} != {e}")
        elif a != e:
            raise AssertionError(f"{name}[{key}]: {a} != {e}")


class TestHelperFunctions:
    """Tests for helper functions that are already well-separated."""
    
    def test_get_neuron_columns(self, sample_df):
        """Test get_neuron_columns returns expected columns."""
        from pca_boxplot import get_neuron_columns
        result = get_neuron_columns(sample_df)
        
        if fixture_exists('neuron_columns'):
            expected = load_fixture('neuron_columns')
            assert result == expected, "get_neuron_columns output changed"
        else:
            save_fixture(result, 'neuron_columns')
    
    def test_get_labeled_neurons(self, sample_df):
        """Test get_labeled_neurons returns expected neurons."""
        from pca_boxplot import get_labeled_neurons, get_neuron_columns
        neuron_cols = get_neuron_columns(sample_df)
        result = get_labeled_neurons(neuron_cols)
        
        if fixture_exists('labeled_neurons'):
            expected = load_fixture('labeled_neurons')
            assert result == expected, "get_labeled_neurons output changed"
        else:
            save_fixture(result, 'labeled_neurons')
    
    def test_count_neuron_datasets(self, sample_df):
        """Test count_neuron_datasets returns expected counts."""
        from pca_boxplot import count_neuron_datasets, get_neuron_columns
        neuron_cols = get_neuron_columns(sample_df)
        result = count_neuron_datasets(sample_df, neuron_cols)
        
        if fixture_exists('dataset_counts'):
            expected = load_fixture('dataset_counts')
            compare_dicts(result, expected, 'dataset_counts')
        else:
            save_fixture(result, 'dataset_counts')
    
    def test_count_neuron_datasets_by_source(self, sample_df):
        """Test count_neuron_datasets_by_source returns expected counts."""
        from pca_boxplot import count_neuron_datasets_by_source, get_neuron_columns
        neuron_cols = get_neuron_columns(sample_df)
        result = count_neuron_datasets_by_source(sample_df, neuron_cols)
        
        if fixture_exists('dataset_counts_by_source'):
            expected = load_fixture('dataset_counts_by_source')
            compare_dicts(result, expected, 'dataset_counts_by_source')
        else:
            save_fixture(result, 'dataset_counts_by_source')
    
    def test_get_neurons_in_group(self, sample_df):
        """Test get_neurons_in_group returns neurons with non-NaN data."""
        from pca_boxplot import get_neurons_in_group, get_neuron_columns
        neuron_cols = get_neuron_columns(sample_df)
        dataset_name = sample_df['dataset_name'].iloc[0]
        group = sample_df[sample_df['dataset_name'] == dataset_name]
        result = get_neurons_in_group(group, neuron_cols)
        
        if fixture_exists('neurons_in_group'):
            expected = load_fixture('neurons_in_group')
            assert result == expected, "get_neurons_in_group output changed"
        else:
            save_fixture(result, 'neurons_in_group')


class TestPCAFunctions:
    """Tests for PCA-related functions."""
    
    def test_pca_per_dataset(self, sample_df):
        """Test pca_per_dataset returns expected results."""
        from pca_boxplot import (
            pca_per_dataset, get_neuron_columns, get_labeled_neurons
        )
        neuron_cols = get_neuron_columns(sample_df)
        labeled_neurons = get_labeled_neurons(neuron_cols)
        
        test_neurons = [n for n in labeled_neurons if n in neuron_cols][:5]
        
        result, nan_log, pc_scores, var_exp = pca_per_dataset(
            sample_df, neuron_cols, test_neurons, 
            anchor_neuron='AVA', n_components=3
        )
        
        if fixture_exists('pca_results'):
            expected = load_fixture('pca_results')
            compare_dicts(result, expected['results'], 'pca_results')
        else:
            save_fixture({
                'results': result,
                'nan_log': nan_log,
                'pc_scores': pc_scores,
                'var_exp': var_exp
            }, 'pca_results')
    
    def test_compute_variance_explained(self, sample_df):
        """Test compute_variance_explained returns expected results."""
        from pca_boxplot import compute_variance_explained, get_neuron_columns
        neuron_cols = get_neuron_columns(sample_df)
        
        result = compute_variance_explained(sample_df, neuron_cols, n_components=5)
        
        if fixture_exists('variance_explained'):
            expected = load_fixture('variance_explained')
            compare_dicts(result, expected, 'variance_explained')
        else:
            save_fixture(result, 'variance_explained')
    
    def test_sort_neurons_by_median(self):
        """Test sort_neurons_by_median sorts correctly."""
        from pca_boxplot import sort_neurons_by_median
        
        df = pd.DataFrame({
            'neuron': ['B', 'A', 'C', 'A', 'B', 'C'],
            'PC1_loading': [1.0, 2.0, 3.0, 1.5, 0.5, 2.5]
        })
        
        result = sort_neurons_by_median(df.copy())
        
        assert list(result['neuron'].cat.categories) == ['C', 'A', 'B']
    
    def test_combine_suffixes_per_dataset(self, sample_df):
        """Test combine_suffixes_per_dataset works correctly."""
        from pca_boxplot import combine_suffixes_per_dataset
        result = combine_suffixes_per_dataset(sample_df.copy())
        
        if fixture_exists('combined_suffixes'):
            expected = load_fixture('combined_suffixes')
            compare_dataframes(result, expected, 'combined_suffixes')
        else:
            save_fixture(result, 'combined_suffixes')
    
    def test_compute_neuron_ava_correlation(self, sample_df):
        """Test compute_neuron_ava_correlation returns expected results."""
        from pca_boxplot import compute_neuron_ava_correlation, get_neuron_columns
        neuron_cols = get_neuron_columns(sample_df)
        neurons = [n for n in neuron_cols if n != 'AVA'][:5]
        
        result = compute_neuron_ava_correlation(sample_df, neurons)
        
        if fixture_exists('ava_correlations'):
            expected = load_fixture('ava_correlations')
            compare_dataframes(result, expected, 'ava_correlations')
        else:
            save_fixture(result, 'ava_correlations')


class TestExtractedFunctions:
    """Tests for functions extracted from main()."""
    
    def test_prepare_dataframe(self):
        """Test prepare_dataframe extracts and prepares data correctly."""
        from pca_boxplot import prepare_dataframe
        
        result = prepare_dataframe(include_hierarchical=False, include_immob=False)
        
        if fixture_exists('prepared_df'):
            expected = load_fixture('prepared_df')
            compare_dataframes(result, expected, 'prepared_df')
        else:
            save_fixture(result, 'prepared_df')
    
    def test_filter_neurons_by_coverage(self, prepared_df):
        """Test filter_neurons_by_coverage returns expected neurons."""
        from pca_boxplot import (
            filter_neurons_by_coverage, get_neuron_columns, get_labeled_neurons
        )
        
        neuron_cols = get_neuron_columns(prepared_df)
        labeled_neurons = get_labeled_neurons(neuron_cols)
        
        result = filter_neurons_by_coverage(
            prepared_df, neuron_cols, labeled_neurons, 
            min_datasets=5, pairwise_min_datasets=3
        )
        
        if fixture_exists('filtered_neurons'):
            expected = load_fixture('filtered_neurons')
            assert set(result['all']) == set(expected['all']), "filtered neurons 'all' mismatch"
        else:
            save_fixture(result, 'filtered_neurons')
    
    def test_build_pca_plot_dataframe(self, prepared_df):
        """Test build_pca_plot_dataframe creates correct DataFrame."""
        from pca_boxplot import (
            build_pca_plot_dataframe, pca_per_dataset, 
            get_neuron_columns, get_labeled_neurons
        )
        
        neuron_cols = get_neuron_columns(prepared_df)
        labeled_neurons = get_labeled_neurons(neuron_cols)
        
        pca_results, _, _, _ = pca_per_dataset(
            prepared_df, neuron_cols, labeled_neurons,
            anchor_neuron='AVA', n_components=1
        )
        
        result = build_pca_plot_dataframe(pca_results, prepared_df)
        
        if fixture_exists('pca_plot_df'):
            expected = load_fixture('pca_plot_df')
            compare_dataframes(result, expected, 'pca_plot_df')
        else:
            save_fixture(result, 'pca_plot_df')
    
    def test_compute_pvalues_for_comparison(self, prepared_df):
        """Test compute_pvalues_for_comparison calculates p-values correctly."""
        from pca_boxplot import (
            compute_pvalues_for_comparison, pca_per_dataset,
            get_neuron_columns, get_labeled_neurons
        )
        
        neuron_cols = get_neuron_columns(prepared_df)
        labeled_neurons = get_labeled_neurons(neuron_cols)
        
        sources = prepared_df['source'].unique()
        if 'flavell' in sources and 'zimmer' in sources:
            df_subset = prepared_df[prepared_df['source'].isin(['flavell', 'zimmer'])]
            neurons = [n for n in labeled_neurons if n in neuron_cols][:5]
            
            pca_results, _, _, _ = pca_per_dataset(
                df_subset, neuron_cols, neurons,
                anchor_neuron='AVA', n_components=1
            )
            
            from pca_boxplot import build_pca_plot_dataframe
            plot_df = build_pca_plot_dataframe(pca_results, df_subset)
            
            result = compute_pvalues_for_comparison(plot_df, 'flavell', 'zimmer')
            
            if fixture_exists('pvalues_fz'):
                expected = load_fixture('pvalues_fz')
                compare_dicts(result, expected, 'pvalues_fz')
            else:
                save_fixture(result, 'pvalues_fz')
    
    def test_make_phase_plot_data(self, prepared_df):
        """Test make_phase_plot_data creates correct DataFrame."""
        from pca_boxplot import (
            make_phase_plot_data, pca_per_dataset,
            get_neuron_columns, get_labeled_neurons
        )
        
        neuron_cols = get_neuron_columns(prepared_df)
        labeled_neurons = get_labeled_neurons(neuron_cols)
        
        _, _, pc_scores, _ = pca_per_dataset(
            prepared_df, neuron_cols, labeled_neurons,
            anchor_neuron='AVA', n_components=2
        )
        
        result = make_phase_plot_data(pc_scores, prepared_df)
        
        if fixture_exists('phase_plot_data'):
            expected = load_fixture('phase_plot_data')
            compare_dataframes(result, expected, 'phase_plot_data')
        else:
            save_fixture(result, 'phase_plot_data')


class TestPlottingFunctions:
    """Tests for plotting functions (just need to run without error)."""
    
    def test_make_boxplot_runs(self, prepared_df):
        """Test make_boxplot executes without error."""
        from pca_boxplot import make_boxplot, pca_per_dataset, get_neuron_columns, get_labeled_neurons
        
        neuron_cols = get_neuron_columns(prepared_df)
        labeled_neurons = get_labeled_neurons(neuron_cols)
        
        pca_results, _, _, _ = pca_per_dataset(
            prepared_df, neuron_cols, labeled_neurons[:5],
            anchor_neuron='AVA', n_components=1
        )
        
        from pca_boxplot import build_pca_plot_dataframe
        plot_df = build_pca_plot_dataframe(pca_results, prepared_df)
        
        make_boxplot(plot_df, 'test_boxplot', 'Test Boxplot')
    
    def test_make_variance_explained_plot_runs(self, prepared_df):
        """Test make_variance_explained_plot executes without error."""
        from pca_boxplot import make_variance_explained_plot, pca_per_dataset, get_neuron_columns, get_labeled_neurons
        
        neuron_cols = get_neuron_columns(prepared_df)
        labeled_neurons = get_labeled_neurons(neuron_cols)
        
        _, _, _, variance_explained = pca_per_dataset(
            prepared_df, neuron_cols, labeled_neurons[:5],
            anchor_neuron='AVA', n_components=3
        )
        
        make_variance_explained_plot(variance_explained, prepared_df, zscore_neurons=False, DEBUG=False, gfp_h5_path=None)
    
    def test_get_comparisons_to_run(self, prepared_df):
        """Test get_comparisons_to_run returns expected comparisons."""
        from pca_boxplot import get_comparisons_to_run, filter_neurons_by_coverage, get_neuron_columns, get_labeled_neurons, build_pca_plot_dataframe, pca_per_dataset
        
        neuron_cols = get_neuron_columns(prepared_df)
        labeled_neurons = get_labeled_neurons(neuron_cols)
        
        filtered = filter_neurons_by_coverage(
            prepared_df, neuron_cols, labeled_neurons,
            min_datasets=5, pairwise_min_datasets=3
        )
        
        pca_results, _, _, _ = pca_per_dataset(
            prepared_df, neuron_cols, labeled_neurons[:5],
            anchor_neuron='AVA', n_components=1
        )
        
        plot_df = build_pca_plot_dataframe(pca_results, prepared_df)
        
        result = get_comparisons_to_run(plot_df, filtered)
        
        if fixture_exists('comparisons'):
            expected = load_fixture('comparisons')
            assert len(result) == len(expected), "Number of comparisons changed"
        else:
            save_fixture(result, 'comparisons')
    
    def test_make_pairwise_comparisons_runs(self, prepared_df):
        """Test make_pairwise_comparisons executes without error."""
        from pca_boxplot import make_pairwise_comparisons, get_comparisons_to_run, filter_neurons_by_coverage, get_neuron_columns, get_labeled_neurons, build_pca_plot_dataframe, pca_per_dataset
        
        neuron_cols = get_neuron_columns(prepared_df)
        labeled_neurons = get_labeled_neurons(neuron_cols)
        
        filtered = filter_neurons_by_coverage(
            prepared_df, neuron_cols, labeled_neurons,
            min_datasets=5, pairwise_min_datasets=3
        )
        
        pca_results, _, _, _ = pca_per_dataset(
            prepared_df, neuron_cols, labeled_neurons[:5],
            anchor_neuron='AVA', n_components=1
        )
        
        plot_df = build_pca_plot_dataframe(pca_results, prepared_df)
        
        comparisons = get_comparisons_to_run(plot_df, filtered)
        
        if comparisons:
            make_pairwise_comparisons(
                prepared_df, neuron_cols, comparisons,
                zscore_neurons=False, min_datasets=5, zscore_text='', DEBUG=False
            )
    
    def test_make_all_phase_plots_runs(self, prepared_df):
        """Test make_all_phase_plots executes without error."""
        from pca_boxplot import make_all_phase_plots, pca_per_dataset, get_neuron_columns, get_labeled_neurons
        
        neuron_cols = get_neuron_columns(prepared_df)
        labeled_neurons = get_labeled_neurons(neuron_cols)
        
        _, _, pc_scores, _ = pca_per_dataset(
            prepared_df, neuron_cols, labeled_neurons[:5],
            anchor_neuron='AVA', n_components=2
        )
        
        make_all_phase_plots(pc_scores, prepared_df, zscore_text='', DEBUG=False)
    
    def test_save_nan_log_runs(self):
        """Test save_nan_log executes without error."""
        from pca_boxplot import save_nan_log
        import tempfile
        import os
        
        nan_log = [
            {'dataset': 'test1', 'neurons_with_partial_nan': ['A', 'B']},
            {'dataset': 'test2', 'neurons_with_partial_nan': ['C']}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_file = f.name
        
        try:
            save_nan_log(nan_log, temp_file, DEBUG=False)
            assert os.path.exists(temp_file), "NaN log file was not created"
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)


class TestEndToEnd:
    """End-to-end tests of the full pipeline."""
    
    def test_main_produces_same_output(self):
        """Test main() produces same output as baseline."""
        if not fixture_exists('main_output'):
            import pca_boxplot
            result = pca_boxplot.main(
                include_hierarchical=False,
                zscore_neurons=False,
                min_datasets=5,
                include_immob=False,
                DEBUG=False
            )
            save_fixture(result, 'main_output')
        else:
            import pca_boxplot
            result = pca_boxplot.main(
                include_hierarchical=False,
                zscore_neurons=False,
                min_datasets=5,
                include_immob=False,
                DEBUG=False
            )
            expected = load_fixture('main_output')
            compare_dataframes(result, expected, 'main_output')


@pytest.fixture(scope='session')
def sample_df():
    """Load a small sample of real data for testing."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from load_neurons import load_neurons
    
    df = load_neurons(include_hierarchical=False, include_immob=False)
    datasets = df['dataset_name'].unique()[:2]
    return df[df['dataset_name'].isin(datasets)].copy()


@pytest.fixture(scope='session')
def prepared_df():
    """Prepare dataframe like main() does."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from load_neurons import load_neurons
    from pca_boxplot import combine_suffixes_per_dataset
    
    df = load_neurons(include_hierarchical=False, include_immob=False)
    df = combine_suffixes_per_dataset(df)
    
    datasets_with_ava = []
    for dataset_name, group in df.groupby('dataset_name'):
        if 'AVA' in group.columns and group['AVA'].notna().sum() > 0:
            datasets_with_ava.append(dataset_name)
    
    return df[df['dataset_name'].isin(datasets_with_ava)].copy()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
