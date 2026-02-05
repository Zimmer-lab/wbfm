import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from functools import reduce
from pathlib import Path
from typing import Dict, List, Union, Tuple
import plotly.express as px
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.linalg import LinAlgError
from scipy.signal import detrend
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.metrics import explained_variance_score
from tqdm.auto import tqdm

from wbfm.utils.external.utils_pandas import split_flattened_index, flatten_multiindex_columns
from wbfm.utils.general.postprocessing.utils_imputation import impute_missing_values_in_dataframe
from wbfm.utils.general.postures.centerline_classes import WormFullVideoPosture
from wbfm.utils.external.utils_matplotlib import corrfunc, paired_boxplot_from_dataframes
from wbfm.utils.general.utils_paper import apply_figure_settings
from wbfm.utils.projects.finished_project_data import load_all_projects_from_list, ProjectData
from wbfm.utils.traces.triggered_averages import FullDatasetTriggeredAverages, \
    clustered_triggered_averages_from_dict_of_projects
from wbfm.utils.general.high_performance_pandas import get_names_from_df
from wbfm.utils.visualization.behavior_comparison_plots import NeuronToMultivariateEncoding
from wbfm.utils.visualization.filtering_traces import remove_outliers_using_std
from wbfm.utils.general.utils_hardcoded import list_of_gas_sensing_neurons, list_neurons_manifold_in_immob
from wbfm.utils.visualization.utils_cca import CCAPlotter


@dataclass
class MultiProjectWrapper:
    """
    A wrapper for multiple projects, allowing for easy access to the same function across all projects

    This dispatches function calls to the project directly
    """
    all_project_paths: list = None
    all_projects: Dict = None

    def __post_init__(self):
        if self.all_projects is None:
            assert self.all_project_paths is not None, "Must pass either projects or paths"
            self.all_projects = load_all_projects_from_list(self.all_project_paths)

    def __getattr__(self, item):
        # Transform all unknown function calls into a loop of calls to the projects
        def method(*args, **kwargs):
            print(f"Dynamically dispatching method: {item}")
            output = {}
            for name, p in tqdm(self.all_projects.items()):
                # if not p.is_valid:
                #     logging.warning(f"Skipping invalid project {name}")
                #     continue
                out = getattr(p, item)(*args, **kwargs)
                output[name] = out
            return output

        return method

    @staticmethod
    def concat_multiple_datasets_long(dict_of_dfs, long_format=True):
        # Works for get_data_for_paired_boxplot
        if long_format:
            df = pd.concat(dict_of_dfs, axis=1)  # Creates a multiindex dataframe
        else:
            df = pd.concat(dict_of_dfs, axis=0).T
        df = df.T.reset_index().drop(columns='level_1')
        if 'dataset_name' in df:
            df.drop('level_0', inplace=True)
        else:
            df = df.rename(columns={'level_0': 'dataset_name'})
        return df.T

    def __repr__(self):
        return f"Multiproject analyzer with {len(self._all_behavior_plotters)} projects"

    def set_for_all_classes(self, updates: dict):
        for key, val in updates.items():
            for p in self.all_projects.values():
                p.__setattr__(key, val)


@dataclass
class MultiProjectWrapperWithBehavior(MultiProjectWrapper):
    """
    A wrapper for multiple projects, allowing for easy access to the same function across all projects

    This does not dispatch function calls to the project directly, but to the behavior plotters
    Most usage of this class is the subclasses: MultiProjectTriggeredAverages and MultiProjectBehaviorPlotter
    """

    class_constructor: callable = None
    constructor_kwargs: dict = field(default_factory=dict)
    use_threading: bool = True

    _all_behavior_plotters: List = None

    def __post_init__(self):
        super().__post_init__()
        # Initialize the behavior plotters
        self._all_behavior_plotters = [self.class_constructor(p, **self.constructor_kwargs) for
                                       p in self.all_projects.values()]

    def __getattr__(self, item):
        # Transform all unknown function calls into a loop of calls to the subobjects
        def method(*args, **kwargs):
            print(f"Dynamically dispatching method: {item}")
            if item == '_all_behavior_plotters':
                return self._all_behavior_plotters
            output = {}
            for p in tqdm(self._all_behavior_plotters):
                if not p.is_valid:
                    logging.warning(f"Skipping invalid project {p.shortened_name}")
                    continue
                out = getattr(p, item)(*args, **kwargs)
                output[p.shortened_name] = out
            return output

        return method

    def set_for_all_classes(self, updates: dict):
        for key, val in updates.items():
            for b in self._all_behavior_plotters:
                b.__setattr__(key, val)


class MultiProjectTriggeredAveragesWithBehavior(MultiProjectWrapperWithBehavior):
    class_constructor: callable = FullDatasetTriggeredAverages


class MultiProjectBehaviorPlotterWithBehavior(MultiProjectWrapperWithBehavior):
    class_constructor: callable = NeuronToMultivariateEncoding

    def pairplot_multi_dataset(self, which_channel='red', include_corr=True,
                               to_save=False):
        """
        Plots a seaborn pairplot for multiple datasets

        Parameters
        ----------
        which_channel
        include_corr
        to_save

        Returns
        -------

        """
        import seaborn as sns

        all_dfs = self.calc_per_neuron_df(which_channel)

        df = pd.concat(all_dfs, axis=0)
        df = df.reset_index().rename(columns={'index': 'neuron_name'})
        g = sns.pairplot(df, hue='dataset_name')
        if include_corr:
            g.map_lower(corrfunc)

        return g

        # if to_save:
        #     fname = '/home/charles/Current_work/presentations/nov_2022'
        #     fname = os.path.join(fname, 'gcamp6f_red_summary.png')
        #     plt.savefig(fname)

    def paired_boxplot_per_neuron_multi_dataset(self, df_start_name='red', df_final_name='ratio'):
        """
        Designed for use with subclass: BehavioralEncoding
            Uses per-neuron dataframes from each dataset

        Parameters
        ----------
        df_start_name
        df_final_name

        Returns
        -------

        """
        all_dfs = self.get_data_for_paired_boxplot(df_final_name, df_start_name)
        df = self.concat_multiple_datasets_long(all_dfs)

        paired_boxplot_from_dataframes(df.iloc[1:, :], [df_start_name, df_final_name])
        plt.title("Maximum correlation to kymograph")
        plt.ylim(0, 0.8)

    def paired_boxplot_overall_multi_dataset(self, df_name='ratio', **kwargs):
        """
        Designed for use with subclass: SpeedEncoding
            Uses full-dataset dataframes from each dataset (one number per dataset)

        Parameters
        ----------
        df_name

        Returns
        -------

        """
        dict_of_dfs = self.calc_dataset_summary_df(df_name, **kwargs)
        df = pd.concat(dict_of_dfs, axis=0).reset_index(drop=True).T

        paired_boxplot_from_dataframes(df)
        if kwargs.get('y_train', None) is not None:
            plt.title(f"Decoding of {kwargs['y_train']}")
        else:
            plt.title(f"Decoding of Speed")
        plt.ylim(-1.0, 1.0)


def build_behavior_time_series_from_multiple_projects(all_projects: Dict[str, ProjectData],
                                                      behavior_names: Union[str, List[str]],
                                                      z_score_beh=False) -> pd.DataFrame:
    """
    Builds a time series of behavior from multiple projects

    See calc_behavior_from_alias for valid values of behavior_name

    Parameters
    ----------
    all_projects: dict - a dictionary of project names to ProjectData objects
    behavior_names: str or list of str - a behavior (or list) of behavior aliases. See calc_behavior_from_alias
    z_score_beh: bool (default False) - whether to z-score the behavior (per dataset)

    Returns
    -------

    """

    if isinstance(behavior_names, str):
        behavior_names = [behavior_names]

    list_of_beh_dfs = []
    for b in behavior_names:
        output_dict = defaultdict(list)
        for dataset_name, p in all_projects.items():
            worm = p.worm_posture_class
            trace = worm.calc_behavior_from_alias(b)
            output_dict[b].extend(trace)
            output_dict['dataset_name'].extend([dataset_name] * len(trace))
            local_time = p.x_for_plots
            if len(local_time) < len(trace):
                raise ValueError(f"Length of local time ({len(local_time)}) is less than trace ({len(trace)})")
            output_dict['local_time'].extend(local_time[:len(trace)])
        # Make sure the final dataframe is sorted correctly
        this_df_beh = pd.DataFrame(output_dict)
        this_df_beh = this_df_beh.sort_values(['dataset_name', 'local_time']).reset_index(drop=True)
        if z_score_beh:
            dataset_names_column = this_df_beh['dataset_name']
            this_df_beh = this_df_beh.groupby('dataset_name', group_keys=False).apply(
                lambda x: (x - x.mean(numeric_only=True)) / x.std(numeric_only=True))
            this_df_beh['dataset_name'] = dataset_names_column
        list_of_beh_dfs.append(this_df_beh)
    # Combine all the dataframes, keeping only a single column of dataset names
    df_beh = pd.concat(list_of_beh_dfs, axis=1)
    df_beh = df_beh.loc[:, ~df_beh.columns.duplicated()]
    return df_beh


def build_trace_time_series_from_multiple_projects(all_projects: Dict[str, ProjectData], **kwargs) -> pd.DataFrame:
    """
    Builds a time series of traces from multiple projects

    This will generate a tall dataframe, where time is repeated for each dataset and stored in the column 'local_time_index'
    Dataset names are stored in the column 'dataset_name'
    Designed to be split using 'row_facet_column=dataset_name' in plotly

    In principle the columns will be a single neuron, but for default named neurons this is not meaningful

    Parameters
    ----------
    all_projects: dict - a dictionary of project names to ProjectData objects

    Returns
    -------

    """
    kwargs['rename_neurons_using_manual_ids'] = kwargs.get('rename_neurons_using_manual_ids', True)

    all_dfs = {}
    for dataset_name, p in all_projects.items():
        df = p.calc_default_traces(**kwargs)
        all_dfs[dataset_name] = df
    df_traces = pd.concat(all_dfs)
    df_traces = df_traces.reset_index(names=['dataset_name', 'local_time'])
    return df_traces


def build_curvature_time_series_from_multiple_projects(all_projects: Dict[str, ProjectData], **kwargs) -> pd.DataFrame:
    """
    Like build_behavior_time_series_from_multiple_projects, but for the curvature
        That this is a full dataframe per dataset, so it doesn't fit the logic of the above function

    Parameters
    ----------
    all_projects
    kwargs

    Returns
    -------

    """

    all_dfs = {}
    for dataset_name, p in all_projects.items():
        df = p.worm_posture_class.curvature(fluorescence_fps=True, reset_index=True)
        all_dfs[dataset_name] = df
    df_curvature = pd.concat(all_dfs)
    df_curvature = df_curvature.reset_index(names=['dataset_name', 'local_time'])
    return df_curvature


def build_cross_dataset_eigenworms(all_projects: Dict[str, ProjectData], i_eigenworm_start=10, i_eigenworm_end=-10,
                                   n_components=5, **kwargs) -> pd.DataFrame:
    """
    Uses build_curvature_time_series_from_multiple_projects, and then recalculates the eigenworms

    Note: different from using ['eigenwormX'] in build_behavior_time_series_from_multiple_projects, because that is
    calculated per-dataset

    Parameters
    ----------
    all_projects
    kwargs

    Returns
    -------

    """

    df_curvature = build_curvature_time_series_from_multiple_projects(all_projects, **kwargs)
    # Remove nan values and other columns
    df_curvature_nonan = df_curvature.replace(np.nan, 0.0).drop(columns=['dataset_name', 'local_time'])
    # Calculate the eigenworms
    df_eigenworms = WormFullVideoPosture.calculate_eigenworms_from_curvature(df_curvature_nonan, n_components,
                                                                             i_eigenworm_start, i_eigenworm_end)
    df_eigenworms = pd.DataFrame(df_eigenworms, columns=[f'eigenworm{i}' for i in range(n_components)])
    df_eigenworms['local_time'] = df_curvature['local_time']
    df_eigenworms['dataset_name'] = df_curvature['dataset_name']
    # Z-score per dataset (currently z-scored for all datasets together)
    df_eigenworms = df_eigenworms.groupby('dataset_name').apply(lambda x: (x - x.mean()) / x.std())
    return df_eigenworms


def build_pca_time_series_from_multiple_projects(all_projects: Dict[str, ProjectData], n_components=2,
                                                 **kwargs) -> pd.DataFrame:
    """
    Builds a time series of the global manifold, i.e. the top 2 PCA modes, from multiple projects
    Note: keeps the pca modes separate

    Similar to build_trace_time_series_from_multiple_projects, but for the calc_pca_modes method

    Note that if you want the pca reconstruction of the traces, you should use build_trace_time_series_from_multiple_projects
    with the residual_mode='pca' option

    Parameters
    ----------
    all_projects: dict - a dictionary of project names to ProjectData objects

    Returns
    -------

    """

    all_dfs = {}
    for dataset_name, p in all_projects.items():
        df, _ = p.calc_pca_modes(n_components=n_components, **kwargs)
        all_dfs[dataset_name] = df
    df_traces = pd.concat(all_dfs)
    df_traces = df_traces.reset_index(names=['dataset_name', 'local_time'])
    return df_traces


def build_cca_time_series_from_multiple_projects(all_projects: Dict[str, ProjectData], n_components=2,
                                                 **trace_kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Like build_pca_time_series_from_multiple_projects, but for CCA modes

    Uses the CCAPlotter class (same as the paper)
    """
    # Options in the paper (continuous behaviors)
    beh_kwargs = dict(additional_behaviors=[f"eigenworm{i}" for i in range(4)])
    cca_opt = dict(truncate_traces_to_n_components=3, preprocess_behavior_using_pca=False, trace_kwargs=trace_kwargs,
                   beh_kwargs=beh_kwargs)

    # Calculate for all projects
    all_dfs_neural = {}
    all_dfs_beh = {}
    for dataset_name, p in all_projects.items():
        p.use_physical_x_axis = True
        cca_obj = CCAPlotter(p, **cca_opt)
        df_neural, df_beh, _ = cca_obj.calc_cca(n_components=n_components, return_dataframes=True)
        all_dfs_neural[dataset_name] = df_neural
        all_dfs_beh[dataset_name] = df_beh
    df_traces_neural = pd.concat(all_dfs_neural)
    df_traces_beh = pd.concat(all_dfs_beh)
    df_traces_neural = df_traces_neural.reset_index(names=['dataset_name', 'local_time'])
    df_traces_beh = df_traces_beh.reset_index(names=['dataset_name', 'local_time'])
    return df_traces_neural, df_traces_beh


def build_dataframe_of_variance_explained(all_projects: Dict[str, ProjectData], n_components=2,
                                          melt=True, **trace_kwargs) -> pd.DataFrame:
    """
    Builds a dataframe of the variance explained by the number of PCA components given per neuron per dataset

    Parameters
    ----------
    all_projects
    n_components
    trace_kwargs

    Returns
    -------

    """
    trace_kwargs['rename_neurons_using_manual_ids'] = trace_kwargs.get('rename_neurons_using_manual_ids', True)

    # Define the residual mode using the given number of components
    if n_components == 2:
        residual_mode = 'pca_global'
    else:
        residual_mode = f'pca_global_{n_components}'
    df_traces_global = build_trace_time_series_from_multiple_projects(all_projects, residual_mode=residual_mode,
                                                                      **trace_kwargs)
    df_traces = build_trace_time_series_from_multiple_projects(all_projects, **trace_kwargs)

    # Group by dataset in each of the dataframes, and calculate the variance explained per neuron
    all_variances = {}
    for dataset_name, df_traces_single in tqdm(df_traces.groupby('dataset_name')):
        df_global_single = df_traces_global.groupby('dataset_name').get_group(dataset_name)
        # Calculate the variance explained for each neuron
        these_variances = {}
        # Drop the columns that have no values
        df_traces_single = df_traces_single.dropna(axis=1, how='all')
        for neuron_name in get_names_from_df(df_traces_single):
            if neuron_name in ['dataset_name', 'local_time']:
                continue
            try:
                neuron_trace = df_traces_single[neuron_name]
                global_trace = df_global_single[neuron_name]
                # These time series should have no nan values
                these_variances[neuron_name] = explained_variance_score(neuron_trace, global_trace)
            except KeyError as e:
                # This shouldn't happen
                these_variances[neuron_name] = np.nan
                print(e)
        all_variances[dataset_name] = these_variances
    # Make a dataframe from the dictionary
    df_variances = pd.DataFrame(all_variances).T

    if melt:
        df_var_exp_gcamp_melt = df_variances.reset_index().melt(id_vars='index')
        df_var_exp_gcamp_melt.rename(columns={'index': 'dataset_name', 'value': 'fraction_variance_explained',
                                              'variable': 'neuron_name'}, inplace=True)

        # Add a column for the simple variance threshold
        df_group = df_traces.drop(columns=['local_time']).groupby('dataset_name')
        df_active = df_group.var()
        df_active_melt = df_active.reset_index().melt(id_vars='dataset_name')
        df_active_melt.rename(columns={'index': 'dataset_name', 'value': 'variance',
                                       'variable': 'neuron_name'}, inplace=True)

        df_var_exp_gcamp_melt = df_var_exp_gcamp_melt.merge(df_active_melt, on=['dataset_name', 'neuron_name'])

        # Add columns for the relevant categories
        gas_sensing_neurons = list_of_gas_sensing_neurons()
        neurons_active_and_manifold_in_immob = list_neurons_manifold_in_immob()

        df_var_exp_gcamp_melt['has_id'] = df_var_exp_gcamp_melt['neuron_name'].apply(
            lambda x: 'neuron' not in x and 'VG_' not in x)
        df_var_exp_gcamp_melt['is_o2'] = df_var_exp_gcamp_melt['neuron_name'].apply(lambda x: x in gas_sensing_neurons)
        df_var_exp_gcamp_melt['active_in_immob'] = df_var_exp_gcamp_melt['neuron_name'].apply(
            lambda x: x in neurons_active_and_manifold_in_immob)

        df_var_exp_gcamp_melt['category'] = np.nan
        df_var_exp_gcamp_melt['category'] = df_var_exp_gcamp_melt['has_id'].replace({True: np.nan, False: 'Not IDed'})
        tmp = df_var_exp_gcamp_melt['is_o2'].replace({True: 'O2 or CO2 sensing', False: np.nan})
        df_var_exp_gcamp_melt['category'].fillna(tmp, inplace=True)
        tmp = df_var_exp_gcamp_melt['active_in_immob'].replace({True: 'Manifold in Immob', False: np.nan})
        df_var_exp_gcamp_melt['category'].fillna(tmp, inplace=True)
        df_var_exp_gcamp_melt['category'].fillna('Other neurons active in FM only', inplace=True)

        return df_var_exp_gcamp_melt
    else:
        return df_variances


def build_time_series_from_multiple_project_clusters(all_projects: Dict[str, ProjectData],
                                                     cluster_opt: dict = None,
                                                     z_score=False, trigger_opt: dict = None,
                                                     num_clusters=10,
                                                     **kwargs):
    """
    Similar to build_time_series_from_multiple_projects, using a clustering object to cluster traces across datasets.
    This object should be built from multiple datasets, and have names that can be split using split_flattened_index

    Note that this has all clusters as column names, and all concatenated time series as rows

    Parameters
    ----------
    all_projects
    z_score : bool - if True, z-scores the traces within each dataset (does not affect the clustering)

    Returns
    -------

    """
    if cluster_opt is None:
        cluster_opt = {'cluster_criterion': 'maxclust', 'linkage_threshold': num_clusters}
    # First build the clustering class
    multi_dataset_clusterer, clustering_intermediates = \
        clustered_triggered_averages_from_dict_of_projects(all_projects,
                                                           cluster_opt=cluster_opt, trigger_opt=trigger_opt, **kwargs)
    all_triggered_average_classes, df_triggered_good, dict_of_triggered_traces = clustering_intermediates

    df_all_clusters, df_imputed = build_dataframe_of_clusters(all_triggered_average_classes, multi_dataset_clusterer,
                                                              z_score)

    return df_imputed, df_all_clusters, multi_dataset_clusterer, all_triggered_average_classes


def build_dataframe_of_clusters(all_triggered_average_classes, multi_dataset_clusterer, z_score: bool = False):
    """

    Parameters
    ----------
    all_triggered_average_classes
    multi_dataset_clusterer
    z_score

    Returns
    -------

    """
    # Loop through clusters, and do two things:
    # 1. Get the average of all neurons per dataset
    # 2. Save in a long dataframe
    # Get a dict from each dataset to the neurons within each cluster in that data
    clust_and_dataset_to_neurons = {}
    for i_clust, combined_names in tqdm(multi_dataset_clusterer.per_cluster_names.items()):

        dataset_to_neurons = defaultdict(list)
        for name in combined_names:
            dataset_name, neuron_name = split_flattened_index([name])[name]
            dataset_to_neurons[dataset_name].append(neuron_name)
        clust_and_dataset_to_neurons[i_clust] = dataset_to_neurons
    list_of_dfs = []
    for i_clust, dataset_to_neurons in tqdm(clust_and_dataset_to_neurons.items()):
        one_clust_dict = defaultdict(list)
        for dataset_name, these_neurons in dataset_to_neurons.items():
            # Each trace is an average of multiple time series (neurons)
            this_class = all_triggered_average_classes[dataset_name]

            traces = this_class.df_traces[these_neurons]
            trace_mean = traces.mean(axis=1)
            if z_score:
                trace_mean = trace_mean - trace_mean.mean()
                trace_mean = trace_mean / trace_mean.std()

            one_clust_dict[f'cluster_{i_clust}'].extend(list(trace_mean))
            one_clust_dict['dataset_name'].extend([dataset_name] * len(trace_mean))

        list_of_dfs.append(pd.DataFrame(one_clust_dict))
    # Make a temp index to keep things in order within each dataframe
    for df in list_of_dfs:
        df['local_time_index'] = df.groupby('dataset_name').cumcount()
    # Merge entire list of dataframes on the dataset name and the new index
    df_all_clusters = reduce(
        lambda left, right: pd.merge(left, right, on=['dataset_name', 'local_time_index'], how='outer'),
        list_of_dfs)
    df_all_clusters = df_all_clusters.sort_values(['dataset_name', 'local_time_index']).reset_index(drop=True)
    # Remove missing values to make valid for sklearn
    try:
        df_imputed = df_all_clusters.dropna(axis=1, thresh=df_all_clusters.shape[0] * 0.9).copy()
        df_imputed.drop(columns=['dataset_name', 'local_time_index'], inplace=True)
        df_imputed = remove_outliers_using_std(df_imputed, std_factor=4)
        df_imputed = impute_missing_values_in_dataframe(df_imputed)
    except (ValueError, LinAlgError) as e:
        df_imputed = None
        logging.warning("Could not impute missing values in dataframe, returning None")
        print(e)
    return df_all_clusters, df_imputed


def get_variance_explained(project_data, trace_kwargs=None):
    if trace_kwargs is None:
        trace_kwargs = dict(use_paper_options=True)
    X = project_data.calc_default_traces(**trace_kwargs, interpolate_nan=True)
    X = detrend(X, axis=0)
    pca = PCA(n_components=20, whiten=False)
    pca.fit(X.T)
    return pca.explained_variance_ratio_


def get_all_variance_explained(all_projects_gcamp, all_projects_gfp, all_projects_immob):
    """
    Gets the variance explained by PCA for all projects up to 20 components

    Parameters
    ----------
    all_projects_gcamp
    all_projects_gfp
    all_projects_immob

    Returns
    -------

    """
    gcamp_var = {name: get_variance_explained(p) for name, p in tqdm(all_projects_gcamp.items())}
    gfp_var = {name: get_variance_explained(p) for name, p in tqdm(all_projects_gfp.items())}
    immob_var = {name: get_variance_explained(p) for name, p in tqdm(all_projects_immob.items())}
    # Cumulative sum
    gcamp_var_sum = np.array([np.cumsum(p) for p in gcamp_var.values()]).T
    gfp_var_sum = np.array([np.cumsum(p) for p in gfp_var.values()]).T
    immob_var_sum = np.array([np.cumsum(p) for p in immob_var.values()]).T

    return gcamp_var, gfp_var, immob_var, gcamp_var_sum, gfp_var_sum, immob_var_sum


def plot_variance_all_neurons(all_projects_gcamp, all_projects_gfp, include_gfp=True, include_legend=True,
                              match_yaxes=True, loop_not_facet_row=False, significance_line_at_95=True,
                              x='fraction_variance_explained', y='acv', names_to_keep_in_simple_id=('VB02', 'DB01'),
                              lag=1, output_folder=None, **kwargs):
    """
    Calculates the autocovariance of all neural traces, and plots a 4-panel figure

    Parameters
    ----------
    all_projects_gcamp
    all_projects_gfp

    Returns
    -------

    """
    df_summary, color_name_mapping, significance_line = calc_summary_dataframe_all_datasets(
        all_projects_gcamp,
        all_projects_gfp,
        significance_line_at_95,
        lag=lag,
        names_to_keep_in_simple_id=names_to_keep_in_simple_id,
        **kwargs
    )

    # Use D3 with the first gray, but skip orange and green (used for immobilized and gfp)
    cmap = px.colors.qualitative.D3.copy()
    green = cmap.pop(2)  # Green
    cmap.pop(1)  # Orange
    cmap.insert(0, px.colors.qualitative.Set1[-1])
    cmap.insert(4, green)  # If gfp is included, then it should be green and will be the 4th color

    # Actually plot
    if not include_gfp:
        print("Excluding gfp")
        df_summary = df_summary[df_summary['Type of data'] != 'gfp']
    scatter_opt = dict(y=y, x=x, symbol='Simple Neuron ID', marginal_y='box', size='multiplex_size', log_y=True,
                       size_max=10)
    if loop_not_facet_row:
        categories = df_summary['Type of data'].unique()
        cmap_copy = cmap.copy()
        all_figs = []
        for c in categories:
            df_subset = df_summary[df_summary['Type of data'] == c]
            range_y = [0.00005, 0.3]
            fig = px.scatter(df_subset, range_y=range_y,
                             color='Genotype and datatype', color_discrete_sequence=cmap_copy, **scatter_opt)
            # Create custom ticks for the y axis: logarithmic, but only on even powers of 10
            min_val_power_10 = np.log10(range_y[0])
            max_val_power_10 = np.log10(range_y[1])
            # min_val_power_10 = np.floor(np.log10(df_subset['acv'].min()))
            # max_val_power_10 = np.ceil(np.log10(df_subset['acv'].max()))
            # Get ticks, but not more than 3
            power_delta = max_val_power_10 - min_val_power_10
            num_to_skip = power_delta // 3
            yticks = [10 ** i for i in range(int(min_val_power_10), int(max_val_power_10) + 1) if i % num_to_skip == 0]
            fig.update_yaxes(tickvals=yticks, tickformat='.0e', tickmode='array')
            # fig.update_yaxes(minor=dict(nticks=0))  # Doesn't work
            cmap_copy.pop(1)
            all_figs.append(fig)
            # Manually change the legend of the "other" category to make sense
            # Also add spaces to the end so the lengths of each subplot are the same
            legend_suffix = ', Other neurons'
            target_length = max(map(len, color_name_mapping.values())) + len(legend_suffix)
            new_legend = f'{color_name_mapping[c]}{legend_suffix}'
            new_legend = new_legend.ljust(target_length + 2)
            fig.data[0].name = new_legend
    else:
        fig = px.scatter(df_summary, facet_row='Type of data',
                         color_discrete_sequence=cmap, range_y=[0.00005, 0.3],
                         color='Genotype and datatype', **scatter_opt)
        all_figs = [fig]

    # Postprocessing
    for i, fig in enumerate(all_figs):
        fig.add_hline(y=significance_line,
                      line_width=2, line_dash="dash",
                      # col=1, #annotation_text="95% line of gfp", annotation_position="bottom left"
                      )
        if loop_not_facet_row:
            # Turn on x and y axis lines
            fig.update_layout(
                xaxis=dict(showline=True, linecolor='black'),
                yaxis=dict(showline=True, linecolor='black')
            )
            # Turn off xaxis label and ticks on all but last figure
            if i != len(all_figs) - 1:
                fig.update_xaxes(title="", showticklabels=False, overwrite=True)
            else:
                fig.update_xaxes(overwrite=True)
            # Turn off most yaxis labels
            if i != 1:
                fig.update_yaxes(title="", overwrite=True)
            else:
                fig.update_yaxes(title="log(autocovariance)", overwrite=True)
        else:
            # Turn off most yaxis labels
            fig.update_yaxes(row=1, title="", overwrite=True)
            fig.update_yaxes(row=3, title="", overwrite=True)

        # Remove white border around individual markers
        # fig.update_traces(marker=dict(line=dict(width=0)))

        if not include_legend:
            fig.update_traces(showlegend=False)

        # Turn off side-titles: https://plotly.com/python/facet-plots/#customizing-subplot-figure-titles
        fig.for_each_annotation(lambda a: a.update(text=""))

        # Turn off legend title
        fig.update_layout(legend_title_text='')

        # Decouple y axes to fully use space
        if not match_yaxes:
            fig.update_yaxes(matches=None)

        height_factor = 0.6 / len(all_figs)
        apply_figure_settings(fig, width_factor=1.0, height_factor=height_factor, plotly_not_matplotlib=True)

        if output_folder is not None:
            fname = os.path.join(output_folder, f'summary_of_neurons_with_signal_covariance-{i}.png')
            fig.write_image(fname, scale=7)
            fname = Path(fname).with_suffix('.svg')
            fig.write_image(fname)

        fig.show()

    return all_figs, df_summary, significance_line, cmap


def calc_summary_dataframe_all_datasets(all_projects_gcamp, all_projects_gfp, significance_line_at_95=True, lag=1,
                                        names_to_keep_in_simple_id=('VB02', 'BAG'),
                                        **trace_kwargs):
    base_marker_size = 0.2
    big_marker_size = 0.5
    str_other_neurons = 'Other neurons'

    all_proj = MultiProjectWrapper(all_projects=all_projects_gcamp)
    all_proj_gfp = MultiProjectWrapper(all_projects=all_projects_gfp)
    # Calculate traces
    trace_opt = dict(rename_neurons_using_manual_ids=True, use_paper_options=True)
    trace_opt.update(trace_kwargs)
    dict_all_traces = all_proj.calc_default_traces(**trace_opt)
    df_all_traces = flatten_multiindex_columns(pd.concat(dict_all_traces, axis=1))
    pca_mode0 = pd.concat(all_proj.calc_correlation_to_pc1(**trace_opt)).values
    # Also residuals
    dict_all_traces = all_proj.calc_default_traces(**trace_opt, residual_mode='pca', interpolate_nan=True)
    df_all_traces_resid = flatten_multiindex_columns(pd.concat(dict_all_traces, axis=1))
    # Also global mode
    dict_all_traces = all_proj.calc_default_traces(**trace_opt, residual_mode='pca_global', interpolate_nan=True)
    df_all_traces_global = flatten_multiindex_columns(pd.concat(dict_all_traces, axis=1))
    # Also for gfp
    dict_all_traces_gfp = all_proj_gfp.calc_default_traces(**trace_opt)
    df_all_traces_gfp = flatten_multiindex_columns(pd.concat(dict_all_traces_gfp, axis=1))
    pca_mode0_gfp = pd.concat(all_proj_gfp.calc_correlation_to_pc1(**trace_opt)).values
    all_summary_dfs = []
    all_trace_dfs = {'gcamp': df_all_traces, 'global gcamp': df_all_traces_global,
                     'residual gcamp': df_all_traces_resid, 'gfp': df_all_traces_gfp}
    # Calculate autocovariance and other metadata
    for name, df_base in all_trace_dfs.items():
        this_var = df_base.var()
        this_mean = df_base.mean()
        this_corr = df_base.apply(lambda col: col.autocorr(lag=lag))
        this_cov = df_base.apply(lambda col: col.autocorr(lag=lag) * col.var())
        df = pd.DataFrame({'var': this_var, 'mean': this_mean, 'acf': this_corr, 'acovf': this_cov})
        df['genotype'] = name
        if name == 'gfp':
            mode = pca_mode0_gfp
        else:
            mode = pca_mode0
        df['pc0'] = mode
        df['pc0_high'] = mode > 0.2
        df['pc0_low'] = mode < -0.2
        all_summary_dfs.append(df)
    df_summary = pd.concat(all_summary_dfs, axis=0)
    df_summary.columns = ['var', 'mean', 'Autocorrelation', 'acv', 'Type of data',
                          'Correlation to PC1', 'pc0_high', 'pc0_low']
    flattened_names = pd.DataFrame(split_flattened_index(df_summary.index)).T
    df_summary['dataset_name'] = flattened_names[0]
    df_summary['neuron_name'] = flattened_names[1]
    df_summary['has_manual_id'] = ['neuron' not in name and name != '' for name in df_summary['neuron_name']]
    # df_summary['bag'] = ['BAG' in name for name in df_summary['neuron_name']]
    df_summary['neuron_name_simple'] = [name[:-1] if name.endswith(('L', 'R')) else name for name in
                                        df_summary['neuron_name']]
    # df_summary['vb02'] = ['VB02' in name for name in df_summary['neuron_name']]

    # Columns related to specifically highlighted neurons
    keep_name_func = lambda query: any([base_name in query for base_name in names_to_keep_in_simple_id])
    df_summary['Neuron ID'] = [name if keep_name_func(name) else str_other_neurons for name in
                               df_summary['neuron_name']]
    df_summary['Simple Neuron ID'] = [name if keep_name_func(name) else str_other_neurons for name in
                                      df_summary['neuron_name_simple']]
    df_summary['multiplex_size'] = [big_marker_size if keep_name_func(name) else base_marker_size for name in
                                    df_summary['neuron_name']]
    # Build the rows that will be in the final plot
    col_name = 'Genotype and datatype'
    color_col = []
    color_name_mapping = {'gcamp': 'Raw', 'global gcamp': 'Global', 'residual gcamp': 'Residual', 'gfp': 'GFP'}
    for i, row in df_summary.iterrows():
        # Put all of the other neurons in the background, regardless of data type
        if row['Neuron ID'] == str_other_neurons:
            color_col.append('Other neurons')
        # These neurons will actually have color
        else:
            color_col.append(color_name_mapping[row['Type of data']])
    df_summary[col_name] = color_col
    # Column relative to gfp
    if significance_line_at_95:
        significance_line = df_summary.groupby('Type of data').quantile(0.95, numeric_only=True).at['gfp', 'acv']
    else:
        significance_line = df_summary.groupby('Type of data').quantile(0.5, numeric_only=True).at['gfp', 'acv']
    df_summary['Significant'] = df_summary['acv'] > significance_line

    # Update formatting to be like others
    df_summary.reset_index(inplace=True)
    df_summary.rename(columns={'index': 'neuron_name_with_dataset'}, inplace=True)

    # Add new things: variance explained by PCA
    df_var_exp_gcamp = build_dataframe_of_variance_explained(all_projects_gcamp, melt=True, **trace_opt)
    df_var_exp_gcamp['Type of data'] = 'gcamp'
    df_var_exp_gfp = build_dataframe_of_variance_explained(all_projects_gfp, melt=True, **trace_opt)
    df_var_exp_gfp['Type of data'] = 'gfp'
    df_var_exp = pd.concat([df_var_exp_gcamp, df_var_exp_gfp], axis=0)

    # Sanity checks
    # Note that the raw sizes aren't the same, because df_summary has the global and residual rows as well
    # But only for gcamp, not gfp
    num_raw_gcamp = df_summary.dropna().groupby('Type of data').count().at['gcamp', 'mean']
    num_raw_gfp = df_summary.dropna().groupby('Type of data').count().at['gfp', 'mean']

    num_raw_gcamp2 = df_var_exp.dropna().groupby('Type of data').count().at['gcamp', 'variance']
    num_raw_gfp2 = df_var_exp.dropna().groupby('Type of data').count().at['gfp', 'variance']

    assert num_raw_gcamp == num_raw_gcamp2, "Mismatch in number of gcamp neurons"
    assert num_raw_gfp == num_raw_gfp2, "Mismatch in number of gfp neurons"

    # This merge works even though the global and residual rows are added for gcamp
    df_summary = df_summary.merge(df_var_exp, on=['dataset_name', 'neuron_name'],
                                  suffixes=('', '_var_exp'))
    # Drop duplicate columns from the var_exp merge
    df_summary.drop(df_summary.filter(regex='_var_exp$').columns, axis=1, inplace=True)

    return df_summary, color_name_mapping, significance_line
