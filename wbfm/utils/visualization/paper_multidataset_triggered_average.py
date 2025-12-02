"""
Designed to plot the triggered average of the paper's datasets.
"""
import itertools
import logging
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass
import random
from typing import Dict, Optional, List, Tuple
import plotly.graph_objects as go

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import stats
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm
from wbfm.utils.external.utils_matplotlib import round_yticks

from wbfm.utils.external.utils_pandas import split_flattened_index, combine_columns_with_suffix
from wbfm.utils.external.utils_plotly import float2rgba, add_annotation_lines
from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes, add_behavior_shading_to_plot
from wbfm.utils.general.utils_paper import apply_figure_settings, paper_trace_settings, plotly_paper_color_discrete_map, \
    plot_box_multi_axis
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.traces.triggered_averages import clustered_triggered_averages_from_dict_of_projects, \
    ClusteredTriggeredAverages, plot_triggered_average_from_matrix_low_level, \
    calc_p_value_using_ttest_triggered_average, FullDatasetTriggeredAverages
from wbfm.utils.general.high_performance_pandas import get_names_from_df
from wbfm.utils.visualization.utils_plot_traces import add_p_value_annotation, convert_channel_mode_to_axis_label
from wbfm.utils.external.custom_errors import NoBehaviorAnnotationsError


@dataclass
class PaperColoredTracePlotter:
    """
    Class to plot the colored traces of the paper's datasets.

    Specifically for raw/global/residual decompositions
    """

    @staticmethod
    def get_color_from_data_type(trigger_type, is_mutant=False, is_hiscl=False, plotly_style=False):
        cmap = plt.get_cmap('tab10')
        if is_mutant:
            # Mutant color is unique (pink)
            return cmap(6)
        if is_hiscl:
            # Unique (darker green)
            return plt.get_cmap('tab20b')(4)
        color_mapping = {'raw_rev': cmap(0),
                         'raw': cmap(0),
                         'raw_fwd': cmap(0),
                         'raw_vt': cmap(0),
                         'raw_dt': cmap(0),
                         'global_rev': cmap(3),
                         'global': cmap(3),
                         'global_fwd': cmap(3),
                         'residual': cmap(4),
                         'residual_collision': cmap(4),
                         'residual_rectified_fwd': cmap(4),
                         'residual_rectified_rev': cmap(4),
                         # Should I use the 'raw' colors for this?
                         'kymo': 'black',
                         'stimulus': cmap(2),
                         'self_collision': cmap(0),
                         'mutant': cmap(6),
                         'immob': plotly_paper_color_discrete_map()['immob'],}
        if trigger_type not in color_mapping:
            raise ValueError(f'Invalid trigger type: {trigger_type}; must be one of {list(color_mapping.keys())}')
        color = color_mapping[trigger_type]
        if plotly_style:
            color = float2rgba(color)
        return color

    def get_trace_opt(self, **kwargs):
        trace_opt = paper_trace_settings()
        trace_opt['use_paper_options'] = True
        trace_opt.update(kwargs)
        return trace_opt

    @classmethod
    def get_behavior_color_from_neuron_name(cls, neuron_name):
        """
        Returns the color of the cluster based on the neuron name.

        Parameters
        ----------
        neuron_name

        Returns
        -------

        """
        color_mapping = dict(
            RIS='tab:blue',
            AVA='tab:orange',
            RIV='tab:green',
            RMEL='tab:green',
            RMER='tab:green',
            SMDV='tab:green',
            RID='tab:red',
            RME='tab:purple',
            VB01='tab:purple',
            VB02='tab:purple',
            VB03='tab:purple',
            DB01='tab:purple',
            DB02='tab:purple',
            AVB='tab:red',
            RIB='tab:red',
            IL1L='black',
            IL2L='black',
        )
        # Add keys by adding the L/R and V/D suffixes
        for k in list(color_mapping.keys()):
            for suffix in ['L', 'R', 'V', 'D']:
                # Only add if not already in the dictionary
                if k + suffix not in color_mapping:
                    color_mapping[k + suffix] = color_mapping[k]
        if neuron_name not in color_mapping:
            logging.warning(f"Neuron name {neuron_name} not found in color mapping; using default color")
            # Use a default color (black)
            color_mapping[neuron_name] = 'black'
        return color_mapping[neuron_name]


@dataclass
class PaperMultiDatasetTriggeredAverage(PaperColoredTracePlotter):
    """
    Class to plot the triggered average of the paper's datasets.

    Specifically designed for residual figures, and uses the proper colors for each type of triggered average.
    """

    all_projects: Dict[str, ProjectData]

    # Options for traces
    min_nonnan: Optional[float] = 0.8

    # Three different sets of parameters: raw, global, and residual
    dataset_clusterer_dict: Dict[str, ClusteredTriggeredAverages] = None
    intermediates_dict: Dict[str, tuple] = None

    trace_opt: Optional[dict] = None
    trigger_opt: dict = None

    # Optional trigger types
    calculate_stimulus: bool = False
    calculate_residual: bool = True
    calculate_global: bool = True
    calculate_turns: bool = True
    calculate_self_collision: bool = False

    verbose: int = 0

    def __post_init__(self):
        # Analyze the project data to get the clusterers and intermediates
        trace_base_opt = self.get_trace_opt(min_nonnan=self.min_nonnan)
        trace_base_opt['use_paper_options'] = True
        if self.trace_opt is not None:
            trace_base_opt.update(self.trace_opt)

        trigger_base_opt = dict(min_duration=4, gap_size_to_remove=4, max_num_points_after_event=40, fixed_num_points_after_event=None)
        if self.trigger_opt is not None:
            trigger_base_opt.update(self.trigger_opt)
        self.trigger_opt = trigger_base_opt

        # Set each project to use physical time
        for proj in self.all_projects.values():
            proj.use_physical_time = True

        self.dataset_clusterer_dict = defaultdict(None)
        # Per trigger type: Dict[str, FullDatasetTriggeredAverages], pd.DataFrame, Dict[str, pd.DataFrame]
        self.intermediates_dict: Tuple[Dict[str, FullDatasetTriggeredAverages], pd.DataFrame, Dict[str, pd.DataFrame]] = (
            defaultdict(lambda: (None, None, None)))

        if self.calculate_residual:
            try:
                # Note: these won't work for immobilized data

                if self.verbose > 0:
                    print("Calculating residual hilbert-triggered averages")
                trigger_opt = dict(use_hilbert_phase=True, state=None)
                trigger_opt.update(self.trigger_opt)
                trace_opt = dict(residual_mode='pca')
                trace_opt.update(trace_base_opt)
                out = clustered_triggered_averages_from_dict_of_projects(self.all_projects, trigger_opt=trigger_opt,
                                                                         trace_opt=trace_opt, verbose=self.verbose)
                self.dataset_clusterer_dict['residual'] = out[0]
                self.intermediates_dict['residual'] = out[1]

                if self.verbose > 0:
                    print("Calculating residual hilbert-triggered averages (rectified for FWD)")
                trigger_opt['only_allow_events_during_state'] = BehaviorCodes.FWD
                cluster_opt = {}
                out = clustered_triggered_averages_from_dict_of_projects(self.all_projects, cluster_opt=cluster_opt,
                                                                         trigger_opt=trigger_opt, trace_opt=trace_opt, verbose=self.verbose)
                self.dataset_clusterer_dict['residual_rectified_fwd'] = out[0]
                self.intermediates_dict['residual_rectified_fwd'] = out[1]

                if self.verbose > 0:
                    print("Calculating residual hilbert-triggered averages (rectified for REV)")
                trigger_opt['only_allow_events_during_state'] = BehaviorCodes.REV
                cluster_opt = {}
                out = clustered_triggered_averages_from_dict_of_projects(self.all_projects, cluster_opt=cluster_opt,
                                                                         trigger_opt=trigger_opt, trace_opt=trace_opt, verbose=self.verbose)
                self.dataset_clusterer_dict['residual_rectified_rev'] = out[0]
                self.intermediates_dict['residual_rectified_rev'] = out[1]

                if self.calculate_self_collision:
                    # Only used for BAG: self-collision triggered
                    trigger_opt = dict(use_hilbert_phase=False, state=BehaviorCodes.SELF_COLLISION)
                    trigger_opt.update(self.trigger_opt)
                    trace_opt = dict(residual_mode='pca')
                    trace_opt.update(trace_base_opt)
                    if self.verbose > 0:
                        print("Calculating residual collision-triggered averages")
                    out = clustered_triggered_averages_from_dict_of_projects(self.all_projects, trigger_opt=trigger_opt,
                                                                             trace_opt=trace_opt, verbose=self.verbose)
                    self.dataset_clusterer_dict['residual_collision'] = out[0]
                    self.intermediates_dict['residual_collision'] = out[1]

            except TypeError as e:
                print("Hilbert triggered averages failed; this may be because the data is immobilized")
                print("Only 'global' triggered averages will be available")

        if self.calculate_global:
            # Slow reversal triggered (global)
            trigger_opt = dict(use_hilbert_phase=False, state=BehaviorCodes.REV)
            trigger_opt.update(self.trigger_opt)
            trace_opt = dict(residual_mode='pca_global')
            trace_opt.update(trace_base_opt)
            if self.verbose > 0:
                print("Calculating global REV-triggered averages")
            out = clustered_triggered_averages_from_dict_of_projects(self.all_projects, trigger_opt=trigger_opt,
                                                                     trace_opt=trace_opt, verbose=self.verbose)
            self.dataset_clusterer_dict['global_rev'] = out[0]
            self.intermediates_dict['global_rev'] = out[1]

            # Slow forward triggered (global)
            trigger_opt = dict(use_hilbert_phase=False, state=BehaviorCodes.FWD)
            trigger_opt.update(self.trigger_opt)
            if self.verbose > 0:
                print("Calculating global FWD-triggered averages")
            out = clustered_triggered_averages_from_dict_of_projects(self.all_projects, trigger_opt=trigger_opt,
                                                                     trace_opt=trace_opt, verbose=self.verbose)
            self.dataset_clusterer_dict['global_fwd'] = out[0]
            self.intermediates_dict['global_fwd'] = out[1]

        # Always calculate: raw
        trigger_dict = {'raw_rev': BehaviorCodes.REV, 'raw_fwd': BehaviorCodes.FWD}
        if self.calculate_turns:
            trigger_dict.update({'raw_vt': BehaviorCodes.VENTRAL_TURN, 'raw_dt': BehaviorCodes.DORSAL_TURN})
        trace_opt = dict(residual_mode=None)
        trace_opt.update(trace_base_opt)
        # Raw reversal triggered and forward triggered
        for trigger_type, state in trigger_dict.items():
            try:
                trigger_opt = dict(use_hilbert_phase=False, state=state)
                trigger_opt.update(self.trigger_opt)
                if self.verbose > 0:
                    print(f"Calculating raw trace {trigger_type}-triggered averages")
                out = clustered_triggered_averages_from_dict_of_projects(self.all_projects, trigger_opt=trigger_opt,
                                                                         trace_opt=trace_opt, verbose=self.verbose)
                self.dataset_clusterer_dict[trigger_type] = out[0]
                self.intermediates_dict[trigger_type] = out[1]
            except (IndexError, KeyError, NoBehaviorAnnotationsError):
                print(f"Trigger type {trigger_type} failed; this may be because the data is immobilized")

        # Optional
        if self.calculate_stimulus:
            trigger_opt = dict(use_hilbert_phase=False, state=BehaviorCodes.STIMULUS)
            trigger_opt.update(self.trigger_opt)
            if self.verbose > 0:
                print(f"Calculating raw trace stimulus-triggered averages")
            out = clustered_triggered_averages_from_dict_of_projects(self.all_projects, trigger_opt=trigger_opt,
                                                                     trace_opt=trace_opt)
            self.dataset_clusterer_dict['stimulus'] = out[0]
            self.intermediates_dict['stimulus'] = out[1]

        if self.calculate_self_collision:
            trigger_opt = dict(use_hilbert_phase=False, state=BehaviorCodes.SELF_COLLISION)
            trigger_opt.update(self.trigger_opt)
            if self.verbose > 0:
                print(f"Calculating raw trace self_collision-triggered averages")
            out = clustered_triggered_averages_from_dict_of_projects(self.all_projects, trigger_opt=trigger_opt,
                                                                     trace_opt=trace_opt)
            self.dataset_clusterer_dict['self_collision'] = out[0]
            self.intermediates_dict['self_collision'] = out[1]

    def valid_trigger_types(self):
        """
        Returns a list of valid trigger types.
        """
        return list(self.dataset_clusterer_dict.keys())

    def get_clusterer_from_trigger_type(self, trigger_type):
        trigger_mapping = self.dataset_clusterer_dict
        if trigger_type not in trigger_mapping:
            raise ValueError(f'Invalid trigger type: {trigger_type}; must be one of {list(trigger_mapping.keys())}')
        return trigger_mapping[trigger_type]

    def get_df_triggered_from_trigger_type(self, trigger_type, return_individual_traces=False) -> pd.DataFrame:
        """
        Returns by default a precalculated dataframe of the form:
        - Columns: Neuron names combined with the dataset name (e.g. 2022-11-23_worm9_BAGL)
        - Rows: Time points

        Parameters
        ----------
        trigger_type
        return_individual_traces - if False, then return the average across events per dataset per neuron (default)
            if True, then return a dictionary of dataframes, where each dataframe is each event within the dataset.
            Note: each dataset has a different number of events (rows), but the same number of time points (columns)

        Returns
        -------

        """
        df_mapping = self.intermediates_dict
        if not return_individual_traces:
            df_mapping = {k: v[1] if v else None for k, v in df_mapping.items()}
        else:
            df_mapping = {k: v[2] if v else None for k, v in df_mapping.items()}
        if trigger_type not in df_mapping:
            raise ValueError(f'Invalid trigger type: {trigger_type}; must be one of {list(df_mapping.keys())}')
        if not return_individual_traces:
            return df_mapping[trigger_type]
        else:
            # Is a dictionary of dataframes
            df_all = pd.concat(df_mapping[trigger_type]).T
            df_all.columns = [f"{dataset_name}_{event_index}" for dataset_name, event_index in df_all.columns]
            return df_all

    def get_df_triggered_from_trigger_type_all_traces_as_df(self, trigger_type, melt_neuron=None):
        """
        Uses get_df_triggered_from_trigger_type, but processes the columns into a multiindex dataframe

        Note: only tested if the triggered average has the same number of time points for each event

        Columns are in the form:
        - Level 0: Neuron name
        - Level 1: Dataset name
        - Level 2: Trial index

        Rows are the time points

        Examples:
            # Plot all trials, all datasets, all time points
            # Do not use 'facet_row' if there are many datasets
            df = self.get_df_triggered_from_trigger_type_all_traces_as_df('raw_rev', melt_neuron='BAGL')
            px.box(df, facet_row='dataset_name', y='value', color='before', points='all', x='trial_idx',
                   category_orders={'before': ['True', 'False']})

            # Collapse time dimension using median
            df_grouped = df_aqr.groupby(['dataset_name', 'trial_idx', 'before']).median().reset_index()

            px.box(df_grouped, x='dataset_name', y='value', color='before', points='all',
                   category_orders={'before': ['True', 'False']})

        Returns
        -------

        """
        df = self.get_df_triggered_from_trigger_type(trigger_type, return_individual_traces=True)
        # Remove neurons with default names
        df = df.drop(columns=df.columns[df.columns.str.contains('neuron')]).copy()

        def split_func(name):
            split_name = name.split('_')
            dataset_name = '_'.join(split_name[:-2])
            neuron_name = split_name[-2]
            trial_idx = split_name[-1]
            return dataset_name, neuron_name, trial_idx

        idx = (split_func(c) for c in df.columns)
        df.columns = pd.MultiIndex.from_tuples(idx, names=['dataset_name', 'neuron_name', 'trial_idx'])
        df = df.swaplevel(i=0, j=1, axis=1)

        if melt_neuron is not None:
            df = df[melt_neuron].melt(ignore_index=False).reset_index().copy()
            df['before'] = df['index'] < 0  # Add a column for before/after the event

        return df

    def get_title_from_trigger_type(self, trigger_type):
        title_mapping = {'raw_rev': 'Raw reversal triggered',
                         'raw_fwd': 'Raw forward triggered',
                         'raw_vt': 'Raw ventral turn triggered',
                         'raw_dt': 'Raw dorsal turn triggered',
                         'global_rev': 'Global reversal triggered',
                         'global_fwd': 'Global forward triggered',
                         'residual': 'Residual undulation triggered',
                         'residual_collision': 'Residual collision triggered',
                         'residual_rectified_fwd': 'Residual (rectified fwd, undulation triggered)',
                         'residual_rectified_rev': 'Residual (rectified rev, undulation triggered)',
                         'kymo': 'Kymograph',
                         'stimulus': 'Stimulus triggered',
                         'self_collision': 'Self-collision triggered'}
        if trigger_type not in title_mapping:
            raise ValueError(f'Invalid trigger type: {trigger_type}; must be one of {list(title_mapping.keys())}')
        return title_mapping[trigger_type]

    def _get_xlabel_from_trigger_type(self, trigger_type):
        title_mapping = {'raw_rev': 'Time relative to\nReversal (s)',
                         'raw_fwd': 'Time relative to\nForward (s)',
                         'raw_vt': 'Time relative to\nVentral Turn (s)',
                         'raw_dt': 'Time relative to\nDorsal Turn (s)',
                         'global_rev': 'Time relative to\nReversal (s)',
                         'global_fwd': 'Time relative to\nForward (s)',
                         'residual': 'Time relative to\nventral undulation (s)',
                         'residual_collision': 'Time relative to\nself-collision (s)',
                         'residual_rectified_fwd': 'Time relative to\nventral undulation (s)',
                         'residual_rectified_rev': 'Time relative to\nventral undulation (s)',
                         'kymo': 'Time (s)',
                         'stimulus': 'Time (s)',
                         'self_collision': 'Time relative to\nself-collision (s)',}
        if trigger_type not in title_mapping:
            raise ValueError(f'Invalid trigger type: {trigger_type}; must be one of {list(title_mapping.keys())}')
        return title_mapping[trigger_type]

    def get_trace_difference_auc(self, neuron0, neuron1, trigger_type, num_iters=100, z_score=False, norm_type='corr',
                                 shuffle_dataset_pairs=True, return_individual_traces=False):
        """
        Calculates the area under the curve of the difference between two neurons.

        Parameters
        ----------
        trigger_type
        neuron0
        neuron1
        num_iters - number of iterations to perform (only used if shuffle_dataset_pairs is True)
        z_score - if True, then we will z-score the traces before calculating the difference
        shuffle_dataset_pairs - if True, then we will shuffle the dataset pairs and subtract them. If False, then we
            will only subtract the same dataset, meaning that both neurons must be present.
        return_individual_traces - if True, then return a list of the difference for each event in each dataset.
            If False, then return a list of the difference for each dataset. (default)
            Mutually exclusive with shuffle_dataset_pairs=True

        Returns
        -------

        """
        df_or_dict = self.get_df_triggered_from_trigger_type(trigger_type,
                                                             return_individual_traces=return_individual_traces)
        if z_score:
            if not return_individual_traces:
                df_or_dict = (df_or_dict - df_or_dict.mean()) / df_or_dict.std()
            else:
                # Then each dataset should be z-scored separately, but across all events simultaneously
                # Numpy will do this automatically (pandas tries to keep the axis)
                keys_to_remove = []
                for _name, df in df_or_dict.items():
                    if df is None:
                        keys_to_remove.append(_name)
                        continue
                    df_or_dict[_name] = (df - np.nanmean(df)) / np.nanstd(df)
                for k in keys_to_remove:
                    del df_or_dict[k]

        if not return_individual_traces:
            names0 = [n for n in list(df_or_dict.columns) if neuron0 in n]
            names1 = [n for n in list(df_or_dict.columns) if neuron1 in n]
        else:
            names0 = [n for n in list(df_or_dict.keys()) if neuron0 in n]
            names1 = [n for n in list(df_or_dict.keys()) if neuron1 in n]
        if len(names0) == 0 or len(names1) == 0:
            raise ValueError(f'Neuron name {neuron0} or {neuron1} not found')

        # Define the summary statistic
        def norm(x, y):
            if norm_type == 'corr':
                # Calculate the correlation, which is what is shown in the clustered matrix
                if isinstance(x, pd.Series):
                    return np.nanmean(x.corr(y))
                else:
                    # Then need to apply the correlation to each row (paired)
                    ind = x.index
                    all_corrs = [x.loc[i, :].corr(y.loc[i, :]) for i in ind]
                    return np.nanmean(all_corrs)
            elif norm_type == 'auc':
                # This is the same as the mean squared difference
                delta = np.nanmean((x - y).pow(2))
                x_norm = np.nanmean(x.pow(2))
                y_norm = np.nanmean(y.pow(2))
                return delta / (x_norm + y_norm)
            else:
                raise ValueError(f"Invalid norm type {norm_type}; must be one of 'corr' or 'auc'")

        if shuffle_dataset_pairs:
            if return_individual_traces:
                raise NotImplementedError("Shuffling dataset pairs not implemented for return_individual_traces=True,"
                                          " because there is no clean way to pair up the traces")
            # There are two lists of neurons, and we want to choose random pairs
            samples = [(random.choice(names0), random.choice(names1)) for _ in range(num_iters)]
            all_norms = []
            for name0, name1 in samples:
                trace0 = df_or_dict[name0]
                trace1 = df_or_dict[name1]
                all_norms.append(norm(trace0, trace1))
        else:
            # Get the names of the datasets
            if not return_individual_traces:
                column_names = get_names_from_df(df_or_dict)
            else:
                column_names = list(df_or_dict.keys())
            split_column_names = split_flattened_index(column_names)
            all_dataset_names = {dataset_name for col_name, (dataset_name, neuron_name) in split_column_names.items()}
            # Loop through datasets, and check if both neurons are present
            all_norms = []
            for dataset_name in all_dataset_names:
                name0 = f"{dataset_name}_{neuron0}"
                name1 = f"{dataset_name}_{neuron1}"
                if name0 in column_names and name1 in column_names:
                    trace0 = df_or_dict[name0]
                    trace1 = df_or_dict[name1]
                    # Regardless of the traces being one trace or a dataframe of events, we compress to a single norm
                    all_norms.append(norm(trace0, trace1))
                else:
                    all_norms.append(np.nan)
            if len(all_norms) == 0:
                raise ValueError(f"Neurons {neuron0} and {neuron1} not found simultaneously in any datasets")

        return all_norms

    def get_trace_difference_auc_multiple_neurons(self, list_of_neurons, trigger_type, norm_type='corr',
                                                  baseline_neuron=None, df_norms=None, **kwargs):
        """
        Use get_trace_difference for pairs of neurons, generated as all combinations of the neurons in list_of_neurons.

        Parameters
        ----------
        trigger_type
        list_of_neurons

        Returns
        -------

        """
        if baseline_neuron is None:
            neuron_combinations = list(itertools.combinations(list_of_neurons, 2))
        else:
            neuron_combinations = [(baseline_neuron, n) for n in list_of_neurons if n != baseline_neuron]
        dict_norms = {}
        for neuron0, neuron1 in tqdm(neuron_combinations, leave=False):
            key = f"{neuron0}-{neuron1}"
            dict_norms[key] = self.get_trace_difference_auc(neuron0, neuron1, trigger_type, norm_type=norm_type,
                                                            **kwargs)

        df_norms = pd.DataFrame(dict_norms)
        return df_norms

    def get_fig_opt(self, height_factor=1, width_factor=1):
        return dict(dpi=300, figsize=(width_factor*10/3, height_factor*10/(2*3)))

    def plot_triggered_average_single_neuron(self, neuron_name, trigger_type, output_folder=None,
                                             fig=None, ax=None, title=None, include_neuron_in_title=False,
                                             xlim=None, ylim=None, min_lines=2, round_y_ticks=False,
                                             show_title=False, show_x_ticks=True, show_y_ticks=True,
                                             show_y_label=True, show_y_label_only_export=False, show_x_label=True, color=None, is_mutant=False,
                                             z_score=False, fig_kwargs=None, annotation_kwargs=None,
                                             legend=False, i_figure=3,
                                             apply_changes_even_if_no_trace=True, show_individual_lines=False,
                                             return_individual_traces=False, use_plotly=False,
                                             df_idx_range=None,
                                             width_factor_addition=0, height_factor_addition=0,
                                             height_factor=None, width_factor=None,
                                             to_show=True, fig_opt=None, DEBUG=False):
        """
        Plot the triggered average for a single neuron.

        Parameters
        ----------
        neuron_name - Name of the neuron to plot; see get_valid_neuron_names for the list of valid names
        trigger_type - Type of the trigger to plot (See valid_trigger_types for the list of types)
        output_folder - Folder to save the figure to (if None, then do not save)
        fig - Figure to plot on (if None, then create a new figure)
        ax - Axes to plot on (if None, then create a new axes)
        title - Title of the plot (if None, then use the default title for the trigger type)
        include_neuron_in_title - If True, then include the neuron name in the title
        xlim - x-axis limits (if None, then do not set limits)
        ylim - y-axis limits (if None, then do not set limits)
        min_lines - Minimum number of lines to plot (if less than this, then do not plot any values for those time points)
        round_y_ticks - If True, then round the y-ticks to the nearest integer
        show_title - If True, then show the title
        show_x_ticks - If True, then show the x-ticks
        show_y_ticks - If True, then show the y-ticks
        show_y_label - If True, then show the y-label
        show_y_label_only_export - If True, then only show the y-label when exporting the figure
        show_x_label - If True, then show the x-label
        color - Color of the trace (if None, then use the default color for the trigger type)
        is_mutant - If True, then use the mutant color (pink)
        z_score - If True, then z-score the traces before plotting
        fig_kwargs - Additional keyword arguments for the figure (if None, then use the default figure options)
        annotation_kwargs - Additional keyword arguments for the annotations (if None, then use the default annotation options)
        legend - If True, then show the legend
        i_figure - Figure index (used for default figure size options when saving the figure)
        apply_changes_even_if_no_trace - If True, then apply the changes even if there is no trace for the neuron
        show_individual_lines - If True, then show the individual lines for each event (default False)
        return_individual_traces - If True, then return the individual traces for each event
        use_plotly - If True, then use Plotly for plotting (default False)
        df_idx_range - If not None, then use this to limit the range of the dataframe
        width_factor_addition - Additional width factor to add to the figure size
        height_factor_addition - Additional height factor to add to the figure size
        height_factor - Height factor for the figure size (if None, then use the default height factor)
        width_factor - Width factor for the figure size (if None, then use the default width factor)
        to_show - If True, then show the figure (default True)
        fig_opt - Additional figure options (if None, then use the default figure options)
        DEBUG - If True, then print debug information
        """
        if isinstance(trigger_type, list):
            # Plot stacked
            # raise NotImplementedError("Not sure why this isn't working (just hspace)")
            _opts = locals().copy()
            fig, axes = plt.subplots(nrows=len(trigger_type), **self.get_fig_opt())
            _opts.pop('trigger_type')
            _opts.pop('self')
            _opts['output_folder'] = None
            _opts['to_show'] = False
            for i, (t, ax) in enumerate(zip(trigger_type, list(axes))):
                _opts['ax'] = ax
                _opts['fig'] = fig
                self.plot_triggered_average_single_neuron(trigger_type=t, **_opts)
            apply_figure_settings(fig=fig, width_factor=width_factor, height_factor=height_factor, plotly_not_matplotlib=False)
            plt.subplots_adjust(hspace=0)
            # Save after last iteration
            self._save_triggered_average(fig, neuron_name, output_folder, show_y_label_only_export, t,
                                         use_plotly, y_label=None, tight_layout=False)
            if to_show:
                plt.show()
            return fig, axes

        if fig_kwargs is None:
            fig_kwargs = {}
        if annotation_kwargs is None:
            annotation_kwargs = {}
        if color is None:
            color = self.get_color_from_data_type(trigger_type, is_mutant=is_mutant)
        df_subset = self.get_traces_single_neuron(neuron_name, trigger_type,
                                                  return_individual_traces=return_individual_traces, DEBUG=DEBUG)

        if df_subset.shape[1] == 0:
            logging.debug(f"Neuron name {neuron_name} not found, skipping")
            triggered_avg = None
        else:
            # Plot the triggered average for each neuron
            is_second_plot = False
            if not use_plotly:
                if ax is None:
                    fig_opt_trigger = self.get_fig_opt(**fig_kwargs)
                    fig, ax = plt.subplots(**fig_opt_trigger)
                else:
                    is_second_plot = True
            else:
                ax = None
                is_second_plot = fig is not None

            if z_score:
                df_subset = (df_subset - df_subset.mean()) / df_subset.std()
            df_subset = df_subset.T

            min_lines = min(min_lines, df_subset.shape[1])
            if DEBUG:
                print('df_subset.index', df_subset.index)
            ax, triggered_avg = plot_triggered_average_from_matrix_low_level(df_subset, 0, min_lines,
                                                                             show_individual_lines=show_individual_lines,
                                                                             is_second_plot=is_second_plot,
                                                                             ax=ax, fig=fig,
                                                                             color=color, label=neuron_name,
                                                                             show_horizontal_line=False,
                                                                             use_plotly=use_plotly, DEBUG=DEBUG)
            if use_plotly:
                fig = ax
                if ax is None:
                    raise ValueError("ax is None for plotly")
                ax = None
            if triggered_avg is None:
                logging.debug(f"Triggered average for {neuron_name} not valid, skipping")

        y_label = convert_channel_mode_to_axis_label(self.trace_opt)

        # Apply additional settings, even if the above failed
        if apply_changes_even_if_no_trace or triggered_avg is not None:
            behavior_shading_type = self._get_shading_from_trigger_name(trigger_type)
            if DEBUG:
                print(f"Behavior shading type: {behavior_shading_type}")
            if behavior_shading_type is not None:
                index_conversion = df_subset.columns
                try:
                    add_behavior_shading_to_plot(ind_preceding=20, index_conversion=index_conversion,
                                                 behavior_shading_type=behavior_shading_type,
                                                 ax=ax if not use_plotly else fig,
                                                 use_plotly=use_plotly, DEBUG=DEBUG)
                except IndexError:
                    print(f"Index error for {neuron_name} and {trigger_type}; skipping shading")

            if xlim is not None and ax is not None:
                ax.set_xlim(xlim)
            if ylim is not None and ax is not None:
                ax.set_ylim(ylim)
            if round_y_ticks and ax is not None:
                round_yticks(ax)

            # Defaults are opposite for each package
            if legend:
                if not use_plotly:
                    plt.legend()
            elif use_plotly and fig is not None:
                fig.update_layout(showlegend=False)

            # Update title and ticks
            if not use_plotly:
                if z_score:
                    ax.set_ylabel("Amplitude (z-scored)")
                else:
                    ax.set_ylabel(y_label)
                if show_title:
                    if title is None:
                        title = self.get_title_from_trigger_type(trigger_type)
                        plt.title(f"{neuron_name} (n={df_subset.shape[1]}) {title}")
                    else:
                        if include_neuron_in_title:
                            plt.title(f"{neuron_name} {title}")
                        else:
                            plt.title(title)
                else:
                    plt.title("")

                proj = self.all_projects[list(self.all_projects.keys())[0]]
                if show_x_ticks:
                    plt.xlabel(proj.x_label_for_plots)
                    height_factor_addition += 0.04
                else:
                    ax.set_xticks([])
                if not show_x_label:
                    ax.set_xlabel('')
                    height_factor_addition -= 0.04
                else:
                    ax.set_xlabel(self._get_xlabel_from_trigger_type(trigger_type))
                # These things affect the width
                if not show_y_ticks:
                    ax.set_yticks([])
                    width_factor_addition -= 0.04
                if not show_y_label:
                    ax.set_ylabel('')
                    width_factor_addition -= 0.04

            else:
                if fig is not None:
                    if show_y_label:
                        fig.update_yaxes(title=y_label)
                    else:
                        fig.update_layout(yaxis_title=None)
                    if show_x_label:
                        fig.update_xaxes(title="Time (s)")
                        height_factor_addition += 0.04
                    else:
                        fig.update_layout(xaxis_title=None)
                    # Add line annotations if there is a dynamic ttest
                    fig = add_annotation_lines(df_idx_range, neuron_name, fig, **annotation_kwargs)

            # Final saving
            if output_folder is not None:
                if fig_opt is None:
                    if i_figure == 0:  # Big
                        fig_opt = dict(width_factor=1.0 + width_factor_addition,
                                       height_factor=0.45 + height_factor_addition)
                    elif i_figure == 3:
                        fig_opt = dict(width_factor=0.5 + width_factor_addition,
                                       height_factor=0.20 + height_factor_addition)
                    elif i_figure > 3:
                        if 'rectified' in trigger_type:
                            fig_opt = dict(width_factor=0.35 + width_factor_addition,
                                           height_factor=0.1 + height_factor_addition)
                        else:
                            fig_opt = dict(width_factor=0.25 + width_factor_addition,
                                           height_factor=0.1 + height_factor_addition)
                    else:
                        raise NotImplementedError(f"i_figure={i_figure} not implemented")
                # Overwrite other options
                if height_factor is not None:
                    fig_opt['height_factor'] = height_factor
                if width_factor is not None:
                    fig_opt['width_factor'] = width_factor
                apply_figure_settings(fig, plotly_not_matplotlib=use_plotly, **fig_opt)

                self._save_triggered_average(fig, neuron_name, output_folder, show_y_label_only_export, trigger_type,
                                             use_plotly, y_label)

        return fig, ax

    def _save_triggered_average(self, fig, neuron_name, output_folder, show_y_label_only_export, trigger_type,
                                use_plotly, y_label, tight_layout=True):
        title = self.get_title_from_trigger_type(trigger_type)
        fname = title.replace(" ", "_").replace(",", "").lower()
        fname = os.path.join(output_folder, f'{neuron_name}-{fname}.png')
        if not use_plotly:
            if tight_layout:
                plt.tight_layout()
            plt.savefig(fname, transparent=True)
            plt.savefig(fname.replace(".png", ".svg"))
        else:
            fname = fname.replace(".png", "-plotly.png")
            fig.write_image(fname.replace(".png", ".svg"))
            fig.write_image(fname, scale=7)

            # Special option to change the returned figure only
            if show_y_label_only_export:
                fig.update_yaxes(title=y_label)

    def _get_shading_from_trigger_name(self, trigger_type):
        if 'rectified_rev' in trigger_type:
            behavior_shading_type = 'both'
        elif 'rectified_fwd' in trigger_type:
            behavior_shading_type = None
        elif 'rev' in trigger_type:
            behavior_shading_type = 'rev'
        elif 'fwd' in trigger_type:
            behavior_shading_type = 'fwd'
        else:
            behavior_shading_type = None
        return behavior_shading_type

    def get_traces_single_neuron(self, neuron_name, trigger_type, return_individual_traces=False, DEBUG=False):
        df = self.get_df_triggered_from_trigger_type(trigger_type, return_individual_traces=return_individual_traces)
        # Get the full names of all the neurons with this name
        # Names will be like '2022-11-23_worm9_BAGL' and we are checking for 'BAGL'
        neuron_names = [n for n in list(df.columns) if neuron_name in n and f'{neuron_name}_' not in n]
        if DEBUG:
            print(f"Found {len(neuron_names)} neurons with name {neuron_name}")
            print(f"Neuron names: {neuron_names}")
        df_subset = df.loc[:, neuron_names]
        return df_subset

    def get_valid_neuron_names(self, trigger_type, remove_nonided_neurons=True):
        """
        Returns a list of valid neuron names for the given trigger type, 
        Neuron names are the columns of the dataframe returned by get_df_triggered_from_trigger_type.

        Parameters
        ----------
        trigger_type

        Returns
        -------

        """
        df = self.get_df_triggered_from_trigger_type(trigger_type)
        
        if remove_nonided_neurons:
            return [n.split('_')[-1] for n in df.columns if 'neuron' not in n]
        else:
            return [n.split('_')[-1] for n in df.columns]

    def plot_triggered_average_multiple_neurons(self, neuron_list, trigger_type, color_list=None,
                                                output_folder=None, **kwargs):
        """
        Uses plot_triggered_average_single_neuron to plot multiple neurons on the same plot.

        Parameters
        ----------
        neuron_list
        trigger_type
        color_list
        title
        output_folder

        Returns
        -------

        """
        if color_list is None:
            # They will all be the same color
            color_list = [self.get_behavior_color_from_neuron_name(n) for n in neuron_list]

        fig, ax = None, None
        for i, (neuron, color) in enumerate(zip(neuron_list, color_list)):
            # Only set the output folder for the last neuron
            if i == len(neuron_list) - 1:
                this_output_folder = output_folder
            else:
                this_output_folder = None
            fig, ax = self.plot_triggered_average_single_neuron(neuron, trigger_type, output_folder=this_output_folder,
                                                                include_neuron_in_title=False, ax=ax, fig=fig,
                                                                fig_kwargs=dict(height_factor=2),
                                                                color=color, **kwargs)
        return fig, ax

    def plot_events_over_trace(self, neuron_name, trigger_type, dataset_name=None, output_foldername=None, **kwargs):
        """
        Plot the full trace with the event

        Loops through individual triggered average objects and plots the full trace with the event.
        """

        these_intermediates = self.intermediates_dict[trigger_type][0]
        for _dataset, triggered_average_class in these_intermediates.items():
            if dataset_name is not None and dataset_name != _dataset:
                continue

            fig, ax = plt.subplots(dpi=100)
            try:
                triggered_average_class.plot_events_over_trace(neuron_name, ax=ax, **kwargs)
                if 'rev' in trigger_type:
                    self.all_projects[_dataset].shade_axis_using_behavior()
                plt.title(f"{neuron_name} - {_dataset}")
                plt.show()
            except KeyError:
                # print(f"Neuron {neuron_name} not found in {name}; skipping")
                continue

    def ttest_before_and_after(self, neuron_name, trigger_type, gap=0):
        """Does a ttest on the traces before and after the event"""
        df_subset = self.get_traces_single_neuron(neuron_name, trigger_type)
        p_value_dict = calc_p_value_using_ttest_triggered_average(df_subset, gap)
        return p_value_dict

    def get_boxplot_before_and_after(self, neuron_name, trigger_type,
                                     summary_function=None,
                                     dynamic_window_center=False, dynamic_window_length=2,
                                     gap=0, same_size_window=False,
                                     return_individual_traces=False, DEBUG=False):
        """
        Preps data for a ttest or other comparison before and after the event, by calculating the median of the traces
        (collapsing the time dimension, and leaving the trial dimension)

        Parameters
        ----------
        neuron_name
        trigger_type
        summary_function - function to apply to the traces before and after the event; must accept the parameter axis=0
            default is np.nanmedian
        gap - time to skip around the event (both before and after)
        dynamic_window_center - if True, then the window of the 'after' values will be centered around the smoothed max value
        dynamic_window_length - if dynamic_window_center is True, then this is the length of the window (half on each side)
            Note: the center will be, at the max: len(trace) - (dynamic_window_length/2)
        same_size_window - if True, then the window before and after the event will be the same size
        return_individual_traces - if True, then do not pool trials within individuals, but treat each trial independently

        Returns
        -------

        """
        if summary_function is None:
            summary_function = np.nanmedian
        df_subset = self.get_traces_single_neuron(neuron_name, trigger_type,
                                                  return_individual_traces=return_individual_traces)
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=RuntimeWarning)
            means_before = summary_function(df_subset.loc[:-gap, :], axis=0)
            if same_size_window:
                # Get last index based on size of means_before
                len_before = len(means_before)
                i_of_0 = df_subset.index.get_loc(0)
                gap_idx = i_of_0 + gap + len_before
                idx_range = (i_of_0 - len_before, gap_idx)
                means_after = summary_function(df_subset.iloc[i_of_0:gap_idx, :], axis=0)
            elif dynamic_window_center:
                assert gap == 0, "Dynamic window center only works with gap=0"
                half_window = int(dynamic_window_length / 2)  # In list units
                idx_half_window = df_subset.index[half_window] - df_subset.index[0]  # In index units
                # Get the smoothed max value
                # Removing the last half of the desired window length (so the window can't be clipped)
                # And only look at positive values (this is the after part)
                # Note that the .loc[:-idx_half_window] gives a bug, thus the need for .iloc
                df_subset_after = df_subset.loc[0:].iloc[half_window:-half_window]
                if df_subset_after.empty:
                    raise ValueError("No data after the event")
                idx_max = df_subset_after.rolling(window=5, center=True).mean().mean(axis=1).idxmax()
                # Get the window around the smoothed max value
                idx_range = [idx_max - idx_half_window, idx_max + idx_half_window]
                # Closest index to the max value
                # idx_range = calc_closest_index(df_subset.index, idx_range)
                means_after = summary_function(df_subset.loc[idx_range[0]:idx_range[1], :], axis=0)
                if DEBUG:
                    print(neuron_name)
                    print(f"idx_max of smoothed time series: {idx_max}")
                    print(f"possible indices: {df_subset_after.index}")
                    print(f"df_subset_after: {df_subset_after}")
                    print(f"idx_half_window: {idx_half_window}")
                    # print(f"Means before: {means_before}; Means after: {means_after}")
            else:
                idx_range = (gap, df_subset.index[-1])  # Not actually used
                means_after = summary_function(df_subset.loc[gap:, :], axis=0)
        return means_before, means_after, idx_range

    def calc_significance_using_mode(self, neuron_names, trigger_type, significance_calculation_method=None, **kwargs):
        """
        Calculates the significance of a neuron using different modes, as implemented in the TriggeredAverage class.

        Returns dictionaries indexed by dataset name, then neuron name
        """

        # Get the triggered average objects (dict for all projects) for the trigger type
        triggered_average_dict = self.intermediates_dict[trigger_type][0]

        all_names_to_keep = {}
        all_all_p_values = {}
        all_all_effect_sizes = {}
        for _dataset, triggered_average_class in tqdm(triggered_average_dict.items()):
            if significance_calculation_method is not None:
                triggered_average_class.significance_calculation_method = significance_calculation_method
            names_to_keep, all_p_values, all_effect_sizes = (
                triggered_average_class.which_neurons_are_significant(neuron_names=neuron_names, verbose=0, **kwargs))
            all_names_to_keep[_dataset] = names_to_keep
            all_all_p_values[_dataset] = all_p_values
            # This is already a dataframe, so making it a nested dictionary doesn't work well
            all_all_effect_sizes[_dataset] = pd.concat(all_effect_sizes)

        return all_names_to_keep, all_all_p_values, all_all_effect_sizes


@dataclass
class PaperExampleTracePlotter(PaperColoredTracePlotter):
    """
    For plotting example traces, specifically a stack of 3 traces:
    - Raw
    - Global
    - Residual
    """

    project: ProjectData

    xlim: Optional[tuple] = (0, 150)
    ylim: Optional[tuple] = None

    trace_options: Optional[dict] = None

    def __post_init__(self):
        self.project.use_physical_time = True

        default_options = self.get_trace_opt()
        if self.trace_options is not None:
            default_options.update(self.trace_options)
        self.trace_options = default_options

        # Load the cache
        self.project.calc_all_paper_traces()

    def get_figure_opt(self):
        return dict(dpi=300, figsize=(10/3, 10/2), gridspec_kw={'wspace': 0.0, 'hspace': 0.0})

    def plot_triple_traces(self, neuron_name, title=False, legend=False, round_y_ticks=False,
                           output_foldername=None, combine_lr=False, **kwargs):
        """
        Plot the three traces (raw, global, residual) on the same plot.
        If output_foldername is not None, save the plot in that folder.

        Parameters
        ----------
        neuron_name
        output_foldername

        Returns
        -------

        """
        df_traces, df_traces_residual, df_traces_global = self._load_triple_traces(combine_lr=combine_lr)

        fig_opt = self.get_figure_opt()
        fig, axes = plt.subplots(**fig_opt, nrows=3, ncols=1)
        xlim = kwargs.get('xlim', self.xlim)
        ylim = kwargs.get('ylim', self.ylim)

        # Do all on one plot
        trace_dict = {'Raw': (df_traces[neuron_name], self.get_color_from_data_type('raw')),
                      'Global': (df_traces_global[neuron_name], self.get_color_from_data_type('global')),
                      'Residual': (df_traces_residual[neuron_name], self.get_color_from_data_type('residual'))}

        for i, (name, vals) in enumerate(trace_dict.items()):
            # Original trace
            ax = axes[i]
            ax.plot(vals[0], color=vals[1], label=name)
            if title and i == 0:
                ax.set_title(neuron_name)
            if legend:
                ax.legend(frameon=False)
            ax.set_ylabel(r"$\Delta R / R_{50}$")
            ax.set_xlim(xlim)
            ax.autoscale(enable=True, axis='y')  # Scale to the actually visible data (leaving x as set)
            if ylim is None:
                # If no given ylim, use the first trace's ylim
                ylim = ax.get_ylim()
            else:
                ax.set_ylim(ylim)

            if i < 2:
                ax.set_xticks([])
            else:
                ax.set_xlabel("Time (s)")
            self.project.shade_axis_using_behavior(ax)

        # Remove space between subplots
        plt.subplots_adjust(hspace=0)

        if round_y_ticks:
            for ax in axes:
                round_yticks(ax, **kwargs.get('round_yticks_kwargs', {}))

        width_factor = kwargs.get('width_factor', 0.25)
        height_factor = kwargs.get('height_factor', 0.3)
        apply_figure_settings(fig, width_factor=width_factor, height_factor=height_factor, plotly_not_matplotlib=False)

        if output_foldername:
            self._save_fig(neuron_name, output_foldername, trigger_type='combined')

        return fig, axes

    def _load_triple_traces(self, combine_lr=False):
        all_traces_dict = self.project.calc_all_paper_traces()

        if self.trace_options.get('channel_mode', 'dr_over_r_50') == 'dr_over_r_20':
            df_traces = all_traces_dict['paper_traces_r20'].copy()
        else:
            df_traces = all_traces_dict['paper_traces'].copy()
        df_traces_residual = all_traces_dict['paper_traces_residual'].copy()
        df_traces_global = all_traces_dict['paper_traces_global'].copy()
        if combine_lr:
            df_traces = combine_columns_with_suffix(df_traces)
            df_traces_residual = combine_columns_with_suffix(df_traces_residual)
            df_traces_global = combine_columns_with_suffix(df_traces_global)
        return df_traces, df_traces_residual, df_traces_global

    def _save_fig(self, neuron_name, output_foldername, trigger_type, plotly_fig=None):
        """
        Save the figure to the output foldername.

        Parameters
        ----------
        neuron_name
        output_foldername
        trigger_type

        Returns
        -------

        """
        if trigger_type == 'combined':
            fname = os.path.join(output_foldername, f'{neuron_name}-combined_traces.png')
        else:
            fname = os.path.join(output_foldername, f'{neuron_name}-{trigger_type}.png')
        if plotly_fig is None:
            plt.savefig(fname, transparent=True)
            plt.savefig(fname.replace(".png", ".svg"))
        else:
            plotly_fig.write_image(fname, scale=3)
            plotly_fig.write_image(fname.replace(".png", ".svg"))

    def get_trace_type_from_trace_options(self, trace_options):
        name_mapping = {'raw': 'raw', 'pca_global': 'global', 'pca': 'residual'}
        if trace_options is None:
            trace_type = 'raw'
        else:
            trace_type = name_mapping[trace_options.get('residual_mode', 'raw')]
        return trace_type

    def plot_single_trace(self, neuron_name, color_type=None, title=False, legend=False, round_y_ticks=False,
                          xlabels=True, ax=None, color=None, output_foldername=None, shading_kwargs=None,
                          trace_options=None,
                          use_plotly=False, **kwargs):
        """
        Plot a single trace.
        If output_foldername is not None, save the plot in that folder.

        Parameters
        ----------
        neuron_name
        trace_type
        color_type
        title
        legend
        round_y_ticks
        xlabels
        ax
        color
        output_foldername
        shading_kwargs
        kwargs

        Returns
        -------

        """
        if shading_kwargs is None:
            shading_kwargs = {}

        default_trace_options = self.trace_options.copy()
        if trace_options is not None:
            default_trace_options.update(trace_options)
        trace_options = default_trace_options
        trace_type = self.get_trace_type_from_trace_options(trace_options)

        df_traces = self.project.calc_default_traces(**trace_options)

        if neuron_name not in df_traces:
            # Try to combine L/R
            df_traces = combine_columns_with_suffix(df_traces)
        if neuron_name not in df_traces:
            raise ValueError(f"Neuron name {neuron_name} not found in traces")
        if color is None:
            if color_type is None:
                color_type = trace_type
            color = self.get_color_from_data_type(color_type, plotly_style=use_plotly)

        if not use_plotly:
            fig_opt = self.get_figure_opt()
            if ax is None:
                fig, ax = plt.subplots(**fig_opt)

            # Plot a single trace
            ax.plot(df_traces[neuron_name],
                    color=color,
                    label=trace_type)

            if title:
                ax.set_title(neuron_name)
            if legend:
                ax.legend(frameon=False)
            label = convert_channel_mode_to_axis_label(trace_options.get('channel_mode', 'dr_over_r_50'))
            ax.set_ylabel(label)

            if xlabels:
                ax.set_xlabel("Time (s)")
                height_factor = 0.14
            else:
                ax.set_xticks([])
                height_factor = 0.1
            ax.set_xlim(kwargs.get('xlim', self.xlim))
            ax.set_ylim(kwargs.get('ylim', self.ylim))
            if round_y_ticks:
                round_yticks(ax)
            self.project.shade_axis_using_behavior(ax, **shading_kwargs)
            fig = None

        else:
            # Same, but plotly
            if ax is not None:
                fig = ax
            else:
                fig = go.Figure()

            # Plot a single trace
            fig.add_traces([go.Scatter(x=df_traces.index, y=df_traces[neuron_name],
                                       mode='lines', line=dict(color=color),
                                       name=trace_type)])

            fig.update_layout(showlegend=legend, title=neuron_name if title else None)
            fig.update_yaxes(title=convert_channel_mode_to_axis_label(trace_options))

            if xlabels:
                fig.update_xaxes(title="Time (s)")
                height_factor = 0.14
            else:
                # fig.update_xaxes(tick_labels=[])
                height_factor = 0.1
            # ax.set_xlim(kwargs.get('xlim', self.xlim))
            # if round_y_ticks:
            #     round_yticks(ax)
            self.project.shade_axis_using_behavior(plotly_fig=fig, **shading_kwargs)
            fig.update_xaxes(range=self.xlim)
            y = df_traces[neuron_name][self.xlim[0]:self.xlim[1]]
            ylim = kwargs.get('ylim', self.ylim)
            if ylim is None:
                fig.update_yaxes(range=[y.min()-np.abs(0.1*y.min()), y.max()+np.abs(0.1*y.max())])
            else:
                fig.update_yaxes(range=ylim)

        width_factor = kwargs.get('width_factor', 0.25)
        height_factor = kwargs.get('height_factor', height_factor)
        apply_figure_settings(fig=fig, width_factor=width_factor, height_factor=height_factor,
                              plotly_not_matplotlib=use_plotly)

        if output_foldername:
            self._save_fig(neuron_name, output_foldername, trigger_type=trace_type, plotly_fig=fig)

        return fig, ax


def plot_ttests_from_triggered_average_classes(neuron_list: List[str],
                                               plotter_classes: List[PaperMultiDatasetTriggeredAverage],
                                               is_mutant_vec: List[bool],
                                               trigger_type: str,
                                               output_dir=None,
                                               ttest_kwargs=None,
                                               df_p_values=None,
                                               to_show=True, DEBUG=False, **kwargs):
    """
    Calculate the data for a t-test on the traces before and after the event.

    Parameters
    ----------
    neuron_list
    plotter_classes
    trigger_type
    gap
    same_size_window
    kwargs

    Returns
    -------

    """
    default_ttest_kwargs = dict(return_individual_traces=False, summary_function=None,
                                same_size_window=False, gap=0)
    if ttest_kwargs is None:
        ttest_kwargs = default_ttest_kwargs
    else:
        default_ttest_kwargs.update(ttest_kwargs)
        ttest_kwargs = default_ttest_kwargs
    # Calculate the basic data for the t-test
    all_boxplot_data_dfs = []
    all_idx_range = []
    # all_df_p_values = []
    for obj, is_mutant in zip(plotter_classes, is_mutant_vec):
        all_boxplot_data_dfs_single_type = []
        all_idx_range_single_type = []
        for neuron in neuron_list:
            means_before, means_after, idx_range = obj.get_boxplot_before_and_after(neuron, trigger_type,
                                                                                    **ttest_kwargs)
            # Sanity check: are any of the lists entirely nan?
            if np.all(np.isnan(means_before)) or np.all(np.isnan(means_after)):
                raise ValueError(f"Neuron {neuron} has all nan values for before or after the event:"
                                 f"means_before:{means_before}, means_after:{means_after}")
            df_before = pd.DataFrame(means_before, columns=['mean']).assign(before=True)
            df_after = pd.DataFrame(means_after, columns=['mean']).assign(before=False)
            df_both = pd.concat([df_before, df_after]).assign(neuron=neuron, is_mutant=is_mutant).assign(
                trigger_type=trigger_type)
            all_boxplot_data_dfs_single_type.append(df_both)
            df_idx_range = pd.DataFrame([idx_range], columns=['start', 'end']).assign(neuron=neuron,
                                                                                      is_mutant=is_mutant)
            all_idx_range_single_type.append(df_idx_range)
            if DEBUG:
                print(f"Neuron {neuron} was tested with indices: {idx_range}")
                print(f"Means before: {means_before}; Means after: {means_after}")
        # Process all neurons for this type, including multiple correction for p values
        # Only multiple correct for one type (mutant or not), not both
        df_boxplot_single_type = pd.concat(all_boxplot_data_dfs_single_type)
        all_boxplot_data_dfs.append(df_boxplot_single_type)
        all_idx_range.append(pd.concat(all_idx_range_single_type))
        # all_df_p_values.append(df_p_values_single_type)

    # Combine
    df_idx_range = pd.concat(all_idx_range)
    df_boxplot = pd.concat(all_boxplot_data_dfs)
    df_boxplot = _add_color_columns_to_df(df_boxplot, trigger_type=trigger_type)
    if df_p_values is None:
        df_p_values = _calc_p_value(df_boxplot, groupby_columns=['neuron', 'is_mutant_str'])  # .reset_index(level=1)
        df_p_values['p_value_corrected'] = multipletests(df_p_values['p_value'].values.squeeze(),
                                                         method='fdr_bh', alpha=0.05)[1]

    # Modify colors to use green for immobilized
    # This is not the only case where is it immobilized, but it is the only one we are plotting
    cmap = plotly_paper_color_discrete_map()
    is_immobilized = 'stimulus' in trigger_type.lower()
    if is_immobilized:
        cmap['Wild Type'] = cmap['immob']
    is_residual = 'residual' in trigger_type.lower()
    if is_residual:
        cmap['Wild Type'] = cmap['residual']

    df_p_values['is_immobilized'] = is_immobilized
    df_boxplot['is_immobilized'] = is_immobilized

    # Actually plot
    all_figs = {}
    for neuron_name in neuron_list:
        # Take the subset of the data for this neuron
        _df = df_boxplot[df_boxplot['neuron'] == neuron_name]

        # if DEBUG:
        #     print(_df)
        fig = plot_box_multi_axis(_df, x_columns_list=['is_mutant_str', 'before_str'], y_column='mean',
                                  color_names=['Wild Type', 'gcy-31;-35;-9'], cmap=cmap, DEBUG=False)

        precalculated_p_values = df_p_values.loc[neuron_name, 'p_value_corrected'].to_dict()
        add_p_value_annotation(fig, x_label='all', show_ns=True, show_only_stars=True, separate_boxplot_fig=False,
                               precalculated_p_values=precalculated_p_values,
                               height_mode='top_of_data', has_multicategory_index=True, DEBUG=False)

        fig.update_layout(showlegend=False, yaxis_title=None, xaxis_title=None)
        # Modify offsetgroup to have only 2 types (rev and fwd), not one for each legend entry
        apply_figure_settings(fig, height_factor=0.1, width_factor=0.25)
        if to_show:
            fig.show()
        all_figs[neuron_name] = fig

        if output_dir is not None:
            fname = os.path.join(output_dir, f'{neuron_name}-{trigger_type}_triggered_average_boxplots.png')
            fig.write_image(fname, scale=3)
            fname = fname.replace('.png', '.svg')
            fig.write_image(fname)

    return all_figs, df_boxplot, df_p_values, df_idx_range


def plot_triggered_averages_from_triggered_average_classes(neuron_list: List[str],
                                                           plotter_classes: List[PaperMultiDatasetTriggeredAverage],
                                                           is_mutant_vec: List[bool],
                                                           trigger_type: str,
                                                           df_idx_range: pd.DataFrame = None,
                                                           output_dir=None,
                                                           to_show=False,
                                                           DEBUG=False,
                                                           **kwargs):
        """
        Plot the triggered averages for a list of neurons.

        Parameters
        ----------
        neuron_list
        plotter_classes
        trigger_type
        output_dir
        to_show
        kwargs

        Returns
        -------

        """
        all_figs = {}
        for neuron_name in neuron_list:
            fig, ax = None, None
            show_x_label = 'URX' in neuron_name
            for obj, is_mutant in zip(plotter_classes, is_mutant_vec):
                fig, ax = obj.plot_triggered_average_single_neuron(neuron_name, trigger_type, is_mutant=is_mutant,
                                                                   fig=fig, ax=ax, show_x_label=show_x_label,
                                                                   output_folder=output_dir, df_idx_range=df_idx_range,
                                                                   **kwargs)

            all_figs[neuron_name] = fig
            if to_show:
                fig.show()

        return all_figs


def _add_color_columns_to_df(df_boxplot, neuron_name=None, trigger_type='rev'):
    # Make a new column with color information based on reversal
    if neuron_name is not None:
        df = df_boxplot[df_boxplot['neuron'] == neuron_name].copy()
    else:
        df = df_boxplot.copy()

    if 'rev' in trigger_type.lower():
        before_str, after_str = 'Fwd', 'Rev'
    elif 'fwd' in trigger_type.lower():
        before_str, after_str = 'Rev', 'Fwd'
    else:
        before_str, after_str = 'Before', 'After'
    df['before_str'] = [before_str if val else after_str for val in df['before']]

    df['color'] = ''
    df.loc[np.logical_and(df['before'], df['is_mutant']), 'color'] = f'{before_str}-Mutant'
    df.loc[np.logical_and(~df['before'], df['is_mutant']), 'color'] = f'{after_str}-Mutant'
    df.loc[np.logical_and(df['before'], ~df['is_mutant']), 'color'] = f'{before_str}-WT'
    df.loc[np.logical_and(~df['before'], ~df['is_mutant']), 'color'] = f'{after_str}-WT'
    df['is_mutant_str'] = 'gcy-31;-35;-9'
    df.loc[~df['is_mutant'], 'is_mutant_str'] = 'Wild Type'

    # Rename columns to the display names
    df['Data Type'] = df['color']
    df['dR/R50'] = df['mean']

    return df


def _calc_p_value(df, groupby_columns=None):
    # func = lambda x: stats.ttest_1samp(x, 0)[1]
    if groupby_columns is None:
        groupby_columns = ['neuron', 'trigger_type']
    func = lambda x: stats.ttest_rel(x[x['before']]['mean'], x[~x['before']]['mean'])[1]
    df_groupby = df.dropna().groupby(groupby_columns)
    df_pvalue = df_groupby.apply(func).to_frame()
    df_pvalue.columns = ['p_value']
    return df_pvalue
