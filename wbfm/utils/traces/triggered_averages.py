import logging
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Callable, Dict, Union
import matplotlib
import plotly.express as px
import numpy as np
import pandas as pd
import scipy
import sklearn
from backports.cached_property import cached_property
from matplotlib import pyplot as plt, cm
from matplotlib.colors import LinearSegmentedColormap
from methodtools import lru_cache
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster import hierarchy
from scipy.stats import permutation_test
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from tqdm.auto import tqdm

from wbfm.utils.external.custom_errors import NeedsAnnotatedNeuronError, NoBehaviorAnnotationsError
from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes, shade_using_behavior, add_behavior_shading_to_plot
from wbfm.utils.external.utils_pandas import get_contiguous_blocks_from_column, remove_short_state_changes, \
    split_flattened_index, count_unique_datasets_from_flattened_index, flatten_multiindex_columns, flatten_nested_dict, \
    calc_surpyval_durations_and_censoring, combine_columns_with_suffix, extend_binary_vector
from wbfm.utils.external.utils_zeta_statistics import calculate_zeta_cumsum, jitter_indices, calculate_p_value_from_zeta
from wbfm.utils.external.utils_matplotlib import paired_boxplot_from_dataframes
from wbfm.utils.external.utils_jupyter import check_plotly_rendering
from wbfm.utils.general.utils_paper import apply_figure_settings
from wbfm.utils.general.hardcoded_paths import neurons_with_confident_ids
from wbfm.utils.general.high_performance_pandas import get_names_from_df
from wbfm.utils.visualization.filtering_traces import filter_gaussian_moving_average
from wbfm.utils.visualization.utils_plot_traces import plot_with_shading, plot_with_shading_plotly


def plot_triggered_average_from_matrix_low_level(triggered_avg_matrix, ind_preceding, min_lines=0,
                                                 show_individual_lines=False, is_second_plot=False,
                                                 ax=None, xlim=None, fig=None,
                                                 show_horizontal_line=True, show_vertical_line=True,
                                                 z_score=False, show_shading=True, use_plotly=False, DEBUG=False,
                                                 **kwargs):
    """
    Plot a triggered average from a matrix of traces

    Parameters
    ----------
    triggered_avg_matrix: shape should be (n_lines, n_timepoints); if a dataframe, then time should be the column names
    ind_preceding: index of the timepoint of the event
    min_lines
    show_individual_lines
    is_second_plot: if true, doesn't plot the reference lines again or set the labels
    ax
    xlim
    z_score
    show_shading
    show_legend
    kwargs

    Returns
    -------

    """
    # Get y values
    raw_trace_mean, triggered_avg, triggered_lower_std, triggered_upper_std, xmax, is_valid = \
        TriggeredAverageIndices.prep_triggered_average_for_plotting(triggered_avg_matrix, min_lines=min_lines,
                                                                    z_score=z_score)

    if DEBUG:
        print(f"triggered_avg: {triggered_avg.iloc[:5]}")

    # Get values that may be used for informational lines
    if isinstance(triggered_avg, pd.Series):
        x_for_vertical_line = 0  # Assume the series has been properly indexed
    else:
        x_for_vertical_line = ind_preceding

    if not is_valid:
        logging.warning("Found invalid neuron (empty triggered average)")
        return ax, None

    if not use_plotly:
        # Plot
        if show_shading:
            ax, lower_shading, upper_shading = plot_with_shading(triggered_avg, triggered_lower_std, xmax, ax=ax, **kwargs,
                                                                 std_vals_upper=triggered_upper_std)#, x=x_vals)
        else:
            ax.plot(triggered_avg[:xmax], **kwargs)
        if show_individual_lines:
            for i, trace in triggered_avg_matrix.iterrows():
                # ax.plot(x_vals[:xmax], trace[:xmax], 'black', alpha=3.0 / (triggered_avg_matrix.shape[0] + 10.0))
                ax.plot(trace.loc[:xmax], 'black', alpha=3.0 / (triggered_avg_matrix.shape[0] + 10.0))
        if not is_second_plot:
            ax.set_ylabel("Activity")
            # Set y limits because sometimes individual traces are very large
            if show_shading:
                ax.set_ylim(np.nanmin(lower_shading), np.nanmax(upper_shading))
            else:
                ax.set_ylim(np.nanmin(triggered_avg), np.nanmax(triggered_avg))
            # Reference points
            if show_horizontal_line:
                ax.axhline(raw_trace_mean, c='black', ls='--')
            if show_vertical_line:
                ax.axvline(x=x_for_vertical_line, color='black', ls='--')
        else:
            ax.autoscale()
        if xlim is not None and ax is not None:
            ax.set_xlim(xlim)
    else:
        # fig = kwargs.get('fig', None)
        # kwargs.pop('fig', None)
        fig, _, _ = plot_with_shading_plotly(triggered_avg, triggered_lower_std, std_vals_upper=triggered_upper_std,
                                             fig=fig, is_second_plot=(fig is not None), **kwargs)
        if show_vertical_line:
            fig.add_shape(type="line", x0=x_for_vertical_line, x1=x_for_vertical_line,
                          y0=0, y1=1, line=dict(color="black", dash="dash"), yref='paper')
        if show_individual_lines:
            for i, trace in triggered_avg_matrix.iterrows():
                fig.add_trace(px.line(trace.loc[:xmax], color_discrete_sequence=['gray']).data[0])
        ax = fig

    return ax, triggered_avg


def calculate_and_filter_triggered_average_indices(binary_state, beh_vec=None, ind_preceding=0, ind_delay=0,
                                                   dict_of_events_to_keep=None,
                                                   min_duration=0, max_duration=None,
                                                   fixed_num_points_after_event=None, max_num_points_after_event=None,
                                                   DEBUG=False):
    """
    Calculates the indices of the triggered averages, and filters them based on a set of criteria

    Specifically:
    1. Minimum duration of the event
    2. Maximum duration of the event
    3. Whether the event is at the edge of the trace
    4. Whether the event starts with a misannotation (Implying the transition wasn't real)
    5. Whether the event is in the dict_of_events_to_keep

    Parameters
    ----------
    binary_state
    beh_vec
    ind_preceding
    ind_delay
    dict_of_events_to_keep
    min_duration
    max_duration
    fixed_num_points_after_event
    max_num_points_after_event
    DEBUG

    Returns
    -------

    """
    if DEBUG:
        print("Arguments: ", locals())
    all_starts, all_ends = get_contiguous_blocks_from_column(binary_state,
                                                             already_boolean=True, skip_boolean_check=True)
    if DEBUG:
        print("Original starts: ", all_starts)
        print("Original ends: ", all_ends)
    if ind_delay > 0:
        # Some of the starts will now be after the ends, but they will be removed by the is_too_short check
        # Also: don't want to mess with an event at exactly 0
        all_starts = [min(i + ind_delay, len(binary_state)-1) if i > 0 else i for i in all_starts]
    if fixed_num_points_after_event is not None:
        # Add this offset to the ends, but make sure they don't go past the next start
        all_ends = []
        for i in range(len(all_starts)):
            i_start = all_starts[i]
            if i == len(all_starts) - 1:
                i_next_start = len(binary_state)
            else:
                i_next_start = all_starts[i + 1]
            i_new_end = min(i_start + fixed_num_points_after_event, i_next_start - 1)
            all_ends.append(i_new_end)
    if max_num_points_after_event is not None:
        # If the next end is too far away, then cut it off
        new_ends = []
        for start, end in zip(all_starts, all_ends):
            if end - start > max_num_points_after_event:
                end = start + max_num_points_after_event
            new_ends.append(end)
        all_ends = new_ends
    if DEBUG:
        print("All starts after fixed_num_points_after_event and max_num_points_after_event processing: ", all_starts)
        print("All ends after fixed_num_points_after_event and max_num_points_after_event processing: ", all_ends)
    # Build all validity checks as a list of callables
    is_too_short = lambda start, end: end - start < min_duration
    is_too_long = lambda start, end: (max_duration is not None) and (end - start > max_duration)
    is_at_edge = lambda start, end: start == 0
    starts_with_misannotation = lambda start, end: (beh_vec is not None) and beh_vec[start - 1] == BehaviorCodes.UNKNOWN
    not_in_dict = lambda start, end: (dict_of_events_to_keep is not None) and \
                                     (dict_of_events_to_keep.get(start, 0) == 0)
    validity_checks = [is_too_short, is_too_long, is_at_edge, starts_with_misannotation, not_in_dict]
    # Build actual indices
    if DEBUG:
        print(f"Names of validity checks: "
              f"[is_too_short, is_too_long, is_at_edge, starts_with_misannotation, not_in_dict]")
    all_ind = build_ind_matrix_from_starts_and_ends(all_starts, all_ends,
                                                    ind_preceding, validity_checks, DEBUG)
    return all_ind


@dataclass
class TriggeredAverageIndices:
    """
    Class for keeping track of all the settings related to a general triggered average
    By default triggered average time points are calculated from the binarized behavioral annotation
        Optionally, a continuous variable can be used
        This is designed to work on a wrapped phase variable, so that a binary variable can be generated by thresholding
    Has all postprocessing functions, so that analysis is consistent when calculated for multiple traces

    Fundamentally, contains the time series of a BehaviorCodes behaviors in behavioral_annotation, and the state to
    trigger on in behavioral_state OR a continuous variable in behavioral_annotation and a threshold in
    behavioral_annotation_threshold

    For main functionality, see:
        triggered_average_indices
        calc_triggered_average_matrix

    For loading with a custom behavioral time series, there are two options:
        1. See BehaviorCodes for the possible states, and use BehaviorCodes.load_using_dict_mapping if loading from a
            vector of integers or strings
        2. Just pass a vector of 0s and 1s, and use BehaviorCodes.CUSTOM as the behavioral_state
            (this can only handle a single annotated state)

    Note:
        The traces themselves are not stored here (see FullDatasetTriggeredAverages for that)
        Does not work well with multiple datasets (edges are not handled at all)
    """
    # Initial calculation of binarized indices
    behavioral_annotation: pd.Series

    behavioral_state: BehaviorCodes = BehaviorCodes.REV  # Note: not used if behavioral_annotation_is_continuous is True

    # Alternative: continuous behavioral annotations
    behavioral_annotation_is_continuous: bool = False
    behavioral_annotation_threshold: float = 0.0  # Not used if behavioral_annotation_is_continuous is False

    # Alternate way to define the start point of each time series
    ind_preceding: int = 10
    ind_delay: int = 0  # Delay the start of the triggered average by this amount of volumes
    trigger_on_downshift: bool = False  # Trigger to the offset instead of the onset

    # Alternate ways to define the end point of each time series
    allowed_succeeding_state: BehaviorCodes = None  # Allow continuation into this state
    fixed_num_points_after_event: int = None  # If not None, then use this number of points after the event (regardless of the end of the state)
    max_num_points_after_event: int = None  # If not None, then cut off the event if it is too long

    # Options for randomly shuffling the events
    max_random_shuffle_offset: int = 0  # If not 0, then shuffle the events by up to this amount (randomly, but all the same offset)

    # Options for filtering the events
    min_duration: int = 0
    max_duration: int = None
    gap_size_to_remove: int = None
    behavioral_annotation_for_rectification: pd.Series = None
    only_allow_events_during_state: int = None  # If not None, only allow events that start during this state

    # Postprocessing the trace matrix (per trace)
    trace_len: int = None
    to_nan_points_of_state_before_point: bool = True
    min_lines: int = 2
    include_censored_data: bool = True  # To include events whose termination is after the end of the data
    dict_of_events_to_keep: dict = None
    mean_subtract: bool = False
    z_score: bool = False
    normalize_amplitude_at_onset: bool = False

    cached_ind: list = field(default=None, init=False, repr=False)
    cache_is_valid: bool = False

    DEBUG: bool = False

    def set_min_duration(self, value):
        self.min_duration = value
        self.cache_is_valid = False

    def set_max_duration(self, value):
        self.max_duration = value
        self.cache_is_valid = False

    def set_beh_vec(self, value):
        self.behavioral_annotation = pd.Series(value)
        self.cache_is_valid = False

    def set_dict_of_events_to_keep(self, value):
        self.dict_of_events_to_keep = value
        self.cache_is_valid = False

    def set_ind_preceding(self, value):
        self.ind_preceding = value
        self.cache_is_valid = False

    def set_ind_delay(self, value):
        self.ind_delay = value
        self.cache_is_valid = False

    def set_fixed_num_points_after_event(self, value):
        self.fixed_num_points_after_event = value
        self.cache_is_valid = False

    def set_max_num_points_after_event(self, value):
        self.max_num_points_after_event = value
        self.cache_is_valid = False
    
    def set_cached_ind(self, value):
        self.cached_ind = value
        self.cache_is_valid = True

    def __post_init__(self):
        # Check the types of the behavioral annotation and state
        if not self.behavioral_annotation_is_continuous and\
                not isinstance(self.behavioral_annotation.iat[0], Enum):
            # Attempt to cast using the 'custom' BehaviorCodes, but only if there is only one nontrivial behavior
            self.behavioral_annotation = pd.Series(self.behavioral_annotation)
            behavior_values = self.behavioral_annotation.unique()
            self.behavioral_state = BehaviorCodes.CUSTOM
            behavior_mapping = {k: BehaviorCodes.NOT_ANNOTATED for k in [-1, 0, '-1', '0', np.nan]}
            # Check if there is an additional behavior to be mapped
            unmapped_behavior = set(behavior_values) - set(behavior_mapping.keys())
            if len(unmapped_behavior) == 1:
                behavior_mapping[unmapped_behavior.pop()] = self.behavioral_state
            else:
                raise ValueError(f"Could not map behavioral annotation to Custom BehaviorCodes. "
                                 f"Unique values: {behavior_values}")
            # Build the Series of BehaviorCodes
            self.behavioral_annotation = self.behavioral_annotation.map(behavior_mapping)

        # Build a dict_of_events_to_keep if only_allow_events_during_state is not None
        state = self.only_allow_events_during_state
        if state is not None:
            if not self.behavioral_annotation_is_continuous:
                logging.warning("Passed only_allow_events_during_state, but behavioral_annotation_is_continuous is False. "
                                "This may give strange results if you are triggering to that state, or an adjacent state")
            if self.dict_of_events_to_keep is not None:
                logging.warning("Passed custom dict_of_events_to_keep, but also passed only_allow_events_during_state. "
                                "Using only_allow_events_during_state to overwrite dict_of_events_to_keep")
            if self.behavioral_annotation_for_rectification is None:
                raise ValueError("Must pass behavioral_annotation_for_rectification if only_allow_events_during_state is not None")
            # Note that we use 'in' not '==' because the beh vector is a flag enum
            beh = self.behavioral_annotation_for_rectification
            self.dict_of_events_to_keep = {i: state in beh.iat[i] for i in range(len(self.behavioral_annotation))}

        if self.ind_delay > 0:
            if self.to_nan_points_of_state_before_point:
                logging.warning("ind_delay is set, but to_nan_points_of_state_before_point is also True. "
                                "Currently the two are incompatible, and the latter will be ignored")
                self.to_nan_points_of_state_before_point = False

        if self.num_events == 0:
            logging.warning(f"No instances of state {self.behavioral_state} found in behavioral annotation!!")

    @property
    def binary_state(self) -> pd.Series:
        if self.behavioral_annotation_is_continuous:
            binary_state = self.behavioral_annotation > self.behavioral_annotation_threshold
        else:
            binary_state = BehaviorCodes.vector_equality(self.behavioral_annotation, self.behavioral_state)
            if self.allowed_succeeding_state is not None:
                # Extend the ends of the state to include the allowed succeeding states
                # But do not modify the starts
                alt_binary_state = BehaviorCodes.vector_equality(self.behavioral_annotation, self.allowed_succeeding_state)
                binary_state = extend_binary_vector(binary_state, alt_binary_state)

        if self.gap_size_to_remove is not None:
            binary_state = remove_short_state_changes(binary_state, self.gap_size_to_remove)

        # Randomly shuffle the events
        if self.max_random_shuffle_offset > 0:
            _state = np.roll(binary_state, self.random_shuffle_offset)
            binary_state = pd.Series(_state, index=binary_state.index)

        return binary_state

    @cached_property
    def random_shuffle_offset(self) -> int:
        return np.random.randint(0, self.max_random_shuffle_offset)

    @property
    def cleaned_binary_state(self) -> pd.Series:
        # I can make this cached, but then it has to be cleared if the binary_state changes
        if self.trace_len is not None:
            return self.binary_state.iloc[:self.trace_len]
        else:
            return self.binary_state

    def triggered_average_indices(self, dict_of_events_to_keep=None, DEBUG=False) -> list:
        """
        Calculates triggered average indices based on a binary state vector saved in this class

        If ind_preceding > 0, then a very early event will lead to negative indices
        Thus in later steps, the trace should be padded with nan at the end to avoid wrapping

        Parameters
        ----------
        dict_of_events_to_keep: Optional dict determining a subset of indices to keep. Key=state starts, value=0 or 1
            Example:
            idx_onsets = [15, 66, 114, 130]
            dict_of_ind_to_keep = {15: 0, 66: 1, 114: 0}

            Note that not all starts need to be in dict_of_ind_to_keep; missing entries are dropped by default

        Returns
        -------

        """
        if dict_of_events_to_keep is None:
            dict_of_events_to_keep = self.dict_of_events_to_keep
        else:
            self.dict_of_events_to_keep = dict_of_events_to_keep
        binary_state = self.cleaned_binary_state.copy()
        # Turn into time series
        opt = dict(min_duration=self.min_duration,
                   max_duration=self.max_duration,
                   beh_vec=self.behavioral_annotation.to_numpy(),
                   dict_of_events_to_keep=dict_of_events_to_keep,
                   ind_preceding=self.ind_preceding,
                   ind_delay=self.ind_delay,
                   fixed_num_points_after_event=self.fixed_num_points_after_event,
                   max_num_points_after_event=self.max_num_points_after_event,
                   DEBUG=DEBUG)

        if self.trigger_on_downshift:
            binary_state = ~binary_state

        # I can't use an LRU cache here because the arguments are not hashable, so just create a single-value cache that invalidates if any of these args change

        if self.cache_is_valid:
            if self.cached_ind is not None:
                all_ind = self.cached_ind
            else:
                all_ind = calculate_and_filter_triggered_average_indices(binary_state, **opt)
                self.cached_ind = all_ind
        else:
            all_ind = calculate_and_filter_triggered_average_indices(binary_state, **opt)
            self.cached_ind = all_ind
        return all_ind

    def calc_triggered_average_matrix(self, raw_trace: pd.Series, custom_ind: List[np.ndarray]=None,
                                      nan_times_with_too_few=False, max_len=None, DEBUG=False,
                                      **ind_kwargs) -> Optional[pd.DataFrame]:
        """
        Uses triggered_average_indices to extract a matrix of traces at each index, with nan padding to equalize the
        lengths of the traces

        If there are no valid indices, returns None

        Parameters
        ----------
        raw_trace
        custom_ind: instead of using self.triggered_average_indices. If not None, ind_kwargs are not used
        nan_times_with_too_few
        max_len: Cut off matrix at a time point. Usually if there aren't enough data points that far
        ind_kwargs

        Returns
        -------

        """
        # If there are multiple neurons with the same name, then sometimes raw_trace can be a matrix, which is a problem
        if raw_trace.ndim > 1:
            raise ValueError("raw_trace must be a vector, not a matrix... "
                             "This is probably caused by a duplicate name in the neuron manual annotation")

        if custom_ind is None:
            all_ind = self.triggered_average_indices(**ind_kwargs)
        else:
            all_ind = custom_ind
        if len(all_ind) == 0:
            return None
        if max_len is None:
            max_len_subset = max(map(len, all_ind))
        else:
            max_len_subset = max_len

        # Preprocessing type 1: change amplitudes
        if self.mean_subtract or self.z_score:
            raw_trace -= raw_trace.mean()
        if self.z_score:
            raw_trace /= raw_trace.std()

        # Pad with nan in case there are negative indices, but only the end
        trace = np.pad(raw_trace, max_len_subset, mode='constant', constant_values=(np.nan, np.nan))[max_len_subset:]
        triggered_avg_matrix = np.zeros((len(all_ind), max_len_subset))
        triggered_avg_matrix[:] = np.nan
        # Save either entire traces, or traces up to a point
        for i, ind in enumerate(all_ind):
            if max_len is not None:
                ind = ind.copy()[:max_len]
            triggered_avg_matrix[i, np.arange(len(ind))] = trace[ind]

        # Postprocessing type 1: change amplitudes
        if self.normalize_amplitude_at_onset:
            # Normalize to the amplitude at the index of the event
            triggered_avg_matrix = triggered_avg_matrix - triggered_avg_matrix[:, [self.ind_preceding]]

        # Postprocessing type 2: remove points
        if self.to_nan_points_of_state_before_point:
            triggered_avg_matrix = self.nan_points_of_state_before_point(triggered_avg_matrix, all_ind, DEBUG=DEBUG)
        if nan_times_with_too_few:
            num_lines_at_each_time = np.sum(~np.isnan(triggered_avg_matrix), axis=0)
            times_to_remove = num_lines_at_each_time < self.min_lines
            triggered_avg_matrix[:, times_to_remove] = np.nan

        # If the trace has a nontrivial index, then use that for the columns of this matrix
        raw_index = raw_trace.index
        # However, the event itself should be at t=0, and times previous to that should be negative
        index = raw_index - raw_index[self.ind_preceding]
        index = index[:triggered_avg_matrix.shape[1]]
        triggered_avg_matrix = pd.DataFrame(triggered_avg_matrix, columns=index)

        return triggered_avg_matrix

    def nan_points_of_state_before_point(self, triggered_average_mat, list_of_triggered_ind,
                                         DEBUG=False):
        """
        Checks points up to a certain level, and nans them if they are invalid. Only checks up to a certain threshold

        Parameters
        ----------
        triggered_average_mat

        Returns
        -------

        """
        if self.behavioral_annotation_is_continuous:
            beh_annotations = self.cleaned_binary_state
        else:
            beh_annotations = self.behavioral_annotation.to_numpy()
        invalid_states = self._get_invalid_states_for_prior_index_removal()
        if DEBUG:
            print(f"Invalid states: {invalid_states}")
        for i_trace in range(len(list_of_triggered_ind)):
            these_ind = list_of_triggered_ind[i_trace]
            if DEBUG:
                print(f"Trace {i_trace} indices: {these_ind}")
            for i_local, i_global in enumerate(these_ind):
                if i_global < 0:
                    continue
                if i_local >= self.ind_preceding:
                    if DEBUG:
                        print(f"Breaking at time {i_global} because of ind_preceding {self.ind_preceding}")
                    break
                if self.behavioral_annotation_is_continuous:
                    this_beh = beh_annotations.iat[i_global]
                else:  # It's numpy
                    this_beh = beh_annotations[i_global]
                if this_beh in invalid_states:
                    # Remove all points before this
                    for i_to_remove in range(i_local + 1):
                        triggered_average_mat[i_trace, i_to_remove] = np.nan
                        if DEBUG:
                            print(f"Removing point {i_to_remove} from trace {i_trace} because "
                                  f"of state {beh_annotations[i_global]} at time {i_global}")
        return triggered_average_mat

    def _get_invalid_states_for_prior_index_removal(self):
        if self.trigger_on_downshift:
            # This flips all of the states that should be removed
            if self.behavioral_annotation_is_continuous:
                invalid_states = {False}
            else:
                # Get all states that are not the one we are triggering on
                all_present_states = BehaviorCodes.convert_to_simple_states_vector(pd.Series(self.behavioral_annotation.unique()))
                invalid_states = set(all_present_states) - {self.behavioral_state}
        else:
            if self.behavioral_annotation_is_continuous:
                invalid_states = {True}
            else:
                invalid_states = {self.behavioral_state, BehaviorCodes.UNKNOWN}

        return invalid_states

    @staticmethod
    def prep_triggered_average_for_plotting(triggered_avg_matrix, min_lines, shorten_to_last_valid=True,
                                            z_score=False):
        triggered_avg, triggered_lower_std, triggered_upper_std, triggered_avg_counts = \
            TriggeredAverageIndices.calc_triggered_average_stats(triggered_avg_matrix, z_score=z_score)
        # Remove points where there are too few lines contributing
        to_remove = triggered_avg_counts < min_lines
        triggered_avg[to_remove] = np.nan
        triggered_lower_std[to_remove] = np.nan
        triggered_upper_std[to_remove] = np.nan
        xmax = pd.Series(triggered_avg).last_valid_index()
        if shorten_to_last_valid:
            # Helps with plotting individual lines, but will likely produce traces of different lengths
            triggered_avg = triggered_avg.loc[:xmax]
            triggered_lower_std = triggered_lower_std.loc[:xmax]
            triggered_upper_std = triggered_upper_std.loc[:xmax]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            raw_trace_mean = np.nanmean(triggered_avg)
        is_valid = len(triggered_avg) > 0 and np.count_nonzero(~np.isnan(triggered_avg)) > 0
        return raw_trace_mean, triggered_avg, triggered_lower_std, triggered_upper_std, xmax, is_valid

    @staticmethod
    def calc_triggered_average_stats(triggered_avg_matrix, z_score=False):
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=RuntimeWarning)
            if z_score:
                triggered_avg_matrix = (triggered_avg_matrix - np.nanmean(triggered_avg_matrix)) \
                                       / np.nanstd(triggered_avg_matrix)
            triggered_avg = np.nanmedian(triggered_avg_matrix, axis=0)
            # Use quantiles that would be same as std if the distribution were normal
            # https://tidsskriftet.no/en/2020/06/medisin-og-tall/mean-and-standard-deviation-or-median-and-quartiles
            # triggered_upper_std = np.nanquantile(triggered_avg_matrix, 0.84, axis=0)
            # triggered_lower_std = np.nanquantile(triggered_avg_matrix, 0.16, axis=0)
            std = np.nanstd(triggered_avg_matrix, axis=0)
            triggered_upper_std = triggered_avg_matrix.mean(axis=0) + std
            triggered_lower_std = triggered_avg_matrix.mean(axis=0) - std
            triggered_avg_counts = np.nansum(~np.isnan(triggered_avg_matrix), axis=0)
        if isinstance(triggered_avg_matrix, pd.DataFrame):
            # Preserve the index, which are actually the columns of the matrix
            triggered_avg = pd.Series(triggered_avg, index=triggered_avg_matrix.columns)
            triggered_lower_std = pd.Series(triggered_lower_std, index=triggered_avg_matrix.columns)
            triggered_upper_std = pd.Series(triggered_upper_std, index=triggered_avg_matrix.columns)
            triggered_avg_counts = pd.Series(triggered_avg_counts, index=triggered_avg_matrix.columns)

        return triggered_avg, triggered_lower_std, triggered_upper_std, triggered_avg_counts

    def calc_significant_points_from_triggered_matrix(self, triggered_avg_matrix):
        """
        Calculates the time points that are (based on the std) "significantly" different from a flat line

        Designed to be used to remove uninteresting traces from triggered average grid plots

        Parameters
        ----------
        triggered_avg_matrix

        Returns
        -------

        """
        raw_trace_mean, triggered_avg, triggered_lower_std, triggered_upper_std, xmax, is_valid = \
            self.prep_triggered_average_for_plotting(triggered_avg_matrix, self.min_lines)
        if not is_valid:
            return []
        x_significant = np.where(np.logical_or(triggered_lower_std > raw_trace_mean, triggered_upper_std < raw_trace_mean))[0]
        return x_significant

    def calc_p_value_using_zeta(self, trace, num_baseline_lines=100, DEBUG=False) -> (
            Tuple[float, Tuple[float, np.ndarray]]):
        """
        See utils_zeta_statistics. Following:
        https://elifesciences.org/articles/71969#

        Parameters
        ----------
        trace
        num_baseline_lines

        Returns
        -------

        """
        # Original triggered average matrix
        triggered_average_indices = self.triggered_average_indices()
        # Set max number of time points based on number of lines present
        # In other words, find the max point in time when there are still enough lines
        if self.min_lines > 0:
            all_lens = np.array(list(map(len, triggered_average_indices)))
            ind_lens_enough = np.argsort(all_lens)[:-self.min_lines]
            max_matrix_length = np.max(all_lens[ind_lens_enough])
        else:
            max_matrix_length = None
        mat = self.calc_triggered_average_matrix(trace, custom_ind=triggered_average_indices,
                                                 max_len=max_matrix_length)
        zeta_line_dat = calculate_zeta_cumsum(mat, DEBUG=DEBUG)

        if DEBUG:
            print(max_matrix_length)

            plt.figure(dpi=100)
            self.plot_triggered_average_from_matrix(mat, show_individual_lines=True)
            plt.title("Triggered average")

            plt.figure(dpi=100)
            plt.plot(np.sum(~np.isnan(mat), axis=0))
            plt.title("Number of lines contributing to each point")
            plt.show()

        # Null distribution
        if max_matrix_length is None:
            mat_len = mat.shape[1]
        else:
            mat_len = max_matrix_length
        baseline_lines = self.calc_null_distribution_of_triggered_lines(mat_len,
                                                                        num_baseline_lines, trace,
                                                                        triggered_average_indices)

        # if DEBUG:
        #     plt.figure(dpi=100)
        #     all_ind_jitter = np.hstack(all_ind_jitter)
        #     plt.hist(all_ind_jitter)
        #     plt.title("Number of times each data point is selected")
        #     plt.show()

        # Normalize by the std of the baseline
        # Note: calc the std across trials, then average across time
        baseline_per_line_std = np.std(baseline_lines, axis=0)
        baseline_std = np.mean(baseline_per_line_std)

        zeta_line_dat /= baseline_std
        baseline_lines /= baseline_std

        if DEBUG:
            plt.figure(dpi=100)
            plt.plot(zeta_line_dat)
            for i_row in range(baseline_lines.shape[0]):
                line = baseline_lines[i_row, :]
                plt.plot(line, 'gray', alpha=0.1)
            plt.ylabel("Deviation (std of baseline)")
            plt.title("Trace zeta line and null distribution")
            plt.show()

        # Calculate individual zeta values (max deviation)
        zeta_dat = np.max(np.abs(zeta_line_dat))
        zetas_baseline = np.max(np.abs(baseline_lines), axis=1)

        # ALT: calculate sum of squares, and plot
        # Idea: maybe I can do chi squared instead
        # Following: https://stats.stackexchange.com/questions/200886/what-is-the-distribution-of-sum-of-squared-errors
        # if DEBUG:
        #     zeta2_dat = np.sum(np.abs(zeta_line_dat)**2.0)
        #     zetas2_baseline = np.sum(np.abs(baseline_lines)**2.0, axis=1)
        #
        #     # What is the df for time series errors?
        #     p2 = 1 - scipy.stats.chi2.cdf(zeta2_dat, 2)
        #
        #     plt.figure(dpi=100)
        #     plt.hist(zetas2_baseline)#, bins=np.arange(0, np.max(zetas2_baseline)))
        #     plt.vlines(zeta2_dat, 0, len(zetas_baseline) / 2, colors='red')
        #     plt.title(f"Distribution of sum of squares, with p={p2}")
        #     plt.show()

        # Final p value
        p = calculate_p_value_from_zeta(zeta_dat, zetas_baseline)

        if DEBUG:
            plt.figure(dpi=100)
            plt.hist(zetas_baseline)
            plt.vlines(zeta_dat, 0, len(zetas_baseline) / 2, colors='red')
            plt.title(f"Distribution of maxima of null, with p value: {p}")
            plt.show()

        return p, (zeta_dat, zetas_baseline)

    def calc_null_distribution_of_triggered_lines(self, max_matrix_length, num_baseline_lines, trace,
                                                  triggered_average_indices):
        baseline_lines = np.zeros((num_baseline_lines, max_matrix_length))
        all_ind_jitter = []
        for i in range(num_baseline_lines):
            ind_jitter = jitter_indices(triggered_average_indices, max_jitter=len(trace), max_len=len(trace))
            mat_jitter = self.calc_triggered_average_matrix(trace, custom_ind=ind_jitter,
                                                            max_len=max_matrix_length)
            zeta_line = calculate_zeta_cumsum(mat_jitter)
            baseline_lines[i, :] = zeta_line
            all_ind_jitter.extend(ind_jitter)
            # if DEBUG:
            #     time.sleep(2)
        return baseline_lines

    def calc_p_value_using_ttest(self, trace, gap=5, DEBUG=False) -> Tuple[float, float]:
        """
        Calculates a p value using a paired t-test on the pre- and post-stimulus time periods

        Note that this is generally sensitive to ind_preceding (in addition to other arguments)

        See calc_p_value_using_ttest_triggered_average for a more general version

        Parameters
        ----------
        trace
        num_baseline_lines

        Returns
        -------

        """
        mat = self.calc_triggered_average_matrix(trace)
        if mat is None:
            return 1, 0
        means_before, means_after = self.split_means_from_triggered_average_matrix(mat, gap=gap)
        p = scipy.stats.ttest_rel(means_before, means_after, nan_policy='omit').pvalue
        effect_size = np.nanmean(means_after) - np.nanmean(means_before)

        if DEBUG:
            self.plot_triggered_average_from_matrix(mat, show_individual_lines=True)
            plt.title(f"P value: {p}")

            df = pd.DataFrame([means_before, means_after]).dropna(axis=1)
            paired_boxplot_from_dataframes(df)
            plt.title(f"P value: {p}")

            plt.show()

        return p, effect_size

    def split_means_from_triggered_average_matrix(self, mat, gap):
        """Gets mean of trace before and after the trigger (same window length)"""
        i_trigger = self.ind_preceding
        num_pts = i_trigger - gap
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=RuntimeWarning)
            means_before = np.nanmean(mat[:, 0:num_pts], axis=1)
            means_after = np.nanmean(mat[:, i_trigger:i_trigger + num_pts], axis=1)
        return means_before, means_after

    def plot_triggered_average_from_matrix(self, triggered_avg_matrix, ax=None,
                                           show_individual_lines=False,
                                           color_significant_times=False,
                                           is_second_plot=False,
                                           **kwargs):
        """
        Core plotting function; must be passed a matrix

        Parameters
        ----------
        triggered_avg_matrix
        ax
        show_individual_lines
        color_significant_times
        kwargs

        Returns
        -------

        """

        min_lines = self.min_lines
        ind_preceding = self.ind_preceding
        ax, triggered_avg = plot_triggered_average_from_matrix_low_level(triggered_avg_matrix,
                                                                         ind_preceding=ind_preceding,
                                                                         min_lines=min_lines,
                                                                         show_individual_lines=show_individual_lines,
                                                                         is_second_plot=is_second_plot, ax=ax, **kwargs)
        if ax is None:
            return
        # Optional orange points
        x_significant = self.calc_significant_points_from_triggered_matrix(triggered_avg_matrix)
        if color_significant_times:
            if len(x_significant) > 0:
                ax.plot(x_significant, triggered_avg[x_significant], 'o', color='tab:orange')

        return ax

    def plot_events_over_trace_from_name(self, trace, ax=None):
        """
        Plots the indices stored here over a trace (for debugging)

        Parameters
        ----------
        trace

        Returns
        -------

        """
        if ax is None:
            fig, ax = plt.subplots(dpi=100)
        ax.plot(trace)
        self.plot_events_over_trace(trace, ax)
        return ax

    def plot_events_over_trace(self, trace: pd.Series, ax=None, vertical_lines=True, dots=True,
                               DEBUG=False):
        """
        Plots dots where the preceding indices start, and vertical lines where the event onsets are

        Parameters
        ----------
        trace
        ax
        vertical_lines
        dots

        Returns
        -------

        """
        trace_idx = list(trace.index)
        if ax is None:
            fig, ax = plt.subplots(dpi=100)
        for idx_list in self.triggered_average_indices():
            i_start = idx_list[self.ind_preceding]
            i_end = idx_list[-1]
            if dots:
                i_start_with_previous = idx_list[0]
                ax.plot(trace_idx[i_start_with_previous], trace.iat[i_start_with_previous], '.', color='tab:orange')
                if DEBUG:
                    print(f"i_start: {i_start}, i_start_clipped: {i_start_with_previous}")
            if vertical_lines:
                ax.axvline(trace_idx[i_start], trace.iat[i_start], linestyle='--', color='tab:green')
                ax.axvline(trace_idx[i_end], trace.iat[i_end], linestyle='--', color='tab:red')

    @property
    def idx_onsets(self):
        """Returns the indices of the onsets"""
        local_idx_of_onset = self.ind_preceding
        idx_onsets = np.array([vec[local_idx_of_onset] for vec in self.triggered_average_indices() if
                               vec[local_idx_of_onset] > 0])
        return idx_onsets

    def onset_vector(self):
        """Binary vector: 0s everywhere except for the onsets"""
        onset_vec = np.zeros(self.trace_len)
        onset_vec[self.idx_onsets] = 1
        return onset_vec

    @property
    def num_events(self):
        return len(self.idx_onsets)

    def all_state_durations(self, include_censored=True):
        # Note that this doesn't account for gaps caused by incorrect annotation
        binary_vec = self.cleaned_binary_state
        all_starts, all_ends = get_contiguous_blocks_from_column(binary_vec, already_boolean=True)
        duration_vec, censored_vec = calc_surpyval_durations_and_censoring(all_starts, all_ends)
        if not include_censored:
            # Remove the censored ones (the ones at the edges)
            duration_vec = np.array(duration_vec)[np.array(censored_vec) == 0]
            duration_vec = list(duration_vec)
        return duration_vec, censored_vec

    def __repr__(self):
        beh_name = self.behavioral_state.name if self.behavioral_state is not None else "None"
        return f"TriggeredAverageIndices: {beh_name} ({self.num_events} events found)"


@dataclass
class FullDatasetTriggeredAverages:
    """
    A class that uses TriggeredAverageIndices to process each trace of a full dataset (Dataframe) into a matrix of
    triggered averages

    For main functionality, see:
        triggered_average_matrix_from_name
        load_from_project

    Also has functions for plotting

    The following is the auto-generated description of all methods
    ----------------------------------------------------------------------------------------------

    """
    df_traces: pd.DataFrame

    # Calculating indices
    ind_class: TriggeredAverageIndices  # Optional
    _ind_preceding: int = None  # Only used if ind_class is None

    # Calculating full average
    mean_subtract_each_trace: bool = False
    min_lines: int = 2
    min_points_for_significance: int = 5
    significance_calculation_method: str = 'ttest'  # Or: 'num_points'

    # Plotting
    show_individual_lines: bool = True
    color_significant_times: bool = True

    @property
    def neuron_names(self):
        names = list(set(self.df_traces.columns.get_level_values(0)))
        names.sort()
        return names

    @cached_property
    def df_left_right_combined(self):
        return combine_columns_with_suffix(self.df_traces)

    def get_df(self, combine_left_right=False):
        if combine_left_right:
            return self.df_left_right_combined
        else:
            return self.df_traces

    def triggered_average_matrix_from_name(self, name, combine_left_right=False, **kwargs):
        """
        Calculates the triggered average matrix (events are rows, time is columns) for a single neuron

        Parameters
        ----------
        name

        Returns
        -------

        """
        return self.ind_class.calc_triggered_average_matrix(self.get_df(combine_left_right)[name], **kwargs)

    def dict_of_all_triggered_averages(self):
        """
        Unlike df_of_all_triggered_averages, returns a dict of np.ndarrays, saving not just the mean but the full matrix
        for each neuron

        Returns
        -------

        """
        dict_triggered = {}
        for name in self.neuron_names:
            mat = self.triggered_average_matrix_from_name(name)
            dict_triggered[name] = mat
        return dict_triggered

    def df_of_all_triggered_averages(self):
        """
        Like triggered_average_matrix_from_name, but just saves the mean of the triggered average
        """
        df_triggered = {}
        for name in self.neuron_names:
            mat = self.triggered_average_matrix_from_name(name)
            if mat is None:
                continue
            raw_trace_mean, triggered_avg, _, _, xmax, is_valid = \
                self.ind_class.prep_triggered_average_for_plotting(mat, min_lines=self.min_lines,
                                                                   shorten_to_last_valid=False)
            if not is_valid:
                continue
            df_triggered[name] = triggered_avg

        df_triggered = pd.DataFrame(df_triggered)
        df_triggered = df_triggered.loc[:df_triggered.last_valid_index()]
        return df_triggered

    def which_neurons_are_significant(self, min_points_for_significance=None, num_baseline_lines=100,
                                      ttest_gap=5, neuron_names=None, combine_left_right=False, verbose=1, DEBUG=False):
        if min_points_for_significance is not None:
            self.min_points_for_significance = min_points_for_significance

        df_traces = self.get_df(combine_left_right=combine_left_right)

        names_to_keep = []
        all_p_values = {}
        all_effect_sizes = {}
        if neuron_names is None:
            neuron_names = self.neuron_names
        for name in tqdm(neuron_names, leave=False):
            if name not in df_traces:
                continue
            if DEBUG:
                print("======================================")
                print(name)

            if self.significance_calculation_method == 'zeta':
                # logging.warning("Zeta calculation is unstable for calcium imaging!")
                trace = df_traces[name]
                p, (zeta_dat, zetas_baseline) = self.ind_class.calc_p_value_using_zeta(trace, num_baseline_lines, DEBUG=DEBUG)
                all_p_values[name] = p
                _df = pd.DataFrame(zetas_baseline, columns=['zeta_value']).assign(baseline=True, neuron_name=name)
                _df2 = pd.DataFrame({'zeta_value': [zeta_dat], 'baseline': False, 'neuron_name': name})
                all_effect_sizes[name] = pd.concat([_df, _df2])
                to_keep = p < 0.05
            elif self.significance_calculation_method == 'num_points':
                # logging.warning("Number of points calculation is not statistically justified!")
                mat = self.triggered_average_matrix_from_name(name)
                x_significant = self.ind_class.calc_significant_points_from_triggered_matrix(mat)
                all_p_values[name] = x_significant
                to_keep = len(x_significant) > self.min_points_for_significance
            elif self.significance_calculation_method == 'ttest':
                trace = df_traces[name]
                p, effect_size = self.ind_class.calc_p_value_using_ttest(trace, ttest_gap, DEBUG=DEBUG)
                all_p_values[name] = p
                all_effect_sizes[name] = effect_size
                to_keep = p < 0.05
            else:
                raise NotImplementedError(f"Unrecognized significance_calculation_method: "
                                          f"{self.significance_calculation_method}")

            if to_keep:
                names_to_keep.append(name)

        if len(names_to_keep) == 0 and verbose >= 1:
            logging.warning("Found no significant neurons, subsequent steps may not work")

        return names_to_keep, all_p_values, all_effect_sizes

    def plot_single_neuron_triggered_average(self, neuron, ax=None, **kwargs):
        if neuron in self.neuron_names:
            y = self.df_traces[neuron]
        elif neuron in get_names_from_df(self.df_left_right_combined):
            y = self.df_left_right_combined[neuron]
        else:
            raise NeedsAnnotatedNeuronError(neuron)
        ax = self.ax_plot_func_for_grid_plot(None, y, ax, neuron, **kwargs)
        if not kwargs.get('use_plotly', False):
            plt.title(f"Triggered average for {neuron}")
        return ax

    def plot_multi_neuron_triggered_average(self, neuron_list, ax=None, skip_if_not_present=True, **kwargs):
        for i, neuron in enumerate(neuron_list):
            opt = dict()
            if i > 0:
                opt['show_horizontal_line'] = False
            try:
                ax = self.plot_single_neuron_triggered_average(neuron, ax, **kwargs, **opt)
            except NeedsAnnotatedNeuronError as e:
                if skip_if_not_present:
                    pass
                else:
                    raise e
        return ax

    def plot_events_over_trace(self, neuron, **kwargs):
        trace = self.df_traces[neuron]
        ax = self.ind_class.plot_events_over_trace_from_name(trace, **kwargs)
        plt.title(f"Trace with events for {neuron}")
        return ax

    def ax_plot_func_for_grid_plot(self, t, y, ax, name, DEBUG=False, **kwargs):
        """Same as ax_plot_func_for_grid_plot, but can be used directly"""
        if kwargs.get('is_second_plot', False):
            # Do not want two legend labels
            if 'label' in kwargs:
                kwargs['label'] = ''
            plot_kwargs = dict(label='')
        else:
            plot_kwargs = dict(label=name)
        plot_kwargs.update(kwargs)

        mat = self.ind_class.calc_triggered_average_matrix(y, DEBUG=DEBUG)
        ax = self.ind_class.plot_triggered_average_from_matrix(mat, ax, **plot_kwargs)
        try:
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("$\Delta R / R_{50}$")
            # ax.axhline(0, c='black', ls='--')
            # ax.plot(self.ind_class.ind_preceding, 0, "r>", markersize=10)
        except AttributeError:
            # Then it is plotly
            pass

        return ax

    @staticmethod
    def load_from_project(project_data, trigger_opt=None, trace_opt=None, triggered_time_series_mode="traces",
                          **kwargs):
        """
        Loads a FullDatasetTriggeredAverages class from a ProjectData class

        Uses the default traces from the ProjectData class, and the default triggered average indices from the
        WormFullVideoPosture class (which uses automatic behavioral annotations)

        If you want to use custom traces or triggers, use trace_opt or trigger_opt
        A specific example is to use pass a custom behavioral annotation using:
            behavioral_annotation = np.array([0, 0, ..., 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, ...])
            trigger_opt = dict(behavioral_annotation=behavioral_annotation, state=1)
            triggered_class = FullDatasetTriggeredAverages.load_from_project(project_data_gcamp,
                                                                             trigger_opt=trigger_opt)

        Parameters
        ----------
        project_data - ProjectData class
        trigger_opt - Passed to WormFullVideoPosture.calc_triggered_average_indices
        trace_opt - Passed to ProjectData.calc_default_traces
        triggered_time_series_mode - how to calculate the time series to be triggered on. Options:
            "traces" - fluorescence traces
            "behavior" - behavioral annotations
            "curvature" - curvature from the kymograph
        kwargs - Passed to FullDatasetTriggeredAverages

        Returns
        -------

        """
        if trigger_opt is None:
            trigger_opt = {}
        if trace_opt is None:
            trace_opt = {}

        trigger_opt_default = dict(min_lines=3, ind_preceding=20)
        trigger_opt_default.update(trigger_opt)
        ind_class = project_data.worm_posture_class.calc_triggered_average_indices(**trigger_opt_default)

        trace_opt_default = dict(use_paper_options=True)
        trace_opt_default.update(trace_opt)
        if triggered_time_series_mode == "traces":
            df_traces = project_data.calc_default_traces(**trace_opt_default)
        elif triggered_time_series_mode == "behavior":
            df_traces = project_data.calc_default_behaviors(**trace_opt_default)
        elif triggered_time_series_mode == "curvature":
            df_traces = project_data.worm_posture_class.curvature(fluorescence_fps=True, reset_index=True,
                                                                  rename_columns=True)
        else:
            raise NotImplementedError(f"Unrecognized triggered_time_series_mode: {triggered_time_series_mode}")

        triggered_averages_class = FullDatasetTriggeredAverages(df_traces, ind_class, **kwargs)

        return triggered_averages_class

    def __repr__(self):
        message = f"FullDatasetTriggeredAverages with {len(self.neuron_names)} neurons"
        if self.ind_class is None:
            message = message + f"; no ind_class is saved, and triggered averages"
        else:
            message = message + f"\n With triggered average class: {self.ind_class}"
        return message


@dataclass
class ClusteredTriggeredAverages:

    df_triggered: pd.DataFrame

    # For plotting individual clusters
    triggered_averages_class: FullDatasetTriggeredAverages = None
    linkage_threshold: float = 100.0  # Not a great default; depends strongly on dataset... designed to get one cluster
    cluster_criterion: str = 'distance'  # Alternate: 'maxclust'
    linkage_method: str = 'average'
    min_correlation: float = 0.0  # Used to filter out weakly correlated neurons

    _R: np.ndarray = None  # The actual dendrogram

    # For plotting or calculating p values with all triggered traces, not just averages
    dict_of_triggered_traces: Dict[str, np.ndarray] = None
    max_trace_len: int = None

    cluster_func: Callable = field(default=hierarchy.fcluster)

    cluster_cmap: Union[str, dict, list] = 'tab10'
    cluster_cmap_is_set: bool = False

    # For plotting individual traces
    _df_traces: pd.DataFrame = None
    _df_behavior: pd.DataFrame = None
    _ind_preceding: int = None

    verbose: int = 0

    def __post_init__(self):

        # Make sure all lists of names are aligned
        self.df_triggered = self.df_triggered.sort_values(by=0, axis='columns')

        if not self.cluster_cmap_is_set:
            self.set_global_scipy_cmap()

    @cached_property
    def df_corr(self) -> pd.DataFrame:
        if self.verbose >= 1:
            print("Calculating correlation")
        df_corr = self.df_triggered.corr()
        return df_corr.replace(np.nan, 0, inplace=False)

    @lru_cache(maxsize=16)
    def _do_clustering(self, linkage_threshold, cluster_criterion, linkage_method):
        if self.verbose >= 1:
            print("Calculating clustering")
        # Assume these don't change, unlike the function args
        df_corr = self.df_corr
        names = self.names
        cluster_func = self.cluster_func

        # Remove neurons that are weakly correlated with everything
        # if self.min_correlation > 0.0:
        #     df = self.df_corr
        #     df_max = df[df < 1].max()
        #     neurons_to_keep = df_max[df_max > self.min_correlation].index
        #     df_corr = df_corr.loc[neurons_to_keep, neurons_to_keep]
            # TODO: fix name offsets later

        # Set random number seeds
        np.random.seed(4242)

        # Calculate clustering
        Z = hierarchy.linkage(df_corr.to_numpy(), method=linkage_method, optimal_ordering=True)
        clust_ind = cluster_func(Z, t=linkage_threshold, criterion=cluster_criterion)

        per_cluster_names = {}
        for i_clust in np.unique(clust_ind):
            per_cluster_names[i_clust] = names[clust_ind == i_clust]
        if self.verbose >= 1:
            print(f"Found {len(per_cluster_names)} clusters")

        return Z, clust_ind, per_cluster_names

    # Set all dependent attributes as properties, which call (cached) _do_clustering each time
    @property
    def clust_args(self):
        return self.linkage_threshold, self.cluster_criterion, self.linkage_method

    @property
    def Z(self) -> np.ndarray:
        Z, clust_ind, per_cluster_names = self._do_clustering(*self.clust_args)
        return Z

    @property
    def clust_ind(self):
        Z, clust_ind, per_cluster_names = self._do_clustering(*self.clust_args)
        return clust_ind

    @property
    def per_cluster_names(self) -> Dict[int, List[str]]:
        Z, clust_ind, per_cluster_names = self._do_clustering(*self.clust_args)
        return per_cluster_names

    @property
    def df_traces(self):
        if self._df_traces is not None:
            return self._df_traces
        elif self.triggered_averages_class is not None:
            return self.triggered_averages_class.df_traces
        else:
            raise ValueError("df_traces is not saved; must provide either _df_traces or triggered_averages_class")

    @property
    def df_behavior(self):
        if self._df_behavior is not None:
            return self._df_behavior
        elif self.triggered_averages_class is not None:
            return self.triggered_averages_class.ind_class.behavioral_annotation
        else:
            raise ValueError("df_behavior is not saved; must provide either _df_behavior or triggered_averages_class")

    @property
    def names(self):
        # return pd.Series(get_names_from_df(self.df_corr, to_sort=False))
        return pd.Series(list(self.df_corr.columns))

    @cached_property
    def number_of_datasets(self):
        """
        Only makes sense if datasets were pooled using flatten_multiindex_columns function

        Returns
        -------

        """
        return count_unique_datasets_from_flattened_index(self.names)

    def plot_clustergram(self, output_folder=None, use_labels=False, show_clusters=True):
        X = self.df_corr.to_numpy()

        # Check for jupyter notebook and large matrices
        static_rendering_required, render_opt = check_plotly_rendering(X)

        dist_fun = lambda X, metric: X  # df_corr is already the distance (similarity)
        import dash_bio
        opt = dict(height=800, width=1000, link_method=self.linkage_method,
                   color_threshold={'row': self.linkage_threshold, 'col': self.linkage_threshold},
                   center_values=False)
        if use_labels:
            opt.update(dict(row_labels=list(self.names), column_labels=list(self.names)))
        else:
            opt.update(dict(hidden_labels=['row', 'col']))
        if not show_clusters:
            # opt.update(dict(color_list={'row': ['blue'], 'col': ['blue']}))
            opt.update(dict(color_threshold={'row': np.inf, 'col': np.inf}))
        clustergram = dash_bio.Clustergram(X, dist_fun=dist_fun, **opt)
        if output_folder is not None:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder, exist_ok=True)
            clustergram.write_image(os.path.join(output_folder, 'clustergram.png'))
        clustergram.show(**render_opt)

        return clustergram

    def plot_clustergram_matplotlib(self, output_folder=None, use_labels=False,
                                    no_dendrogram=True, ax=None, figsize=(5, 2)):
        """
        Similar to plot_clustergram but uses matplotlib (not interactive) instead of plotly

        Parameters
        ----------
        output_folder
        use_labels

        Returns
        -------

        """
        df = self.df_corr

        if ax is None:
            if no_dendrogram:
                figsize = (figsize[0] / 2, figsize[1])
                fig, ax = plt.subplots(dpi=200, figsize=figsize)
                ax_dend = None
                # ax.set_title("Dendrogram-matched correlations")
            else:
                fig, (ax_dend, ax) = plt.subplots(2, 1, dpi=200, figsize=figsize)
                ax.set_box_aspect(1)
                ax_dend.set_box_aspect(1)
                ax_dend.set_title("Dendrogram-matched correlations")

        R = self.recalculate_dendrogram(self.linkage_threshold, no_plot=no_dendrogram, ax=ax_dend)
        dendrogram_idx = [int(i) for i in R['ivl']]
        # Needs to be reversed if the orientation is left
        # dendrogram_idx.reverse()

        im = ax.imshow(df.iloc[dendrogram_idx, dendrogram_idx], cmap='BrBG')
        if not use_labels:
            ax.set_xticks([])
            ax.set_yticks([])
        # From: https://stackoverflow.com/questions/32462881/add-colorbar-to-existing-axis
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
        ax.set_title("Correlation")

        self._save_plot("clustergram.png", output_folder)

    def plot_dendrogram_matplotlib(self, linkage_threshold=None, output_folder=None, show_xticks=False):
        if linkage_threshold is None:
            linkage_threshold = self.linkage_threshold

        # link_color_func = good_dataset_clusterer.map_list_of_cluster_ids_to_colors(split_dict)
        # R = hierarchy.dendrogram(Z, orientation='left', no_labels=True, link_color_func=link_color_func)

        fig, ax = plt.subplots(dpi=200, figsize=(5/2, 2))
        no_plot = False
        self.recalculate_dendrogram(linkage_threshold, no_plot, ax)
        if not show_xticks:
            plt.axis('off')
            plt.xticks([])

        self._save_plot("dendrogram.png", output_folder)

    def recalculate_dendrogram(self, linkage_threshold=None, no_plot=False, ax=None):
        if linkage_threshold is None:
            linkage_threshold = self.linkage_threshold
        Z = self.Z
        if self.cluster_criterion != 'distance':
            logging.warning("Cluster criterion is not distance, dendrogram will not be matched to clusters")
        R = hierarchy.dendrogram(Z, orientation='top', no_labels=True, color_threshold=linkage_threshold,
                                 above_threshold_color='black', ax=ax, no_plot=no_plot)
        self._R = R
        return R

    def plot_subcluster_clustergram(self, i_clust, linkage_threshold=None, ax=None):
        """
        Reclusters a single cluster and plots a plotly (interactive) clustergram

        Parameters
        ----------
        i_clust
        linkage_threshold

        Returns
        -------

        """
        if linkage_threshold is None:
            linkage_threshold = self.linkage_threshold
        # redo clustering of a single cluster
        names = self.per_cluster_names[i_clust]
        df_corr = self.df_corr.loc[names, names]
        X = df_corr.to_numpy()
        static_rendering_required, render_opt = check_plotly_rendering(X)

        if not static_rendering_required:
            dist_fun = lambda X, metric: X  # df_corr is already the distance (similarity)
            import dash_bio
            opt = dict(height=800, width=800, link_method=self.linkage_method,
                       color_threshold={'row': linkage_threshold, 'col': linkage_threshold},
                       center_values=False)
            # opt.update(dict(row_labels=[], column_labels=[]))
            # opt['row_labels'] = []
            # opt['column_labels'] = []
            clustergram = dash_bio.Clustergram(X, dist_fun=dist_fun, **opt)
            clustergram.show(**render_opt)

            Z, R = None, None
        else:
            # Sometimes the colormap with matplotlib breaks dash_bio, so just go with matplotlib
            Z = hierarchy.linkage(df_corr.to_numpy(), method=self.linkage_method, optimal_ordering=True)
            R = hierarchy.dendrogram(Z, orientation='top', no_labels=True, color_threshold=linkage_threshold,
                                     above_threshold_color='black', ax=ax, no_plot=False)

        return Z, R

    def plot_all_clusters(self):
        ind_class = self.triggered_averages_class.ind_class
        for i_clust, name_list in self.per_cluster_names.items():
            fig, ax = plt.subplots(dpi=200, figsize=(5/2, 2))
            # Build a pseudo-triggered average matrix, made of the means of each neuron
            pseudo_mat = []
            for name in name_list:
                triggered_avg = self.df_triggered[name].copy()
                pseudo_mat.append(triggered_avg)
            # Normalize the traces to be similar to the correlation, i.e. z-score them
            pseudo_mat = np.stack(pseudo_mat)
            # Ignore runtime warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                pseudo_mat = pseudo_mat - np.nanmean(pseudo_mat, axis=1, keepdims=True)
                pseudo_mat = pseudo_mat / np.nanstd(pseudo_mat, axis=1, keepdims=True)
            # Plot
            ind_class.plot_triggered_average_from_matrix(pseudo_mat, ax, show_individual_lines=True)
            plt.title(f"Triggered Averages of cluster {i_clust} ({pseudo_mat.shape[0]} traces)")

    def plot_all_clusters_simple(self, min_lines=2, ind_preceding=None, xlim=None, z_score=False,
                                 output_folder=None, use_individual_triggered_events=False,
                                 per_cluster_names=None, **kwargs):
        """Like plot_all_clusters, but doesn't require a triggered_averages_class to be saved"""
        if ind_preceding is None:
            ind_preceding = self._ind_preceding
        if per_cluster_names is None:
            per_cluster_names = self.per_cluster_names
        if use_individual_triggered_events:
            get_matrix_from_names = lambda names: self.get_triggered_matrix_all_events_from_names(names)[0]
        else:
            get_matrix_from_names = self.get_subset_triggered_average_matrix

        self.plot_clusters_from_names(get_matrix_from_names, per_cluster_names, min_lines, ind_preceding, xlim, z_score,
                                      output_folder, **kwargs)

    def set_global_scipy_cmap(self, force_reset=False):
        if not force_reset and self.cluster_cmap_is_set:
            return
        # Set the cluster color map depending on type of cluster_cmap
        if isinstance(self.cluster_cmap, str):
            cmap = matplotlib.colormaps[self.cluster_cmap]
            # This is a global variable... probably shouldn't be reset every time
            cmap_hex = [matplotlib.colors.rgb2hex(rgba) for rgba in cmap.colors[1:]]
            hierarchy.set_link_color_palette(cmap_hex)
        elif isinstance(self.cluster_cmap, list):
            hierarchy.set_link_color_palette(self.cluster_cmap)
        elif isinstance(self.cluster_cmap, dict):
            # We need a list, but we want to remove any keys associated with singleton clusters
            cmap_list = [self.cluster_cmap[i] for i in self.cluster_cmap.keys() if len(self.per_cluster_names[i]) > 1]
            hierarchy.set_link_color_palette(cmap_list)
        else:
            raise ValueError("cluster_cmap must be a string, list or dict")

        self.cluster_cmap_is_set = True

    def cluster_color_func(self, i):
        # By default, self.cluster_cmap is a string, but map_clusters_using_paper_cmap changes it to a list
        if isinstance(self.cluster_cmap, str):
            # The dendrogram has a funny default, where the 0 color is reserved for non-clusters, and is skipped
            # So in principle I want the modular division of i, but if i > 10 I have to add 1
            # We want to skip 0, 10, 20, etc.
            i = int((i + (i - 1) // 9) % 10)
            cmap = matplotlib.colormaps[self.cluster_cmap]
            color = cmap(i)
        elif isinstance(self.cluster_cmap, list):
            color = self.cluster_cmap[i]
        elif isinstance(self.cluster_cmap, dict):
            color = self.cluster_cmap[i]
        else:
            raise ValueError(f"Unknown cluster_cmap type {type(self.cluster_cmap)}")
        return color

    def reset_cluster_cmap(self):
        self.cluster_cmap = 'tab10'
        self.set_global_scipy_cmap(force_reset=True)

    def map_clusters_using_paper_cmap(self, base_cmap=None, other_color_offset=None, verbose=0):
        """
        Uses hard-coded colors as determined by belonging of target neurons in an example dataset

        Sets other clusters to be black if other_color_offset is None, otherwise sets them to be the same colormap, but
        offset by other_color_offset (specifically so as to not overlap with with another clustering)

        Returns
        -------

        """
        if base_cmap is None:
            base_cmap = matplotlib.colormaps['tab10']

        # Hard code the mapping of specific neurons to clusters
        # Prioritization goes from top to bottom, if there are multiple neurons within the same cluster
        neuron2color = {
            "neuron_056": 4,  # Unknown FWD neuron, possibly RIBL, purple... alt: neuron_029
            "AVBL": 4,  # Unknown FWD neuron, possibly RIBL, purple... alt: neuron_029
            "neuron_008": 0,  # RIS, blue
            "RIS": 0,  # RIS, blue
            "neuron_060": 1,  # AVAL, orange
            "AVAL": 1,  # AVAL, orange
            # "neuron_033": 2,  # Ventral turning neuron, green
            "RIVL": 2,  # Ventral turning neuron, green
            "neuron_076": 3,  # RID (after turn), red
            "RID": 3,  # RID (after turn), red
        }
        # Prepend project name to keys
        project_name = "ZIM2165_Gcamp7b_worm1-2022_11_28"
        neuron2color = {f"{project_name}_{k}": v for k, v in neuron2color.items()}
        # Get which cluster each neuron belongs to
        # Note that each cluster could have multiple neurons... for now, allow them to overwrite each other
        custom_cmap = {}
        other_colors_used = 0
        custom_colors_used = max(neuron2color.values())
        for i_clust, names_in_clust in self.per_cluster_names.items():
            names_in_clust = set(names_in_clust)
            for neuron_name, color in neuron2color.items():
                if neuron_name in names_in_clust:
                    custom_cmap[i_clust] = base_cmap(color)
                    if verbose >= 1:
                        print(f"Setting cluster {i_clust} ({len(names_in_clust)} traces) to color {color} ({neuron_name})")
                    break
            else:
                if other_color_offset is None:
                    if verbose >= 2:
                        print(f"Setting unknown cluster {i_clust} to black")
                    custom_cmap[i_clust] = (0, 0, 0, 1)
                else:
                    custom_cmap[i_clust] = base_cmap(1 + custom_colors_used + other_colors_used + other_color_offset)
                    other_colors_used += 1
                    if verbose >= 2:
                        print(f"Setting unknown cluster {i_clust} to color {custom_cmap[i_clust]}")
        # Convert from floats to hex strings
        custom_cmap = {k: matplotlib.colors.to_hex(v) for k, v in custom_cmap.items()}

        # Finally set the cmap
        self.cluster_cmap = custom_cmap
        self.set_global_scipy_cmap(force_reset=True)

    def plot_clusters_from_names(self, get_matrix_from_names, per_cluster_names, min_lines=2,
                                 ind_preceding=None, xlim=None, z_score=False, output_folder=None,
                                 show_individual_lines=True, cluster_color_func: Callable = None,
                                 fig_opt=None, to_show=True, behavior_shading_type=None, **kwargs):

        if ind_preceding is None:
            ind_preceding = self._ind_preceding

        if fig_opt is None:
            fig_opt = {}
        default_fig_opt = dict(dpi=200)
        default_fig_opt.update(fig_opt)

        if cluster_color_func is None:
            cluster_color_func = self.cluster_color_func
        i_clust_non_singleton = 0  # The scipy function skips colors for singleton clusters
        for i_clust, name_list in per_cluster_names.items():
            name_list = list(name_list)
            if len(name_list) > 1:
                i_clust_non_singleton += 1
            if len(name_list) < min_lines:
                print(f"Skipping cluster {i_clust} with {len(name_list)} lines")
                continue
            fig, ax = plt.subplots(**default_fig_opt)
            pseudo_mat = get_matrix_from_names(name_list)
            # Normalize the traces to be similar to the correlation, i.e. z-score them
            if z_score:
                pseudo_mat = pseudo_mat - np.nanmean(pseudo_mat, axis=1, keepdims=True)
                pseudo_mat = pseudo_mat / np.nanstd(pseudo_mat, axis=1, keepdims=True)
            # Plot
            if isinstance(pseudo_mat, pd.DataFrame) and np.isnan(pseudo_mat).values.all():
                continue
            elif isinstance(pseudo_mat, np.ndarray) and np.isnan(pseudo_mat).all():
                continue
            plot_opt = dict(show_individual_lines=show_individual_lines, is_second_plot=False, xlim=xlim)
            if cluster_color_func is not None:
                # I need to account for singleton clusters here, unless it is a dictionary
                if isinstance(self.cluster_cmap, dict):
                    cluster_color = cluster_color_func(i_clust)
                else:
                    cluster_color = cluster_color_func(i_clust_non_singleton)
                plot_opt['color'] = cluster_color
            # these_corr = self.df_corr.loc[name_list[0], name_list[1:]]
            # avg_corr = these_corr.mean()
            plot_triggered_average_from_matrix_low_level(pseudo_mat, ind_preceding, min_lines,
                                                         ax=ax, **plot_opt)
            plt.title(f"Triggered Averages of cluster {i_clust} ({pseudo_mat.shape[0]} traces)")
            plt.xlabel("Time (seconds)")
            plt.tight_layout()

            add_behavior_shading_to_plot(ind_preceding, pseudo_mat.columns, behavior_shading_type, ax)

            base_fname = f"cluster_{i_clust}.png"
            self._save_plot(base_fname, output_folder)
            if to_show:
                plt.show()

    def _save_plot(self, base_fname, output_folder):

        plt.tight_layout()
        if output_folder is not None:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder, exist_ok=True)
            plt.savefig(os.path.join(output_folder, base_fname), dpi=200, transparent=True)
            # Also save .svg
            plt.savefig(os.path.join(output_folder, base_fname.replace(".png", ".svg")), dpi=200, transparent=True)

    def plot_all_clusters_grid_plot(self, i_clust, num_columns=1, **kwargs):
        """Like plot_all_clusters, but plots the full times series instead of the triggered average"""
        if self.df_traces is None:
            raise ValueError("df_traces is None, cannot plot")
        name_list = list(self.per_cluster_names[i_clust])
        # Use the grid plot function to plot
        from wbfm.utils.visualization.plot_traces import make_grid_plot_from_dataframe
        fig, axes = make_grid_plot_from_dataframe(self.df_traces, name_list, num_columns=num_columns, **kwargs)

    def plot_all_clusters_grid_plot_multi_project(self, i_clust, all_projects,
                                                  min_neurons_per_project=1, num_columns=1, **kwargs):
        """
        Like plot_all_clusters_grid_plot, but assumes that traces come from different projects, which need different shading
        """
        from wbfm.utils.visualization.plot_traces import make_grid_plot_from_dataframe

        if self.df_traces is None:
            raise ValueError("df_traces is None, cannot plot")

        # Use the grid plot function to plot
        for project_name, project in all_projects.items():
            bh = self.df_behavior[project_name]
            shade_plot_func = lambda ax: shade_using_behavior(bh, ax=ax)

            # Only get the names that are in this project
            this_name_list = self.get_neurons_in_cluster_and_project(i_clust, project_name)
            if len(this_name_list) < min_neurons_per_project:
                continue

            fig, axes = make_grid_plot_from_dataframe(self.df_traces, this_name_list,
                                                      num_columns=num_columns, shade_plot_func=shade_plot_func, **kwargs)
            plt.show()

    def get_neurons_in_cluster_and_project(self, i_clust, project_name):
        """Get the names of the neurons in a cluster and project"""
        if self.df_traces is None:
            raise ValueError("df_traces is None")
        name_list = list(self.per_cluster_names[i_clust])
        # Only get the names that are in this project
        this_name_list = [name for name in name_list if project_name in name]
        return this_name_list

    def plot_multiple_clusters_simple(self, i_clust_list: List[int], min_lines=2, ind_preceding=None, z_score=False,
                                      show_individual_lines=False, show_shading_error_bars=True, xlim=None,
                                      use_dendrogram_colors=True, output_folder=None, behavior_shading_type=None,
                                      show_guide_lines=True, legend=False, **plot_kwargs):

        if ind_preceding is None:
            ind_preceding = self._ind_preceding

        # if xlim is not None:
        #     xlim = np.array(xlim) / fps

        #
        already_plotted_clusters = []
        fig, ax = plt.subplots(dpi=200, figsize=(5, 4))
        for i_clust in i_clust_list:
            if i_clust in already_plotted_clusters:
                continue
            name_list = list(self.per_cluster_names[i_clust])
            pseudo_mat = self.get_subset_triggered_average_matrix(name_list)
            # Normalize the traces to be similar to the correlation, i.e. z-score them
            if z_score:
                pseudo_mat = pseudo_mat - np.nanmean(pseudo_mat, axis=1, keepdims=True)
                pseudo_mat = pseudo_mat / np.nanstd(pseudo_mat, axis=1, keepdims=True)
            # Plot
            if use_dendrogram_colors:
                color = self.cluster_color_func(i_clust)
                plot_kwargs['color'] = color
            if show_guide_lines:
                is_second_plot = already_plotted_clusters != []
            else:
                is_second_plot = True
            ax, _ = plot_triggered_average_from_matrix_low_level(pseudo_mat, ind_preceding, min_lines,
                                                                 show_individual_lines=show_individual_lines,
                                                                 is_second_plot=is_second_plot, ax=ax,
                                                                 show_shading=show_shading_error_bars,
                                                                 xlim=xlim,
                                                                 label=f"Cluster {i_clust}", **plot_kwargs)
            already_plotted_clusters.append(i_clust)
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude (z-score)")
        plt.title(f"Cluster triggered average")
        if legend:
            plt.legend()

        index_conversion = pseudo_mat.columns  # Should always exist
        add_behavior_shading_to_plot(ind_preceding, index_conversion, behavior_shading_type, ax)

        self._save_plot(f"multiple_clusters_{i_clust_list}.png", output_folder)

    def get_optimal_clusters_using_hdbscan(self, min_cluster_size=10):
        """
        This is a different clustering algorithm that automatically detects the number of clusters

        Returns
        -------

        """

        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='precomputed')
        cluster_labels = clusterer.fit_predict(1 - self.df_corr)
        unique_labels, label_counts = np.unique(cluster_labels, return_counts=True)

        return len(unique_labels) - 1, label_counts

    def get_subset_triggered_average_matrix(self, name_list: List[str] = None) -> pd.DataFrame:
        # Build a pseudo-triggered average matrix, made of the means of each neuron
        pseudo_mat = []
        if name_list is None:
            name_list = list(self.df_triggered.keys())
        for name in name_list:
            triggered_avg = self.df_triggered[name].copy()
            pseudo_mat.append(triggered_avg)
        pseudo_mat = pd.concat(pseudo_mat, axis=1)
        pseudo_mat = pseudo_mat.T
        return pseudo_mat

    def get_triggered_matrix_all_events_from_names(self, name_list: List[str] = None) -> \
            Tuple[np.ndarray, Dict[str, List[int]]]:
        # Gets all the individual events from all neurons in the list
        if self.dict_of_triggered_traces is None:
            get_df_trigger = self.triggered_averages_class.triggered_average_matrix_from_name
        else:
            get_df_trigger = self.dict_of_triggered_traces.get
        if name_list is None:
            name_list = list(self.df_triggered.keys())
        clust_traces = [get_df_trigger(name) for name in name_list]
        # If there is a maximum length set, loop through and reduce the length of the traces
        # Loop through and pad array with nan if some arrays are shorter
        if self.max_trace_len is None:
            max_len = max([x.shape[1] for x in clust_traces])
        else:
            max_len = self.max_trace_len
        for i, trace in enumerate(clust_traces):
            if trace.shape[1] < max_len:
                clust_traces[i] = np.pad(trace, ((0, 0), (0, max_len - trace.shape[1])), mode='constant',
                                         constant_values=np.nan)
            elif trace.shape[1] > max_len:
                clust_traces[i] = trace[:, :max_len]
        # Build a dictionary mapping each neuron name to a list of indices in the matrix
        ind_of_each_neuron = {}
        offset = 0
        for name, traces in zip(name_list, clust_traces):
            ind_of_each_neuron[name] = list(range(offset, offset + traces.shape[0]))
            offset += traces.shape[0]
        # Stack along neuron axis, not time axis
        return np.vstack(clust_traces), ind_of_each_neuron

    def calculate_p_value_all_clusters(self, n_resamples=300):
        """
        Uses a permutation test on the max difference between traces to calculate a p value

        Note that permutation is done on individual triggered events, not of full neurons

        Returns
        -------

        """
        n = len(self.per_cluster_names)
        all_p_values = 0.05*np.ones((n, n))
        rng = np.random.default_rng()
        # Loop through all pairs of clusters
        for i_clust0 in tqdm(range(n), leave=False):
            name_list0 = list(self.per_cluster_names[i_clust0 + 1])
            traces0, name2ind0 = self.get_triggered_matrix_all_events_from_names(name_list0)
            for i_clust1 in range(i_clust0+1, n):
                name_list1 = list(self.per_cluster_names[i_clust1 + 1])
                traces1, name2ind1 = self.get_triggered_matrix_all_events_from_names(name_list1)
                pvalue = self.calculate_p_value_two_clusters(traces0, name2ind0, traces1, name2ind1,
                                                             rng, n_resamples)
                # Force symmetry
                all_p_values[i_clust0, i_clust1] = pvalue
                all_p_values[i_clust1, i_clust0] = pvalue

        return all_p_values

    def get_p_value_for_next_split(self, tree, min_size=4, n_resamples=300, min_fraction_datasets=0.8,
                                   verbose=0, DEBUG=False, **kwargs):
        """
        Meant to be called recursively from build_unsplittable_tree_dict

        Does several sanity checks, and returns a fake p value if any fail.
        Specifically, returns 1.0 for no split, and 0.0 for split

        If these pass, then it calculates the p value for the split using a permutation test

        Parameters
        ----------
        tree
        min_size
        verbose

        Returns
        -------

        """
        rng = np.random.default_rng()

        if tree.is_leaf():
            if verbose >= 1:
                print("Reached leaf; can't split")
            return 1.0
        if verbose >= 1:
            print(f"Checking split: {tree.id} -> {tree.left.id} + {tree.right.id}")

        # Get left and right sub clusters, then check p value
        left_ids = tree.left.pre_order(lambda x: x.id)
        right_ids = tree.right.pre_order(lambda x: x.id)

        # Check one: automatically split if ONLY one side is too small to calculate p_values
        # If they are both small, then leave it unsplit
        right_is_small = len(right_ids) < min_size
        left_is_small = len(left_ids) < min_size
        if right_is_small ^ left_is_small:
            if verbose >= 1:
                print("Single small cluster; splitting")
            return 0.0
        elif right_is_small and left_is_small:
            if verbose >= 1:
                print("Two small clusters; not splitting")
            return 1.0

        left_names = self.names[left_ids]
        left_traces, name2ind0 = self.get_triggered_matrix_all_events_from_names(left_names)

        right_names = self.names[right_ids]
        right_traces, name2ind1 = self.get_triggered_matrix_all_events_from_names(right_names)

        # Check two: if not enough datasets are represented in each side, then don't split
        total_number_of_datasets = self.number_of_datasets
        min_datasets_needed = int(min_fraction_datasets * total_number_of_datasets)
        num_left_datasets = count_unique_datasets_from_flattened_index(left_names)
        num_right_datasets = count_unique_datasets_from_flattened_index(right_names)
        print(f"num_left_datasets: {num_left_datasets}, num_right_datasets: {num_right_datasets}")
        if num_left_datasets < min_datasets_needed or num_right_datasets < min_datasets_needed:
            if verbose >= 1:
                print(f"Too few datasets in one side; not splitting")
            return 1.0

        # If all above tests are passed, then calculate p value using a permutation test
        p_value = self.calculate_p_value_two_clusters(left_traces, name2ind0,
                                                      right_traces, name2ind1, rng=rng,
                                                      n_resamples=n_resamples,
                                                      **kwargs,
                                                      DEBUG=DEBUG,
                                                      names0=left_names, names1=right_names,
                                                      DEBUG_str=f"{tree.id} -> {tree.left.id} + {tree.right.id}")
        if verbose >= 1:
            print(f"p-value: {p_value}")
        return p_value

    def build_clusters_using_p_values(self, tree=None, split_dict=None, p_value_threshold=0.05,
                                      recursion_level=0, verbose=0, DEBUG=False, **kwargs):
        """
        Returns a dictionary with keys of the tree ids, and values as a tuple:
        - the tree corresponding to the entire cluster
        - p-value at which the splitting stopped (should be > 0.05)

        Parameters
        ----------
        tree
        split_dict
        recursion_level
        verbose

        Returns
        -------

        """
        if tree is None:
            tree = hierarchy.to_tree(self.Z)

        if split_dict is None:
            split_dict = {}
        if verbose >= 1:
            print(f"Checking tree {tree.id} at recursion level {recursion_level}")
        p_value = self.get_p_value_for_next_split(tree, verbose=verbose - 1, DEBUG=DEBUG, **kwargs)

        if p_value > p_value_threshold:
            # Then it can't be split, and we stop the dfs
            split_dict[tree.id] = (tree, p_value)
        else:
            # Then it can be split, and we recurse using left and right
            opt = dict(p_value_threshold=p_value_threshold, recursion_level=recursion_level + 1,
                       verbose=verbose, DEBUG=DEBUG)
            opt.update(kwargs)
            split_dict = self.build_clusters_using_p_values(tree.left, split_dict=split_dict, **opt)
            split_dict = self.build_clusters_using_p_values(tree.right, split_dict=split_dict, **opt)

        return split_dict

    def cluster2neuron_from_split_dict(self, split_dict):
        """
        Turn the output of build_clusters_using_p_values into a dictionary mapping cluster ids to a list of neuron ids

        Parameters
        ----------
        split_dict

        Returns
        -------

        """
        names = self.names
        per_cluster_names = {}
        for clust_id, (tree, _) in split_dict.items():
            neuron_ids = tree.pre_order(lambda x: x.id)
            per_cluster_names[clust_id] = [names[i] for i in neuron_ids]
        return per_cluster_names

    @staticmethod
    def map_list_of_cluster_ids_to_colors(split_dict, cmap=None, min_size=3) -> Callable:
        """
        Turn cluster ids into colors, but need the ids of the linkage combinations, not the neurons

        split_dict is the output of build_clusters_using_p_values

        Meant to be used with link_color_func of hierarchy.dendrogram as:
        link_color_func = self.map_list_of_cluster_ids_to_colors(...)
        hierarchy.dendrogram(..., link_color_func=link_color_func)

        Parameters
        ----------
        split_dict
        cmap
        min_size

        Returns
        -------

        """
        if cmap is None:
            cmap = px.colors.qualitative.Plotly

        def assign_color_recursively(link_color_dict, tree, col):
            link_color_dict[tree.id] = col
            if not tree.is_leaf():
                assign_color_recursively(link_color_dict, tree.left, col)
                assign_color_recursively(link_color_dict, tree.right, col)
            return link_color_dict

        link_color_dict = {}
        keys = list(split_dict.keys())
        for i_clust, k in enumerate(keys):
            col = cmap[i_clust % len(cmap)]
            tree = split_dict[k][0]
            # If the cluster is too small, set the color to black
            if len(tree.pre_order(lambda x: x.id)) < min_size:
                col = 'black'
            # Assign color to top level split, then recursively for all leaves
            link_color_dict = assign_color_recursively(link_color_dict, tree, col)

        def link_color_func(node_id):
            return link_color_dict.get(node_id, 'black')

        return link_color_func

    @staticmethod
    def calculate_p_value_two_clusters(traces0, name2ind0, traces1, name2ind1,
                                       rng=None, n_resamples=300, z_score=False,
                                       DEBUG=False, DEBUG_str="",
                                       names0=None, names1=None):
        if rng is None:
            rng = np.random.default_rng()
        # all_traces should be an iterable of arrays, each row of each array being a trace
        all_traces = (traces0.T, traces1.T)
        from wbfm.utils.general.utils_clustering import ks_statistic
        res = permutation_test(all_traces, ks_statistic, vectorized=True,
                               n_resamples=n_resamples, axis=1, random_state=rng)
        if DEBUG:
            
            fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10))

            # Traces for each cluster with shading
            opt = dict(show_individual_lines=False, ax=axes[0], min_lines=2, ind_preceding=20,
                       z_score=z_score)
            plot_triggered_average_from_matrix_low_level(traces0, is_second_plot=False, **opt)
            plot_triggered_average_from_matrix_low_level(traces1, is_second_plot=True, **opt)

            # Histogram for permutation test
            axes[1].hist(res.null_distribution, bins=50)
            axes[1].vlines(x=ks_statistic(*all_traces, axis=-1), ymin=0, ymax=20, color='r')
            axes[1].set_title(f"Permutation distribution of test statistic (p={res.pvalue}) {DEBUG_str}")
            axes[1].set_xlabel("Value of Statistic")
            axes[1].set_ylabel("Frequency")

            # Histogram showing how many datasets are represented in each cluster

            # First, split the names of the traces to get the dataset name, then count
            # the number of times each dataset appears
            all_dataset_counts = {}
            for i, names in enumerate([names0, names1]):
                if names is None:
                    continue
                unflattened_dict0 = split_flattened_index(names)
                # Count how many times each dataset name appears
                dataset_counts = defaultdict(int)
                for key, (dataset_name, neuron_name) in unflattened_dict0.items():
                    dataset_counts[dataset_name] += 1
                all_dataset_counts[i] = dataset_counts
            df_counts = pd.DataFrame(all_dataset_counts)
            # Plot histogram
            df_counts.plot.bar(ax=axes[2])

            plt.show()
            
        return res.pvalue

    def plot_cluster_silhouette_scores(self, max_n_clusters=10, plot_individual_neuron_scores=False):
        """
        Plots silhouette scores for different number of clusters, for each neuron

        See: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

        Returns
        -------

        """
        Z = self.Z
        X = 1.0 - self.df_corr.to_numpy()
        range_n_clusters = np.arange(2, max_n_clusters)

        all_scores = []
        for n_clusters in range_n_clusters:
            if plot_individual_neuron_scores:
                fig, ax1 = plt.subplots(1, 1)
                fig.set_size_inches(18, 7)
                ax1.set_xlim([-0.1, 1])
                ax1.set_ylim([0, len(Z) + (n_clusters + 1) * 10])

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            cluster_labels = self.cluster_func(Z, t=n_clusters, criterion='maxclust')

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(X, cluster_labels, metric='precomputed')
            all_scores.append(silhouette_avg)

            if plot_individual_neuron_scores:
                # Compute the silhouette scores for each sample
                sample_silhouette_values = silhouette_samples(X, cluster_labels)

                y_lower = 10
                for i in range(n_clusters):
                    # Aggregate the silhouette scores for samples belonging to
                    # cluster i, and sort them
                    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i + 1]

                    ith_cluster_silhouette_values.sort()

                    size_cluster_i = ith_cluster_silhouette_values.shape[0]
                    y_upper = y_lower + size_cluster_i

                    color = cm.nipy_spectral(float(i) / n_clusters)
                    ax1.fill_betweenx(
                        np.arange(y_lower, y_upper),
                        0,
                        ith_cluster_silhouette_values,
                        facecolor=color,
                        edgecolor=color,
                        alpha=0.7,
                    )

                    # Label the silhouette plots with their cluster numbers at the middle
                    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                    # Compute the new y_lower for next plot
                    y_lower = y_upper + 10  # 10 for the 0 samples

                ax1.set_title(f"Silhouette scores for {n_clusters} clusters with average score {silhouette_avg:.2f}"
                              f" for {len(Z)} neurons")
                ax1.set_xlabel("Silhouette coefficient values (higher is better)")
                ax1.set_ylabel("Cluster label")

                # The vertical line for average silhouette score of all the values
                ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

                ax1.set_yticks([])  # Clear the yaxis labels / ticks
                ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.show()
        df = pd.DataFrame(dict(n_clusters=range_n_clusters, silhouette_score=all_scores))
        fig = px.line(df, title="Silhouette scores for different number of clusters (higher is better)", x="n_clusters",
                      y="silhouette_score")
        fig.show()

    def plot_silhouette_scores_and_points(self, max_n_clusters=10):
        """
        Plots silhouette scores for different number of clusters, for each neuron

        Includes a 2d pca plot of the neurons, with the silhouette scores as color

        See: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

        Returns
        -------

        """

        Z = self.Z
        X = self.df_corr.to_numpy()
        # X_normalized = StandardScaler().fit_transform(X)
        X_normalized = X
        # Build a 2d tsne embedding
        # from tsnecuda import TSNE
        # tsne = TSNE()
        # X_pca = tsne.fit_transform(X_normalized)
        X_pca = PCA(n_components=2).fit_transform(X_normalized)

        range_n_clusters = np.arange(2, max_n_clusters)
        for n_clusters in range_n_clusters:
            # Create a subplot with 1 row and 2 columns
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)

            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            cluster_labels = self.cluster_func(Z, t=n_clusters, criterion='maxclust')

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(X, cluster_labels)
            print(
                "For n_clusters =",
                n_clusters,
                "The average silhouette_score is :",
                silhouette_avg,
            )

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(X, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i + 1]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i + 1) / n_clusters)
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.scatter(
                X_pca[:, 0], X_pca[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
            )

            # Labeling the clusters
            # centers = clusterer.cluster_centers_
            # # Draw white circles at cluster centers
            # ax2.scatter(
            #     centers[:, 0],
            #     centers[:, 1],
            #     marker="o",
            #     c="white",
            #     alpha=1,
            #     s=200,
            #     edgecolor="k",
            # )

            # for i, c in enumerate(centers):
            #     ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")

            plt.suptitle(
                "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
                % n_clusters,
                fontsize=14,
                fontweight="bold",
            )

        plt.show()

    def plot_cluster_alternate_score(self, max_n_clusters=10, score_func='calinski_harabasz_score'):
        """
        See: plot_cluster_silhouette_scores

        Note that silhouette uses the correlation as a distance metric, while calinski_harabasz
        uses the euclidean distance between points, which should be z-scored

        Returns
        -------

        """
        if score_func == 'calinski_harabasz_score':
            func = sklearn.metrics.calinski_harabasz_score
            better_str = "(higher is better)"
        elif score_func == 'davies_bouldin_score':
            func = sklearn.metrics.davies_bouldin_score
            better_str = "(lower is better)"
        else:
            raise ValueError("score_func must be either 'calinski_harabasz_score' or 'davies_bouldin_score'")
        Z = self.Z
        X = self.df_triggered.to_numpy()
        # z-score the data
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        X = X.T
        range_n_clusters = np.arange(2, max_n_clusters)

        all_scores = []
        for n_clusters in range_n_clusters:
            cluster_labels = self.cluster_func(Z, t=n_clusters, criterion='maxclust')
            score = func(X, cluster_labels)
            all_scores.append(score)

        df = pd.DataFrame(dict(n_clusters=range_n_clusters, score=all_scores))
        fig = px.line(df, title=f"{score_func} for different number of clusters {better_str}", x="n_clusters",
                      y="score")
        fig.show()

    @staticmethod
    def load_from_project(project_data, trigger_opt=None, trace_opt=None, to_filter=True,
                          **kwargs):
        """
        Convenience function to load from a project

        Default: triggers to FWD. To change this, pass in trigger_opt. Example:
            trigger_opt = dict(min_duration=10, state=BehaviorCodes.REV, ind_preceding=30)

        See FullDatasetTriggeredAverages for full documentation of trigger_opt

        Parameters
        ----------
        project_data
        trigger_opt
        trace_opt
        to_filter
        kwargs

        Returns
        -------

        """
        if trigger_opt is None:
            trigger_opt = {}
        if trace_opt is None:
            trace_opt = {}
        # default_trigger_opt = dict(min_duration=10, state=BehaviorCodes.REV, ind_preceding=30)
        default_trigger_opt = {}
        default_trigger_opt.update(trigger_opt)
        default_trace_opt = {}
        default_trace_opt.update(trace_opt)
        triggered_averages_class = FullDatasetTriggeredAverages.load_from_project(project_data,
                                                                                  trigger_opt=default_trigger_opt,
                                                                                  trace_opt=default_trace_opt)

        # Strongly filter to clean up the correlation matrix
        df = triggered_averages_class.df_traces.copy()
        if to_filter:
            triggered_averages_class.df_traces = filter_gaussian_moving_average(df, std=2)

        return ClusteredTriggeredAverages.load_from_triggered_average_class(triggered_averages_class, **kwargs)

    @staticmethod
    def load_from_triggered_average_class(triggered_averages_class, **kwargs):
        df_triggered = triggered_averages_class.df_of_all_triggered_averages()
        return ClusteredTriggeredAverages(df_triggered, triggered_averages_class=triggered_averages_class,
                                          **kwargs)

    def calc_dataframe_of_manual_ids_per_cluster(self, all_projects=None):
        """
        Calculate a dataframe of manual ids per cluster

        Parameters
        ----------
        all_projects

        Returns
        -------

        """
        per_cluster_names = self.per_cluster_names
        per_id_counts_per_cluster = defaultdict(lambda: defaultdict(int))
        for key_clust, clust_names in per_cluster_names.items():
            for name in clust_names:
                dataset_name, neuron_name = split_flattened_index([name])[name]

                if all_projects is None:
                    # Assume that the neurons are already renamed to have the manual id, if any
                    manual_name = None if 'neuron' in neuron_name else neuron_name
                else:
                    # Get project, and check if this neuron has a manually annotated name
                    p = all_projects[dataset_name]
                    mapping = p.neuron_name_to_manual_id_mapping(confidence_threshold=1, remove_unnamed_neurons=True)
                    manual_name = mapping.get(neuron_name, None)
                if manual_name is not None:
                    # clust_names_manual.append(manual_name)
                    per_id_counts_per_cluster[manual_name][f"cluster_{key_clust:02d}"] += 1
        df_id_counts = pd.DataFrame(per_id_counts_per_cluster).sort_index()
        return df_id_counts

    def plot_manual_ids_per_cluster(self, all_projects, use_bar_plot=True, neuron_threshold=0,
                                    neuron_subset='paper', normalize_by_number_of_ids=False, legend=False,
                                    combine_left_right=False, allow_temporary_names=False,
                                    output_folder=None, **kwargs):
        """
        Plots a bar chart of the number of neurons per manual ID per cluster

        Parameters
        ----------
        all_projects
        use_bar_plot
        neuron_threshold - if > 0, only plot neurons with at least this many identifications
        normalize_by_max - if True, normalize each neuron by the max number of identifications it has
        output_folder
        kwargs

        Returns
        -------

        """
        # Does not need the projects, because names are already renamed to have the manual id
        df_id_counts = self.calc_dataframe_of_manual_ids_per_cluster(all_projects=None)
        if not allow_temporary_names:
            # Remove names with an underscore, which are temporary names
            df_id_counts = df_id_counts.loc[:, [i for i in df_id_counts.columns if '_' not in i]]

        if neuron_subset is not None:
            if isinstance(neuron_subset, str) and neuron_subset == 'paper':
                neuron_subset = neurons_with_confident_ids()
            df_id_counts = df_id_counts.loc[:, neuron_subset]

        if not use_bar_plot:
            fig = px.imshow(df_id_counts, title=f"Number of neurons per manual ID per cluster", **kwargs)
            fig.show()
        else:
            df_id_counts_sparse = pd.melt(df_id_counts, ignore_index=False).dropna().reset_index()
            df_id_counts_sparse.columns = ['cluster', 'neuron', 'count']

            # Plotly wants a string for the color
            # https://stackoverflow.com/questions/63460213/how-to-define-colors-in-a-figure-using-plotly-graph-objects-and-plotly-express
            # color_map = {name: f"rgba{self.cluster_color_func(i + 1)}" for i, name in enumerate(cluster_names)}
            # fig.show()

            # We want a color for every cluster, even if there isn't a plot here
            cluster_names = list(self.per_cluster_names.keys())
            # Remove singleton clusters
            cluster_names = [i for i in cluster_names if len(self.per_cluster_names[i]) > 1]
            color_sequence = [self.cluster_color_func(i + 1) for i, name in enumerate(cluster_names)]
            custom_cmap = LinearSegmentedColormap.from_list('clusters', color_sequence)

            # However, we also need to add an empty row for any clusters that are not present
            # But these ids are renamed to be strings like 'cluster_01', so we need to convert back from ints
            for i in cluster_names:
                name = f"cluster_{i:02d}"
                if name not in df_id_counts.index:
                    df_id_counts.loc[name, :] = np.nan
            # Then re-sort
            df_id_counts = df_id_counts.sort_index()

            # Postprocess: filter out too few, and normalize
            if neuron_threshold > 0:
                df_id_counts = df_id_counts.loc[:, df_id_counts.sum(axis=0) >= neuron_threshold]
            df_id_counts = df_id_counts.dropna(axis='columns', how='all')

            if combine_left_right:
                df_id_counts = combine_columns_with_suffix(df_id_counts, suffixes=['L', 'R'], how='sum')

            if normalize_by_number_of_ids:
                df_id_counts = df_id_counts / df_id_counts.sum(axis=0)

            # Sort the columns alphabetically
            df_id_counts = df_id_counts.reindex(sorted(df_id_counts.columns), axis=1)

            # Final plot
            fig, ax = plt.subplots(dpi=200, figsize=(5, 2))
            df_id_counts.T.plot(kind='bar', stacked=True, colormap=custom_cmap, ax=ax)
            if not legend:
                ax.get_legend().remove()

            plt.xlabel("Manual ID")
            plt.title("Cluster Membership")
            if normalize_by_number_of_ids:
                plt.ylabel("Fraction")
            else:
                plt.ylabel("Count")

            # Apply paper settings
            apply_figure_settings(fig, width_factor=0.5, height_factor=0.2, plotly_not_matplotlib=False)

            self._save_plot(f"manual_ids_per_cluster.png", output_folder=output_folder)
        return df_id_counts


def ax_plot_func_for_grid_plot(t, y, ax, name, project_data, state, min_lines=4, **kwargs):
    """
    Designed to be used with make_grid_plot_using_project with the arg ax_plot_func=ax_plot_func
    Note that you must create a closure to remove the following args, and pass a lambda:
        project_data
        state
        min_lines

    Example:
    from functools import partial
    func = partial(ax_plot_func_for_grid_plot, project_data=p, state=1)

    Parameters
    ----------
    state: the state whose onset is calculated as the trigger
    min_lines: the minimum number of lines that must exist for a line to be plotted
    project_data
    t: time vector (unused)
    y: full trace (1d)
    ax: matplotlib axis
    name: neuron name
    kwargs

    Returns
    -------

    """
    plot_kwargs = dict(label=name)
    plot_kwargs.update(kwargs)

    ind_preceding = 20
    worm_class = project_data.worm_posture_class

    ind_class = worm_class.calc_triggered_average_indices(state=state, ind_preceding=ind_preceding,
                                                          min_lines=min_lines)
    mat = ind_class.calc_triggered_average_matrix(y)
    ind_class.plot_triggered_average_from_matrix(mat, ax, **plot_kwargs)


def assign_id_based_on_closest_onset_in_split_lists(class1_onsets, class0_onsets, rev_onsets) -> dict:
    """
    Assigns each reversal a class based on which list contains an event closes to that reversal

    Note if a reversal has no previous forward, it will be removed!

    Parameters
    ----------
    class1_onsets
    class0_onsets
    rev_onsets

    Returns
    -------

    """
    raise ValueError("Not working! See test")
    dict_of_rev_with_id = {}
    for rev in rev_onsets:
        # For both forward lists, get the previous indices
        these_class0 = class0_onsets.copy() - rev
        these_class0 = these_class0[these_class0 < 0]

        these_class1 = class1_onsets.copy() - rev
        these_class1 = these_class1[these_class1 < 0]

        # Then the smaller absolute one (closer in time) one gives the class
        only_prev_short = len(these_class0) == 0 and len(these_class1) > 0
        only_prev_long = len(these_class1) == 0 and len(these_class0) > 0
        # Do not immediately calculate, because the list may be empty
        short_is_closer = lambda: np.min(np.abs(these_class0)) < np.min(np.abs(these_class1))
        if only_prev_short:
            dict_of_rev_with_id[rev] = 0
        elif only_prev_long:
            dict_of_rev_with_id[rev] = 1
        elif short_is_closer():
            # Need to check the above two conditions before trying to evaluate this
            dict_of_rev_with_id[rev] = 0
        else:
            dict_of_rev_with_id[rev] = 1

        # Optimization: Finally, remove the used one from the fwd onset list

    return dict_of_rev_with_id


def build_ind_matrix_from_starts_and_ends(all_starts: List[int], all_ends: List[int], ind_preceding: int,
                                          validity_checks=None, DEBUG=False):
    """
    Builds a matrix of indices, where each row is a block of indices corresponding to a start and end



    Parameters
    ----------
    all_ends
    all_starts
    ind_preceding
    validity_checks
    DEBUG

    Returns
    -------

    """
    if validity_checks is None:
        validity_checks = []
    all_ind = []
    for start, end in zip(all_starts, all_ends):
        if DEBUG:
            print("Checking block: ", start, end)
        # Check validity
        validity_vec = [check(start, end) for check in validity_checks]
        if any(validity_vec):
            if DEBUG:
                print("Skipping because: ", validity_vec)
            continue
        elif DEBUG:
            print("***Keeping***")
        ind = np.arange(start - ind_preceding, end)
        all_ind.append(ind)
    if DEBUG:
        print(f"Final indices: {all_ind}")
    return all_ind


def calc_time_series_from_starts_and_ends(all_starts, all_ends, num_pts, min_duration=0, only_onset=False):
    """
    Calculates a time series from a list of starts and ends

    Example:
    all_starts = [0, 10, 20]
    all_ends = [5, 15, 25]
    num_pts = 30
    min_duration = 0
    only_onset = False

    Then the output will be:
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, ...]

    Parameters
    ----------
    all_starts
    all_ends
    num_pts
    min_duration
    only_onset

    Returns
    -------

    """
    state_trace = np.zeros(num_pts)
    for start, end in zip(all_starts, all_ends):
        if end - start < min_duration:
            continue

        if not only_onset:
            state_trace[start:end] = 1
        else:
            state_trace[start] = 1
    return state_trace


def clustered_triggered_averages_from_dict_of_projects(all_projects: dict, cluster_opt=None, verbose=0, **kwargs) \
        -> Tuple[ClusteredTriggeredAverages,
        Tuple[Dict[str, FullDatasetTriggeredAverages], pd.DataFrame, Dict[str, pd.DataFrame]]]:
    """
    See ClusteredTriggeredAverages.load_from_project for kwargs

    if trigger_opt is in kwargs, it will be passed to calc_triggered_average_indices
    if trace_opt is in kwargs, it will be passed to calc_default_traces

    Parameters
    ----------
    all_projects
    kwargs

    Returns
    -------

    """
    # First calculate triggered average classes for each project
    if cluster_opt is None:
        cluster_opt = {}
    all_triggered_average_classes = {}
    trigger_opt_default = {'state': BehaviorCodes.FWD}
    if 'trigger_opt' in kwargs:
        if kwargs['trigger_opt'] is not None:
            trigger_opt_default.update(kwargs['trigger_opt'])
        kwargs.pop('trigger_opt')

    for name, p in tqdm(all_projects.items(), leave=False, desc='Precalculating traces and triggered average indices'):
        try:
            triggered_averages_class = FullDatasetTriggeredAverages.load_from_project(p, trigger_opt=trigger_opt_default,
                                                                                      **kwargs)
            all_triggered_average_classes[name] = triggered_averages_class
        except NoBehaviorAnnotationsError:
            print(f"Skipping {p.project_dir} for behavior {trigger_opt_default['state']} because it has no or incomplete behavior annotations")

    if len(all_triggered_average_classes) == 0:
        raise NoBehaviorAnnotationsError("No datasets had the requested behavior annotation")

    tqdm_opt = dict(leave=False, disable=not verbose)
    # Combine all triggered averages dataframes, renaming to contain dataset information
    df_triggered_good = pd.concat(
        {name: c.df_of_all_triggered_averages() for name, c in tqdm(all_triggered_average_classes.items(), **tqdm_opt, desc='Combining triggered averages')}, axis=1)
    df_triggered_good = flatten_multiindex_columns(df_triggered_good)

    # Combine all full traces dataframes, renaming to contain dataset information
    df_traces_good = pd.concat(
        {name: c.df_traces for name, c in tqdm(all_triggered_average_classes.items(), **tqdm_opt, desc='Combining full traces')}, axis=1)
    df_traces_good = flatten_multiindex_columns(df_traces_good)

    # Combine all behavior time series, renaming to contain dataset information
    df_behavior = pd.concat(
        {name: c.ind_class.behavioral_annotation for name, c in tqdm(all_triggered_average_classes.items(), **tqdm_opt, desc='Combining behavior annotations')}, axis=1)
    # This one doesn't need to be flattened, because each dataset only has one column

    # Build a map back to the original data
    dict_of_triggered_traces = {}
    for name, c in tqdm(all_triggered_average_classes.items(), **tqdm_opt, desc='Building map to original data'):
        c.ind_class.z_score = False
        dict_of_triggered_traces[name] = c.dict_of_all_triggered_averages()
    dict_of_triggered_traces = flatten_nested_dict(dict_of_triggered_traces)

    # Check that the ind_preceding is the same between all ind_class, and save it
    ind_preceding = None
    for name, c in tqdm(all_triggered_average_classes.items(), **tqdm_opt, desc='Validating indices'):
        if ind_preceding is None:
            ind_preceding = c.ind_class.ind_preceding
        else:
            assert ind_preceding == c.ind_class.ind_preceding, "ind_preceding must be the same for all datasets"

    # Build a combined class
    # I'm not actually using the clustering functionality, this is just an old class
    default_cluster_opt = dict(linkage_threshold=12, verbose=1)
    default_cluster_opt.update(cluster_opt)
    if verbose > 1:
        print("Building final triggered average class")
    good_dataset_clusterer = ClusteredTriggeredAverages(df_triggered_good, **default_cluster_opt,
                                                        dict_of_triggered_traces=dict_of_triggered_traces,
                                                        _df_traces=df_traces_good,
                                                        _df_behavior=df_behavior,
                                                        _ind_preceding=ind_preceding)

    return good_dataset_clusterer, (all_triggered_average_classes, df_triggered_good, dict_of_triggered_traces)


def calc_p_value_using_ttest_triggered_average(df_triggered, gap=0):
    """
    Given a dataframe with a triggered event at index=0, calculate the p-value of the mean difference between before and
    after the event

    Parameters
    ----------
    df_triggered
    gap

    Returns
    -------

    """
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        means_before = np.nanmean(df_triggered.loc[:-gap, :], axis=1)
        means_after = np.nanmean(df_triggered.loc[gap:, :], axis=1)
    p = scipy.stats.ttest_rel(means_before, means_after, nan_policy='omit').pvalue
    effect_size = np.nanmean(means_after) - np.nanmean(means_before)

    return dict(p_value=p, effect_size=effect_size, means_before=means_before, means_after=means_after)
