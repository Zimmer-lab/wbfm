import logging
import os
import threading
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Dict, Tuple
from copy import deepcopy
import napari
import numpy as np
import pandas as pd
import sklearn.linear_model
import zarr
from backports.cached_property import cached_property
from wbfm.utils.external.utils_pandas import cast_int_or_nan
from matplotlib import pyplot as plt

from wbfm.utils.general.utils_piecewise import predict_using_rolling_ransac_filter_single_trace
from wbfm.utils.tracklets.high_performance_pandas import get_names_from_df
from wbfm.utils.tracklets.utils_tracklets import get_time_overlap_of_candidate_tracklet, \
    split_tracklet_within_sparse_dataframe
from wbfm.utils.tracklets.tracklet_class import DetectedTrackletsAndNeurons
from segmentation.util.utils_metadata import DetectedNeurons
from segmentation.util.utils_postprocessing import split_neuron_interactive
from wbfm.gui.utils.utils_gui import build_tracks_from_dataframe
from wbfm.utils.projects.project_config_classes import SubfolderConfigFile
from wbfm.utils.projects.utils_filenames import read_if_exists, pickle_load_binary, get_sequential_filename
from wbfm.utils.visualization.filtering_traces import trace_from_dataframe_factory, \
    filter_rolling_mean, filter_linear_interpolation, remove_outliers_using_std, filter_exponential_moving_average
from wbfm.utils.traces.bleach_correction import detrend_exponential_lmfit
from wbfm.utils.visualization.utils_plot_traces import correct_trace_using_linear_model


@dataclass
class TracePlotter:
    # Data
    red_traces: pd.DataFrame
    green_traces: pd.DataFrame
    final_tracks: pd.DataFrame

    # Settings
    channel_mode: str
    calculation_mode: str

    remove_outliers: bool = False
    filter_mode: str = 'no_filtering'
    bleach_correct: bool = True
    min_confidence: float = None
    background_per_pixel: float = None
    preprocess_volume_correction: bool = False  # Alternate way to subtract background

    tspan: list = None

    # For experimental methods of trace calculation
    alternate_dataframe_folder: str = None
    alternate_dataframe_mode: str = None
    alternate_column_name: str = None

    verbose: int = 1

    def __post_init__(self):
        if self.tspan is None:
            self.tspan = list(range(self.red_traces.shape[0]))

    def calculate_traces(self, neuron_name: str) -> pd.Series:
        """
        First step when plotting, with the mode saved as a class variable

        For extending with new calculation modes, make a function with the signature:
            y = f(neuron_name, df_traces)
        Where df_traces is, for example, project_data.red_traces
        and y is a pd.Series

        Parameters
        ----------
        neuron_name

        Returns
        -------

        """
        valid_modes = ['green', 'red', 'ratio', 'linear_model',
                       'df_over_f_20', 'ratio_df_over_f_20', 'dr_over_r_20', 'dr_over_r_50',
                       'linear_model_then_ratio', 'ratio_then_linear_model', 'high_order_linear_model',
                       'cross_term_linear_model',
                       'green_rolling_ransac', 'ratio_rolling_ransac',
                       'top_pixels_10_percent']
        assert (self.channel_mode in valid_modes), \
            f"Unknown channel mode {self.channel_mode}, must be one of {valid_modes}"

        if self.verbose >= 3:
            print(f"Calculating {self.channel_mode} trace for {neuron_name} for {self.calculation_mode} mode")

        ##
        ## Function for getting a single time series (with preprocessing)
        ##
        # Format: y = f(neuron_name, traces_dataframe)
        if self.alternate_column_name is None:
            column_name = 'intensity_image'
        else:
            column_name = self.alternate_column_name
        single_trace_preprocessed = trace_from_dataframe_factory(self.calculation_mode,
                                                                 self.background_per_pixel,
                                                                 self.bleach_correct,
                                                                 self.preprocess_volume_correction,
                                                                 column_name=column_name)

        if not self.bleach_correct:
            def calc_single_df_over_f(i, _df) -> pd.Series:
                _y = single_trace_preprocessed(i, _df)
                y0 = np.nanquantile(_y, 0.2)
                return pd.Series((_y-y0) / y0)
        else:
            def calc_single_df_over_f(i, _df) -> pd.Series:
                _y = single_trace_preprocessed(i, _df)
                _y, _ = pd.Series(detrend_exponential_lmfit(_y))
                y0 = np.nanquantile(_y, 0.2)
                return pd.Series((_y-y0) / y0)

        ##
        ## Function for getting final y value from above functions
        ##
        # How to combine channels, or which channel to choose
        if self.calculation_mode == 'likelihood':
            # First: use the tracks dataframe, not the traces ones
            df = self.final_tracks

            def calc_y(i) -> pd.Series:
                return single_trace_preprocessed(i, df)

        elif self.channel_mode in ['red', 'green', 'df_over_f_20']:
            # Second: use a single traces dataframe (red OR green)
            df = self.get_single_dataframe_for_traces()

            if self.channel_mode in ['red', 'green']:
                def calc_y(i) -> pd.Series:
                    return single_trace_preprocessed(i, df)
            elif self.channel_mode == 'df_over_f_20':
                def calc_y(i) -> pd.Series:
                    return calc_single_df_over_f(i, df)
            else:
                raise NotImplementedError

        elif self.channel_mode in ['ratio', 'ratio_df_over_f_20', 'dr_over_r_20', 'dr_over_r_50'] or \
                'linear_model' in self.channel_mode or 'ransac' in self.channel_mode:
            # Third: use both traces dataframes (red AND green)
            df_red, df_green = self.get_two_dataframes_for_traces()

            if self.channel_mode == 'ratio':
                def calc_y(i) -> pd.Series:
                    return single_trace_preprocessed(i, df_green) / single_trace_preprocessed(i, df_red)

            elif self.channel_mode == 'ratio_df_over_f_20':
                def calc_y(i) -> pd.Series:
                    return calc_single_df_over_f(i, df_green) / calc_single_df_over_f(i, df_red)

            elif self.channel_mode == 'linear_model':
                assert self.alternate_dataframe_mode is None, "Not yet implemented"

                def calc_y(_neuron_name) -> pd.Series:
                    y_result_including_na = correct_trace_using_linear_model(df_red, df_green, _neuron_name)
                    return y_result_including_na

            elif self.channel_mode == 'linear_model_then_ratio':
                def calc_y(_neuron_name) -> pd.Series:
                    # predictor_names = ['t', 'area', 'intensity_image_over_area']
                    # predictor_names = ['x', 'y', 'z', 't', 'area', 'intensity_image_over_area']
                    predictor_names = ['x', 'y', 'z', 't', 't_squared', 'area', 'area_squared',
                                       'intensity_image_over_area']
                    opt = dict(predictor_names=predictor_names, neuron_name=_neuron_name, remove_intercept=False)
                    y_green = correct_trace_using_linear_model(df_red, df_green, **opt)
                    y_red = correct_trace_using_linear_model(df_red, df_red, **opt)
                    y_result_including_na = y_green / y_red
                    return y_result_including_na

            elif self.channel_mode == 'ratio_then_linear_model':

                def calc_y(_neuron_name) -> pd.Series:
                    y_ratio = single_trace_preprocessed(_neuron_name, df_green) / \
                              single_trace_preprocessed(_neuron_name, df_red)
                    predictor_names = ['x', 'y', 'z', 't', 't_squared', 'area', 'area_squared']
                    opt = dict(predictor_names=predictor_names, neuron_name=_neuron_name, remove_intercept=False)
                    y_result_including_na = correct_trace_using_linear_model(df_red, y_ratio, **opt)
                    return y_result_including_na

            elif self.channel_mode == 'high_order_linear_model':

                def calc_y(_neuron_name) -> pd.Series:
                    predictor_names = ['x', 'y', 'z', 'z_squared',
                                       't', 't_squared', 't_cubed',
                                       'area', 'area_squared', 'area_cubed',
                                       'intensity_image', 'intensity_image_squared', 'intensity_image_cubed']
                    opt = dict(predictor_names=predictor_names, neuron_name=_neuron_name, remove_intercept=False)
                    y_result_including_na = correct_trace_using_linear_model(df_red, df_green, **opt)
                    return y_result_including_na

            elif self.channel_mode == 'cross_term_linear_model':

                def calc_y(_neuron_name) -> pd.Series:
                    predictor_names = ['x', 'y', 'z', 'z_squared',
                                       't', 't_squared',
                                       'area', 'area_squared', 'area_cubed',
                                       'intensity_image', 'intensity_image_squared', 'intensity_image_cubed',
                                       'area_times_intensity_image', 'z_times_intensity_image']
                    opt = dict(predictor_names=predictor_names, neuron_name=_neuron_name, remove_intercept=False)
                    y_result_including_na = correct_trace_using_linear_model(df_red, df_green, **opt)
                    return y_result_including_na

            elif self.channel_mode == 'dr_over_r_20':
                def calc_y(i) -> pd.Series:
                    ratio = single_trace_preprocessed(i, df_green) / single_trace_preprocessed(i, df_red)
                    r0 = np.nanquantile(ratio, 0.2)
                    dr_over_r = (ratio - r0) / r0
                    return pd.Series(dr_over_r)
            elif self.channel_mode == 'dr_over_r_50':
                def calc_y(i) -> pd.Series:
                    ratio = single_trace_preprocessed(i, df_green) / single_trace_preprocessed(i, df_red)
                    r0 = np.nanmedian(ratio)
                    dr_over_r = (ratio - r0) / r0
                    return pd.Series(dr_over_r)

            elif self.channel_mode == 'ratio_rolling_ransac':
                def calc_y(i) -> pd.Series:
                    _green = single_trace_preprocessed(i, df_green)
                    _red = single_trace_preprocessed(i, df_red)
                    # Remove nan
                    valid_ind = _red.dropna().index.intersection(_green.dropna().index)
                    green_predicted = predict_using_rolling_ransac_filter_single_trace(_red[valid_ind],
                                                                                       _green[valid_ind])
                    return pd.Series(_green[valid_ind] / green_predicted, index=valid_ind)
            elif self.channel_mode == 'green_rolling_ransac':
                def calc_y(i) -> pd.Series:
                    _green = single_trace_preprocessed(i, df_green)
                    _red = single_trace_preprocessed(i, df_red)
                    valid_ind = _red.dropna().index.intersection(_green.dropna().index)
                    green_predicted = predict_using_rolling_ransac_filter_single_trace(_red[valid_ind],
                                                                                       _green[valid_ind])
                    return pd.Series(_green[valid_ind] - green_predicted, index=valid_ind)
            else:
                raise NotImplementedError

        else:
            raise ValueError("Unknown calculation or channel mode")

        y = calc_y(neuron_name)

        ##
        ## Other postprocessing
        ##
        # Then remove outliers and / or filter
        if self.min_confidence is not None:
            low_confidence = self.final_tracks[neuron_name]['likelihood'] < self.min_confidence
            nan_confidence = np.isnan(self.final_tracks[neuron_name]['likelihood'])
            outliers_from_tracking = np.logical_or(low_confidence, nan_confidence)
            y[outliers_from_tracking] = np.nan

        if self.remove_outliers:
            # y = remove_outliers_via_rolling_mean(y, window=9)
            y = remove_outliers_using_std(y, std_factor=5)

        if self.filter_mode == "rolling_mean":
            y = filter_rolling_mean(y, window=5)
        elif self.filter_mode == "linear_interpolation":
            y = filter_linear_interpolation(y, window=15)
        elif self.filter_mode == "3d_pca":
            y = filter_exponential_moving_average(y)
        elif self.filter_mode == "no_filtering":
            pass
        else:
            logging.warning(f"Unrecognized filter mode: {self.filter_mode}")

        return y

    def calculate_traces_full_dataframe(self, min_percent_nonzero=None, df=None, names=None):
        """Uses saved options to calculate the traces for all neurons"""

        if names is None:
            if df is None:
                if min_percent_nonzero is not None:
                    thresh = min_percent_nonzero * self.red_traces.shape[0]
                    df = self.red_traces.dropna(axis=1, thresh=thresh, inplace=False)
                    names = get_names_from_df(df)
                else:
                    names = get_names_from_df(self.red_traces)
            else:
                names = get_names_from_df(df)

        trace_dict = {}
        for name in names:
            trace_dict[name] = self.calculate_traces(name)

        df_traces = pd.DataFrame(trace_dict)
        return df_traces

    @cached_property
    def alternate_dataframes(self):
        # First, read the new dataframes from the files
        if self.alternate_dataframe_mode == 'top_pixels_10_percent':
            self.alternate_column_name = 'top10percent'
            red_fname = os.path.join(self.alternate_dataframe_folder, f'df_top_0-1_red.h5')
            green_fname = os.path.join(self.alternate_dataframe_folder, f'df_top_0-1_green.h5')
        elif self.alternate_dataframe_mode == 'top_pixels_25_percent':
            self.alternate_column_name = 'top25percent'
            red_fname = os.path.join(self.alternate_dataframe_folder, f'df_top_0-25_red.h5')
            green_fname = os.path.join(self.alternate_dataframe_folder, f'df_top_0-25_green.h5')
        elif self.alternate_dataframe_mode == 'top_pixels_50_percent':
            self.alternate_column_name = 'top50percent'
            red_fname = os.path.join(self.alternate_dataframe_folder, f'df_top_0-5_red.h5')
            green_fname = os.path.join(self.alternate_dataframe_folder, f'df_top_0-5_green.h5')
        else:
            raise NotImplementedError(f"Unknown type: {self.channel_mode}")
        df_red, df_green = pd.read_hdf(red_fname), pd.read_hdf(green_fname)

        # Second, concatenate with the old metadata
        df_red.columns = pd.MultiIndex.from_product([df_red.columns, [self.alternate_column_name]])
        df_green.columns = pd.MultiIndex.from_product([df_green.columns, [self.alternate_column_name]])

        df_red_full = pd.concat([df_red, self.red_traces], axis=1)
        df_green_full = pd.concat([df_green, self.green_traces], axis=1)

        return df_red_full, df_green_full

    def get_single_dataframe_for_traces(self):
        """If the trace uses only a single dataframe, this switches between which base"""
        if self.alternate_dataframe_mode is None:
            if self.channel_mode == 'red':
                df = self.red_traces
            else:
                df = self.green_traces
        else:
            df_red, df_green = self.alternate_dataframes
            if self.channel_mode == 'red':
                df = df_red
            else:
                df = df_green
        return df.copy()

    def get_two_dataframes_for_traces(self):
        """If the trace uses both dataframes, this switches between which base"""
        if self.alternate_dataframe_mode is None:
            df_red, df_green = self.red_traces, self.green_traces
        else:
            df_red, df_green = self.alternate_dataframes
        return df_red.copy(), df_green.copy()


@dataclass
class TrackletAndSegmentationAnnotator:
    df_tracklet_obj: DetectedTrackletsAndNeurons
    global2tracklet: Dict[str, List[str]]
    segmentation_metadata: DetectedNeurons

    # Same as global2tracklet, but updated with all manual changes
    _combined_global2tracklet_dict: Dict[str, List[str]] = None

    # Annotation
    manual_global2tracklet_names: Dict[str, List[str]] = None
    manual_global2tracklet_removals: Dict[str, List[str]] = None
    current_neuron: str = None
    current_tracklet_name: Union[str, None] = None
    previous_tracklet_name: Union[str, None] = None

    tracklet_split_names: Dict[str, List[str]] = None
    tracklet_split_times: Dict[str, List[Tuple[int, int]]] = None

    gt_mismatches: dict = None

    # Saving
    training_cfg: SubfolderConfigFile = None
    tracking_cfg: SubfolderConfigFile = None

    output_match_fname: str = None
    output_df_fname: str = None
    tracklet_split_names_fname: str = None
    tracklet_split_times_fname: str = None

    saving_lock: threading.Lock = threading.Lock()

    # New: for interactive segmentation splitting and merging
    candidate_mask: np.ndarray = None
    time_of_candidate: int = None
    indices_of_original_neurons: List[int] = None

    buffer_masks: zarr.Array = None
    t_buffer_masks: List[int] = None

    segmentation_options: dict = None

    # Visualization options
    segmentation_callbacks: List[callable] = None
    tracklet_callbacks: List[callable] = None
    to_add_layer_to_viewer: bool = True
    verbose: int = 1

    last_clicked_position: list = None
    z_to_xy_ratio: float = 1.0

    is_currently_interactive: bool = True

    logger: logging.Logger = None

    def __post_init__(self):
        if self.manual_global2tracklet_names is None:
            self.manual_global2tracklet_names = defaultdict(list)
        if self.manual_global2tracklet_removals is None:
            self.manual_global2tracklet_removals = defaultdict(list)
        if self.t_buffer_masks is None:
            self.t_buffer_masks = []
        if self._combined_global2tracklet_dict is None:
            self._combined_global2tracklet_dict = deepcopy(self.global2tracklet)

        if not self.df_tracklet_obj.interactive_mode:
            self.df_tracklet_obj.setup_interactivity()

        match_fname = self.tracking_cfg.resolve_relative_path_from_config('manual_correction_global2tracklet_fname')
        self.output_match_fname = get_sequential_filename(match_fname)
        df_fname = self.tracking_cfg.resolve_relative_path_from_config('manual_correction_tracklets_df_fname')
        self.output_df_fname = get_sequential_filename(df_fname)

        # Read metadata (if found) and save as same name with suffix
        splits_names_fname = Path(df_fname).parent.joinpath("split_names.pickle")
        splits_times_fname = splits_names_fname.with_name("split_times.pickle")

        reader = pickle_load_binary
        self.tracklet_split_names = read_if_exists(splits_names_fname, reader=reader)
        if self.tracklet_split_names is None:
            self.tracklet_split_names = defaultdict(list)
        # else:
        #     # Do not overwrite old splits?
        #     self.tracklet_split_names_fname = get_sequential_filename(str(splits_names_fname))
        self.tracklet_split_times = read_if_exists(splits_times_fname, reader=reader)
        if self.tracklet_split_times is None:
            self.tracklet_split_times = defaultdict(list)

        self.tracklet_split_names_fname = str(splits_names_fname)
        self.tracklet_split_times_fname = str(splits_times_fname)

        # self.tracklet_split_times_fname = get_sequential_filename(str(splits_times_fname))

        if self.segmentation_options is None:
            self.segmentation_options = dict(
                x_split_local_coord=None
            )

        if self.indices_of_original_neurons is None:
            self.indices_of_original_neurons = []

        self.logger.debug(
            f"Output files for annotator: {match_fname}, {df_fname}, {splits_names_fname}, {splits_times_fname}")

    def initialize_gt_model_mismatches(self, project_data):
        from wbfm.utils.projects.finished_project_data import calc_all_mismatches_between_ground_truth_and_pairs
        try:
            self.gt_mismatches = calc_all_mismatches_between_ground_truth_and_pairs(project_data, minimum_confidence=0.7)
        except ModuleNotFoundError:
            pass

    @property
    def combined_global2tracklet_dict(self):
        # tmp = deepcopy(self.global2tracklet)
        # for k in tmp.keys():
        #     tmp[k].extend(self.manual_global2tracklet_names[k].copy())
        #     [tmp[k].remove(neuron) for neuron in self.manual_global2tracklet_removals[k] if neuron in tmp[k]]
        # if self.current_tracklet_name is not None:
        #     pass
        #     # logging.warning("Currently active tracklet not included in combined dict")
        # return tmp
        return self._combined_global2tracklet_dict

    @property
    def current_tracklet(self):
        if self.current_tracklet_name is None:
            return None
        df_single_track = self.df_tracklet_obj.df_tracklets_zxy[self.current_tracklet_name]
        return df_single_track

    def set_current_tracklet(self, tracklet_name):
        self.previous_tracklet_name = self.current_tracklet_name
        self.current_tracklet_name = tracklet_name

    def clear_current_tracklet(self):
        if self.current_tracklet_name is not None:
            self.logger.debug(f"Cleared tracklet {self.current_tracklet_name} from the annotator")
            self.set_current_tracklet(None)
        else:
            self.logger.debug("No current tracklet; this button did nothing")

    def calculate_tracklets_for_neuron(self, neuron_name=None) -> \
            Tuple[Dict[str, pd.DataFrame], pd.DataFrame, str]:
        # Note: does NOT save this neuron as self.current_neuron
        if neuron_name is None:
            neuron_name = self.current_neuron
        if neuron_name is None:
            raise ValueError("Must pass neuron name explicitly or have one saved in the object")
        # Returns a list of pd.DataFrames with columns x, y, z, and likelihood, which can be plotted in a loop
        these_names = self.global2tracklet[neuron_name].copy()

        these_names.extend(self.manual_global2tracklet_names[neuron_name])
        [these_names.remove(name) for name in self.manual_global2tracklet_removals[neuron_name] if name in these_names]

        if self.verbose >= 1:
            self.print_current_status(neuron_name)
        these_tracklets = {name: self.df_tracklet_obj.df_tracklets_zxy[name] for name in these_names}
        current_name = self.current_tracklet_name
        if current_name is not None:
            current_tracklet = self.df_tracklet_obj.df_tracklets_zxy[current_name]
        else:
            current_tracklet = None

        return these_tracklets, current_tracklet, current_name

    def get_neuron_name_of_conflicting_match(self, tracklet_name=None):
        # The tracklet shouldn't be in the manually annotated match, because it can't be added if there are conflicts
        if tracklet_name is None:
            tracklet_name = self.current_tracklet_name
        for k, v in self.combined_global2tracklet_dict.items():
            if tracklet_name in v:
                return k
        return None

    def is_tracklet_already_matched(self, tracklet_name=None):
        if tracklet_name is None:
            tracklet_name = self.current_tracklet_name
        name = self.get_neuron_name_of_conflicting_match(tracklet_name)
        if name is None:
            return False
        else:
            return True

    def get_dict_of_tracklet_time_conflicts(self, candidate_tracklet_name=None) -> Union[Dict[str, list], None]:
        # The tracklet shouldn't be in the manually annotated match, because it can't be added if there are conflicts
        if candidate_tracklet_name is None:
            candidate_tracklet_name = self.current_tracklet_name
        if candidate_tracklet_name is None:
            return None
        tracklet_dict = self.combined_global2tracklet_dict
        current_tracklet_names = tracklet_dict[self.current_neuron]
        df_tracklets = self.df_tracklet_obj.df_tracklets_zxy

        return get_time_overlap_of_candidate_tracklet(candidate_tracklet_name, current_tracklet_names, df_tracklets)

    def time_of_next_conflict(self, i_start=0) -> Tuple[Union[int, None], Union[str, None]]:
        conflicts = self.get_dict_of_tracklet_time_conflicts()
        if conflicts is None:
            return None, None

        next_conflict_time = np.inf
        neuron_conflict = None
        for neuron, times in conflicts.items():
            t = np.min(times)
            if i_start < t < next_conflict_time:
                next_conflict_time = t
                neuron_conflict = neuron

        if neuron_conflict is not None:
            self.logger.debug(f"Found next conflict with {neuron_conflict} at time {next_conflict_time}")
            return int(next_conflict_time), neuron_conflict
        else:
            return None, None

    def end_time_of_current_tracklet(self):
        tracklet = self.current_tracklet
        if tracklet is None:
            return None
        else:
            return int(tracklet.last_valid_index())

    def start_time_of_current_tracklet(self):
        tracklet = self.current_tracklet
        if tracklet is None:
            return None
        else:
            return int(tracklet.first_valid_index())

    def tracklet_has_time_overlap(self, tracklet_name=None):
        if tracklet_name is None:
            tracklet_name = self.current_tracklet_name
        name_list = self.get_dict_of_tracklet_time_conflicts(tracklet_name)
        if name_list is None:
            return False
        else:
            return True

    @property
    def is_current_tracklet_confict_free(self):
        has_match = self.is_tracklet_already_matched()
        has_time_overlap = self.tracklet_has_time_overlap()
        if not has_match and not has_time_overlap:
            return True
        else:
            self.print_tracklet_conflicts()
            return False

    def get_types_of_conflicts(self):
        types_of_conflicts = []
        conflicting_match = self.get_neuron_name_of_conflicting_match()
        if conflicting_match and conflicting_match == self.current_neuron:
            types_of_conflicts.append("Already added")
        else:
            # It's possible for a human to force an already added tracklet to have multiple tracklets...
            if conflicting_match:
                types_of_conflicts.append("Identity")
            if self.get_dict_of_tracklet_time_conflicts():
                types_of_conflicts.append("Time")
            if len(types_of_conflicts) == 0:
                types_of_conflicts.append("No conflicts")
        return types_of_conflicts

    def print_tracklet_conflicts(self):
        name = self.get_neuron_name_of_conflicting_match()
        if name is not None:
            self.logger.warning(f"Tracklet {self.current_tracklet_name} is already matched to other neuron: {name}")
        name_dict = self.get_dict_of_tracklet_time_conflicts()
        if name_dict is not None:
            self.logger.warning(f"Current tracklet {self.current_tracklet_name} has time conflict with tracklet(s):")
            for name, times in name_dict.items():
                self.logger.info(f"{name} at times {times}")

    def add_tracklet_to_neuron(self, tracklet_name, neuron_name):
        previously_added = self.manual_global2tracklet_names[neuron_name]
        previously_removed = self.manual_global2tracklet_removals[neuron_name]
        if tracklet_name in previously_added:
            self.logger.debug(f"Tracklet {tracklet_name} already in {neuron_name}; nothing added")
        else:
            self.manual_global2tracklet_names[neuron_name].append(tracklet_name)
            self.add_tracklet_to_global2tracklet_dict(tracklet_name, neuron_name)
            self.logger.info(f"Successfully added tracklet {tracklet_name} to {neuron_name}")

        if tracklet_name in previously_removed:
            self.logger.debug(f"Tracklet was in the to-remove list, but was removed")
            self.manual_global2tracklet_removals[neuron_name].remove(tracklet_name)
            self.add_tracklet_to_global2tracklet_dict(tracklet_name, neuron_name)

    def remove_tracklet_from_global2tracklet_dict(self, tracklet_name, neuron_name):
        if tracklet_name in self._combined_global2tracklet_dict[neuron_name]:
            self._combined_global2tracklet_dict[neuron_name].remove(tracklet_name)
        else:
            logging.warning("Tried to remove tracklet, but is not added")

    def add_tracklet_to_global2tracklet_dict(self, tracklet_name, neuron_name):
        if tracklet_name not in self._combined_global2tracklet_dict[neuron_name]:
            self._combined_global2tracklet_dict[neuron_name].append(tracklet_name)
        else:
            logging.warning("Tried to add tracklet, but is already added")

    def remove_tracklet_from_neuron(self, tracklet_name, neuron_name=None):
        if neuron_name is None:
            neuron_name = self.current_neuron
        previously_added = self.manual_global2tracklet_names[neuron_name]
        previously_removed = self.manual_global2tracklet_removals[neuron_name]
        if tracklet_name in previously_removed:
            self.logger.debug(f"Tracklet {tracklet_name} already removed from {neuron_name}; nothing removed")
        else:
            self.manual_global2tracklet_removals[neuron_name].append(tracklet_name)
            self.remove_tracklet_from_global2tracklet_dict(tracklet_name, neuron_name)
            self.logger.info(f"Successfully added {tracklet_name} to removal list of {neuron_name}")

        if tracklet_name in previously_added:
            self.logger.debug(f"{tracklet_name} was in the manually to-add list, but was removed")
            self.manual_global2tracklet_names[neuron_name].remove(tracklet_name)
            self.remove_tracklet_from_global2tracklet_dict(tracklet_name, neuron_name)

    def remove_tracklet_from_all_matches(self, tracklet_name=None):
        if tracklet_name is None:
            tracklet_name = self.current_tracklet_name
        other_match = self.get_neuron_name_of_conflicting_match(tracklet_name)
        if other_match is not None:
            with self.saving_lock:
                self.remove_tracklet_from_neuron(tracklet_name, other_match)
        else:
            self.logger.debug("Already unmatched")
        return tracklet_name

    def remove_tracklets_with_time_conflicts(self):
        tracklet_name = self.current_tracklet_name
        all_overlap_dict = self.get_dict_of_tracklet_time_conflicts(tracklet_name)
        if all_overlap_dict is not None:
            with self.saving_lock:
                conflicting_names = list(all_overlap_dict.keys())
                for conflicting_tracklet_name in conflicting_names:
                    self.remove_tracklet_from_neuron(conflicting_tracklet_name, self.current_neuron)
                    # self.manual_global2tracklet_removals[self.current_neuron].append(conflicting_tracklet_name)
            # assert not self.tracklet_has_time_overlap(tracklet_name), f"Clean up of {tracklet_name} failed"
        else:
            self.logger.debug("Already not conflicting")
        return conflicting_names

    def remove_all_tracklets_after_time(self, t):
        self.logger.warning(f"Removing all tracklets attached to {self.current_neuron} after t={t}")
        conflicting_names = []
        with self.saving_lock:
            for tracklet_name in self.combined_global2tracklet_dict[self.current_neuron]:
                t_start = self.df_tracklet_obj.df_tracklets_zxy[tracklet_name].first_valid_index()
                if t_start > t:
                    self.remove_tracklet_from_neuron(tracklet_name, self.current_neuron)
                    conflicting_names.append(tracklet_name)
        return conflicting_names

    def save_current_tracklet_to_current_neuron(self):
        if self.current_tracklet_name is None:
            self.logger.info("No neuron selected")
            return None
        if self.is_current_tracklet_confict_free:
            with self.saving_lock:
                self.add_tracklet_to_neuron(self.current_tracklet_name, self.current_neuron)
                tracklet_name = self.current_tracklet_name
                self.set_current_tracklet(None)
            return tracklet_name
        else:
            self.logger.warning("Current tracklet has conflicts, please resolve before saving as a match")
            return None

    def print_current_status(self, neuron_name=None):
        if neuron_name is None:
            neuron_name = self.current_neuron

        if neuron_name is None:
            print("No neuron selected")
        else:
            these_names = self.global2tracklet[self.current_neuron]
            self.logger.debug(f"Initial tracklets for {self.current_neuron}: {these_names}")
            self.logger.debug(f"Previous manually added tracklets: {self.manual_global2tracklet_names[neuron_name]}")
            self.logger.debug(
                f"Previous manually removed tracklets: {self.manual_global2tracklet_removals[neuron_name]}")
            self.logger.debug(f"Currently selected (not yet added) tracklet: {self.current_tracklet_name}")

    def save_manual_matches_to_disk_dispatch(self):
        # Saves the new dataframe (possibly with split tracklets) and the new matches
        self.logger.warning("Saving tracklet dataframe, DO NOT QUIT")
        self.logger.info("Note: the GUI might still respond, but you can't split or save any tracklets")
        # self.saving_lock.acquire(blocking=True)
        self.logger.info("Acquired saving lock; currently saving")
        t = threading.Thread(target=self.save_manual_matches_to_disk)
        t.start()

    def save_manual_matches_to_disk(self):
        with self.saving_lock:
            self.tracking_cfg.pickle_data_in_local_project(self.combined_global2tracklet_dict, self.output_match_fname)
            match_fname = self.tracking_cfg.unresolve_absolute_path(self.output_match_fname)
            self.tracking_cfg.config.update({'manual_correction_global2tracklet_fname': match_fname})

            # Note: sparse matrices can only be pickled
            # self.tracking_cfg.h5_in_local_project(self.df_tracklet_obj.df_tracklets_zxy, self.output_df_fname)
            if self.df_tracklet_obj.use_custom_padded_dataframe:
                df = self.df_tracklet_obj.df_tracklets_zxy.return_sparse_dataframe()
            else:
                df = self.df_tracklet_obj.df_tracklets_zxy
            self.tracking_cfg.pickle_data_in_local_project(df, self.output_df_fname,
                                                           custom_writer=pd.to_pickle)
            df_fname = self.tracking_cfg.unresolve_absolute_path(self.output_df_fname)
            self.tracking_cfg.config.update({'manual_correction_tracklets_df_fname': df_fname})

            self.tracking_cfg.pickle_data_in_local_project(self.tracklet_split_names, self.tracklet_split_names_fname)
            self.tracking_cfg.pickle_data_in_local_project(self.tracklet_split_times, self.tracklet_split_times_fname)

            self.tracking_cfg.update_self_on_disk()
        self.logger.info("Saving successful! You may now quit")

    def split_current_tracklet(self, i_split, set_new_half_to_current=True):
        """
        The current time is included in the "new half" of the tracklet
        The newer half is added as a new index in the df_tracklet dataframe

        And finally, the newer half is set as the current tracklet (if set_new_half_to_current)

        Parameters
        ----------
        i_split
        set_new_half_to_current

        Returns
        -------

        """
        if self.current_tracklet_name is None:
            print("No current tracklet!")
            return False

        with self.saving_lock:
            # Left half stays as old name
            old_name = self.current_tracklet_name
            all_tracklets = self.df_tracklet_obj.df_tracklets_zxy

            if self.df_tracklet_obj.use_custom_padded_dataframe:
                successfully_split, all_tracklets, left_name, right_name = \
                    all_tracklets.split_tracklet(i_split, old_name)
            else:
                successfully_split, all_tracklets, left_name, right_name = \
                    split_tracklet_within_sparse_dataframe(all_tracklets, i_split, old_name)

            if not successfully_split:
                self.logger.warning("Did not successfully split; check logs")
                return False

            self.df_tracklet_obj.df_tracklets_zxy = all_tracklets
            if set_new_half_to_current:
                self.set_current_tracklet(right_name)
            else:
                self.set_current_tracklet(left_name)

            # Save a record of the split
            self.tracklet_split_names[left_name].append(right_name)
            self.tracklet_split_times[left_name].append((i_split - 1, i_split))

            # Update the callback dictionary
            self.df_tracklet_obj.update_callback_dictionary_for_single_tracklet(left_name)
            self.df_tracklet_obj.update_callback_dictionary_for_single_tracklet(right_name)

        return True

    def segmentation_updated_callbacks(self):
        [callback() for callback in self.segmentation_callbacks]

    def tracklet_updated_callbacks(self):
        [callback() for callback in self.tracklet_callbacks]

    def connect_tracklet_clicking_callback(self, layer_to_add_callback, viewer: napari.Viewer,
                                           added_segmentation_callbacks,
                                           added_tracklet_callbacks,
                                           max_dist=1.0):
        self.segmentation_callbacks = added_segmentation_callbacks
        self.tracklet_callbacks = added_tracklet_callbacks

        @layer_to_add_callback.mouse_drag_callbacks.append
        def on_click(layer, event):
            if not self.is_currently_interactive:
                logging.warning("Click received, but interactivity is turned off")
                return
            self.click_callback(event, layer, viewer)

    def click_callback(self, event, layer, viewer):
        self.last_clicked_position = event.position
        invalid_target, seg_index, time_index = self._unpack_click_event(event, layer)
        if invalid_target:
            print("Event triggered on background; returning")
        else:
            if self.verbose >= 1:
                self.logger.debug(f"Event triggered on segmentation {seg_index} at time {int(event.position[0])} "
                                  f"and position {event.position[1:]}")

            # Decide which mode: segmentation or tracklet

            # The modifiers field is a list of Key objects
            # Class definition: https://github.com/vispy/vispy/blob/ef982591e223fff09d91d8c2697489c7193a85aa/vispy/util/keys.py
            # print([m.name for m in event.modifiers])
            click_modifiers = [m.name.lower() for m in event.modifiers]
            if 'control' in click_modifiers:
                # Just add neuron, no automatic splitting
                self.append_segmentation_to_list(time_index, seg_index)
                segment_mode_not_tracklet_mode = True
            elif 'alt' in click_modifiers:
                # Shortcut for clearing all neurons, adding this one, and attempting to split
                self.set_selected_segmentation(time_index, seg_index)
                self.split_current_neuron_and_add_napari_layer(viewer, split_method="Gaussian")
                segment_mode_not_tracklet_mode = True
            else:
                segment_mode_not_tracklet_mode = False

            if not segment_mode_not_tracklet_mode:
                # Split tracklet, not segmentation
                tracklet_name = self.df_tracklet_obj.get_tracklet_from_segmentation_index(
                    i_time=time_index,
                    seg_ind=seg_index
                )
                if self.verbose >= 1:
                    self.logger.debug(f"Neuron is part of tracklet {tracklet_name}")

                if tracklet_name:
                    self.set_current_tracklet(tracklet_name)
                    self.add_current_tracklet_to_viewer(viewer)
                    if self.current_neuron is not None:
                        self.tracklet_updated_callbacks()
                else:
                    self.set_selected_segmentation(time_index, seg_index)
                    if self.verbose >= 1:
                        self.logger.debug(f"Tracklet not found; adding segmentation only")

    def _unpack_click_event(self, event, layer):
        # Get information about clicked-on neuron
        seg_index = layer.get_value(
            position=event.position,
            view_direction=event.view_direction,
            dims_displayed=event.dims_displayed,
            world=True
        )
        time_index = int(event.position[0])
        invalid_target = seg_index is None or seg_index == 0
        return invalid_target, seg_index, time_index

    def add_current_tracklet_to_viewer(self, viewer):
        df_single_track = self.current_tracklet
        if self.verbose >= 1:
            self.logger.debug(f"Adding tracklet of length {df_single_track['z'].count()}")
        if self.to_add_layer_to_viewer:
            all_tracks_array, track_of_point, to_remove = build_tracks_from_dataframe(df_single_track,
                                                                                      z_to_xy_ratio=self.z_to_xy_ratio)
            viewer.add_tracks(track_of_point, name=self.current_tracklet_name,
                              tail_width=10, head_length=0, tail_length=4,
                              colormap='hsv', blending='opaque', opacity=1.0)
        if self.verbose >= 2:
            print(df_single_track.dropna(inplace=False))

    def split_current_neuron_and_add_napari_layer(self, viewer, split_method):
        # Note: keeps both indices saved in the object
        seg_index = self.indices_of_original_neurons
        if len(seg_index) > 1:
            self.logger.warning("Multiple neurons selected, splitting is ambiguous... returning")
            return
        else:
            seg_index = seg_index[0]
            time_index = self.time_of_candidate

        full_mask = viewer.layers['Raw segmentation'].data[time_index]
        red_volume = viewer.layers['Red data'].data[time_index]
        new_full_mask = split_neuron_interactive(full_mask, red_volume, seg_index,
                                                 min_separation=2,
                                                 verbose=3,
                                                 method=split_method,
                                                 **self.segmentation_options)
        plt.show()
        split_succeeded = new_full_mask is not None
        if split_succeeded:
            self.add_candidate_mask_layer(viewer, new_full_mask)

    def merge_current_neurons(self, viewer):
        # NOTE: will keep the index of the first selected neuron
        if len(self.indices_of_original_neurons) <= 1:
            self.logger.warning(f"Too few neurons selected ({len(self.indices_of_original_neurons)}), aborting")
            return
        elif len(self.indices_of_original_neurons) > 2:
            self.logger.warning(f"Merging than 2 neurons not supported, aborting")
            return

        time_index = self.time_of_candidate
        new_full_mask = viewer.layers['Raw segmentation'].data[time_index].copy()

        indices_to_overwrite = self.indices_of_original_neurons[1:]
        target_index = self.indices_of_original_neurons[0]
        for i in indices_to_overwrite:
            new_full_mask[new_full_mask == i] = target_index

        self.add_candidate_mask_layer(viewer, new_full_mask)

        # Keep the saved indices as both
        # self.set_selected_segmentation(self.time_of_candidate, target_index)

    def add_candidate_mask_layer(self, viewer, new_full_mask=None):
        if new_full_mask is None:
            # Add copy of current point
            t = viewer.dims.current_step[0]
            new_full_mask = viewer.layers['Raw segmentation'].data[t].copy()
        # Add as a new candidate layer
        layer_name = f"Candidate_mask"
        viewer.add_labels(new_full_mask, name=layer_name, opacity=1.0,
                          scale=(self.z_to_xy_ratio, 1.0, 1.0))
        # Save for later combining with original mask
        self.candidate_mask = new_full_mask

    def modify_buffer_segmentation(self, t, new_mask):
        self.buffer_masks[t, ...] = new_mask
        self.t_buffer_masks.append(t)

    def update_segmentation_layer_using_buffer(self, layer, t=None):
        if t is None:
            # Default to most recent buffer change
            t = self.t_buffer_masks[-1]
        elif t not in self.t_buffer_masks:
            self.logger.warning(f"Tried to update segmentation layer at t={t}, but no buffer mask found")
            return
        layer.data[t] = self.buffer_masks[t]

    def clear_currently_selected_segmentations(self, do_callbacks=True):
        self.time_of_candidate = None
        self.indices_of_original_neurons = []
        self.invalidate_candidate_mask()
        if do_callbacks:
            self.segmentation_updated_callbacks()

    def append_segmentation_to_list(self, time_index, seg_index):
        if self.time_of_candidate is None:
            self.time_of_candidate = time_index
            self.indices_of_original_neurons = [seg_index]
            self.invalidate_candidate_mask()
            self.segmentation_updated_callbacks()
        else:
            if self.time_of_candidate == time_index:
                self.indices_of_original_neurons.append(seg_index)
                self.invalidate_candidate_mask()
                self.segmentation_updated_callbacks()
                self.logger.info(f"Added neuron to list; current neurons: {self.indices_of_original_neurons}")
            else:
                self.logger.warning("Attempt to add segmentations of different time points; not supported")

    def invalidate_candidate_mask(self):
        # Make sure the metadata and so on is synced to the saved mask
        self.candidate_mask = None

    def set_selected_segmentation(self, time_index, seg_index):
        """Like append_segmentation_to_list, but forces a single neuron. Also properly accounts for callbacks"""
        self.clear_currently_selected_segmentations(do_callbacks=False)
        self.append_segmentation_to_list(time_index, seg_index)

    def toggle_highlight_selected_neuron(self, viewer):
        layer = viewer.layers['Raw segmentation']
        ind = self.indices_of_original_neurons
        if len(ind) > 1:
            self.logger.warning("Selection not implemented if more than one neuron is selected")
            return
        elif len(ind) == 1:
            layer.show_selected_label = True
            layer.selected_label = ind[0]
        else:
            layer.show_selected_label = False

    def attach_current_segmentation_to_current_tracklet(self):
        t, tracklet_name, mask_ind, flag = \
            self.check_validity_of_tracklet_and_segmentation(can_be_attached_to_tracklet=False)
        if not flag:
            return flag

        with self.saving_lock:
            self.df_tracklet_obj.update_tracklet_metadata_using_segmentation_metadata(
                t, tracklet_name, mask_ind=mask_ind, likelihood=1.0
            )

        self.clear_currently_selected_segmentations(do_callbacks=False)
        self.segmentation_updated_callbacks()
        self.tracklet_updated_callbacks()

        return True

    def delete_current_segmentation_from_tracklet(self):
        t, tracklet_name, mask_ind, flag = self.check_validity_of_tracklet_and_segmentation()
        if not flag:
            return flag

        # Check that the segmentation is actually attached to this neuron
        old_mask_ind = cast_int_or_nan(self.df_tracklet_obj.df_tracklets_zxy[tracklet_name]['raw_segmentation_id'][t])
        if np.isnan(old_mask_ind) or old_mask_ind != mask_ind:
            self.logger.warning(f"Deletion of segmentation {mask_ind} from {tracklet_name} attempted, "
                                f"but current mask is instead {old_mask_ind}; aborting")
            return False

        with self.saving_lock:
            self.df_tracklet_obj.delete_data_from_tracklet_at_time(t, tracklet_name)

        self.clear_currently_selected_segmentations(do_callbacks=False)
        self.segmentation_updated_callbacks()
        # self.tracklet_updated_callbacks()
        return True

    def check_validity_of_tracklet_and_segmentation(self, can_be_attached_to_tracklet=True):
        """
        1) segmentation must be unique (not 2 segmentations selected)
        2) segmentation must not be attached to a tracklet
        3) A segmentation and a tracklet must both be selected
        """
        flag = True
        t = self.time_of_candidate
        tracklet_name = self.current_tracklet_name

        if len(self.indices_of_original_neurons) != 1:
            self.logger.warning(f"Selected segmentation must be unique; "
                                f"found {len(self.indices_of_original_neurons)} segmentations")
            flag = False
            mask_ind = None
        else:
            mask_ind = self.indices_of_original_neurons[0]

        # Check if segmentation already has a tracklet
        allowed_tracklet_attachment_state = True
        if not can_be_attached_to_tracklet:
            previous_tracklet_name = self.df_tracklet_obj.get_tracklet_from_segmentation_index(
                i_time=t,
                seg_ind=mask_ind
            )
            if previous_tracklet_name:
                allowed_tracklet_attachment_state = False

        if not allowed_tracklet_attachment_state:
            self.logger.warning(f"Selected segmentation must not be attached to a tracklet")
            flag = False
        else:
            # Get known data, then rebuild the other metadata from this
            if tracklet_name is None:
                self.logger.warning("No tracklet selected, can't modify using segmentation")
                flag = False

        if flag:
            self.logger.info(f"Modifying {tracklet_name} using segmentation {mask_ind} at t={t}")

        return t, tracklet_name, mask_ind, flag
