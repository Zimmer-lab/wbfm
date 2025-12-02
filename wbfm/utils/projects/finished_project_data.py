import concurrent
import logging
import shutil
from collections import defaultdict

from wbfm.utils.external.utils_pandas import combine_columns_with_suffix

import tables
from methodtools import lru_cache
from pathlib import Path
from matplotlib import pyplot as plt
from scipy import signal
from scipy.signal import detrend
from sklearn.decomposition import PCA

from wbfm.utils.general.utils_paper import PaperDataCache, apply_figure_settings, paper_trace_settings
from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
from wbfm.utils.external.utils_jupyter import executing_in_notebook
from wbfm.utils.external.utils_zarr import zarr_reader_folder_or_zipstore
from wbfm.utils.external.custom_errors import NoMatchesError, NoNeuronsError, NoBehaviorAnnotationsError, \
    IncompleteConfigFileError, DataSynchronizationError
from wbfm.utils.general.postprocessing.utils_imputation import impute_missing_values_in_dataframe
from wbfm.utils.general.postures.centerline_classes import WormFullVideoPosture
from wbfm.utils.neuron_matching.class_reference_frame import ReferenceFrame
from wbfm.utils.neuron_matching.matches_class import MatchesWithConfidence, get_mismatches
from wbfm.utils.external.utils_neuron_names import int2name_neuron
from wbfm.utils.general.utils_paper import behavior_name_mapping
import os
from dataclasses import dataclass
from typing import Tuple, Dict, Union, List, Optional
import numpy as np
import dask.array as da
import pandas as pd
import zarr
from tqdm.auto import tqdm

from wbfm.utils.projects.neuropal_manager import NeuropalManager
from wbfm.utils.traces.triggered_averages import plot_triggered_average_from_matrix_low_level
from wbfm.utils.general.hardcoded_paths import read_names_of_neurons_to_id, neurons_with_confident_ids
from wbfm.utils.external.utils_pandas import dataframe_to_numpy_zxy_single_frame, df_to_matches, \
    get_column_name_from_time_and_column_value, fix_extra_spaces_in_dataframe_columns, \
    get_contiguous_blocks_from_column, make_binary_vector_from_starts_and_ends, fill_missing_indices_with_nan, \
    ffill_using_raw_data
from wbfm.utils.neuron_matching.class_frame_pair import FramePair
from wbfm.utils.projects.physical_units import PhysicalUnitConversion
from wbfm.utils.projects.utils_project_status import get_project_status
from wbfm.utils.traces.residuals import calculate_residual_subtract_pca, calculate_residual_subtract_nmf
from wbfm.utils.general.high_performance_pandas import get_names_from_df
from wbfm.utils.tracklets.utils_tracklets import fix_global2tracklet_full_dict, check_for_unmatched_tracklets
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from wbfm.utils.tracklets.tracklet_class import DetectedTrackletsAndNeurons
from wbfm.utils.projects.plotting_classes import TracePlotter, TrackletAndSegmentationAnnotator
from wbfm.utils.segmentation.util.utils_metadata import DetectedNeurons
from wbfm.utils.projects.project_config_classes import ModularProjectConfig, SubfolderConfigFile
from wbfm.utils.general.utils_filenames import read_if_exists, pickle_load_binary, \
    load_file_according_to_precedence, pandas_read_any_filetype, get_sequential_filename
from wbfm.utils.projects.utils_project import safe_cd
# from functools import cached_property # Only from python>=3.8
from backports.cached_property import cached_property

from wbfm.utils.visualization.filtering_traces import fast_slow_decomposition, filter_trace_using_mode, \
    fill_nan_in_dataframe


@dataclass
class ProjectData:
    """
    Project data class that collects all important final data from a whole brain freely moving dataset
    Also exposes methods to load intermediate data

    ######## Important fields #########
    project_config: ModularProjectConfig  # Custom class (used for loading intermediate results)

    red_data: zarr.Array  # Preprocessed video (~100 GB)
    green_data: zarr.Array  # Preprocessed video (~100 GB)

    raw_segmentation: zarr.Array = None  # Full-sized segmentation (before tracking)
    segmentation: zarr.Array  # Full-sized segmentation (after tracking -> colors are aligned)
    segmentation_metadata: DetectedNeurons  # Easy conversion between segmentation ID and position

    # Traces as calculated from the segmentation
    red_traces: pd.DataFrame
    green_traces: pd.DataFrame

    ######## Important properties (loaded on demand) #########
    intermediate_global_tracks
    final_tracks

    raw_frames
    raw_matches

    df_all_tracklets

    """
    project_dir: str
    project_config: ModularProjectConfig  # Custom class (used for loading intermediate results)

    red_data: zarr.Array = None  # Preprocessed video (~100 GB)
    green_data: zarr.Array = None  # Preprocessed video (~100 GB)

    raw_segmentation: zarr.Array = None  # Full-sized segmentation (before tracking)
    segmentation: zarr.Array = None  # Full-sized segmentation (after tracking -> colors are aligned)
    segmentation_metadata: DetectedNeurons = None  # Easy conversion between segmentation ID and position

    # OLD
    df_training_tracklets: pd.DataFrame = None
    reindexed_masks_training: zarr.Array = None

    # Traces as calculated from the segmentation
    red_traces: pd.DataFrame = None
    green_traces: pd.DataFrame = None

    # For plotting and visualization
    background_per_pixel: float = None  # Simple version of background correction
    likelihood_thresh: float = None  # When plotting, plot gaps for low-confidence time points

    all_used_fnames: list = None  # Save a list of all paths to raw data
    verbose: int = 2  # How much to print when running methods

    # Precedence when multiple are available
    precedence_global2tracklet: list = None
    precedence_df_tracklets: list = None
    precedence_tracks: list = None

    # Will be set as loaded
    final_tracks_fname: str = None
    intermediate_tracks_fname: str = None
    global2tracklet_fname: str = None
    df_all_tracklets_fname: str = None
    force_tracklets_to_be_sparse: bool = False
    set_up_tracklet_interactivity: bool = True

    _custom_frame_indices: list = None

    # Classes for more functionality
    _trace_plotter: TracePlotter = None
    physical_unit_conversion: PhysicalUnitConversion = None

    # Values for ground truth annotation (reading from excel or .csv)
    finished_neurons_column_name: str = "Finished?"
    df_manual_tracking_fname: str = None

    # EXPERIMENTAL (but tested)
    use_custom_padded_dataframe: bool = False
    _use_physical_time: bool = False  # Relies on hardcoded volumes per second

    # Caching
    data_cacher: PaperDataCache = None

    def __post_init__(self):
        """
        Load values from disk config files if the user did not set them
        """
        if self.project_config.has_valid_self_path:
            track_cfg = self.project_config.get_tracking_config()
            if self.precedence_global2tracklet is None:
                self.precedence_global2tracklet = track_cfg.config['precedence_global2tracklet']
            if self.precedence_df_tracklets is None:
                self.precedence_df_tracklets = track_cfg.config['precedence_df_tracklets']
            if self.precedence_tracks is None:
                self.precedence_tracks = track_cfg.config['precedence_tracks']

            self.data_cacher = PaperDataCache(self)
        else:
            self.logger.debug("ProjectData initialized without a project_config object; "
                                "if this is from a NWB file, this is expected")

    @cached_property
    def neuropal_manager(self):
        try:
            neuropal_config = self.project_config.get_neuropal_config()
        except (FileNotFoundError, IncompleteConfigFileError):
            return NeuropalManager(None)
        return NeuropalManager(neuropal_config)

    @cached_property
    def intermediate_global_tracks(self) -> pd.DataFrame:
        """
        Dataframe of tracks produced by the global tracker alone (no tracklets)

        Names are aligned with the final traces
        """
        if not self.project_config.has_valid_self_path:
            self.logger.debug("ProjectData initialized without a project_config object; intermediate tracks can't be loaded")
            return None
        tracking_cfg = self.project_config.get_tracking_config()

        # Manual annotations take precedence by default
        fname = tracking_cfg.config['leifer_params']['output_df_fname']
        fname = tracking_cfg.resolve_relative_path(fname, prepend_subfolder=False)
        self.logger.debug(f"Loading intermediate global tracks from {fname}")
        self.all_used_fnames.append(fname)

        global_tracks = read_if_exists(fname)
        self.intermediate_tracks_fname = fname
        return global_tracks

    @cached_property
    def initial_pipeline_tracks(self) -> pd.DataFrame:
        """
        Dataframe of tracks produced by the global tracker with tracklets

        Does not include any manual annotations

        Uses the initial filename in the config file, under ['final_3d_postprocessing']['output_df_fname']
        """
        tracking_cfg = self.project_config.get_tracking_config()

        fname = tracking_cfg.config['final_3d_postprocessing']['output_df_fname']
        fname = tracking_cfg.resolve_relative_path(fname, prepend_subfolder=False)
        self.logger.debug(f"Loading initial pipeline tracks from {fname}")
        self.all_used_fnames.append(fname)

        global_tracks = read_if_exists(fname)
        return global_tracks

    def single_reference_frame_tracks(self, i_frame: int = 0) -> pd.DataFrame:
        """
        Dataframe of tracks produced by the global tracker, but before combining with other reference frames

        Assumes the filename is of the format f"df_tracks_superglue_template-{i_frame}.h5"
        """
        tracking_cfg = self.project_config.get_tracking_config()

        fname = os.path.join('postprocessing', f"df_tracks_superglue_template-{i_frame}.h5")
        fname = tracking_cfg.resolve_relative_path(fname, prepend_subfolder=True)
        self.logger.debug(f"Loading single reference frame tracks from: {fname}")
        self.all_used_fnames.append(fname)

        global_tracks = read_if_exists(fname)
        return global_tracks

    @cached_property
    def final_tracks(self) -> pd.DataFrame:
        """
        Dataframe of tracks produced by combining the global tracker and tracklets

        Names are aligned with the final traces
        """
        if not self.project_config.has_valid_self_path:
            self.logger.debug("ProjectData initialized without a project_config object; final tracks can't be loaded")
            return None
        tracking_cfg = self.project_config.get_tracking_config()

        # Manual annotations take precedence by default
        fname = tracking_cfg.config['leifer_params']['output_df_fname']
        possible_fnames = dict(automatic=tracking_cfg.resolve_relative_path_from_config('final_3d_tracks_df'),
                               imputed=tracking_cfg.resolve_relative_path_from_config('missing_data_imputed_df'),
                               fdnc=tracking_cfg.resolve_relative_path(fname, prepend_subfolder=False))

        fname_precedence = self.precedence_tracks
        final_tracks, fname = load_file_according_to_precedence(fname_precedence, possible_fnames,
                                                                reader_func=read_if_exists)
        self.final_tracks_fname = fname
        self.logger.debug(f"Loading final global tracks from {fname}")
        self.all_used_fnames.append(fname)
        return final_tracks

    def get_final_tracks_only_finished_neurons(self) -> Tuple[Optional[pd.DataFrame], List[str]]:
        """
        See get_ground_truth_annotations()

        Dataframe with subset of columns (neurons)
        """
        df_gt, finished_neurons = self.get_list_of_finished_neurons()
        if df_gt is None:
            return None, []
        else:
            return df_gt.loc[:, finished_neurons], finished_neurons

    def get_list_of_finished_neurons(self):
        """Get the finished neurons and dataframe that will be subset-ed"""
        df_gt = self.final_tracks
        finished_neurons = self.finished_neuron_names()
        return df_gt, finished_neurons

    def _build_global2tracklet(self) -> dict:
        """Builds the global2tracklet object"""
        tracking_cfg = self.project_config.get_tracking_config()

        # Manual annotations take precedence by default
        possible_fnames = dict(
            manual=tracking_cfg.resolve_relative_path_from_config('manual_correction_global2tracklet_fname'),
            automatic=tracking_cfg.resolve_relative_path_from_config('global2tracklet_matches_fname'))

        fname_precedence = self.precedence_global2tracklet
        global2tracklet, fname = load_file_according_to_precedence(fname_precedence, possible_fnames,
                                                                   reader_func=pickle_load_binary)
        self.global2tracklet_fname = fname
        self.logger.debug(f"Loading global2tracklet from {fname}")
        if global2tracklet is not None:
            global2tracklet = fix_global2tracklet_full_dict(self.df_all_tracklets, global2tracklet)
        return global2tracklet

    @cached_property
    def global2tracklet(self) -> dict:
        """Dictionary of matches between neuron names (str) and tracklets (str)"""
        return self._build_global2tracklet()

    @cached_property
    def raw_frames(self) -> List[ReferenceFrame]:
        """
        List of ReferenceFrame objects

        This can become desynced if the user modifies segmentation
        """
        if self.verbose >= 1:
            self.logger.info("First time loading the raw frames, may take a while...")
        train_cfg = self.project_config.get_training_config()
        fname = os.path.join('raw', 'frame_dat.pickle')
        fname = train_cfg.resolve_relative_path(fname, prepend_subfolder=True)
        self.logger.debug(f"Loading raw_frames from {fname}")
        frames = pickle_load_binary(fname)
        self.all_used_fnames.append(fname)
        return frames

    @cached_property
    def raw_matches(self) -> Dict[tuple, FramePair]:
        """
        Dict of FramePair objects

        This can become desynced if the user modifies segmentation
        """
        if self.verbose >= 1:
            self.logger.info("First time loading the raw matches, may take a while...")
        train_cfg = self.project_config.get_training_config()
        fname = os.path.join('raw', 'match_dat.pickle')
        fname = train_cfg.resolve_relative_path(fname, prepend_subfolder=True)
        self.logger.debug(f"Loading raw_matches from {fname}")
        matches = pickle_load_binary(fname)
        self.all_used_fnames.append(fname)
        return matches

    @cached_property
    def _raw_clust(self) -> pd.DataFrame:
        """
        Legacy custom dataframe format, before transforming into tracklets.

        Use not suggested
        """
        if self.verbose >= 1:
            self.logger.info("First time loading the raw cluster dataframe, may take a while...")
        train_cfg = self.project_config.get_training_config()
        fname = os.path.join('raw', 'clust_df_dat.pickle')
        fname = train_cfg.resolve_relative_path(fname, prepend_subfolder=True)
        clust = pickle_load_binary(fname)
        self.all_used_fnames.append(fname)
        return clust

    @cached_property
    def df_all_tracklets(self) -> pd.DataFrame:
        """Sparse Dataframe of all tracklets"""
        # TODO: Should this just use _load_df_tracklets?

        df_all_tracklets, fname = self._load_df_tracklets()
        self.df_all_tracklets_fname = fname
        self.all_used_fnames.append(fname)
        train_cfg = self.project_config.get_training_config()

        # Loosely check if there has been any manual annotation
        if train_cfg.config['df_3d_tracklets'] != "2-training_data/all_tracklets.pickle":
            self.force_tracklets_to_be_sparse = True

        if self.force_tracklets_to_be_sparse:
            # This check itself takes so long that it's not worth it
            # if not check_if_fully_sparse(df_all_tracklets):
            self.logger.warning("Casting tracklets as sparse, may take a minute")
            # df_all_tracklets = to_sparse_multiindex(df_all_tracklets)
            df_all_tracklets = df_all_tracklets.astype(pd.SparseDtype("float", np.nan))
            # if True:
            #     self.logger.warning("Casting tracklets as sparse, may take a minute")
            #     # df_all_tracklets = to_sparse_multiindex(df_all_tracklets)
            #     df_all_tracklets = df_all_tracklets.astype(pd.SparseDtype("float", np.nan))
            # else:
            #     self.logger.info("Found sparse matrix")
        self.logger.debug(f"Finished loading tracklets from {fname}")

        return df_all_tracklets

    def _load_df_tracklets(self, dryrun=False) -> Tuple[pd.DataFrame, str]:
        if self.verbose >= 1 and not dryrun:
            self.logger.info("First time loading all the tracklets, may take a while...")
        if not self.project_config.has_valid_self_path:
            self.logger.debug("ProjectData initialized without a project_config object; tracklets can't be loaded")
            return None, None
        train_cfg = self.project_config.get_training_config()
        track_cfg = self.project_config.get_tracking_config()
        # Manual annotations take precedence by default
        possible_fnames = dict(
            manual=track_cfg.resolve_relative_path_from_config('manual_correction_tracklets_df_fname'),
            wiggle=track_cfg.resolve_relative_path_from_config('wiggle_split_tracklets_df_fname'),
            automatic=train_cfg.resolve_relative_path_from_config('df_3d_tracklets'))
        fname_precedence = self.precedence_df_tracklets
        df_all_tracklets, fname = load_file_according_to_precedence(fname_precedence, possible_fnames,
                                                                    reader_func=pandas_read_any_filetype,
                                                                    dryrun=dryrun)
        return df_all_tracklets, fname

    def has_tracklets(self):
        return self._load_df_tracklets(dryrun=True)[1] is not None

    def has_global2tracklet(self):
        return self._load_df_tracklets(dryrun=True)[1] is not None

    @cached_property
    def tracklet_annotator(self) -> TrackletAndSegmentationAnnotator:
        """Custom class that implements manual modification of tracklets and segmentation"""
        if not self.project_config.has_valid_self_path:
            self.logger.debug("ProjectData initialized without a project_config object; tracklet annotator can't be loaded")
            return None
        tracking_cfg = self.project_config.get_tracking_config()
        training_cfg = self.project_config.get_training_config()
        # fname = tracking_cfg.resolve_relative_path_from_config('global2tracklet_matches_fname')

        obj = TrackletAndSegmentationAnnotator(
            lambda: self.tracklets_and_neurons_class,  # This is the function, not the property
            lambda: self.global2tracklet,  # This is the function, not the property
            segmentation_metadata=self.segmentation_metadata,
            tracking_cfg=tracking_cfg,
            training_cfg=training_cfg,
            z_to_xy_ratio=self.physical_unit_conversion.z_to_xy_ratio,
            buffer_masks=zarr.zeros_like(self.raw_segmentation),
            to_setup_interactivity=self.set_up_tracklet_interactivity,
            logger=self.logger,
            verbose=self.verbose,
        )

        if self.set_up_tracklet_interactivity:
            # Requires tracklets to be loaded
            obj.initialize_gt_model_mismatches(self)
        return obj

    def _build_tracklets_and_neurons_class(self) -> DetectedTrackletsAndNeurons:
        """Class that connects tracklets with raw neuron segmentation"""
        _ = self.df_all_tracklets  # Make sure it is loaded
        return DetectedTrackletsAndNeurons(self.df_all_tracklets, self.segmentation_metadata,
                                           dataframe_output_filename=self.df_all_tracklets_fname,
                                           use_custom_padded_dataframe=self.use_custom_padded_dataframe)

    @cached_property
    def tracklets_and_neurons_class(self) -> DetectedTrackletsAndNeurons:
        """Class that connects tracklets with raw neuron segmentation"""
        return self._build_tracklets_and_neurons_class()

    @cached_property
    def worm_posture_class(self) -> WormFullVideoPosture:
        """
        Class with all functionality related to behavior

        For example, allows coloring the traces using behavioral annotation, and correlations to behavioral time series
        """
        return WormFullVideoPosture.load_from_project(self)

    @cached_property
    def tracked_worm_class(self):
        """Class that connects tracklets and final neurons using global tracking"""
        if self.verbose >= 1:
            self.logger.warning(" First time loading tracked worm object, may take a while")
        tracking_cfg = self.project_config.get_tracking_config()
        fname = tracking_cfg.resolve_relative_path('raw/worm_obj.pickle', prepend_subfolder=True)
        return pickle_load_binary(fname)

    @property
    def logger(self) -> logging.Logger:
        if self.project_config is None:
            return logging.getLogger("ProjectData")
        else:
            return self.project_config.logger

    def load_tracklet_related_properties(self):
        """Helper function for loading cached properties"""
        _ = self.df_all_tracklets
        _ = self.tracked_worm_class

    def load_interactive_properties(self):
        """Helper function for loading cached properties"""
        self.set_up_tracklet_interactivity = True
        _ = self.tracklet_annotator

    def load_frame_related_properties(self):
        """Helper function for loading cached properties"""
        _ = self.raw_frames
        _ = self.raw_matches

    def load_segmentation_related_properties(self):
        """Helper function for loading cached properties"""
        _ = self.segmentation_metadata.segmentation_metadata
        self.all_used_fnames.append(self.segmentation_metadata.detection_fname)

    @cached_property
    def num_frames(self) -> int:
        """Note that this is cached so that a user can overwrite the number of frames"""
        num_frames = None
        if self.project_config is not None:
            try:
                num_frames = self.project_config.num_frames
            except (AttributeError, FileNotFoundError):
                num_frames = None
        if num_frames is None:
            # Then try calculate from the processed data, not the raw
            if self.red_data is not None:
                num_frames = self.red_data.shape[0]
        if num_frames is None:
            # Loads the raw data, which may be slow
            num_frames = self.project_config.get_num_frames_robust()
        return num_frames
    

    def num_frames_minus_tracking_failures(self, **kwargs):
        tracking_failures = self.estimate_tracking_failures_from_project(**kwargs)
        if tracking_failures is not None:
            return self.num_frames - len(tracking_failures)
        else:
            return self.num_frames

    def custom_frame_indices(self) -> list:
        """For overriding the normal iterator over frames, for skipping problems etc."""
        if self._custom_frame_indices is not None:
            return self._custom_frame_indices
        else:
            return list(range(self.num_frames))

    def get_frame_index_generator(self):
        """Generator yielding values from custom_frame_indices"""
        for val in self.custom_frame_indices():
            yield val

    @property
    def which_training_frames(self) -> list:
        """Used to determine reference frames for posture matching during tracking"""
        train_cfg = self.project_config.get_training_config()
        return train_cfg.config['training_data_3d']['which_frames']

    @property
    def num_training_frames(self) -> int:
        return len(self.which_training_frames)

    def check_data_desyncing(self, raise_error=True):
        """
        Checks desynchronization in the tracklet-neuron matching database

        See: check_for_unmatched_tracklets
        """
        self.logger.info("Checking for database desynchronization")
        unmatched_tracklets = check_for_unmatched_tracklets(self.df_all_tracklets, self.global2tracklet,
                                                            raise_error=raise_error)

    @staticmethod
    def unpack_config_file(project_path: Union[str, ModularProjectConfig]):
        """Unpack config file into its components"""
        if isinstance(project_path, ModularProjectConfig):
            cfg = project_path
        else:
            opt = {'log_to_file': not executing_in_notebook()}
            cfg = ModularProjectConfig(project_path, **opt)

        project_dir = cfg.project_dir

        segment_cfg = cfg.get_segmentation_config()
        train_cfg = cfg.get_training_config()
        tracking_cfg = cfg.get_tracking_config()
        traces_cfg = cfg.get_traces_config()

        return cfg, segment_cfg, train_cfg, tracking_cfg, traces_cfg, project_dir

    @staticmethod
    def _load_data_from_configs(cfg: ModularProjectConfig,
                                segment_cfg: SubfolderConfigFile,
                                train_cfg: SubfolderConfigFile,
                                tracking_cfg: SubfolderConfigFile,
                                traces_cfg: SubfolderConfigFile,
                                project_dir,
                                to_load_tracklets=False,
                                to_load_interactivity=False,
                                to_load_frames=False,
                                to_load_segmentation_metadata=False,
                                initialization_kwargs=None,
                                verbose=1):
        """Load all data (Dataframes, etc.) from disk using filenames defined in config files"""
        # Initialize object in order to use cached properties
        if initialization_kwargs is None:
            initialization_kwargs = {}
        else:
            cfg.logger.debug(f"Initialized project with custom settings: {initialization_kwargs}")

        obj = ProjectData(project_dir, cfg, **initialization_kwargs)

        obj.all_used_fnames = []
        preprocessing_settings = cfg.get_preprocessing_class(do_background_subtraction=False)

        red_dat_fname = preprocessing_settings.get_path_to_preprocessed_data(red_not_green=True)
        green_dat_fname = preprocessing_settings.get_path_to_preprocessed_data(red_not_green=False)
        red_traces_fname = traces_cfg.resolve_relative_path(traces_cfg.config['traces']['red'])
        green_traces_fname = traces_cfg.resolve_relative_path(traces_cfg.config['traces']['green'])

        # df_training_tracklets_fname = train_cfg.resolve_relative_path_from_config('df_training_3d_tracks')
        # reindexed_masks_training_fname = train_cfg.resolve_relative_path_from_config('reindexed_masks')

        # final_tracks_fname = tracking_cfg.resolve_relative_path_from_config('final_3d_tracks_df')
        seg_fname_raw = segment_cfg.resolve_relative_path_from_config('output_masks')
        seg_fname = traces_cfg.resolve_relative_path_from_config('reindexed_masks')

        # Metadata uses class from wbfm.utils.segmentation package, which does lazy loading itself
        seg_metadata_fname = segment_cfg.resolve_relative_path_from_config('output_metadata')
        obj.segmentation_metadata = DetectedNeurons(seg_metadata_fname)
        obj.physical_unit_conversion = PhysicalUnitConversion.load_from_config(cfg)

        # Read ahead of time because they may be needed for classes in the threading environment
        _ = obj.final_tracks
        zarr_reader_readwrite = lambda fname: zarr.open(fname, mode='r+')
        cfg.logger.debug("Starting threads to read data...")

        with safe_cd(cfg.project_dir):
            with concurrent.futures.ThreadPoolExecutor() as ex:
                if to_load_tracklets:
                    ex.submit(obj.load_tracklet_related_properties)
                if to_load_interactivity:
                    ex.submit(obj.load_interactive_properties)
                if to_load_frames:
                    ex.submit(obj.load_frame_related_properties)
                if to_load_segmentation_metadata:
                    ex.submit(obj.load_segmentation_related_properties)
                red_data = ex.submit(read_if_exists, red_dat_fname, zarr_reader_folder_or_zipstore).result()
                green_data = ex.submit(read_if_exists, green_dat_fname, zarr_reader_folder_or_zipstore).result()
                red_traces = ex.submit(read_if_exists, red_traces_fname).result()
                green_traces = ex.submit(read_if_exists, green_traces_fname).result()
                raw_segmentation = ex.submit(read_if_exists, seg_fname_raw, zarr_reader_readwrite).result()
                segmentation = ex.submit(read_if_exists, seg_fname, zarr_reader_folder_or_zipstore).result()

            if red_traces is not None:
                red_traces.replace(0, np.nan, inplace=True)
                green_traces.replace(0, np.nan, inplace=True)

        obj.all_used_fnames.extend([red_dat_fname, green_dat_fname, red_traces_fname, green_traces_fname,
                                    seg_fname_raw, seg_fname])
        cfg.logger.debug(f"Read all data from files: {obj.all_used_fnames}")

        background_per_pixel = preprocessing_settings.reset_background_per_pixel
        likelihood_thresh = traces_cfg.config['visualization']['likelihood_thresh']

        # Return a full object
        obj.red_data = red_data
        obj.green_data = green_data
        obj.raw_segmentation = raw_segmentation
        obj.segmentation = segmentation
        obj.red_traces = red_traces
        obj.green_traces = green_traces
        obj.background_per_pixel = background_per_pixel
        obj.likelihood_thresh = likelihood_thresh
        if verbose >= 1:
            cfg.logger.info(obj)
        else:
            cfg.logger.debug(obj)
        return obj

    @staticmethod
    def load_final_project_data(project_path, allow_hybrid_loading=False, **kwargs):
        """
        Wrapper function that dispatches to the correct loading function based on the input:
        load_final_project_data_from_nwb:
            - str (path to .nwb file)
        (otherwise): load_final_project_data_from_config:
            - ModularProjectConfig
            - str (path to project directory)
            - str (path to project_config.yaml)

        If it is already a project, return it as-is


        Parameters
        ----------
        project_path
        kwargs

        Returns
        -------

        """
        loaded_via_nwb = False
        if isinstance(project_path, str):
            if project_path.endswith('.nwb'):
                # Most kwargs are ignored here
                initialization_kwargs = kwargs.get('initialization_kwargs', dict())
                assert kwargs.get('load_tracklets', False) is False, "Tracklets are not supported for NWB files"
                project_data = ProjectData.load_final_project_data_from_nwb(project_path, **initialization_kwargs)
                loaded_via_nwb = True
            else:
                project_data = ProjectData.load_final_project_data_from_config(project_path, **kwargs)
        else:
            project_data = ProjectData.load_final_project_data_from_config(project_path, **kwargs)

        # Hybrid loading using both styles:
        #   If the project was loaded via a config file path, but has steps missing, then try to also load the nwb
        if not loaded_via_nwb and allow_hybrid_loading:
            try:
                # First check if there is an nwb file at all
                cfg_nwb = project_data.project_config.get_nwb_config()
            except PermissionError as e:
                project_data.logger.warning(f"Hybrid loading was set to True, but got a permission error; unable to load nwb. If there is no nwb file, this is not a problem. Full error: {e}")
                allow_hybrid_loading = False

        if not loaded_via_nwb and allow_hybrid_loading:

            nwb_filename = cfg_nwb.resolve_relative_path_from_config('nwb_filename')
            if nwb_filename is not None and os.path.exists(nwb_filename):
                project_data.logger.info(f"Found nwb file at {nwb_filename}; attempting to load additional data or analysis")
                initialization_kwargs = kwargs.get('initialization_kwargs', dict())
                project_data_nwb = ProjectData.load_final_project_data_from_nwb(nwb_filename, **initialization_kwargs)
                
                # First: raw data loaded from the nwb file
                nwb_preprocessing_class = project_data_nwb.project_config.get_preprocessing_class()
                preprocessing_class = project_data.project_config.get_preprocessing_class()

                def _update_project_with_nwb_file_handles():
                    project_data._nwb_io = project_data_nwb._nwb_io
                    project_data._nwb_obj = project_data_nwb._nwb_obj

                if not preprocessing_class.has_raw_data:
                    if nwb_preprocessing_class.has_raw_data:
                        # Merge the classes (data from nwb, proper config file links from project)
                        nwb_preprocessing_class.cfg_preprocessing = preprocessing_class.cfg_preprocessing
                        nwb_preprocessing_class.cfg_project = preprocessing_class.cfg_project
                        project_data.project_config._preprocessing_class = nwb_preprocessing_class
                        _update_project_with_nwb_file_handles()
                        project_data.logger.info(f"Successfully imported raw data from nwb file")
                    else:
                        project_data.logger.info(f"Did not find raw data in nwb file, continuing")

                # Define processing fields to attempt to load from the NWB project
                fields_to_load = [
                    ("red_data", True),
                    ("green_data", True),
                    ("red_traces", True),
                    ("green_traces", True),
                    ("segmentation", True),
                    ("raw_segmentation", True),
                    ("final_tracks", False),
                    ("intermediate_global_tracks", False),
                ]

                for field, update_handles in fields_to_load:
                    if getattr(project_data, field, None) is None:
                        nwb_value = getattr(project_data_nwb, field, None)
                        if nwb_value is not None:
                            setattr(project_data, field, nwb_value)
                            if update_handles:
                                _update_project_with_nwb_file_handles()
                            project_data.logger.info(
                                f"Successfully loaded {field.replace('_', ' ')} from nwb file"
                                f"{' (no metadata loaded)' if field in ['red_data', 'green_data', 'raw_segmentation'] else ''}"
                            )
                        else:
                            project_data.logger.info(
                                f"Did not find {field.replace('_', ' ')} in nwb file, continuing"
                            )
            else:
                project_data.logger.debug(f"Found no nwb file, continuing")

        return project_data

    @staticmethod
    def load_final_project_data_from_config(project_path: Union[str, os.PathLike, ModularProjectConfig],
                                            force_reload=False, **kwargs):
        """
        Main constructor that accepts multiple input formats
        This includes an already initialized ProjectData class, in which case this function returns

        valid input formats are:
            - path to project directory
            - path to project directory + project_config.yaml
            - ModularProjectConfig class (initialized with path to project directory)
            - ProjectData class (already initialized; this is returned as-is)

        valid kwargs are:
            to_load_tracklets=False,
            to_load_interactivity=False,
            to_load_frames=False,
            to_load_segmentation_metadata=False,
            initialization_kwargs=None
        """
        if isinstance(project_path, (str, os.PathLike)):
            if Path(project_path).is_dir():
                project_path = Path(project_path).joinpath('project_config.yaml')
            args = ProjectData.unpack_config_file(project_path)
            return ProjectData._load_data_from_configs(*args, **kwargs)
        elif isinstance(project_path, ModularProjectConfig):
            args = ProjectData.unpack_config_file(project_path)
            return ProjectData._load_data_from_configs(*args, **kwargs)
        elif isinstance(project_path, ProjectData):
            if force_reload:
                # Reload fully from disk, not a project_config class
                project_path = project_path.project_dir
                return ProjectData.load_final_project_data_from_config(project_path, **kwargs)
            else:
                return project_path
        else:
            raise TypeError("Must pass pathlike or already loaded project data")

    @staticmethod
    def load_final_project_data_from_nwb(nwb_path: Union[str, os.PathLike], **kwargs):
        """
        Like load_final_project_data_from_config, but for NWB files

        See https://github.com/focolab/NWBelegans for more information
        """

        # Do not use a with block in order to keep the file open
        # This allows "weak" closing of the file
        # https://docs.h5py.org/en/stable/high/file.html#closing-files

        from pynwb import NWBHDF5IO, NWBFile
        from wbfm.utils.general.preprocessing.utils_preprocessing import PreprocessingSettings
        from wbfm.utils.nwb.utils_nwb_export import convert_nwb_to_trace_dataframe

        nwb_io = NWBHDF5IO(nwb_path, mode='r', load_namespaces=True)
        if isinstance(nwb_io, NWBFile):
            obj.logger.debug('NWB file loaded successfully')
            nwb_obj = nwb_io
        else:
            nwb_obj = nwb_io.read()

        # Dummy project_config class (will give warnings)
        project_config = ModularProjectConfig(None)
        obj = ProjectData(nwb_path, project_config, **kwargs)

        # Fluorescence data
        preprocessing_settings = PreprocessingSettings()
        obj.project_config._preprocessing_class = preprocessing_settings
        if 'CalciumImageSeries' in nwb_obj.acquisition:
            # Transpose data from TXYZC to TZXY (splitting the channel)
            dat = nwb_obj.acquisition['CalciumImageSeries'].data
            chunks = (1, ) + dat.shape[1:-1] + (1,)
            if len(dat.shape) == 5 and dat.shape[-1] >= 2:
                obj.red_data = da.from_array(dat, chunks=chunks)[..., 0].transpose((0, 3, 1, 2))
                obj.green_data = da.from_array(dat, chunks=chunks)[..., 1].transpose((0, 3, 1, 2))
                obj.logger.debug(f"Loaded red and green data from NWB file: {obj.red_data.shape}")

                # Load this into the raw data as well; needed for certain steps
                preprocessing_settings._raw_red_data = da.from_array(dat, chunks=chunks)[..., 0].transpose((0, 3, 1, 2))
                preprocessing_settings._raw_green_data = da.from_array(dat, chunks=chunks)[..., 1].transpose((0, 3, 1, 2))

                if dat.shape[-1] > 2:
                    obj.logger.warning(f"Expected 1 or 2 channels in CalciumImageSeries, found {dat.shape[-1]}; ignoring extras")
            elif (len(dat.shape) == 5 and dat.shape[-1] == 1) or (len(dat.shape) == 4):
                # Then only one channel present; set both to be this data
                obj.red_data = da.from_array(dat, chunks=chunks).squeeze().transpose((0, 3, 1, 2))
                obj.green_data = da.from_array(dat, chunks=chunks).squeeze().transpose((0, 3, 1, 2))

                preprocessing_settings._raw_red_data = da.from_array(dat, chunks=chunks).squeeze().transpose((0, 3, 1, 2))
                preprocessing_settings._raw_green_data = da.from_array(dat, chunks=chunks).squeeze().transpose((0, 3, 1, 2))
                obj.logger.debug("WARNING, only one video channel found; setting both channels as this data")
                obj.logger.debug(f"Loaded data from NWB file: {obj.red_data.shape}")

        if 'RawCalciumImageSeries' in nwb_obj.acquisition:
            # Load this, but it's not actually part of the main ProjectData class
            # Transpose data from TXYZC to TZXY (splitting the channel)
            chunks = (1, ) + nwb_obj.acquisition['RawCalciumImageSeries'].data.shape[1:-1] + (1,)
            preprocessing_settings._raw_red_data = da.from_array(nwb_obj.acquisition['RawCalciumImageSeries'].data, chunks=chunks)[..., 0].transpose((0, 3, 1, 2))
            preprocessing_settings._raw_green_data = da.from_array(nwb_obj.acquisition['RawCalciumImageSeries'].data, chunks=chunks)[..., 1].transpose(
                (0, 3, 1, 2))

        # Traces
        try:
            # Load the traces, and the tracks using the same dataframes (they should all have xyz info)
            both_df_traces = convert_nwb_to_trace_dataframe(nwb_obj)
            # There may not be a reference, but there is always the signal
            obj.green_traces = both_df_traces['Signal']
            obj.red_traces = both_df_traces.get('Reference', obj.green_traces.copy())
            obj.final_tracks = obj.red_traces.copy()

        except KeyError as e:
            obj.logger.info(f"Could not load traces from NWB file: {e}")

        # Tracking (overwrites final_tracks above, if found)
        try:
            activity = nwb_obj.processing['CalciumActivity']

            from wbfm.utils.nwb.utils_nwb_export import load_per_neuron_position
            df_tracking = load_per_neuron_position(activity['NeuronCentroids'])
            obj.final_tracks = df_tracking
            obj.intermediate_global_tracks = df_tracking
        except KeyError as e:
            obj.logger.debug(f"Could not load tracks (centroids) from NWB file: {e}")

        try:
            activity = nwb_obj.processing['CalciumActivity']
            df_seg_id = activity['NeuronSegmentationID'].to_dataframe()
            # This is a dataframe with neuron names as columns, and segmentation ids as values
            # We want to join this as a subcolumn within the final_tracks dataframe, but first it needs to be made multiindexed with the column name 'raw_segmentation_id'
            df_seg_id.columns = pd.MultiIndex.from_product([df_seg_id.columns, ['raw_segmentation_id']])
            df_tracking = obj.final_tracks
            if df_tracking is not None:
                df_seg_id.index = df_tracking.index
                df_tracking = pd.concat([df_tracking, df_seg_id], axis=1)
            else:
                obj.logger.info("No final tracks found, saving only segmentation ids")
                df_tracking = df_seg_id
            obj.final_tracks = df_tracking
        except KeyError as e:
            obj.logger.info(f"Could not load segmentation ids from NWB file: {e}")


        # Segmentation
        try:
            activity = nwb_obj.processing['CalciumActivity']
            # Transpose data from TXYZ to TZXY
            dat = da.squeeze(activity['CalciumSeriesSegmentation'].data)
            chunks = (1, ) + dat.shape[1:]
            obj.segmentation = da.from_array(dat, chunks=chunks).transpose((0, 3, 1, 2))
        except (KeyError, AttributeError) as e:
            obj.logger.info(f"Could not load segmentation from NWB file: {e}")

        try:
            activity = nwb_obj.processing['CalciumActivity']
            # Transpose data from TXYZ to TZXY
            dat = da.squeeze(activity['CalciumSeriesSegmentationUntracked'].data)
            chunks = (1, ) + dat.shape[1:]
            obj.raw_segmentation = da.from_array(dat, chunks=chunks).transpose((0, 3, 1, 2))
        except (KeyError, AttributeError) as e:
            # Set to be equal to the segmentation, if it exists
            if obj.segmentation is not None:
                obj.raw_segmentation = obj.segmentation
                obj.logger.info(f"Could not load raw segmentation from NWB file ({e}); using tracked segmentation instead")
            else:
                obj.logger.info(f"Could not load raw segmentation from NWB file: {e}")

        # Other metadata
        p = PhysicalUnitConversion()
        if 'CalciumImageSeries' in nwb_obj.acquisition:
            p.volumes_per_second = nwb_obj.acquisition['CalciumImageSeries'].rate
            grid_spacing = nwb_obj.acquisition['CalciumImageSeries'].imaging_volume.grid_spacing
            if len(grid_spacing) == 3 and not grid_spacing[0] == grid_spacing[1]:
                logging.error(f"Grid should be isotropic in xy; found {grid_spacing}. Ignoring position 1, so errors in visualization may occur")
            xy_spacing = grid_spacing[0]
            p.zimmer_fluroscence_um_per_pixel_xy = xy_spacing
            p.zimmer_um_per_pixel_z = grid_spacing[-1]
            obj.physical_unit_conversion = p

        # Save the raw object
        obj._nwb_io = nwb_io
        obj._nwb_obj = nwb_obj

        return obj

    def calculate_traces(self, channel_mode: str,
                         calculation_mode: str,
                         neuron_name: str,
                         remove_outliers: bool = False,
                         filter_mode: str = 'no_filtering',
                         bleach_correct: bool = True,
                         residual_mode: str = None,
                         **kwargs) -> Tuple[list, pd.Series]:
        """
        Uses TracePlotter class to calculate traces

        In other words, creates a class, then uses a method from that class

        Parameters
        ----------
        channel_mode - red, green, ratio, df_over_f_20, ratio_df_over_f_20, dr_over_r_20, linear_model
        calculation_mode - integration (raw sum of pixels), volume, mean, z, likelihood (from the tracks dataframe)
        neuron_name - example: 'neuron_001'
        remove_outliers - try to remove spiking outliers
        filter_mode - try to filter
        min_confidence - if confidence below this, plot a gap
        residual_mode - for compatibility; intentionally unused here

        Returns
        -------

        time (as vector), y (as vector)

        """
        if 'background_per_pixel' in kwargs:
            self.background_per_pixel = kwargs['background_per_pixel']
        else:
            kwargs['background_per_pixel'] = self.background_per_pixel

        if self.project_config.has_valid_self_path:
            alternate_dataframe_folder = self.project_config.get_visualization_config().absolute_subfolder
        else:
            alternate_dataframe_folder = None

        self._trace_plotter = TracePlotter(
            self.red_traces,
            self.green_traces,
            self.final_tracks,
            channel_mode=channel_mode,
            calculation_mode=calculation_mode,
            remove_outliers=remove_outliers,
            filter_mode=filter_mode,
            bleach_correct=bleach_correct,
            alternate_dataframe_folder=alternate_dataframe_folder,
            **kwargs
        )
        y = self._trace_plotter.calculate_traces(neuron_name)
        return self._trace_plotter.tspan, y

    @property
    def neuron_names(self):
        """All names of neurons"""
        return get_names_from_df(self.red_traces)

    def well_tracked_neuron_names(self, min_nonnan=0.5, remove_invalid_neurons=False,
                                  confidence_threshold=1,
                                  rename_neurons_using_manual_ids=False, always_keep_manual_ids=False):
        """
        Subset of neurons that pass a given tracking threshold
        """

        min_nonnan = int(min_nonnan * self.num_frames)
        df_tmp = self.red_traces.dropna(axis=1, thresh=min_nonnan)
        neuron_names = get_names_from_df(df_tmp)
        if remove_invalid_neurons:
            invalid_names = self.finished_neuron_names(finished_not_invalid=False)
            neuron_names = [n for n in neuron_names if n not in invalid_names]

        # Optional: rename columns to use manual ids, if found
        mapping = self.neuron_name_to_manual_id_mapping(confidence_threshold=confidence_threshold,
                                                        remove_unnamed_neurons=True)
        if rename_neurons_using_manual_ids:
            neuron_names = [mapping[n] if n in mapping else n for n in neuron_names]

        # Optional: keep manual ids even if the neurons are not well tracked
        if always_keep_manual_ids:
            neuron_names = neuron_names + [n for n in mapping.keys() if n not in neuron_names]

        return neuron_names

    def calc_default_behaviors(self, min_nonnan: Optional[float] = None, interpolate_nan: bool = False,
                               raise_error_on_empty: bool = True,
                               neuron_names: tuple = None,
                               residual_mode: Optional[str] = None,
                               nan_tracking_failure_points: bool = False,
                               nan_using_ppca_manifold: bool = False,
                               remove_invalid_neurons: bool = True,
                               return_fast_scale_separation: bool = False,
                               return_slow_scale_separation: bool = False,
                               rename_neurons_using_manual_ids: bool = False,
                               z_score: bool = False,
                               binary_behaviors: bool = False,
                               add_constant: bool = False,
                               verbose=0,
                               **kwargs):
        """
        Like calc_default_traces, but for behaviors

        Returns a dataframe of behaviors, by default:
            - signed speed
            - angular speed
            - summed curvature
            - head curvature

        Options are kept for compatibility with calc_default_traces, but most are not used
        """

        if binary_behaviors:
            behavior_codes = ['rev', 'ventral_turn', 'dorsal_turn', 'pause', 'self_collision']
        else:
            behavior_codes = ['signed_middle_body_speed', 'ventral_only_body_curvature', 'ventral_only_head_curvature',
                              'dorsal_only_body_curvature', 'dorsal_only_head_curvature']
                              #'worm_nose_peak_frequency', 'worm_head_peak_frequency',
                              #'worm_speed_body_peak_frequency']
        behavior_dict = {}
        for code in behavior_codes:
            behavior_dict[code] = self.calculate_behavior_trace(code, **kwargs)[1]

        df = pd.DataFrame(behavior_dict)
        # Optional: nan time points that are estimated to have a tracking error (either global or per-neuron)
        if nan_tracking_failure_points:
            invalid_ind = self.estimate_tracking_failures_from_project(pad_nan_points=5)
            if invalid_ind is not None:
                df.loc[invalid_ind, :] = np.nan

        # Optional: fill all gaps
        if interpolate_nan:
            df_filtered = df.rolling(window=3, center=True, min_periods=2).mean()  # Removes size-1 holes
            for i in range(5):
                # Sometimes svd randomly doesn't converge; try again
                try:
                    df = impute_missing_values_in_dataframe(df_filtered, d=int(0.9*df.shape[1]))  # Removes larger holes
                    break
                except np.linalg.LinAlgError:
                    if i == 0:
                        self.logger.warning("SVD did not converge, trying again")
                    continue

        # Optional: z-score
        if z_score:
            df = (df - df.mean(axis=0)) / df.std(axis=0)

        # Optional: add a constant column
        if add_constant:
            df['constant'] = 1

        return df

    @lru_cache(maxsize=128)
    def calculate_behavior_trace(self, behavior_code, **kwargs):
        y = self.worm_posture_class.calc_behavior_from_alias(behavior_code).reset_index(drop=True)
        if "filter_mode" in kwargs:
            y = filter_trace_using_mode(y, kwargs['filter_mode'])
        return self.x_for_plots, y

    @lru_cache(maxsize=128)
    def calc_default_traces(self,
                            min_nonnan: Optional[float] = 0.75,
                            interpolate_nan: bool = False,
                            raise_error_on_empty: bool = True,
                            neuron_names: tuple = None,
                            residual_mode: Optional[str] = None,
                            nan_tracking_failure_points: bool = False,
                            nan_using_ppca_manifold: bool = False,
                            remove_invalid_neurons: bool = True,
                            return_fast_scale_separation: bool = False,
                            return_slow_scale_separation: bool = False,
                            rename_neurons_using_manual_ids: bool = False,
                            manual_id_confidence_threshold: int = 1,
                            use_physical_time: Optional[bool] = None,
                            remove_tail_neurons: bool = True,
                            always_keep_manual_ids: bool = True,
                            use_paper_options: bool = False,
                            only_keep_confident_ids: bool = False,
                            interpolate_nan_with_ppca: bool = False,
                            verbose=0,
                            **kwargs):
        """
        Core function for calculating dataframes of traces using various preprocessing options

        Note that all steps that can be calculated per-trace are implemented in the TracePlotter class.
            Other steps that require the full dataframe are implemented in this function

        Uses the currently recommended 'best' settings (which are the default):
        opt = dict(
            channel_mode='dr_over_r_50',
            calculation_mode='integration',
            remove_outliers=True,
            filter_mode='rolling_mean',
            high_pass_bleach_correct=False
        )

        if interpolate_nan is True, then additionally (after dropping empty neurons and removing outliers):
            1. Filter
            2. PPCA to fill in all gaps

        Parameters
        ----------
        min_nonnan: drops neurons with too few nonnan points, in this case 75%
        interpolate_nan: bool, see above
        raise_error_on_empty: if empty AFTER dropping, raise an error
        neuron_names: a subset of names to do
        nan_tracking_failure_points: Uses a simple heuristic (count number of neurons) to determine points of complete
            tracking failure, and removes all activity at those times
        nan_using_ppca_manifold: Uses a dimensionality heuristic to remove single-neuron mistakes. See OutlierRemover
            Note: iterative algorithm that takes around a minute
        high_pass_bleach_correct: Filters by removing very slow drifts, i.e. a gaussian of sigma = num_frames / 5
        remove_tail_neurons: Removes neurons that are annotated with "tail" in the manual annotation
        verbose
        kwargs: Args to pass to calculate_traces; updates the default 'opt' dict above
            See TracePlotter for options

        Returns
        -------

        """
        # Loads data from cache (actually used for the paper)
        # There are currently several cached versions of the data, depending on the residual, interpolation,
        # and channel options
        if use_paper_options:
            channel_mode = kwargs.get('channel_mode', 'dr_over_r_50')
            df = self.calc_paper_traces(channel_mode=channel_mode, residual_mode=residual_mode,
                                        interpolate_nan=interpolate_nan)
            # Most postprocessing is not done on these traces, but these are allowed
            if remove_tail_neurons:
                tail_names = self.tail_neuron_names()
                tail_names = [n for n in tail_names if n in get_names_from_df(df)]
                df = df.drop(columns=tail_names)
            return df

        opt = dict(
            channel_mode='dr_over_r_50',
            calculation_mode='integration',
            remove_outliers=True,
            filter_mode='rolling_mean',
            high_pass_bleach_correct=False
        )
        opt.update(kwargs)

        if neuron_names is None:
            user_passed_neuron_names = False
            neuron_names = tuple(self.neuron_names)
        else:
            user_passed_neuron_names = True
        if remove_invalid_neurons:
            invalid_names = self.finished_neuron_names(finished_not_invalid=False)
            neuron_names = tuple([n for n in neuron_names if n not in invalid_names])

        # TODO: this doesn't work if the only neuron name passed is a manually id'ed name
        df = self.calc_raw_traces(neuron_names, **opt).copy()

        # Shorten dataframe to only use expected number of time points
        df = df.iloc[:self.num_frames, :]

        # Optional: check neurons to remove
        if min_nonnan is not None:
            if not user_passed_neuron_names:
                names = self.well_tracked_neuron_names(min_nonnan, remove_invalid_neurons,
                                                       always_keep_manual_ids=always_keep_manual_ids)
                df_drop = df.loc[:, names].copy()
                # df_drop = df[names].copy()
            else:
                self.logger.warning("min_nonnan was passed, but neuron_names was also passed. Ignoring min_nonnan")
                df_drop = df
        else:
            df_drop = df

        if verbose >= 1:
            print(f"Dropped {df.shape[1] - df_drop.shape[1]} neurons with threshold {min_nonnan}/{df.shape[0]}")

        if df_drop.shape[1] == 0:
            msg = f"All neurons were dropped with a threshold of {min_nonnan}; check project.num_frames."\
                  f"If a video has very large gaps, num_frames should be set lower. For now, returning all"
            if raise_error_on_empty:
                raise NoNeuronsError(msg)
            else:
                logging.warning(msg)
                # Do not return dropped version
        else:
            df = df_drop

        # Optional: nan time points that are estimated to have a tracking error (either global or per-neuron)
        if nan_tracking_failure_points:
            invalid_ind = self.estimate_tracking_failures_from_project(pad_nan_points=5)
            if invalid_ind is not None:
                df.loc[invalid_ind, :] = np.nan
                # if interpolate_nan:
                #     self.logger.warning("Requested nan interpolation, but then nan were added due to tracking failures")

        if nan_using_ppca_manifold:
            try:
                to_remove_all_names = self.calc_indices_to_remove_using_ppca()
                # Subset the full removal matrix to only the neurons in this dataframe
                # to_remove_all_names is a matrix, so we can't directly index using pandas syntax
                if to_remove_all_names is not None and len(to_remove_all_names) > 0:
                    # We can cache a failure as None or empty matrix, so we need to check for that
                    names = get_names_from_df(df)
                    original_names = self.neuron_names
                    # Get the mapping between the names that have survived so far and the original names
                    name_ind = [original_names.index(n) for n in names]
                    to_remove = to_remove_all_names[:, name_ind]
                    df[to_remove] = np.nan
            except ValueError as e:
                self.logger.warning(f"PPCA failed with error: {e}, skipping manifold-based outlier removal")

        # Optional: fill all gaps (required for some steps)
        if residual_mode is not None and residual_mode != 'none':
            if not interpolate_nan:
                interpolate_nan = True
                self.logger.warning("Residual mode only works if nan are interpolated, enforcing that setting")
        if interpolate_nan:
            # if self._trace_plotter.filter_mode == 'no_filtering':
            #     df = df.rolling(window=3, center=True, min_periods=2).mean()  # Removes size-1 holes

            if interpolate_nan_with_ppca:
                for i in range(5):
                    # Sometimes svd randomly doesn't converge; try again
                    try:
                        df = impute_missing_values_in_dataframe(df, d=int(0.9*df.shape[1]))  # Removes larger holes
                        break
                    except np.linalg.LinAlgError:
                        if i == 0:
                            self.logger.warning("SVD did not converge, trying again")
                        continue
            # Finally, really make sure there are no nan
            df = fill_nan_in_dataframe(df, do_filtering=False)

        # Optional: reindex to physical time
        df.index = self.x_for_plots
        if use_physical_time:
            # Force reindexing, even if the class doesn't have the flag set
            df.index = self._x_physical_time

        # Optional: substract a dominant mode to get a residual
        if residual_mode is not None and residual_mode != 'none':
            if residual_mode == 'pca':
                df, _ = calculate_residual_subtract_pca(df, n_components=2)
            elif residual_mode == 'pca_global':
                _, df = calculate_residual_subtract_pca(df, n_components=2)
            elif residual_mode == 'nmf':
                df, _ = calculate_residual_subtract_nmf(df, n_components=2)
            elif 'pca_global' in residual_mode:
                # Then assume there is a number after pca, separate by an underscore
                n_components = int(residual_mode.split('_')[-1])
                _, df = calculate_residual_subtract_pca(df, n_components=n_components)
            else:
                raise NotImplementedError(f"Unrecognized residual mode: {residual_mode}")

        # Optional: separate fast and slow components, and return only one
        if return_fast_scale_separation and return_slow_scale_separation:
            raise ValueError("Cannot return both fast and slow scale separation")
        if return_fast_scale_separation or return_slow_scale_separation:
            df_fast, df_slow = fast_slow_decomposition(df)
            if return_fast_scale_separation:
                df = df_fast
            else:
                df = df_slow

        # Optional: remove neurons with specific manual annotations
        if remove_tail_neurons:
            tail_names = self.tail_neuron_names()
            tail_names = [n for n in tail_names if n in get_names_from_df(df)]
            df = df.drop(columns=tail_names)

        # Optional: rename columns to use manual ids, if found
        if rename_neurons_using_manual_ids:
            mapping = self.neuron_name_to_manual_id_mapping(confidence_threshold=manual_id_confidence_threshold,
                                                            only_include_confident_labels=only_keep_confident_ids)
            df = df.rename(columns=mapping)

        # Optional: set the index to be physical units
        if self.use_physical_time:
            df.index = self.x_for_plots

        return df

    def calc_indices_to_remove_using_ppca(self):
        return self.data_cacher.calc_indices_to_remove_using_ppca()

    def calc_paper_traces(self, channel_mode='dr_over_r_50', residual_mode=None, **kwargs):
        return self.data_cacher.paper_trace_dispatcher(channel_mode=channel_mode, residual_mode=residual_mode, **kwargs)

    def calc_all_paper_traces(self):
        all_methods = self.data_cacher.list_of_paper_trace_methods()
        all_methods_fnames = self.data_cacher.list_of_paper_trace_methods(return_simple_names=True)
        all_traces_dict = {name: method() for name, method in zip(all_methods_fnames, all_methods)}
        return all_traces_dict

    def copy_paper_traces_to_main_folder(self):
        """Copy all paper traces from the .cache folder to the traces folder"""
        source_folder = self.data_cacher.cache_dir
        target_folder = self.project_config.get_traces_config().absolute_subfolder
        # Copy every .h5 file
        for fname in os.listdir(source_folder):
            if fname.endswith('.h5'):
                source_path = os.path.join(source_folder, fname)
                target_path = os.path.join(target_folder, fname)
                shutil.copy(source_path, target_path)

    @lru_cache(maxsize=16)
    def calc_raw_traces(self, neuron_names: tuple, **opt: dict):
        """
        Calculates traces for a list of neurons in parallel
        Similar to calc_default_traces, but does not do any post-processing

        Parameters
        ----------
        neuron_names
        opt

        Returns
        -------

        """
        # Initialize the trace calculator class and get the initial dataframe
        _ = self.calculate_traces(neuron_name=neuron_names[0], **opt)
        trace_dict = dict()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(self._trace_plotter.calculate_traces, n): n for n in neuron_names}
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                trace_dict[name] = future.result()
        df = pd.DataFrame(trace_dict)
        df = df.reindex(sorted(df.columns), axis=1)
        return df

    def calc_pca_modes(self, n_components=10, flip_pc1_to_have_reversals_high=True, return_pca_weights=False,
                       return_pca_object=False, multiply_by_variance=False, combine_left_right=False, **trace_kwargs) \
            -> Tuple[pd.DataFrame, np.array]:
        """
        Calculates the PCA modes of the traces, and optionally flips the first mode to have reversals high
        This allows comparison of PC1 across datasets

        Returns the modes (or weights if return_pca_weights is True) and the explained variance as a vector
        OR: returns the PCA object itself, if return_pca_object is True

        Parameters
        ----------
        n_components
        flip_pc1_to_have_reversals_high
        return_pca_weights
        trace_kwargs

        Returns
        -------

        """
        trace_kwargs['interpolate_nan'] = True
        trace_kwargs['rename_neurons_using_manual_ids'] = True

        X = self.calc_default_traces(**trace_kwargs)
        if combine_left_right:
            X = combine_columns_with_suffix(X, suffixes=['L', 'R'], how='mean')
        X = fill_nan_in_dataframe(X, do_filtering=False)
        X -= X.mean()
        pca = PCA(n_components=n_components, whiten=False)
        if return_pca_weights:
            pca.fit(X)
            pca_weights = pca.components_.T
        pca.fit(X.T)
        pca_modes = pca.components_.T
        if multiply_by_variance:
            pca_modes *= pca.explained_variance_

        if return_pca_object:
            return pca

        if flip_pc1_to_have_reversals_high:
            # Calculate the speed, and define the sign of the first PC to be anticorrelated to speed
            reversal_time_series = None
            try:
                reversal_time_series = self.worm_posture_class.worm_speed(fluorescence_fps=True, reset_index=True,
                                                                          signed=True)
            except NoBehaviorAnnotationsError:
                pass

            # Instead of behavior, see if there is an ID'ed AVA neuron
            if reversal_time_series is None:
                for candidate_name in ['AVA', 'AVAL', 'AVAR']:
                    if candidate_name in X:
                        reversal_time_series = -X[candidate_name]
                        break
                else:
                    self.logger.warning("Could not calculate speed or AVA, so not flipping PC1")

            # If we have a reversal time series, flip the first PC to be anticorrelated with it
            if reversal_time_series is not None:
                correlation = np.corrcoef(pca_modes[:, 0], reversal_time_series)[0, 1]
                if correlation > 0:
                    if return_pca_weights:
                        pca_weights[:, 0] = -pca_weights[:, 0]
                    else:
                        pca_modes[:, 0] = -pca_modes[:, 0]

        if return_pca_weights:
            return pd.DataFrame(pca_weights, index=X.columns), pca.explained_variance_ratio_
        else:
            return pd.DataFrame(pca_modes, index=X.index), pca.explained_variance_ratio_

    def calc_correlation_to_pc1(self, **trace_kwargs):
        """
        Calculates the correlation of a trace to the first PC

        Parameters
        ----------
        name
        trace_kwargs

        Returns
        -------

        """
        trace_kwargs['interpolate_nan'] = True
        X = self.calc_default_traces(**trace_kwargs)
        X = fill_nan_in_dataframe(X, do_filtering=False)
        X -= X.mean()
        pca = PCA(n_components=1, whiten=True)
        pca.fit(X.T)
        pca_modes = pca.components_.T
        pc1 = pd.Series(pca_modes[:, 0])
        pc1.index = X.index
        correlation = X.corrwith(pc1)
        return correlation

    def calc_plateau_state_using_pc1(self, replace_nan=True, DEBUG=False, **trace_kwargs):
        # Get the trace that will be used to calculate the plateau state
        pca_modes, _ = self.calc_pca_modes(n_components=1, **trace_kwargs)
        pc1 = pd.Series(pca_modes.loc[:, 0])
        # Calculate plateaus using worm posture class method
        plateaus, working_pw_fits = self.worm_posture_class.calc_plateau_state_from_trace(pc1, n_breakpoints=2,
                                                                                          replace_nan=replace_nan,
                                                                                          DEBUG=DEBUG)
        return plateaus, working_pw_fits

    def plot_neuron_with_kymograph(self, neuron_name: str, **kwargs):
        """
        Plots a subplot with a neuron trace and the kymograph, if found

        Parameters
        ----------
        neuron_name

        Returns
        -------

        """
        # Note: neuron_name can't be a manually ided name at first
        name_mapping = self.neuron_name_to_manual_id_mapping(flip_names_and_ids=True,
                                                             remove_unnamed_neurons=True,
                                                             confidence_threshold=0)
        original_neuron_name = name_mapping.get(neuron_name, neuron_name)

        df = self.calc_default_traces(channel_mode='ratio', calculation_mode='integration',
                                      neuron_names=(original_neuron_name, ), **kwargs)
        # At this point, get the name of the neuron that was actually used
        t, y = df.index, df[original_neuron_name]
        df_kymo = self.worm_posture_class.curvature(fluorescence_fps=True)
        df_kymo.index = df.index

        fig, axes = plt.subplots(nrows=2, figsize=(30, 10), sharex=True)
        axes[0].imshow(df_kymo.T, origin="upper", cmap='seismic', extent=[0, df_kymo.shape[0], df_kymo.shape[1], 0],
                       aspect='auto', vmin=-0.06, vmax=0.06)
        axes[0].set_ylabel("Segment (0=nose)")
        axes[1].plot(t, y)
        axes[1].set_ylabel("Ratio (green/red)")
        plt.xlabel("Time (frames)")
        plt.xlim(0, self.num_frames)
        self.shade_axis_using_behavior()

    def save_fig_in_project(self, suffix='', overwrite=False):
        """
        Saves current figure within the project visualization directory, with optional suffix

        Parameters
        ----------
        suffix - suffix of the filename

        Returns
        -------
        Nothing

        """
        out_fname = f'fig-{suffix}.png'
        foldername = self.project_config.get_visualization_config(make_subfolder=True).absolute_subfolder
        out_fname = os.path.join(foldername, out_fname)
        if not overwrite:
            out_fname = get_sequential_filename(out_fname)

        plt.savefig(out_fname)

    def calculate_tracklets(self, neuron_name: str) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, str]:
        """
        Calculates tracklets using the tracklet_annotator class

        Returns all tracklets already attached, as well as any currently selected tracklets
        """
        y_dict, y_current, y_current_name = self.tracklet_annotator.get_tracklets_for_neuron(neuron_name)
        return y_dict, y_current, y_current_name

    def modify_confidences_of_frame_pair(self, pair, gamma, mode) -> list:
        """
        Postprocessing function to be applied to frame pairs that incorporates similarity in image space

        Parameters
        ----------
        pair
        gamma
        mode

        Returns
        -------

        """
        frame_match = self.raw_matches[pair]

        matches = frame_match.modify_confidences_using_image_features(self.segmentation_metadata,
                                                                      gamma=gamma,
                                                                      mode=mode)
        frame_match.final_matches = matches
        return matches

    def modify_confidences_of_all_frame_pairs(self, gamma, mode):
        """
        Loops modify_confidences_of_frame_pair across all frames

        Parameters
        ----------
        gamma
        mode

        Returns
        -------

        """
        frame_matches = self.raw_matches
        opt = dict(metadata=self.segmentation_metadata, gamma=gamma, mode=mode)
        for pair, obj in frame_matches.items():
            matches = obj.modify_confidences_using_image_features(**opt)
            obj.final_matches = matches

    def modify_segmentation_using_manual_correction(self, t=None, new_mask=None):
        """
        Modifies single segmentation, but does NOT update the disk, but rather a buffer zarr array in the
        tracklet_annotator class

        NOTE: will invalidate raw_frames and raw_matches if a new mask is created (not if one is just modified)

        Parameters
        ----------
        t
        new_mask

        Returns
        -------

        """
        if new_mask is None or t is None:
            new_mask = self.tracklet_annotator.candidate_mask
            t = self.tracklet_annotator.time_of_candidate
        if new_mask is None:
            self.logger.warning("Modification attempted, but no valid candidate mask exists; aborting")
            self.logger.warning("HINT: if you produce a mask but then click different neurons, it invalidates the mask!")
            self.logger.warning("===================================================================")
            self.logger.warning("DID NOT SAVE ANYTHING!!!")
            self.logger.warning("===================================================================")
            return
        if t is None:
            self.logger.warning("Modification attempted, but no valid time exists; aborting")
            self.logger.warning("===================================================================")
            self.logger.warning("DID NOT SAVE ANYTHING!!!")
            self.logger.warning("===================================================================")
            return

        affected_masks = self.tracklet_annotator.indices_of_original_neurons
        if len(affected_masks) == 0:
            self.logger.info("No saved affected masks found; this is fine if masks were changed manually")
            this_seg = self.raw_segmentation[t, ...]
            affected_ind = (this_seg - new_mask) != 0
            affected_masks = np.hstack([np.unique(this_seg[affected_ind]),np.unique(new_mask[affected_ind])])

        self.logger.info(f"Updating raw segmentation at t = {t}; affected masks={affected_masks}")
        self.tracklet_annotator.modify_buffer_segmentation(t, new_mask)
        # self.raw_segmentation[t, ...] = new_mask

        self.logger.info(f"Updating metadata at t, but NOT writing to disk...")
        red_volume = self.red_data[t, ...]
        self.segmentation_metadata.modify_segmentation_metadata(t, new_mask, red_volume)

        self.logger.info("Updating affected tracklets, but NOT writing to disk")
        for m in affected_masks:
            if m == 0:
                continue
            # Explicitly check to see if there actually was a tracklet before the segmentation was changed
            # Note that this metadata refers to the old masks, even if the mask is deleted above
            tracklet_name = self.tracklets_and_neurons_class.get_tracklet_from_segmentation_index(t, m)
            if tracklet_name is not None:
                self.tracklets_and_neurons_class.update_tracklet_metadata_using_segmentation_metadata(
                    t, tracklet_name=tracklet_name, mask_ind=m, likelihood=1.0, verbose=1
                )
                self.logger.info(f"Updating {tracklet_name} corresponding to segmentation {m}")
            else:
                self.logger.info(f"No tracklet corresponding to segmentation {m}; not updated")
        self.logger.debug("Segmentation and tracklet metadata modified successfully")

    def modify_segmentation_on_disk_using_buffer(self):
        """
        Modifies single segmentation ON DISK using tracklet_annotator class

        NOTE: invalidates raw_frames and raw_matches!!

        See: modify_segmentation_using_manual_correction

        """

        if self.tracklet_annotator.segmentation_updated:
            self.logger.info(f"Updating masks at t = {self.tracklet_annotator.t_buffer_masks}")
            self.logger.warning(f"Invalidates raw_frames and raw_matches; if tracking is needed, re-run all steps after"
                                f" segmentation")
            for t in self.tracklet_annotator.t_buffer_masks:
                self.raw_segmentation[t, ...] = self.tracklet_annotator.buffer_masks[t, ...]
            return True
        else:
            self.logger.info("No segmentation changes to save")
            return None

    def shade_axis_using_behavior(self, ax=None, plotly_fig=None, **kwargs):
        """
        Shades the currently active matplotlib axis using externally annotated behavior annotation
        OR, if plotly_fig is passed, shade a plotly figure

        Note: only works if self.beh_annotation is found

        Example with a loaded project:

        t, y = project_data.calculate_traces(channel_mode='ratio',
                                             calculation_mode='integration',
                                             neuron_name='neuron_001')
        plt.plot(t, y)
        project_data.shade_axis_using_behavior()
        plt.show()

        Parameters
        ----------
        ax - optional; takes the current one by default
        behaviors_to_ignore - integer of behaviors to ignore when shading

        Returns
        -------

        """
        if self.use_physical_time:
            index_conversion = self._x_physical_time
        else:
            index_conversion = None
        self.worm_posture_class.shade_using_behavior(ax=ax, plotly_fig=plotly_fig,
                                                     index_conversion=index_conversion, **kwargs)

    def get_centroids_as_numpy(self, i_frame):
        """Original format of metadata is a dataframe of tuples; this returns a normal np.array"""
        return self.segmentation_metadata.detect_neurons_from_file(i_frame)

    def get_centroids_as_numpy_training(self, i_frame: int, is_relative_index=True) -> np.ndarray:
        """Original format of metadata is a dataframe of tuples; this returns a normal np.array"""
        assert is_relative_index, "Only relative supported"

        return dataframe_to_numpy_zxy_single_frame(self.df_training_tracklets, t=i_frame)

    def get_distance_to_closest_neuron(self, i_frame, target_pt, nbr_obj=None) -> float:
        """
        Get the 3d distance between a target neuron and its closest neighbor

        Parameters
        ----------
        i_frame
        target_pt
        nbr_obj

        Returns
        -------

        """
        if nbr_obj is None:
            segmented_pts = self.get_centroids_as_numpy(i_frame)
            nbr_obj = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(segmented_pts)

        # Get point
        # target_pt = df_interp2[which_neuron].iloc[i_frame].to_numpy()[:3]

        # Get closest neighbor
        if not any(np.isnan(target_pt)):
            imputed_dist, _ = nbr_obj.kneighbors([target_pt], n_neighbors=1)
            dist = imputed_dist[0][0]
        else:
            dist = np.nan

        return dist

    def correct_relative_training_index(self, i) -> int:
        """Converts a relative index within the training set into a global index"""
        return self.which_training_frames[i]

    def napari_of_single_match(self, pair, which_matches='final_matches', this_match: FramePair = None,
                               rigidly_align_volumetric_images=False, min_confidence=0.0):
        """
        Visualize the matches between two volumes, including the raw segmentation

        Parameters
        ----------
        pair
        which_matches
        this_match
        rigidly_align_volumetric_images
        min_confidence

        Returns
        -------

        """
        from wbfm.utils.visualization.napari_from_project_data_class import NapariLayerInitializer
        v = NapariLayerInitializer.napari_of_single_match(self, pair, which_matches, this_match,
                                                          rigidly_align_volumetric_images, min_confidence)
        return v

    def napari_tracks_layer_of_single_neuron_match(self, neuron_name, t):
        """Helper function to get tracks layer between t and t+1. See also napari_of_single_match"""
        neuron_ind_in_list = self.final_tracks.loc[t, (neuron_name, 'raw_neuron_ind_in_list')]
        if np.isnan(neuron_ind_in_list):
            self.logger.debug(f"No match for {neuron_name} at t={t}")
        else:
            neuron_ind_in_list = int(neuron_ind_in_list)

        this_pair = self.raw_matches[(t, t + 1)]
        list_of_matches = this_pair.final_matches
        matches_class = MatchesWithConfidence.matches_from_array(list_of_matches)
        # If match is not found, use -1
        this_match = [[neuron_ind_in_list, matches_class.get_mapping_0_to_1(unique=True).get(neuron_ind_in_list, -1)]]
        tracks = this_pair.napari_tracks_of_matches(this_match)
        return tracks

    def add_layers_to_viewer(self, viewer=None, which_layers: Union[str, List[Union[str, Tuple]]] = 'all',
                             to_remove_flyback=False, check_if_layers_exist=False,
                             dask_for_segmentation=True, **kwargs):
        """
        Add layers corresponding to any analysis steps to a napari viewer object

        If no viewer is passed, then this creates a new one.

        By default, these layers are added to the viewer:
            ['Red data',
            'Green data',
            'Raw segmentation',
            'Neuron IDs',
            'Intermediate global IDs']

        An additional option that is not added:
            'GT IDs' (only exists if ground truth annotation is present)

        Additional special type, which should be passed as a tuple:
            ('heatmap', 'count_nonnan')
            ('heatmap', 'max_of_red')
            ('heatmap', 'max_of_green')
            ('heatmap', 'std_of_green')

        See: NapariPropertyHeatMapper for up-to-date list of options (all methods are valid)

        Parameters
        ----------
        viewer
        which_layers
        to_remove_flyback
        check_if_layers_exist
        dask_for_segmentation

        Returns
        -------

        """
        from wbfm.utils.visualization.napari_from_project_data_class import NapariLayerInitializer
        v = NapariLayerInitializer.add_layers_to_viewer(self, viewer, which_layers,
                                                        to_remove_flyback, check_if_layers_exist,
                                                        dask_for_segmentation, error_if_missing_layers=False,
                                                        **kwargs)
        return v

    def get_desynced_seg_and_frame_object_frames(self, verbose=1) -> List[int]:
        """
        Return frame objects that are obviously desynced from the segmentation

        Desyncing is flagged by two different numbers of objects stored in the different metadata
        """
        desynced_frames = []
        for t in range(self.num_frames):
            pts_from_seg = self.get_centroids_as_numpy(t)
            pts_from_frame = self.raw_frames[t].neuron_locs

            if pts_from_seg.shape != pts_from_frame.shape:
                desynced_frames.append(t)
        if verbose >= 1:
            self.logger.warning(f"Found {len(desynced_frames)} desynchronized frames")
        return desynced_frames

    @cached_property
    def df_manual_tracking(self) -> Optional[pd.DataFrame]:
        """
        Load a dataframe corresponding to manual tracking, i.e. which neurons have been manually corrected or ID'ed

        """
        # Manual annotations take precedence by default
        if not self.project_config.has_valid_self_path:
            self.logger.debug("No project config found; cannot load manual annotations")
            return None
        excel_fname = self.get_default_manual_annotation_fname()
        try:
            possible_fnames = dict(excel=excel_fname,
                                   csv_old=Path(excel_fname).with_suffix('.csv'),
                                   csv=Path(excel_fname).with_name(self.shortened_name).with_suffix('.csv'),
                                   h5=Path(excel_fname).with_name(self.shortened_name).with_suffix('.h5'),
                                   h5_backup=Path(excel_fname).with_suffix('.h5'))
        except ValueError:
            self.df_manual_tracking_fname = ''
            return None
        possible_fnames = {k: str(v) for k, v in possible_fnames.items()}
        # fname_precedence = ['excel', 'csv', 'h5']  # This precedence is used because humans may have added things
        fname_precedence = ['newest']
        try:
            df_manual_tracking, fname = load_file_according_to_precedence(fname_precedence, possible_fnames,
                                                                          reader_func=read_if_exists, na_filter=False)
        except ValueError:
            # Then the file was corrupted... try to load from the h5 file (don't worry about other file types)
            fname = possible_fnames['h5_backup']
            if Path(fname).exists():
                df_manual_tracking = pd.read_hdf(fname)
                # Confirm that the excel file exists but was corrupt
                if Path(excel_fname).exists():
                    try:
                        _ = pd.read_excel(excel_fname)
                        raise DataSynchronizationError("Corrupt excel file, but was able to read it the second time")
                    except ValueError:
                        # This is expected from the first read, and we want to fix the file
                        df_manual_tracking.to_excel(excel_fname, index=False)
                        self.logger.warning(
                            f"Found corrupted manual annotation file ({excel_fname}), overwriting from h5 file")

                        # And then actually read this file, so we don't touch the backup
                        df_manual_tracking, fname = load_file_according_to_precedence(fname_precedence, possible_fnames,
                                                                                      reader_func=read_if_exists, na_filter=False)
            else:
                self.logger.warning(
                    f"Found corrupted manual annotation file ({excel_fname}), with no backup h5 file; returning None")

                fname = ''
                df_manual_tracking = None
        self.df_manual_tracking_fname = fname
        return df_manual_tracking

    def get_default_manual_annotation_fname(self, neuropal_subproject=False) -> Optional[str]:
        if self.project_config is None:
            raise IncompleteConfigFileError("No project config found; cannot load or save manual annotations")
        if neuropal_subproject:
            excel_fname = self.neuropal_manager.get_default_manual_annotation_fname()
        else:
            track_cfg = self.project_config.get_tracking_config()
            excel_fname = track_cfg.resolve_relative_path("manual_annotation/manual_annotation.xlsx",
                                                          prepend_subfolder=True)
        return excel_fname

    def build_neuron_editor_gui(self, neuropal_subproject=False):
        """
        Initialize a QT table interface for editing neurons

        Designed to be used with NapariTraceExplorer

        Returns
        -------

        """
        if neuropal_subproject:
            df = self.neuropal_manager.df_ids
        else:
            df = self.df_manual_tracking

        if df is None:
            # Generate a default dataframe, with hardcoded names
            df = pd.DataFrame(columns=['Neuron ID', 'Finished?', 'ID1', 'ID2', 'Certainty', 'does_activity_match_ID',
                                       'paired_neuron', 'Interesting_not_IDd', 'Notes'])
            # Fill the first column with the default names
            df['Neuron ID'] = self.neuropal_manager.neuron_names if neuropal_subproject else self.neuron_names
            df['Certainty'] = 0
            try:
                fname = self.get_default_manual_annotation_fname(neuropal_subproject=neuropal_subproject)
            except IncompleteConfigFileError:
                return None
        else:
            # Ensure formatting of output name
            if neuropal_subproject:
                fname = self.neuropal_manager.df_id_fname
            else:
                fname = self.df_manual_tracking_fname

            if not fname.endswith('.xlsx'):
                # Make sure the output is excel even if the input isn't
                self.logger.warning(f"Found manual annotation file at ({fname})")
                fname = str(Path(fname).with_suffix('.xlsx'))
                self.logger.warning(f", but will save as excel file at ({fname})")

        # Enforce certain datatypes
        df['Neuron ID'] = df['Neuron ID'].astype(str)
        df['ID1'] = df['ID1'].fillna(df['Neuron ID']).astype(str)
        df['ID2'] = df['ID2'].astype(str)
        # Replace other NaNs with empty strings
        df = df.fillna('')
        df = df.replace('nan', '')
        # Certainty should be floats, first remove any non-numeric characters (but print a warning if found)
        try:
            df['Certainty'] = df['Certainty'].replace('', 0).replace('nan', 0).fillna(0).astype(int)
        except ValueError:
            self.logger.warning("Found non-numeric entries in Certainty column; replacing with 0")
            non_numeric = df['Certainty'].apply(lambda x: not isinstance(x, (int, float, np.integer)))
            df.loc[non_numeric, 'Certainty'] = 0
            df['Certainty'] = df['Certainty'].fillna(0).astype(int)
        
        # Get the list of neuron ids that the user wants to id, if found
        vis_config = self.project_config.get_visualization_config()
        neurons_to_id = vis_config.config.get('neurons_to_id', None)
        if neurons_to_id is None:
            # Try to load from the cluster, if mounted linux-style
            neurons_to_id = read_names_of_neurons_to_id()
        else:
            neurons_to_id = list(neurons_to_id)
        if neurons_to_id is None:
            self.logger.warning("Could not find neurons_to_id; that column will not work")

        # Actually build the class
        try:
            from wbfm.gui.utils.utils_gui import NeuronNameEditor
            manual_neuron_name_editor = NeuronNameEditor(neurons_to_id=neurons_to_id)
            manual_neuron_name_editor.import_dataframe(df, fname)
        except (PermissionError, tables.exceptions.HDF5ExtError):
            self.logger.warning(f"Could not open manual annotation file at ({fname}); "
                                f"will not be able to save, thus this GUI will not be opened")
            manual_neuron_name_editor = None

        return manual_neuron_name_editor

    def dict_numbers_to_neuron_names(self, neuropal_subproject=False) -> Dict[str, Tuple[str, int]]:
        """
        Uses df_manual_tracking to map neuron numbers to names and confidence. Example:
            dict_numbers_to_neuron_names['neuron_001'] -> ['AVAL', 2]

        Assumes the following column names:
            output keys = 'Neuron ID'
            output values = ['ID1', 'Certainty']

        Certainty goes from 0-2, with 2 being the most certain

        Some files my have an 'ID2' column for less certain ID's; this is ignored

        Returns
        -------

        """
        # Read each column, and create a dictionary
        if neuropal_subproject:
            df = self.neuropal_manager.df_ids
        else:
            df = self.df_manual_tracking
        if df is None:
            return {}
        # Strip white space from column names
        df.columns = df.columns.str.strip()

        # Get the automatically assigned (meaningless) neuron names
        neuron_names = df['Neuron ID'].values
        neuron_names = [str(i) for i in neuron_names]

        # Get the most confident ID. If not present (nan), will be an empty string
        neuron_ids = df['ID1'].values
        neuron_ids = ['' if isinstance(i, float) and np.isnan(i) else str(i) for i in neuron_ids]

        # Get the certainty. If not present (nan), will be 0
        neuron_certainty = df['Certainty'].values
        # First check that they are a good type, otherwise replace with nan
        neuron_certainty = [i if isinstance(i, (float, int, np.integer)) else np.nan for i in neuron_certainty]
        neuron_certainty = [0 if np.isnan(i) else int(i) for i in neuron_certainty]

        # Create a dictionary
        neuron_dict = dict(zip(neuron_names, zip(neuron_ids, neuron_certainty)))
        return neuron_dict

    def neuron_name_to_manual_id_mapping(self, confidence_threshold=2, remove_unnamed_neurons=False,
                                         flip_names_and_ids=False, error_on_duplicate=False,
                                         remove_duplicates=True, neuropal_subproject=False,
                                         only_include_confident_labels=False) -> \
            Dict[str, str]:
        """
        Note: if confidence_threshold is 0, then non-id'ed neuron names will be removed because
        dict_numbers_to_neuron_names has a blank string at confidence 0

        Note also that neurons that do not exist in the trace dictionary can be present in the df_manual_tracking and
        therefore in the neuron_name_to_manual_id_mapping. This is because we are reading from an excel file, which
        may have been manually edited

        Note that only_include_confident_labels will remove unnamed neurons

        Parameters
        ----------
        confidence_threshold
        remove_unnamed_neurons
        flip_names_and_ids
        error_on_duplicate
        only_include_confident_labels

        Returns
        -------

        """
        name_ids = self.dict_numbers_to_neuron_names(neuropal_subproject=neuropal_subproject).copy()
        if len(name_ids) == 0:
            return {}
        name_mapping = {k: (v[0] if (v[1] >= confidence_threshold and v[0] != '') else k) for k, v in name_ids.items()}
        if only_include_confident_labels:
            confident_labels = neurons_with_confident_ids()
            name_mapping = {k: v for k, v in name_mapping.items() if v in confident_labels}
        # Check that there are no duplicates within the values
        value_counts = pd.Series(name_mapping.values()).value_counts()
        message = f"Duplicate values found in neuron_name_to_manual_id_mapping: " \
                  f"{list(value_counts[value_counts > 1].index)} (dataset: {self.project_dir})"
        if len(value_counts[value_counts > 1]) > 0:
            if error_on_duplicate:
                raise ValueError(message)
            else:
                # Check if the only duplicate is the empty string
                if len(value_counts[value_counts > 1]) == 1 and '' in value_counts:
                    pass
                else:
                    if self.verbose >= 1:
                        self.logger.warning(message)
                    else:
                        self.logger.debug(message)
                    if remove_duplicates:
                        # Keep the first instance of any duplicate, replacing the name with the original name
                        replaced_names = []
                        for (k_map, v_map), (k_orig, v_orig) in zip(name_mapping.items(), name_ids.items()):
                            if list(name_mapping.values()).count(v_map) > 1 and k_map not in replaced_names:
                                name_mapping[k_map] = k_orig
                                replaced_names.append(k_map)

                        message = f"Removed duplicates, leaving only the first instance of each"
                        if self.verbose >= 1:
                            self.logger.warning(message)
                        else:
                            self.logger.debug(message)
        if remove_unnamed_neurons:
            name_mapping = {k: v for k, v in name_mapping.items() if k != v}
        if flip_names_and_ids:
            # If the neuron name is '', then save the key instead
            name_mapping = {k if v == '' else v: k for k, v in name_mapping.items()}
        return name_mapping

    def tail_neuron_names(self):
        """
        Searches the "Notes" column of the manual annotation file for neurons that have been marked as "tail"

        Returns
        -------

        """
        if self.df_manual_tracking is not None and 'Notes' in self.df_manual_tracking:
            tail_ids = self.df_manual_tracking['Notes'].str.contains('tail', case=False)
            names = list(self.df_manual_tracking['Neuron ID'][tail_ids].values)
        else:
            names = []
        return names

    def estimate_tracking_failures_from_project(self, pad_nan_points=3, contamination=0.1,#'auto',
                                                min_decrease_threshold=40,
                                                DEBUG=False):
        """
        Uses sudden dips in the number of detected objects to guess where the tracking might fail

        Additionally, pads contiguous regions of tracking failure, assuming that the tracking was incorrect before and after
        the times it was detected

        Parameters
        ----------
        pad_nan_points
        contamination
        min_decrease_threshold - minimum number of objects that must be lost to be considered a tracking failure

        Returns
        -------

        """
        try:
            all_vol = [self.segmentation_metadata.get_all_volumes(i) for i in range(self.num_frames)]
        except (AttributeError, FileNotFoundError) as e:
            self.logger.warning(f"Error with reading segmentation, may be due to python version: {e}")
            return None
        all_num_objs = np.array(list(map(len, all_vol)))
        all_num_objs = detrend(all_num_objs)
        model = LocalOutlierFactor(contamination=contamination)
        vals = model.fit_predict(all_num_objs.reshape(-1, 1))
        # Get outliers, but only care about decreases in objects, not increases
        vals[all_num_objs > 0] = 1
        # Additionally filter by an absolute large-enough decrease in objects
        all_object_deviations = all_num_objs - np.mean(all_num_objs)
        vals[np.abs(all_object_deviations) < min_decrease_threshold] = 1

        if DEBUG:
            print(np.where(vals == -1))

        # Pad the discovered blocks
        if pad_nan_points is not None:
            df_vals = pd.Series(vals == -1)
            starts, ends = get_contiguous_blocks_from_column(df_vals, already_boolean=True)
            if DEBUG:
                print(starts, ends)
            idx_boolean = make_binary_vector_from_starts_and_ends(starts, ends, vals).astype(bool)
        else:
            idx_boolean = vals == -1

        # Get outliers, but only care about decreases in objects, not increases
        invalid_idx = np.where(idx_boolean)[0]
        invalid_idx = np.array([i for i in invalid_idx if all_num_objs[i] < 0])

        return invalid_idx

    @cached_property
    def df_exported_format(self) -> Optional[pd.DataFrame]:
        """
        Loads a previously exported summary dataframe, including behavior and multiple calculations of traces

        returns None if the file doesn't exist

        See project_export.save_all_final_dataframes
        """

        fname = Path(self.project_dir).joinpath('final_dataframes/df_final.h5')
        try:
            df_final = pd.read_hdf(fname)
            return df_final
        except FileNotFoundError:
            return None

    # @lru_cache(maxsize=2)
    def finished_neuron_names(self, finished_not_invalid=True) -> List[str]:
        """
        Uses df_manual_tracking to get a list of the neuron names that have been fully corrected, or are invalid

        By default, returns the finished neurons, corresponding to the column 'Finished?'
            Otherwise, uses the column 'Invalid?'

        The manual annotation file is expected to be a .csv in the following format:
        Neuron ID, Finished?, Invalid?
        neuron_001, False, True
        neuron_002, True, False
        ...

        Extra columns are not a problem, but extra rows are
        """
        df_manual_tracking = self.df_manual_tracking
        if df_manual_tracking is None:
            return []

        if finished_not_invalid:
            column_name = None
        else:
            column_name = 'Invalid?'

        try:
            neurons_finished_mask = self._check_format_and_unpack(df_manual_tracking, column_name=column_name)
            neurons_in_column = list(df_manual_tracking[neurons_finished_mask]['Neuron ID'])

            # Filter to make sure they are the proper format
            tmp = []
            for col_name in neurons_in_column:
                if isinstance(col_name, str):
                    tmp.append(col_name)
                else:
                    self.logger.warning(f"Found and removed improper column name in manual annotation : {col_name}")
            neurons_in_column = tmp
        except KeyError:
            if column_name != "Invalid?":
                # This one is commonly missing
                self.logger.warning(f"Requested manual annotation column not found: {column_name}")
            neurons_in_column = []

        return neurons_in_column

    @cached_property
    def neurons_with_ids(self) -> Optional[pd.DataFrame]:
        """
        Loads excel file with information about manually id'ed neurons

        Note: in newer datasets, this information is stored in df_manual_tracking instead of a separate file

        Returns
        -------

        """

        # File should have the same name as the folder
        vis_config = self.project_config.get_visualization_config()
        fname = f"{self.shortened_name}.xlsx"
        fname = vis_config.resolve_relative_path(fname, prepend_subfolder=True)

        if Path(fname).exists():
            return fix_extra_spaces_in_dataframe_columns(pd.read_excel(fname))
        else:
            return None

    def _check_format_and_unpack(self, df_manual_tracking, column_name=None):
        if column_name is None:
            column_name = self.finished_neurons_column_name
        neurons_finished_mask = df_manual_tracking[column_name]
        # If it is boolean, then we are done
        # Otherwise, check for simple strings
        if neurons_finished_mask.dtype != bool:
            neurons_finished_mask = neurons_finished_mask.map({'True': True, 'False': False,
                                                               'TRUE': True, 'FALSE': False,
                                                               'true': True, 'false': False,
                                                               True: True, False: False,
                                                               '': False, 'nan': False, 'NaN': False, 'NAN': False,})
            neurons_finished_mask = neurons_finished_mask.fillna(False).astype(bool)
        if 'Neuron ID' not in df_manual_tracking[neurons_finished_mask]:
            self.logger.warning("Did not find expected column name ('Neuron ID') for the neuron ids... "
                                "check the formatting of the manual annotation file")
        return neurons_finished_mask

    @property
    def shortened_name(self):
        """Returns the project directory name of the project"""
        return str(Path(self.project_dir).name)

    @property
    def more_shortened_name(self):
        """
        Returns a shortened version project directory name of the project

        Expects a name like 'ZIM2319_GFP_worm1-2022-12-10', and removes the date and splits by underscores
        In that example, returns 'worm1'
        """
        name = self.shortened_name.split('-')[0]
        if len(name.split('_')) > 1:
            name = name.split('_')[-1]
        return name

    # Functions for printing analysis statistics
    def print_seg_statistics(self):
        print("=======================================\n")
        print("Expectation: frames equal to total video")
        print("Expectation: ~140-180 neurons; ~200 is okay")
        print("Expectation: Brightness 300000-400000")
        print("Expectation: Volume: 600-800")
        self.segmentation_metadata.print_statistics(detail_level=2)
        print()

    def print_tracklet_statistics(self):
        print("=======================================\n")
        print("Expectation: ~140-180 neurons; ~200 is okay")
        print("Expectation: ~90-95% matched")

        # Check raw frames and matches, then check actual tracklets
        print(f"Found {len(self.raw_frames)} Frame objects")
        for i in range(5):
            print(self.raw_frames[i])

        print(f"Found {len(self.raw_matches)} Matches objects")
        for i in range(5):
            key = (i, i+1)
            print(self.raw_matches[key])

        print(self.tracklets_and_neurons_class)
        print()

    def print_tracking_statistics(self):
        print("=======================================\n")
        worm = self.tracked_worm_class
        worm.verbose = 2

        print(worm)
        print()

    @property
    def use_physical_time(self) -> bool:
        """Whether to reindex returned traces to physical time"""
        return self._use_physical_time

    @use_physical_time.setter
    def use_physical_time(self, value: bool):
        assert isinstance(value, bool), "use_physical_time must be a boolean"
        self._use_physical_time = value
        # Also set the internal behavioral class to use physical time
        self.worm_posture_class.use_physical_time = value

    @property
    def _x_physical_time(self):
        """Helper for reindexing plots from volumes to seconds"""
        x = np.arange(self.num_frames)
        x = x / self.physical_unit_conversion.volumes_per_second
        return x

    @property
    def x_for_plots(self):
        """
        Helper for reindexing plots from volumes to seconds

        Uses self._x_physical_time and self.use_physical_x_axis to return the desired time
        """
        if self.use_physical_time:
            x = self._x_physical_time
        else:
            x = np.arange(self.num_frames)
        return x

    @property
    def x_label_for_plots(self) -> str:
        """
        Helper for reindexing plots from volumes to seconds

        Uses self._x_physical_time and self.use_physical_x_axis to return the desired string
        """
        if self.use_physical_time:
            label = "Time (s)"
        else:
            label = "Time (volumes)"
        return label

    @property
    def raw_data_dir(self):
        if self.project_config is None:
            return None
        try:
            return self.project_config.get_folder_for_all_channels()
        except IncompleteConfigFileError:
            self.logger.warning("No project config found; cannot load raw data directory")
            return None

    @property
    def x_lim(self):
        """
        Returns first and last element of self.x_for_plots
        """
        x = self.x_for_plots
        return [x[0], x[-1]]
    
    # Functions for the current status of a project (and which steps have been completed)
    def check_preprocessed_data(self):
        return (self.red_data is not None) and (self.green_data is not None)
    
    def check_segmentation(self):
        return (self.raw_segmentation is not None) and (self.check_segmentation_metadata())
    
    def check_segmentation_metadata(self):
        has_metadata = True
        try:
            _ = self.segmentation_metadata.num_frames
        except FileNotFoundError:
            has_metadata = False
        return has_metadata
    
    def check_tracking(self):
        return self.final_tracks is not None
    
    def check_traces(self):
        return (self.red_traces is not None) and (self.green_traces is not None) and (self.segmentation is not None)

    def __repr__(self):
        return f"=======================================\n\
Project data for directory:\n\
{self.project_dir} \n\
With raw data in directory:\n\
{self.raw_data_dir} \n\
See self.worm_posture_class for information on behavioral parameters\
\n\
Found the following data files:\n\
============Preprocessed Videos========\n\
red_data:                 {self.red_data is not None}\n\
green_data:               {self.green_data is not None}\n\
============Segmentation===============\n\
raw_segmentation:         {self.raw_segmentation is not None}\n\
colored_segmentation:     {self.segmentation is not None}\n\
============Tracking===================\n\
tracklets:                {self.has_tracklets()}\n\
final_tracks:             {self.check_tracking()}\n\
manual_tracking:          {self.df_manual_tracking is not None}\n\
============Traces=====================\n\
traces:                   {self.check_traces()}\n\n"


def template_matches_to_dataframe(project_data: ProjectData,
                                  all_matches: list,
                                  null_value=-1):
    """
    Extracts locations of the matches given by the second column of all_matches, and names them like the first column

    Parameters
    ----------
    project_data
    all_matches - Correct null value is []
    null_value

    Returns
    -------

    """
    num_frames = len(all_matches)
    coords = ['z', 'x', 'y', 'likelihood', 'raw_neuron_ind_in_list']
    sz = (num_frames, len(coords))
    neuron_arrays = defaultdict(lambda: np.zeros(sz))

    for i_frame, these_matches in enumerate(tqdm(all_matches, leave=False)):
        pts = project_data.get_centroids_as_numpy(i_frame)
        # For each match, save location
        for m in these_matches:
            this_unscaled_pt = pts[int(m[1])]
            this_template_idx = m[0]

            # These columns must match the order of 'coords' above
            if null_value in m:
                neuron_arrays[this_template_idx][i_frame, :] = np.nan
            else:
                neuron_arrays[this_template_idx][i_frame, :3] = this_unscaled_pt
                neuron_arrays[this_template_idx][i_frame, 3] = m[2]  # Match confidence
                neuron_arrays[this_template_idx][i_frame, 4] = int(m[1])  # Match index

    # Convert to pandas multiindexing formatting
    new_dict = {}
    for i_template, data in neuron_arrays.items():
        for i_col, coord_name in enumerate(coords):
            # NOTE: these neuron names are final for all subsequent steps
            k = (int2name_neuron(int(i_template + 1)), coord_name)
            new_dict[k] = data[:, i_col]

    df = pd.DataFrame(new_dict)

    return df


def calc_mismatch_between_ground_truth_and_pairs(all_matches: Dict[tuple, FramePair],
                                                 df_gt: pd.DataFrame,
                                                 t0: int,
                                                 minimum_confidence: float):
    """
    Helper function for calculating the mismatches between a dict of matches (t to t+1), and a ground truth dataframe
    """
    pair = (t0, t0 + 1)

    model_matches = all_matches[pair].final_matches
    # model_matches = all_matches[pair].feature_matches
    gt_matches = df_to_matches(df_gt, t0)

    model_obj = MatchesWithConfidence.matches_from_array(model_matches, minimum_confidence=minimum_confidence)
    gt_obj = MatchesWithConfidence.matches_from_array(gt_matches, 1)
    correct_matches, gt_matches_different_model, model_matches_different_gt, model_matches_no_gt, gt_matches_no_model = \
        get_mismatches(gt_obj, model_obj)

    return gt_matches_different_model, model_matches_different_gt, model_matches_no_gt, gt_matches_no_model


def calc_all_mismatches_between_ground_truth_and_pairs(project_data: ProjectData,
                                                       minimum_confidence=0.0) -> Dict[str, list]:
    """
    Calculates all mismatches between the ground truth and the matches class

    Return a dictionary indexed by a tuple:
        (time, neuron_name, tracklet_name) = bool

    (In principle this could be a list)

    See: calc_mismatch_between_ground_truth_and_pairs

    Parameters
    ----------
    minimum_confidence
    project_data

    Returns
    -------

    """
    all_mismatches = defaultdict(list)
    all_matches = project_data.raw_matches
    df_gt, finished_neurons = project_data.get_final_tracks_only_finished_neurons()
    if len(finished_neurons) == 0:
        return all_mismatches

    project_data.logger.info("Calculating mismatches between ground truth and automatic matches")
    for t0 in tqdm(range(project_data.num_frames - 2)):
        # Only need the first of the pair here, i.e. the index on t0, not t0+1
        try:
            gt_matches_different_model, model_matches_different_gt, _, _ = \
                calc_mismatch_between_ground_truth_and_pairs(all_matches, df_gt, t0,
                                                             minimum_confidence=minimum_confidence)
        except NoMatchesError:
            continue

        if gt_matches_different_model:
            for gt_mismatch, model_mismatch in zip(gt_matches_different_model, model_matches_different_gt):
                raw_neuron_ind_in_list = gt_mismatch[0]
                ind, neuron_name = get_column_name_from_time_and_column_value(df_gt, t0, raw_neuron_ind_in_list,
                                                                              col_name='raw_neuron_ind_in_list')

                # Convert to segmentation to use the tracklet class pre-allocated dict
                try:
                    mask_ind = project_data.segmentation_metadata.i_in_array_to_mask_index(t0, raw_neuron_ind_in_list)
                    tracklet_name = project_data.tracklets_and_neurons_class.get_tracklet_from_segmentation_index(t0,
                                                                                                                  mask_ind)
                    all_mismatches[neuron_name].append((t0, tracklet_name, model_mismatch, gt_mismatch))
                except IndexError:
                    logging.warning(f"Index error for {t0, raw_neuron_ind_in_list}, implies data desynchronization")

    logging.info(f"Found {sum(map(len, all_mismatches.values()))} mismatches")

    return all_mismatches


def print_project_statistics(project_config: ModularProjectConfig):
    """
    Prints basic statistics of the output of each step that is completed

    Parameters
    ----------
    project_config

    Returns
    -------

    """

    last_finished_step = get_project_status(project_config)
    # Load everything possible, because we will use it
    project_data = ProjectData.load_final_project_data_from_config(project_config,
                                                                   to_load_frames=True,
                                                                   to_load_segmentation_metadata=True,
                                                                   to_load_tracklets=True)

    step_check_functions = {1: 'print_seg_statistics',
                            2: 'print_tracklet_statistics',
                            3: 'print_tracking_statistics'}

    for i_step, func_name in step_check_functions.items():
        if i_step > last_finished_step:
            break

        getattr(project_data, func_name)()


def load_all_projects_in_folder(folder_name: str, **kwargs) -> Dict[str, ProjectData]:
    """
    Loads all projects from a folder with the given options

    Uses load_all_projects_from_list
    """
    list_of_project_folders = list(Path(folder_name).iterdir())
    all_projects = load_all_projects_from_list(list_of_project_folders, **kwargs)
    return all_projects


def load_all_projects_from_list(list_of_project_folders: List[Union[str, Path]], only_load_paths=False,
                                **kwargs) -> Dict[str, Union[ProjectData]]:
    """
    Loads all projects from a list.

    Note: can't be easily multithreaded because each project loads itself using multiple threads
    """
    all_projects_dict = {}
    if 'verbose' not in kwargs:
        kwargs['verbose'] = 0

    def check_folder_and_load(_folder):
        if Path(_folder).is_file():
            return None
        for file in Path(_folder).iterdir():
            if "project_config.yaml" in file.name and not file.name.startswith('.'):
                if not only_load_paths:
                    return ProjectData.load_final_project_data(file, **kwargs)
                else:
                    return file
        return None

    for folder in tqdm(list_of_project_folders, leave=False):
        proj = check_folder_and_load(folder)
        if proj is not None:
            if not only_load_paths:
                all_projects_dict[proj.shortened_name] = proj
            else:
                all_projects_dict[Path(proj).parent.stem] = proj

    return all_projects_dict


def plot_pca_modes_from_project(project_data: ProjectData, n_components=3, trace_kwargs=None, title="",
                                to_save=True):
    """
    Plots pca modes across time of traces. Not a phase plot.

    Interpolates nan by default, but does not remove estimated tracking errors
    """
    if trace_kwargs is None:
        trace_kwargs = {}

    pca_modes, var_explained = project_data.calc_pca_modes(n_components=n_components, **trace_kwargs)

    # Use physical time axis
    x = project_data.x_for_plots

    plt.figure(dpi=100, figsize=(15, 3))

    offsets = 1.5*np.arange(n_components)
    plt.plot(x, pca_modes / pca_modes.max() - offsets, label=[f"mode {i+1}" for i in range(n_components)])
    plt.legend(loc='lower right')
    project_data.shade_axis_using_behavior()
    plt.yticks([])
    plt.xlim(project_data.x_lim)

    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Normalized activity (ratio)")

    if to_save:
        vis_cfg = project_data.project_config.get_visualization_config(make_subfolder=True)
        fname = 'pca_modes.png'
        fname = vis_cfg.resolve_relative_path(fname, prepend_subfolder=True)
        plt.savefig(fname)


def plot_pca_projection_3d_from_project(project_data: ProjectData, trace_kwargs=None, t_start=None, t_end=None,
                                        include_time_series_subplot=True, fig=None, fig_opt=None,
                                        states_to_remove=None, include_slowing=False, verbose=0):
    """
    Similar to plot_pca_modes_from_project, but 3d. Plots times series and 3d axis
    
    Parameters
    ----------
    project_data
    trace_kwargs
    t_start
    t_end
    include_time_series_subplot
    fig
    verbose

    Returns
    -------

    """
    if fig_opt is None:
        fig_opt = dict(figsize=(15, 15), dpi=200)
    if trace_kwargs is None:
        trace_kwargs = {}
    if fig is None:
        fig = plt.figure(**fig_opt)
    if include_time_series_subplot:
        ax = fig.add_subplot(211, projection='3d')
    else:
        ax = fig.add_subplot(111, projection='3d')
    if states_to_remove is None:
        states_to_remove = [BehaviorCodes.UNKNOWN]

    pca_proj, var_explained = project_data.calc_pca_modes(**trace_kwargs, interpolate_nan=True)
    var_explained *= 100

    if t_end is not None:
        pca_proj = pca_proj[:t_end, :]
    if t_start is not None:
        pca_proj = pca_proj[t_start:, :]

    # Color the lines by behavior annotation using proper colormap
    beh_annotation = dict(fluorescence_fps=True, reset_index=True, include_collision=False, include_turns=True,
                          include_head_cast=False, include_pause=True, include_slowing=include_slowing,
                          use_pause_to_exclude_other_states=True, simplify_states=True)
    ethogram_cmap_kwargs = dict(use_plotly_style_strings=False)
    # beh_annotation.update(beh_annotation_kwargs)
    state_vec = project_data.worm_posture_class.beh_annotation(**beh_annotation)
    pca_proj['state'] = state_vec.values
    ethogram_cmap_kwargs.setdefault('include_turns', beh_annotation['include_turns'])
    ethogram_cmap_kwargs.setdefault('include_quiescence', beh_annotation['include_pause'])
    ethogram_cmap_kwargs.setdefault('include_collision', beh_annotation['include_collision'])
    ethogram_cmap = BehaviorCodes.ethogram_cmap(**ethogram_cmap_kwargs)

    # modes_to_plot = [0, 1, 2]
    # df_out, col_names = modify_dataframe_to_allow_gaps_for_plotly(pca_proj, modes_to_plot, 'state')
    state_codes = pca_proj['state'].unique()

    for i, state_code in enumerate(state_codes):
        if state_code in states_to_remove:
            continue
        # Plot the state
        state_df = pca_proj[pca_proj['state'] == state_code].copy()
        # Fill with nan so matplotlib doesn't connect the gaps
        state_df, _ = fill_missing_indices_with_nan(state_df, expected_index=pca_proj.index)
        # For every first nan value after a gap, add in the real data in order to connect states
        state_df = ffill_using_raw_data(state_df, pca_proj)
        # Transform state_code to simple string
        state_name = behavior_name_mapping()[BehaviorCodes.convert_to_simple_states(state_code).full_name]

        ax.plot(state_df[0], state_df[1], state_df[2], c=ethogram_cmap[state_code], label=state_name)

    label_func = lambda i: f'Component {i} ({var_explained[i - 1]:.0f}%)'
    ax.set_xlabel(label_func(1))
    ax.set_ylabel(label_func(2))
    ax.set_zlabel(label_func(3))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Also plot the simple time series
    if include_time_series_subplot:
        ax2 = fig.add_subplot(212)
        for i in range(3):
            ax2.plot(pca_proj[:, i] / np.max(pca_proj[:, i]) - i, label=f'mode {i+1}')
        plt.legend()
        ax2.set_title("PCA modes")

    return fig, ax, pca_proj


def calc_frequency_spectrum(project_data, neuron_name, z_score=False, **kwargs):
    """
    Uses Welch's method to calculate a frequency spectrum

    Parameters
    ----------
    project_data
    neuron_name

    Returns
    -------

    """

    df_traces = project_data.calc_default_traces(**kwargs)
    y = df_traces[neuron_name]
    if z_score:
        y = (y - y.mean()) / y.std()

    fs = project_data.physical_unit_conversion.volumes_per_second
    frequencies, Pxx_den = signal.welch(y, fs)
    return frequencies, Pxx_den


def plot_frequencies_for_fm_and_immob_projects(all_projects_wbfm, all_projects_immob, neuron_name,
                                               output_folder=None, **kwargs):
    """
    Calculates the frequency spectrum for a neuron in all projects, and plots them

    Parameters
    ----------
    all_projects_wbfm
    all_projects_immob
    neuron_name
    kwargs

    Returns
    -------

    """
    # Calculate all frequencies
    all_pxx_wbfm = {}
    all_pxx_immob = {}
    for name, proj in all_projects_wbfm.items():
        try:
            frequency_wbfm, all_pxx_wbfm[name] = calc_frequency_spectrum(proj, neuron_name, use_paper_options=True,
                                                                         **kwargs)
        except KeyError:
            pass
    for name, proj in all_projects_immob.items():
        try:
            frequency_immob, all_pxx_immob[name] = calc_frequency_spectrum(proj, neuron_name, use_paper_options=True,
                                                                           **kwargs)
        except KeyError:
            pass

    df_pxx_wbfm = pd.DataFrame(all_pxx_wbfm, index=frequency_wbfm).T
    df_pxx_immob = pd.DataFrame(all_pxx_immob, index=frequency_immob).T

    # Normalize them to have the same initial (after the trivial i=0) value
    df_pxx_wbfm = df_pxx_wbfm.div(df_pxx_wbfm.iloc[:, 1], axis=0)
    df_pxx_immob = df_pxx_immob.div(df_pxx_immob.iloc[:, 1], axis=0)

    # Plot like the triggered averages (median plus shading)
    opt = dict(ind_preceding=0)
    ax, _ = plot_triggered_average_from_matrix_low_level(df_pxx_wbfm, label='Freely moving', is_second_plot=True, **opt)
    ax, _ = plot_triggered_average_from_matrix_low_level(df_pxx_immob, ax=ax, is_second_plot=True, label='Immobilized',
                                                         **opt)
    plt.legend()
    ax.set_xlim(0.01, 0.5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized Power")
    ax.set_title(f"Frequency spectrum for {neuron_name}")

    apply_figure_settings(width_factor=0.4, height_factor=0.25, plotly_not_matplotlib=False)

    if output_folder is not None:
        plt.tight_layout()
        fname = f"frequency_spectrum_{neuron_name}.png"
        fname = Path(output_folder).joinpath(fname)
        plt.savefig(fname, transparent=True)
        fname = Path(fname).with_suffix('.svg')
        plt.savefig(fname)

    return df_pxx_wbfm, df_pxx_immob, all_pxx_wbfm, all_pxx_immob


def rename_manual_ids_in_project(project_data: ProjectData, name_mapping: Dict[str, str],
                                 always_update_cached_dataframes=True):
    """
    Renames a set of manual ids in project_data using the dictionary name_mapping

    Parameters
    ----------
    project_data
    name_mapping

    Returns
    -------

    """

    # Make sure project is loaded
    project_data = ProjectData.load_final_project_data(project_data)

    # Load the previous mappings
    previous2raw = project_data.neuron_name_to_manual_id_mapping(confidence_threshold=0, flip_names_and_ids=True,
                                                                 remove_unnamed_neurons=True)
    previous2new = {}

    # Check if any updates are applicable
    for previous_manual_id, new_manual_id in name_mapping.items():
        if previous_manual_id in previous2raw.keys():
            previous2new[previous_manual_id] = new_manual_id
        else:
            project_data.logger.debug(f"Did not find previous id {previous_manual_id} in previous2raw")

    # If any updates, then also need to update the cached dataframes with traces
    if len(previous2new) > 0:
        # Update the manual annotation file
        df = project_data.df_manual_tracking
        if df is None:
            raise NoNeuronsError("No manual annotation file found")
        for col in ['ID1', 'ID2']:
            df[col] = df[col].replace(previous2new)
        manual_annotation_fname = project_data.df_manual_tracking_fname
        print(f"Saving manual annotation file to {manual_annotation_fname} with size:")
        df.info(memory_usage='deep')
        df.to_excel(manual_annotation_fname, index=False)

    if len(previous2new) > 0 or always_update_cached_dataframes:
        # Make sure the project gets the new manual ids from disk
        project_data = ProjectData.load_final_project_data(project_data, force_reload=True)
        rename_manual_ids_from_excel_in_project(project_data)


def rename_manual_ids_from_excel_in_project(project_data: ProjectData, dryrun=False):
    """
    Similar to rename_manual_ids_in_project, but instead of a given update dictionary, assumes that the excel file is
    correct. Then it updates all the cached dataframes with the new names.

    Note that in order to fix duplicate ids, it

    Parameters
    ----------
    project_data

    Returns
    -------

    """

    # Make sure project is loaded
    project_data = ProjectData.load_final_project_data(project_data)

    # Instead of using the dictionary (which has updates), use the current (outdated) columns of the trace dataframe
    # Then convert back to original names, then finally to the new updated names
    opt = paper_trace_settings()
    df_old_manual_names = project_data.calc_default_traces(**opt)
    opt['rename_neurons_using_manual_ids'] = False
    df_raw_names = project_data.calc_default_traces(**opt)

    raw2new = project_data.neuron_name_to_manual_id_mapping(confidence_threshold=0)

    # Combine the mappings
    raw2previous = {k: v for k, v in zip(df_raw_names.columns, df_old_manual_names.columns)}
    previous2new = {}
    for raw_id in raw2new.keys():
        previous_id = raw2previous.get(raw_id, raw_id)  # This has some dropped names, which should be the same
        new_id = raw2new[raw_id]
        previous2new[previous_id] = new_id

    # Update cached traces
    if not dryrun:
        project_data.data_cacher.rename_columns_in_existing_cached_dataframes(previous2new)
    else:
        print(f"Modifications to be made: {[(k, v) for k, v in previous2new.items() if k!=v]}")
