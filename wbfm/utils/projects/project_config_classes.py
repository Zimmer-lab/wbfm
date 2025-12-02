import glob
import json
import logging
import os
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import pprint
from imutils import MicroscopeDataReader

from wbfm.utils.external.utils_pandas import ensure_dense_dataframe
from wbfm.utils.external.custom_errors import NoBehaviorDataError, IncompleteConfigFileError
from wbfm.utils.general.utils_logging import setup_logger_object, setup_root_logger
from wbfm.utils.general.utils_filenames import check_exists, resolve_mounted_path_in_current_os, \
    get_sequential_filename, get_location_of_new_project_defaults, is_absolute_in_any_os, \
    get_location_of_alternative_project_defaults
from wbfm.utils.projects.utils_project import safe_cd, update_project_config_path, \
    update_snakemake_config_path, RawFluorescenceData
from wbfm.utils.external.utils_yaml import edit_config, load_config
from wbfm.utils.general.hardcoded_paths import default_raw_data_config
from wbfm.utils.external.custom_errors import RawDataFormatError


@dataclass
class ConfigFileWithProjectContext:
    """
    Top-level configuration file class

    Knows how to:
    1. update itself on disk
    2. save new data inside the relevant project using pickle or pandas
    3. change filepaths between relative and absolute
    """

    _self_path: str
    _config: dict = None
    project_dir: str = None

    _logger: logging.Logger = None
    log_to_file: bool = False

    def __post_init__(self):
        if self._self_path is None:
            logging.debug("self_path is None; some functionality will not work")
            self._config = dict()
        else:
            if Path(self._self_path).is_dir():
                # Then it was a folder, and we should find the config file inside
                self.project_dir = self._self_path
                self._self_path = str(Path(self._self_path).joinpath('project_config.yaml'))
            else:
                self.project_dir = str(Path(self._self_path).parent)
            self._config = load_config(self._self_path)
            if self._config is None:
                if not Path(self._self_path).exists():
                    raise FileNotFoundError(f"Could not find config file {self._self_path}")
                else:
                    raise ValueError(f"Found empty file at {self._self_path}; probably yaml crashed and deleted the file. "
                                     f"There is no way to recover the data, so the file must be recreated manually.")
            # Convert to default dict, for backwards compatibility with deprecated keys
            # Actually: this gives problems with pickling, so do not do this
            # self.config = defaultdict(lambda: defaultdict(lambda: None), self.config)

    @property
    def config(self):
        if self.has_valid_self_path:
            return self._config
        else:
            raise IncompleteConfigFileError("No valid self_path was found")

    @property
    def has_valid_self_path(self):
        return self._self_path is not None

    @property
    def logger(self):
        if self._logger is None:
            self.setup_logger('ConfigFile.log')
        return self._logger

    def setup_logger(self, relative_log_filename: str):
        if self.has_valid_self_path:
            log_filename = self.resolve_relative_path(os.path.join('log', relative_log_filename))
            self._logger = setup_logger_object(log_filename, self.log_to_file)
        else:
            self._logger = logging.getLogger('project_config')
        return self._logger

    def setup_global_logger(self, relative_log_filename: str):
        log_filename = self.resolve_relative_path(os.path.join('log', relative_log_filename))
        logger = setup_root_logger(log_filename)
        return logger

    def update_self_on_disk(self):
        fname = self.absolute_self_path
        self.logger.info(f"Updating config file {fname} on disk")
        # Make sure none of the values are Path objects, which will crash the yaml dump and leave an empty file!
        for key in self.config:
            if isinstance(self.config[key], Path):
                self.config[key] = str(self.config[key])
                self.logger.warning(f"Found Path object in config file (probably a bug), converting to string: {key}")
        self.logger.debug(f"Updated values: {self.config}")
        try:
            edit_config(fname, self.config)
        except PermissionError as e:
            if Path(self._self_path).is_absolute():
                self.logger.debug(f"Skipped updating nonlocal file: {fname}")
            else:
                # Then it was a local file, and the error was real
                raise e

    def resolve_relative_path_from_config(self, key) -> str:
        val = self.config.get(key, None)
        return self.resolve_relative_path(val)

    def resolve_relative_path(self, val: str) -> Optional[str]:
        if val is None:
            return val
        if is_absolute_in_any_os(val):
            return resolve_mounted_path_in_current_os(val)
        relative_path = Path(self.project_dir).joinpath(val)
        # Replace any windows slashes with unix slashes
        relative_path = str(relative_path.resolve()).replace('\\', '/')
        return relative_path

    def unresolve_absolute_path(self, val: str, raise_if_not_relative=False) -> Optional[str]:
        if val is None:
            return val
        # NOTE: is_relative_to() only works for python >= 3.9
        # if Path(val).is_relative_to(self.project_dir):
        project_dir = self.project_dir
        try:
            return str(Path(val).relative_to(project_dir))
        except ValueError:
            try:
                # As of October 2023, the cluster has /lisc/data/scratch and/lisc/data/scratch mapping to the same point
                # Both should be removed if possible from the path; this will always check the /lisc/data/scratch version
                project_dir = Path(project_dir).resolve()
                return str(Path(val).relative_to(project_dir))
            except ValueError:
                if raise_if_not_relative:
                    raise ValueError(f"Could not make path {val} relative to {project_dir}")
                return val

    @property
    def absolute_self_path(self):
        return self.resolve_relative_path(self._self_path)

    @property
    def relative_self_path(self):
        return self.unresolve_absolute_path(self._self_path)

    def to_json(self):
        return json.dumps(vars(self))

    def pickle_data_in_local_project(self, data, relative_path: str,
                                     allow_overwrite=True, make_sequential_filename=False,
                                     custom_writer=None,
                                     **kwargs):
        """
        For objects larger than 4GB and python<3.8, protocol=4 must be specified directly

        https://stackoverflow.com/questions/29704139/pickle-in-python3-doesnt-work-for-large-data-saving
        """
        abs_path = self.resolve_relative_path(relative_path)
        Path(abs_path).parent.mkdir(parents=True, exist_ok=True)
        if not abs_path.endswith('.pickle'):
            abs_path += ".pickle"
        if make_sequential_filename:
            abs_path = get_sequential_filename(abs_path)
        self.logger.info(f"Saving at: {self.unresolve_absolute_path(abs_path)}")
        check_exists(abs_path, allow_overwrite)
        if custom_writer:
            # Useful for pickling dataframes
            custom_writer(data, abs_path, **kwargs)
        else:
            with open(abs_path, 'wb') as f:
                pickle.dump(data, f, **kwargs)

        return abs_path

    def save_data_in_local_project(self, data: pd.DataFrame, relative_path: str,
                                   suffix='.h5',
                                   allow_overwrite=True, make_sequential_filename=False, also_save_csv=False,
                                   **kwargs):
        abs_path = self.resolve_relative_path(relative_path)
        Path(abs_path).parent.mkdir(parents=True, exist_ok=True)
        if not abs_path.endswith(suffix):
            abs_path += suffix
        if make_sequential_filename:
            abs_path = get_sequential_filename(abs_path)
        self.logger.info(f"Saving at local path: {self.unresolve_absolute_path(abs_path)}")
        check_exists(abs_path, allow_overwrite)
        if suffix == '.h5':
            ensure_dense_dataframe(data).to_hdf(abs_path, key="df_with_missing")
        elif suffix == '.csv':
            data.to_csv(abs_path)
        elif suffix == '.xlsx':
            data.to_excel(abs_path, **kwargs)
        else:
            raise ValueError(f"Unknown suffix: {suffix}")

        if also_save_csv:
            csv_fname = Path(abs_path).with_suffix('.csv')
            data.to_csv(csv_fname)

        return abs_path

    def __repr__(self):
        pp = pprint.PrettyPrinter(indent=2)
        return pp.pformat(self.config)


@dataclass
class SubfolderConfigFile(ConfigFileWithProjectContext):
    """
    Configuration file (loaded from .yaml) that knows the project it should be executed in

    In principle this config file is associated with a subfolder and single step of a project

    Light wrapper around ConfigFileWithProjectContext, with the added functionality of:
    1. Saving in the correct subfolder
    2. Resolving paths to the correct subfolder

    Note that this subfolder does not need to be nested within the main project folder
    """

    subfolder: str = None

    @property
    def absolute_subfolder(self):
        return self.resolve_relative_path(self.subfolder)

    def __post_init__(self):
        pass

    def resolve_relative_path(self, raw_path: Optional[str], prepend_subfolder=False) -> Optional[str]:
        if raw_path is None:
            return None
        if is_absolute_in_any_os(raw_path):
            return resolve_mounted_path_in_current_os(raw_path)

        final_path = self._prepend_subfolder(raw_path, prepend_subfolder)
        # Replace any windows slashes with unix slashes
        final_path = str(Path(final_path).resolve()).replace('\\', '/')
        return final_path

    def _prepend_subfolder(self, val, prepend_subfolder):
        if prepend_subfolder:
            final_path = os.path.join(self.project_dir, self.subfolder, val)
        else:
            final_path = os.path.join(self.project_dir, val)
        return final_path

    def save_data_in_local_project(self, data: pd.DataFrame, relative_path: str, prepend_subfolder=False,
                                   **kwargs):
        path = self._prepend_subfolder(relative_path, prepend_subfolder)
        abs_path = super().save_data_in_local_project(data, path, **kwargs)
        return abs_path


@dataclass
class ModularProjectConfig(ConfigFileWithProjectContext):
    """
    Add functionality to get individual config files using the main project config filepath

    Returns config_file_with_project_context objects, instead of raw dictionaries for the subconfig files

    Knows how to:
    1. find the individual config files of the substeps
    2. initialize the physical unit conversion class
    3. loading other options classes
    4. find the files used for the behavior pipeline
    5. open the raw data (external to the project)

    In principle, any usage of non-local files (raw data) should go through this class

    """

    _preprocessing_class: RawFluorescenceData = None

    def get_project_config_of_remote_project(self, step=None) -> Optional[str]:
        # Checks if any of the subfolder configs are absolute paths, and if so, returns the corresponding project path
        remote_project_dir = None
        for fname, config_path in self.config['subfolder_configs'].items():
            if step is not None and fname != step:
                continue
            if os.path.isabs(config_path):
                remote_project_dir = os.path.dirname(os.path.dirname(config_path))
                break
        return remote_project_dir

    def get_segmentation_config(self) -> SubfolderConfigFile:
        fname = Path(self.config['subfolder_configs']['segmentation'])
        return SubfolderConfigFile(**self._check_path_and_load_config(fname))

    def get_training_config(self) -> SubfolderConfigFile:
        fname = Path(self.config['subfolder_configs']['training_data'])
        return SubfolderConfigFile(**self._check_path_and_load_config(fname))

    def get_tracking_config(self) -> SubfolderConfigFile:
        fname = Path(self.config['subfolder_configs']['tracking'])
        return SubfolderConfigFile(**self._check_path_and_load_config(fname))
    
    def get_snakemake_config(self) -> SubfolderConfigFile:
        """Uniquely, this config file must be local; it is hardcoded here instead of read from subfolder_configs"""
        fname = Path(os.path.join('snakemake', 'snakemake_config.yaml'))
        return SubfolderConfigFile(**self._check_path_and_load_config(fname))

    def get_preprocessing_config(self) -> SubfolderConfigFile:
        """
        Not often used, except for updating the file.

        Note: NOT a subfolder
        """
        fname = Path(self.get_preprocessing_config_filename())
        return SubfolderConfigFile(**self._check_path_and_load_config(fname))

    def get_preprocessing_class(self, do_background_subtraction=None):
        # Note: this can't be pickled!
        # https://github.com/cloudpipe/cloudpickle/issues/178

        if self._preprocessing_class is not None:
            return self._preprocessing_class

        fname = self.get_preprocessing_config_filename()
        from wbfm.utils.general.preprocessing.utils_preprocessing import PreprocessingSettings
        preprocessing_settings = PreprocessingSettings.load_from_yaml(fname, do_background_subtraction)
        preprocessing_settings.cfg_preprocessing = self.get_preprocessing_config()
        preprocessing_settings.cfg_project = self
        if not preprocessing_settings.background_is_ready:
            try:
                preprocessing_settings.find_background_files_from_raw_data_path()
            except FileNotFoundError:
                self.logger.warning("Did not find background; turning off background subtraction")
                preprocessing_settings.do_background_subtraction = False

        self._preprocessing_class = preprocessing_settings

        return preprocessing_settings

    def get_preprocessing_config_filename(self):
        # In newer versions, it is in the dat folder and has an entry in the main config file
        fname = self.config['subfolder_configs'].get('preprocessing', None)
        fname = self.resolve_relative_path(fname)
        if fname is None or not Path(fname).exists():
            # In older versions, it was in the main folder
            fname = str(Path(self.project_dir).joinpath('preprocessing_config.yaml'))
        return fname

    def get_behavior_config(self) -> SubfolderConfigFile:
        if not self.has_valid_self_path:
            raise FileNotFoundError
        fname = Path(self.project_dir).joinpath('behavior', 'behavior_config.yaml')
        if not fname.exists():
            raise FileNotFoundError
        return SubfolderConfigFile(**self._check_path_and_load_config(fname))

    def get_traces_config(self) -> SubfolderConfigFile:
        fname = Path(self.config['subfolder_configs']['traces'])
        return SubfolderConfigFile(**self._check_path_and_load_config(fname))

    def get_raw_data_config(self) -> SubfolderConfigFile:
        """
        This is a different kind of config file, which is present in the raw data folder, and not in the local project

        Returns
        -------

        """
        fname = None
        try:
            fname = self.get_folder_for_all_channels()
            if fname is None:
                raise FileNotFoundError
            fname = Path(fname).joinpath('config.yaml')
            return SubfolderConfigFile(**self._check_path_and_load_config(fname))
        except FileNotFoundError:
            # Allow a hardcoded default... fragile, but necessary for projects with deleted raw data
            cfg = default_raw_data_config()
            self._logger.debug(f"Could not find file {fname}; "
                               f"Using hardcoded default raw data config: {cfg}")
            return SubfolderConfigFile(_self_path=None, _config=cfg, project_dir=self.project_dir)

    def get_nwb_config(self, make_subfolder=True) -> SubfolderConfigFile:
        fname = self.config['subfolder_configs'].get('nwb', None)
        if fname is None:
            fname = Path(self.project_dir).joinpath('nwb', 'nwb_config.yaml')  # Default (local) path; old projects don't have this subfolder_config entry
            if not fname.exists():
                if make_subfolder:
                    self.initialize_nwb_folder()
                else:
                    raise FileNotFoundError("No path to a nwb config file was found in the project_config.yaml file")
        else:
            fname = Path(fname)
        return SubfolderConfigFile(**self._check_path_and_load_config(fname))

    def _get_neuropal_dir(self, make_subfolder=True, raise_error=False):
        # Directory which is not part of a default project
        foldername = Path(self.project_dir).joinpath('neuropal')
        if make_subfolder:
            try:
                foldername.mkdir(exist_ok=True)
            except PermissionError as e:
                self.logger.warning(f"Could not create neuropal folder (continuing): {e}")
                if raise_error:
                    raise e
        return str(foldername)

    def get_neuropal_config(self) -> SubfolderConfigFile:
        fname = self.config['subfolder_configs'].get('neuropal', None)
        if fname is None:
            fname = Path(self.project_dir).joinpath(self._get_neuropal_dir(), 'neuropal_config.yaml')
            if not fname.exists():
                raise FileNotFoundError("No path to a neuropal config file was found in the project_config.yaml file")
        else:
            fname = Path(fname)
        return SubfolderConfigFile(**self._check_path_and_load_config(fname))

    def initialize_neuropal_subproject(self) -> SubfolderConfigFile:
        # Nearly the same as getting a subfolder config, but expects the folder to not exist
        foldername = self._get_neuropal_dir(make_subfolder=True, raise_error=True)
        # Copy contents of the neuropal folder from the github project to the local project
        source_folder = Path(get_location_of_alternative_project_defaults()).joinpath('neuropal_subproject')
        for content in source_folder.iterdir():
            if content.is_file():
                shutil.copy(content, foldername)
            else:
                raise FileNotFoundError(f"Found a folder in the default neuropal folder: {content}")
        # Add this config path to the main project config
        neuropal_config = self.get_neuropal_config()
        self.config['subfolder_configs']['neuropal'] = neuropal_config.relative_self_path
        self.update_self_on_disk()
        return neuropal_config

    def initialize_nwb_folder(self) -> SubfolderConfigFile:
        # Nearly the same as getting a subfolder config, but expects the folder to not exist
        foldername = Path(self.project_dir).joinpath('nwb')
        foldername.mkdir(exist_ok=True)
        # Copy contents of the config file from the github project to the local project
        source_folder = Path(get_location_of_new_project_defaults()).joinpath('nwb')
        for content in source_folder.iterdir():
            if content.is_file():
                shutil.copy(content, foldername)
            else:
                raise FileNotFoundError(f"Found a folder in the default nwb folder: {content}")
        # Add this config path to the main project config
        nwb_config = self.get_nwb_config()
        self.config['subfolder_configs']['nwb'] = nwb_config.relative_self_path
        self.update_self_on_disk()
        return nwb_config

    def _check_path_and_load_config(self, subconfig_path: Path,
                                    allow_config_to_not_exist: bool = False) -> Dict:
        if is_absolute_in_any_os(str(subconfig_path)):
            project_dir = Path(resolve_mounted_path_in_current_os(str(subconfig_path.parent.parent)))
        else:
            project_dir = Path(self.absolute_self_path).parent

        if is_absolute_in_any_os(subconfig_path):
            subconfig_path = Path(resolve_mounted_path_in_current_os(str(subconfig_path)))

        with safe_cd(project_dir):
            try:
                cfg = load_config(subconfig_path)
                if not isinstance(cfg, dict):
                    # If files just have a string comment, they won't be loaded as a dict
                    # In that case, the file is empty and we should just return an empty dict
                    cfg = dict()
            except FileNotFoundError as e:
                if allow_config_to_not_exist:
                    cfg = dict()
                else:
                    raise e
            except OSError as e:
                logging.error(f"Could not load config file {subconfig_path} in project {project_dir}: {e}")
                raise e
        subfolder = subconfig_path.parent

        args = dict(_self_path=str(subconfig_path),
                    _config=cfg,
                    project_dir=str(project_dir),
                    _logger=self.logger,
                    subfolder=str(subfolder))
        return args

    def get_log_dir(self) -> str:
        foldername = Path(self.project_dir).joinpath('log')
        foldername.mkdir(exist_ok=True)
        return str(foldername)

    def _get_visualization_dir(self) -> str:
        # Directory which is not part of a default project
        foldername = Path(self.project_dir).joinpath('visualization')
        try:
            foldername.mkdir(exist_ok=True)
        except PermissionError as e:
            self.logger.warning(f"Could not create visualization folder (continuing): {e}")
        return str(foldername)

    def get_visualization_config(self, make_subfolder=False) -> SubfolderConfigFile:
        fname = self.config['subfolder_configs'].get('visualization', None)
        if fname is None:
            # Assume the local folder is correct
            fname = os.path.join(self._get_visualization_dir(), 'visualization_config.yaml')
        fname = Path(fname)
        cfg = SubfolderConfigFile(**self._check_path_and_load_config(fname, allow_config_to_not_exist=True))
        if make_subfolder:
            try:
                Path(cfg.subfolder).mkdir(exist_ok=True)
            except PermissionError as e:
                self.logger.warning(f"Could not create visualization folder (continuing): {e}")
        return cfg

    def resolve_mounted_path_in_current_os(self, key) -> Optional[Path]:
        path = self.config.get(key, None)
        if path is None:
            return None
        return Path(resolve_mounted_path_in_current_os(path))

    def get_folder_for_entire_day(self) -> Optional[Path]:
        fname = self.get_folder_for_all_channels()
        if fname is None:
            return None
        else:
            return Path(fname).parent

    def get_folder_for_all_channels(self, verbose=0) -> Optional[Path]:
        red_fname, is_btf = self.get_raw_data_fname(red_not_green=True)
        if red_fname is None:
            if verbose >= 1:
                print("Could not find red_bigtiff_fname, aborting")
            return None
        if is_btf:
            folder_for_all_channels = Path(red_fname).parents[1]
        else:
            folder_for_all_channels = Path(red_fname).parents[0]

        if not folder_for_all_channels.exists():
            if verbose >= 1:
                print(f"Could not find main data folder {folder_for_all_channels}, aborting")
            return None
        return folder_for_all_channels

    def get_folder_with_background(self) -> Path:
        folder_for_entire_day = self.get_folder_for_entire_day()
        if folder_for_entire_day is not None:
            folder_for_background = folder_for_entire_day.joinpath('background')
        else:
            raise FileNotFoundError("Could not find behavior folder for entire day")
        if not folder_for_background.exists():
            raise FileNotFoundError(f"Could not find background folder {folder_for_background}")

        return folder_for_background

    def get_folder_with_calibration(self):
        folder_for_entire_day = self.get_folder_for_entire_day()
        folder_for_calibration = folder_for_entire_day.joinpath('calibration')
        if not folder_for_calibration.exists():
            raise FileNotFoundError(f"Could not find calibration folder {folder_for_calibration}")

        return folder_for_calibration

    def get_folder_with_alignment(self):
        folder_for_entire_day = self.get_folder_for_entire_day()
        if folder_for_entire_day is not None:
            folder_for_alignment = folder_for_entire_day.joinpath('alignment')
        else:
            raise FileNotFoundError("Could not find behavior folder for entire day")
        if not folder_for_alignment.exists():
            raise FileNotFoundError(f"Could not find alignment folder {folder_for_alignment}")

        return folder_for_alignment

    def get_red_and_green_dot_alignment_bigtiffs(self) -> Tuple[Optional[str], Optional[str]]:
        folder_for_alignment = self.get_folder_with_alignment()

        red_btf_fname, green_btf_fname = None, None
        for subfolder in folder_for_alignment.iterdir():
            if subfolder.is_dir():
                if subfolder.name.endswith('alignment_Ch0'):
                    red_btf_fname = self._extract_btf_from_folder(subfolder)
                elif subfolder.name.endswith('alignment_Ch1'):
                    green_btf_fname = self._extract_btf_from_folder(subfolder)

        return red_btf_fname, green_btf_fname

    def get_raw_data_fname(self, red_not_green) -> Tuple[Optional[str], bool]:
        is_btf = True
        if red_not_green:
            fname = self.resolve_mounted_path_in_current_os('red_bigtiff_fname')
            if fname is None:
                fname = self.resolve_mounted_path_in_current_os('red_fname')
                is_btf = False
        else:
            fname = self.resolve_mounted_path_in_current_os('green_bigtiff_fname')
            if fname is None:
                fname = self.resolve_mounted_path_in_current_os('green_fname')
                is_btf = False
        return fname, is_btf

    @property
    def start_volume(self):
        """Only for backwards compatibility; function is now in PreprocessingSettings"""
        return self.get_preprocessing_class().start_volume

    @property
    def num_slices(self):
        """Only for backwards compatibility; function is now in PreprocessingSettings"""
        return self.get_preprocessing_class().num_slices

    def get_num_slices_robust(self):
        """Only for backwards compatibility; function is now in PreprocessingSettings"""
        return self.get_preprocessing_class().get_num_slices_robust()

    @property
    def num_frames(self):
        """Only for backwards compatibility; function is now in PreprocessingSettings"""
        return self.get_preprocessing_class().num_frames

    def get_num_frames_robust(self):
        """Only for backwards compatibility; function is now in PreprocessingSettings"""
        return self.get_preprocessing_class().get_num_frames_robust()

    def get_red_and_green_grid_alignment_bigtiffs(self) -> Tuple[List[str], List[str]]:
        """
        Find bigtiffs for the grid pattern, for alignment. Expects 5 files, all with the pattern:
        {date}_alignment-3D-{location}_{channel}
        Example:/lisc/data/scratch/neurobiology/zimmer/ulises/wbfm/20220913/2022-09-13_11-55_alignment-3D-TopLeft_Ch0

        The locations are:
        center, TopLeft, TopRight, BottomRight, BottomLeft

        Returns a list of filenames, in this order

        """
        # Note: this will probably change in the future
        folder_for_entire_day = self.get_folder_for_entire_day()
        folder_for_alignment = folder_for_entire_day.parent

        red_btf_fname, green_btf_fname = {}, {}
        prefix_list = ['center', 'TopLeft', 'TopRight', 'BottomRight', 'BottomLeft']

        for subfolder in folder_for_alignment.iterdir():
            if subfolder.is_dir():
                if subfolder.name.endswith('_Ch0'):
                    this_fname = self._extract_btf_from_folder(subfolder, allow_ome_tif=True)
                    try:
                        this_key = prefix_list[np.where([p in subfolder.name for p in prefix_list])[0][0]]
                        red_btf_fname[this_key] = this_fname
                    except IndexError:
                        # Not one of the ones we care about
                        pass
                elif subfolder.name.endswith('_Ch1'):
                    this_fname = self._extract_btf_from_folder(subfolder, allow_ome_tif=True)
                    try:
                        this_key = prefix_list[np.where([p in subfolder.name for p in prefix_list])[0][0]]
                        green_btf_fname[this_key] = this_fname
                    except IndexError:
                        # Not one of the ones we care about
                        pass

        red_btf_fname = [red_btf_fname[k] for k in prefix_list if red_btf_fname[k] is not None]
        green_btf_fname = [green_btf_fname[k] for k in prefix_list if green_btf_fname[k] is not None]
        if len(red_btf_fname) < len(prefix_list):
            logging.warning(f"Expected 5 alignment files, but only found {len(red_btf_fname)} : {red_btf_fname}")
        return red_btf_fname, green_btf_fname

    @staticmethod
    def _extract_btf_from_folder(subfolder, allow_ome_tif=False):
        btf_fname = None
        for file in subfolder.iterdir():
            if file.name.endswith('btf'):
                btf_fname = str(file)
            elif allow_ome_tif and file.name.endswith('ome.tif'):
                btf_fname = str(file)
        return btf_fname

    def get_behavior_raw_file_from_red_fname(self):
        """If the user did not set the behavior foldername, try to infer it from the red"""
        behavior_subfolder, flag = self.get_behavior_raw_parent_folder_from_red_fname()
        if not flag:
            return None
        # Second, get the file itself
        for content in behavior_subfolder.iterdir():
            if content.is_file():
                # UK spelling, and there may be preprocessed bigtiffs in the folder
                if str(content).endswith('-BH_bigtiff.btf'):
                    behavior_fname = behavior_subfolder.joinpath(content)
                    break
        else:
            print(f"Found no behavior file in {behavior_subfolder}, aborting")
            return None

        return behavior_fname, behavior_subfolder

    def get_behavior_raw_parent_folder_from_red_fname(self, verbose=0) -> Tuple[Optional[Path], bool]:
        """
        This searches for the behavior folder in the parent folder of the red file
        The assumed structure is:
        - main_data_folder
            - behavior_subfolder
                - video_files
            - red_subfolder
                - ...
            - green_subfolder
                - ...

        However, there are two options for the red_fname from get_raw_data_fname:
        1. ndtiff, meaning red_fname = red_subfolder
        2. btf, meaning red_fname = inside red_subfolder
        Thus, the function must check for both cases, taking the parent folder in the first case and the grandparent in
        the second, and then searching for the behavior subfolder

        Parameters
        ----------
        verbose

        Returns
        -------

        """
        folder_for_all_channels = self.get_folder_for_all_channels(verbose=verbose)
        if folder_for_all_channels is None:
            return None, False
        # First, get the subfolder
        for content in folder_for_all_channels.iterdir():
            if content.is_dir():
                # Ulises uses UK spelling
                if content.name.endswith('behaviour') or content.name.endswith('BH'):
                    behavior_subfolder = folder_for_all_channels.joinpath(content)
                    flag = True
                    break
        else:
            if verbose >= 1:
                print(f"Found no behavior subfolder in {folder_for_all_channels}, aborting")
            flag = False
            behavior_subfolder = None
        if verbose >= 1:
            print(f"Found behavior subfolder: {behavior_subfolder}")
        return behavior_subfolder, flag

    def get_folders_for_behavior_pipeline(self, crash_if_no_background=True):
        """
        Requires the raw behavior folder and the behavior folder in the project

        Raises a NoBehaviorDataError if the entire behavior folder cannot be found, which is expected for immobilized
        recordings.
        Raises a RawDataFormatError if the raw data is not in the expected format

        Note that background_video is a file if it is a .btf, and a folder if it is ndtiff

        Returns
        -------

        """
        # General folders
        behavior_raw_folder, flag = self.get_behavior_raw_parent_folder_from_red_fname()
        if not flag:
            raise NoBehaviorDataError()
        behavior_parent_folder = str(behavior_raw_folder.parent)
        multiday_parent_folder = str(Path(behavior_parent_folder).parent)
        behavior_raw_folder = str(behavior_raw_folder)

        # Second
        try:
            beh_cfg = self.get_behavior_config()
        except FileNotFoundError:
            raise NoBehaviorDataError(f"Could not find behavior_config.yaml in {self.project_dir}/behavior... "
                                      f"Did the user delete it?")
        behavior_output_folder = beh_cfg.subfolder

        if not Path(behavior_output_folder).exists():
            update_path_to_behavior_in_config(self)

        # See if the .btf file has already been produced... unfortunately this is a modification of the raw data folder
        # Assume that any .btf file is the correct one, IF it doesn't have 'AVG' in the name (refers to background subtraction)
        btf_file = [f for f in os.listdir(behavior_raw_folder) if f.endswith(".btf") and 'AVG' not in f and
                    os.path.isfile(f)]
        if len(btf_file) == 1:
            btf_file = btf_file[0]
            self.logger.info(f".btf file already produced: {btf_file}")
            btf_file = os.path.join(behavior_raw_folder, btf_file)
        elif len(btf_file) > 1:
            raise RawDataFormatError(f"There is more than one .btf file in {behavior_raw_folder}")
        else:
            # Then it will need to be produced
            btf_file = os.path.join(behavior_raw_folder, 'raw_stack.btf')
            self.logger.warning(f"No .btf file found, will produce it in the raw data folder: {btf_file}, "
                                f"UNLESS you have updated to NDTIFF. In that case, don't worry")

        # Look for background image
        background_parent_folder = self._get_background_parent_folder(multiday_parent_folder,
                                                                      crash_if_no_background=crash_if_no_background)
        # First, try to just load with the ndtiff reader
        if background_parent_folder is not None:
            try:
                # Same reading style as stack_z_projection
                _ = MicroscopeDataReader(background_parent_folder, as_raw_tiff=True, raw_tiff_num_slices=1)
                background_video = background_parent_folder
            except TypeError:
                try:
                    _ = MicroscopeDataReader(background_parent_folder, as_raw_tiff=False)
                    background_video = background_parent_folder
                except (TypeError, FileNotFoundError):
                    logging.info(f"Tried to read background using MicroscopeDataReader, but failed: "
                                 f"{background_parent_folder}... falling back to glob")
                    background_video = self._find_individual_background_files(background_parent_folder,
                                                                              crash_if_no_background)

            # Name of the background image is the same, but with 'AVG' prepended
            if background_video is None:
                background_img = None
            else:
                background_img = os.path.join(behavior_output_folder, 'AVG' + os.path.basename(background_video))
                background_img = str(Path(background_img).with_suffix('.tif'))
        else:
            background_video, background_img = None, None

        return behavior_parent_folder, behavior_raw_folder, behavior_output_folder, \
            background_img, background_video, btf_file

    def _get_background_parent_folder(self, multiday_parent_folder, suffix='BH', crash_if_no_background=True) -> \
            Optional[str]:
        # First, get intermediate folder: anything with 'background' (only folders)
        background_parent_folder = [f for f in os.listdir(multiday_parent_folder) if 'background' in f and
                                    os.path.isdir(os.path.join(multiday_parent_folder,f))]
        found_one_folder = True
        msg = ''
        if len(background_parent_folder) > 1:
            found_one_folder = False
            msg = f"Found more than one background folders in the parent folder: {background_parent_folder}"
        elif len(background_parent_folder) == 0:
            found_one_folder = False
            msg = f"Found no background folder in the parent folder: {background_parent_folder}"
        if not found_one_folder:
            self.logger.error(msg)
            if crash_if_no_background:
                raise RawDataFormatError(msg)
            else:
                return None
        # Second, actual background folder (only folders)
        background_parent_folder = os.path.join(multiday_parent_folder, background_parent_folder[0])
        specific_background_parent_folder = [f for f in os.listdir(background_parent_folder) if 'background' in f and
                                             suffix in f and os.path.isdir(os.path.join(background_parent_folder,f))]
        if len(specific_background_parent_folder) != 1:
            msg = (f"Found no or more than one specific background folder(s) for channel {suffix}: "
                   f"{specific_background_parent_folder}")
            if crash_if_no_background:
                raise RawDataFormatError(msg)
            else:
                self.logger.warning(msg)
                return None
        else:
            specific_background_parent_folder = os.path.join(background_parent_folder,
                                                             specific_background_parent_folder[0])
        return specific_background_parent_folder

    def _find_individual_background_files(self, background_parent_folder, crash_if_no_background):
        # Otherwise, try to find specific files
        background_video = glob.glob(f"{background_parent_folder}/*background*")
        # Remove any with AVG in the name
        background_video = [f for f in background_video if 'AVG' not in f]
        background_video = [f for f in background_video if 'metadata' not in f]
        if len(background_video) == 1:
            background_video = background_video[0]
            background_video = str(Path(background_video).resolve())  # This is needed because the path is relative
            print("This is the background video: ", background_video)

        elif len(background_video) > 1:
            msg = f"There is more than one background video: {background_video}"
            if crash_if_no_background:
                raise RawDataFormatError(msg)
            else:
                self.logger.warning(msg)
                background_video = None
        else:
            msg = f"No background videos found in {background_parent_folder}/"
            if crash_if_no_background:
                raise RawDataFormatError(msg)
            else:
                self.logger.warning(msg)
                background_video = None
        return background_video


def update_path_to_segmentation_in_config(cfg: ModularProjectConfig) -> SubfolderConfigFile:
    # For now, does NOT overwrite anything on disk

    segment_cfg = cfg.get_segmentation_config()
    train_cfg = cfg.get_training_config()

    metadata_path = segment_cfg.resolve_relative_path_from_config('output_metadata')
    # Add external detections
    if not os.path.exists(metadata_path):
        raise FileNotFoundError("Could not find external annotations")
    train_cfg.config['tracker_params']['external_detections'] = segment_cfg.unresolve_absolute_path(metadata_path)

    return train_cfg


def update_path_to_behavior_in_config(cfg: ModularProjectConfig):
    """
    Used to update old projects that were not initialized with a behavior folder

    Parameters
    ----------
    cfg

    Returns
    -------

    """

    try:
        _ = cfg.get_behavior_config()
        print("Project already has a behavior config; update cannot be clearly automated")
        return
    except FileNotFoundError:
        pass

    # Then make a new folder, file, and fill it
    fname = Path(cfg.project_dir).joinpath('behavior', 'nwb_config.yaml')
    fname.parent.mkdir(exist_ok=False)
    behavior_cfg = SubfolderConfigFile(_self_path=str(fname), _config={}, project_dir=cfg.project_dir, subfolder='behavior')

    # Fill variable 1: Try to find behavior annotations
    raw_behavior_foldername, flag = cfg.get_behavior_raw_parent_folder_from_red_fname()
    if not flag:
        cfg.logger.warning("Could not find behavior foldername, so will create empty config file")
    else:
        annotated_behavior_csb = raw_behavior_foldername.joinpath('beh_annotation.csv')
        if not annotated_behavior_csb.exists():
            cfg.logger.warning(f"Could not find behavior file {annotated_behavior_csb}, so will create empty config file")
        else:
            behavior_cfg.config['manual_behavior_annotation'] = str(annotated_behavior_csb)

    # After filling (or not), write to disk
    behavior_cfg.update_self_on_disk()


def rename_variable_in_config(project_path: str, vars_to_rename: dict):
    """
    Renames variables, especially for updating variable names

    Overwrites the config file on disk

    Parameters
    ----------
    project_path
    vars_to_rename - nested dict with following levels
        key0 = project_config name (None is main level)
            (project, preprocessing, segmentation, training, tracking, traces)
        key1 = Variable name. Can be nested. If this isn't found, then skip
        key2 = If vars_to_rename[key0][key1] is not a dict, then this is the new variable name
            else, then recurse

    Returns
    -------

    """

    cfg = ModularProjectConfig(project_path)

    for file_key, vars_dict in vars_to_rename.items():
        if file_key == 'project':
            loaded_cfg = cfg
        else:
            # Note: this creates coupling with the naming convention...
            load_function_name = f'get_{file_key}_config'
            loaded_cfg = getattr(cfg, load_function_name)

        for old_name0, new_name0 in vars_dict.items():
            _cfg_to_update = loaded_cfg.config
            _update_config_value(file_key, _cfg_to_update, old_name0=old_name0, new_name0=new_name0)
        loaded_cfg.update_self_on_disk()


def update_variable_in_config(project_path: str, vars_to_update: dict):
    """
    Like rename_variable_in_config, but only updates the value, not the name

    This means that key2 should be different, but everything else about vars_to_update is the same as vars_to_rename

    Overwrites the config file on disk

    Parameters
    ----------
    project_path
    vars_to_update - nested dict with following levels
        key0 = project_config name (None is main level)
            (project, preprocessing, segmentation, training, tracking, traces)
        key1 = Variable name. Can be nested. If this isn't found, then skip
        key2 = If vars_to_rename[key0][key1] is not a dict, then this is the new value else, recurse

    Returns
    -------

    """

    cfg = ModularProjectConfig(project_path)

    for file_key, vars_dict in vars_to_update.items():
        if file_key == 'project':
            loaded_cfg = cfg
        else:
            # Note: this creates coupling with the naming convention...
            load_function_name = f'get_{file_key}_config'
            loaded_cfg = getattr(cfg, load_function_name)

        for key, val in vars_dict.items():
            _cfg_to_update = loaded_cfg.config
            _update_config_value(file_key, _cfg_to_update, old_name0=key, new_value=val)
        loaded_cfg.update_self_on_disk()


def _update_config_value(file_key, cfg_to_update, old_name0, new_name0=None, new_value=None):
    if cfg_to_update is None:
        return

    if new_value is None and new_name0 is None:
        raise ValueError("Must pass either new_value0 or new_name0")

    if new_name0 is not None and isinstance(new_name0, dict):
        for _old_name1, _new_name1 in new_name0.items():
            _cfg_to_update1 = cfg_to_update.get(_old_name1, None)
            _update_config_value(file_key, _cfg_to_update1, old_name0, new_name0)
        return

    if old_name0 not in cfg_to_update:
        msg = f"{old_name0} not found in config {file_key}, skipping"
        logging.warning(msg)
    else:
        if new_name0 is not None:
            new_val = cfg_to_update[old_name0]
            if new_name0 in cfg_to_update:
                msg = f"New name {new_name0} already found in config {file_key}!"
                raise NotImplementedError(msg)
            else:
                cfg_to_update[new_name0] = new_val
        else:
            cfg_to_update[old_name0] = new_value
    return cfg_to_update


def make_project_name_like(project_path: str, target_directory: str, target_suffix: str = None, 
                           new_project_name: str = None, verbose=1) -> Path:
    old_project_dir = Path(project_path).parent
    if new_project_name is None:
        new_project_name = old_project_dir.name
    if target_suffix is not None:
        new_project_name = f"{new_project_name}{target_suffix}"
    target_project_name = Path(target_directory).joinpath(new_project_name)
    return target_project_name, old_project_dir


def make_project_like(project_path: str, target_directory: str,
                      steps_to_keep: list = None,
                      target_suffix: str = None,
                      new_project_name: str = None, 
                      verbose=1):
    """
    Copy all config files from a project, i.e. only the files that would exist in a new project

    Parameters
    ----------
    project_path - project to copy
    target_directory - parent folder within which to create the new project
    steps_to_keep - steps, if any, to keep absolute paths connecting to the old project.
        Should be the full name of the step, not just a number (and not including the number). Example:
        steps_to_keep="['segmentation']" (note that the surrounding quotes are needed for the cli)
    target_suffix - suffix for filename. Default is none
    new_project_name - optional new name for project. Default is same as old
    verbose

    Returns
    -------

    """

    assert project_path.endswith('.yaml'), f"Must pass a valid config file: {project_path}"
    assert os.path.exists(project_path), f"Must pass a project that exists: {project_path}"
    assert os.path.exists(target_directory), f"Must pass a folder that exists: {target_directory}"


    target_project_name, old_project_dir = make_project_name_like(project_path, target_directory,
                                                              target_suffix=target_suffix,
                                                              new_project_name=new_project_name,
                                                              verbose=verbose)
    if os.path.exists(target_project_name):
        raise FileExistsError(f"There is already a project at: {target_project_name}")
    if verbose >= 1:
        print(f"Copying project in directory {old_project_dir} with new name {target_project_name}")

    # Get a list of all files that should be present, relative to the project directory
    src = get_location_of_new_project_defaults()
    template_fnames = list(Path(src).rglob('**/*'))
    if len(template_fnames) == 0:
        print(f"Found no template files, something went wrong with trying to find project template. Searched here: {src}")
        raise FileNotFoundError

    # Convert them to relative
    template_fnames = {str(fname.relative_to(src)) for fname in template_fnames}
    if verbose >= 3:
        print(f"Found template files: {template_fnames}")

    # Also get the filenames of the target folder
    old_project_fnames = list(Path(old_project_dir).rglob('**/*'))
    if verbose >= 3:
        print(f"Found files in the old project (to copy): {old_project_fnames}")
    if len(old_project_fnames) == 0:
        logging.warning("Found no files to copy; perhaps the old project is incorrectly formatted")

    # Check each initial project fname, and if it is in the initial set, copy it
    for fname in old_project_fnames:
        if fname.is_dir():
            continue
        rel_fname = fname.relative_to(old_project_dir)
        new_fname = target_project_name.joinpath(rel_fname)
        if str(rel_fname) in template_fnames:
            os.makedirs(new_fname.parent, exist_ok=True)
            shutil.copy(fname, new_fname)

            if verbose >= 1:
                print(f"Copying {rel_fname}")
        elif verbose >= 2:
            print(f"Not copying {rel_fname}")

    # Update the copied project config with the new dest folder
    update_project_config_path(target_project_name)

    # Connect the new project to old project config files, if any
    old_cfg = ModularProjectConfig(project_path)
    old_project_dir = old_cfg.project_dir
    if steps_to_keep is not None:
        project_updates = dict(subfolder_configs=dict())
        all_steps = list(old_cfg.config['subfolder_configs'].keys())

        for step in all_steps:
            subcfg_fname = old_cfg.config['subfolder_configs'].get(step, None)
            if subcfg_fname is None:
                raise NotImplementedError(step)

            if step in steps_to_keep:
                # Must make it absolute
                if Path(subcfg_fname).is_absolute():
                    project_updates['subfolder_configs'][step] = subcfg_fname
                else:
                    project_updates['subfolder_configs'][step] = os.path.join(old_project_dir, subcfg_fname)

            else:
                # Must explicitly include the relative path, otherwise it will be deleted
                if Path(subcfg_fname).is_absolute():
                    subcfg_fname = Path(subcfg_fname)
                    project_updates['subfolder_configs'][step] = os.path.join(subcfg_fname.parent.name, subcfg_fname.name)
                else:
                    project_updates['subfolder_configs'][step] = subcfg_fname

        dest_fname = 'project_config.yaml'
        project_fname = os.path.join(target_project_name, dest_fname)
        project_fname = str(Path(project_fname).resolve())
        edit_config(project_fname, project_updates)
    else:
        print("All new steps")

    # Also update the snakemake file with the project directory
    update_snakemake_config_path(target_project_name)

    # Specific check: if the data should be included, the old style might still have relative paths
    if 'preprocessing' in steps_to_keep:
        # Check if the preprocessed data is found (reload the project)
        project_config = ModularProjectConfig(str(target_project_name))
        preprocessing_class = project_config.get_preprocessing_class()
        fname = preprocessing_class.get_path_to_preprocessed_data(red_not_green=True)
        if fname is None:
            # Then there was no preprocessed data, ignore
            pass
        elif not os.path.exists(fname):
            # Then it is not found, and we should update the main config file with absolute paths
            # But it is the old style, so we need to update project_config not the preprocessing_config
            # But actually: check that it really is the old style
            red_fname = project_config.config.get('preprocessed_red', None)
            assert red_fname is not None, "Expected preprocessed_red in main config file (old style)"
            assert not Path(red_fname).is_absolute(), f"Expected relative path: {red_fname}"
            # Make the files absolute, pointing to the old project
            project_config.config['preprocessed_red'] = os.path.join(old_project_dir, red_fname)
            green_fname = project_config.config.get('preprocessed_green', None)
            project_config.config['preprocessed_green'] = os.path.join(old_project_dir, green_fname)
            project_config.update_self_on_disk()

    return target_project_name
