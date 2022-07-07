import json
import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
import pprint

from DLC_for_WBFM.utils.general.utils_logging import setup_logger_object, setup_root_logger
from DLC_for_WBFM.utils.projects.physical_units import PhysicalUnitConversion
from DLC_for_WBFM.utils.projects.utils_filenames import check_exists, resolve_mounted_path_in_current_os, \
    get_sequential_filename
from DLC_for_WBFM.utils.projects.utils_project import load_config, edit_config, safe_cd


@dataclass
class ConfigFileWithProjectContext:
    """
    Top-level configuration file class

    Knows how to:
    1. update itself on disk
    2. save new data inside the relevant project
    3. change filepaths between relative and absolute
    """

    self_path: str
    config: dict = None
    project_dir: str = None

    _logger: logging.Logger = None
    log_to_file: bool = True

    def __post_init__(self):
        self.config = load_config(self.self_path)
        self.project_dir = str(Path(self.self_path).parent)

    @property
    def logger(self):
        if self._logger is None:
            self.setup_logger('ConfigFile.log')
        return self._logger

    def setup_logger(self, relative_log_filename: str):
        log_filename = self.resolve_relative_path(os.path.join('log', relative_log_filename))
        self._logger = setup_logger_object(log_filename, self.log_to_file)
        return self._logger

    def setup_global_logger(self, relative_log_filename: str):
        log_filename = self.resolve_relative_path(os.path.join('log', relative_log_filename))
        logger = setup_root_logger(log_filename)
        return logger

    def update_self_on_disk(self):
        fname = self.resolve_relative_path(self.self_path)
        edit_config(fname, self.config)
        self.logger.info(f"Updating config file {fname} on disk")

    def resolve_relative_path_from_config(self, key) -> str:
        val = self.config.get(key, None)
        return self.resolve_relative_path(val)

    def resolve_relative_path(self, val: str) -> str:
        if val is None:
            return None
        relative_path = Path(self.project_dir).joinpath(val)
        return str(relative_path.resolve())

    def unresolve_absolute_path(self, val: str) -> str:
        if val is None:
            return None
        # NOTE: is_relative_to() only works for python >= 3.9
        # if Path(val).is_relative_to(self.project_dir):
        try:
            return str(Path(val).relative_to(self.project_dir))
        except ValueError:
            return val

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

    def h5_data_in_local_project(self, data: pd.DataFrame, relative_path: str,
                                 allow_overwrite=True, make_sequential_filename=False, also_save_csv=False):
        abs_path = self.resolve_relative_path(relative_path)
        if not abs_path.endswith('.h5'):
            abs_path += ".h5"
        if make_sequential_filename:
            abs_path = get_sequential_filename(abs_path)
        self.logger.info(f"Saving at: {self.unresolve_absolute_path(abs_path)}")
        check_exists(abs_path, allow_overwrite)
        data.to_hdf(abs_path, key="df_with_missing")

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

    In principle this config file is associated with a subfolder (and single step) of a project
    """

    subfolder: str = None

    def __post_init__(self):
        pass

    def resolve_relative_path(self, val: str, prepend_subfolder=False) -> str:
        if val is None:
            return None

        if prepend_subfolder:
            final_path = os.path.join(self.project_dir, self.subfolder, val)
        else:
            final_path = os.path.join(self.project_dir, val)
        return str(Path(final_path).resolve())


@dataclass
class ModularProjectConfig(ConfigFileWithProjectContext):
    """
    Add functionality to get individual config files using the main project config filepath

    Returns config_file_with_project_context objects, instead of raw dictionaries for the subconfig files

    Knows how to:
    1. find the individual config files of the substeps
    2. initialize the physical unit conversion class
    3. and loading other options classes

    """

    def get_segmentation_config(self):
        fname = Path(self.config['subfolder_configs']['segmentation'])
        return SubfolderConfigFile(**self._check_path_and_load_config(fname))

    def get_training_config(self):
        fname = Path(self.config['subfolder_configs']['training_data'])
        return SubfolderConfigFile(**self._check_path_and_load_config(fname))

    def get_tracking_config(self):
        fname = Path(self.config['subfolder_configs']['tracking'])
        return SubfolderConfigFile(**self._check_path_and_load_config(fname))

    def get_preprocessing_config(self):
        """
        Not often used, except for updating the file.

        Note: NOT a subfolder
        """
        fname = str(Path(self.project_dir).joinpath('preprocessing_config.yaml'))
        return ConfigFileWithProjectContext(fname)

    def get_behavior_config(self):
        fname = Path(self.project_dir).joinpath('behavior', 'behavior_config.yaml')
        if not fname.exists():
            self.logger.warning("Project does not have a behavior config file")
            raise FileNotFoundError
        return SubfolderConfigFile(**self._check_path_and_load_config(fname))

    def get_traces_config(self):
        fname = Path(self.config['subfolder_configs']['traces'])
        return SubfolderConfigFile(**self._check_path_and_load_config(fname))

    def get_physical_unit_conversion_class(self) -> PhysicalUnitConversion:
        if 'physical_units' in self.config:
            return PhysicalUnitConversion(**self.config['physical_units'])
        else:
            self.logger.warning("Using default physical unit conversions")
            return PhysicalUnitConversion()

    def _check_path_and_load_config(self, subconfig_path: Path) -> Dict:
        if subconfig_path.is_absolute():
            project_dir = subconfig_path.parent.parent
        else:
            project_dir = Path(self.self_path).parent
        with safe_cd(project_dir):
            cfg = load_config(subconfig_path)
        subfolder = subconfig_path.parent

        args = dict(self_path=str(subconfig_path),
                    config=cfg,
                    project_dir=str(project_dir),
                    _logger=self.logger,
                    subfolder=str(subfolder))
        return args

    def get_log_dir(self) -> str:
        foldername = Path(self.project_dir).joinpath('log')
        foldername.mkdir(exist_ok=True)
        return str(foldername)

    def get_visualization_dir(self) -> str:
        foldername = Path(self.project_dir).joinpath('visualization')
        foldername.mkdir(exist_ok=True)
        return str(foldername)

    def resolve_mounted_path_in_current_os(self, key) -> Optional[Path]:
        path = self.config[key]
        if path is None:
            return None
        return Path(resolve_mounted_path_in_current_os(path))

    def get_folder_with_calibration_for_day(self) -> Optional[Path]:
        raw_bigtiff_filename = self.resolve_mounted_path_in_current_os('red_bigtiff_fname')
        if raw_bigtiff_filename is None:
            raise FileNotFoundError(raw_bigtiff_filename)
        raw_bigtiff_filename = Path(raw_bigtiff_filename)
        folder_for_entire_day = raw_bigtiff_filename.parents[2]  # 3 folders up

        if not Path(folder_for_entire_day).exists():
            raise FileNotFoundError(folder_for_entire_day)
        return folder_for_entire_day

    def get_folder_with_background(self) -> Path:
        folder_for_entire_day = self.get_folder_with_calibration_for_day()
        folder_for_background = folder_for_entire_day.joinpath('background')
        if not folder_for_background.exists():
            raise FileNotFoundError(f"Could not find background folder {folder_for_background}")

        return folder_for_background

    def get_folder_with_calibration(self):
        folder_for_entire_day = self.get_folder_with_calibration_for_day()
        folder_for_calibration = folder_for_entire_day.joinpath('calibration')
        if not folder_for_calibration.exists():
            raise FileNotFoundError(f"Could not find calibration folder {folder_for_calibration}")

        return folder_for_calibration

    def get_behavior_raw_file_from_red_fname(self):
        """If the user did not set the behavior foldername, try to infer it from the red"""
        behavior_subfolder, flag = self.get_behavior_raw_parent_folder_from_red_fname()
        if not flag:
            return None
        # Second, get the file itself
        for content in behavior_subfolder.iterdir():
            if content.is_file():
                # UK spelling, and there may be preprocessed bigtiffs in the folder
                if str(content).endswith('-behaviour-bigtiff.btf'):
                    behavior_fname = behavior_subfolder.joinpath(content)
                    break
        else:
            print(f"Found no behavior file in {behavior_subfolder}, aborting")
            return None

        return behavior_fname, behavior_subfolder

    def get_behavior_raw_parent_folder_from_red_fname(self) -> Tuple[Optional[Path], bool]:
        red_fname = self.config['red_bigtiff_fname']
        if red_fname is None:
            return None, False
        main_data_folder = Path(red_fname).parents[1]
        if not main_data_folder.exists():
            return None, False
        # First, get the subfolder
        for content in main_data_folder.iterdir():
            if content.is_dir():
                # UK spelling
                if 'behaviour' in content.name:
                    behavior_subfolder = main_data_folder.joinpath(content)
                    flag = True
                    break
        else:
            print(f"Found no behavior subfolder in {main_data_folder}, aborting")
            flag = False
            behavior_subfolder = None
        return behavior_subfolder, flag


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
    # Used to update old projects that were not initialized with a behavior folder
    try:
        _ = cfg.get_behavior_config()
        print("Project already has a behavior config; update cannot be clearly automated")
        return
    except FileNotFoundError:
        pass

    # Then make a new folder, file, and fill it
    fname = Path(cfg.project_dir).joinpath('behavior', 'behavior_config.yaml')
    fname.parent.mkdir(exist_ok=False)
    behavior_cfg = SubfolderConfigFile(self_path=str(fname), config={}, project_dir=cfg.project_dir, subfolder='behavior')

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
