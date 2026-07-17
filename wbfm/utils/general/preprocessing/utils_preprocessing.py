import concurrent
import concurrent.futures
import dataclasses
import logging
import pickle
import threading
from dataclasses import dataclass, field
from typing import List, Optional
from skimage.transform import resize

from imutils import MicroscopeDataReader
from methodtools import lru_cache

import dask.array as da
import numpy as np
import zarr
from backports.cached_property import cached_property
from ruamel.yaml import YAML
from scipy import ndimage as ndi
from tifffile import tifffile
from tqdm.auto import tqdm
from types import SimpleNamespace

from wbfm.utils.external.utils_zarr import zarr_reader_folder_or_zipstore
from wbfm.utils.external.custom_errors import MustBeFiniteError, TiffFormatError
from wbfm.utils.general.preprocessing.deconvolution import ImageScaler, CustomPSF, sharpen_volume_using_dog, \
    sharpen_volume_using_bilateral
from wbfm.utils.external.utils_rigid_alignment import align_stack_to_middle_slice, \
    cumulative_alignment_of_stack, apply_alignment_matrix_to_stack, calculate_alignment_matrix_two_stacks
from wbfm.utils.projects.project_config_classes import ModularProjectConfig, ConfigFileWithProjectContext
from wbfm.utils.general.utils_filenames import add_name_suffix
from wbfm.utils.external.utils_yaml import edit_config
from wbfm.utils.general.video_and_data_conversion.import_video_as_array import get_single_volume
from wbfm.utils.projects.utils_project import RawFluorescenceData


def background_subtract_single_channel(raw_fname, background_fname, num_frames, num_slices, preprocessing_settings,
                                       DEBUG=False):

    raw_data = zarr_reader_folder_or_zipstore(raw_fname)
    background_video_list = read_background(background_fname, num_frames, num_slices,
                                            preprocessing_settings)
    # Add a new truly constant background value, to keep anything from going negative
    new_background = preprocessing_settings.reset_background_per_pixel
    # Get a single image, because that's the physical camera
    background_video_mean = np.mean(np.mean(background_video_list, axis=0), axis=0) - new_background
    # Don't try to modify the data as read; it is read-only
    fname_subtracted = add_name_suffix(raw_fname, '_background_subtracted')
    logging.info(f"Creating data copy at {fname_subtracted}")
    store = zarr.DirectoryStore(fname_subtracted)
    background_subtracted = zarr.zeros_like(raw_data, store=store)
    # Loop so that not all is loaded in memory... should I use dask?
    for i, volume in enumerate(tqdm(raw_data)):
        background_subtracted[i, ...] = np.array(np.maximum(volume - background_video_mean, 0), dtype=volume.dtype)
        if DEBUG and i > 5:
            break
    # zarr.save_array(background_subtracted, fname_subtracted)

    return fname_subtracted


def read_background(background_fname, num_frames, num_slices, preprocessing_settings=None):
    background_video_list = []
    with tifffile.TiffFile(background_fname) as background_tiff:
        for i in tqdm(range(num_frames)):
            try:
                background_volume = get_single_volume(background_tiff, i, num_slices, dtype='uint16')
            except IndexError:
                # The file should be found, but isn't long enough
                break
            # Note: this will do rigid rotation
            background_volume = perform_preprocessing(background_volume, preprocessing_settings, i)
            background_video_list.append(background_volume)
    logging.info(f"Read background tiff file of shape: {background_video_list[0].shape}")
    return background_video_list


@dataclass
class PreprocessingSettings(RawFluorescenceData):
    """
    Holds settings that will be applied to a video (.tiff or .btf)

    Designed to be used with the ReferenceFrame class
    """

    # Path to bounding boxes
    bounding_boxes_fname: str = None

    # Plane removal, especially flyback
    raw_number_of_planes: int = None
    starting_plane: int = None
    rescale_to_target_z: int = None

    # Filtering
    do_filtering: bool = False
    filter_opt: List = field(default_factory=lambda: {'high_freq': 2.0, 'low_freq': 5000.0})

    do_background_subtraction: bool = True
    background_fname_red: str = None
    background_fname_green: str = None

    reset_background: bool = True
    reset_background_per_pixel: int = 75

    # Mini max
    do_mini_max_projection: bool = False
    mini_max_size: int = 3

    # Mirroring, usually between green and red channels
    do_mirroring: bool = False

    # Rigid alignment (slices to each other)
    do_rigid_alignment: bool = False

    # Rigid alignment (green to red channel)
    align_green_red_cameras: bool = True
    camera_alignment_method: str = 'data'
    _camera_alignment_matrix: np.array = None
    path_to_camera_alignment_matrix: str = None
    gauss_filt_sigma: float = None
    max_project_across_time: bool = True

    # Deconvolution and other things (experimental)
    do_deconvolution: bool = False
    do_sharpening: bool = False
    do_sharpening_bilateral: bool = False

    sharpening_kwargs: dict = field(default_factory=dict)

    # Datatypes and scaling
    initial_dtype: str = 'uint16'  # Filtering etc. will act on this
    final_dtype: str = 'uint16'
    uint8_only_for_opencv: bool = True
    alpha: float = 0.15  # Deprecated

    alpha_red: float = None
    alpha_green: float = None

    # For updating self on disk
    cfg_preprocessing: ConfigFileWithProjectContext = None
    cfg_project: ModularProjectConfig = None

    # Load results of a separate preprocessing step, if available
    to_save_warp_matrices: bool = True
    to_use_previous_warp_matrices: bool = False
    path_to_previous_warp_matrices: str = None  # This file may not exist, and will be written the first time

    # Temporary variable to store warp matrices across time
    all_warp_matrices: dict = field(default_factory=dict)

    # Final output (preprocessed data)
    preprocessed_red_fname: str = None
    preprocessed_green_fname: str = None

    verbose: int = 0

    def __post_init__(self):
        if not self.alpha_is_ready:
            if self.cfg_preprocessing is not None:
                self.cfg_preprocessing.logger.warning("Alpha is not set in the yaml; will have to be calculated from data")

    @property
    def background_is_ready(self):
        if not self.do_background_subtraction:
            return True
        else:
            return self.background_red is not None

    @cached_property
    def deconvolution_scaler(self):
        return ImageScaler(self.reset_background_per_pixel)

    @cached_property
    def psf(self):
        return CustomPSF()

    @cached_property
    def background_red(self):
        if self.background_fname_red is None:
            return None
        try:
            return self.load_background(self.background_fname_red)
        except FileNotFoundError:
            return None

    @cached_property
    def background_green(self):
        if self.background_fname_green is None:
            return None
        try:
            return self.load_background(self.background_fname_green)
        except FileNotFoundError:
            return None

    def find_background_files_from_raw_data_path(self, force_search=False):
        if self.background_fname_red is not None:
            logging.info(f"Already has background at {self.background_fname_red}")
            if not force_search:
                return

        folder_for_background = self.cfg_project.get_folder_with_background()
        logging.info(f"Attempting to find new background files in parent folder {folder_for_background}")

        for subfolder in folder_for_background.iterdir():

            n = subfolder.name
            if 'BH' in n:
                # This is a behavior folder, which is not what we want
                continue

            subfolder_is_ch0 = ('background' in n and '_Ch0' in n) or ('background-channel-0' in n)
            subfolder_is_ch1 = ('background' in n and '_Ch1' in n) or ('background-channel-1' in n)

            if subfolder.is_dir() and subfolder_is_ch0:
                # Red channel
                for file in subfolder.iterdir():
                    if file.is_file() and 'background' in file.name and file.name.endswith('.btf'):
                        self.background_fname_red = str(file)
                        logging.info(f"Found red channel background at: {file}")
                        break
                else:
                    raise FileNotFoundError(f"Could not find red background file within {folder_for_background}")
            elif subfolder.is_dir() and subfolder_is_ch1:
                # Green channel
                for file in subfolder.iterdir():
                    if file.is_file() and 'background' in file.name and file.name.endswith('.btf'):
                        self.background_fname_green = str(file)
                        logging.info(f"Found green channel background at: {file}")
                        break
                else:
                    raise FileNotFoundError(f"Could not find green background file within {folder_for_background}")

        # Also update the preprocessing file on disk
        if not self.background_fname_red or not self.background_fname_green:
            raise FileNotFoundError(f"{self.background_fname_red} and {self.background_fname_green}")
        self.cfg_preprocessing.config['background_fname_red'] = self.background_fname_red
        self.cfg_preprocessing.config['background_fname_green'] = self.background_fname_green
        self.cfg_preprocessing.update_self_on_disk()

        # Actually load data
        _, _ = self.background_red, self.background_green

    def load_background(self, background_fname):
        num_frames = 10
        try:
            background_video_list = read_background(background_fname, num_frames, self.raw_number_of_planes,
                                                    preprocessing_settings=None)
        except IndexError:
            logging.warning(f"Found the background file at {background_fname}, but it was empty")
            return None
        # Add a new truly constant background value, to keep anything from going negative
        new_background = self.reset_background_per_pixel
        # Get a single image, because that's the physical camera
        background_video_mean = np.mean(np.mean(background_video_list, axis=0), axis=0)
        if self.reset_background:
            min_background_val = np.min(background_video_mean)
            if new_background > min_background_val:
                logging.warning(f"Can't set new background value to requested {new_background}, "
                                f"because some values would be negative. Using {min_background_val} instead")
                new_background = min_background_val

            background_video_mean -= new_background

        background_video_mean = background_video_mean.astype(self.initial_dtype)
        logging.info(f"Loaded background with mean: {np.mean(background_video_mean)}")

        return background_video_mean

    @property
    def alpha_is_ready(self):
        if self.alpha_red is not None and self.alpha_green is not None:
            return True
        else:
            return False

    def calculate_alpha_from_data(self, video, which_channel='red', num_volumes_to_load=None):
        # Note that this doesn't take into account background subtraction
        # Note: dask isn't faster than just numpy, but manages memory much better
        logging.info(f"Calculating alpha from data for channel {which_channel}; may take ~2 minutes per video")
        if num_volumes_to_load is None:
            # Load the entire video; only really works with zarr
            current_max_value = da.from_zarr(video).max().compute()
            targeted_max_value = 254.0
        else:
            # Assumed to be tiff files; note that the stack axis doesn't matter
            # Requires setting an estimated maximum, to give clipping room
            this_dat = np.vstack([self.get_single_volume(video, i) for i in range(num_volumes_to_load)])
            current_max_value = np.max(this_dat)
            targeted_max_value = 200.0

        alpha = float(targeted_max_value / current_max_value)
        logging.info(f"Calculated alpha={alpha} for channel {which_channel}")

        if which_channel == 'red':
            self.alpha_red = alpha
            self.cfg_preprocessing.config['alpha_red'] = alpha
        elif which_channel == 'green':
            self.alpha_green = alpha
            self.cfg_preprocessing.config['alpha_green'] = alpha

        self.cfg_preprocessing.update_self_on_disk()

    @staticmethod
    def load_from_yaml(fname, do_background_subtraction=None):
        with open(fname, 'r') as f:
            preprocessing_dict = YAML().load(f)
        if do_background_subtraction is not None:
            preprocessing_dict['do_background_subtraction'] = do_background_subtraction
        return PreprocessingSettings(**preprocessing_dict)

    @property
    def camera_alignment_matrix(self):
        if self._camera_alignment_matrix is None:
            return None
        else:
            return self._camera_alignment_matrix
        # warp_mat = get_precalculated_camera_alignment_matrix()

    def calculate_warp_mat(self, project_config):
        valid_methods = ['data', 'dots', 'grid']
        assert self.camera_alignment_method in valid_methods, \
            f"Invalid method found {self.camera_alignment_method}, must be one of {valid_methods}"

        if self.camera_alignment_method == 'data':
            self.calculate_warp_mat_from_btf_files()
        elif self.camera_alignment_method == 'dots':
            self.calculate_warp_mat_from_dot_overlay(project_config)
        if self.camera_alignment_method == 'grid':
            self.calculate_warp_mat_from_grid_overlay(project_config)

        # Check basic validity
        if np.isnan(self.camera_alignment_matrix).any():
            raise MustBeFiniteError(self.camera_alignment_matrix)

    def calculate_warp_mat_from_dot_overlay(self, project_config):
        # Find calibration videos, if present
        red_btf_fname, green_btf_fname = project_config.get_red_and_green_dot_alignment_bigtiffs()
        if red_btf_fname is None or green_btf_fname is None:
            raise NotImplementedError("Tried to calculate alignment from dot overlay, but it wasn't found.")
            # self.calculate_warp_mat_from_data(project_data.red_data, project_data.green_data)
            # return

        green_align = tifffile.imread(green_btf_fname)
        red_align = tifffile.imread(red_btf_fname)

        warp_mat = calculate_alignment_matrix_two_stacks(red_align, green_align)

        # Save in this object
        self._camera_alignment_matrix = warp_mat

    def calculate_warp_mat_from_grid_overlay(self, project_config):
        # Find calibration videos, if present
        red_btf_fnames, green_btf_fnames = project_config.get_red_and_green_grid_alignment_bigtiffs()
        if red_btf_fnames is None or green_btf_fnames is None:
            raise NotImplementedError("Tried to calculate alignment from dot overlay, but it wasn't found.")

        red_align = None
        # The num_slices shouldn't matter too
        tiff_opt = dict(which_vol=0, num_slices=22, dtype='uint16')
        for fname in red_btf_fnames:
            if red_align is None:
                red_align = get_single_volume(fname, **tiff_opt)
            else:
                red_align += get_single_volume(fname, **tiff_opt)

        green_align = None
        for fname in green_btf_fnames:
            if green_align is None:
                green_align = get_single_volume(fname, **tiff_opt)
            else:
                green_align += get_single_volume(fname, **tiff_opt)

        warp_mat = calculate_alignment_matrix_two_stacks(red_align, green_align)

        # Save in this object
        self._camera_alignment_matrix = warp_mat

    def calculate_warp_mat_from_data(self, red_data, green_data):
        """
        Calculate a matrix for aligning two channels of data (designed to change second input matrix)

        Does NOT apply the matrix

        Parameters
        ----------
        red_data
        green_data

        Returns
        -------
        Nothing

        """
        # Get representative volumes (in theory) and max project
        tspan = np.arange(10, red_data.shape[0], 250, dtype=int)
        red_vol_subset = np.array([np.max(red_data[t], axis=0) for t in tspan])
        green_vol_subset = np.array([np.max(green_data[t], axis=0) for t in tspan])

        # Calculate (average over above volumes)
        warp_mat = calculate_alignment_matrix_two_stacks(red_vol_subset, green_vol_subset,
                                                         use_only_first_pair=False)

        # Save in this object
        self._camera_alignment_matrix = warp_mat

    def calculate_warp_mat_from_btf_files(self):
        # Directly read and subsample
        red_dat = self.open_raw_data_as_4d_dask(red_not_green=True)
        green_dat = self.open_raw_data_as_4d_dask(red_not_green=False)
        num_frames = red_dat.shape[0]

        tspan = np.arange(10, num_frames, 50, dtype=int)

        # Max project across z
        red_vol_subset = np.array(np.max(red_dat[tspan], axis=1))
        green_vol_subset = np.array(np.max(green_dat[tspan], axis=1))

        if self.max_project_across_time:
            # Alignment is more stable for large misalignments if we max project across time as well as z
            red_vol_subset = [np.max(red_vol_subset, axis=0)]
            green_vol_subset = [np.max(green_vol_subset, axis=0)]

        # Calculate (average over above volumes)
        warp_mat = calculate_alignment_matrix_two_stacks(red_vol_subset, green_vol_subset,
                                                         use_only_first_pair=False,
                                                         gauss_filt_sigma=self.gauss_filt_sigma)

        # Save in this object
        self._camera_alignment_matrix = warp_mat

    def write_to_yaml(self, fname):
        edit_config(fname, dataclasses.asdict(self))

    def save_all_warp_matrices(self):
        with open(self.path_to_previous_warp_matrices, 'wb') as f:
            pickle.dump(self.all_warp_matrices, f)
        with open(self.path_to_camera_alignment_matrix, 'wb') as f:
            pickle.dump(self.camera_alignment_matrix, f)

    def load_all_warp_matrices(self):
        """
        Loads warp matrices from disk

        Note that if the same preprocessing_settings object is used to process multiple files, then this is not needed
        """
        with open(self.path_to_previous_warp_matrices, 'rb') as f:
            self.all_warp_matrices = pickle.load(f)

    def get_single_volume(self, video_dat_4d, i_time: int, num_slices=None):
        if num_slices is not None:
            raise NotImplementedError("Should set PreprocessingSettings.raw_number_of_planes")
        raw_volume = get_single_volume(video_dat_4d, i_time, self.raw_number_of_planes, dtype=self.initial_dtype)
        return raw_volume

    @lru_cache(maxsize=4)
    def _open_raw_data(self, red_not_green=True, actually_open=True) -> Optional[MicroscopeDataReader]:
        """
        Open the raw data file, which used to be a .btf file but is now an ndtiff folder

        Note: this returns a MicroscopeDataReader object, and the user should call .dask_array to get the data
            BUT: this is 6d, not 4d

        Parameters
        ----------
        red_not_green - if True, opens the red file, else the green file

        Returns
        -------

        """
        # First check filename style
        if self.cfg_project is not None:
            fname, is_btf = self.cfg_project.get_raw_data_fname(red_not_green)
        else:
            fname, is_btf = None, False

        # If not found, check if the data is directly loaded to this class
        if fname is None:
            if red_not_green and self._raw_red_data is not None:
                return SimpleNamespace(dask_array=self._raw_red_data)
            elif not red_not_green and self._raw_green_data is not None:
                return SimpleNamespace(dask_array=self._raw_green_data)
            else:
                raise FileNotFoundError("Could not find raw data file")

        # Open using the new DataReader
        if is_btf:
            z_slices = self.num_slices
            if z_slices is None:
                raise TiffFormatError("Could not find number of z slices in config file; "
                                      "Required if using .btf files")
            if actually_open:
                try:
                    dat = MicroscopeDataReader(fname, as_raw_tiff=True, raw_tiff_num_slices=z_slices,
                                            verbose=0)
                except KeyError:
                    logging.warning(f"Could not open {fname} as a MicroscopeDataReader; "
                                    f"possibly it is not a valid ndtiff folder")
                    dat = None
            else:
                dat = None
        else:
            # Has metadata already
            if actually_open:
                try:
                    dat = MicroscopeDataReader(fname, as_raw_tiff=False, verbose=0)
                except (KeyError, tifffile.TiffFileError):
                    logging.warning(f"Could not open {fname} as a MicroscopeDataReader; "
                                    f"possibly it is not a valid ndtiff folder")
                    dat = None
            else:
                dat = None

        return dat

    def open_raw_data_as_4d_dask(self, red_not_green=True) -> Optional[da.Array]:
        """Opens the raw fluorescence data as a 4d dask array, with shape (time, z, y, x)"""
        dat = self._open_raw_data(red_not_green)
        if dat is None:
            return None
        dat_out = da.squeeze(dat.dask_array)
        if dat_out.ndim != 4:
            raise ValueError(f"Expected 4d data, got {dat_out.ndim}d data")
        return dat_out

    @property
    def has_raw_data(self):
        try:
            _ = self.open_raw_data_as_4d_dask()
            return True
        except FileNotFoundError:
            return False

    @property
    def num_slices(self):
        """Just for backwards compability: Checks for either the old or new key"""
        num_slices = self.cfg_project.config['dataset_params'].get('num_slices', None)
        if num_slices is None and 'deprecated_dataset_params' in self.cfg_project.config:
            num_slices = self.cfg_project.config['deprecated_dataset_params'].get('num_slices', None)
        return num_slices

    def get_num_slices_robust(self):
        """
        Tries to read from the config file, but if that fails then read the raw data file

        Returns
        -------

        """
        num_slices = self.num_slices
        if num_slices is None:
            dat = self.open_raw_data_as_4d_dask()
            num_slices = dat.shape[1]
        return num_slices

    @property
    def start_volume(self):
        """Just for backwards compability: Checks for either the old or new key"""

        num_slices = self.cfg_project.config['dataset_params'].get('start_volume', 0)
        if num_slices is None and 'deprecated_dataset_params' in self.cfg_project.config:
            num_slices = self.cfg_project.config['deprecated_dataset_params'].get('start_volume', 0)
        return num_slices

    @property
    def num_frames(self):
        # Checks for either the old or new key
        num_frames = self.cfg_project.config['dataset_params'].get('num_frames', None)
        if num_frames is None and 'deprecated_dataset_params' in self.cfg_project.config:
            num_frames = self.cfg_project.config['deprecated_dataset_params'].get('num_frames', None)
        return num_frames

    def get_num_frames_robust(self):
        """
        Tries to read from the config file, but if that fails then read the raw data file

        If the raw data file is not found, then raises FileNotFoundError

        Returns
        -------

        """
        num_slices = self.num_frames
        if num_slices is None:
            dat = self.open_raw_data_as_4d_dask()
            num_slices = dat.shape[0]
        return num_slices

    def get_path_to_preprocessed_data(self, red_not_green=True, DEBUG=False) -> str:
        """
        Like get_raw_data_fname, but for preprocessed data

        Was previously just a direct config file lookup, but it has been moved and now checks both places

        Parameters
        ----------
        red_not_green

        Returns
        -------

        """
        if red_not_green:
            fname = self.cfg_preprocessing.resolve_relative_path_from_config('preprocessed_red_fname')
        else:
            fname = self.cfg_preprocessing.resolve_relative_path_from_config('preprocessed_green_fname')

        if fname is None:
            if DEBUG:
                self.cfg_preprocessing.logger.warning(f"Could not find preprocessed data for channel {red_not_green}; "
                                                      f"Checking old style")
            # Then check old style
            if red_not_green:
                fname = str(self.cfg_project.resolve_relative_path_from_config('preprocessed_red'))
            else:
                fname = str(self.cfg_project.resolve_relative_path_from_config('preprocessed_green'))
        else:
            if DEBUG:
                self.cfg_preprocessing.logger.warning(f"Found preprocessed data using new style at {fname}")

        return str(fname)

    def __repr__(self):
        return f"========================================================= \n\
Preprocessing settings object with settings: \n\
    Data settings: \n\
        raw_number_of_planes = {self.raw_number_of_planes} \n\
        starting_plane = {self.starting_plane} \n\
    Filtering:  \n\
        do_background_subtraction = {self.do_background_subtraction} \n\
        reset_background = {self.reset_background} \n\
        reset_background_per_pixel = {self.reset_background_per_pixel} \n\
    Rigid alignment (slices to each other): \n\
        do_rigid_alignment = {self.do_rigid_alignment} \n\
        Rigid alignment (green to red channel) \n\
        align_green_red_cameras = {self.align_green_red_cameras} \n\
        camera_alignment_method = {self.camera_alignment_method} \n\
    Deconvolution and other things (experimental): \n\
        do_sharpening = {self.do_sharpening} \n\
        sharpening_kwargs = {self.sharpening_kwargs} \n\
    Has raw data: \n\
         {self.has_raw_data}\n\
"


def perform_preprocessing(single_volume_raw: np.ndarray,
                          preprocessing_settings: PreprocessingSettings,
                          which_frame: int = None,
                          which_channel: str = 'red') -> np.ndarray:
    """
    Performs all preprocessing as set by the fields of preprocessing_settings

    See PreprocessingSettings for valid options

    Parameters
    ----------
    single_volume_raw
    preprocessing_settings
    which_frame
    which_channel

    Returns
    -------
    numpy array

    """

    s = preprocessing_settings
    if s is None:
        return single_volume_raw
    if s.final_dtype == 'uint8':
        logging.warning("This dtype should be uint16, not uint8. For now, it is ignored")

    if which_channel == 'red':
        alpha = s.alpha_red
        background = s.background_red
    elif which_channel == 'green':
        alpha = s.alpha_green
        background = s.background_green
    else:
        raise NotImplementedError(f"Unrecognized channel: {which_channel}")

    if s.starting_plane is not None:
        single_volume_raw = single_volume_raw[s.starting_plane:, ...]

    if s.do_background_subtraction:
        try:
            single_volume_raw = uint_safe_subtraction(single_volume_raw, background)
        except ValueError:
            logging.warning(f"The background {background.shape} was not the correct shape {single_volume_raw.shape}")

        except TypeError:
            logging.warning(f"Background was incorrect type: {background}")
        finally:
            logging.warning("Setting 'do_background_subtraction' to False")
            s.do_background_subtraction = False

    if s.do_filtering:
        raise NotImplementedError
        # single_volume_raw = filter_stack(single_volume_raw, s.filter_opt)

    if s.do_mirroring:
        single_volume_raw = np.flip(single_volume_raw, axis=-1)

    if s.do_rigid_alignment:
        if not s.to_use_previous_warp_matrices:
            import cv2
            try:
                single_volume_raw, warp_matrices_dict = align_stack_to_middle_slice(single_volume_raw)
            except cv2.error as e:
                logging.warning(f"When rigidly aligning in z, encountered opencv error {e}; leaving unaligned")
                warp_matrices_dict = {}
            if s.to_save_warp_matrices:
                s.all_warp_matrices[which_frame] = warp_matrices_dict
        else:
            assert len(s.all_warp_matrices) > 0, ("No warp matrices found, but to_use_previous_warp_matrices is True... "
                                                  "Possibly the red channel was not processed first, or the code crashed"
                                                  " before saving the matrices. Try deleting any preprocessed data in "
                                                  "the dat folder and rerunning preprocessing")
            warp_matrices_dict = s.all_warp_matrices[which_frame]
            if len(warp_matrices_dict) > 0:
                single_volume_raw = cumulative_alignment_of_stack(single_volume_raw, warp_matrices_dict)

    if s.align_green_red_cameras and which_channel == 'green':
        # Matrix should be precalculated
        alignment_mat = s.camera_alignment_matrix
        if alignment_mat is None:
            logging.warning("Requested red-green alignment, but no matrix was found")
        else:
            single_volume_raw = apply_alignment_matrix_to_stack(single_volume_raw, alignment_mat)

    # if s.do_deconvolution:
    #     single_volume_raw = s.psf.deconvolve_single_volume_2d(single_volume_raw)
    #     s.psf.scaler.reset()

    if s.do_sharpening:
        scaler = ImageScaler()
        single_volume_raw = sharpen_volume_using_dog(scaler.scale_volume(single_volume_raw), s.sharpening_kwargs,
                                                     verbose=0)
        single_volume_raw = scaler.unscale_volume(single_volume_raw)

    if s.do_sharpening_bilateral:
        scaler = ImageScaler()
        single_volume_raw = sharpen_volume_using_bilateral(scaler.scale_volume(single_volume_raw))
        single_volume_raw = scaler.unscale_volume(single_volume_raw)

    if s.do_mini_max_projection:
        mini_max_size = s.mini_max_size
        single_volume_raw = ndi.maximum_filter(single_volume_raw, size=(mini_max_size, 1, 1))

    if s.rescale_to_target_z is not None:
        target_shape = (s.rescale_to_target_z, single_volume_raw.shape[1], single_volume_raw.shape[2])
        single_volume_raw = resize(single_volume_raw, target_shape, order=3, preserve_range=True)

    # Do not actually change datatype
    if not s.uint8_only_for_opencv:
        raise DeprecationWarning("uint8 should not be saved directly, but converted on demand for opencv")

    return single_volume_raw


def preprocess_all_frames_using_config(config: ModularProjectConfig,
                                       preprocessing_settings: PreprocessingSettings = None, which_frames: list = None,
                                       which_channel: str = None, out_fname: str = None, verbose: int = 0,
                                       DEBUG: bool = False) -> zarr.Array:
    """
    Preprocesses all frames that will be analyzed as per config

    NOTE: expects 'preprocessing_config' and 'dataset_params' to be in config
    OR for the PreprocessingSettings object to be passed directly

    Loads but does not process frames before config['dataset_params']['start_volume']
        (to keep the indices the same as the original dataset)

    Parameters
    ----------
    config: config class loaded from yaml
    preprocessing_settings: class with preprocessing settings
    which_frames: list of frames to analyze. Optional
    which_channel: red or green
    out_fname: filename of output video
    verbose
    DEBUG

    Returns
    -------

    """
    if preprocessing_settings is None:
        p = config.get_preprocessing_class()
    else:
        p = preprocessing_settings

    video_dat_4d = p.open_raw_data_as_4d_dask(red_not_green=(which_channel == 'red'))
    return preprocess_all_frames(video_dat_4d, p, which_channel, out_fname, DEBUG=DEBUG)


def preprocess_all_frames(video_dat_4d: da, p: PreprocessingSettings, which_channel: str, out_fname: str,
                          DEBUG: bool = False) -> zarr.Array:
    """
    Preprocesses all frames using multithreading, saving directly to an output zarr file

    Parameters
    ----------
    DEBUG
    p
    which_channel - Needed to get the correct alpha and warp matrices
    out_fname

    Returns
    -------

    """

    total_sz = video_dat_4d.shape
    if p.rescale_to_target_z is not None:
        total_sz = list(total_sz)
        total_sz[1] = p.rescale_to_target_z
        total_sz = tuple(total_sz)

    chunk_sz = (1,) + total_sz[1:]

    try:
        store = zarr.DirectoryStore(path=out_fname)
    except AttributeError:
        store = zarr.storage.LocalStore(path=out_fname)
    logging.info(f"Preprocessing all frames and saving to {out_fname}, with settings: "
                 f"total_sz={total_sz}, chunk_sz={chunk_sz}")
    if p.final_dtype == np.uint8:
        raise DeprecationWarning("uint16 should be saved directly")
    preprocessed_dat = zarr.zeros(total_sz, chunks=chunk_sz, dtype=p.initial_dtype,
                                  synchronizer=zarr.ThreadSynchronizer(),
                                  store=store, overwrite=True)
    read_lock = threading.Lock()

    max_workers = 32
    if p.do_deconvolution:
        max_workers = 1
    # Load data and preprocess
    num_total_frames = video_dat_4d.shape[0]
    frame_list = list(range(num_total_frames))

    # Note: this saves alpha to disk
    p.calculate_alpha_from_data(video_dat_4d, which_channel=which_channel, num_volumes_to_load=10)
    start_volume = 0

    with tqdm(total=num_total_frames) as pbar:
        def parallel_func(i):
            preprocessed_dat[i, ...] = get_and_preprocess(i, p, start_volume, video_dat_4d,
                                                          which_channel, read_lock)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(parallel_func, i): i for i in frame_list}
            for future in concurrent.futures.as_completed(futures):
                future.result()
                pbar.update(1)

    return preprocessed_dat


def _get_video_options(config, video_fname):
    import tifffile
    if video_fname.endswith('.tif') or video_fname.endswith('.btf'):
        with tifffile.TiffFile(video_fname) as tif:
            sz = tif.pages[0].shape
    elif video_fname.endswith('.zarr'):
        sz = zarr.open(video_fname).shape[2:]
    else:
        raise FileNotFoundError("Must pass .zarr or .tif or .btf file")
    # vid_opt = {'fps': config['dataset_params']['fps'],
    #            'frame_height': sz[0],
    #            'frame_width': sz[1],
    #            'is_color': False}
    # if 'training_data_2d' in config:
    #     vid_opt['is_color'] = config['training_data_2d'].get('is_color', False)
    return sz


def get_and_preprocess(i, p, start_volume, video_dat_4d, which_channel, read_lock=None):
    """
    See perform_preprocessing

    Note: the preprocessing class must know the number of planes in a volume

    Parameters
    ----------
    i: frame index (time)
    p: preprocessing class
    start_volume
    video_dat_4d
    which_channel
    read_lock

    Returns
    -------

    """
    if read_lock is None:
        single_volume_raw = p.get_single_volume(video_dat_4d, i)
    else:
        with read_lock:
            single_volume_raw = p.get_single_volume(video_dat_4d, i)
    # Don't preprocess data that we didn't even segment!
    if i >= start_volume:
        return perform_preprocessing(single_volume_raw, p, i, which_channel=which_channel)
    else:
        return single_volume_raw


def uint_safe_subtraction(vol_raw, background):
    """
    Subtracts uint values with clipping instead of overflow

    Assumes uint16

    Parameters
    ----------
    vol_raw
    background

    Returns
    -------

    """
    vol_int = vol_raw.astype(int) - background

    max_val = 65536
    vol_int = np.clip(vol_int, 0, max_val).astype('uint16')
    return vol_int
