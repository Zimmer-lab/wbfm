"""
The metadata generator for the segmentation pipeline
"""
import concurrent.futures
import logging
import pickle
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
from wbfm.utils.external.custom_errors import MissingAnalysisError
from wbfm.utils.general.postprocessing.utils_metadata import regionprops_one_volume
from wbfm.utils.general.utils_filenames import pickle_load_binary
from wbfm.utils.external.utils_neuron_names import name2int_neuron_and_tracklet, int2name_neuron

import dask
from dask.diagnostics import ProgressBar
import numpy as np
import pandas as pd
import zarr
from wbfm.utils.segmentation.util.utils_config_files import _unpack_config_file
from skimage.measure import label, regionprops
from tqdm import tqdm


def OLD_get_metadata_dictionary(masks, original_vol):
    """
    Creates a dataframe with metadata ('total_brightness', 'neuron_volume', 'centroids') for a volume
    Parameters
    ----------
    masks : 3D numpy array
        contains the segmented and stitched masks
    original_vol : 3d numpy array
        original volume of the recording, which was segmented into 'masks'.
        Contains the actual brightness values.
    planes = : dict(list)
        Dictionary containing a list of Z-slices on which a neuron is present
        neuron_ID = 1
        planes[neuron_ID] # [12, 13, 14]

    Returns
    -------
    df : pandas dataframe
        for each neuron (=row) the total brightness, volume and centroid is saved
        dataframe(columns = 'total_brightness', 'neuron_volume', 'centroids';
                  rows = neuron #)
    """
    # metadata_dict = {(Vol #, Neuron #) = [Total brightness, neuron volume, centroids]}
    neurons_list = np.unique(masks)
    neurons_list = np.delete(neurons_list, np.where(neurons_list == 0))
    neurons = []
    # create list of integers for each neuron ID (instead of float)
    for x in neurons_list:
        neurons.append(int(x))

    # create dataframe with
    # cols = (total brightness, volume, centroids, all_values (full histogram))
    # rows = neuron ID
    df = pd.DataFrame(index=neurons,
                      columns=['total_brightness', 'neuron_volume', 'centroids', 'all_values'])

    # TODO: refactor regionprops outside of list
    for n in neurons:
        neuron_mask = masks == n
        neuron_vol = np.count_nonzero(neuron_mask)

        original_vals = original_vol[neuron_mask]
        total_brightness = np.sum(original_vals)
        vals = np.ravel(original_vals)
        # vals, counts = np.unique(original_vals, return_counts=True)

        neuron_label = label(neuron_mask)
        # NOTE: for skimage>0.19 this property changes name
        try:
            centroids = regionprops(neuron_label, intensity_image=original_vol)[0].weighted_centroid
        except AttributeError:
            centroids = regionprops(neuron_label, intensity_image=original_vol)[0].centroid_weighted

        df.at[n, 'total_brightness'] = total_brightness
        df.at[n, 'neuron_volume'] = neuron_vol
        df.at[n, 'centroids'] = centroids
        df.at[n, 'all_values'] = vals
        # df.at[n, 'pixel_counts'] = counts

    return df


def get_metadata_dictionary(masks, original_vol, name_mode='neuron', props_to_save=None,
                            raise_if_too_many_neurons=False):
    if props_to_save is None:
        props_to_save = ['area', 'weighted_centroid', 'intensity_image', 'label', 'bbox', 'intensity_mean']
    props = regionprops_one_volume(masks, original_vol, props_to_save, name_mode=name_mode)

    # Convert back to old (Niklas) style
    dict_of_rows = defaultdict(list)
    for k, v in props.items():
        idx = name2int_neuron_and_tracklet(k[0])

        # Assume the entries were originally added in regionprops order
        dict_of_rows[idx].append(v)

    # NOTE: deprecates "all_values"
    new_names = ['neuron_volume', 'centroids', 'total_brightness', 'label', 'bbox', 'intensity_mean']
    df_metadata = pd.DataFrame.from_dict(dict_of_rows, orient='index', columns=new_names)

    return df_metadata


def centroids_from_dict_of_dataframes(dict_of_dataframes, i_volume) -> np.ndarray:
    vol0_zxy = dict_of_dataframes[i_volume]['centroids'].to_numpy()
    return np.array([np.array(m) for m in vol0_zxy])


@dataclass
class DetectedNeurons:

    detection_fname: str

    _segmentation_metadata: Dict[str, pd.DataFrame] = None
    _num_frames: int = None

    _brightnesses_cache: dict = None
    _volumes_cache: dict = None

    def __post_init__(self):
        if self._brightnesses_cache is None:
            self._brightnesses_cache = {}
        if self._volumes_cache is None:
            self._volumes_cache = {}

    def setup(self):
        _ = self.segmentation_metadata

    @property
    def segmentation_metadata(self):
        """
        Main data structure for metadata, saved according to an old standard:
        dict of dataframes, with columns as renamed versions of regionprops output

        Returns
        -------

        """
        if not Path(self.detection_fname).exists():
            raise FileNotFoundError(f"{self.detection_fname} doesn't exist!")
        
        if self._segmentation_metadata is None:
            # Note: dict of dataframes
            try:
                self._segmentation_metadata = pickle_load_binary(self.detection_fname)
            except EOFError:
                backup_fname = Path(self.detection_fname).with_name("backup_metadata.pickle")
                self._segmentation_metadata = pickle_load_binary(backup_fname)
                logging.error(f"Could not load {self.detection_fname}, likely corrupted. "
                              f"Loaded backup at {backup_fname}, but it may be out of sync")
                self.detection_fname = str(backup_fname)
        return self._segmentation_metadata

    @property
    def num_frames(self):
        if self._num_frames is None:
            self._num_frames = len(self.segmentation_metadata)
        return self._num_frames

    @property
    def which_frames(self):
        ind = list(self.segmentation_metadata.keys())
        ind.sort()
        return ind

    @property
    def volumes_with_no_neurons(self) -> list:
        empty_ind = []
        for t in self.which_frames:
            if len(self.detect_neurons_from_file(int(t))) == 0:
                empty_ind.append(t)
        return empty_ind

    def get_all_brightnesses(self, i_volume: int, is_relative_index=False) -> pd.Series:
        if is_relative_index:
            i_volume = self.correct_relative_index(i_volume)
        if i_volume not in self._brightnesses_cache:
            self._brightnesses_cache[i_volume] = self.segmentation_metadata[i_volume]['total_brightness']
        return self._brightnesses_cache[i_volume]

    def get_all_volumes(self, i_volume: int, is_relative_index=False) -> pd.Series:
        if is_relative_index:
            i_volume = self.correct_relative_index(i_volume)
        if i_volume not in self._volumes_cache:
            self._volumes_cache[i_volume] = self.segmentation_metadata[i_volume]['neuron_volume']
        return self._volumes_cache[i_volume]

    def intensity_background_subtracted(self, i_volume: int, background_per_pixel=14, is_relative_index=False):
        if is_relative_index:
            i_volume = self.correct_relative_index(i_volume)
        y = self.get_all_brightnesses(i_volume)
        vol = self.get_all_volumes(i_volume)
        return y - background_per_pixel*vol

    def intensity_mean_over_volume(self, i_volume: int, is_relative_index=False):
        if is_relative_index:
            i_volume = self.correct_relative_index(i_volume)
        y = self.get_all_brightnesses(i_volume)
        vol = self.get_all_volumes(i_volume)
        return y / vol

    def correct_relative_index(self, i):
        return self.which_frames[i]

    def modify_segmentation_metadata(self, i_volume, new_masks, red_volume):
        self.segmentation_metadata[i_volume] = get_metadata_dictionary(new_masks, red_volume)

        self._volumes_cache.pop(i_volume, None)
        self._brightnesses_cache.pop(i_volume, None)

    def get_all_metadata_for_single_time(self, mask_ind, t, likelihood=1.0):
        """
        Gets all metadata for a single neuron (via the mask_index) at a single time point

        Correct null value is None

        See also: get_all_neuron_metadata_for_single_time
        """
        ind_in_list = self.mask_index_to_i_in_array(t, mask_ind)
        column_names = self._column_names
        if ind_in_list is None:
            return None, column_names
        zxy = self.mask_index_to_zxy(t, mask_ind)
        red = float(self.get_all_brightnesses(t).iat[ind_in_list])
        vol = int(self.get_all_volumes(t).iat[ind_in_list])
        row_data = [zxy[0], zxy[1], zxy[2], likelihood, ind_in_list, mask_ind, red, vol]
        return row_data, column_names

    @property
    def _column_names(self):
        return ['z', 'x', 'y', 'likelihood', 'raw_neuron_ind_in_list', 'raw_segmentation_id',
                'brightness_red', 'volume']

    def get_all_neuron_metadata_for_single_time(self, t, likelihood=1.0, use_mean_intensity=False,
                                                as_dataframe=True) -> Tuple[list, list]:
        """
        Returns all metadata for all neurons at a single time

        If there are no neurons, the returns two empty lists
        """
        if t in self.volumes_with_no_neurons:
            return [], []
        all_metadata = self.segmentation_metadata[t].copy()
        column_names = self._column_names.copy()
        # Reformat using the new column names
        zxy = np.array(all_metadata['centroids'].values.tolist())
        if len(zxy.shape) > 2:
            # Then there are multiple channels, and we want to average them
            zxy = zxy.mean(axis=-1)
        red = all_metadata['total_brightness'].to_numpy()
        vol = all_metadata['neuron_volume'].to_numpy()
        mask_ind = all_metadata['label']
        ind_in_list = [self.mask_index_to_i_in_array(t, i) for i in mask_ind]
        likelihood_vec = [likelihood] * len(mask_ind)
        if use_mean_intensity:
            mean = np.array(all_metadata['intensity_mean'].values.tolist())
            if len(mean.shape) > 1:
                # Then there are multiple channels, and we need to loop for intensity
                # And take the mean for the centroid
                row_data = [zxy[:, 0], zxy[:, 1], zxy[:, 2], likelihood_vec, ind_in_list, mask_ind, red, vol]
                for i in range(mean.shape[1]):
                    column_names.append(f'mean_intensity_{i}')
                    row_data.append(mean[:, i])
            else:
                row_data = [zxy[:, 0], zxy[:, 1], zxy[:, 2], likelihood_vec, ind_in_list, mask_ind, mean, vol]
        else:
            row_data = [zxy[:, 0], zxy[:, 1], zxy[:, 2], likelihood_vec, ind_in_list, mask_ind, red, vol]
        if as_dataframe:
            # err
            data_dict = {
                (int2name_neuron(i+1), attribute_name): attribute_val
                for attribute_name, attribute_vec in zip(column_names, row_data)
                for i, attribute_val in enumerate(attribute_vec)
            }
            df = pd.DataFrame(data_dict, index=[t])
            return df
        else:
            return row_data, column_names

    def overwrite_original_detection_file(self):
        backup_fname = Path(self.detection_fname).with_name("backup_metadata.pickle")
        try:
            if not backup_fname.exists():
                shutil.copy(self.detection_fname, backup_fname)
            else:
                # Assume the backup was already copied
                pass
        except (PermissionError, OSError):
            logging.warning("Could not create backup copy, will still attempt to overwrite original file")

        try:
            with open(self.detection_fname, 'wb') as f:
                # Note: dict of dataframes
                pickle.dump(self._segmentation_metadata, f)
            logging.warning(f"Overwriting original file; backup saved at {backup_fname}")
        except (PermissionError, OSError) as e:
            logging.warning("Could not overwrite original file; if you want to save changes to segmentation, "
                            "you must have permissions. Original error: "
                            "\n" + str(e))
            return False
        return True

    def detect_neurons_from_file(self, i_volume: int, numpy_not_list=True) -> np.ndarray:
        """
        Designed to be used with centroids detected using a different pipeline
        """
        if numpy_not_list:
            neuron_locs = centroids_from_dict_of_dataframes(self.segmentation_metadata, i_volume)
        else:
            neuron_locs = self.segmentation_metadata[i_volume]['centroids']
            neuron_locs = np.array([n for n in neuron_locs])

        if len(neuron_locs) > 0:
            pass
            # neuron_locs = neuron_locs[:, [0, 2, 1]]
        else:
            neuron_locs = []

        return neuron_locs

    def i_in_array_to_mask_index(self, i_time, i_index):
        # Given the row index in the position matrix, return the corresponding mask label integer
        return self.segmentation_metadata[i_time].iloc[i_index].name

    def mask_index_to_i_in_array(self, i_time, mask_index):
        # Inverse of i_in_array_to_mask_index
        # Return index of seg array given the mask index, IF found
        these_indices = list(self.segmentation_metadata[i_time].index)
        if mask_index in these_indices:
            return these_indices.index(mask_index)
        else:
            return None

    def mask_index_to_zxy(self, i_time, mask_index):
        # See mask_index_to_i_in_array
        # Return position given the mask index
        seg_index = self.mask_index_to_i_in_array(i_time, mask_index)
        return np.array(self.segmentation_metadata[i_time].iloc[seg_index]['centroids'])

    def print_statistics(self, detail_level=1):
        print(self)
        if detail_level >= 1:
            print(f"Number of empty volumes: {len(self.volumes_with_no_neurons)}")
        else:
            return

        if detail_level >= 2:
            t = 10
            b = self.get_all_brightnesses(t)
            v = self.get_all_volumes(t)
            print(f"Found {len(v)} neurons at t={t}")
            print(f"Mean brightness: {np.mean(b)}")
            print(f"Mean volume: {np.mean(v)}")

    def __repr__(self):
        return f"DetectedNeurons object with {self.num_frames} frames"


def recalculate_metadata_from_config(project_cfg, name_mode, DEBUG=False,
                                     **project_kwargs):
    """

    Given a project that contains a segmentation, recalculate the metadata

    Parameters
    ----------
    DEBUG
    project_cfg
    segment_cfg

    Returns
    -------
    Saves metadata.pickle to disk (within folder 1-segmentation)

    See also:
        segment_video_using_config_3d

    """
    from wbfm.utils.projects.finished_project_data import ProjectData

    project_data = ProjectData.load_final_project_data(project_cfg, **project_kwargs)
    segment_cfg = project_data.project_config.get_segmentation_config()

    # Load from the project directly instead of passing the config files
    masks_zarr = project_data.raw_segmentation
    if masks_zarr is None:
        raise MissingAnalysisError("No segmentation masks found in the project data. "
                                   "Please run the segmentation pipeline first.")
    video_dat = project_data.red_data
    if video_dat is None:
        raise MissingAnalysisError("No video data found in the project data. "
                                             "Please run the preprocessing pipeline first.")

    frame_list = list(range(video_dat.shape[0]))

    metadata_fname = segment_cfg.resolve_relative_path_from_config('output_metadata')

    logging.info(f"Read zarr with size {masks_zarr.shape}")
    logging.info(f"Read video with size {video_dat.shape}")

    if DEBUG:
        frame_list = frame_list[:2]

    calc_metadata_full_video(frame_list, masks_zarr, video_dat, metadata_fname, name_mode=name_mode)


def calc_metadata_full_video(frame_list: list, masks_zarr: zarr.Array, video_dat: zarr.Array,
                             metadata_fname: str = None, name_mode='neuron', props_to_save=None) -> dict:
    """
    Calculates metadata once segmentation is finished

    Assume the masks are indexed from 0 and the video is indexed using frame_list

    Parameters
    ----------
    name_mode
    frame_list
    masks_zarr
    video_dat
    metadata_fname
    """
    metadata = dict()

    # Check if inputs are dask arrays
    is_dask = hasattr(masks_zarr, "compute") and hasattr(video_dat, "compute")

    if is_dask:
        logging.info("Using Dask to parallelize metadata computation")
        # If dask, use delayed to parallelize the computation
        def process_metadata(masks, volume, i_vol):
            return (i_vol, get_metadata_dictionary(masks, volume, name_mode=name_mode, props_to_save=props_to_save))

        tasks = []
        for i_mask, i_vol in enumerate(frame_list):
            masks = masks_zarr[i_mask, ...]
            volume = video_dat[i_vol, ...]
            tasks.append(dask.delayed(process_metadata)(masks, volume, i_vol))

        with ProgressBar():
            results = dask.compute(*tasks)

        for i_vol, meta in results:
            metadata[i_vol] = meta
    else:
        logging.info("Using ThreadPoolExecutor to parallelize metadata computation")
        # If not dask, use ThreadPoolExecutor to parallelize the computation
        with tqdm(total=len(frame_list)) as pbar:
            def parallel_func(i_both):
                i_mask, i_vol = i_both
                masks = masks_zarr[i_mask, ...]
                volume = video_dat[i_vol, ...]
                metadata[i_vol] = get_metadata_dictionary(masks, volume, name_mode=name_mode, props_to_save=props_to_save)

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(parallel_func, i): i for i in enumerate(frame_list)}
                for future in concurrent.futures.as_completed(futures):
                    future.result()
                    pbar.update(1)

    # saving metadata and settings
    if metadata_fname is not None:
        with open(metadata_fname, 'wb') as meta_save:
            pickle.dump(metadata, meta_save)

    return metadata
