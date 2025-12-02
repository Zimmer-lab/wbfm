import logging
import os
import threading
from typing import Union

import cv2
import dask.array as da
from skimage.measure import regionprops
from wbfm.utils.external.custom_errors import NoMatchesError
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.general.utils_filenames import add_name_suffix
from wbfm.utils.projects.utils_project_status import check_all_needed_data_for_step
from numcodecs import blosc

import wbfm.utils.segmentation.util.utils_postprocessing as post
import numpy as np
from tqdm import tqdm
# preprocessing
from wbfm.utils.general.video_and_data_conversion.import_video_as_array import get_single_volume
from wbfm.utils.projects.project_config_classes import ModularProjectConfig, ConfigFileWithProjectContext
from wbfm.utils.general.preprocessing.utils_preprocessing import perform_preprocessing
# metadata
from wbfm.utils.segmentation.util.utils_config_files import _unpack_config_file
from wbfm.utils.segmentation.util.utils_metadata import get_metadata_dictionary, calc_metadata_full_video
import zarr
import concurrent.futures


def segment_video_using_config_3d(preprocessing_cfg: ConfigFileWithProjectContext,
                                  segment_cfg: ConfigFileWithProjectContext,
                                  project_cfg: ModularProjectConfig,
                                  continue_from_frame: int =None,
                                  DEBUG: bool = False) -> None:
    """

    Parameters
    ----------
    _config - dict
        Parameters as loaded from a .yaml file. See segment3d.py for documentation
    continue_from_frame - int or None
        For example, if the segmentation crashed, then continue from this frame instead of starting anew

    Returns
    -------
    Saves masks and metadata in the project subfolder 1-segmentation

    """

    (mask_fname, metadata_fname, stardist_model_name, verbose, video_path, _,
     all_bounding_boxes, sum_red_and_green_channels, segment_on_green_channel) = _unpack_config_file(
        preprocessing_cfg, segment_cfg, project_cfg, DEBUG)

    # Open the file
    project_dat = ProjectData.load_final_project_data_from_config(project_cfg)
    red_dat = project_dat.red_data
    video_dat = red_dat

    # Image preprocessing
    if sum_red_and_green_channels or segment_on_green_channel:
        green_dat = project_dat.green_data
        if sum_red_and_green_channels:
            # Cannot directly sum zarr Arrays, so need to convert to dask and lazy sum
            import dask.array as da
            video_dat = da.from_zarr(video_dat) + da.from_zarr(green_dat)
        elif segment_on_green_channel:
            video_dat = green_dat
    num_frames = video_dat.shape[0]
    frame_list = list(range(num_frames))
    logging.info(f"Found video of shape {video_dat.shape}")

    # Other initialization
    sd_model = initialize_stardist_model(stardist_model_name, verbose)
    # For now don't worry about postprocessing the first volume
    opt_postprocessing = segment_cfg.config['postprocessing_params']  # Most options are 2d-only
    # Do first volume outside the parallelization loop to initialize keras and zarr
    masks_zarr = _do_first_volume3d(frame_list, mask_fname, num_frames,
                                    sd_model, verbose, video_dat, all_bounding_boxes, continue_from_frame)
    # Main function
    segmentation_options = {'masks_zarr': masks_zarr, 'opt_postprocessing': opt_postprocessing,
                            'all_bounding_boxes': all_bounding_boxes,
                            'sd_model': sd_model, 'verbose': verbose}

    # Will always be at least continuing after the first frame
    if continue_from_frame is None:
        continue_from_frame = 1
    else:
        continue_from_frame += 1
        project_cfg.logger.info(f"Continuing from frame {continue_from_frame}")

    _segment_full_video_3d(segment_cfg, frame_list, mask_fname, num_frames, verbose, video_dat,
                           segmentation_options, continue_from_frame)

    # Note that metadata is calculated on the red channel only
    calc_metadata_full_video(frame_list, masks_zarr, red_dat, metadata_fname)


def _segment_full_video_3d(segment_cfg: ConfigFileWithProjectContext,
                           frame_list: list, mask_fname: str, num_frames: int, verbose: int,
                           video_dat: zarr.Array,
                           opt: dict, continue_from_frame: int) -> None:
    # Parallel version: threading
    keras_lock = threading.Lock()
    read_lock = threading.Lock()
    opt['keras_lock'] = keras_lock
    opt['read_lock'] = read_lock

    import tensorflow as tf
    is_cuda_gpu_available = tf.test.is_gpu_available(cuda_only=True)

    def parallel_func(_i_both):
        i_out, i_vol = _i_both
        segment_and_save3d(i_out + continue_from_frame, i_vol, video_dat=video_dat, **opt)

    if is_cuda_gpu_available:
        segment_cfg.logger.info("Found cuda! Running single process")

        for i_both in enumerate(tqdm(frame_list[continue_from_frame:])):
            parallel_func(i_both)
    else:
        max_workers = 8
        segment_cfg.logger.info("Did not find cuda, running in multi-threaded mode")

        with tqdm(total=num_frames - continue_from_frame) as pbar:

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(parallel_func, i): i for i in enumerate(frame_list[continue_from_frame:])}
                for future in concurrent.futures.as_completed(futures):
                    future.result()
                    pbar.update(1)

    segment_cfg.update_self_on_disk()
    if verbose >= 1:
        segment_cfg.logger.info(f'Done with segmentation pipeline! Mask data saved at {mask_fname}')

##
## 2d pipeline (stitch to get 3d)
##


def segment_video_using_config_2d(preprocessing_cfg: ConfigFileWithProjectContext,
                                  segment_cfg: ConfigFileWithProjectContext,
                                  project_cfg: ModularProjectConfig,
                                  continue_from_frame: int = None,
                                  DEBUG: bool = False) -> None:
    """
    Full pipeline based on only a config file

    See segment2d.py for parameter documentation
    """

    (mask_fname, metadata_fname, stardist_model_name, verbose, video_path, zero_out_borders,
     all_bounding_boxes, sum_red_and_green_channels) = _unpack_config_file(
        preprocessing_cfg, segment_cfg, project_cfg, DEBUG)

    # Open the file
    project_dat = ProjectData.load_final_project_data_from_config(project_cfg)
    video_dat = project_dat.red_data
    num_frames = video_dat.shape[0]
    frame_list = list(range(num_frames))

    sd_model = initialize_stardist_model(stardist_model_name, verbose)
    # Do first volume outside the parallelization loop to initialize keras and zarr
    opt_postprocessing = segment_cfg.config['postprocessing_params']
    if verbose > 1:
        print("Postprocessing settings: ")
        print(opt_postprocessing)
    # Force BLOSC (compression within zarr) to not be multi-threaded
    # On large jobs (not remote, not dask) I get this error: https://github.com/pangeo-data/pangeo/issues/196
    os.environ["BLOSC_NTHREADS"] = "1"
    # See also: https://zarr.readthedocs.io/en/stable/tutorial.html#tutorial-tips-blosc
    blosc.use_threads = False
    masks_zarr = _do_first_volume2d(frame_list, mask_fname, num_frames,
                                    sd_model, verbose, video_dat, zero_out_borders,
                                    all_bounding_boxes,
                                    continue_from_frame, opt_postprocessing)

    # Main function
    segmentation_options = {'masks_zarr': masks_zarr, 'opt_postprocessing': opt_postprocessing,
                            'sd_model': sd_model, 'verbose': verbose, 'zero_out_borders': zero_out_borders,
                            'all_bounding_boxes': all_bounding_boxes}

    # Will always be at least continuing after the first frame
    if continue_from_frame is None:
        continue_from_frame = 1
    else:
        continue_from_frame += 1
        print(f"Continuing from frame {continue_from_frame}")

    _segment_full_video_2d(segment_cfg, frame_list, mask_fname, num_frames, verbose, video_dat,
                           segmentation_options, continue_from_frame)

    # Same 2d and 3d
    calc_metadata_full_video(frame_list, masks_zarr, video_dat, metadata_fname)


def initialize_stardist_model(stardist_model_name, verbose):
    from wbfm.utils.segmentation.util.utils_model import get_stardist_model
    sd_model = get_stardist_model(stardist_model_name, verbose=verbose - 1)
    # Not fully working for multithreaded scenario
    # Discussion about finalizing: https://stackoverflow.com/questions/40850089/is-keras-thread-safe/43393252#43393252
    # Dicussion about making the predict function: https://github.com/jaromiru/AI-blog/issues/2
    sd_model.keras_model.make_predict_function()
    return sd_model


def _segment_full_video_2d(segment_cfg: ConfigFileWithProjectContext,
                           frame_list: list, mask_fname: str, num_frames: int, verbose: int,
                           video_dat: zarr.Array,
                           opt: dict, continue_from_frame: int) -> None:

    # Parallel version: threading
    keras_lock = threading.Lock()
    read_lock = threading.Lock()
    opt['keras_lock'] = keras_lock
    opt['read_lock'] = read_lock

    with tqdm(total=num_frames - continue_from_frame) as pbar:
        def parallel_func(i_both):
            i_out, i_vol = i_both
            segment_and_save2d(i_out + continue_from_frame, i_vol, video_dat=video_dat, **opt)

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = {executor.submit(parallel_func, i): i for i in enumerate(frame_list[continue_from_frame:])}
            for future in concurrent.futures.as_completed(futures):
                future.result()
                pbar.update(1)

    segment_cfg.update_self_on_disk()
    if verbose >= 1:
        print(f'Done with segmentation pipeline! Mask data saved at {mask_fname}')


def _do_first_volume2d(frame_list: list, mask_fname: str, num_frames: int,
                       sd_model, verbose: int, video_dat: zarr.Array,
                       zero_out_borders: bool,
                       all_bounding_boxes: list = None,
                       continue_from_frame: int = None, opt_postprocessing: dict = None) -> zarr.Array:
    # Do first loop to initialize the zarr data
    if continue_from_frame is None:
        i = 0
        mode = 'w-'
    else:
        i = continue_from_frame
        # Old file MUST exist in this case
        mode = 'r+'
    i_volume = frame_list[i]
    volume = get_volume_using_bbox(all_bounding_boxes, i_volume, video_dat)

    from wbfm.utils.segmentation.util.utils_model import segment_with_stardist_2d
    final_masks = segment_with_stardist_2d(volume, sd_model, zero_out_borders, verbose=verbose - 1)
    _, num_slices, x_sz, y_sz = video_dat.shape
    masks_zarr = _create_or_continue_zarr(mask_fname, num_frames, num_slices, x_sz, y_sz, mode=mode)
    final_masks = perform_post_processing_2d(final_masks,
                                             volume,
                                             **opt_postprocessing,
                                             verbose=verbose - 1)
    save_volume_using_bbox(all_bounding_boxes, final_masks, i, i_volume, masks_zarr)
    return masks_zarr


def get_volume_using_bbox(all_bounding_boxes: dict, i_volume: int, video_dat: Union[zarr.Array, da.Array]) \
        -> np.ndarray:
    if all_bounding_boxes is None:
        raise NotImplementedError("Bounding box not found; this may cause significant artifacts")
        # volume = video_dat[i_volume, ...]
    else:
        bbox = all_bounding_boxes[i_volume]
        if bbox is None or len(bbox) == 0:
            volume = video_dat[i_volume, ...]
        else:
            volume = video_dat[i_volume, :, bbox[0]:bbox[2], bbox[1]:bbox[3]]
    # Check if the volume is a dask object; if so, actually load it
    if isinstance(volume, da.Array):
        volume = volume.compute()
    return volume


def _do_first_volume3d(frame_list: list, mask_fname: str, num_frames: int,
                       sd_model, verbose: int, video_dat: Union[zarr.Array, da.Array],
                       all_bounding_boxes: list = None,
                       continue_from_frame: int = None) -> zarr.Array:
    # Do first loop to initialize the zarr data
    if continue_from_frame is None:
        i = 0
        mode = 'w-'
    else:
        i = continue_from_frame
        # Old file MUST exist in this case
        mode = 'r+'
    i_volume = frame_list[i]
    volume = get_volume_using_bbox(all_bounding_boxes, i_volume, video_dat)

    from wbfm.utils.segmentation.util.utils_model import segment_with_stardist_3d
    final_masks = segment_with_stardist_3d(volume, sd_model, verbose=verbose - 1)
    _, num_slices, x_sz, y_sz = video_dat.shape
    masks_zarr = _create_or_continue_zarr(mask_fname, num_frames, num_slices, x_sz, y_sz, mode=mode)

    save_volume_using_bbox(all_bounding_boxes, final_masks, i, i_volume, masks_zarr)
    return masks_zarr


def _create_or_continue_zarr(output_fname, num_frames, num_slices, x_sz, y_sz, mode='w'):
    """
    Creates a new zarr file of the correct file, or, if it already exists, check for a stopping point

    Parameters
    ----------
    output_fname - path to new file; will crash if that file exists
    num_frames
    num_slices
    x_sz
    y_sz
    mode

    Returns
    -------
    Initialized (with zeros) zarr array

    """
    sz = (num_frames, num_slices, x_sz, y_sz)
    chunks = (1, num_slices, x_sz, y_sz)
    print(f"Opening zarr at: {output_fname}")
    try:
        masks_zarr = zarr.open(output_fname, mode=mode,
                               shape=sz, chunks=chunks, dtype=np.uint16,
                               fill_value=0,
                               synchronizer=zarr.ThreadSynchronizer())
    except zarr.errors.ContainsArrayError as e:
        print("???????????????????????????????????????????????????????")
        print("Array exists; did you mean to pass continue_from_frame?")
        print("???????????????????????????????????????????????????????")
        raise e
    return masks_zarr


def segment_and_save3d(i, i_volume, masks_zarr, opt_postprocessing,
                       all_bounding_boxes,
                       sd_model, verbose, video_dat, keras_lock=None, read_lock=None):
    """
    Segments a single volume using bounding boxes, and saves in an open zarr file

    Parameters
    ----------
    i - the new index, which may be offset from the raw volume index
    i_volume - the index in the original video
    masks_zarr - the output class (zarr)
    all_bounding_boxes - a dictionary of bounding boxes, indexed by i_volume
    sd_model - stardist model object
    verbose
    video_dat - the raw video (zarr)
    keras_lock - a lock object (optional)
    read_lock

    Returns
    -------

    """
    volume = get_volume_using_bbox(all_bounding_boxes, i_volume, video_dat)
    from wbfm.utils.segmentation.util.utils_model import segment_with_stardist_3d
    if keras_lock is None:
        segmented_masks = segment_with_stardist_3d(volume, sd_model, verbose=verbose - 1)
    else:
        with keras_lock:  # Keras is not thread-safe in the end
            segmented_masks = segment_with_stardist_3d(volume, sd_model, verbose=verbose - 1)

    final_masks = perform_post_processing_3d(segmented_masks,
                                             volume,
                                             **opt_postprocessing,
                                             verbose=verbose - 1)
    save_volume_using_bbox(all_bounding_boxes, final_masks, i, i_volume, masks_zarr)


def segment_and_save2d(i, i_volume, masks_zarr, opt_postprocessing,
                       zero_out_borders,
                       all_bounding_boxes,
                       sd_model, verbose, video_dat, keras_lock=None, read_lock=None):
    volume = get_volume_using_bbox(all_bounding_boxes, i_volume, video_dat)
    from wbfm.utils.segmentation.util.utils_model import segment_with_stardist_2d
    if keras_lock is None:
        segmented_masks = segment_with_stardist_2d(volume, sd_model, zero_out_borders, verbose=verbose - 1)
    else:
        with keras_lock:  # Keras is not thread-safe in the end
            segmented_masks = segment_with_stardist_2d(volume, sd_model, zero_out_borders, verbose=verbose - 1)

    final_masks = perform_post_processing_2d(segmented_masks,
                                             volume,
                                             **opt_postprocessing,
                                             verbose=verbose - 1)
    save_volume_using_bbox(all_bounding_boxes, final_masks, i, i_volume, masks_zarr)


def save_volume_using_bbox(all_bounding_boxes, final_masks, i, i_volume, masks_zarr):
    if all_bounding_boxes is None or all_bounding_boxes[i_volume] is None or len(all_bounding_boxes[i_volume]) == 0:
        masks_zarr[i, :, :, :] = final_masks
    else:
        bbox = all_bounding_boxes[i_volume]
        masks_zarr[i, :, bbox[0]:bbox[2], bbox[1]:bbox[3]] = final_masks


def _get_and_prepare_volume(i, num_slices, preprocessing_settings, video_path, read_lock=None):
    # use get single volume function from charlie
    import_opt = {'which_vol': i, 'num_slices': num_slices, 'alpha': 1.0, 'dtype': 'uint16'}
    if read_lock is None:
        volume = get_single_volume(video_path, **import_opt)
    else:
        with read_lock:
            volume = get_single_volume(video_path, **import_opt)
    volume = perform_preprocessing(volume, preprocessing_settings)
    return volume


def save_masks_and_metadata(final_masks, i, i_volume, masks_zarr, metadata, volume):
    # Add masks to zarr file; automatically saves
    masks_zarr[i, :, :, :] = final_masks
    # metadata dictionary; also modified by reference
    meta_df = get_metadata_dictionary(final_masks, volume)
    metadata[i_volume] = meta_df


def perform_post_processing_3d(mask_array: np.ndarray, img_volume: np.ndarray, to_remove_dim_slices=False,
                               max_number_of_objects=None,
                               **kwargs) -> np.ndarray:
    """
    Post processes volumes in 3d. Similar to perform_post_processing_2d, but much simpler
        Note: the function signature is designed to be similar, thus unused args are fine

    Currently, only entirely removing dim objects is supported, based on keeping the brightest num_to_keep

    Parameters
    ----------
    mask_array
    img_volume
    to_remove_dim_slices
    kwargs

    Returns
    -------

    """
    if to_remove_dim_slices and max_number_of_objects is not None:
        props = regionprops(mask_array, intensity_image=img_volume)
        if len(props) > max_number_of_objects:
            all_intensities = np.array([p.intensity_mean for p in props])
            ind_sorted = np.argsort(-all_intensities)
            for i in ind_sorted[max_number_of_objects:]:
                if props[i].label == 0:
                    continue
                # Numpy wants individual lists
                c = props[i].coords
                z, x, y = c[:, 0], c[:, 1], c[:, 2]
                try:
                    mask_array[z, x, y] = 0
                except IndexError:
                    # The above should work, but it's giving me errors
                    logging.debug("Initial indexing failed")
                    try:
                        mask_array[mask_array == props[i].label] = 0
                    except IndexError as e:
                        print(z, x, y)
                        print(mask_array.shape)
                        raise e

    return mask_array


def perform_post_processing_2d(mask_array: np.ndarray, img_volume: np.ndarray, border_width_to_remove,
                               to_remove_border=True,
                               upper_length_threshold=12, lower_length_threshold=3,
                               to_remove_dim_slices=False,
                               stitch_via_watershed=False,
                               min_separation=0,
                               already_stitched=False,
                               also_split_using_centroids=False,
                               verbose=0,
                               DEBUG=False,
                               **kwargs):
    """
    Performs some post-processing steps including: Splitting long neurons, removing short neurons and
    removing too large areas

    Parameters
    ----------
    mask_array : 3D numpy array
        array of segmented masks
    img_volume : 3D numpy array
        array of original image with brightness values
    border_width_to_remove : int
        within that distance to border, artefacts/masks will be removed
    to_remove_border : boolean
        if true, a certain width
    upper_length_threshold : int
        masks longer than this will be (tried to) split
    lower_length_threshold : int
        masks shorter than this will be removed
    to_remove_dim_slices : bool
        Before stitching, removes stardist segments that are too dim
    stitch_via_watershed : bool
        Default is False, which means stitching via bipartite matching and a lot of post-processing
    verbose : int
        flag for print statements. Increasing by 1, increase depth by 1

    Returns
    -------
    final_masks : 3D numpy array
        3D array of masks after post-processing

    """
    if verbose >= 1:
        print(f"Starting preprocessing with {len(np.unique(mask_array)) - 1} neurons")
        print("Note: not yet stitched in z")
    masks = post.remove_large_areas(mask_array, verbose=verbose)
    if to_remove_dim_slices:
        masks = post.remove_dim_slices(masks, img_volume, verbose=verbose)
    if verbose >= 1:
        print(f"After large area removal: {len(np.unique(masks)) - 1}")

    if not stitch_via_watershed:
        if not already_stitched:
            try:
                stitched_masks, intermediates = post.bipartite_stitching(masks, verbose=verbose)
            except NoMatchesError:
                logging.warning("No between-plane matches found; skipping this volume")
                return np.zeros_like(masks)
        else:
            stitched_masks = masks.copy()
        if verbose >= 1:
            print(f"After stitching: {len(np.unique(stitched_masks)) - 1}")
        neuron_lengths = post.get_neuron_lengths_dict(stitched_masks)

        # calculate brightnesses and their global Z-plane
        brightnesses, neuron_planes, neuron_centroids = post.calc_brightness(img_volume, stitched_masks)
        # split too long neurons
        current_global_neuron = len(neuron_lengths)
        split_masks, split_lengths, split_brightnesses, current_global_neuron, split_neuron_planes = \
            post.split_long_neurons(stitched_masks,
                                    neuron_lengths,
                                    brightnesses,
                                    neuron_centroids,
                                    current_global_neuron,
                                    upper_length_threshold,
                                    neuron_planes,
                                    min_separation,
                                    also_split_using_centroids,
                                    verbose=verbose - 1)
        if verbose >= 1:
            print(f"After splitting: {len(np.unique(split_masks)) - 1}")

        final_masks, final_neuron_lengths, final_brightness, final_neuron_planes, removed_neurons_list = \
            post.remove_short_neurons(split_masks,
                                      split_lengths,
                                      lower_length_threshold,
                                      split_brightnesses,
                                      split_neuron_planes)
        if verbose >= 1:
            print(f"After short neuron removal: {len(np.unique(final_masks)) - 1}")
    else:
        if verbose >= 1:
            print("Stitching using watershed")
        final_masks = post.stitch_via_watershed(masks, img_volume)

    if to_remove_border:
        final_masks = post.remove_border(final_masks, border_width_to_remove)

    if verbose >= 1:
        print(f"After border removal: {len(np.unique(final_masks))}")
        print("Postprocessing finished")

    if DEBUG:
        from wbfm.utils.projects.utils_debugging import shelve_full_workspace
        fname = 'stardist_2d_postprocessing.out'
        shelve_full_workspace(fname, list(dir()), locals())

    return final_masks


def resplit_masks_in_z_from_config(preprocessing_cfg: ConfigFileWithProjectContext,
                                   segment_cfg: ConfigFileWithProjectContext,
                                   project_cfg: ModularProjectConfig,
                                   continue_from_frame: int = None, DEBUG=False) -> None:
    """
    Similar to segment_full_video_2d, but this assumes a previously segmented video

    """

    (mask_fname, metadata_fname, _, verbose, video_path, zero_out_borders,
     all_bounding_boxes, sum_red_and_green_channels) = _unpack_config_file(
        preprocessing_cfg, segment_cfg, project_cfg, DEBUG)

    # Get data: needs both segmentation and raw video
    check_all_needed_data_for_step(project_cfg, 2)
    masks_old = zarr.open(mask_fname, mode='r')
    # masks_old = np.array(masks_zarr[:num_frames, ...])  # TEST

    # Do not overwrite old file
    new_fname = str(add_name_suffix(mask_fname, '1'))
    masks_zarr = zarr.open_like(masks_old, path=new_fname)
    video_dat = zarr.open(video_path, mode='r')

    num_frames = video_dat.shape[0]
    frame_list = list(range(num_frames))

    if DEBUG:
        print(video_dat)
        print(f"Opened video at {video_path}")
        print(masks_zarr)
        print(f"Found segmentation at {mask_fname} with shape {masks_old.shape}")

    opt_postprocessing = segment_cfg.config['postprocessing_params']  # Unique to 2d
    opt = {'opt_postprocessing': opt_postprocessing,
           'verbose': verbose,
           'all_bounding_boxes': all_bounding_boxes}
    read_lock = threading.Lock()
    write_lock = threading.Lock()
    opt['read_lock'] = read_lock
    opt['write_lock'] = write_lock
    if continue_from_frame is None:
        # Note that this does NOT have a separate 'do first volume' function
        continue_from_frame = 0

    for i_out, i_vol in enumerate(tqdm(frame_list[continue_from_frame:])):
        _only_postprocess2d(i_out + continue_from_frame, i_vol, video_dat=video_dat,
                            masks_old=masks_old, masks_zarr=masks_zarr, **opt)

    #... GENUINELY NO IDEA WHY THREADS DON'T WORK HERE
    # Actually split
    # with tqdm(total=num_frames - continue_from_frame) as pbar:
    #     def parallel_func(i_both):
    #         i_out, i_vol = i_both
    #         _only_postprocess2d(i_out + continue_from_frame, i_vol, video_dat=video_dat, masks_zarr=masks_zarr,
    #                             masks_old=masks_old, **opt)
    #
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    #         futures = {executor.submit(parallel_func, i): i for i in enumerate(frame_list[continue_from_frame:])}
    #         for future in concurrent.futures.as_completed(futures):
    #             future.result()
    #             pbar.update(1)

    # Update metadata
    segment_cfg.update_self_on_disk()
    if verbose >= 1:
        print(f'Done with segmentation pipeline! Mask data saved at {mask_fname}')

    # Same 2d and 3d
    calc_metadata_full_video(frame_list, masks_zarr, video_dat, metadata_fname)


def _only_postprocess2d(i, i_volume, masks_zarr, opt_postprocessing,
                       all_bounding_boxes, read_lock, write_lock, masks_old,
                       verbose, video_dat):
    volume = get_volume_using_bbox(all_bounding_boxes, i_volume, video_dat)
    # Read mask directly from previously segmented volume, but copy it
    with read_lock:
        segmented_masks = np.array(masks_old[i_volume, :, :, :])
        # segmented_masks = np.array(masks_zarr[i, :, :, :])
    if verbose >= 1:
        print(f"Analyzing {i}, {i_volume}")
        if verbose >= 3:
            print(f"Mean values: {np.mean(segmented_masks)}, {np.mean(volume)}")
    final_masks = perform_post_processing_2d(segmented_masks,
                                             volume,
                                             **opt_postprocessing,
                                             verbose=verbose - 1)
    with read_lock:
        save_volume_using_bbox(all_bounding_boxes, final_masks, i_volume, i_volume, masks_zarr)


#gaussian blur functions

def gaussian_blur_volume(volume, kernel=(5, 5)):

    """ takes volume """
    restored = volume.copy()
    for z in tqdm(range(volume.shape[0])):
        restored[z, :, :] = cv2.GaussianBlur(volume[z, :, :], kernel, 0)

    return restored


def gaussian_blur_video(video, fname, kernel=(5, 5)):
    """takes video"""

    restored_video = _create_or_continue_zarr(fname + ".zarr", num_frames=video.shape[0], num_slices=video.shape[1],
                                              x_sz=video.shape[2], y_sz=video.shape[3], mode='w-')

    for i in tqdm(range(video.shape[0])):
        volume = gaussian_blur_volume(video[i, :, :], kernel=kernel)
        restored_video[i, :, :, :] = volume

    return restored_video


def gaussian_blur_using_config(project_cfg, fname_for_saving_red, fname_for_saving_green, kernel=(5, 5)):
    """takes config file"""
    # Open the file
    project_dat = ProjectData.load_final_project_data_from_config(project_cfg)
    video_dat_red = project_dat.red_data
    video_dat_green = project_dat.green_data

    gaussian_blur_video(video_dat_red, fname=fname_for_saving_red, kernel=kernel)
    gaussian_blur_video(video_dat_green, fname=fname_for_saving_green, kernel=kernel)