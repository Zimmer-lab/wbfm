import concurrent.futures
import logging
import os
from collections import defaultdict
from pathlib import Path
from re import A

import numpy as np
import pandas as pd
import zarr
from skimage.measure import regionprops

from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.external.utils_neuron_names import int2name_neuron
from wbfm.utils.general.high_performance_pandas import get_names_from_df
from wbfm.utils.external.custom_errors import NoMatchesError
from tqdm.auto import tqdm

from wbfm.utils.projects.project_config_classes import SubfolderConfigFile, ModularProjectConfig
from wbfm.utils.general.utils_filenames import pickle_load_binary
from wbfm.utils.projects.utils_project import safe_cd
from wbfm.utils.tracklets.training_data_from_tracklets import build_subset_df_from_tracklets, \
    get_or_recalculate_which_frames, _unpack_config_training_data_conversion


def reindex_segmentation(DEBUG, all_matches, seg_masks, new_masks, min_confidence, max_workers=4):
    all_lut = all_matches_to_lookup_tables(all_matches, min_confidence=min_confidence)
    all_lut_keys = all_lut.keys()
    if DEBUG:
        all_lut_keys = [0, 1]
        print("DEBUG mode: only doing first 2 volumes")
    # Apply lookup tables to each volume
    with tqdm(total=len(all_lut), desc="Reindexing segmentation") as pbar:
        def parallel_func(i):
            lut = all_lut[i]
            try:
                new_masks[i, ...] = lut[seg_masks[i, ...]]
            except IndexError as e:
                logging.error(f"IndexError for volume {i}: {e}; removing segmentation for that image")
                # If the index is out of bounds, then probably the image is corrupted
                new_masks[i, ...] = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_results = {executor.submit(parallel_func, i): i for i in all_lut_keys}
            for future in concurrent.futures.as_completed(future_results):
                # _ = future_results[future]
                _ = future.result()
                pbar.update(1)

    return

def _unpack_config_reindexing(traces_cfg, raw_seg_masks, project_cfg):

    relative_path = traces_cfg.config['reindexed_masks']
    out_fname = project_cfg.resolve_relative_path(relative_path)
    if str(out_fname).endswith('.zip'):
        # Then it was already zipped, and should be written normally for this first step
        out_fname = str(Path(out_fname).with_suffix(''))
    project_cfg.logger.info(f"Saving masks at {out_fname}")
    # Check if the raw_seg_masks are zarr or dask
    try:
        # If it is a zarr or numpy array, then we can just open it like this
        chunks = (1, ) + raw_seg_masks.shape[1:]
        new_masks = zarr.open_like(raw_seg_masks, chunks=chunks, path=str(out_fname))
        assert new_masks.chunks[0] == 1, "Chunking must be (1, ..., ...) for safe parallel writes"
    except (AttributeError, TypeError):
        # Otherwise we need to copy the metadata manually
        project_cfg.logger.info(f"Raw segmentation masks are not zarr, but {type(raw_seg_masks)}; creating new zarr array")
        # This is the case for dask arrays
        chunks = list(raw_seg_masks.chunksize)
        # Replace the first element with 1, i.e. one chunk per time slice
        chunks = tuple([1] + chunks[1:])
        new_masks = zarr.open(
            str(out_fname),
            shape=raw_seg_masks.shape,
            dtype=raw_seg_masks.dtype,
            chunks=chunks,
            mode='w'
        )

    # Get tracking (dataframe) with neuron names
    matches_fname = traces_cfg.resolve_relative_path_from_config('all_matches')
    all_matches = pd.read_pickle(matches_fname)
    # Format: dict with i_volume -> Nx3 array of [dlc_ind, segmentation_ind, confidence] triplets

    min_confidence = traces_cfg.config['traces']['min_confidence']

    return all_matches, new_masks, min_confidence, out_fname


def create_spherical_segmentation(this_config, sphere_radius, DEBUG=False):
    """
    Creates a new psuedo-segmentation, which is just a sphere centered on the tracking point
    """
    track_cfg = this_config['track_cfg']
    seg_cfg = this_config['segment_cfg']

    # Required if using in multiple processes
    # from zarr import blosc
    # blosc.use_threads = False

    with safe_cd(Path(this_config['project_path']).parent):
        # Get original segmentation, just for shaping
        seg_fname = seg_cfg['output_masks']
        seg_masks = zarr.open(seg_fname)

        # Initialize the masks at 0
        out_fname = os.path.join("3-tracking", "segmentation_from_tracking.zarr")
        print(f"Saving masks at {out_fname}")
        new_masks = zarr.open_like(seg_masks, path=out_fname,
                                   synchronizer=zarr.ThreadSynchronizer())
        mask_sz = new_masks.shape

        # Get the 3d DLC tracks
        df_fname = track_cfg['final_3d_tracks_df']
        df = pd.read_hdf(df_fname)

    neuron_names = df.columns.levels[0]
    num_frames = mask_sz[0]
    chunk_sz = new_masks.chunks

    # Generate spheres for each neuron, for all time
    cube_sz = [2, 4, 4]

    def get_clipped_sizes(this_sz, sz, total_sz):
        lower_dim = int(np.clip(this_sz - sz, a_min=0, a_max=total_sz))
        upper_dim = int(np.clip(this_sz + sz + 1, a_max=total_sz, a_min=0))
        return lower_dim, upper_dim

    def parallel_func(i_time: int, ind_neuron: int, this_df: pd.DataFrame):
        # X=col, Y=row
        z, col, row = [int(this_df['z'][i_time]), int(this_df['x'][i_time]), int(this_df['y'][i_time])]
        # Instead do a cube (just for visualization)
        z0, z1 = get_clipped_sizes(z, cube_sz[0], chunk_sz[1])
        row0, row1 = get_clipped_sizes(row, cube_sz[1], chunk_sz[2])
        col0, col1 = get_clipped_sizes(col, cube_sz[2], chunk_sz[3])
        new_masks[i_time, z0:z1, row0:row1, col0:col1] = ind_neuron + 1  # Skip 0

    for ind_neuron, neuron in tqdm(enumerate(neuron_names), total=len(neuron_names)):
        # for i in tqdm(range(num_frames), total=num_frames, leave=False):
        #     parallel_func(i, this_df=df[neuron])

        with tqdm(total=num_frames, leave=False) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
                future_results = {executor.submit(parallel_func, i, ind_neuron=ind_neuron, this_df=df[neuron]): i for i
                                  in range(num_frames)}
                for future in concurrent.futures.as_completed(future_results):
                    _ = future.result()
                    pbar.update(1)
        if DEBUG:
            break

    # cube_sz = [3, 7, 7]

    # def create_cube(i_time, ind_neuron, neuron):
    #     # Inner loop: one time and one neuron
    #     this_df = df[neuron]
    #     # FLIP XY
    #     z, y, x = [int(this_df['z'][i_time]), int(this_df['x'][i_time]), int(this_df['y'][i_time])]
    #     # Instead do a cube (just for visualization)
    #     z, x, y = get_crop_coords3d([z, x, y], cube_sz, chunk_sz)
    #     new_masks[i_time, z[0]:z[-1]+1, x[0]:x[-1]+1, y[0]:y[-1]+1] = ind_neuron
    #
    # def process_single_volume(i_time):
    #     # Outer loop: process one full volume
    #     for i_neuron, neuron in enumerate(neuron_names):
    #         create_cube(i_time, i_neuron, neuron)
    #
    # # Instead do a process pool that finishes one file at a time
    # # NOTE: blosc can hang when doing the multiprocesssing :(
    # with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
    #     with tqdm(total=num_frames) as progress:
    #         futures = []
    #
    #         for t in range(num_frames):
    #             future = pool.submit(process_single_volume, t)
    #             future.add_done_callback(lambda p: progress.update())
    #             futures.append(future)
    #
    #         results = []
    #         for future in futures:
    #             result = future.result()
    #             results.append(result)
    #
    # # Reset to automatic detection
    # blosc.use_threads = None


def all_matches_to_lookup_tables(all_matches: dict, min_confidence=None) -> dict:
    """
    Convert a dictionary of match arrays into a lookup table

    Match format:
        all_matches[i_volume] = [[new_ind, old_ind],...]
        Shape: Nx2+ (can be >2)

    Output usage:
        lut = all_lut[i]
        new_masks[i, ...] = lut[old_masks[i, ...]]
    """
    # Convert dataframe to lookup tables, per volume
    # Note: if not all neurons are in the dataframe, then they are set to 0
    all_lut = defaultdict(np.array)
    lut_size = 1000
    # lut_size = max([max(np.array(m)[:, 1]) for m in all_matches.values()]) + 1
    for i_volume, match in all_matches.items():
        lut = np.zeros(lut_size, dtype=int)  # TODO: Should be more than the maximum local index
        try:
            if len(match) == 0:
                raise NoMatchesError
            # TODO: are the matches always the same length?
            match = np.array(match)
            dlc_ind = match[:, 0].astype(int)
            seg_ind = match[:, 1].astype(int)
            if match.shape[1] > 2:
                conf = match[:, 2]
            else:
                conf = np.ones_like(dlc_ind)
                if min_confidence is not None:
                    logging.warning("Confidence threshold passed, but confidence isn't saved; skipping")
            for dlc, seg, c in zip(dlc_ind, seg_ind, conf):
                # Raw indices of the lut should match the local index
                if min_confidence is None or c > min_confidence:
                    lut[seg] = dlc
                    # Otherwise keep as 0
            # lut[seg_ind] = dlc_ind
            if np.max(seg_ind) > lut_size:
                raise ValueError("Lookup-table size is too small; increase this (in code) or fix it!")
        except NoMatchesError:
            # Some volumes may be empty
            pass
        all_lut[i_volume] = lut
    return all_lut


def reindex_segmentation_only_training_data(cfg: ModularProjectConfig,
                                            segment_cfg: SubfolderConfigFile,
                                            training_cfg: SubfolderConfigFile,
                                            keep_raw_segmentation_index=True,
                                            add_one_to_raw_tracklet_index=True,
                                            DEBUG=False):
    """
    Using tracklets and full segmentation, produces a small video (zarr) with neurons colored by track

    Note: the tracklet indices will NOT be the same as the original dataframe
    ... but they will be the same as the segmentation
    """
    if not add_one_to_raw_tracklet_index:
        raise NotImplementedError("Currently, 1 must be added to the index")

    logging.info("Reindexing segmentation (only training volumes)")

    num_frames = cfg.config['dataset_params']['num_frames']

    df_tracklets, df_clust, min_length_to_save, segmentation_metadata = _unpack_config_training_data_conversion(
        training_cfg, segment_cfg)

    # Get ALL matches to the segmentation, then subset
    with safe_cd(cfg.project_dir):

        # Get the frames chosen as training data, or recalculate
        which_frames = get_or_recalculate_which_frames(DEBUG, df_clust, num_frames, training_cfg)

        # Build a sub-df with only the relevant neurons; all time slices
        subset_df = build_subset_df_from_tracklets(df_tracklets, which_frames)

        # TODO: refactor using DetectedNeurons class
        fname = segment_cfg.resolve_relative_path_from_config('output_metadata')
        segmentation_metadata = pickle_load_binary(fname)

        fname = segment_cfg.resolve_relative_path_from_config('output_masks')
        masks = zarr.open(fname)

    logging.info("Convert dataframe to matches per frame")
    # NOTE: only works with updated tracklet dataframe
    tracklet_names = get_names_from_df(subset_df)

    all_matches = {}
    for t in which_frames:
        matches = []
        for i, name in enumerate(tracklet_names):
            neuron_df = subset_df[name]
            raw_neuron_id = neuron_df['raw_neuron_ind_in_list'].at[t]
            if keep_raw_segmentation_index:
                # Do keep the (very large) index from the tracklet df
                # BUT, this can't be 0 because it is the same as the segmentation index (background is 0)
                global_ind = raw_neuron_id + 1
            else:
                # These will NOT be the final names of the neurons if utils_fdnc is used
                global_ind = i + 1
            matches.append([global_ind, int(raw_neuron_id)])
        all_matches[t] = matches

    # all_matches = {}
    # for i, i_frame in tqdm(enumerate(which_frames)):
    #     matches = []
    #     for i_row, neuron_df in subset_df.iterrows():
    #         # i_tracklet = neuron_df['all_ind_local'][i].astype(int)
    #         i_tracklet = int(neuron_df['all_ind_local'][i])
    #         seg_ind = segmentation_metadata[i_frame].index[i_tracklet].astype(int)
    #         if keep_raw_segmentation_index:
    #             # Do keep the (very large) index from the tracklet df
    #             # BUT, this can't be 0 because it is the same as the segmentation index (background is 0)
    #             if add_one_to_raw_tracklet_index:
    #                 global_ind = neuron_df['clust_ind'] + 1
    #             else:
    #                 raise NotImplementedError("Currently, 1 must be added")
    #         else:
    #             # These will NOT be the final names of the neurons if utils_fdnc is used
    #             global_ind = i_row + 1
    #         matches.append([global_ind, seg_ind])
    #     all_matches[i_frame] = np.array(matches)

    # Reindex using look-up table
    all_lut = all_matches_to_lookup_tables(all_matches)

    # Initialize new array
    new_sz = list(masks.shape)
    new_sz[0] = len(which_frames)
    out_fname = os.path.join('2-training_data', 'reindexed_masks.zarr')
    out_fname = cfg.resolve_relative_path(out_fname)
    new_masks = zarr.open_like(masks, path=out_fname, shape=new_sz)

    logging.info("Reindexing segmentation and writing to disk")
    for i, (i_volume, lut) in tqdm(enumerate(all_lut.items())):
        new_masks[i, ...] = lut[masks[i_volume, ...]]

    # Automatically saves


def extract_list_of_pixel_values_from_config(project_path: str):
    # Format:
    # Dict of dict of list; calculate using regionprops
    project_data = ProjectData.load_final_project_data_from_config(project_path)

    dict_of_dict_of_vals_red = {}
    dict_of_dict_of_vals_green = {}

    for t in tqdm(range(project_data.num_frames)):
        vol_red, masks = project_data.red_data[t], project_data.segmentation[t]
        vol_green = project_data.green_data[t]

        dict_for_this_time_red = {}
        dict_for_this_time_green = {}

        props_red = regionprops(masks, intensity_image=vol_red)
        props_green = regionprops(masks, intensity_image=vol_green)

        for prop in props_red:
            vol_of_values = prop['intensity_image']
            label = prop['label']
            dict_for_this_time_red[int2name_neuron(label)] = vol_of_values[vol_of_values > 0]

        for prop in props_green:
            vol_of_values = prop['intensity_image']
            label = prop['label']
            dict_for_this_time_green[int2name_neuron(label)] = vol_of_values[vol_of_values > 0]

        dict_of_dict_of_vals_red[t] = dict_for_this_time_red
        dict_of_dict_of_vals_green[t] = dict_for_this_time_green

    # Save
    config = project_data.project_config
    Path(config.resolve_relative_path('visualization')).mkdir(exist_ok=True)

    fname_red = os.path.join('visualization', 'pixel_values_all_neurons_red.pickle')
    config.pickle_data_in_local_project(dict_of_dict_of_vals_red, fname_red)

    fname_green = os.path.join('visualization', 'pixel_values_all_neurons_green.pickle')
    config.pickle_data_in_local_project(dict_of_dict_of_vals_green, fname_green)

    return dict_of_dict_of_vals_red, dict_of_dict_of_vals_green
