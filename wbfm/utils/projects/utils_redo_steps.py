import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from traitlets import default
import zarr

from wbfm.utils.external.utils_pandas import cast_int_or_nan
from wbfm.utils.neuron_matching.utils_matching import calc_bipartite_from_positions
from wbfm.utils.projects.finished_project_data import ProjectData
from skimage.measure import regionprops
from tqdm.auto import tqdm
from wbfm.utils.segmentation.util.utils_metadata import DetectedNeurons

from wbfm.utils.projects.project_config_classes import ModularProjectConfig
from wbfm.utils.general.utils_filenames import add_name_suffix
from wbfm.utils.general.high_performance_pandas import get_names_from_df


def remap_tracklets_to_new_segmentation(project_data: ProjectData,
                                        new_segmentation_suffix,
                                        path_to_new_segmentation,
                                        path_to_new_metadata,
                                        DEBUG=False):

    if path_to_new_segmentation is None and new_segmentation_suffix is not None:
        seg_cfg = project_data.project_config.get_segmentation_config()
        old_seg_name = seg_cfg.config['output_masks']
        old_meta_name = seg_cfg.config['output_metadata']
        path_to_new_segmentation = add_name_suffix(old_seg_name, new_segmentation_suffix)
        path_to_new_metadata = add_name_suffix(old_meta_name, new_segmentation_suffix)

    new_meta = DetectedNeurons(path_to_new_metadata)
    print(new_meta)
    old_seg = project_data.raw_segmentation
    new_seg = zarr.open(path_to_new_segmentation)
    red = project_data.red_data
    num_frames = project_data.num_frames
    old_df = project_data.df_all_tracklets
    try:
        logging.info("Converting dataframe to dense form, may take a while")
        if DEBUG:
            new_df = old_df[:5].sparse.to_dense()
        else:
            new_df = old_df.sparse.to_dense()
    except AttributeError:
        new_df = old_df.copy()
    names = get_names_from_df(old_df)

    if DEBUG:
        num_frames = 5

    all_old2new_idx, all_old2new_labels = match_two_segmentations(new_seg, num_frames, old_seg, red)

    logging.info("Updating tracklet-segmentation indices using mapping")
    col_name1 = 'raw_segmentation_id'
    col_name2 = 'raw_neuron_ind_in_list'
    for n in tqdm(names):
        tracklet = old_df[n].dropna(axis=0)
        ind, new_col1, new_col2 = get_new_column_values_using_mapping(all_old2new_idx, all_old2new_labels, col_name1,
                                                                      col_name2, num_frames, tracklet)

        new_df.loc[ind, (n, col_name1)] = new_col1
        new_df.loc[ind, (n, col_name2)] = new_col2

    # Note; this loop could be combined with above if needed
    logging.info("Updating metadata using new indices")
    cols_to_replace = ['z', 'x', 'y', 'brightness_red', 'volume']
    for n in tqdm(names):
        new_columns = defaultdict(list)
        tracklet = new_df[n].dropna(axis=0)
        ind = tracklet.index
        ind = ind[ind < num_frames]

        for t in ind:
            mask_ind = tracklet.at[t, 'raw_segmentation_id']
            row_data, column_names = new_meta.get_all_metadata_for_single_time(mask_ind, t, None)

            # Only need certain columns
            for val, col_name in zip(row_data, column_names):
                if col_name in cols_to_replace:
                    new_columns[col_name].append(val)
        # Update all columns at once
        for col_name in cols_to_replace:
            new_df.loc[ind, (n, col_name)] = new_columns[col_name]

    _save_new_tracklets_and_update_config_file(new_df, path_to_new_metadata, path_to_new_segmentation, project_data,
                                               DEBUG)

    return new_df, all_old2new_idx, all_old2new_labels


def _save_new_tracklets_and_update_config_file(new_df, path_to_new_metadata, path_to_new_segmentation, project_data,
                                               DEBUG=False):
    # Save
    logging.info(f"Saving with debug mode: {DEBUG}")
    track_cfg = project_data.project_config.get_tracking_config()
    df_to_save = new_df.astype(pd.SparseDtype("float", np.nan))
    output_df_fname = os.path.join('3-tracking', 'postprocessing', 'df_resegmented.pickle')
    track_cfg.pickle_data_in_local_project(df_to_save, output_df_fname, custom_writer=pd.to_pickle)
    # logging.warning("Overwriting name of manual correction tracklets, assuming that was the most recent")
    df_fname = track_cfg.unresolve_absolute_path(output_df_fname)
    track_cfg.config.update({'manual_correction_tracklets_df_fname': df_fname})
    if not DEBUG:
        track_cfg.update_self_on_disk()
    segmentation_cfg = project_data.project_config.get_segmentation_config()
    fname = segmentation_cfg.unresolve_absolute_path(path_to_new_segmentation)
    segmentation_cfg.config['output_masks'] = fname
    fname = segmentation_cfg.unresolve_absolute_path(path_to_new_metadata)
    segmentation_cfg.config['output_metadata'] = fname
    if not DEBUG:
        segmentation_cfg.update_self_on_disk()


def match_two_segmentations(new_seg, num_frames, old_seg, red):
    logging.info("Create mapping from old to new segmentation")
    all_old2new_idx = {}
    all_old2new_labels = {}
    for t in tqdm(range(num_frames)):
        this_img = red[t]

        new_centroids, new_labels = _get_props(new_seg[t], this_img)
        old_centroids, old_labels = _get_props(old_seg[t], this_img)

        new_c_array = np.array(list(new_centroids.values()))
        old_c_array = np.array(list(old_centroids.values()))

        old2new_idx, conf, _ = calc_bipartite_from_positions(old_c_array, new_c_array)
        old2new_labels = {old_labels[i1]: new_labels[i2] for i1, i2 in old2new_idx}

        all_old2new_idx[t] = dict(old2new_idx)
        all_old2new_labels[t] = old2new_labels
    return all_old2new_idx, all_old2new_labels


def get_new_column_values_using_mapping(all_old2new_idx, all_old2new_labels, col_name1, col_name2, num_frames,
                                        tracklet):
    old_col1 = tracklet[col_name1]
    old_col2 = tracklet[col_name2]
    ind = old_col1.index
    ind = ind[ind < num_frames]
    new_col1 = []
    new_col2 = []
    for t in ind:
        if t >= num_frames:
            break
        new_col1.append(all_old2new_labels[t].get(int(old_col1[t]), np.nan))
        new_col2.append(all_old2new_idx[t].get(int(old_col2[t]), np.nan))
    return ind, new_col1, new_col2


def _get_props(this_seg, this_img=None):
    props = regionprops(this_seg.copy(), intensity_image=this_img.copy())
    centroids = {}
    labels = {}
    for i, p in enumerate(props):
        centroids[i] = p.weighted_centroid
        labels[i] = p.label
    return centroids, labels


def remap_tracklets_to_new_segmentation_using_config(project_path: str,
                                                     new_segmentation_suffix=None,
                                                     path_to_new_segmentation=None,
                                                     path_to_new_metadata=None,
                                                     DEBUG=False):
    project_data = ProjectData.load_final_project_data_from_config(project_path)

    remap_tracklets_to_new_segmentation(project_data,
                                        new_segmentation_suffix,
                                        path_to_new_segmentation,
                                        path_to_new_metadata,
                                        DEBUG)


def correct_tracks_dataframe_using_frame_class(project_cfg: ModularProjectConfig, overwrite=False):
    """
    Just checks to make sure all of the IDs in the dataframe are within the number of physical neurons, nothing else

    Similar to: build_ground_truth_neuron_feature_spaces
    """
    if overwrite:
        logging.warning("Using frame objects to correct dataframe... "
                        "you must be sure that the frame objects are not invalid!")

    project_data = ProjectData.load_final_project_data_from_config(project_cfg, to_load_frames=True)
    return correct_tracks_dataframe_using_project(project_data, overwrite)


def correct_tracks_dataframe_using_project(project_data: ProjectData, overwrite: bool, actually_save: bool = True):
    all_frames = project_data.raw_frames
    df = project_data.final_tracks
    df_fname = project_data.final_tracks_fname
    num_frames = df.shape[0]
    neurons = get_names_from_df(df)
    updated_neurons_and_times = defaultdict(list)
    # Get stored feature spaces from Frame objects
    for neuron in tqdm(neurons):
        this_neuron = df[neuron]

        for t in range(num_frames):
            ind_within_frame = this_neuron['raw_neuron_ind_in_list'][t]
            if np.isnan(ind_within_frame):
                continue
            else:
                ind_within_frame = int(ind_within_frame)
            frame = all_frames[t]

            if ind_within_frame >= frame.all_features.shape[0]:
                updated_neurons_and_times[neuron].append(t)
                logging.warning(f"Neuron not found within frame; deleting {neuron} at t={t}")
                df.loc[t, neuron] = np.nan
                # insert_value_in_sparse_df(df, index=t, columns=neuron, val=np.nan)
    # Save
    if len(updated_neurons_and_times) > 0 and actually_save:
        tracking_cfg = project_data.project_config.get_tracking_config()
        tracking_cfg.save_data_in_local_project(df, df_fname, allow_overwrite=overwrite,
                                                make_sequential_filename=~overwrite)
    else:
        print("No updates needed")
    return df


def add_metadata_to_df_raw_ind(df_raw_ind, segmentation_metadata: DetectedNeurons,
                               raise_error=True):
    """
    Given a dataframe with only the raw_ind_in_list, add metadata to it

    Expect that df_raw_ind has a multiindex column, with the top level as neuron names and the final column names should be:
        brightness_red 	likelihood 	raw_neuron_ind_in_list 	raw_segmentation_id 	volume 	x 	y 	z 	raw_tracklet_id

    Parameters
    ----------
    df_raw_ind
    segmentation_metadata

    Returns
    -------

    """
    # Each column will be the same length as df_raw_ind
    def make_new_col():
        col = np.zeros(df_raw_ind.shape[0])
        col[:] = np.nan
        return col
    new_df_values = defaultdict(make_new_col)

    # Iterate over each column (slightly slow but better than messing with column names and indices)
    top_level_names = df_raw_ind.columns.get_level_values(0).unique()
    for neuron_name in tqdm(top_level_names, leave=False):
        this_col = df_raw_ind.loc[:, (neuron_name, 'raw_neuron_ind_in_list')]
        try:
            likelihood = df_raw_ind.loc[:, (neuron_name, 'likelihood')]
        except KeyError:
            likelihood = np.ones(t)

        for t in range(len(this_col)):
            raw_ind = cast_int_or_nan(this_col.iat[t])
            if np.isnan(raw_ind):
                continue
            try:
                mask_ind = segmentation_metadata.i_in_array_to_mask_index(t, raw_ind)
            except IndexError as e:
                print(f"Index error for neuron {neuron_name} at t={t}, raw_ind={raw_ind}, "
                      f"with detected number of objects {len(segmentation_metadata.segmentation_metadata[t])}")
                if raise_error:
                    raise e
                continue
            row_data, column_names = segmentation_metadata.get_all_metadata_for_single_time(mask_ind, t, likelihood=likelihood[t])
            for val, col_name in zip(row_data, column_names):
                key = (neuron_name, col_name)
                new_df_values[key][t] = val

    # Now, convert to a dataframe
    new_df = pd.DataFrame(new_df_values)
    return new_df


def combine_metadata_from_two_dataframes(df_raw_ind, df_with_metadata, column_to_match='raw_neuron_ind_in_list', raise_error=True,
                                         DEBUG=False):
    """
    Given a dataframe with only the raw_ind_in_list, add metadata to it

    Expect that df_raw_ind has a multiindex column, with the top level as neuron names and the final column names should be at least:
        raw_neuron_ind_in_list 	raw_segmentation_id  	x 	y 	z

    Parameters
    ----------
    df_raw_ind
    segmentation_metadata

    Returns
    -------

    """
    # Prepare result DataFrame (copy to avoid modifying input)
    def make_new_col():
        col = np.zeros(df_raw_ind.shape[0])
        col[:] = np.nan
        return col
    dict_result = defaultdict(make_new_col)

    # Get unique neuron names from both frames
    neurons_raw = df_raw_ind.columns.get_level_values(0).unique()

    # Loop 1: over each neuron in df_raw_ind
    for neuron in tqdm(neurons_raw, desc="Combining metadata per neuron"):
        if column_to_match not in df_raw_ind[neuron].columns:
            if raise_error:
                raise ValueError(f"Column '{column_to_match}' not found in neuron '{neuron}' of df_raw_ind")
            else:
                print(f"Warning: Column '{column_to_match}' not found in neuron '{neuron}' of df_raw_ind, skipping")
                continue
        nonnan_times = df_raw_ind.loc[:, (neuron, column_to_match)].dropna().index.astype(int)

        for t in nonnan_times:
        # Loop 2: over each non-nan time point in df_raw_ind
            this_row = df_raw_ind.loc[t, neuron]
            match_value = this_row[column_to_match]

            # Find matching row in df_with_metadata for this neuron, across top-level objects
            _df = df_with_metadata.loc[t, (slice(None), column_to_match)]
            original_match = _df.index.get_level_values(0)[(_df == match_value).values][0]
            original_row = df_with_metadata.loc[t, original_match]

            # Generate new column names and values
            for col in this_row.index:
                if col == column_to_match:
                    continue
                dict_result[(neuron, col)][t] = this_row[col]
            for col in original_row.index:
                dict_result[(neuron, col)][t] = original_row[col]
        if DEBUG:
            print(this_row.index)
            print(original_row.index)
            break

    df_result = pd.DataFrame(dict_result)
    return df_result
