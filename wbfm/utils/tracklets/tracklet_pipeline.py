import logging
import os
import os.path as osp
from collections import defaultdict
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from wbfm.utils.neuron_matching.class_frame_pair import FramePair, FramePairOptions
from wbfm.utils.nn_utils.superglue import SuperGlueUnpackerWithTemplate
from wbfm.utils.nn_utils.worm_with_classifier import PostprocessedFeatureSpaceTemplateMatcher, SuperGlueFullVideoTrackerWithTemplate
from wbfm.utils.neuron_matching.utils_candidate_matches import fit_umap_using_frames

from wbfm.utils.neuron_matching.feature_pipeline import match_all_adjacent_frames
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.external.utils_neuron_names import name2int_neuron_and_tracklet, int2name_tracklet
from wbfm.utils.general.high_performance_pandas import delete_tracklets_using_ground_truth, PaddedDataFrame, \
    get_names_from_df, check_if_heterogenous_columns, get_next_name_generator, split_multiple_tracklets
from wbfm.utils.segmentation.util.utils_metadata import DetectedNeurons
from wbfm.utils.tracklets.tracklet_class import TrackedWorm
from wbfm.utils.tracklets.utils_tracklets import build_tracklets_dfs, \
    remove_tracklets_from_dictionary_without_database_match
from wbfm.utils.projects.project_config_classes import ModularProjectConfig, SubfolderConfigFile
from wbfm.utils.general.utils_filenames import pickle_load_binary
from wbfm.utils.projects.utils_project import safe_cd
from wbfm.utils.general.utils_hardcoded import load_hardcoded_neural_network_paths

###
### For use with produces tracklets (step 2 of traces)
###
from tqdm.auto import tqdm


def match_all_adjacent_frames_using_config(project_config: ModularProjectConfig,
                                           training_config: SubfolderConfigFile,
                                           DEBUG: bool = False) -> None:
    """
    Main pipeline step after the frames exist, but not the matches

    Can also be used if the matches are corrupted or need to be redone
    """

    project_data = ProjectData.load_final_project_data_from_config(project_config)
    project_config.logger.info(f"Matching frames (pairwise)")

    # Load the previous step
    all_frame_dict = project_data.raw_frames

    # Intermediate products: pairwise matches between frames
    _, tracker_params, frame_pair_options, _ = _unpack_config_frame2frame_matches(project_data, training_config,
                                                                                  DEBUG)
    start_volume = tracker_params['start_volume']
    end_volume = start_volume + tracker_params['num_frames']
    project_config.logger.info(f"Calculating Frame pairs for frames: {start_volume + 1} to {end_volume}")

    if frame_pair_options.use_superglue:
        all_frame_pairs = build_frame_pairs_using_superglue(all_frame_dict, frame_pair_options, project_data)
    else:
        # Use UMAP embedding by default
        umap = fit_umap_using_frames(all_frame_dict)
        all_matcher_dict = {k: PostprocessedFeatureSpaceTemplateMatcher(template_frame=v, postprocesser=umap.transform) for k, v in all_frame_dict.items()}
        all_frame_pairs = match_all_adjacent_frames(all_matcher_dict, start_volume, end_volume, frame_pair_options, use_tracker_class=True)

    with safe_cd(project_config.project_dir):
        _save_matches_and_frames(all_frame_dict, all_frame_pairs, training_config)


def build_frame_pairs_using_superglue(all_frame_dict, frame_pair_options, project_data,
                                      match_using_additional_methods=True):
    # Load hardcoded path to model
    path_dict = load_hardcoded_neural_network_paths()
    superglue_parent_folder = path_dict['tracking_paths']['model_parent_folder']
    superglue_model_name = path_dict['tracking_paths']['tracklet_model_name']
    superglue_path = os.path.join(superglue_parent_folder, superglue_model_name)

    if os.path.exists(superglue_path):
        path_to_model = superglue_path
    else:
        raise FileNotFoundError(superglue_path)

    superglue_unpacker = SuperGlueUnpackerWithTemplate(project_data=project_data)
    tracker = SuperGlueFullVideoTrackerWithTemplate(superglue_unpacker=superglue_unpacker, path_to_model=path_to_model)
    num_frames = project_data.num_frames - 1
    all_frame_pairs = {}
    for t in tqdm(range(num_frames)):
        frame0, frame1 = all_frame_dict[t], all_frame_dict[t + 1]
        frame_pair = FramePair(options=frame_pair_options, frame0=frame0, frame1=frame1)
        if frame_pair.check_both_frames_valid():
            # Use new method to match
            matches_class = tracker.match_two_time_points(t, t + 1)
            frame_pair.feature_matches = matches_class.array_matches_with_conf.tolist()
            # Explicitly load data to prevent frame class using original video path
            dat0, dat1 = project_data.red_data[t], project_data.red_data[t+1]
            frame_pair.load_raw_data(dat0, dat1)

            if match_using_additional_methods:
                frame_pair.match_using_all_methods()
        else:
            frame_pair.feature_matches = []
        all_frame_pairs[(t, t + 1)] = frame_pair
    return all_frame_pairs


def postprocess_matches_to_tracklets(all_frame_dict, all_frame_pairs, z_threshold, min_confidence, logger,
                                     verbose=0):
    # Also updates the matches of the object
    opt = dict(z_threshold=z_threshold, min_confidence=min_confidence)
    logger.info(
        f"Postprocessing pairwise matches using confidence threshold {min_confidence} and z threshold: {z_threshold}")
    all_matches_dict = {k: pair.calc_final_matches(**opt)
                        for k, pair in tqdm(all_frame_pairs.items())}
    logger.info("Extracting locations of neurons")
    all_zxy = {k: f.neuron_locs for k, f in all_frame_dict.items()}
    logger.info("Building tracklets")
    return build_tracklets_dfs(all_matches_dict, all_zxy, verbose=verbose)


def save_all_tracklets(df, df_multi_index_format, training_config):
    with safe_cd(training_config.project_dir):
        # Custom format for pairs
        subfolder = osp.join('2-training_data', 'raw')
        fname = osp.join(subfolder, 'clust_df_dat.pickle')
        training_config.pickle_data_in_local_project(df, fname)

        # Update to save as sparse from the beginning
        out_fname = training_config.config['df_3d_tracklets']
        logging.info("Converting dataframe to sparse format")
        df_multi_index_format = df_multi_index_format.astype(pd.SparseDtype("float", np.nan))
        training_config.pickle_data_in_local_project(df_multi_index_format, out_fname, custom_writer=pd.to_pickle)


def unpack_config_for_tracklets(training_config, segmentation_config):
    params = training_config.config['pairwise_matching_params']
    z_threshold = params.get('z_threshold', None)
    min_confidence = params.get('min_confidence', 0.1)
    # matching_method = params['matching_method']

    fname = os.path.join('raw', 'match_dat.pickle')
    fname = training_config.resolve_relative_path(fname, prepend_subfolder=True)
    all_frame_pairs = pickle_load_binary(fname)

    fname = os.path.join('raw', 'frame_dat.pickle')
    fname = training_config.resolve_relative_path(fname, prepend_subfolder=True)
    all_frame_dict = pickle_load_binary(fname)

    seg_metadata_fname = segmentation_config.resolve_relative_path_from_config('output_metadata')
    segmentation_metadata = DetectedNeurons(seg_metadata_fname)

    postprocessing_params = training_config.config['postprocessing_params']

    return all_frame_dict, all_frame_pairs, z_threshold, min_confidence, segmentation_metadata, postprocessing_params


def _unpack_config_frame2frame_matches(project_data, training_config, DEBUG):
    project_config = project_data.project_config
    # Get options
    tracker_params = training_config.config['tracker_params'].copy()
    tracker_params['project_config'] = project_config
    if 'num_frames' in training_config.config['tracker_params']:
        tracker_params['num_frames'] = training_config.config['tracker_params']['num_frames']
    else:
        tracker_params['num_frames'] = project_data.num_frames
    if DEBUG:
        tracker_params['num_frames'] = 5
    if 'start_volume' in training_config.config['tracker_params']:
        tracker_params['start_volume'] = training_config.config['tracker_params']['start_volume']
    else:
        logging.warning("Using deprecated dataset parameter for start volume")
        tracker_params['start_volume'] = project_config.start_volume

    frame_pair_options = FramePairOptions.load_from_config_file(project_config, training_config)
    frame_pair_options.apply_tanh_to_confidence = False
    # pairwise_matches_params = project_config.get_frame_pair_options(training_config)
    preprocessing_class = project_config.get_preprocessing_class()
    tracker_params['preprocessing_settings'] = preprocessing_class

    video_fname = preprocessing_class.get_path_to_preprocessed_data(red_not_green=True)

    track_on_green_channel = project_config.config['dataset_params'].get('segment_and_track_on_green_channel', False)

    return video_fname, tracker_params, frame_pair_options, track_on_green_channel


def _save_matches_and_frames(all_frame_dict: dict, all_frame_pairs: Union[dict, None],
                             training_config: SubfolderConfigFile) -> None:
    subfolder = osp.join('2-training_data', 'raw')
    Path(subfolder).mkdir(exist_ok=True)

    if all_frame_dict is not None:
        fname = osp.join(subfolder, 'frame_dat.pickle')
        [frame.prep_for_pickle() for frame in all_frame_dict.values()]
        training_config.pickle_data_in_local_project(all_frame_dict, fname)

    if all_frame_pairs is not None:
        fname = osp.join(subfolder, 'match_dat.pickle')
        [p.prep_for_pickle() for p in all_frame_pairs.values()]
        training_config.pickle_data_in_local_project(all_frame_pairs, fname)
    else:
        training_config.logger.warning(f"all_frame_pairs is None; this step will need to be rerun")


def filter_tracklets_using_volume(df_all_tracklets, volume_percent_threshold, min_length_to_save, verbose=0,
                                  DEBUG=False):
    """
    Split the tracklets based on a threshold on the percentage change in volume

    Usually, if the volume changes by a lot, it is because there is a segmentation error
    """
    # Get the split points
    df_only_volume = df_all_tracklets.xs('volume', level=1, axis=1)
    df_percent_changes = df_only_volume.diff() / df_only_volume

    df_split_points = df_percent_changes.abs() > volume_percent_threshold
    t_split_points, i_tracklet_split_points = df_split_points.to_numpy().nonzero()

    # Reformat the split points to be a dict per-tracklet
    all_names = get_names_from_df(df_only_volume)
    tracklet2split = defaultdict(list)
    for t, i_tracklet in zip(t_split_points, i_tracklet_split_points):
        tracklet_name = all_names[i_tracklet]
        tracklet2split[tracklet_name].append(t)

    # Get all the candidate tracklets, including the raw ones if no split detected
    all_new_tracklets = []
    all_names.sort()
    i_next_name = name2int_neuron_and_tracklet(all_names[-1])
    if verbose >= 1:
        print(f"New tracklets starting at index: {i_next_name + 1}")
    # convert_to_sparse = lambda x: pd.arrays.SparseArray(np.squeeze(x.values))
    for name in tqdm(all_names, leave=False):
        this_tracklet = df_all_tracklets[[name]]
        # this_tracklet.loc[name] = this_tracklet.groupby(level=1, axis=1).apply(convert_to_sparse)
        if name in tracklet2split:
            split_points = tracklet2split[name]
            these_candidates = split_multiple_tracklets(this_tracklet, split_points)
            # Remove short ones, and rename
            these_candidates = [c for c in these_candidates if c[name]['z'].count() >= min_length_to_save]
            for i, c in enumerate(these_candidates):
                if i == 0:
                    # The first tracklet keeps the original name
                    all_new_tracklets.append(c)
                    continue
                i_next_name += 1
                all_new_tracklets.append(c.rename(mapper={name: int2name_tracklet(i_next_name)}, axis=1))

        else:
            all_new_tracklets.append(this_tracklet)
        if DEBUG:
            print(tracklet2split[name])
            print(all_new_tracklets)
            break

    if verbose >= 1:
        print(f"Split {len(all_names)} raw tracklets into {len(all_new_tracklets)} new tracklets")
        print("Now concatenating...")

    # Convert to sparse datatype
    # all_converted_tracklets = [t.groupby(level=1, axis=1).apply(convert_to_sparse) for t in tqdm(all_new_tracklets, leave=False)]

    # Remake original all-tracklet dataframe
    # df = pd.concat(all_converted_tracklets)
    df = pd.concat(all_new_tracklets, axis=1)
    if verbose >= 1:
        print("Finished")
    return df, tracklet2split


def split_tracklets_using_neuron_match_conflicts(project_cfg: ModularProjectConfig, DEBUG=False):
    # Assume that step 3b has been run, and use the raw saved objects that contain matches with conflicts
    project_data = ProjectData.load_final_project_data_from_config(project_cfg, to_load_tracklets=True)
    initial_empty_cols = 500000
    if DEBUG:
        initial_empty_cols = 10

    # Unpack
    logging.info("Initializing worm object with conflicting tracklet matches")
    tracking_cfg = project_cfg.get_tracking_config()
    minimum_confidence = tracking_cfg.config['tracklet_splitting_postprocessing']['min_confidence']
    fname = os.path.join('raw', 'final_matching_with_conflict.pickle')
    fname = tracking_cfg.resolve_relative_path(fname, prepend_subfolder=True)
    final_matching_with_conflict = pickle_load_binary(fname)

    tracklets_and_neurons_class = project_data.tracklets_and_neurons_class
    worm_obj = TrackedWorm(detections=tracklets_and_neurons_class)
    worm_obj.reinitialize_all_neurons_from_final_matching(final_matching_with_conflict)

    logging.info("Calculating all needed split points")
    split_list_dict = worm_obj.get_conflict_time_dictionary_for_all_neurons(minimum_confidence=minimum_confidence)

    logging.info("Initializing tracklet splitting class")
    df_tracklets = project_data.df_all_tracklets
    df_padded = PaddedDataFrame.construct_from_basic_dataframe(df_tracklets, name_mode='tracklet',
                                                               initial_empty_cols=initial_empty_cols)

    df_split, name_mapping = PaddedDataFrame.split_using_dict_of_points(df_padded, split_list_dict)
    # Do not explicitly use the name_mapping dict, because step 3b should just be rerun
    df_final = df_split.return_sparse_dataframe()

    # Save and update configs
    # training_cfg = project_cfg.get_training_config()
    out_fname = os.path.join('3-tracking', 'all_tracklets_after_conflict_splitting.pickle')
    tracking_cfg.pickle_data_in_local_project(df_final, relative_path=out_fname, custom_writer=pd.to_pickle)

    # TODO: update the name of wiggle_split_tracklets_df_fname
    tracking_cfg.config['wiggle_split_tracklets_df_fname'] = out_fname
    tracking_cfg.update_self_on_disk()

    logging.info("The tracklets have been split, but now step 3b should be rerun to regenerate the matches "
                 "(Using previous matches should be fine)")


def overwrite_tracklets_using_ground_truth(project_cfg: ModularProjectConfig,
                                           keep_new_tracklet_matches=False,
                                           update_only_finished_neurons=False,
                                           use_original_tracklets=False, DEBUG=False):
    """
    Overwrites tracklet database using a smaller number of ground truth tracks

    Note: deletes any tracklet matches from neurons that don't have any ground truth

    Parameters
    ----------
    project_cfg
    keep_new_tracklet_matches
    update_only_finished_neurons
    use_original_tracklets
    DEBUG

    Returns
    -------

    """
    project_data = ProjectData.load_final_project_data_from_config(project_cfg, to_load_tracklets=True)
    training_cfg = project_cfg.get_training_config()
    tracking_cfg = project_cfg.get_tracking_config()

    # Unpack
    if use_original_tracklets:
        # Get the tracklets directly from step 2
        fname = training_cfg.resolve_relative_path_from_config('df_3d_tracklets')
        df_tracklets = pd.read_pickle(fname)
    else:
        df_tracklets = project_data.df_all_tracklets

    df_gt = project_data.final_tracks
    sanity_checks_on_dataframes(df_gt, df_tracklets)

    if update_only_finished_neurons:
        neurons_that_are_finished = project_data.finished_neuron_names()
    else:
        project_data.logger.info("Assuming partially tracked neurons are correct")
        neurons_that_are_finished = None

    # Delete conflicting tracklets, then concat
    # If step 3c has been run, this should do nothing?
    df_tracklets_no_conflict, _, _ = delete_tracklets_using_ground_truth(df_gt, df_tracklets,
                                                                         gt_names=neurons_that_are_finished,
                                                                         DEBUG=DEBUG)

    if neurons_that_are_finished is not None:
        df_to_concat = df_gt.loc[:, neurons_that_are_finished]
    else:
        df_to_concat = df_gt
    neuron_names = get_names_from_df(df_to_concat)
    name_gen = get_next_name_generator(df_tracklets_no_conflict, name_mode='tracklet')
    gtneuron2tracklets = {name: new_name for name, new_name in zip(neuron_names, name_gen)}
    df_to_concat = df_to_concat.rename(mapper=gtneuron2tracklets, axis=1)

    project_data.logger.info("Large pandas concat, may take a while...")
    df_including_tracks = pd.concat([df_tracklets_no_conflict, df_to_concat], axis=1)

    project_data.logger.info("Splitting non-contiguous tracklets using custom dataframe class")
    df_padded = PaddedDataFrame.construct_from_basic_dataframe(df_including_tracks, name_mode='tracklet',
                                                               initial_empty_cols=10000)
    df_split, name_mapping = df_padded.split_all_tracklets_using_mode(split_mode='gap', verbose=0)

    # Keep the names as they are in the ground truth track
    global2tracklet_new = update_global2tracklet_dictionary(df_split, gtneuron2tracklets, name_mapping)

    df_final = df_split.return_sparse_dataframe()

    if keep_new_tracklet_matches:
        raise NotImplementedError
        # Need to have a way to match these new neuron names to the old ones
        # tracking_cfg = project_cfg.get_tracking_config()
        # fname = tracking_cfg.resolve_relative_path_from_config('global2tracklet_matches_fname')
        # old_global2tracklet = pickle_load_binary(fname)
        #
        # offset = 1
        # for i, old_matches in enumerate(old_global2tracklet.values()):
        #     new_neuron_name = int2name_neuron(i + offset)
        #     while new_neuron_name in global2tracklet_tmp:
        #         offset += 1
        #         new_neuron_name = int2name_neuron(i + offset)
        #     global2tracklet_tmp[new_neuron_name] = old_matches

    # Save and update configs
    training_cfg = project_cfg.get_training_config()
    out_fname = os.path.join('2-training_data', 'all_tracklets_with_ground_truth.pickle')
    training_cfg.pickle_data_in_local_project(df_final, relative_path=out_fname, custom_writer=pd.to_pickle)

    global2tracklet_matches_fname = os.path.join('3-tracking', 'global2tracklet_with_ground_truth.pickle')
    tracking_cfg.pickle_data_in_local_project(global2tracklet_new, global2tracklet_matches_fname)

    tracking_cfg.config['global2tracklet_matches_fname'] = global2tracklet_matches_fname
    training_cfg.config['df_3d_tracklets'] = out_fname
    training_cfg.update_self_on_disk()
    tracking_cfg.update_self_on_disk()

    return df_including_tracks, global2tracklet_new


def update_global2tracklet_dictionary(df_split, global2tracklet_original, name_mapping):
    logging.info("Updating the dictionary that matches the neurons and tracklets")
    # Start with the original matches
    global2tracklet_tmp = {}
    for neuron_name, single_match in global2tracklet_original.items():
        if single_match in name_mapping:
            global2tracklet_tmp[neuron_name] = list(name_mapping[single_match])
        else:
            global2tracklet_tmp[neuron_name] = [single_match]
    global2tracklet_new = remove_tracklets_from_dictionary_without_database_match(df_split, global2tracklet_tmp)
    return global2tracklet_new


def sanity_checks_on_dataframes(df_gt, df_tracklets):
    try:
        df_tracklets.drop(level=1, columns='raw_tracklet_id', inplace=True)
    except KeyError:
        pass
    check_if_heterogenous_columns(df_tracklets, raise_error=True)
    try:
        df_gt.drop(level=1, columns='raw_tracklet_id', inplace=True)
    except KeyError:
        pass
    check_if_heterogenous_columns(df_gt, raise_error=True)
