import os

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import tensorflow as tf

tf.__version__  # tf must be imported, see https://github.com/pytorch/pytorch/issues/81140
from wbfm.utils.neuron_matching.utils_candidate_matches import fit_umap_using_frames

from wbfm.utils.external.utils_pandas import fill_missing_indices_with_nan
from wbfm.utils.neuron_matching.long_range_matching import _unpack_for_track_tracklet_matching, \
    extend_tracks_using_global_tracking, greedy_matching_using_node_class, \
    combine_tracklets_using_matching, _save_graphs_and_combined_tracks

from wbfm.utils.neuron_matching.utils_candidate_matches import rename_columns_using_matching, \
    combine_dataframes_using_bipartite_matching
from wbfm.utils.nn_utils.superglue import SuperGlueUnpackerWithTemplate
from wbfm.utils.nn_utils.worm_with_classifier import DirectFeatureSpaceTemplateMatcher, PostprocessedFeatureSpaceTemplateMatcher, _unpack_project_for_global_tracking, \
    SuperGlueFullVideoTrackerWithTemplate, track_using_template
from wbfm.utils.external.random_templates import generate_random_valid_template_frames
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.projects.project_config_classes import ModularProjectConfig
from wbfm.utils.projects.utils_project import safe_cd
from wbfm.utils.tracklets.tracklet_class import TrackedWorm, DetectedTrackletsAndNeurons
from wbfm.utils.tracklets.utils_tracklets import split_all_tracklets_at_once
from wbfm.utils.external.utils_pandas import crop_to_same_time_length


def track_using_using_config(project_cfg, use_superglue_tracker=False, DEBUG=False):
    all_frames, num_frames, num_random_templates, project_data, t_template, tracking_cfg, use_multiple_templates = _unpack_project_for_global_tracking(
        DEBUG, project_cfg)

    # Do this by default for barlow embeddings
    use_umap_preprocessing = not use_superglue_tracker

    # Create the helper classes that actually do the matching
    if use_superglue_tracker:
        superglue_unpacker = SuperGlueUnpackerWithTemplate(project_data=project_data, t_template=t_template)
        tracker = SuperGlueFullVideoTrackerWithTemplate(superglue_unpacker=superglue_unpacker)
        model = tracker.model # Save for future tracker initialization

        def _init_tracker(t):
            superglue_unpacker = SuperGlueUnpackerWithTemplate(project_data=project_data, t_template=t)
            tracker = SuperGlueFullVideoTrackerWithTemplate(superglue_unpacker=superglue_unpacker,  model=model)
            return tracker
        
    elif use_umap_preprocessing:
        umap = fit_umap_using_frames(all_frames)
        def _init_tracker(t):
            return PostprocessedFeatureSpaceTemplateMatcher(template_frame=all_frames[t], confidence_gamma=100, postprocesser=umap.transform)
        
        tracker = _init_tracker(t=t_template)
    
    else:
        # Simplest; direct matching in the feature space
        def _init_tracker(t):
            return DirectFeatureSpaceTemplateMatcher(template_frame=all_frames[t], confidence_gamma=100)

    min_neurons_for_template = 50
    all_dfs_raw = []

    if not use_multiple_templates:
        df_final = track_using_template(all_frames, num_frames, project_data, tracker)
    else:
        # Ensure the reference frames are actually good by checking they have a minimum number of neurons
        all_templates = generate_random_valid_template_frames(all_frames, min_neurons_for_template, num_frames,
                                                              t_template, num_random_templates)

        project_cfg.logger.info(f"Using {num_random_templates} templates at t={all_templates}")
        # All subsequent dataframes will have their names mapped to this
        df_base = track_using_template(all_frames, num_frames, project_data, tracker)
        all_dfs_names_aligned = [df_base]
        all_dfs_raw = [df_base]
        for i, t in enumerate(tqdm(all_templates[1:])):
            tracker = _init_tracker(t=t)
            df = track_using_template(all_frames, num_frames, project_data, tracker)
            df_name_aligned, _, _, _ = rename_columns_using_matching(df_base, df, try_to_fix_inf=True)#, column='raw_segmentation_id')
            all_dfs_names_aligned.append(df_name_aligned)
            all_dfs_raw.append(df)

        tracking_cfg.config['t_templates'] = all_templates
        df_final = combine_dataframes_using_bipartite_matching(all_dfs_names_aligned)

    # Save
    out_fname = '3-tracking/postprocessing/df_tracks_postprocessed.h5'
    out_fname = tracking_cfg.save_data_in_local_project(df_final, out_fname, also_save_csv=True,
                                                        make_sequential_filename=True)
    out_fname = tracking_cfg.unresolve_absolute_path(out_fname)
    tracking_cfg.config['leifer_params']['output_df_fname'] = str(out_fname)

    # Also save the intermediate dataframes
    if use_multiple_templates:
        out_fname = '3-tracking/postprocessing/df_tracks_template-0.h5'
        for df in all_dfs_raw:
            out_fname = tracking_cfg.save_data_in_local_project(df, out_fname, also_save_csv=False,
                                                                make_sequential_filename=True)

    tracking_cfg.update_self_on_disk()


def match_two_projects_using_superglue_using_config(project_cfg_base: ModularProjectConfig,
                                                    project_cfg_target: ModularProjectConfig, to_save=True,
                                                    use_multiple_templates=True, only_match_same_time_points=False,
                                                    DEBUG=False):
    """
    Matches two projects using the main tracking pipeline

    Tracks from template frames of the base project to all frames of the target project

    Saves pickle files, .h5, and .xlsx files in the visualization subfolder of the target and base projects.


    """
    all_frames_base, _, num_random_templates, project_data_base, t_template, tracking_cfg, _ = _unpack_project_for_global_tracking(
        DEBUG, project_cfg_base)
    # Also unpack second config
    all_frames_target, num_frames, _, project_data_target, _, _, _ = _unpack_project_for_global_tracking(
        DEBUG, project_cfg_target)

    superglue_unpacker = SuperGlueUnpackerWithTemplate(project_data=project_data_base, t_template=t_template)
    tracker_base = SuperGlueFullVideoTrackerWithTemplate(superglue_unpacker=superglue_unpacker)
    model = tracker_base.model  # Save for later initialization
    min_neurons_for_template = 50
    if only_match_same_time_points:
        num_random_templates = 100

    if not use_multiple_templates and not only_match_same_time_points:
        df_final = track_using_template(all_frames_target, num_frames, project_data_target, tracker_base)
    elif use_multiple_templates:
        # Ensure the reference frames are actually good by checking they have a minimum number of neurons
        all_templates = generate_random_valid_template_frames(all_frames_base, min_neurons_for_template, num_frames,
                                                              t_template, num_random_templates)

        project_data_base.logger.info(f"Using {num_random_templates} templates at t={all_templates}")

        # All subsequent dataframes will have their names mapped to this
        df_base = track_using_template(all_frames_target, num_frames, project_data_target, tracker_base)
        all_dfs_names_aligned = [df_base]
        all_dfs_raw = [df_base]
        for i, t in enumerate(tqdm(all_templates[1:])):
            superglue_unpacker = SuperGlueUnpackerWithTemplate(project_data=project_data_base, t_template=t)
            tracker = SuperGlueFullVideoTrackerWithTemplate(superglue_unpacker=superglue_unpacker, model=model)
            if only_match_same_time_points:
                _all_frames_target = all_frames_target[t]
            else:
                _all_frames_target = all_frames_target

            df = track_using_template(all_frames_target, num_frames, project_data_target, tracker)
            df_name_aligned, _, _, _ = rename_columns_using_matching(df_base, df, try_to_fix_inf=True)
            all_dfs_names_aligned.append(df_name_aligned)
            all_dfs_raw.append(df)

        tracking_cfg.config['t_templates'] = all_templates
        df_final = combine_dataframes_using_bipartite_matching(all_dfs_names_aligned)

    # Ensure that there are the same number of time points
    df_final, df_original = crop_to_same_time_length(df_final, project_data_target.final_tracks, axis=0)

    _, matches, conf, name_mapping = rename_columns_using_matching(df_final, df_original, try_to_fix_inf=True)

    if to_save:
        # Save in target AND base folders
        fname = f'match_{project_data_base.shortened_name}_{project_data_target.shortened_name}.pickle'
        fname = os.path.join('visualization', fname)
        project_cfg_base.pickle_data_in_local_project(matches, fname)
        project_cfg_target.pickle_data_in_local_project(matches, fname)

        fname = f'conf_{project_data_base.shortened_name}_{project_data_target.shortened_name}.pickle'
        fname = os.path.join('visualization', fname)
        project_cfg_base.pickle_data_in_local_project(conf, fname)
        project_cfg_target.pickle_data_in_local_project(conf, fname)

        fname = f'name_mapping_{project_data_base.shortened_name}_{project_data_target.shortened_name}.pickle'
        fname = os.path.join('visualization', fname)
        project_cfg_base.pickle_data_in_local_project(name_mapping, fname)
        project_cfg_target.pickle_data_in_local_project(name_mapping, fname)

        fname = f'name_mapping_{project_data_base.shortened_name}_{project_data_target.shortened_name}.xlsx'
        fname = os.path.join('visualization', fname)
        df_mapping = pd.DataFrame(name_mapping, index=["Immobilized Match"]).T
        fname1 = os.path.join(project_data_base.project_dir, fname)
        df_mapping.to_excel(fname1)
        fname2 = os.path.join(project_data_target.project_dir, fname)
        df_mapping.to_excel(fname2)

        # Also update the manual annotation file for the target project, if there are any manual ids
        manual_ids_before = project_before.neuron_name_to_manual_id_mapping(confidence_threshold=0,
                                                                            remove_unnamed_neurons=True,
                                                                            flip_names_and_ids=True)
        if len(manual_ids_before) > 0:
            manual_ids_after = {name_mapping.get(k, k): v for k, v in manual_ids_before.items()}
            df_after_manual_tracking = project_after.df_manual_tracking.copy()
            fname = project_after.df_manual_tracking_fname
            # Save a backup of the original file
            df_after_manual_tracking.to_hdf(fname.replace('.xlsx', '_backup.xlsx'))
            # Rename neurons, and save with the original filename
            df_after_manual_tracking['ID1'] = df_after_manual_tracking['Neuron ID'].map(map_only_named)
            df_after_manual_tracking.to_xlsx(fname)

    return df_final, matches, conf, name_mapping


def _map_only_named(x):
    new_name = name_mapping_manual_ids.get(x, '')
    if 'neuron' not in new_name:
        return new_name
    else:
        return ''


def match_tracks_and_tracklets_using_config(project_config: ModularProjectConfig, to_save=True, verbose=0,
                                            DEBUG=False):
    """Replaces: final_tracks_from_tracklet_matches_from_config"""
    # Initialize project data and unpack
    logger = project_config.logger
    project_data = ProjectData.load_final_project_data_from_config(project_config, to_load_tracklets=True)

    df_global_tracks, min_confidence, min_overlap, num_neurons, only_use_previous_matches, outlier_threshold, \
    previous_matches, t_template, track_config, tracklets_and_neurons_class, use_multiple_templates, \
    use_previous_matches, tracklet_splitting_iterations, auto_split_conflicts = _unpack_for_track_tracklet_matching(project_data)

    # Add initial tracklets to neurons, then add matches (if any found before)
    logger.info(f"Initializing worm class with settings: \n"
                f"only_use_previous_matches={only_use_previous_matches}\n"
                f"use_previous_matches={use_previous_matches}\n"
                f"use_multiple_templates={use_multiple_templates}")

    def _initialize_worm(tracklets_obj, verbose=verbose):
        _worm_obj = TrackedWorm(detections=tracklets_obj, logger=logger, verbose=verbose)
        if only_use_previous_matches:
            _worm_obj.initialize_neurons_using_previous_matches(previous_matches)
        else:
            _worm_obj.initialize_neurons_at_time(t=t_template, num_expected_neurons=num_neurons,
                                                 df_global_tracks=df_global_tracks)
            if use_previous_matches:
                _worm_obj.add_previous_matches(previous_matches)
        # _worm_obj.initialize_all_neuron_tracklet_classifiers()
        if verbose >= 1:
            logger.info(f"Initialized worm object: {_worm_obj}")
        return _worm_obj

    worm_obj = _initialize_worm(tracklets_and_neurons_class)

    # Note: need to load this after the worm object is initialized, because the df may be modified
    df_tracklets = worm_obj.detections.df_tracklets_zxy
    df_tracklets_split = None

    extend_tracks_opt = dict(min_overlap=min_overlap, min_confidence=min_confidence,
                             outlier_threshold=outlier_threshold, verbose=verbose, DEBUG=DEBUG)
    if not only_use_previous_matches:
        logger.info("Adding all tracklet candidates to neurons")
        extend_tracks_using_global_tracking(df_global_tracks, df_tracklets, worm_obj, **extend_tracks_opt)

        # Build candidate graph, then postprocess it
        global_tracklet_neuron_graph = worm_obj.compose_global_neuron_and_tracklet_graph()
        if not auto_split_conflicts:
            logger.info("Greedy matching for each time slice subgraph")
            final_matching_with_conflict = greedy_matching_using_node_class(global_tracklet_neuron_graph,
                                                                            node_class_to_match=1)
            # Final step to remove time conflicts
            worm_obj.reinitialize_all_neurons_from_final_matching(final_matching_with_conflict)
            worm_obj.remove_conflicting_tracklets_from_all_neurons()
        else:
            # For metadata saving (the original worm with conflicts is otherwise not saved)
            logger.info("Calculating node matches for metadata purposes")
            final_matching_with_conflict = greedy_matching_using_node_class(global_tracklet_neuron_graph,
                                                                            node_class_to_match=1)

            logger.info("Iteratively splitting tracklets using track matching conflicts")
            for i_split in tqdm(range(tracklet_splitting_iterations)):
                split_list_dict = worm_obj.get_conflict_time_dictionary_for_all_neurons(
                    minimum_confidence=min_confidence)
                if len(split_list_dict) == 0:
                    logger.info(f"Found no further tracklet conflicts on iteration i={i_split}")
                    break
                else:
                    logger.info(f"Found conflicts on {len(split_list_dict)} tracklets")
                df_tracklets, df_tracklets_split, worm_obj = _split_tracklets_and_reinitialize_worm(
                    _initialize_worm,
                    df_global_tracks,
                    df_tracklets,
                    df_tracklets_split,
                    extend_tracks_opt,
                    i_split,
                    project_data,
                    split_list_dict,
                    tracklet_splitting_iterations,
                    worm_obj
                )

        logger.info("Removing tracklets that have time conflicts on a single neuron ")
        worm_obj.remove_conflicting_tracklets_from_all_neurons()
        worm_obj.update_time_covering_ind_for_all_neurons()
    else:
        global_tracklet_neuron_graph = None
        final_matching_with_conflict = None

    no_conflict_neuron_graph = worm_obj.compose_global_neuron_and_tracklet_graph()
    logger.info("Final matching to prevent the same tracklet assigned to multiple neurons")
    final_matching_no_conflict = greedy_matching_using_node_class(no_conflict_neuron_graph, node_class_to_match=1)
    df_new = combine_tracklets_using_matching(df_tracklets, final_matching_no_conflict)

    num_frames = df_global_tracks.shape[0]
    df_final, num_added = fill_missing_indices_with_nan(df_new, expected_max_t=num_frames)
    if num_added > 0:
        logger.warning(f"Some time points {num_added} are completely empty of tracklets, and are added as nan")

    # SAVE
    if to_save:
        with safe_cd(project_data.project_dir):
            _save_graphs_and_combined_tracks(df_final, final_matching_no_conflict, final_matching_with_conflict,
                                             global_tracklet_neuron_graph,
                                             track_config, worm_obj,
                                             df_tracklets_split)
    return df_final, final_matching_no_conflict, global_tracklet_neuron_graph, worm_obj


def _split_tracklets_and_reinitialize_worm(_initialize_worm, df_global_tracks, df_tracklets, df_tracklets_split,
                                           extend_tracks_opt, i_split, project_data, split_list_dict,
                                           tracklet_splitting_iterations, worm_obj):
    df_tracklets_split, all_new_tracklets, name_mapping = split_all_tracklets_at_once(df_tracklets, split_list_dict)
    tracklets_and_neurons_class2 = DetectedTrackletsAndNeurons(df_tracklets_split,
                                                               project_data.segmentation_metadata,
                                                               dataframe_output_filename=project_data.df_all_tracklets_fname)
    worm_obj2 = _initialize_worm(tracklets_and_neurons_class2, verbose=0)
    if i_split == tracklet_splitting_iterations - 1:
        # On the last iteration, allow short tracklets to be matched
        extend_tracks_opt['min_overlap'] = 1
    conf2 = extend_tracks_using_global_tracking(df_global_tracks, df_tracklets_split, worm_obj2,
                                                **extend_tracks_opt)
    # Overwrite original object, and continue
    worm_obj = worm_obj2
    df_tracklets = df_tracklets_split.astype(pd.SparseDtype("float", np.nan))
    return df_tracklets, df_tracklets_split, worm_obj
