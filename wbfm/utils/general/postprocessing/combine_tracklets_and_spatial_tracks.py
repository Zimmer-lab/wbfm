import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
from wbfm.utils.general.high_performance_pandas import get_names_from_df
from wbfm.utils.tracklets.utils_tracklets import fix_global2tracklet_full_dict
from wbfm.utils.projects.finished_project_data import ProjectData
from scipy.spatial.distance import squareform, pdist
from tqdm.auto import tqdm

from wbfm.utils.projects.project_config_classes import SubfolderConfigFile, ModularProjectConfig
from wbfm.utils.general.utils_filenames import read_if_exists
from wbfm.utils.projects.utils_project import safe_cd


def combine_matched_tracklets(these_tracklet_names: List[str],
                              neuron_name: str,
                              df_tracklet: pd.DataFrame,
                              dlc_tracks: pd.DataFrame,
                              verbose=1) -> pd.DataFrame:
    """Combines a covering of short tracklets and a gappy DLC track into a final DLC-style track"""
    coords = ['z', 'x', 'y', 'likelihood']

    # these_tracklet_names = fix_matches_to_use_keys_not_int(df_tracklet, these_tracklet_ind)
    logging.info(f"Found {len(these_tracklet_names)} tracklets for {neuron_name}")

    if len(these_tracklet_names) == 0:
        # Then no tracklets were found, so we pass an empty column
        one_col_shape = dlc_tracks[neuron_name].shape
        summed_tracklet_array = np.empty(one_col_shape)
        summed_tracklet_array[:] = np.nan
        if verbose >= 1:
            logging.warning(f"No tracklets found for {neuron_name}")

    else:
        # Combine tracklets (one dataframe, multiple names)
        new_df = df_tracklet[these_tracklet_names].copy()
        numpy_columns = []
        for c in coords:
            this_column = np.nansum([new_df[name][c] for name in these_tracklet_names], axis=0)
            # My tracker often does not track the very last frames, so fill with nan
            if len(this_column) < len(dlc_tracks):
                tmp = np.zeros(len(dlc_tracks) - len(this_column))
                tmp[:] = np.nan
                this_column = np.hstack([this_column, tmp])

            numpy_columns.append(this_column)
            if verbose >= 1 and c == 'z':
                logging.info(f"Matching neuron {neuron_name} with tracklets {these_tracklet_names}")
                logging.info(f"summed_tracklet_array: {this_column}")
                logging.info(f"Nonzero entries: {np.where(this_column>0)}")
        summed_tracklet_array = np.stack(numpy_columns, axis=1)

    # Morph to DLC format
    cols = [[neuron_name], coords]
    cols = pd.MultiIndex.from_product(cols)
    summed_tracklet_df = pd.DataFrame(data=summed_tracklet_array, columns=cols)

    return summed_tracklet_df


def combine_global_and_tracklet_coverings(global2tracklet: Dict[str, List[str]],
                                          df_tracklet: pd.DataFrame,
                                          df_global_tracks: pd.DataFrame,
                                          keep_only_tracklets_in_final_tracks: bool,
                                          verbose=0):
    """
    Combines coverings of all tracklets and globally-tracked neurons

    If there is a tracklet covering for a time point, then it overwrites the global track.
    If keep_only_tracklets_in_final_tracks is false, then points without a tracklet are filled by the global
        track position, if any. Otherwise, only tracklets survive to the final dataframe
    """
    all_df = []
    logging.info(f"Found {len(global2tracklet)} tracklet-track combinations")

    # Build new tracklets into intermediate column
    for neuron_name, these_tracklet_names in global2tracklet.items():
        df = combine_matched_tracklets(these_tracklet_names, neuron_name, df_tracklet, df_global_tracks,
                                       verbose=verbose-1)
        all_df.append(df)
    new_tracklet_df = pd.concat(all_df, axis=1)
    if verbose >= 3:
        print(all_df)

    # Produce new dataframe
    if not keep_only_tracklets_in_final_tracks:
        final_track_df = combine_dlc_and_tracklets(new_tracklet_df, df_global_tracks)
    else:
        final_track_df = new_tracklet_df

    return final_track_df, new_tracklet_df


def combine_dlc_and_tracklets(new_tracklet_df, dlc_tracks):
    # Combine new and old tracklets (two dataframes, same neuron names)
    # Note: needs a loop because combine_first() doesn't work for multiindexes
    new_tracklet_df.replace(0, np.NaN, inplace=True)
    final_track_df = new_tracklet_df.copy()
    all_neuron_names = get_names_from_df(new_tracklet_df)
    for name in all_neuron_names:
        final_track_df[name] = new_tracklet_df[name].combine_first(dlc_tracks[name])
    return final_track_df


def final_tracks_from_tracklet_matches_from_config(track_config: SubfolderConfigFile,
                                                   training_cfg: SubfolderConfigFile,
                                                   project_cfg: ModularProjectConfig,
                                                   use_imputed_df,
                                                   start_from_manual_matches,
                                                   DEBUG=False):

    d_max, df_global_tracks, df_tracklets, min_overlap, output_df_fname, \
        keep_only_tracklets_in_final_tracks, global2tracklet, used_indices, allowed_tracklet_endpoint_wiggle\
        = _unpack_tracklets_for_combining(
            project_cfg, training_cfg, track_config, use_imputed_df, start_from_manual_matches)

    # Rename to be sequential, like the reindexed segmentation
    logging.info("Concatenating tracklets")
    combined_df, new_tracklet_df = combine_global_and_tracklet_coverings(global2tracklet,
                                                                         df_tracklets,
                                                                         df_global_tracks,
                                                                         keep_only_tracklets_in_final_tracks,
                                                                         verbose=0)
    _save_combined_dataframe(DEBUG, combined_df, output_df_fname, project_cfg.project_dir, track_config)


def get_already_covered_indices(df_tracklets, previous_matches):
    if len(previous_matches) > 0:
        all_tracklet_names = get_names_from_df(df_tracklets)
        all_tracklet_ind = df_tracklets.index
        covering_time_points = []
        for i2 in previous_matches:
            if type(i2) == str:
                tracklet_name = i2
            else:
                tracklet_name = all_tracklet_names[i2]
            is_nan = df_tracklets[tracklet_name]['x'].isnull()
            covering_time_points.extend(list(all_tracklet_ind[~is_nan]))
    else:
        covering_time_points = None
    return covering_time_points


def _save_combined_dataframe(DEBUG, combined_df, output_df_fname, project_dir, track_config):
    with safe_cd(project_dir):
        # Actually save
        track_config.save_data_in_local_project(combined_df, output_df_fname, also_save_csv=True)
        # logging.info(f"Saving to: {output_df_fname}")
        # combined_df.to_hdf(output_df_fname, key='df_with_missing')

        # csv_fname = Path(output_df_fname).with_suffix('.csv')
        # combined_df.to_csv(csv_fname)

        if not DEBUG:
            # Save only df_fname in yaml; don't overwrite other fields
            updates = {'final_3d_tracks_df': str(output_df_fname)}
            track_config.config.update(updates)
            track_config.update_self_on_disk()


def _unpack_tracklets_for_combining(project_cfg: ModularProjectConfig,
                                    training_cfg: SubfolderConfigFile,
                                    track_config: SubfolderConfigFile,
                                    use_imputed_df,
                                    use_manual_matches):
    d_max = track_config.config['final_3d_postprocessing']['max_dist']
    min_overlap = track_config.config['final_3d_postprocessing']['min_overlap_dlc_and_tracklet']
    # min_dlc_confidence = track_config.config['final_3d_postprocessing']['min_dlc_confidence']
    allowed_tracklet_endpoint_wiggle = track_config.config['final_3d_postprocessing']['allowed_tracklet_endpoint_wiggle']
    keep_only_tracklets_in_final_tracks = track_config.config['final_3d_postprocessing'][
        'keep_only_tracklets_in_final_tracks']
    output_df_fname = track_config.config['final_3d_postprocessing']['output_df_fname']

    # Use main object to load
    project_data = ProjectData(project_cfg.project_dir, project_cfg)
    df_tracklets = project_data.df_all_tracklets
    if not use_manual_matches:
        project_data.precedence_global2tracklet = ['automatic', 'manual']
    if use_imputed_df:
        project_data.precedence_tracks = ['imputed', 'automatic', 'fdnc']

    with safe_cd(project_cfg.project_dir):

        subfolder = os.path.join('3-tracking', 'postprocessing')
        Path(subfolder).mkdir(exist_ok=True)
        # tracklet_fname = training_cfg.resolve_relative_path('all_tracklets.h5', prepend_subfolder=True)
        # if not use_imputed_df:
        #     global_tracks_fname = track_config.resolve_relative_path_from_config('final_3d_tracks_df')
        # else:
        #     global_tracks_fname = track_config.resolve_relative_path_from_config('missing_data_imputed_df')
        #     logging.info(f"Reading from the imputed data, not the original global tracks: {global_tracks_fname}")
        # df_tracklets: pd.DataFrame = read_if_exists(tracklet_fname)
        # df_global_tracks: pd.DataFrame = read_if_exists(global_tracks_fname)

        df_global_tracks = project_data.final_tracks
        logging.info(f"Combining {int(df_tracklets.shape[1]/4)} tracklets with {int(df_global_tracks.shape[1]/4)} neurons")
        df_global_tracks.replace(0, np.NaN, inplace=True)

    # Check for previous matches, and start from them
    global2tracklet = project_data.global2tracklet
    if global2tracklet is None:
        logging.info(f"Did not find previous tracklet matches")
        global2tracklet = defaultdict(list)
        used_names = set()
    else:
        used_names = set()
        [used_names.update(names) for names in global2tracklet.values()]
        num_tracklets = len(get_names_from_df(df_tracklets))
        # logging.info(f"Found previous tracklet matches with {len(used_names)}/{num_tracklets} matches")
        # Should be fixed: don't allow these to be integers from the beginning
        global2tracklet = fix_global2tracklet_full_dict(df_tracklets, global2tracklet)

    # fname = track_config.resolve_relative_path_from_config('global2tracklet_matches_fname')
    # if Path(fname).exists():
    #     logging.info(f"Found previous tracklet matches at {fname}")
    #     global2tracklet = pickle_load_binary(fname)
    #     used_indices = set()
    #     [used_indices.update(ind) for ind in global2tracklet.values()]
    # else:
    #     logging.info(f"Did not find previous tracklet matches")
    #     global2tracklet = defaultdict(list)
    #     used_indices = set()

    return d_max, df_global_tracks, df_tracklets, min_overlap, output_df_fname, \
        keep_only_tracklets_in_final_tracks, global2tracklet, used_names, allowed_tracklet_endpoint_wiggle


def remove_overmatching(df, tol=1e-3):
    """
    Checks the tracklet + tracks combined dataframe for multiple neuron names being assigned to the same location, and
    removes the one with lower confidence

    Returns
    -------

    """
    all_neurons = get_names_from_df(df)
    coords = ['z', 'x', 'y']
    tspan = list(df.index)

    for i_time in tqdm(tspan):
        all_zxy = np.array([df[n][coords].iloc[i_time].to_numpy() for n in all_neurons])
        dist_mat = squareform(pdist(all_zxy))
        dist_mat[np.diag_indices_from(dist_mat)] = np.nan

        min_dist = np.nanmin(dist_mat, axis=1)
        all_overmatched = []
        # Get neurons that are extremely close
        for i0, d in enumerate(min_dist):
            if np.abs(d) < tol:
                i1 = np.nanargmin(dist_mat[i0, :])
                all_overmatched.append([all_neurons[i0], all_neurons[i1]])

        # Remove the neuron with lower confidence
        for n0, n1 in all_overmatched:
            conf0, conf1 = df[n0]['likelihood'].iloc[i_time], df[n1]['likelihood'].iloc[i_time]
            d_conf = conf0 - conf1
            if np.isnan(d_conf):
                continue
            if d_conf == 0:
                raise ValueError
            if d_conf > 0:
                df[n1].iloc[i_time] = np.nan
            else:
                df[n0].iloc[i_time] = np.nan

    return df


def remove_overmatched_tracks_using_config(track_cfg: SubfolderConfigFile):

    df_fname = track_cfg.resolve_relative_path_from_config('final_3d_tracks_df')
    df = read_if_exists(df_fname)

    df = remove_overmatching(df)

    # Overwrite
    df.to_hdf(df_fname, key='df_with_missing')
    df.to_csv(Path(df_fname).with_suffix('.csv.'))
