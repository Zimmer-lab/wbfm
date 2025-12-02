import concurrent
import logging
import os
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

from wbfm.utils.general.high_performance_pandas import get_names_from_df
from wbfm.utils.external.custom_errors import NoMatchesError, NoNeuronsError
from wbfm.utils.general.utils_networkx import calc_bipartite_from_candidates
from wbfm.utils.projects.physical_units import PhysicalUnitConversion
from wbfm.utils.general.postprocessing.postprocessing_utils import remove_outliers_to_combine_tracks
from wbfm.utils.neuron_matching.matches_class import matches_between_tracks
from wbfm.utils.general.utils_filenames import get_sequential_filename
from wbfm.utils.projects.utils_project import safe_cd
from fDNC.src.DNC_predict import pre_matt, predict_matches, filter_matches, predict_label
from tqdm.auto import tqdm
import torch

from wbfm.utils.projects.finished_project_data import ProjectData, template_matches_to_dataframe
from wbfm.utils.projects.project_config_classes import ModularProjectConfig, SubfolderConfigFile
from wbfm.utils.external.utils_neuron_names import int2name_neuron

default_package_path = "/lisc/data/scratch/neurobiology/zimmer/fieseler/github_repos/fDNC_Neuron_ID"


def load_fdnc_options_and_template(custom_template=None, path_to_folder=None):
    prediction_options = load_fdnc_options(path_to_folder)
    template, template_label = load_fdnc_template(custom_template)

    return prediction_options, template, template_label


def load_fdnc_template(custom_template=None):
    if custom_template is None:
        temp_fname = os.path.join(default_package_path, 'Data', 'Example', 'template.data')
        temp = pre_matt(temp_fname)
        template = temp['pts']
        template_label = temp['name']
    else:
        template = custom_template
        template_label = None
    return template, template_label


def load_fdnc_options(path_to_folder=None):
    if path_to_folder is None:
        path_to_folder = default_package_path
    model_path = os.path.join(path_to_folder, 'model', 'model.bin')
    prediction_options = dict(
        cuda=torch.cuda.is_available(),
        model_path=model_path
    )
    if prediction_options['cuda']:
        logging.info("Found cuda!")
    else:
        logging.info("Did not find cuda, using cpu")
    return prediction_options


def track_using_fdnc(project_data: ProjectData,
                     prediction_options,
                     template,
                     match_confidence_threshold,
                     full_video_not_training=True,
                     physical_unit_conversion: PhysicalUnitConversion = None) -> list:
    if full_video_not_training:
        num_frames = project_data.num_frames

        def get_pts(i):
            these_pts = project_data.get_centroids_as_numpy(i)
            if len(these_pts) == 0:
                raise NoNeuronsError
            return physical_unit_conversion.zimmer2leifer(these_pts)
    else:
        num_frames = project_data.num_training_frames

        def get_pts(i):
            these_pts = project_data.get_centroids_as_numpy_training(i)
            return physical_unit_conversion.zimmer2leifer(these_pts)

    all_matches = []
    for i_frame in tqdm(range(num_frames), total=num_frames, leave=False):
        try:
            pts_scaled = get_pts(i_frame)
            matches, _ = predict_matches(test_pos=pts_scaled, template_pos=template, **prediction_options)
            matches = filter_matches(matches, match_confidence_threshold)
            all_matches.append(matches)
        except NoNeuronsError:
            all_matches.append([])
    return all_matches


def generate_templates_from_training_data(project_data: ProjectData, physical_unit_conversion: PhysicalUnitConversion):
    all_templates = []
    num_templates = project_data.num_training_frames

    for i in range(num_templates):
        custom_template = project_data.get_centroids_as_numpy_training(i)
        all_templates.append(physical_unit_conversion.zimmer2leifer(custom_template))
    return all_templates


def generate_random_templates(project_data: ProjectData, num_templates,
                              physical_unit_conversion: PhysicalUnitConversion, seed=42):
    rng = np.random.default_rng(seed=seed)
    template_ind = rng.integers(low=0, high=project_data.num_frames, size=num_templates)

    all_templates = []
    for i in template_ind:
        custom_template = project_data.get_centroids_as_numpy(i)
        all_templates.append(physical_unit_conversion.zimmer2leifer(custom_template))
    return all_templates, template_ind


def track_using_fdnc_multiple_templates(project_data: ProjectData,
                                        base_prediction_options,
                                        match_confidence_threshold,
                                        num_templates=None,
                                        physical_unit_conversion: PhysicalUnitConversion = None):
    all_templates = generate_templates_from_training_data(project_data,
                                                          physical_unit_conversion=physical_unit_conversion)

    def _parallel_func(template):
        return track_using_fdnc(project_data, base_prediction_options, template, match_confidence_threshold,
                                physical_unit_conversion=physical_unit_conversion)

    max_workers = round(project_data.num_training_frames / 2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_jobs = [executor.submit(_parallel_func, template) for template in all_templates]
        matches_per_template = [job.result() for job in submitted_jobs]

    # Combine the matches between each frame and template
    final_matches = combine_multiple_template_matches(matches_per_template)

    return final_matches


def combine_multiple_template_matches(matches_per_template, min_conf=0.1):
    """Correct value for empty matches in matches_per_template is []"""
    final_matches = []
    num_frames = len(matches_per_template[0])
    num_templates = len(matches_per_template)
    for i_frame in tqdm(range(num_frames), leave=False):
        candidate_matches = []
        for i_template in range(num_templates):
            candidate_matches.extend(matches_per_template[i_template][i_frame])
        # Reduce individual confidences so they are an average, not a sum
        candidate_matches = [(m[0], m[1], m[2] / num_templates) for m in candidate_matches]

        try:
            matches, conf, _ = calc_bipartite_from_candidates(candidate_matches, min_confidence_after_sum=min_conf)
            match_and_conf = [(m[0], m[1], c) for m, c in zip(matches, conf)]
        except NoMatchesError:
            match_and_conf = []
        final_matches.append(match_and_conf)

    return final_matches


def track_using_fdnc_from_config(project_cfg: ModularProjectConfig,
                                 tracks_cfg: SubfolderConfigFile,
                                 DEBUG=False):
    match_confidence_threshold, prediction_options, template, project_data, use_multiple_templates, \
    _, physical_unit_conversion, i_template = \
        _unpack_for_fdnc(project_cfg, tracks_cfg, DEBUG)

    if use_multiple_templates:
        logging.info("Tracking using multiple templates")
        all_matches = track_using_fdnc_multiple_templates(project_data, prediction_options, match_confidence_threshold,
                                                          physical_unit_conversion=physical_unit_conversion)
    else:
        logging.info("Tracking using single template")
        all_matches = track_using_fdnc(project_data, prediction_options, template, match_confidence_threshold,
                                       physical_unit_conversion=physical_unit_conversion)
    # Force the template to be actually correct
    num_neurons = template.shape[0]
    all_matches[i_template] = [(i, i, 1.0) for i in range(num_neurons)]

    logging.info("Converting matches to dataframe format")
    df = template_matches_to_dataframe(project_data, all_matches)

    logging.info("Saving tracks and matches")
    with safe_cd(project_cfg.project_dir):
        output_df_fname = tracks_cfg.config['leifer_params']['output_df_fname']

        output_pickle_fname = Path(output_df_fname).with_name('fdnc_matches.pickle')
        project_cfg.pickle_data_in_local_project(all_matches, str(output_pickle_fname))
        _save_final_tracks(df, tracks_cfg, output_df_fname)


def track_using_fdnc_random_from_config(project_cfg: ModularProjectConfig,
                                        tracks_cfg: SubfolderConfigFile,
                                        DEBUG=False):
    """WIP; not currently used"""
    match_confidence_threshold, prediction_options, _, project_data, _, num_templates, physical_unit_conversion = \
        _unpack_for_fdnc(project_cfg, tracks_cfg, DEBUG)

    all_templates, template_ind = generate_random_templates(project_data, num_templates=num_templates,
                                                            physical_unit_conversion=physical_unit_conversion)

    # Track using one template at a time, and save them to disk
    all_dfs = []
    all_all_matches = []
    logging.info("Tracking using multiple random templates")
    for i, template in tqdm(enumerate(all_templates)):
        all_matches = track_using_fdnc(project_data, prediction_options, template, match_confidence_threshold)
        df = template_matches_to_dataframe(project_data, all_matches)

        all_dfs.append(df)
        all_all_matches.append(all_matches)

    with safe_cd(project_cfg.project_dir):
        for i, (df, all_matches) in enumerate(zip(all_dfs, all_all_matches)):
            default_df_fname = Path(tracks_cfg.config['leifer_params']['output_df_fname'])
            base_fname, suffix_fname = default_df_fname.stem, default_df_fname.suffix
            new_base_fname = f"{base_fname}-{i}"
            this_df_fname = default_df_fname.with_name(f"{new_base_fname}{str(suffix_fname)}")

            pickle_fname = Path(default_df_fname).with_name('random_template_matches.pickle')
            output_pickle_fname = get_sequential_filename(str(pickle_fname))
            project_cfg.pickle_data_in_local_project(all_matches, output_pickle_fname)

            _save_final_tracks(df, tracks_cfg, this_df_fname)

    # Then use the positions to create a dictionary of inter-template names
    # TODO: Use multiple dataframes as the starting point
    all_mappings = []
    df0 = all_dfs[0]
    for i, df1 in enumerate(all_dfs[1:]):
        mapping = matches_between_tracks(df0, df1, user_inlier_mode=True,
                                         inlier_gamma=100.0)
        # Remove entirely unmatched neurons, and map confidence-thresholded neurons to temporary names
        renaming_dict = mapping.get_mapping_1_to_0_with_unmatched_names()
        old_names = set(get_names_from_df(df1))
        df1 = df1[old_names.intersection(mapping.names1)]

        df1.rename(columns=renaming_dict, level=0, inplace=True)
        all_mappings.append(mapping)

    with safe_cd(project_cfg.project_dir):
        fname = os.path.join('3-tracking', 'random_template_matches.pickle')
        tracks_cfg.pickle_data_in_local_project(all_mappings, fname)

    # Combine to make final tracks
    df_combined = remove_outliers_to_combine_tracks(all_dfs)

    with safe_cd(project_cfg.project_dir):
        df_fname = Path(tracks_cfg.config['leifer_params']['output_df_fname'])
        _save_final_tracks(df_combined, tracks_cfg, df_fname)


def _save_final_tracks(df, tracks_cfg, output_df_fname):
    Path(output_df_fname).parent.mkdir(exist_ok=True)

    tracks_cfg.save_data_in_local_project(df, output_df_fname, also_save_csv=True)
    tracks_cfg.config['final_3d_tracks_df'] = str(output_df_fname)
    tracks_cfg.update_self_on_disk()


def _unpack_for_fdnc(project_cfg, tracks_cfg, DEBUG):
    use_zimmer_template = tracks_cfg.config['leifer_params']['use_zimmer_template']
    use_multiple_templates = tracks_cfg.config['leifer_params']['use_multiple_templates']
    i_template = tracks_cfg.config['final_3d_tracks']['template_time_point']
    num_templates = tracks_cfg.config['leifer_params'].get('num_random_templates', None)
    project_data = ProjectData.load_final_project_data_from_config(project_cfg)
    physical_unit_conversion = PhysicalUnitConversion.load_from_config(project_cfg)
    if use_zimmer_template:
        custom_template = project_data.get_centroids_as_numpy(i_template)
        custom_template = physical_unit_conversion.zimmer2leifer(custom_template)
    else:
        custom_template = None
    prediction_options, template, _ = load_fdnc_options_and_template(custom_template=custom_template)
    fdnc_updates = tracks_cfg.config['leifer_params']['core_options']
    prediction_options.update(fdnc_updates)
    match_confidence_threshold = tracks_cfg.config['leifer_params']['match_confidence_threshold']

    return match_confidence_threshold, prediction_options, template, project_data, \
           use_multiple_templates, num_templates, physical_unit_conversion, i_template


def get_putative_names_from_config(project_config: ModularProjectConfig):
    """

    Parameters
    ----------
    project_config

    Returns
    -------

    """

    project_data = ProjectData.load_final_project_data_from_config(project_config)
    prediction_options, template, template_label = load_fdnc_options_and_template()
    physical_unit_conversion = PhysicalUnitConversion.load_from_config(project_config)

    all_only_top_dict = defaultdict(list)
    num_templates = project_data.num_training_frames

    for i_template in tqdm(range(num_templates)):
        pts = project_data.get_centroids_as_numpy_training(i_template)
        pts = physical_unit_conversion.zimmer2leifer(pts)
        labels = predict_label(test_pos=pts, template_pos=template, template_label=template_label, **prediction_options)
        template_top1 = labels[0]

        for i_neuron, neuron_top_candidate in enumerate(template_top1):
            name = int2name_neuron(i_neuron + 1)
            neuron_key = (name,)
            for i_name_or_conf in range(2):
                if i_name_or_conf == 0:
                    key = neuron_key + ('name',)
                else:
                    key = neuron_key + ('likelihood',)
                val = neuron_top_candidate[i_name_or_conf]
                all_only_top_dict[key].append(val)

    df_candidate_names = pd.DataFrame(all_only_top_dict)
    raw_names = get_names_from_df(df_candidate_names)
    all_match_dict = {}
    all_conf_dict = {}
    for n in raw_names:

        this_df = df_candidate_names[n]
        candidate_names = this_df['name'].unique()
        candidate_conf = np.zeros_like(candidate_names)

        conf_func = lambda x: sum(x) / df_candidate_names.shape[0]

        for i, c in enumerate(candidate_names):
            ind = this_df['name'] == c
            candidate_conf[i] = conf_func(this_df['likelihood'].loc[ind])

        # Just take the max value
        i_max = np.argmax(candidate_conf)
        this_name = candidate_names[i_max]
        if this_name != '':
            all_match_dict[n] = this_name
        else:
            all_match_dict[n] = n
        all_conf_dict[n] = candidate_conf[i_max]

    df_out1 = pd.DataFrame.from_dict(all_match_dict, orient='index', columns=['name'])
    df_out2 = pd.DataFrame.from_dict(all_conf_dict, orient='index', columns=['likelihood'])
    df_out = pd.concat([df_out1, df_out2], axis=1)

    # Save
    out_fname = os.path.join('4-traces', 'names_from_leifer_template.h5')
    out_fname = project_config.resolve_relative_path(out_fname)
    df_out.to_hdf(out_fname, key='df_with_missing')

    out_fname = Path(out_fname).with_suffix('.csv')
    df_out.to_csv(str(out_fname))
