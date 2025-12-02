import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from wbfm.utils.neuron_matching.utils_matching import filter_matches
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from wbfm.utils.external.utils_pandas import df_to_matches, accuracy_of_matches
from wbfm.utils.general.postprocessing.postprocessing_utils import filter_dataframe_using_likelihood
from wbfm.utils.external.utils_matplotlib import paired_boxplot_from_dataframes
from wbfm.utils.general.utils_paper import apply_figure_settings
from wbfm.utils.neuron_matching.utils_matching import calc_bipartite_from_positions, calc_nearest_neighbor_matches
from wbfm.utils.general.high_performance_pandas import get_names_from_df


def calc_true_positive(gt: dict, test: dict):
    num_tp = 0
    for k, v in gt.items():
        if test.get(k, None) == v:
            num_tp += 1
    return num_tp


def calc_mismatches(gt: dict, test: dict):
    num_mm = 0
    for k, v in test.items():
        if gt.get(k, None) != v:
            num_mm += 1
    return num_mm


def calc_missing_matches(gt: dict, test: dict):
    num_mm = 0
    for k, v in gt.items():
        if k not in test:
            num_mm += 1
    return num_mm


def calc_summary_scores_for_training_data(m_final,
                                          min_confidence=0.0,
                                          max_possible=None):
    """
    Assumes the true matches are trivial, e.g. (1,1)

    max_possible defaults to assuming the maximum match in the first column (template) is all that is possible
    """
    m_final = filter_matches(m_final, min_confidence)
    if max_possible is None:
        max_possible = np.max(m_final[:, 0]).astype(int) + 1
    m0to1_dict = {m[0]: m[1] for m in m_final}

    num_tp = 0
    num_outliers = 0
    for m0, m1 in m0to1_dict.items():
        if m0 > max_possible:
            continue
        if m0 == m1:
            num_tp += 1
        else:
            num_outliers += 1
    num_missing = max_possible - num_tp - num_outliers

    return num_tp, num_outliers, num_missing, max_possible


def get_confidences_of_tp_and_outliers(m_final):
    """
    Assumes the true matches are trivial, e.g. (1,1)

    max_possible defaults to assuming the maximum match in the first column (template) is all that is possible
    """
    m0to1_dict = {m[0]: m[1] for m in m_final}
    m0toconf_dict = {m[0]: m[2] for m in m_final}

    conf_tp = []
    conf_outliers = []
    for m0, m1 in m0to1_dict.items():
        if m0 == m1:
            conf_tp.append(m0toconf_dict[m0])
        else:
            conf_outliers.append(m0toconf_dict[m0])

    return conf_tp, conf_outliers


def calc_all_dist(df1, df2):
    # Check if they are same neuron, i.e. right on top of each other
    df1.replace(0.0, np.nan, inplace=True)
    df2.replace(0.0, np.nan, inplace=True)
    df_norm = np.sqrt(np.square(df1 - df2).sum(axis=1, min_count=1))

    num_total1 = df1.count()[0]
    num_total2 = df2.count()[0]
    num_total_total = df_norm.count()

    return df_norm.to_numpy(), num_total1, num_total2, num_total_total


def calc_accuracy(all_dist, dist_tol=1e-2):
    # Due to nan values, num_matches + num_mismatches != num_total
    num_matches = len(np.where(all_dist < dist_tol)[0])
    num_mismatches = len(np.where(all_dist > dist_tol)[0])
    num_nan_total = len(np.where(np.isnan(all_dist))[0])

    return num_matches, num_mismatches, num_nan_total


def calculate_column_of_differences(df_gt, df_test,
                                    column_to_check='raw_neuron_ind_in_list', neurons_that_are_finished=None):
    """
    Compares the neuron indices between a ground truth and a test dataframe
    Returns a list of columns with name 'true_neuron_ind' that can be concatenated to the original dataframe using:
    ```
    df_list.insert(0, df_test)
    df_with_true_neuron_ind = pd.concat(df_list, axis=1)
    ```

    Note that df_test can be a dataframe of tracks or tracklets
    """
    if neurons_that_are_finished is None:
        lookup = df_gt.loc[:, (slice(None), column_to_check)]
    else:
        lookup = df_gt.loc[:, (neurons_that_are_finished, column_to_check)]

    names = get_names_from_df(neurons_that_are_finished)

    df_list = []
    for name in tqdm(names):
        track = df_test[name][column_to_check]

        # Note: nan evaluates to false
        mask = lookup.apply(lambda col: col == track)
        idx, true_neuron_ind = np.where(mask)

        # Remove duplicates
        idx, idx_unique = np.unique(idx, return_index=True)
        true_neuron_ind = true_neuron_ind[idx_unique]

        df_list.append(pd.DataFrame(data=true_neuron_ind, columns=[(name, 'true_neuron_ind')], index=idx))

    # Construct full dataframe with these new columns
    # df_list.insert(0, df_test)
    # df_with_true_neuron_ind = pd.concat(df_list, axis=1)
    return df_list

##
## Plotting
##


def plot_histogram_at_likelihood_thresh(df1, df2, likelihood_thresh):
    """Assumes that the neurons have the same name; see rename_columns_using_matching"""
    df2_filter = filter_dataframe_using_likelihood(df2, likelihood_thresh)
    df_all_acc = calculate_accuracy_from_dataframes(df1, df2_filter)

    dat = [df_all_acc['matches_to_gt_nonnan'], df_all_acc['mismatches'], df_all_acc['nan_in_gt']]
    import seaborn as sns

    sns.histplot(dat, common_norm=False, stat="percent", multiple="stack")
    # sns.histplot(dat, multiple="stack")
    plt.ylabel("Percent of true neurons")
    plt.xlabel("Accuracy (various metrics)")

    plt.title(f"Likelihood threshold: {likelihood_thresh}")

    return dat


def calculate_accuracy_from_dataframes(df_gt: pd.DataFrame, df2_filter: pd.DataFrame, column_names=None) -> pd.DataFrame:
    """
    Calculates accuracy of two dataframes assuming they have the same column names (i.e. neuron names).
    Format of the dataframes should be the tracking format as produced by the pipeline, i.e. columns are multiindexed with
    neuron names and the column name 'raw_neuron_ind_in_list' containing the neuron indices

    In principle these dataframes should be loaded via the ProjectData class (final_tracks or intermediate_global_tracks), but this is not strictly necessary.

    Return dataframe has columns:
        matches: fraction of matches to ground truth
        matches_to_gt_nonnan: fraction of matches to ground truth, excluding ground truth time points that are nan
        mismatches: fraction of mismatches to ground truth
        nan_in_gt: fraction of ground truth time points that are nan

    Parameters
    ----------
    df_gt
    df2_filter

    Returns
    -------

    """
    if column_names is None:
        column_names = ['z', 'x', 'y']
    tracked_names = get_names_from_df(df_gt)

    all_dist_dict, all_total1, all_total2 = calculate_distance_pair_of_dataframes(df_gt, df2_filter, column_names)

    num_t = df_gt.shape[0]
    all_acc_dict = defaultdict(list)
    for name in tqdm(tracked_names, leave=False):
        if name not in all_dist_dict:
            all_acc_dict['matches'].append(np.nan)
            all_acc_dict['matches_to_gt_nonnan'].append(np.nan)
            all_acc_dict['mismatches'].append(np.nan)
            all_acc_dict['nan_in_gt'].append(np.nan)
            continue
        matches, mismatches, nan = calc_accuracy(all_dist_dict[name])
        num_total1, num_total2 = all_total1[name], all_total2[name]
        all_acc_dict['matches'].append(matches / num_t)
        all_acc_dict['matches_to_gt_nonnan'].append(matches / num_total1)
        all_acc_dict['mismatches'].append(mismatches / num_t)
        all_acc_dict['nan_in_gt'].append((num_t - num_total2) / num_t)
    df_all_acc = pd.DataFrame(all_acc_dict, index=tracked_names)
    return df_all_acc


def calculate_distance_pair_of_dataframes(df_gt, df2_filter, column_names):
    # Calculate distance between neuron positions in two dataframes with the SAME COLUMN NAMES

    tracked_names = get_names_from_df(df_gt)
    all_dist_dict = defaultdict(int)
    all_total1 = defaultdict(int)
    all_total2 = defaultdict(int)
    for name in tqdm(tracked_names, leave=False):
        if name not in df2_filter:
            continue
        this_df_gt, this_df2 = df_gt[name][column_names].copy(), df2_filter[name][column_names].copy()
        all_dist_dict[name], all_total1[name], all_total2[name], _ = calc_all_dist(this_df_gt, this_df2)
    return all_dist_dict, all_total1, all_total2


def calculate_confidence_of_mismatches(df_gt: pd.DataFrame, df2_filter: pd.DataFrame, column_names=None) -> pd.DataFrame:
    """
    Returns all confidence values, instead of summary statistics (see: calculate_accuracy_from_dataframes)

    Returns an extended dataframe df2_filter, which has added columns ('is_correct') corresponding to correctness:
        0 - mismatch
        1 - match
        2 - ground truth was nan

    Parameters
    ----------
    df_gt
    df2_filter

    Returns
    -------

    """
    if column_names is None:
        column_names = ['z', 'x', 'y']
    tracked_names = get_names_from_df(df_gt)

    all_dist_dict, all_total1, all_total2 = calculate_distance_pair_of_dataframes(df_gt, df2_filter, column_names)

    df2_with_classes = df2_filter.copy()

    dist_tol = 1e-2
    num_t = df2_filter.shape[0]
    for name in tqdm(tracked_names, leave=False):
        this_dist = all_dist_dict[name]

        # ind_matches = np.where(this_dist < dist_tol)[0]
        ind_mismatches = np.where(this_dist > dist_tol)[0]
        ind_nan = np.where(np.isnan(this_dist))[0]

        new_col = np.ones(num_t, dtype=int)
        new_col[ind_mismatches] = 0
        new_col[ind_nan] = 2

        df2_with_classes.loc[:, (name, 'is_correct')] = new_col

    return df2_with_classes.copy()  # To defragment


# Specific tests for tracklets
def test_baseline_and_new_matcher_on_vgg_features(project_data, desc0=None, desc1=None, t0=0, t1=1,
                                                  calculate_superglue_matches=True):
    """

    Optional: directly pass a custom feature space

    Extracts vgg features as saved in two Frame classes, and compares 3 ways of matching:
    1. Superglue postprocessing (what I'm doing)
    2. Bipartite matching directly on the feature space
    3. Greedy matching directly on the feature space
    """
    from wbfm.utils.nn_utils.superglue import SuperGlue
    f0 = project_data.raw_frames[t0]
    f1 = project_data.raw_frames[t1]
    df_gt = project_data.get_final_tracks_only_finished_neurons()[0]

    # Unpack
    if desc0 is None:
        desc0 = f0.all_features
    if desc1 is None:
        desc1 = f1.all_features
    desc0 = torch.tensor(desc0).float()
    desc1 = torch.tensor(desc1).float()

    # For now, remove z
    # kpts0 = torch.tensor(f0.neuron_locs)[:, 1:].float()
    # kpts1 = torch.tensor(f1.neuron_locs)[:, 1:].float()
    kpts0 = torch.tensor(f0.neuron_locs).float()
    kpts1 = torch.tensor(f1.neuron_locs).float()

    scores0 = torch.ones((kpts0.shape[0], 1)).float()
    scores1 = torch.ones((kpts1.shape[0], 1)).float()

    image0 = np.expand_dims(np.expand_dims(np.zeros_like(project_data.red_data[t0]), axis=0), axis=0)
    image1 = np.expand_dims(np.expand_dims(np.zeros_like(project_data.red_data[t1]), axis=0), axis=0)

    all_matches = torch.unsqueeze(torch.tensor(df_to_matches(df_gt, t0, t1)), dim=0)

    # Repack
    if calculate_superglue_matches:
        data = dict(descriptors0=desc0, descriptors1=desc1, keypoints0=kpts0, keypoints1=kpts1, all_matches=all_matches,
                    image0=image0, image1=image1,
                    scores0=scores0, scores1=scores1)

        model = SuperGlue(config=dict(descriptor_dim=desc0.shape[1], match_threshold=0.0))

        out = model(data)
        new_matches = [[i, m0] for i, m0 in enumerate(out['matches0'].detach().numpy())]
        acc_pipeline = accuracy_of_matches(all_matches, new_matches)
    else:
        # Use the matches from the object as already calculated
        new_matches = project_data.raw_matches[(t0, t1)].final_matches
        acc_pipeline = accuracy_of_matches(all_matches, new_matches)

    acc_baseline_bipartite, acc_baseline_greedy = test_baseline_feature_space_matchers(desc0, desc1, all_matches)

    return acc_pipeline, acc_baseline_bipartite, acc_baseline_greedy


def test_baseline_feature_space_matchers(project_data, t0, t1):
    f0 = project_data.raw_frames[t0]
    f1 = project_data.raw_frames[t1]
    df_gt = project_data.get_final_tracks_only_finished_neurons()[0]

    desc0 = f0.all_features
    desc1 = f1.all_features
    desc0 = torch.tensor(desc0).float()
    desc1 = torch.tensor(desc1).float()

    all_matches = torch.unsqueeze(torch.tensor(df_to_matches(df_gt, t0, t1)), dim=0)

    baseline_matches, conf, _ = calc_bipartite_from_positions(desc0, desc1)
    baseline_matches2, conf = calc_nearest_neighbor_matches(desc0, desc1, max_dist=1000.0)

    # Accuracy
    acc_baseline_bipartite = accuracy_of_matches(all_matches, baseline_matches)
    acc_baseline_greedy = accuracy_of_matches(all_matches, baseline_matches2)

    return acc_baseline_bipartite, acc_baseline_greedy


def test_baseline_and_new_matcher_on_embeddings(project_data, t0=0, t1=1):
    """
    Same as test_baseline_and_new_matcher_on_vgg_features, but uses the Superglue feature space instead of VGG

    Parameters
    ----------
    project_data
    t0
    t1

    Returns
    -------

    """
    from wbfm.utils.nn_utils.superglue import SuperGlue
    from wbfm.utils.nn_utils.worm_with_classifier import SuperglueFeatureSpaceTemplateMatcher
    f0 = project_data.raw_frames[t0]
    f1 = project_data.raw_frames[t1]
    df_gt = project_data.get_final_tracks_only_finished_neurons()[0]

    # Unpack
    tracker = SuperglueFeatureSpaceTemplateMatcher(f0)

    kpts0 = torch.tensor(f0.neuron_locs).float()
    kpts1 = torch.tensor(f1.neuron_locs).float()

    scores0 = torch.ones((kpts0.shape[0], 1)).float()
    scores1 = torch.ones((kpts1.shape[0], 1)).float()

    image0 = np.expand_dims(np.expand_dims(np.zeros_like(project_data.red_data[t0]), axis=0), axis=0)
    image1 = np.expand_dims(np.expand_dims(np.zeros_like(project_data.red_data[t1]), axis=0), axis=0)

    all_matches = torch.unsqueeze(torch.tensor(df_to_matches(df_gt, t0, t1)), dim=0)

    # Repack

    with torch.no_grad():
        desc0_embed = tracker.embedding_template.detach()
        desc1_embed = tracker.embed_target_frame(f1).detach()

    data = dict(descriptors0=desc0_embed, descriptors1=desc1_embed, keypoints0=kpts0, keypoints1=kpts1,
                all_matches=all_matches,
                image0=image0, image1=image1,
                scores0=scores0, scores1=scores1)
    model = SuperGlue(config=dict(descriptor_dim=120, match_threshold=0.0))

    out = model(data)
    new_matches = [[i, m0] for i, m0 in enumerate(out['matches0'].detach().numpy())]
    baseline_matches, conf, _ = calc_bipartite_from_positions(desc0_embed, desc1_embed)
    baseline_matches2, conf = calc_nearest_neighbor_matches(desc0_embed, desc1_embed, max_dist=1000.0)

    # Accuracy
    acc_new = accuracy_of_matches(all_matches, new_matches)
    acc_baseline_bipartite = accuracy_of_matches(all_matches, baseline_matches)
    acc_baseline_greedy = accuracy_of_matches(all_matches, baseline_matches2)

    return acc_new, acc_baseline_bipartite, acc_baseline_greedy


## Just the track-tracklet comparisons
def calc_tracklet_track_mismatch(project_data, to_plot=False):
    df_final = project_data.final_tracks.loc[:, (slice(None), 'raw_neuron_ind_in_list')].droplevel(level=1,
                                                                                                   axis=1).T.sort_index().T
    df_intermediate = project_data.intermediate_global_tracks.loc[:, (slice(None), 'raw_neuron_ind_in_list')].droplevel(
        level=1, axis=1)

    df_diff = (df_intermediate - df_final)
    num_t = df_diff.shape[0]

    df_nonequal_tracks = (df_diff != 0).apply(pd.value_counts).loc[True, :] / num_t
    df_nan_in_final = (num_t - df_final.count()) / num_t
    df_mismatched = df_nonequal_tracks - df_nan_in_final

    if to_plot:
        # plt.figure(dpi=150)
        # plt.plot(df_nonequal_tracks, label='nonequal')
        # plt.plot(df_mismatched, label='nonequal without nan')
        # plt.legend()
        # plt.xticks([]);

        plt.figure()
        plt.hist([df_nan_in_final, df_mismatched], label=['nan in final tracks', 'nonequal without nan'], bins=20);
        plt.legend()

    return df_nan_in_final, df_mismatched


def calc_accuracy_of_pipeline_steps(project_data_gcamp, remove_gt_nan=True, output_folder=None):
    neuron_names = project_data_gcamp.finished_neuron_names()
    if len(neuron_names) == 0:
        print("No finished neurons; quitting")
        return

    # Get the outputs of each pipeline step
    df_global = project_data_gcamp.intermediate_global_tracks[neuron_names]
    df_single_reference = project_data_gcamp.single_reference_frame_tracks(0)[neuron_names]
    df_pipeline = project_data_gcamp.initial_pipeline_tracks[neuron_names]
    df_gt = project_data_gcamp.final_tracks[neuron_names]

    # Get each accuracy
    opt = dict(column_names=['raw_neuron_ind_in_list'])
    df_acc_global = calculate_accuracy_from_dataframes(df_gt, df_global, **opt)
    df_acc_pipeline = calculate_accuracy_from_dataframes(df_gt, df_pipeline, **opt)
    df_acc_single_reference = calculate_accuracy_from_dataframes(df_gt, df_single_reference, **opt)

    if remove_gt_nan:
        col_name = 'matches_to_gt_nonnan'
    else:
        col_name = 'matches'

    df_acc = pd.DataFrame({'Single reference frame': df_acc_single_reference[col_name],
                           'Multiple reference frames': df_acc_global[col_name],
                           'Full pipeline': df_acc_pipeline[col_name]})

    fig = plt.figure(dpi=200)
    paired_boxplot_from_dataframes(df_acc.T, num_rows=3, fig=fig, add_median_line=False)

    plt.title("Tracklets significantly improve tracking quality")
    plt.ylabel("Fraction of correctly tracked points")
    apply_figure_settings(fig, width_factor=1, height_factor=0.5, plotly_not_matplotlib=False)

    if output_folder is not None:
        fname = os.path.join(output_folder, "paired_boxplot_trackers.png")
        plt.savefig(fname, transparent=True)
        fname = str(Path(fname).with_suffix('.svg'))
        plt.savefig(fname)

    return df_acc
