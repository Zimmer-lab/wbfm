from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.community import k_clique_communities
from tqdm.auto import tqdm

from wbfm.utils.general.utils_networkx import calc_bipartite_matches_using_networkx, build_digraph_from_matches, \
    unpack_node_name, is_one_neuron_per_frame
from wbfm.utils.neuron_matching.utils_matching import calc_bipartite_from_positions, calc_bipartite_from_ids
from scipy.sparse import coo_matrix
from wbfm.utils.general.high_performance_pandas import get_names_from_df


##
## Convinience function
##

def calc_all_bipartite_matches(candidates, min_edge_weight=0.5):
    """
    Multi-pair wrapper around calc_bipartite_matches

    Assumes 'candidates' is a dictionary of all 3-element candidates for each pairing
        Last element is the weight (confidence)
    """
    bp_match_dict = {}
    for key in candidates:
        these_candidates = [c for c in candidates[key] if c[-1] > min_edge_weight]
        bp_matches = calc_bipartite_matches_using_networkx(these_candidates)
        bp_match_dict[key] = bp_matches

    return bp_match_dict


##
## Build communities from large network of matches
##

def calc_neurons_using_k_cliques(all_matches,
                                 k_values=[5, 4, 3],
                                 list_min_sizes=[450, 400, 350, 300, 250],
                                 max_size=500,
                                 min_conf=0.0,
                                 verbose=1):
    # Do a list of descending clique sizes
    G = build_digraph_from_matches(all_matches, verbose=0, min_conf=min_conf).to_undirected()

    # Precompute cliques... doesn't work if nodes are removed
    # all_cliques = list(nx.find_cliques(G))

    all_communities = []
    # Multiple passes: take largest communities first
    for min_size in list_min_sizes:
        for k in k_values:
            communities = list(k_clique_communities(G, k=k))  # , cliques=all_cliques))
            nodes_to_remove = []
            for c in communities:
                if len(c) > min_size and len(c) < max_size:
                    nodes_to_remove.extend(c)
                    all_communities.append(c)
            G.remove_nodes_from(nodes_to_remove)
            if verbose >= 1:
                print(f"{len(G.nodes)} nodes remaining")
        max_size = min_size

    return all_communities


def calc_neuron_using_voronoi(all_matches,
                              dist,
                              total_frames,
                              target_size_vec=None,
                              verbose=0):
    # Cluster using voronoi cells
    DG = build_digraph_from_matches(all_matches, dist, verbose=0)
    # Indices may not start at 0
    # Syntax: https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
    all_pairs = all_matches.keys()
    unique_nodes = set(node for pair in all_pairs for node in pair)

    global2local = {}
    global_current_ind = 0
    if target_size_vec is None:
        stop_size = max(1, int(total_frames / 2))
        target_size_vec = list(range(total_frames, stop_size, -1))

    for target_size in target_size_vec:
        for start_vol in unique_nodes:
            # Get simple centers: all neurons in a "start" volume
            center_nodes = []
            for n in DG.nodes():
                frame_ind, _ = unpack_node_name(n)
                if frame_ind == start_vol:
                    center_nodes.append(n)
            if len(center_nodes) == 0:
                continue
            cells = nx.voronoi_cells(DG, center_nodes)

            # Heuristic
            # If the cells have a unique node in each frame, then take it as true
            # ENHANCE: removal of outliers
            for k, v in cells.items():
                if is_one_neuron_per_frame(v, min_size=target_size, total_frames=total_frames):
                    global2local[global_current_ind] = v
                    global_current_ind += 1
                    DG.remove_nodes_from(v)
            if verbose >= 2:
                print(f"{len(DG)} nodes remaining (across all frames)")
        if verbose >= 1:
            print(f"Found {global_current_ind} neurons of size at least {target_size}")

    return global2local


##
## Utilities
##

##
## Networkx conversion
##

def convert_labels_to_matches(labels, offset=None, max_frames=None, DEBUG=False):
    """
    Turns a dict of classes per neuron (labels) into framewise matches
        Note: not every neuron needs to be labeled
    Format:
        labels[node_ind] = global_neuron_label

    Assumes the node indices can be unpacked using unpack_node_name()
    """

    match_dict = defaultdict(list)
    unique_labels = np.unique(list(labels.values()))
    if DEBUG:
        print(unique_labels)

    for name in unique_labels:
        # Get nodes of this class
        these_ind = []
        for node_ind, lab in labels.items():
            if lab != name:
                continue
            frame_ind, local_ind = unpack_node_name(node_ind)
            these_ind.append((frame_ind, local_ind))
        if DEBUG:
            print(these_ind)
        # Build matches that know the starting frame
        for i_f0, i_l0 in these_ind:
            for i_f1, i_l1 in these_ind:
                if i_l0 == i_l1 and i_f0 == i_f1:
                    continue
                if offset is not None:
                    k = (i_f0 - offset, i_f1 - offset)
                else:
                    k = (i_f0, i_f1)
                if max_frames is not None:
                    if k[0] >= max_frames or k[1] >= max_frames:
                        continue
                match_dict[k].append([i_l0, i_l1])

    return match_dict


def community_to_matches(all_communities):
    """See calc_neurons_using_k_cliques()"""

    community_dict = {}
    for i, c in enumerate(all_communities):
        name = f"neuron_{i}"
        for neuron in c:
            community_dict[neuron] = name
    clique_matches = convert_labels_to_matches(community_dict, offset=50, max_frames=500)

    return clique_matches


def fix_candidates_without_confidences(candidates):
    """
    All candidate matches should be 3d, i.e. (node0, node1, confidence)
    However, the gaussian process formatting currently doesn't output a confidence
    Also: sometimes the node indices are cast as floats, when they should be ints
    """
    new_candidates = {}
    for k, these_matches in candidates.items():
        new_matches = []
        for m in these_matches:
            if len(m) == 3:
                m = (int(m[0]), int(m[1]), m[2])
            else:
                m = (int(m[0]), int(m[1]), 1.0)
            new_matches.append(m)
        new_candidates[k] = new_matches
    return new_candidates


def matches_to_sparse_matrix(matches_with_conf, shape=None):
    matches_with_conf = np.array([np.array(m) for m in matches_with_conf])
    row, col, data = \
        matches_with_conf[:, 0].astype(int), matches_with_conf[:, 1].astype(int), matches_with_conf[:, 2].astype(float)
    if shape is None:
        shape = (max(row)+1, max(col)+1)
    return coo_matrix((data, (row, col)), shape=shape, dtype=float)


def rename_columns_using_matching(df_base, df_to_rename, column='raw_neuron_ind_in_list',
                                  try_to_fix_inf=False):
    """
    Aligns the names of df_to_rename with the names of df_base based on bipartite matching on values of 'column' variable
    Drops columns without a match

    Note: can't really handle nan or inf values in either matrix unless try_to_fix_inf=True

    Parameters
    ----------
    df_base
    df_to_rename
    column
    try_to_fix_inf

    Returns
    -------
    df1_renamed
    matches - list of 2-element matches
    conf
    name_mapping - neuron name dict of 'matches', with:
        key = old name of df_to_rename
        value = new name (same as df_base)
    """

    names0 = get_names_from_df(df_base)
    names1 = get_names_from_df(df_to_rename)

    # Note: the bipartite index has to match the name indices
    df0_ind = df_base.loc(axis=1)[names0, column].to_numpy()
    df1_ind = df_to_rename.loc(axis=1)[names1, column].to_numpy()

    if column=="raw_neuron_ind_in_list" or column=="raw_segmentation_id":
        matches, conf, _ = calc_bipartite_from_ids(df1_ind.T, df0_ind.T)
    else:
        matches, conf, _ = calc_bipartite_from_positions(df1_ind.T, df0_ind.T, try_to_fix_inf=try_to_fix_inf)

    # Start with default
    name_mapping = {n: 'unmatched_neuron' for n in names1}
    for m in matches:
        name_mapping[names1[m[0]]] = names0[m[1]]

    # Note: drops columns with no match!
    df1_renamed = df_to_rename.rename(columns=name_mapping, inplace=False)
    if len(names1) > len(names0):
        df1_renamed.drop(columns='unmatched_neuron', inplace=True)

    return df1_renamed, matches, conf, name_mapping


def combine_dataframes_using_max_of_column(df0, df1, column='likelihood'):
    names0 = get_names_from_df(df0)
    names1 = get_names_from_df(df1)
    shared_names = list(set(names0).intersection(names1))

    df0_like = df0.loc(axis=1)[shared_names, column].to_numpy()
    df1_like = df1.loc(axis=1)[shared_names, column].to_numpy()
    comparison = df1_like > df0_like

    new_df = df0.copy()
    for i_col, name in enumerate(shared_names):
        ind_mask = comparison[:, i_col]

        new_df.loc[ind_mask, name] = df1.loc[ind_mask, name].values

    return new_df


def combine_dataframes_using_mode(all_dfs, column='raw_neuron_ind_in_list', i_base=0):
    """Assumes that all_dfs[i_base] is the base dataframe, and all others are to be combined into it; only works well with >2 dataframes"""
    names = get_names_from_df(all_dfs[i_base])
    new_df = all_dfs[i_base].copy()

    for neuron_name in tqdm(names, leave=False):

        dfs_containing_this_neuron = [df for df in all_dfs if neuron_name in df]
        df_tmp_names = [f'df_{i}' for i in range(len(dfs_containing_this_neuron))]

        # Get argmode
        df_ids = pd.DataFrame(
            {n: df.loc[:, (neuron_name, column)].values for n, df in zip(df_tmp_names, dfs_containing_this_neuron)})
        column_id_mode = df_ids.mode(axis=1)[0]
        df_id_mode = pd.DataFrame({n: column_id_mode.values for n in df_tmp_names})

        df_argmode = df_id_mode == df_ids

        # Update this neuron with the voted-on index
        new_df_one_neuron = new_df[neuron_name].copy()  # Copy to prevent warnings
        for df, tmp_name in zip(dfs_containing_this_neuron[1:], df_tmp_names[1:]):
            col_mask = df_argmode[tmp_name]

            new_df_one_neuron.loc[col_mask] = df.loc[col_mask, neuron_name].values
        new_df[neuron_name] = new_df_one_neuron

    return new_df


def combine_dataframes_using_bipartite_matching(all_dfs, column='raw_neuron_ind_in_list', i_base=0):
    """
    Combines a list of dataframes using time-slice bipartite matching. Does not use likelihood, but simple voting

    Note that the names must be aligned (as best they can be), probably using rename_columns_using_matching
    """
    names = get_names_from_df(all_dfs[i_base])
    new_df = all_dfs[i_base].copy()

    graphs_all_times = defaultdict(nx.Graph)
    for i, neuron_name in enumerate(tqdm(names)):

        dfs_containing_this_neuron = [df for df in all_dfs if neuron_name in df]

        for df in dfs_containing_this_neuron:
            ind_at_all_time = df.loc[:, (neuron_name, column)].values
            for t, ind in enumerate(ind_at_all_time):
                if np.isnan(ind):
                    continue
                node_name = int(ind)
                edge = [neuron_name, node_name]
                g = graphs_all_times[t]
                if neuron_name in g and node_name in g[neuron_name]:
                    g[neuron_name][node_name]['weight'] += 1
                else:
                    g.add_edge(*edge, weight=1)

    for t, g in tqdm(graphs_all_times.items()):
        top_nodes = [n for n in g if type(n) == str and 'neuron' in n]
        matching = nx.bipartite.maximum_matching(g, top_nodes=top_nodes)

        for name, ind in matching.items():
            # Note: matching includes both directions
            if type(name) != str:
                continue
            ind = matching[name]
            new_df.loc[t, (name, column)] = ind

    return new_df


def combine_and_rename_multiple_dataframes(all_raw_dfs, i_base):
    df_base = all_raw_dfs[i_base]
    all_dfs = [df_base]
    for i, df in enumerate(all_raw_dfs):
        if i == i_base:
            continue
        df_renamed, *_ = rename_columns_using_matching(df_base, df, try_to_fix_inf=True)
        all_dfs.append(df_renamed)
    # Combine to one dataframe
    if len(all_dfs) > 1:
        # df_combined = combine_dataframes_using_mode(all_dfs)
        df_combined = combine_dataframes_using_bipartite_matching(all_dfs)
    else:
        df_combined = all_dfs[0]
    return df_combined


def fit_umap_using_frames(all_frames):
    print("Pretraining UMAP for global space embedding")
    from umap import UMAP
    X_all_neurons = []

    for f in all_frames.values():
        if f.all_features is not None:
            X_all_neurons.append(f.all_features)

    X_all_neurons = np.vstack(X_all_neurons)
    opt_umap = dict(n_components=10, n_neighbors=10, min_dist=0)
    umap = UMAP(**opt_umap)
    umap.fit(X_all_neurons)
    return umap
