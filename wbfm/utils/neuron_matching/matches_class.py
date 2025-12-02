import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict

import networkx as nx
import numpy as np
from networkx import Graph, NetworkXError
from tqdm.asyncio import tqdm

from wbfm.utils.external.utils_pandas import accuracy_of_matches
from wbfm.utils.external.custom_errors import NoMatchesError
from wbfm.utils.general.distance_functions import dist2conf
from wbfm.utils.external.utils_neuron_names import int2name_neuron, int2name_using_mode
from scipy.optimize import linear_sum_assignment

from wbfm.utils.general.high_performance_pandas import get_names_from_df
from wbfm.utils.general.postprocessing.postprocessing_utils import distance_between_2_tracks, num_inliers_between_tracks


@dataclass
class MatchesWithConfidence:

    indices0: list = None
    indices1: list = None
    confidence: list = None

    dist2conf_gamma: float = None
    indices_have_offset: List[bool] = None
    int2name_funcs: List[callable] = None

    match_metadata: Dict = None

    @property
    def names0(self):
        if self.indices_have_offset[0]:
            return [self.int2name_funcs[0](i + 1) for i in self.indices0]
        else:
            return [self.int2name_funcs[0](i) for i in self.indices0]

    @property
    def names1(self):
        if self.indices_have_offset[1]:
            return [self.int2name_funcs[1](i + 1) for i in self.indices1]
        else:
            return [self.int2name_funcs[1](i) for i in self.indices1]

    def __post_init__(self):
        if self.indices0 is None:
            self.indices0 = []
        if self.indices1 is None:
            self.indices1 = []
        if self.confidence is None:
            self.confidence = []
        if self.int2name_funcs is None:
            self.int2name_funcs = [int2name_neuron, int2name_neuron]
        if self.match_metadata is None:
            self.match_metadata = {}

        try:
            # Should be two element list, but may be passed as a bool
            if len(self.indices_have_offset) == 1:
                self.indices_have_offset = [self.indices_have_offset[0], self.indices_have_offset[0]]
        except TypeError:
            self.indices_have_offset = [self.indices_have_offset, self.indices_have_offset]

    def add_match(self, new_match, metadata=""):
        assert len(new_match) == 3

        self.indices0.append(new_match[0])
        self.indices1.append(new_match[1])
        self.confidence.append(new_match[2])

        self.match_metadata[(new_match[0], new_match[1])] = metadata

    def match_already_exists(self, new_match):
        map0_1 = self.get_mapping_0_to_1(unique=False)
        return new_match[1] in map0_1[new_match[0]]

    def get_mapping_0_to_1(self, conf_threshold=0.0, unique=False):
        if unique:
            if self.confidence is None:
                return {n0: n1 for n0, n1 in zip(self.indices0, self.indices1)}
            else:
                return {n0: n1 for n0, n1, c in zip(self.indices0, self.indices1, self.confidence) if c > conf_threshold}
        else:
            mapping = defaultdict(list)
            if self.confidence is None:
                [mapping[n0].append(n1) for n0, n1 in zip(self.indices0, self.indices1)]
            else:
                [mapping[n0].append(n1) for n0, n1, c in zip(self.indices0, self.indices1, self.confidence)
                 if c > conf_threshold]
        return mapping

    def get_mapping_1_to_0(self, conf_threshold=0.0):
        if self.confidence is None:
            return {n1: n0 for n0, n1 in zip(self.indices0, self.indices1)}
        else:
            return {n1: n0 for n0, n1, c in zip(self.indices0, self.indices1, self.confidence) if c > conf_threshold}

    def get_mapping_0_to_1_names(self, conf_threshold=0.0):
        if self.confidence is None:
            return {n0: n1 for n0, n1 in zip(self.names0, self.names1)}
        else:
            return {n0: n1 for n0, n1, c in zip(self.names0, self.names1, self.confidence) if c > conf_threshold}

    def get_mapping_1_to_0_names(self, conf_threshold=0.0):
        if self.confidence is None:
            return {n1: n0 for n0, n1 in zip(self.names0, self.names1)}
        else:
            return {n1: n0 for n0, n1, c in zip(self.names0, self.names1, self.confidence) if c > conf_threshold}

    def get_mapping_pair_to_conf(self, conf_threshold=0.0):
        if self.confidence is None:
            return None
        else:
            return {(n0, n1): c for n0, n1, c in zip(self.indices0, self.indices1, self.confidence)
                    if c > conf_threshold}

    def get_mapping_0_to_1_with_unmatched_names(self, conf_threshold=0.0):
        # Generates a new name with suffix '_unmatched' if the neuron is found in index0, but has no match
        mapping = self.get_mapping_0_to_1_names(conf_threshold)
        update_mapping = {n0: f"{n0}_unmatched" for n0 in self.names0 if n0 not in mapping}
        mapping.update(update_mapping)
        return mapping

    def get_mapping_1_to_0_with_unmatched_names(self, conf_threshold=0.0):
        mapping = self.get_mapping_1_to_0_names(conf_threshold)
        update_mapping = {n1: f"{n1}_unmatched" for n1 in self.names1 if n1 not in mapping}
        mapping.update(update_mapping)
        return mapping

    def get_mapping_pair_to_conf_names(self, conf_threshold=0.0):
        if self.confidence is None:
            return None
        else:
            return {(n0, n1): c for n0, n1, c in zip(self.names0, self.names1, self.confidence)
                    if c > conf_threshold}

    def get_num_matches(self, conf_threshold=0.0):
        return len(self.get_mapping_0_to_1(conf_threshold))

    @property
    def array_matches_with_conf(self) -> np.ndarray:
        return np.array(np.stack([self.indices0, self.indices1, self.confidence], axis=1))

    @property
    def array_matches_without_conf(self) -> np.ndarray:
        return np.array(np.stack([self.indices0, self.indices1], axis=1))

    @staticmethod
    def matches_from_array(matches_with_conf, confidence=None, minimum_confidence=0.0, invalid_value=-1):
        """
        Initialize object from a nx2 or nx3 array that lists the matches. Confidence can be passed as an additional vector; otherwise nx3 shape is assumed with the 3rd column as the confidence

        Parameters
        ----------
        matches_with_conf
        confidence
        minimum_confidence
        invalid_value

        Returns
        -------

        """
        if matches_with_conf is None or len(matches_with_conf) == 0:
            raise NoMatchesError
        # if minimum_confidence > 0.0:
        #     ... this doesn't work for lists!
        #     assert matches_with_conf.shape[0] >= 2, "Must pass an array with confidence if using minimum_confidence"

        def _cast(vec, do_numpy):
            if do_numpy:
                return [int(m) for m in np.array(vec)]
            else:
                return [int(m) for m in vec]

        def _cast_both(mat, do_numpy):
            i1_vec, i2_vec = _cast(mat[:, 0], do_numpy), _cast(mat[:, 1], do_numpy)
            _keep_vec = [(_i1 != invalid_value) and (_i2 != invalid_value) for _i1, _i2 in zip(i1_vec, i2_vec)]
            if minimum_confidence > 0.0:
                _keep_vec = [k and (conf > minimum_confidence) for k, conf in zip(_keep_vec, mat[:, 2])]

            i1_out = [_i1 for _i1, keep in zip(i1_vec, _keep_vec) if keep]
            i2_out = [_i2 for _i2, keep in zip(i2_vec, _keep_vec) if keep]
            return i1_out, i2_out, _keep_vec

        try:
            i1, i2, keep_vec = _cast_both(matches_with_conf, do_numpy=False)
        except TypeError:
            try:
                i1, i2, keep_vec = _cast_both(matches_with_conf, do_numpy=True)
            except TypeError:
                matches_with_conf = np.array(matches_with_conf)
                i1, i2, keep_vec = _cast_both(matches_with_conf, do_numpy=True)
        if confidence is None:
            confidence = [c for c, keep in zip(matches_with_conf[:, 2], keep_vec) if keep]
        elif np.isscalar(confidence):
            confidence = confidence * np.ones(len(i1))

        return MatchesWithConfidence(i1, i2, confidence)

    @staticmethod
    def matches_from_list_of_gt_ids(gt_ids0, gt_ids1, confidence=None):
        if confidence is None:
            confidence = 1.0

        gt_matches = []
        for i, gt0 in enumerate(gt_ids0):
            if gt0 == -1:
                continue
            match_ind1 = np.where(gt_ids1 == gt0)[0]
            if len(match_ind1) > 0:
                match = (i, match_ind1[0])
                gt_matches.append(match)

        return MatchesWithConfidence.matches_from_array(gt_matches, confidence)

    @staticmethod
    def matches_from_distance_matrix(dist, gamma=1.0):
        row_i, col_i = linear_sum_assignment(dist)
        all_dist = [dist[i, j] for i, j in zip(row_i, col_i)]
        conf = dist2conf(np.array(all_dist), gamma=gamma)
        return MatchesWithConfidence(row_i, col_i, conf, gamma)

    @staticmethod
    def matches_from_bipartite_graph(graph: nx.Graph):
        """Assumes neurons and tracklets, and places neuron in class 0 """
        assert nx.is_bipartite(graph), "Graph must be bipartite!"
        matches = MatchesWithConfidence()

        for neuron, neuron_data in graph.nodes(data=True):
            if neuron_data['bipartite'] == 1:
                continue
            for tracklet, tracklet_data in dict(graph[neuron]).items():
                conf = graph.get_edge_data(neuron, tracklet)['weight']
                new_match = [neuron_data['metadata'], tracklet_data['metadata'], conf]
                matches.add_match(new_match)
        return matches

    def __repr__(self):
        return f"MatchesWithConfidence class with {len(self.get_mapping_0_to_1())} class A and " \
               f"{len(self.get_mapping_1_to_0())} class B matched objects, with {len(self.indices0)} edges"


class MatchesAsGraph(Graph):
    # ind2names: Dict[Tuple, str]  # Tuple notation is (frame #, neuron #)
    #
    # offset_convention: List[bool]  # Whether has offset or not
    # naming_convention: List[str]  # Will name nodes to keep them unique; can also be tracklet
    # name_prefixes: List[str]

    _raw2network_names: Dict[str, str]

    def __init__(self, ind2names=None, name_prefixes=None, naming_convention=None, offset_convention=None):
        if ind2names is None:
            ind2names = {}
        if name_prefixes is None:
            name_prefixes = ['frame', 'frame']
        if naming_convention is None:
            naming_convention = ['neuron', 'neuron']
        if offset_convention is None:
            offset_convention = [True, True]

        self._raw2network_names: Dict[str, str] = {}

        self.ind2names: Dict[Tuple, str] = ind2names
        self.name_prefixes: List[str] = name_prefixes
        # Will name nodes to keep them unique; can also be tracklet
        self.naming_convention: List[str] = naming_convention
        self.offset_convention: List[bool] = offset_convention

        super().__init__()

    @property
    def names2ind(self):
        return {v: k for k, v in self.ind2names.items()}

    def __contains__(self, n):
        if self.has_node(n):  # Calls the super class
            return True
        else:
            try:
                if self.has_node(self.tuple2name(n[0], n[1])):
                    return True
            except IndexError:
                return False
            finally:
                return False

    def tuple2name(self, bipartite_ind, local_ind, group_ind=None):
        """

        Parameters
        ----------
        bipartite_ind - Either 0 or 1; used with self.naming_convention and related fields
        ind - The second index; any integer, e.g. which neuron within a frame, or tracklet index
        group_ind - The first index, if different from bipartite_ind, e.g. which frame the neuron belongs to

        Returns
        -------

        Examples:
        if naming_convention == ['neuron', 'neuron']:
            bipartite_0_frame_1_neuron_001

        if subgroup_ind == 4:
            bipartite_1_frame_1_neuron_123

        """
        if self.offset_convention[bipartite_ind]:
            local_ind += 1
        naming_convention = self.naming_convention[bipartite_ind]
        if callable(naming_convention):
            name = naming_convention(local_ind)
        elif isinstance(naming_convention, str):
            name = int2name_using_mode(local_ind, naming_convention)
        else:
            raise TypeError(f"naming_convention must be callable or string; was {naming_convention}")
        prefix = f"bipartite_{bipartite_ind}_{self.name_prefixes[bipartite_ind]}"
        if group_ind is None:
            node_name = f"{prefix}_{bipartite_ind}_{name}"
        else:
            node_name = f"{prefix}_{bipartite_ind}_{name}"
        # NOTE: this name is actually generated new, thus may not be the true raw name
        # ... but should be, if the naming conventions are the same
        # self.node2raw_names[node_name] = name
        # INSTEAD: use the 'metadata' field of the nodes
        return node_name

    def name2tuple(self, name):
        # NOTE: doesn't tell you which bipartite element this came from
        # Sometimes this is the same as group_ind, but may not be
        n = self.nodes[name]
        bipartite_ind, group_ind, local_ind = n['bipartite'], n['group_ind'], n['local_ind']
        # bipartite_ind, group_ind, ind = int(name.split('_')[1]), int(name.split('_')[3]), int(name.split('_')[3])
        # if self.offset_convention[group_ind]:
        #     ind -= 1
        return bipartite_ind, group_ind, local_ind

    def add_match_if_not_present(self, new_match, group_ind0=0, group_ind1=1,
                                 node0_metadata=None,
                                 node1_metadata=None,
                                 edge_metadata=None,
                                 convert_ind_to_names=True):
        assert len(new_match) == 3

        n0, n1, conf = new_match

        if convert_ind_to_names:
            name0 = self.tuple2name(bipartite_ind=0, group_ind=group_ind0, local_ind=n0)
            name1 = self.tuple2name(bipartite_ind=1, group_ind=group_ind1, local_ind=n1)
        else:
            name0 = n0
            name1 = n1

        if self.has_edge(name0, name1):
            return False
        else:
            self.add_node(name0, bipartite=0, group_ind=group_ind0, local_ind=n0, metadata=node0_metadata)
            self.add_node(name1, bipartite=1, group_ind=group_ind0, local_ind=n0, metadata=node1_metadata)
            self.add_weighted_edges_from([(name0, name1, conf)], metadata=edge_metadata)
            return True

    def get_unique_match(self, group_and_ind=None, name=None):
        name = self.process_query(group_and_ind, name)

        matches = self.get_all_matches(name=name)
        if matches is None:
            return None
        elif len(matches) > 1:
            print("More than one match found")
            raise NotImplementedError
        else:
            return matches[0]

    def get_all_matches(self, group_and_ind=None, name=None):
        name = self.process_query(group_and_ind, name)
        try:
            return list(self.neighbors(name))
        except NetworkXError:
            logging.debug("No match found")
            return None

    def process_query(self, group_and_ind, name):
        if name is not None and group_and_ind is not None:
            print("Specify either the indices as a tuple, or the full name, not both")
            raise NotImplementedError
        if group_and_ind is not None:
            name = self.tuple2name(*group_and_ind)
        return name

    def raw_name_to_network_name(self, raw_name):
        # May be needed because I have some objects initialized with the old code
        # if self._raw2network_names is not None:
        #     if raw_name in self._raw2network_names:
        #         return self._raw2network_names[raw_name]
        # else:
        #     self._raw2network_names = {}

        # node_names = list(self)
        nodes = dict(self.nodes(data=True))
        for name, node in nodes.items():
            if raw_name == node['metadata']:
                # self._raw2network_names[raw_name] = name
                return name
        else:
            return None

    def network_name_to_raw_name(self, network_name):
        return dict(self.nodes(data=True))[network_name]['metadata']

    def get_nodes_of_class(self, i_class):
        return {n for n, d in self.nodes(data=True) if d["bipartite"] == i_class}

    def draw(self):
        top = self.get_nodes_of_class(0)
        pos = nx.bipartite_layout(self, top)
        nx.draw(self, pos=pos)

    def __repr__(self):
        return f"MatchesAsGraph object with {len(self.get_nodes_of_class(0))} class A nodes and " \
               f"{len(self.get_nodes_of_class(1))} class B nodes, with {len(self.edges)} edges"

    def pretty_print_single_match(self, node):
        dat = dict(self[node])
        print(f"{len(dat)} matches for {node}")
        for key, val in dat.items():
            print(f"{key} with weight={val['weight']}")


def get_mismatches(gt_matches: MatchesWithConfidence, model_matches: MatchesWithConfidence, verbose=0):
    """
    Get mismatches of different types:
    1. Match is different between model and gt:
        1a. The match that the model gave
        1b. The match that the gt gave
    2. Match doesn't exist in gt, but does in model
    3. Match exists in gt, but doesn't in model

    Parameters
    ----------
    gt_matches
    model_matches

    Returns
    -------
    gt_matches_different_model
    model_matches_different_gt
    model_matches_no_gt
    gt_matches_no_model

    """

    dict_of_model_matches = model_matches.get_mapping_0_to_1(unique=True)
    dict_of_gt_matches = gt_matches.get_mapping_0_to_1(unique=True)
    list_of_model_matches: List[list] = model_matches.array_matches_without_conf.tolist()
    list_of_gt_matches: List[list] = gt_matches.array_matches_without_conf.tolist()
    inverse_dict_of_model_matches = model_matches.get_mapping_1_to_0()

    model_matches_no_gt = []
    correct_matches = []
    gt_matches_no_model = []
    gt_matches_different_model = []
    model_matches_different_gt = []

    for gt_m in list_of_gt_matches:
        if gt_m in list_of_model_matches:
            if verbose >= 3:
                print(f"{gt_m} in {list_of_model_matches}")
            correct_matches.append(gt_m)
            continue
        elif gt_m[0] != inverse_dict_of_model_matches.get(gt_m[1], gt_m[0]):
            # The first time point had the wrong match
            gt_matches_different_model.append(gt_m)
            model_matches_different_gt.append([inverse_dict_of_model_matches[gt_m[1]], gt_m[1]])
        elif gt_m[1] != dict_of_model_matches.get(gt_m[0], gt_m[1]):
            # The second time point had the wrong match
            gt_matches_different_model.append(gt_m)
            model_matches_different_gt.append([gt_m[0], dict_of_model_matches[gt_m[0]]])
        else:
            # Rare; usually the model has many more matches
            gt_matches_no_model.append(gt_m)

    for model_m in list_of_model_matches:
        if model_m[0] not in dict_of_gt_matches:
            model_matches_no_gt.append(model_m)

    return correct_matches, gt_matches_different_model, model_matches_different_gt, model_matches_no_gt, gt_matches_no_model


def accuracy_of_matches_from_classes(gt_matches: MatchesWithConfidence, model_matches: MatchesWithConfidence):
    gt_m = gt_matches.array_matches_without_conf
    model_m = model_matches.array_matches_without_conf

    return accuracy_of_matches(gt_m, model_m)


def matches_between_tracks(df1, df2, user_inlier_mode=False,
                           dist2conf_gamma=1.0,
                           inlier_gamma=10.0) -> MatchesWithConfidence:
    coords = ['z', 'x', 'y']

    # Find matches between neuron names
    leifer_names = get_names_from_df(df1)
    track_names = get_names_from_df(df2)

    zxy1 = [df1[name][coords].to_numpy() for name in leifer_names]
    zxy2 = [df2[name][coords].to_numpy() for name in track_names]

    num_i, num_j = len(zxy1), len(zxy2)
    all_dist = np.zeros((num_i, num_j))

    # Have to do a custom loop because the input is 3d and cdist crashes
    if user_inlier_mode:
        f = lambda i, j: distance_between_2_tracks(zxy1[i], zxy2[j])
    else:
        f = lambda i, j: inlier_gamma / num_inliers_between_tracks(zxy1[i], zxy2[j])
    for i in tqdm(range(num_i), leave=False):
        for j in range(num_j):
            all_dist[i, j] = f(i, j)

    matches_with_conf = MatchesWithConfidence.matches_from_distance_matrix(all_dist, gamma=dist2conf_gamma)

    return matches_with_conf
