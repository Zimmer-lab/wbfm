import logging
from typing import Tuple

import numpy as np
import scipy.special
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

from wbfm.utils.general.distance_functions import dist2conf, calc_confidence_from_distance_array_and_matches


def calc_nearest_neighbor_matches(zxy0: np.ndarray,
                                  zxy1: np.ndarray,
                                  max_dist: float = None,
                                  n_neighbors: int = 1,
                                  softmax_for_confidence: bool = True):
    """
    Calculates single nearest neighbor matches

    NOT guaranteed to be one-to-one

    Parameters
    ----------
    softmax_for_confidence
    n_neighbors
    zxy0
    zxy1
    max_dist

    Returns
    -------
    matches_with_confidence

    """

    nan_val = 1e6
    zxy0 = np.nan_to_num(zxy0, nan=nan_val)
    zxy1 = np.nan_to_num(zxy1, nan=nan_val)

    algorithm = 'brute'
    neighbors_of_1 = NearestNeighbors(n_neighbors=n_neighbors, radius=max_dist, algorithm=algorithm).fit(zxy1)

    # Easier to just get the closest and postprocess the distance, vs. returning all neighbors in a ball and sorting
    all_dist, all_ind_1 = neighbors_of_1.kneighbors(zxy0, n_neighbors=n_neighbors)
    # all_dist, all_ind_1 = neighbors_of_1.radius_neighbors(zxy0, radius=max_dist)

    to_keep = all_dist < max_dist
    if n_neighbors == 1:
        all_ind_0 = np.array(range(len(all_ind_1)), dtype=int)
        all_ind_0 = all_ind_0[:, np.newaxis]
    else:
        # Then all_ind_1 is nested
        base_ind = np.array([1] * n_neighbors)
        all_ind_0 = np.array([base_ind * i for i in range(len(all_ind_1))])

    all_ind_0, all_ind_1, all_dist = all_ind_0[to_keep], all_ind_1[to_keep], all_dist[to_keep]
    # Doing the subset here automatically flattens the arrays
    matches = np.array([[i0, i1] for i0, i1 in zip(all_ind_0, all_ind_1)])
    if softmax_for_confidence and n_neighbors > 1:
        # Softmax doesn't make sense for only one distance
        conf = scipy.special.softmax(all_dist, axis=1)
    else:
        conf = dist2conf(all_dist)
    return matches, conf


def calc_matches_from_positions_using_softmax(query_embedding, trained_embedding):
    import torch
    distances = torch.cdist(trained_embedding, query_embedding)
    confidences = torch.softmax(torch.sigmoid(1.0 / distances), dim=1)
    i_trained, i_query = linear_sum_assignment(confidences, maximize=True)
    matches = list(zip(i_query, i_trained))
    return matches


def calc_bipartite_from_ids(xyz0: np.ndarray, xyz1: np.ndarray,
                                  gamma: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Uses scipy implementation of linear_sum_assignment to calculate best matches based on mismatch counts.

    Parameters
    ==============
    xyz0 - array-like; shape=(n0,m)
        Array of identifiers or values for the first set
    xyz1 - array-like; shape=(n1,m)
        Array of identifiers or values for the second set
    max_dist - float (default=1e9)
        Distance over which to remove matches (currently unused)
    gamma - float (default=1.0)
        Confidence scaling factor
    """
    import numba
    @numba.jit(nopython=True)
    def nandist(u, v):
        """
        Distance = sum of mismatches:
        - +1 for each element that differs
        - +1 if one is NaN and the other is not
        """
        dist = 0
        for i in range(u.shape[0]):
            if np.isnan(u[i]) and np.isnan(v[i]):
                continue
            if np.isnan(u[i]) or np.isnan(v[i]) or u[i] != v[i]:
                dist += 1
        return float(dist)

    cost_matrix = cdist(np.array(xyz0), np.array(xyz1), nandist)

    try:
        matches = linear_sum_assignment(cost_matrix)
    except ValueError:
        logging.warning("Value error: inf or nan detected in cost matrix.")
        raise ValueError

    raw_matches = [[m0, m1] for (m0, m1) in zip(matches[0], matches[1])]
    matches = np.array(raw_matches)

    # Confidence based on mismatch cost matrix
    conf = calc_confidence_from_distance_array_and_matches(cost_matrix, matches, gamma)

    return matches, conf, np.array(raw_matches)



def calc_bipartite_from_positions(xyz0: np.ndarray, xyz1: np.ndarray,
                                  max_dist: float = None,
                                  gamma: float = 1.0,
                                  try_to_fix_inf = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Uses scipy implementation of linear_sum_assignment to calculate best matches

    Parameters
    ==============
    xyz0 - array-like; shape=(n0,m)
        The 3d positions of a point cloud
        Note that m==3 is not required
    xyz1 - array-like; shape=(n1,m)
        The 3d positions of a second point cloud
    max_dist - float or None (default)
        Distance over which to remove matches

    """
    # ENHANCE: use sparse distance matrix: https://stackoverflow.com/questions/52366421/how-to-do-n-d-distance-and-nearest-neighbor-calculations-on-numpy-arrays
    if max_dist is None:
        max_dist = 1e6

    if not try_to_fix_inf:
        cost_matrix = cdist(np.array(xyz0), np.array(xyz1), 'euclidean')
    else:
        # Scipy can't deal with np.inf, so we want to maximize, not minimize
        # (And set impossible values to 0.0)
        # inv_cost_matrix = gamma / (cost_matrix + 1e-6)
        # np.where(inv_cost_matrix < (gamma / max_dist), 0.0, inv_cost_matrix)
        # inv_cost_matrix = np.nan_to_num(inv_cost_matrix)
        #
        # try:
        #     matches = linear_sum_assignment(inv_cost_matrix, maximize=True)
        # except ValueError:
        #     raise ValueError

        # Slower, so don't use by default
        import numba

        @numba.jit(nopython=True)
        def nandist(u, v):
            return np.sqrt(np.nansum((u - v) ** 2))
        cost_matrix = cdist(np.array(xyz0), np.array(xyz1), nandist)

    try:
        matches = linear_sum_assignment(cost_matrix)
    except ValueError:
        logging.warning("Value error probably means there were inf or nan detected; try try_to_fix_inf=True")
        raise ValueError

    raw_matches = [[m0, m1] for (m0, m1) in zip(matches[0], matches[1])]
    matches = raw_matches.copy()

    # Postprocess to remove distance matches
    # if max_dist is not None:
    #     match_dist = [cost_matrix[i, j] for (i, j) in matches]
    #     to_remove = [i for i, d in enumerate(match_dist) if d > max_dist]
    #     to_remove.reverse()
    #     [matches.pop(i) for i in to_remove]

    matches = np.array(matches)
    # if try_to_fix_inf:
    #     conf = calc_confidence_from_distance_array_and_matches(inv_cost_matrix, matches, use_dist2conf=False)
    # else:
    conf = calc_confidence_from_distance_array_and_matches(cost_matrix, matches, gamma)
    # conf = [conf_func(d) for d in match_dist]

    # Return matches twice to fit old function signature
    return matches, conf, np.array(raw_matches)


def filter_matches(matches, threshold):
    return [m for m in matches if m[2] > threshold]
