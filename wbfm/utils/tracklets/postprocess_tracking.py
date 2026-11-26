import warnings
from dataclasses import dataclass
from typing import List
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import dask
import numpy as np
from sklearn.preprocessing import StandardScaler
from ppca import PPCA
from sklearn.metrics.pairwise import nan_euclidean_distances
from tqdm.auto import tqdm

from wbfm.utils.general.high_performance_pandas import get_names_from_df


def normalize_3d(all_dist):
    # Normalize a t x n x n matrix by a neuron dimension
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        all_dist_mean = np.nanmean(all_dist, axis=0)
        all_dist_std = np.nanstd(all_dist, axis=0)
        all_dist_norm = (all_dist - all_dist_mean) / all_dist_std
    return all_dist_norm


@dataclass
class OutlierRemoval:
    names: List[str]

    ppca_dimension: int = 3  # Chosen via swaps on ground truth tracks
    learning_rate: float = 1.0
    std_threshold_min: float = 2.0
    std_threshold_factor: float = 3.0
    immediate_removal_threshold: float = 6.0

    num_outliers_tol: int = 20

    verbose: int = 1

    all_zxy_3d_original: np.ndarray = None
    df_traces: pd.DataFrame = None

    _all_zxy_3d: np.ndarray = None
    _all_dist_flattened: np.ndarray = None
    _all_dist: np.ndarray = None
    _all_dist_diff: np.ndarray = None
    total_matrix_to_remove: np.ndarray = None

    all_matrices_to_remove: List[np.ndarray] = None
    all_outlier_values: List[np.ndarray] = None

    def __post_init__(self):
        self.all_matrices_to_remove = []
        self.all_outlier_values = []

    @staticmethod
    def load_from_project(project_data, names=None, min_nonnan=None, verbose=0, **kwargs):

        coords = ['z', 'x', 'y']
        if names is None:
            names = project_data.well_tracked_neuron_names(min_nonnan=min_nonnan)
        df_traces = project_data.calc_default_traces(channel_mode='ratio', neuron_names=tuple(names))

        all_zxy = project_data.red_traces.loc[:, (names, coords)].copy()
        z_to_xy_ratio = project_data.physical_unit_conversion.z_to_xy_ratio
        all_zxy.loc[:, (slice(None), 'z')] = z_to_xy_ratio * all_zxy.loc[:, (slice(None), 'z')]

        obj = OutlierRemoval.load_from_arrays(all_zxy, coords, df_traces, names, verbose, **kwargs)
        return obj

    @staticmethod
    def load_from_arrays(all_zxy, coords, df_traces, names, verbose, **kwargs):
        all_zxy = all_zxy.sort_index(axis='columns')
        all_zxy_3d = all_zxy.to_numpy().reshape(all_zxy.shape[0], -1, len(coords))
        # Save
        obj = OutlierRemoval(names,
                             verbose=verbose,
                             df_traces=df_traces,
                             all_zxy_3d_original=all_zxy_3d.copy(),
                             _all_zxy_3d=all_zxy_3d,
                             **kwargs)
        obj.get_pairwise_distances()
        return obj

    @staticmethod
    def load_from_df(df_tracks, df_traces=None, verbose=0, **kwargs):

        coords = ['z', 'x', 'y']
        names = get_names_from_df(df_tracks, to_sort=False)

        all_zxy = df_tracks.loc[:, (names, coords)].copy()
        obj = OutlierRemoval.load_from_arrays(all_zxy, coords, df_traces, names, verbose, **kwargs)
        return obj

    def get_pairwise_distances(self):
        all_zxy_3d = self._all_zxy_3d
        # Get all pairwise distances
        output = []
        for time_zxy in all_zxy_3d:
            output.append(dask.delayed(nan_euclidean_distances)(time_zxy))
        total = dask.delayed(np.stack)(output)
        all_dist = total.compute()
        all_dist_flattened = all_dist.reshape(all_dist.shape[0], -1)
        self._all_dist_flattened = all_dist_flattened
        self._all_dist = all_dist
        return all_dist, all_dist_flattened

    def calc_outlier_indices_using_ppca(self) -> np.ndarray:
        ppca_dimension = self.ppca_dimension
        learning_rate = self.learning_rate
        verbose = self.verbose

        all_dist_flattened = self._all_dist_flattened
        all_dist = self._all_dist
        names = self.names

        if verbose > 1:
            print("Building low dimensional manifold...")
        scaler = StandardScaler()
        scaler.fit(all_dist_flattened)
        dat_normalized = scaler.transform(all_dist_flattened)

        # PPCA.fit silently drops columns (neuron pairs) with fewer than min_obs observations, e.g. two tracks that are
        # never present at the same time; its reconstruction then has fewer columns than the scaler, which used to
        # raise and skip outlier removal for the whole project. Fit on the kept columns and leave the others NaN.
        min_obs = 10
        valid_columns = np.sum(~np.isnan(dat_normalized), axis=0) >= min_obs
        ppca = PPCA()
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=RuntimeWarning)
            ppca.fit(data=dat_normalized, d=ppca_dimension, tol=0.02, min_obs=min_obs, verbose=(verbose > 1))

        if verbose > 1:
            print("Project data to that manifold...")
        full_time_reconstruction = np.full(dat_normalized.shape, np.nan)
        full_time_reconstruction[:, valid_columns] = np.dot(ppca.transform(), ppca.C.T)
        # V = ma.masked_invalid(ppca.C)
        # full_time_mode_weights = ma.dot(ma.masked_invalid(dat_normalized), V)
        # full_time_reconstruction = ma.dot(full_time_mode_weights, V.T)
        dat_imputed_flattened = scaler.inverse_transform(full_time_reconstruction)
        all_dist_imputed = dat_imputed_flattened.reshape(all_dist.shape)

        # Calculate the sum of distances over all paired neurons
        # But normalize the pairwise distances by the std across time (reduce importance of far away neurons)
        all_dist_imputed_norm = normalize_3d(all_dist_imputed)
        all_dist_norm = normalize_3d(all_dist)
        all_dist_diff = np.abs(all_dist_imputed_norm - all_dist_norm)
        # Normalize again (reduce importance of highly variable neurons)
        all_dist_diff = normalize_3d(all_dist_diff)

        # all_dist_diff = (all_dist_imputed - all_dist) ** 2.0
        # all_dist_diff = normalize_3d(all_dist_diff)

        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=RuntimeWarning)
            dat_outliers = np.nanmedian(all_dist_diff, axis=2)

        # Get individual outliers, and decide which to remove in this iteration
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=RuntimeWarning)
            all_thresholds = self.std_threshold_factor * np.nanstd(dat_outliers, axis=0)
            all_thresholds = np.stack([all_thresholds, self.std_threshold_min * np.ones(len(all_thresholds))])
            all_thresholds = np.max(all_thresholds, axis=0)

        matrix_to_remove = np.zeros_like(dat_outliers, dtype=bool)
        for i, name in enumerate(names):
            these_values = dat_outliers[:, i]
            # First immediately remove some
            ind_immediate = np.where(these_values - np.nanmean(these_values) > self.immediate_removal_threshold)[0]
            if len(ind_immediate) > 0:
                matrix_to_remove[ind_immediate, i] = True
                these_values[ind_immediate] = np.nan

            # Take the top percentage of the time points over the threshold
            num_outliers = np.sum(these_values - np.nanmean(these_values) > all_thresholds[i])
            if num_outliers == 0:
                continue
            num_to_remove = int(num_outliers*learning_rate)
            if num_to_remove == 0:
                num_to_remove = 1
            ind_sort = np.argsort(-these_values)
            ind_to_remove = ind_sort[:num_to_remove]

            matrix_to_remove[ind_to_remove, i] = True

            # if name == 'neuron_060':
            #     print(num_outliers, num_to_remove)

        self.all_matrices_to_remove.append(matrix_to_remove)
        self.all_outlier_values.append(dat_outliers)
        self._all_dist_diff = all_dist_diff
        if self.total_matrix_to_remove is None:
            self.total_matrix_to_remove = matrix_to_remove.copy()
        else:
            self.total_matrix_to_remove = self.total_matrix_to_remove | matrix_to_remove

        return matrix_to_remove

    def remove_outliers_from_zxy(self):
        matrix_to_remove = self.all_matrices_to_remove[-1]
        self._all_zxy_3d[matrix_to_remove, :] = np.nan

    def iteratively_remove_outliers_using_ppca(self, max_iter=8, DEBUG=False, DEBUG_name='neuron_017'):
        """

        Parameters
        ----------
        max_iter: Chosen based on ground truth swaps
        DEBUG
        DEBUG_name

        Returns
        -------

        """
        # Do not assume it was set up initially; start from all_zxy_3d
        for _ in tqdm(range(max_iter), leave=False):
            self.get_pairwise_distances()
            self.calc_outlier_indices_using_ppca()
            self.remove_outliers_from_zxy()

            if DEBUG:
                # self.plot_outlier_all_lines(DEBUG_name)
                self.plot_outlier_values(DEBUG_name)

            num_removed = np.sum(self.all_matrices_to_remove[-1])

            if self.verbose > 0:
                print(f"Removed {num_removed} outliers "
                      f"(total={np.sum(self.total_matrix_to_remove)})")

            if num_removed <= self.num_outliers_tol:
                if self.verbose > 0:
                    print("Reached tolerance")
                break
        else:
            if self.verbose > 0:
                print("Outlier removal ended before convergence (this is normal)")

    def plot_before_after(self, neuron_name, z_not_traces=True):
        i_trace = self.names.index(neuron_name)

        mask_remove = self.total_matrix_to_remove[:, i_trace]
        ind_remove = np.where(mask_remove)[0]

        if z_not_traces:
            y0 = self.all_zxy_3d_original[:, i_trace, 0]
            y1 = self._all_zxy_3d[:, i_trace, 0]
        else:
            y0 = self.df_traces[neuron_name]
            y1 = y0.copy()
            y1[mask_remove] = np.nan
        df = pd.DataFrame([y0, y1], index=['Original', 'Filtered']).T
        y_remove = y0[mask_remove]

        fig = px.line(df, title=f"Num removed = {len(ind_remove)}")
        # print(ind_remove)
        fig.add_trace(go.Scatter(x=ind_remove, y=y_remove, mode='markers'))
        fig.show()

    def plot_outlier_values(self, neuron_name):
        i_trace = self.names.index(neuron_name)

        trace = self.all_outlier_values[0][:, i_trace].copy()
        trace -= np.nanmean(trace)

        fig = px.line(trace)
        fig.add_hline(np.max([self.std_threshold_factor * np.nanstd(trace), self.std_threshold_min]))

        # Show which points were removed in which iteration
        for i, iteration_mask_remove in enumerate(self.all_matrices_to_remove):
            mask_remove = iteration_mask_remove[:, i_trace]
            y_remove = trace[mask_remove]
            ind_remove = np.where(mask_remove)[0]
            fig.add_trace(go.Scatter(x=ind_remove, y=y_remove, mode='markers', name=f'iteration {i}'))

        fig.show()

    def plot_outlier_all_lines(self, neuron_name):
        i_trace = self.names.index(neuron_name)

        trace_mat = self._all_dist_diff[:, i_trace, :].copy()
        trace_mat = trace_mat - np.nanmean(trace_mat, axis=0)
        trace_mat = pd.DataFrame(trace_mat, columns=self.names)

        mask_remove = self.total_matrix_to_remove[:, i_trace]
        y_remove = trace_mat[neuron_name][mask_remove]
        ind_remove = np.where(mask_remove)[0]

        fig = px.line(trace_mat, title=f"Num removed = {len(ind_remove)}")
        fig.add_hline(self.std_threshold_min)
        fig.add_trace(go.Scatter(x=ind_remove, y=y_remove, mode='markers'))
        fig.show()
