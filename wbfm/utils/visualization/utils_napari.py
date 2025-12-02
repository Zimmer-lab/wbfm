from ast import Index
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt

from wbfm.utils.external.utils_pandas import cast_int_or_nan
from wbfm.utils.general.high_performance_pandas import get_names_from_df
from wbfm.utils.external.utils_neuron_names import name2int_neuron_and_tracklet

from wbfm.utils.external.bleach_correction import detrend_exponential_iter
from sklearn.linear_model import LinearRegression


def napari_labels_from_traces_dataframe(df, neuron_name_dict=None, label_using_column_name=False,
                                        z_to_xy_ratio=1, automatic_label_by_default=True, include_time=True,
                                        DEBUG=False):
    """
    Expects dataframe with positions, with column names either:
        legacy format: ['z_dlc', 'x_dlc', 'y_dlc']
        current format: ['z', 'x', 'y']

        And optionally: 'i_reindexed_segmentation' or 'label'
        (note: additional columns do not matter)

    Returns napari-ready format:
        A dict of options, with a nested dict 'properties' and a list 'data'
        'properties' has one entry, 'labels' = long list with all points at all time
        'dat' is a list of equal length with all the dimensions (tzxy)

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe with positions and labels
    neuron_name_dict: dict
        Dictionary with neuron names as keys and custom names as values
    label_using_column_name: bool
        If True, uses the column name as the label. Otherwise tries to detect the label from the dataframe
    z_to_xy_ratio: float
        Ratio of z to xy dimensions (for proper visualization in napari)
    automatic_label_by_default: bool
        If True, napari uses the automatic label as the text. Otherwise uses the custom label

    Returns
    -------

    """
    df.replace(0, np.NaN, inplace=True)  # DLC uses all zeros as failed tracks

    if neuron_name_dict is None:
        neuron_name_dict = {}
    all_neurons = get_names_from_df(df)
    t_vec = np.expand_dims(np.array(list(df.index), dtype=int), axis=1)
    # label_vec = np.ones(len(df.index), dtype=int)
    if include_time:
        all_t_zxy = np.array([[0, 0, 0, 0]], dtype=int)
    else:
        all_t_zxy = np.array([[0, 0, 0]], dtype=int)
    properties = dict(automatic_label=[], custom_label=[])
    for n in all_neurons:
        coords = ['z', 'x', 'y']
        zxy = np.array(df[n][coords]).astype(float)
        # Note that this messes up the 2d view, because z values will be in-between real planes
        zxy[:, 0] *= z_to_xy_ratio
        if include_time:
            t_zxy = np.hstack([t_vec, zxy])
        else:
            t_zxy = zxy

        # Add two label fields: one for the automatic label, and one for the (optional) custom label
        this_gt_name = neuron_name_dict.get(n, '')
        if 'neuron_' in this_gt_name:
            this_gt_name = this_gt_name.split('_')[1]
        label_vec_gt = [this_gt_name] * len(df.index)
        if DEBUG:
            print(f"Found named neuron: {n} = {label_vec_gt[0]}")
        properties['custom_label'].extend(label_vec_gt)

        # Get the index from the dataframe, or try to convert the column name into a label
        if label_using_column_name:
            label_vec = [name2int_neuron_and_tracklet(n) for _ in range(t_vec.shape[0])]
        elif 'i_reindexed_segmentation' in df[n]:
            # For old style
            label_vec = list(map(int, df[n]['i_reindexed_segmentation']))
        elif 'label' in df[n]:
            # For traces dataframe
            label_vec = [i for i in df[n]['label']]
        elif 'raw_neuron_ind_in_list' in df[n]:
            # For tracks dataframe
            label_vec = [i for i in df[n]['raw_neuron_ind_in_list']]
        else:
            label_vec = [name2int_neuron_and_tracklet(n) for _ in range(t_vec.shape[0])]
            # raise ValueError("Could not find a label column in the dataframe, and label_using_column_name is False")
        properties['automatic_label'].extend(label_vec)

        # This should synchronize with any label fields
        all_t_zxy = np.vstack([all_t_zxy, t_zxy])

    # Remove invalid positions
    # Some points are negative instead of nan
    all_t_zxy = np.where(all_t_zxy < 0, np.nan, all_t_zxy)
    to_keep = ~np.isnan(all_t_zxy).any(axis=1)
    all_t_zxy = all_t_zxy[to_keep, :]
    all_t_zxy = all_t_zxy[1:, :]  # Remove dummy starter point
    for key in properties.keys():
        properties[key] = [p for p, good in zip(properties[key], to_keep[1:]) if good]
    # properties['automatic_label'] = [p for p, good in zip(properties['automatic_label'], to_keep[1:]) if good]
    # properties['custom_label'] = [p for p, good in zip(properties['custom_label'], to_keep[1:]) if good]
    # Additionally remove invalid names
    try:
        to_keep = np.array([not np.isnan(p) for p in properties['automatic_label']])
        all_t_zxy = all_t_zxy[to_keep, :]
        properties['automatic_label'] = [cast_int_or_nan(p) for p, good in zip(properties['automatic_label'], to_keep) if good]
        properties['custom_label'] = [p for p, good in zip(properties['custom_label'], to_keep) if good]
    except (TypeError, IndexError):
        # Then the user is passing a non-int custom name, so just skip this
        pass
    # More info on text: https://github.com/napari/napari/blob/main/examples/add_points_with_text.py
    # If additional properties are added, can be accessed with fstring syntax
    label_str = '{automatic_label}' if automatic_label_by_default else '{custom_label}'
    text = {'string': label_str}
    # Final package
    options = {'data': all_t_zxy, 'face_color': 'transparent', 'edge_color': 'transparent',
               'text': text,  # Can add color or size here
               'properties': properties, 'name': 'Neuron IDs', 'blending': 'additive',
               'visible': False}
    return options


@dataclass
class NapariPropertyHeatMapper:
    """
    Builds dictionaries to map segmentation labels to various neuron properties (e.g. average or max brightness)

    NOTE: any custom values should be sorted according to names
    """

    red_traces: pd.DataFrame
    green_traces: pd.DataFrame
    curvature_fluorescence_fps: pd.DataFrame = pd.DataFrame([np.nan])

    names: List[str] = None

    @property
    def vec_of_labels(self):
        return np.nanmean(self.df_labels.to_numpy(), axis=0).astype(int)

    @property
    def df_labels(self) -> pd.DataFrame:
        return self.red_traces.loc[:, (self.names, 'label')]

    @property
    def mean_red(self):
        tmp1 = self.red_traces.loc[:, (self.names, 'intensity_image')]
        tmp1.columns = self.names
        tmp2 = self.red_traces.loc[:, (self.names, 'area')]
        tmp2.columns = self.names
        return tmp1 / tmp2

    @property
    def mean_green(self):
        tmp1 = self.green_traces.loc[:, (self.names, 'intensity_image')]
        tmp1.columns = self.names
        tmp2 = self.green_traces.loc[:, (self.names, 'area')]
        tmp2.columns = self.names
        return tmp1 / tmp2

    def corrcoef_kymo(self):
        if self.curvature(fluorescence_fps=True) .isnull().values.all():
            return [np.nan]

        if not self.curvature(fluorescence_fps=True) .isnull().values.all():
            corrcoefs = []
            for neuron in self.names:
                vector = np.abs(np.corrcoef(self.curvature(fluorescence_fps=True) .assign(
                    neuron_to_test=self.red_traces[neuron]["intensity_image"]).dropna(axis="rows").T)[100, :99])
                c = np.max(vector)
                corrcoefs.append(c)
            val_to_plot = corrcoefs
            return property_vector_to_colormap(val_to_plot, self.vec_of_labels)

    def anchor_corr_red(self, anchor="neuron_028"):
        corrcoefs = []
        anchor_trace_raw = detrend_exponential_iter(self.red_traces[anchor]["intensity_image"])[0]
        for neuron in self.names:
            try:
                neuron_trace_raw = detrend_exponential_iter(self.red_traces[neuron]["intensity_image"])[0]
                remove_nan = np.logical_and(np.invert(np.isnan(anchor_trace_raw)),
                                            np.invert(np.isnan(neuron_trace_raw)))
                anchor_trace = anchor_trace_raw[remove_nan]
                neuron_trace = neuron_trace_raw[remove_nan]
                vol_anchor = self.red_traces[anchor]["area"][remove_nan]
                vol_neuron = self.red_traces[neuron]["area"][remove_nan]

                model_anchor = LinearRegression()
                model_anchor.fit(np.array(vol_anchor).reshape(-1, 1), anchor_trace)
                anchor_trace_corrected = anchor_trace - model_anchor.predict(np.array(vol_anchor).reshape(-1, 1))

                model_neuron = LinearRegression()
                model_neuron.fit(np.array(vol_neuron).reshape(-1, 1), neuron_trace)
                neuron_trace_corrected = neuron_trace - model_neuron.predict(np.array(vol_neuron).reshape(-1, 1))

                corrcoefs.append(np.corrcoef(anchor_trace_corrected, neuron_trace_corrected)[0][1])
            except ValueError:
                print(neuron, "skipped")
                corrcoefs.append(0)

            val_to_plot = np.array(corrcoefs)

        return property_vector_to_colormap(val_to_plot, self.vec_of_labels)

    def count_nonnan(self) -> Dict[int, float]:
        num_nonnan = self.df_labels.count()
        val_to_plot = np.array(num_nonnan) / self.df_labels.shape[0]
        return property_vector_to_colormap(val_to_plot, self.vec_of_labels)

    def max_of_red(self):
        val_to_plot = list(self.mean_red.max())
        return property_vector_to_colormap(val_to_plot, self.vec_of_labels, scale_to_minus_1_and_1=False)

    def std_of_red(self):
        val_to_plot = list(self.mean_red.std())
        return property_vector_to_colormap(val_to_plot, self.vec_of_labels, scale_to_minus_1_and_1=False)

    def max_of_green(self):
        val_to_plot = list(self.mean_green.max())
        return property_vector_to_colormap(val_to_plot, self.vec_of_labels, scale_to_minus_1_and_1=False)

    def std_of_green(self):
        val_to_plot = list(self.mean_green.std())
        return property_vector_to_colormap(val_to_plot, self.vec_of_labels, scale_to_minus_1_and_1=False)

    def max_of_ratio(self):
        val_to_plot = list((self.mean_green / self.mean_red).max())
        return property_vector_to_colormap(val_to_plot, self.vec_of_labels, scale_to_minus_1_and_1=False)

    def std_of_ratio(self):
        val_to_plot = list((self.mean_green / self.mean_red).std())
        return property_vector_to_colormap(val_to_plot, self.vec_of_labels, scale_to_minus_1_and_1=False)

    def custom_val_to_plot(self, val_to_plot: pd.Series, scale_to_minus_1_and_1):
        prop_dict = property_vector_to_colormap(val_to_plot, self.vec_of_labels,
                                                scale_to_minus_1_and_1=scale_to_minus_1_and_1)
        return prop_dict


def property_vector_to_colormap(val_to_plot, vec_of_labels, cmap=plt.cm.RdBu,
                                scale_to_minus_1_and_1=True, create_colorbar=False) -> Dict[int, float]:
    """
    Takes a vector of values and a vector of labels, and returns a dictionary with the labels as keys and the values
    as colors

    The colormap is scaled assuming that the values are in the range [-1, 1], e.g. a correlation coefficient
        In this case, white = 0
    Alternatively, if scale_to_minus_1_and_1 is False, the values are assumed to be in the range [xmin, xmax],
    and white has a special meaning only if xmin < 0 and xmax > 0

    Parameters
    ----------
    val_to_plot
    vec_of_labels
    cmap

    Returns
    -------

    """
    prop = np.array(val_to_plot)
    # Check if the values are already in the range [-1, 1]
    if scale_to_minus_1_and_1:
        if np.nanmax(prop) > 1 or np.nanmin(prop) < -1:
            raise ValueError("Values should be in the range [-1, 1] for this colormap")
        # matplotlib cmaps need values in [0, 1]
        prop_scaled = (prop + 1) / 2
        # Use TwoSlopeNorm to ensure white is at 0; only used for colorbar
        scaler = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        
    elif np.nanmax(prop) > 0 > np.nanmin(prop):
        # Then we have a special case where white is 0
        scaler = mcolors.TwoSlopeNorm(vmin=np.nanmin(prop), vcenter=0, vmax=np.nanmax(prop))
        prop_scaled = scaler(prop)
    else:
        # Then white doesn't have a special meaning
        scaler = mcolors.Normalize(vmin=np.nanmin(prop), vmax=np.nanmax(prop))
        prop_scaled = scaler(prop)

    colors = cmap(prop_scaled)
    prop_dict = dict(zip(vec_of_labels, colors))

    if create_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=scaler)
        sm.set_array([])
        fig = plt.figure()
        plt.colorbar(sm)
        plt.show()
        return prop_dict, fig
    else:
        return prop_dict


def dlc_to_napari_tracks(df, likelihood_thresh=0.4):
    """
    Convert a deeplabcut-style track to an array that can be visualized using:
        napari.view_tracks(dat)
    """

    # Convert tracks to napari style
    neuron_names = df.columns.remove_unused_levels().levels[0]
    # 5 columns:
    # track_id, t, z, y, x
    coords = ['z', 'y', 'x']
    all_tracks_list = []
    for i, name in enumerate(neuron_names):
        zxy_array = np.array(df[name][coords])
        t_array = np.expand_dims(np.arange(zxy_array.shape[0]), axis=1)

        # Remove low likelihood
        if 'likelihood' in df[name]:
            to_keep = df[name]['likelihood'] > likelihood_thresh
            zxy_array = zxy_array[to_keep, :]
            t_array = t_array[to_keep, :]
        id_array = np.ones_like(t_array) * i

        all_tracks_list.append(np.hstack([id_array, t_array, zxy_array]))

    return np.vstack(all_tracks_list)


def napari_tracks_from_match_list(list_of_matches, n0_zxy_raw, n1_zxy_raw, null_value=-1, t0=0):
    """
    Create a list of lists to be used with napari.add_tracks (or viewer.add_tracks)

    Parameters
    ----------
    list_of_matches
    n0_zxy_raw
    n1_zxy_raw
    null_value
    t0

    Returns
    -------

    """
    all_tracks_list = []
    for i_track, m in enumerate(list_of_matches):
        if null_value in m:
            continue

        track_m0 = [i_track, t0]
        track_m0.extend(n0_zxy_raw[int(m[0])])
        track_m1 = [i_track, t0 + 1]
        track_m1.extend(n1_zxy_raw[int(m[1])])

        all_tracks_list.append(track_m0)
        all_tracks_list.append(track_m1)
    return all_tracks_list


def napari_labels_from_frames(all_frames: dict, num_frames=1, to_flip_zxy=True) -> dict:

    all_t_zxy = np.array([[0, 0, 0, 0]], dtype=int)
    properties = {'label': []}
    for i_frame, frame in all_frames.items():
        if i_frame >= num_frames:
            break
        zxy = frame.neuron_locs
        if to_flip_zxy:
            zxy = zxy[:, [0, 2, 1]]
        num_neurons = zxy.shape[0]
        t_vec = np.ones((num_neurons, 1)) * i_frame
        t_zxy = np.hstack([t_vec, zxy])

        label_vec = list(range(num_neurons))

        all_t_zxy = np.vstack([all_t_zxy, t_zxy])
        properties['label'].extend(label_vec)

    all_t_zxy = all_t_zxy[1:, :]  # Remove dummy starter point
    options = {'data': all_t_zxy, 'face_color': 'transparent', 'edge_color': 'transparent', 'text': 'label',
               'properties': properties, 'name': 'Raw IDs'}

    return options
