from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from backports import cached_property
from matplotlib import pyplot as plt

from wbfm.utils.external.utils_pandas import cast_int_or_nan
from wbfm.utils.tracklets.high_performance_pandas import get_names_from_df
from wbfm.utils.projects.utils_neuron_names import name2int_neuron_and_tracklet
from wbfm.utils.visualization.utils_plot_traces import detrend_exponential_iter
from sklearn.linear_model import LinearRegression()


def napari_labels_from_traces_dataframe(df, neuron_name_dict=None,
                                        z_to_xy_ratio=1, DEBUG=False):
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
    z_to_xy_ratio
    df
    neuron_name_dict
    DEBUG

    Returns
    -------

    """
    df.replace(0, np.NaN, inplace=True)  # DLC uses all zeros as failed tracks

    if neuron_name_dict is None:
        neuron_name_dict = {}
    all_neurons = get_names_from_df(df)
    t_vec = np.expand_dims(np.array(list(df.index), dtype=int), axis=1)
    # label_vec = np.ones(len(df.index), dtype=int)
    all_t_zxy = np.array([[0, 0, 0, 0]], dtype=int)
    properties = dict(label=[])
    for n in all_neurons:
        coords = ['z', 'x', 'y']
        zxy = np.array(df[n][coords])

        # if round_in_z:
        #     zxy[:, 0] = np.round(zxy[:, 0])
        zxy[:, 0] *= z_to_xy_ratio
        # zxy = df[n][zxy_names].to_numpy(dtype=int)
        t_zxy = np.hstack([t_vec, zxy])
        if n in neuron_name_dict:
            # label_vec[:] = this_name
            label_vec = [neuron_name_dict[n]] * len(df.index)
            if DEBUG:
                print(f"Found named neuron: {n} = {label_vec[0]}")
        else:
            # Get the index from the dataframe, or try to convert the column name into a label
            if 'i_reindexed_segmentation' in df[n]:
                label_vec = list(map(int, df[n]['i_reindexed_segmentation']))
            elif 'label' in df[n]:
                # For traces dataframe
                label_vec = [i for i in df[n]['label']]
            elif 'raw_neuron_ind_in_list' in df[n]:
                # For tracks dataframe
                label_vec = [i for i in df[n]['raw_neuron_ind_in_list']]
            else:
                label_vec = [name2int_neuron_and_tracklet(n) for _ in range(t_vec.shape[0])]

        all_t_zxy = np.vstack([all_t_zxy, t_zxy])
        properties['label'].extend(label_vec)
    # Remove invalid positions
    # Some points are negative instead of nan
    all_t_zxy = np.where(all_t_zxy < 0, np.nan, all_t_zxy)
    to_keep = ~np.isnan(all_t_zxy).any(axis=1)
    all_t_zxy = all_t_zxy[to_keep, :]
    all_t_zxy = all_t_zxy[1:, :]  # Remove dummy starter point
    properties['label'] = [p for p, good in zip(properties['label'], to_keep[1:]) if good]
    # Additionally remove invalid names
    try:
        to_keep = np.array([not np.isnan(p) for p in properties['label']])
        all_t_zxy = all_t_zxy[to_keep, :]
        properties['label'] = [cast_int_or_nan(p) for p, good in zip(properties['label'], to_keep) if good]
    except TypeError:
        # Then the user is passing a non-int custom name, so just skip this
        pass
    # More info on text: https://github.com/napari/napari/blob/main/examples/add_points_with_text.py
    options = {'data': all_t_zxy, 'face_color': 'transparent', 'edge_color': 'transparent',
               'text': {'text': 'label'},  # Can add color or size here
               'properties': properties, 'name': 'Neuron IDs', 'blending': 'additive',
               'visible': False}

    return options


@dataclass
class NapariPropertyHeatMapper:
    """Builds dictionaries to map segmentation labels to various neuron properties (e.g. average or max brightness)"""

    red_traces: pd.DataFrame
    green_traces: pd.DataFrame
    curvature_fluorescence_fps: pd.DataFrame = pd.DataFrame([np.nan])

    @property
    def names(self):
        return get_names_from_df(self.red_traces)

    @property
    def vec_of_labels(self):
        return np.nanmean(self.df_labels.to_numpy(), axis=0).astype(int)

    @property
    def df_labels(self) -> pd.DataFrame:
        return self.red_traces.loc[:, (slice(None), 'label')]

    @property
    def mean_red(self):
        tmp1 = self.red_traces.loc[:, (slice(None), 'intensity_image')]
        tmp1.columns = self.names
        tmp2 = self.red_traces.loc[:, (slice(None), 'area')]
        tmp2.columns = self.names
        return tmp1 / tmp2

    @property
    def mean_green(self):
        tmp1 = self.green_traces.loc[:, (slice(None), 'intensity_image')]
        tmp1.columns = self.names
        tmp2 = self.green_traces.loc[:, (slice(None), 'area')]
        tmp2.columns = self.names
        return tmp1 / tmp2

    def corrcoef_kymo(self):
        if self.curvature_fluorescence_fps.isnull().values.all():
            return [np.nan]

        if not self.curvature_fluorescence_fps.isnull().values.all():
            corrcoefs = []
            for neuron in self.names:
                vector = np.abs(np.corrcoef(self.curvature_fluorescence_fps.assign(
                    neuron_to_test=self.red_traces[neuron]["intensity_image"]).dropna(axis="rows").T)[100, :99])
                c = np.max(vector)
                corrcoefs.append(c)
            val_to_plot = corrcoefs
            return property_vector_to_colormap(val_to_plot, self.vec_of_labels)

    def anchor_corr_red(self,anchor="neuron_028"):
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
                print(neuron, "skiped")
                corrcoefs.append(0)

            val_to_plot = np.array(corrcoefs)

        return property_vector_to_colormap(val_to_plot, self.vec_of_labels)

    def count_nonnan(self) -> Dict[int, float]:
        num_nonnan = self.df_labels.count()
        val_to_plot = np.array(num_nonnan) / self.df_labels.shape[0]
        return property_vector_to_colormap(val_to_plot, self.vec_of_labels)

    def max_of_red(self):
        val_to_plot = list(self.mean_red.max())
        return property_vector_to_colormap(val_to_plot, self.vec_of_labels)

    def std_of_red(self):
        val_to_plot = list(self.mean_red.std())
        return property_vector_to_colormap(val_to_plot, self.vec_of_labels)

    def max_of_green(self):
        val_to_plot = list(self.mean_green.max())
        return property_vector_to_colormap(val_to_plot, self.vec_of_labels)

    def std_of_green(self):
        val_to_plot = list(self.mean_green.std())
        return property_vector_to_colormap(val_to_plot, self.vec_of_labels)

    def max_of_ratio(self):
        val_to_plot = list((self.mean_green / self.mean_red).max())
        return property_vector_to_colormap(val_to_plot, self.vec_of_labels)

    def std_of_ratio(self):
        val_to_plot = list((self.mean_green / self.mean_red).std())
        return property_vector_to_colormap(val_to_plot, self.vec_of_labels)


def property_vector_to_colormap(val_to_plot, vec_of_labels, cmap=plt.cm.plasma):
    prop = np.array(val_to_plot)
    prop_scaled = (
            (prop - prop.min()) / (prop.max() - prop.min())
    )  # matplotlib cmaps need values in [0, 1]
    colors = cmap(prop_scaled)
    prop_dict = dict(zip(vec_of_labels, colors))
    return prop_dict
