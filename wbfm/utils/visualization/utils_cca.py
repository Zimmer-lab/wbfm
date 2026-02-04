from ast import Not
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, Union
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from tqdm.auto import tqdm

from wbfm.utils.external.utils_pandas import combine_columns_with_suffix
from wbfm.utils.general.utils_paper import apply_figure_settings, behavior_name_mapping
from wbfm.utils.general.utils_filenames import get_sequential_filename
from wbfm.utils.visualization.filtering_traces import fill_nan_in_dataframe
from wbfm.utils.visualization.utils_plot_traces import modify_dataframe_to_allow_gaps_for_plotly
from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
import plotly.graph_objects as go
from methodtools import lru_cache
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.general.utils_hardcoded import neurons_with_confident_ids


@dataclass
class CCAPlotter:
    """
    Dataclass for CCA visualization
    """

    project_data: ProjectData

    _df_traces: pd.DataFrame = None
    _df_beh: pd.DataFrame = None
    df_beh_binary: pd.DataFrame = None

    # Preprocessing options. Goal: more interpretable CCA weights
    preprocess_traces_using_pca: bool = True
    truncate_traces_to_n_components: int = None  # Not used if preprocess_traces_using_pca is False
    preprocess_behavior_using_pca: bool = True
    truncate_behavior_to_n_components: int = None  # Not used if preprocess_behavior_using_pca is False
    _df_traces_truncated: pd.DataFrame = None
    _df_beh_truncated: pd.DataFrame = None
    _pca_traces: PCA = None
    _pca_beh: PCA = None

    trace_kwargs: dict = None
    beh_kwargs: dict = None

    def __post_init__(self):
        # Default traces and behaviors
        opt = dict(filter_mode='rolling_mean', interpolate_nan=True, nan_tracking_failure_points=True,
                   use_physical_time=True, rename_neurons_using_manual_ids=True)
        if self.trace_kwargs is not None:
            opt.update(self.trace_kwargs)
        self.trace_kwargs = opt  # Complete options, including user modifications

        df_traces = self.project_data.calc_default_traces(**opt)
        self._df_traces = df_traces
        if self.preprocess_traces_using_pca:
            X, pca = self._truncate_using_pca(df_traces, n_components=self.truncate_traces_to_n_components)
            self._df_traces_truncated = pd.DataFrame(X, index=df_traces.index)
            self._pca_traces = pca

        opt = opt.copy()
        opt.update(self.beh_kwargs or {})
        df_beh = self.project_data.calc_default_behaviors(**opt, add_constant=False)
        # Standardize, but do not fully z-score, the behaviors
        beh_std = df_beh.std()
        beh_std[beh_std == 0] = 1  # There may be a constant column, which should not be divided by 0
        df_beh = df_beh / beh_std
        self._df_beh = df_beh
        if self.preprocess_behavior_using_pca:
            X, pca = self._truncate_using_pca(df_beh, n_components=self.truncate_behavior_to_n_components,
                                              subtract_mean=False)
            self._df_beh_truncated = pd.DataFrame(X, index=df_beh.index)
            self._pca_beh = pca
        # No filtering
        self.df_beh_binary = self.project_data.calc_default_behaviors(binary_behaviors=True)

    def _truncate_using_pca(self, df_traces, n_components=None, subtract_mean=True):
        X = fill_nan_in_dataframe(df_traces, do_filtering=False)

        # Make a pipeline that subtracts the mean and then does PCA
        if subtract_mean:
            pipe = Pipeline([
                ('subtract_mean', StandardScaler(with_mean=True, with_std=False)),
                ('pca', PCA(n_components=n_components, whiten=False))
            ])
        else:
            pipe = Pipeline([
                ('pca', PCA(n_components=n_components, whiten=False))
            ])
        X = pipe.fit_transform(X)

        return X, pipe

    @property
    def df_traces(self):
        if self.preprocess_traces_using_pca:
            return self._df_traces_truncated
        else:
            return self._df_traces

    @property
    def df_beh(self):
        if self.preprocess_behavior_using_pca:
            return self._df_beh_truncated
        else:
            return self._df_beh

    @lru_cache(maxsize=16)
    def calc_cca(self, n_components=3, binary_behaviors=False, sparse_tau=None, return_dataframes=False,
                 anchor_modes_with_defined_behaviors=True):
        """
        Does regular or sparse CCA. Returns the transformed data and the CCA object

        if sparse_tau is None, then regular CCA is performed (default)
        Note that because this is a cached function, sparse_tau should be a tuple, not a list

        Parameters
        ----------
        n_components
        binary_behaviors
        sparse_tau

        Returns
        -------

        """

        X = self.df_traces
        Y = self._get_beh_df(binary_behaviors)

        if sparse_tau is None:
            cca = CCA(n_components=n_components)
            X_r, Y_r = cca.fit_transform(X, Y)
        else:
            raise NotImplementedError("Sparse CCA not implemented")
        
        if anchor_modes_with_defined_behaviors:
            # Given known information about what the first and second modes should be, flip the modes if needed
            anchor_col = 'speed' if not binary_behaviors else 'rev'
            # First mode should be forward movement (speed)
            first_mode_beh_weights = cca.y_weights_[:, 0]
            anchor_idx = Y.columns.get_loc(anchor_col)
            if (binary_behaviors and first_mode_beh_weights[anchor_idx] < 0) or \
                (not binary_behaviors and first_mode_beh_weights[anchor_idx] > 0):
                # We want the 'rev' behavior to be positively weighted, but the velocity to be negatively weighted
                X_r[:, 0] *= -1
                Y_r[:, 0] *= -1
                cca.x_weights_[:, 0] *= -1
                cca.y_weights_[:, 0] *= -1
            
            # Second mode should be ventral turning
            anchor_col = 'ventral_only_body_curvature' if not binary_behaviors else 'ventral_turn'
            second_mode_beh_weights = cca.y_weights_[:, 1]
            anchor_idx = Y.columns.get_loc(anchor_col)
            # We want both to be positively weighted this time
            if second_mode_beh_weights[anchor_idx] < 0:
                X_r[:, 1] *= -1
                Y_r[:, 1] *= -1
                cca.x_weights_[:, 1] *= -1
                cca.y_weights_[:, 1] *= -1
        
        if return_dataframes:
            # Basically just set up the indices to be the same as the input dataframes
            X_r = pd.DataFrame(X_r, index=X.index, columns=[f'CCA_neural_mode_{i+1}' for i in range(X_r.shape[1])])
            Y_r = pd.DataFrame(Y_r, index=Y.index, columns=[f'CCA_behavior_mode_{i+1}' for i in range(Y_r.shape[1])])

        return X_r, Y_r, cca

    def calc_cca_scores(self, n_components=3, binary_behaviors=False, sparse_tau=None):
        """
        Does regular or sparse CCA. Returns the score on the CCA object

        if sparse_tau is None, then regular CCA is performed (default)
        Note that because this is a cached function, sparse_tau should be a tuple, not a list

        Parameters
        ----------
        n_components
        binary_behaviors
        sparse_tau

        Returns
        -------

        """

        X = self.df_traces
        Y = self._get_beh_df(binary_behaviors)
        _, _, cca = self.calc_cca(n_components=n_components, binary_behaviors=binary_behaviors, sparse_tau=sparse_tau)
        return cca.score(X, Y)

    def calc_cca_reconstruction(self, **kwargs):
        X_r, Y_r, cca = self.calc_cca(**kwargs)
        return cca.inverse_transform(X_r, Y_r)

    def calc_r_squared(self, use_pca=False, n_components: Union[int, list] = 1, use_behavior=False,
                       DEBUG=False, **kwargs):
        """
        Calculate the r-squared of reconstruction using either PCA or CCA

        If use_pca is True and use_behavior is True, then we use the PCA latent space from the neurons to reconstruct the behaviors
        In any case (pca or cca), the neuronal latent space is used to reconstruct; only the target space changes
        """
        if isinstance(n_components, list):
            all_r_squared, r_squared_per_row = {}, {}
            for i in n_components:
                all_r_squared[i], r_squared_per_row[i] = self.calc_r_squared(use_pca, i, use_behavior=use_behavior, **kwargs)
            return all_r_squared, r_squared_per_row
        
        # First, get the target to reconstruct
        if use_behavior:
            binary_behaviors = kwargs.get('binary_behaviors', False)
            y = self._get_beh_df(binary_behaviors=binary_behaviors, raw_not_truncated=True)
        else:
            y = self._df_traces  # Use the non-truncated traces
        
        # Second, get the latent space to reconstruct from, and use linear regression to reconstruct
        if use_pca:
            if use_behavior:
                # Get the PCA modes from the neuronal activity
                X_r, _ = self.project_data.calc_pca_modes(n_components=n_components, multiply_by_variance=False, **self.trace_kwargs)
                # Transform the neuronal PCA modes to behavior space using linear regression
                reg = LinearRegression().fit(X_r, y)
                y_r_recon = reg.predict(X_r)

            else:
                # Just reconstruct the traces
                pipe = self.project_data.calc_pca_modes(n_components=n_components, multiply_by_variance=False, 
                                                        return_pca_object=True, **self.trace_kwargs)
                y_r_recon = pipe.inverse_transform(pipe.fit_transform(y))

        else:
            # Only use the neuronal latent space to reconstruct
            X_r_recon, _ = self.calc_cca_reconstruction(n_components=n_components, **kwargs)
            reg = LinearRegression().fit(X_r_recon, y)
            y_r_recon = reg.predict(X_r_recon)
            # if use_behavior:
            #     # Binary behaviors are not transformed using PCA
            #     binary_behaviors = kwargs.get('binary_behaviors', False)
            #     if self.preprocess_behavior_using_pca and not binary_behaviors:
            #         # Transform the reconstructed behaviors back to the original space
            #         Y_r_recon = self._pca_beh.inverse_transform(Y_r_recon)
            #     # Later, only X is used
            #     X_r_recon = Y_r_recon
            # else:
            #     if self.preprocess_traces_using_pca:
            #         # Transform the reconstructed traces back to the original space
            #         X_r_recon = self._pca_traces.inverse_transform(X_r_recon)

        # Then, calculate the r-squared (either behavior or traces)
        residual_variance = (y - y_r_recon).var().sum()
        total_variance = y.var().sum()
        r_squared = 1 - residual_variance / total_variance

        # Also calculate the variance explained per row
        residual_variance_per_row = (y - y_r_recon).var(axis=0)
        total_variance_per_row = y.var(axis=0)
        r_squared_per_row = 1 - residual_variance_per_row / total_variance_per_row
        if DEBUG:
            print(f"Settings: binary_behaviors={kwargs.get('binary_behaviors', False)}, use_pca={use_pca}, n_components={n_components}")
            print(f"total_variance_per_row: {total_variance_per_row}")
            print(f"residual_variance_per_row: {residual_variance_per_row}")
            print(f"mean per row: {y.mean(axis=0)}")
            print(f"mean reconstruction per row: {X_r_recon.mean(axis=0)}")

        return r_squared, r_squared_per_row

    def calc_mode_dot_product(self, mode=0, binary_behaviors=False, **kwargs):
        """
        Calculate the dot product between a cca mode and the equivalent pca mode

        Parameters
        ----------
        mode
        binary_behaviors
        kwargs

        Returns
        -------

        """
        # CCA mode
        _, _, cca = self.calc_cca(n_components=mode+1, binary_behaviors=binary_behaviors, **kwargs)
        df_x, _ = self.get_weights_from_cca(cca, binary_behaviors, **kwargs)
        # PCA mode
        df_x_pca, _ = self.calc_pca_mode(mode, return_pca_weights=True)
        # Normalize the modes
        df_x = df_x / np.linalg.norm(df_x)
        df_x = df_x.iloc[mode, :]
        df_x_pca = df_x_pca / np.linalg.norm(df_x_pca)  # PCA is already 1 dimensional
        return df_x.values.dot(df_x_pca.values)[0]

    def _get_beh_df(self, binary_behaviors, raw_not_truncated=None):
        if raw_not_truncated is None:
            raw_not_truncated = not self.preprocess_behavior_using_pca

        if binary_behaviors:
            Y = self.df_beh_binary
        else:
            if raw_not_truncated:
                Y = self._df_beh
            else:
                # This will be truncated if the behavior is preprocessed using PCA
                Y = self.df_beh
        return Y

    def visualize_modes(self, i=1, binary_behaviors=False, **kwargs):
        X_r, Y_r, cca = self.calc_cca(n_components=i, binary_behaviors=binary_behaviors, **kwargs)

        df = pd.DataFrame({'Latent X': X_r[:, i], 'Latent Y': Y_r[:, i]})
        fig = px.line(df)
        self.project_data.shade_axis_using_behavior(plotly_fig=fig)
        fig.show()

        return fig

    def visualize_modes_and_weights(self, n_components=1, binary_behaviors=False,
                                    sort_trace_weights=True, sort_beh_weights=True,
                                    **kwargs):

        X_r, Y_r, cca = self.calc_cca(n_components=n_components, binary_behaviors=binary_behaviors, **kwargs)
        df_x, df_y = self.get_weights_from_cca(cca, binary_behaviors, **kwargs)

        def f(i=0):
            df = pd.DataFrame({'Latent X': X_r[:, i] / X_r[:, i].max(),
                               'Latent Y': Y_r[:, i] / Y_r[:, i].max()})
            fig = px.line(df, title=f'Latent mode {i+1}, correlation={pd.Series(X_r[:, i]).corr(pd.Series(Y_r[:, i]))}')
            self.project_data.shade_axis_using_behavior(plotly_fig=fig)
            fig.show()

            y = df_y.iloc[i, :]
            if sort_beh_weights:
                y = y.sort_values(ascending=False)
            fig = px.bar(y)
            fig.show()

            x = df_x.iloc[i, :]
            if sort_trace_weights:
                x = x.sort_values(ascending=False)
            fig = px.bar(x)
            fig.show()

            if self.preprocess_traces_using_pca:
                fig = px.line(self._df_traces_truncated, title=f'PCA modes from traces')
                self.project_data.shade_axis_using_behavior(plotly_fig=fig)
                fig.show()

            fig = px.line(self._df_beh, title=f'Raw behavior data')
            self.project_data.shade_axis_using_behavior(plotly_fig=fig)
            fig.show()

            # Correlation between behaviors and traces
            df_beh = self._get_beh_df(binary_behaviors, raw_not_truncated=True)
            df_pca = self.calc_pca_mode(i)[0]
            df_corr = pd.concat([df_beh, df_pca], axis=1).corr()
            fig = px.imshow(df_corr, height=400, width=600)
            fig.show()

        from ipywidgets import interact
        interact(f, i=(0, X_r.shape[1] - 1))

    def get_weights_from_cca(self, cca, binary_behaviors, **kwargs):
        """Returns trace and behavioral weight dataframes from a CCA object, with the correct column names"""
        df_beh = self._get_beh_df(binary_behaviors)
        df_traces = self.df_traces
        if 'sparse_tau' in kwargs:
            df_y = pd.DataFrame(cca.weights[1], index=df_beh.columns).T
            df_x = pd.DataFrame(cca.weights[0], index=df_traces.columns).T
        else:
            df_y = pd.DataFrame(cca.y_weights_, index=df_beh.columns).T
            df_x = pd.DataFrame(cca.x_weights_, index=df_traces.columns).T
        # Convert the weights to the original neuron space, if using PCA preprocessing
        if self.preprocess_traces_using_pca:
            df_x = pd.DataFrame(self._pca_traces.inverse_transform(df_x), columns=self._df_traces.columns)
        if self.preprocess_behavior_using_pca and not binary_behaviors:
            df_y = pd.DataFrame(self._pca_beh.inverse_transform(df_y), columns=self._df_beh.columns)
        return df_x, df_y

    def plot_single_mode(self, i_mode=0, binary_behaviors=False, use_pca=False, show_legend=True,
                         output_folder=None, **kwargs):

        if use_pca:
            df, var_explained = self.calc_pca_mode(i_mode)
        else:
            X_r, Y_r, cca = self.calc_cca(binary_behaviors=binary_behaviors, **kwargs)

            df = pd.DataFrame({f'Latent neural mode {i_mode+1}': X_r[:, i_mode] / X_r[:, i_mode].max(),
                               f'Latent behavior mode {i_mode+1}': Y_r[:, i_mode] / Y_r[:, i_mode].max()})
        df.index = self.df_traces.index
        fig = px.line(df)
        self.project_data.shade_axis_using_behavior(plotly_fig=fig)
        fig.update_yaxes(title='Amplitude', range=[-0.7, 1.2])
        fig.update_xaxes(title='Time (s)')
        if not show_legend:
            fig.update_layout(showlegend=False)
        apply_figure_settings(fig, width_factor=0.35, height_factor=0.15, plotly_not_matplotlib=True)
        fig.show()

        if output_folder is not None:
            fname = self._get_fig_filename(binary_behaviors, plot_3d=False, use_pca=use_pca, single_mode=True)
            fname = os.path.join(output_folder, fname)
            self._save_plotly_all_formats(fig, fname)

    def calc_pca_mode(self, i_mode, return_pca_weights=False) -> Tuple[pd.DataFrame, np.array]:
        X_r, var_explained = self.project_data.calc_pca_modes(n_components=i_mode + 1, multiply_by_variance=True,
                                                              return_pca_weights=return_pca_weights, **self.trace_kwargs)
        X_r = np.array(X_r)
        df = pd.DataFrame({f'PCA mode {i_mode + 1}': X_r[:, i_mode] / X_r[:, i_mode].max()})
        return df, var_explained

    def _save_plotly_all_formats(self, fig, fname):
        fig.write_html(fname)
        fname = fname.replace('.html', '.png')
        fig.write_image(fname, scale=4)
        fname = fname.replace('.png', '.svg')
        fig.write_image(fname)

    def plot(self, binary_behaviors=False, modes_to_plot=None, use_pca=False, use_X_r=True, sparse_tau=None,
             plot_3d=True, show_legend=True, output_folder=None, show_grid=False, use_paper_options=True,
             color_by_discrete_behavior=True, color_by_continuous_behavior=None, overwrite_file=True,
             DEBUG=False, ethogram_cmap_kwargs=None, beh_annotation_kwargs=None):
        """
        Plot modes with lots of options, colored by discrete behaviors

        Parameters
        ----------
        binary_behaviors
        modes_to_plot
        use_pca
        use_X_r
        sparse_tau
        plot_3d
        show_legend
        output_folder
        show_grid
        use_paper_options
        DEBUG
        ethogram_cmap_kwargs
        beh_annotation_kwargs

        Returns
        -------

        """
        if ethogram_cmap_kwargs is None:
            ethogram_cmap_kwargs = {}
        if beh_annotation_kwargs is None:
            beh_annotation_kwargs = {}
        if modes_to_plot is None:
            modes_to_plot = [0, 1, 2]
        if use_pca:
            # Multiply just to help the plotting
            X_r, var_explained = self.project_data.calc_pca_modes(n_components=3, multiply_by_variance=False)
            X_r = 3*X_r
            var_explained = 100*var_explained
            df_latents = pd.DataFrame(X_r)
        else:
            X_r, Y_r, cca = self.calc_cca(n_components=3, binary_behaviors=binary_behaviors, sparse_tau=sparse_tau)
            n_components = list(np.arange(1, modes_to_plot[-1] + 1))
            var_explained_cumulative, _ = self.calc_r_squared(use_pca=False, n_components=n_components,
                                                              binary_behaviors=binary_behaviors)
            # Undo the cumulative calculation
            # Note: unlike PCA, the first component is 1-indexed
            var_explained = var_explained_cumulative.copy()
            for i in n_components:
                if i == 1:
                    var_explained[i] = 100*var_explained_cumulative[i]
                else:
                    var_explained[i] = 100*(var_explained_cumulative[i] - var_explained_cumulative[i-1])
            if use_X_r:
                df_latents = pd.DataFrame(X_r)
            else:
                df_latents = pd.DataFrame(Y_r)
                logging.warning("Variance explained is only calculate for neuronal space")

        # Need these variables even if continuous coloring is used
        col_names, df_out, ethogram_cmap, state_codes = self._build_discrete_behavior_dataframe(
            df_latents,
            modes_to_plot,
            beh_annotation_kwargs,
            ethogram_cmap_kwargs
        )

        if color_by_discrete_behavior:
            phase_plot_list = []
            # Loop over behaviorally-colored short segments and plot
            for i, state_code in enumerate(state_codes):
                scatter_opt = dict(mode='lines', name=state_code.full_name,
                                   line=dict(color=ethogram_cmap.get(state_code, None), width=4))
                if plot_3d:
                    phase_plot_list.append(
                        go.Scatter3d(x=df_out[col_names[0][i]], y=df_out[col_names[1][i]], z=df_out[col_names[2][i]],
                                     **scatter_opt))
                else:
                    phase_plot_list.append(
                        go.Scatter(x=df_out[col_names[0][i]], y=df_out[col_names[1][i]],
                                   **scatter_opt))

            # Need to manually add the list of traces instead of plotly express
            fig = go.Figure(layout=dict(height=800, width=1000))
            fig.add_traces(phase_plot_list)
        elif color_by_continuous_behavior is not None:
            # Get the behavior to use as coloring, and plot it using go.Scatter
            beh_color = self.project_data.worm_posture_class.calc_behavior_from_alias(color_by_continuous_behavior)
            df_latents['color'] = beh_color.values
            # If I do a plotly line, it can only do discrete colormaps
            scatter_opt = dict(title=f'Coloring by {color_by_continuous_behavior}', color='color')

            if plot_3d:
                fig = px.scatter(df_latents, x=0, y=1, **scatter_opt)
            else:
                fig = px.scatter_3d(df_latents, x=0, y=1, z=2, **scatter_opt)

        else:
            raise NotImplementedError('Must pass color_by_continuous_behavior or set color_by_discrete_behavior=True')

        # Transparent background; needed for 3d plots
        if plot_3d:
            for axis in ['xaxis', 'yaxis', 'zaxis']:
                fig.update_layout(scene={axis: dict(backgroundcolor="rgba(0, 0, 0, 0)", zerolinecolor="black")})

        if show_grid:
            # Change the tick values to be evenly spaced
            xtick_min = np.floor(df_out[col_names[0][0]].min())
            xtick_max = np.ceil(df_out[col_names[0][0]].max())
            xtick_range = np.arange(xtick_min, xtick_max, 1)
            ytick_min = np.floor(df_out[col_names[1][0]].min())
            ytick_max = np.ceil(df_out[col_names[1][0]].max())
            ytick_range = np.arange(ytick_min, ytick_max, 1)
            # Hacky: https://community.plotly.com/t/scatter3d-background-plot-color/38838/4
            list_of_axes = ['xaxis', 'yaxis']
            list_of_ranges = [xtick_range, ytick_range]
            # for axis, tick_range in zip(list_of_axes, list_of_ranges):
            #     fig.update_layout({axis: dict(tickvals=tick_range, gridcolor='black')})

            if plot_3d:
                ztick_min = np.floor(df_out[col_names[2][0]].min())
                ztick_max = np.ceil(df_out[col_names[2][0]].max())
                ztick_range = np.arange(ztick_min, ztick_max, 1)

                list_of_axes = ['xaxis', 'yaxis', 'zaxis']
                list_of_ranges = [xtick_range, ytick_range, ztick_range]
                for axis, tick_range in zip(list_of_axes, list_of_ranges):
                    fig.update_layout(scene={axis: dict(tickvals=xtick_range, gridcolor='black', showbackground=True)})

                # From: https://stackoverflow.com/questions/73187799/truncated-figure-with-plotly?noredirect=1#comment129258910_73187799
                # Note that this is hard to do in jupyter and then see the settings, but can be done with dash:
                # https://community.plotly.com/t/how-to-get-change-current-scene-camera-in-3d-plot-inside-jupyter-notebook-python/1912/4
        else:
            if plot_3d:
                list_of_axes = ['xaxis', 'yaxis', 'zaxis']
                for axis in list_of_axes:
                    fig.update_layout(scene={axis: dict(showticklabels=False)})
            else:
                list_of_axes = ['xaxis', 'yaxis']
                # Do not update the 'scene'
                for axis in list_of_axes:
                    fig.update_layout({axis: dict(showticklabels=False)})

        # Remove legend
        if not show_legend:
            fig.update_layout(showlegend=False)

        # Add margins (the axis labels are often cut off)
        if plot_3d:
            camera = dict(eye=dict(x=1.8, y=1.8, z=1.8))
            fig.update_layout(scene_camera=camera)
        # if plot_3d:
            # fig.update_xaxes(automargin=True)
            # fig.update_layout(
            #     margin=dict(l=0, r=0, b=0, t=0),
            # )
            # fig.update_traces(cliponaxis=False)

        # Transparent background
        if use_paper_options:
            apply_figure_settings(fig, width_factor=0.4, height_factor=0.3, plotly_not_matplotlib=True)

        # Get base string to use for modes
        if use_pca:
            axis_title_func = lambda i: f'Neuronal component {i}<br>(PCA; {var_explained[i-1]:.0f}%)'
        elif binary_behaviors:
            axis_title_func = lambda i: f'Discrete Behavioral and<br>Neuronal component {i} (CCA; {var_explained[i]:.0f}%)'
        else:
            axis_title_func = lambda i: f'Behavioral and<br>Neuronal component {i} (CCA; {var_explained[i]:.0f}%)'
        # Get a shorter version
        # simple_base_axis_title = f"{base_axis_title.split('(')[1][:-1]} {{}}"

        # For paper
        if plot_3d:
            for axis in ['xaxis', 'yaxis', 'zaxis']:
                fig.update_layout(scene={axis: dict(showgrid=True)})

        else:
            # There is a bug with plotly moving the x axis title to the top when labels are turned off
            # https://github.com/plotly/plotly.js/issues/6552
            # Instead, make the labels transparent
            fig.update_xaxes(showticklabels=True, tickfont=dict(color="rgba(0,0,0,0)", size=1))

            opt = dict(showline=True, linecolor='black')#, font=dict(color='black', size=10))
            fig.update_layout(
                xaxis=dict(title=axis_title_func(1), side='bottom', **opt),
                yaxis=dict(title=axis_title_func(2), side='left', **opt),
            )

        if output_folder is not None:
            fname = self._get_fig_filename(binary_behaviors, plot_3d, use_pca, single_mode=False)
            fname = os.path.join(output_folder, fname)

            if not overwrite_file:
                fname = get_sequential_filename(fname, verbose=0)

            self._save_plotly_all_formats(fig, fname)

        fig.show()
        return fig

    def _build_discrete_behavior_dataframe(self, df_latents, modes_to_plot, beh_annotation_kwargs, ethogram_cmap_kwargs):
        # Color the lines by behavior annotation (not necessarily the same as the behavior used for CCA)
        beh_annotation = dict(fluorescence_fps=True, reset_index=True, include_collision=False, include_turns=True,
                              include_head_cast=False, include_pause=False, include_slowing=False)
        beh_annotation.update(beh_annotation_kwargs)
        state_vec = self.project_data.worm_posture_class.beh_annotation(**beh_annotation)
        df_latents['state'] = state_vec.values  # Ignore the index here, since it may not be physical time
        ethogram_cmap_kwargs.setdefault('include_turns', beh_annotation['include_turns'])
        ethogram_cmap_kwargs.setdefault('include_quiescence', beh_annotation['include_pause'])
        ethogram_cmap_kwargs.setdefault('include_collision', beh_annotation['include_collision'])
        ethogram_cmap = BehaviorCodes.ethogram_cmap(**ethogram_cmap_kwargs)
        df_out, col_names = modify_dataframe_to_allow_gaps_for_plotly(df_latents, modes_to_plot, 'state')
        state_codes = df_latents['state'].unique()
        return col_names, df_out, ethogram_cmap, state_codes

    def _get_fig_filename(self, binary_behaviors, plot_3d, use_pca, single_mode):
        # Build name based on options used
        if use_pca:
            fname = 'pca'
        else:
            fname = 'cca'
            if binary_behaviors:
                fname += '-binary'
            else:
                fname += '-continuous'
        if plot_3d:
            fname += '-3d'
        else:
            fname += '-2d'
        if single_mode:
            fname += '-mode'
        fname += '.html'
        return fname

    def calc_mode_correlation(self, **kwargs):
        """
        Calculates the correlation between the modes of the traces and behaviors, which is what CCA is maximizing

        Returns
        -------

        """

        X_r, Y_r, cca = self.calc_cca(**kwargs)
        corr_list = []
        for i in range(X_r.shape[1]):
            corr_list.append(pd.Series(X_r[:, i]).corr(pd.Series(Y_r[:, i])))
        return corr_list


def calc_r_squared_for_all_projects(all_projects, r_squared_kwargs=None, melt=True, **kwargs):
    if r_squared_kwargs is None:
        r_squared_kwargs = {}

    if r_squared_kwargs.get('n_components', None) is not None:
        if isinstance(r_squared_kwargs['n_components'], list):
            # Then do recursively
            all_r_squared = []
            all_r_squared_per_row = {}
            n_components_list = r_squared_kwargs['n_components'].copy()
            for n_components in tqdm(n_components_list):
                r_squared_kwargs['n_components'] = n_components
                all_cca_classes, df_r_squared, r_squared_per_row = calc_r_squared_for_all_projects(all_projects, r_squared_kwargs, melt, **kwargs)
                df_r_squared['n_components'] = n_components
                all_r_squared.append(df_r_squared)
                all_r_squared_per_row[n_components] = r_squared_per_row

            # Flatten the nested dictionary and then convert to a DataFrame
            flat_data = []
            beh_name_mapping = behavior_name_mapping(shorten=True)
            for outer_key, level1_dict in all_r_squared_per_row.items():
                for level1_key, level2_dict in level1_dict.items():
                    for level2_key, level3_dict in level2_dict.items():
                        for level3_key, value in level3_dict.items():
                            flat_data.append({
                                'Components': outer_key,
                                'Dataset Name': level1_key,
                                'Method': level2_key,
                                'Behavior Variable': beh_name_mapping.get(level3_key, level3_key),
                                'Cumulative Variance explained': value
                            })
            df_all_r_squared_per_row = pd.DataFrame(flat_data)
            return all_cca_classes, pd.concat(all_r_squared), df_all_r_squared_per_row

    all_cca_classes = {}
    all_r_squared = defaultdict(dict)
    all_r_squared_per_row = defaultdict(dict)

    opt_dict = {'PCA': dict(use_pca=True),
                'CCA': dict(use_pca=False),
                'CCA Discrete': dict(use_pca=False, binary_behaviors=True)}

    for name, p in tqdm(all_projects.items(), leave=False):
        cca_plotter = CCAPlotter(p, **kwargs)
        all_cca_classes[name] = cca_plotter
        for opt_name, opt in opt_dict.items():
            opt.update(r_squared_kwargs)
            all_r_squared[name][opt_name], all_r_squared_per_row[name][opt_name] = cca_plotter.calc_r_squared(**opt)

    df_r_squared = pd.DataFrame(all_r_squared).T

    if melt:
        df_r_squared = df_r_squared.melt(var_name='Method', value_name='Variance Explained')

    return all_cca_classes, df_r_squared, all_r_squared_per_row


def calc_mode_correlation_for_all_projects(all_projects, correlation_kwargs=None, **kwargs):
    if correlation_kwargs is None:
        correlation_kwargs = {}
    all_cca_classes = {}
    all_mode_correlations = defaultdict(dict)

    opt_dict = {'cca': dict(),
                'cca_binary': dict(binary_behaviors=True)}

    for name, p in tqdm(all_projects.items()):
        cca_plotter = CCAPlotter(p, **kwargs)
        all_cca_classes[name] = cca_plotter
        for opt_name, opt in opt_dict.items():
            opt.update(correlation_kwargs)
            all_mode_correlations[opt_name][name] = cca_plotter.calc_mode_correlation(**opt)

    df_mode_correlations = pd.DataFrame(all_mode_correlations['cca'])
    df_mode_correlations_binary = pd.DataFrame(all_mode_correlations['cca_binary'])

    return all_cca_classes, df_mode_correlations, df_mode_correlations_binary


def calc_cca_weights_for_all_projects(all_projects, which_mode=0, weights_kwargs=None,
                                      neural_not_behavioral=True, correct_sign_using_top_weight=True,
                                      drop_unlabeled_neurons=True, min_datasets_present=5,
                                      combine_left_and_right=False,
                                      **kwargs):
    if weights_kwargs is None:
        weights_kwargs = {}
    all_cca_classes = {}
    all_weights = defaultdict(dict)

    opt_dict = {'cca': dict(binary_behaviors=False),
                'cca_binary': dict(binary_behaviors=True)}

    for name, p in all_projects.items():
        cca_plotter = CCAPlotter(p, **kwargs)
        all_cca_classes[name] = cca_plotter
        for opt_name, opt in opt_dict.items():
            opt.update(weights_kwargs)
            _, _, cca = cca_plotter.calc_cca(**opt)
            trace_weights, behavior_weights = cca_plotter.get_weights_from_cca(cca, **opt)
            if neural_not_behavioral:
                all_weights[opt_name][name] = trace_weights.loc[which_mode, :]
            else:
                all_weights[opt_name][name] = behavior_weights.loc[which_mode, :]

    df_weights = pd.DataFrame(all_weights['cca']).T
    df_weights_binary = pd.DataFrame(all_weights['cca_binary']).T

    if neural_not_behavioral:
        # Drop all neurons that contain 'neuron' in the name
        if drop_unlabeled_neurons:
            df_weights = df_weights.loc[:, ~df_weights.columns.str.contains('neuron')]
            df_weights_binary = df_weights_binary.loc[:, ~df_weights_binary.columns.str.contains('neuron')]
        # Remove neurons that are not present in at least min_datasets_present datasets
        if min_datasets_present > 0:
            df_weights = df_weights.loc[:, df_weights.notnull().sum() >= min_datasets_present]
            df_weights_binary = df_weights_binary.loc[:, df_weights_binary.notnull().sum() >= min_datasets_present]

    # The weights have sign ambiguity. Correct this by setting the top weight to be positive
    if correct_sign_using_top_weight:
        sign_vec = np.sign(df_weights.iloc[:, df_weights.abs().sum().argmax()])
        df_weights = df_weights.mul(sign_vec, axis='index')
        sign_vec = np.sign(df_weights_binary.iloc[:, df_weights_binary.abs().sum().argmax()])
        df_weights_binary = df_weights_binary.mul(sign_vec, axis='index')

    # Combine the left and right neurons (optional)
    if combine_left_and_right:
        df_weights = combine_columns_with_suffix(df_weights)
        df_weights_binary = combine_columns_with_suffix(df_weights_binary)

    # Sort them by the signed median weight
    df_weights = df_weights.reindex(df_weights.median().sort_values(ascending=False).index, axis=1)
    df_weights_binary = df_weights_binary.reindex(df_weights_binary.median().sort_values(ascending=False).index, axis=1)

    return all_cca_classes, df_weights, df_weights_binary


def calc_pca_weights_for_all_projects(all_projects, which_mode=0, correct_sign_using_top_weight=True,
                                      neuron_names=None, include_only_confident_ids=False,
                                      drop_unlabeled_neurons=True,
                                      min_datasets_present=0, combine_left_right=False,
                                      **kwargs):
    """
    Similar to calc_cca_weights_for_all_projects, but for PCA

    Parameters
    ----------
    all_projects
    which_mode
    correct_sign_using_top_weight
    drop_unlabeled_neurons
    min_datasets_present
    kwargs

    Returns
    -------

    """
    if include_only_confident_ids and neuron_names is not None:
        raise NotImplementedError("Cannot use include_only_confident_ids and neuron_names together")
    all_weights = defaultdict(dict)
    trace_opt = kwargs.copy()

    for name, p in tqdm(all_projects.items()):
        trace_weights, _ = p.calc_pca_modes(return_pca_weights=True, combine_left_right=combine_left_right, **trace_opt)
        all_weights[name] = trace_weights.T.loc[which_mode, :]

    df_weights = pd.DataFrame(all_weights).T

    # Keep a subset of neurons if neuron_names is specified
    if neuron_names is not None:
        # Allow neuron_names to include additional column names
        df_weights = df_weights.loc[:, df_weights.columns.isin(neuron_names)]
    # Keep a subset of neurons that have confident ids
    if include_only_confident_ids:
        df_weights = df_weights.loc[:, df_weights.columns.isin(neurons_with_confident_ids(combine_left_right=combine_left_right))]
    # Drop all neurons that contain 'neuron' in the name
    if drop_unlabeled_neurons:
        df_weights = df_weights.loc[:, ~df_weights.columns.str.contains('neuron')]
    # Remove neurons that are not present in at least min_datasets_present datasets
    if min_datasets_present > 0:
        df_weights = df_weights.loc[:, df_weights.notnull().sum() >= min_datasets_present]

    # The weights have sign ambiguity. Correct this by setting the top weight to be positive
    if correct_sign_using_top_weight:
        sign_vec = np.sign(df_weights.iloc[:, df_weights.abs().sum().argmax()])
        df_weights = df_weights.mul(sign_vec, axis='index')

    # Sort them by the signed median weight
    df_weights = df_weights.reindex(df_weights.median().sort_values(ascending=False).index, axis=1)

    return df_weights
