import logging
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tqdm.auto import tqdm

from wbfm.utils.external.utils_matplotlib import export_legend
from wbfm.utils.general.hardcoded_paths import load_paper_datasets

from wbfm.utils.utils_cache import cache_to_disk_class
from wbfm.utils.external.utils_plotly import pastelize_color, mute_color


def paper_trace_settings():
    """
    The settings used in the paper.

    Returns
    -------

    """
    opt = dict(interpolate_nan=True,
               filter_mode='rolling_mean',
               min_nonnan=0.75,
               nan_tracking_failure_points=True,
               nan_using_ppca_manifold=True,
               channel_mode='dr_over_r_50',
               use_physical_time=True,
               rename_neurons_using_manual_ids=True,
               always_keep_manual_ids=True,
               only_keep_confident_ids=True,
               manual_id_confidence_threshold=0,
               high_pass_bleach_correct=False)
    return opt


def plotly_paper_color_discrete_map():
    """
    To be used with the color_discrete_map argument of plotly.express functions

    # TODO: this sometimes returns hex, and sometimes rgba... unfortunately, plotly has the same inconsistency

    Parameters
    ----------

    Returns
    -------

    """
    base_cmap = px.colors.qualitative.D3
    pca_cmap = px.colors.qualitative.Safe
    # mode_cmap = px.colors.qualitative.Plotly
    mode_cmap = px.colors.qualitative.Safe
    from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
    beh_cmap = BehaviorCodes.ethogram_cmap(include_collision=True, include_quiescence=True, include_reversal_turns=True,
                                           include_custom=True, include_stimulus=True)

    cmap_dict = {'gcamp': base_cmap[0], 'wbfm': base_cmap[0],
                 'Active in Freely Moving only': base_cmap[0], 'Manifold in Freely Moving only': base_cmap[0],
                 'Freely Moving (GCaMP)': base_cmap[0], 'Freely Moving': base_cmap[0], 'Wild Type': base_cmap[0],
                 # Skip orange... don't like it!
                 'immob': base_cmap[2], 'Active in Immob': base_cmap[2], 'Manifold in Immob': base_cmap[2],
                 'Intrinsic (shared with immobilized)': base_cmap[2],
                 'Immobilized (GCaMP)': base_cmap[2], 'Immobilized': base_cmap[2],
                 'gfp': base_cmap[7], 'Reversal State': base_cmap[7],  # Gray
                 'Freely Moving (GFP)': base_cmap[7],
                 'Freely Moving (GFP, residual)': base_cmap[7],
                 'global': base_cmap[3],
                 'residual': base_cmap[4],
                 'Freely Moving (GCaMP, residual)': base_cmap[4],
                 'O2 or CO2 sensing': base_cmap[5],  # Brown
                 'Not IDed': base_cmap[7], 'Not Identified': base_cmap[7],'Undetermined': base_cmap[7],  # Same as gfp; shouldn't ever be on same plot
                 'mutant': base_cmap[6], 'Freely Moving (gcy-31, gcy-35, gcy-9)': base_cmap[6],
                 'gcy-31, gcy-35, gcy-9': base_cmap[6], 'Mutant': base_cmap[6], 'gcy-31; -35; -9': base_cmap[6],
                 'gcy-31;-35;-9': base_cmap[6],  # Pink
                 # Colors for hierarchy
                 'No oscillations': base_cmap[7], 'No Behavior or Hierarchy': base_cmap[7],  # Same as gfp
                 'Hierarchy only': base_cmap[0],  # Same as raw
                 'Behavior only': base_cmap[1],  # Similar to raw, but brighter (teal)
                 'Hierarchical Behavior': base_cmap[3],  # New: orange
                 # PCA and CCA, which are a different colormap
                 'PCA': pca_cmap[4],
                 'CCA': pca_cmap[3], 'Continuous': pca_cmap[3],
                 'CCA Discrete': pca_cmap[5], 'CCA\n Discrete': pca_cmap[5], 'Discrete': pca_cmap[5],
                 # Individual modes, which are again different
                 1: mode_cmap[4], 2: mode_cmap[5], 3: mode_cmap[0], 4: mode_cmap[10], 5: mode_cmap[8],
                 # Role types, which are connected to behavior
                 'Inter, fwd': beh_cmap[BehaviorCodes.FWD], 'Inter, Forward': beh_cmap[BehaviorCodes.FWD],
                 'Motor, Forward': beh_cmap[BehaviorCodes.FWD], 'Forward': beh_cmap[BehaviorCodes.FWD],
                 'Inter, rev': beh_cmap[BehaviorCodes.REV], 'Inter, Reverse': beh_cmap[BehaviorCodes.REV],
                 'Motor, Reverse': beh_cmap[BehaviorCodes.REV], 'Reverse': beh_cmap[BehaviorCodes.REV],
                 'Sensory': beh_cmap[BehaviorCodes.SELF_COLLISION], 'Other': beh_cmap[BehaviorCodes.SELF_COLLISION],
                 'Interneuron': beh_cmap[BehaviorCodes.STIMULUS],
                 'Motor': beh_cmap[BehaviorCodes.QUIESCENCE], 'Interneuron, Motor': beh_cmap[BehaviorCodes.QUIESCENCE],
                 'Motor, Ventral': beh_cmap[BehaviorCodes.VENTRAL_TURN], 'Ventral': beh_cmap[BehaviorCodes.VENTRAL_TURN],
                 'Ventral body': beh_cmap[BehaviorCodes.VENTRAL_TURN], 'Ventral head': beh_cmap[BehaviorCodes.VENTRAL_TURN],
                 'Motor, Dorsal': beh_cmap[BehaviorCodes.DORSAL_TURN], 'Dorsal': beh_cmap[BehaviorCodes.DORSAL_TURN],
                 'Dorsal body': beh_cmap[BehaviorCodes.DORSAL_TURN], 'Dorsal head': beh_cmap[BehaviorCodes.DORSAL_TURN],
                 }
    # Add alternative names
    for k, v in data_type_name_mapping().items():
        cmap_dict[v] = cmap_dict[k]
    return cmap_dict


def intrinsic_categories_color_discrete_map(return_hex=True, mix_fraction = 0.0):
    d3 = px.colors.qualitative.D3
    cmap = {'Intrinsic': d3[4], #d3[1],           # Purple (try to emphasize)
            'No manifold': d3[7],         # Gray
            'Freely moving only': d3[9],  # Light blue, close to the raw blue
            'Immobilized only': d3[5],    # Bleh green, close to the immobilized green
            'Rev in FM only': d3[0],
            'Fwd in both': d3[4],
            'Rev in immob only': d3[2],
            'Fwd in immob only': d3[2],
            'Encoding switches': d3[6] # Pink
            }
    # Map everything to be more pastel
    if mix_fraction is not None and mix_fraction != 0:
        if mix_fraction < 0:
            add_alpha = lambda hex_col: mute_color(hex_col, -mix_fraction, return_hex=return_hex)
        else:
            add_alpha = lambda hex_col: pastelize_color(hex_col, mix_fraction, return_hex=return_hex)
        cmap = {k: add_alpha(v) for k, v in cmap.items()}
    return cmap

def export_legend_for_paper(fname=None, frameon=True, ethogram=False, reversal_shading=False,
                            include_self_collision=False, triple_plots=False, o2=True, bayesian_supp=False, o2_supp=False):
    if ethogram:
        from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
        cmap = BehaviorCodes.ethogram_cmap(use_plotly_style_strings=False)
        labels = [BehaviorCodes.FWD, BehaviorCodes.REV, BehaviorCodes.VENTRAL_TURN, BehaviorCodes.DORSAL_TURN,
                  BehaviorCodes.PAUSE]
        colors = [cmap[k] for k in labels]
        labels = [behavior_name_mapping()[k.name] for k in labels]
    elif triple_plots:
        from wbfm.utils.visualization.paper_multidataset_triggered_average import PaperColoredTracePlotter
        cmap = PaperColoredTracePlotter.get_color_from_data_type
        labels = ['Raw', 'Global', 'Residual']
        colors = [PaperColoredTracePlotter.get_color_from_data_type(l.lower()) for l in labels]
    elif bayesian_supp:
        from wbfm.utils.visualization.paper_multidataset_triggered_average import PaperColoredTracePlotter
        cmap = PaperColoredTracePlotter.get_color_from_data_type
        labels = ['Global (Freely moving)']
        colors = [PaperColoredTracePlotter.get_color_from_data_type('global')]
        cmap2 = plotly_paper_color_discrete_map()
        labels2 = ['Freely Moving', 'Immobilized']
        colors2 = [cmap2[l] for l in labels2]
        labels.extend(labels2)
        colors.extend(colors2)
    elif o2_supp:
        cmap = plotly_paper_color_discrete_map()
        labels = ['Freely Moving', 'Immobilized', 'gcy-31, gcy-35, gcy-9']
        colors = [cmap[l] for l in labels]
    elif o2:
        cmap = plotly_paper_color_discrete_map()
        labels = ['Freely Moving', 'gcy-31, gcy-35, gcy-9']
        colors = [cmap[l] for l in labels]
    else:
        from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
        # Just plot the gray background
        labels = ['Backward Crawling']
        colors = [BehaviorCodes.shading_cmap_func(BehaviorCodes.REV)]
        if include_self_collision:
            labels.append('Self-collision')
            colors.append(BehaviorCodes.shading_cmap_func(BehaviorCodes.SELF_COLLISION,
                                                          additional_shaded_states=[BehaviorCodes.SELF_COLLISION]))

    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]

    handles = [f("s", colors[i]) for i in range(len(labels))]
    legend = plt.legend(handles, labels, loc=3, framealpha=1, frameon=frameon)

    if fname is not None:
        export_legend(legend=legend, fname=fname)


def data_type_name_mapping(include_mutant=False):
    mapping = {'wbfm': 'Freely Moving (GCaMP)',
               'gcamp': 'Freely Moving (GCaMP)',
               'immob': 'Immobilized (GCaMP)',
               'gfp': 'Freely Moving (GFP)'}
    if include_mutant:
        mapping['mutant'] = 'Freely Moving (gcy-31;-35;-9)'
        mapping['immob_mutant_o2'] = 'Immobilized with O2 stimulus (gcy-31;-35;-9)'
        mapping['immob_o2'] = 'Immobilized with O2 stimulus (GCaMP)'
        mapping['immob_o2_hiscl'] = 'Immobilized with O2 stimulus (HisCl)'
    return mapping


# Basic settings based on the physical dimensions of the paper
dpi = 96
# column_width_inches = 6.5  # From 3p elsevier template
column_width_inches = 8.5  # Full a4 page
column_width_pixels = column_width_inches * dpi
# column_height_inches = 8.6  # From 3p elsevier template
column_height_inches = 11  # Full a4 page
column_height_pixels = column_height_inches * dpi
pixels_per_point = dpi / 72.0
font_size_points = 10  # I think the default is 10, but since I am doing a no-margin image I need to be a bit larger
font_size_pixels = font_size_points * pixels_per_point


def paper_figure_page_settings(height_factor=1, width_factor=1):
    """Settings for a full column width, full height. Will be multiplied later"""
    # Note: changes this globally
    # plt.rcParams["font.family"] = "arial"

    matplotlib_opt = dict(figsize=(column_width_inches*width_factor,
                                   column_height_inches*height_factor), dpi=dpi)
    matplotlib_font_opt = dict(fontsize=font_size_points)
    plotly_opt = dict(width=round(column_width_pixels*width_factor),
                      height=round(column_height_pixels*height_factor))
    # See: https://stackoverflow.com/questions/67844335/what-is-the-default-font-in-python-plotly
    plotly_font_opt = dict(font=dict(size=font_size_pixels, color='black'), font_family="arial")

    opt = dict(matplotlib_opt=matplotlib_opt, plotly_opt=plotly_opt,
               matplotlib_font_opt=matplotlib_font_opt, plotly_font_opt=plotly_font_opt)
    return opt


def apply_figure_settings(fig=None, width_factor=1, height_factor=1, plotly_not_matplotlib=True):
    """
    Apply settings for the paper, per figure. Note that this does not change the size settings, only font sizes and
    background colors (transparent).

    Parameters
    ----------
    fig - Figure to modify. If None, will use plt.gcf(), which assumes that the figure is the current matplotlib figure
    width_factor - Fraction of an A4 page to use (width)
    height_factor - Fraction of an A4 page to use (height)
    plotly_not_matplotlib - If True, will modify the figure using plotly syntax. Otherwise, will use matplotlib syntax

    Returns
    -------

    """
    if fig is None:
        if not plotly_not_matplotlib:
            fig = plt.gcf()
        else:
            raise NotImplementedError("Only matplotlib is supported if the figure is not directly passed for now")
    figure_opt = paper_figure_page_settings(width_factor=width_factor, height_factor=height_factor)

    if plotly_not_matplotlib:
        font_dict = figure_opt['plotly_font_opt']
        size_dict = figure_opt['plotly_opt']
        # Update font size
        fig.update_layout(**font_dict, **size_dict, title=font_dict, autosize=False)
        # Transparent background
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        # Remove background grid lines
        fig.update_xaxes(showgrid=False, zeroline=False)
        fig.update_yaxes(showgrid=False, zeroline=False)
        # Remove margin
        fig.update_layout(margin=dict(l=2, r=0, t=0, b=2))
        # Add black lines on edges of plot (only left and bottom)
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black')
    else:
        font_dict = figure_opt['matplotlib_font_opt']
        size_dict = figure_opt['matplotlib_opt']
        # Change size
        fig.set_size_inches(size_dict['figsize'])
        fig.set_dpi(size_dict['dpi'])

        # Get ax from figure
        ax = fig.axes[0]

        # Title font size
        title = ax.title
        title.set_fontsize(font_dict['fontsize'])

        # X-axis and Y-axis label font sizes
        xlabel = ax.xaxis.label
        ylabel = ax.yaxis.label
        xlabel.set_fontsize(font_dict['fontsize'])
        ylabel.set_fontsize(font_dict['fontsize'])

        # Tick label font sizes
        for tick in ax.get_xticklabels():
            tick.set_fontsize(font_dict['fontsize'])
        for tick in ax.get_yticklabels():
            tick.set_fontsize(font_dict['fontsize'])

        plt.tight_layout()


def behavior_name_mapping(shorten=False):
    name_mapping = dict(
        signed_middle_body_speed='Velocity',
        dorsal_only_head_curvature='Dorsal head curvature',
        ventral_only_head_curvature='Ventral head curvature',
        dorsal_only_body_curvature='Dorsal body curvature',
        ventral_only_body_curvature='Ventral body curvature',
        FWD='Forward crawling',
        REV='Backward crawling',
        VENTRAL_TURN='Ventral turning',
        DORSAL_TURN='Dorsal turning',
        UNKNOWN='Unknown',
        rev='Fwd-Bwd State',
        dorsal_turn='Dorsal turn',
        ventral_turn='Ventral turn',
        self_collision='Self-collision',
        head_cast='Head cast',
        slowing='Slowing',
        SLOWING='Slowing',
        pause='Pause',
        PAUSE='Pause',
        # Eigenworms are counted from 0 in python, but the paper wants them from 1
        eigenworm_0='Eigenworm 1',
        eigenworm_1='Eigenworm 2',
        eigenworm_2='Eigenworm 3',
        eigenworm_3='Eigenworm 4',
    )
    if shorten:
        name_mapping = {k: v.replace(' curvature', '') for k, v in name_mapping.items()}
        # name_mapping_short = dict(
        #     rev='REV',
        #     dorsal_turn='DT',
        #     ventral_turn='VT',
        #     self_collision='Touch',
        #     head_cast='Cast',
        #     slowing='Slow',
        # )
    return name_mapping


class PaperDataCache:
    """
    Class for caching data generated by the project data, and to be used in the figures of the paper.

    """

    def __init__(self, project_data):

        from wbfm.utils.projects.finished_project_data import ProjectData
        self.project_data: ProjectData = project_data

    @cache_to_disk_class('invalid_indices_cache_fname',
                         func_save_to_disk=np.save,
                         func_load_from_disk=np.load)
    def calc_indices_to_remove_using_ppca(self):
        from wbfm.utils.tracklets.postprocess_tracking import OutlierRemoval
        names = self.project_data.neuron_names
        coords = ['z', 'x', 'y']
        all_zxy = self.project_data.red_traces.loc[:, (slice(None), coords)].copy()
        z_to_xy_ratio = self.project_data.physical_unit_conversion.z_to_xy_ratio
        all_zxy.loc[:, (slice(None), 'z')] = z_to_xy_ratio * all_zxy.loc[:, (slice(None), 'z')]
        outlier_remover = OutlierRemoval.load_from_arrays(all_zxy, coords, df_traces=None, names=names, verbose=0)
        try:
            outlier_remover.iteratively_remove_outliers_using_ppca(max_iter=3)
            to_remove = outlier_remover.total_matrix_to_remove
        except ValueError as e:
            logging.warning(f"PPCA failed with error: {e}, skipping outlier removal and saving empty array")
            to_remove = np.array([])
        return to_remove

    def invalid_indices_cache_fname(self):
        if self.cache_dir is None:
            return None
        return os.path.join(self.cache_dir, 'invalid_indices.npy')

    def paper_trace_dispatcher(self, channel_mode='dr_over_r_50', residual_mode=None, interpolate_nan=True,
                               **kwargs):
        """
        Dispatches the calculation of traces based on the arguments. Currently, **kwargs are ignored

        Parameters
        ----------
        channel_mode
        residual_mode
        kwargs

        Returns
        -------

        """
        if interpolate_nan:
            if residual_mode is None:
                if channel_mode == 'dr_over_r_50':
                    return self.calc_paper_traces()
                elif channel_mode == 'dr_over_r_20':
                    return self.calc_paper_traces_r20()
                elif channel_mode == 'red':
                    return self.calc_paper_traces_red()
                elif channel_mode == 'green':
                    return self.calc_paper_traces_green()
                else:
                    raise ValueError(f"Unknown channel mode: {channel_mode}")
            elif residual_mode == 'pca':
                return self.calc_paper_traces_residual()
            elif residual_mode == 'pca_global':
                return self.calc_paper_traces_global()
            elif residual_mode == 'pca_global_1':
                return self.calc_paper_traces_global_1()
            else:
                raise ValueError(f"Unknown residual mode: {residual_mode}")
        else:
            if residual_mode is not None:
                raise ValueError("All residual modes require nan interpolation; "
                                 f"got incompatible residual_mode: {residual_mode} with interpolate_nan=False")
            if channel_mode != 'dr_over_r_50':
                raise ValueError(f"Only dr_over_r_50 is supported without nan interpolation; "
                                 f"got incompatible channel_mode: {channel_mode}")
            return self.calc_paper_traces_no_interpolation()

    def list_of_paper_trace_methods(self, return_filenames=False, return_simple_names=False):
        """
        A list of the class methods that can be used to calculate traces for the paper

        Note that via ._decorator_args, the arguments used to cache the data are also saved
        """
        method_names = [self.calc_paper_traces, self.calc_paper_traces_r20, self.calc_paper_traces_red,
                        self.calc_paper_traces_green, self.calc_paper_traces_no_interpolation,
                        self.calc_paper_traces_residual, self.calc_paper_traces_global, self.calc_paper_traces_global_1]
        if return_simple_names or return_filenames:
            list_cache_filename_methods = [m._decorator_args['cache_filename_method'] for m in method_names]
            if return_filenames:
                return [getattr(self, m)() for m in list_cache_filename_methods]
            else:
                return [m.replace('_cache_fname', '') for m in list_cache_filename_methods]
        else:
            return method_names

    def rename_columns_in_existing_cached_dataframes(self, previous2new: Dict[str, str]):
        """
        Renames columns in all cached dataframes that have already been generated by the paper_trace_dispatcher

        Used for fixing incorrectly ID'ed neurons

        Parameters
        ----------
        previous2new

        Returns
        -------

        """

        all_possible_cached_methods = self.list_of_paper_trace_methods()
        # self.project_data.logger.info(f'Updating cached dataframes with name mapping: {previous2new}')

        for cache_method in all_possible_cached_methods:
            # Get the filename that would be used to save the file
            cache_filename_method = cache_method._decorator_args['cache_filename_method']
            cache_kwargs = cache_method._decorator_args['cache_kwargs']
            cache_filename = getattr(self, cache_filename_method)(**cache_kwargs)
            if cache_filename is not None and os.path.exists(cache_filename):
                self.project_data.logger.debug(f'Updating cached dataframe at {cache_filename}')
                # Actually load the file (do NOT recalculate)
                df = cache_method()
                # Rename columns
                df = df.rename(columns=previous2new)
                # Save the file
                func_save_to_disk = cache_method._decorator_args['func_save_to_disk']
                func_save_to_disk(cache_filename, df)

    @cache_to_disk_class('paper_traces_cache_fname',
                         func_save_to_disk=lambda filename, data: data.to_hdf(filename, key='df_with_missing'),
                         func_load_from_disk=pd.read_hdf)
    def calc_paper_traces(self):
        """
        Uses calc_default_traces to calculate traces according to settings used for the paper.
        See paper_trace_settings() for details

        Returns
        -------

        """
        opt = paper_trace_settings()
        assert not opt.get('use_paper_traces', False), \
            "paper_trace_settings should have use_paper_traces=False (recursion error)"
        df = self.project_data.calc_default_traces(**opt)
        if df is None:
            raise ValueError(f"Paper traces for project {self.project_data.project_dir} is None")
        return df

    def paper_traces_cache_fname(self):
        if self.cache_dir is None:
            return None
        return os.path.join(self.cache_dir, 'paper_traces.h5')

    @cache_to_disk_class('paper_traces_cache_fname_r20',
                         func_save_to_disk=lambda filename, data: data.to_hdf(filename, key='df_with_missing'),
                         func_load_from_disk=pd.read_hdf)
    def calc_paper_traces_r20(self):
        """
        Uses calc_default_traces to calculate traces according to settings used for the paper.
        See paper_trace_settings() for details

        Returns
        -------

        """
        opt = paper_trace_settings()
        opt['channel_mode'] = 'dr_over_r_20'
        assert not opt.get('use_paper_traces', False), \
            "paper_trace_settings should have use_paper_traces=False (recursion error)"
        df = self.project_data.calc_default_traces(**opt)
        if df is None:
            raise ValueError(f"Paper traces for project {self.project_data.project_dir} is None")
        return df

    def paper_traces_cache_fname_r20(self):
        if self.cache_dir is None:
            return None
        return os.path.join(self.cache_dir, 'paper_traces_r20.h5')

    @cache_to_disk_class('paper_traces_no_interpolation_cache_fname',
                         func_save_to_disk=lambda filename, data: data.to_hdf(filename, key='df_with_missing'),
                         func_load_from_disk=pd.read_hdf)
    def calc_paper_traces_no_interpolation(self):
        """
        Changes two options: interpolate_nan=False and nan_using_ppca_manifold=False

        Thus, is not suitable for pca/cca analysis, but can be used with Bayesian analysis

        Returns
        -------

        """
        opt = paper_trace_settings()
        opt['interpolate_nan'] = False
        opt['nan_using_ppca_manifold'] = False
        assert not opt.get('use_paper_traces', False), \
            "paper_trace_settings should have use_paper_traces=False (recursion error)"
        df = self.project_data.calc_default_traces(**opt)
        if df is None:
            raise ValueError(f"Paper traces for project {self.project_data.project_dir} is None")
        return df

    def paper_traces_no_interpolation_cache_fname(self):
        if self.cache_dir is None:
            return None
        return os.path.join(self.cache_dir, 'paper_traces_no_interpolation.h5')

    @cache_to_disk_class('paper_traces_cache_fname_red',
                         func_save_to_disk=lambda filename, data: data.to_hdf(filename, key='df_with_missing'),
                         func_load_from_disk=pd.read_hdf)
    def calc_paper_traces_red(self):
        """
        Uses calc_default_traces to calculate traces according to settings used for the paper.
        See paper_trace_settings() for details

        Returns
        -------

        """
        opt = paper_trace_settings()
        opt['channel_mode'] = 'red'
        assert not opt.get('use_paper_traces', False), \
            "paper_trace_settings should have use_paper_traces=False (recursion error)"
        df = self.project_data.calc_default_traces(**opt)
        if df is None:
            raise ValueError(f"Paper traces for project {self.project_data.project_dir} is None")
        return df

    def paper_traces_cache_fname_red(self):
        if self.cache_dir is None:
            return None
        return os.path.join(self.cache_dir, 'paper_traces_red.h5')

    @cache_to_disk_class('paper_traces_cache_fname_green',
                         func_save_to_disk=lambda filename, data: data.to_hdf(filename, key='df_with_missing'),
                         func_load_from_disk=pd.read_hdf)
    def calc_paper_traces_green(self):
        """
        Uses calc_default_traces to calculate traces according to settings used for the paper.
        See paper_trace_settings() for details

        Returns
        -------

        """
        opt = paper_trace_settings()
        opt['channel_mode'] = 'green'
        assert not opt.get('use_paper_traces', False), \
            "paper_trace_settings should have use_paper_traces=False (recursion error)"
        df = self.project_data.calc_default_traces(**opt)
        if df is None:
            raise ValueError(f"Paper traces for project {self.project_data.project_dir} is None")
        return df

    def paper_traces_cache_fname_green(self):
        if self.cache_dir is None:
            return None
        return os.path.join(self.cache_dir, 'paper_traces_green.h5')

    @cache_to_disk_class('paper_traces_residual_cache_fname',
                         func_save_to_disk=lambda filename, data: data.to_hdf(filename, key='df_with_missing'),
                         func_load_from_disk=pd.read_hdf)
    def calc_paper_traces_residual(self):
        """
        Like calc_paper_traces but adds the residual mode.
        """
        opt = paper_trace_settings()
        opt['residual_mode'] = 'pca'
        opt['interpolate_nan'] = True
        assert not opt.get('use_paper_traces', False), \
            "paper_trace_settings should have use_paper_traces=False (recursion error)"
        df = self.project_data.calc_default_traces(**opt)
        if df is None:
            raise ValueError(f"Paper traces (residual) for project {self.project_data.project_dir} is None")
        return df

    def paper_traces_residual_cache_fname(self):
        if self.cache_dir is None:
            return None
        return os.path.join(self.cache_dir, 'paper_traces_residual.h5')

    @cache_to_disk_class('paper_traces_global_cache_fname',
                         func_save_to_disk=lambda filename, data: data.to_hdf(filename, key='df_with_missing'),
                         func_load_from_disk=pd.read_hdf)
    def calc_paper_traces_global(self):
        """
        Like calc_paper_traces but for the global mode.
        """
        opt = paper_trace_settings()
        opt['residual_mode'] = 'pca_global'
        opt['interpolate_nan'] = True
        assert not opt.get('use_paper_traces', False), \
            "paper_trace_settings should have use_paper_traces=False (recursion error)"
        df = self.project_data.calc_default_traces(**opt)
        if df is None:
            raise ValueError(f"Paper traces (global) for project {self.project_data.project_dir} is None")
        return df

    @cache_to_disk_class('paper_traces_global_1_cache_fname',
                         func_save_to_disk=lambda filename, data: data.to_hdf(filename, key='df_with_missing'),
                         func_load_from_disk=pd.read_hdf)
    def calc_paper_traces_global_1(self):
        """
        Like calc_paper_traces but for the global mode.
        """
        opt = paper_trace_settings()
        opt['residual_mode'] = 'pca_global_1'
        opt['interpolate_nan'] = True
        assert not opt.get('use_paper_traces', False), \
            "paper_trace_settings should have use_paper_traces=False (recursion error)"
        df = self.project_data.calc_default_traces(**opt)
        if df is None:
            raise ValueError(f"Paper traces (global) for project {self.project_data.project_dir} is None")
        return df

    def paper_traces_global_cache_fname(self):
        if self.cache_dir is None:
            return None
        return os.path.join(self.cache_dir, 'paper_traces_global.h5')

    def paper_traces_global_1_cache_fname(self):
        if self.cache_dir is None:
            return None
        return os.path.join(self.cache_dir, 'paper_traces_global_1.h5')

    @property
    def cache_dir(self):
        fname = os.path.join(self.project_data.project_dir, '.cache')
        if not os.path.exists(fname):
            try:
                os.makedirs(fname)
            except PermissionError:
                print(f"Could not create cache directory {fname}")
                fname = None
        return fname

    def clear_disk_cache(self, delete_traces=True, delete_invalid_indices=True,
                         dry_run=False, verbose=1):
        """
        Deletes all cached files generated using the cache_to_disk_class decorator

        Returns
        -------

        """
        possible_fnames = []
        if delete_traces:
            possible_fnames.extend(self.list_of_paper_trace_methods(return_filenames=True))
        if delete_invalid_indices:
            possible_fnames.append(self.invalid_indices_cache_fname())
        for fname in possible_fnames:
            if fname is not None and os.path.exists(fname):
                if verbose >= 1:
                    print(f"Deleting {fname}")
                if not dry_run:
                    os.remove(fname)


def plot_box_multi_axis(df, x_columns_list, y_column, color_names=None, cmap=None, DEBUG=False):
    """
    Plots a box plot with multiple x labels

    https://plotly.com/python/categorical-axes/#multicategorical-axes
    """
    # Create boxplot using graph objects directly
    fig = go.Figure()

    # Sample data
    x = [list(df[col]) for col in x_columns_list]
    y = list(df[y_column])
    if color_names is None:
        color_names = np.unique(x[0])

    # Create a column for each color, assuming that the color name is in the first column (as a datatype)
    for c in color_names:
        num_total_pts = len(x[0])
        this_y = [y[i] for i in range(num_total_pts) if x[0][i] == c]
        # Need to keep both x columns
        x0 = [c] * len(this_y)
        x1 = [x[1][i] for i in range(num_total_pts) if x[0][i] == c]
        if DEBUG:
            print(f"Color: {c}, len: {len(this_y)}")
            print(f"X0: {x0}")
            print(f"X1: {x1}")
            print(f"Y: {this_y}")

        fig.add_trace(go.Box(x=[x0, x1], y=this_y, name=c))

    # Mapping between x labels (categories) and colors
    if cmap is None:
        cmap = plotly_paper_color_discrete_map()

    # Update the colormap based on the mapping
    for trace in fig.data:
        if trace.name in cmap:
            trace.marker.color = cmap[trace.name]
    return fig


def package_bayesian_df_for_plot(df, df_normalization=None,
                                 min_num_datapoints=0):
    # The scores should be calculated from the diff column, and the se of that, i.e. dse
    # However, the order of the models may be different, and thus the subtraction may not be what I want
    # So I could recalculate the loo for the pairs of models I actually want to compare
    # ... but I don't have the loo_dictionary, so I'll just set things to 0 if they aren't higher than the less complex models
    df_diff = df.pivot(columns='model_type', index='neuron_name', values='elpd_diff').copy()  # .reset_index()
    df_diff = df_diff / df.pivot(columns='model_type', index='neuron_name', values='dse')
    # Here each score is 'offset', such that the best model is 0, and the others are worse by the relevant amount
    # For example, if hierarchical_pca is rank 0 (should be), then the column 'nonhierarchical' is the improvement
    df_diff['Relative Hierarchy Score'] = df_diff['nonhierarchical']  # Check for order issues later
    df_diff['Hierarchy Score'] = df_diff['null']

    # Alternative: take the actual log likelihood, normalized by the number of data points
    if df_normalization is not None:
        df_elpd = df.pivot(columns='model_type', index='neuron_name', values='elpd_loo').copy().dropna()
        df_elpd = df_elpd.divide(df_normalization.count(), axis=0).dropna()
        # Add suffix to make it obvious these are processed columns
        df_elpd.columns = [f"{col}_normalized" for col in df_elpd.columns]
        df_diff = pd.concat([df_diff, df_elpd], axis=1)
        if min_num_datapoints > 0:
            has_enough_datapoints = df_normalization.count() > min_num_datapoints
            # This has more rows than df_diff, so we need to filter
            has_enough_datapoints = has_enough_datapoints.loc[df_diff.index]
            df_diff = df_diff.loc[has_enough_datapoints, :]

    # But the behavior score is the difference between the null and the nonhierarchical, which we don't directly have
    # Note that if the nonhierarchical is the best, this is still correct because that column is 0, and the null column
    # is exactly what we want
    df_diff['nonhierarchical'].fillna(0, inplace=True)
    df_diff['Behavior Score'] = df_diff['null'] - df_diff['nonhierarchical']

    # If any neurons have 'hierarchical_pca' with a rank < 0, then the hierarchy score is 0
    # This is because the hierarchical_pca model should always be the best unless there is overfitting
    idx_hierarchy = df['model_type'] == 'hierarchical_pca'
    rank_of_hierarchy_models = df.loc[idx_hierarchy, 'rank']
    idx_of_non_first_hierarchy_models = df[idx_hierarchy].loc[rank_of_hierarchy_models > 0, 'neuron_name']
    # We may have dropped some rows from df_diff, so ensure the index is still valid
    idx_of_non_first_hierarchy_models = idx_of_non_first_hierarchy_models[idx_of_non_first_hierarchy_models.isin(df_diff.index)]
    df_diff.loc[idx_of_non_first_hierarchy_models, 'Hierarchy Score'] = 0
    df_diff.loc[idx_of_non_first_hierarchy_models, 'Relative Hierarchy Score'] = 0

    # If any neurons have 'null' with a rank = 0, then both scores are 0
    # This is because the null model should always be the worst
    idx_null = df['model_type'] == 'null'
    rank_of_null_models = df.loc[idx_null, 'rank']
    idx_of_first_null_models = df[idx_null].loc[rank_of_null_models == 0, 'neuron_name']
    # We may have dropped some rows from df_diff, so ensure the index is still valid
    idx_of_first_null_models = idx_of_first_null_models[idx_of_first_null_models.isin(df_diff.index)]
    df_diff.loc[idx_of_first_null_models, 'Behavior Score'] = 0  # The hierarchy is already set to 0

    x, y = df_diff['Hierarchy Score'], df_diff['Behavior Score']
    text_labels = pd.Series(list(x.index), index=x.index)
    # no_label_idx = np.logical_and(x < 5, y < 8)  # Displays some blue-only text
    # no_label_idx = y < 8
    # text_labels[no_label_idx] = ''

    df_to_plot = df_diff.copy()
    df_to_plot['text'] = text_labels
    df_to_plot['neuron_name'] = df_to_plot.index
    # df_to_plot = pd.DataFrame({'Hierarchy Score': x, 'Behavior Score': y,
    #                            'text': text_labels, 'neuron_name': x.index})
    # df_to_plot = df_to_plot[df_to_plot.index.isin(neurons_with_confident_ids())]
    return df_to_plot


def add_figure_panel_references_to_df(df):
    """For each type of data, add the relevant figure panel references"""
    ref = 'Figure panel references'
    df.at['num_datasets_freely_moving_gcamp', ref] = '1K; 2C; 3A-L; 4A-E; S2C; S5E; S7A-O; S8A-F'
    df.at['raw_rev', ref] = '4A; S8A'
    df.at['raw_fwd', ref] = '4B,C; S8B-F'
    df.at['self_collision', ref] = '4E'
    df.at['residual', ref] = '4E'
    df.at['residual_rectified_fwd', ref] = '3E-H'
    df.at['residual_rectified_rev', ref] = '3E-H'

    df.at['num_datasets_immob_gcamp', ref] = '2C,G; S2C; S7A-H; S8A-C,F'
    df.at['num_datasets_mutant_immob', ref] = '4A-C; S8A-C,F'
    df.at['immob-stimulus', ref] = '4A-C; S8B,C,F'
    df.at['immob_mutant-stimulus', ref] = 'S8B,C,F'
    df.at['immob_downshift-stimulus', ref] = 'S9A'
    df.at['immob_mutant_downshift-stimulus', ref] = 'S9A'
    df.at['immob_hiscl-stimulus', ref] = 'S8D,E'

    df.at['num_datasets_gfp', ref] = '3I; S4D-E'


if __name__ == '__main__':
    # Generate the paper plots for the main paper projects
    all_projects_gcamp = load_paper_datasets(genotype=['gcamp', 'hannah_O2_fm'])
    all_projects_gfp = load_paper_datasets(genotype=['gfp', 'hannah_O2_fm'])
    all_projects_immob = load_paper_datasets(genotype=['immob'])

    for project_dict in [all_projects_immob, all_projects_gcamp, all_projects_gfp]:
        for project_name, project_data in tqdm(project_dict.items()):
            # For now, only calculate the non-interpolated traces, because the other ones are too slow
            project_data.calc_default_traces(use_paper_options=True, interpolate_nan=False)
