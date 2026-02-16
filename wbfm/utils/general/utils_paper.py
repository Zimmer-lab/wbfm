from calendar import c
from collections import defaultdict
import logging
import os
from typing import Dict
from scipy import stats
from statsmodels.stats.multitest import multipletests
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tqdm.auto import tqdm
from pathlib import Path

from wbfm.utils.external.utils_plotly import colored_text
from wbfm.utils.external.utils_matplotlib import export_legend
from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
from wbfm.utils.general.utils_hardcoded import get_neuron_base, load_paper_datasets, neuron_groups, intrinsic_definition, intrinsic_categories_short_description, neurons_with_confident_ids, neurons_with_less_confident_ids

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
                 'Freely Moving (GCaMP)': base_cmap[0], 'Freely Moving': base_cmap[0], 'Wild Type': base_cmap[0], '488': base_cmap[0], 488: base_cmap[0],
                 'No Light': base_cmap[1], '505': base_cmap[1], 505: base_cmap[1],
                 # Skip orange... don't like it!
                 'immob': base_cmap[2], 'Active in Immob': base_cmap[2], 'Manifold in Immob': base_cmap[2],
                 'Intrinsic (shared with immobilized)': base_cmap[2],
                 'Immobilized (GCaMP)': base_cmap[2], 'Immobilized': base_cmap[2],
                 'gfp': base_cmap[7], 'Reversal State': base_cmap[7],  # Gray
                 'Inactive': base_cmap[7], 'Active': base_cmap[0],
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
                 'Hierarchical Behavior': base_cmap[3], 'Hierarchy': base_cmap[3], # New: orange
                 # PCA and CCA, which are a different colormap
                 'PCA': pca_cmap[4],
                 'CCA': pca_cmap[3], 'Continuous': pca_cmap[3],
                 'CCA Discrete': pca_cmap[5], 'CCA\n Discrete': pca_cmap[5], 'Discrete': pca_cmap[5],
                 # Individual modes, which are again different
                 1: mode_cmap[4], 2: mode_cmap[5], 3: mode_cmap[0], 4: mode_cmap[10], 5: mode_cmap[8],
                 # Role types, which are connected to behavior
                 'Inter, fwd': beh_cmap[BehaviorCodes.FWD], 'Inter, Forward': beh_cmap[BehaviorCodes.FWD],
                 'Motor, Forward': beh_cmap[BehaviorCodes.FWD], 'Forward': beh_cmap[BehaviorCodes.FWD],
                 'Inter, rev': beh_cmap[BehaviorCodes.REV], 'Inter, Reverse': beh_cmap[BehaviorCodes.REV], 'Inter, Backward': beh_cmap[BehaviorCodes.REV],
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
            'Intrinsic (modulated)': d3[6],          # Pink (try to emphasize)
            'No manifold': d3[7],         # Gray
            'Freely moving only': d3[9],  # Light blue, close to the raw blue
            'Immobilized only': d3[5],    # Bleh green, close to the immobilized green
            'Rev in FM only': d3[0],
            'Fwd in both': d3[4],
            'Rev in immob only': d3[2],
            'Fwd in immob only': d3[2],
            'Encoding switches': d3[8] # Bleh Green
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

    # Also turn the axis ticks off
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])

    # Set transparent background
    ax.set_facecolor((0, 0, 0, 0))
    plt.tight_layout()

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
        eigenworm0='Eigenworm 1',
        eigenworm1='Eigenworm 2',
        eigenworm2='Eigenworm 3',
        eigenworm3='Eigenworm 4',
        TRACKING_FAILURE='Tracking failure',
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


def convert_cv_results_to_bayesian_format(df_cv):
    """
    Convert cross-validation results to the format expected by package_bayesian_df_for_plot.
    
    Parameters
    ----------
    df_cv : pd.DataFrame
        Input dataframe with columns:
        neuron_name, model_type, fold, group_id, test_ll, train_ll, test_size, train_size, model
    
    Returns
    -------
    pd.DataFrame
        Converted dataframe with columns compatible with package_bayesian_df_for_plot:
        neuron_name, model_type, rank, elpd_loo, p_loo, elpd_diff, weight, se, dse, warning, scale
    """
    df = df_cv.copy()
    
    # 1. Strip 'cv_fold' from neuron_name
    df['neuron_name'] = df['neuron_name'].str.replace('_cv_fold.*', '', regex=True)
    
    # 2. Normalize test_ll and train_ll by their respective sizes
    df['test_ll_normalized'] = df['test_ll'] / df['test_size']
    df['train_ll_normalized'] = df['train_ll'] / df['train_size']
    
    # 3. Calculate mean and std across folds for each neuron and model_type
    grouped = df.groupby(['neuron_name', 'model']).agg({
        'test_ll_normalized': ['mean', 'std'],
        'train_ll_normalized': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    # Note: model is renamed model_type
    grouped.columns = ['neuron_name', 'model_type', 'test_ll_mean', 'test_ll_std', 
                       'train_ll_mean', 'train_ll_std']
    
    # 4. Map to the expected output format
    # elpd_loo corresponds to the test set performance (left-one-out CV proxy)
    grouped['elpd_loo'] = grouped['test_ll_mean']
    grouped['elpd_loo_train'] = grouped['train_ll_mean']
    grouped['elpd_loo_se'] = grouped['test_ll_std']
    grouped['elpd_loo_train_se'] = grouped['train_ll_std']
    
    # 5. Add ranking within each neuron (best model gets rank 0)
    grouped['rank'] = grouped.groupby('neuron_name')['elpd_loo'].rank(method='first', ascending=False) - 1

    # Calculate elpd_diff relative to the best model for each neuron
    best_elpd = grouped.groupby('neuron_name')['elpd_loo'].transform('max')
    grouped['elpd_diff'] = best_elpd - grouped['elpd_loo']
    
    # 6. Add placeholder columns
    grouped['p_loo'] = np.nan
    grouped['weight'] = np.nan
    grouped['warning'] = ''
    grouped['scale'] = np.nan
    grouped['se'] = np.nan
    grouped['dse'] = np.nan
    
    return grouped.sort_values(['neuron_name', 'rank']).reset_index(drop=True)


def package_bayesian_df_for_plot(df, df_normalization=None, val_name='elpd_diff', take_absolute_value=False,
                                 min_num_datapoints=0, normalize_by_dse=True, DEBUG=False):
    """
    Builds a score to be plotted with the following logic:
    - Hierarchy Score: ELPD improvement of hierarchical_pca over null model
    - Behavior Score: ELPD improvement of nonhierarchical over null model

    Either way, assumes that the 'elpd_diff' column is the difference between the best model and the given model,
    such that the best model has elpd_diff = 0, and worse models have positive elpd_diff.
    """
    
    # The scores should be calculated from the diff column, and the se of that, i.e. dse
    # However, the order of the models may be different, and thus the subtraction may not be what I want
    # So I could recalculate the loo for the pairs of models I actually want to compare
    # ... but I don't have the loo_dictionary, so I'll just set things to 0 if they aren't higher than the less complex models
    df = df.copy()
    if take_absolute_value:
        df[val_name] = df[val_name].abs()
    df_diff = df.pivot(columns='model_type', index='neuron_name', values=val_name)
    
    if normalize_by_dse:
        # Copy the raw diff dataframe columns before normalizing
        df_diff_raw = df_diff.copy()
        df_diff_raw.columns = [f"{col}_raw" for col in df_diff_raw.columns]
        # This normalizes by the standard error, i.e. converts it to a z-score like metric
        df_diff = df_diff / df.pivot(columns='model_type', index='neuron_name', values='dse')

        df_diff = pd.concat([df_diff, df_diff_raw], axis=1)  # Add raw columns back in

    # Here each score is 'offset', such that the best model is 0, and the others are worse by the relevant amount
    # For example, if hierarchical_pca is rank 0 (should be), then the column 'nonhierarchical' is the improvement
    df_diff['Relative Hierarchy Score'] = df_diff['nonhierarchical']  # Check for order issues later
    df_diff['Hierarchy Score'] = df_diff['null']
    if normalize_by_dse:
        df_diff['Relative Hierarchy Score (raw)'] = df_diff_raw['nonhierarchical_raw']
        df_diff['Hierarchy Score (raw)'] = df_diff_raw['null_raw']

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
            # Print which neurons were removed
            print(f"Removed neurons with insufficient data points: "
                  f"{list(has_enough_datapoints.index[~has_enough_datapoints])}")

    # But the behavior score is the difference between the null and the nonhierarchical, which we don't directly have
    # Note that if the nonhierarchical is the best, this is still correct because that column is 0, and the null column
    # is exactly what we want
    df_diff['nonhierarchical'].fillna(0, inplace=True)
    df_diff['Behavior Score'] = df_diff['null'] - df_diff['nonhierarchical']
    if normalize_by_dse:
        df_diff['nonhierarchical_raw'].fillna(0, inplace=True)
        df_diff['Behavior Score (raw)'] = df_diff_raw['null_raw'] - df_diff_raw['nonhierarchical_raw']

    # If any neurons have 'hierarchical_pca' with a rank > 0, then the hierarchy score is 0
    # This is because the hierarchical_pca model should always be the best unless there is overfitting
    idx_hierarchy = df['model_type'] == 'hierarchical_pca'
    rank_of_hierarchy_models = df.loc[idx_hierarchy, 'rank']
    idx_of_non_first_hierarchy_models = df[idx_hierarchy].loc[rank_of_hierarchy_models > 0, 'neuron_name']
    # We may have dropped some rows from df_diff, so ensure the index is still valid
    idx_of_non_first_hierarchy_models = idx_of_non_first_hierarchy_models[idx_of_non_first_hierarchy_models.isin(df_diff.index)]
    df_diff.loc[idx_of_non_first_hierarchy_models, 'Hierarchy Score'] = 0
    df_diff.loc[idx_of_non_first_hierarchy_models, 'Relative Hierarchy Score'] = 0
    if normalize_by_dse:
        df_diff.loc[idx_of_non_first_hierarchy_models, 'Hierarchy Score (raw)'] = 0
        df_diff.loc[idx_of_non_first_hierarchy_models, 'Relative Hierarchy Score (raw)'] = 0
    if DEBUG:
        print(f"Neurons with non-best hierarchical_pca models: {idx_of_non_first_hierarchy_models}")

    # If any neurons have 'null' with a rank = 0, then both scores are 0
    # This is because the null model should always be the worst
    idx_null = df['model_type'] == 'null'
    rank_of_null_models = df.loc[idx_null, 'rank']
    idx_of_first_null_models = df[idx_null].loc[rank_of_null_models == 0, 'neuron_name']
    # We may have dropped some rows from df_diff, so ensure the index is still valid
    idx_of_first_null_models = idx_of_first_null_models[idx_of_first_null_models.isin(df_diff.index)]
    df_diff.loc[idx_of_first_null_models, 'Behavior Score'] = 0  # The hierarchy is already set to 0
    if normalize_by_dse:
        df_diff.loc[idx_of_first_null_models, 'Behavior Score (raw)'] = 0
    if DEBUG:
        print(f"Neurons with null models as the best: {idx_of_first_null_models}")

    x, y = df_diff['Hierarchy Score'], df_diff['Behavior Score']
    text_labels = pd.Series(list(x.index), index=x.index)

    df_to_plot = df_diff.copy()
    df_to_plot['text'] = text_labels
    df_to_plot['neuron_name'] = df_to_plot.index

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
    all_projects_gcamp = load_paper_datasets(data_type=['gcamp', 'hannah_O2_fm'])
    all_projects_gfp = load_paper_datasets(data_type=['gfp', 'hannah_O2_fm'])
    all_projects_immob = load_paper_datasets(data_type=['immob'])

    for project_dict in [all_projects_immob, all_projects_gcamp, all_projects_gfp]:
        for project_name, project_data in tqdm(project_dict.items()):
            # For now, only calculate the non-interpolated traces, because the other ones are too slow
            project_data.calc_default_traces(use_paper_options=True, interpolate_nan=False)



def plot_foldchange_boxes(
        df: pd.DataFrame,
        behavior_col: str,
        groups: list,
        rows_col: str = "Neuron",
        subtitle_behavior_col: str = None,
        value_col: str = "log2_fc",
        cmap: str = "Blues",
        behavior_hspace: float = 1.0,
        row_vspace: float = 0.33,
        subtitle_hgap: float = 0.5,
        group_vgap: float = 1.0,
        box_size: float = 1.0,
        edge_lw: float = 1.2,
        margin: float = 0.02,
        figsize: tuple = (6, 6),
        use_pval_log10: bool = False,
        pval_threshold: float = 0.1,
        vmax: float = None,
        vmin: float = None,
        nonsig_color: str = "lightgray",
        neuron_order: list = None,
        add_text: bool = False,
        center_at_zero: bool = False,
        DEBUG=False
):
    """
    Custom box-grid plot (neurons x behaviors) with group ordering.
    Can show either log2_fc (default) or signed -log10(p_value_adj).

    Originally designed by Itamar Lev
    """

    def format_group_label(name: str) -> str:
        """
        Replace underscores with line breaks for prettier multi-line group labels.
        Example:
            "motor_rev_tail" -> "motor\nrev\ntail"
        """
        return name.replace("_", "\n")

    df = df.copy()
    if rows_col not in df.columns:
        df[rows_col] = df.index.to_series().str.split("_").str[0]
    
    if behavior_col not in df.columns:
        raise ValueError(f"DataFrame must contain behavior column '{behavior_col}'")
    behaviors = df[behavior_col].unique().tolist()

    # --- if using pval log10, create transformed column ---
    if use_pval_log10:
        if "p_value_adj" not in df.columns:
            raise ValueError("DataFrame must contain 'p_value_adj' when use_pval_log10=True.")
        if "log2_fc" not in df.columns:
            raise ValueError("DataFrame must contain 'log2_fc' when use_pval_log10=True.")

        def signed_log10(row):
            if pd.isna(row["p_value_adj"]) or pd.isna(row["log2_fc"]):
                return np.nan
            if row["p_value_adj"] >= pval_threshold:
                return 0.0  # mark as nonsignificant
            sign = np.sign(row["log2_fc"])
            return sign * (-np.log10(row["p_value_adj"]))

        df["signed_pval_log10"] = df.apply(signed_log10, axis=1)
        value_col = "signed_pval_log10"

    # === Assign & sort neurons ===
    ordered_neurons, neuron_assignment = assign_and_sort_neurons(df, groups=groups, value_col=value_col, rows_col=rows_col,
                                                                 neuron_order=neuron_order)

    # --- Build data dictionary ---
    # Determine fill value for missing entries
    fill_value = 0.0 if use_pval_log10 else np.nan

    # Build data_dict to include all neurons in ordered_neurons
    # Change: data_dict is a flat dict with multi-index keys, not a nested dict
    data_df = df.groupby([rows_col, behavior_col, subtitle_behavior_col] if subtitle_behavior_col is not None else [rows_col, behavior_col])[value_col].median()
    # data_df = df.set_index([rows_col, behavior_col, subtitle_behavior_col]) if subtitle_behavior_col is not None else df.set_index([rows_col, behavior_col])
    data_dict = defaultdict(lambda: fill_value, data_df.to_dict())
    if DEBUG:
        print("Data dict keys:")
        for k in data_dict.keys():
            print(k)
            print(f"  Value: {data_dict[k]}")

    # Gather all values for colormap normalization
    all_values = list(data_dict.values())
    if len(all_values) == 0:
        raise ValueError("No numeric values found to color boxes.")

    if use_pval_log10 or center_at_zero:
        # force symmetric color scale around 0
        abs_max = np.max(np.abs(all_values))
        v_min, v_max = -abs_max, abs_max
    else:
        v_min, v_max = np.nanmin(all_values), np.nanmax(all_values)

    vmin = vmin if vmin is not None else v_min
    vmax = vmax if vmax is not None else v_max

    cmap_obj = plt.get_cmap(cmap)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    if DEBUG:
        print(f"Color scale vmin: {vmin}, vmax: {vmax}")

    # Geometry
    bw, bh = box_size, box_size
    n_beh = len(behaviors)

    # --- Compute neuron positions with group gaps ---
    ypos_map = {}
    ytick_positions, ytick_labels = [], []
    ycursor = 0.0
    group_positions = {}
    if isinstance(groups, dict):
        groups = list(groups.keys())
    else:
        raise ValueError("groups must be a list or dict")

    for g in groups + ["other"]:
        group_neurons = [n for n in ordered_neurons if neuron_assignment[n] == g]
        if not group_neurons:
            continue
        start_y = ycursor
        for n in group_neurons:
            ypos_map[n] = ycursor
            ytick_positions.append(ycursor + bh / 2.0)
            ytick_labels.append(n)
            ycursor += bh + row_vspace
        end_y = ycursor - row_vspace
        group_positions[g] = (start_y, end_y)
        ycursor += group_vgap

    # x positions also
    xpos_map, xtick_positions, xtick_labels, behavior_positions, total_width = compute_xpos_map(df, behavior_col=behavior_col, sub_behavior_col=subtitle_behavior_col,
                                                                                                box_size=bw, hspace=behavior_hspace)

    # total_width = n_beh * bw + (n_beh - 1) * behavior_hspace
    total_height = ycursor - group_vgap

    fig, ax = plt.subplots(figsize=figsize)

    # Draw boxes
    for key in data_dict.keys():
        if len(key) == 2:
            neuron, beh_key = key
        elif len(key) == 3:
            neuron, beh_key, subbeh_key = key
            beh_key = (beh_key, subbeh_key)
        else:
            raise ValueError("data_dict keys must be of length 2 or 3")
        
        ypos = ypos_map[neuron]
        xpos = xpos_map[beh_key]
        val = data_dict[key]

        if use_pval_log10 and nonsig_color is not None:
            # Missing or nonsignificant → gray
            color = nonsig_color if np.isnan(val) or val == 0.0 else cmap_obj(norm(val))
        else:
            # Fold-change mode: missing values are gray
            color = nonsig_color if np.isnan(val) else cmap_obj(norm(val))
        if DEBUG:
            print(f"Drawing box for neuron {neuron}, behavior {beh_key} at ({xpos}, {ypos}) with value {val} and color {color}")
        rect = Rectangle(
            (xpos, ypos),
            bw,
            bh,
            facecolor=color,
            edgecolor="black",
            linewidth=edge_lw,
            clip_on=False
        )
        ax.add_patch(rect)

        if add_text:
            threshold = 0.5 * (vmax - vmin)
            ax.text(
                xpos + bw/2,      # x: center of box
                ypos + bh/2,      # y: center of box
                f"{val:.2f}",     # text to display
                # f"{neuron[0]},{beh_key[-1]}\n{val:.2f}",     # text to display
                ha='center',       # horizontal alignment
                va='center',       # vertical alignment
                fontsize=8,
                color='white' if abs(val) > threshold else 'black',  # contrast with background
                weight='bold'
            )
    # X ticks
    # xtick_positions = [xi * (bw + behavior_hspace) + bw / 2.0 for xi in range(n_beh)]
    # print(xtick_positions)
    if DEBUG:
        print("X tick positions and labels:")
        print(xtick_positions)
        print(xtick_labels)
    if 'sub' in xtick_positions:
        ax.set_xticks(xtick_positions['sub'])
        ax.set_xticklabels(xtick_labels['sub'])
        # Set main as additional labels not using an axis
        for pos, label in zip(xtick_positions['main'], xtick_labels['main']):
            ax.text(
                pos,
                total_height + (bh + behavior_hspace) * 0.5,
                label,
                ha="center",
                va="bottom",
                fontsize=12,
            )
    else:
        ax.set_xticks(xtick_positions['main'])
        ax.set_xticklabels(xtick_labels['main'])
    # for k in xtick_positions.keys():
    #     ax.set_xticks(xtick_positions[k])
    #     ax.set_xticklabels(xtick_labels[k])
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", which="both", length=0, pad=8)

    # Y ticks
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels)
    ax.tick_params(axis="y", which="both", length=0)

    # Group labels
    if DEBUG:
        print("Group positions:")
        print(group_positions)
    for g, (start_y, end_y) in group_positions.items():
        ymid = (start_y + end_y) / 2.0 + bh / 2.0
        ax.text(
            -(behavior_hspace + bw) * 2.5,
            ymid,
            format_group_label(g),
            ha="center",
            va="center",
            fontsize=12,
            rotation=90
        )

    # Limits
    ax.set_xlim(-margin, total_width + margin)
    ax.set_ylim(-margin, total_height + margin)
    # ax.set_aspect("equal")

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    if DEBUG:
        print(f"Creating colorbar with vmin: {vmin}, vmax: {vmax}")
        print(cmap_obj)
        print(norm)
    # Don't set array - let the norm handle the mapping
    
    cbar = fig.colorbar(sm, ax=ax)
    # cbar = fig.colorbar(sm, ax=ax, fraction=0.06, pad=0.03)
    cbar_label = "Signed -log10(adj p-value)" if use_pval_log10 else value_col.replace("_", " ").title()
    cbar.set_label(cbar_label, fontsize=10)

    # --- Modify colorbar tick labels if clipping happened ---
    ticks = cbar.get_ticks()
    ticklabels = [f"{t:.2g}" for t in ticks]

    # Check if data exceeded manual limits
    if v_min < vmin:
        ticklabels[0] = f"<= {ticklabels[0]}"
    if v_max > vmax:
        ticklabels[-1] = f">= {ticklabels[-1]}"

    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticklabels)

    return ax, fig, ordered_neurons


def assign_and_sort_neurons(
        df: pd.DataFrame,
        groups: list,
        value_col: str = "log2_fc",
        rows_col: str = "Neuron",
        neuron_order: list = None,  # <-- optional input
):
    """
    Assign neurons to user-specified groups and optionally order them.

    - df neurons are assumed to already be in base format.
    - Group definitions are normalized with get_neuron_base.
    - If a neuron belongs to multiple groups, assign to the smallest one.
    - Neurons not in any group go into 'other'.
    - Within each group, neurons are ordered by descending fold change unless neuron_order is provided.

    Returns:
        ordered_neurons: list of neurons in plotting order
        neuron_assignment: dict neuron -> group
    """

    # Ensure neuron column exists
    if rows_col not in df.columns:
        df = df.copy()
        df[rows_col] = df.index.to_series().str.split("_").str[0]

    # Normalize group definitions (convert L/R → base)
    if isinstance(groups, list):
        group_dict = {g: set(get_neuron_base(n) for n in neuron_groups(g)) for g in groups}
    elif isinstance(groups, dict):
        # Assume format is correct
        group_dict = groups.copy()
        groups = list(group_dict.keys())
    else:
        raise ValueError("groups must be a list or dict")

    # Build neuron -> candidate groups map
    neuron_to_groups = {}
    for g, members in group_dict.items():
        for n in members:
            neuron_to_groups.setdefault(n, []).append(g)

    # Resolve conflicts + add "other"
    neuron_assignment = {}
    neurons_to_assign = neuron_order if neuron_order is not None else df[rows_col].unique()
    for neuron in neurons_to_assign:
        if neuron not in neuron_to_groups:
            neuron_assignment[neuron] = "other"
        else:
            g_list = neuron_to_groups[neuron]
            if len(g_list) == 1:
                neuron_assignment[neuron] = g_list[0]
            else:
                chosen = min(g_list, key=lambda g: len(group_dict[g]))
                neuron_assignment[neuron] = chosen

    # Compute max fold change per neuron for sorting
    fc_map = df.groupby(rows_col)[value_col].max().to_dict()

    # Build ordered list
    if neuron_order is not None:
        # Keep the user-provided order
        ordered_neurons = neuron_order.copy()
        # Include any extra neurons in df not in neuron_order at the end
        for n in df[rows_col].unique():
            if n not in ordered_neurons:
                ordered_neurons.append(n)
    else:
        # Sort within groups by descending fold change
        ordered_neurons = []
        for g in groups + ["other"]:
            group_neurons = [n for n, grp in neuron_assignment.items() if grp == g]
            group_neurons_sorted = sorted(group_neurons, key=lambda n: fc_map.get(n, -np.inf), reverse=True)
            ordered_neurons.extend(group_neurons_sorted)

    return ordered_neurons, neuron_assignment


def compute_xpos_map(df, behavior_col='behavior', sub_behavior_col=None,
                     box_size=1.0, hspace=0.33, behavior_gap=1.0):
    """
    Compute x-positions for hierarchical behavior structure.
    
    Parameters:
        sub_behavior_col: Optional. If None, uses flat behavior structure.
    
    Returns:
        xpos_map: dict with keys (main_behavior, sub_behavior) -> x_position
                  OR just behavior -> x_position if sub_behavior_col is None
        xtick_positions: dict with 'main' and 'sub' tick positions (or just 'main' if no sub)
        xtick_labels: dict with 'main' and 'sub' tick labels (or just 'main' if no sub)
        behavior_positions: dict mapping main_behavior -> (start_x, end_x)
        total_width: total width of the plot
    """
    
    bw = box_size
    
    # Case 1: No sub-behaviors (flat structure)
    if sub_behavior_col is None:
        behaviors = df[behavior_col].unique()
        
        xpos_map = {}
        tick_positions = []
        tick_labels = []
        behavior_positions = {}
        
        xcursor = 0.0
        
        for beh in behaviors:
            xpos_map[beh] = xcursor
            tick_positions.append(xcursor + bw / 2.0)
            tick_labels.append(beh)
            behavior_positions[beh] = (xcursor, xcursor + bw)
            xcursor += bw + hspace
        
        total_width = xcursor - hspace
        
        return (xpos_map, 
                {'main': tick_positions}, 
                {'main': tick_labels}, 
                behavior_positions, 
                total_width)
    
    # Case 2: Hierarchical structure with sub-behaviors
    behavior_hierarchy = df[[behavior_col, sub_behavior_col]].drop_duplicates()
    grouped = behavior_hierarchy.groupby(behavior_col, sort=False)[sub_behavior_col].apply(list)
    
    xpos_map = {}
    sub_tick_positions = []
    sub_tick_labels = []
    main_tick_positions = []
    main_tick_labels = []
    behavior_positions = {}
    sub_beh_factor = 0.10
    
    xcursor = 0.0
    
    for main_beh in grouped.index:
        sub_behaviors = grouped[main_beh]
        start_x = xcursor
        
        for sub_beh in sub_behaviors:
            xpos_map[(main_beh, sub_beh)] = xcursor
            sub_tick_positions.append(xcursor + bw / 2.0)
            sub_tick_labels.append(sub_beh)
            xcursor += bw + hspace * sub_beh_factor
        
        xcursor = xcursor - hspace * sub_beh_factor
        end_x = xcursor
        
        mid_x = (start_x + end_x) / 2.0 + bw / 2.0
        main_tick_positions.append(mid_x)
        main_tick_labels.append(main_beh)
        
        behavior_positions[main_beh] = (start_x, end_x)
        xcursor += behavior_gap
    
    total_width = xcursor - behavior_gap
    
    xtick_positions = {
        'main': main_tick_positions,
        'sub': sub_tick_positions
    }
    
    xtick_labels = {
        'main': main_tick_labels,
        'sub': sub_tick_labels
    }
    
    return xpos_map, xtick_positions, xtick_labels, behavior_positions, total_width


def split_time_series_with_laser_switches(df_green: pd.DataFrame, background_per_pixel: float = 100, brightness_threshold: float = 0, minimum_period_length: int = 10,
                                          DEBUG=False):
    """
    Detects periods where the laser is OFF based on total green fluorescence intensity.

    Note that this will also detect tracking failures if the entire worm is missing; hopefully those are short enough to be filtered with minimum_period_length.
    """
    total_green = df_green.loc[:, (slice(None), 'intensity_image')].T.sum() - background_per_pixel*df_green.loc[:, (slice(None), 'area')].T.sum()
    if DEBUG:
        fig = px.line(total_green, title='Total green fluorescence after background subtraction')
        fig.add_hline(y=brightness_threshold, line_dash='dash', line_color='red', annotation_text='Brightness threshold', annotation_position='top left')
        fig.show()
    # Get the time points with laser on/off switches, based on negative (near-zero) values after background subtraction
    is_laser_off = total_green < brightness_threshold
    # Convert to starts and stops of laser ON periods
    laser_on_periods = []
    in_laser_on = False
    for i in range(len(is_laser_off)):
        if not is_laser_off.iloc[i] and not in_laser_on:
            # Laser just turned ON
            start_idx = i
            in_laser_on = True
        elif is_laser_off.iloc[i] and in_laser_on:
            # Laser just turned OFF
            end_idx = i
            laser_on_periods.append((start_idx, end_idx))
            in_laser_on = False
    # Handle case where laser is ON until the end
    if in_laser_on:
        laser_on_periods.append((start_idx, len(is_laser_off)))
    # Remove short periods
    laser_on_periods = [(start, end) for start, end in laser_on_periods if (end - start) >= minimum_period_length]
    return laser_on_periods


def plot_trajectory(project_data, beh_annotation_kwargs=None, to_save=True):
    from wbfm.utils.visualization.utils_plot_traces import modify_dataframe_to_allow_gaps_for_plotly

    xy = project_data.worm_posture_class.calc_behavior_from_alias('worm_center_position').copy()
    xy = xy - xy.iloc[0, :]

    if beh_annotation_kwargs is None:
        beh_annotation_kwargs = {}
    beh_annotation_defaults = dict(fluorescence_fps=True, simplify_states=True,
                                   include_head_cast=False, include_collision=False, include_pause=True)
    beh_annotation_defaults.update(beh_annotation_kwargs)

    beh = project_data.worm_posture_class.beh_annotation(**beh_annotation_defaults)

    df_xy = xy
    df_xy['Behavior'] = beh.values
    # df_xy['Behavior'] = df_xy['Behavior'].map(lambda x: behavior_name_mapping()[x.name])
    df_xy.head()
    df_xy['size'] = 1
    ethogram_cmap = BehaviorCodes.ethogram_cmap(include_turns=True, include_reversal_turns=False)
    df_out, col_names = modify_dataframe_to_allow_gaps_for_plotly(df_xy, ['X', 'Y'], 'Behavior')

    # Loop to prep each line, then plot
    state_codes = df_xy['Behavior'].unique()
    phase_plot_list = []
    for i, state_code in enumerate(state_codes):
        if state_code == BehaviorCodes.UNKNOWN:
            continue
        phase_plot_list.append(
                    go.Scatter(x=df_out[col_names[0][i]], y=df_out[col_names[1][i]], mode='lines',
                                name=behavior_name_mapping()[state_code.full_name], line=dict(color=ethogram_cmap.get(state_code, None), width=4)))

    fig = go.Figure()
    fig.add_traces(phase_plot_list)

    # fig = px.scatter(df_xy, x='X', y='Y', color='Behavior')
    fig.update_yaxes(dict(title="Distance (mm)"), scaleanchor= 'x')
    fig.update_xaxes(dict(title="Distance (mm)"), )#range=[-1, 1])

    fig.add_trace(go.Scatter(x=[0], y=[0], marker=dict(
                        color='black', symbol='x',
                        size=10
                    ), name='start'))

    fig.add_trace(go.Scatter(x=[xy.iloc[-1, 0]], y=[xy.iloc[-1, 1]], marker=dict(
                        color='black',
                        size=10), name='end'
                    ))
    apply_figure_settings(fig, width_factor=0.5, height_factor=0.25, plotly_not_matplotlib=True)

    fig.update_traces(marker=dict(line=dict(width=0)))
    fig.show()

    if to_save:        
        fname = f"trajectory.png"
        fname = os.path.join("behavior", fname)
        fig.write_image(fname, scale=3)
        fig.write_image(fname.replace(".png", ".svg"))

# Functions for behavior supp figures
def calc_net_displacement(p):
    # Units: mm
    i_seg = 50

    df = p.worm_posture_class.centerline_absolute_coordinates()
    xy0 = df.loc[0, :][i_seg]
    xy1 = df.iloc[-1, :][i_seg]
    
    return np.linalg.norm(xy0 - xy1)
    
def calc_cumulative_displacement(p):
    # Units: mm
    i_seg = 50

    df = p.worm_posture_class.centerline_absolute_coordinates()[i_seg]
    dist = np.sqrt((df['X'] - df['X'].shift())**2 + (df['Y'] - df['Y'].shift())**2)
    line_integral = np.nansum(dist)
    
    return line_integral

def calc_displacement_dataframes(all_projects):
    
    all_displacements = defaultdict(dict)
    for name, p in tqdm(all_projects.items()):
        all_displacements['net'][name] = calc_net_displacement(p)
        all_displacements['cumulative'][name] = calc_cumulative_displacement(p)
    df_displacement_gcamp = pd.DataFrame(all_displacements)
    
    return df_displacement_gcamp


def calc_p_values_for_pca_weights(wbfm_weights: pd.DataFrame, immob_weights: pd.DataFrame,
                                  intrinsic_categories_fname=None, add_parentheses_for_less_confident=True):
    """
    Calculate p values for PCA weights between two datasets; neurons should be id'ed in both datasets.
    """

    ##
    ## Initial calculation of p values with multiple comparison correction
    ##
    opts_multipletests = dict(method='fdr_bh', alpha=0.05)

    names_to_keep = set(wbfm_weights.columns).intersection(immob_weights.columns)
    wbfm_melt = wbfm_weights.melt(var_name='neuron_name', value_name='PC1 weight').assign(dataset_type='gcamp')
    immob_melt = immob_weights.melt(var_name='neuron_name', value_name='PC1 weight').assign(dataset_type='immob')
    df_both = pd.concat([wbfm_melt, immob_melt], axis=0)
    df_both = df_both[df_both['neuron_name'].isin(names_to_keep)]
    df_both['Dataset Type'] = df_both['dataset_type'].map(data_type_name_mapping())

    # Update the neuron names to include parentheses if they are less confident
    if add_parentheses_for_less_confident:
        mapping = neurons_with_less_confident_ids(combine_left_right=True, return_mapping=True)
        df_both['neuron_name'] = df_both['neuron_name'].map(lambda x: mapping.get(x, x))

    # Significantly different from 0... need a permutation version, so use an extra function
    # From: https://stackoverflow.com/questions/73569894/permutation-based-alternative-to-scipy-stats-ttest-1samp
    # def _t_statistic(x, axis=-1):
    #     # return stats.ttest_1samp(x, popmean=0, axis=axis).statistic
    #     return stats.ttest_1samp(x, popmean=0).statistic

    # def t_statistic_permutation(x):
    #     return stats.permutation_test((x.values, ), _t_statistic, permutation_type='samples', ).pvalue

    def t_statistic_permutation(x):
        return stats.wilcoxon(x.values).pvalue

    # func = lambda x: stats.ttest_1samp(x, 0)[1]
    df_groupby = df_both.dropna().groupby(['neuron_name', 'dataset_type'])
    df_pvalue = df_groupby['PC1 weight'].apply(t_statistic_permutation).to_frame()
    df_pvalue.columns = ['p_value']

    # Multiple comparison correction in the same way for all tests
    output = multipletests(df_pvalue.values.squeeze(), **opts_multipletests)
    df_pvalue['p_value_corrected'] = output[1]
    df_pvalue['significance_corrected'] = output[0]

    # Sign of medians
    df_medians_gcamp = df_groupby['PC1 weight'].median()[(slice(None), 'gcamp')]
    df_medians_immob = df_groupby['PC1 weight'].median()[(slice(None), 'immob')]

    # Significantly different from each other (should be exact same as the boxplot)
    df_groupby = df_both.dropna().groupby(['neuron_name'])
    func = lambda x: stats.ttest_ind(x[x['dataset_type']=='gcamp']['PC1 weight'], x[x['dataset_type']=='immob']['PC1 weight'], 
                                    equal_var=False, permutations=1000)[1]
    df_significant_diff = df_groupby.apply(func).to_frame()
    df_significant_diff.columns = ['p_value_diff']
    # Multiple comparison correction in the same way for all tests
    output = multipletests(df_significant_diff.values.squeeze(), **opts_multipletests)
    df_significant_diff['p_value_corrected_diff'] = output[1]
    df_significant_diff['significance_corrected_diff'] = output[0]

    ##
    ## Conversion to interpretable categories
    ##
    # Process p value comparisons to 0
    df_pvalue_thresh = df_pvalue['significance_corrected'].reset_index()

    # Collect signficance calculations per datatype
    df_pivot = df_pvalue_thresh.pivot_table(index='neuron_name', columns='dataset_type', values='significance_corrected', aggfunc='first')
    df_4states_complex = df_pivot.astype(str).radd(df_pivot.columns + '_')
    df_4states_complex = (df_4states_complex['gcamp'] + '_' + df_4states_complex['immob'])#.reset_index()

    # Add suffix to the state: are both medians on the same side?
    df_medians_gcamp.name = 'same_sign'
    df_medians_immob.name = 'same_sign'
    df_medians_same_sign = ((df_medians_gcamp>0) == (df_medians_immob>0)).astype(str).radd(df_medians_gcamp.name + '_')
    df_4states_complex = df_4states_complex.to_frame().join(df_medians_same_sign)#.reset_index()

    # Add suffix to the state: is the difference between them significant?
    df_4states_complex = df_4states_complex.join(df_significant_diff['significance_corrected_diff'].astype(str).radd('diff_'))

    # Combine into final categories
    df_4states_complex.columns = ['pvalue_result', 'diff_sign', 'diff_sig']
    df_4states = (df_4states_complex['pvalue_result'] + '_' + df_4states_complex['diff_sign'] + '_' + df_4states_complex['diff_sig']).to_frame()
    df_4states.columns = ['Result']

    df_4states_counts = df_4states['Result'].value_counts().reset_index()
    df_4states_counts['Result_simple'] = df_4states_counts['Result'].map(intrinsic_definition)
    df_4states['Result_simple'] = df_4states['Result'].map(intrinsic_definition)

    df_4states['Result_description'] = df_4states['Result'].map(intrinsic_categories_short_description())

    # Also add the original booleans that lead to these categories
    df_categories = df_4states.copy().join(df_4states_complex.loc[:, ['pvalue_result', 'diff_sign', 'diff_sig']]).drop(columns='Result')

    # Color xticks by later pie chart colors
    # NOTE: IF UPDATING NEURONS: this will remove neurons, which then will not get into the pie chart later
    # df_categories = pd.read_excel('fig3/intrinsic_categories.xlsx')
    df_categories['Result_simple_color'] = df_categories['Result_simple'].map(intrinsic_categories_color_discrete_map(return_hex=False))
    df_both = pd.merge(df_both, df_categories, on='neuron_name', validate='many_to_one')
    df_both['neuron_name_html'] = df_both.apply(lambda x: colored_text(x['neuron_name'], x['Result_simple_color'], bold=True), axis=1)
    if intrinsic_categories_fname is not None:
        df_4states.sort_values(by='Result_description').to_excel(intrinsic_categories_fname)

    return df_both, df_significant_diff, df_4states_counts


def calculate_bayesian_model_categories(x, y, df_to_plot_gfp, df_to_plot_gcamp, remove_names_of_ns=True):
    # Add a couple names back in
    df_to_plot_gfp = df_to_plot_gfp.copy()
    rename_func = lambda x: f'{x} (gfp)' if x != '' else ''
    df_to_plot_gfp.loc[:, 'text'] = df_to_plot_gfp.loc[:, 'text'].apply(rename_func)

    df_to_plot = pd.concat([df_to_plot_gcamp, df_to_plot_gfp])
    df_to_plot['Dataset Type'] = df_to_plot['datatype']
    df_to_plot['Size'] = 1

    x_max_gfp = df_to_plot_gfp[x].max()
    y_max_gfp = df_to_plot_gfp[y].max()
    print('GFP thresholds: ', y_max_gfp, x_max_gfp)

    def categorize_row(row):
        if row[y] > y_max_gfp and row[x] > x_max_gfp:
            return 'Hierarchical Behavior'
        elif row[y] <= y_max_gfp and row[x] > x_max_gfp:
            return 'Behavior only'
        elif row[y] > y_max_gfp and row[x] <= x_max_gfp:
            return 'Hierarchy only'
        else:
            return 'No Behavior or Hierarchy'

    # Apply function to create new column
    df_to_plot_gcamp['Category_raw'] = df_to_plot_gcamp.apply(categorize_row, axis=1)
    df_to_plot['Category_raw'] = df_to_plot.apply(categorize_row, axis=1)
    _df = df_to_plot[df_to_plot.index.isin(neurons_with_confident_ids())]

    # Final categories: combine 'Hierarchy only' and 'Hierarchical Behavior' into 'Hierarchy', and split GFP into 'GFP' category
    def simplify_category(row):
        if row['Dataset Type'] == 'gfp':
            return 'GFP'
        elif row['Category_raw'] in ['Hierarchical Behavior', 'Hierarchy only']:
            return 'Hierarchy'
        else:
            return row['Category_raw']
    df_to_plot['Category'] = df_to_plot.apply(simplify_category, axis=1)

    df_to_plot['text_raw'] = df_to_plot['text']
    if remove_names_of_ns:
        df_to_plot.loc[df_to_plot[y] <= y_max_gfp, 'text'] = ''
    
    return df_to_plot, _df, y_max_gfp, x_max_gfp


def plot_bayesian_model_comparison(x, y, df_to_plot=None, y_max_gfp=None, df_to_plot_gfp=None, df_to_plot_gcamp=None, 
                                   output_folder=None, remove_names_of_ns=True, display_text=True, to_show=True, **kwargs):
    """
    Plot Bayesian model comparison with GFP thresholds indicated.
    """

    if df_to_plot is None or y_max_gfp is None:
        df_to_plot, _df, y_max_gfp, x_max_gfp = calculate_bayesian_model_categories(x, y, df_to_plot_gfp, df_to_plot_gcamp,remove_names_of_ns=remove_names_of_ns)
    
    fig = px.scatter(df_to_plot, 
                     y=y, x=x, #range_x=[-2, 60],
                     text=df_to_plot['text'] if display_text else None, 
                     color='Dataset Type',
                     color_discrete_map=plotly_paper_color_discrete_map(), 
                     size_max=10,
                     hover_data=['Category'],
                     **kwargs
                    )
    fig.update_traces(textposition='middle left')

    apply_figure_settings(fig, width_factor=1.0, height_factor=0.3)

    fig.add_shape(type="line",
                  x0=0, y0=y_max_gfp,  # start of the line (bottom of the plot)
                  x1=1, y1=y_max_gfp,  # end of the line (top of the plot)
                  line=dict(color="black", width=1, dash="dash"),
                  xref='paper',
                  yref='y')
    # Diagonal line, only if not plotting a relative score
    if 'Relative' not in x and 'Relative' not in y:
        xy_max = np.min([df_to_plot[x].max(), df_to_plot[y].max()])
        fig.add_shape(type="line",
                    x0=0, y0=0,  # start of the line (bottom of the plot)
                    x1=xy_max, y1=xy_max,  # end of the line (top of the plot)
                    line=dict(color="black", width=1, dash="dash"),
                    xref='x',
                    yref='y')
    fig.update_layout(legend=dict(
        yanchor="top",
        y=1.02,
        xanchor="left",
        x=0.02
    ))
    fig.update_xaxes(title=f'{x}')# over Behavior model')
    fig.update_yaxes(title=f'{y}')# <br>over Trivial model')
    if output_folder is not None:
        ##
        # Make a figure for presentations with fewer names
        ##
        apply_figure_settings(fig, height_factor=0.4, width_factor=0.5)
        fname = os.path.join(output_folder, 'hierarchy_behavior_score_with_gfp_presentation.png')
        fig.write_image(fname, scale=7)

    apply_figure_settings(fig, height_factor=0.25, width_factor=0.5)

    if output_folder is not None:
        fname = os.path.join(output_folder, f'x-{x}_y-{y}.png')
        fig.write_image(fname, scale=3)
        fname = Path(fname).with_suffix('.svg')
        fig.write_image(fname)
        fname = Path(fname).with_suffix('.html')
        fig.write_html(fname)
        
    if to_show:
        fig.show()
    
    return fig, df_to_plot
