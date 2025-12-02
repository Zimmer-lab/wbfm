import logging
import math
import os
from functools import partial
from pathlib import Path
from typing import Optional, Union, Callable, List
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import scipy.io
from sklearn.decomposition import PCA

from wbfm.utils.general.utils_paper import paper_trace_settings, paper_figure_page_settings, \
    apply_figure_settings, behavior_name_mapping
from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes, options_for_ethogram, shade_using_behavior
from wbfm.utils.external.custom_errors import NoNeuronsError, NoBehaviorAnnotationsError
from wbfm.utils.external.utils_matplotlib import get_twin_axis
from wbfm.utils.external.utils_neuron_names import int2name_neuron, name2int_neuron_and_tracklet
from wbfm.utils.external.utils_pandas import cast_int_or_nan
from matplotlib import transforms, pyplot as plt
from matplotlib.ticker import NullFormatter, MultipleLocator
from tqdm.auto import tqdm
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.projects.utils_project import safe_cd
from wbfm.utils.traces.triggered_averages import FullDatasetTriggeredAverages
from wbfm.utils.general.high_performance_pandas import get_names_from_df
import matplotlib.style as mplstyle
from plotly.subplots import make_subplots
from plotly import graph_objects as go
from wbfm.utils.visualization.filtering_traces import fill_nan_in_dataframe
from wbfm.utils.visualization.utils_plot_traces import modify_dataframe_to_allow_gaps_for_plotly
import plotly.express as px
from wbfm.utils.general.utils_paper import plotly_paper_color_discrete_map


##
## New functions for use with project_config files
##

def make_grid_plot_from_project(project_data: ProjectData,
                                channel_mode: str = 'ratio',
                                calculation_mode: str = 'integration',
                                neuron_names_to_plot: list = None,
                                filter_mode: str = 'no_filtering',
                                color_using_behavior=True,
                                remove_outliers=False,
                                bleach_correct=True,
                                behavioral_correlation_shading=None,
                                direct_shading_dict=None,
                                df_traces: pd.DataFrame=None,
                                postprocessing_func: Optional[callable] = None,
                                min_nonnan=None,
                                share_y_axis=False,
                                to_save=True,
                                savename_suffix="",
                                title_str = None,
                                trace_kwargs=None,
                                **kwargs):
    """

    See project_data.calculate_traces for details on the arguments, and TracePlotter for even more detail

    See make_grid_plot_from_dataframe for a lower-level function that doesn't require a project object

    Design of this function:
        Build a dataframe of traces.
            With channel_mode=all, plots for multiple dataframes
        Build a set of functions to apply to each subplot based on the traces
        Plot all subplots

    Parameters
    ----------
    project_data: full project data
    channel_mode: trace calculation option; see calc_default_traces
    calculation_mode: trace calculation option; see calc_default_traces
    neuron_names_to_plot: subset of neurons to plot
    filter_mode: trace calculation option; see calc_default_traces
    color_using_behavior: if behavioral annotation exists, shade background for reversals and turns
    remove_outliers: trace calculation option; see calc_default_traces
    bleach_correct: trace calculation option; see calc_default_traces
    behavioral_correlation_shading: correlate to a particular behavior; see factory_correlate_trace_to_behavior_variable
    direct_shading_dict: instead of dynamic calculation using behavioral_correlation_shading, pass a value per neuron
    share_y_axis: subplot option
    min_nonnan: minimum tracking performance to include
    df_traces: traces dataframe that replaces all trace calculation options
    postprocessing_func: Callable that must accept the output of calculate_traces, and give the same type of output
    to_save: to export png within the project 4-traces folder; the name is based on the channel_mode and calculation_mode
    savename_suffix: for saving
    kwargs: passed to make_grid_plot_from_callables

    Returns
    -------

    """
    # Evaluate possible recursion
    if trace_kwargs is None:
        trace_kwargs = {}
    if channel_mode == 'all':
        # First, completely raw data
        all_modes = ['red', 'green']
        opt = dict(project_data=project_data,
                   calculation_mode=calculation_mode,
                   color_using_behavior=color_using_behavior,
                   bleach_correct=bleach_correct)
        for mode in all_modes:
            make_grid_plot_from_project(channel_mode=mode, **opt)
        # Second, remove outliers and filter
        all_modes = ['ratio']
        opt['remove_outliers'] = True
        opt['filter_mode'] = 'rolling_mean'
        for mode in all_modes:
            make_grid_plot_from_project(channel_mode=mode, **opt)
        return

    # Set up initial variables
    if neuron_names_to_plot is not None:
        neuron_names = neuron_names_to_plot
    else:
        if isinstance(min_nonnan, float):
            neuron_names = project_data.well_tracked_neuron_names(min_nonnan)
        else:
            neuron_names = project_data.neuron_names
    neuron_names.sort()

    # Build dataframe of all traces
    if df_traces is None:
        trace_options = {'channel_mode': channel_mode, 'calculation_mode': calculation_mode, 'filter_mode': filter_mode,
                         'remove_outliers': remove_outliers, 'bleach_correct': bleach_correct,
                         'neuron_names': tuple(neuron_names), 'min_nonnan': min_nonnan}
        trace_options.update(trace_kwargs)
        df_traces = project_data.calc_default_traces(**trace_options)
        # Recalculate the neuron names; they may have been renamed if rename_neurons_using_manual_ids was used
        neuron_names = get_names_from_df(df_traces)

    # Build functions to make a single subplot
    shade_plot_func = lambda axis, **kwargs: project_data.shade_axis_using_behavior(axis, **kwargs)
    logger = project_data.logger

    # Optional function: correlate to a behavioral variable or passed list
    assert direct_shading_dict is None or behavioral_correlation_shading is None, "Can't shade in both ways"
    if direct_shading_dict is None:
        background_shading_value_func = factory_correlate_trace_to_behavior_variable(project_data,
                                                                                     behavioral_correlation_shading)
    else:
        background_shading_value_func = lambda y, name: direct_shading_dict.get(name, None)

    ##
    # Actually make grid plot
    ##
    fig, _ = make_grid_plot_from_dataframe(df_traces, neuron_names,
                                           shade_plot_func=shade_plot_func,
                                           color_using_behavior=color_using_behavior,
                                           background_shading_value_func=background_shading_value_func,
                                           logger=logger,
                                           share_y_axis=share_y_axis, **kwargs)
    if title_str is None:
        title_str = project_data.shortened_name
    plt.suptitle(title_str, y=1.02, fontsize='xx-large')
    plt.tight_layout()

    # Save final figure and dataframe used to produce it
    if not savename_suffix.startswith('-'):
        savename_suffix = f"-{savename_suffix}"
    if to_save:
        if neuron_names_to_plot is None:
            prefix = f"{channel_mode}_{calculation_mode}"
            if remove_outliers:
                prefix = f"{prefix}_outliers_removed"
            if filter_mode != "no_filtering":
                prefix = f"{prefix}_{filter_mode}"
            if share_y_axis:
                prefix = f"{prefix}_sharey"
            if behavioral_correlation_shading is not None:
                if isinstance(behavioral_correlation_shading, str):
                    prefix = f"{prefix}_beh_{behavioral_correlation_shading}"
                else:
                    prefix = f"{prefix}_beh-custom"
            if 'shade_plot_kwargs' in kwargs:
                f"{prefix}_background_shading-custom"
            if trace_kwargs.get('rename_neurons_using_manual_ids', False):
                prefix = f"{prefix}_manual_ids"
            if trace_kwargs.get('residual_mode', False):
                prefix = f"{prefix}_residual"
            fname = f"{prefix}-grid{savename_suffix}.png"
        else:
            fname = f"{len(neuron_names_to_plot)}neurons_{channel_mode}_{calculation_mode}_grid_plot.png"
        traces_cfg = project_data.project_config.get_traces_config()
        out_fname = traces_cfg.resolve_relative_path(fname, prepend_subfolder=True)
        save_grid_plot(out_fname)

        fname = Path(fname).with_suffix('.h5')
        try:
            traces_cfg.save_data_in_local_project(df_traces, str(fname), prepend_subfolder=True)
        except ValueError as e:
            logger.warning(f"Couldn't save dataframe to {fname}; likely due to a duplicate column")
            logger.warning(f"Error: {e}")

    return fig


def make_grid_plot_from_dataframe(df: pd.DataFrame, neuron_names_to_plot: list = None, **kwargs):
    """
    Plots a grid of subplots, with each showing a single trace. By default plots all, but a sublist can be specified.

    Many options for sorting and shading the background are available.

    See make_grid_plot_using_project for a more high-level function
    Note: this function does NOT accept a project object; use make_grid_plot_using_project for that

    Parameters
    ----------
    df
    neuron_names_to_plot
    kwargs - see make_grid_plot_from_callables

    Returns
    -------

    """

    if neuron_names_to_plot is not None:
        neuron_names = neuron_names_to_plot
    else:
        neuron_names = get_names_from_df(df)
    neuron_names.sort()

    # Build functions to make a single subplot
    tspan = np.arange(df.shape[0])

    def get_data_func(neuron_name):
        # Make sure only a single column is returned
        y = df[neuron_name]
        if y.ndim > 1:
            y = y.iloc[:, 0]
            logging.warning(f"Multiple columns found for {neuron_name}; using only the first")
        return tspan, y

    fig, original_axes = make_grid_plot_from_callables(get_data_func, neuron_names, **kwargs)

    return fig, original_axes


def make_grid_plot_from_two_dataframes(df0, df1, twinx_when_reusing_figure=True, **kwargs):
    """

    Parameters
    ----------
    df0 - first trace (blue)
    df1 - second trace (orange)
    twinx_when_reusing_figure - Whether to plot the second trace on its own yaxis, or keep the same
    kwargs

    Returns
    -------

    """
    # Original grid plot
    fig, original_axes = make_grid_plot_from_dataframe(df0, **kwargs)
    # Don't want to pass these to the second plot
    if 'neuron_names_to_plot' not in kwargs:
        kwargs['neuron_names_to_plot'] = get_names_from_df(df0)
    if 'sort_using_shade_value' in kwargs:
        del kwargs['sort_using_shade_value']
    if 'background_shading_value_func' in kwargs:
        del kwargs['background_shading_value_func']
    # Plot second trace
    fig, _ = make_grid_plot_from_dataframe(df1, fig=fig, twinx_when_reusing_figure=twinx_when_reusing_figure, **kwargs)
    # Align y axes
    if kwargs.get('share_y_axis', False):
        twinned_axes = [get_twin_axis(ax) for ax in original_axes]
        # From: https://www.tutorialspoint.com/how-to-share-secondary-y-axis-between-subplots-in-matplotlib
        twinned_axes[0].get_shared_y_axes().join(*twinned_axes)
    return fig


def factory_correlate_trace_to_behavior_variable(project_data,
                                                 behavioral_correlation_shading: Union[str, Callable])\
        -> Optional[Callable]:
    """

    Parameters
    ----------
    project_data
    behavioral_correlation_shading

    Returns
    -------

    """
    valid_behavioral_shadings = ['absolute_speed', 'speed', 'positive_speed', 'negative_speed', 'curvature',
                                 'pc1']
    posture_class = project_data.worm_posture_class
    y = None
    if behavioral_correlation_shading is None:
        y = None
    elif isinstance(behavioral_correlation_shading, Callable):
        y = behavioral_correlation_shading(project_data)
    elif behavioral_correlation_shading == 'absolute_speed':
        y = posture_class.worm_speed(fluorescence_fps=True)
    elif behavioral_correlation_shading == 'speed':
        y = posture_class.worm_speed(fluorescence_fps=True, signed=True)
    elif behavioral_correlation_shading == 'positive_speed':
        y = posture_class.worm_speed(fluorescence_fps=True, signed=True)
        beh_ind = posture_class.beh_annotation(fluorescence_fps=True)
        rev_ind = BehaviorCodes.vector_equality(beh_ind, BehaviorCodes.REV)
        y[rev_ind] = 0
    elif behavioral_correlation_shading == 'negative_speed':
        y = posture_class.worm_speed(fluorescence_fps=True, signed=True)
        beh_ind = posture_class.beh_annotation(fluorescence_fps=True)
        fwd_ind = BehaviorCodes.vector_equality(beh_ind, BehaviorCodes.FWD)
        y[fwd_ind] = 0
    elif behavioral_correlation_shading == 'curvature':
        y = posture_class.summed_curvature_from_kymograph(fluorescence_fps=True)
    elif behavioral_correlation_shading == 'pc1':
        # Note: this does not require the kymograph
        model = PCA(n_components=1)
        try:
            df = project_data.calc_default_traces(interpolate_nan=True, min_nonnan=0.9)
        except NoNeuronsError:
            df = project_data.calc_default_traces(interpolate_nan=True, min_nonnan=0.5)
        df -= df.mean()
        model.fit(df.T)
        y = np.squeeze(model.components_)
    else:
        # Try to calculate via an alias
        try:
            y = project_data.worm_posture_class.calc_behavior_from_alias(behavioral_correlation_shading)
        except NotImplementedError:
            assert behavioral_correlation_shading in valid_behavioral_shadings, \
                f"Must pass None or one of: {valid_behavioral_shadings}"

    if y is None:
        return None

    def background_shading_value_func(X, name):
        ind = np.where(~np.isnan(X))[0]
        return np.corrcoef(X[ind], y[:len(X)][ind])[0, 1]

    return background_shading_value_func


def make_grid_plot_from_leifer_file(fname: str,
                                    channel_mode: str = 'all',
                                    color_using_behavior=True):
    if channel_mode == 'all':
        all_modes = ['rRaw', 'gRaw', 'Ratio2']
        opt = dict(fname=fname,
                   color_using_behavior=color_using_behavior)
        for mode in all_modes:
            make_grid_plot_from_leifer_file(channel_mode=mode, **opt)
        return

    assert channel_mode in ['rRaw', 'gRaw', 'Ratio2']

    data = scipy.io.loadmat(fname)

    ethogram = [cast_int_or_nan(d) for d in data['behavior'][0][0][0]]
    # ethogram_names = {-1: 'Reversal', 1: 'Forward', 2: 'Turn'}
    ethogram_cmap = {-1: 'darkgray', 0: None, 1: None, 2: 'red'}

    num_neurons, t = data[channel_mode].shape
    neuron_names = [int2name_neuron(i + 1) for i in range(num_neurons)]

    # Build functions to make a single subplot
    get_data_func = lambda neuron_name: (np.arange(t), data[channel_mode][name2int_neuron_and_tracklet(neuron_name) - 1])
    shade_plot_func = lambda axis: shade_using_behavior(ethogram, axis, cmap=ethogram_cmap)
    logger = logging.getLogger()

    make_grid_plot_from_callables(get_data_func, neuron_names, shade_plot_func,
                                  color_using_behavior=color_using_behavior, logger=logger)

    # Save final figure
    out_fname = f"leifer_{channel_mode}_grid_plot.png"
    out_fname = Path(fname).with_name(out_fname)

    save_grid_plot(out_fname)


def save_grid_plot(out_fname):
    plt.subplots_adjust(left=0,
                        bottom=0,
                        right=1,
                        top=1,
                        wspace=0.0,
                        hspace=0.0)
    logging.info(f"Saving figure at: {out_fname}")
    plt.savefig(out_fname, bbox_inches='tight', pad_inches=0)


def make_grid_plot_from_callables(get_data_func: callable,
                                  neuron_names: list,
                                  shade_plot_func: callable = None,
                                  background_shading_value_func: callable = None,
                                  color_using_behavior: bool = True,
                                  shade_plot_kwargs: dict = None,
                                  share_y_axis: bool = False,
                                  logger: logging.Logger = None,
                                  num_columns: int = 5,
                                  twinx_when_reusing_figure: bool = False,
                                  fig=None,
                                  sort_using_shade_value=False,
                                  sort_without_shading=False,
                                  ax_plot_func: Optional[Union[callable, List[callable]]] = None,
                                  fig_opt=None,
                                  **plot_kwargs):
    """
    Makes a grid plot from callables. Designed to plot either raw time series, or a function that processes time series
    and returns another time series (get_data_func or ax_plot_func)

    Parameters
    ----------
    get_data_func: function that accepts a neuron name and returns a tuple of (t, y)
    neuron_names: list of neurons to plot
    shade_plot_func: function that accepts an axis object and shades the plot
    background_shading_value_func: function to get a value to shade the background, e.g. correlation to behavior
    color_using_behavior: whether to use the shade_plot_func
    shade_plot_kwargs: kwargs to pass to shade_plot_func
    ax_plot_func: signature: (t, y, ax, name, **kwargs) -> None [should plot something on the given axis]
    share_y_axis: whether to share the y axis between all subplots
    logger: logger object (optional)
    num_columns: number of columns in the grid
    twinx_when_reusing_figure: whether to plot the second trace on its own xaxis, or keep the same
    sort_using_shade_value: whether to sort the neurons based on the background_shading_value_func
    sort_without_shading: whether to sort the neurons based on the background_shading_value_func, but not shade
    fig: matplotlib figure object to use
    fig_opt: kwargs to pass to plt.subplots. Only used if fig is None
    plot_kwargs: kwargs to pass as part of ax_opt to _ax_plot_func(), e.g. alpha 
    

    Example:
    get_data_func = lambda neuron_name: project_data.calculate_traces(neuron_name=neuron_name, **options)
    shade_plot_func = project_data.shade_axis_using_behavior

    Returns
    -------

    """
    
    
    if fig_opt is None:
        fig_opt = {}
    if shade_plot_kwargs is None:
        shade_plot_kwargs = {}
    if ax_plot_func is None:
        _ax_plot_func = lambda t, y, ax, name, **kwargs: ax.plot(t, y, **kwargs)
    elif isinstance(ax_plot_func, list):
        _ax_plot_func = lambda *args, **kwargs: [f(*args, **kwargs) for f in ax_plot_func]
    else:
        # Assume ax_plot_func has the correct signature
        _ax_plot_func = ax_plot_func

    # Get the data in the beginning, to make sure there are no errors
    all_y_t = {}
    for name in neuron_names:
        try:
            t, y = get_data_func(name)
            all_y_t[name] = [t, y]
        except ValueError:
            continue

    # Set up the colormap of the background, if any
    if background_shading_value_func is not None:
        # From: https://stackoverflow.com/questions/59638155/how-to-set-0-to-white-at-a-uneven-color-ramp

        # First get all the traces, so that the entire cmap can be scaled
        assert len(all_y_t) > 1, "Cannot calculate a colormap for a single trace"
        all_vals = [background_shading_value_func(yt[1], name) for name, yt in all_y_t.items()]

        if sort_using_shade_value or sort_without_shading:
            # Sort descending
            ind = np.argsort(-np.array(all_vals))
            # neuron_names = [neuron_names[i] for i in ind]
            keys = list(all_y_t.keys())
            all_y_t = {keys[i]: all_y_t[keys[i]] for i in ind}
            all_vals = [all_vals[i] for i in ind]

        norm = TwoSlopeNorm(vmin=np.nanmin(all_vals), vcenter=0, vmax=np.nanmax(all_vals))
        # norm.autoscale(all_vals)
        values_normalized = norm(all_vals)
        correlation_shading_colors = plt.cm.PiYG(values_normalized)

    else:
        correlation_shading_colors = []
        all_vals = None

    # Set up grid of subplots
    num_neurons = len(all_y_t)
    num_rows = int(np.ceil(num_neurons / float(num_columns)))
    default_fig_opt = dict(dpi=150, figsize=(25, 25), sharey=share_y_axis, sharex=True)
    default_fig_opt.update(fig_opt)
    if logger is not None:
        logger.info(f"Found {num_neurons} neurons; shaping to grid of shape {(num_rows, num_columns)}")
    if fig is None:
        fig, original_axes = plt.subplots(num_rows, num_columns, **default_fig_opt)
        new_fig = True
    else:
        original_axes = fig.axes
        new_fig = False

    # Loop through neurons and: plot, label, and shade, or apply custom function
    for i, (neuron_name, yt) in tqdm(enumerate(all_y_t.items())):
        ax = fig.axes[i]
        if twinx_when_reusing_figure and not new_fig:
            ax = ax.twinx().twiny()
            ax_opt = dict(color='tab:orange', **plot_kwargs)
        else:
            ax_opt = dict()
        t, y = yt

        if not new_fig:
            # ax.plot(t, y, **ax_opt)
            _ax_plot_func(t, y, ax, neuron_name, **ax_opt)
        else:
            _ax_plot_func(t, y, ax, neuron_name, label=neuron_name)
            # ax.plot(t, y, label=neuron_name)
            # For removing the lines from the legends:
            # https://stackoverflow.com/questions/25123127/how-do-you-just-show-the-text-label-in-plot-legend-e-g-remove-a-labels-line
            leg = ax.legend(loc='upper left', handlelength=0, handletextpad=0, fancybox=True, framealpha=0.0)
            for item in leg.legendHandles:
                item.set_visible(False)
            ax.set_frame_on(False)
            ax.set_axis_off()

            # Additional layers of information on the axes
            if color_using_behavior and shade_plot_func is not None:
                shade_plot_func(ax, **shade_plot_kwargs)

            if background_shading_value_func is not None and not sort_without_shading:
                color, val = correlation_shading_colors[i], background_shading_value_func(y, neuron_name)
                ax.axhspan(y.min(), y.max(), xmax=len(y), facecolor=color, alpha=0.25, zorder=-100)
                ax.set_title(f"Shaded value (below): {val:0.2f}")

    return fig, original_axes


def _plot_subplots(y1, y2):
    fig, axes = plt.subplots(ncols=2, figsize=(15, 5), dpi=100)

    ax1 = axes[0]
    ax1.plot(y1, label='Original trace')
    ax1_2 = ax1.twinx()
    ax1_2.plot(y2, 'tab:orange')
    ax1.legend()

    window = [500, 700]
    for w in window:
        ax1.plot([w, w], [np.nanmin(y1), np.nanmax(y1)], color='black', lw=3)

    ax2 = axes[1]
    ax2.plot(y1[window[0]:window[1]], lw=2)
    ax2_2 = ax2.twinx()
    ax2_2.plot(y2[window[0]:window[1]], 'tab:orange', lw=2, label='Modified trace')
    ax2_2.legend()

    return ax1, ax2


def title_from_params(params):
    t = ''
    for key, val in params.items():
        if isinstance(val, str):
            k = key.split('_')[0]
            t += f'{k}={val}-'

    return t[:-1]


def plot_compare_two_calculation_methods(project_data, neuron_name, variable_dict=None, **kwargs):
    """
    kwargs:
        channel_mode: str,
        calculation_mode: str,
        neuron_name: str,
        filter_mode: str = 'no_filtering'
    """

    default_kwargs = dict(
        channel_mode='dr_over_r_20',
        calculation_mode='integration',
        remove_outliers=True
    )
    default_kwargs.update(kwargs)

    t, y1 = project_data.calculate_traces(neuron_name=neuron_name, **default_kwargs)

    for key, val_list in variable_dict.items():
        for val in val_list:
            default_kwargs[key] = val
            t, y2 = project_data.calculate_traces(neuron_name=neuron_name, **default_kwargs)

            ax1, ax2 = _plot_subplots(y1, y2)

            title = title_from_params(default_kwargs)
            ax1.set_title(title)
            ax2.set_title(neuron_name)

##
## Generally plotting
##


def plot3d_with_max(dat, z, t, max_ind, vmin=100, vmax=400):
    plt.imshow(dat[:, :, z, t], vmin=vmin, vmax=vmax)
    plt.colorbar()
    x, y = max_ind[t, 1], max_ind[t, 0]
    if z == max_ind[t, 2]:
        plt.scatter(x, y, marker='x', c='r')
    plt.title(f"Max for t={t} is on z={max_ind[t, 2]}, xy={x},{y}")


def plot3d_with_max_and_hist(dat, z, t, max_ind):
    # From: https://matplotlib.org/2.0.2/examples/pylab_examples/scatter_hist.html
    rot = transforms.Affine2D().rotate_deg(90)
    nullfmt = NullFormatter()  # no labels

    plt.figure(1, figsize=(8, 8))

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    axIm = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # Actually display
    frame = dat[:, :, z, t]
    axIm.imshow(frame, vmin=0, vmax=400)
    x, y = max_ind[t, 1], max_ind[t, 0]
    #     if z == max_ind[t,2]:
    #         plt.scatter(x, y, marker='x', c='r')
    #     plt.title(f"Max for t={t} is on z={max_ind[t,2]}, xy={x},{y}")

    axHistx.plot(np.max(frame, axis=0))

    #     base = plt.gca().transData
    axHisty.plot(np.flip(np.max(frame, axis=1)), range(frame.shape[0]))  # , transform=base+rot)


##
## Helper functions
##


def get_tracking_channel(t_dict):
    try:
        dat = t_dict['mcherry']
    except KeyError:
        dat = t_dict['red']
    return dat


def get_measurement_channel(t_dict):
    try:
        dat = t_dict['gcamp']
    except KeyError:
        dat = t_dict['green']
    return dat


##
## For interactivity
##

class ClickableGridPlot:
    def __init__(self, project_data, verbose=3):

        # Set up grid plot
        opt = dict(channel_mode='ratio',
                   calculation_mode='integration',
                   filter_mode='rolling_mean',
                   to_save=False)

        mplstyle.use('fast')
        with safe_cd(project_data.project_dir):
            fig = make_grid_plot_from_project(project_data, **opt)

        self.fig = fig
        self.project_data = project_data

        # Set up metadata objects
        names = project_data.neuron_names
        self.selected_neurons = {n: {"List ID": 0, "Proposed Name": n} for n in names}
        self.current_list_index = 1
        self.current_selected_label = None
        self.verbose = verbose

        # Set up text box for modifying names
        # plt.subplots_adjust(bottom=0.2)
        # axbox = plt.axes([0.1, 0.05, 0.8, 0.075])
        # self.text_box = TextBox(axbox, 'Modify neuron name', initial="initial_text")
        # self.text_box.on_submit(self.modify_neuron_name)

        # Finish
        self.connect()
        # Load file and add initial colors, if any
        self.load_previous_file()
        plt.show()

    def connect(self):
        cid = self.fig.canvas.mpl_connect('button_press_event', self.shade_selected_subplot_callback)
        cid = self.fig.canvas.mpl_connect('key_press_event', self.update_current_list_index)
        cid = self.fig.canvas.mpl_connect('close_event', self.write_file)

    def update_current_list_index(self, event):
        if event.key in ['1', '2', '3']:
            self.current_list_index = int(event.key)
        else:
            self.current_list_index = 0

        print(f"Current list index: {self.current_list_index}")

    def modify_neuron_name(self, text):
        self.selected_neurons[self.current_selected_label]["Proposed Name"] = text

    def update_selected_label(self, new_label):
        self.current_selected_label = new_label
        # self.text_box.set_val(new_label)

    def get_color_from_list_index(self):
        print(f"Getting color: {self.current_list_index}")
        if self.current_list_index == 1:
            return 'green'
        elif self.current_list_index == 2:
            return 'blue'
        else:
            return 'red'

    def shade_selected_subplot_callback(self, event):
        ax = event.inaxes
        if self.verbose >= 3:
            print(event)
            print(ax)
        if ax is None or len(ax.lines) == 0:
            return
        button_pressed = event.button

        self.shade_selected_subplot(ax, button_pressed)

    def shade_selected_subplot(self, ax, button_pressed):

        line = ax.lines[0]
        label = line.get_label()
        self.update_selected_label(label)

        # Button codes: https://matplotlib.org/stable/api/backend_bases_api.html#matplotlib.backend_bases.MouseButton
        if button_pressed == 1:
            # Left click = select neuron
            if self.selected_neurons[label]["List ID"] == self.current_list_index:
                print(f"{label} already selected")
            else:
                print(f"Selecting {label}")
                self._reset_shading(ax)

                y = line.get_ydata()
                color = self.get_color_from_list_index()

                shading = ax.axhspan(np.nanmin(y), np.nanmax(y), xmax=len(y), facecolor=color, alpha=0.25, zorder=-100)
                ax.draw_artist(shading)

                self.selected_neurons[label]["List ID"] = self.current_list_index

        elif button_pressed == 3:
            # Right click = deselect
            if self.selected_neurons[label]["List ID"] == 0:
                print(f"{label} not selected")
            else:
                print(f"Deselecting {label}")
                self._reset_shading(ax)
                plt.draw()
                self.selected_neurons[label]["List ID"] = 0
        else:
            print("Button press detected, but did nothing")
        # From: https://stackoverflow.com/questions/29277080/efficient-matplotlib-redrawing
        ax.figure.canvas.blit(ax.bbox)
        # if verbose >= 2:
        #     print("Currently selected neuron:")
        #     print(self.selected_neurons)

    def _reset_shading(self, ax):
        if len(ax.patches) > 0:
            [p.remove() for p in ax.patches]
            # ax.patches = []

    def write_file(self, event):
        log_dir = self.project_data.project_config.get_visualization_config(make_subfolder=True).absolute_subfolder
        fname = os.path.join(log_dir, 'selected_neurons.csv')
        # fname = get_sequential_filename(fname)
        print(f"Saving: {fname}")

        df = pd.DataFrame(self.selected_neurons)
        df.T.to_csv(path_or_buf=fname, index=True)
        fname = Path(fname).with_suffix('.xlsx')
        df.T.to_excel(fname, index=True)
        # df = pd.DataFrame(self.selected_neurons, index=[0])
        # df.to_csv(path_or_buf=fname, header=True, index=False)

        print(df.T)

    def load_previous_file(self):
        visualization_directory = self.project_data.project_config.get_visualization_config().absolute_subfolder
        fname = os.path.join(visualization_directory, 'selected_neurons.csv')
        if not os.path.exists(fname):
            print(f"Did not find previous state at: {fname}")
            return
        else:
            # plt.show(block=False)
            self.fig.canvas.draw()
            print(f"Reading previous state from: {fname}")
            df = pd.read_csv(fname, index_col=0)

            axes = self.fig.axes
            button_pressed = 1

            for ax, (name, list_index) in zip(axes, df.iterrows()):
                if list_index[0] == 0:
                    continue

                # Add the shading to this axis
                print(f"Shading {name} with index {list_index[0]} and name {list_index[1]}")
                self.current_list_index = list_index[0]
                self.shade_selected_subplot(ax, button_pressed)

                # Also add the info to the dict
                self.selected_neurons[name]["List ID"] = list_index[0]
                self.selected_neurons[name]["Proposed Name"] = list_index[1]

        plt.draw()
        self.current_list_index = 1


def make_heatmap_using_project(project_data: ProjectData, to_save=True, plot_kwargs=None, trace_kwargs=None,
                               also_plot_zscore=False, neuron_names_to_plot=None):
    """
    Uses seaborn to make a heatmap, including clustering of the traces

    Parameters
    ----------
    trace_kwargs
    plot_kwargs
    project_data
    to_save

    Returns
    -------

    """

    default_trace_kwargs = dict(interpolate_nan=True, filter_mode='rolling_mean', channel_mode='dr_over_r_20')
    if trace_kwargs is not None:
        default_trace_kwargs.update(trace_kwargs)
    trace_kwargs = default_trace_kwargs

    default_plot_kwargs = dict(metric="correlation", cmap='jet', figsize=(15, 10), col_cluster=False,
                               cbar_pos=(-0.01, 0.2, 0.02, 0.4))
    if plot_kwargs is not None:
        default_plot_kwargs.update(plot_kwargs)
    plot_kwargs = default_plot_kwargs

    # Calculate
    try:
        df = project_data.calc_default_traces(**trace_kwargs).T
    except ValueError:
        logging.warning("Value error when interpolating traces; probably this means there wasn't enough data")
        return

    if neuron_names_to_plot is not None:
        df = df.loc[neuron_names_to_plot, :]

    if 'vmin' not in plot_kwargs:
        plot_kwargs['vmin'] = 2*np.quantile(df.values, 0.1)
    if 'vmax' not in plot_kwargs:
        plot_kwargs['vmax'] = 2*np.quantile(df.values, 0.95)

    # Plot
    import seaborn as sns
    fig = sns.clustermap(df, **plot_kwargs)
    if project_data.use_physical_time:
        ax = fig.ax_heatmap
        ax.xaxis.set_major_locator(MultipleLocator(project_data.physical_unit_conversion.volumes_per_second*60))
        x = ax.get_xticks()
        x = [int(i) for i in x if (project_data.num_frames > i >= 0)]
        labels = [int(np.round(project_data.x_for_plots[i])) for i in x]
        ax.set_xticks(ticks=x, labels=labels)
        ax.set_xlabel("Time (seconds)")
    fig.ax_heatmap.set_title(f"Heatmap for project {project_data.shortened_name}")

    if also_plot_zscore:
        plot_kwargs['z_score'] = 0
        fig_zscore = sns.clustermap(df, **plot_kwargs)

    # Save
    if to_save:
        traces_cfg = project_data.project_config.get_traces_config()

        fname = 'heatmap.png'
        fname = traces_cfg.resolve_relative_path(fname, prepend_subfolder=True)
        fig.savefig(fname)

        if also_plot_zscore:
            fname = 'heatmap_zscore.png'
            fname = traces_cfg.resolve_relative_path(fname, prepend_subfolder=True)
            fig_zscore.savefig(fname)

    return fig


def make_default_summary_plots_using_config(proj_dat: ProjectData):
    # Note: reloads the project data to properly read the new trace h5 files
    logger = proj_dat.logger
    logger.info("Making default grid plots")
    grid_opt = paper_trace_settings()
    grid_opt['channel_mode'] = 'all'
    grid_opt['min_nonnan'] = None
    grid_opt['interpolate_nan'] = False
    grid_opt['rename_neurons_using_manual_ids'] = False
    try:
        make_grid_plot_from_project(proj_dat, **grid_opt)
    except NoNeuronsError:
        pass
    # Also save a heatmap and a colored plot
    try:
        make_heatmap_using_project(proj_dat, to_save=True)
    except (NoNeuronsError, ValueError):
        pass
    # Also save a PC1-correlated grid plot
    grid_opt['only_keep_confident_ids'] = False
    grid_opt['rename_neurons_using_manual_ids'] = True
    grid_opt['behavioral_correlation_shading'] = 'pc1'
    grid_opt['sort_using_shade_value'] = True
    grid_opt['interpolate_nan'] = True
    try:
        make_grid_plot_from_project(proj_dat, **grid_opt)
    except (np.linalg.LinAlgError, ValueError, NoNeuronsError) as e:
        # For test projects, this will fail due to too little data
        logger.warning("Failed to make PC1 grid plot; if this is a test project this may be expected")
        logger.info(e)
        pass


def make_default_triggered_average_plots(project_cfg, to_save=True):

    # Load data class
    project_data = ProjectData.load_final_project_data_from_config(project_cfg)
    vis_cfg = project_data.project_config.get_visualization_config()
    project_data.verbose = 0
    all_triggers = dict(reversal=BehaviorCodes.REV, forward=BehaviorCodes.FWD)
    # Build triggered average class
    trace_opt = dict(channel_mode='ratio', calculation_mode='integration', min_nonnan=0.8)
    df = project_data.calc_default_traces(**trace_opt)
    trigger_opt = dict(min_lines=2, ind_preceding=20, state=None)
    min_significant = 20
    ind_class = project_data.worm_posture_class.calc_triggered_average_indices(**trigger_opt)
    triggered_averages_class = FullDatasetTriggeredAverages(df, ind_class, min_points_for_significance=min_significant)

    # Options for the traces within the grid plot
    trace_and_plot_opt = dict(to_save=False, color_using_behavior=False, share_y_axis=False,
                              behavioral_correlation_shading='pc1', sort_without_shading=True)
    trace_and_plot_opt.update(trace_opt)

    # Loop
    for name, state in all_triggers.items():
        # Change option within class
        triggered_averages_class.ind_class.behavioral_state = state
        _make_three_triggered_average_grid_plots(name, project_data, to_save, trace_and_plot_opt,
                                                 triggered_averages_class, vis_cfg)


def make_fwd_and_turn_triggered_average_plots(project_cfg, turn_state=BehaviorCodes.VENTRAL_TURN,
                                              to_save=True):
    """
    Makes a grid plot with forward and ventral turn triggered averages plotted on top of each other

    Parameters
    ----------
    project_cfg
    turn_state
    to_save

    Returns
    -------

    """

    # Load data class
    project_data = ProjectData.load_final_project_data_from_config(project_cfg)
    project_data.verbose = 0
    # Options for the traces
    trace_opt = dict(rename_neurons_using_manual_ids=True)
    # Build triggered average class
    trigger_class_fwd = FullDatasetTriggeredAverages.load_from_project(project_data,
                                                                       trigger_opt=dict(state=BehaviorCodes.FWD),
                                                                       trace_opt=trace_opt)
    trigger_class_turn = FullDatasetTriggeredAverages.load_from_project(project_data,
                                                                        trigger_opt=dict(state=turn_state),
                                                                        trace_opt=trace_opt)

    # Make sure there are any events
    if trigger_class_turn.ind_class.num_events == 0:
        print("Skipping turn triggered average plot because there are no events")
        return

    # Actually make the plot
    func1 = trigger_class_fwd.ax_plot_func_for_grid_plot
    func2 = lambda *args, **kwargs: trigger_class_turn.ax_plot_func_for_grid_plot(*args, is_second_plot=True, **kwargs)

    df_traces = trigger_class_turn.df_traces.copy()
    fig, original_axes = make_grid_plot_from_dataframe(df_traces, ax_plot_func=func1)
    fig, original_axes = make_grid_plot_from_dataframe(df_traces, ax_plot_func=func2, fig=fig)

    title_str = project_data.shortened_name
    plt.suptitle(title_str, y=1.02, fontsize='xx-large')
    plt.tight_layout()

    # Save
    if to_save:
        project_data.save_fig_in_project(suffix='fwd_vt_triggered_average', overwrite=True)


def _make_three_triggered_average_grid_plots(name, project_data, to_save, trace_and_plot_opt,
                                             triggered_averages_class, vis_cfg):
    # First, simple gridplot
    func = triggered_averages_class.ax_plot_func_for_grid_plot
    make_grid_plot_from_project(project_data, **trace_and_plot_opt, ax_plot_func=func)
    if to_save:
        project_data.save_fig_in_project(suffix='triggered_average_simple', overwrite=True)
    # Second, gridplot with "significant" points marked
    func = partial(triggered_averages_class.ax_plot_func_for_grid_plot,
                   show_individual_lines=True, color_significant_times=True)
    make_grid_plot_from_project(project_data, **trace_and_plot_opt, ax_plot_func=func)
    if to_save:
        project_data.save_fig_in_project(suffix='triggered_average_significant_points_marked', overwrite=True)
    # Finally, a smaller subset of the grid plot (only neurons with enough signficant points)
    subset_neurons, _, _ = triggered_averages_class.which_neurons_are_significant()
    func = partial(triggered_averages_class.ax_plot_func_for_grid_plot)
    make_grid_plot_from_project(project_data, **trace_and_plot_opt, ax_plot_func=func,
                                neuron_names_to_plot=subset_neurons)
    if to_save:
        project_data.save_fig_in_project(suffix='triggered_average_neuron_subset', overwrite=True)


def make_pirouette_split_triggered_average_plots(project_cfg, to_save=True):

    # Load data class
    project_data = ProjectData.load_final_project_data_from_config(project_cfg)
    vis_cfg = project_data.project_config.get_visualization_config()
    project_data.verbose = 0
    all_triggers = dict(reversal=1, forward=0)
    # Build triggered average class
    trace_opt = dict(channel_mode='ratio', calculation_mode='integration', min_nonnan=0.8)
    df = project_data.calc_default_traces(**trace_opt)
    trigger_opt = dict(min_lines=2, ind_preceding=20)
    min_significant = 5
    ind_rev_pirouette, ind_rev_non_pirouette = \
        project_data.worm_posture_class.calc_triggered_average_indices_with_pirouette_split(**trigger_opt)
    if len(ind_rev_pirouette.num_events) == 0 or len(ind_rev_non_pirouette.num_events) == 0:
        project_data.logger.warning("No events found; not plotting pirouette split triggered averages")
        return
    triggered_averages_pirouette = FullDatasetTriggeredAverages(df, ind_rev_pirouette,
                                                                min_points_for_significance=min_significant)
    triggered_averages_non_pirouette = FullDatasetTriggeredAverages(df, ind_rev_non_pirouette,
                                                                    min_points_for_significance=min_significant)

    # Options for the traces within the grid plot
    trace_and_plot_opt = dict(to_save=False, color_using_behavior=False, share_y_axis=False,
                              behavioral_correlation_shading='pc1', sort_without_shading=True)
    trace_and_plot_opt.update(trace_opt)

    # Plot and save
    name = "pirouette"
    triggered_averages_class = triggered_averages_pirouette
    _make_three_triggered_average_grid_plots(name, project_data, to_save, trace_and_plot_opt,
                                             triggered_averages_class, vis_cfg)

    name = "non_pirouette"
    triggered_averages_class = triggered_averages_non_pirouette
    _make_three_triggered_average_grid_plots(name, project_data, to_save, trace_and_plot_opt,
                                             triggered_averages_class, vis_cfg)


def make_summary_interactive_heatmap_with_pca(project_cfg, to_save=True, to_show=False, trace_opt=None,
                                              output_folder=None, **kwargs):

    base_font_size = 18

    project_data = ProjectData.load_final_project_data_from_config(project_cfg)
    num_pca_modes_to_plot = 3
    column_widths, ethogram_opt, heatmap, heatmap_opt, kymograph, kymograph_opt, phase_plot_list, phase_plot_list_opt, row_heights, subplot_titles, trace_list, trace_opt_list, trace_shading_opt, var_explained_line, var_explained_line_opt, weights_list, weights_opt_list = build_all_plot_variables_for_summary_plot(
        project_data, num_pca_modes_to_plot, trace_opt=trace_opt, **kwargs)

    rows = 1 + num_pca_modes_to_plot + 2
    cols = 1 + num_pca_modes_to_plot

    # Build figure

    ### First column: x axis is time
    fig = make_subplots(rows=rows, cols=cols, shared_xaxes=False, shared_yaxes=False,
                        row_heights=row_heights, column_widths=column_widths,
                        horizontal_spacing=0.04, vertical_spacing=0.05,
                        subplot_titles=subplot_titles,
                        specs=[[{}, {}, {}, {}],
                               [{}, {"rowspan": 4, "colspan": 3, "type": "scene"}, None, None],
                               [{}, None, None, None],
                               [{}, None, None, None],
                               [{}, None, None, None],
                               [{}, {"rowspan": 1, "colspan": 3}, None, None]])

    fig.add_trace(heatmap, **heatmap_opt)
    for opt in ethogram_opt:
        fig.add_shape(**opt, row=2, col=1)
    for trace, trace_opt in zip(trace_list, trace_opt_list):
        fig.add_trace(trace, **trace_opt)
        num_before_adding_shapes = len(fig.layout.shapes)
        for shade_opt in trace_shading_opt:
            shade_opt['y1'] = 0.5 # Will be half the overall plot
            fig.add_shape(**shade_opt, row=trace_opt['row'], col=trace_opt['col'])
        # Force yref in all of these new shapes, which doesn't really work for subplots
        # But here it is hardcoded as 50% of the overall plot (extending across subplots)
        for i in range(num_before_adding_shapes, len(fig.layout.shapes)):
            fig.layout.shapes[i]['yref'] = 'paper'

    ### Second column
    for trace, trace_opt in zip(weights_list, weights_opt_list):
        fig.add_trace(trace, **trace_opt)
    fig.add_traces(phase_plot_list, **phase_plot_list_opt)
    fig.add_trace(var_explained_line, **var_explained_line_opt)

    ### Final updates
    fig.update_xaxes(dict(showticklabels=False, showgrid=False), col=1, overwrite=True, matches='x')
    fig.update_yaxes(dict(showticklabels=False, showgrid=False), col=1, overwrite=True)

    fig.update_xaxes(dict(showticklabels=False), row=1, overwrite=True)
    fig.update_yaxes(dict(showticklabels=False), row=1, overwrite=True, matches='y')

    # Remove ticks on the 3d plot
    # https://community.plotly.com/t/dont-show-ticks-on-3d-scatter/35695/2
    fig.update_layout(
        scene=dict(
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(showticklabels=False),
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3'
        )
    )

    fig.update_xaxes(dict(showticklabels=True, title='Time (seconds)'), row=6, col=1, overwrite=True,)
    # fig.update_yaxes(dict(showticklabels=True), row=6, col=1, overwrite=True)

    fig.update_layout(showlegend=True, autosize=False, width=1.5*1000, height=1.5*800)

    # Add a single legend for behavior colors
    fig.update_layout(
        legend=dict(
            itemsizing='constant',  # Display legend items as colored boxes and text
            x=0.63,  # Adjust the x position of the legend
            y=0.5, #0.54,  # Adjust the y position of the legend
            bgcolor='rgba(0, 0, 0, 0.00)',  # Set the background color of the legend
            bordercolor='Black',  # Set the border color of the legend
            borderwidth=1,  # Set the border width of the legend
            font=dict(size=base_font_size)  # Set the font size of the legend text
        )
    )
    # Transparent background and remove lines
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    # Fonts
    fig.update_layout(font=dict(size=base_font_size),
                      title=dict(font=dict(size=base_font_size+2)))
    # Unclear why the colorscale is not jet, but this fixes it
    fig.update_layout(coloraxis1=dict(colorscale='jet'))
    # Fix the location and size of the colorbar
    fig.update_coloraxes(colorbar=dict(
        x=-0.07,  # Adjust the x position to move the colorbar to the right (1 is the rightmost position)
        len=0.45,  # Set the length to determine the size of the colorbar
        y=0.8,  # Adjust the y position to center the colorbar vertically
        title=dict(text='dR/R20', font=dict(size=base_font_size))
    ))

    if to_show:
        fig.show()

    if to_save:
        _save_plotly_all_types(fig, project_data, fname='summary_trace_plot.html', output_folder=output_folder)

    return fig


def make_summary_heatmap_and_subplots(project_cfg, to_save=True, to_show=False, trace_opt=None,
                                      include_speed_subplot=True, ethogram_on_top=False,
                                      output_folder=None, base_width=0.5, base_height=0.3, **kwargs):
    """
    Similar to make_summary_interactive_heatmap_with_pca, but saves each subplot separately for more control

    Parameters
    ----------
    project_cfg
    to_save
    to_show
    trace_opt
    kwargs

    Returns
    -------

    """

    if trace_opt is None:
        trace_opt = paper_trace_settings()

    if not isinstance(base_height, list):
        base_height = [base_height, base_height*(2/3)]
    if not isinstance(base_width, list):
        base_width = [base_width, base_width*0.8]

    # Get figure options
    figure_opt = paper_figure_page_settings()
    font_dict = figure_opt['plotly_font_opt']
    # plotly_opt = figure_opt['plotly_opt']

    project_data = ProjectData.load_final_project_data_from_config(project_cfg)
    num_pca_modes_to_plot = 3
    column_widths, ethogram_opt, heatmap, heatmap_opt, kymograph, kymograph_opt, phase_plot_list, phase_plot_list_opt, row_heights, subplot_titles, trace_list, trace_opt_list, trace_shading_opt, var_explained_line, var_explained_line_opt, weights_list, weights_opt_list = build_all_plot_variables_for_summary_plot(
        project_data, num_pca_modes_to_plot, trace_opt=trace_opt, **kwargs)

    # Build figure 1: heatmap
    if ethogram_on_top:
        _opt = dict(rows=2, row_heights=[0.9, 0.1], vertical_spacing=0.0)
    else:
        _opt = dict(rows=1)
    fig1 = make_subplots(**_opt)
    fig1.add_trace(heatmap, **heatmap_opt)
    fig1.update_layout(showlegend=False, autosize=False, #**plotly_opt,
                       coloraxis=dict(colorscale="jet"))
    apply_figure_settings(fig1, width_factor=base_width[0], height_factor=base_height[0], plotly_not_matplotlib=True)
    # Keep ticks (only last row) when plotting separately
    fig1.update_xaxes(dict(showticklabels=False, showgrid=False), col=1, overwrite=True, matches='x')
    fig1.update_xaxes(dict(showticklabels=True, showgrid=False), title='Seconds',
                      col=1, row=2 if ethogram_on_top else 1, overwrite=True, matches='x')
    # Remove ticks
    fig1.update_yaxes(dict(showticklabels=False, showgrid=True), title="Neurons",
                      col=1, row=1, overwrite=True)
    fig1.update_yaxes(dict(showticklabels=False, showgrid=False), title="",
                      col=1, row=2, overwrite=True)
    fig1.update_coloraxes(cmin=-0.25, cmax=0.75, colorbar=dict(
        # thickness=10,
        title=dict(text=r'ΔR / R₅₀', **font_dict)
        # title=dict(text=r'$\frac{\Delta R}{R_{50}}$', **font_dict)
    ))
    # fig1.update_traces(colorbar=dict(thickness=5))

    # Build figure 2: Ethogram with PCA modes
    num_ethogram_rows = 4
    if include_speed_subplot:
        num_ethogram_rows += 1
    if ethogram_on_top:
        num_ethogram_rows -= 1
    subplot_titles = [""]*num_ethogram_rows
    fig2 = make_subplots(rows=num_ethogram_rows, cols=1, shared_xaxes=True, shared_yaxes=False,
                         subplot_titles=subplot_titles, vertical_spacing=0.0)

    # Add to top or bottom
    if ethogram_on_top:
        _fig = fig1
        ethogram_row = 2
        ethogram_trace_offset = 1
    else:
        _fig = fig2
        ethogram_row = 1
        ethogram_trace_offset = 2
    for opt in ethogram_opt:
        _fig.add_shape(**opt, row=ethogram_row, col=1)
    # All on second
    for i, (trace, trace_opt) in enumerate(zip(trace_list, trace_opt_list)):
        if not include_speed_subplot and i >= num_pca_modes_to_plot:
            break
        trace_opt.pop('row')
        fig2.add_trace(trace, **trace_opt, row=i+ethogram_trace_offset)
        num_before_adding_shapes = len(fig2.layout.shapes)
        for shade_opt in trace_shading_opt:
            shade_opt['y1'] = 1.0
            fig2.add_shape(**shade_opt)
        # Force yref in all of these new shapes, which doesn't really work for subplots
        # But here it is hardcoded as 50% of the overall plot (extending across subplots)
        for _i in range(num_before_adding_shapes, len(fig2.layout.shapes)):
            fig2.layout.shapes[_i]['yref'] = 'paper'
    fig2.update_layout(showlegend=False, autosize=False)#, **plotly_opt)
    apply_figure_settings(fig2, width_factor=base_width[1], height_factor=base_height[1], plotly_not_matplotlib=True)
    # Remove ticks
    fig2.update_xaxes(dict(showticklabels=False, showgrid=False), col=1, overwrite=True, matches='x')
    fig2.update_yaxes(dict(showticklabels=False, showgrid=False), col=1, overwrite=True)

    fig2.update_xaxes(dict(showticklabels=False), row=1, overwrite=True)

    # Turn ticks on for speed, but decrease number of ticks because the size is too small
    fig2.update_yaxes(dict(showticklabels=True, showgrid=True, griddash='dash', gridcolor='black'),
                      range=[-0.22, 0.22], tickmode='array', tickvals=[-0.2, 0, 0.2],
                      row=5, overwrite=True)

    fig2.update_xaxes(dict(showticklabels=True, title='Time (seconds)'), row=num_ethogram_rows, col=1, overwrite=True, )
    # Remove black lines for all but bottom subplot
    for i in range(1, num_ethogram_rows):
        fig2.update_xaxes(showline=False, row=i, overwrite=True)

    # Add a horizontal "divider" line to separate speed
    # fig2.add_hline(y=0.2, line=dict(color='black', width=2), row=5, col=1)
    fig2.update_xaxes(showline=True, linewidth=2, linecolor='black', row=4, overwrite=True)

    # Move the titles down
    # fig2.update_annotations(yshift=-19, xshift=-40)

    # Use the y axis as titles... but make them smaller
    # fig2.update_yaxes(dict(title=dict(text='PCA <br> modes', font=dict(size=12))), row=3, overwrite=True)
    fig2.update_yaxes(dict(title=dict(text='.<br>1', font=dict(size=12))), row=2, col=1, overwrite=True)  # Note the extra '.'... otherwise the <br> doens't render for some reason
    fig2.update_yaxes(dict(title=dict(text='PCA modes<br>2', font=dict(size=12))), row=3, col=1, overwrite=True)
    fig2.update_yaxes(dict(title=dict(text='.<br>3', font=dict(size=12))), row=4, col=1, overwrite=True)
    fig2.update_yaxes(dict(title=dict(text='Speed <br> (mm/s)', font=dict(size=12))), row=5, overwrite=True)

    # For now, don't build the rest of the figures
    if to_show:
        fig1.show()
        fig2.show()
    if to_save:
        _save_plotly_all_types(fig1, project_data, fname='summary_only_heatmap_plot.html', output_folder=output_folder)
        _save_plotly_all_types(fig2, project_data, fname='summary_only_pca_plot.html', output_folder=output_folder)

    return fig1, fig2


def _save_plotly_all_types(fig, project_data, fname='summary_trace_plot.html', output_folder=None):
    trace_cfg = project_data.project_config.get_traces_config()
    # Save in the actual project
    fname = trace_cfg.resolve_relative_path(fname, prepend_subfolder=True)
    fig.write_html(str(fname))
    fname = Path(fname).with_suffix('.png')
    fig.write_image(str(fname))
    fname = Path(fname).with_suffix('.svg')
    fig.write_image(str(fname))
    # Save in a local folder
    if output_folder is not None:
        fname = Path(fname).with_suffix('.svg')
        fname = os.path.join(output_folder, fname.name)
        fig.write_image(str(fname), scale=1)
        fname = Path(fname).with_suffix('.png')
        fname = os.path.join(output_folder, fname.name)
        fig.write_image(str(fname), scale=4)
    # eps isn't working:
    # ValueError: Transform failed with error code 256: PDF to EPS conversion failed
    # fname = Path(fname).with_suffix('.eps')
    # fig.write_image(str(fname))


def make_summary_interactive_kymograph_with_behavior(project_cfg, to_save=True, to_show=False, keep_reversal_turns=False,
                                                     crop_x_axis=True, row_heights=None, x_range=None,
                                                     apply_figure_size_settings=True, discrete_behaviors=False,
                                                     showlegend=True, eigenworm_behaviors=False,
                                                     **kwargs):
    """
    Similar to make_summary_interactive_heatmap_with_pca, but with a kymograph instead of the neural traces

    In the end, only includes behavioral variables, not neural traces

    Parameters
    ----------
    project_cfg
    to_save
    to_show

    Returns
    -------

    """
    project_data = ProjectData.load_final_project_data_from_config(project_cfg)
    if x_range is None:
        fps = project_data.physical_unit_conversion.frames_per_second
        x_range = [25000/fps, 29000/fps]
    project_data.use_physical_time = True
    behavior_alias_dict = _get_behavior_dict(discrete_behaviors, eigenworm_behaviors)

    num_modes_to_plot = len(behavior_alias_dict)
    behavior_kwargs = dict(fluorescence_fps=False, reset_index=False)
    behavior_kwargs.update(kwargs['behavior_kwargs']) if 'behavior_kwargs' in kwargs else {}
    kwargs['behavior_kwargs'] =behavior_kwargs
    additional_shaded_states = []#[BehaviorCodes.SLOWING, BehaviorCodes.HEAD_CAST]
    column_widths, ethogram_opt, heatmap, heatmap_opt, kymograph, kymograph_opt, phase_plot_list, phase_plot_list_opt, _row_heights, subplot_titles, trace_list, trace_opt_list, trace_shading_opt, var_explained_line, var_explained_line_opt, weights_list, weights_opt_list = build_all_plot_variables_for_summary_plot(
        project_data, num_modes_to_plot, use_behavior_traces=True, behavior_alias_dict=behavior_alias_dict,
        additional_shaded_states=additional_shaded_states, showlegend=showlegend, **kwargs)

    # One column with a heatmap, (short) ethogram, and kymograph
    rows = 1 + num_modes_to_plot + 1
    if not discrete_behaviors and not eigenworm_behaviors:
        # Will add speed manually
        rows += 1
    cols = 1
    if row_heights is None:
        row_heights = _row_heights[:rows]
    else:
        row_heights = row_heights

    # Build figure
    ## Kymograph and ethogram (large image subplots)
    subplot_titles = ['', '']
    subplot_titles.extend(list(behavior_alias_dict.keys()))
    # subplot_titles.append('Speed')
    fig = make_subplots(rows=rows, cols=cols, shared_xaxes=False, shared_yaxes=False,
                        row_heights=row_heights, vertical_spacing=0.02,
                        #subplot_titles=subplot_titles
                        )
    fig.update_layout(
        width=800,
        height=600
    )

    for opt in ethogram_opt:
        fig.add_shape(**opt, row=2, col=1)
    kymograph_opt['row'] = 1
    fig.add_trace(kymograph, **kymograph_opt)

    ## Add behavior traces
    # I am adding multiple traces to a single plot, which the original function wasn't designed for
    # So I have to manually check the size of behavior_alias_dict and add the traces correctly
    i_num_traces_used = 0
    # Add dummy variable at the end for speed, which is always calculated and should be last
    if not discrete_behaviors and not eigenworm_behaviors:
        behavior_alias_dict['speed'] = ['speed']
    for k, v in behavior_alias_dict.items():
        if not isinstance(v, list):
            v = [v]
        # Loop over each trace
        for _ in range(len(v)):
            # Use the global variable to index in the simple lists
            trace, trace_opt = trace_list[i_num_traces_used], trace_opt_list[i_num_traces_used]
            i_num_traces_used += 1

            fig.add_trace(trace, **trace_opt)
            num_before_adding_shapes = len(fig.layout.shapes)
            for shade_opt in trace_shading_opt:
                shade_opt['y1'] = 1-row_heights[0]  # Default is half the overall plot
                fig.add_shape(**shade_opt, row=trace_opt['row'], col=trace_opt['col'])
            # Force yref in all of these new shapes, which doesn't really work for subplots
            # But here it is hardcoded as 50% of the overall plot (extending across subplots)
            for i in range(num_before_adding_shapes, len(fig.layout.shapes)):
                fig.layout.shapes[i]['yref'] = 'paper'

    # for trace, trace_opt in zip(trace_list, trace_opt_list):
    #     fig.add_trace(trace, **trace_opt)
    #     num_before_adding_shapes = len(fig.layout.shapes)
    #     for shade_opt in trace_shading_opt:
    #         shade_opt['y1'] = 0.5  # Will be half the overall plot
    #         fig.add_shape(**shade_opt, row=trace_opt['row'], col=trace_opt['col'])
    #     # Force yref in all of these new shapes, which doesn't really work for subplots
    #     # But here it is hardcoded as 50% of the overall plot (extending across subplots)
    #     for i in range(num_before_adding_shapes, len(fig.layout.shapes)):
    #         fig.layout.shapes[i]['yref'] = 'paper'

    ### Final updates
    fig.update_xaxes(dict(showticklabels=False, showgrid=False), col=1, overwrite=True, matches='x')
    if crop_x_axis:
        fig.update_xaxes(dict(range=x_range), row=1, col=1, overwrite=True)
    fig.update_yaxes(dict(showticklabels=False, showgrid=False), col=1, overwrite=True)
    fig.update_xaxes(dict(showticklabels=True, title=project_data.x_label_for_plots),
                     row=5, col=1, overwrite=True,)

    # Flip the kymograph
    fig.update_yaxes(dict(autorange='reversed'), col=1, row=1, overwrite=True)
    # Note: specific to the paper figure
    fig.update_yaxes(dict(showticklabels=True, showgrid=False, title='Body<br>Segment'), col=1, row=1)
    if discrete_behaviors:
        fig.update_yaxes(dict(showticklabels=False, showgrid=False), col=1, row=1)
        fig.update_yaxes(dict(showticklabels=False, showgrid=True, title='Turn<br>Annotations'), col=1, row=3)
        fig.update_yaxes(dict(showticklabels=False, showgrid=True, title='Other<br>Annotations'), col=1, row=4)
        fig.update_yaxes(dict(showticklabels=False, showgrid=True, title='Backwards<br>Annotation'), col=1, row=5)
    elif eigenworm_behaviors:
        fig.update_yaxes(dict(showticklabels=False, showgrid=False, title='Body Segment'), col=1, row=1)
        fig.update_yaxes(dict(showticklabels=False, showgrid=True, title='Eigenworms'), col=1, row=3)
        fig.update_yaxes(dict(showticklabels=False, showgrid=True, title='Eigenworms'), col=1, row=4)
        fig.update_yaxes(dict(showticklabels=False, showgrid=True, title='Backwards'), col=1, row=5)
    else:
        fig.update_yaxes(dict(showticklabels=True, showgrid=True, title='Head<br>Curvature'), col=1, row=3)
        fig.update_yaxes(dict(showticklabels=True, showgrid=True, title='Body<br>Curvature'), col=1, row=4)
        fig.update_yaxes(dict(showticklabels=True, showgrid=True, title='Velocity<br>(mm/s)', range=[-0.25, 0.15]), col=1, row=5)
    # Move the subplot titles down
    # fig.update_annotations(yshift=-7)

    if not discrete_behaviors:
        fig.update_layout(showlegend=False, overwrite=True)
    if apply_figure_size_settings:
        if showlegend:
            width_factor = 0.35
        else:
            width_factor = 0.3
        apply_figure_settings(fig, width_factor=width_factor, height_factor=0.45, plotly_not_matplotlib=True)

    # Add zero line to the speed plot
    if not discrete_behaviors and not eigenworm_behaviors:
        fig.update_yaxes(dict(showticklabels=True, showgrid=True, griddash='dash', gridcolor='black'),
                         range=[-0.22, 0.14],
                         tickmode='array', tickvals=[-0.22, 0],
                         row=5, overwrite=True)

    # Get the colormaps and legends in the right places, and not overlapping
    fig.update_layout(
        coloraxis2=dict(colorscale='RdBu',
                        colorbar=dict(
                            len=0.5,
                            yanchor='middle',
                            y=0.75,
                            xanchor='left',
                            x=1.01,
                            title=dict(text=r'Curvature (1/mm)', font=dict(size=14))
                        )),
    )
    fig.update_layout(
        showlegend=True,
        legend=dict(
          yanchor="middle",
          y=0.25,
          xanchor="left",
          x=1.01
        )
    )
    if not showlegend:
        fig.update_coloraxes(showscale=False)

    # Add a single legend for behavior colors
    # ... this doesn't work here because the shapes don't and can't have legends...
    # fig.update_layout(
    #     legend=dict(
    #         itemsizing='constant',  # Display legend items as colored boxes and text
    #         x=0.63,  # Adjust the x position of the legend
    #         y=0.54,  # Adjust the y position of the legend
    #         bgcolor='rgba(0, 0, 0, 0.00)',  # Set the background color of the legend
    #         bordercolor='Black',  # Set the border color of the legend
    #         borderwidth=1,  # Set the border width of the legend
    #         font=dict(size=base_font_size)  # Set the font size of the legend text
    #     )
    # )

    if to_show:
        fig.show()

    if to_save:
        # Change fname depending on whether we're keeping reversal turns
        if keep_reversal_turns:
            fname = 'summary_behavior_plot_kymograph_with_reversal_turns.html'
        else:
            fname = 'summary_behavior_plot_kymograph.html'
        _save_plotly_all_types(fig, project_data, fname=fname)

    return fig


def _get_behavior_dict(discrete_behaviors, eigenworm_behaviors):
    if discrete_behaviors:
        behavior_alias_dict = {'Turns': ['dorsal_turn', 'ventral_turn'],
                               'Other': ['self_collision', 'head_cast'],
                               'Rev': ['rev']}
    elif eigenworm_behaviors:
        behavior_alias_dict = {'Eigenworms1': ['eigenworm_0', 'eigenworm_1'],
                               'Eigenworms2': ['eigenworm_2', 'eigenworm_3'],
                               'Rev': ['rev']}
    else:
        behavior_alias_dict = {'Head curvature': ['dorsal_only_head_curvature', 'ventral_only_head_curvature'],
                               'Body curvature': ['ventral_only_body_curvature', 'dorsal_only_body_curvature']}
    return behavior_alias_dict


def make_summary_interactive_heatmap_with_kymograph(project_cfg, to_save=True, to_show=False, **kwargs):
    """
    Similar to make_summary_interactive_heatmap_with_pca, but with a kymograph instead of PCA modes
    The total effect is to remove all but the first column

    Parameters
    ----------
    project_cfg
    to_save
    to_show

    Returns
    -------

    """

    project_data = ProjectData.load_final_project_data_from_config(project_cfg)
    num_pca_modes_to_plot = 3
    kwargs['behavior_kwargs'] = dict(fluorescence_fps=False, reset_index=False)
    column_widths, ethogram_opt, heatmap, heatmap_opt, kymograph, kymograph_opt, phase_plot_list, phase_plot_list_opt, row_heights, subplot_titles, trace_list, trace_opt_list, trace_shading_opt, var_explained_line, var_explained_line_opt, weights_list, weights_opt_list = build_all_plot_variables_for_summary_plot(
        project_data, num_pca_modes_to_plot, **kwargs)

    # One column with a heatmap, (short) ethogram, and kymograph
    rows = 3
    cols = 1

    row_heights = row_heights[:2]
    row_heights.append(row_heights[0])

    # Build figure

    ### Column: x axis is time
    subplot_titles = ['Traces sorted by PC1', 'Ethogram', 'Kymograph']
    fig = make_subplots(rows=rows, cols=cols, shared_xaxes=False, shared_yaxes=False,
                        row_heights=row_heights, vertical_spacing=0.05,
                        subplot_titles=subplot_titles)

    fig.add_trace(heatmap, **heatmap_opt)
    for opt in ethogram_opt:
        fig.add_shape(**opt, row=2, col=1)
    fig.add_trace(kymograph, **kymograph_opt)

    ### Final updates
    fig.update_xaxes(dict(showticklabels=False, showgrid=False), col=1, overwrite=True, matches='x')
    fig.update_yaxes(dict(showticklabels=False, showgrid=False), col=1, overwrite=True)

    fig.update_layout(showlegend=False, autosize=False, width=1.5*1000, height=1.5*800)

    # Flip the kymograph in the y direction, so that the head is on top
    fig.update_yaxes(autorange="reversed", col=1, row=3, overwrite=True)

    # Fonts
    fig.update_layout(font=dict(size=18))
    # Get the colormaps in the right places
    fig.update_layout(
        coloraxis1=dict(colorscale='jet',  colorbar=dict(
            len=0.5,
            yanchor='middle',
            y=0.75,
            xanchor='right',
            x=1.1)),
        coloraxis2=dict(colorscale='RdBu', colorbar=dict(
            len=0.5,
            yanchor='middle',
            y=0.25,
            xanchor='right',
            x=1.1
        ),),
    )

    if to_show:
        fig.show()

    if to_save:
        # Change fname depending on whether we're keeping reversal turns
        if keep_reversal_turns:
            fname = 'summary_trace_plot_kymograph_with_reversal_turns.html'
        else:
            fname = 'summary_trace_plot_kymograph.html'
        _save_plotly_all_types(fig, project_data, fname=fname)

    return fig


def make_full_summary_interactive_plot(project_cfg, to_save=True, to_show=False, keep_reversal_turns=False,
                                                     crop_x_axis=True, row_heights=None, x_range=None,
                                                     apply_figure_size_settings=True,
                                                     showlegend=True,
                                                     **kwargs):
    """
    Similar to make_summary_interactive_heatmap_with_pca, but with all relevant information:
    1. Heatmap
    2. Ethogram
    3. Kymograph
    4. Head curvature
    5. Body curvature
    6. Turn annotation
    7. Other annotations
    8. Eigenworm 1/2
    9. Eigenworm 3/4
    10. Speed

    On the right, there are smaller visualizations:
    1. PCA plot
    2. Trajectory

    Things that are not plotted:
    1. PCA mode time series

    Parameters
    ----------
    project_cfg
    to_save
    to_show

    Returns
    -------

    """
    project_data = ProjectData.load_final_project_data_from_config(project_cfg)
    if x_range is None:
        fps = project_data.physical_unit_conversion.frames_per_second
        x_range = [25000/fps, 29000/fps]
    project_data.use_physical_time = True

    kwargs['behavior_kwargs'] = dict(fluorescence_fps=False, reset_index=False)
    trace_kwargs = dict(use_paper_traces=True)
    additional_shaded_states = []
    if row_heights is None:
        row_heights = [0.25, 0.05, 0.1]
        row_heights.extend([0.05]*7)
        row_heights.extend([0.3])
    col_widths = [0.388, 0.388, 0.15]  # Make the bottom row two columns actually square
    num_modes_to_plot = 2
    # Use the same function as all individual plots, but loop to get all the variables

    # Helper function to get everything
    def _get_all_variables(**kwargs):
        return build_all_plot_variables_for_summary_plot(project_data, num_modes_to_plot,
                                                         additional_shaded_states=additional_shaded_states,
                                                         showlegend=showlegend, **kwargs)

    # Behavior, kymograph, and trace heatmap
    # Also has the 3d pca phase plot
    behavior_alias_dict = {'Head curv.': ['dorsal_only_head_curvature', 'ventral_only_head_curvature'],
                           'Body curv.': ['ventral_only_body_curvature', 'dorsal_only_body_curvature'],
                           'Other annot.': ['self_collision', 'head_cast'],
                           'Eigen- worms 1/2': ['eigenworm_0', 'eigenworm_1'],
                           'Eigen- worms 3/4': ['eigenworm_2', 'eigenworm_3'],
                           'Speed': ['speed']}
    opt = dict(use_behavior_traces=True, behavior_alias_dict=behavior_alias_dict)
    (_, ethogram_opt, heatmap, heatmap_opt, kymograph, kymograph_opt, phase_plot_list, phase_plot_list_opt, _, _,
     trace_list, trace_opt_list, trace_shading_opt, _, _, _, _) = _get_all_variables(**opt)

    for i, _opt in enumerate(trace_opt_list):
        _opt['row'] += 1  # Start at the right row

    # Right side: trajectory
    try:
        trajectory_plot_list = _make_trajectory_plot(project_data, )
    except NoBehaviorAnnotationsError:
        trajectory_plot_list = []

    # One column with a heatmap, (short) ethogram, and kymograph
    rows = len(row_heights)
    cols = len(col_widths)

    # Build figure
    specs = [[{"colspan": 3}, None, None]]*(len(row_heights)-1)
    specs.append([{}, {}, {}])
    fig = make_subplots(rows=rows, cols=cols, shared_xaxes=False, shared_yaxes=False,
                        row_heights=row_heights, column_widths=col_widths,
                        vertical_spacing=0.0, horizontal_spacing=0.075,
                        specs=specs)

    ## 1: Heatmap
    fig.add_trace(heatmap, **heatmap_opt)

    ## 2. Ethogram
    for opt in ethogram_opt:
        fig.add_shape(**opt, row=2, col=1)

    ## 3. Kymograph
    if kymograph is not None:
        fig.add_trace(kymograph, **kymograph_opt)

    ## 4-10. Behavior traces
    # I am adding multiple traces to a single plot, which the original function wasn't designed for
    # So I have to manually check the size of behavior_alias_dict and add the traces correctly
    i_num_traces_used = 0
    legend_entries_to_remove = ['Head curv.',
                                'Body curv.', 'Speed'
                                ]
    for y_axis_title, trace_names in behavior_alias_dict.items():
        if len(trace_list) == 0:
            break
        if not isinstance(trace_names, list):
            trace_names = [trace_names]
        # Loop over each trace
        for trace_name in trace_names:
            # Use the global variable to index in the simple lists
            trace, trace_opt = trace_list[i_num_traces_used], trace_opt_list[i_num_traces_used]
            if y_axis_title in legend_entries_to_remove:
                trace['showlegend'] = False  # That is the plotly object, but it acts like a dict
            i_num_traces_used += 1

            fig.add_trace(trace, **trace_opt)
            # Add the yaxis label
            fig.update_yaxes(dict(showticklabels=False,  #'annotations' not in y_axis_title,
                                  title=dict(text=y_axis_title.replace(' ', '<br>'),
                                             font=dict(size=12), )),
                             **trace_opt)
            # Shrink traces
            # fig.update_traces(, **trace_opt)

            num_before_adding_shapes = len(fig.layout.shapes)
            for shade_opt in trace_shading_opt:
                shade_opt['y1'] = 1-row_heights[0]  # Default is half the overall plot
                shade_opt['y0'] = row_heights[-1]  # Do not shade the last row
                fig.add_shape(**shade_opt, row=trace_opt['row'], col=trace_opt['col'])
            # Force yref in all of these new shapes, which doesn't really work for subplots
            # But here it is hardcoded as 50% of the overall plot (extending across subplots)
            for i in range(num_before_adding_shapes, len(fig.layout.shapes)):
                fig.layout.shapes[i]['yref'] = 'paper'

    ## Below everything (pca phase plot and trajectory)
    phase_plot_opt = dict(row=len(row_heights), col=1)
    for trace in phase_plot_list:
        fig.add_trace(trace, **phase_plot_opt)

    trajectory_plot_list_opt = dict(row=len(row_heights), col=2)
    for trace in trajectory_plot_list:
        trace['showlegend'] = False
        fig.add_trace(trace, **trajectory_plot_list_opt)
    fig.update_xaxes(overwrite=True, **trajectory_plot_list_opt, title=dict(text='Distance (mm)', font=dict(size=14)))
    fig.update_yaxes(overwrite=True, **trajectory_plot_list_opt, title=dict(text='Distance (mm)', font=dict(size=14)),
                     scaleanchor='x12', scaleratio=1  # Also make the axes square
                     )

    ### Final updates
    fig.update_xaxes(dict(showticklabels=False, showgrid=False),
                     col=1, overwrite=True, matches='x')
    fig.update_xaxes(dict(showgrid=False), row=len(row_heights), col=1, overwrite=True,
                     matches=None)
    if crop_x_axis:
        fig.update_xaxes(dict(range=x_range), row=1, col=1, overwrite=True)
    # for i_row in range(2, len(row_heights)):
    #     fig.update_yaxes(dict(showticklabels=True), row=i_row, col=1, overwrite=True)
    fig.update_xaxes(dict(showticklabels=True, title=project_data.x_label_for_plots),
                     row=len(row_heights)-2, col=1, overwrite=True,)

    # fig.update_annotations(#yshift=-19,
    #                        xshift=-40)

    # Update top yaxes
    fig.update_yaxes(dict(showticklabels=False, showgrid=False, ),#title='Ethogram'),
                     row=2, col=1, overwrite=True)
    fig.update_yaxes(dict(showticklabels=True, showgrid=False, ),#title='Body<br>Segment'),
                     row=3, col=1, overwrite=True)
    fig.update_yaxes(dict(autorange='reversed'), row=3, col=1, overwrite=True)
    fig.update_yaxes(dict(showticklabels=False, showgrid=False,),
                     **heatmap_opt, overwrite=True)

    apply_figure_settings(fig, width_factor=1.0, height_factor=1.0, plotly_not_matplotlib=True)

    # Update bottom rows
    var_explained = 100 * phase_plot_list_opt['pca_obj'].explained_variance_ratio_
    phase_plot_opt['overwrite'] = True
    fig.update_xaxes(dict(showticklabels=True), **phase_plot_opt, title=dict(text=f'PC1 ({var_explained[0]:2.1f}%)', font=dict(size=14)))
    fig.update_yaxes(dict(showticklabels=True), **phase_plot_opt, title=dict(text=f'PC2 ({var_explained[1]:2.1f}%)', font=dict(size=14)))

    fig.update_yaxes(dict(showticklabels=True, showgrid=True), **trajectory_plot_list_opt)

    # Add zero line to the speed plot
    fig.update_yaxes(dict(showticklabels=True, showgrid=True, griddash='dash', gridcolor='black'),
                     range=[-0.16, 0.16],
                     tickmode='array', tickvals=[-0.1, 0, 0.1],
                     row=len(row_heights)-2, overwrite=True)
    # fig.update_xaxes(dict(showticklabels=True), row=len(row_heights)-2, overwrite=True)
    # Get the colormaps and legends in the right places, and not overlapping
    # Kymograph
    fig.update_layout(
        coloraxis2=dict(colorscale='RdBu',
                        colorbar=dict(
                            len=0.2,
                            yanchor='bottom',
                            y=0.61,
                            xanchor='left',
                            x=-0.13,
                            title=dict(text=r'Body<br>Segment<br>Curvature<br>(1/mm)', font=dict(size=14))
                        )),
    )
    # Traces
    fig.update_layout(
        coloraxis=dict(colorscale='jet',
                        colorbar=dict(
                            len=0.2,
                            yanchor='bottom',
                            y=0.8,
                            xanchor='left',
                            x=-0.13,
                            title=dict(text=r'Neuronal<br>Activity<br>dR/R50', font=dict(size=14))
                        ), cmin=-0.5, cmax=1.5),
    )
    fig.update_layout(
        showlegend=True,
        legend=dict(
          yanchor="bottom",
          y=-0.02,
          xanchor="left",
          x=sum(col_widths[:-1]) + 0.02
        )
    )
    if not showlegend:
        fig.update_coloraxes(showscale=False)

    if to_show:
        fig.show()

    if to_save:
        fname = 'summary_plot_with_everything.html'
        _save_plotly_all_types(fig, project_data, fname=fname)

    return fig


def build_all_plot_variables_for_summary_plot(project_data, num_pca_modes_to_plot=3, keep_reversal_turns=False,
                                              use_manual_annotations=False, use_behavior_traces=False,
                                              behavior_alias_dict=None, behavior_kwargs=None, showlegend=True,
                                              additional_shaded_states=None, trace_opt: dict = None, DEBUG=False):
    if behavior_kwargs is None:
        behavior_kwargs = dict(fluorescence_fps=True, reset_index=False)
    if behavior_alias_dict is None:
        behavior_alias_dict = {}
    default_trace_opt = dict(use_paper_options=True)
    if trace_opt is not None:
        default_trace_opt.update(trace_opt)
    df_traces = project_data.calc_default_traces(**default_trace_opt)
    x_for_plots_volumes = project_data.x_for_plots
    # x_for_plots_behavior = project_data.worm_posture_class.x_physical_time_frames

    df_traces_no_nan = fill_nan_in_dataframe(df_traces.copy(), do_filtering=True)
    # Calculate pca modes, and use them to sort
    pca_weights = PCA(n_components=10)
    pca_modes = PCA(n_components=10)
    df_mean_subtracted = df_traces_no_nan - df_traces_no_nan.mean()
    pca_weights.fit(df_mean_subtracted)
    pca_modes.fit(df_mean_subtracted.T)
    # Preprocess
    df_tmp = df_traces
    # df_tmp -= df_tmp.min()
    ind_sort = np.argsort(pca_weights.components_[0, :])
    dat = df_tmp.T.iloc[ind_sort, :]
    neuron_names = list(dat.index)
    df_pca_modes = pd.DataFrame(pca_modes.components_[0:num_pca_modes_to_plot, :].T)
    col_names = [f'mode {i}' for i in range(num_pca_modes_to_plot)]
    df_pca_modes.columns = col_names
    df_pca_modes.set_index(x_for_plots_volumes, inplace=True)

    has_behavior = True
    try:
        if not behavior_kwargs['fluorescence_fps']:
            _opt = dict(strong_smoothing_before_derivative=True)
        else:
            _opt = dict()
        speed = project_data.worm_posture_class.worm_speed(**behavior_kwargs, **_opt, use_stage_position=False,
                                                           signed=True)
    except NoBehaviorAnnotationsError:
        speed = pd.Series(np.zeros(df_pca_modes.shape[0]))
        has_behavior = False
    # TODO: move the reindexing to the worm posture class itself
    speed = pd.DataFrame(speed)
    try:
        speed.set_index(x_for_plots_volumes, inplace=True)
    except ValueError:
        # Then we are working in behavioral space, and the x axis should be set properly in the worm_posture class
        pass

    df_pca_weights = pd.DataFrame(pca_weights.components_[0:num_pca_modes_to_plot, :].T)
    col_names = [f'mode {i}' for i in range(num_pca_modes_to_plot)]
    df_pca_weights.columns = col_names
    df_pca_weights.index = neuron_names
    df_pca_weights = df_pca_weights.iloc[ind_sort, :].reset_index(drop=True)
    var_explained = pca_modes.explained_variance_ratio_[:7]
    # Initialize options for all subplots
    subplot_titles = ['Traces sorted by PC1', '', 'PCA weights', '', '', 'Phase plot',
                      'PCA modes', '', '', 'Middle Body Speed', 'Variance Explained']
    # Relies on num_pca_modes_to_plot being 3
    row_heights = [0.55, 0.05, 0.1, 0.1, 0.1, 0.1]
    column_widths = [0.7, 0.1, 0.1, 0.1]

    ### Main heatmap
    heatmap = go.Heatmap(y=dat.index, z=dat, x=x_for_plots_volumes,
                         zmin=-0.25, zmax=1.25, colorscale='jet', xaxis="x", yaxis="y",
                         coloraxis='coloraxis1')
    heatmap_opt = dict(row=1, col=1)

    ### Alternate: Kymograph heatmap
    try:
        kymo_dat = project_data.worm_posture_class.curvature(**behavior_kwargs).T
        # Instead of zmin and zmax on the plot, actually modify the data (options seem to not propagate to the plot)
        kymo_thresh = 0.04 / project_data.physical_unit_conversion.zimmer_behavior_um_per_pixel_xy
        kymo_dat[kymo_dat < -kymo_thresh] = -kymo_thresh
        kymo_dat[kymo_dat > kymo_thresh] = kymo_thresh
        # Convert to 1/mm
        kymo_dat = kymo_dat.iloc[3:-3, :] * 1000
        kymograph = go.Heatmap(x=kymo_dat.columns, y=kymo_dat.index, z=kymo_dat,
                               colorscale='RdBu', xaxis="x", yaxis="y", coloraxis='coloraxis2')
        kymograph_opt = dict(row=3, col=1)
    except NoBehaviorAnnotationsError:
        kymograph = None
        kymograph_opt = dict()

    ### Individual traces modes
    mode_colormap = plotly_paper_color_discrete_map()
    beh_colormap = BehaviorCodes.ethogram_cmap(include_custom=True)
    trace_list = []
    trace_opt_list = []
    if not use_behavior_traces:
        for i, col in enumerate(col_names):
            # Traces are pca modes
            trace_list.append(go.Scatter(y=df_pca_modes[col], x=df_pca_modes.index,
                                         line=dict(color=mode_colormap[i+1], width=2), showlegend=False))
            trace_opt_list.append(dict(row=i + 3, col=1, secondary_y=False))

    else:
        try:
            for i, (name_key, name_list) in enumerate(behavior_alias_dict.items()):
                # They may be a list of behaviors
                if not isinstance(name_list, list):
                    name_list = [name_list]
                if 'Eigen' in name_key:
                    legendgroup = 'Eigenworms'
                elif 'Rev' in name_key:
                    legendgroup = 'Backwards'
                else:
                    legendgroup = name_key

                for single_name in name_list:
                    y = project_data.worm_posture_class.calc_behavior_from_alias(single_name, **behavior_kwargs)
                    # Do not control the line colors here, because we want different ones on one plot
                    # Actually: convert to main behavior-related colors
                    # But first, need to convert to the relevant behavior code (this is a longer string)
                    if 'ventral' in single_name:
                        code = BehaviorCodes.VENTRAL_TURN
                        y *= 1000  # Move to mm instead of um
                    elif 'dorsal' in single_name:
                        code = BehaviorCodes.DORSAL_TURN
                        y *= 1000  # Move to mm instead of um
                    else:
                        code = BehaviorCodes.UNKNOWN

                    legend_name = behavior_name_mapping().get(single_name, single_name)
                    trace_list.append(go.Scatter(y=y, x=y.index,
                                                 name=legend_name, showlegend=showlegend,
                                                 legendgroup=legendgroup, legendgrouptitle=dict(text=legendgroup),
                                                 marker=dict(color=beh_colormap[code]),
                                                 line=dict(width=1)))

                    # Same options, but additional entries to match length of trace_list
                    trace_opt_list.append(dict(row=i + 3, col=1, secondary_y=False))
                    if DEBUG:
                        print(f'Adding trace for {single_name} with color {beh_colormap[code]}')
        except NoBehaviorAnnotationsError:
            pass
    #### Shading on top of the PCA modes
    try:
        beh_vec = project_data.worm_posture_class.beh_annotation(**behavior_kwargs, include_pause=True)
        beh_vec = pd.DataFrame(beh_vec)
        # Check lengths; sometimes beh_vec is one too short
        if len(beh_vec) == df_pca_modes.shape[0] - 1:
            beh_vec = pd.concat([beh_vec, pd.Series([BehaviorCodes.UNKNOWN])])
        try:
            beh_vec.set_index(x_for_plots_volumes, inplace=True)
        except ValueError:
            # Then we are working in behavioral space, and we don't need this
            pass
        trace_shading_opt = options_for_ethogram(beh_vec, shading=True)
    except NoBehaviorAnnotationsError:
        trace_shading_opt = dict()
    if has_behavior:
        ### Speed plot (below pca modes)
        trace_list.append(go.Scatter(y=speed.iloc[:, 0], x=speed.index, showlegend=False))
        trace_opt_list.append(dict(row=num_pca_modes_to_plot + 3, col=1, secondary_y=False))
    ### PCA weights (same names as pca modes)
    mode_colormap = px.colors.qualitative.Plotly
    weights_list = []
    weights_opt_list = []
    for i, col in enumerate(col_names):
        weights_list.append(go.Bar(x=df_pca_weights[col], y=df_pca_weights.index, orientation='h',
                                   marker=dict(color=mode_colormap[i]),
                                   hovertext=neuron_names,
                                   hoverinfo="text", showlegend=False))
        weights_opt_list.append(dict(row=1, col=2 + i, secondary_y=False))
    ### Ethogram
    # Include manual annotations, if any
    beh_vec = None
    if use_manual_annotations:
        try:
            beh_vec = project_data.worm_posture_class.manual_beh_annotation(**behavior_kwargs)
            logging.info('Using manual annotations')
        except NoBehaviorAnnotationsError:
            logging.warning('No manual annotations found')
            beh_vec = None
    if beh_vec is None:
        beh_vec = project_data.worm_posture_class.beh_annotation(**behavior_kwargs, include_pause=True)
    ethogram_cmap_opt = dict(include_reversal_turns=keep_reversal_turns, include_pause=True)
    if beh_vec is None:
        # If still none, that means there are no annotations (e.g. it is immobilized)
        beh_vec = pd.Series([BehaviorCodes.UNKNOWN for i in range(df_pca_modes.shape[0])])
        ethogram_opt = dict()
    else:
        beh_vec = pd.DataFrame(beh_vec)
        # Check lengths; sometimes beh_vec is one too short
        if len(beh_vec) == df_pca_modes.shape[0] - 1:
            beh_vec = pd.concat([beh_vec, pd.Series([BehaviorCodes.UNKNOWN])])
        try:
            beh_vec.set_index(x_for_plots_volumes, inplace=True)
        except ValueError:
            # Then we are working in behavioral space, and we don't need this
            pass
        # print(f'Unique state codes for ethogram: {[s.full_name for s in beh_vec.iloc[:, 0].unique()]}')
        ethogram_opt = options_for_ethogram(beh_vec, **ethogram_cmap_opt, include_turns=True,
                                            to_extend_short_states=True,
                                            additional_shaded_states=additional_shaded_states, DEBUG=False)
    ### 3d phase plot
    ethogram_cmap = BehaviorCodes.ethogram_cmap(**ethogram_cmap_opt)
    # Use the same behaviors as the ethogram
    try:
        df_pca_modes['behavior'] = list(beh_vec.iloc[:, 0])
        df_out, col_names = modify_dataframe_to_allow_gaps_for_plotly(df_pca_modes,
                                                                      ['mode 0', 'mode 1'],
                                                                      'behavior')

        state_codes = beh_vec.iloc[:, 0].unique()  # Get unique state codes; there is only one column
        phase_plot_list = []
        # print(f'Unique state codes: {[s.full_name for s in state_codes]}')
        for i, state_code in enumerate(state_codes):
            try:
                # Only show the legend if the behavior is FWD or REV
                showlegend = state_code.full_name in {'FWD', 'REV',
                                                      'VENTRAL_TURN and FWD', 'FWD and VENTRAL_TURN',
                                                      'DORSAL_TURN and FWD', 'FWD and DORSAL_TURN',
                                                      'PAUSE'}
                name = state_code.name
                if name is None:
                    # If there is a complex state
                    name = state_code.full_name.split(' and ')[0]
                name = behavior_name_mapping().get(name, name)
                # print(f'Adding phase plot for {name} with color {ethogram_cmap[state_code]}')
                phase_plot_list.append(
                    go.Scatter(x=df_out[col_names[0][i]], y=df_out[col_names[1][i]],
                               mode='lines',
                               name=name, line=dict(color=ethogram_cmap[state_code], width=4),
                               showlegend=showlegend, legendgroup='Ethogram', legendgrouptitle=dict(text='Ethogram')),)
            except KeyError as e:
                # print(f'KeyError: {e} on behavior {state_code.full_name}')
                pass

    except ValueError as e:
        # Then we are working in behavioral space, and we don't need a phase plot
        print(f'ValueError: {e}; if only the behavior is being plotted, this is not a problem')
        phase_plot_list = []

    phase_plot_list_opt = dict(row=3, col=2, pca_obj=pca_weights)
    ### Variance explained
    var_explained_line = go.Scatter(x=np.arange(1, len(var_explained)+1), y=var_explained, showlegend=False)
    var_explained_line_opt = dict(row=6, col=2, secondary_y=False)

    return column_widths, ethogram_opt, heatmap, heatmap_opt, kymograph, kymograph_opt, phase_plot_list, \
        phase_plot_list_opt, row_heights, subplot_titles, trace_list, trace_opt_list, trace_shading_opt, \
        var_explained_line, var_explained_line_opt, weights_list, weights_opt_list


def make_summary_hilbert_triggered_average_grid_plot(project_cfg, i_body_segment=41,
                                                     return_fast_scale_separation=False, residual_mode=None,
                                                     to_save=True,
                                                     **kwargs):
    """
    Make a grid plot of the hilbert-phase triggered average for a given body segment

    Neurons that oscillate should show a strong signal, and others will be flat.
    This signal should be present regardless of the segment chosen

    kwargs are passed to calc_default_traces

    Parameters
    ----------
    project_cfg
    i_body_segment
    kwargs

    Returns
    -------

    """
    project_data = ProjectData.load_final_project_data_from_config(project_cfg)

    # Get traces
    trace_opt = dict(interpolate_nan=False,
                     filter_mode='rolling_mean',
                     min_nonnan=0.9,
                     nan_tracking_failure_points=True,
                     nan_using_ppca_manifold=False,
                     channel_mode='dr_over_r_20',
                     rename_neurons_using_manual_ids=True,
                     residual_mode=residual_mode,
                     return_fast_scale_separation=return_fast_scale_separation)
    trace_opt.update(kwargs)

    df_traces = project_data.calc_default_traces(**trace_opt)

    phase = project_data.worm_posture_class.hilbert_phase(fluorescence_fps=True, reset_index=True)
    phase = phase.loc[:, i_body_segment].copy() % (2 * math.pi)
    phase -= np.nanmean(phase)

    trigger_opt = dict(min_duration=0, min_lines=10, ind_preceding=10, behavioral_annotation=phase,
                       behavioral_annotation_is_continuous=True, )
    min_significant = 10
    ind_class_fast = project_data.worm_posture_class.calc_triggered_average_indices(**trigger_opt)
    triggered_averages_class = FullDatasetTriggeredAverages(df_traces, ind_class_fast,
                                                            min_points_for_significance=min_significant)

    # Options for the traces within the grid plot
    trace_and_plot_opt = dict(color_using_behavior=False, share_y_axis=False)
    subset_neurons = project_data.well_tracked_neuron_names(0.9, rename_neurons_using_manual_ids=trace_opt['rename_neurons_using_manual_ids'])

    func = partial(triggered_averages_class.ax_plot_func_for_grid_plot,
                   show_individual_lines=False, color_significant_times=True)
    fig = make_grid_plot_from_dataframe(df_traces, neuron_names_to_plot=subset_neurons, **trace_and_plot_opt,
                                        ax_plot_func=func)

    # Make a title for the plot based on the options, and save in the project
    if to_save:
        fname = f'hilbert_triggered_average_grid_plot-fast_{return_fast_scale_separation}-residual_{residual_mode}.png'
        project_data.save_fig_in_project(fname, overwrite=True)

    return fig


def plot_raw_global_residual(neuron_name):
    """
    Plots the ratio, global, and residual component for a given neuron

    Parameters
    ----------
    neuron_name

    Returns
    -------

    """

    pass


def _make_trajectory_plot(project_data, **kwargs):
    """
    Make a trajectory plot for the worm

    Parameters
    ----------
    project_data
    to_save
    to_show
    kwargs

    Returns
    -------

    """
    behavior_kwargs = dict(fluorescence_fps=False)
    # xy = project_data.worm_posture_class.stage_position(fluorescence_fps=True).copy()
    xy = project_data.worm_posture_class.calc_behavior_from_alias('worm_center_position', **behavior_kwargs).copy()
    xy = xy - xy.iloc[0, :]

    beh = project_data.worm_posture_class.beh_annotation(simplify_states=True,
                                                         include_head_cast=False, include_collision=False,
                                                         include_pause=False, **behavior_kwargs)

    df_xy = xy
    df_xy['Behavior'] = beh.values

    df_xy['size'] = 1
    ethogram_cmap = BehaviorCodes.ethogram_cmap(include_turns=True, include_reversal_turns=False,
                                                include_quiescence=True)
    df_out, col_names = modify_dataframe_to_allow_gaps_for_plotly(df_xy, ['X', 'Y'], 'Behavior')

    # Loop to prep each line, then plot
    state_codes = df_xy['Behavior'].unique()
    phase_plot_list = []
    for i, state_code in enumerate(state_codes):
        if state_code == BehaviorCodes.UNKNOWN:
            continue
        phase_plot_list.append(
            go.Scatter(x=df_out[col_names[0][i]], y=df_out[col_names[1][i]], mode='lines',
                       name=state_code.full_name, line=dict(color=ethogram_cmap.get(state_code, None), width=4)))

    # Add start and end
    phase_plot_list.append(go.Scatter(x=[0], y=[0], marker=dict(
        color='black', symbol='x',
        size=10
    ), name='start'))
    phase_plot_list.append(go.Scatter(x=[xy.iloc[-1, 0]], y=[xy.iloc[-1, 1]], marker=dict(
        color='black',
        size=10), name='end'
                             ))

    return phase_plot_list
