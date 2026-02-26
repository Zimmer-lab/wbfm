from typing import List

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def plotly_boxplot_colored_boxes(df, color_list):
    """
    Plotly can't color individual boxes using plotly express, so they have to be made one by one

    See https://towardsdatascience.com/applying-a-custom-colormap-with-plotly-boxplots-5d3acf59e193

    Parameters
    ----------
    df
    color_list

    Returns
    -------

    """

    fig = go.Figure()

    columns = df.columns

    for i, (column, color) in enumerate(zip(columns, color_list)):
        fig.add_trace(go.Box(name=column, y=df[column], marker_color=color))

    fig.update_layout(showlegend=False)
    # Make one grid line at 0, and make it black
    fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black")

    return fig


def add_trendline_annotation(fig, x_offset=0, y_offset=0):
    """
    Given a scatter plot with a trendline added, add an annotation with the slope, R² and p-value of the trendline

    Parameters
    ----------
    fig
    df

    Returns
    -------

    """
    # Extract trendline results
    results = px.get_trendline_results(fig)
    trendline = results.iloc[0].px_fit_results
    slope = trendline.params[1]
    r2 = trendline.rsquared
    pvalue = trendline.pvalues[1]

    # Get a reasonable position (top right) for the annotation using the x and y max data points
    x = np.nanmin(fig.data[0].x) + x_offset
    y = np.nanmax(fig.data[0].y) + y_offset

    # Add the annotation
    annotation_text = f'Slope: {slope:.2f}<br>R²: {r2:.2f}<br>p-value: {pvalue:.2e}'
    fig.add_annotation(
        x=x, y=y,
        text=annotation_text,
        showarrow=False,
        bordercolor='black',
        borderwidth=1,
        borderpad=4,
        bgcolor='white',
        opacity=0.8
    )

    return fig


def plotly_plot_mean_and_shading(df, x, y, color=None, line_name='Mean', shade_style='std',
                                 add_individual_lines=False,
                                 cmap=None, x_intersection_annotation=None, annotation_kwargs=None,
                                 annotation_position=None, fig=None, is_second_plot=False, **kwargs):
    """
    Plot the mean of a y column for each x value, and shade the standard deviation

    Note that this requires identical x values for each group

    Format expected for the dataframe (with dummy column names):
    x   y   color
    0   1   A
    0   2   A
    0   3   A

    Parameters
    ----------
    df
    x
    y
    color - if not None, will plot a separate line and shading for each unique value in this column
    line_name - name for the mean line in the legend
    shade_style - 'std' to shade one standard deviation, 'quantile' to shade the interquartile range
    add_individual_lines - whether to add lines for each individual group
    cmap - optional dictionary mapping line_name to color for the mean line and shading
    x_intersection_annotation - if not None, will add vertical and horizontal lines at the intersection of the mean line with the specified x value, and annotate with the y value at the intersection
    annotation_kwargs - additional keyword arguments to pass to the annotation (like font size)
    annotation_position - position for the annotation of the intersection point, options are 'top left', 'top right', 'bottom left', 'bottom right'
    fig - optional existing figure to add to, if None will create a new figure
    is_second_plot - whether this is the second plot in a figure, which affects whether the vertical line for the intersection annotation is plotted (to avoid plotting it twice in the same figure)

    Returns
    -------

    """
    if annotation_kwargs is None:
        annotation_kwargs = dict()
    if annotation_position is None:
        annotation_position = 'top left'

    if color is not None and len(df[color].unique()) > 1:
        # Assume we want to subset the dataframe by the color list
        fig = None
        default_position_list = ['top left', 'bottom right', 'top right', 'bottom left']
        for i, group in enumerate(df[color].unique()):
            _df = df[df[color] == group]
            # Alternate annotation_position by default
            annotation_position = default_position_list[i % len(default_position_list)]

            fig = plotly_plot_mean_and_shading(_df, x, y, color=color, line_name=group,
                                               add_individual_lines=False, cmap=cmap, fig=fig,
                                               x_intersection_annotation=x_intersection_annotation,
                                               annotation_position=annotation_position, is_second_plot=is_second_plot)
            is_second_plot = True
        return fig

    # Calculate mean and std dev for each x value
    grouped = df.groupby(x)
    mean_y = grouped[y].mean()
    if shade_style == 'std':
        shade_y = grouped[y].std()
        upper_y = mean_y + shade_y
        lower_y = mean_y - shade_y
    elif shade_style == 'quantile':
        upper_y = grouped[y].quantile(0.75)
        lower_y = grouped[y].quantile(0.25)
    else:
        raise ValueError(f"Unknown shade_style: {shade_style}; valid options are 'std' and 'quantile'")

    if fig is None:
        fig = go.Figure()

    if add_individual_lines:
        for group in df[color].unique():
            df_subset = df[df[color] == group]
            fig.add_trace(go.Scatter(
                x=df_subset[x], y=df_subset[y], mode='lines', name=f'Group {group}', line=dict(width=1), **kwargs
            ))

    # Add the mean line
    opt = dict()
    if cmap is not None:
        opt['line'] = dict(color=cmap[line_name])
    fig.add_trace(go.Scatter(
        x=mean_y.index, y=mean_y, mode='lines', name=str(line_name), **opt
    ))

    # Shade the standard deviation area
    opt = dict()
    if cmap is not None:
        if '#' in cmap[line_name]:
            opt['fillcolor'] = hex2rgba(cmap[line_name])
        else:
            opt['fillcolor'] = add_alpha_to_rgb(cmap[line_name])
    else:
        opt['fillcolor'] = 'rgba(0,100,80,0.2)'

    # Add two lines in a unique group that will have shading between them
    fill_opt = dict(hoverinfo="skip", showlegend=False,
                    line=dict(color='rgba(255,255,255,0)'),
                    **opt)

    # First one, which doesn't show up
    fig.add_trace(go.Scatter(x=mean_y.index, y=upper_y, **fill_opt))
    # Second one, which does show up
    fig.add_trace(go.Scatter(x=mean_y.index, y=lower_y, fill='tonexty', **fill_opt))

    if x_intersection_annotation is not None:
        y_value_at_x = mean_y.loc[x_intersection_annotation]
        # Add vertical line at x=x_intersection_annotation
        if not is_second_plot:
            fig.add_shape(type="line",
                          x0=x_intersection_annotation, y0=mean_y.min(), x1=x_intersection_annotation, y1=mean_y.max(),
                          line=dict(color="Black", width=1, dash="dash"),
                          )

        # Add horizontal line at the intersection with mean_y
        fig.add_shape(type="line",
                      x0=mean_y.index.min(), y0=y_value_at_x, x1=mean_y.index.max(), y1=y_value_at_x,
                      line=dict(color="Black", width=1, dash="dash"),
                      )

        # Add text annotation at the intersection point
        x = 0.5*x_intersection_annotation if 'left' in annotation_position else 1.5*x_intersection_annotation
        y = 1.1*y_value_at_x if 'top' in annotation_position else 0.9*y_value_at_x
        fig.add_annotation(
            x=x, y=y,
            text=f"y={y_value_at_x:.2f}",
            showarrow=False,
            bgcolor=opt['fillcolor'],
            **annotation_kwargs
            # showarrow=True,
            # arrowhead=2,
            # ax=-40, ay=-10  # Position the text relative to the point
        )

    return fig


def hex2rgba(hex_color, alpha=0.2, return_tuple=False):
    fillcolor = tuple(int(hex_color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
    if alpha is not None:
        fillcolor = fillcolor + (alpha,)
    if not return_tuple:
        fillcolor = f"rgba{fillcolor}"
    return fillcolor


def add_alpha_to_rgb(rgb_color, alpha=0.2):
    return hex2rgba(rgba2hex(rgb_color), alpha=alpha)


def rgba2hex(rgba_color):
    rgba_color = rgba_color.replace('rgba', '').replace('rgb', '').replace(' ', '').replace('(', '').replace(')', '')
    rgba_color = rgba_color.split(',')
    hex_color = '#'
    # Alpha is prepended
    if len(rgba_color) == 4:
        hex_color = f'{int(float(rgba_color[3])*255):02x}'
    for i in range(3):
        hex_color += f'{int(rgba_color[i]):02x}'
    return hex_color


def rgba2float(rgba_color):
    """Converts string of rgba color to list of floats, like matplotlib expects"""
    rgba_color = rgba_color.replace('rgba', '').replace('rgb', '').replace(' ', '').replace('(', '').replace(')', '')
    rgba_color = rgba_color.split(',')
    return [float(c) / 255 for c in rgba_color]


def pastelize_color(hex_color, mix_fraction=0.2, return_hex=True):
    """Make a color more pastel by mixing it with white."""
    rgb_color = hex2rgba(hex_color, alpha=None, return_tuple=True)
    mix_color = (255, 255, 255)
    mixed_rgb = mix_rgba(rgb_color, mix_color, mix_fraction, return_hex)
    return mixed_rgb


def darken_color(hex_color, mix_fraction=0.2, return_hex=True):
    """Make a color darker by mixing it with black."""
    rgb_color = hex2rgba(hex_color, alpha=None, return_tuple=True)
    mix_color = (0, 0, 0)
    mixed_rgb = mix_rgba(rgb_color, mix_color, mix_fraction, return_hex)
    return mixed_rgb


def mute_color(hex_color, mix_fraction=0.2, return_hex=True):
    """Make a color darker by mixing it with black."""
    rgb_color = hex2rgba(hex_color, alpha=None, return_tuple=True)
    mix_color = (128, 128, 128)
    mixed_rgb = mix_rgba(rgb_color, mix_color, mix_fraction, return_hex)
    return mixed_rgb


def mix_rgba(rgb_color, mix_color, mix_fraction, return_hex=True):
    mixed_rgb = tuple(
        int(np.clip((1 - mix_fraction) * c + mix_fraction * m, 0, 255)) for c, m in zip(rgb_color, mix_color)
    )
    mixed_rgb = f"rgba{mixed_rgb}"
    if return_hex:
        mixed_rgb = rgba2hex(mixed_rgb)
    return mixed_rgb


def float2rgba(float_color, alpha=0.2):
    # Convert list of float values to string rgba color
    if len(float_color) == 3:
        float_color = float_color + [alpha]
    fillcolor = f"rgba{tuple(int(255 * c) if i < 3 else c for i, c in enumerate(float_color))}"
    # fillcolor = fillcolor.replace(' ', '')
    return fillcolor


def get_nonoverlapping_text_positions(x, y, all_text, fig, weight=100, k=None, add_nodes_with_no_text=True,
                                      x_range=None, y_range=None, **kwargs):
    """

    Parameters
    ----------
    x
    y
    all_text
    fig
    weight - weight of the edge between the data and the text (attraction)
    k - optimal distance between nodes
    add_nodes_with_no_text
    x_range
    y_range
    kwargs

    Returns
    -------

    """
    import networkx as nx
    positions = np.array(list(zip(x, y)))
    G = nx.Graph()

    # Add nodes
    fixed_nodes = []
    text_index_mapping = dict()
    for i, (pos, text) in enumerate(zip(positions, all_text)):
        if len(text) == 0:
            # Skip empty text
            if not add_nodes_with_no_text:
                continue
        text_index_mapping[text] = i
        #     continue
        G.add_node(text, pos=pos)
        G.add_node(f"data_{i}", pos=pos)  # Will be fixed in place
        G.add_edge(f"data_{i}", text, weight=weight)  # Try to keep the text near the data
        fixed_nodes.append(f"data_{i}")

    # Add edges independent of distance
    # for i in range(len(positions)):
    #     G.add_edge(f"data_{i}", f"text_{i}", weight=weight)  # Try to keep the text near the data
        # for j in range(i + 1, len(positions)):
        #     dist = np.linalg.norm(positions[i] - positions[j])
        #     G.add_edge(f"text_{i}", f"text_{j}", weight=1.0 / (dist + 1e-4))
        # print(1.0 / (dist + 1e-4))

    # Apply force-directed layout
    new_positions = nx.spring_layout(G, pos=nx.get_node_attributes(G, 'pos'), fixed=fixed_nodes,
                                     weight='weight',  k=k,)

    # Update the plot with new text positions
    # adjusted_text_positions = np.array([new_positions[k] for k in new_positions.keys() if k in all_text])

    # Create new scatter plot with adjusted text positions
    if fig is None:
        adjusted_scatter = go.Scatter(
            x=x, y=y,
            mode='markers',
        )
        fig = go.Figure(data=[adjusted_scatter])

    # for i, (t, (x_new, y_new)) in enumerate(zip(text, adjusted_text_positions)):
    for t, i in text_index_mapping.items():
        # i is the index in the original data list
        if len(t) == 0:
            # Skip empty text
            continue
        _x, _y = x.iat[i], y.iat[i]
        x_new, y_new = new_positions[t]
        if x_range is not None:
            x_new = max(x_range[0], min(x_range[1], x_new))
        if y_range is not None:
            y_new = max(y_range[0], min(y_range[1], y_new))
        fig.add_annotation(x=_x, y=_y, ax=x_new, ay=y_new,  # arrowhead=2,
                           text=t, xref="x", yref="y", axref="x", ayref="y", font=dict(**kwargs))

    return fig


def combine_plotly_figures(all_figs, show_legends: List[bool] = None, force_yref_paper=True,
                           horizontal=True, DEBUG=False, **kwargs):
    """
    Combine multiple plotly figures into a single figure, all on one row

    Does not work if figures are already subplots

    Parameters
    ----------
    all_figs

    Returns
    -------

    """

    if horizontal:
        opt = dict(rows=1, cols=len(all_figs), shared_yaxes=True, horizontal_spacing=0.01)
    else:
        opt = dict(rows=len(all_figs), cols=1, shared_xaxes=True, vertical_spacing=0.01)

    fig = make_subplots(
        **opt, **kwargs
    )
    if DEBUG:
        print(f"Creating subplots with {len(all_figs)} columns")

    for old_fig, i_col in zip(all_figs, range(1, len(all_figs) + 1)):
        if horizontal:
            opt = dict(row=1, col=i_col)
        else:
            opt = dict(row=i_col, col=1)

        for trace in old_fig.data:
            if show_legends is not None:
                trace.showlegend = show_legends[i_col - 1]
            fig.add_trace(trace, **opt)
            if DEBUG:
                print(f"Adding trace to row 1, col {i_col}")
        for annotation in old_fig.layout.annotations:
            fig.add_annotation(annotation, **opt)
        for shape in old_fig.layout.shapes:
            fig.add_shape(shape, **opt)
        fig.update_xaxes(old_fig.layout.xaxis, **opt)
        fig.update_yaxes(old_fig.layout.yaxis, **opt)

    # Force the yref for shapes to be 'paper', which is turned off by default in subplots
    # https://community.plotly.com/t/drawing-vertical-line-on-histogram-in-subplot-but-yref-paper-is-not-working/31581/3
    if force_yref_paper:
        for shape in fig.layout.shapes:
            shape['yref'] = 'paper'

    return fig


def add_annotation_lines(df_idx_range, neuron_name, fig, is_immobilized=False, is_residual=False, DEBUG=False):
    """Based on a dataframe with start and end times for annotations, add bars to a plotly figure"""
    if df_idx_range is not None:
        # If there is a dynamic time window used for the ttest, then add a bar as an annotation
        this_idx = df_idx_range[df_idx_range['neuron'] == neuron_name]
        # Add a bar for the dynamic window for each type (mutant and not)
        from wbfm.utils.general.utils_paper import plotly_paper_color_discrete_map
        _cmap = plotly_paper_color_discrete_map()
        for i, row in this_idx.iterrows():
            y0 = 0.9
            if row['is_mutant']:
                color = _cmap['gcy-31;-35;-9']
                y0 = 0.95
            elif is_immobilized:
                color = _cmap['immob']
            elif is_residual:
                color = _cmap['residual']
            else:
                color = _cmap['Wild Type']
            if DEBUG:
                print(f"Adding bar for {neuron_name} with color {color}")
                print(f"At location {row['start']} to {row['end']}")
            fig.add_shape(type="rect", x0=row['start'], y0=y0, x1=row['end'], y1=y0,
                          line=dict(color=color, width=2), xref='x', yref='paper', layer='below')
    return fig


def colored_text(text, color, bold=False):
    """
    Figure should be updated by extracting original text, defining colors, and then updating the layout:

    ticktext = [colored_text(t, c) for t, c in text2colors.items()]
    fig.update_layout(
    yaxis=dict(tickmode='array', ticktext=ticktext, tickvals=ticks)
    )

    Parameters
    ----------
    text
    color

    Returns
    -------

    From: https://stackoverflow.com/questions/58183962/how-to-color-ticktext-in-plotly
    """
    if not bold:
        return f"<span style='color:{str(color)}'> {str(text)} </span>"
    else:
        return f"<span style='color:{str(color)}'> <b>{str(text)}</b> </span>"
