from typing import Dict

import dash
import pandas as pd
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go


def dashboard_from_two_dataframes(df_summary: pd.DataFrame, raw_dfs: Dict[str, pd.DataFrame], is_jupyter=False):
    """
    Create a dashboard from two dataframes. The first dataframe is used to create a line plot, and the second dataframe
    is used to create a scatter plot. The scatter plot has a clickData output, which is used to update the line plot.

    The index of the summary dataframe is the identifier of each point, and must correspond to a column in the original
    dataframe (raw_dfs). The summary dataframe must have at least two columns, which will be used to create the scatter plot.

    df_summary must have a column called 'index' which is the column name of each raw_dfs dataframe

    Parameters
    ----------
    raw_dfs
    df_summary

    Returns
    -------

    """
    if is_jupyter:
        from jupyter_dash import JupyterDash
        app = JupyterDash(__name__)
    else:
        app = dash.Dash(__name__)

    # Get the column names of the summary dataframe
    column_names = df_summary.columns
    column_names_with_none = ['None'] + list(column_names)

    keys = list(raw_dfs.keys())
    initial_clickData = {'points': [{'customdata': [raw_dfs[keys[0]].columns[0]]}]}

    # Create a dropdown menu to choose each column of the summary dataframe
    dropdown_menu_x = dcc.Dropdown(
        id="dropdown_x",
        options=[{"label": i, "value": i} for i in column_names],
        value=column_names[0],
        clearable=False
    )
    dropdown_menu_y = dcc.Dropdown(
        id="dropdown_y",
        options=[{"label": i, "value": i} for i in column_names],
        value=column_names[1],
        clearable=False
    )
    dropdown_menu_color = dcc.Dropdown(
        id="dropdown_color",
        options=[{"label": i, "value": i} for i in column_names_with_none],
        value=None,
        clearable=False
    )

    dropdown_style = {'display': 'inline-block', 'width': '33%'}
    row_style = {'display': 'inline-block', 'width': '100%'}
    # Create the layout
    app.layout = html.Div([
        # Create two rows, with a label on top of a dropdown menu
        html.Div([
            html.Div([
                html.Label("X axis (scatterplot)"),
            ], style=dropdown_style),
            html.Div([
                html.Label("Y axis (scatterplot)"),
            ], style=dropdown_style),
            html.Div([
                html.Label("Color splitting (scatterplot)"),
            ], style=dropdown_style),
        ]),
        html.Div([
            html.Div([
                dropdown_menu_x
            ], style=dropdown_style),
            html.Div([
                dropdown_menu_y
            ], style=dropdown_style),
            html.Div([
                dropdown_menu_color
            ], style=dropdown_style),
        ]),

        html.Div([
            dcc.Graph(id="scatter", clickData=initial_clickData),
        ], style=row_style),
        html.Div([
            dcc.Graph(id="line")
        ], style=row_style)
    ])

    # Create a callback for clicking on the scatterplot. The callback should update the line plot
    @app.callback(
        Output("line", "figure"),
        Input("scatter", "clickData")
    )
    def update_line(clickData):
        # Do not update if the clickData is None
        if clickData is None:
            return dash.no_update
        click_name = clickData["points"][0]["customdata"][0]

        fig = line_plot_from_dict_of_dataframes(raw_dfs, y=click_name)
        
        return fig

    # Create a callback for updating the scatterplot using both dropdown menus
    # The scatter plot must have custom data, which is the name of the stock
    @app.callback(
        Output("scatter", "figure"),
        Input("dropdown_x", "value"),
        Input("dropdown_y", "value"),
        Input("dropdown_color", "value"),
        Input("scatter", "clickData")
    )
    def update_scatter(x, y, color, clickData):
        selected_row = clickData["points"][0]["customdata"][0]
        fig = _build_scatter_plot(df_summary, x, y, selected_row, color=color)
        return fig

    return app


def _build_scatter_plot(df_summary, x, y, selected_row, **kwargs):
    df_summary['selected'] = 1
    df_summary.loc[selected_row, 'selected'] = 5
    fig = px.scatter(df_summary, x=x, y=y, hover_data=["index"], custom_data=["index"],
                     size='selected',
                     title="Click on a point to update the line plot",
                     marginal_y="violin",
                     **kwargs)
    fig.update_layout(font=dict(size=18))
    return fig


def line_plot_from_dict_of_dataframes(dict_of_dfs: dict, y: str):
    # Create a single figure from a list of dataframes
    fig = go.Figure()
    for k, df in dict_of_dfs.items():
        fig.add_scatter(y=df[y], name=k, )
    fig.update_layout(xaxis_title="Time", yaxis_title="Amplitude", font=dict(size=18))
    return fig


if __name__ == "__main__":
    # Create a test dataframe
    df = px.data.stocks()
    raw_dfs = {'test': df}

    # Create a dataframe with the average and std of each stock as columns
    df_std = df.std()
    df_avg = df.mean()
    df_summary = pd.DataFrame({"std": df_std, "avg": df_avg})

    # Create a new column with the name of the stock
    df_summary["stock"] = df_summary.index

    app = dashboard_from_two_dataframes(df_summary, raw_dfs)
    app.run_server(debug=True)
