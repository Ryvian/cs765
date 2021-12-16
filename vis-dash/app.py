import dash
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from dash import Input, Output, dcc, html
from scipy.spatial import KDTree
from lib import *

N_NEIGHBORS = 15

raw_data = pd.read_csv("./static/mnist_test.csv")
raw_data = raw_data[:1000]
t = list(raw_data.columns)
t.remove("label")
raw_features = raw_data[t]
raw_tree = KDTree(raw_features.values)
raw_neighbors = find_neighbors(raw_features.values, raw_tree, N_NEIGHBORS)
dr_data = {}
fig_data = {}

dr_methods = ["PCA", "TSNE", "UMAP"]
highlight_methods = ["neighbors", "same label"]

process_data_pca(raw_features, dr_data, raw_data, raw_neighbors, N_NEIGHBORS)
process_data_tsne(raw_features, dr_data, raw_data, raw_neighbors, N_NEIGHBORS)
process_data_umap(raw_features, dr_data, raw_data, raw_neighbors, N_NEIGHBORS)
# functions
def make_scatterplot_pca():
    df = dr_data['pca']
    fig = make_scatterplot(df)
    fig_data["pca"] = fig
    return fig
def make_scatterplot_tsne():
    df = dr_data['tsne']
    fig = make_scatterplot(df)
    fig_data["tsne"] = fig
    return fig
def make_scatterplot_umap():
    df = dr_data['umap']
    fig = make_scatterplot(df)
    fig_data["umap"] = fig
    return fig

def make_barplot(method: str):
    df = dr_data[method]
    return make_barplot_aux(df)

def make_boxplot(method: str):
    df = dr_data[method]
    return make_boxplot_aux(df)

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

corr = calculate_correspondence(raw_features.values, dr_data['pca'][["x", "y"]].values, N_NEIGHBORS)
# import ipdb; ipdb.set_trace()
corr_tsne = calculate_correspondence(raw_features.values, dr_data['tsne'][["x", "y"]].values, N_NEIGHBORS)
corr_umap = calculate_correspondence(raw_features.values, dr_data['umap'][["x", "y"]].values, N_NEIGHBORS)
order = np.argsort([-corr, -corr_tsne, -corr_umap])

order_dict = {0:"PCA",1:"TSNE",2:"UMAP"}
order_dict_lower = {0:"pca",1:"tsne",2:"umap"}
order_trust={0:corr, 1:corr_tsne, 2:corr_umap}
function_dict = {0:make_scatterplot_pca(),1:make_scatterplot_tsne(),2:make_scatterplot_umap()}

method = dbc.Card(
    [
        html.Div(
            [
                dbc.Label("Method:"),
                html.Div(order_dict[order[0]]),
            ], 
        ),
        html.Div( [
                dbc.Label("highlight:"),
                dcc.Dropdown(
                    id=order_dict_lower[order[0]]+"-highlight-choice",
                    options=[
                        {"label": col, "value": col} for col in highlight_methods
                    ],
                    value=highlight_methods[0],
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label("Trustworthiness"),
                html.Div(f"{order_trust[order[0]]:.3f}", id=order_dict_lower[order[0]]+"-correspondence")
            ]
        ),
    ],
    body=True,
)

method1 = dbc.Card(
    [
        html.Div(
            [
                dbc.Label("Method:"),
                html.Div(order_dict[order[1]]),
            ],
        ),
        html.Div( [
                dbc.Label("highlight:"),
                dcc.Dropdown(
                    id=order_dict_lower[order[1]]+"-highlight-choice",
                    options=[
                        {"label": col, "value": col} for col in highlight_methods
                    ],
                    value=highlight_methods[0],
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label("Trustworthiness"),
                html.Div(f"{order_trust[order[1]]:.3f}", id=order_dict_lower[order[1]]+"-correspondence")
            ]
        ),
    ],
    body=True,
)

method2 = dbc.Card(
    [
        html.Div(
            [
                dbc.Label("Method:"),
                html.Div(order_dict[order[2]]),
            ],
        ),
        html.Div( [
                dbc.Label("highlight:"),
                dcc.Dropdown(
                    id=order_dict_lower[order[2]]+"-highlight-choice",
                    options=[
                        {"label": col, "value": col} for col in highlight_methods
                    ],
                    value=highlight_methods[0],
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label("Trustworthiness"),
                html.Div(f"{order_trust[order[2]]:.3f}", id=order_dict_lower[order[2]]+"-correspondence")
            ]
        ),
    ],
    body=True,
)

rows = [
    dbc.Row(html.H1("&nbsp;")),
    dbc.Row([
        dbc.Col(html.H2("Method", className="text-center"), md=2),
        dbc.Col(html.H2("Scatter Plot", className="text-center"), md=5),
        dbc.Col(html.H2("Overlap", className="text-center"), md=5)
    ], align="center"),
    html.Hr(),
    dbc.Row([
        dbc.Col(method, md=2),
        dbc.Col(dcc.Graph(
            id = "method-"+order_dict_lower[order[0]]+"-scatterplot",
            figure=function_dict[order[0]],
        ), md=5),
        dbc.Col(dcc.Graph(
            # id="method-pca-bar",
            # figure=make_barplot('pca'),
            id="method-"+order_dict_lower[order[0]]+"-box",
            figure=make_boxplot(order_dict_lower[order[0]]),

        ), md=5),
    ], align="center"),
    html.Hr(),
    dbc.Row([
        dbc.Col(method1, md=2),
        dbc.Col(dcc.Graph(
            id = "method-"+order_dict_lower[order[1]]+"-scatterplot",
            figure=function_dict[order[1]],
        ), md=5),
        dbc.Col(dcc.Graph(
            # id="method-pca-bar",
            # figure=make_barplot('pca'),
            id="method-"+order_dict_lower[order[1]]+"-box",
            figure=make_boxplot(order_dict_lower[order[1]]),

        ), md=5),
    ], align="center"),
    html.Hr(),

    dbc.Row([
        dbc.Col(method2, md=2),
        dbc.Col(dcc.Graph(
            id = "method-"+order_dict_lower[order[2]]+"-scatterplot",
            figure=function_dict[order[2]],
        ), md=5),
        dbc.Col(dcc.Graph(
            # id="method-pca-bar",
            # figure=make_barplot('pca'),
            id="method-"+order_dict_lower[order[2]]+"-box",
            figure=make_boxplot(order_dict_lower[order[2]]),

        ), md=5),
    ], align="center")
]


navbar = dbc.NavbarSimple(
    brand="Dimensional Reduction Visualization", 
    class_name="navbar navbar-expand-md navbar-dark fixed-top bg-dark",
    fluid=True, 
    dark=True,
    children=[
    dbc.NavItem("mnist.csv", class_name="navbar-text"),
    dbc.NavItem("&nbsp;"),
    dbc.Button("Upload", class_name="btn btn-primary")
])

page_layout = dbc.Container(
    [
        navbar,
        *rows
    ],
    fluid=True,
)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@app.callback(
    Output('method-pca-scatterplot', 'figure'),
    Input('method-pca-scatterplot', 'clickData'),
    Input('pca-highlight-choice', 'value')
)
def update_selection_pca(clicked, highlight):
    if clicked:
        clicked = clicked["points"][0]['customdata']
        clicked_index = clicked[0]
        clicked_label = clicked[1]
        if highlight == "neighbors":
            selection = [i for i in raw_neighbors[clicked_index]]
            selection.append(clicked)
        else: # same label
            selection = list(raw_data[raw_data['label'] == clicked_label].index)
    else:
        selection = []    
  
    fig = fig_data["pca"]
    fig.update_traces(selectedpoints=selection)
    return fig


@app.callback(
    Output('method-umap-scatterplot', 'figure'),
    Input('method-umap-scatterplot', 'clickData'),
    Input('umap-highlight-choice', 'value')
)
def update_selection_umap(clicked, highlight):
    if clicked:
        clicked = clicked["points"][0]['customdata']
        clicked_index = clicked[0]
        clicked_label = clicked[1]
        if highlight == "neighbors":
            selection = [i for i in raw_neighbors[clicked_index]]
            selection.append(clicked)
        else:  # same label
            selection = list(raw_data[raw_data['label'] == clicked_label].index)
    else:
        selection = []

    fig = fig_data["umap"]
    fig.update_traces(selectedpoints=selection)
    return fig


@app.callback(
    Output('method-tsne-scatterplot', 'figure'),
    Input('method-tsne-scatterplot', 'clickData'),
    Input('tsne-highlight-choice', 'value')
)
def update_selection_tsne(clicked, highlight):
    if clicked:
        clicked = clicked["points"][0]['customdata']
        clicked_index = clicked[0]
        clicked_label = clicked[1]
        if highlight == "neighbors":
            selection = [i for i in raw_neighbors[clicked_index]]
            selection.append(clicked)
        else:  # same label
            selection = list(raw_data[raw_data['label'] == clicked_label].index)
    else:
        selection = []

    fig = fig_data["tsne"]
    fig.update_traces(selectedpoints=selection)
    return fig

@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/cs765':
        return page_layout
    else:
        return '404'


if __name__ == "__main__":
    app.run_server(debug=True, port=8050)