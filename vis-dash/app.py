import dash
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from dash import Input, Output, dcc, html
# from sklearn import datasets
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
from scipy.spatial import KDTree
from copy import copy

from lib import *

N_NEIGHBORS = 15

# iris_raw = datasets.load_iris()
# iris = pd.DataFrame(iris_raw["data"], columns=iris_raw["feature_names"])
raw_data = pd.read_csv("./static/mnist_test.csv")
raw_data = raw_data[:1000]
t = list(raw_data.columns)
t.remove("label")
raw_features = raw_data[t]
raw_tree = KDTree(raw_features.values)
raw_neighbors = find_neighbors(raw_features.values, raw_tree, N_NEIGHBORS)
dr_data = {}
fig_data = {}

dr_methods = ["PCA", "T-SNE"]
highlight_methods = ["neighbors", "same label"]

process_data_pca(raw_features, dr_data, raw_data, raw_neighbors, N_NEIGHBORS)

# functions
def make_scatterplot_pca():
    df = dr_data['pca']
    fig = make_scatterplot(df)
    fig_data["pca"] = fig
    return fig

def make_barplot(method: str):
    df = dr_data[method]
    return make_barplot_aux(df)

def make_boxplot(method: str):
    df = dr_data[method]
    return make_boxplot_aux(df)

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])



corr = calculate_correspondence(raw_features.values, dr_data['pca'][["x", "y"]].values, N_NEIGHBORS)
method = dbc.Card(
    [
        html.Div(
            [
                dbc.Label("Method:"),
                html.Div("PCA"),
            ], 
        ),
        html.Div( [
                dbc.Label("highlight:"),
                dcc.Dropdown(
                    id="pca-highlight-choice",
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
                html.Div(f"{corr:.3f}", id="pca-correspondence")
            ]
        ),
    ],
    body=True,
)


rows = [
    dbc.Row(html.H1("&nbsp;")),
    dbc.Row([
        dbc.Col(html.H2("Method"), md=2),
        dbc.Col(html.H2("Scatter Plot"), md=5),
        dbc.Col(html.H2("Overlap"), md=5)
    ], align="center"),
    dbc.Row([
        dbc.Col(method, md=2),
        dbc.Col(dcc.Graph(
            id = "method-pca-scatterplot",
            figure=make_scatterplot_pca(),
        ), md=5),
        dbc.Col(dcc.Graph(
            # id="method-pca-bar",
            # figure=make_barplot('pca'),
            id="method-pca-box",
            figure=make_boxplot('pca'),

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

app.layout = dbc.Container(
    [
        # html.H1("Iris k-means clustering"),
        # html.Hr(),
        navbar,
        # dbc.Row(
        #     [
        #         dbc.Col(controls, md=4),
        #         dbc.Col(dcc.Graph(id="cluster-graph"), md=8),
        #     ],
        #     align="center",
        # ),
        *rows
    ],
    fluid=True,
)

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



if __name__ == "__main__":
    app.run_server(debug=True, port=8050)