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

# functions
def make_scatterplot_pca():
    df = process_data_pca(raw_features, dr_data, raw_data, raw_neighbors, N_NEIGHBORS)
    fig = make_scatterplot(df)
    fig_data["pca"] = fig
    return fig

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# controls = dbc.Card(
#     [
#         html.Div(
#             [
#                 dbc.Label("X variable"),
#                 dcc.Dropdown(
#                     id="x-variable",
#                     options=[
#                         {"label": col, "value": col} for col in iris.columns
#                     ],
#                     value="sepal length (cm)",
#                 ),
#             ]
#         ),
#         html.Div(
#             [
#                 dbc.Label("Y variable"),
#                 dcc.Dropdown(
#                     id="y-variable",
#                     options=[
#                         {"label": col, "value": col} for col in iris.columns
#                     ],
#                     value="sepal width (cm)",
#                 ),
#             ]
#         ),
#         html.Div(
#             [
#                 dbc.Label("Cluster count"),
#                 dbc.Input(id="cluster-count", type="number", value=3),
#             ]
#         ),
#     ],
#     body=True,
# )

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
                    id="highlight-choice",
                    options=[
                        {"label": col, "value": col} for col in highlight_methods
                    ],
                    value=highlight_methods[0],
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label("Correspondence"),
                html.Div("0.0", id="method-correspondence")
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
            id = "method-pca-graph",
            figure=make_scatterplot_pca(),
        ), md=5),
        dbc.Col([], md=5)
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
    Output('method-pca-graph', 'figure'),
    Input('method-pca-graph', 'clickData')
)
def update_selection_pca(clicked):
    if clicked:
        clicked = clicked["points"]
        clicked = clicked[0]['customdata'][0]
        selection = [i for i in raw_neighbors[clicked]]
        selection.append(clicked)
    else:
        selection = []        
    fig = fig_data["pca"]
    fig.update_traces(selectedpoints=selection)
    return fig



# @app.callback(
#     Output("cluster-graph", "figure"),
#     [
#         Input("x-variable", "value"),
#         Input("y-variable", "value"),
#         Input("cluster-count", "value"),
#     ],
# )
# def make_graph(x, y, n_clusters):
#     # minimal input validation, make sure there's at least one cluster
#     km = KMeans(n_clusters=max(n_clusters, 1))
#     df = iris.loc[:, [x, y]]
#     km.fit(df.values)
#     df["cluster"] = km.labels_

#     centers = km.cluster_centers_

#     data = [
#         go.Scatter(
#             x=df.loc[df.cluster == c, x],
#             y=df.loc[df.cluster == c, y],
#             mode="markers",
#             marker={"size": 8},
#             name="Cluster {}".format(c),
#         )
#         for c in range(n_clusters)
#     ]

#     data.append(
#         go.Scatter(
#             x=centers[:, 0],
#             y=centers[:, 1],
#             mode="markers",
#             marker={"color": "#000", "size": 12, "symbol": "diamond"},
#             name="Cluster centers",
#         )
#     )

#     layout = {"xaxis": {"title": x}, "yaxis": {"title": y}}

#     return go.Figure(data=data, layout=layout)




# # make sure that x and y values can't be the same variable
# def filter_options(v):
#     """Disable option v"""
#     return [
#         {"label": col, "value": col, "disabled": col == v}
#         for col in iris.columns
#     ]


# # functionality is the same for both dropdowns, so we reuse filter_options
# app.callback(Output("x-variable", "options"), [Input("y-variable", "value")])(
#     filter_options
# )
# app.callback(Output("y-variable", "options"), [Input("x-variable", "value")])(
#     filter_options
# )


if __name__ == "__main__":
    app.run_server(debug=True, port=8050)