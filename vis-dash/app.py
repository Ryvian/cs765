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
raw_data = pd.read_csv("~/Downloads/mnist_test.csv")
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
import numpy as np
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
        dbc.Col(html.H2("Method"), md=2),
        dbc.Col(html.H2("Scatter Plot"), md=5),
        dbc.Col(html.H2("Overlap"), md=5)
    ], align="center"),
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