import dash
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from dash import Input, Output, dcc, html
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import trustworthiness
import umap
from scipy.spatial import KDTree

def find_neighbors(data: np.ndarray, tree: KDTree, n):
    neighbors = {}
    for i in range(data.shape[0]):
        _, inds = tree.query(data[i, :], k=n+1)
        inds = set(inds)
        inds.remove(i)
        neighbors[i] = inds
    return neighbors

def find_overlaps(neighbors1, neighbors2):
    overlaps = {}
    for i in neighbors1.keys():
        overlap = len(neighbors1[i].intersection(neighbors2[i]))
        overlaps[i] = overlap
    return overlaps

def process_data(dr: np.ndarray, method: str, processed_data: pd.DataFrame, raw_data: pd.DataFrame, raw_neighbors, n):
    data = processed_data
    data[method] = pd.DataFrame()
    df = data["pca"]
    df["label"] = raw_data["label"]
    df["x"] = dr[:, 0]
    df["y"] = dr[:, 1]
    dr_tree = KDTree(dr)
    dr_neighbors = find_neighbors(dr, dr_tree, n)
    overlaps = find_overlaps(raw_neighbors, dr_neighbors)
    df["overlap"] = [overlaps[i] for i in df.index]
    df["index"] = df.index
    return df

def process_data_tsne1(dr: np.ndarray, method: str, processed_data: pd.DataFrame, raw_data: pd.DataFrame, raw_neighbors, n):
    data = processed_data
    data[method] = pd.DataFrame()
    df = data["tsne"]
    df["label"] = raw_data["label"]
    df["x"] = dr[:, 0]
    df["y"] = dr[:, 1]
    dr_tree = KDTree(dr)
    dr_neighbors = find_neighbors(dr, dr_tree, n)
    overlaps = find_overlaps(raw_neighbors, dr_neighbors)
    df["overlap"] = [overlaps[i] for i in df.index]
    df["index"] = df.index
    return df

def process_data_umap1(dr: np.ndarray, method: str, processed_data: pd.DataFrame, raw_data: pd.DataFrame, raw_neighbors, n):
    data = processed_data
    data[method] = pd.DataFrame()
    df = data["umap"]
    df["label"] = raw_data["label"]
    df["x"] = dr[:, 0]
    df["y"] = dr[:, 1]
    dr_tree = KDTree(dr)
    dr_neighbors = find_neighbors(dr, dr_tree, n)
    overlaps = find_overlaps(raw_neighbors, dr_neighbors)
    df["overlap"] = [overlaps[i] for i in df.index]
    df["index"] = df.index
    return df
def calculate_correspondence(orignal: np.ndarray, embedded: np.ndarray, n_neighbors: int):
    return trustworthiness(orignal, embedded, n_neighbors=n_neighbors)

def process_data_pca(raw_features: pd.DataFrame, processed_data: pd.DataFrame, raw_data: pd.DataFrame, raw_neighbors, n):
    pca = PCA(n_components=2)
    dr = pca.fit_transform(raw_features.to_numpy())
    return process_data(dr, "pca", processed_data, raw_data, raw_neighbors, n)

def process_data_tsne(raw_features: pd.DataFrame, processed_data: pd.DataFrame, raw_data: pd.DataFrame, raw_neighbors, n):
    tsne = TSNE(n_components=2)
    dr = tsne.fit_transform(raw_features.to_numpy())
    return process_data_tsne1(dr, "tsne", processed_data, raw_data, raw_neighbors, n)

def process_data_umap(raw_features: pd.DataFrame, processed_data: pd.DataFrame, raw_data: pd.DataFrame, raw_neighbors, n):
    umap1 = umap.UMAP()
    dr = umap1.fit_transform(raw_features.to_numpy())
    return process_data_umap1(dr, "umap", processed_data, raw_data, raw_neighbors, n)

def make_scatterplot(data: pd.DataFrame):
    fig = px.scatter(data, x="x", y="y", color="overlap", hover_name="label", color_continuous_scale='Bluered_r', hover_data={
            'x': True,
            'y': True,
            'overlap': True,
            'index': True
        },
        custom_data=['index', 'label']
        )
    return fig

def make_barplot_aux(data: pd.DataFrame):
    df = data[["label", "overlap"]]
    overlaps = df.groupby("label").mean()
    overlaps["Label"] = overlaps.index
    overlaps.rename(columns={"overlap": "Mean Overlap"}, inplace=True)
    fig = px.bar(overlaps, x='Label', y='Mean Overlap', )
    return fig

def make_boxplot_aux(data: pd.DataFrame):
    df = data[["label", "overlap"]]
    fig = px.box(df, x='label', y='overlap', )
    return fig