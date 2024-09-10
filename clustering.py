import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
import plotly.express as px

def cluster_genres(genre_data, n_clusters=10):
    """
    Cluster genre data using KMeans.

    Args:
    genre_data (pd.DataFrame): DataFrame containing genre features
    n_clusters (int): Number of clusters for KMeans

    Returns:
    pd.DataFrame: genre_data with added 'cluster' column
    """
    try:
        cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=n_clusters))])
        X = genre_data.select_dtypes(np.number)
        cluster_pipeline.fit(X)
        genre_data['cluster'] = cluster_pipeline.predict(X)
        logging.info(f"Successfully clustered genres into {n_clusters} clusters.")
        return genre_data
    except Exception as e:
        logging.error(f"Error in clustering genres: {str(e)}")
        return None

def visualize_genre_clusters(genre_data):
    """
    Visualize genre clusters using t-SNE.

    Args:
    genre_data (pd.DataFrame): DataFrame containing genre features and cluster labels

    Returns:
    plotly.graph_objs._figure.Figure: Scatter plot of genre clusters
    """
    try:
        X = genre_data.select_dtypes(np.number)
        tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, verbose=1))])
        genre_embedding = tsne_pipeline.fit_transform(X)
        projection = pd.DataFrame(columns=['x', 'y'], data=genre_embedding)
        projection['genres'] = genre_data['genres']
        projection['cluster'] = genre_data['cluster']

        fig = px.scatter(
            projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'genres'],
            title='Genre Clusters Visualization'
        )
        logging.info("Successfully created genre clusters visualization.")
        return fig
    except Exception as e:
        logging.error(f"Error in visualizing genre clusters: {str(e)}")
        return None

def create_song_clusters(data, n_clusters=20):
    """
    Create song clusters using KMeans algorithm.

    Args:
    data (pd.DataFrame): The dataset containing song features.
    n_clusters (int): Number of clusters to create.

    Returns:
    pd.DataFrame: The input dataframe with an additional 'cluster_label' column.
    """
    try:
        song_cluster_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('kmeans', KMeans(n_clusters=n_clusters, verbose=False))
        ], verbose=False)

        X = data.select_dtypes(np.number)
        number_cols = list(X.columns)
        song_cluster_pipeline.fit(X)
        song_cluster_labels = song_cluster_pipeline.predict(X)
        data['cluster_label'] = song_cluster_labels
        logging.info(f"Successfully created {n_clusters} song clusters.")
        return data
    except Exception as e:
        logging.error(f"Error in creating song clusters: {str(e)}")
        return None

def visualize_song_clusters(data):
    """
    Visualize song clusters using PCA.

    Args:
    data (pd.DataFrame): The dataset containing song features and cluster labels.

    Returns:
    plotly.graph_objs._figure.Figure: A scatter plot of song clusters.
    """
    try:
        from sklearn.decomposition import PCA

        X = data.select_dtypes(np.number)
        pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
        song_embedding = pca_pipeline.fit_transform(X)
        projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
        projection['title'] = data['name']
        projection['cluster'] = data['cluster_label']

        fig = px.scatter(
            projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'title'],
            title="Song Clusters Visualization"
        )
        logging.info("Successfully created song clusters visualization.")
        return fig
    except Exception as e:
        logging.error(f"Error in visualizing song clusters: {str(e)}")
        return None
