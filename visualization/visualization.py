import logging
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from yellowbrick.target import FeatureCorrelation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_decade(year):
    """Convert a year to its corresponding decade."""
    period_start = int(year/10) * 10
    return f'{period_start}s'

def visualize_decades(data):
    """Visualize the distribution of songs across decades."""
    try:
        data['decade'] = data['year'].apply(get_decade)
        plt.figure(figsize=(11, 6))
        sns.countplot(x='decade', data=data)
        plt.title('Distribution of Songs Across Decades')
        plt.xlabel('Decade')
        plt.ylabel('Number of Songs')
        plt.savefig('decades_distribution.png')
        plt.close()
        logging.info("Decades distribution visualization saved as 'decades_distribution.png'")
    except Exception as e:
        logging.error(f"Error in visualize_decades: {str(e)}")

def visualize_sound_features(year_data):
    """Visualize the evolution of sound features over the years."""
    try:
        sound_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'valence']
        fig = px.line(year_data, x='year', y=sound_features)
        fig.update_layout(title='Evolution of Sound Features Over Time',
                          xaxis_title='Year',
                          yaxis_title='Feature Value')
        fig.write_image('sound_features_evolution.png')
        logging.info("Sound features evolution visualization saved as 'sound_features_evolution.png'")
    except Exception as e:
        logging.error(f"Error in visualize_sound_features: {str(e)}")

def visualize_top_genres(genre_data, n=10):
    """
    Visualize the top N genres based on popularity.

    Args:
    genre_data (pd.DataFrame): DataFrame containing genre data
    n (int): Number of top genres to visualize (default: 10)

    Returns:
    plotly.graph_objs._figure.Figure: The generated bar plot
    """
    try:
        top_n_genres = genre_data.nlargest(n, 'popularity')
        fig = px.bar(top_n_genres, x='genres', y=['valence', 'energy', 'danceability', 'acousticness'],
                     barmode='group', title=f'Top {n} Genres by Popularity')
        fig.write_image('top_genres.png')
        logging.info("Top genres visualization saved as 'top_genres.png'")
        return fig
    except Exception as e:
        logging.error(f"Error in visualizing top genres: {str(e)}")
        return None

def visualize_feature_correlation(data):
    """
    Visualize feature correlation with the target 'popularity'.
    """
    try:
        logging.info("Visualizing feature correlation")
        feature_names = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                         'liveness', 'loudness', 'speechiness', 'tempo', 'valence',
                         'duration_ms', 'explicit', 'key', 'mode', 'year']

        X, y = data[feature_names], data['popularity']
        features = np.array(feature_names)

        visualizer = FeatureCorrelation(labels=features)

        plt.figure(figsize=(20, 20))
        visualizer.fit(X, y)
        visualizer.show()
        plt.savefig('feature_correlation.png')
        plt.close()
        logging.info("Feature correlation visualization saved as 'feature_correlation.png'")
    except KeyError as e:
        logging.error(f"Key error in feature correlation visualization: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error in feature correlation visualization: {str(e)}")
