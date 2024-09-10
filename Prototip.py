#!/usr/bin/env python
# coding: utf-8

import os
import sys
import logging
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CHUNK_SIZE = 40960
DATA_SOURCE_MAPPING = 'spotify-dataset:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F1800580%2F2936818%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240505%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240505T010052Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D7a095318cf8957bfd4914ad66ca3cc13ddf6bc1134cf7c51390b61242cb24b66a2ed57ef750e7985bb85d9ed9f605a5beb55ae65f822c9e313095e749aaa7e918d628438b6688d56e5dfb1348ae56cffea9f624619eada59e74a67920b51ec08f6daf795e9d06257346ea5cf0be1b3f09106afdbd2a7add2d459df99e0e61b6917617b22d690d548475f625f1c8bc5b8e62a6965b1df59eec070eeaed6b20ed970c213a041584dc06a4ce3b2f25f470bdb63b6000f1b337efc162760ff4dc44fd0649313fb80397b35a34a64ce13ea1c7039f389e04d2b8c024f12adb6a2adbc218a7f8e63b30f11de6479b6f73a99b3f9d45a55b48bb6f73b5e0b6a9c024774'

KAGGLE_INPUT_PATH = './kaggle/input'
KAGGLE_WORKING_PATH = './kaggle/working'

# Initialize language model
MODEL_NAME = "distilgpt2"  # A smaller, faster model for demonstration
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
nlp = pipeline("text-generation", model=model, tokenizer=tokenizer)

def setup_directories():
    """Set up the necessary directories for the project."""
    shutil.rmtree(KAGGLE_INPUT_PATH, ignore_errors=True)
    os.makedirs(KAGGLE_INPUT_PATH, exist_ok=True)
    os.makedirs(KAGGLE_WORKING_PATH, exist_ok=True)

    for path, name in [(KAGGLE_INPUT_PATH, 'input'), (KAGGLE_WORKING_PATH, 'working')]:
        try:
            os.symlink(path, os.path.join("..", name), target_is_directory=True)
        except FileExistsError:
            logging.warning(f"Symlink for {name} already exists.")

def main():
    """Main function to run the script."""
    setup_directories()
    # Add other main functions here as we refactor the code

if __name__ == "__main__":
    main()

for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
    directory, download_url_encoded = data_source_mapping.split(':')
    download_url = unquote(download_url_encoded)
    filename = urlparse(download_url).path
    destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
    try:
        with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
            total_length = fileres.headers['content-length']
            print(f'Downloading {directory}, {total_length} bytes compressed')
            dl = 0
            data = fileres.read(CHUNK_SIZE)
            while len(data) > 0:
                dl += len(data)
                tfile.write(data)
                done = int(50 * dl / int(total_length))
                sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded")
                sys.stdout.flush()
                data = fileres.read(CHUNK_SIZE)
            if filename.endswith('.zip'):
              with ZipFile(tfile) as zfile:
                zfile.extractall(destination_path)
            else:
              with tarfile.open(tfile.name) as tarfile:
                tarfile.extractall(destination_path)
            print(f'\nDownloaded and uncompressed: {directory}')
    except HTTPError as e:
        print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
        continue
    except OSError as e:
        print(f'Failed to load {download_url} to path {destination_path}')
        continue

print('Data source import complete.')


"""
Spotify Song Recommendation System using Data Analysis and Machine Learning
"""

import os
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Configure matplotlib for non-interactive environments
plt.switch_backend('agg')

def load_data():
    """Load and return the datasets."""
    base_path = './data/'
    datasets = {
        'main': "data.csv",
        'genre': 'data_by_genres.csv',
        'year': 'data_by_year.csv'
    }
    loaded_data = {}

    for name, filename in datasets.items():
        path = os.path.join(base_path, filename)
        try:
            logging.info(f"Loading {name} dataset from {path}...")
            df = pd.read_csv(path)
            loaded_data[name] = df
            logging.info(f"{name.capitalize()} dataset shape: {df.shape}")
            logging.info(f"{name.capitalize()} dataset info:")
            logging.info(df.info())
        except FileNotFoundError:
            logging.error(f"File not found: {path}")
        except pd.errors.EmptyDataError:
            logging.error(f"Empty CSV file: {path}")
        except pd.errors.ParserError:
            logging.error(f"Error parsing CSV file: {path}")
        except Exception as e:
            logging.error(f"Unexpected error loading {name} dataset: {str(e)}")

    if len(loaded_data) != len(datasets):
        logging.warning("Not all datasets were loaded successfully.")
        if not loaded_data:
            logging.error("No datasets were loaded. Please check the data directory and file names.")
            return None

    return loaded_data

def visualize_feature_correlation(data):
    """
    Visualize feature correlation with the target 'popularity'.
    """
    try:
        logging.info("Visualizing feature correlation")
        from yellowbrick.target import FeatureCorrelation

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

def process_data():
    """Load data and perform initial processing."""
    loaded_data = load_data()

    if 'main' in loaded_data:
        visualize_feature_correlation(loaded_data['main'])
    else:
        logging.warning("Main dataset not available for feature correlation visualization.")

    return loaded_data

# Load and process the datasets
datasets = process_data()

# Assign datasets to variables for easier access
data = datasets.get('main')
genre_data = datasets.get('genre')
year_data = datasets.get('year')

if data is None or genre_data is None or year_data is None:
    logging.error("One or more required datasets are missing. Please check the data loading process.")
    sys.exit(1)

# # **Veri Görselleştirmesi**

# # **Yıllara Göre Müzik**
# 
# Yıllara göre gruplandırılmış verileri kullanarak müziğin 1921 ve 2020 yılları arasındaki değişimini görmekteyiz.

# In[ ]:


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

# Call the visualization functions
if 'main' in datasets:
    visualize_decades(datasets['main'])
else:
    logging.warning("Main dataset not available for decades visualization.")

if 'year' in datasets:
    visualize_sound_features(datasets['year'])
else:
    logging.warning("Year dataset not available for sound features visualization.")


# # **Türlere Atanan Parametreler**
# 

# In[ ]:


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
        return fig
    except Exception as e:
        logging.error(f"Error in visualizing top genres: {str(e)}")
        return None

# Visualize top genres
if 'genre' in datasets:
    top_genres_fig = visualize_top_genres(datasets['genre'])
    if top_genres_fig:
        top_genres_fig.show()
        top_genres_fig.write_image('top_genres.png')
        logging.info("Top genres visualization saved as 'top_genres.png'")
    else:
        logging.warning("Failed to generate top genres visualization.")
else:
    logging.warning("Genre dataset not available for top genres visualization.")


# # **Şarkı Türlerinin Görselleştirilmesi**
# 

# In[ ]:


# scikit-learn should be included in the project's requirements

# In[252]:


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

# Implement genre clustering and visualization
clustered_genre_data = cluster_genres(datasets['genre'])
if clustered_genre_data is not None:
    fig = visualize_genre_clusters(clustered_genre_data)
    if fig is not None:
        fig.write_image('genre_clusters.png')
        logging.info("Genre clusters visualization saved as 'genre_clusters.png'")
    else:
        logging.warning("Failed to generate genre clusters visualization.")
else:
    logging.warning("Failed to cluster genre data.")


# # **Şarkı İsimlerine Göre Veri Görselleştirmesi**

# In[ ]:


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

# Song clustering and visualization will be moved to the main flow of the script
# This code block is intentionally left empty and will be implemented in the main function


# In[ ]:


# Spotipy should be installed as part of the project requirements
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# In[254]:


import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Spotify client
try:
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id="aa96343daac04feb9210c59311e4ef7a",
        client_secret="22b975444578402d9dd7f740fe0d1cd7"
    ))
except Exception as e:
    logging.error(f"Failed to initialize Spotify client: {str(e)}")
    sp = None

def find_song(name, year):
    if sp is None:
        logging.error("Spotify client is not initialized")
        return None

    song_data = defaultdict()
    try:
        results = sp.search(q=f'track: {name} year: {year}', limit=1)
        if not results['tracks']['items']:
            logging.warning(f"No results found for {name} ({year})")
            return None

        track = results['tracks']['items'][0]
        audio_features = sp.audio_features(track['id'])[0]

        song_data['name'] = [name]
        song_data['year'] = [year]
        song_data['explicit'] = [int(track['explicit'])]
        song_data['duration_ms'] = [track['duration_ms']]
        song_data['popularity'] = [track['popularity']]

        song_data.update({key: [value] for key, value in audio_features.items()})

        return pd.DataFrame(song_data)
    except Exception as e:
        logging.error(f"Error finding song {name} ({year}): {str(e)}")
        return None

number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
               'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

def get_song_data(song, spotify_data):
    try:
        return spotify_data[(spotify_data['name'] == song['name']) &
                            (spotify_data['year'] == song['year'])].iloc[0]
    except IndexError:
        return find_song(song['name'], song['year'])

def get_mean_vector(song_list, spotify_data):
    song_vectors = []
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            logging.warning(f"Skipping {song['name']} ({song['year']}): Not found in Spotify or database")
            continue
        song_vector = song_data[number_cols].values
        if len(song_vector) == len(number_cols):
            song_vectors.append(song_vector)
        else:
            logging.warning(f"Skipping {song['name']}: Unexpected number of features")

    if not song_vectors:
        logging.error("No valid songs found to create mean vector")
        return None

    return np.mean(np.array(song_vectors), axis=0)

def flatten_dict_list(dict_list):
    return {key: [d[key] for d in dict_list] for key in dict_list[0].keys()}

def recommend_songs(song_list, spotify_data, n_songs=10):
    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)

    song_center = get_mean_vector(song_list, spotify_data)
    if song_center is None:
        return []

    try:
        # Define song_cluster_pipeline
        song_cluster_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('kmeans', KMeans(n_clusters=20, verbose=False))
        ])

        # Fit the pipeline on the spotify_data
        song_cluster_pipeline.fit(spotify_data[number_cols])

        scaler = song_cluster_pipeline.named_steps['scaler']
        scaled_data = scaler.transform(spotify_data[number_cols])
        scaled_song_center = scaler.transform(song_center.reshape(1, -1))
        distances = cdist(scaled_song_center, scaled_data, 'cosine')
        index = list(np.argsort(distances)[:, :n_songs][0])

        rec_songs = spotify_data.iloc[index]
        rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
        return rec_songs[metadata_cols].to_dict(orient='records')
    except Exception as e:
        logging.error(f"Error in recommending songs: {str(e)}")
        return []

def is_music_related(message):
    """
    Determine if the message is music-related using simple keyword matching.
    """
    music_keywords = ['song', 'music', 'artist', 'album', 'playlist', 'genre', 'recommend', 'similar']
    return any(keyword in message.lower() for keyword in music_keywords)

def generate_general_response(message):
    """
    Generate a general response using a pre-trained language model.
    """
    try:
        inputs = tokenizer.encode(f"Human: {message}\nAI:", return_tensors="pt")
        outputs = model.generate(inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("AI:")[-1].strip()
    except Exception as e:
        logging.error(f"Error in generating general response: {str(e)}")
        return "I'm sorry, I couldn't generate a response. Can you please try rephrasing your question?"

def chatbot_recommend_songs(message, history):
    """
    Function to be called by the chatbot for song recommendations and general queries.
    """
    try:
        if not is_music_related(message):
            return generate_general_response(message) + "\n\nRemember, I'm primarily a music recommendation chatbot. Feel free to ask about songs or artists!"

        song_list = parse_song_input(message)

        if not song_list:
            return "I couldn't find any valid song requests in your message. Please provide songs in the format 'Song Name, Year'. Or, you can ask me about music in general!"

        recommendations = recommend_songs(song_list, data)
        if not recommendations:
            return "I'm sorry, I couldn't find any recommendations based on those songs. Could you try with different songs?"

        rec_list = [f"{song['name']} ({song['year']}) by {', '.join(song['artists'])}" for song in recommendations]
        response = "Based on your song choices, here are some recommendations:\n" + "\n".join(rec_list)

        return response + "\n\nWould you like more recommendations or information about any of these songs?"
    except Exception as e:
        logging.error(f"Error in chatbot recommendation: {str(e)}")
        return "I'm sorry, there was an error while processing your request. Could you please try again with a different set of songs or question?"

# Example usage (for testing purposes)
if __name__ == "__main__":
    test_message = "Come As You Are, 1991\nWhat I've Done, 2007"
    print(chatbot_recommend_songs(test_message, []))


# In[ ]:


import gradio as gr
import logging
import re

def parse_song_input(input_text):
    song_list = []
    pattern = r'(.*?)\s*(?:\((\d{4})\))?'
    matches = re.findall(pattern, input_text)
    for match in matches:
        name, year = match
        name = name.strip()
        if name:
            try:
                year = int(year) if year else None
                song_list.append({"name": name, "year": year})
            except ValueError:
                logging.warning(f"Invalid year format for song: {name}")
    return song_list

def recommend_songs_interface(song_input):
    try:
        song_list = parse_song_input(song_input)
        if not song_list:
            return "I couldn't identify any songs in your message. Could you please provide song names, optionally with years?"

        recommendations = recommend_songs(song_list, data)
        if not recommendations:
            return "I'm sorry, I couldn't find any recommendations based on those songs. Could you try with different songs?"

        formatted_recommendations = [f"{song['name']} ({song['year']}) by {', '.join(song['artists'])}" for song in recommendations]
        return "Based on your input, here are some song recommendations:\n" + "\n".join(formatted_recommendations)
    except Exception as e:
        logging.error(f"Error in recommendation process: {str(e)}")
        return "I'm sorry, but I encountered an error while processing your request. Could you please try again?"

def chatbot_interface(message, history):
    if not message.strip():
        return "Hello! I'm your music recommendation chatbot. I can suggest songs and answer general music-related questions. What can I help you with today?"

    # Check if the message is a song recommendation request or a general query
    if is_song_request(message):
        response = recommend_songs_interface(message)
    else:
        response = generate_general_response(message)

    # Always remind the user about the primary function
    return response + "\n\nRemember, I'm here to help you discover new music. Feel free to ask for song recommendations anytime!"

def is_song_request(message):
    # Simple heuristic to determine if the message is asking for song recommendations
    keywords = ['recommend', 'suggestion', 'similar to', 'like', 'song', 'track', 'artist']
    return any(keyword in message.lower() for keyword in keywords)

def generate_general_response(message):
    # Use the language model to generate a response for general queries
    inputs = tokenizer(f"Human: {message}\nAI:", return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=150, num_return_sequences=1, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("AI:")[-1].strip()

interface = gr.ChatInterface(
    fn=chatbot_interface,
    title="Music Recommendation Chatbot",
    description="Chat with me about songs you like, and I'll recommend similar ones! You can mention song names with or without years.",
    examples=[
        ["I like Come As You Are by Nirvana"],
        ["Recommend me songs similar to What I've Done by Linkin Park from 2007"],
        ["I enjoy listening to Bohemian Rhapsody and Stairway to Heaven. Any recommendations?"]
    ],
)

def main():
    # Load and process data
    datasets = process_data()

    # Perform visualizations
    if 'main' in datasets:
        visualize_decades(datasets['main'])
    if 'year' in datasets:
        visualize_sound_features(datasets['year'])
    if 'genre' in datasets:
        top_genres_fig = visualize_top_genres(datasets['genre'])
        if top_genres_fig:
            top_genres_fig.show()

    # Perform clustering
    if 'genre' in datasets:
        clustered_genre_data = cluster_genres(datasets['genre'])
        if clustered_genre_data is not None:
            genre_cluster_fig = visualize_genre_clusters(clustered_genre_data)
            if genre_cluster_fig is not None:
                genre_cluster_fig.show()

    if 'main' in datasets:
        clustered_song_data = create_song_clusters(datasets['main'])
        if clustered_song_data is not None:
            song_cluster_fig = visualize_song_clusters(clustered_song_data)
            if song_cluster_fig is not None:
                song_cluster_fig.show()

    # Launch the Gradio interface
    interface.launch()

if __name__ == "__main__":
    main()


# In[ ]:


import pandas as pd

data = pd.read_csv("/content/data/data.csv")

print(data.head())

