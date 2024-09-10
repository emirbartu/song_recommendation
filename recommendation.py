import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from collections import defaultdict
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

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

def recommend_songs(song_list, spotify_data, n_songs=10):
    metadata_cols = ['name', 'year', 'artists']
    song_dict = {key: [d[key] for d in song_list] for key in song_list[0].keys()}

    song_center = get_mean_vector(song_list, spotify_data)
    if song_center is None:
        return []

    try:
        song_cluster_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('kmeans', KMeans(n_clusters=20, verbose=False))
        ])

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

# Define number_cols here as it's used in get_mean_vector
number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
               'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']
