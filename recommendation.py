"""
recommendation.py

This module provides functionality for recommending songs based on user input.
It uses the Spotify API for fetching song data and implements a recommendation
algorithm using clustering and cosine similarity.
"""

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
import ast
import re
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define number_cols here as it's used in get_mean_vector
number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
               'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Spotify client
try:
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=os.getenv('SPOTIFY_CLIENT_ID'),
        client_secret=os.getenv('SPOTIFY_CLIENT_SECRET')
    ))
except Exception as e:
    logging.error(f"Failed to initialize Spotify client: {str(e)}")
    sp = None

def find_song(name, year):
    if sp is None:
        logging.error("Spotify client is not initialized")
        return None

    song_data = defaultdict(lambda: [None])
    try:
        logging.info(f"Searching Spotify for track: {name}, year: {year}")
        results = sp.search(q=f'track: {name} year: {year}', limit=1)
        logging.debug(f"Full Spotify search results: {results}")
        if not results['tracks']['items']:
            logging.warning(f"No results found for {name} ({year})")
            return None

        track = results['tracks']['items'][0]
        logging.info(f"Found track: {track['name']} by {', '.join([artist['name'] for artist in track['artists']])}")
        logging.debug(f"Full track data: {track}")

        audio_features = sp.audio_features(track['id'])[0]
        logging.debug(f"Full audio features: {audio_features}")

        # Ensure all required columns are present with default values
        for col in number_cols:
            if col == 'year':
                song_data[col] = [year]
            elif col == 'explicit':
                song_data[col] = [int(track.get('explicit', False))]
            elif col == 'popularity':
                song_data[col] = [track.get('popularity', 0)]
            elif col in audio_features:
                song_data[col] = [audio_features[col]]
            else:
                song_data[col] = [0]  # Default value for missing features
            logging.debug(f"Column {col}: {song_data[col]}")

        song_data['name'] = [name]
        song_data['artists'] = [', '.join([artist['name'] for artist in track['artists']])]

        df = pd.DataFrame(song_data)
        logging.info(f"Created DataFrame with columns: {df.columns}")
        logging.debug(f"DataFrame shape: {df.shape}")
        logging.debug(f"DataFrame content:\n{df.to_string()}")

        missing_cols = set(number_cols) - set(df.columns)
        if missing_cols:
            logging.warning(f"Missing columns in retrieved data: {missing_cols}")

        final_df = df[['name', 'artists'] + number_cols]
        logging.info(f"Final DataFrame shape: {final_df.shape}")
        logging.debug(f"Final DataFrame content:\n{final_df.to_string()}")
        return final_df
    except Exception as e:
        logging.error(f"Error finding song {name} ({year}): {str(e)}")
        return None

def get_song_data(song, spotify_data):
    logging.info(f"Attempting to get data for song: {song['name']} ({song['year']})")
    try:
        result = spotify_data[(spotify_data['name'] == song['name']) &
                              (spotify_data['year'] == song['year'])].iloc[0]
        logging.info(f"Found song data in dataset: {song['name']} ({song['year']})")
        logging.debug(f"Song data from dataset: {result}")
        return result
    except IndexError:
        logging.info(f"Song not found in dataset, searching Spotify API: {song['name']} ({song['year']})")
        return find_song(song['name'], song['year'])

def get_mean_vector(song_list, spotify_data):
    song_vectors = []
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            logging.warning(f"Skipping {song['name']} ({song.get('year', 'Unknown')}): Not found in Spotify or database")
            continue

        try:
            # Ensure song_data is a DataFrame
            if isinstance(song_data, pd.Series):
                song_data = song_data.to_frame().T

            # Check if all required columns are present
            missing_cols = set(number_cols) - set(song_data.columns)
            if missing_cols:
                logging.warning(f"Missing columns for {song['name']}: {missing_cols}")
                # Add missing columns with default values
                for col in missing_cols:
                    song_data[col] = 0

            # Extract features in the correct order and handle missing or invalid data
            song_vector = []
            for col in number_cols:
                value = song_data[col].values[0]
                if pd.isna(value) or np.isinf(value) or value is None:
                    logging.warning(f"Invalid value for {col} in {song['name']}, using default")
                    if col == 'year':
                        value = song.get('year') or 2000
                    else:
                        value = 0
                try:
                    song_vector.append(float(value))
                except ValueError:
                    logging.warning(f"Could not convert {col} to float for {song['name']}, using 0")
                    song_vector.append(0.0)

            logging.info(f"Processing song: {song['name']} ({song.get('year', 'Unknown')})")
            logging.debug(f"Song data shape: {song_data.shape}, Song vector shape: {len(song_vector)}")
            logging.debug(f"Song data columns: {song_data.columns.tolist()}")
            logging.debug(f"Song vector content: {song_vector}")

            if len(song_vector) == len(number_cols):
                song_vectors.append(song_vector)
                logging.info(f"Successfully added song vector for {song['name']}")
            else:
                logging.warning(f"Skipping {song['name']}: Unexpected number of features. Expected {len(number_cols)}, got {len(song_vector)}")
                logging.debug(f"Expected columns: {number_cols}")
                logging.debug(f"Actual columns: {song_data.columns.tolist()}")
        except Exception as e:
            logging.error(f"Error processing {song['name']}: {str(e)}")
            logging.exception("Detailed error information:")

    if not song_vectors:
        logging.error("No valid songs found to create mean vector")
        return None

    try:
        mean_vector = np.mean(np.array(song_vectors), axis=0)
        logging.info(f"Created mean vector with shape: {mean_vector.shape}")
        logging.debug(f"Mean vector content: {mean_vector.tolist()}")
        return mean_vector
    except Exception as e:
        logging.error(f"Error creating mean vector: {str(e)}")
        return None

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

        # Clean up artist names
        def clean_artists(artists):
            if isinstance(artists, str):
                try:
                    # Try to safely evaluate the string as a list
                    artist_list = ast.literal_eval(artists)
                    if isinstance(artist_list, list):
                        artists = artist_list
                    else:
                        raise ValueError
                except (ValueError, SyntaxError):
                    # If evaluation fails, split the string
                    artists = re.split(r'[,&]', artists)
            elif not isinstance(artists, list):
                artists = [str(artists)]

            # Clean and join artist names
            cleaned = ', '.join(artist.strip() for artist in artists if artist.strip())
            # Remove any remaining brackets or quotes
            cleaned = re.sub(r'[\[\]\'"]', '', cleaned)
            return ', '.join(artist.strip() for artist in cleaned.split(',') if artist.strip())

        rec_songs['artists'] = rec_songs['artists'].apply(clean_artists)

        recommendations = rec_songs[metadata_cols].to_dict(orient='records')
        return [{'name': song['name'], 'year': song['year'], 'artists': song['artists']} for song in recommendations]
    except Exception as e:
        logging.error(f"Error in recommending songs: {str(e)}")
        return []

# Define number_cols here as it's used in get_mean_vector
number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
               'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']
