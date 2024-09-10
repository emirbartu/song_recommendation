#!/usr/bin/env python
# coding: utf-8

import logging
from data_processing import process_data
from chatbot_interface import launch_chatbot
from recommendation import recommend_songs

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Main function to run the script."""
    # Load and process data
    datasets = process_data()

    if datasets:
        # Initialize the recommendation system
        recommendation_system = lambda song_list: recommend_songs(song_list, datasets['main'])

        # Launch the chatbot interface with the recommendation system
        launch_chatbot(recommendation_system)
    else:
        logging.error("Failed to load datasets. Please check the data processing module.")

if __name__ == "__main__":
    main()
