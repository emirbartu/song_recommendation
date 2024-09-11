# fizy_muzik_onerme (Music Recommendation System)

## Project Overview
fizy_muzik_onerme is an advanced music recommendation system developed for the Turkcell Hackathon Fizy application. This project extends a Kaggle-based music recommendation system, enhancing it with a user-focused experience and integrating a chatbot interface using Gradio. The system leverages machine learning techniques and the Spotify API to provide personalized music recommendations through an interactive chatbot.

## File Structure
The project has been refactored for improved modularity and maintainability:
- `main.py`: Main script that orchestrates the entire application
- `data_processing.py`: Handles data loading and initial processing
- `recommendation.py`: Core recommendation engine
- `chatbot_interface.py`: Manages the Gradio-based chatbot interface

## Features
- Personalized song recommendations based on user input
- Interactive chatbot interface for music discovery
- Integration with Spotify API for up-to-date song information
- Natural language processing for understanding user queries
- Comprehensive music recommendation system based on song similarities

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/your-username/fizy_muzik_onerme.git
   cd fizy_muzik_onerme
   ```
2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Set up the .env file:
   - Create a file named `.env` in the root directory of the project
   - Obtain Spotify API credentials:
     a. Go to https://developer.spotify.com/ and log in or create an account
     b. Create a new application to get your `client_id` and `client_secret`
   - Add your Spotify API credentials to the `.env` file in the following format:
     ```
     SPOTIFY_CLIENT_ID=your_client_id_here
     SPOTIFY_CLIENT_SECRET=your_client_secret_here
     ```
   - Keep your `.env` file secure and never share it publicly
   - Note: The `.env` file is crucial for the project to interact with the Spotify API

## How to Use
1. Obtain Spotify API credentials:
   - Create a Spotify Developer account at https://developer.spotify.com/
   - Create a new application to get your `client_id` and `client_secret`
   - Add these credentials to your `.env` file as described in the installation steps

2. Prepare your data:
   - Place the Spotify dataset CSV files in the `./data/` directory
   - Ensure you have `data.csv`, `data_by_genres.csv`, and `data_by_year.csv`

3. Run the main script:
   ```
   python main.py
   ```

4. The script will perform the following actions:
   - Load and process the data
   - Launch the Gradio interface

5. Open the provided Gradio interface URL in your web browser

6. Interact with the chatbot:
   - Enter song names (optionally with years) for recommendations
   - Ask general music-related questions
   - Explore recommended songs and their features

## Dependencies
- pandas, numpy: Data manipulation and analysis
- scikit-learn: Machine learning algorithms for preprocessing
- spotipy: Spotify API integration
- gradio: Web interface for the chatbot
- transformers, torch: Natural language processing for the chatbot

## Data Sources
The project uses the Spotify dataset available on Kaggle. You can find more information about the original dataset [here](https://www.kaggle.com/code/vatsalmavani/music-recommendation-system-using-spotify-dataset/notebook).

## Development
This project is continuously evolving. Future plans include:
- Enhanced chatbot integration for a more intuitive user experience
- Improved recommendation algorithms incorporating user feedback
- Extended genre and mood-based recommendations
- Integration with more music streaming platforms

## Visualization for Researchers
While the main functionality of fizy_muzik_onerme focuses on the chatbot interface and music recommendations, the project originally included various data visualization features that may be of interest to researchers in the field of music information retrieval and recommendation systems.

These visualization features, although not part of the main chatbot functionality, can provide valuable insights into music trends, genre characteristics, and the underlying structure of the dataset. They include:

1. Distribution of songs across decades
2. Evolution of sound features over time
3. Top genres based on popularity
4. Feature correlation with song popularity
5. Clustering of genres and songs

Researchers interested in exploring these visualizations can refer to the `visualization.py` and `clustering.py` files in the project repository. These modules contain functions for creating various plots and performing clustering analyses.

To use these visualization features:
1. Ensure you have the necessary additional dependencies installed (matplotlib, seaborn, plotly, yellowbrick)
2. Import the required functions from `visualization.py` and `clustering.py`
3. Call these functions with the processed dataset to generate visualizations

Please note that these visualization features are optional and separate from the main recommendation system. They are provided for research purposes and to offer a deeper understanding of the dataset and the factors influencing music popularity and recommendations.

## Contributing
Contributions to improve fizy_muzik_onerme are welcome. Please feel free to submit pull requests or open issues to discuss potential enhancements.

## License
[MIT]

## Acknowledgements
- Original Kaggle project by Vatsal Mavani
- Spotify API for providing music data
- Gradio for the web interface framework
- The open-source community for various libraries and tools used in this project
