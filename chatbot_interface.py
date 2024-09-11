import logging
import gradio as gr
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import re
import pandas as pd
from recommendation import recommend_songs

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Spotify client
import os

try:
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=os.environ.get("SPOTIFY_CLIENT_ID"),
        client_secret=os.environ.get("SPOTIFY_CLIENT_SECRET")
    ))
except Exception as e:
    logging.error(f"Failed to initialize Spotify client: {str(e)}")
    sp = None

# Initialize language model
MODEL_NAME = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
nlp = pipeline("text-generation", model=model, tokenizer=tokenizer)

def parse_song_input(input_text):
    song_list = []
    logging.info(f"Original input: {input_text}")

    # Remove common phrases and unnecessary words
    input_text = re.sub(r'\b(i like|i enjoy|recommend me songs similar to|any recommendations|listening to)\b', '', input_text, flags=re.IGNORECASE)
    input_text = input_text.strip()
    logging.info(f"Input after removing common phrases: {input_text}")

    # Remove punctuation except for parentheses, hyphens, and commas
    input_text = re.sub(r'[^\w\s(),\-]', '', input_text)

    # Split input by common separators
    raw_songs = re.split(r'\s*,\s*|\s+and\s+|\s*\|\s*', input_text)
    logging.info(f"Split raw songs: {raw_songs}")

    for raw_song in raw_songs:
        # Simplified pattern to match song name with optional year and artist
        pattern = r'^(?P<name>[^(]+)(?:\s*\((?P<year>\d{4})\))?(?:\s*(?:by|-)?\s*(?P<artist>.+))?$'
        match = re.match(pattern, raw_song.strip(), re.IGNORECASE)

        if match:
            song_info = match.groupdict()
            song_info['name'] = song_info['name'].strip()
            if song_info['name']:
                song_info['year'] = int(song_info['year']) if song_info.get('year') else None
                song_info['artist'] = song_info['artist'].strip() if song_info.get('artist') else None
                song_list.append(song_info)
                logging.info(f"Successfully parsed song: {song_info}")
            else:
                logging.warning(f"Empty song name after parsing: {raw_song}")
        else:
            logging.warning(f"Could not parse song information from: {raw_song}")

    logging.info(f"Final parsed song list: {song_list}")
    return song_list

def is_music_related(message):
    music_keywords = ['song', 'music', 'artist', 'album', 'playlist', 'genre', 'recommend', 'similar']
    return any(keyword in message.lower() for keyword in music_keywords)

def generate_general_response(message):
    try:
        inputs = tokenizer.encode(f"Human: {message}\nAI:", return_tensors="pt")
        outputs = model.generate(inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("AI:")[-1].strip()
    except Exception as e:
        logging.error(f"Error in generating general response: {str(e)}")
        return "I'm sorry, I couldn't generate a response. Can you please try rephrasing your question?"

def chatbot_recommend_songs(message, history, recommendation_system):
    try:
        logging.info(f"Received input message: {message}")

        if not is_music_related(message):
            return generate_general_response(message) + "\n\nBy the way, I'm great at recommending music! Feel free to ask me about songs or artists you enjoy."

        song_list = parse_song_input(message)
        logging.info(f"Parsed song list: {song_list}")

        if not song_list:
            return "I couldn't quite catch any song names there. Could you please mention some songs you like? For example, you could say 'I love Bohemian Rhapsody by Queen'."

        recommendations = recommendation_system(song_list)
        logging.info(f"Received recommendations: {recommendations}")

        if not recommendations:
            return "Hmm, I'm having trouble finding recommendations based on those songs. How about we try with some different tracks? What other music do you enjoy?"

        rec_list = [f"• {song['name']} ({song['year']}) by {song['artists']}" for song in recommendations[:5]]
        response = f"Great choice! I've got some exciting recommendations for you. The results are like this:\n\n" + "\n".join(rec_list)

        return response + "\n\nWhat do you think about these suggestions? Would you like to hear more about any of them, or should I recommend some different tracks?"
    except Exception as e:
        logging.error(f"Error in chatbot recommendation: {str(e)}")
        return "Oops! It seems I hit a wrong note there. Could we start over? Tell me about some songs or artists you've been enjoying lately."

# The global interface definition has been removed as it's no longer needed.
# The interface is now created within the launch_chatbot function.

def launch_chatbot(recommendation_system):
    global interface
    interface = gr.ChatInterface(
        fn=lambda message, history: chatbot_recommend_songs(message, history, recommendation_system),
        title="Music Recommendation Chatbot",
        description="Chat with me about songs you like, and I'll recommend similar ones! You can mention song names with or without years.",
        examples=[
            ["I like Come As You Are by Nirvana"],
            ["Recommend me songs similar to What I've Done by Linkin Park from 2007"],
            ["I enjoy listening to Bohemian Rhapsody and Stairway to Heaven. Any recommendations?"]
        ],
    )
    interface.launch()

if __name__ == "__main__":
    import pandas as pd
    from recommendation import recommend_songs
    import os

    # Load the Spotify dataset
    try:
        data = pd.read_csv("./data/data.csv")
        logging.info("Successfully loaded Spotify dataset.")
    except FileNotFoundError:
        logging.error("Spotify dataset not found. Please ensure 'data.csv' is in the './data/' directory.")
        exit(1)
    except pd.errors.EmptyDataError:
        logging.error("The Spotify dataset file is empty.")
        exit(1)
    except Exception as e:
        logging.error(f"An error occurred while loading the Spotify dataset: {str(e)}")
        exit(1)

    # Define the actual recommendation system
    def recommendation_system(song_list):
        try:
            return recommend_songs(song_list, data)
        except Exception as e:
            logging.error(f"Error in recommendation system: {str(e)}")
            return []

    # Check if Spotify credentials are set
    if not os.environ.get("SPOTIFY_CLIENT_ID") or not os.environ.get("SPOTIFY_CLIENT_SECRET"):
        logging.warning("Spotify credentials not set. Some features may not work properly.")

    launch_chatbot(recommendation_system)
