import logging
import gradio as gr
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import re

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

# Initialize language model
MODEL_NAME = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
nlp = pipeline("text-generation", model=model, tokenizer=tokenizer)

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

def chatbot_recommend_songs(message, history):
    try:
        if not is_music_related(message):
            return generate_general_response(message) + "\n\nRemember, I'm primarily a music recommendation chatbot. Feel free to ask about songs or artists!"

        song_list = parse_song_input(message)

        if not song_list:
            return "I couldn't find any valid song requests in your message. Please provide songs in the format 'Song Name, Year'. Or, you can ask me about music in general!"

        # This function needs to be implemented in the recommendation module
        from recommendation import recommend_songs
        recommendations = recommend_songs(song_list, sp)

        if not recommendations:
            return "I'm sorry, I couldn't find any recommendations based on those songs. Could you try with different songs?"

        rec_list = [f"{song['name']} ({song['year']}) by {', '.join(song['artists'])}" for song in recommendations]
        response = "Based on your song choices, here are some recommendations:\n" + "\n".join(rec_list)

        return response + "\n\nWould you like more recommendations or information about any of these songs?"
    except Exception as e:
        logging.error(f"Error in chatbot recommendation: {str(e)}")
        return "I'm sorry, there was an error while processing your request. Could you please try again with a different set of songs or question?"

interface = gr.ChatInterface(
    fn=chatbot_recommend_songs,
    title="Music Recommendation Chatbot",
    description="Chat with me about songs you like, and I'll recommend similar ones! You can mention song names with or without years.",
    examples=[
        ["I like Come As You Are by Nirvana"],
        ["Recommend me songs similar to What I've Done by Linkin Park from 2007"],
        ["I enjoy listening to Bohemian Rhapsody and Stairway to Heaven. Any recommendations?"]
    ],
)

def launch_chatbot():
    interface.launch()

if __name__ == "__main__":
    launch_chatbot()
