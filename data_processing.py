import os
import logging
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def process_data():
    """Load data and perform initial processing."""
    loaded_data = load_data()
    return loaded_data

def get_decade(year):
    """Convert a year to its corresponding decade."""
    period_start = int(year/10) * 10
    return f'{period_start}s'

if __name__ == "__main__":
    datasets = process_data()
    if datasets:
        print("Data loaded successfully.")
    else:
        print("Failed to load data.")
