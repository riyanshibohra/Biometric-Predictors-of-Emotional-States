import pandas as pd
import numpy as np
from pathlib import Path
from transformers import pipeline, set_seed
import random
import nltk
from nltk.corpus import wordnet as wn
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Function to load data
def load_data():
    """
    Load the datasets from specified paths.
    :return DataFrames for activity, interaction, and health data.
    """
    # Define data paths
    DATA_ROOT = Path(__file__).parents[0] / "data"
    PATH_TO_ACTIVITY_DATA = (DATA_ROOT / "activity_environment_data.csv").resolve()
    PATH_TO_HEALTH_DATA = (DATA_ROOT / "personal_health_data.csv").resolve()
    PATH_TO_INTERACTION_DATA = (DATA_ROOT / "digital_interaction_data.csv").resolve()

    # Load and return datasets
    activity = pd.read_csv(PATH_TO_ACTIVITY_DATA)
    interaction = pd.read_csv(PATH_TO_INTERACTION_DATA)
    health = pd.read_csv(PATH_TO_HEALTH_DATA)
    return activity, interaction, health

# Function for text generation
def text_generation():
    """
    Initializes the text generation pipeline.
    :return a text generation pipeline object.
    """
    nltk.download('wordnet')
    generator = pipeline('text-generation', model='gpt2')
    set_seed(42)
    return generator

# Function to get synonyms of a word from NLTK's WordNet
def synonyms(word):
    """
    Gets the synonyms for a given word using NLTK's WordNet.
    :param word: The input word for which synonyms are generated.
    :return list: A list of synonyms for the given word.
    """
    syn = set()
    for i in wn.synsets(word):
        for j in i.lemmas():
            syn.add(j.name())
    return list(syn)

# Function to generate prompt
def generate_prompt(mood):
    """
    Generates a prompt based on a mood.
    :param mood: The mood for which prompt is generated.
    :return str: Generated prompt.
    """

    # Define lists of sentence starters and mood-specific words

    # List of sentence starters
    sentence_starters = [
    "Today I'm feeling",
    "Right now, it seems like",
    "Looking around, I notice",
    "This moment feels",
    "I've been thinking and",
    "There's a sense that",
    "Reflecting on the day, I feel",
    "I find myself feeling",
    "The atmosphere around me is",
    "In this situation, I am",
    "Feeling a bit",
    "The day brings me",
    "As I sit here, I'm",
    "My mood is",
    "There's this feeling that",
    "Currently, my state of mind is"
     ]

    # Dictionary mapping moods to related words
    mood_words = {
    "Happy": ["joyful", "cheerful", "uplifted"],
    "Sad": ["downcast", "sorrowful", "dismal"],
    "Neutral": ["routine", "usual", "normal"],
    "Anxious": ["restless", "uneasy", "worried"]
     }

    start = random.choice(sentence_starters)
    mood_word = random.choice(mood_words[mood])
    mood_synonyms = synonyms(mood_word)

    if mood_synonyms:
        mood_word = random.choice(mood_synonyms)

    prompt = f"{start} {mood_word}"
    return prompt

# Function to generate batch descriptions
def batch_generate(batch, generator):
    """
    Generates text descriptions for a batch of data.
    :param batch (DataFrame): Batch of data.
    :param generator (pipeline): Text generation pipeline.
    :return: List of generated text descriptions.
    """
    descriptions = []
    for _, row in tqdm(batch.iterrows(), total=batch.shape[0], desc="Generating Descriptions"):
        prompt = generate_prompt(row['Mood'])
        max_tokens = random.randint(10, 15)
        generated_texts = generator(prompt, max_tokens=max_tokens, num_return_sequences=1)
        generated_text = generated_texts[0]['generated_text'].strip()
        tokens = generated_text.split()
        new_text = ' '.join(tokens[:15])
        descriptions.append(new_text)
    return descriptions

# Main function to orchestrate the script execution
def main():
    """
    Main function to perform the data processing, text generation, and saving results.
    """
    # Load data
    activity, interaction, health = load_data()

    # Merge datasets
    combined_data = pd.merge(health, activity, on=['User_ID', 'Timestamp'], how='outer')
    combined_data = pd.merge(combined_data, interaction, on=['User_ID', 'Timestamp'], how='outer')
    combined_data = combined_data.drop(['User_ID', 'Timestamp', 'Notifications_Received'], axis=1)

    # Initialize text generation pipeline
    generator = text_generation()

    # Process in batches
    size = 100
    num_batches = len(combined_data) // size
    combined_data['Mood_Description'] = pd.NA

    for i in range(num_batches + 1):
        start = i * size
        end = start + size
        batch = combined_data.iloc[start:end]
        descriptions = batch_generate(batch, generator)
        combined_data.loc[start:start + len(descriptions) - 1, 'Mood_Description'] = descriptions

    # Save results
    combined_data.to_csv('mood_descriptions.csv', index=False)

if __name__ == "__main__":
    main()
