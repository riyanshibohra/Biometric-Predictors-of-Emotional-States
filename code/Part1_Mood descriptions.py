## Importing Libraries

# Load necessary libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# TensorFlow and Keras libraries for building and training neural network models
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense,  Flatten
from transformers import BertTokenizer, TFBertModel

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Function to load and transform data
def load_and_transform_data():
    """
    Loads and transforms the dataset.
    :return DataFrame: The transformed data, with one-hot encoding applied to the 'Mood' column and unnecessary columns dropped.
    """

    DATA_ROOT = Path(__file__).parents[0] / "data"
    PATH_TO_COMBINED_DATA = (DATA_ROOT / "mood_descriptions.csv").resolve()
    
    combined_data = pd.read_csv(PATH_TO_COMBINED_DATA)

    # Perform data transformations using one-hot encoding
    mood_encoded = pd.get_dummies(combined_data['Mood'])
    combined_data = combined_data.join(mood_encoded)
    combined_data = combined_data[['Mood_Description', 'Happy', 'Neutral', 'Anxious', 'Sad']]

    return combined_data

# Data preprocessing function
def preprocess_data(combined_data):
    """
    Preprocesses the combined data for NLP tasks.
    :param: combined_data (DataFrame): The combined dataset containing the 'Mood_Description' and mood labels.
    Returns:
    :param tuple: Contains preprocessed text data, labels, vocabulary size, and maximum sequence length.
    """

    # Convert all text in the 'Mood_Description' series to lowercase to standardize the data
    description = combined_data['Mood_Description'].str.lower()
    
    labels = combined_data[['Neutral', 'Anxious', 'Happy', 'Sad']].values

    # Tokenization and vectorization
    tokenizer = Tokenizer()                 # Initialize a tokenizer
    tokenizer.fit_on_texts(description)     # Fit the tokenizer on the training data
    sequences = tokenizer.texts_to_sequences(description)      # Convert the text into sequences

    # Calculate the vocabulary size and maximum length
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(seq) for seq in sequences)

    # Padding sequences to ensure uniform length
    padded_seq = pad_sequences(sequences, maxlen=max_length, padding='post')

    return padded_seq, labels, vocab_size, max_length, tokenizer

# Function to train ensemble models
def train_ensemble_models(X_train, y_train, max_length, vocab_size):
    """
    Trains ensemble models, one for each mood class.
    :param X_train: Training features (text sequences).
    :param y_train: Training labels.
    :param max_length: Maximum length of the text sequences.
    :param vocab_size: Size of the vocabulary.
    :return list: A list of trained Sequential models, one for each class.
    """

    early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1)     # To prevent overfitting
    models = []             # Creating an empty list to store models for each class

    # Looping over each class to create and train a separate model
    for i in range(4):
        model = Sequential()       # initialize a Sequential model
        model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=max_length))          # adding an embedding layer
        model.add(LSTM(100))       # adding an LSTM layer with 100 units
        model.add(Dense(1, activation='sigmoid'))           # adding a dense layer for output
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Extracting the labels for the current class
        y_train_class = y_train[:,i]

        model.fit(X_train, y_train_class, epochs=3, batch_size=32, callbacks=[early_stopping], validation_split=0.1)
        models.append(model)

    return models

# Function to evaluate ensemble models
def evaluate_ensemble_models(models, X_test, y_test):
    """
    Evaluates ensemble models on the test dataset.
    :param models: A list of trained models.
    :param X_test: Test features.
    :param y_test: Test labels.
    :return tuple: Contains precision, recall, and F1-score for each class.
    """

    # Using the trained models to predict on the test data
    ensemble_pred = np.zeros((len(X_test), 4))

    for i, model in enumerate(models):
        ensemble_pred[:, i] = model.predict(X_test).flatten()

    # Finding the index of the maximum value in each row of ensemble_pred
    max_indices = np.argmax(ensemble_pred, axis=1)
    result = np.zeros_like(ensemble_pred)
    result[np.arange(len(X_test)), max_indices] = 1

    # Evaluating Model performance using Precision, Recall and F1 score
    precision = []
    recall = []
    f1 = []
    for i in range(4):
        precision.append(precision_score(y_test[:, i], result[:, i]))
        recall.append(recall_score(y_test[:, i], result[:, i]))
        f1.append(f1_score(y_test[:, i], result[:, i]))

    return precision, recall, f1

# Function to train BERT model
def train_bert_model(X_train, y_train, max_length):
    """
    Trains a BERT model for mood classification.
    :param X_train: Training features (text).
    :param y_train: Training labels.
    :param max_length: Maximum length for tokenized sequences.
    :return model: A trained Sequential model using BERT embeddings.
    """
    # Load the pre-trained BERT model and tokenizer   
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    X_train_tokens = tokenizer(X_train.tolist(), padding=True, truncation=True, return_tensors='tf', max_length=max_length)
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')

    # Extract BERT embeddings
    bert_output = bert_model(X_train_tokens)
    bert_embeddings = bert_output.last_hidden_state       # Selecting the last hidden state

    model = Sequential()            # constructing a Sequential model to fit on top of BERT embeddings
    model.add(Flatten(input_shape=(bert_embeddings.shape[1], bert_embeddings.shape[2])))          # adding a Flatten layer to convert 3D tensor to 2D tensor
    model.add(Dense(100, activation='relu'))            # adding a Dense layer with 100 neurons and 'relu' activation
    model.add(Dense(4, activation='softmax'))           # final output layer with 'softmax' activation
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(bert_embeddings, y_train, epochs=3, batch_size=32)

    return model

# Function to evaluate BERT model
def evaluate_bert_model(model, X_test, y_test, max_length):
    """
    Evaluates the BERT model on the test dataset.
    :param model: The trained BERT model.
    :param X_test: Test features.
    :param y_test: Test labels.
    :param max_length: Maximum length for tokenized sequences.
    :return tuple: Contains precision, recall, and F1-score for each class.
    """

    # Tokenizing and converting the test data
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    X_test_tokens = tokenizer(X_test.tolist(), padding=True, truncation=True, return_tensors='tf', max_length=max_length)
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')

    # Extract BERT embeddings for test data
    bert_output_test = bert_model(X_test_tokens)
    bert_embeddings_test = bert_output_test.last_hidden_state

    bert_pred = model.predict(bert_embeddings_test)

    # Finding the index of the maximum value in each row of bert_pred
    max_indices = np.argmax(bert_pred, axis=1)
    result = np.zeros_like(bert_pred)
    result[np.arange(bert_pred.shape[0]), max_indices] = 1

    # Evaluating Model performance using Precision, Recall and F1 score
    precision = []
    recall = []
    f1 = []

    for i in range(4):
        precision.append(precision_score(y_test[:, i], result[:, i]))
        recall.append(recall_score(y_test[:, i], result[:, i]))
        f1.append(f1_score(y_test[:, i], result[:, i]))

    return precision, recall, f1

# Function to create and save plots
def create_and_save_plots(precision, recall, f1):
    """
    Creates and saves plots for the evaluation metrics.
    :param precision: List of precision values for each class.
    :param recall: List of recall values for each class.
    :param f1: List of F1-score values for each class.
    """

    labels = ['Neutral', 'Anxious', 'Happy', 'Sad']
    bar_width = 0.2

    # Plot Precision
    r1 = np.arange(len(labels))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    plt.bar(r1, precision, width=bar_width, color='#E63946', label='Precision')
    plt.bar(r2, recall, width=bar_width, color='#ffa600', label='Recall')
    plt.bar(r3, f1, width=bar_width, color='#1f77b4', label='F1-Score')
    plt.xlabel('Labels')
    plt.ylabel('Scores')
    plt.title('Model Evaluation Metrics')
    plt.xticks([r + bar_width for r in range(len(labels))], labels)
    plt.legend()
    plt.savefig('model_metrics.png', bbox_inches='tight', dpi=300)
    plt.show()

# Main function
def main():
    # Load and transform data
    combined_data = load_and_transform_data()

    # Preprocess data
    X, y, vocab_size, max_length, tokenizer = preprocess_data(combined_data)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train ensemble models
    ensemble_models = train_ensemble_models(X_train, y_train, max_length, vocab_size)

    # Evaluate ensemble models
    ensemble_precision, ensemble_recall, ensemble_f1 = evaluate_ensemble_models(ensemble_models, X_test, y_test)

    # Train BERT model
    bert_model = train_bert_model(X_train, y_train, max_length)

    # Evaluate BERT model
    bert_precision, bert_recall, bert_f1 = evaluate_bert_model(bert_model, X_test, y_test, max_length)

    # Create and save comparison plots
    create_and_save_plots(ensemble_precision + bert_precision, ensemble_recall + bert_recall, ensemble_f1 + bert_f1)

if __name__ == "__main__":
    main()


