
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence

# Load trained model
model = tf.keras.models.load_model(
    "artifacts/sentiment_rnn.keras"
)

# Load saved vocabulary
with open(
    "artifacts/word_index.json",
    "r",
    encoding="utf-8"
) as file:
    word_index = json.load(file)

with open(
    "artifacts/config.json",
    "r",
    encoding="utf-8"
) as file:
    config = json.load(file)

VOCAB_SIZE = config["vocab_size"]
MAX_LENGTH = config["max_length"]


def encode_review(text):
    words = text_to_word_sequence(text)
    sequence = [1]  # IMDB start token

    for word in words:
        index = word_index.get(word, 2) + 3

        # Reserve 0 for padding, 1 for start, 2 for OOV
        if index >= VOCAB_SIZE:
            index = 2

        sequence.append(index)

    return pad_sequences(
        [sequence],
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )


def predict_sentiment(text):
    encoded = encode_review(text)

    probability = float(
        model.predict(encoded, verbose=0)[0][0]
    )

    sentiment = (
        "Positive" if probability >= 0.5
        else "Negative"
    )

    return sentiment, probability


if __name__ == "__main__":
    review = input("Enter a movie review: ")

    sentiment, probability = predict_sentiment(review)

    print("Predicted sentiment:", sentiment)
    print(f"Positive sentiment probability: {probability:.2%}")
