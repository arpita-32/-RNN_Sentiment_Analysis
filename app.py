
import os
import json
import numpy as np
import streamlit as st
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence

ARTIFACT_DIR = "artifacts"

st.set_page_config(
    page_title="Movie Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)


@st.cache_resource
def load_model_and_vocabulary():
    model = tf.keras.models.load_model(
        os.path.join(ARTIFACT_DIR, "sentiment_rnn.keras")
    )

    with open(
        os.path.join(ARTIFACT_DIR, "word_index.json"),
        "r",
        encoding="utf-8"
    ) as file:
        word_index = json.load(file)

    with open(
        os.path.join(ARTIFACT_DIR, "config.json"),
        "r",
        encoding="utf-8"
    ) as file:
        config = json.load(file)

    return model, word_index, config


def encode_review(text, word_index, config):
    words = text_to_word_sequence(text)
    sequence = [1]

    for word in words:
        index = word_index.get(word, 2) + 3

        if index >= config["vocab_size"]:
            index = 2

        sequence.append(index)

    return pad_sequences(
        [sequence],
        maxlen=config["max_length"],
        padding="post",
        truncating="post"
    )


st.title("🎬 Movie Review Sentiment Analysis")

st.write(
    "Enter an English movie review to predict whether "
    "its sentiment is positive or negative."
)

try:
    model, word_index, config = load_model_and_vocabulary()
except Exception as error:
    st.error(
        "Unable to load model files. "
        "Please run train.py first and check the artifacts folder."
    )
    st.stop()

review = st.text_area(
    "Enter your movie review",
    placeholder=(
        "Example: The story was amazing and the actors "
        "gave an excellent performance."
    ),
    height=150
)

if st.button("Analyze Sentiment", type="primary"):
    if not review.strip():
        st.warning("Please enter a movie review.")
    else:
        encoded = encode_review(review, word_index, config)

        probability = float(
            model.predict(encoded, verbose=0)[0][0]
        )

        st.subheader("Prediction Result")

        if probability >= 0.5:
            st.success("Positive sentiment 😊")
        else:
            st.error("Negative sentiment 😞")

        st.metric(
            "Positive sentiment probability",
            f"{probability:.2%}"
        )

        st.progress(probability)

        st.caption(
            "This is a model estimate. It may not correctly "
            "interpret sarcasm, mixed opinions, or unfamiliar words."
        )
