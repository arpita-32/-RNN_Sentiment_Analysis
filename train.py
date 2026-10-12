
import os
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -------------------------------------------------
# 1. Configuration
# -------------------------------------------------
VOCAB_SIZE = 10000
MAX_LENGTH = 200
EMBEDDING_DIM = 32
BATCH_SIZE = 128
EPOCHS = 5

np.random.seed(42)
tf.random.set_seed(42)

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# -------------------------------------------------
# 2. Load IMDB dataset
# -------------------------------------------------
print("Loading IMDB dataset...")

(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=VOCAB_SIZE
)

print("Training reviews:", len(X_train))
print("Testing reviews:", len(X_test))
print("First encoded review:", X_train[0][:20])
print("First sentiment:", y_train[0])

# -------------------------------------------------
# 3. Pad sequences to a fixed length
# -------------------------------------------------
X_train = pad_sequences(
    X_train,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

X_test = pad_sequences(
    X_test,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

print("Training input shape:", X_train.shape)
print("Testing input shape:", X_test.shape)

# -------------------------------------------------
# 4. Build Simple RNN model
# -------------------------------------------------
model = Sequential([
    Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBEDDING_DIM,
        mask_zero=True
    ),
    SimpleRNN(64),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

# -------------------------------------------------
# 5. Compile the model
# -------------------------------------------------
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------------------------------
# 6. Train the model
# -------------------------------------------------
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping],
    verbose=1
)

# -------------------------------------------------
# 7. Evaluate on the unseen test set
# -------------------------------------------------
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

probabilities = model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=0
).ravel()

predictions = (probabilities >= 0.5).astype(int)

print("\nTest loss:", round(float(test_loss), 4))
print("Test accuracy:", round(float(test_accuracy), 4))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["Negative", "Positive"]
    )
)

# -------------------------------------------------
# 8. Save the trained model
# -------------------------------------------------
model.save(
    os.path.join(ARTIFACT_DIR, "sentiment_rnn.keras")
)

# -------------------------------------------------
# 9. Save IMDB vocabulary for raw-text prediction
# -------------------------------------------------
word_index = imdb.get_word_index()

with open(
    os.path.join(ARTIFACT_DIR, "word_index.json"),
    "w",
    encoding="utf-8"
) as file:
    json.dump(word_index, file)

# Save configuration for use by the app
config = {
    "vocab_size": VOCAB_SIZE,
    "max_length": MAX_LENGTH,
    "embedding_dim": EMBEDDING_DIM
}

with open(
    os.path.join(ARTIFACT_DIR, "config.json"),
    "w",
    encoding="utf-8"
) as file:
    json.dump(config, file)

print("\nTraining complete!")
print("Saved model and vocabulary in:", ARTIFACT_DIR)
