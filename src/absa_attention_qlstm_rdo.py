# =========================================================
# Attention-Guided Quantum LSTM with Red Deer Optimization
# for Aspect-Based Sentiment Analysis
# =========================================================

# =========================================================
# 1. Import Required Libraries
# =========================================================

import pandas as pd
import numpy as np
import re
import random

import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import class_weight
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import MiniBatchKMeans

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout,
    Bidirectional, GlobalAveragePooling1D,
    Concatenate, SpatialDropout1D, BatchNormalization, Layer
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# 2. Load Dataset
# =========================================================

df = pd.read_csv("app_reviews.csv")

# Keep required columns
df = df[['content', 'score']]
df.dropna(inplace=True)

# =========================================================
# 3. Convert Ratings to Sentiment Labels
# =========================================================

def score_to_sentiment(score):
    """Convert rating score into sentiment label."""
    if score >= 4:
        return 1   # Positive
    elif score <= 2:
        return 0   # Negative
    else:
        return None

df["sentiment"] = df["score"].apply(score_to_sentiment)
df.dropna(inplace=True)
df["sentiment"] = df["sentiment"].astype(int)

# =========================================================
# 4. Text Preprocessing
# =========================================================

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    """Clean and normalize text."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df["cleaned_content"] = df["content"].apply(clean_text)

# =========================================================
# 5. Tokenization and Sequence Padding
# =========================================================

X = df["cleaned_content"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

max_words = 10000
max_len = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len)
X_test_pad = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_len)

# =========================================================
# 6. Baseline Model: Standard LSTM
# =========================================================

lstm_model = Sequential([
    Embedding(max_words, 128),
    LSTM(128),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

lstm_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history_lstm = lstm_model.fit(
    X_train_pad,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1
)

y_pred = (lstm_model.predict(X_test_pad) > 0.5).astype(int)
lstm_acc = accuracy_score(y_test, y_pred)

print("LSTM Accuracy:", lstm_acc)

# =========================================================
# 7. Quantum-Inspired Embedding Layer
# =========================================================

class QuantumInspiredEmbedding(Layer):
    """
    Custom embedding layer that creates
    quantum-inspired magnitude and phase features.
    """

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.real_embedding = Embedding(vocab_size, embed_dim)
        self.imag_embedding = Embedding(vocab_size, embed_dim)

    def call(self, inputs):

        real_part = self.real_embedding(inputs)
        imag_part = self.imag_embedding(inputs)

        magnitude = tf.sqrt(tf.square(real_part) + tf.square(imag_part))
        phase = tf.math.atan2(imag_part, real_part)

        return tf.concat([magnitude, phase], axis=-1)

# =========================================================
# 8. Quantum LSTM Model
# =========================================================

qlstm_model = Sequential([
    QuantumInspiredEmbedding(max_words, 64),
    LSTM(128),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

qlstm_model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history_qlstm = qlstm_model.fit(
    X_train_pad,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1
)

y_pred_qlstm = (qlstm_model.predict(X_test_pad) > 0.5).astype(int)
qlstm_acc = accuracy_score(y_test, y_pred_qlstm)

print("QLSTM Accuracy:", qlstm_acc)

# =========================================================
# 9. Red Deer Optimization (RDO)
# Hyperparameter tuning
# =========================================================

def fitness_function(units, dropout, lr):
    """Evaluate model fitness using validation loss."""

    model = Sequential([
        QuantumInspiredEmbedding(max_words, 64),
        LSTM(units),
        Dropout(dropout),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train_pad,
        y_train,
        epochs=3,
        batch_size=128,
        validation_split=0.05,
        verbose=0
    )

    return -min(history.history["val_loss"])

# Random initialization
population = [{
    "units": random.choice([64, 128]),
    "dropout": round(random.uniform(0.35, 0.55), 2),
    "lr": random.choice([0.001, 0.0005])
} for _ in range(6)]

best_params = None
best_fitness = -np.inf

for deer in population:

    fitness = fitness_function(
        deer["units"],
        deer["dropout"],
        deer["lr"]
    )

    if fitness > best_fitness:
        best_fitness = fitness
        best_params = deer

print("Best Parameters:", best_params)

# =========================================================
# 10. Attention Layer
# =========================================================

class AttentionLayer(Layer):
    """Custom attention mechanism."""

    def build(self, input_shape):

        self.W = self.add_weight(
            shape=(input_shape[-1], input_shape[-1]),
            initializer="glorot_uniform"
        )

        self.b = self.add_weight(
            shape=(input_shape[-1],),
            initializer="zeros"
        )

        self.u = self.add_weight(
            shape=(input_shape[-1],),
            initializer="glorot_uniform"
        )

    def call(self, x):

        uit = tf.tanh(tf.matmul(x, self.W) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        a = tf.nn.softmax(ait)

        return tf.reduce_sum(x * tf.expand_dims(a, -1), axis=1)

# =========================================================
# 11. Attention-Based QLSTM Model
# =========================================================

def build_attention_model():

    inputs = tf.keras.Input(shape=(max_len,))

    x = QuantumInspiredEmbedding(max_words, 100)(inputs)
    x = SpatialDropout1D(0.3)(x)

    lstm_out = Bidirectional(LSTM(128, return_sequences=True))(x)

    att = AttentionLayer()(lstm_out)
    avg = GlobalAveragePooling1D()(lstm_out)

    merged = Concatenate()([att, avg])

    x = Dense(128, activation="relu")(merged)
    x = Dropout(0.4)(x)
    x = BatchNormalization()(x)

    outputs = Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs)

attention_model = build_attention_model()

attention_model.compile(
    optimizer=Adam(0.0005),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# =========================================================
# 12. Aspect Extraction
# =========================================================

aspects = {
    "price": ["price","cost","cheap","expensive"],
    "quality": ["quality","durable","material"],
    "delivery": ["delivery","shipping","late"],
    "service": ["service","support"],
    "performance": ["performance","speed"]
}

def extract_aspect(text):

    for aspect, keywords in aspects.items():
        for word in keywords:
            if word in text:
                return aspect

    return "general"

df["aspect"] = df["cleaned_content"].apply(extract_aspect)

# =========================================================
# 13. Visualization
# =========================================================

plt.figure(figsize=(6,4))
df["sentiment"].value_counts().plot(kind="bar")
plt.title("Sentiment Distribution")
plt.show()
