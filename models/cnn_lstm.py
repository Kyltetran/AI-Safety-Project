import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from utils.preprocess import clean_text
from .base_model import BaseModel
import numpy as np


class CNNTextModel(BaseModel):
    def __init__(self, vocab_size=10000, max_len=100):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token="<UNK>")
        self.model = None

    def prepare_data(self, texts, labels=None):
        texts = [clean_text(t) for t in texts]
        seqs = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(seqs, maxlen=self.max_len)
        return padded, np.array(labels) if labels is not None else None

    def build_model(self):
        model = Sequential([
            Embedding(self.vocab_size, 128, input_length=self.max_len),
            Conv1D(64, 5, activation='relu'),
            GlobalMaxPooling1D(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        self.model = model

    def train(self, X_train, y_train, X_val, y_val):
        self.tokenizer.fit_on_texts(X_train)
        X_train, y_train = self.prepare_data(X_train, y_train)
        X_val, y_val = self.prepare_data(X_val, y_val)

        self.build_model()
        self.model.fit(X_train, y_train, validation_data=(
            X_val, y_val), epochs=3, batch_size=32)

    def predict(self, text):
        x, _ = self.prepare_data([text])
        pred = self.model.predict(x)
        return "real" if pred[0][0] > 0.5 else "fake"
