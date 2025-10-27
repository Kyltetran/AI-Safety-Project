# models/cnn_lstm.py
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class CNNLSTMTrainer:
    def __init__(self,
                 vocab_size=20000,
                 max_len=64,
                 embedding_dim=128,
                 model_dir="output/models",
                 tokenizer_path="output/models/tokenizer.pkl"):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.tokenizer = None
        self.model = None
        self.model_dir = model_dir
        self.tokenizer_path = tokenizer_path
        os.makedirs(self.model_dir, exist_ok=True)

    def prepare_tokenizer(self, texts):
        self.tokenizer = Tokenizer(
            num_words=self.vocab_size, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(texts)

    def texts_to_padded(self, texts):
        seqs = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(seqs, maxlen=self.max_len,
                               padding="post", truncating="post")
        return padded

    def build_model(self):
        model = Sequential([
            Embedding(input_dim=self.vocab_size,
                      output_dim=self.embedding_dim, input_length=self.max_len),
            Conv1D(filters=128, kernel_size=5,
                   activation="relu", padding="same"),
            MaxPooling1D(pool_size=2),
            Bidirectional(LSTM(128, return_sequences=False)),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer="adam",
                      loss="binary_crossentropy", metrics=["accuracy"])
        self.model = model
        return model

    def train(self, X_train_texts, y_train, X_val_texts, y_val,
              epochs=10, batch_size=64, checkpoint_name="cnn_lstm_best.h5"):
        # Ensure tokenizer exists
        if self.tokenizer is None:
            self.prepare_tokenizer(X_train_texts)

        X_train = self.texts_to_padded(X_train_texts)
        X_val = self.texts_to_padded(X_val_texts)

        if self.model is None:
            self.build_model()

        checkpoint_path = os.path.join(self.model_dir, checkpoint_name)
        callbacks = [
            ModelCheckpoint(checkpoint_path, monitor="val_loss",
                            save_best_only=True, verbose=1),
            EarlyStopping(monitor="val_loss", patience=3,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=2, verbose=1)
        ]

        history = self.model.fit(X_train, np.array(y_train),
                                 validation_data=(X_val, np.array(y_val)),
                                 epochs=epochs, batch_size=batch_size,
                                 callbacks=callbacks, verbose=2)
        return history

    def evaluate(self, X_texts, y_true):
        X = self.texts_to_padded(X_texts)
        preds_prob = self.model.predict(X).ravel()
        preds = (preds_prob >= 0.5).astype(int)
        return preds, preds_prob

    def save(self, model_name="cnn_lstm_model.h5"):
        path = os.path.join(self.model_dir, model_name)
        self.model.save(path)
        # save tokenizer
        with open(self.tokenizer_path, "wb") as f:
            pickle.dump(self.tokenizer, f)

    def load(self, model_path=None, tokenizer_path=None):
        if model_path is None:
            model_path = os.path.join(self.model_dir, "cnn_lstm_model.h5")
        if tokenizer_path is None:
            tokenizer_path = self.tokenizer_path

        self.model = tf.keras.models.load_model(model_path)
        with open(tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)
