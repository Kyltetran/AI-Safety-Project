import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from utils.preprocess import clean_text
from .base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', LogisticRegression(max_iter=200))
        ])

    def train(self, X_train, y_train, X_val, y_val):
        X_train = X_train.apply(clean_text)
        X_val = X_val.apply(clean_text)
        self.pipeline.fit(X_train, y_train)
        preds = self.pipeline.predict(X_val)
        print(classification_report(y_val, preds))

    def predict(self, text):
        text = clean_text(text)
        return self.pipeline.predict([text])[0]

    def save(self, path="output/models/logreg_model.pkl"):
        joblib.dump(self.pipeline, path)

    def load(self, path="output/models/logreg_model.pkl"):
        self.pipeline = joblib.load(path)
