import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.preprocess import clean_text
from .base_model import BaseModel


class BERTModel(BaseModel):
    def __init__(self, model_name="distilbert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2)

    def train(self, X_train, y_train, X_val=None, y_val=None):
        # For simplicity, training code omitted — assume pretrained
        pass

    def predict(self, text):
        text = clean_text(text)
        inputs = self.tokenizer(text, return_tensors="pt",
                                truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
        return "real" if pred == 1 else "fake"
