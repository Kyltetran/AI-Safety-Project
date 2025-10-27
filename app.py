from flask import Flask, request, jsonify, render_template
from datetime import datetime
import os
import csv
from models.logistic_regression import LogisticRegressionModel

app = Flask(__name__, static_folder="frontend", template_folder="frontend")

# Load model
model = LogisticRegressionModel()
model.load("output/models/logreg_model.pkl")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    result = model.predict(text)

    # log user input
    os.makedirs("data/raw", exist_ok=True)
    with open("data/raw/user_inputs.csv", "a") as f:
        csv.writer(f).writerow([datetime.now(), text, result])

    return jsonify({"prediction": result})


if __name__ == "__main__":
    app.run(debug=True)
