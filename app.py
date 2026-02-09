from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("means_model.pkl", "rb"))
scaler = pickle.load(open("scaler (1).pkl", "rb"))


@app.route("/")
def home():
    return "K-Means Customer Segmentation API Running"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Expecting input features
        # Example: Age, Annual Income, Spending Score
        age = data["age"]
        income = data["income"]
        spending = data["spending"]

        # Convert to array
        features = np.array([[age, income, spending]])

        # Scale input
        scaled_features = scaler.transform(features)

        # Predict cluster
        cluster = model.predict(scaled_features)

        return jsonify({
            "cluster": int(cluster[0])
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)