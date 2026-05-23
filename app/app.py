from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model + features
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")

@app.route('/')
def home():
    return "Breast Cancer Prediction API Running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Convert input to DataFrame
    input_data = pd.DataFrame([data])

    # Ensure correct feature order
    input_data = input_data[features]

    prediction = model.predict(input_data)

    return jsonify({
        "prediction": int(prediction[0])
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)