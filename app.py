from flask import Flask, request, jsonify
import numpy as np
import joblib
import pandas as pd
from flask_cors import CORS
import traceback

app = Flask(__name__)
CORS(app)

# Load models and transformers
try:
    model = joblib.load("random_forest_model.pkl")
    nb_model = joblib.load("naive_bayes_model.pkl")
    encoder = joblib.load("encoder.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    print("❌ Failed to load models or preprocessors:", e)
    traceback.print_exc()

# For predicting future expense
@app.route("/predict_expense", methods=["POST"])
def predict_expense():
    try:
        data = request.get_json()
        print("🔹 Received Request: ", data)

        input_data = pd.DataFrame([[
            float(data["person"]),
            float(data["month_1"]),
            float(data["month_2"]),
            float(data["month_3"])
        ]], columns=["person", "2020", "2021", "2022"])

        print("🔹 Transformed Input Data:\n", input_data)

        predicted_value = model.predict(input_data)[0]
        print(f"✅ Prediction: {predicted_value}")

        return jsonify({"predicted_next_month": round(predicted_value, 2)})

    except Exception as e:
        print("❌ Error in /predict_expense:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

# For clustering prediction
@app.route("/predict", methods=["POST"])
def predict_cluster():
    try:
        data = request.get_json()
        amount = float(data["amount"])
        credit_debit = data["credit_debit"]
        transaction_type = data["transaction_type"]

        print("Received Input Data:", data)

        expected_features = ["Credit/Debit", "Transaction Type"]
        sample_category_df = pd.DataFrame([[credit_debit, transaction_type]], columns=expected_features)

        try:
            sample_category_encoded = encoder.transform(sample_category_df)
        except ValueError as e:
            print("⚠️ Warning:", e)
            return jsonify({"error": "Unknown category in input. Please retrain encoder with more data."}), 400

        sample_numeric_scaled = scaler.transform(np.array([[amount]]))
        sample_final = np.hstack((sample_numeric_scaled, sample_category_encoded))

        predicted_cluster = nb_model.predict(sample_final)

        return jsonify({"predicted_cluster": int(predicted_cluster[0])})

    except Exception as e:
        print("❌ Error in /predict:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

# Optional: only for local dev, not for Gunicorn
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
