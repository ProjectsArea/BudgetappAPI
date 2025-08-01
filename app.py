from flask import Flask, request, jsonify
import numpy as np
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Load models from S3
model = ("random_forest_model.pkl")
nb_model = ("naive_bayes_model.pkl")
encoder = ("encoder.pkl")
scaler = ("scaler.pkl")
expected_features = ["Credit/Debit", "Transaction Type"]
# Prediction endpoint
@app.route("/predict_expense", methods=["POST"])
def predict_expense():
    try:
        # Parse input JSON data
        data = request.get_json()
        print("🔹 Received Request: ", data)

        # Map input month data to match trained model's feature names
        input_data = pd.DataFrame([[
            float(data["person"]),
            float(data["month_1"]),  # Map to "2020"
            float(data["month_2"]),  # Map to "2021"
            float(data["month_3"])   # Map to "2022"
        ]], columns=["person", "2020", "2021", "2022"])  # Match trained model feature names

        print("🔹 Transformed Input Data:\n", input_data)

        # Make a prediction
        predicted_value = model.predict(input_data)[0]
        print(f"✅ Prediction: {predicted_value}")

        # Return JSON response
        return jsonify({"predicted_next_month": round(predicted_value, 2)})

    except Exception as e:
        print("❌ Error processing request: ", str(e))
        return jsonify({"error": str(e)}), 400


@app.route("/predict", methods=["POST"])
def predict_cluster():
    try:
        # Get JSON input
        data = request.get_json()
        amount = float(data["amount"])
        credit_debit = data["credit_debit"]
        transaction_type = data["transaction_type"]

        print("Received Input Data:", data)

        # Ensure correct column names
        expected_features = ["Credit/Debit", "Transaction Type"]
        sample_category_df = pd.DataFrame([[credit_debit, transaction_type]], columns=expected_features)

        # ⚠️ Handle Unknown Categories
        try:
            sample_category_encoded = encoder.transform(sample_category_df)
        except ValueError as e:
            print("⚠️ Warning:", str(e))
            return jsonify({"error": "Unknown category in input. Please retrain encoder with more data."}), 400

        sample_numeric_scaled = scaler.transform(np.array([[amount]]))

        # Combine features
        sample_final = np.hstack((sample_numeric_scaled, sample_category_encoded))

        predicted_cluster = nb_model.predict(sample_final)

        return jsonify({"predicted_cluster": int(predicted_cluster[0])})

    except Exception as e:
        print("Prediction Error:", str(e))
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
