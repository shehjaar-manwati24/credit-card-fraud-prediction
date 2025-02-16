from fastapi import FastAPI
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

# Load pre-trained model and preprocessing artifacts
model = joblib.load("fraud_detection_model.pkl")
scaler = joblib.load("scaler.pkl")
category_encoder = joblib.load("category_encoder.pkl")
merchant_stats = joblib.load("merchant_stats.pkl")

# Initialize FastAPI
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running!"}

@app.post("/predict/")
def predict(transaction: dict):
    try:
        # Extract raw input values
        amt = transaction["amt"]
        merchant = transaction["merchant"]
        category = transaction["category"]
        trans_time = transaction["trans_date_trans_time"]  # Format: "YYYY-MM-DD HH:MM:SS"

        # --- Preprocessing ---

        # Scale 'amt' using pre-trained scaler
        amt_scaled = scaler.transform([[amt]])[0][0]

        # Encode 'category' using pre-trained LabelEncoder
        try:
            category_encoded = category_encoder.transform([category])[0]
        except ValueError:  # If the category is unseen
            category_encoded = -1

        # Extract merchant-based features
        merchant_avg_amt = merchant_stats.get(merchant, {}).get("merchant_avg_amt", 0)
        merchant_freq = merchant_stats.get(merchant, {}).get("merchant_freq", 0)
        merchant_risk = merchant_stats.get(merchant, {}).get("merchant_risk", 0)

        # Extract date-related features
        trans_date = datetime.strptime(trans_time, "%Y-%m-%d %H:%M:%S")
        day_of_week = trans_date.weekday()  # Monday = 0, Sunday = 6
        hour = trans_date.hour  # Hour of transaction
        day_of_month = trans_date.day  # Day of the month

        # Prepare the final input array
        input_features = np.array([[amt_scaled, merchant_freq, merchant_avg_amt, merchant_risk,
                                    category_encoded, day_of_week, hour, day_of_month]])

        # --- Model Prediction ---
        prediction = model.predict(input_features)[0]
        
        # Try getting probability, if available
        fraud_probability = None
        if hasattr(model, "predict_proba"):
            fraud_probability = model.predict_proba(input_features)[0][1]

        return {
            "fraud_prediction": int(prediction),
            "fraud_probability": fraud_probability
        }

    except Exception as e:
        return {"error": str(e)}
