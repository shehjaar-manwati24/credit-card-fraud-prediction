from fastapi import FastAPI
import pandas as pd
import joblib
import numpy as np

# Load the trained fraud detection model
model = joblib.load("fraud_detection_model.pkl")

# Load preprocessing artifacts
scaler = joblib.load("scaler.pkl")  # MinMaxScaler object
category_encoder = joblib.load("category_encoder.pkl")  # LabelEncoder object
merchant_stats = joblib.load("merchant_stats.pkl")  # Dictionary with merchant stats

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
        trans_date_trans_time = pd.to_datetime(transaction["trans_date_trans_time"])  # Ensure it's a datetime object

        # Extract time-based features
        day_of_week = trans_date_trans_time.dayofweek  # Monday=0, Sunday=6
        hour = trans_date_trans_time.hour  # Transaction hour
        day_of_month = trans_date_trans_time.day  # Day of the month

        # --- Preprocessing ---

        # Scale 'amt' using the MinMaxScaler
        amt_scaled = scaler.transform(np.array([[amt]]))[0][0]  # Ensure it's a 2D array

        # Encode 'category' using pre-trained LabelEncoder
        if category in category_encoder.classes_:
            category_encoded = category_encoder.transform([category])[0]  # Use `transform` instead of dictionary lookup
        else:
            category_encoded = -1  # Assign -1 if the category is unseen

        # Extract merchant-based features (avg_amt, freq, risk)
        if merchant in merchant_stats:
            merchant_avg_amt = merchant_stats[merchant]["avg_amt"]
            merchant_freq = merchant_stats[merchant]["freq"]
            merchant_risk = merchant_stats[merchant]["risk"]
        else:
            merchant_avg_amt, merchant_freq, merchant_risk = 0, 0, 0  # Defaults for unknown merchants

        # Prepare the final input as an array
        input_features = np.array([[amt_scaled, category_encoded, merchant_avg_amt, merchant_freq, merchant_risk, day_of_week, hour, day_of_month]])

        # --- Model Prediction ---
        prediction = model.predict(input_features)[0]
        probability = model.predict_proba(input_features)[0][1]  # Probability of fraud

        return {
            "fraud_prediction": int(prediction),
            "fraud_probability": probability
        }

    except Exception as e:
        return {"error": str(e)}
