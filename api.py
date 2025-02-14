from fastapi import FastAPI
import pandas as pd
import joblib

# Load the trained fraud detection model
model = joblib.load("fraud_detection_model.pkl")

# Initialize FastAPI
app = FastAPI()

@app.get("/")
def home():
    
    return {"message": "Fraud Detection API is running!"}

@app.post("/predict/")
def predict(transaction: dict):
    """
    Accepts transaction data as JSON and returns fraud prediction.
    """
    df = pd.DataFrame([transaction])  # Convert input data to DataFrame
    prediction = model.predict(df)[0]  # Predict fraud (0 = legit, 1 = fraud)
    
    return {"fraud_prediction": int(prediction)}

# Run the API with: uvicorn api:app --reload
