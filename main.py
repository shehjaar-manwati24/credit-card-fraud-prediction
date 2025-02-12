import pandas as pd

# Load dataset (Replace with your actual file path)
df = pd.read_csv("credit_card_transactions.csv")

# Show first 5 rows
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Convert transaction date column to datetime format
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"], format="%d-%m-%Y %H:%M")

# Extract features from the date
df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek  # Monday=0, Sunday=6
df["hour"] = df["trans_date_trans_time"].dt.hour  # Extract transaction hour
df["day_of_month"] = df["trans_date_trans_time"].dt.day  # Extract day of the month

# Show updated data
print(df)

from sklearn.preprocessing import MinMaxScaler

# Initialize scaler
scaler = MinMaxScaler()

# Apply Min-Max Scaling
df["amt_scaled"] = scaler.fit_transform(df[["amt"]])

# Show before & after transformation
print(df[["amt", "amt_scaled"]].head())

from sklearn.preprocessing import LabelEncoder

# Initialize label encoders
merchant_encoder = LabelEncoder()
category_encoder = LabelEncoder()

# Apply encoding
df["merchant_encoded"] = merchant_encoder.fit_transform(df["merchant"])
df["category_encoded"] = category_encoder.fit_transform(df["category"])

# Show results
print(df[["merchant", "merchant_encoded", "category", "category_encoded"]].head())

from sklearn.model_selection import train_test_split

# Selecting features (excluding raw categorical columns)
features = ["amt_scaled", "merchant_encoded", "category_encoded", "day_of_week", "hour", "day_of_month"]
target = "is_fraud"  # We will first build a fraud detection model

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Show dataset sizes
print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

from sklearn.model_selection import train_test_split

# Selecting features (excluding raw categorical columns)
features = ["amt_scaled", "merchant_encoded", "category_encoded", "day_of_week", "hour", "day_of_month"]
target = "is_fraud"  # We will first build a fraud detection model

# Compute Merchant Frequency (How often does a user shop here?)
df["merchant_freq"] = df.groupby("merchant")["merchant"].transform("count")

# Compute Merchant Average Transaction Amount
df["merchant_avg_amt"] = df.groupby("merchant")["amt"].transform("mean")

# Compute Merchant Fraud Rate (Optional for fraud detection)
merchant_fraud_rates = df.groupby("merchant")["is_fraud"].mean()
df["merchant_risk"] = df["merchant"].map(merchant_fraud_rates)

# Show results
print(df[["merchant", "merchant_freq", "merchant_avg_amt", "merchant_risk"]].head())

from sklearn.model_selection import train_test_split

# Define features for model training
features = [
    "amt_scaled", "merchant_freq", "merchant_avg_amt", "merchant_risk",  # Merchant-based financial insights
    "category_encoded",  # Encoded category
    "day_of_week", "hour", "day_of_month"  # Time-based features
]

target = "is_fraud"  # First, we'll train a fraud detection model

# Split dataset into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Show dataset sizes
print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
 
import imblearn
from imblearn.under_sampling import RandomUnderSampler


# Initialize Undersampler
undersampler = RandomUnderSampler(sampling_strategy=0.5, random_state=42)  # Reduce non-fraud cases

# Apply undersampling
X_train_under, y_train_under = undersampler.fit_resample(X_train, y_train)

# Check new class distribution
print("Original class distribution:", y_train.value_counts().to_dict())
print("After Undersampling:", y_train_under.value_counts().to_dict())

# Train a new model on undersampled data
model_under = RandomForestClassifier(n_estimators=100, random_state=42)
model_under.fit(X_train_under, y_train_under)

# Evaluate recall
recall_under = recall_score(y_test, model_under.predict(X_test))
print(f"Recall after Undersampling: {recall_under:.4f}")

import joblib

# Save the best model
joblib.dump(model_under, "fraud_detection_model.pkl")

print("Model saved successfully!")









