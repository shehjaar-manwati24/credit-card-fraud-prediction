# 💳 Credit Card Fraud Detection

This project focuses on identifying fraudulent credit card transactions using machine learning. Built on a highly imbalanced dataset of 284,807 real transactions, the model uses a Random Forest classifier and is evaluated using metrics beyond accuracy to reflect real-world fraud detection priorities.

---

## 📂 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Features**:
  - `Time`: Seconds since the first transaction
  - `V1–V28`: PCA-transformed, anonymized features
  - `Amount`: Transaction amount
  - `Class`: Target (0 = Legit, 1 = Fraud)

---

## 🧠 Project Workflow

### 1. Data Exploration
- Analyzed class imbalance (fraud cases ~0.17%)
- Explored transaction patterns and amount differences between classes

### 2. Preprocessing
- Cleaned dataset, handled missing values
- Split into features (`X`) and labels (`y`)
- Used an 80/20 train-test split

### 3. Model Training
- Trained a `RandomForestClassifier` with tuned hyperparameters:
  - `n_estimators=20`, `max_depth=8`, `min_samples_split=10`, `n_jobs=-1`
- Handled class imbalance without oversampling (baseline model)

### 4. Evaluation
- Evaluated using:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Matthews Correlation Coefficient (MCC)
- Visualized confusion matrix

---

## 📈 Model Performance

| Metric      | Value     |
|-------------|-----------|
| Accuracy    | 99.94%    |
| Precision   | 95.83%    |
| Recall      | 70.41%    |
| F1-Score    | 81.18%    |
| MCC         | 0.8212    |

> 💡 High precision ensures very few false fraud alerts. The recall indicates room for improvement in catching all fraudulent activity.

---

## 🧰 Tech Stack

- Python 3
- Pandas, NumPy
- Scikit-learn (RandomForest, evaluation metrics)
- Matplotlib & Seaborn for data visualization

---

## 🚀 Future Improvements

- Handle class imbalance using SMOTE or undersampling
- Try ensemble models (XGBoost, LightGBM)
- Add model explainability using SHAP
- Deploy as a REST API with FastAPI for real-time fraud alerts

