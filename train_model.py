"""
train_model.py
End-to-End Machine Learning Model Deployment - Heart Disease Prediction
Task 1: Data Understanding and Preprocessing
Task 2: Model Development
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ----------------------------------------------------------------------
# Task 1: Data Understanding and Preprocessing
# ----------------------------------------------------------------------

# 1. Load the dataset using Pandas
df = pd.read_csv("heart.csv")

# 2. Display the first five records
print("First 5 records:")
print(df.head())

# 3. Identify numerical features and target variable
target_variable = "target"
numerical_features = [col for col in df.columns if col != target_variable]

print("\nNumerical Features:", numerical_features)
print("Target Variable:", target_variable)

# 4. Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# 5. Split the dataset into 80% training and 20% testing
X = df[numerical_features]
y = df[target_variable]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# ----------------------------------------------------------------------
# Task 2: Model Development
# ----------------------------------------------------------------------

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model using Joblib
joblib.dump(model, "model.pkl")
# Also persist the exact feature order used for training, so the API
# can build correctly-ordered input for prediction.
joblib.dump(numerical_features, "feature_names.pkl")

print("\nModel saved as model.pkl")
print("Feature order saved as feature_names.pkl")
