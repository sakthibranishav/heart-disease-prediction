
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("=" * 50)
print("CARDIOVASCULAR PROJECT - FINAL CHECK")
print("=" * 50)

# Check required files
required_files = [
    "heart.csv",
    "heart_cleaned.csv",
    "heart_model.pkl",
    "app.py",
    "dashboard.py",
    "data_cleaning.py",
    "random_forest.py"
]

print("\n1. FILE CHECK")

for file in required_files:
    if os.path.exists(file):
        print(f"[OK] {file}")
    else:
        print(f"[MISSING] {file}")

# Load cleaned dataset
print("\n2. DATASET CHECK")

data = pd.read_csv("heart_cleaned.csv")

print("Dataset shape:", data.shape)
print("Missing values:", int(data.isnull().sum().sum()))
print("Duplicate rows:", int(data.duplicated().sum()))
print("\nTarget distribution:")
print(data["cardio"].value_counts().sort_index())

# Load saved model
print("\n3. MODEL CHECK")

saved_data = joblib.load("heart_model.pkl")
model = saved_data["model"]
features = saved_data["features"]

missing_features = [
    feature for feature in features
    if feature not in data.columns
]

if missing_features:
    print("ERROR: Missing model features:", missing_features)
else:
    print("All model features are present.")

    X = data[features]
    y = data["cardio"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    predictions = model.predict(X_test)

    print("\n4. MODEL EVALUATION")
    print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

print("\nFINAL CHECK COMPLETED")
