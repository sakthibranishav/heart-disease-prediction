
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print("=" * 50)
print(" CARDIOVASCULAR DISEASE PREDICTION")
print("=" * 50)

# 1. Load dataset
data = pd.read_csv("heart.csv", sep=";")

# 2. Remove duplicate rows
data = data.drop_duplicates()

# 3. Convert age from days to years
data["age_years"] = data["age"] / 365.25

# 4. Remove ID and original age column
X = data.drop(columns=["id", "age", "cardio"])
y = data["cardio"]

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Create and train model
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

print("\nTraining model...")
model.fit(X_train, y_train)

# 7. Make predictions
predictions = model.predict(X_test)

# 8. Evaluate model
accuracy = accuracy_score(y_test, predictions)

print("\nModel training completed!")
print(f"Test accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nEducational model evaluation completed.")
print("This model is not a medical diagnostic tool.")
