
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("=" * 55)
print(" ORIGINAL VS CLEANED DATASET COMPARISON")
print("=" * 55)

# Load both datasets
original = pd.read_csv("heart.csv", sep=";")
cleaned = pd.read_csv("heart_cleaned.csv")

# Prepare original dataset
original["age_years"] = original["age"] / 365.25

X_original = original.drop(columns=["id", "age", "cardio"])
y_original = original["cardio"]

# Prepare cleaned dataset
X_cleaned = cleaned.drop(columns=["id", "age", "cardio"])
y_cleaned = cleaned["cardio"]

# Use the same random seed and test proportion
X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(
    X_original, y_original,
    test_size=0.2, random_state=42, stratify=y_original
)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cleaned, y_cleaned,
    test_size=0.2, random_state=42, stratify=y_cleaned
)

def train_and_evaluate(X_train, X_test, y_train, y_test, name):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"\n{name}")
    print("-" * 40)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, predictions))

    return accuracy

original_accuracy = train_and_evaluate(
    X_train_o, X_test_o, y_train_o, y_test_o,
    "ORIGINAL DATASET"
)

cleaned_accuracy = train_and_evaluate(
    X_train_c, X_test_c, y_train_c, y_test_c,
    "CLEANED DATASET"
)

print("\nFINAL COMPARISON")
print(f"Original accuracy: {original_accuracy * 100:.2f}%")
print(f"Cleaned accuracy: {cleaned_accuracy * 100:.2f}%")
print("Compare recall and precision too, not accuracy alone.")
