
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Heart Disease Model Dashboard",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Cardiovascular Disease Prediction")
st.subheader("Machine Learning Evaluation Dashboard")

st.info(
    "Educational project only. This model is not a medical "
    "diagnostic tool."
)

# Load dataset and saved model
data = pd.read_csv("heart_cleaned.csv")
saved_data = joblib.load("heart_model.pkl")

model = saved_data["model"]
features = saved_data["features"]

# Prepare the same dataset features
X = data[features]
y = data["cardio"]

# Reproduce the model's test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Evaluate saved model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(
    y_test, predictions, zero_division=0
)
recall = recall_score(
    y_test, predictions, zero_division=0
)
f1 = f1_score(
    y_test, predictions, zero_division=0
)

# Performance metrics
st.header("1. Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{accuracy * 100:.2f}%")
col2.metric("Precision (Class 1)", f"{precision * 100:.2f}%")
col3.metric("Recall (Class 1)", f"{recall * 100:.2f}%")
col4.metric("F1-Score (Class 1)", f"{f1 * 100:.2f}%")

st.caption(
    "Metrics are calculated on the held-out test split. "
    "Precision, recall and F1-score refer to class 1."
)

# Confusion matrix
st.header("2. Confusion Matrix")

cm = confusion_matrix(y_test, predictions, labels=[0, 1])

fig, ax = plt.subplots()
image = ax.imshow(cm, interpolation="nearest")
ax.set_title("Confusion Matrix")
fig.colorbar(image, ax=ax)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Class 0", "Class 1"])
ax.set_yticklabels(["Class 0", "Class 1"])
ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")

for i in range(2):
    for j in range(2):
        ax.text(
            j, i, str(cm[i, j]),
            ha="center", va="center"
        )

st.pyplot(fig)
plt.close(fig)

st.write(
    "Rows show actual labels; columns show predicted labels. "
    "The diagonal cells are correct predictions."
)

# Feature importance
st.header("3. Feature Importance")

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

fig2, ax2 = plt.subplots()
ax2.barh(
    importance["Feature"][::-1],
    importance["Importance"][::-1]
)
ax2.set_xlabel("Importance")
ax2.set_title("Random Forest Feature Importance")
fig2.tight_layout()

st.pyplot(fig2)
plt.close(fig2)

st.caption(
    "Feature importance shows how the model uses features "
    "for its predictions; it does not establish causation."
)

# Dataset overview
st.header("4. Dataset Overview")

total_records = len(data)
class_0 = int((data["cardio"] == 0).sum())
class_1 = int((data["cardio"] == 1).sum())

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", f"{total_records:,}")
col2.metric("Class 0 Records", f"{class_0:,}")
col3.metric("Class 1 Records", f"{class_1:,}")

st.write("Class distribution")

distribution = pd.DataFrame({
    "Class": ["Class 0", "Class 1"],
    "Records": [class_0, class_1]
}).set_index("Class")

st.bar_chart(distribution)

# Classification report
with st.expander("View detailed classification report"):
    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )
    st.dataframe(pd.DataFrame(report).transpose())

st.success("Dashboard loaded successfully!")

# ---------------- MODEL COMPARISON ----------------

st.header("5. Model Comparison")

try:
    comparison = pd.read_csv("model_comparison_results.csv")

    st.write(
        "Comparison of three machine-learning models "
        "evaluated on the same test dataset."
    )

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Accuracy Comparison")

    chart_data = comparison.set_index("Model")[["Accuracy (%)"]]
    st.bar_chart(chart_data)

    best_model = comparison.loc[
        comparison["Accuracy (%)"].idxmax()
    ]

    st.success(
        f"Best accuracy in this comparison: "
        f"{best_model['Model']} "
        f"({best_model['Accuracy (%)']:.2f}%)"
    )

except FileNotFoundError:
    st.warning(
        "Run model_comparison.py first to generate "
        "model_comparison_results.csv."
    )
