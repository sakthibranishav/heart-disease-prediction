
import base64
import gzip
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="CardioPredict | ML Project",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent


# =====================================================
# 2. LOAD TRAINED MODEL
# =====================================================
@st.cache_resource
def load_model():
    model_path = BASE_DIR / "heart_model_compressed.pkl.gz"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path.name}. "
            "Upload heart_model_compressed.pkl.gz "
            "to the same folder as app.py."
        )

    with gzip.open(model_path, "rb") as file:
        saved = joblib.load(file)

    return saved["model"], saved["features"]


try:
    model, features = load_model()
except Exception as error:
    st.error(
        "Could not load the trained model. "
        "Check that heart_model_compressed.pkl.gz "
        "is uploaded beside app.py."
    )
    st.exception(error)
    st.stop()


# =====================================================
# 3. HEART BACKGROUND
# =====================================================
def set_background(image_file):
    if not image_file.exists():
        st.warning(
            "Heart background image not found. "
            "Place heart_background.png beside app.py."
        )
        return

    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image:
                linear-gradient(
                    rgba(8, 22, 40, 0.78),
                    rgba(8, 22, 40, 0.78)
                ),
                url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        [data-testid="stSidebar"] {{
            background-color: rgba(7, 20, 36, 0.96);
        }}

        [data-testid="stHeader"] {{
            background-color: transparent;
        }}

        h1, h2, h3, h4, p, label,
        [data-testid="stMarkdownContainer"] {{
            color: #F5F9FF;
        }}

        div[data-testid="stMetric"] {{
            background-color: rgba(12, 35, 55, 0.78);
            padding: 16px;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.15);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


set_background(BASE_DIR / "heart_background.png")


# =====================================================
# 4. SIDEBAR NAVIGATION
# =====================================================
st.sidebar.title("❤️ CardioPredict")
st.sidebar.caption("Cardiovascular ML Project")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Prediction",
        "Model Dashboard",
        "About the Project",
    ],
)

st.sidebar.divider()
st.sidebar.caption(
    "Built using Python, Pandas, Scikit-learn and Streamlit"
)


# =====================================================
# 5. PREDICTION PAGE
# =====================================================
if page == "Prediction":

    st.title("❤️ Cardiovascular Disease Prediction")

    st.write(
        "Enter the details below to obtain a classification "
        "from the trained Random Forest model."
    )

    st.warning(
        "Educational demonstration only. This model has not "
        "been clinically validated and must not be used for "
        "medical diagnosis or treatment decisions."
    )

    st.subheader("Enter the details")

    with st.form("prediction_form"):

        left, right = st.columns(2)

        with left:
            st.markdown("#### Personal details")

            age = st.number_input(
                "Age (years)",
                min_value=1,
                max_value=100,
                value=45,
            )

            gender = st.selectbox(
                "Gender (dataset code)",
                options=[1, 2],
                help="Uses the numeric codes in the original dataset.",
            )

            height = st.number_input(
                "Height (cm)",
                min_value=100,
                max_value=250,
                value=165,
            )

            weight = st.number_input(
                "Weight (kg)",
                min_value=25.0,
                max_value=300.0,
                value=65.0,
                step=0.5,
            )

        with right:
            st.markdown("#### Health-related inputs")

            ap_hi = st.number_input(
                "Systolic blood pressure",
                min_value=70,
                max_value=250,
                value=120,
            )

            ap_lo = st.number_input(
                "Diastolic blood pressure",
                min_value=40,
                max_value=150,
                value=80,
            )

            cholesterol = st.selectbox(
                "Cholesterol (dataset level)",
                options=[1, 2, 3],
                format_func=lambda x: {
                    1: "1 - Normal",
                    2: "2 - Above normal",
                    3: "3 - Well above normal",
                }[x],
            )

            gluc = st.selectbox(
                "Glucose (dataset level)",
                options=[1, 2, 3],
                format_func=lambda x: {
                    1: "1 - Normal",
                    2: "2 - Above normal",
                    3: "3 - Well above normal",
                }[x],
            )

        st.markdown("#### Lifestyle inputs")

        col1, col2, col3 = st.columns(3)

        with col1:
            smoke = st.selectbox("Smoking", ["No", "Yes"])

        with col2:
            alco = st.selectbox(
                "Alcohol consumption",
                ["No", "Yes"],
            )

        with col3:
            active = st.selectbox(
                "Physically active",
                ["Yes", "No"],
            )

        submitted = st.form_submit_button(
            "Generate Prediction",
            type="primary",
            use_container_width=True,
        )

    if submitted:

        if ap_hi <= ap_lo:
            st.error(
                "Please check the blood pressure values. "
                "For this project's input validation, systolic "
                "pressure must be greater than diastolic pressure."
            )

        else:
            input_values = {
                "gender": gender,
                "height": height,
                "weight": weight,
                "ap_hi": ap_hi,
                "ap_lo": ap_lo,
                "cholesterol": cholesterol,
                "gluc": gluc,
                "smoke": 1 if smoke == "Yes" else 0,
                "alco": 1 if alco == "Yes" else 0,
                "active": 1 if active == "Yes" else 0,
                "age_years": age,
            }

            input_df = pd.DataFrame([input_values])

            try:
                # Match the exact feature order used during training.
                input_df = input_df[features]
                prediction = model.predict(input_df)[0]

                st.divider()
                st.subheader("Prediction Result")

                if prediction == 1:
                    st.error(
                        "Model output: Class 1",
                        icon="⚠️",
                    )
                else:
                    st.success(
                        "Model output: Class 0",
                        icon="✅",
                    )

                st.caption(
                    "Class 0 and Class 1 are labels from the dataset. "
                    "This result is not a diagnosis or a reliable "
                    "assessment of an individual's actual medical risk."
                )

            except Exception as error:
                st.error("Could not generate a prediction.")
                st.exception(error)


# =====================================================
# 6. MODEL DASHBOARD
# =====================================================
elif page == "Model Dashboard":

    st.title("📊 Model Evaluation Dashboard")

    st.write(
        "Explore the performance of the machine-learning models "
        "and the dataset used in this project."
    )

    st.info(
        "All reported metrics describe this educational dataset "
        "and do not establish clinical usefulness."
    )

    # -------------------------------------------------
    # Model comparison
    # -------------------------------------------------
    st.header("1. Model Comparison")

    results_path = BASE_DIR / "model_comparison_results.csv"

    if results_path.exists():

        try:
            results = pd.read_csv(results_path)

            if results.empty:
                st.warning("The model comparison file is empty.")

            else:
                st.dataframe(
                    results,
                    use_container_width=True,
                    hide_index=True,
                )

                if (
                    "Model" in results.columns
                    and "Accuracy (%)" in results.columns
                ):
                    st.subheader("Accuracy Comparison")

                    accuracy_chart = results.set_index("Model")[
                        ["Accuracy (%)"]
                    ]

                    st.bar_chart(accuracy_chart)

                    best_model = results.loc[
                        results["Accuracy (%)"].idxmax()
                    ]

                    st.success(
                        f"Highest accuracy in this comparison: "
                        f"{best_model['Model']} "
                        f"({best_model['Accuracy (%)']:.2f}%)"
                    )

                metric_columns = [
                    column
                    for column in [
                        "Precision (%)",
                        "Recall (%)",
                        "F1 Score (%)",
                    ]
                    if column in results.columns
                ]

                if (
                    "Model" in results.columns
                    and metric_columns
                ):
                    st.subheader(
                        "Precision, Recall and F1-Score"
                    )

                    metric_chart = results.set_index("Model")[
                        metric_columns
                    ]

                    st.bar_chart(metric_chart)

        except Exception as error:
            st.warning("Could not read the model comparison file.")
            st.exception(error)

    else:
        st.warning(
            "Model comparison file not found. "
            "Include model_comparison_results.csv in your project."
        )

    # -------------------------------------------------
    # Dataset overview
    # -------------------------------------------------
    st.header("2. Dataset Overview")

    data_path = BASE_DIR / "heart_cleaned.csv"
    data = None

    if data_path.exists():

        try:
            data = pd.read_csv(data_path)

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Records", f"{len(data):,}")

            if "cardio" in data.columns:
                col2.metric(
                    "Class 0 Records",
                    f"{int((data['cardio'] == 0).sum()):,}",
                )

                col3.metric(
                    "Class 1 Records",
                    f"{int((data['cardio'] == 1).sum()):,}",
                )

                st.subheader("Target Class Distribution")

                distribution = data["cardio"].value_counts().sort_index()
                distribution.index = [
                    f"Class {value}" for value in distribution.index
                ]

                st.bar_chart(distribution)

            st.write(
                "Missing values:",
                int(data.isnull().sum().sum()),
            )

        except Exception as error:
            st.warning("Could not read the cleaned dataset.")
            st.exception(error)

    else:
        st.warning(
            "heart_cleaned.csv was not found. "
            "Upload it beside app.py to display the dataset overview."
        )

    # -------------------------------------------------
    # Confusion matrix
    # -------------------------------------------------
    st.header("3. Confusion Matrix")

    st.write(
        "The confusion matrix shows actual versus predicted "
        "classes for the saved Random Forest model."
    )

    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        from sklearn.model_selection import train_test_split

        if data is not None:

            required_columns = features + ["cardio"]
            missing_columns = [
                column
                for column in required_columns
                if column not in data.columns
            ]

            if missing_columns:
                st.warning(
                    "Dataset is missing required columns: "
                    + ", ".join(missing_columns)
                )

            else:
                X = data[features]
                y = data["cardio"]

                _, X_test, _, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=42,
                    stratify=y,
                )

                predictions = model.predict(X_test)

                cm = confusion_matrix(
                    y_test,
                    predictions,
                    labels=[0, 1],
                )

                fig, ax = plt.subplots(figsize=(5, 4))

                image = ax.imshow(
                    cm,
                    interpolation="nearest",
                )

                ax.set_title("Random Forest Confusion Matrix")
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
                            j,
                            i,
                            f"{cm[i, j]:,}",
                            ha="center",
                            va="center",
                        )

                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                st.caption(
                    "Rows represent actual labels; columns represent "
                    "predicted labels. Diagonal cells are correct "
                    "classifications."
                )

    except Exception as error:
        st.warning("Could not generate the confusion matrix.")
        st.caption(str(error))

    # -------------------------------------------------
    # Feature importance
    # -------------------------------------------------
    st.header("4. Feature Importance")

    try:
        importance = pd.DataFrame(
            {
                "Feature": features,
                "Importance": model.feature_importances_,
            }
        ).sort_values(
            "Importance",
            ascending=False,
        )

        st.bar_chart(
            importance.set_index("Feature")[["Importance"]]
        )

        st.caption(
            "Feature importance describes how the Random Forest "
            "uses features. It does not prove that a feature "
            "causes cardiovascular disease."
        )

    except (AttributeError, ValueError) as error:
        st.warning(
            "Feature importance is unavailable for this model."
        )
        st.caption(str(error))


# =====================================================
# 7. ABOUT THE PROJECT
# =====================================================
elif page == "About the Project":

    st.title("ℹ️ About CardioPredict")

    st.write(
        "CardioPredict is an educational machine-learning project "
        "that classifies records in a cardiovascular dataset."
    )

    st.header("Project Workflow")

    st.markdown("""
    1. **Data collection:** Load the cardiovascular dataset.
    2. **Data preprocessing:** Inspect data quality and apply
       documented cleaning rules.
    3. **Exploratory analysis:** Examine distributions and patterns.
    4. **Model training:** Train classification algorithms.
    5. **Model evaluation:** Compare models on a held-out test set.
    6. **Web application:** Display predictions and evaluation charts.
    """)

    st.header("Technologies Used")

    st.markdown("""
    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - Streamlit
    - Matplotlib
    - Joblib
    """)

    st.header("Current Model Result")

    st.metric(
        "Random Forest test accuracy",
        "73.26%",
    )

    st.caption(
        "Measured on the project's cleaned dataset using a held-out "
        "test split. This is a dataset evaluation result, not evidence "
        "of clinical validity."
    )

    st.header("Project Limitations")

    st.markdown("""
    - The model is an educational prototype.
    - Data-cleaning choices may affect performance and generalizability.
    - Accuracy alone does not establish model reliability.
    - The model must not be used to diagnose, rule out, or treat disease.
    """)
