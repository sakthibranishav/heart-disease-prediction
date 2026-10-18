# Cardiovascular Disease Prediction Using Machine Learning

## Project Overview
This project uses machine learning to predict cardiovascular disease class based on health-related data. It includes data preprocessing, exploratory data analysis, model comparison, and a Streamlit web application for demonstrating predictions.

**Note:** This project is for educational purposes only and is not a medical diagnostic tool.

## Objectives
- Clean and preprocess the cardiovascular dataset.
- Explore health-related features and their relationships.
- Train and compare multiple machine learning algorithms.
- Evaluate model performance using classification metrics.
- Develop an interactive web application using Streamlit.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit

## Machine Learning Algorithms
The following algorithms were compared:
1. Logistic Regression
2. Decision Tree
3. Random Forest

## Model Results

| Algorithm | Accuracy |
|---|---:|
| Logistic Regression | 72.33% |
| Decision Tree | 72.25% |
| Random Forest | 73.26% |

Random Forest achieved the highest accuracy among the three models tested, at 73.26% on the selected test dataset.

## Dataset
The project uses a cardiovascular disease dataset containing health-related features such as age, gender, height, weight, blood pressure, cholesterol, glucose, smoking, alcohol consumption, and physical activity.

The original dataset contains 70,000 records. After applying data-cleaning rules, 68,636 records remained.

## Web Application Features
- Interactive prediction form
- Model performance dashboard
- Model comparison results
- Feature importance visualization
- Project information page

## Project Structure
```text
Heart_Disease_Prediction/
├── app.py
├── heart.csv
├── heart_cleaned.csv
├── heart_model.pkl
├── model_comparison_results.csv
├── heart_background.png
├── data_cleaning.py
├── random_forest.py
├── model_comparison.py
├── save_model.py
└── requirements.txt
```

The files listed above should be included only if they exist in your repository. Add any other files required by your application.

## Installation and Setup

### 1. Clone the repository
```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd heart-disease-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
streamlit run app.py
```

The application will open in your browser.

## Deployment
The application can be deployed using Streamlit Community Cloud by connecting the GitHub repository and selecting `app.py` as the main file.

## Limitations
- The model's performance is based on the selected dataset and test split.
- The model has not been clinically validated.
- Predictions must not be used to make medical decisions.

## Author
Sakthi Branisha V

## Acknowledgment
This project was developed as part of machine learning project work to gain practical experience in data preprocessing, model evaluation, and web application development.
