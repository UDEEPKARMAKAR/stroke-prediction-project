# Stroke Prediction and Risk Analysis

A comprehensive machine learning project focused on predicting stroke risk using clinical and demographic patient data, built as part of Minor Project 4.

## Project Overview
Stroke is a leading cause of death and disability globally. Early identification of high-risk individuals can significantly improve prevention and treatment outcomes. This project implements an end-to-end Machine Learning pipeline utilizing the **Kaggle Stroke Prediction Dataset** to predict whether a patient is likely to suffer a stroke based on parameters like age, hypertension, heart disease, average glucose level, and BMI.

---

## Key Features & Methodology

* **Exploratory Data Analysis (EDA):** Uncovered patterns regarding age correlations, lifestyle attributes, and handled heavy class imbalance (~4.87% positive stroke cases).
* **Data Preprocessing:** 
  * Median imputation for missing `bmi` values to preserve distribution skewness.
  * One-hot encoding for categorical variables (`drop_first=True`).
  * Stratified 80:20 train-test split to maintain class ratio integrity.
* **Model Development & Tuning:** Trained a **Random Forest Classifier** optimized via **GridSearchCV** with 3-fold cross-validation, prioritizing metrics robust to class imbalance.
* **Evaluation Metrics:** Evaluated using Confusion Matrix, Precision, Recall, F1-Score, ROC-AUC Curve, and Feature Importance analysis.

---

## Repository Structure
```text
├── dataset/                  # Contains the raw and processed CSV data files
├── notebooks/                # Jupyter Notebooks containing EDA, training, and evaluation
├── reports/                  # Detailed project report (PDF/DOCX)
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies
