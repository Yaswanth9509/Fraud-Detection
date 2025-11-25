# Fraud-Detection

Fraud Detection Model

This repository contains an end-to-end Fraud Detection System designed for highly imbalanced datasets, such as credit card transactions. The system supports dataset ingestion, interactive fallback options, exploratory analysis, preprocessing with sampling techniques, model training, evaluation, hyperparameter tuning, feature importance analysis, and real-time fraud prediction.

Overview

The core component of this project is the FraudDetectionModel class, which encapsulates the complete machine learning pipeline for fraud classification. It includes flexible dataset handling (CSV, Excel, or synthetic generation), scalable preprocessing, multiple modeling approaches, anomaly detection support, and detailed evaluation capabilities.

Features
1. Dataset Management

Supports CSV and Excel file formats.

Validates dataset structure and target column requirements.

Supports renaming or specifying the target column when missing.

Handles binary label validation and automatic mapping.

Provides an interactive fallback menu when dataset loading fails.

Includes a synthetic dataset generator modeled on typical fraud detection data characteristics.

2. Exploratory Data Analysis

Dataset summary and information inspection.

Class imbalance overview and fraud percentage.

Missing value detection and handling.

Visualizations including:

Class distribution

Amount and Time distributions

Correlation heatmap

Box plots and feature–target correlations

<img width="1919" height="1017" alt="Screenshot 2025-11-25 083410" src="https://github.com/user-attachments/assets/2121ab95-e1e8-452e-9b91-dd5f96c03ed5" />

3. Preprocessing and Sampling

Numeric feature extraction and filtering.

Feature scaling using StandardScaler.

Train–test splitting with stratified sampling.

Optional sampling techniques:

SMOTE oversampling

Random undersampling

4. Model Training

The system trains and manages multiple models:

Logistic Regression

Random Forest

SVM

Isolation Forest (unsupervised anomaly detection)

All trained models are stored and accessible for evaluation and inference.

5. Model Evaluation

Generates predictions, classification reports, accuracy, and F1 score for each model.

Confusion matrix visualizations.

ROC curve and AUC computation for models with probability outputs.

Automatic selection of the best model based on F1 score.

<img width="1919" height="1014" alt="Screenshot 2025-11-25 083429" src="https://github.com/user-attachments/assets/7378eedb-ddad-462b-83fd-3f25946d250b" />

6. Hyperparameter Tuning

Supports grid-search hyperparameter tuning for:

Random Forest

Logistic Regression

Uses stratified sampling and cross-validation for efficient evaluation.

7. Feature Importance Analysis

Extracts and ranks feature importance using Random Forest.

Generates bar charts for top features.

Computes cumulative feature influence.

8. Real-Time Fraud Detection

Provides an interface to evaluate new transaction records:

Returns predictions

Returns probabilities (when available)

Supports both supervised and isolation-forest-based detection

Project Structure
Fraud_Detection.py               # Core fraud detection system
Fraud_Detection.csv              # Optional dataset uploaded by the user

Technologies Used

Python 3.x

NumPy

Pandas

Matplotlib

Seaborn

scikit-learn

imbalanced-learn (SMOTE, undersampling)

Usage
Initialization
fraud_detector = FraudDetectionModel()

Loading Data
fraud_detector.load_data("path/to/fraud_dataset.csv")

Exploratory Analysis
fraud_detector.explore_data()

Preprocessing
fraud_detector.preprocess_data(use_sampling='smote')

Model Training
fraud_detector.train_models()

Model Evaluation
results = fraud_detector.evaluate_models()

Hyperparameter Tuning
fraud_detector.hyperparameter_tuning('Random Forest')

Feature Importance
fraud_detector.feature_importance_analysis()

Fraud Detection on New Data
new_data = fraud_detector.data.drop('Class', axis=1).sample(1)
fraud_detector.detect_fraud(new_data)

Synthetic Data Generation

If no dataset is provided or loading fails, the system can generate a synthetic dataset with:

Normal and fraud distributions

Time and Amount variables

28 anonymized PCA-like features

Highly imbalanced class ratio similar to real credit card datasets

This allows full demonstration without requiring external data.

Use Cases

Credit card fraud detection

Online transaction monitoring

Banking and fintech security systems

Fraud analytics research

Model comparison and benchmarking
