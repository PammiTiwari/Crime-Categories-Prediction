# CrimeCast: Predict Crime Categories using Machine Learning

## Overview

CrimeCast is a machine learning challenge where participants aim to predict the type of crime based on various incident-level features. The dataset includes temporal, geographical, and demographic information that forms the basis for building predictive models.

## Objective

The primary objective is to develop a machine learning model that can accurately predict the **crime category** based on features such as time, location, victim details, and crime context.

## Dataset

The dataset consists of three files:
- **train.csv**: Contains the training data with crime details and labeled categories.
- **test.csv**: Contains the test data where crime categories need to be predicted.
- **sample_submission.csv**: A sample submission format.

### Key Features
- **Date & Time**: When the crime occurred.
- **Location**: Including latitude, longitude, area ID, and location description.
- **Victim Details**: Age, gender, and descent.
- **Crime Context**: Weapon used, modus operandi, and premise description.
- **Target**: `Crime_Category` (the crime type to be predicted).

## Approach

### 1. Data Loading and Exploration
   - Loaded the dataset and performed exploratory data analysis (EDA).
   - Explored distribution of crime categories, time patterns, and area-wise hotspots.

### 2. Data Preprocessing
   - **Datetime Features**: Extracted hour, day, weekday, and month from date and time fields.
   - **Categorical Encoding**: Applied Label Encoding and One-Hot Encoding to categorical variables.
   - **Missing Value Handling**: Used imputation or "Unknown" tagging for incomplete fields.
   - **Standardization**: Normalized numerical features like victim age.

### 3. Feature Engineering
   - Created derived features such as:
     - `is_night_time`
     - `is_weekend`
     - `weapon_used` (binary)
     - `location_cluster` (using latitude & longitude)
   - Combined rare categories and reduced high-cardinality fields for model simplicity.

### 4. Model Building
   Multiple models were tested to predict crime categories:
   - **Logistic Regression**
   - **Random Forest**
   - **XGBoost**
   - **LightGBM**
   - **Multilayer Perceptron (MLP)**
   - **Support Vector Machine (SVM)**

   Additionally, an **Ensemble Stacking Classifier** was used to combine model outputs for improved performance.

### 5. Model Tuning
   - **Hyperparameter Tuning**: Applied `GridSearchCV` and `RandomizedSearchCV` for optimal performance.
   - **Feature Importance Analysis**: Used feature selection techniques to reduce noise and overfitting.

### 6. Evaluation
   The models were evaluated using:
   - **Accuracy Score**: Main metric used in the competition.
   - **F1 Score**: To balance performance across all crime categories.
   - **Confusion Matrix**: To assess class-wise prediction capability.

## Results

The final model achieved an accuracy score of **0.92275** on the test dataset.

This result placed me at **Rank 14** on the public leaderboard. The model performed well across major crime categories and generalized effectively across both frequent and rare classes.
