# Recipe for Rating: Predict Food Ratings using ML

## Overview

"Recipe for Rating" is a machine learning challenge where participants aim to predict how people rate recipes based on available data. The dataset includes recipe names, reviews, and several other features that serve as the foundation for developing predictive models.

## Objective

The primary objective is to develop a machine learning model that can accurately predict the rating of recipes based on various features.

## Dataset

The dataset consists of three files:
- **train.csv**: Contains the training data with recipe details and ratings.
- **test.csv**: Contains the test data where ratings need to be predicted.
- **sample.csv**: A sample submission file.

### Key Features
- **Recipe Name**: The name or title of the recipe.
- **Reviews**: User reviews associated with each recipe.
- **Additional Features**: Other features such as the ingredients and preparation steps, which may also affect the ratings.

## Approach

### 1. Data Loading and Exploration
   - Loaded the dataset and conducted basic exploratory data analysis (EDA).
   - Examined the distribution of ratings, feature correlations, and data quality (e.g., handling missing values).

### 2. Data Preprocessing
   - **Text Preprocessing**: Cleaned the reviews by removing unwanted characters, stopwords, and applying tokenization.
   - **Feature Extraction**: Used **TF-IDF Vectorization** and **Count Vectorization** to convert text features into numerical form.
   - **Label Encoding**: Categorical variables such as recipe types were encoded into numerical labels.
   - **Train-Test Split**: Split the dataset into training and validation sets to ensure model evaluation.

### 3. Model Building
   Several machine learning models were tried and tested to predict food ratings:
   - **Logistic Regression**
   - **Random Forest Classifier**
   - **Naive Bayes**
   - **Gradient Boosting Classifier**
   - **Support Vector Machines (SVM)**
   - **Multilayer Perceptron (MLP)**

   **Stacking Classifier** was also implemented, combining different models to improve predictive performance.

### 4. Model Tuning
   - **Hyperparameter Tuning**: Used `GridSearchCV` for model selection and fine-tuning of hyperparameters to optimize the performance of the models.
   - **Feature Selection**: Applied statistical methods like `chi2` to select the most important features for model training.

### 5. Evaluation
   The models were evaluated using:
   - **Accuracy Score**: The main metric used for model evaluation in this competition.
   - **Confusion Matrix**: To evaluate model performance on individual classes.
   - **F1 Score**: For a balance between precision and recall, particularly important in cases of class imbalance.

   After several experiments, **Random Forest** and **Gradient Boosting** provided the best results in terms of accuracy and generalization across the dataset.

## Results

The final model achieved an accuracy score of **0.78156** (highest was 0.80444) on the test dataset . Additionally, the confusion matrix showed strong performance in predicting the higher ratings, with slight underperformance in distinguishing between lower ratings.
