# Titanic-Survivors-Prediction

This project builds a machine learning model to predict whether a passenger survived the Titanic disaster. The model uses key features such as age, gender, ticket class, and fare. The goal is to preprocess the dataset, train a classification model, and evaluate its performance.

Dataset Source: https://www.kaggle.com/c/titanic

## Project Overview
The project follows a standard machine learning workflow:
1. Load the Titanic dataset
2. Clean and preprocess the data
3. Encode categorical variables
4. Select relevant features
5. Split the data into training and testing sets
6. Train a Logistic Regression model
7. Evaluate model performance

## Features Used
The model predicts survival using:
- Age  
- Sex  
- Pclass  
- Fare  

Target variable:
- Survived (0 = No, 1 = Yes)

## Project Structure
titanic_survivors_prediction/
│
├── train.csv  
├── titanic_model.py  
├── .gitignore  
└── README.md  

The `.gitignore` file is included to keep the repository clean by excluding IDE files, Python cache files, virtual environments, and other unnecessary items from version control.

## Data Preprocessing Steps
- Dropped the Cabin column due to too many missing values  
- Filled missing Age values using the mean  
- Filled missing Embarked values using the mode  
- Converted Sex and Embarked into numerical form  
- Selected Age, Sex, Pclass, and Fare as features for the model

## Model Training
A Logistic Regression model is used to perform binary classification.  
The dataset is split into training and testing sets (80/20).  
The model is trained on the training data and evaluated on both sets.

## Evaluation
The script prints:
- Training accuracy  
- Testing accuracy  

These scores help determine how well the model performs on both seen and unseen data.

## How to Run the Project
1. Download the Titanic dataset from Kaggle and place train.csv in the project directory  
2. Install required libraries:

pip install pandas scikit-learn

3. Run the script:

python titanic_model.py

## Requirements
- Python 3.8+  
- pandas  
- scikit-learn  

## Notes
This project provides a clean baseline model. It can be extended with more advanced preprocessing, feature engineering, or different machine learning algorithms.
