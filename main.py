"""
Titanic Survivors Prediction

This script trains a machine learning model to predict whether a passenger
survived or not using features such as age, gender, ticket class, and fare.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------
# The file path points to the project folder in PyCharm.
titanic_data = pd.read_csv(
    r"C:\Users\lenovo\PycharmProjects\titanic_survivors_prediction\train.csv"
)


# ------------------------------------------------------------
# Data cleaning and preprocessing
# ------------------------------------------------------------

# Drop columns that are not useful for prediction
titanic_data.drop(columns=["Cabin"], inplace=True)

# Fill missing ages with the mean age
titanic_data["Age"].fillna(titanic_data["Age"].mean(), inplace=True)

# Replace missing embarked values with the most common value
titanic_data["Embarked"].fillna(titanic_data["Embarked"].mode()[0], inplace=True)

# Convert categorical values to numeric labels
titanic_data.replace({
    "Sex": {"male": 0, "female": 1},
    "Embarked": {"S": 0, "C": 1, "Q": 2}
}, inplace=True)


# ------------------------------------------------------------
# Select features for training
# ------------------------------------------------------------
X = titanic_data[["Age", "Sex", "Pclass", "Fare"]]   # Input features
Y = titanic_data["Survived"]                         # Target variable


# ------------------------------------------------------------
# Split into training and testing sets
# ------------------------------------------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)


# ------------------------------------------------------------
# Train the logistic regression model
# ------------------------------------------------------------
model = LogisticRegression(max_iter=200)
model.fit(X_train, Y_train)


# ------------------------------------------------------------
# Model evaluation
# ------------------------------------------------------------
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

train_accuracy = accuracy_score(Y_train, train_predictions)
test_accuracy = accuracy_score(Y_test, test_predictions)


# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------
print("\nTitanic Survival Prediction Model")
print("----------------------------------")
print(f"Training Accuracy : {train_accuracy:.4f}")
print(f"Testing Accuracy  : {test_accuracy:.4f}\n")
