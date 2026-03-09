# Bagging Ensemble Example
# Author: Example ML Implementation

# Import required libraries
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()

# Features and target
X = data.data
y = data.target

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Base model
base_model = DecisionTreeClassifier()

# Bagging Ensemble Model
bagging_model = BaggingClassifier(
    estimator=base_model,   # base learner
    n_estimators=10,        # number of trees
    random_state=42
)

# Train the model
bagging_model.fit(X_train, y_train)

# Make predictions
y_pred = bagging_model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("Bagging Model Accuracy:", accuracy)