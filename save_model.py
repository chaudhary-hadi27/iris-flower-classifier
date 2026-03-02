# -*- coding: utf-8 -*-
"""
Run this file ONCE to train and save the model.
Ye file sirf ek baar chalao — model.pkl ban jayega.
Then push model.pkl to GitHub along with app.py
"""

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train
dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X_train, y_train)

# Accuracy check
y_pred = dtc.predict(X_test)
print(f"✅ Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save model
joblib.dump(dtc, "iris_model.pkl")
print("✅ Model saved as iris_model.pkl")
