from model import PolynomialBostonPredictor
import pandas as pd
from sklearn.datasets import load_boston
import joblib

# Load the Boston Housing Price dataset
boston = load_boston()
X = boston.data
y = boston.target

# Instantiate the model
model = PolynomialBostonPredictor(degree=2)

# Train the model
model.train(X, y)

# Save the trained model to a file
joblib.dump(model, 'polynomial_boston_model.pkl')
