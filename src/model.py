from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
import numpy as np

class PolynomialBostonPredictor:
    def __init__(self, degree=2):
        self.degree = degree
        self.model = None

    def train(self, X, y):
        # Create polynomial features and scale the data
        poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        scaler = StandardScaler()
        X_poly_scaled = scaler.fit_transform(X_poly)
        
        # Create a Linear Regression model
        self.model = Pipeline([('scaler', scaler), ('linear', LinearRegression())])
        self.model.fit(X_poly_scaled, y)

    def evaluate(self, X, y, cv=10):
        # Perform cross-validation
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        r2_scores = cross_val_score(self.model, X, y, cv=kf, scoring='r2')
        mean_r2 = np.mean(r2_scores)

        mse_scores = -cross_val_score(self.model, X, y, cv=kf, scoring='neg_mean_squared_error')
        mean_mse = np.mean(mse_scores)

        mae_scores = -cross_val_score(self.model, X, y, cv=kf, scoring='neg_mean_absolute_error')
        mean_mae = np.mean(mae_scores)

        rmse_scores = np.sqrt(mse_scores)
        mean_rmse = np.mean(rmse_scores)

        return {
            'R2 Score': mean_r2,
            'Mean Squared Error': mean_mse,
            'Mean Absolute Error': mean_mae,
            'Root Mean Squared Error': mean_rmse
        }

    def predict(self, X):
        return self.model.predict(X)
