import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

#load the dataset
boston = load_boston()

#convert the dataset to pandas dataframe
df = pd.DataFrame(data=boston.data, columns = boston.feature_names)

#add the target to the dataframe in column "Price"
df['Price'] = boston.target

#separate the features and the target for the modelling
x_data = df.drop('Price', axis=1)
y_data = df['Price']

#Random Forest with Cross validation 
regressorOHT = RandomForestRegressor(n_estimators=100, max_leaf_nodes=None, min_samples_leaf=1, random_state=42)

# Perform k-fold cross-validation
k = 5  # Number of folds
r2_scores = cross_val_score(regressorOHT, x_data, y_data, cv=k, scoring='r2')
mae_scores = -cross_val_score(regressorOHT, x_data, y_data, cv=k, scoring='neg_mean_absolute_error')
mse_scores = -cross_val_score(regressorOHT, x_data, y_data, cv=k, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(mse_scores)

# Calculate the mean of each metric
mean_r2 = np.mean(r2_scores)
mean_mae = np.mean(mae_scores)
mean_mse = np.mean(mse_scores)
mean_rmse = np.mean(rmse_scores)

print(f"R-squared (Cross-Validation): {mean_r2:.2f}")
print(f"Mean Absolute Error (Cross-Validation): {mean_mae:.2f}")
print(f"Mean Squared Error (Cross-Validation): {mean_mse:.2f}")
print(f"Root Mean Squared Error (Cross-Validation): {mean_rmse:.2f}")