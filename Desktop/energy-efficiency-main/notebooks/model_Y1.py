from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import DMatrix, cv  # Ensure DMatrix and cv are imported
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import numpy as np

# Fetch dataset
energy_efficiency = fetch_ucirepo(id=242)

# Data (as pandas dataframes)
X = energy_efficiency.data.features
y = energy_efficiency.data.targets

# Metadata
print(energy_efficiency.metadata)

# Variable information
print(energy_efficiency.variables)

# Add synthetic/mock features
np.random.seed(42)

# Insulation Quality: 0=Poor, 1=Average, 2=Good
X['Insulation_Quality'] = np.random.choice([0, 1, 2], size=len(X))
# Building Age: 1-100 years
X['Building_Age'] = np.random.randint(1, 101, size=len(X))
# Occupancy Level: 1-10 people
X['Occupancy_Level'] = np.random.randint(1, 11, size=len(X))
# HVAC System Type: 0=Central, 1=Split, 2=Window, 3=None
X['HVAC_System_Type'] = np.random.choice([0, 1, 2, 3], size=len(X))
# Climate Zone: 0=Cold, 1=Temperate, 2=Hot
X['Climate_Zone'] = np.random.choice([0, 1, 2], size=len(X))
# Appliance Efficiency: 0=Low, 1=Medium, 2=High
X['Appliance_Efficiency'] = np.random.choice([0, 1, 2], size=len(X))

# Combine 'X' and 'y' into a single DataFrame
combined_df = pd.concat([X, y], axis=1)

# Display the DataFrame
print(combined_df.head())  # .head() shows the first 5 rows

# Splitting the data into training and testing sets, focusing on 'Y1' as the target variable
X_train, X_test, y_train, y_test = train_test_split(
    combined_df.drop(['Y1', 'Y2'], axis=1),  # Features (excluding both Y1 and Y2)
    combined_df['Y1'],  # Target variable (Y1 - Heating Load)
    test_size=0.2,  # 20% of the data for testing
    random_state=42  # For reproducibility
)

# Convert data into DMatrix, an optimized data format for XGBoost
dtrain = DMatrix(X_train, label=y_train)

# Parameters for XGBoost
params = {
    'objective': 'reg:squarederror',
    'n_estimators': 100,
    'seed': 42
}

# Perform Cross-Validation
cv_results = cv(
    dtrain=dtrain,
    params=params,
    nfold=5,  # Number of folds in K-Fold Cross-Validation
    num_boost_round=100,  # Number of boosting rounds, equivalent to n_estimators
    metrics='rmse',  # Root Mean Square Error as evaluation metric
    as_pandas=True,  # Return results as a Pandas DataFrame
    seed=42
)

# Display Cross-Validation results
print(cv_results)

# Initialize the XGBoost regressor with the best parameters found
xg_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, seed=42)

# Train the model
xg_reg.fit(X_train, y_train)

# Predictions on the test set
y_pred = xg_reg.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Extract feature importance from the XGBoost model
feature_importance = xg_reg.feature_importances_
features = X_train.columns

# Create a DataFrame for feature importance
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})

# Sort the DataFrame by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Create a bar chart with Plotly
fig = px.bar(importance_df,
             x='Importance',
             y='Feature',
             orientation='h',
             title='Feature Importance')

# Display the chart
fig.show()

import pickle

# Save Model as Pickle-File
with open('xgboost_regression_model1.pkl', 'wb') as file:
    pickle.dump(xg_reg, file)
