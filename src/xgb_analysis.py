import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np


import pickle

# Load the data
df = pd.read_csv('values/merged_data_treated.csv')

# Define features and target
features = ['Mean_Sentiment_Score_Corrected','GDP Change','Mean_File_Count','Mean_Word_Count','Mean_Positive_Corrected','Mean_Negative_Corrected','Median_Sentiment_Score_Corrected','StdDev_Sentiment_Score_Corrected', 'Treasury Rate', 'Inflation Rate', 'Unemployment Rate']

target = 'Interest Rate Change'

X = df[features]
y = df[target] * 100  # to get the percentage

#y, lambda_val = stats.yeojohnson(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define the model
model = XGBRegressor(reg_alpha=0.1) #with regulatization parameter to avoid overfitting

# Define the grid of hyperparameters to search
hyperparameter_grid = {
    'n_estimators': [100, 500, 900, 1100, 1500],
    'max_depth':[2, 3, 5, 10, 15],
    'learning_rate':[0.05,0.1,0.15,0.2],
    'min_child_weight':[1,2,3,4],
    'booster':['gbtree','gblinear'],
    'base_score':[0.25,0.5,0.75,1]
    }

# Set up the grid search with 5-fold cross validation
grid_cv = GridSearchCV(estimator=model, param_grid=hyperparameter_grid, cv=5, n_jobs =-1, verbose = 1)
#show progress
print('Starting Grid Search')
grid_cv.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_test, y_test)], verbose=1)
print('Grid Search Finished')

# Get the best parameters
best_parameters = grid_cv.best_params_

print(f'Best parameters: {best_parameters}')

# Train the model with the best parameters
best_model = XGBRegressor(**best_parameters)

filename = 'finalized_model.pkl'
pickle.dump(best_model, open(filename, 'wb'))

best_model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_test, y_test)])

# Make predictions
y_pred = best_model.predict(X_test)

#y_pred = inv_boxcox(y_pred_transformed, lambda_val)
# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f'MAE: {mae}, MSE: {mse}, RMSE: {rmse}')


# Get feature importances
importances = best_model.feature_importances_

# Convert the importances into a DataFrame
importances_df = pd.DataFrame({'feature':X_train.columns, 'importance':importances})

# Sort the DataFrame by importance
importances_df = importances_df.sort_values('importance', ascending=False)




print(importances_df)

def plot_importances(importances_df):
    plt.figure(figsize=(10, 6))
    plt.barh(importances_df['feature'], importances_df['importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances')
    plt.gca().invert_yaxis()
    plt.show()
