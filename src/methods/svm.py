from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import TensorDataset, DataLoader
import shap
# Define the model

def svm():
    # Load the data
    df = pd.read_csv('values/merged_data_treated.csv')

    # Define features and target
    features = ['Mean_Sentiment_Score_Corrected','GDP Change','Mean_File_Count','Mean_Word_Count','Mean_Positive_Corrected','Mean_Negative_Corrected','Median_Sentiment_Score_Corrected','StdDev_Sentiment_Score_Corrected', 'Treasury Rate', 'Inflation Rate', 'Unemployment Rate']

    target = 'Interest Rate Change'

    X = df[features]
    y = df[target] * 100  # to get the percentage


    #y, lambda_val = stats.yeojohnson(y)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)


    model = SVR(kernel='linear')

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
# Assuming 'model' is your trained SVM model and 'features' is a list of your feature names
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': abs(model.coef_[0])
    })


    return rmse


svm()