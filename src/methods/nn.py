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


def nnet():
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


    # Convert data to tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float).view(-1, 1)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float).view(-1, 1)

    # Create a DataLoader
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_data, batch_size=10)

    # Define the model architecture
    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )

    # Define the loss function and the optimizer
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters())

    # Train the model
    for epoch in range(50):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # Make predictions
    with torch.no_grad():
        y_pred = model(X_test_tensor)

    # Convert tensors to numpy arrays for evaluation
    y_test = y_test_tensor.numpy()
    y_pred = y_pred.numpy()



    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    print(f'Neural Network \n MAE: {mae}, MSE: {mse}, RMSE: {rmse}')

    return rmse
nnet()