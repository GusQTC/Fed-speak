import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import shap

def rf():
    # Load the data
    df = pd.read_csv('values/merged_data_treated.csv')

    # Define features and target
    features = ['Mean_Sentiment_Score_Corrected','GDP Change','Mean_File_Count','Mean_Word_Count','Mean_Positive_Corrected','Mean_Negative_Corrected','Median_Sentiment_Score_Corrected','StdDev_Sentiment_Score_Corrected', 'Treasury Rate', 'Inflation Rate', 'Unemployment Rate']

    target = 'Interest Rate Change'

    features = df[features]
    target = df[target] * 100  # to get the percentage

    # Split the data into training and testing sets
    features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Create and train the Random Forest model
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(features_train, target_train)

    # Make predictions on the test set
    predictions = rf_model.predict(features_test)

    # Calculate the root mean squared error (RMSE)
    mse = mean_squared_error(target_test, predictions)
    rmse = mse ** 0.5


    # Print the RMSE
    print('Root Mean Squared Error (RMSE):', rmse)

    importance_df = pd.DataFrame({
        'Feature': features.columns,
        'Importance': rf_model.feature_importances_
    })
    print(importance_df)

   
    return rmse
    result = '''Root Mean Squared Error (RMSE): 22.735971132400156
                                Feature  Importance
    0     Mean_Sentiment_Score_Corrected    0.157786
    1                         GDP Change    0.102963
    2                    Mean_File_Count    0.017456
    3                    Mean_Word_Count    0.175560
    4            Mean_Positive_Corrected    0.108628
    5            Mean_Negative_Corrected    0.059466
    6   Median_Sentiment_Score_Corrected    0.029391
    7   StdDev_Sentiment_Score_Corrected    0.057984
    8                      Treasury Rate    0.126129
    9                     Inflation Rate    0.031969
    10                 Unemployment Rate    0.132668
    '''
rf()