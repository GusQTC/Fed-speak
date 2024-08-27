import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel

import shap


def xgb():
    # Load the data
    df = pd.read_csv('new_values/result_economic_corrected_ner_fred.csv')

    # Handling NaN values by repeating previous
    df.bfill(inplace=True)
    # Define features and target
    features=["Mean_File_Count","Mean_Word_Count","Mean_Positive","Mean_Negative","Mean_Sentiment_Score","Median_Sentiment_Score","StdDev_Sentiment_Score","Mean_Sentiment_Score_Corrected","Mean_Positive_Corrected","Mean_Negative_Corrected","Median_Sentiment_Score_Corrected","StdDev_Sentiment_Score_Corrected","ORG","GPE","PERSON","WORK_OF_ART","PRODUCT","NORP","LOC","LAW","FAC","EVENT","LANGUAGE","treasury_rate","inflation_rate","unemployment_rate","gdp_growth","S&P-500"]


    target = 'interest_rate_change'

    X = df[features]
    y = df[target] * 100  # to get the percentage


    #y, lambda_val = stats.yeojohnson(y)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)


    # Define the model
    xgb_model = XGBRegressor(reg_alpha=0.1) #with regulatization parameter to avoid overfitting
    
    
    # Create the pipeline
    pipeline = Pipeline([
        ('feature_selection', SelectFromModel(XGBRegressor())),
        ('regression', xgb_model)
    ])
    
    param_grid = {
    'feature_selection__estimator__max_depth': [3, 5, 7],
    'feature_selection__threshold': ['mean', 'median', '1.25*mean'],
    'regression__n_estimators': [50, 100, 200],
    'regression__learning_rate': [0.01, 0.05, 0.1],
    'regression__max_depth': [3, 5, 7],
    'regression__min_child_weight': [1, 3, 5],
    'regression__base_score':[0.25,0.5,0.75,1],
    'regression__booster':['gbtree','gblinear']

}

    # Set up the grid search with 5-fold cross validation
    grid_cv = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)    #show progress
    print('Starting Grid Search')
    grid_cv.fit(X_train, y_train)
    print('Grid Search Finished')

    # Get the best parameters
    best_parameters = grid_cv.best_params_

    print(f'Best parameters: {best_parameters}')

    # Train the model with the best parameters
    # Best parameters: {'base_score': 0.75, 'booster': 'gbtree', 'learning_rate': 0.05, 'max_depth': 3, 'min_child_weight': 3, 'n_estimators': 100}
    best_model = grid_cv.best_estimator_
    
    selected_features = best_model.named_steps['feature_selection'].get_support()
    
    feature_names = np.array(features)[selected_features]

    X_train_selected = X_train[feature_names]

    
    best_model.fit(X_train_selected, y_train)
    
    X_test_selected = X_test[feature_names]
    


    # Make predictions
    y_pred = best_model.predict(X_test_selected)

    #y_pred = inv_boxcox(y_pred_transformed, lambda_val)
    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    print(f'XGB \n MAE: {mae}, MSE: {mse}, RMSE: {rmse}')


    # Get feature importances
    importances = best_model.named_steps['regression'].feature_importances_
    
    #importances_selected = importances[selected_features]


    # Convert the importances into a DataFrame
    importances_df = pd.DataFrame({'feature': feature_names, 'importance': importances})

    # Sort the DataFrame by importance
    importances_df = importances_df.sort_values('importance', ascending=False)

    print(importances_df)
    
    plot_importances(importances_df)
    
    importances_df.to_csv('new_values/importances_REF.csv', index=False)

    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test_selected)
    shap.summary_plot(shap_values, X_test_selected, feature_names=features)
    return rmse

def plot_importances(importances_df):
    plt.figure(figsize=(10, 6))
    plt.barh(importances_df['feature'], importances_df['importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances')
    plt.gca().invert_yaxis()
    plt.show()

xgb()