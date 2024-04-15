import pandas as pd

import numpy as np
import stats
from scipy.stats import shapiro, normaltest
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot


df = pd.read_csv('values/merged_data.csv')
features = ['Mean_Sentiment_Score_Corrected','Median_Sentiment_Score_Corrected','StdDev_Sentiment_Score_Corrected', 'Treasury Rate', 'Inflation Rate', 'Unemployment Rate']

def calculate_statistics(df, features):
    # Loop through the columns
    for column in features:
        # Calculate the statistic and p-value for Shapiro-Wilk test
        stat, p = shapiro(df[column])
        print(f'Shapiro-Wilk Test for {column}: Statistic={stat}, p={p}')

        # Calculate the statistic and p-value for D'Agostino's K^2 Test
        stat, p = normaltest(df[column])
        print(f"D'Agostino's K^2 Test for {column}: Statistic={stat}, p={p}")

        # Plot a histogram
        sns.histplot(df[column], kde=True)
        plt.title(f'Histogram for {column}')
        plt.show()

        # Plot a Q-Q plot
        qqplot(df[column], line='s')
        plt.title(f'Q-Q Plot for {column}')
        plt.show()


def remove_outliers(df, features):
    for feature in features:
        # Calculate Q1, Q2 and IQR
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1

        # Define the upper and lower bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter out the outliers
        df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]

    return df

df_treated = remove_outliers(df, features)

print(df_treated.head())

df_treated.to_csv('values/merged_data_treated.csv', index=False)

# whats the diff between the df and df_treated

diff = df.shape[0] - df_treated.shape[0]
print(f'The number of rows removed is {diff}')

