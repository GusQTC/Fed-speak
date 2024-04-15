import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv('economic_lexycon/comparison_interest_sentiment_speeches.csv', index_col=0, parse_dates=True)

correlation = df['Value'].corr(df['Interest Rate Change'])
print('Correlation:', correlation)

#monetary Correlation: -0.4278455710632934
# Minute Correlation: -0.15239296988007217

#speeches Correlation: 0.20867533173174313
#speeches change Correlation: -0.08212339183700984
