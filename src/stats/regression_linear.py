import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Read the example data into a DataFrame
data = pd.read_csv('economic_lexycon/comparison_interest_sentiment_speeches.csv')

def categorize_change(change):
    if change > 0:
        return +1
    elif change < 0:
        return -1
    else:
        return 0

"""
,Value,Interest Rate
2019-01-01,-186958,2.4
2019-03-01,-168877,2.41
2019-05-01,-110558,2.39
2019-06-01,-158766,2.38
2019-07-01,-124577,2.4
2019-09-01,-275322,2.04
2019-10-01,-307610,1.83
2019-12-01,-152728,1.55
"""

data['Interest Rate Change'] = data['Interest Rate'].diff()

data.dropna(inplace=True)

data['Interest Rate Binary'] = data['Interest Rate Change'].apply(categorize_change)

data.to_csv('comparison_interest_sentiment_speeches.csv', sep=',', index=True)
# Prepare the data
X = data[['Value']]  # Features
y = data['Interest Rate Binary']  # Target variable

# Create and train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

print(predictions)

# Calculate evaluation metrics
mse = mean_squared_error(y, predictions)
mae = mean_absolute_error(y, predictions)

print('Predicted interest rate variation:', predictions[0])
print('Mean squared error (MSE):', mse)
print('Mean absolute error (MAE):', mae)


#
# monetary
#Predicted interest rate: 1.7510924335017894
#Mean squared error (MSE): 2.912787454858025
#Mean absolute error (MAE): 1.436320139064976

"""
Predicted interest rate: 0.06802499536798534
Mean squared error (MSE): 0.11869024227490574
Mean absolute error (MAE): 0.22611805658272016
"""

# minutes
#Predicted interest rate: 2.094347894228965
#Mean squared error (MSE): 3.7683115919454164
#Mean absolute error (MAE): 1.662860397875994

"""
Predicted interest rata: 0.12808323159889737
Mean squared error (MSE): 0.09923549007293046
Mean absolute error (MAE): 0.19440649535199778
"""

#Speeches
'''
Predicted interest rate variation: 0.08385064779513893
Mean squared error (MSE): 0.1664587426223797
Mean absolute error (MAE): 0.23585597302016412
'''
#speeches change in interest rate
### Predicted interest rate variation: 0.020440404521670744
# Mean squared error (MSE): 0.1650770970403212
# Mean absolute error (MAE): 0.23512486857260817