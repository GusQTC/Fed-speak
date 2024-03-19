import pandas as pd


df = pd.read_csv('tokens/result_statements_economic.csv')


# Calculate skewness
skewness = df['Value'].skew()

print("Skewness: ", skewness)