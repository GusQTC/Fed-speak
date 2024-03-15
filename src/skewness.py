import pandas as pd


df = pd.read_csv('tokens/result_minutes_economic.csv')


# Calculate skewness
skewness = df['Value'].skew()

print("Skewness: ", skewness)