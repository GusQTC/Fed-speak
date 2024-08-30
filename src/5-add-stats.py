#add fredapi economic data to new_values/result_economic_corrected_ner.csv
import pandas as pd
from fredapi import Fred
import os

# Load the data
df = pd.read_csv('new_values/result_economic_corrected_ner.csv')
df['Year'] = df['Year'].astype(int)
df['Month'] = df['Month'].astype(int)
# Define the FRED API key

FRED_API_KEY= "55ddb1dfa49988e7cd6ecaa8095a5143"
fred = Fred(api_key=FRED_API_KEY)

interest_rate = fred.get_series('FEDFUNDS', observation_start='01/01/2000', observation_end='01/01/2024' )

treasury_rate = fred.get_series('DGS10', observation_start='01/01/2000', observation_end='01/01/2024' )

inflation_rate = fred.get_series('CPIAUCSL', observation_start='01/01/2000', observation_end='01/01/2024' )

unemployment_rate = fred.get_series('UNRATE', observation_start='01/01/2000', observation_end='01/01/2024' )

gpd = fred.get_series('GDP', observation_start='01/01/2000', observation_end='01/01/2024' )

stocks = fred.get_series('SP500', observation_start='01/01/2000', observation_end='01/01/2024' )
print(inflation_rate)
print(inflation_rate)

# Ensure each series has a name set before resampling
interest_rate.name = 'interest_rate' if interest_rate.name is None else interest_rate.name
treasury_rate.name = 'treasury_rate' if treasury_rate.name is None else treasury_rate.name
inflation_rate.name = 'inflation_rate' if inflation_rate.name is None else inflation_rate.name
stocks.name = 'stocks' if stocks.name is None else stocks.name

# Resample each series to the start of the month and calculate the mean
interest_rate_resampled = interest_rate.resample('MS').mean()
interest_rate_change_resampled = interest_rate_resampled.pct_change()
treasury_rate_resampled = treasury_rate.resample('MS').mean()
inflation_rate_resampled = inflation_rate.resample('MS').mean()*100
unemployment_rate_resampled = unemployment_rate.resample('MS').mean()
gdp_resampled = gpd.resample('MS').mean().pct_change()
stocks_resampled = stocks.resample('MS').mean().pct_change()

# +1, 0, -1 for interest rate change
interest_rate_label = interest_rate_change_resampled.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

# Combine the resampled series into a DataFrame
economic_data = pd.DataFrame({
    'interest_rate': interest_rate_resampled,
   'interest_rate_change': interest_rate_change_resampled,
    'treasury_rate': treasury_rate_resampled,
   'inflation_rate': inflation_rate_resampled,
    'unemployment_rate': unemployment_rate_resampled,
    'gdp_growth': gdp_resampled,
   'S&P-500': stocks_resampled,
    'interest_rate_label': interest_rate_label
})
economic_data['Year'] = economic_data.index.year
economic_data['Month'] = economic_data.index.month
economic_data.reset_index(inplace=True)

economic_data.set_index(['Year', 'Month'], inplace=True)

# Merge the economic data with the existing DataFrame
#df = pd.merge(df, economic_data, on=['Year', 'Month'], how='left')

# Save the combined DataFrame
economic_data.to_csv('economic_data.csv', index=False)
