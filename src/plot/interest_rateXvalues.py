import pandas as pd
import matplotlib.pyplot as plt
from fredapi import Fred

from scipy import stats

# Create a Fred instance



FRED_API_KEY= "55ddb1dfa49988e7cd6ecaa8095a5143"
fred = Fred(api_key=FRED_API_KEY)



# Read the file into a DataFrame
df = pd.read_csv('economic_lexycon/result_lexycon_speeches_economic.csv')

#df = df[df['File'].str.contains('monetary')]
#df = df[df['File'].str.contains('fomcminutes')]

df = df.set_index('File')

interest_rate = fred.get_series('FEDFUNDS', observation_start='01/01/2013', observation_end='01/01/2024' )

treasury_rate = fred.get_series('DGS10', observation_start='01/01/2013', observation_end='01/01/2024' )

inflation_rate = fred.get_series('CPIAUCSL', observation_start='01/01/2013', observation_end='01/01/2024' )
print(inflation_rate)
inflation_rate = inflation_rate.pct_change()
print(inflation_rate)



# Resample each series to the start of the month and calculate the mean
interest_rate_resampled = interest_rate.resample('MS').mean()
treasury_rate_resampled = treasury_rate.resample('MS').mean()
inflation_rate_resampled = inflation_rate.resample('MS').mean()*100

# Combine the resampled series into a DataFrame
economic_data = pd.DataFrame({'Interest Rate': interest_rate_resampled, 'Treasury Rate': treasury_rate_resampled, 'Inflation Rate': inflation_rate_resampled})

economic_data.index = pd.to_datetime(interest_rate.index, format='%Y-%m')
# Remove the '.htm.txt' from the 'File' column
# Convert the index to a string
df.index = df.index.astype(str)

# Remove the '.htm.txt' from the 'File' index

df['Date'] = df.index.str.extract(r'(\d{8})', expand=False)

#df.index = df.index.str.replace('fomcminutes', '')



df['Date']= pd.to_datetime(df['Date'], format='%Y%m%d')

df_agg = df.groupby('Date').mean()


# group the value average by month
df_agg = df_agg.resample('MS').mean()

# Convert 'File' index to datetime format

# Adjust 'File' dates to the first day of the month
df_agg.index = df_agg.index.values.astype('datetime64[M]')

# Convert 'File' to datetime format

df_agg = pd.merge(df_agg, economic_data, left_index=True, right_index=True, how='left')

df_agg = df_agg.dropna()

# Normalize the 'Value' column
df_agg['Value'], fitted_lambda = stats.yeojohnson(df_agg['Value'])
print('Fitted lambda:', fitted_lambda)


df_agg = df_agg.dropna()
df_agg['Interest Rate Lower Bound'] = df_agg['Interest Rate'] - 0.25
df_agg['Interest Rate Upper Bound'] = df_agg['Interest Rate'] + 0.25

# inflation bounds
df_agg['Inflation Rate Lower Bound'] = df_agg['Inflation Rate'] - 0.25
df_agg['Inflation Rate Upper Bound'] = df_agg['Inflation Rate'] + 0.25

# Set 'File' as the index



df_agg.to_csv('economic_lexycon/comparison_interest_sentiment_speeches.csv', sep=',', index=True)

fig, ax1 = plt.subplots()

# Plot the 'Value' column
ax1.plot(df_agg['Value'], color='lightblue')
ax1.set_ylabel('Value', color='lightblue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create a second y-axis that shares the same x-axis
ax2 = ax1.twinx()

# Plot the 'Interest Rate' column on the second y-axis
ax2.plot(df_agg['Interest Rate'], color='red')

ax2.fill_between(df_agg.index, df_agg['Interest Rate'], df_agg['Inflation Rate Upper Bound'], 
                 where=df_agg['Interest Rate'] < df_agg['Inflation Rate Upper Bound'], 
                 color='red', alpha=0.5)


ax2.set_ylabel('Interest Rate', color='red')
ax2.tick_params(axis='y', labelcolor='red')

#plot inflation
ax2.plot(df_agg['Inflation Rate'], color='orange')
# inflation bounds
ax2.plot(df_agg['Inflation Rate Lower Bound'], color='#FFA500', linestyle='--')
ax2.plot(df_agg['Inflation Rate Upper Bound'], color='#FFA500', linestyle='--')

#plot lower bound
ax2.plot(df_agg['Interest Rate Lower Bound'], color='black', linestyle='--')

#plot upper bound
ax2.plot(df_agg['Interest Rate Upper Bound'], color='black', linestyle='--')

#plot treasury rate
ax2.plot(df_agg['Treasury Rate'], color='green')

#ax2.plot(df_agg['Interest Rate Change'], color='green')

# Show the plot

plt.savefig('economic_lexycon/interest_change_speeches.png')

plt.show()
