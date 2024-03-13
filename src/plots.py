import pandas as pd
import matplotlib.pyplot as plt
from fredapi import Fred

# Create a Fred instance



FRED_API_KEY= "55ddb1dfa49988e7cd6ecaa8095a5143"
fred = Fred(api_key=FRED_API_KEY)



# Read the file into a DataFrame
df = pd.read_csv('economic_lexycon/result_lexycon_speeches_economic.csv')

#df = df[df['File'].str.contains('monetary')]
#df = df[df['File'].str.contains('fomcminutes')]

df = df.set_index('File')

interest_rate = fred.get_series('FEDFUNDS', observation_start='01/01/2013', observation_end='01/01/2024' )

interest_rate = pd.DataFrame({'Interest Rate': interest_rate})

interest_rate = interest_rate.resample('MS').mean()
interest_rate.index = pd.to_datetime(interest_rate.index, format='%Y-%m')

interest_rate_growth = interest_rate.pct_change()
# Remove the '.htm.txt' from the 'File' column
# Convert the index to a string
df.index = df.index.astype(str)

# Remove the '.htm.txt' from the 'File' index

df['Date'] = df.index.str.extract(r'(\d{8})', expand=False)

#df.index = df.index.str.replace('fomcminutes', '')



df['Date']= pd.to_datetime(df['Date'], format='%Y%m%d')

df_agg = df.groupby('Date').mean()




# Convert 'File' index to datetime format

# Adjust 'File' dates to the first day of the month
df_agg.index = df_agg.index.values.astype('datetime64[M]')

# Convert 'File' to datetime format

df_agg = pd.merge(df_agg, interest_rate, left_index=True, right_index=True, how='left')

df_agg['Interest Rate Change'] = df_agg['Interest Rate'].pct_change()
# Set 'File' as the index



df_agg.to_csv('economic_lexycon/comparison_interest_sentiment_speeches.csv', sep=',', index=True)

fig, ax1 = plt.subplots()

# Plot the 'Value' column
ax1.plot(df_agg['Value'], color='blue')
ax1.set_ylabel('Value', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create a second y-axis that shares the same x-axis
ax2 = ax1.twinx()

# Plot the 'Interest Rate' column on the second y-axis
ax2.plot(df_agg['Interest Rate Change'], color='red')
ax2.set_ylabel('Interest Rate Change', color='red')
ax2.tick_params(axis='y', labelcolor='red')

#ax2.plot(df_agg['Interest Rate Change'], color='green')

# Show the plot
plt.show()

plt.savefig('economic_lexycon/interest_change_speeches.png')