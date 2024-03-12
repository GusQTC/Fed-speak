import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('comparison_interest_sentiment_speeches.csv', index_col= 1, parse_dates=True)


fig, ax1 = plt.subplots()

# Plot the 'Value' column
ax1.plot(df['Value'], color='blue')
ax1.set_ylabel('Value', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create a second y-axis that shares the same x-axis
ax2 = ax1.twinx()

# Plot the 'Interest Rate' column on the second y-axis
ax2.plot(df['Interest Rate Binary'], color='red')
ax2.set_ylabel('Interest Rate', color='red')
ax2.tick_params(axis='y', labelcolor='red')

#ax2.plot(df_agg['Interest Rate Change'], color='green')

# Show the plot
plt.show()