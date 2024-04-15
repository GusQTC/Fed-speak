import pandas as pd
import numpy as np
from scipy import stats

# Read the CSV file
df = pd.read_csv('values/result_economic.csv')

# Group by 'Year' and 'Month' and calculate the average values
aggregated_df = df.groupby(['Year', 'Month']).agg(
    Mean_File_Count=('File', 'count'),
    Mean_Word_Count=('Word Count', 'mean'),
    Mean_Positive=('Positive', 'mean'),
    Mean_Negative=('Negative', 'mean'),
    Mean_Sentiment_Score=('sentiment_score_economic', 'mean'),
    Median_Sentiment_Score=('sentiment_score_economic', 'median'),
    StdDev_Sentiment_Score=('sentiment_score_economic', 'std')
).reset_index()

aggregated_df['Date'] = pd.to_datetime(aggregated_df[['Year', 'Month']].assign(DAY=1))

aggregated_df['Mean_Sentiment_Score_Corrected'] = aggregated_df['Mean_Sentiment_Score'] / aggregated_df['Mean_Word_Count']
aggregated_df['Mean_Positive_Corrected'] = aggregated_df['Mean_Positive'] / aggregated_df['Mean_Word_Count']
aggregated_df['Mean_Negative_Corrected'] = aggregated_df['Mean_Negative'] / aggregated_df['Mean_Word_Count']
aggregated_df['Median_Sentiment_Score_Corrected'] = aggregated_df['Median_Sentiment_Score'] / aggregated_df['Mean_Word_Count']
aggregated_df['StdDev_Sentiment_Score_Corrected'] = aggregated_df['StdDev_Sentiment_Score'] / aggregated_df['Mean_Word_Count']

#remove year and month
aggregated_df = aggregated_df.drop(columns=['Year', 'Month'])

# Save the aggregated DataFrame to a new CSV file
aggregated_df.to_csv('values/result_economic_corrected.csv', index=False)


