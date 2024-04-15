import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data into a pandas DataFrame
df = pd.read_csv('values/result_economic_filtered.csv')

#normalize sentiment score
#using min-max normalization
df['Mean_Sentiment_Score'] = (df['Mean_Sentiment_Score'] - df['Mean_Sentiment_Score'].min()) / (df['Mean_Sentiment_Score'].max() - df['Mean_Sentiment_Score'].min())


def plot_sentiment_score_mean(df):
    plt.figure(figsize=(10,6))
    plt.plot(df['Date'], df['Mean_Sentiment_Score']/df['Mean_Word_Count'])
    plt.xlabel('Date')
    plt.ylabel('Sentiment Score Mean')
    plt.title('Sentiment Score Mean over Time')
    plt.show()

def plot_sentiment_word_count(df):
    # Plot the Positive and Negative word counts over time
    plt.figure(figsize=(10,6))
    plt.plot(df['Date'], df['Mean_Positive']/df['Mean_Word_Count'], label='Positive')
    plt.plot(df['Date'], df['Mean_Negative']/df['Mean_Word_Count'], label='Negative')
    plt.xlabel('Date')
    plt.ylabel('Word Count')
    plt.title('Positive and Negative Word Counts over Time')
    plt.legend()
    plt.show()

def plot_sentiment_score(df):
    # Plot the Sentiment Score Mean and Standard Deviation over time
    ax1 = df.plot(x='Date', y='Mean_Sentiment_Score', legend=True)
    ax1.set_ylabel('Sentiment Score Mean', color='b')
    ax2 = df.plot(x='Date', y='StdDev_Sentiment_Score', secondary_y=True, ax=ax1, legend=True)
    ax2.set_ylabel('Standard Deviation Sentiment Score', color='g')
    plt.title('Sentiment Score Mean and Standard Deviation over Time')
    plt.show()

def plot_word_file_count(df):
    # Plot word Count and file count over time
    #two axis
    ax1 = df.plot(x='Date', y='Mean_File_Count', legend=True)
    ax1.set_ylabel('File Count', color='b')
    ax2 = df.plot(x='Date', y='Mean_Word_Count', secondary_y=True, ax=ax1, legend=True)
    ax2.set_ylabel('Word Count', color='g')
    plt.title('Word Count and File Count over Time')
    plt.show()
