import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd

from collections import Counter, defaultdict

# Download NLTK resources (only required for the first run)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

df_dict_big = pd.read_csv('lexycon/Loughran-McDonald_MasterDictionary_1993-2023.csv')

df_dict_big['Word'] = df_dict_big['Word'].str.lower()

economic_dict = pd.read_csv('lexycon/Economic_Lexicon.csv')

# Preprocess and tokenize function
def preprocess_and_tokenize(text):
    # Remove HTML tags, URLs, and special characters
    clean_text = re.sub('<[^<]+?>|https?://\S+|[^A-Za-z0-9]+', ' ', text)

    # Convert to lowercase
    clean_text = clean_text.lower()

    # Tokenize the text
    tokens = word_tokenize(clean_text)

    # Remove stopwords, perform lemmatization
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Remove stopwords and perform lemmatization
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

    return tokens


def calculate_word_frequencies(tokens):
    # Calculate word frequencies
    word_frequencies = Counter(tokens)
    
    # Calculate word densities
    word_densities = {word: count / len(tokens) for word, count in word_frequencies.items()}
    
    return word_frequencies, word_densities

def calculate_sentiment_score(tokens, df_dict):
    sentiment_score = 0
    positive_count = 0
    negative_count = 0
    # Iterate through each token
    for token in tokens:
        token_sentiment = 0

        #if token in df_dict['Word'].values:
        if token in df_dict['token'].values:
            token_sentiment += df_dict[df_dict['token'] == token]['sentiment'].values[0]

            if token_sentiment > 0:
                positive_count += 1
            elif token_sentiment < 0:
                negative_count += 1

            #token_sentiment += df_dict[df_dict['Word'] == token]['Positive'].values[0]
            #token_sentiment -= df_dict[df_dict['Word'] == token]['Negative'].values[0]


        sentiment_score += token_sentiment

    return sentiment_score, positive_count, negative_count

dir = 'speeches'

# Example usage

result = pd.DataFrame(columns=['Value'])
count = pd.DataFrame(columns=['Positive', 'Negative'])
yearly_word_frequencies = defaultdict(Counter)

for file in os.listdir(dir):

    file.replace('.htm.txt', 'txt')

    text = open(f'{dir}/{file}', 'r', encoding='utf8').read()
    text_tokens = preprocess_and_tokenize(text)
    sentiment_score_economic, positives, negatives = calculate_sentiment_score(text_tokens, economic_dict)
    word_frequencies, word_densities = calculate_word_frequencies(text_tokens)
    year = file[0:4]
    date = file[0:-4]
    yearly_word_frequencies[year] += word_frequencies

    print(f'{file}: {sentiment_score_economic}')

    count = pd.concat([count, pd.DataFrame({'Positive':[positives], 'Negative': [negatives]})])
    count.to_csv(f'tokens/count_{dir}_economic.csv', sep=',', index=False)

    result = pd.concat([result, pd.DataFrame({'Date': [date], 'Value': [sentiment_score_economic]})])
    result.to_csv(f'tokens/result_{dir}_economic.csv', sep=',', index=False)

    for year, frequencies in yearly_word_frequencies.items():
        print(f'{year}: {file}')
        pd.DataFrame(frequencies.most_common(100)).to_csv(f'word_frequency_speeches/economic_{year}.csv', sep=',', index=False)
