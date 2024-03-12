import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd

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


def calculate_sentiment_score(tokens, df_dict):
    sentiment_score = 0

    # Iterate through each token
    for token in tokens:
        token_sentiment = 0

        if token in df_dict['Word'].values:
            #token_sentiment += df_dict[df_dict['token'] == token]['sentiment'].values[0]

            token_sentiment += df_dict[df_dict['Word'] == token]['Positive'].values[0]
            token_sentiment -= df_dict[df_dict['Word'] == token]['Negative'].values[0]


        sentiment_score += token_sentiment

    return sentiment_score

dir = 'speeches'

# Example usage

result = pd.DataFrame(columns=['File', 'Value'])

for file in os.listdir(dir):

    file.replace('.htm.txt', 'txt')

    text = open(f'{dir}/{file}', 'r', encoding='utf8').read()
    text_tokens = preprocess_and_tokenize(text)
    sentiment_score_economic = calculate_sentiment_score(text_tokens, df_dict_big)
    print(f'{file}: {sentiment_score_economic}')
    result = pd.concat([result, pd.DataFrame({'File': [file], 'Value': [sentiment_score_economic]})])
    result.to_csv('result_lexycon_speeches.csv', sep=',', index=False)
