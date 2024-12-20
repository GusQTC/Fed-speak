import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import zlib
import base64
import spacy
import json

from collections import Counter, defaultdict

# Download NLTK resources (only required for the first run)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000  # Set the desired maximum length


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

dirs = ['compressed_statements', "compressed_speeches", 'compressed_fomc_minutes' ]
yearly_word_frequencies = defaultdict(Counter)
year_month_entities = defaultdict(lambda: defaultdict(Counter))

# Example usage

result = pd.DataFrame(columns=['File', 'Word Count', 'Positive', 'Negative', 'sentiment_score_economic', 'Year', 'Month'])
yearly_word_frequencies = defaultdict(Counter)
for current_dir in dirs:
    for file in os.listdir(current_dir):

        file.replace('.htm.txt', 'txt')

        text = open(f'{current_dir}/{file}', 'r', encoding='utf8').read()

        decompressed = zlib.decompress(base64.b64decode(text.encode())).decode()
            # Tokenize the text and perform NER

        max_chunk_length = 200000  # Adjust as needed

            # Split the text into smaller chunks
        chunks = [decompressed[i:i+max_chunk_length] for i in range(0, len(decompressed), max_chunk_length)]

# Process each chunk separately
        for chunk in chunks:
            doc = nlp(chunk)
            text_tokens = [token.text for token in doc]

            # Extract named entities
            named_entities = [(ent.text, ent.label_) for ent in doc.ents]

            sentiment_score_economic, positives, negatives = calculate_sentiment_score(text_tokens, economic_dict)
            word_frequencies, word_densities = calculate_word_frequencies(text_tokens)

            word_count = len(text_tokens)

            #get only the numbers in the file name
            file = re.sub('[^0-9]', '', file)
            year = file[0:4]
            year = str(year)
            date = file[0:-4]
            # month
            month = file[4:6]
            month = str(month)

            count = 0
            for entity, label in named_entities:
                entity_label = f"{entity} - {label}"

                if label not in ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']:
                    year_month_entities.setdefault(year, {}).setdefault(month, {}).setdefault(entity_label, 0)
                    year_month_entities[year][month][entity_label] += 1
                    #save the entities to a file
                    with open(f'NER/year_month_entities.json', 'w') as f:
                            json.dump(year_month_entities, f)


                else:
                    print(f"Skipping entity '{entity}' with label '{label}'")


            print(f'{file}: {sentiment_score_economic}')
            
            
            

            result = pd.concat([result, pd.DataFrame([[file, word_count, positives, negatives, sentiment_score_economic, year, month]], columns=['File', 'Word Count','Positive', 'Negative', 'sentiment_score_economic', 'Year', 'Month'])])
            result.to_csv(f'new_data/result_economic.csv', sep=',', index=False)

