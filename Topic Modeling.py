import re
from gensim import corpora
import gensim
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import zlib
import base64
import os
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

dir_speeches  = 'compressed_speeches'


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and non-alphabetic characters
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize
    words = word_tokenize(text)
    # Remove stop words and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return words

documents = {}
#read all files in the directory
for filename in os.listdir(dir_speeches):
    if filename.endswith('.txt'):
        with open(os.path.join(dir_speeches, filename), 'r', encoding="utf8") as file:
            text = file.read()
            #print('Original Text:', text)
            decompressed = zlib.decompress(base64.b64decode(text.encode())).decode()
            decompressed = preprocess(decompressed)
            
            documents[filename] = decompressed
            



# Let's start with some sample documents
#separate by year
# group by year and month
texts = {}

for filename in documents:
    #  ____20160112.txt
    year = filename[-13:-9]
    month = filename[-9:-7]
    year_month = year + month
    texts[year_month] = texts.get(year_month, '') + documents[filename]

# Preprocess the data


# for each month, create a list of documents
# for each document, split the text into words

tokens = []

topics_year_month = {}
for year_month in texts:
    print('Year_Month:', year_month)
    print('Text:', len(texts[year_month]))
    
    tokens = texts[year_month]

    
    dictionary = corpora.Dictionary(tokens)
    
    corpus = [dictionary.doc2bow(text) for text in tokens]
    
    lda = gensim.models.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15)
    
    topics = lda.print_topics(num_words=10)
    for topic in topics:
        print(topic)
        topics_year_month[year_month] = topics_year_month.get(year_month, []) + [topic]




# Create a dictionary representation of the documents.

# Convert document (a list of words) into the bag-of-words format = list of (token_id, token_count) tuples.

# Train the model on the corpus.

# Print the keyword in the 2 topics
