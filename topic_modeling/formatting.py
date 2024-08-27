import csv
from collections import defaultdict
import pandas as pd

#read csv

data = defaultdict(list)

with open("compressed_speeches_topics.csv", 'r', encoding='utf8') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header row
    for row in reader:
        year_month, topic_id, topic_words = row
        # The topic_words string looks like this: "0.018*"also" + ..."
        # We only need the first number (the topic probability), so we'll split the string and take the first part
        topic_prob = float(topic_words.split('*')[0])
        data[year_month].append(topic_prob)
        
        
df = pd.DataFrame.from_dict(data, orient='index')

df.sort_index(inplace=True)

X = df