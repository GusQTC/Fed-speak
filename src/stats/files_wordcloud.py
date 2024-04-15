from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import os
'''
for file in os.listdir('word_frequency_speeches'):
    year = file[-8:-4]
    word_frequencies = pd.read_csv(f'word_frequency_speeches/{file}').set_index('0').to_dict()['1']
    wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(word_frequencies)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Frequencies in {year}')
    plt.show()
    wordcloud.to_file(f'wordcloud_{year}.png')'''

# Read the file into a DataFrame
df = pd.read_csv('word_frequency_minutes/economic_2013.csv')

year = 2013
# Create a WordCloud object
wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(df.set_index('0').to_dict()['1'])
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title(f'Word Frequencies in {year}')
plt.show()
wordcloud.to_file(f'wordcloud_{year}.png')
