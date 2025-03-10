import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer


df = pd.read_csv("/Users/ayaan/Downloads/data-amazon/Reviews.csv")

# print(df.head())

# print(df.shape)

#EDA
ax = df['Score'].value_counts().sort_index() \
    .plot(kind='bar',
          title='Count of Reviews by Stars',
          figsize=(10, 5))
ax.set_xlabel('Review Stars')
# plt.show()

#NLTK
example = df['Text'][50]
print(example)

tokens = nltk.word_tokenize(example)
print(tokens[:10])

tagged = nltk.pos_tag(tokens)
print(tagged[:10])

entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()

#Vader Sentiment Analysis
sia = SentimentIntensityAnalyzer()
print(sia.polarity_scores('I am so happy!'))
print(sia.polarity_scores('This is the worst song ever.'))
print(sia.polarity_scores(example))

res = {}
for i, row in df.iterrows():
    myid = row['Id']
    text = row['Text']
    res[myid] = sia.polarity_scores(text)

count = 0
for key, value in res.items():
    if count >= 3:  # Stop after 3
        break
    print(f"{key}: {value}")
    count += 1