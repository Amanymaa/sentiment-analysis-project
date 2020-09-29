######sentiment analysis project
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib as plt
import seaborn as sns
import re
import os
import tweepy
import json

################part1
import nltk
#nltk.download_shell()
plt.style.use('fivethirtyeight')
df = pd.read_csv('/Users/Magic Systems/Desktop/tweet_dataset.csv')
oldText= pd.DataFrame(df['old_text'])

print(df.head())
print(oldText.head())
print(df.shape)
print('\n')

Tweets = [line.rstrip() for line in df]
print(len(Tweets))
print(df.describe())
print(df.info())

#Let's use groupby to use describe by sentiment
print(df.groupby('sentiment').describe())
#print the first ten tweets
for tweet_no, tweet in enumerate(df['text'][:10]):
    print(tweet_no,  tweet)
    print('\n')
#Let's make a new column to detect how long the tweets are:
df['length'] =df['old_text'].apply(len)
print(df.head())

#Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

df['length'] .plot(bins=50, kind='hist')
plt.show()

#comment on the plot about long tweet:we found that the max length is 167
print(df.length.describe())
#lets try to find the tweet
print(df[df['length'] == 167]['old_text'].iloc[0])

#classify sentiment
df.hist(column='length', by='new_sentiment', bins=50,figsize=(12,11))
plt.show()
###############################part 2
#Clean the tweets:
nltk.download('punkt')
def clean(text):
    text = re.sub(r'@[A-Za-z0–9]+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'RT[\s]+', '', text) # to remove retweets
    text = re.sub(r'https?:\/\/\S+', '', text)
    text = re.sub(r'[^\w\s]','',text)
    text = text.replace(text, text.lower())

    return text

df['old_text'] = df['old_text'].apply(clean)
df['tokenized_text'] = df.apply(lambda row: nltk.word_tokenize(row['old_text']), axis=1)
readf = df.drop(['old_text'],axis=1)
print(readf.head())

