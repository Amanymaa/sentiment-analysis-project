######sentiment analysis project

import string
import pandas as pd
import numpy as np
import self as self
import sklearn as sk
import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import json
import csv
import tweepy as tw
import os
from nltk import text
from sklearn.linear_model import LogisticRegression
from wordcloud import WordCloud
from textblob import TextBlob
import sys
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import twython
from twython import Twython
plt.style.use('fivethirtyeight')
################part1
#nltk.download_shell()
#API
import oauth
consumer_key =""
consumer_secret=""
access_token=""
access_token_secret = ""

auth =tw.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)
api =tw.API(auth,wait_on_rate_limit=True,wait_on_rate_limit_notify=True)
# Instantiate an object
joker_tweets = Twython(consumer_key, consumer_secret)
# Create our query
query = {'q': '#jokermovie',
        'count': 100,
        'lang': 'en',
        }
# Search tweets
dict_ = {'user': [], 'date': [], 'text': [], 'favorite_count': []}
for status in joker_tweets.search(**query)['statuses']:
    dict_['user'].append(status['user']['screen_name'])
    dict_['date'].append(status['created_at'])
    dict_['text'].append(status['text'])
    dict_['favorite_count'].append(status['favorite_count'])
# Structure data in a pandas DataFrame for easier manipulation
df = pd.DataFrame(dict_)
df.sort_values(by='favorite_count', inplace=True, ascending=False)
print(df.head())
print(df.describe())
df.to_csv('/sentimentAnalysis/joker.csv')

tweet_text=df['text']
print(tweet_text.head())
print(tweet_text.tail())
print(df.shape)
print('\n')

Tweets = [line.rstrip() for line in df]
print(len(Tweets))
print(df.info())

#Let's use groupby to use describe by sentiment
print(df.groupby('text').describe())
#print the first ten tweets
for tweet_no, tweet in enumerate(df['text'][:10]):
    print(tweet_no,  tweet)
    print('\n')
#Let's make a new column to detect how long the tweets are:
df['length'] =df['text'].apply(len)
print(df.head())

#Data Visualization
df['length'] .plot(bins=50, kind='hist')
plt.show()

#comment on the plot about long tweet:we found that the max length is 140
print(df.length.describe())
#lets try to find the tweet
print(df[df['length'] == 140]['text'].iloc[0])

#Clean the tweets:
#nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
text=df['text']
def clean(text):
    text = re.sub(r'@[A-Za-z0–9]+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'RT[\s]+', '', text) # to remove retweets
    text = re.sub(r'https?:\/\/\S+', '', text)
    text = re.sub(r'[^\w\s]','',text)
    text = text.replace(text, text.lower())

    return text

def delstopw(t):
    return [w for w in t if w not in stopwords.words('english')]

df['text'] = df['text'].apply(clean)
df['tokenized_text'] = df.apply(lambda row: nltk.word_tokenize(row['text']), axis=1)
df['tokenized_text'] = df['tokenized_text'].apply(delstopw)
print(df.head())
readf = df.drop(['text'], axis=1)
print(readf.head())

#plot the word cloud
import matplotlib.pyplot as plt
allWords=' '.join([tweet for tweet in df['text']])
wordCloud=WordCloud(width=500,height=300,random_state=21,max_font_size=119).generate(allWords)
plt.imshow(wordCloud,interpolation="bilinear")
plt.axis("off")
plt.show()

#subjectivity
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity
df['subjectivity']=df['text'].apply(getSubjectivity)

# polarity
def getPolarity(text):
    return TextBlob(text).sentiment.polarity
df['polarity']=df['text'].apply(getPolarity)
print(df.head())

#create a function to count negative , positive and neutral
def getAnalysis(score):
    if score<0:
        return "negative"
    elif score>0:
        return "positive"
    else :
        return "neutral"
    
df['Analysis']=df['polarity'].apply(getAnalysis)
print(df.head())

#plot polarity & subjectivity
plt.figure(figsize=(8,6))
for i in range(0,df.shape[0]):
    plt.scatter(df['polarity'][i],df['subjectivity'][i],color='green')
plt.title('Sentiment Analysis')
plt.xlabel('polarity')
plt.ylabel('subjectivity')
plt.show()
#classify sentiment
df.hist(column='length', by='Analysis', bins=50,figsize=(10,8))
plt.show()

#positive tweets
pos_tweets=df[df.Analysis =='positive']
pos_tweets=pos_tweets['text']
print(pos_tweets)
#number of pos_tweets
print(round((pos_tweets.shape[0] / df.shape[0]) *100 , 1))
#number of neg_tweets
neg_tweets=df[df.Analysis =='negative']
neg_tweets=neg_tweets['text']
print(neg_tweets)
print(round((neg_tweets.shape[0] / df.shape[0]) *100 , 1))

#value count
c=df['Analysis'].value_counts()
plt.xlabel('sentiment')
plt.ylabel('counts')
c.plot(kind='bar')
plt.show()

#part #2

#cleaning is done

# Part #3
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

bow_transformer = CountVectorizer(analyzer=delstopw).fit(df['Analysis'])
# Model Validation
from sklearn.model_selection import train_test_split

tweet_train, tweet_test, sentiment_train, sentiment_test = train_test_split(df['tokenized_text'], df['Analysis'],
                                                                            test_size=0.2)

#
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
#
# print(classification_report(predictions, sentiment_test))
# print(confusion_matrix(predictions, sentiment_test))
# print(accuracy_score(predictions, sentiment_test))


# Training the model using CountVectorizer
bow_transformer = CountVectorizer(analyzer=delstopw)
bow_transformer.fit(tweet_train, sentiment_train)
tweet_bow = bow_transformer.transform(df['Analysis'])

predictions = bow_transformer.predict(tweet_test)
print(classification_report(predictions, sentiment_test))
print(confusion_matrix(predictions, sentiment_test))
print(accuracy_score(predictions, sentiment_test))


# Training the model using TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
vectorizer.fit(tweet_train, sentiment_train)
tfidf_transformer = TfidfTransformer().fit(tweet_bow)
tweet_tfidf = tfidf_transformer.transform(tweet_bow)

predictions = vectorizer.predict(tweet_test)
print(classification_report(predictions, sentiment_test))
print(confusion_matrix(predictions, sentiment_test))
print(accuracy_score(predictions, sentiment_test))


# # Training the model using MultinomialNB
spam_detect_model = MultinomialNB().fit(tweet_tfidf, df['Analysis'])
spam_detect_model.fit(tweet_train, sentiment_train)

predictions = spam_detect_model.predict(tweet_test)
print(classification_report(predictions, sentiment_test))
print(confusion_matrix(predictions, sentiment_test))
print(accuracy_score(predictions, sentiment_test))


# Training the model using LogisticRegression
model = LogisticRegression()
model.fit(tweet_train, sentiment_train)

predictions = model.predict(tweet_test)
print(classification_report(predictions, sentiment_test))
print(confusion_matrix(predictions, sentiment_test))
print(accuracy_score(predictions, sentiment_test))
