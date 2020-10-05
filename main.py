######sentiment analysis project 
import string
from msilib import sequence
from sre_parse import Tokenizer

import gensim as gensim
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
from sklearn import pipeline, naive_bayes
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
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

joker_tweets = Twython(consumer_key, consumer_secret)

# Create our query
max_id="1312763298098667525"
max_iters=10
for call in range(0,max_iters):
    query = {'q': '#jokermovie',
            'count': 100,
            'lang': 'en',
            'max_id': max_id
            }
# Search tweets
dict_ = {'id': [], 'date': [], 'text': [], 'favorite_count': []}
for status in joker_tweets.search(**query)['statuses']:
    dict_['id'].append(status['id'])
    dict_['date'].append(status['created_at'])
    dict_['text'].append(status['text'])
    dict_['favorite_count'].append(status['favorite_count'])
# Structure data in a pandas DataFrame for easier manipulation
df = pd.DataFrame(dict_)
df.sort_values(by='favorite_count', inplace=True, ascending=False)
print(df.head())
print(df.describe())
df.to_csv('/sentimentAnalysis/joker.csv')
df=pd.read_csv('/sentimentAnalysis/joker.csv')
tweet_text=df['text']
print(tweet_text.head())
print(df.shape)
print('\n')

Tweets = [line.rstrip() for line in df]
print(len(Tweets))
print(df.info())

#Let's use groupby to use describe by sentiment
print('grouping')
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
#plt.show()

#comment on the plot about long tweet:we found that the max length is 140
print(df.length.describe())
#lets try to find the tweet
print(df[df['length'] == 140]['text'].iloc[0])

#Clean the tweets:
#nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

text=df['text']
def clean(text):
    text = re.sub(r'@[A-Za-z0–9]+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'RT[\s]+', '', text) # to remove retweets
    text = re.sub(r'https?:\/\/\S+', '', text)
    text = re.sub(r'[^\w\s]','',text)
    text = text.replace(text, text.lower())
    return text
stopset=set(stopwords.words('english'))
def delstopw(t):
    return [w for w in t if w not in stopwords.words('english')]

df['text'] = df['text'].apply(clean)
df['tokenized_text'] = df.apply(lambda row: nltk.word_tokenize(row['text']), axis=1)
df['tokenized_text'] = df['tokenized_text'].apply(delstopw)
print(df.head())
readf = df.drop(['text'], axis=1)
print(readf.head())

#normalization and limatization
tokenized_text=df['tokenized_text']
def normalization(tokenized_text):

    lem = nltk.WordNetLemmatizer()
    normalized_tweet = []
    for word in text:
        normalized_text = lem.lemmatize(word, 'v')
        normalized_tweet.append(normalized_text)
    return normalized_tweet
print('normalized :')
print(normalization(tokenized_text))

#plot the word cloud
import matplotlib.pyplot as plt
allWords=' '.join([tweet for tweet in df['text']])
wordCloud=WordCloud(width=500,height=300,random_state=21,max_font_size=119).generate(allWords)
plt.imshow(wordCloud,interpolation="bilinear")
plt.axis("off")
#plt.show()

#sentiment analysis:
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

#transform sentiment to binary to use in the model :
encoded_labels = [1 if label =='positive' else 0 for label in df['Analysis']]
encoded_labels = np.array(encoded_labels)
print(encoded_labels)
df.insert(4,"pos_or_neg",encoded_labels,True)
print(df["pos_or_neg"].head())

#plot polarity & subjectivity
plt.figure(figsize=(8,6))
for i in range(0,df.shape[0]):
    plt.scatter(df['polarity'][i],df['subjectivity'][i],color='green')
plt.title('Sentiment Analysis')
plt.xlabel('polarity')
plt.ylabel('subjectivity')
#plt.show()
#classify sentiment
df.hist(column='length', by='Analysis', bins=50,figsize=(10,8))
#plt.show()

print("How people are reacting on JOKER movie by analyzing  100  tweets:" + '\n')
#positive tweets
pos_tweets=df[df.Analysis =='positive']
pos_tweets=pos_tweets['text']
print('positive tweets are : '+ '\n')
print(pos_tweets)
#neg tweets
neg_tweets=df[df.Analysis =='negative']
neg_tweets=neg_tweets['text']
print('negative tweets are : '+ '\n')
print(neg_tweets)
#number of pos_tweets
print('%positive tweets :')
print(round((pos_tweets.shape[0] / df.shape[0]) *100 , 1))
#number of neg_tweets
print('%neg tweets:')
print(round((neg_tweets.shape[0] / df.shape[0]) *100 , 1) )

#value count
c=df['Analysis'].value_counts()
plt.xlabel('sentiment')
plt.ylabel('counts')
c.plot(kind='bar')
#plt.show()

#part #2

#cleaning is done

# # Part #3#

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
vectorizer=TfidfVectorizer(use_idf=True,lowercase=True,strip_accents='ascii',stop_words=stopset)
x=vectorizer.fit_transform(normalization(tokenized_text))
y=df.pos_or_neg
tweet_train, tweet_test, sentiment_train, sentiment_test = train_test_split(x, y,random_state=42,test_size=0.2)

# training naivebayes model
clf=naive_bayes.MultinomialNB()
clf.fit(tweet_train,sentiment_train)
#accuracy
print("accuracy of naive model is: " )
classifier_model = MultinomialNB().fit(tweet_train,sentiment_train)
all_predictions = classifier_model.predict(tweet_test)
print(classification_report(all_predictions,sentiment_test))
print(confusion_matrix(all_predictions,sentiment_test))
print(accuracy_score(all_predictions,sentiment_test))
# # #

# # Using Logistic Regression Classifier
classifier_model = LogisticRegression().fit(tweet_train,sentiment_train)
all_predictions = classifier_model.predict(tweet_test)
print("logistic Regression score is:")
print(classification_report(all_predictions,sentiment_test))
print(confusion_matrix(all_predictions,sentiment_test))
print(accuracy_score(all_predictions,sentiment_test))

#Rating Joker movie:
rating = round((pos_tweets.shape[0] / (pos_tweets.shape[0] + neg_tweets.shape[0])) * 10)
if 2.0 > rating >= 0:
    print("Movie rating = 1 star")
elif 2 <= rating < 3:
    print("Movie rating = 2 star")
elif 3 <= rating < 4:
    print("Movie rating = 2.5 star")
elif 4 <= rating < 5:
    print("Movie_rating = 3 star")
elif 5 <= rating < 6:
    print("Movie rating = 3.5 star")
elif 6 <= rating < 8:
    print("Movie rating = 4 star")
elif 8 <= rating < 9:
    print('Movie rating = 4.5 star')
elif 9 <= rating <= 10:
    print("Movie rating = 5 star")

# # END OF THE CODE
