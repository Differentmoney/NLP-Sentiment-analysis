import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import re
import time
import string
import warnings
# for all NLP related operations on text
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from nltk.classify import NaiveBayesClassifier
from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

import tweepy
from tweepy import OAuthHandler 

# To identify the sentiment of text
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from textblob.np_extractors import ConllExtractor

# For Deploy
import pickle
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.pipeline import make_pipeline
from nltk.tokenize import RegexpTokenizer

import twitter_client as tc
import helper as hp

def start():
    twitter_client = tc.TwitterClient()
    tweets_df = twitter_client.get_tweets('Elon Musk', maxTweets=100000)
    
    # Fetch sentiments
    sentiments_using_SIA = tweets_df.tweets.apply(lambda tweet: hp.fetch_sentiment_using_SIA(tweet))
    pd.DataFrame(sentiments_using_SIA.value_counts())

    # Clean tweets
    tweets_df['tidy_tweets'] = np.vectorize(hp.remove_pattern)(tweets_df['tweets'], "@[\w]*: | *RT*")
    # Remove links
    cleaned_tweets = []

    for index, row in tweets_df.iterrows():
        # Here we are filtering out all the words that contains link, RT, @, #
        words_without_links = [word for word in row.tidy_tweets.split() if 'http' not in word]
        cleaned_tweets.append(' '.join(words_without_links))

    tweets_df['tidy_tweets'] = cleaned_tweets
    tweets_df = tweets_df[tweets_df['tidy_tweets']!='']
    tweets_df.drop_duplicates(subset=['tidy_tweets'], keep=False)
    tweets_df = tweets_df.reset_index(drop=True)
    tweets_df['absolute_tidy_tweets'] = tweets_df['tidy_tweets'].str.replace("[^a-zA-Z# ]", "")
    stopwords_set = set(stopwords)
    cleaned_tweets = []

    for index, row in tweets_df.iterrows():
        
        # filerting out all the stopwords 
        words_without_stopwords = [word for word in row.absolute_tidy_tweets.split() if not word in stopwords_set and '#' not in word.lower()]
        
        # finally creating tweets list of tuples containing stopwords(list) and sentimentType 
        cleaned_tweets.append(' '.join(words_without_stopwords))
        
    tweets_df['absolute_tidy_tweets'] = cleaned_tweets

    # Torkenize Tweets
    tokenized_tweets = tweets_df['absolute_tidy_tweets'].apply(lambda x: x.split())
    # Lemmatize Tweets  
    word_lemmatizer = WordNetLemmatizer()
    tokenized_tweet = tokenized_tweet.apply(lambda x: [word_lemmatizer.lemmatize(i) for i in x])
    # joining all the words in a tweet
    for i, tokens in enumerate(tokenized_tweet):
        tokenized_tweet[i] = ' '.join(tokens)

    tweets_df['absolute_tidy_tweets'] = tokenized_tweet

    #Grammer Rule
    sentence_re = r'(?:(?:[A-Z])(?:.[A-Z])+.?)|(?:\w+(?:-\w+)*)|(?:\$?\d+(?:.\d+)?%?)|(?:...|)(?:[][.,;"\'?():-_`])'
    grammar = r"""
        NBAR:
            {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns
            
        NP:
            {<NBAR>}
            {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
    """
    chunker = nltk.RegexpParser(grammar)

    # Extracting Noun Phrases
    textblob_key_phrases = []
    extractor = ConllExtractor()

    for index, row in tweets_df.iterrows():
        # filerting out all the hashtags
        words_without_hash = [word for word in row.tidy_tweets.split() if '#' not in word.lower()]
        
        hash_removed_sentence = ' '.join(words_without_hash)
        
        blob = TextBlob(hash_removed_sentence, np_extractor=extractor)
        textblob_key_phrases.append(list(blob.noun_phrases))

    tweets_df['key_phrases'] = textblob_key_phrases

    all_words = ' '.join([text for text in tweets_df['absolute_tidy_tweets'][tweets_df.sentiment == 'pos']])
    hp.generate_wordcloud(all_words)
    all_words = ' '.join([text for text in tweets_df['absolute_tidy_tweets'][tweets_df.sentiment == 'neg']])
    hp.generate_wordcloud(all_words)

    hashtags = hp.hashtag_extract(tweets_df['tidy_tweets'])
    hashtags = sum(hashtags, [])

    hp.generate_hashtag_freqdist(hashtags)
    # Keep only non empty tweets
    tweets_df2 = tweets_df[tweets_df['key_phrases'].str.len()>0]
    # feature extraction
    # BOW features
    bow_word_vectorizer = CountVectorizer(max_df=0.90, min_df=2, stop_words='english')
    # bag-of-words feature matrix
    bow_word_feature = bow_word_vectorizer.fit_transform(tweets_df2['absolute_tidy_tweets'])

    # TF-IDF features
    tfidf_word_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, stop_words='english')
    # TF-IDF feature matrix
    tfidf_word_feature = tfidf_word_vectorizer.fit_transform(tweets_df2['absolute_tidy_tweets'])

    phrase_sents = tweets_df2['key_phrases'].apply(lambda x: ' '.join(x))

    # BOW phrase features
    bow_phrase_vectorizer = CountVectorizer(max_df=0.90, min_df=2)
    bow_phrase_feature = bow_phrase_vectorizer.fit_transform(phrase_sents)

    # TF-IDF phrase feature
    tfidf_phrase_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2)
    tfidf_phrase_feature = tfidf_phrase_vectorizer.fit_transform(phrase_sents)
    
    # Build model
    target_variable = tweets_df2['sentiment'].apply(lambda x: 0 if x=='neg' else 1)

    # Split Train test
    X_train, X_test, y_train, y_test = train_test_split(bow_word_feature, target_variable, test_size=0.3, random_state=272)
    hp.naive_model(X_train, X_test, y_train, y_test)   
    
    X_train, X_test, y_train, y_test = train_test_split(tfidf_word_feature, target_variable, test_size=0.3, random_state=272)
    hp.naive_model(X_train, X_test, y_train, y_test)

    X_train, X_test, y_train, y_test = train_test_split(bow_phrase_feature, target_variable, test_size=0.3, random_state=272)
    hp.naive_model(X_train, X_test, y_train, y_test)

    X_train, X_test, y_train, y_test = train_test_split(tfidf_phrase_feature, target_variable, test_size=0.3, random_state=272)
    hp.naive_model(X_train, X_test, y_train, y_test)

    tweets_df2['sentiment'] = tweets_df2['sentiment'].apply(lambda x: 0 if x=='neg' else 1)

    # Train and test model
    pipeline_ls = make_pipeline(CountVectorizer(max_df=0.90, min_df=2, tokenizer = RegexpTokenizer(r'[A-Za-z]+').tokenize,stop_words='english'), LogisticRegression())
    X_train, X_test, y_train, y_test = train_test_split(tweets_df2.tweets, tweets_df2.sentiment)
    pipeline_ls.fit(X_train, y_train)
    pipeline_ls.score(X_test,y_test) 

