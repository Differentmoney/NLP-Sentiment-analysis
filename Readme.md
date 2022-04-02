This program uses NLP to perform sentiment analysis on Twitter Tweets about TSLA. The program uses the Twitter API to collect tweets and then uses the Natural Language Processing library to perform sentiment analysis on the tweets. The program then displays the results in a table. The program also displays the results in a graph. 

A gaussian naive based model was trained on the scraped tweets. The model was then used to predict the sentiment of the tweets. The results of the model as an accuracy score, achieving a rating of 87%.

To run the program, you must have the following python dependencies installed:
    nltk
    tweepy
    numpy
    matplotlib
    pandas
    seaborn
    sklearn
    textblob

Prior to running the program, to gain access to the twitter api you must update twitter_client.py with your access credentials. To obtain access to Twitter API use following link:
https://developer.twitter.com/en/docs/authentication/oauth-1-0a/obtaining-user-access-tokens

To run the program simply run main.py

python3 main.py

[]: # Language: python
[]: # Path: sentiment_analysis.py



