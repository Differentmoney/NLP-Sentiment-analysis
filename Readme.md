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

Tweets prior and after preprocess:
Prior:
![Screen Shot 2022-04-02 at 1 26 25 PM](https://user-images.githubusercontent.com/56705341/161394281-1be4e412-f540-40fc-b312-0dd80dc70e9c.png)
Preprocessed:
![Screen Shot 2022-04-02 at 1 27 46 PM](https://user-images.githubusercontent.com/56705341/161394339-434921fe-3840-4b6d-9a6d-d5fa9c6b3fe7.png)

Word frequency for positive Sentiments
![Screen Shot 2022-04-02 at 1 21 19 PM](https://user-images.githubusercontent.com/56705341/161394095-f70c5475-cf14-45cc-a42e-47de4000d0e9.png)
Words frequency for Negative Sentiments
![Screen Shot 2022-04-02 at 1 22 19 PM](https://user-images.githubusercontent.com/56705341/161394132-4d46a559-3328-40e8-96f7-0f3b029d818c.png)


