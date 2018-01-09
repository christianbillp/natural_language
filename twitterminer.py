# TwitterMiner
# Purpose: Data collector
# Details: Connects to Twitter and fetches tagged tweets
#
# %% Imports and definitions
import tweepy
import nltk
import pandas as pd
import datetime
from tweepy import OAuthHandler
from textblob import TextBlob
pd.set_option('max_colwidth', 400)


class TwitterMiner():
    '''For mining tags from twitter'''

    def __init__(self, consumer_key, consumer_secret, access_token, access_secret):
        self.auth = OAuthHandler(consumer_key, consumer_secret)
        self.auth.set_access_token(access_token, access_secret)
        self.api = tweepy.API(self.auth)

    def get_recent(self, n):
        '''Get [n] latest tweets from home feed'''
        rl = [status.text for status in tweepy.Cursor(self.api.home_timeline).items(n)]
        return rl

    def clean_tokenize(self, input_tweet):
        '''Removes tags and hyperlinks from tweet'''
        wl = [word for word in input_tweet.split(' ') if word.startswith('#h') == False and
                                                         word.startswith('http') == False]
        rl = nltk.word_tokenize(' '.join(wl))

        return rl

    def get_tagged(self, n, tag):
        '''Get [n] tweets tagged with [tag]'''
        rl = [status.text for status in tweepy.Cursor(self.api.search, q=tag).items(n)]
        self.df = pd.DataFrame(rl, columns=['tweet'])
        self.df['posix_time'] = datetime.datetime.now().timestamp()
        self.df['clean tokens'] = self.df['tweet'].apply(self.clean_tokenize)
        self.df['polarity'] = [TextBlob(tweet).sentiment[0] for tweet in self.df['tweet']]
        self.df['subjectivity'] = [TextBlob(tweet).sentiment[1] for tweet in self.df['tweet']]
        self.df['tag'] = tag.lower()

        return self.df

    def drop_pickle(self, filename):
        '''Saves data as pickle'''
        self.df.to_pickle(filename)


if __name__ == '__main__':
    '''Tests connection to twitter, extract tweets and drops data pickle'''

    # Load authentication details from configuration file
    with open('twitter_conf.txt', 'r') as f:
        consumer_key, consumer_secret, access_token, access_secret = f.read().split(',')

    # Establish connection to Twitter
    tm = TwitterMiner(consumer_key, consumer_secret, access_token, access_secret)

    # Get tweets
    df = tm.get_tagged(n=300, tag='Trump')

    # Save data as pickle
    tm.drop_pickle('twitter_temp.pickle')

# %% End of file
