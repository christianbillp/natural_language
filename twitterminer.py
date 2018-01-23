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
from sqlalchemy import create_engine

pd.set_option('max_colwidth', 400)


class TwitterMiner():
    '''For mining tags from twitter'''

    def __init__(self, consumer_key, consumer_secret, access_token, access_secret):
        self.auth = OAuthHandler(consumer_key, consumer_secret)
        self.auth.set_access_token(access_token, access_secret)
        self.api = tweepy.API(self.auth)
        
        with open('sql.conf', 'r') as f:
            self.username, self.password, self.hostname, self.db_name = f.read().split(',')

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
        self.df['mining_time'] = datetime.datetime.now().timestamp()
        self.df['polarity'] = [TextBlob(tweet).sentiment[0] for tweet in self.df['tweet']]
        self.df['subjectivity'] = [TextBlob(tweet).sentiment[1] for tweet in self.df['tweet']]

        self.df['clean tokens'] = self.df['tweet'].apply(self.clean_tokenize)
        self.df['tag'] = tag.lower()
        
        return self.df
    
    def test(self, tag, n):
        o = tweepy.Cursor(self.api.search, q=tag).items(n)
        tweets = [status for status in o]
        j = [tweet._json for tweet in tweets]
        self.df = pd.DataFrame(j)
        
        return self.df
    
    def send_to_sql(self, table_name):
        '''Converts internal dataframe to strings and uploads to sql database'''
        engine = create_engine(f'mysql+pymysql://{self.username}:{self.password}@{self.hostname}/{self.db_name}', encoding='utf-8')
        dfs = self.df.applymap(lambda x: str(x).encode('utf-8','ignore'))
        dfs.to_sql(table_name, engine, if_exists='append')

    def drop_pickle(self, filename):
        '''Saves data as pickle'''
        self.df.to_pickle(filename)


if __name__ == '__main__':
    '''Tests connection to twitter, extract tweets and drops data pickle'''

    # Load authentication details from configuration file
    with open('twitter.conf', 'r') as f:
        consumer_key, consumer_secret, access_token, access_secret = f.read().split(',')

    # Establish connection to Twitter
    tm = TwitterMiner(consumer_key, consumer_secret, access_token, access_secret)

    # Get tweets
#    df = tm.get_tagged(n=30, tag='Trump')
    df = tm.test(tag='#trump', n=100)

    # Save data as pickle
    tm.send_to_sql('twitter')
#    tm.drop_pickle('twitter_temp.pickle')
    
# %% End of file
