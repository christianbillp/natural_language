# TwitterMiner
# Purpose: Data collector
# Details: Connects to Twitter and fetches tagged tweets
#
# %% Imports and definitions
import tweepy
from tweepy import OAuthHandler
from pymongo import MongoClient

class TwitterMiner():
    '''For mining tags from twitter'''

    def __init__(self, consumer_key, consumer_secret, access_token, access_secret):
        self.auth = OAuthHandler(consumer_key, consumer_secret)
        self.auth.set_access_token(access_token, access_secret)
        self.api = tweepy.API(self.auth)
        
        with open('sql.conf', 'r') as f:
            self.username, self.password, self.hostname, self.db_name = f.read().split(',')

        
    def mine_and_send(self, tag, n_items):
        '''Mines [n_items] tweets with [tag]'''
        tweets = [status for status in tweepy.Cursor(self.api.search, q=tag).items(n_items)]
    
        with open('mongo.conf', 'r') as f:
            username, password, hostname, port, db_name = f.read().strip('\n').split(',')
        
        with MongoClient(f'mongodb://{username}:{password}@{hostname}:{port}/{db_name}') as client:
            [client['nlp']['twitter'].insert_one(item._json) for item in [tweet for tweet in tweets]]        
    
if __name__ == '__main__':
    '''Tests connection to twitter, extract tweets and drops data pickle'''

    # Load authentication details from configuration file
    with open('twitter.conf', 'r') as f:
        consumer_key, consumer_secret, access_token, access_secret = f.read().split(',')

    # Establish connection to Twitter
    tm = TwitterMiner(consumer_key, consumer_secret, access_token, access_secret)

    # Get tweets and save directly to mongoDB    
    tm.mine_and_send('#trump', 100)

# %% End of file
