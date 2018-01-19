# Main program
# Purpose: Analyse sentiments: Trump vs Hillary
# Details:
#
# %% Imports and definitions
from twitterminer import TwitterMiner
from data_processor import DataProcessor
from twitter_analyzer import TwitterAnalyzer
from news_analyzer import NewsAnalyzer
import pandas as pd
pd.set_option('max_colwidth', 400)

# %% - TwitterMiner as data source
#
# Load authentication details from configuration file
with open('twitter_conf.txt', 'r') as f:
    consumer_key, consumer_secret, access_token, access_secret = f.read().split(',')

# Establish connection to Twitter
tm = TwitterMiner(consumer_key, consumer_secret, access_token, access_secret)

# Get tweets
df = tm.get_tagged(n=300, tag='hillary')

# Save data as pickle
tm.drop_pickle('twitter_temp.pickle')

# %% - Append new Twitter data to main Twitter database
#
# Create dataprocessor
dp = DataProcessor()

# Append temporary file to database
temp_df = dp.append_to_database('twitter_temp.pickle', 'twitter_db.pickle')

# Shows tags
dp.show_tags('twitter_db.pickle')

# %% - Set up LanguageAnalyzer for twitter_db
#
ta = TwitterAnalyzer('twitter_db.pickle')

# %% - Show sentiment for musk
ta.show_sentiment('musk')

# %% - Show sentiment for trump
ta.show_sentiment('trump')

# %% - Show sentiment for hillary
ta.show_sentiment('hillary')

# %% - Compare sentiment (polarity) for trump and hillary
ta.compare_tags(['trump', 'hillary'], aspect='polarity')

# %% - Show concordance for the word "Trump" in tweets tagget with "trump"
ta.show_concordance(10, 'trump', 'Trump')

# %% - Show the 10 most frequent words in tweets tagged with "trump"
ta.show_frequent_words(10, 'trump')

# %% - Set up NewsAnalyzer for news_db
#
na = NewsAnalyzer('news_db.pickle')

# %% - Generate tokens and show most frequent words used by tv2
na.generate_tokens()
print(na.show_frequent_words(20, 'bt'))

# %% End of file

#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 300

fig = plt.figure(figsize=(8,8))
ax1 = plt.subplot(211)
plt.subplots_adjust(hspace=0.4)  
ta.show_sentiment('trump')

ax2 = plt.subplot(212)
ta.show_sentiment('hillary')


