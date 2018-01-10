# TwitterAnalyzer
# Purpose: Analyses and interprets gathered data from Twitter
# Details: Visualization and tables
#
# %% Imports and definitions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('max_colwidth', 400)
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from sklearn.cluster import KMeans
import string

class NewsAnalyzer():
    '''Main class for NewsAnalyzer. Construct with database filename'''

    def __init__(self, filename):
        self.df = pd.read_pickle(filename)

    def show_frequent_words(self, n, tag):
        '''Shows [n] most frequent words in corpus with [tag]'''
        # Additional stopwords may be needed depending on topic
        sw = stopwords.words('danish')
        sw = sw + [char for char in string.punctuation]
        sw = sw+ ['\'s', 'RT', '\'\'', 'The', '``', '’', '“', "n't", 'http', 'https', 'I',
                  'kan', 'to', '...', 'nbsp']


        ts = self.df[self.df['tag'] == tag]['clean tokens'].values
        rl = [word for wordlist in ts for word in wordlist if word.lower() not in sw]
        text = nltk.word_tokenize(' '.join(rl))
        fdist = nltk.FreqDist(text)
        plt.title('{} Most frequent words in tweets with tag: "{}"'.format(n, tag))
        fdist.plot(n)

        return fdist.most_common(n)

    def clean_tokenize(self, input_tweet):
        '''Removes tags and hyperlinks from tweet'''
        wl = [word for word in input_tweet.split(' ') if word.startswith('#h') == False and
                                                         word.startswith('http') == False]
        rl = nltk.word_tokenize(' '.join(wl))

        return rl

    def generate_tokens(self):
        self.df['clean tokens'] = self.df['Content'].apply(self.clean_tokenize)


if __name__ == '__main__':
    ''''''
    na = NewsAnalyzer('news_db.pickle')
    na.generate_tokens()
    print(na.show_frequent_words(20, 'tv2'))

#%%





















